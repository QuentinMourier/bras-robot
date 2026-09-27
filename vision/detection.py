#!/usr/bin/env python3
"""Repérer une cible colorée dans l'image, avec sa position ET son orientation.

Pourquoi la couleur plutôt qu'un réseau de neurones : la boucle de suivi doit
rendre une position en quelques millisecondes pour que le bras colle à la
cible. Un seuillage HSV coûte 2 à 4 ms sur une image 1280x720, là où yolo11n en
NCNN demande plusieurs centaines de millisecondes sur ce Pi — l'écart décide à
lui seul de la fluidité du suivi.

L'orientation compte autant que la position : une pince à deux mors doit
s'écarter perpendiculairement au grand axe de l'objet, sinon elle glisse le
long. On mesure donc le rectangle d'aire minimale entourant la tache, dont le
grand côté donne cet axe (voir `cinematique.orientation`).

Le teint de la peau fonctionne comme n'importe quelle couleur, mais mal : il
est peu saturé et proche du bois ou du carton clair. Pour suivre un doigt,
mieux vaut lui mettre un capuchon ou une gommette franchement colorée.
"""

from collections import namedtuple

import numpy as np

# Plages HSV, en convention OpenCV (teinte sur 0-179).
#
# Rappel hérité de detection_picam.py : la Picamera2 est configurée en RGB888,
# qu'OpenCV traite comme du BGR. Les couleurs sont donc permutées par rapport à
# la réalité, et ces seuils sont réglés sur cette convention — ce qui est
# nommé « rouge » ci-dessous correspond à ce qu'OpenCV voit, pas à l'oeil.
PLAGES = {
    "rouge": [((0, 120, 80), (10, 255, 255)),        # le rouge est à cheval
              ((170, 120, 80), (180, 255, 255))],    # sur les deux bouts
    "vert":  [((35, 80, 60), (85, 255, 255))],
    "bleu":  [((90, 120, 60), (130, 255, 255))],
    "jaune": [((20, 110, 90), (35, 255, 255))],
    "rose":  [((140, 80, 90), (170, 255, 255))],
}

SURFACE_MIN = 400        # pixels ; en deçà, c'est du bruit
NOYAU = np.ones((5, 5), np.uint8)

Cible = namedtuple("Cible", "u v aire angle largeur longueur")


class DetecteurCouleur:
    """Cherche la plus grande tache d'une couleur donnée.

        detecteur = DetecteurCouleur("rouge")
        cible = detecteur.detecter(image)      # None si rien d'assez gros
    """

    def __init__(self, couleur="rouge", surface_min: int = SURFACE_MIN):
        if couleur not in PLAGES:
            raise ValueError(f"couleur inconnue : {couleur}. "
                             f"Disponibles : {', '.join(PLAGES)}")
        self.couleur = couleur
        self.plages = PLAGES[couleur]
        self.surface_min = surface_min

    def masque(self, image) -> np.ndarray:
        """Masque binaire des pixels de la couleur cherchée."""
        import cv2

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        total = None
        for bas, haut in self.plages:
            part = cv2.inRange(hsv, np.array(bas), np.array(haut))
            total = part if total is None else (total | part)
        # Ouverture : enlève les pixels isolés sans ronger la tache utile.
        return cv2.morphologyEx(total, cv2.MORPH_OPEN, NOYAU)

    def detecter_plusieurs(self, image, maximum: int = 6):
        """Les taches de la couleur, de la plus grande à la plus petite.

        Rendre plusieurs candidats, et non la seule plus grande, est ce qui
        permet de suivre le bon objet quand la scène en contient d'autres de la
        même couleur. Observé le 27 septembre 2026 : une bouteille rouge et un
        livre rouge, tous deux hors de la zone atteignable, éclipsaient le
        bouchon que l'on voulait suivre. L'appelant peut désormais parcourir
        les candidats et retenir le premier qui soit à portée du bras.
        """
        import cv2

        contours, _ = cv2.findContours(self.masque(image), cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        gardes = [c for c in contours
                  if cv2.contourArea(c) >= self.surface_min]
        gardes.sort(key=cv2.contourArea, reverse=True)
        return [self._mesurer(c) for c in gardes[:maximum]]

    @staticmethod
    def _mesurer(contour):
        import cv2

        aire = float(cv2.contourArea(contour))
        (u, v), (a, b), angle_deg = cv2.minAreaRect(contour)
        if a < b:
            a, b = b, a
            angle_deg += 90.0
        return Cible(u=float(u), v=float(v), aire=aire,
                     angle=float(np.radians(angle_deg)),
                     largeur=float(b), longueur=float(a))

    def detecter(self, image):
        """Plus grande tache de la couleur, ou None. Voir `Cible`.

        `angle` est l'orientation du GRAND axe dans l'image, en radians, et
        `largeur`/`longueur` les côtés du rectangle englobant — leur rapport
        dit si l'objet est franchement allongé, donc si son orientation a un
        sens. Une tache ronde donne un angle arbitraire : c'est au code
        appelant d'en tenir compte (voir `orientation_fiable`).
        """
        import cv2

        candidats = self.detecter_plusieurs(image, maximum=1)
        return candidats[0] if candidats else None


def orientation_fiable(cible, rapport_min: float = 1.3) -> bool:
    """Vrai si l'objet est assez allongé pour que son orientation ait un sens.

    Sur une tache quasi ronde, `minAreaRect` rend un angle qui saute d'une
    image à l'autre. S'en servir pour orienter la pince la ferait pivoter sans
    raison entre deux images : mieux vaut alors garder l'orientation courante.
    """
    return cible is not None and cible.largeur > 0 \
        and cible.longueur / cible.largeur >= rapport_min


def direction_sur_table(cible, homographie) -> float:
    """Direction du grand axe de l'objet dans le repère du bras, en radians.

    L'angle mesuré dans l'image ne vaut pas celui sur la table : la caméra
    regarde de biais, ce qui déforme les angles. On transporte donc deux points
    pris le long du grand axe à travers l'homographie, et c'est leur écart sur
    la table qui donne la direction utile.
    """
    from vision.etalonnage_camera import pixel_vers_table

    demi = cible.longueur / 2 * 0.8      # un peu en deçà des bouts
    dx, dy = np.cos(cible.angle) * demi, np.sin(cible.angle) * demi
    x1, y1 = pixel_vers_table(cible.u - dx, cible.v - dy, homographie)
    x2, y2 = pixel_vers_table(cible.u + dx, cible.v + dy, homographie)
    return float(np.arctan2(y2 - y1, x2 - x1))


if __name__ == "__main__":
    import cv2

    print("Vérification sur des images de synthèse — aucun matériel requis.\n")
    detecteur = DetecteurCouleur("bleu")

    for nom, (centre, taille, rotation) in {
        "barre horizontale": ((640, 360), (200, 40), 0),
        "barre à 30°":       ((500, 300), (200, 40), 30),
        "barre à 75°":       ((700, 400), (200, 40), 75),
        "tache ronde":       ((640, 360), (90, 90), 0),
    }.items():
        image = np.zeros((720, 1280, 3), np.uint8)
        boite = cv2.boxPoints((centre, taille, rotation)).astype(np.int32)
        # En convention BGR, « bleu » est le premier canal.
        cv2.fillPoly(image, [boite], (200, 60, 60))

        cible = detecteur.detecter(image)
        if cible is None:
            print(f"  {nom:<20} rien détecté")
            continue
        attendu = rotation % 180
        obtenu = np.degrees(cible.angle) % 180
        ecart = min(abs(obtenu - attendu), 180 - abs(obtenu - attendu))
        fiable = "oui" if orientation_fiable(cible) else "non"
        print(f"  {nom:<20} centre ({cible.u:6.1f}, {cible.v:6.1f}) "
              f"angle {obtenu:5.1f}° (attendu {attendu:5.1f}°, écart "
              f"{ecart:4.1f}°)  allongé : {fiable}")

    print("\nLa tache ronde doit être marquée « non allongé » : son angle\n"
          "n'a pas de sens, et s'en servir ferait pivoter la pince au hasard.")

#!/usr/bin/env python3
"""Éprouve la chaîne de suivi complète sans caméra ni bras.

    venv/bin/python -m apps.test_suivi

Une caméra factice fabrique l'image d'une cible colorée dont on choisit la
position sur la table ; elle est projetée dans l'image par l'homographie
INVERSE de celle qu'utilise le suivi. La chaîne testée est donc la vraie :

    image de synthèse -> DetecteurCouleur -> homographie -> Asservissement
                      -> BrasSimule

Et comme on sait où l'on a posé la cible, on peut mesurer exactement ce que le
bras devrait rattraper. C'est la seule façon d'éprouver le suivi une nuit sans
matériel, et cela vérifie précisément ce qu'une simulation du bras seule ne
verrait pas : la conversion pixel -> mètre, le lissage, le rejet des sauts
aberrants et le comportement quand la cible bouge.
"""

import sys
import time

import numpy as np

from materiel.asservissement import Asservissement, SortieSimulee
from materiel.simulation import BrasSimule
from vision.detection import DetecteurCouleur, orientation_fiable
from vision.etalonnage_camera import charger_homographie


class CameraFactice:
    """Fabrique l'image d'une barre colorée posée à un endroit de la table."""

    def __init__(self, homographie, largeur=1280, hauteur=720):
        import cv2

        self.cv2 = cv2
        self.forme = (hauteur, largeur, 3)
        # Pour dessiner, il faut aller des mètres vers les pixels : c'est
        # l'inverse de ce que fait le suivi.
        self.vers_pixels = np.linalg.inv(homographie)
        self.position = np.array([0.26, -0.05])
        self.orientation = 0.0          # grand axe sur la table, rad

    def _pixel(self, x, y):
        p = self.cv2.perspectiveTransform(
            np.array([[[float(x), float(y)]]]), self.vers_pixels)[0][0]
        return float(p[0]), float(p[1])

    def capturer(self):
        image = np.zeros(self.forme, np.uint8)
        # Une barre de 6 cm sur 2 cm, dessinée par ses quatre coins projetés :
        # la perspective est ainsi respectée, comme pour un vrai objet.
        demi_l, demi_e = 0.030, 0.010
        c, s = np.cos(self.orientation), np.sin(self.orientation)
        coins = []
        for dl, de in ((+1, +1), (+1, -1), (-1, -1), (-1, +1)):
            x = self.position[0] + dl * demi_l * c - de * demi_e * s
            y = self.position[1] + dl * demi_l * s + de * demi_e * c
            coins.append(self._pixel(x, y))
        self.cv2.fillPoly(image, [np.array(coins, np.int32)], (200, 60, 60))
        return image

    def close(self):
        pass


def main():
    try:
        homographie, hauteur = charger_homographie()
    except FileNotFoundError:
        print("Pas d'homographie : lancez d'abord `etalonnage_camera ajuster`.")
        return 1

    from apps.suivi import Vision

    camera = CameraFactice(homographie)
    detecteur = DetecteurCouleur("bleu")

    print("=== 1. La cible de synthèse est-elle retrouvée là où on l'a mise ? ===")
    ecarts = []
    for x, y, angle in [(0.22, -0.12, 0.0), (0.30, 0.02, np.pi / 4),
                        (0.26, -0.05, np.pi / 2), (0.33, -0.15, -np.pi / 6)]:
        camera.position = np.array([x, y])
        camera.orientation = angle
        cible = detecteur.detecter(camera.capturer())
        if cible is None:
            print(f"  ({x:+.2f},{y:+.2f}) : NON DÉTECTÉE")
            ecarts.append(np.inf)
            continue
        from vision.etalonnage_camera import pixel_vers_table
        rx, ry = pixel_vers_table(cible.u, cible.v, homographie)
        ecart = np.hypot(rx - x, ry - y) * 1000
        ecarts.append(ecart)
        print(f"  posée ({x:+.2f},{y:+.2f}) -> retrouvée ({rx:+.3f},{ry:+.3f})"
              f"   écart {ecart:5.2f} mm   allongée : "
              f"{'oui' if orientation_fiable(cible) else 'non'}")
    print(f"  écart maximal de la boucle image<->table : {max(ecarts):.2f} mm\n")

    print("=== 2. Le bras rejoint-il une cible fixe ? ===")
    bras = BrasSimule(bavard=False)
    sortie = SortieSimulee(bras)
    camera.position = np.array([0.26, -0.05])
    vision = Vision(camera, detecteur, homographie, hauteur)
    vision.start()
    time.sleep(0.4)

    with Asservissement(sortie) as asser:
        debut = time.monotonic()
        atteint = False
        while time.monotonic() - debut < 12:
            position, _ = vision.derniere()
            if position is not None:
                asser.viser(float(position[0]), float(position[1]), hauteur)
                if asser.erreur() < 0.003:
                    atteint = True
                    break
            time.sleep(0.02)
        duree = time.monotonic() - debut
        print(f"  {'rejointe' if atteint else 'NON rejointe'} en {duree:.2f} s, "
              f"écart final {asser.erreur() * 1000:.2f} mm")

        print("\n=== 3. Le bras suit-il une cible qui bouge ? ===")
        print(f"  {'temps':>7} {'cible':>18} {'pince':>18} {'retard':>9}")
        retards = []
        depart = time.monotonic()
        while time.monotonic() - depart < 6.0:
            t = time.monotonic() - depart
            # Un aller-retour lent, comme un doigt qu'on promène.
            camera.position = np.array([0.26 + 0.05 * np.sin(t * 0.8),
                                        -0.05 + 0.06 * np.cos(t * 0.8)])
            position, _ = vision.derniere()
            if position is not None:
                asser.viser(float(position[0]), float(position[1]), hauteur)
                p = asser.etat()["pince_xyz"]
                retard = np.linalg.norm(camera.position - p[:2]) * 1000
                retards.append(retard)
                if len(retards) % 40 == 0:
                    print(f"  {t:6.1f}s ({camera.position[0]:+.3f},"
                          f"{camera.position[1]:+.3f}) "
                          f"({p[0]:+.3f},{p[1]:+.3f}) {retard:7.1f} mm")
            time.sleep(0.02)

        stables = retards[len(retards) // 3:]      # après le rattrapage initial
        print(f"\n  retard moyen une fois accroché : {np.mean(stables):.1f} mm")
        print(f"  retard maximal                 : {np.max(stables):.1f} mm")
        etat = asser.etat()
        print(f"  {etat['cycles']} cycles d'asservissement, "
              f"{etat['retards']} en retard de cadence")
        print(f"  vision : {vision.images} images, "
              f"{1 / max(vision.periode, 1e-6):.1f} img/s")

    vision.tourne = False
    print("\nLe retard en suivi vient surtout du lissage et de la vitesse\n"
          "maximale : il se règle par LISSAGE et --vitesse.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

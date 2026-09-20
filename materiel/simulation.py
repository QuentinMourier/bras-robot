#!/usr/bin/env python3
"""Back-end `Bras` simulé : le bras piloté sans le matériel.

Complète `materiel/servos.py` (les vrais servos) derrière la même interface
`materiel.interface.Bras`. Le code appelant — vision, logique de saisie, futures
applications de `apps/` — ne voit aucune différence : il appelle `aller_a`,
`ouvrir_pince`, `lire_positions` de la même façon.

Ce qui est simulé fidèlement
----------------------------
Tout ce qui relève de la *décision*, et c'est le point de ce module : la
simulation rejoue les mêmes contrôles que le bras réel, en réutilisant le même
code (`cinematique.securite.ik_est_valide`, `etalonnage.limiter`, et les
constantes de `servos.py`) plutôt qu'en les réécrivant :

  - la cinématique inverse et son seuil de convergence ;
  - les butées mesurées à la main (LIMITES_DEG), pas celles de l'URDF ;
  - le refus d'un écart de plus de ECART_MAX_DEG sur un axe en un seul
    `aller_a`, signe habituel d'un changement de configuration de l'IK ;
  - le découpage du mouvement en paliers de PALIER_DEG ;
  - l'arrêt d'urgence par fichier témoin (`touch /tmp/bras-stop`).

Conséquence utile : une séquence acceptée en simulation le sera aussi par le
bras réel. C'est ce qui permet de préparer une application sans le matériel,
puis de la rejouer sur le bras en ne changeant qu'une ligne.

Ce qui n'est PAS simulé
-----------------------
Il n'y a ici aucune physique : ni gravité, ni contact, ni couple.

  - `charge()` renvoie toujours 0. Or la surveillance de charge est le
    garde-fou principal du bras réel : la simulation ne peut donc pas détecter
    une collision, et une trajectoire validée ici peut parfaitement traverser
    la table ou le support.
  - Le biais de suivi des servos (0,1° sur le roulis à vide, jusqu'à 1,0° sur
    le coude en charge) n'existe pas : la simulation arrive exactement sur sa
    consigne, là où `BrasServos._affiner` doit rattraper l'écart.
  - `pince_tient()` renvoie toujours False : il n'y a pas d'objet à saisir.

Autrement dit, la simulation valide la géométrie et la logique, pas la sûreté
physique. Elle ne dispense pas de surveiller le premier passage sur le bras.

Pour de la vraie physique — contacts, préhension, objet qui tombe — il faudrait
MuJoCo et les maillages STL, absents de `urdf/assets/`.

Utilisation
-----------
    from materiel.simulation import BrasSimule

    with BrasSimule() as bras:
        bras.aller_a(0.25, 0.05, 0.15)
        bras.fermer_pince()
        bras.tracer("trajet.png")

En ligne de commande, toutes les commandes de test_bras.py acceptent --simu :

    venv/bin/python -m materiel.test_bras --simu etat
    venv/bin/python -m materiel.test_bras --simu aller 0.25 0.05 0.15 --vue trajet.png
"""

import math
import os
import time

import numpy as np

from cinematique.ik import angles_utiles, position_de, vecteur_complet
from cinematique.securite import ik_est_valide, interpoler
from materiel.etalonnage import (AXES_BRAS, AXE_PINCE, LIMITES_DEG,
                                 NOMS_ARM101, PINCE_FERMEE, PINCE_OUVERTE,
                                 SENS_LATERAUX_VERIFIES, TOUS_LES_AXES,
                                 ZERO_DEG, angles_vers_degres,
                                 degres_vers_angles, limiter)
from materiel.interface import Bras
# Les constantes de mouvement sont importées du back-end réel, jamais recopiées :
# c'est ce qui garantit que la simulation refuse exactement ce qu'il refuse.
from materiel.servos import (ECART_MAX_DEG, PALIER_DEG, TOLERANCE_FINALE_DEG,
                             VITESSE)
from materiel.sts3215 import FICHIER_STOP, arret_demande

# Vitesse du bras réel, en degrés par seconde : VITESSE est en pas/s, et un tour
# complet fait 4096 pas. Sert uniquement au mode `temps_reel`, qui fait durer un
# mouvement simulé aussi longtemps qu'il durerait sur le bras.
DEGRES_PAR_SECONDE = VITESSE * 360.0 / 4096.0

# Pose de départ par défaut : celle où tous les angles URDF valent zéro, soit le
# bras tendu vers l'avant, pince à (0,391 ; 0 ; 0,227) m. C'est la référence
# naturelle du modèle, et elle tombe bien dans les butées mesurées.
POSE_ZERO_URDF = {sid: ZERO_DEG[sid] for sid in AXES_BRAS}


class BrasSimule(Bras):
    """Le SO-ARM101 sans matériel : même interface, mêmes refus, aucune physique."""

    def __init__(self, pose_deg: dict | None = None, bavard: bool = True,
                 temps_reel: bool = False, ecart_max_deg: float = ECART_MAX_DEG,
                 **_ignores):
        """`pose_deg` : position de départ en degrés servo (défaut POSE_ZERO_URDF).

        `temps_reel` : si vrai, un mouvement dure aussi longtemps qu'il durerait
        sur le bras réel. Pratique pour répéter une séquence au rythme réel,
        inutile (et pénible) pour une simple vérification géométrique.

        Les arguments propres au matériel (`port`, `baud`…) sont acceptés et
        ignorés, pour que la simulation soit un remplacement direct de
        BrasServos dans du code déjà écrit.
        """
        self._degres = dict(pose_deg) if pose_deg else dict(POSE_ZERO_URDF)
        self._degres.setdefault(AXE_PINCE, PINCE_OUVERTE)

        for sid, degres in self._degres.items():
            _, bride = limiter(sid, degres)
            if bride:
                bas, haut = LIMITES_DEG[sid]
                raise ValueError(
                    f"pose de départ hors butées : axe {sid} "
                    f"({NOMS_ARM101[sid]}) à {degres:+.1f}°, "
                    f"attendu dans [{bas:+.0f}°, {haut:+.0f}°]")

        self.bavard = bavard
        self.temps_reel = temps_reel
        self.ecart_max_deg = ecart_max_deg
        self._stop_demande = False
        # Historique des poses traversées, une par palier : c'est ce que trace
        # `tracer()` et ce qui permet de rejouer un mouvement après coup.
        self.trajectoire = [dict(self._degres)]

    # ------------------------------------------------------------------
    # Cycle de vie
    # ------------------------------------------------------------------
    def close(self) -> None:
        """Sans objet en simulation ; présent pour l'homogénéité avec BrasServos."""

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def _dire(self, message: str) -> None:
        if self.bavard:
            print(f"[simu] {message}")

    # ------------------------------------------------------------------
    # Lecture
    # ------------------------------------------------------------------
    def degres_servo(self, axes=TOUS_LES_AXES) -> dict:
        """Position actuelle de chaque axe, en degrés servo (0 = centre)."""
        return {sid: self._degres[sid] for sid in axes}

    def lire_positions(self):
        """Angles des 5 articulations (rad, repère URDF), ordre AXES_BRAS."""
        return np.array(degres_vers_angles(self._degres))

    def position_pince(self):
        """Position (x, y, z) du point de préhension, en mètres."""
        return position_de(vecteur_complet(self.lire_positions()))

    def charge(self, sid: int) -> int:
        """Toujours 0 : sans physique, il n'y a pas d'effort à mesurer.

        Ce n'est pas un détail d'implémentation mais la limite principale de ce
        back-end — voir l'en-tête du module.
        """
        return 0

    # ------------------------------------------------------------------
    # Mouvement
    # ------------------------------------------------------------------
    def aller_a(self, x: float, y: float, z: float) -> bool:
        """Amène le point de préhension en (x, y, z), en mètres. Voir Bras."""
        depart = self.lire_positions()
        ok, solution, erreur_mm, message = ik_est_valide(
            x, y, z, angles_initiaux=vecteur_complet(depart))
        if not ok:
            self._dire(f"mouvement refusé : {message}")
            return False

        self._dire(f"cible ({x:.3f}, {y:.3f}, {z:.3f}) m, "
                   f"résidu d'IK {erreur_mm:.1f} mm")
        if abs(y) > 0.02 and not SENS_LATERAUX_VERIFIES:
            self._dire("rappel : le sens des axes 1 et 5 n'est pas vérifié sur "
                       "le bras réel, donc le signe de y reste à confirmer "
                       "(voir etalonnage.SENS_LATERAUX_VERIFIES). Sans effet "
                       "en simulation, où le modèle fait foi.")
        return self.aller_angles(angles_utiles(solution))

    def aller_angles(self, angles) -> bool:
        """Amène les 5 articulations aux angles donnés (rad, repère URDF).

        Reproduit les contrôles de BrasServos.aller_angles : butées mesurées
        puis limite d'écart sur un seul mouvement.
        """
        cibles = angles_vers_degres(angles)
        depart = self.degres_servo(AXES_BRAS)

        for sid, degres in cibles.items():
            borne, bride = limiter(sid, degres)
            cibles[sid] = borne
            if bride:
                bas, haut = LIMITES_DEG[sid]
                self._dire(f"mouvement refusé : axe {sid} ({NOMS_ARM101[sid]}) "
                           f"demandé à {degres:+.1f}°, hors des butées mesurées "
                           f"[{bas:+.0f}°, {haut:+.0f}°]")
                return False

        ecarts = {sid: cibles[sid] - depart[sid] for sid in AXES_BRAS}
        sid_max = max(ecarts, key=lambda s: abs(ecarts[s]))
        if abs(ecarts[sid_max]) > self.ecart_max_deg:
            self._dire(
                f"mouvement refusé : il faudrait {ecarts[sid_max]:+.0f}° sur "
                f"l'axe {sid_max} ({NOMS_ARM101[sid_max]}), plus que les "
                f"{self.ecart_max_deg:.0f}° autorisés d'un coup. L'IK a "
                "probablement changé de configuration ; approchez la cible en "
                "plusieurs étapes.")
            return False

        return self._executer(depart, cibles)

    def _executer(self, depart: dict, cibles: dict) -> bool:
        """Parcourt le mouvement par paliers, en enregistrant chaque pose."""
        if arret_demande():
            self._dire(f"mouvement refusé : arrêt d'urgence armé. "
                       f"`rm {FICHIER_STOP}` pour réarmer.")
            return False

        self._stop_demande = False
        amplitude = max(abs(cibles[sid] - depart[sid]) for sid in AXES_BRAS)
        if amplitude < TOLERANCE_FINALE_DEG:
            self._dire("déjà en position")
            return True

        nombre = max(1, math.ceil(amplitude / PALIER_DEG))
        etapes = interpoler([depart[sid] for sid in AXES_BRAS],
                            [cibles[sid] for sid in AXES_BRAS], n_pas=nombre)
        self._dire(f"{amplitude:.0f}° à parcourir, {nombre} paliers")

        duree_palier = (amplitude / nombre) / DEGRES_PAR_SECONDE
        for i, etape in enumerate(etapes[1:], start=1):
            # L'arrêt d'urgence est relu à chaque palier, comme sur le bras réel
            # où la boucle de surveillance le relit toutes les 30 ms.
            if self._stop_demande or arret_demande():
                self._dire(f"arrêt au palier {i}/{nombre} : arrêt d'urgence")
                return False
            if self.temps_reel:
                time.sleep(duree_palier)

            self._degres.update(dict(zip(AXES_BRAS, etape)))
            self.trajectoire.append(dict(self._degres))

        # Pas de rattrapage de biais à faire : sans couple, on arrive pile.
        self._degres.update(cibles)
        self.trajectoire.append(dict(self._degres))
        self._dire("arrivé (écart résiduel nul : aucun servo à rattraper)")
        return True

    def stop(self) -> None:
        """Arrêt d'urgence : le bras reste là où il en est de sa trajectoire."""
        self._stop_demande = True
        self._dire("arrêt d'urgence : le bras est figé sur place")

    def liberer(self) -> None:
        """Sans effet : sans gravité, un bras sans couple ne retombe pas."""
        self._dire("couple coupé (sans effet en simulation : pas de gravité)")

    # ------------------------------------------------------------------
    # Pince
    # ------------------------------------------------------------------
    def ouvrir_pince(self) -> None:
        """Ouvre la pince. Rien ne peut la gêner en simulation."""
        self._degres[AXE_PINCE] = PINCE_OUVERTE
        self.trajectoire.append(dict(self._degres))
        self._dire(f"pince ouverte à {PINCE_OUVERTE:+.1f}°")

    def fermer_pince(self) -> None:
        """Ferme la pince, qui va toujours jusqu'au bout faute d'objet."""
        self._degres[AXE_PINCE] = PINCE_FERMEE
        self.trajectoire.append(dict(self._degres))
        self._dire(f"pince fermée à {PINCE_FERMEE:+.1f}° — à vide "
                   "(la simulation ne contient aucun objet)")

    def pince_tient(self) -> bool:
        """Toujours False : il n'y a rien à tenir dans cette simulation."""
        return False

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------
    def poses_urdf(self, pas: int = 1):
        """Trajectoire enregistrée, convertie en angles URDF (rad).

        `pas` permet d'alléger le tracé en ne gardant qu'une pose sur n.
        """
        return [np.array(degres_vers_angles(pose))
                for pose in self.trajectoire[::pas]]

    def tracer(self, fichier: str | None = None, cible=None, pas: int = 1,
               titre: str | None = None):
        """Dessine la trajectoire parcourue en 3D. Voir `tracer_poses`."""
        return tracer_poses(self.poses_urdf(pas), fichier=fichier, cible=cible,
                            titre=titre)

    def animer(self, fichier: str = "trajet_simule.gif", pas: int = 1,
               titre: str | None = None, images_par_seconde: int = 12):
        """Fabrique un GIF animé du mouvement. Voir `animer_poses`."""
        return animer_poses(self.poses_urdf(pas), fichier=fichier, titre=titre,
                            images_par_seconde=images_par_seconde)


def squelette(angles) -> np.ndarray:
    """Positions (x, y, z) des 7 points de la chaîne, de la base à la pince.

    C'est la cinématique directe de chaque lien, et non plus seulement du point
    de préhension : de quoi dessiner le bras entier.
    """
    from cinematique.ik import bras
    return np.array([m[:3, 3] for m in
                     bras.forward_kinematics(vecteur_complet(angles),
                                             full_kinematics=True)])


def tracer_poses(poses, fichier: str | None = None, cible=None,
                 titre: str | None = None):
    """Trace une ou plusieurs poses du bras en 3D.

    La pose finale est dessinée en plein, les précédentes en gris pâle, et le
    chemin parcouru par la pince en trait plein : le mouvement se lit d'un coup
    d'œil. (Le tracé maison remplace `ikpy.chain.plot`, qui donne à chaque pose
    une couleur vive et rend une trajectoire illisible dès quelques dizaines de
    poses.)

    Sans `fichier`, la fenêtre matplotlib s'ouvre — il faut alors un écran ;
    avec, l'image est écrite sur disque, ce qui marche aussi en SSH.

    Retourne le chemin du fichier écrit, ou None si la figure a été affichée.
    """
    import matplotlib
    # Sans écran (SSH, service systemd), matplotlib doit basculer sur un moteur
    # de rendu hors écran avant que pyplot ne soit importé.
    ecran = os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
    if fichier or not ecran:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    squelettes = [squelette(angles) for angles in poses]

    figure = plt.figure(figsize=(9, 7.5))
    axes = figure.add_subplot(111, projection="3d")

    for os_bras in squelettes[:-1]:
        axes.plot(os_bras[:, 0], os_bras[:, 1], os_bras[:, 2],
                  color="0.75", linewidth=1.0, zorder=1)

    chemin_pince = np.array([s[-1] for s in squelettes])
    axes.plot(chemin_pince[:, 0], chemin_pince[:, 1], chemin_pince[:, 2],
              color="tab:orange", linewidth=2.0, zorder=3,
              label="chemin de la pince")

    final = squelettes[-1]
    axes.plot(final[:, 0], final[:, 1], final[:, 2], color="tab:blue",
              linewidth=3.0, marker="o", markersize=5, zorder=4,
              label="pose finale")

    axes.scatter(*chemin_pince[0], color="tab:green", s=60, zorder=5,
                 label="départ")
    axes.scatter(*chemin_pince[-1], color="tab:red", s=60, zorder=5,
                 label="arrivée")
    if cible is not None:
        axes.scatter(*cible, marker="*", color="black", s=140, zorder=6,
                     label="cible")

    # La base du bras, pour situer l'échelle.
    axes.scatter(0, 0, 0, marker="s", color="black", s=40, zorder=5)

    axes.set_xlabel("x (m)")
    axes.set_ylabel("y (m)")
    axes.set_zlabel("z (m)")
    axes.set_title(titre or f"Trajectoire simulée — {len(poses)} poses")
    axes.legend(loc="upper left", fontsize=9)

    # Repère orthonormé : sans cela matplotlib étire les axes indépendamment et
    # le bras paraît disproportionné.
    tous = np.vstack(squelettes)
    centre = (tous.max(axis=0) + tous.min(axis=0)) / 2
    portee = (tous.max(axis=0) - tous.min(axis=0)).max() / 2
    axes.set_xlim(centre[0] - portee, centre[0] + portee)
    axes.set_ylim(centre[1] - portee, centre[1] + portee)
    axes.set_zlim(min(0, centre[2] - portee), centre[2] + portee)
    axes.view_init(elev=22, azim=-125)

    if fichier:
        figure.savefig(fichier, dpi=110, bbox_inches="tight")
        plt.close(figure)
        return fichier
    plt.show()
    return None


# Cycle de démonstration : aller chercher un objet posé en (0,25 ; 0,10 ; 0,07)
# et le déposer 15 cm plus loin. C'est la forme qu'aura la première application
# de `apps/`, et elle tient entièrement dans l'enveloppe validée par
# cinematique/enveloppe.py.
#
# Les points de passage ne sont pas décoratifs : depuis le bras tendu, viser
# directement la cible demande plus de 60° sur un axe, ce que `aller_angles`
# refuse (à raison — voir ECART_MAX_DEG). On approche donc en plusieurs pas,
# comme il faudra le faire sur le bras réel.
CYCLE_DEMO = [
    ("approche haute",   (0.36, 0.00, 0.22), None),
    ("approche",         (0.33, 0.00, 0.21), None),
    ("au-dessus objet",  (0.28, 0.05, 0.16), "ouvrir"),
    ("descente",         (0.26, 0.10, 0.12), None),
    ("sur l'objet",      (0.25, 0.10, 0.07), "fermer"),
    ("levée",            (0.26, 0.10, 0.14), None),
    ("transfert",        (0.28, -0.02, 0.16), None),
    ("dépose",           (0.28, -0.03, 0.08), "ouvrir"),
    ("dégagement",       (0.29, -0.03, 0.16), None),
]


def animer_poses(poses, fichier: str = "trajet_simule.gif",
                 titre: str | None = None, images_par_seconde: int = 12):
    """Écrit un GIF animé où le bras rejoue la trajectoire, pose par pose.

    Le chemin déjà parcouru par la pince reste tracé derrière elle, ce qui
    permet de suivre le mouvement même sur une image arrêtée.

    Retourne le chemin du fichier écrit.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import animation

    squelettes = [squelette(angles) for angles in poses]
    chemin_pince = np.array([s[-1] for s in squelettes])

    figure = plt.figure(figsize=(8, 7))
    axes = figure.add_subplot(111, projection="3d")

    # Cadre fixe : sans cela les axes seraient recalculés à chaque image et le
    # bras semblerait grandir et rétrécir au lieu de bouger.
    tous = np.vstack(squelettes)
    centre = (tous.max(axis=0) + tous.min(axis=0)) / 2
    portee = (tous.max(axis=0) - tous.min(axis=0)).max() / 2
    axes.set_xlim(centre[0] - portee, centre[0] + portee)
    axes.set_ylim(centre[1] - portee, centre[1] + portee)
    axes.set_zlim(min(0, centre[2] - portee), centre[2] + portee)
    axes.set_xlabel("x (m)")
    axes.set_ylabel("y (m)")
    axes.set_zlabel("z (m)")
    axes.view_init(elev=22, azim=-125)
    axes.scatter(0, 0, 0, marker="s", color="black", s=40)

    trace_bras, = axes.plot([], [], [], color="tab:blue", linewidth=3.0,
                            marker="o", markersize=5)
    trace_chemin, = axes.plot([], [], [], color="tab:orange", linewidth=1.8)

    def image(i):
        os_bras = squelettes[i]
        trace_bras.set_data(os_bras[:, 0], os_bras[:, 1])
        trace_bras.set_3d_properties(os_bras[:, 2])
        passe = chemin_pince[:i + 1]
        trace_chemin.set_data(passe[:, 0], passe[:, 1])
        trace_chemin.set_3d_properties(passe[:, 2])
        axes.set_title(f"{titre or 'Trajectoire simulée'} — pose "
                       f"{i + 1}/{len(squelettes)}")
        return trace_bras, trace_chemin

    anim = animation.FuncAnimation(figure, image, frames=len(squelettes),
                                   blit=False)
    anim.save(fichier, writer=animation.PillowWriter(fps=images_par_seconde))
    plt.close(figure)
    return fichier


if __name__ == "__main__":
    with BrasSimule(bavard=False) as bras_simule:
        depart = bras_simule.position_pince()
        print(f"Départ : ({depart[0]:+.3f}, {depart[1]:+.3f}, "
              f"{depart[2]:+.3f}) m\n")

        for etape, cible, pince in CYCLE_DEMO:
            if not bras_simule.aller_a(*cible):
                bras_simule.bavard = True       # rejoue l'étape en expliquant
                bras_simule.aller_a(*cible)
                print(f"\nArrêt à l'étape « {etape} ».")
                break
            atteint = bras_simule.position_pince()
            ecart = np.linalg.norm(np.array(cible) - atteint) * 1000
            print(f"  {etape:<16} ({atteint[0]:+.3f}, {atteint[1]:+.3f}, "
                  f"{atteint[2]:+.3f}) m   à {ecart:.1f} mm de la cible")
            if pince == "ouvrir":
                bras_simule.ouvrir_pince()
                print(f"  {'  → pince ouverte':<16}")
            elif pince == "fermer":
                bras_simule.fermer_pince()
                print(f"  {'  → pince fermée':<16}")

        chemin = bras_simule.tracer(
            "trajet_simule.png", pas=2,
            titre="Cycle de saisie simulé — SO-ARM101")
        print(f"\n{len(bras_simule.trajectoire)} poses parcourues, "
              f"tracé écrit dans {chemin}")

        gif = bras_simule.animer(titre="Cycle de saisie simulé")
        print(f"Animation écrite dans {gif}")

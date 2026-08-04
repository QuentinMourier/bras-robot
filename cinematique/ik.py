"""Cinématique directe et inverse du SO-ARM101, à partir de l'URDF officiel.

La géométrie exacte vient de urdf/so101_new_calib.urdf, chargée par ikpy
(remplace l'ancienne approximation manuelle, cf. project_bras_robot_cinematique).
"""

from pathlib import Path

import numpy as np
from ikpy.chain import Chain

URDF_PATH = Path(__file__).parent.parent / "urdf" / "so101_new_calib.urdf"

# ikpy suit automatiquement la branche gripper_frame_joint (point de préhension)
# plutôt que la branche "gripper" (mâchoire mobile) : c'est la bonne chaîne pour
# le positionnement, la mâchoire étant pilotée séparément (voir materiel/interface.py).
bras = Chain.from_urdf_file(str(URDF_PATH))

# Les deux liens fixes (origine + point de préhension) ne sont pas des degrés de
# liberté : on les exclut des angles à faire varier pendant l'IK.
MASQUE_LIENS_ACTIFS = [
    False if link.joint_type == "fixed" else True for link in bras.links
]
bras.active_links_mask = MASQUE_LIENS_ACTIFS

# ikpy fait de l'optimisation locale : partir toujours d'angles à zéro peut
# converger vers un mauvais minimum local sur certaines cibles pourtant
# atteignables (observé jusqu'à 72 mm d'erreur lors du benchmark). On limite
# ce risque avec plusieurs essais depuis des points de départ différents,
# et on s'arrête dès qu'un essai est assez précis (pas la peine de tous les
# faire à chaque fois).
N_ESSAIS_MULTI_DEPART = 6
SEUIL_ARRET_ANTICIPE_M = 0.001  # 1 mm : au-delà, pas la peine d'essayer un autre départ
_rng = np.random.default_rng()


def _borner(angles):
    """Ramène un vecteur d'angles dans les butées déclarées par l'URDF.

    La position réelle du bras peut les dépasser de un ou deux degrés : les
    courses mesurées à la main sur le bras monté sont un peu plus larges que
    celles du modèle (l'épaule va jusqu'à 101,6° pour 100° déclarés). Or
    scipy refuse un point de départ hors bornes. Comme ce n'est qu'un point de
    départ d'optimisation, le ramener dans les bornes est sans conséquence.
    """
    bornes = []
    for angle, link in zip(angles, bras.links):
        bas, haut = link.bounds
        if bas is not None:
            angle = max(bas, angle)
        if haut is not None:
            angle = min(haut, angle)
        bornes.append(angle)
    return np.array(bornes)


def _points_de_depart(angles_initiaux):
    """Construit la liste des vecteurs de départ à essayer pour l'IK, dans l'ordre :
    1. les angles initiaux fournis par l'appelant (souvent la position actuelle du
       bras : le meilleur départ possible, la cible n'étant généralement pas loin) ;
    2. tous les angles à zéro (le défaut historique) ;
    3. des tirages aléatoires dans les butées de chaque articulation, pour
       échapper aux minima locaux propres à la pose zéro.
    """
    departs = []
    if angles_initiaux is not None:
        departs.append(_borner(np.array(angles_initiaux, dtype=float)))
    departs.append(np.zeros(len(bras.links)))
    while len(departs) < N_ESSAIS_MULTI_DEPART:
        tirage = [0.0]
        for link in bras.links[1:-1]:
            bas, haut = link.bounds
            tirage.append(_rng.uniform(bas, haut))
        tirage.append(0.0)
        departs.append(np.array(tirage))
    return departs


def angles_pour(x: float, y: float, z: float, angles_initiaux=None) -> np.ndarray:
    """Cinématique inverse : position cible (x, y, z) en mètres -> angles des 5 servos (rad).

    Essaie plusieurs points de départ (voir _points_de_depart) et retourne la
    meilleure solution trouvée, pour limiter le risque de mauvaise convergence.
    Retourne le vecteur complet des liens ikpy (avec des 0 aux liens fixes) ;
    utiliser angles_utiles(...) pour les 5 angles moteurs utiles.
    """
    cible = np.array([x, y, z])
    meilleure_solution, meilleure_erreur = None, np.inf

    for depart in _points_de_depart(angles_initiaux):
        solution = bras.inverse_kinematics(cible, initial_position=depart)
        erreur = np.linalg.norm(bras.forward_kinematics(solution)[:3, 3] - cible)
        if erreur < meilleure_erreur:
            meilleure_solution, meilleure_erreur = solution, erreur
        if meilleure_erreur < SEUIL_ARRET_ANTICIPE_M:
            break

    return meilleure_solution


def position_de(angles) -> np.ndarray:
    """Cinématique directe : angles (vecteur complet ikpy) -> position (x, y, z) atteinte."""
    return bras.forward_kinematics(angles)[:3, 3]


# --- Conversions entre le vecteur complet ikpy (7 valeurs, liens fixes inclus)
# et les 5 angles moteurs utiles (shoulder_pan, shoulder_lift, elbow_flex,
# wrist_flex, wrist_roll) manipulés par materiel.interface.Bras. ---

def vecteur_complet(angles_5) -> np.ndarray:
    """5 angles moteurs -> vecteur ikpy complet (0 aux liens fixes, indices 0 et 6)."""
    return np.array([0.0, *angles_5, 0.0])


def angles_utiles(vecteur_complet_ikpy) -> np.ndarray:
    """Vecteur ikpy complet -> 5 angles moteurs utiles (retire les liens fixes)."""
    return np.array(vecteur_complet_ikpy[1:6])


if __name__ == "__main__":
    cible = [0.20, 0.05, 0.15]
    sol = angles_pour(*cible)
    atteint = position_de(sol)
    erreur_mm = np.linalg.norm(np.array(cible) - atteint) * 1000
    print("Angles (rad) :", np.round(sol, 4))
    print("Position atteinte :", np.round(atteint, 4))
    print(f"Erreur : {erreur_mm:.2f} mm")

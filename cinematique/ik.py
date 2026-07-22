"""Cinématique directe et inverse du SO-ARM101, à partir de l'URDF officiel.

Remplace l'ancien bras.py (dimensions approximatives codées à la main) :
ici la géométrie exacte vient de urdf/so101_new_calib.urdf, chargé par ikpy.
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


def angles_pour(x: float, y: float, z: float, angles_initiaux=None) -> np.ndarray:
    """Cinématique inverse : position cible (x, y, z) en mètres -> angles des 5 servos (rad).

    Retourne le vecteur complet des liens ikpy (avec des 0 aux liens fixes) ;
    utiliser angles[1:6] pour les 5 angles moteurs utiles.
    """
    if angles_initiaux is None:
        angles_initiaux = [0] * len(bras.links)
    return bras.inverse_kinematics([x, y, z], initial_position=angles_initiaux)


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

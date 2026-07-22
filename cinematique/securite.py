"""Vérifications de sécurité avant d'envoyer un mouvement au bras réel ou simulé."""

import numpy as np

from cinematique.ik import bras, angles_pour, position_de

SEUIL_ERREUR_MM = 5.0


def ik_est_valide(x: float, y: float, z: float, angles_initiaux=None):
    """Résout l'IK pour (x, y, z) et vérifie la convergence + les butées articulaires.

    Retourne (ok: bool, angles: np.ndarray, erreur_mm: float, message: str).
    """
    angles = angles_pour(x, y, z, angles_initiaux=angles_initiaux)
    atteint = position_de(angles)
    erreur_mm = float(np.linalg.norm(np.array([x, y, z]) - atteint) * 1000)

    if erreur_mm >= SEUIL_ERREUR_MM:
        return False, angles, erreur_mm, f"Convergence IK insuffisante ({erreur_mm:.1f} mm >= {SEUIL_ERREUR_MM} mm) : cible probablement hors de portée."

    for i, (angle, link) in enumerate(zip(angles, bras.links)):
        if link.joint_type == "fixed":
            continue
        bas, haut = link.bounds
        if not (bas <= angle <= haut):
            return False, angles, erreur_mm, f"Angle hors butée sur '{link.name}' : {angle:.3f} rad (bornes [{bas:.3f}, {haut:.3f}])"

    return True, angles, erreur_mm, "OK"


def interpoler(angles_depart, angles_arrivee, n_pas: int):
    """Découpe un mouvement en n_pas étapes linéaires entre deux jeux d'angles.

    Retourne une liste de n_pas + 1 vecteurs d'angles (départ inclus, arrivée incluse),
    à envoyer un par un aux servos pour un mouvement fluide plutôt qu'un saut brusque.
    """
    depart = np.array(angles_depart, dtype=float)
    arrivee = np.array(angles_arrivee, dtype=float)
    return [depart + (arrivee - depart) * t for t in np.linspace(0, 1, n_pas + 1)]


if __name__ == "__main__":
    ok, angles, erreur_mm, message = ik_est_valide(0.20, 0.05, 0.15)
    print(f"Valide: {ok} | erreur: {erreur_mm:.2f} mm | {message}")

    ok2, angles2, erreur_mm2, message2 = ik_est_valide(1.0, 1.0, 1.0)  # hors de portée
    print(f"Valide: {ok2} | erreur: {erreur_mm2:.2f} mm | {message2}")

    etapes = interpoler([0] * len(angles), angles, n_pas=5)
    print(f"{len(etapes)} étapes d'interpolation générées")

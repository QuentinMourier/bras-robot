"""Mesure le temps moyen de résolution IK sur des cibles aléatoires atteignables.

Utilisé pour vérifier que la résolution IK reste rapide sur le Raspberry Pi
(par rapport au PC de développement). Génère les cibles par cinématique
directe à partir d'angles aléatoires tirés dans les butées de chaque
articulation, ce qui garantit qu'elles sont bien atteignables.
"""

import time

import numpy as np

from cinematique.ik import bras, angles_pour, position_de, vecteur_complet

N_TIRAGES = 100
rng = np.random.default_rng(42)


def cible_aleatoire_atteignable():
    """Tire des angles aléatoires dans les butées, retourne la position résultante."""
    angles = [0.0]
    for link in bras.links[1:-1]:  # exclut les 2 liens fixes (origine, point de préhension)
        bas, haut = link.bounds
        angles.append(rng.uniform(bas, haut))
    angles.append(0.0)
    return position_de(angles)


if __name__ == "__main__":
    durees = []
    erreurs = []

    for _ in range(N_TIRAGES):
        cible = cible_aleatoire_atteignable()
        t0 = time.perf_counter()
        solution = angles_pour(*cible)
        t1 = time.perf_counter()

        atteint = position_de(solution)
        erreur_mm = np.linalg.norm(cible - atteint) * 1000

        durees.append(t1 - t0)
        erreurs.append(erreur_mm)

    durees = np.array(durees) * 1000  # ms
    erreurs = np.array(erreurs)

    print(f"{N_TIRAGES} résolutions IK :")
    print(f"  temps moyen   : {durees.mean():.2f} ms")
    print(f"  temps médian  : {np.median(durees):.2f} ms")
    print(f"  temps min/max : {durees.min():.2f} / {durees.max():.2f} ms")
    print(f"  erreur moyenne: {erreurs.mean():.3f} mm")
    print(f"  erreur max    : {erreurs.max():.3f} mm")

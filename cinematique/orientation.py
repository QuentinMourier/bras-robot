#!/usr/bin/env python3
"""Orienter la pince avant une saisie : quel roulis de poignet pour ces mors ?

Le problème
-----------
Une pince à deux mors ne saisit bien que dans un sens : il faut que les mors
s'écartent PERPENDICULAIREMENT au grand axe de l'objet, sinon ils glissent le
long au lieu de le pincer. Or `aller_a` ne commande qu'une position, jamais une
orientation — et wrist_roll est même gelé dans l'IK depuis le 5 septembre,
précisément pour qu'il cesse d'être choisi au hasard par l'optimiseur.

Ce module le rend à sa vraie fonction : orienter les mors, une fois la position
décidée.

La géométrie, lue dans l'URDF
-----------------------------
L'articulation `gripper` a pour origine rpy = (pi/2, 0, 0) par rapport à
`gripper_link`, et pour axe (0, 0, 1) dans son propre repère : la charnière du
mors mobile est donc portée par le -y de `gripper_link`. Le repère de
préhension (`gripper_frame_joint`, rpy = (0, pi, 0)) conserve ce y.

Il s'ensuit que, dans le repère de préhension :
    y  = axe de la charnière,
    z  = direction d'approche,
    x  = direction d'ECARTEMENT des mors.

C'est ce x qu'il faut amener perpendiculairement au grand axe de l'objet.

Méthode
-------
La relation entre le roulis et la direction des mors n'a pas d'expression
simple : elle dépend de toute la posture du bras. On l'échantillonne donc — la
cinématique directe rapide coûte 0,2 ms, donc balayer la course du roulis et
garder le meilleur reste négligeable, et c'est bien plus robuste qu'une
inversion analytique qui se tromperait de branche.

Réserve, faute de bras cette nuit
---------------------------------
La déduction ci-dessus vient de l'URDF, pas d'une observation. Si à l'usage les
mors se referment à 90° de l'attendu, c'est que l'écartement se fait selon y et
non selon x : passer AXE_MACHOIRES de 0 à 1 suffit alors à tout corriger.
"""

import numpy as np

from cinematique.jacobienne import reperes

# Colonne du repère de préhension portant l'écartement des mors : 0 = x.
AXE_MACHOIRES = 0

# Course du roulis explorée, en radians (les butées mesurées valent -165/+156
# en degrés servo ; on reste un peu en deçà).
ROULIS_MIN = np.radians(-150.0)
ROULIS_MAX = np.radians(150.0)
ECHANTILLONS = 181


def direction_machoires(angles) -> np.ndarray:
    """Vecteur unitaire, dans le repère du monde, selon lequel les mors s'écartent."""
    repere = reperes(angles)[-1]
    return repere[:3, AXE_MACHOIRES]


def angle_machoires(angles) -> float:
    """Orientation des mors projetée sur le plan de la table, en radians.

    On projette parce que c'est dans ce plan que l'objet est posé et que sa
    direction est mesurée : la composante verticale des mors ne change rien à
    la qualité de la prise.
    """
    direction = direction_machoires(angles)
    return float(np.arctan2(direction[1], direction[0]))


def roulis_pour_aligner(angles, direction_objet: float,
                        roulis_min: float = ROULIS_MIN,
                        roulis_max: float = ROULIS_MAX,
                        echantillons: int = ECHANTILLONS):
    """Roulis amenant les mors perpendiculaires à `direction_objet` (rad).

    `direction_objet` est l'angle du grand axe de l'objet dans le plan de la
    table, mesuré depuis l'axe x du bras. Retourne (roulis, écart résiduel en
    radians) : l'écart dit à quel point l'alignement voulu est atteignable — un
    écart important signale que la posture ne permet pas cette prise, et qu'il
    vaut mieux aborder l'objet autrement.

    Le critère est tridimensionnel, et il a fallu une correction pour l'être.
    Comparer les seules projections sur la table ne marche pas : le bras aborde
    l'objet presque à l'horizontale, donc les mors s'écartent dans un plan
    quasi VERTICAL, dont la projection au sol est dégénérée — mesuré, elle
    valait ±90° quel que soit le roulis, sans aucune information.

    On cherche donc le roulis qui rapproche le plus l'axe des mors de la
    direction horizontale perpendiculaire au grand axe de l'objet. Un objet et
    son demi-tour se saisissant pareil, on compare des directions non
    orientées : d'où la valeur absolue du produit scalaire.
    """
    angles = np.asarray(angles, dtype=float)
    voulu = np.array([np.cos(direction_objet + np.pi / 2),
                      np.sin(direction_objet + np.pi / 2), 0.0])

    essai = angles.copy()
    meilleur, meilleur_produit = float(angles[4]), -1.0
    for roulis in np.linspace(roulis_min, roulis_max, echantillons):
        essai[4] = roulis
        produit = abs(float(np.dot(direction_machoires(essai), voulu)))
        if produit > meilleur_produit:
            meilleur, meilleur_produit = float(roulis), produit
    ecart = float(np.arccos(np.clip(meilleur_produit, 0.0, 1.0)))
    return meilleur, ecart


if __name__ == "__main__":
    from materiel.etalonnage import AXES_BRAS, ZERO_DEG, degres_vers_angles

    repos = np.array(degres_vers_angles({s: ZERO_DEG[s] for s in AXES_BRAS}))

    print("=== Le roulis fait-il bien tourner l axe des mors ? ===")
    print(f"{'roulis':>9}   axe des mors (x, y, z)")
    for degres in (-90, -45, 0, 45, 90):
        essai = repos.copy()
        essai[4] = np.radians(degres)
        d = direction_machoires(essai)
        print(f"{degres:>+8}°   ({d[0]:+.3f}, {d[1]:+.3f}, {d[2]:+.3f})")

    print("\n=== Trouve-t-il le roulis pour un objet donné ? ===")
    print(f"{'objet':>8} {'roulis trouvé':>15} {'mors':>10} {'écart':>9}")
    for objet in (0, 30, 60, 90, 120, 150):
        roulis, ecart = roulis_pour_aligner(repos, np.radians(objet))
        essai = repos.copy()
        essai[4] = roulis
        mors = np.degrees(angle_machoires(essai))
        print(f"{objet:>7}° {np.degrees(roulis):>14.1f}° {mors:>9.1f}° "
              f"{np.degrees(ecart):>8.2f}°")

    print("\nUn écart proche de zéro signifie que l'alignement demandé est\n"
          "atteignable ; les mors sont alors perpendiculaires au grand axe.")

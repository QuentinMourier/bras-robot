#!/usr/bin/env python3
"""Cinématique différentielle : suivre une cible en continu, sans optimiseur.

Pourquoi ce module existe
-------------------------
`cinematique.ik.angles_pour` résout l'IK par optimisation numérique avec six
points de départ. C'est robuste pour « va à cet endroit », et c'est ce qu'il
faut pour un grand déplacement. Mais le banc d'essai mesure 18 ms en médiane,
38 ms en moyenne et jusqu'à 328 ms au pire — impossible d'asservir le bras sur
une cible qui bouge à ce rythme-là.

Or, pour suivre, on n'a pas besoin de résoudre le problème complet à chaque
image : la cible n'a bougé que de quelques millimètres depuis la fois
précédente. On cherche donc un petit déplacement articulaire dq qui produise le
petit déplacement cartésien dx voulu, ce qui est un simple problème linéaire :

    dx = J(q) . dq        J = la jacobienne, une matrice 3 x n

Coût : une passe de cinématique directe et l'inversion d'une matrice 3 x 3,
soit quelques dizaines de microsecondes. Trois ordres de grandeur plus rapide.

Avantage inattendu, et sans doute le plus important : cette méthode **ne peut
pas changer de configuration**. Elle part des angles actuels et les fait varier
continûment. Les sauts de 150° sur le poignet qui bloquaient la capture
d'étalonnage, et les +162° de roulis corrigés le 5 septembre, sont par
construction impossibles ici.

Limites, en contrepartie
------------------------
  - Ne vaut que pour de petits déplacements : au-delà, la linéarisation dérive
    et il faut repasser par `ik.angles_pour`.
  - Près d'une singularité (bras tendu, ou axes alignés), la jacobienne devient
    mal conditionnée. D'où l'amortissement ci-dessous, qui échange un peu de
    précision de suivi contre des consignes articulaires bornées.

Vérifié dans `__main__` contre la cinématique directe d'ikpy, par différences
finies : c'est le seul contrôle qui puisse prendre la jacobienne en défaut.
"""

import numpy as np

from cinematique.ik import bras
from cinematique.enveloppe import _rotation_rpy

# Amortissement de Levenberg-Marquardt, en unités de la jacobienne (m/rad).
# Plus il est grand, plus les consignes sont douces près des singularités, et
# moins le suivi est fidèle. 0,02 laisse le suivi au millimètre loin des
# singularités tout en bornant dq à quelques degrés quand le bras est tendu.
AMORTISSEMENT = 0.02

# Articulations pilotées par défaut : les quatre premières. wrist_roll en est
# exclu pour la même raison qu'il est gelé dans ik.py — il ne déplace la pince
# que de 1 mm par 10°, donc il ne sert à rien pour positionner, et il sert à
# tout pour orienter la pince (voir apps/saisie.py).
AXES_POSITION = (0, 1, 2, 3)


def _liaisons_fixes():
    """Matrices 4x4 de chaque liaison, hors rotation de l'articulation."""
    fixes = []
    for lien in bras.links:
        matrice = np.eye(4)
        matrice[:3, :3] = _rotation_rpy(
            getattr(lien, "origin_orientation", np.zeros(3)))
        matrice[:3, 3] = getattr(lien, "origin_translation", np.zeros(3))
        fixes.append(matrice)
    return np.array(fixes)


# La géométrie du bras ne change jamais : on la calcule une fois pour toutes.
_FIXES = _liaisons_fixes()
_MOBILES = [i for i, lien in enumerate(bras.links) if lien.joint_type != "fixed"]


def reperes(angles) -> np.ndarray:
    """Repères 4x4 successifs de la chaîne, pour 5 angles (rad).

    Retourne un tableau (n_liens + 1, 4, 4) : le repère du monde, puis celui
    obtenu après chaque liaison. Le dernier est celui du point de préhension.
    """
    angles = np.asarray(angles, dtype=float)
    courant = np.eye(4)
    suite = [courant]
    for indice, lien in enumerate(bras.links):
        etape = _FIXES[indice]
        if lien.joint_type != "fixed":
            theta = angles[indice - 1]
            c, s = np.cos(theta), np.sin(theta)
            rotation = np.eye(4)
            rotation[0, 0], rotation[0, 1] = c, -s
            rotation[1, 0], rotation[1, 1] = s, c
            etape = etape @ rotation
        courant = courant @ etape
        suite.append(courant)
    return np.array(suite)


def position(angles) -> np.ndarray:
    """Position (x, y, z) du point de préhension. Équivalent rapide de ik.position_de."""
    return reperes(angles)[-1][:3, 3]


def position_et_jacobienne(angles, axes=AXES_POSITION):
    """Position de la pince ET jacobienne, en une seule passe. -> (p, J).

    Les deux sortent du même parcours de la chaîne : les calculer séparément
    la parcourrait deux fois, ce qui doublait le coût d'un pas d'asservissement.

    Colonne j de J : de combien la pince se déplace quand l'articulation j
    tourne d'un radian. Pour une articulation rotoïde, c'est le produit
    vectoriel de son axe par le bras de levier jusqu'à la pince :

        dp/dq = z  x  (p_pince - p_articulation)

    z et p sont pris dans le repère qui précède la rotation, celui où l'axe de
    l'articulation est le z local.
    """
    suite = reperes(angles)
    p_pince = suite[-1][:3, 3]

    colonnes = []
    for j in axes:
        indice = _MOBILES[j]
        # Repère avant rotation = repère précédent, composé de la liaison fixe.
        avant = suite[indice] @ _FIXES[indice]
        colonnes.append(np.cross(avant[:3, 2], p_pince - avant[:3, 3]))
    return p_pince, np.column_stack(colonnes)


def jacobienne(angles, axes=AXES_POSITION) -> np.ndarray:
    """Jacobienne de position seule. Voir `position_et_jacobienne`."""
    return position_et_jacobienne(angles, axes)[1]


def _resoudre(J, deplacement, axes, amortissement):
    """dq = J^T (J J^T + lambda^2 I)^-1 . dx, replacé sur les 5 articulations.

    La matrice à inverser est 3 x 3 quel que soit le nombre d'articulations :
    c'est ce qui rend l'appel négligeable en temps de calcul.
    """
    carre = J @ J.T + (amortissement ** 2) * np.eye(3)
    dq_axes = J.T @ np.linalg.solve(carre, np.asarray(deplacement, dtype=float))

    dq = np.zeros(len(_MOBILES))
    for valeur, j in zip(dq_axes, axes):
        dq[j] = valeur
    return dq


def increment(angles, deplacement, axes=AXES_POSITION,
              amortissement: float = AMORTISSEMENT) -> np.ndarray:
    """Variation articulaire (rad) produisant le déplacement cartésien voulu.

    Retourne un vecteur de 5 valeurs (une par articulation du bras), nul sur
    les axes non pilotés, directement additionnable aux angles courants.
    """
    return _resoudre(jacobienne(angles, axes), deplacement, axes, amortissement)


def suivre(angles, cible, axes=AXES_POSITION, gain: float = 1.0,
           pas_max_rad: float = np.radians(4.0),
           amortissement: float = AMORTISSEMENT):
    """Un pas d'asservissement vers `cible`. Retourne (nouveaux angles, erreur m).

    `gain` sous 1 amortit la réponse — utile quand la cible est bruitée, ce qui
    est toujours le cas d'une cible issue de la vision. `pas_max_rad` borne la
    consigne d'un cycle, dernier garde-fou contre une cible aberrante.
    """
    angles = np.asarray(angles, dtype=float)
    p, J = position_et_jacobienne(angles, axes)
    ecart = np.asarray(cible, dtype=float) - p
    dq = _resoudre(J, ecart * gain, axes, amortissement)

    plus_grand = np.abs(dq).max()
    if plus_grand > pas_max_rad:
        dq *= pas_max_rad / plus_grand
    return angles + dq, float(np.linalg.norm(ecart))


if __name__ == "__main__":
    import time

    from cinematique.ik import angles_pour, position_de, vecteur_complet

    rng = np.random.default_rng(0)

    print("=== La cinématique directe rapide dit-elle comme ikpy ? ===")
    ecarts = []
    for _ in range(200):
        a = rng.uniform(-1.2, 1.2, 5)
        ecarts.append(np.abs(position(a) - position_de(vecteur_complet(a))).max())
    print(f"  écart maximal : {max(ecarts) * 1e9:.2f} nm\n")

    print("=== La jacobienne dit-elle comme les différences finies ? ===")
    pires = []
    for _ in range(200):
        a = rng.uniform(-1.2, 1.2, 5)
        J = jacobienne(a)
        h = 1e-7
        for colonne, j in enumerate(AXES_POSITION):
            decale = a.copy()
            decale[j] += h
            numerique = (position(decale) - position(a)) / h
            pires.append(np.abs(J[:, colonne] - numerique).max())
    print(f"  écart maximal : {max(pires) * 1e6:.3f} µm/rad\n")

    print("=== Combien de temps, comparé à l'IK complète ? ===")
    a = rng.uniform(-0.8, 0.8, 5)
    cible = position(a) + np.array([0.004, -0.003, 0.002])

    debut = time.perf_counter()
    for _ in range(2000):
        suivre(a, cible)
    rapide = (time.perf_counter() - debut) / 2000

    debut = time.perf_counter()
    for _ in range(20):
        angles_pour(*cible, angles_initiaux=vecteur_complet(a))
    lent = (time.perf_counter() - debut) / 20

    print(f"  différentielle : {rapide * 1e6:8.1f} µs")
    print(f"  ik.angles_pour : {lent * 1e6:8.1f} µs")
    print(f"  rapport        : {lent / rapide:8.0f} x plus rapide\n")

    print("=== Converge-t-elle sur une cible à quelques centimètres ? ===")
    a = rng.uniform(-0.5, 0.5, 5)
    cible = position(a) + np.array([0.03, 0.02, -0.02])
    for iteration in range(1, 41):
        a, erreur = suivre(a, cible)
        if iteration in (1, 2, 5, 10, 20, 40):
            print(f"  itération {iteration:>2} : erreur {erreur * 1000:7.3f} mm")

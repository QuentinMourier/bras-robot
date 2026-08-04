#!/usr/bin/env python3
"""Enveloppe réellement atteignable par la pince, compte tenu des butées mesurées.

L'URDF déclare des butées beaucoup plus larges que celles du bras monté : s'y
fier donnerait une enveloppe fantaisiste. On échantillonne donc au hasard dans
`etalonnage.LIMITES_DEG` — les butées relevées à la main — et on regarde où le
point de préhension peut se trouver.

L'échantillonnage se fait en cinématique *directe*, pas inverse : c'est exact
(pas de question de convergence) et assez rapide pour brasser 200 000 poses,
la chaîne étant recalculée d'un coup en numpy plutôt qu'une pose à la fois.

    venv/bin/python -m cinematique.enveloppe
    venv/bin/python -m cinematique.enveloppe --hauteur 0.05 --n 300000

ATTENTION : « atteignable » ne veut pas dire « sans danger ». Ce calcul ignore
la table, le support, les câbles, les objets posés autour et les collisions du
bras avec lui-même. Il donne la limite haute de ce qui est possible, pas une
autorisation.
"""

import argparse
import sys

import numpy as np

from cinematique.ik import bras
from materiel.etalonnage import AXES_BRAS, LIMITES_DEG, servo_vers_urdf


def _transformations_fixes():
    """Matrices 4x4 de chaque liaison, hors rotation de l'articulation."""
    fixes = []
    for lien in bras.links:
        matrice = np.eye(4)
        # Le lien d'origine n'a ni translation ni orientation : il vaut l'identité.
        matrice[:3, :3] = _rotation_rpy(
            getattr(lien, "origin_orientation", np.zeros(3)))
        matrice[:3, 3] = getattr(lien, "origin_translation", np.zeros(3))
        fixes.append(matrice)
    return np.array(fixes)


def _rotation_rpy(rpy):
    """Convention URDF : rotation fixe X puis Y puis Z (roulis, tangage, lacet)."""
    r, p, y = rpy
    cr, sr, cp, sp, cy, sy = (np.cos(r), np.sin(r), np.cos(p),
                              np.sin(p), np.cos(y), np.sin(y))
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp,     cp * sr,                cp * cr],
    ])


def positions_pince(angles):
    """Cinématique directe vectorisée. angles : (n, 5) en radians -> (n, 3) en m."""
    fixes = _transformations_fixes()
    n = len(angles)
    courant = np.tile(np.eye(4), (n, 1, 1))

    # Les liens 0 et 6 sont fixes ; les 5 autres tournent autour de leur axe z.
    for indice, lien in enumerate(bras.links):
        etape = np.tile(fixes[indice], (n, 1, 1))
        if lien.joint_type != "fixed":
            theta = angles[:, indice - 1]
            c, s = np.cos(theta), np.sin(theta)
            rotation = np.tile(np.eye(4), (n, 1, 1))
            rotation[:, 0, 0], rotation[:, 0, 1] = c, -s
            rotation[:, 1, 0], rotation[:, 1, 1] = s, c
            etape = etape @ rotation
        courant = courant @ etape
    return courant[:, :3, 3]


def _verifier_contre_ikpy(generateur, n=20):
    """Compare la cinématique vectorisée à celle d'ikpy, qui fait référence.

    La version numpy réimplémente la chaîne pour aller vite ; autant s'assurer
    qu'elle dit exactement la même chose que celle qui pilote le bras.
    """
    angles = generateur.uniform(-1.0, 1.0, (n, len(AXES_BRAS)))
    rapide = positions_pince(angles)
    lent = np.array([bras.forward_kinematics(np.array([0.0, *a, 0.0]))[:3, 3]
                     for a in angles])
    return np.abs(rapide - lent).max()


def echantillonner(n, generateur):
    """Tire n poses au hasard dans les butées mesurées, et retourne les positions."""
    tirage = np.empty((n, len(AXES_BRAS)))
    for colonne, sid in enumerate(AXES_BRAS):
        bas, haut = LIMITES_DEG[sid]
        degres = generateur.uniform(bas, haut, n)
        tirage[:, colonne] = [servo_vers_urdf(sid, d) for d in degres]
    return positions_pince(tirage)


def carte(points, axe_horizontal, axe_vertical, tranche, epaisseur,
          largeur=62, hauteur=24):
    """Dessine en caractères la coupe des points dont l'axe restant est dans la tranche."""
    axe_restant = ({0, 1, 2} - {axe_horizontal, axe_vertical}).pop()
    garde = np.abs(points[:, axe_restant] - tranche) <= epaisseur
    retenus = points[garde]
    if not len(retenus):
        print("  (aucun point dans cette tranche)")
        return

    h, v = retenus[:, axe_horizontal], retenus[:, axe_vertical]
    hmin, hmax = h.min(), h.max()
    vmin, vmax = v.min(), v.max()
    grille = np.zeros((hauteur, largeur), dtype=int)
    ih = ((h - hmin) / max(hmax - hmin, 1e-9) * (largeur - 1)).astype(int)
    iv = ((v - vmin) / max(vmax - vmin, 1e-9) * (hauteur - 1)).astype(int)
    np.add.at(grille, (hauteur - 1 - iv, ih), 1)

    seuil = max(1, int(np.percentile(grille[grille > 0], 40)))
    noms = "xyz"
    for ligne_index, ligne in enumerate(grille):
        valeur = vmax - (vmax - vmin) * ligne_index / (hauteur - 1)
        dessin = "".join("#" if c > seuil else ("." if c else " ") for c in ligne)
        print(f"  {valeur:+6.2f} |{dessin}|")
    print(f"  {'':6s} +{'-' * largeur}+")
    print(f"  {'':6s}  {hmin:+.2f}{noms[axe_horizontal]:^{largeur - 10}}{hmax:+.2f}")


def carte_acceptation(z, portee=0.45, pas=0.05):
    """Carte des cibles que `BrasServos.aller_a` accepterait à cette hauteur.

    Rejoue exactement ce que fait le back-end : convergence de l'IK, puis
    conversion en degrés servo et contrôle des butées mesurées. Plus sévère
    que l'enveloppe atteignable, et c'est bien le but.
    """
    from cinematique.ik import angles_utiles
    from cinematique.securite import ik_est_valide
    from materiel.etalonnage import angles_vers_degres, dans_les_limites

    valeurs = np.arange(-portee, portee + pas / 2, pas)
    print(f"  {'':7s}" + "".join(f"{x:+5.2f}" for x in valeurs))
    for y in valeurs[::-1]:
        cases = []
        for x in valeurs:
            ok, solution, _, _ = ik_est_valide(x, y, z)
            if not ok:
                cases.append("  .  ")
                continue
            degres = angles_vers_degres(angles_utiles(solution))
            dedans = all(dans_les_limites(sid, d) for sid, d in degres.items())
            cases.append("  O  " if dedans else "  x  ")
        print(f"  {y:+6.2f} " + "".join(cases))
    print("\n  O = accepté     x = hors butées mesurées     . = IK ne converge pas")


def main():
    parseur = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parseur.add_argument("--n", type=int, default=200_000,
                         help="nombre de poses tirées (défaut 200000)")
    parseur.add_argument("--hauteur", type=float, default=0.05,
                         help="hauteur de la coupe horizontale, en m (défaut 0.05)")
    parseur.add_argument("--epaisseur", type=float, default=0.02,
                         help="demi-épaisseur des coupes, en m (défaut 0.02)")
    parseur.add_argument("--acceptation", action="store_true",
                         help="carte des cibles acceptées par le back-end "
                              "à la hauteur --hauteur (lent : IK par point)")
    args = parseur.parse_args()

    if args.acceptation:
        print(f"Cibles acceptées par aller_a() à z = {args.hauteur:.2f} m :\n")
        carte_acceptation(args.hauteur)
        return 0

    generateur = np.random.default_rng(0)
    ecart = _verifier_contre_ikpy(generateur)
    if ecart > 1e-9:
        print(f"ANOMALIE : la cinématique vectorisée s'écarte d'ikpy de "
              f"{ecart * 1000:.3f} mm. Résultats non fiables.")
        return 1

    points = echantillonner(args.n, generateur)
    rayon = np.hypot(points[:, 0], points[:, 1])

    print(f"{args.n} poses tirées dans les butées mesurées "
          f"(cinématique vérifiée contre ikpy à {ecart * 1e6:.1f} nm près).\n")
    print("Enveloppe globale du point de préhension (m) :")
    for indice, nom in enumerate("xyz"):
        print(f"  {nom} : {points[:, indice].min():+.3f} à "
              f"{points[:, indice].max():+.3f}")
    print(f"  distance à l'axe de la base : {rayon.min():.3f} à {rayon.max():.3f}")

    print(f"\nCoupe verticale, tranche |y| <= {args.epaisseur:.2f} m "
          "(profil du bras vu de côté) :")
    carte(points, 0, 2, 0.0, args.epaisseur)

    print(f"\nCoupe horizontale à z = {args.hauteur:.2f} m "
          f"+/- {args.epaisseur:.2f} (vue de dessus) :")
    carte(points, 0, 1, args.hauteur, args.epaisseur)

    print("\nRappel : ce calcul ignore la table, le support, les câbles et les")
    print("collisions du bras avec lui-même. Vérifiez toujours une cible avec")
    print("`test_bras.py essai x y z`, qui ne bouge pas, avant `aller`.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Banc d'essai du back-end `BrasServos` : le bras piloté en coordonnées.

Là où `test_servos.py` parle aux servos un par un, en degrés, cet outil-ci
travaille dans le repère cartésien de l'URDF, en passant par la cinématique
inverse et par toutes les sécurités de materiel/servos.py.

À lancer depuis la racine du dépôt, avec le venv (ikpy y est installé) :

    venv/bin/python -m materiel.test_bras etat
    venv/bin/python -m materiel.test_bras essai 0.20 0.05 0.15   # sans bouger
    venv/bin/python -m materiel.test_bras monter                 # vérification
    venv/bin/python -m materiel.test_bras aller 0.20 0.05 0.15
    venv/bin/python -m materiel.test_bras pince ouvrir

Toutes ces commandes acceptent --simu, qui pilote le bras simulé de
materiel/simulation.py au lieu des servos : aucun matériel n'est requis, et les
mêmes sécurités s'appliquent. Les options générales se placent avant la
commande :

    venv/bin/python -m materiel.test_bras --simu etat
    venv/bin/python -m materiel.test_bras --simu --vue trajet.png aller 0.30 0.05 0.15
    venv/bin/python -m materiel.test_bras --simu --temps-reel aller 0.30 0.05 0.15

Attention : la simulation ne connaît ni la gravité ni les contacts. Elle valide
la géométrie et la logique, pas la sûreté physique — voir l'en-tête de
materiel/simulation.py.

Pour arrêter un mouvement en cours, depuis n'importe où :

    ssh pi touch /tmp/bras-stop     # fige le bras, couple maintenu
    ssh pi rm /tmp/bras-stop        # réarme

Le repère : origine au centre de la base, z vers le haut, distances en mètres.
L'orientation de x et y dans le monde réel se déduit de la commande `etat`,
qui affiche où le modèle croit que se trouve la pince.
"""

import argparse
import sys

import numpy as np

from cinematique.ik import angles_utiles, vecteur_complet
from cinematique.securite import ik_est_valide
from materiel.etalonnage import (AXES_BRAS, AXE_PINCE, LIMITES_DEG,
                                 NOMS_ARM101, SENS, SENS_LATERAUX_VERIFIES,
                                 SENS_VERIFIES, ZERO_DEG, angles_vers_degres,
                                 limiter)
from materiel.servos import (BAUD_DEFAUT, ECART_MAX_DEG, PORT_DEFAUT,
                             BrasServos)
from materiel.simulation import BrasSimule
from materiel.sts3215 import ErreurServo


def cmd_etat(bras, args):
    """Où se trouve le bras, en degrés servo, en angles URDF et en (x, y, z)."""
    degres = bras.degres_servo()
    angles = bras.lire_positions()

    print(f"{'id':>2}  {'nom':<14} {'servo':>9} {'urdf':>9} {'zéro':>8} "
          f"{'sens':>5} {'charge':>7}")
    for sid in AXES_BRAS:
        print(f"{sid:>2}  {NOMS_ARM101[sid]:<14} {degres[sid]:>+8.1f}° "
              f"{np.degrees(angles[sid - 1]):>+8.1f}° {ZERO_DEG[sid]:>+7.1f}° "
              f"{SENS[sid]:>+5d} {bras.charge(sid):>+7d}")
    print(f"{AXE_PINCE:>2}  {NOMS_ARM101[AXE_PINCE]:<14} "
          f"{degres[AXE_PINCE]:>+8.1f}° {'-':>9} {'-':>8} {'-':>5} "
          f"{bras.charge(AXE_PINCE):>+7d}")

    x, y, z = bras.position_pince()
    print(f"\nPoint de préhension : x {x:+.3f}  y {y:+.3f}  z {z:+.3f}  (m)")
    print(f"  distance à l'axe de la base : {np.hypot(x, y) * 100:.1f} cm")
    print(f"  hauteur au-dessus de la base : {z * 100:.1f} cm")
    if not SENS_VERIFIES:
        print("\nSens de rotation non vérifiés (etalonnage.SENS_VERIFIES) : "
              "cette position est calculée, pas confirmée.\n"
              "Lancez `monter` pour la vérifier par un petit mouvement.")
    elif not SENS_LATERAUX_VERIFIES:
        print("\nHauteur et forme du bras vérifiées ; en revanche le sens des "
              "axes 1 (base) et 5 (roulis) ne l'est pas.\nSi l'un des deux est "
              "inversé, y est à lire au signe près.")
    return 0


def cmd_essai(bras, args):
    """Résout l'IK pour une cible et montre ce qui serait envoyé, sans bouger."""
    depart_rad = bras.lire_positions()
    depart_deg = bras.degres_servo(AXES_BRAS)

    ok, solution, erreur_mm, message = ik_est_valide(
        args.x, args.y, args.z, angles_initiaux=vecteur_complet(depart_rad))
    print(f"Cible ({args.x:+.3f}, {args.y:+.3f}, {args.z:+.3f}) m — "
          f"résidu {erreur_mm:.1f} mm")
    if not ok:
        print(f"REFUSÉ : {message}")
        return 1

    cibles = angles_vers_degres(angles_utiles(solution))
    print(f"\n{'id':>2}  {'nom':<14} {'départ':>9} {'cible':>9} {'écart':>9}"
          f"   butées")
    refus = []
    for sid in AXES_BRAS:
        _, bride = limiter(sid, cibles[sid])
        bas, haut = LIMITES_DEG[sid]
        marque = "  HORS BUTÉES" if bride else ""
        ecart = cibles[sid] - depart_deg[sid]
        if abs(ecart) > ECART_MAX_DEG:
            marque += f"  ÉCART > {ECART_MAX_DEG:.0f}°"
        if marque:
            refus.append(sid)
        print(f"{sid:>2}  {NOMS_ARM101[sid]:<14} {depart_deg[sid]:>+8.1f}° "
              f"{cibles[sid]:>+8.1f}° {ecart:>+8.1f}°   "
              f"[{bas:+.0f}, {haut:+.0f}]{marque}")

    if refus:
        print("\nCe mouvement serait refusé par le back-end.")
        return 1
    print("\nMouvement recevable. `aller` l'exécuterait.")
    return 0


def cmd_aller(bras, args):
    """Déplace le point de préhension vers une position donnée.

    Par la boucle d'asservissement : mouvement continu, sans les paliers
    suivis d'attente de l'ancien chemin, visiblement saccadés sur le bras réel.
    """
    from materiel.approche import aller_asservi
    from materiel.asservissement import asservissement_pour

    with asservissement_pour(bras) as asser:
        if not aller_asservi(asser, args.x, args.y, args.z):
            return 1
    x, y, z = bras.position_pince()
    ecart_mm = np.linalg.norm(np.array([x, y, z]) -
                              np.array([args.x, args.y, args.z])) * 1000
    print(f"Position atteinte : ({x:+.3f}, {y:+.3f}, {z:+.3f}) m, "
          f"à {ecart_mm:.1f} mm de la cible.")
    return 0


def cmd_monter(bras, args):
    """Vérifie l'étalonnage en levant la pince de quelques centimètres.

    C'est le contrôle qui manque avant de faire confiance au modèle : un
    déplacement purement vertical ne dépend d'aucune convention d'orientation,
    donc l'observation est sans ambiguïté — la pince monte, ou le modèle est
    faux. Si un sens de rotation était inversé, le bras partirait ailleurs (et
    la surveillance de charge l'arrêterait s'il poussait sur la table).
    """
    depart = bras.position_pince()
    cible = depart + np.array([0.0, 0.0, args.cm / 100])
    print(f"Pince en ({depart[0]:+.3f}, {depart[1]:+.3f}, {depart[2]:+.3f}) m "
          f"d'après le modèle.")
    print(f"Consigne : la même chose {args.cm:.0f} cm plus haut, "
          "sans rien changer d'autre.\n")

    from materiel.approche import aller_asservi
    from materiel.asservissement import asservissement_pour

    with asservissement_pour(bras) as asser:
        if not aller_asservi(asser, *cible):
            print("\nMouvement refusé ou interrompu — voir le message "
                  "ci-dessus. L'étalonnage reste à vérifier.")
            return 1

    arrivee = bras.position_pince()
    print(f"\nModèle : la pince est montée de "
          f"{(arrivee[2] - depart[2]) * 100:.1f} cm.")
    print("Regardez le bras :")
    print(f"  - si la pince s'est bien levée d'environ {args.cm:.0f} cm sans "
          "partir de côté,\n    l'étalonnage est bon : passez SENS_VERIFIES à "
          "True dans materiel/etalonnage.py ;")
    print("  - sinon, notez ce qu'elle a fait (montée à l'envers, rotation de "
          "la base,\n    translation latérale) : c'est le sens de l'axe "
          "correspondant qui est faux.")
    return 0


def cmd_pince(bras, args):
    """Ouvre ou ferme la pince."""
    if args.action == "ouvrir":
        bras.ouvrir_pince()
    else:
        bras.fermer_pince()
    return 0


def construire_parseur():
    parseur = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parseur.add_argument("--port", default=PORT_DEFAUT)
    parseur.add_argument("--baud", type=int, default=BAUD_DEFAUT)
    parseur.add_argument("--vitesse", type=int, default=None,
                         help="vitesse des servos en pas/s")
    parseur.add_argument("--simu", action="store_true",
                         help="pilote le bras simulé au lieu du bras réel "
                              "(aucun matériel requis)")
    parseur.add_argument("--temps-reel", action="store_true",
                         help="avec --simu : fait durer le mouvement aussi "
                              "longtemps que sur le bras réel")
    parseur.add_argument("--vue", metavar="FICHIER.png", default=None,
                         help="avec --simu : écrit le tracé 3D du trajet")
    sous = parseur.add_subparsers(dest="commande", required=True)

    p = sous.add_parser("etat", help="position actuelle, servo et cartésienne")
    p.set_defaults(fonction=cmd_etat, bouge=False)

    p = sous.add_parser("essai", help="résout l'IK d'une cible sans bouger")
    p.add_argument("x", type=float)
    p.add_argument("y", type=float)
    p.add_argument("z", type=float)
    p.set_defaults(fonction=cmd_essai, bouge=False)

    p = sous.add_parser("aller", help="déplace la pince vers (x, y, z)")
    p.add_argument("x", type=float)
    p.add_argument("y", type=float)
    p.add_argument("z", type=float)
    p.set_defaults(fonction=cmd_aller, bouge=True)

    p = sous.add_parser("monter", help="vérifie l'étalonnage en levant la pince")
    p.add_argument("--cm", type=float, default=3.0,
                   help="hauteur du mouvement de contrôle (défaut 3 cm)")
    p.set_defaults(fonction=cmd_monter, bouge=True)

    p = sous.add_parser("pince", help="ouvre ou ferme la pince")
    p.add_argument("action", choices=["ouvrir", "fermer"])
    p.set_defaults(fonction=cmd_pince, bouge=True)

    return parseur


def main():
    args = construire_parseur().parse_args()

    options = {}
    if args.vitesse is not None:
        options["vitesse"] = args.vitesse
    # `monter` est justement la commande qui sert à vérifier les sens : elle a
    # le droit de bouger avant qu'ils ne le soient. Les lectures aussi, elles
    # ne commandent rien.
    if args.commande in ("monter", "etat", "essai"):
        options["accepter_sens_non_verifies"] = True

    if args.simu:
        constructeur = lambda: BrasSimule(temps_reel=args.temps_reel, **options)
    else:
        constructeur = lambda: BrasServos(args.port, args.baud, **options)

    try:
        with constructeur() as bras:
            try:
                code = args.fonction(bras, args)
            except KeyboardInterrupt:
                # Un Ctrl-C en plein mouvement laisserait les servos rejoindre
                # leur dernière consigne : on les fige explicitement.
                bras.stop()
                return 130
            if args.simu and args.vue:
                chemin = bras.tracer(args.vue)
                print(f"\nTracé du trajet ({len(bras.trajectoire)} poses) "
                      f"écrit dans {chemin}")
            return code
    except RuntimeError as erreur:
        print(erreur)
        return 1
    except ErreurServo as erreur:
        print(f"Erreur de communication : {erreur}")
        return 1
    except OSError as erreur:
        print(f"Port {args.port} inaccessible : {erreur}")
        return 1
    except KeyboardInterrupt:
        print("\nInterrompu.")
        return 130


if __name__ == "__main__":
    sys.exit(main())

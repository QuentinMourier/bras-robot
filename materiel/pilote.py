#!/usr/bin/env python3
"""Pilotage interactif du bout de la pince, au clavier (« jog » cartésien).

On ne commande pas des angles : on déplace une CIBLE dans l'espace, et la
cinématique inverse se charge de trouver les angles qui l'atteignent. Une
touche = un pas de quelques centimètres.

    venv/bin/python -m materiel.pilote --simu     # sans matériel, sans risque
    venv/bin/python -m materiel.pilote            # bras réel

Commandes :

      flèche haut / bas    avancer / reculer      (x)
      flèche gauche/droite aller à gauche/droite  (y)
      m / d                monter / descendre     (z)
      o / f                ouvrir / fermer la pince
      + / -                pas plus grand / plus petit
      r                    revenir à la pose de départ
      p                    afficher la position courante
      q                    quitter

La cible ne bouge que si le mouvement réussit : un refus (hors butées, IK qui
ne converge pas, obstacle) la laisse où elle était, pour que l'affichage reste
le reflet de la réalité.

Sécurité, sur le bras réel :
  - le pas vaut 1 cm par défaut, donc chaque mouvement est petit ;
  - toutes les protections de BrasServos s'appliquent (butées mesurées,
    paliers de 10°, surveillance de charge) ;
  - `touch /tmp/bras-stop` depuis un autre terminal fige tout ;
  - Ctrl-C fige le bras sur place.

À FAIRE UNE FOIS avec cet outil : vérifier le sens latéral. Appuyez sur la
flèche gauche et regardez de quel côté part réellement le bras. Si la flèche
« gauche » l'envoie à droite, c'est que le sens de l'axe 1 est inversé — voir
etalonnage.SENS_LATERAUX_VERIFIES, encore à False à ce jour.
"""

import argparse
import sys
import termios
import tty

import numpy as np

from materiel.etalonnage import SENS_LATERAUX_VERIFIES
from materiel.servos import BAUD_DEFAUT, PORT_DEFAUT, BrasServos
from materiel.simulation import BrasSimule
from materiel.sts3215 import ErreurServo

PAS_DEFAUT = 0.01      # m, soit 1 cm : petit, donc sûr
PAS_MIN = 0.002
PAS_MAX = 0.05

# Touche -> déplacement (dx, dy, dz) en multiples du pas courant.
DEPLACEMENTS = {
    "HAUT":   (+1, 0, 0),
    "BAS":    (-1, 0, 0),
    "GAUCHE": (0, +1, 0),
    "DROITE": (0, -1, 0),
    "m":      (0, 0, +1),
    "d":      (0, 0, -1),
}

FLECHES = {"[A": "HAUT", "[B": "BAS", "[C": "DROITE", "[D": "GAUCHE"}


def lire_touche():
    """Lit une touche au vol (sans attendre Entrée), flèches comprises.

    Les flèches n'envoient pas un caractère mais une séquence d'échappement de
    trois octets (ESC, '[', puis une lettre) : il faut donc lire la suite quand
    le premier octet est un ESC.
    """
    premier = sys.stdin.read(1)
    if premier != "\x1b":
        return premier
    suite = sys.stdin.read(2)
    return FLECHES.get(suite, "ESC")


def afficher_aide():
    print("""
    flèches ↑↓    avancer / reculer      (x)
    flèches ←→    gauche / droite        (y)
    m / d         monter / descendre     (z)
    o / f         ouvrir / fermer la pince
    + / -         régler le pas
    r             revenir à la pose de départ
    p             position courante
    q             quitter
""")


def piloter(bras, pas: float, simu: bool) -> None:
    """Boucle de pilotage. Le terminal est déjà en mode caractère par caractère."""
    depart = np.array(bras.position_pince(), dtype=float)
    cible = depart.copy()

    print(f"Pince à ({cible[0]:+.3f}, {cible[1]:+.3f}, {cible[2]:+.3f}) m — "
          f"pas de {pas * 100:.1f} cm")
    if not simu and not SENS_LATERAUX_VERIFIES:
        print("\n  /!\\  Le sens de l'axe 1 n'est pas vérifié : la flèche "
              "« gauche » pourrait\n       envoyer le bras à droite. Regardez "
              "le bras au premier mouvement latéral.")
    afficher_aide()

    while True:
        touche = lire_touche()

        # Chaîne vide = fin de l'entrée (tuyau épuisé) : on quitte proprement.
        if touche in ("", "q", "ESC", "\x03"):
            print("\nArrêt du pilotage. Le bras garde sa position.")
            return

        if touche == "p":
            reelle = bras.position_pince()
            print(f"  cible  ({cible[0]:+.3f}, {cible[1]:+.3f}, {cible[2]:+.3f}) m")
            print(f"  réelle ({reelle[0]:+.3f}, {reelle[1]:+.3f}, "
                  f"{reelle[2]:+.3f}) m")
            continue

        if touche in ("+", "-"):
            pas = min(PAS_MAX, pas * 1.5) if touche == "+" \
                else max(PAS_MIN, pas / 1.5)
            print(f"  pas = {pas * 100:.1f} cm")
            continue

        if touche == "o":
            bras.ouvrir_pince()
            print("  pince ouverte")
            continue

        if touche == "f":
            bras.fermer_pince()
            tenue = " — objet tenu" if bras.pince_tient() else ""
            print(f"  pince fermée{tenue}")
            continue

        if touche == "r":
            if bras.aller_a(*depart):
                cible = depart.copy()
                print(f"  retour au départ ({cible[0]:+.3f}, {cible[1]:+.3f}, "
                      f"{cible[2]:+.3f}) m")
            else:
                print("  retour refusé — approchez en plusieurs pas")
            continue

        if touche not in DEPLACEMENTS:
            continue

        demande = cible + np.array(DEPLACEMENTS[touche], dtype=float) * pas
        if bras.aller_a(*demande):
            cible = demande
            print(f"  ({cible[0]:+.3f}, {cible[1]:+.3f}, {cible[2]:+.3f}) m")
        else:
            # La cible reste où elle était : l'affichage ne doit jamais
            # prétendre que le bras est allé quelque part où il n'est pas.
            print(f"  refusé -> ({demande[0]:+.3f}, {demande[1]:+.3f}, "
                  f"{demande[2]:+.3f}) m  (cible inchangée)")


def main():
    parseur = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parseur.add_argument("--simu", action="store_true",
                         help="pilote le bras simulé (aucun matériel requis)")
    parseur.add_argument("--pas", type=float, default=PAS_DEFAUT * 100,
                         help="pas initial en centimètres (défaut 1)")
    parseur.add_argument("--port", default=PORT_DEFAUT)
    parseur.add_argument("--baud", type=int, default=BAUD_DEFAUT)
    parseur.add_argument("--vue", metavar="FICHIER.png", default=None,
                         help="avec --simu : trace le trajet en quittant")
    args = parseur.parse_args()

    try:
        bras = BrasSimule(bavard=False) if args.simu \
            else BrasServos(args.port, args.baud, bavard=False)
    except RuntimeError as erreur:
        print(erreur)
        return 1
    except ErreurServo as erreur:
        print(f"Erreur de communication : {erreur}")
        return 1
    except OSError as erreur:
        print(f"Port {args.port} inaccessible : {erreur}")
        return 1

    print("=== Pilotage du bout de la pince ===  "
          + ("SIMULATION" if args.simu else "BRAS RÉEL"))
    if not args.simu:
        print("Arrêt d'urgence : touch /tmp/bras-stop  (depuis un autre terminal)")

    # Mode « caractère par caractère » : sans cela le terminal attendrait
    # Entrée avant de livrer la moindre touche. Ne s'applique qu'à un vrai
    # terminal — l'entrée peut être un tuyau, pour rejouer une séquence de
    # touches en simulation (utile pour tester sans matériel).
    interactif = sys.stdin.isatty()
    descripteur = sys.stdin.fileno()
    reglages = termios.tcgetattr(descripteur) if interactif else None
    try:
        if interactif:
            tty.setcbreak(descripteur)
        piloter(bras, args.pas / 100, args.simu)
    except KeyboardInterrupt:
        bras.stop()
        print("\nInterrompu : le bras est figé sur place.")
        return 130
    finally:
        if interactif:
            termios.tcsetattr(descripteur, termios.TCSADRAIN, reglages)
        if args.simu and args.vue:
            print(f"Tracé écrit dans {bras.tracer(args.vue)}")
        bras.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())

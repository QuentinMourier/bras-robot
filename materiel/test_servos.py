#!/usr/bin/env python3
"""Banc de test des servos STS3215, à utiliser avant le montage du bras.

Un servo neuf sort d'usine avec l'identifiant 1 : il faut donc les brancher
UN PAR UN pour leur attribuer les identifiants 1 à 6 du SO-ARM101, de la base
vers la pince :

    1 shoulder_pan   2 shoulder_lift   3 elbow_flex
    4 wrist_flex     5 wrist_roll      6 gripper

Exemples :
    python3 test_servos.py scan
    python3 test_servos.py info 1
    python3 test_servos.py id 1 3          # renomme le servo 1 en 3
    python3 test_servos.py prep 3          # remet à zéro et centre avant montage
    python3 test_servos.py pos 3 -45       # va à -45° par rapport au centre
    python3 test_servos.py test 3          # aller-retour doux + relevé
    python3 test_servos.py libre 3         # coupe le couple (servo mobile à la main)
"""

import argparse
import sys
import time

from sts3215 import (BAUDS, CENTRE, ADR_ANGLE_MAX, ADR_ANGLE_MIN, ADR_MODE,
                     ADR_OFFSET, BusServos, ErreurServo, degres_en_pas,
                     pas_en_degres)

PORT_DEFAUT = "/dev/ttyACM0"
BAUD_DEFAUT = 1_000_000

NOMS_ARM101 = {1: "shoulder_pan", 2: "shoulder_lift", 3: "elbow_flex",
               4: "wrist_flex", 5: "wrist_roll", 6: "gripper"}
MODES = {0: "position", 1: "roue libre", 2: "PWM", 3: "pas à pas"}

# Plages autorisées, en degrés servo autour du centre (2048 pas).
#
# Les amplitudes viennent de urdf/so101_new_calib.urdf. Attention : le signe du
# servo ne correspond pas forcément au signe URDF. Pour les axes 2, 3 et 4 la
# plage est symétrique, donc la question ne se pose pas. Pour la base, la sonde
# du 1er août a montré que le sens positif va vers la pince du support : on la
# bride donc à +20° tant que le bras est monté sur ce support temporaire.
# La pince (axe 6) est asymétrique et son sens n'est pas encore vérifié : sa
# plage reste volontairement étroite en attendant.
LIMITES_DEG = {
    1: (-110.0, 20.0),    # design ±110°, bridé côté pince du support
    2: (-100.0, 100.0),
    3: (-96.8, 96.8),
    4: (-95.0, 95.0),
    5: (-157.2, 157.2),
    6: (-100.0, 10.0),    # sens présumé inversé par rapport à l'URDF
}

CHARGE_MAX = 400  # pour-mille du couple maxi : seuil d'arrêt sur obstacle
PAS_INCREMENT = 10.0  # amplitude d'un palier, en degrés


def limiter(sid, degres):
    """Ramène un angle dans la plage autorisée. Retourne (angle, bridé)."""
    if sid not in LIMITES_DEG:
        return degres, False
    bas, haut = LIMITES_DEG[sid]
    borne = max(bas, min(haut, degres))
    return borne, borne != degres


def cmd_scan(bus, args):
    """Liste les servos présents sur le bus."""
    trouves = bus.scanner()
    if not trouves:
        print("Aucun servo ne répond.")
        print("  - alimentation 12 V branchée et allumée ?")
        print("  - câble du servo bien enfoncé côté carte et côté servo ?")
        print(f"  - bonne vitesse de transmission ? (essai à {args.baud} bauds,")
        print("    relancez avec --tous-bauds pour balayer les autres)")
        return 1
    for sid in trouves:
        nom = NOMS_ARM101.get(sid, "")
        etiquette = f"  ({nom})" if nom else ""
        print(f"servo id={sid}{etiquette}")
    if len(trouves) > 1:
        print(f"\n{len(trouves)} servos sur le bus.")
    return 0


def cmd_scan_tous_bauds(args):
    """Balaye toutes les vitesses de transmission connues."""
    for index, baud in sorted(BAUDS.items(), key=lambda kv: -kv[1]):
        try:
            with BusServos(args.port, baud) as bus:
                trouves = bus.scanner()
        except OSError as erreur:
            print(f"{baud:>9} bauds : port inaccessible ({erreur})")
            continue
        if trouves:
            print(f"{baud:>9} bauds : servos {trouves}  <-- registre baud = {index}")
        else:
            print(f"{baud:>9} bauds : rien")
    return 0


def cmd_bilan(bus, args):
    """Tableau récapitulatif de toute la chaîne, en lecture seule."""
    presents = bus.scanner()
    if not presents:
        print("Aucun servo ne répond.")
        return 1

    print(f"{'id':>2}  {'nom':<14} {'position':>9} {'angle':>8} {'tension':>8} "
          f"{'temp':>5}  {'mode':<10} {'butées':>10} {'offset':>7}  couple")
    anomalies = []
    for sid in presents:
        etat = bus.etat(sid)
        butees = f"{etat['angle_min']}-{etat['angle_max']}"
        print(f"{sid:>2}  {NOMS_ARM101.get(sid, '?'):<14} {etat['position']:>9} "
              f"{pas_en_degres(etat['position']):>+7.1f}° {etat['tension']:>7.1f}V "
              f"{etat['temperature']:>4}°C  {MODES.get(etat['mode'], '?'):<10} "
              f"{butees:>10} {etat['offset']:>7}  "
              f"{'actif' if etat['couple_actif'] else 'coupé'}")

        if etat["tension"] < 9.0:
            anomalies.append(f"servo {sid} : {etat['tension']:.1f} V, trop bas")
        if etat["temperature"] > 55:
            anomalies.append(f"servo {sid} : {etat['temperature']} °C, trop chaud")
        if etat["mode"] != 0:
            anomalies.append(f"servo {sid} : mode {MODES.get(etat['mode'])}, "
                             "attendu position")

    manquants = [sid for sid in NOMS_ARM101 if sid not in presents]
    if manquants:
        noms = ", ".join(f"{sid} ({NOMS_ARM101[sid]})" for sid in manquants)
        anomalies.append(f"absents du bus : {noms}")

    if anomalies:
        print("\nAnomalies :")
        for ligne in anomalies:
            print(f"  - {ligne}")
        return 1
    print(f"\n{len(presents)} servos, aucune anomalie.")
    return 0


def cmd_info(bus, args):
    """Relevé complet d'un servo."""
    etat = bus.etat(args.id)
    nom = NOMS_ARM101.get(args.id)
    print(f"Servo {args.id}" + (f" ({nom})" if nom else ""))
    print(f"  position      {etat['position']:>5} pas "
          f"({pas_en_degres(etat['position']):+.1f}° par rapport au centre)")
    print(f"  tension       {etat['tension']:.1f} V")
    print(f"  température   {etat['temperature']} °C")
    print(f"  charge        {etat['charge']}")
    print(f"  courant       {etat['courant_mA']:.0f} mA")
    print(f"  couple        {'actif' if etat['couple_actif'] else 'coupé'}")
    print(f"  mode          {MODES.get(etat['mode'], etat['mode'])}")
    print(f"  butées        {etat['angle_min']} à {etat['angle_max']} pas")
    print(f"  offset        {etat['offset']}")
    print(f"  liaison       {etat['baud']} bauds")

    if etat["tension"] < 9.0:
        print(f"\n  ATTENTION : {etat['tension']:.1f} V, c'est bas pour un "
              "modèle 12 V. Vérifiez l'alimentation.")
    if etat["temperature"] > 55:
        print(f"\n  ATTENTION : {etat['temperature']} °C, laissez refroidir.")
    return 0


def cmd_id(bus, args):
    """Change l'identifiant d'un servo."""
    presents = bus.scanner()
    if args.ancien not in presents:
        print(f"Le servo {args.ancien} ne répond pas (présents : {presents}).")
        return 1
    if len(presents) > 1:
        print(f"Plusieurs servos sur le bus : {presents}.")
        print("Branchez-les un par un pour attribuer les identifiants, "
              "sinon les servos qui partagent un identifiant réagiront tous.")
        return 1
    if args.nouveau in presents and args.nouveau != args.ancien:
        print(f"L'identifiant {args.nouveau} est déjà pris.")
        return 1

    bus.changer_id(args.ancien, args.nouveau)
    time.sleep(0.2)
    if bus.ping(args.nouveau):
        nom = NOMS_ARM101.get(args.nouveau)
        print(f"Servo {args.ancien} -> {args.nouveau}"
              + (f" ({nom})" if nom else "") + " : OK")
        return 0
    print(f"Le servo ne répond pas à l'identifiant {args.nouveau}. "
          "Coupez puis rétablissez l'alimentation et relancez un scan.")
    return 1


def cmd_prep(bus, args):
    """Remet un servo dans l'état attendu au montage, puis le centre."""
    etat = bus.etat(args.id)
    print(f"Servo {args.id} : {etat['position']} pas, {etat['tension']:.1f} V")

    if etat["mode"] != 0:
        print(f"  mode {MODES.get(etat['mode'])} -> position")
        bus.regler_eprom(args.id, ADR_MODE, 0)
    if etat["offset"] != 0:
        print(f"  offset {etat['offset']} -> 0")
        bus.regler_eprom(args.id, ADR_OFFSET, 0, 2)
    if (etat["angle_min"], etat["angle_max"]) != (0, 4095):
        print(f"  butées {etat['angle_min']}-{etat['angle_max']} -> 0-4095")
        bus.regler_eprom(args.id, ADR_ANGLE_MIN, 0, 2)
        bus.regler_eprom(args.id, ADR_ANGLE_MAX, 4095, 2)

    print(f"  centrage sur {CENTRE} pas...")
    bus.tenir(args.id)
    bus.aller_a(args.id, CENTRE, vitesse=args.vitesse, accel=args.accel)
    atteint = bus.attendre_arret(args.id)
    ecart = atteint - CENTRE
    print(f"  position finale {atteint} pas (écart {ecart:+d})")
    if abs(ecart) > 30:
        print("  ATTENTION : le servo n'atteint pas le centre. "
              "Quelque chose le bloque, ou le couple est insuffisant.")
        return 1
    print("\nLe palonnier est centré : montez-le sans le faire tourner.")
    print("Le couple reste actif pour tenir la position "
          f"(python3 {sys.argv[0]} libre {args.id} pour relâcher).")
    return 0


def cmd_pos(bus, args):
    """Envoie le servo à un angle donné, en degrés autour du centre."""
    cible = degres_en_pas(args.degres)
    bus.tenir(args.id)
    bus.aller_a(args.id, cible, vitesse=args.vitesse, accel=args.accel)
    atteint = bus.attendre_arret(args.id)
    print(f"cible {cible} pas ({args.degres:+.1f}°) -> "
          f"atteint {atteint} pas ({pas_en_degres(atteint):+.1f}°)")
    return 0


def cmd_test(bus, args):
    """Aller-retour doux autour du centre, avec relevé de la charge."""
    etat = bus.etat(args.id)
    print(f"Servo {args.id} : {etat['tension']:.1f} V, "
          f"{etat['temperature']} °C, position {etat['position']} pas")
    if etat["tension"] < 9.0:
        print("Tension trop basse, test annulé.")
        return 1

    bus.tenir(args.id)
    for angle in [0, args.amplitude, 0, -args.amplitude, 0]:
        cible = degres_en_pas(angle)
        bus.aller_a(args.id, cible, vitesse=args.vitesse, accel=args.accel)
        atteint = bus.attendre_arret(args.id)
        courant = bus.etat(args.id)
        print(f"  -> {pas_en_degres(cible):+7.1f}° : atteint "
              f"{pas_en_degres(atteint):+7.1f}°, charge {courant['charge']:>5}, "
              f"{courant['courant_mA']:>5.0f} mA, {courant['temperature']} °C")
        if abs(atteint - cible) > 40:
            print("     écart important : le servo force ou est bloqué.")

    print("\nTest terminé. Le servo tient encore la position ; "
          f"python3 {sys.argv[0]} libre {args.id} pour le relâcher.")
    return 0


def cmd_axe(bus, args):
    """Amène une articulation du bras assemblé à un angle, par paliers surveillés."""
    cible, bride = limiter(args.id, args.degres)
    if bride:
        bas, haut = LIMITES_DEG[args.id]
        print(f"Consigne {args.degres:+.1f}° hors plage, ramenée à {cible:+.1f}° "
              f"(limites {bas:+.0f}° / {haut:+.0f}°).")

    depart = pas_en_degres(bus.tenir(args.id, vitesse=args.vitesse,
                                     accel=args.accel))
    nom = NOMS_ARM101.get(args.id, "?")
    print(f"Axe {args.id} ({nom}) : {depart:+.1f}° -> {cible:+.1f}°")

    # Découpage en paliers : un obstacle est ainsi rencontré à faible élan.
    ecart = cible - depart
    nombre = max(1, int(abs(ecart) / PAS_INCREMENT + 0.999))
    for i in range(1, nombre + 1):
        palier = depart + ecart * i / nombre
        atteint, pic, obstacle = bus.aller_a_surveille(
            args.id, degres_en_pas(palier), vitesse=args.vitesse,
            accel=args.accel, charge_max=args.charge_max)
        reel = pas_en_degres(atteint)
        etat = "OBSTACLE" if obstacle else f"charge max {pic}"
        print(f"  palier {i}/{nombre} : visé {palier:+7.1f}°, "
              f"atteint {reel:+7.1f}°  ({etat})")

        if obstacle:
            print(f"\nArrêt : la charge a dépassé {args.charge_max} ‰ à "
                  f"{reel:+.1f}°. Le bras tient sa position, l'effort est relâché.")
            print("Vérifiez ce qui bloque avant de relancer.")
            return 1

    final = bus.etat(args.id)
    print(f"\nArrivé à {pas_en_degres(final['position']):+.1f}°, "
          f"{final['temperature']} °C, {final['courant_mA']:.0f} mA")
    return 0


def cmd_tenir(bus, args):
    """Engage le couple là où le bras se trouve, sans aucun mouvement."""
    cibles = [args.id] if args.id is not None else bus.scanner()
    for sid in cibles:
        position = bus.tenir(sid, vitesse=args.vitesse, accel=args.accel)
        print(f"servo {sid:>2} ({NOMS_ARM101.get(sid, '?'):<14}) : couple engagé "
              f"sur {position} pas ({pas_en_degres(position):+.1f}°)")
    print("\nLe bras tient sa position actuelle. Il ne bougera que sur "
          "consigne explicite.")
    return 0


def cmd_libre(bus, args):
    """Coupe le couple : le servo se manipule à la main."""
    cibles = [args.id] if args.id is not None else bus.scanner()
    for sid in cibles:
        bus.couple(sid, False)
        print(f"servo {sid} : couple coupé")
    return 0


def construire_parseur():
    parseur = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parseur.add_argument("--port", default=PORT_DEFAUT)
    parseur.add_argument("--baud", type=int, default=BAUD_DEFAUT)
    parseur.add_argument("--vitesse", type=int, default=800,
                         help="vitesse du mouvement, en pas/s (défaut 800)")
    parseur.add_argument("--accel", type=int, default=30,
                         help="accélération (défaut 30)")
    sous = parseur.add_subparsers(dest="commande", required=True)

    p = sous.add_parser("scan", help="liste les servos présents")
    p.add_argument("--tous-bauds", action="store_true",
                   help="balaye toutes les vitesses de transmission")
    p.set_defaults(fonction=cmd_scan)

    p = sous.add_parser("bilan", help="tableau récapitulatif de la chaîne")
    p.set_defaults(fonction=cmd_bilan)

    p = sous.add_parser("info", help="relevé complet d'un servo")
    p.add_argument("id", type=int)
    p.set_defaults(fonction=cmd_info)

    p = sous.add_parser("id", help="change l'identifiant d'un servo")
    p.add_argument("ancien", type=int)
    p.add_argument("nouveau", type=int)
    p.set_defaults(fonction=cmd_id)

    p = sous.add_parser("prep", help="remet à zéro et centre avant montage")
    p.add_argument("id", type=int)
    p.set_defaults(fonction=cmd_prep)

    p = sous.add_parser("pos", help="va à un angle donné (degrés, 0 = centre)")
    p.add_argument("id", type=int)
    p.add_argument("degres", type=float)
    p.set_defaults(fonction=cmd_pos)

    p = sous.add_parser("test", help="aller-retour doux avec relevé")
    p.add_argument("id", type=int)
    p.add_argument("--amplitude", type=float, default=30,
                   help="amplitude en degrés (défaut 30)")
    p.set_defaults(fonction=cmd_test)

    p = sous.add_parser("axe", help="bouge une articulation du bras assemblé")
    p.add_argument("id", type=int)
    p.add_argument("degres", type=float)
    p.add_argument("--charge-max", type=int, default=CHARGE_MAX,
                   help=f"seuil d'arrêt sur obstacle (défaut {CHARGE_MAX} ‰)")
    p.set_defaults(fonction=cmd_axe)

    p = sous.add_parser("tenir", help="engage le couple sans bouger")
    p.add_argument("id", type=int, nargs="?")
    p.set_defaults(fonction=cmd_tenir)

    p = sous.add_parser("libre", help="coupe le couple")
    p.add_argument("id", type=int, nargs="?")
    p.set_defaults(fonction=cmd_libre)

    return parseur


def main():
    args = construire_parseur().parse_args()

    if args.commande == "scan" and args.tous_bauds:
        return cmd_scan_tous_bauds(args)

    try:
        with BusServos(args.port, args.baud) as bus:
            return args.fonction(bus, args)
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

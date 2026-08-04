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
import json
import sys
import time

# Les butées mesurées, les noms d'axes et les positions de la pince vivent dans
# etalonnage.py : c'est le même étalonnage qui sert ici et au back-end
# materiel/servos.py, il n'existe qu'en un seul exemplaire.
from etalonnage import LIMITES_DEG, NOMS_ARM101, limiter  # noqa: F401
from sts3215 import (BAUDS, CENTRE, ADR_ANGLE_MAX, ADR_ANGLE_MIN, ADR_MODE,
                     ADR_OFFSET, ADR_POS, BusServos, ErreurServo,
                     degres_en_pas, pas_en_degres)

PORT_DEFAUT = "/dev/ttyACM0"
BAUD_DEFAUT = 1_000_000

MODES = {0: "position", 1: "roue libre", 2: "PWM", 3: "pas à pas"}

CHARGE_MAX = 400  # pour-mille du couple maxi : seuil d'arrêt sur obstacle
PAS_INCREMENT = 10.0  # amplitude d'un palier, en degrés


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


def _chercher_butee(bus, sid, vers, args):
    """Avance par paliers vers `vers` jusqu'à sentir une butée.

    Retourne (angle atteint, butée touchée). L'arrêt vient soit de la charge
    qui dépasse le seuil, soit de l'axe qui cesse d'avancer.
    """
    courant = pas_en_degres(bus.etat(sid)["position"])
    while abs(vers - courant) > 0.5:
        avance = min(PAS_INCREMENT, abs(vers - courant))
        prochain = courant + (avance if vers > courant else -avance)
        atteint, pic, obstacle = bus.aller_a_surveille(
            sid, degres_en_pas(prochain), vitesse=args.vitesse,
            accel=args.accel, charge_max=args.charge_max)
        reel = pas_en_degres(atteint)
        print(f"    {reel:+7.1f}°  charge {pic:>4}")

        if obstacle:
            return reel, True
        if abs(reel - courant) < 0.5:  # n'avance plus sans déclencher le seuil
            return reel, True
        courant = reel
    return courant, False


def cmd_course(bus, args):
    """Mesure la course réelle d'un axe en effleurant ses deux butées.

    À réserver aux axes qui peuvent toucher leur butée sans dommage : la pince
    typiquement. Sur les axes qui portent le bras, une butée « mécanique » peut
    en réalité être la table ou une autre pièce du robot.
    """
    bas, haut = LIMITES_DEG.get(args.id, (-160.0, 160.0))
    nom = NOMS_ARM101.get(args.id, "?")
    depart = pas_en_degres(bus.tenir(args.id, vitesse=args.vitesse,
                                     accel=args.accel))
    print(f"Axe {args.id} ({nom}) : départ {depart:+.1f}°, "
          f"exploration entre {bas:+.0f}° et {haut:+.0f}°, "
          f"seuil de charge {args.charge_max} ‰\n")

    print(f"  vers le positif (jusqu'à {haut:+.0f}°) :")
    maxi, butee_haute = _chercher_butee(bus, args.id, haut, args)
    print(f"  vers le négatif (jusqu'à {bas:+.0f}°) :")
    mini, butee_basse = _chercher_butee(bus, args.id, bas, args)

    print(f"\nCourse mesurée : {mini:+.1f}° à {maxi:+.1f}° "
          f"(amplitude {maxi - mini:.1f}°)")
    print(f"  extrémité positive : "
          f"{'butée mécanique' if butee_haute else 'limite logicielle'}")
    print(f"  extrémité négative : "
          f"{'butée mécanique' if butee_basse else 'limite logicielle'}")

    milieu = (mini + maxi) / 2
    print(f"\nRetour au milieu de la course ({milieu:+.1f}°)...")
    bus.aller_a_surveille(args.id, degres_en_pas(milieu), vitesse=args.vitesse,
                          accel=args.accel, charge_max=args.charge_max)
    return 0


def cmd_calibrer(bus, args):
    """Enregistre les butées réelles pendant que l'opérateur bouge le bras à la main.

    Plus fiable que de chercher les butées au moteur : la main sent la butée
    avant de forcer, et surtout elle sait distinguer une vraie fin de course
    d'une collision avec la table ou avec une autre pièce du bras.
    """
    presents = bus.scanner()
    if not presents:
        print("Aucun servo ne répond.")
        return 1

    for sid in presents:
        bus.couple(sid, False)
    print(f"Couple coupé sur {len(presents)} axes : le bras se manipule "
          "librement.\n")
    print(f"Bougez chaque articulation d'une butée à l'autre pendant "
          f"{args.duree:.0f} secondes.")
    print("Tenez le bras pour qu'il ne retombe pas d'un coup.\n")

    releves = {}
    for sid in presents:
        angle = pas_en_degres(bus.etat(sid)["position"])
        releves[sid] = {"min": angle, "max": angle}

    debut = time.monotonic()
    dernier_affichage = 0.0
    lectures = 0
    while True:
        ecoule = time.monotonic() - debut
        if ecoule >= args.duree:
            break
        for sid in presents:
            try:
                angle = pas_en_degres(bus.lire(sid, ADR_POS, 2))
            except ErreurServo:
                continue  # lecture perdue : sans importance, on rééchantillonne
            releves[sid]["min"] = min(releves[sid]["min"], angle)
            releves[sid]["max"] = max(releves[sid]["max"], angle)
        lectures += 1

        if ecoule - dernier_affichage >= 3.0:
            dernier_affichage = ecoule
            etendues = "  ".join(
                f"{sid}:{releves[sid]['max'] - releves[sid]['min']:>5.0f}°"
                for sid in presents)
            restant = args.duree - ecoule
            print(f"  [{restant:>3.0f}s restantes]  amplitudes  {etendues}",
                  flush=True)

    print(f"\n{lectures} relevés par axe.\n")
    print(f"{'id':>2}  {'nom':<14} {'min':>9} {'max':>9} {'amplitude':>10}"
          f"   {'config':>9}")
    resultat = {}
    for sid in presents:
        bas, haut = releves[sid]["min"], releves[sid]["max"]
        amplitude = haut - bas
        # Rappel de l'amplitude actuellement autorisée par LIMITES_DEG, pour
        # voir d'un coup d'oeil si la mesure deborde de ce qui est configure.
        config = LIMITES_DEG.get(sid)
        ref = f"{config[1] - config[0]:.0f}°" if config else "?"
        print(f"{sid:>2}  {NOMS_ARM101.get(sid, '?'):<14} {bas:>+8.1f}° "
              f"{haut:>+8.1f}° {amplitude:>9.1f}°   {ref:>9}")
        resultat[str(sid)] = {"nom": NOMS_ARM101.get(sid), "min_deg": round(bas, 2),
                              "max_deg": round(haut, 2),
                              "amplitude_deg": round(amplitude, 2)}

    maigres = [sid for sid in presents
               if releves[sid]["max"] - releves[sid]["min"] < 20]
    if maigres:
        noms = ", ".join(f"{sid} ({NOMS_ARM101.get(sid)})" for sid in maigres)
        print(f"\nAxes à peine bougés, mesure peu fiable : {noms}")

    chemin = args.sortie
    with open(chemin, "w", encoding="utf-8") as fichier:
        json.dump({"date": time.strftime("%Y-%m-%d %H:%M:%S"),
                   "axes": resultat}, fichier, indent=2, ensure_ascii=False)
    print(f"\nEnregistré dans {chemin}")
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

    p = sous.add_parser("course", help="mesure la course réelle d'un axe")
    p.add_argument("id", type=int)
    p.add_argument("--charge-max", type=int, default=200,
                   help="seuil de détection de butée (défaut 200 ‰)")
    p.set_defaults(fonction=cmd_course)

    p = sous.add_parser("calibrer",
                        help="enregistre les butées en bougeant le bras à la main")
    p.add_argument("--duree", type=float, default=90.0,
                   help="durée d'enregistrement en secondes (défaut 90)")
    p.add_argument("--sortie", default="calibration.json")
    p.set_defaults(fonction=cmd_calibrer)

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

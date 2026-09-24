#!/usr/bin/env python3
"""Vérifie les trames émises par le pilote STS3215, sans aucun matériel.

    venv/bin/python -m materiel.test_protocole

Pourquoi ce fichier existe : `ecrire_groupe` construit une trame que personne
n'acquitte — le SYNC WRITE ne renvoie rien. Une erreur de longueur ou de somme
de contrôle y passerait donc totalement inaperçue à l'exécution : les servos
ignoreraient la trame en silence, et le bras resterait immobile sans le moindre
message. La seule façon de s'en assurer sans brancher le bras est de comparer
les octets produits à une trame calculée à la main.

Les trames attendues ci-dessous ont été posées à la main depuis la
spécification du protocole (en-tête 0xFF 0xFF, longueur = nombre de paramètres
+ 2, somme de contrôle = complément à un de la somme du corps).
"""

import sys


class SerieFactice:
    """Remplace un port série : retient ce qu'on lui écrit, ne répond jamais."""

    def __init__(self):
        self.ecrit = bytearray()

    def write(self, trame):
        self.ecrit.extend(trame)

    def read(self, n=1):
        return b""

    def reset_input_buffer(self):
        pass

    def close(self):
        pass


def bus_factice():
    """Un BusServos branché sur un port série factice."""
    from materiel.sts3215 import BusServos

    import threading

    bus = BusServos.__new__(BusServos)   # sans __init__, donc sans port réel
    bus.ser = SerieFactice()
    bus.timeout = 0.0
    bus._verrou = threading.RLock()      # normalement posé par __init__
    return bus


def hexa(octets) -> str:
    return " ".join(f"{o:02X}" for o in octets)


def verifier(nom, obtenu, attendu) -> bool:
    if bytes(obtenu) == bytes(attendu):
        print(f"  OK   {nom}\n         {hexa(obtenu)}")
        return True
    print(f"  ECHEC {nom}\n         obtenu  {hexa(obtenu)}"
          f"\n         attendu {hexa(attendu)}")
    return False


def main():
    from materiel.sts3215 import ADR_ACCEL, ADR_POS_CIBLE, ADR_VITESSE_CIBLE

    succes = []

    # --- 1. Une écriture simple, dont la forme est déjà éprouvée sur le bras
    # réel : elle sert ici de témoin, pour s'assurer que le banc lui-même dit
    # vrai avant de juger le SYNC WRITE.
    bus = bus_factice()
    try:
        bus.ecrire(1, ADR_POS_CIBLE, 2048, 2)
    except Exception:
        pass    # le port factice n'acquitte rien ; la trame est déjà émise
    # corps = 01 05 03 2A 00 08 ; somme = 0x3B ; complément = 0xC4
    succes.append(verifier("écriture simple, servo 1, position 2048",
                           bus.ser.ecrit,
                           bytes([0xFF, 0xFF, 0x01, 0x05, 0x03,
                                  0x2A, 0x00, 0x08, 0xC4])))

    # --- 2. SYNC WRITE de position sur deux servos
    bus = bus_factice()
    bus.positions_groupe({1: 2048, 2: 1024})
    # params = 2A 02 | 01 00 08 | 02 00 04      (adresse, taille, puis par servo)
    # corps  = FE 0A 83 2A 02 01 00 08 02 00 04 ; somme = 454 -> 0xC6 ; ~ = 0x39
    succes.append(verifier("SYNC WRITE position, servos 1 et 2",
                           bus.ser.ecrit,
                           bytes([0xFF, 0xFF, 0xFE, 0x0A, 0x83, 0x2A, 0x02,
                                  0x01, 0x00, 0x08,
                                  0x02, 0x00, 0x04, 0x39])))

    # --- 3. Les cinq axes du bras d'un coup : c'est le cas réel de la boucle
    # de suivi. On vérifie surtout la longueur annoncée et la somme.
    bus = bus_factice()
    bus.positions_groupe({sid: 2048 + 100 * sid for sid in (1, 2, 3, 4, 5)})
    trame = bytes(bus.ser.ecrit)
    corps = trame[2:-1]
    longueur_ok = corps[1] == len(corps) - 3 + 2      # champ longueur
    somme_ok = trame[-1] == (~sum(corps)) & 0xFF
    entier = len(trame) == 2 + 3 + 2 + 5 * 3 + 1      # en-tête+corps+données+somme
    print(f"  {'OK  ' if longueur_ok and somme_ok and entier else 'ECHEC'} "
          f"SYNC WRITE 5 axes : {len(trame)} octets, longueur annoncée "
          f"{corps[1]}, somme {'juste' if somme_ok else 'FAUSSE'}")
    print(f"         {hexa(trame)}")
    succes.append(longueur_ok and somme_ok and entier)

    # --- 4. Préparation vitesse/accélération : deux trames distinctes, les
    # registres 41 et 46 n'étant pas contigus.
    bus = bus_factice()
    bus.preparer_groupe((1, 2), vitesse=1500, accel=30)
    trames = bytes(bus.ser.ecrit)
    deux = trames.count(b"\xff\xff") == 2
    accel_ok = trames[5] == ADR_ACCEL
    coupe = trames.find(b"\xff\xff", 2)
    vitesse_ok = trames[coupe + 5] == ADR_VITESSE_CIBLE
    print(f"  {'OK  ' if deux and accel_ok and vitesse_ok else 'ECHEC'} "
          f"preparer_groupe : {trames.count(b'\xff\xff')} trames, "
          f"adresses {trames[5]} puis {trames[coupe + 5]} "
          f"(attendu {ADR_ACCEL} puis {ADR_VITESSE_CIBLE})")
    succes.append(deux and accel_ok and vitesse_ok)

    # --- 5. Refus d'un groupe mal formé : des tailles différentes produiraient
    # une trame que les servos interpréteraient de travers, sans rien signaler.
    bus = bus_factice()
    try:
        bus.ecrire_groupe(ADR_POS_CIBLE, {1: b"\x00\x08", 2: b"\x00"})
        print("  ECHEC tailles inégales : aurait dû être refusé")
        succes.append(False)
    except ValueError:
        print("  OK   tailles inégales refusées")
        succes.append(True)

    # --- 6. Un groupe vide n'émet rien du tout.
    bus = bus_factice()
    bus.ecrire_groupe(ADR_POS_CIBLE, {})
    succes.append(verifier("groupe vide : aucune trame", bus.ser.ecrit, b""))

    print(f"\n{sum(succes)}/{len(succes)} vérifications passées.")
    if all(succes):
        print("\nLes trames sont conformes à la spécification. Reste à les\n"
              "confronter au bras : une trame juste sur le papier peut encore\n"
              "buter sur une particularité du firmware.")
    return 0 if all(succes) else 1


if __name__ == "__main__":
    sys.exit(main())

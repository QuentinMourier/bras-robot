"""Pilote bas niveau des servomoteurs Feetech STS3215 (bus série half-duplex).

Le protocole est celui de la famille SMS/STS : trames identiques à celles de
Dynamixel 1.0 (en-tête 0xFF 0xFF, checksum = complément à un de la somme) mais
les mots de 16 bits sont en little-endian (octet de poids faible d'abord).

Ce module ne dépend que de pyserial, volontairement : il sert à diagnostiquer
le matériel avant même d'installer le SDK Feetech ou LeRobot.
"""

import time

import serial

# --- Instructions du protocole ---
PING = 0x01
READ = 0x02
WRITE = 0x03

# --- Table mémoire du STS3215 (adresse du premier octet) ---
# EPROM (persistant, nécessite le déverrouillage du registre LOCK pour écrire)
ADR_ID = 5
ADR_BAUD = 6
ADR_ANGLE_MIN = 9  # 2 octets
ADR_ANGLE_MAX = 11  # 2 octets
ADR_TEMP_MAX = 13
ADR_OFFSET = 31  # 2 octets, correction de position
ADR_MODE = 33  # 0 = position, 1 = roue libre, 2 = PWM, 3 = pas à pas
# SRAM (volatil)
ADR_COUPLE = 40  # torque enable
ADR_ACCEL = 41
ADR_POS_CIBLE = 42  # 2 octets
ADR_VITESSE_CIBLE = 46  # 2 octets
ADR_VERROU = 55  # LOCK : 0 = EPROM déverrouillée, 1 = verrouillée
ADR_POS = 56  # 2 octets
ADR_VITESSE = 58  # 2 octets, signé
ADR_CHARGE = 60  # 2 octets, signé
ADR_TENSION = 62  # 1 octet, en dixièmes de volt
ADR_TEMPERATURE = 63  # 1 octet, en °C
ADR_EN_MOUVEMENT = 66
ADR_COURANT = 69  # 2 octets, unité 6,5 mA

# Index de vitesse de transmission tel que stocké dans le registre 6
BAUDS = {0: 1_000_000, 1: 500_000, 2: 250_000, 3: 128_000,
         4: 115_200, 5: 76_800, 6: 57_600, 7: 38_400}

RESOLUTION = 4096  # pas par tour complet
CENTRE = 2048  # position médiane, celle du montage du SO-ARM101

# Bits du champ « erreur » renvoyé dans chaque réponse
ERREURS = [(0x01, "tension"), (0x02, "angle"), (0x04, "surchauffe"),
           (0x08, "courant"), (0x10, "checksum"), (0x20, "surcharge"),
           (0x40, "instruction")]


class ErreurServo(Exception):
    """Communication impossible ou réponse invalide."""


def decrit_erreur(code: int) -> str:
    """Traduit le champ d'erreur d'une réponse en texte lisible."""
    if code == 0:
        return "aucune"
    noms = [nom for bit, nom in ERREURS if code & bit]
    return ", ".join(noms) or f"inconnue (0x{code:02x})"


def _signe(valeur: int, bit_signe=15) -> int:
    """Décode un entier en signe/magnitude.

    Le STS3215 n'utilise pas le complément à deux : un bit dédié porte le sens.
    Il est en position 15 pour la vitesse, mais en position 10 pour la charge
    (dont la magnitude ne va que de 0 à 1000, en pour-mille du couple maxi).
    """
    masque = 1 << bit_signe
    return -(valeur & (masque - 1)) if valeur & masque else valeur


class BusServos:
    """Un bus série sur lequel sont chaînés un ou plusieurs servos."""

    def __init__(self, port="/dev/ttyACM0", baud=1_000_000, timeout=0.05):
        self.ser = serial.Serial(port, baud, timeout=timeout)
        self.timeout = timeout

    def close(self) -> None:
        self.ser.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    # ------------------------------------------------------------------
    # Couche trame
    # ------------------------------------------------------------------
    def _transaction(self, sid: int, instruction: int, params=b""):
        """Envoie une trame et retourne (erreur, paramètres) de la réponse.

        Retourne None si le servo n'a pas répondu (absent, hors tension, ou
        mauvaise vitesse de transmission).
        """
        corps = bytes([sid, len(params) + 2, instruction]) + params
        trame = b"\xff\xff" + corps + bytes([(~sum(corps)) & 0xFF])

        self.ser.reset_input_buffer()
        self.ser.write(trame)

        if sid == 0xFE:  # diffusion générale : aucun servo ne répond
            return None

        # Certains adaptateurs half-duplex renvoient un écho de ce qu'on émet ;
        # on ignore alors la trame identique à celle qu'on vient d'envoyer.
        fin = time.monotonic() + self.timeout * 4
        while time.monotonic() < fin:
            reponse = self._lire_trame(fin)
            if reponse is None:
                return None
            brut, sid_recu, erreur, donnees = reponse
            if brut == trame:
                continue
            if sid_recu != sid:
                continue
            return erreur, donnees
        return None

    def _lire_trame(self, fin):
        """Lit une trame complète. Retourne (brut, id, erreur, paramètres)."""
        entete = b""
        while time.monotonic() < fin:
            octet = self.ser.read(1)
            if not octet:
                return None
            entete = (entete + octet)[-2:]
            if entete == b"\xff\xff":
                break
        else:
            return None

        tete = self.ser.read(2)
        if len(tete) < 2:
            return None
        sid, longueur = tete
        if longueur < 2:
            return None
        reste = self.ser.read(longueur)
        if len(reste) < longueur:
            return None

        erreur, donnees, checksum = reste[0], reste[1:-1], reste[-1]
        attendu = (~(sid + longueur + erreur + sum(donnees))) & 0xFF
        if attendu != checksum:
            return None
        return b"\xff\xff" + tete + reste, sid, erreur, donnees

    # ------------------------------------------------------------------
    # Opérations élémentaires
    # ------------------------------------------------------------------
    def ping(self, sid: int) -> bool:
        """Teste la présence d'un servo à cet identifiant."""
        return self._transaction(sid, PING) is not None

    def lire(self, sid: int, adresse: int, taille: int) -> int:
        """Lit un registre de 1 ou 2 octets."""
        reponse = self._transaction(sid, READ, bytes([adresse, taille]))
        if reponse is None:
            raise ErreurServo(f"pas de réponse du servo {sid}")
        _, donnees = reponse
        if len(donnees) != taille:
            raise ErreurServo(f"réponse tronquée du servo {sid}")
        return donnees[0] if taille == 1 else donnees[0] | (donnees[1] << 8)

    def ecrire(self, sid: int, adresse: int, valeur: int, taille: int = 1):
        """Écrit un registre de 1 ou 2 octets."""
        if taille == 1:
            octets = bytes([valeur & 0xFF])
        else:
            octets = bytes([valeur & 0xFF, (valeur >> 8) & 0xFF])
        reponse = self._transaction(sid, WRITE, bytes([adresse]) + octets)
        if reponse is None and sid != 0xFE:
            raise ErreurServo(f"écriture non acquittée par le servo {sid}")

    # ------------------------------------------------------------------
    # Confort
    # ------------------------------------------------------------------
    def scanner(self, jusqu_a=20):
        """Retourne la liste des identifiants qui répondent sur le bus."""
        return [sid for sid in range(jusqu_a + 1) if self.ping(sid)]

    def etat(self, sid: int) -> dict:
        """Relève complet d'un servo (position, tension, température...)."""
        return {
            "id": sid,
            "position": self.lire(sid, ADR_POS, 2),
            "vitesse": _signe(self.lire(sid, ADR_VITESSE, 2)),
            "charge": _signe(self.lire(sid, ADR_CHARGE, 2), bit_signe=10),
            "tension": self.lire(sid, ADR_TENSION, 1) / 10,
            "temperature": self.lire(sid, ADR_TEMPERATURE, 1),
            "courant_mA": self.lire(sid, ADR_COURANT, 2) * 6.5,
            "couple_actif": bool(self.lire(sid, ADR_COUPLE, 1)),
            "en_mouvement": bool(self.lire(sid, ADR_EN_MOUVEMENT, 1)),
            "mode": self.lire(sid, ADR_MODE, 1),
            "angle_min": self.lire(sid, ADR_ANGLE_MIN, 2),
            "angle_max": self.lire(sid, ADR_ANGLE_MAX, 2),
            "offset": _signe(self.lire(sid, ADR_OFFSET, 2)),
            "baud": BAUDS.get(self.lire(sid, ADR_BAUD, 1), "?"),
        }

    def couple(self, sid: int, actif: bool) -> None:
        """Active ou coupe le maintien en position (couple moteur)."""
        self.ecrire(sid, ADR_COUPLE, 1 if actif else 0)

    def aller_a(self, sid: int, position: int, vitesse=1000, accel=30) -> None:
        """Envoie une consigne de position, en pas (0 à 4095)."""
        position = max(0, min(RESOLUTION - 1, int(position)))
        self.ecrire(sid, ADR_ACCEL, accel)
        self.ecrire(sid, ADR_VITESSE_CIBLE, vitesse, 2)
        self.ecrire(sid, ADR_POS_CIBLE, position, 2)

    def attendre_arret(self, sid: int, delai_max=8.0) -> int:
        """Attend la fin du mouvement et retourne la position atteinte.

        On ne peut pas se fier au seul drapeau « en mouvement » : il vaut
        encore 0 pendant les quelques millisecondes qui séparent l'envoi de la
        consigne du démarrage effectif, ce qui ferait croire à un arrêt
        immédiat. On exige donc en plus que la position ait cessé de bouger.
        """
        fin = time.monotonic() + delai_max
        precedente = self.lire(sid, ADR_POS, 2)
        stables = 0
        while time.monotonic() < fin:
            time.sleep(0.05)
            position = self.lire(sid, ADR_POS, 2)
            bouge = self.lire(sid, ADR_EN_MOUVEMENT, 1)
            if not bouge and abs(position - precedente) <= 2:
                stables += 1
                if stables >= 3:
                    return position
            else:
                stables = 0
            precedente = position
        return precedente

    # --- Écritures en EPROM (persistantes) ---
    def _deverrouiller(self, sid: int) -> None:
        self.ecrire(sid, ADR_VERROU, 0)

    def _verrouiller(self, sid: int) -> None:
        self.ecrire(sid, ADR_VERROU, 1)

    def changer_id(self, ancien: int, nouveau: int) -> None:
        """Réattribue l'identifiant d'un servo (écriture persistante)."""
        self._deverrouiller(ancien)
        self.ecrire(ancien, ADR_ID, nouveau)
        time.sleep(0.1)
        self._verrouiller(nouveau)

    def regler_eprom(self, sid: int, adresse: int, valeur: int, taille=1):
        """Écrit un registre EPROM en gérant le verrou."""
        self._deverrouiller(sid)
        self.ecrire(sid, adresse, valeur, taille)
        time.sleep(0.05)
        self._verrouiller(sid)


def pas_en_degres(pas: int) -> float:
    """Convertit une position en pas vers un angle en degrés (0 au centre)."""
    return (pas - CENTRE) * 360 / RESOLUTION


def degres_en_pas(degres: float) -> int:
    """Convertit un angle en degrés (0 au centre) vers une position en pas."""
    return int(round(CENTRE + degres * RESOLUTION / 360))

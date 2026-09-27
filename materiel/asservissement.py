#!/usr/bin/env python3
"""Boucle de commande continue : un mouvement fluide, et qui suit une cible mobile.

Le défaut auquel ce module répond
---------------------------------
`BrasServos.aller_a` découpe le trajet en paliers de 10°, et surtout **attend
l'arrivée** à chaque palier avant d'envoyer le suivant. Le bras décélère donc
jusqu'à l'arrêt une dizaine de fois par mouvement : ce sont les à-coups, et ce
sont eux qui secouent la mécanique. S'y ajoutent quinze allers-retours série
par palier (accélération, position et vitesse, écrites servo par servo), soit
des dizaines de millisecondes pendant lesquelles les axes partent en ordre
dispersé.

Le principe retenu ici est l'inverse : on ne commande jamais « va là-bas puis
arrête-toi », mais on rafraîchit une consigne qui glisse, à cadence fixe
(50 Hz par défaut). Le servo a toujours devant lui une cible légèrement en
avance, donc il ne décélère jamais. Trois briques :

  1. Un **générateur de consigne** qui limite vitesse et accélération
     articulaires. C'est lui qui garantit la douceur : une cible qui saute de
     20 cm ne produit plus un à-coup mais une rampe.
  2. La **cinématique différentielle** (`cinematique.jacobienne`), qui convertit
     l'écart cartésien en écart articulaire en 0,2 ms au lieu des 18 à 328 ms
     de l'IK complète — et qui, par construction, ne peut pas changer de
     configuration du bras en cours de route.
  3. Le **SYNC WRITE** (`BusServos.positions_groupe`), une trame unique pour
     les cinq axes au lieu de quinze allers-retours : ils reçoivent leur
     consigne dans la même milliseconde et démarrent ensemble.

La cible peut être déplacée à tout moment depuis un autre fil d'exécution, ce
qui est exactement ce qu'il faut pour suivre un objet vu par la caméra : la
vision tourne à son rythme (10 à 15 images/s), l'asservissement au sien.

Sécurités conservées
--------------------
Aller plus vite ne doit pas vouloir dire aller moins prudemment :
  - toute consigne est bornée par `etalonnage.LIMITES_DEG` avant émission ;
  - le fichier témoin `/tmp/bras-stop` est relu à chaque cycle ;
  - la charge est relevée en continu, un axe par cycle à tour de rôle, donc
    chaque axe toutes les 100 ms à 50 Hz ; au-delà du seuil, tout s'arrête ;
  - vitesse et accélération sont bornées, ce que l'ancien code ne faisait pas.

Ce qui n'a PAS pu être vérifié
------------------------------
Écrit dans la nuit du 24 au 25 septembre 2026, sans accès au bras. Tout est
éprouvé en simulation et sur des trames calculées à la main
(`materiel/test_protocole.py`), mais rien n'a tourné sur les servos réels. Au
premier essai : garder `touch /tmp/bras-stop` sous la main, commencer avec
`--vitesse 20` et l'alimentation à portée.
"""

import threading
import time

import numpy as np

from cinematique.jacobienne import (AMORTISSEMENT, AXES_POSITION, _resoudre,
                                    position, position_et_jacobienne)
from materiel.etalonnage import (AXES_BRAS, LIMITES_DEG, NOMS_ARM101,
                                 angles_vers_degres, degres_vers_angles,
                                 limiter)
from materiel.sts3215 import (ADR_POS, FICHIER_STOP, RESOLUTION, arret_demande,
                              degres_en_pas, pas_en_degres)

FREQUENCE = 50.0            # Hz, cadence de la boucle
VITESSE_MAX_DEG_S = 60.0    # par axe ; l'ancien code tournait à ~35 deg/s
ACCEL_MAX_DEG_S2 = 240.0    # rampe : 0 a 60 deg/s en un quart de seconde
CHARGE_MAX = 400            # pour-mille, comme BrasServos
VITESSE_CART_M_S = 0.15     # m/s, vitesse de la pince dans l'espace
ACCEL_CART_M_S2 = 0.35      # m/s2, rampe correspondante
MARGE_VITESSE = 1.35        # le servo vise un peu au-dela du profil
VITESSE_SERVO_MINI = 8      # pas/s : 0 signifierait « vitesse maximale »
# La loi de freinage suppose que la décélération demandée est réalisable. Elle
# ne l'est pas toujours : passée par la jacobienne et amortie, elle se heurte
# au plafond articulaire. Mesuré en simulation, viser 0,50 m/s2 faisait
# dépasser l'accélération articulaire de 9 % (262 pour un plafond de 240) et
# laissait un dépassement de cible ; 0,35 tient le plafond exactement, pour
# 0,05 s de plus sur un trajet de 1,7 s.
TOLERANCE_ARRIVEE_M = 0.002  # 2 mm : on considère la cible atteinte


class Arret(Exception):
    """La boucle s'est arrêtée d'elle-même (butée, charge, arrêt d'urgence)."""


# ----------------------------------------------------------------------
# Sorties : là où atterrissent les consignes
# ----------------------------------------------------------------------
class SortieServos:
    """Écrit sur les vrais servos, en une trame par cycle."""

    def __init__(self, bus, vitesse_servo: int = 1500, accel_servo: int = 40):
        self.bus = bus
        # Vitesse et accélération se règlent une fois : ensuite chaque cycle ne
        # transmet plus que la position, deux octets par axe. La vitesse est
        # volontairement généreuse — c'est le générateur de consigne qui mène le
        # mouvement, le servo ne doit jamais être le facteur limitant, sans quoi
        # il traîne derrière et le mouvement redevient saccadé.
        bus.preparer_groupe(AXES_BRAS, vitesse=vitesse_servo, accel=accel_servo)

    def positions(self) -> dict:
        return {sid: pas_en_degres(self.bus.lire(sid, ADR_POS, 2))
                for sid in AXES_BRAS}

    def ecrire(self, consignes_deg: dict, vitesses_deg_s: dict = None) -> None:
        if vitesses_deg_s is None:
            self.bus.positions_groupe({sid: degres_en_pas(deg)
                                       for sid, deg in consignes_deg.items()})
            return
        # Vitesse servo = vitesse du profil, majoree d'une marge : le servo doit
        # etre encore en mouvement quand la consigne suivante arrive, sans pour
        # autant devancer le profil.
        self.bus.consignes_groupe({
            sid: (degres_en_pas(deg),
                  max(VITESSE_SERVO_MINI,
                      abs(vitesses_deg_s.get(sid, 0.0)) * MARGE_VITESSE
                      * RESOLUTION / 360.0))
            for sid, deg in consignes_deg.items()})

    def charge(self, sid: int) -> int:
        return self.bus.charge(sid)


class SortieSimulee:
    """Applique les consignes à un `BrasSimule`, pour éprouver la boucle à sec."""

    def __init__(self, bras_simule):
        self.bras = bras_simule
        self.journal = []      # historique des consignes, pour analyse

    def positions(self) -> dict:
        return self.bras.degres_servo(AXES_BRAS)

    def ecrire(self, consignes_deg: dict, vitesses_deg_s: dict = None) -> None:
        self.bras._degres.update(consignes_deg)
        self.journal.append(dict(consignes_deg))

    def charge(self, sid: int) -> int:
        return 0


# ----------------------------------------------------------------------
# Générateur de consigne
# ----------------------------------------------------------------------
def vitesse_d_approche(ecart: float, vitesse_max: float, accel_max: float) -> float:
    """Vitesse à tenir pour rejoindre `ecart` et s'y arrêter pile.

    C'est la loi de freinage : à la distance d d'un but, la vitesse maximale
    dont on peut encore s'arrêter avec la décélération a vaut sqrt(2.a.d). En
    la respectant, on arrive à vitesse nulle sans jamais dépasser — donc sans
    le demi-tour qui suit tout dépassement, et qui est un à-coup de plus.
    """
    return float(np.clip(np.sign(ecart) * np.sqrt(2 * accel_max * abs(ecart)),
                         -vitesse_max, vitesse_max))


class Profil:
    """Intègre une consigne de vitesse articulaire en bornant l'accélération.

    Le générateur ne reçoit plus un but mais une VITESSE voulue, qu'il laisse
    varier d'au plus `accel_max * dt` par cycle. C'est cette borne qui supprime
    les à-coups : la vitesse ne peut pas sauter, donc le couple non plus.

    La première version faisait glisser la consigne vers un but, lequel était
    lui-même recalculé à chaque cycle par la cinématique différentielle. Deux
    régulateurs en cascade, le second poursuivant un but que le premier
    déplaçait : mesuré en simulation, cela dépassait la cible puis repartait en
    sens inverse à 37 deg/s. Commander directement la vitesse supprime la
    cascade, et avec elle l'oscillation.
    """

    def __init__(self, depart_deg: dict, vitesse_max: float, accel_max: float):
        self.consigne = dict(depart_deg)
        self.vitesse = {sid: 0.0 for sid in depart_deg}
        self.vitesse_max = vitesse_max
        self.accel_max = accel_max

    def appliquer(self, vitesses_voulues: dict, dt: float) -> dict:
        """Un pas de dt seconde. Retourne la nouvelle consigne articulaire."""
        saut = self.accel_max * dt
        for sid, voulue in vitesses_voulues.items():
            voulue = float(np.clip(voulue, -self.vitesse_max, self.vitesse_max))
            self.vitesse[sid] = float(np.clip(voulue,
                                              self.vitesse[sid] - saut,
                                              self.vitesse[sid] + saut))
            self.consigne[sid] += self.vitesse[sid] * dt
        return dict(self.consigne)

    def figer(self, positions_deg: dict) -> None:
        """Recale la consigne sur le réel et annule les vitesses."""
        self.consigne.update(positions_deg)
        for sid in self.vitesse:
            self.vitesse[sid] = 0.0


# ----------------------------------------------------------------------
# La boucle
# ----------------------------------------------------------------------
class Asservissement:
    """Maintient le bras sur une cible, rafraîchie à volonté, à cadence fixe.

        with Asservissement(sortie) as asser:
            asser.viser(0.26, -0.05, 0.05)
            asser.attendre_arrivee()

    La boucle tourne dans son propre fil : `viser` rend la main aussitôt, et
    la cible peut être changée en cours de route sans rien interrompre.
    """

    def __init__(self, sortie, frequence: float = FREQUENCE,
                 vitesse_max: float = VITESSE_MAX_DEG_S,
                 accel_max: float = ACCEL_MAX_DEG_S2,
                 charge_max: int = CHARGE_MAX,
                 vitesse_cartesienne: float = VITESSE_CART_M_S,
                 accel_cartesienne: float = ACCEL_CART_M_S2,
                 axes_ik=AXES_POSITION, bavard: bool = False):
        self.sortie = sortie
        self.dt = 1.0 / frequence
        self.charge_max = charge_max
        self.vitesse_cartesienne = vitesse_cartesienne
        self.accel_cartesienne = accel_cartesienne
        self.axes_ik = axes_ik
        self.bavard = bavard

        depart = sortie.positions()
        self.profil = Profil(depart, vitesse_max, accel_max)
        self._angles_but = np.array(degres_vers_angles(depart))
        self._cible_xyz = None

        self._verrou = threading.Lock()
        self._fil = None
        self._tourne = False
        self.motif_arret = None
        self.cycles = 0
        self.retards = 0          # cycles n'ayant pas tenu la cadence

    # -- pilotage ------------------------------------------------------
    def viser(self, x: float, y: float, z: float) -> None:
        """Fixe une cible cartésienne. La boucle s'y dirige par la jacobienne."""
        with self._verrou:
            self._cible_xyz = np.array([x, y, z], dtype=float)

    def viser_angles(self, angles) -> None:
        """Fixe un but articulaire (rad), court-circuitant la cinématique."""
        with self._verrou:
            self._cible_xyz = None
            self._angles_but = np.array(angles, dtype=float)

    def tourner_poignet(self, roulis_rad: float) -> None:
        """Change le roulis du poignet sans toucher au reste.

        Le roulis ne sert pas à positionner (il ne déplace la pince que de 1 mm
        par 10°) mais à orienter les mors avant une saisie. Il est donc piloté
        à part, comme la pince elle-même.
        """
        with self._verrou:
            self._angles_but = self._angles_but.copy()
            self._angles_but[4] = roulis_rad

    def etat(self) -> dict:
        with self._verrou:
            angles = np.array(degres_vers_angles(self.profil.consigne))
            return {
                "consigne_deg": dict(self.profil.consigne),
                "pince_xyz": position(angles),
                "cible_xyz": None if self._cible_xyz is None
                             else self._cible_xyz.copy(),
                "tourne": self._tourne,
                "motif_arret": self.motif_arret,
                "cycles": self.cycles,
                "retards": self.retards,
            }

    def erreur(self) -> float:
        """Distance (m) entre la pince commandée et la cible. inf si sans cible."""
        etat = self.etat()
        if etat["cible_xyz"] is None:
            return float("inf")
        return float(np.linalg.norm(etat["cible_xyz"] - etat["pince_xyz"]))

    def attendre_arrivee(self, tolerance: float = TOLERANCE_ARRIVEE_M,
                         delai: float = 15.0) -> bool:
        """Attend que la pince soit à `tolerance` de la cible. False si délai."""
        fin = time.monotonic() + delai
        while time.monotonic() < fin:
            if not self._tourne:
                return False
            if self.erreur() <= tolerance:
                return True
            time.sleep(self.dt)
        return False

    # -- cycle de vie --------------------------------------------------
    def demarrer(self) -> None:
        if self._tourne:
            return
        self._tourne = True
        self.motif_arret = None
        self._fil = threading.Thread(target=self._boucle, daemon=True,
                                     name="asservissement")
        self._fil.start()

    def arreter(self, motif: str = "demandé") -> None:
        """Arrête la boucle et fige le bras sur sa position réelle."""
        if not self._tourne:
            return
        self._tourne = False
        if self._fil is not None:
            self._fil.join(timeout=2.0)
        self.motif_arret = self.motif_arret or motif
        try:
            reelles = self.sortie.positions()
            self.profil.figer(reelles)
            self.sortie.ecrire(reelles)      # consigne = position : arrêt net
        except Exception:
            pass                              # sur un bus mort, rien à faire

    def __enter__(self):
        self.demarrer()
        return self

    def __exit__(self, *_):
        self.arreter("fin du bloc")

    # -- le coeur ------------------------------------------------------
    def _boucle(self) -> None:
        prochain = time.monotonic()
        tour_de_charge = 0
        while self._tourne:
            prochain += self.dt
            try:
                self._un_cycle(tour_de_charge)
            except Arret as arret:
                self.motif_arret = str(arret)
                self._tourne = False
                if self.bavard:
                    print(f"[asservissement] arrêt : {arret}")
                break
            except Exception as erreur:            # bus coupé, etc.
                self.motif_arret = f"erreur : {erreur}"
                self._tourne = False
                break

            tour_de_charge = (tour_de_charge + 1) % len(AXES_BRAS)
            self.cycles += 1

            reste = prochain - time.monotonic()
            if reste > 0:
                time.sleep(reste)
            else:
                self.retards += 1
                prochain = time.monotonic()       # on ne rattrape pas le retard

    def _un_cycle(self, tour_de_charge: int) -> None:
        if arret_demande():
            raise Arret(f"arrêt d'urgence ({FICHIER_STOP})")

        with self._verrou:
            cible = None if self._cible_xyz is None else self._cible_xyz.copy()
            but = self._angles_but.copy()

        # Tout se calcule depuis la CONSIGNE, jamais depuis la position mesurée :
        # la consigne est lisse, la mesure est bruitée, et asservir sur du bruit
        # ferait vibrer le bras.
        angles = np.array(degres_vers_angles(self.profil.consigne))

        if cible is not None:
            # Vitesse cartésienne voulue : dirigée vers la cible, et plafonnée
            # par la loi de freinage pour arriver sans dépasser.
            p, J = position_et_jacobienne(angles, self.axes_ik)
            ecart = cible - p
            distance = float(np.linalg.norm(ecart))
            if distance > 1e-6:
                norme = vitesse_d_approche(distance, self.vitesse_cartesienne,
                                           self.accel_cartesienne)
                v_cartesienne = ecart / distance * norme
            else:
                v_cartesienne = np.zeros(3)
            # La jacobienne étant linéaire, lui passer une vitesse donne une
            # vitesse : rad/s en sortie, sans autre transformation.
            v_articulaire = _resoudre(J, v_cartesienne, self.axes_ik,
                                      AMORTISSEMENT)
        else:
            v_articulaire = np.zeros(5)

        # Le roulis est toujours mené vers son but propre, cible cartésienne ou
        # non : il n'entre pas dans le positionnement.
        ecart_roulis = np.degrees(but[4] - angles[4])
        vitesses = {}
        for indice, sid in enumerate(AXES_BRAS):
            if cible is not None and indice != 4:
                vitesses[sid] = np.degrees(v_articulaire[indice])
            elif indice == 4:
                vitesses[sid] = vitesse_d_approche(
                    ecart_roulis, self.profil.vitesse_max, self.profil.accel_max)
            else:
                vitesses[sid] = vitesse_d_approche(
                    np.degrees(but[indice] - angles[indice]),
                    self.profil.vitesse_max, self.profil.accel_max)

        consigne = self.profil.appliquer(vitesses, self.dt)

        # Butées : dernier rempart, appliqué à ce qui part réellement sur le bus.
        bornees = {}
        for sid, degres in consigne.items():
            borne, bride = limiter(sid, degres)
            if bride:
                self.profil.consigne[sid] = borne
                self.profil.vitesse[sid] = 0.0    # inutile de pousser la butée
            bornees[sid] = borne

        self.sortie.ecrire(bornees, dict(self.profil.vitesse))

        # Surveillance de charge : un axe par cycle, donc chacun toutes les
        # 100 ms à 50 Hz. Les lire tous à chaque cycle coûterait cinq
        # allers-retours et ruinerait la cadence.
        sid = AXES_BRAS[tour_de_charge]
        charge = abs(self.sortie.charge(sid))
        if charge > self.charge_max:
            raise Arret(f"charge {charge} pour mille sur l'axe {sid} "
                        f"({NOMS_ARM101[sid]}) — quelque chose bloque")


def asservissement_pour(bras, **options) -> Asservissement:
    """Construit un `Asservissement` adapté au back-end fourni.

    Évite que chaque outil ait à connaître la différence entre un bras réel et
    un bras simulé : ils passent leur `Bras`, ils reçoivent une boucle prête.

        with asservissement_pour(bras) as asser:
            aller_asservi(asser, 0.26, -0.05, 0.05)
    """
    sortie = SortieSimulee(bras) if hasattr(bras, "_degres") \
        else SortieServos(bras.bus)
    return Asservissement(sortie, **options)

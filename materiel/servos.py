"""Back-end `Bras` pilotant les servos Feetech STS3215 réels (SO-ARM101).

Fait le lien entre trois briques déjà écrites :
  - `cinematique/` calcule les angles articulaires (rad, repère URDF) ;
  - `materiel/etalonnage.py` les convertit en degrés servo ;
  - `materiel/sts3215.py` parle au bus série.

Les précautions ci-dessous ne sont pas décoratives : elles viennent des essais
du 1er août 2026 sur le bras monté, où une consigne brutale a fait rattraper
50° à l'épaule d'un coup.

  1. Le couple n'est jamais activé directement : `bus.tenir()` recopie d'abord
     la position réelle dans la consigne (cf. sts3215.tenir).
  2. Un mouvement est découpé en paliers de 10° : un obstacle est ainsi
     rencontré à faible élan.
  3. La charge de chaque axe est surveillée pendant tout le trajet ; au-delà du
     seuil, tous les axes sont figés sur place et le mouvement échoue.
  4. Les butées retenues sont celles mesurées à la main (`LIMITES_DEG`), pas
     celles de l'URDF, trop larges (le coude y autorise +96,8° pour une butée
     réelle à +47,7°).
  5. Un arrêt d'urgence est possible depuis n'importe quelle session, sans
     toucher au bus série : `touch /tmp/bras-stop` (voir sts3215.FICHIER_STOP).
     Le témoin est relu à chaque tour de surveillance, soit toutes les 30 ms.
  6. Une solution d'IK qui demanderait un grand écart sur un axe est refusée :
     c'est en général le signe que l'optimiseur a basculé vers une autre
     configuration du bras (coude en haut plutôt qu'en bas), ce qui donnerait
     un grand mouvement inattendu pour une petite cible.

Utilisation, depuis la racine du dépôt :

    from materiel.servos import BrasServos

    with BrasServos() as bras:
        bras.aller_a(0.20, 0.05, 0.15)
        bras.fermer_pince()
"""

import math
import time

import numpy as np

from cinematique.ik import angles_utiles, position_de, vecteur_complet
from cinematique.securite import ik_est_valide, interpoler
from materiel.etalonnage import (AXES_BRAS, AXE_PINCE, LIMITES_DEG,
                                 NOMS_ARM101, PINCE_FERMEE, PINCE_OUVERTE,
                                 SENS_LATERAUX_VERIFIES, SENS_VERIFIES,
                                 TOUS_LES_AXES, angles_vers_degres,
                                 degres_vers_angles, limiter)
from materiel.interface import Bras
from materiel.sts3215 import (ADR_POS, FICHIER_STOP, BusServos, ErreurServo,
                              arret_demande, degres_en_pas, pas_en_degres)

PORT_DEFAUT = "/dev/ttyACM0"
BAUD_DEFAUT = 1_000_000

VITESSE = 400        # pas/s, soit ~35°/s : lent, on privilégie la sûreté
ACCEL = 20
CHARGE_MAX = 400     # pour-mille du couple maxi, seuil d'arrêt sur obstacle
PALIER_DEG = 10.0    # amplitude d'un palier de mouvement
ECART_MAX_DEG = 60.0  # écart maxi accepté sur un axe pour un seul aller_a()

TOLERANCE_PALIER_DEG = 2.0   # « palier atteint », avant correction du biais
TOLERANCE_FINALE_DEG = 0.4   # précision visée en fin de mouvement
CORRECTIONS_MAX = 3          # tentatives de rattrapage du biais de suivi
BIAIS_MAX_DEG = 4.0          # correction cumulée maxi (garde-fou)

PERIODE_SURVEILLANCE = 0.03  # s entre deux relevés position + charge
DELAI_PALIER = 6.0           # s avant de déclarer un palier en échec

MARGE_PRISE_DEG = 5.0  # écart mini à la fermeture à vide pour parler de prise
CHARGE_PRISE = 80      # ‰ : effort de maintien au-delà duquel la pince serre


class BrasServos(Bras):
    """Pilotage du SO-ARM101 réel : 5 articulations + pince, sur un bus série."""

    def __init__(self, port: str = PORT_DEFAUT, baud: int = BAUD_DEFAUT,
                 vitesse: int = VITESSE, accel: int = ACCEL,
                 charge_max: int = CHARGE_MAX,
                 ecart_max_deg: float = ECART_MAX_DEG,
                 bavard: bool = True,
                 accepter_sens_non_verifies: bool = False):
        if not SENS_VERIFIES and not accepter_sens_non_verifies:
            raise RuntimeError(
                "Le sens de rotation des axes n'est pas encore vérifié sur le "
                "bras réel (etalonnage.SENS_VERIFIES = False). Lancez\n"
                "    python3 -m materiel.test_bras monter\n"
                "qui fait un petit mouvement de contrôle, puis passez "
                "SENS_VERIFIES à True.")

        self.bus = BusServos(port, baud)
        self.vitesse = vitesse
        self.accel = accel
        self.charge_max = charge_max
        self.ecart_max_deg = ecart_max_deg
        self.bavard = bavard
        self._stop_demande = False

        absents = [sid for sid in TOUS_LES_AXES if not self.bus.ping(sid)]
        if absents:
            self.bus.close()
            noms = ", ".join(f"{sid} ({NOMS_ARM101[sid]})" for sid in absents)
            raise ErreurServo(f"servos absents du bus {port} : {noms}")

        tension = self.bus.etat(AXES_BRAS[0])["tension"]
        if tension < 9.0 and self.bavard:
            self._dire(f"ATTENTION : {tension:.1f} V sur le bus, c'est bas pour "
                       "un modèle 12 V. Le bras risque de ne pas se tenir.")

    # ------------------------------------------------------------------
    # Cycle de vie
    # ------------------------------------------------------------------
    def close(self) -> None:
        """Ferme le bus série. Ne coupe pas le couple : le bras reste tenu."""
        self.bus.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def _dire(self, message: str) -> None:
        if self.bavard:
            print(f"[bras] {message}")

    # ------------------------------------------------------------------
    # Lecture
    # ------------------------------------------------------------------
    def degres_servo(self, axes=TOUS_LES_AXES) -> dict:
        """Position actuelle de chaque axe, en degrés servo (0 = centre)."""
        return {sid: pas_en_degres(self.bus.lire(sid, ADR_POS, 2))
                for sid in axes}

    def lire_positions(self):
        """Angles des 5 articulations (rad, repère URDF), ordre AXES_BRAS."""
        return np.array(degres_vers_angles(self.degres_servo(AXES_BRAS)))

    def position_pince(self):
        """Position (x, y, z) du point de préhension, en mètres."""
        return position_de(vecteur_complet(self.lire_positions()))

    def charge(self, sid: int) -> int:
        """Charge instantanée d'un axe, en pour-mille du couple maxi (signée)."""
        return self.bus.charge(sid)

    # ------------------------------------------------------------------
    # Mouvement
    # ------------------------------------------------------------------
    def aller_a(self, x: float, y: float, z: float) -> bool:
        """Amène le point de préhension en (x, y, z), en mètres. Voir Bras."""
        depart = self.lire_positions()
        ok, solution, erreur_mm, message = ik_est_valide(
            x, y, z, angles_initiaux=vecteur_complet(depart))
        if not ok:
            self._dire(f"mouvement refusé : {message}")
            return False

        self._dire(f"cible ({x:.3f}, {y:.3f}, {z:.3f}) m, "
                   f"résidu d'IK {erreur_mm:.1f} mm")
        if abs(y) > 0.02 and not SENS_LATERAUX_VERIFIES:
            self._dire(f"attention : y = {y:+.3f} m, alors que le sens de la "
                       "rotation de base n'est pas vérifié. Si l'axe 1 est "
                       "inversé, le bras ira du côté opposé — sans danger "
                       "(les butées sont en degrés servo), mais au mauvais "
                       "endroit. Voir etalonnage.SENS_LATERAUX_VERIFIES.")
        return self.aller_angles(angles_utiles(solution))

    def aller_angles(self, angles) -> bool:
        """Amène les 5 articulations aux angles donnés (rad, repère URDF).

        Sert de point de passage unique pour tous les mouvements du bras :
        c'est ici que sont appliquées les butées mesurées et la limite d'écart.
        """
        cibles = angles_vers_degres(angles)
        depart = self.degres_servo(AXES_BRAS)

        for sid, degres in cibles.items():
            borne, bride = limiter(sid, degres)
            cibles[sid] = borne  # colle sur la butée les dépassements négligeables
            if bride:
                bas, haut = LIMITES_DEG[sid]
                self._dire(f"mouvement refusé : axe {sid} ({NOMS_ARM101[sid]}) "
                           f"demandé à {degres:+.1f}°, hors des butées mesurées "
                           f"[{bas:+.0f}°, {haut:+.0f}°]")
                return False

        ecarts = {sid: cibles[sid] - depart[sid] for sid in AXES_BRAS}
        sid_max = max(ecarts, key=lambda s: abs(ecarts[s]))
        if abs(ecarts[sid_max]) > self.ecart_max_deg:
            self._dire(
                f"mouvement refusé : il faudrait {ecarts[sid_max]:+.0f}° sur "
                f"l'axe {sid_max} ({NOMS_ARM101[sid_max]}), plus que les "
                f"{self.ecart_max_deg:.0f}° autorisés d'un coup. L'IK a "
                "probablement changé de configuration ; approchez la cible en "
                "plusieurs étapes.")
            return False

        return self._executer(depart, cibles)

    def _executer(self, depart: dict, cibles: dict) -> bool:
        """Découpe le mouvement en paliers surveillés et l'exécute."""
        if arret_demande():
            self._dire(f"mouvement refusé : arrêt d'urgence armé. "
                       f"`rm {FICHIER_STOP}` pour réarmer.")
            return False

        self._stop_demande = False
        for sid in AXES_BRAS:
            self.bus.tenir(sid, vitesse=self.vitesse, accel=self.accel)

        amplitude = max(abs(cibles[sid] - depart[sid]) for sid in AXES_BRAS)
        if amplitude < TOLERANCE_FINALE_DEG:
            self._dire("déjà en position")
            return True

        nombre = max(1, math.ceil(amplitude / PALIER_DEG))
        etapes = interpoler([depart[sid] for sid in AXES_BRAS],
                            [cibles[sid] for sid in AXES_BRAS], n_pas=nombre)
        self._dire(f"{amplitude:.0f}° à parcourir, {nombre} paliers")

        for i, etape in enumerate(etapes[1:], start=1):
            palier = dict(zip(AXES_BRAS, etape))
            for sid, degres in palier.items():
                self.bus.aller_a(sid, degres_en_pas(degres),
                                 vitesse=self.vitesse, accel=self.accel)
            atteint, _, motif = self._surveiller(palier, TOLERANCE_PALIER_DEG,
                                                 DELAI_PALIER)
            if not atteint:
                self._figer()
                self._dire(f"arrêt au palier {i}/{nombre} : {motif}")
                return False

        return self._affiner(cibles)

    def _surveiller(self, cibles: dict, tolerance: float, delai: float):
        """Attend que les axes atteignent `cibles` en surveillant la charge.

        Retourne (atteint, incident, motif). `incident` distingue ce qui exige
        d'abandonner le mouvement (charge excessive, arrêt demandé) du simple
        délai dépassé, qui n'est fatal que pendant un palier.
        """
        fin = time.monotonic() + delai
        while time.monotonic() < fin:
            time.sleep(PERIODE_SURVEILLANCE)
            if self._stop_demande:
                return False, True, "arrêt demandé par stop()"
            if arret_demande():
                return False, True, f"arrêt demandé ({FICHIER_STOP})"

            arrive = True
            for sid, cible in cibles.items():
                degres = pas_en_degres(self.bus.lire(sid, ADR_POS, 2))
                charge = abs(self.charge(sid))
                if charge > self.charge_max:
                    return False, True, (f"charge {charge} ‰ sur l'axe {sid} "
                                         f"({NOMS_ARM101[sid]}) à "
                                         f"{degres:+.1f}° — quelque chose bloque")
                if abs(degres - cible) > tolerance:
                    arrive = False
            if arrive:
                return True, False, "atteint"
        return False, False, f"position non atteinte en {delai:.0f} s"

    def _affiner(self, cibles: dict) -> bool:
        """Rattrape le biais de suivi des servos en fin de mouvement.

        Un STS3215 s'arrête là où son couple équilibre la charge, donc un peu
        avant la consigne : biais mesuré de 0,1° sur le roulis à vide, jusqu'à
        1,0° sur le coude quand il porte l'avant-bras. On rejoue donc la
        consigne décalée de l'erreur observée, ce qui revient à demander au
        servo de viser légèrement au-delà pour arriver au bon endroit.
        """
        consignes = dict(cibles)
        for _ in range(CORRECTIONS_MAX):
            reel = self.degres_servo(AXES_BRAS)
            erreurs = {sid: cibles[sid] - reel[sid] for sid in AXES_BRAS}
            pire = max(AXES_BRAS, key=lambda s: abs(erreurs[s]))
            if abs(erreurs[pire]) <= TOLERANCE_FINALE_DEG:
                break

            for sid in AXES_BRAS:
                vise = consignes[sid] + erreurs[sid]
                # Garde-fous : ne jamais s'éloigner de la cible de plus que le
                # biais attendu, et rester dans les butées mesurées.
                vise = max(cibles[sid] - BIAIS_MAX_DEG,
                           min(cibles[sid] + BIAIS_MAX_DEG, vise))
                consignes[sid], _ = limiter(sid, vise)
                self.bus.aller_a(sid, degres_en_pas(consignes[sid]),
                                 vitesse=self.vitesse, accel=self.accel)
            _, incident, motif = self._surveiller(cibles, TOLERANCE_FINALE_DEG,
                                                  1.5)
            if incident:
                self._figer()
                self._dire(f"arrêt pendant le rattrapage final : {motif}")
                return False

        reel = self.degres_servo(AXES_BRAS)
        ecart = max(abs(cibles[sid] - reel[sid]) for sid in AXES_BRAS)
        self._dire(f"arrivé, écart résiduel {ecart:.2f}° au pire des axes")
        return True

    def _figer(self) -> None:
        """Recopie la position réelle dans la consigne : le bras s'arrête net.

        Différent de couper le couple, qui ferait retomber le bras.
        """
        for sid in AXES_BRAS:
            self.bus.aller_a(sid, self.bus.lire(sid, ADR_POS, 2),
                             vitesse=self.vitesse, accel=self.accel)

    def stop(self) -> None:
        """Arrêt d'urgence : fige le bras là où il est, couple maintenu."""
        self._stop_demande = True
        self._figer()
        self._dire("arrêt d'urgence : le bras est figé sur place")

    def liberer(self) -> None:
        """Coupe le couple de tous les axes. ATTENTION : le bras retombe."""
        for sid in TOUS_LES_AXES:
            self.bus.couple(sid, False)
        self._dire("couple coupé sur tous les axes")

    # ------------------------------------------------------------------
    # Pince
    # ------------------------------------------------------------------
    def ouvrir_pince(self) -> None:
        """Ouvre la pince, en cédant si un obstacle gêne l'ouverture."""
        self.bus.tenir(AXE_PINCE, vitesse=self.vitesse, accel=self.accel)
        atteint, pic, obstacle = self.bus.aller_a_surveille(
            AXE_PINCE, degres_en_pas(PINCE_OUVERTE), vitesse=self.vitesse,
            accel=self.accel, charge_max=self.charge_max)
        etat = "bloquée" if obstacle else "ouverte"
        self._dire(f"pince {etat} à {pas_en_degres(atteint):+.1f}° "
                   f"(charge max {pic} ‰)")

    def fermer_pince(self) -> None:
        """Ferme la pince et laisse le servo caler sur ce qu'il rencontre.

        Volontairement sans surveillance de charge, contrairement à
        l'ouverture : c'est le calage du servo sur l'objet qui fait la prise.
        Relâcher l'effort dès que la charge monte, ce serait lâcher l'objet.
        """
        self.bus.tenir(AXE_PINCE, vitesse=self.vitesse, accel=self.accel)
        self.bus.aller_a(AXE_PINCE, degres_en_pas(PINCE_FERMEE),
                         vitesse=self.vitesse, accel=self.accel)
        atteint = pas_en_degres(self.bus.attendre_arret(AXE_PINCE))
        charge = abs(self.charge(AXE_PINCE))
        self._dire(f"pince fermée à {atteint:+.1f}°, charge {charge} ‰"
                   + (" — objet tenu" if self.pince_tient() else " — à vide"))

    def pince_tient(self) -> bool:
        """Vrai si la pince est calée sur un objet plutôt que refermée à vide.

        Deux signes concordants sont exigés : elle s'est arrêtée nettement
        avant sa position de fermeture, et elle continue de forcer pour y
        rester. Les deux seuils comptent — refermée à vide, la pince affiche
        déjà une vingtaine de pour-mille au repos, et elle marque des temps
        d'arrêt de quelques degrés en cours de course.

        Seuils à revoir quand on aura serré de vrais objets : ils sont réglés
        sur la fermeture à vide, faute de mieux pour l'instant.
        """
        degres = pas_en_degres(self.bus.lire(AXE_PINCE, ADR_POS, 2))
        return (degres > PINCE_FERMEE + MARGE_PRISE_DEG
                and abs(self.charge(AXE_PINCE)) > CHARGE_PRISE)

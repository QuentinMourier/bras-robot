#!/usr/bin/env python3
"""Le bras suit une cible colorée en direct, et la saisit sur commande.

    venv/bin/python -m apps.suivi --couleur rouge
    venv/bin/python -m apps.suivi --couleur bleu --simu    # caméra réelle, bras simulé

Commandes, au clavier, sans valider :

      s        suivre / suspendre
      a        attraper la cible
      o        ouvrir la pince
      r        revenir à la pose de repos
      p        état détaillé
      q        quitter

Arrêt d'urgence depuis un autre terminal, à tout moment :  touch /tmp/bras-stop

Comment c'est agencé, et pourquoi
---------------------------------
Trois rythmes cohabitent, et c'est justement ce qui rend le suivi fluide :

  - la CAMÉRA tourne dans son fil, à la cadence que lui permettent la capture
    et le seuillage (10 à 20 images/s) ;
  - l'ASSERVISSEMENT tourne dans le sien, à 50 Hz, et ne dépend jamais de la
    caméra : entre deux détections il continue d'avancer vers la dernière cible
    connue, au lieu de s'arrêter en l'attendant ;
  - le CLAVIER occupe le fil principal.

L'ancienne chaîne faisait tout à la suite : détecter, résoudre l'IK complète
(18 à 328 ms), puis parcourir le trajet par paliers en attendant l'arrivée à
chacun. D'où des à-coups, et un retard de plusieurs secondes sur une cible
mobile. Ici, la détection ne fait que déplacer un but ; c'est l'asservissement
qui mène le bras, sans jamais s'arrêter.

Écrit dans la nuit du 24 au 25 septembre 2026 sans accès au matériel : éprouvé
en simulation et sur images de synthèse, jamais sur le bras. Premier essai
prudent, main sur l'alimentation.
"""

import argparse
import select
import sys
import termios
import threading
import time
import tty

import numpy as np

from cinematique.ik import angles_utiles
from cinematique.orientation import roulis_pour_aligner
from cinematique.securite import ik_est_valide
from materiel.etalonnage import angles_vers_degres, limiter
from materiel.asservissement import (Asservissement, SortieServos,
                                     SortieSimulee)
from materiel.etalonnage import AXES_BRAS, ZERO_DEG, degres_vers_angles
from materiel.sts3215 import FICHIER_STOP, ErreurServo
from vision.detection import (DetecteurCouleur, direction_sur_table,
                              orientation_fiable)
from vision.etalonnage_camera import charger_homographie, pixel_vers_table

PERIODE_VISION = 0.0         # s : aucun plafond, la caméra donne sa cadence
LISSAGE = 0.45               # 0 = cible figée, 1 = aucun lissage
RECUL_SUIVI_M = 0.015        # la pince se tient en retrait de la cible
SAUT_ABERRANT_M = 0.12       # au-delà, on tient la détection pour fausse
ECART_ORIENTATION_MAX = np.radians(35.0)
RECUL_SAISIE_M = 0.05        # distance de pré-saisie, en retrait de la cible


class Vision(threading.Thread):
    """Détecte la cible en continu et tient à jour sa position sur la table."""

    def __init__(self, camera, detecteur, homographie, hauteur,
                 lissage: float = LISSAGE, filtre=None):
        super().__init__(daemon=True, name="vision")
        self.camera = camera
        self.detecteur = detecteur
        self.homographie = homographie
        self.hauteur = hauteur
        self.lissage = lissage
        self.filtre = filtre     # (x, y) -> bool : la cible est-elle a portee ?

        self._verrou = threading.Lock()
        self._position = None       # (x, y) lissé, repère du bras
        self._direction = None      # grand axe sur la table, rad
        self._vue_a = 0.0
        self.images = 0
        self.detections = 0
        self.periode = 0.0          # durée d'un tour, mesurée
        self.tourne = True

    def run(self):
        while self.tourne:
            debut = time.monotonic()
            try:
                image = self.camera.capturer()
            except Exception:
                time.sleep(0.2)
                continue
            self.images += 1

            # On parcourt les candidats du plus gros au plus petit et on
            # retient le premier que le bras puisse atteindre : la plus grosse
            # tache n'est pas forcément la bonne (bouteille, livre...).
            for cible in self.detecteur.detecter_plusieurs(image):
                if self._integrer(cible):
                    break
            self.periode = time.monotonic() - debut

            reste = PERIODE_VISION - self.periode
            if reste > 0:
                time.sleep(reste)

    def _integrer(self, cible) -> bool:
        x, y = pixel_vers_table(cible.u, cible.v, self.homographie)
        nouvelle = np.array([x, y], dtype=float)
        if self.filtre is not None and not self.filtre(x, y):
            return False

        with self._verrou:
            ancienne = self._position
            # Un saut brutal vient presque toujours d'un reflet ou d'un objet
            # de la même couleur qui entre dans le champ : on l'ignore une fois
            # plutôt que d'envoyer le bras à l'autre bout de la table.
            if ancienne is not None and \
                    np.linalg.norm(nouvelle - ancienne) > SAUT_ABERRANT_M:
                return False
            self._position = nouvelle if ancienne is None \
                else ancienne + (nouvelle - ancienne) * self.lissage
            if orientation_fiable(cible):
                self._direction = direction_sur_table(cible, self.homographie)
            self._vue_a = time.monotonic()
            self.detections += 1
        return True

    def derniere(self, fraicheur_max: float = 1.0):
        """(position, direction) si la détection est récente, sinon (None, None)."""
        with self._verrou:
            if self._position is None or \
                    time.monotonic() - self._vue_a > fraicheur_max:
                return None, None
            return self._position.copy(), self._direction


_dernier_refus = [0.0]


def maintenant_ignore(_application) -> bool:
    """Limite l'affichage des refus à un par seconde, pour ne pas noyer l'écran."""
    if time.monotonic() - _dernier_refus[0] < 1.0:
        return False
    _dernier_refus[0] = time.monotonic()
    return True


def lire_touche(delai: float = 0.2):
    """Une touche si elle est là, sinon None. N'immobilise pas la boucle."""
    if not sys.stdin.isatty():
        return None
    if select.select([sys.stdin], [], [], delai)[0]:
        return sys.stdin.read(1)
    return None


class Application:
    def __init__(self, args):
        self.args = args
        self.homographie, self.hauteur = charger_homographie()
        self.simu = args.simu

        from vision.camera import Camera
        self.camera = Camera()

        if args.simu:
            from materiel.simulation import BrasSimule
            self.bras = BrasSimule(bavard=False)
            sortie = SortieSimulee(self.bras)
        else:
            from materiel.servos import BrasServos
            self.bras = BrasServos(args.port, bavard=False)
            sortie = SortieServos(self.bras.bus)

        self.asser = Asservissement(sortie, frequence=args.frequence,
                                    vitesse_max=args.vitesse,
                                    bavard=True)
        self.vision = Vision(self.camera, DetecteurCouleur(args.couleur),
                             self.homographie, self.hauteur,
                             lissage=args.lissage,
                             filtre=lambda x, y: self.atteignable(
                                 x, y, self.hauteur))
        self.suit = False

    # -- actions -------------------------------------------------------
    def atteignable(self, x: float, y: float, z: float) -> bool:
        """Vrai si le back-end accepterait vraiment cette cible.

        Contrôle qui manquait, et dont l'absence a coûté cher le 27 septembre
        2026 : la détection prend la plus grosse tache de la couleur cherchée,
        or la table en portait d'autres — une bouteille, un livre — situées
        HORS de la zone étalonnée. L'homographie y est extrapolée, donc elle
        rend une position fantaisiste, vers laquelle l'asservissement s'est mis
        à tirer jusqu'à écraser la pince sur la table.

        Une cible que le bras ne peut pas atteindre n'est pas une cible : on
        l'ignore, et le bras reste où il est.
        """
        ok, solution, _, _ = ik_est_valide(x, y, z)
        if not ok:
            return False
        degres = angles_vers_degres(angles_utiles(solution))
        return not any(limiter(sid, degres[sid])[1] for sid in AXES_BRAS)

    def en_retrait(self, position):
        """Point situé `recul` mètres en avant de la cible, côté bras.

        On ne vise jamais le doigt lui-même : la pince s'y cognerait. Le
        retrait se prend le long du rayon partant de la base, direction selon
        laquelle le bras aborde naturellement un point de la table.
        """
        distance = float(np.linalg.norm(position))
        if distance < 1e-6:
            return position
        return position - position / distance * self.args.recul

    def angles_consigne(self):
        return np.array(degres_vers_angles(self.asser.profil.consigne))

    def attraper(self):
        """Oriente les mors, ouvre, va sur la cible, referme."""
        position, direction = self.vision.derniere()
        if position is None:
            print("  aucune cible fraîche — rien à attraper")
            return
        self.suit = False
        x, y = float(position[0]), float(position[1])
        if not self.atteignable(x, y, self.hauteur):
            print(f"  cible ({x:+.3f}, {y:+.3f}) hors d'atteinte — saisie "
                  "abandonnée")
            return
        print(f"  saisie de ({x:+.3f}, {y:+.3f})")

        if direction is not None:
            roulis, ecart = roulis_pour_aligner(self.angles_consigne(), direction)
            if ecart <= ECART_ORIENTATION_MAX:
                print(f"  objet orienté à {np.degrees(direction):+.0f}° : "
                      f"roulis {np.degrees(roulis):+.0f}°")
                self.asser.tourner_poignet(roulis)
                time.sleep(1.0)
            else:
                # L'axe des mors ne peut pas être amené perpendiculaire à
                # l'objet depuis cette posture : le dire, et saisir quand même
                # dans l'orientation courante plutôt que de serrer de travers.
                print(f"  orientation demandée hors d'atteinte "
                      f"({np.degrees(ecart):.0f}° d'écart) — prise telle quelle")
        else:
            print("  objet trop rond pour avoir une orientation : prise telle quelle")

        self.bras.ouvrir_pince()

        # Pré-saisie en retrait, puis approche : la pince arrive sur l'objet
        # par un mouvement franc au lieu de le pousser de biais.
        pre = np.array([x, y]) - np.array([x, y]) / np.linalg.norm([x, y]) \
            * RECUL_SAISIE_M
        self.asser.viser(pre[0], pre[1], self.hauteur)
        self.asser.attendre_arrivee(delai=8)
        self.asser.viser(x, y, self.hauteur)
        self.asser.attendre_arrivee(delai=8)

        self.bras.fermer_pince()
        tenu = self.bras.pince_tient()
        print(f"  pince refermée — objet tenu : {'oui' if tenu else 'non'}")

    def repos(self):
        self.suit = False
        self.asser.viser_angles(np.array(degres_vers_angles(
            {s: ZERO_DEG[s] for s in AXES_BRAS})))
        print("  retour au repos")

    def etat(self):
        e = self.asser.etat()
        p = e["pince_xyz"]
        position, direction = self.vision.derniere()
        print(f"  pince   ({p[0]:+.3f}, {p[1]:+.3f}, {p[2]:+.3f}) m")
        print(f"  cible   " + ("aucune" if position is None else
              f"({position[0]:+.3f}, {position[1]:+.3f}) m" +
              ("" if direction is None
               else f", axe {np.degrees(direction):+.0f}°")))
        print(f"  vision  {self.vision.images} images, "
              f"{self.vision.detections} détections, "
              f"{1 / max(self.vision.periode, 1e-6):.1f} img/s")
        print(f"  boucle  {e['cycles']} cycles, {e['retards']} en retard")

    # -- boucle principale ---------------------------------------------
    def tourner(self):
        self.vision.start()
        self.asser.demarrer()
        print(f"=== Suivi ({'SIMULATION' if self.simu else 'BRAS RÉEL'}) — "
              f"couleur {self.args.couleur} ===")
        print(f"Arrêt d'urgence : touch {FICHIER_STOP}")
        print("s suivre/suspendre | a attraper | o ouvrir | r repos | "
              "p état | q quitter\n")

        dernier_affichage = 0.0
        while True:
            touche = lire_touche()
            if touche in ("q", "\x03"):
                break
            if touche == "s":
                self.suit = not self.suit
                print(f"  suivi {'actif' if self.suit else 'suspendu'}")
            elif touche == "a":
                self.attraper()
            elif touche == "o":
                self.bras.ouvrir_pince()
                print("  pince ouverte")
            elif touche == "r":
                self.repos()
            elif touche == "p":
                self.etat()

            if not self.asser._tourne:
                print(f"\n  asservissement arrêté : {self.asser.motif_arret}")
                break

            if self.suit:
                position, _ = self.vision.derniere()
                if position is not None:
                    self.asser.viser(float(position[0]), float(position[1]),
                                     self.hauteur)

            maintenant = time.monotonic()
            if self.suit and maintenant - dernier_affichage > 1.0:
                dernier_affichage = maintenant
                e = self.asser.etat()
                p = e["pince_xyz"]
                position, _ = self.vision.derniere()
                if position is not None:
                    reste = np.linalg.norm(position - p[:2])
                    print(f"  pince ({p[0]:+.3f}, {p[1]:+.3f})  "
                          f"cible ({position[0]:+.3f}, {position[1]:+.3f})  "
                          f"écart {reste * 1000:5.1f} mm  "
                          f"{1 / max(self.vision.periode, 1e-6):4.1f} img/s")

    def fermer(self):
        self.vision.tourne = False
        self.asser.arreter("fin de session")
        try:
            self.camera.close()
        except Exception:
            pass
        try:
            self.bras.close()
        except Exception:
            pass


def main():
    parseur = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parseur.add_argument("--couleur", default="rouge")
    parseur.add_argument("--simu", action="store_true")
    parseur.add_argument("--port", default="/dev/ttyACM0")
    parseur.add_argument("--frequence", type=float, default=50.0)
    parseur.add_argument("--recul", type=float, default=RECUL_SUIVI_M,
                         help="retrait de la pince devant la cible, en m "
                              f"(défaut {RECUL_SUIVI_M})")
    parseur.add_argument("--lissage", type=float, default=LISSAGE,
                         help="0 = cible figée, 1 = aucun lissage "
                              "(monter pour plus de réactivité)")
    parseur.add_argument("--vitesse", type=float, default=60.0,
                         help="vitesse articulaire maxi, deg/s "
                              "(commencer bas au premier essai)")
    args = parseur.parse_args()

    try:
        application = Application(args)
    except FileNotFoundError:
        print("Pas d'homographie : lancez d'abord\n"
              "    venv/bin/python -m vision.etalonnage_camera ajuster")
        return 1
    except (ErreurServo, OSError) as erreur:
        print(f"Bras injoignable : {erreur}")
        return 1

    interactif = sys.stdin.isatty()
    reglages = termios.tcgetattr(sys.stdin) if interactif else None
    try:
        if interactif:
            tty.setcbreak(sys.stdin.fileno())
        application.tourner()
    except KeyboardInterrupt:
        print("\nInterrompu.")
    finally:
        if interactif:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, reglages)
        application.fermer()
    return 0


if __name__ == "__main__":
    sys.exit(main())

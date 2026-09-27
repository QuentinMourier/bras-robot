#!/usr/bin/env python3
"""Rejoindre une cible lointaine en plusieurs pas.

`Bras.aller_a` refuse tout mouvement demandant plus de ECART_MAX_DEG (60°) sur
un axe : c'est une sécurité, pas un défaut. Un grand écart signale en général
que l'IK a changé de configuration du bras (coude en haut plutôt qu'en bas), et
l'exécuter donnerait un grand mouvement imprévu pour une petite cible.

La parade est celle que le message d'erreur recommande lui-même : découper le
trajet en points de passage. Ce module le fait, pour les deux back-ends — c'est
du pur appel à l'interface `Bras`, il ne sait pas s'il pilote des servos ou une
simulation.

    from materiel.approche import aller_par_etapes

    aller_par_etapes(bras, 0.26, 0.03, 0.05)

À noter : les points intermédiaires sont interpolés en ligne droite dans
l'espace, sans aucune connaissance des obstacles. Comme partout ailleurs dans
ce dépôt, c'est la surveillance de charge du bras réel qui protège des
collisions, pas la géométrie.
"""

import json
import time
from pathlib import Path

import numpy as np

from materiel.etalonnage import (AXES_BRAS, NOMS_ARM101, ZERO_DEG,
                                 degres_vers_angles)

ETAPES_MAX = 8

# Pose de repli : celle où tous les angles URDF valent zéro, bras tendu vers
# l'avant. Son intérêt n'est pas son allure mais sa position dans l'espace
# ARTICULAIRE — chaque axe y est loin de ses butées, donc l'IK lancée de là
# trouve des solutions franches plutôt que des contorsions.
POSE_NEUTRE_DEG = {sid: ZERO_DEG[sid] for sid in AXES_BRAS}

# Pose de REPOS : celle où l'on laisse le bras en fin de session. Distincte de
# la pose neutre, qui sert à l'IK et non au rangement.
#
# La neutre a le bras tendu à l'horizontale, ce qui est mauvais pour se reposer
# — couple coupé, il tombe de haut. Une bonne pose de repos est basse et
# repliée, pour que relâcher les moteurs ne provoque presque aucune chute.
#
# Elle n'est pas calculée mais RELEVÉE : la chercher par optimisation donne des
# postures contorsionnées ou en contact avec le socle, que le modèle ignore
# puisqu'il ne connaît ni le support ni les collisions du bras avec lui-même.
# On préfère donc la poser à la main, moteurs relâchés, et l'enregistrer.
FICHIER_REPOS = Path(__file__).parent.parent / "pose_repos.json"


def pose_repos():
    """Pose de repos relevée, ou la pose neutre si aucune n'a été enregistrée."""
    try:
        donnees = json.loads(FICHIER_REPOS.read_text(encoding="utf-8"))
        return {int(sid): float(v) for sid, v in donnees["degres"].items()}
    except (FileNotFoundError, KeyError, ValueError):
        return dict(POSE_NEUTRE_DEG)


def enregistrer_pose_repos(degres: dict, note: str = "") -> None:
    """Retient la pose courante comme pose de repos."""
    FICHIER_REPOS.write_text(json.dumps(
        {"date": time.strftime("%Y-%m-%d %H:%M:%S"),
         "note": note or "posée à la main, moteurs relâchés",
         "degres": {str(sid): round(float(v), 2)
                    for sid, v in degres.items()}},
        indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def aller_au_repos_asservi(asser, delai: float = 25.0,
                           tolerance_deg: float = 2.0) -> bool:
    """Amène le bras à la pose de repos relevée, par la boucle continue."""
    return _rejoindre_angles(asser, pose_repos(), delai, tolerance_deg)

PAS_ARTICULAIRE_DEG = 45.0


def aller_par_etapes(bras, x: float, y: float, z: float,
                     etapes_max: int = ETAPES_MAX, bavard: bool = True) -> bool:
    """Amène la pince en (x, y, z), en insérant des points de passage au besoin.

    Tente d'abord le mouvement direct — le cas courant, et le plus rapide. S'il
    est refusé, recommence en coupant le trajet en 2, puis 3, etc. jusqu'à
    `etapes_max`. Retourne True dès qu'un découpage passe entièrement.

    Un découpage qui échoue en cours de route laisse le bras là où il s'est
    arrêté : c'est voulu, on ne tente pas de revenir en arrière tout seul.
    """
    cible = np.array([x, y, z], dtype=float)

    if bras.aller_a(*cible):
        return True

    for n in range(2, etapes_max + 1):
        depart = np.array(bras.position_pince(), dtype=float)
        if bavard:
            print(f"[approche] nouvel essai en {n} étapes")
        # On saute le premier point (t=0), qui est la position actuelle.
        fractions = np.linspace(0.0, 1.0, n + 1)[1:]
        if all(bras.aller_a(*(depart + (cible - depart) * t)) for t in fractions):
            return True

    if bavard:
        print(f"[approche] cible ({x:+.3f}, {y:+.3f}, {z:+.3f}) m non atteinte "
              f"même en {etapes_max} étapes.")
    return False


def aller_angles_par_etapes(bras, cibles_deg: dict,
                            pas_max_deg: float = PAS_ARTICULAIRE_DEG,
                            bavard: bool = True) -> bool:
    """Rejoint des angles donnés en interpolant dans l'espace ARTICULAIRE.

    Aucune cinématique inverse n'intervient : on connaît déjà les angles
    voulus. Le bras ne peut donc pas changer de configuration en route, ce qui
    est exactement le défaut que cette fonction sert à contourner.

    Le trajet est coupé en pas de `pas_max_deg` pour rester sous la limite
    d'écart de `aller_angles`.
    """
    depart = bras.degres_servo(AXES_BRAS)
    ecart = max(abs(cibles_deg[sid] - depart[sid]) for sid in AXES_BRAS)
    nombre = max(1, int(np.ceil(ecart / pas_max_deg)))
    if bavard and nombre > 1:
        print(f"[approche] {ecart:.0f}° à parcourir en articulaire, "
              f"{nombre} étapes")

    for t in np.linspace(0.0, 1.0, nombre + 1)[1:]:
        etape = {sid: depart[sid] + (cibles_deg[sid] - depart[sid]) * t
                 for sid in AXES_BRAS}
        if not bras.aller_angles(degres_vers_angles(etape)):
            if bavard:
                print("[approche] étape articulaire refusée")
            return False
    return True


def revenir_au_neutre(bras, bavard: bool = True) -> bool:
    """Ramène le bras à POSE_NEUTRE_DEG, bras tendu et axes loin des butées.

    À utiliser quand une cible est refusée : le bras s'est en général coincé
    dans un recoin de son espace articulaire, d'où la cible n'est joignable
    qu'en retournant un axe de 150°. Repartir du neutre remet l'IK en terrain
    franc.
    """
    if bavard:
        print("[approche] retour à la pose neutre")
    return aller_angles_par_etapes(bras, POSE_NEUTRE_DEG, bavard=bavard)


def aller_en_repassant_au_neutre(bras, x: float, y: float, z: float,
                                 bavard: bool = True) -> bool:
    """`aller_par_etapes`, et si cela échoue, réessaie depuis la pose neutre."""
    if aller_par_etapes(bras, x, y, z, bavard=bavard):
        return True
    if not revenir_au_neutre(bras, bavard=bavard):
        return False
    return aller_par_etapes(bras, x, y, z, bavard=bavard)


# ----------------------------------------------------------------------
# Versions asservies : les mêmes déplacements, mais fluides
# ----------------------------------------------------------------------
#
# Tout ce qui précède passe par `Bras.aller_a` et ses paliers de 10 degrés
# suivis d'une attente d'arrivée, puis d'un rattrapage de biais en fin de
# course. Observé sur le bras réel le 27 septembre 2026 : ce chemin est
# visiblement saccadé, là où la boucle d'asservissement, sur le même trajet,
# est fluide. L'oeil l'a tranché avant l'instrumentation, qui ne regardait
# qu'un axe et n'avait rien vu.
#
# Les fonctions ci-dessous font le même travail à travers un `Asservissement`
# déjà démarré. À préférer partout où l'on en a un sous la main.

def aller_asservi(asser, x: float, y: float, z: float,
                  tolerance: float = 0.004, delai: float = 20.0) -> bool:
    """Rejoint (x, y, z) par la boucle continue. True si la cible est atteinte."""
    asser.viser(x, y, z)
    return asser.attendre_arrivee(tolerance=tolerance, delai=delai)


def _rejoindre_angles(asser, cible_deg: dict, delai: float,
                      tolerance_deg: float) -> bool:
    """Vise des ANGLES et attend que la consigne les ait rejoints.

    On vise des angles et non un point : ces poses sont définies dans l'espace
    articulaire, et y aller par la cinématique inverse rendrait une posture
    différente pour la même position de pince.
    """
    asser.viser_angles(np.array(degres_vers_angles(cible_deg)))
    fin = time.monotonic() + delai
    while time.monotonic() < fin:
        consigne = asser.etat()["consigne_deg"]
        if max(abs(consigne[sid] - cible_deg[sid])
               for sid in AXES_BRAS) <= tolerance_deg:
            return True
        time.sleep(asser.dt)
    return False


def revenir_au_neutre_asservi(asser, delai: float = 20.0,
                              tolerance_deg: float = 1.5) -> bool:
    """Ramène le bras à POSE_NEUTRE_DEG par la boucle continue."""
    return _rejoindre_angles(asser, POSE_NEUTRE_DEG, delai, tolerance_deg)

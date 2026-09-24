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

import numpy as np

from materiel.etalonnage import (AXES_BRAS, NOMS_ARM101, ZERO_DEG,
                                 degres_vers_angles)

ETAPES_MAX = 8

# Pose de repli : celle où tous les angles URDF valent zéro, bras tendu vers
# l'avant. Son intérêt n'est pas son allure mais sa position dans l'espace
# ARTICULAIRE — chaque axe y est loin de ses butées, donc l'IK lancée de là
# trouve des solutions franches plutôt que des contorsions.
POSE_NEUTRE_DEG = {sid: ZERO_DEG[sid] for sid in AXES_BRAS}

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

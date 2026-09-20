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

ETAPES_MAX = 8


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

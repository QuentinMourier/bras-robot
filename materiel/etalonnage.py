"""Correspondance entre les angles du modèle URDF et les positions des servos.

C'est la pièce qui manquait entre `cinematique/` (qui raisonne en radians dans
le repère de urdf/so101_new_calib.urdf) et `materiel/sts3215.py` (qui parle en
pas de servo, 0 à 4095, ou en degrés autour du centre 2048).

La relation est affine, un servo par articulation :

    angle_urdf = sens x (degres_servo - zero)      (sens vaut +1 ou -1)

--- Le zéro (`ZERO_DEG`) ---

L'URDF utilisé est la version « new calib » du dépôt SO-ARM100, dont chaque
articulation a son zéro **au milieu de sa plage** (Simulation/SO101/README.md).
Le milieu de la course réelle, mesuré à la main le 1er août 2026 et enregistré
dans calibration.json, donne donc directement la position servo du zéro URDF.

Deux articulations font exception et ne sont pas centrées sur zéro dans l'URDF
(`MILIEU_URDF_DEG`) : le roulis du poignet (-157,2° / +162,8°) et la pince
(-10° / +100°).

--- Le sens (`SENS`) ---

Une plage mesurée est compatible avec les deux sens de rotation : la mesure
seule ne peut pas trancher. Trois sources l'ont fait, le 2 août 2026 :

1. **La gravité, sans rien bouger.** L'URDF porte la masse et le centre de
   gravité de chaque pièce ; le bras au repos, couple coupé, était retenu par
   ses butées. La gravité doit donc pousser chaque axe *dans* la butée où il
   repose. Le coude y répond sans ambiguïté (+0,38 N·m contre sa butée haute,
   dix fois le bruit) : son sens est +1. L'épaule et le poignet, eux, sortent
   entre 0,02 et 0,08 N·m, soit dans la bande de friction du réducteur 1:345 —
   indécidable ainsi.

2. **Un mouvement calculé à travers la convention elle-même.** C'est le seul
   type d'essai qui puisse la mettre en défaut : commander un axe en degrés
   servo produit le même mouvement quel que soit le sens supposé, seule
   l'interprétation change. On a donc demandé « la pince, 3 cm plus haut »
   (commande `monter` de test_bras.py), consigne obtenue par IK puis reconvertie
   en degrés servo. Observé à la caméra : la pince est montée de 2,8 cm pour
   3,0 prédits. Passées au crible de ce seul relevé, 28 des 32 combinaisons de
   signes sont éliminées — l'hypothèse concurrente prédisait une descente de
   7,6 cm et un écart latéral de 5,2 cm.

3. Restent quatre combinaisons, qui ne diffèrent que par les axes 1 et 5 (voir
   `SENS_LATERAUX_VERIFIES` plus bas).
"""

import math

# --- Identification des axes ---
NOMS_ARM101 = {1: "shoulder_pan", 2: "shoulder_lift", 3: "elbow_flex",
               4: "wrist_flex", 5: "wrist_roll", 6: "gripper"}

# Les 5 articulations du bras, dans l'ordre de la chaîne ikpy et de
# materiel.interface.Bras.lire_positions().
AXES_BRAS = (1, 2, 3, 4, 5)
AXE_PINCE = 6
TOUS_LES_AXES = AXES_BRAS + (AXE_PINCE,)

# Plages autorisées, en degrés servo autour du centre (2048 pas).
#
# Mesurées à la main le 1er août 2026 sur le bras monté (commande `calibrer`,
# 20 000 relevés par axe, résultats bruts dans calibration.json), puis rentrées
# 3° en deçà des butées réelles pour ne jamais venir taper dedans.
#
# Les amplitudes obtenues correspondent toutes au design de
# urdf/so101_new_calib.urdf à quelques degrés près, ce qui valide le montage.
# En revanche le zéro servo n'est pas au milieu de la course sur l'épaule
# (+40,9°) ni sur le coude (-49,2°) : c'est voulu par le design, ces deux
# articulations ayant un débattement utile asymétrique. Ne pas chercher à
# « corriger » ce décalage, il fait partie de la géométrie du bras.
#
# Ces limites-là font foi, pas celles de l'URDF, beaucoup trop larges.
LIMITES_DEG = {
    # Axe 1 re-mesuré le 20 septembre 2026, après déplacement et re-serrage du
    # bras sur son support : la course accessible est passée de
    # [-108,8 ; +2,2] à [-46,3 ; +69,2], soit la même amplitude (115° contre
    # 111°) décalée d'environ 65°. Les deux extrémités relevées sont désormais
    # à l'intérieur de la course interne du bras (±110° d'après l'URDF) : ce
    # sont donc deux blocages du support, et non des butées du bras lui-même.
    #
    # Le zéro, lui, n'a PAS bougé : un décalage du palonnier aurait déplacé le
    # repère du servo de 65°, ce qui se serait vu à l'œil. Vérifié le 20
    # septembre en amenant l'axe à +1,4° (soit l'angle URDF zéro d'après
    # MILIEU_MESURE_DEG) : le bras pointe bien droit devant. D'où deux chiffres
    # de statuts différents pour cet axe — la butée change avec le support, le
    # zéro non.
    #
    # Révisé le 27 septembre 2026, après fixation définitive du socle dans une
    # pince. Ces bornes-ci ne sont NI des butées mécaniques NI des blocages du
    # support : ce sont les limites de l'espace réellement dégagé autour du
    # bras, relevées à la main (-38,4 / +45,9) et arrondies vers l'intérieur.
    # Une multiprise d'un côté, les câbles de l'autre, installés à demeure.
    #
    # La distinction compte pour la suite : un jour où cet encombrement
    # changera, ces deux valeurs seront à revoir, alors que les butées des
    # autres axes, elles, sont internes au bras et ne bougeront jamais. Les
    # obstacles imprévus, eux, restent du ressort de la surveillance de charge.
    #
    # Coût sur la zone atteignable, mesuré : le coeur (y de -0,10 à +0,10)
    # reste accessible à 100 %, les bords latéraux tombent à 43-57 % au-delà
    # de |y| = 0,20.
    1: (-38.0, 45.0),
    2: (-57.0, 139.0),    # butées réelles -60,7° / +142,5°
    3: (-143.0, 44.0),    # butées réelles -146,2° / +47,7°
    4: (-94.0, 92.0),     # butées réelles -97,7° / +95,9°
    5: (-165.0, 156.0),   # butées réelles -168,0° / +159,3°
    6: (-95.0, 15.0),     # butées réelles -98,1° / +18,0°
}

# Pince, mesuré sur le bras monté le 1er août 2026 (commande `course`) :
#
# -98,8° est une vraie butée mécanique, mors contre mors : la charge y monte à
# 124 ‰ et l'axe cesse d'avancer. C'est le zéro naturel de la pince.
#
# La butée d'ouverture est à +18,0°, relevée à la main : la recherche au moteur
# ne l'avait pas trouvée, elle s'était arrêtée sur la limite logicielle d'alors.
# La course totale fait donc 116°, soit 5° de plus que ce que déclare l'URDF.
#
# Pour saisir un objet, viser PINCE_FERMEE et laisser le servo caler dessus :
# c'est la charge qui fait la prise, pas la position atteinte.
PINCE_FERMEE = -96.0    # 2° avant la butée, pour ne pas forcer en permanence
PINCE_OUVERTE = 15.0    # 3° avant la butée d'ouverture mesurée à +18,0°

# --- Correspondance URDF <-> servo ---

# Milieu de la course réelle, en degrés servo (calibration.json).
# L'axe 1 n'a qu'une seule butée mesurable (-108,8°), la pince du support
# bloquant l'autre côté : son milieu est déduit de l'amplitude de l'URDF (220°).
MILIEU_MESURE_DEG = {1: 1.19, 2: 40.87, 3: -49.22, 4: -0.88, 5: -4.35,
                     6: -40.04}

# Milieu de la plage déclarée par l'URDF, en degrés. Zéro partout sauf pour le
# roulis et la pince, dont les butées ne sont pas symétriques.
MILIEU_URDF_DEG = {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 2.79, 6: 45.0}

# Sens de rotation de chaque axe : +1 si les degrés servo croissent dans le même
# sens que l'angle URDF, -1 sinon. Voir l'en-tête du module.
SENS = {1: +1, 2: +1, 3: +1, 4: +1, 5: +1, 6: +1}

# Vérifié sur le bras réel le 2 août 2026 pour les axes 2, 3 et 4 — ceux qui
# déterminent la forme du bras et tout son comportement vertical. Voir la
# méthode en tête de module. Tant que c'était False, BrasServos refusait de
# bouger : un sens inversé aurait envoyé le bras dans une pose radicalement
# différente de celle calculée, tout en restant dans les butées.
SENS_VERIFIES = True

# Axe 1 (rotation de base) VÉRIFIÉ le 20 septembre 2026, par la méthode que ce
# commentaire réclamait : un déplacement latéral commandé, observé à l'œil.
#
# Protocole : `python -m materiel.pilote` sur le bras réel, puis flèche gauche,
# qui demande +1 cm sur y. Observé : la base tourne dans le sens trigonométrique
# vu de dessus — soit, pour un opérateur qui fait face au bras, vers sa droite
# (les côtés s'inversent quand on se fait face : l'opérateur regarde selon -x,
# donc sa droite est le +y du bras).
#
# C'est exactement ce que prédit le modèle : dans un repère direct avec x vers
# l'avant et z vers le haut, aller vers +y, c'est tourner dans le sens
# trigonométrique. Le bras est donc allé là où le modèle l'envoyait, et
# SENS[1] = +1 est confirmé. y et -y ne sont pas échangés.
#
# Axe 5 (roulis du poignet) : toujours pas vérifié par un essai dédié, le test
# vertical d'août ne le distinguant pas (il ne déplace le point de préhension
# que de 1,3 cm). Mais c'est devenu sans effet sur le positionnement : depuis la
# correction du 5 septembre, cinematique/ik.py gèle wrist_roll et ne le commande
# plus jamais (voir MASQUE_LIENS_ACTIFS). Son sens ne redeviendra une question
# que le jour où l'on orientera la pince explicitement avant une saisie.
SENS_LATERAUX_VERIFIES = True

# Position servo, en degrés, à laquelle l'angle URDF vaut zéro.
ZERO_DEG = {sid: MILIEU_MESURE_DEG[sid] - SENS[sid] * MILIEU_URDF_DEG[sid]
            for sid in TOUS_LES_AXES}


def servo_vers_urdf(sid: int, degres: float) -> float:
    """Degrés servo (0 = centre, 2048 pas) -> angle de l'articulation URDF (rad)."""
    return math.radians(SENS[sid] * (degres - ZERO_DEG[sid]))


def urdf_vers_servo(sid: int, radians: float) -> float:
    """Angle de l'articulation URDF (rad) -> degrés servo (0 = centre)."""
    return ZERO_DEG[sid] + SENS[sid] * math.degrees(radians)


def angles_vers_degres(angles) -> dict:
    """Les 5 angles du bras (rad, ordre AXES_BRAS) -> {id servo: degrés}."""
    return {sid: urdf_vers_servo(sid, angle)
            for sid, angle in zip(AXES_BRAS, angles)}


def degres_vers_angles(degres: dict) -> list:
    """{id servo: degrés} -> les 5 angles du bras (rad, ordre AXES_BRAS)."""
    return [servo_vers_urdf(sid, degres[sid]) for sid in AXES_BRAS]


# Un angle qui ne dépasse la butée que de cette valeur est ramené dessus sans
# être déclaré hors plage. Sans cette tolérance, viser exactement une butée est
# refusé : l'aller-retour degrés -> radians -> degrés d'un angle rond ressort à
# 1e-14 près, ce qui suffit à le faire passer pour un dépassement.
TOLERANCE_BUTEE_DEG = 1e-6


def limiter(sid: int, degres: float):
    """Ramène un angle servo dans la plage autorisée. Retourne (angle, bridé)."""
    if sid not in LIMITES_DEG:
        return degres, False
    bas, haut = LIMITES_DEG[sid]
    borne = max(bas, min(haut, degres))
    return borne, abs(borne - degres) > TOLERANCE_BUTEE_DEG


def dans_les_limites(sid: int, degres: float) -> bool:
    """Vrai si cet angle servo est dans la plage autorisée."""
    _, bride = limiter(sid, degres)
    return not bride

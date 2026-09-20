#!/usr/bin/env python3
"""Leçon exécutable : comment marche une simulation de bras robotisé.

À lancer depuis la racine du dépôt :

    venv/bin/python comprendre_simulation.py

Ce fichier n'est utilisé par rien d'autre : il est là pour être lu et exécuté,
pas pour faire tourner le bras. Le vrai code de simulation est dans
materiel/simulation.py — beaucoup plus court qu'on ne le croit, une fois qu'on
a compris ce qui suit.

L'idée à retenir, et tout le reste en découle :

    Une simulation, c'est trois choses.
      1. UN ÉTAT      : quelques nombres qui décrivent le système.
      2. DES RÈGLES   : comment ces nombres changent quand on commande.
      3. UN REGARD    : comment on transforme ces nombres en quelque chose
                        d'observable (une position, une image).

    Le bras réel a exactement les mêmes trois choses. La seule différence est
    l'endroit où vit l'état : dans le monde physique pour le vrai bras (on le
    lit sur le port série), dans une variable Python pour la simulation.

    C'est pour ça que le même code peut piloter les deux (materiel/interface.py).
"""

import numpy as np

from cinematique.ik import angles_pour, angles_utiles, bras, vecteur_complet
from materiel.etalonnage import AXES_BRAS, NOMS_ARM101, degres_vers_angles
from materiel.simulation import BrasSimule, squelette


def titre(numero, texte):
    print(f"\n{'=' * 70}\n  ÉTAPE {numero} — {texte}\n{'=' * 70}\n")


# ----------------------------------------------------------------------
def etape_1_etat():
    """L'état : la simulation entière tient dans six nombres."""
    titre(1, "L'ÉTAT : six nombres, et c'est tout")

    simu = BrasSimule(bavard=False)

    print("Voici l'intégralité de ce que la simulation « sait » du bras :\n")
    for sid, degres in simu.degres_servo().items():
        print(f"    axe {sid} ({NOMS_ARM101[sid]:<14}) = {degres:+8.2f}°")

    print("""
C'est un simple dictionnaire Python (`simu._degres`). Il n'y a pas de bras
caché quelque part : ces six angles SONT le bras simulé.

Le vrai bras a le même état, mais rangé ailleurs — dans la position physique de
ses six moteurs. `BrasServos.degres_servo()` va la lire sur le port série ;
`BrasSimule.degres_servo()` lit le dictionnaire. Même question, même réponse,
deux sources différentes.""")
    return simu


# ----------------------------------------------------------------------
def etape_2_regard(simu):
    """Le regard : transformer les angles en position. La cinématique directe."""
    titre(2, "LE REGARD : des angles vers une position (cinématique directe)")

    print("""Six angles, c'est bien, mais la question utile est « où est la pince ? ».
Y répondre s'appelle la CINÉMATIQUE DIRECTE (forward kinematics).

Le principe est mécanique, au sens propre. Le bras est une chaîne de segments
rigides. Chaque articulation ajoute deux choses :
    - une ROTATION   (l'angle du moteur) ;
    - une TRANSLATION (la longueur du segment qui suit, fixe).

On part de la base, et on avance segment par segment. Voici la chaîne, dans
l'ordre, pour la pose actuelle :\n""")

    angles = simu.lire_positions()
    points = squelette(angles)
    noms = [l.name for l in bras.links]

    precedent = None
    for nom, point in zip(noms, points):
        if precedent is None:
            print(f"    {nom:<20} ({point[0]:+.3f}, {point[1]:+.3f}, "
                  f"{point[2]:+.3f}) m   <- origine")
        else:
            longueur = np.linalg.norm(point - precedent) * 100
            print(f"    {nom:<20} ({point[0]:+.3f}, {point[1]:+.3f}, "
                  f"{point[2]:+.3f}) m   +{longueur:.1f} cm")
        precedent = point

    print(f"""
Le dernier point est la pince. La cinématique directe, c'est exactement ce
parcours : partir de (0, 0, 0) et enchaîner les étapes.

En pratique, chaque étape est une matrice 4x4 (une « transformation
homogène ») : elle range dans un seul objet la rotation et le déplacement. Les
enchaîner revient à les multiplier. Voici celle qui mène à la pince :\n""")

    matrice = bras.forward_kinematics(vecteur_complet(angles))
    for ligne in matrice:
        print("        [" + "  ".join(f"{v:+7.3f}" for v in ligne) + "]")

    position = f"({matrice[0, 3]:+.3f}, {matrice[1, 3]:+.3f}, {matrice[2, 3]:+.3f})"
    print(f"""
    Le bloc 3x3 en haut à gauche  = l'ORIENTATION de la pince.
    La colonne de droite (3 1ers) = sa POSITION : {position} m.
    La dernière ligne [0 0 0 1] est une commodité mathématique, qui permet
    justement d'écrire rotation + translation dans une seule multiplication.

Point important : cette étape est de la GÉOMÉTRIE PURE. Aucune notion de
temps, de masse ni de force n'intervient. On demande « si les moteurs sont à
ces angles, où est la pince ? », et la réponse est exacte — pas approchée.""")

    print("\n--- Rendons la chaîne tangible : bougeons UN seul moteur ---\n")
    for axe_bouge in AXES_BRAS:
        degres = simu.degres_servo(AXES_BRAS)
        avant = squelette(np.array(degres_vers_angles(degres)))[-1]
        degres[axe_bouge] += 10.0
        apres = squelette(np.array(degres_vers_angles(degres)))[-1]
        deplacement = np.linalg.norm(apres - avant) * 100
        print(f"    +10° sur l'axe {axe_bouge} ({NOMS_ARM101[axe_bouge]:<14})"
              f" -> la pince bouge de {deplacement:5.1f} cm")

    print("""
Les axes proches de la base déplacent beaucoup la pince, ceux du poignet peu :
ils portent moins de segments derrière eux. C'est ce déséquilibre qui explique
le défaut d'IK corrigé dans cinematique/ik.py — le roulis du poignet ne
déplaçant presque pas la pince, l'optimiseur le mettait n'importe où.""")


# ----------------------------------------------------------------------
def etape_3_regles():
    """Les règles : la boucle de simulation elle-même."""
    titre(3, "LES RÈGLES : la boucle de simulation")

    print("""Voilà le cœur de toute simulation, et il est décevant de simplicité :

    pour chaque petit pas du mouvement :
        etat = etat + (cible - etat) * fraction    # on avance un peu
        enregistrer(etat)                          # on garde la trace

C'est tout. Dans materiel/simulation.py, cela se lit dans `_executer` :
`interpoler()` fabrique la liste des étapes intermédiaires, puis la boucle
recopie chacune dans `self._degres`.

Pourquoi découper au lieu de sauter directement à la cible ? Deux raisons, et
la seconde est la vraie :
    - le tracé serait vide entre le départ et l'arrivée ;
    - SURTOUT, le bras réel, lui, DOIT y aller par paliers (10° à la fois,
      constante PALIER_DEG) pour rencontrer un obstacle à faible vitesse.
      La simulation reproduit ce découpage pour rester fidèle à son modèle.

Regardons un mouvement pas à pas :\n""")

    simu = BrasSimule(bavard=False)
    avant = simu.position_pince()
    simu.aller_a(0.36, 0.00, 0.22)

    print(f"    départ  : ({avant[0]:+.3f}, {avant[1]:+.3f}, {avant[2]:+.3f}) m")
    for i, pose in enumerate(simu.trajectoire[1:], start=1):
        point = squelette(np.array(degres_vers_angles(pose)))[-1]
        print(f"    pas {i}   : ({point[0]:+.3f}, {point[1]:+.3f}, "
              f"{point[2]:+.3f}) m")
    print("""
Chaque ligne est un état successif. Mises bout à bout, elles forment la
trajectoire — c'est exactement ce que dessine `tracer()`.""")


# ----------------------------------------------------------------------
def etape_4_inverse():
    """Le problème inverse : de la position vers les angles."""
    titre(4, "LE PROBLÈME INVERSE : d'une position vers les angles")

    print("""On ne veut pas commander des angles, on veut dire « va à cet endroit ».
Il faut donc faire le chemin DANS L'AUTRE SENS : de (x, y, z) vers les angles.
C'est la CINÉMATIQUE INVERSE (IK), et c'est beaucoup plus dur.

Pourquoi ? La cinématique directe a toujours UNE réponse, calculable d'un coup.
L'inverse peut en avoir zéro (cible hors de portée), une, plusieurs (coude en
haut ou coude en bas) — voire une infinité.

Faute de formule générale, ikpy procède par TÂTONNEMENT INTELLIGENT : il part
d'une pose, calcule où tombe la pince, mesure l'écart à la cible, corrige, et
recommence jusqu'à ce que l'écart soit négligeable. C'est de l'optimisation
numérique.

Deux conséquences pratiques, visibles dans ton code :
    - c'est LENT comparé à la directe (des millisecondes, pas des microsecondes) ;
    - ça peut ÉCHOUER en se coinçant dans un mauvais minimum local. D'où les
      6 points de départ différents de `_points_de_depart` dans cinematique/ik.py.

Vérifions les deux sens l'un après l'autre :\n""")

    cible = (0.30, 0.05, 0.15)
    solution = angles_pour(*cible)
    retour = bras.forward_kinematics(solution)[:3, 3]

    print(f"    cible demandée        : ({cible[0]:+.3f}, {cible[1]:+.3f}, "
          f"{cible[2]:+.3f}) m")
    print("    angles trouvés par l'IK :")
    for sid, angle in zip(AXES_BRAS, angles_utiles(solution)):
        print(f"        {NOMS_ARM101[sid]:<14} {np.degrees(angle):+8.2f}°")
    print(f"    ces angles repassés en directe : ({retour[0]:+.3f}, "
          f"{retour[1]:+.3f}, {retour[2]:+.3f}) m")
    print(f"    écart : {np.linalg.norm(retour - np.array(cible)) * 1000:.3f} mm"
          "  <- la boucle est bouclée")


# ----------------------------------------------------------------------
def etape_5_limites():
    """Ce que cette simulation ne sait pas faire."""
    titre(5, "CE QUE CETTE SIMULATION NE SAIT PAS FAIRE")

    print("""Tout ce qui précède est de la GÉOMÉTRIE. C'est une simulation
CINÉMATIQUE : elle répond à « où ? », jamais à « avec quelle force ? ».

Elle ignore donc complètement :
    - la GRAVITÉ        : le bras simulé tient dans n'importe quelle pose,
                          même celles où le vrai s'effondrerait ;
    - les CONTACTS      : rien n'empêche la pince de traverser la table ;
    - la MASSE et l'INERTIE : le mouvement est instantané et sans élan ;
    - la PRÉHENSION     : `pince_tient()` renvoie toujours False, faute
                          d'objet à serrer.

C'est pour cela que `charge()` renvoie toujours 0. Sur le bras réel, la charge
mesurée est le principal garde-fou : c'est elle qui détecte qu'on pousse contre
un obstacle. La simulation ne peut pas la fabriquer — il n'y a rien à pousser.

    >>> À retenir : une trajectoire validée en simulation peut parfaitement
    >>> traverser la table. La simulation valide la GÉOMÉTRIE et la LOGIQUE,
    >>> jamais la sûreté physique.

L'étage au-dessus s'appelle une simulation DYNAMIQUE (MuJoCo, Gazebo, Simscape).
Elle ajoute masses, forces et contacts, puis intègre les équations du mouvement
pas de temps par pas de temps. C'est plus fidèle, bien plus lourd, et cela
demande un modèle complet : masses, inerties, frottements, et les maillages 3D
qui manquent aujourd'hui dans urdf/assets/.

Ordre d'apprentissage conseillé : maîtriser d'abord la cinématique (ci-dessus),
la dynamique ensuite. Beaucoup de projets de saisie n'ont jamais besoin de la
seconde.""")


# ----------------------------------------------------------------------
def main():
    print(__doc__)
    simu = etape_1_etat()
    etape_2_regard(simu)
    etape_3_regles()
    etape_4_inverse()
    etape_5_limites()

    print(f"\n{'=' * 70}")
    print("""
POUR ALLER PLUS LOIN, dans l'ordre :

    materiel/interface.py     les 5 questions qu'on pose à un bras (44 lignes)
    materiel/simulation.py    la simulation complète
    cinematique/ik.py         directe et inverse
    cinematique/enveloppe.py  la cinématique directe réécrite à la main en
                              numpy, sans ikpy : très instructif à lire

Pour voir le bras bouger :

    venv/bin/python -m materiel.simulation
    venv/bin/python -m materiel.test_bras --simu --vue trajet.png aller 0.30 0.05 0.15
""")


if __name__ == "__main__":
    main()

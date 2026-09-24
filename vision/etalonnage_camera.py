#!/usr/bin/env python3
"""Étalonnage œil-main : relier les pixels de la caméra aux mètres du bras.

Le problème, et pourquoi il faut le résoudre
--------------------------------------------
La caméra sait dire « objet rouge au pixel (712, 383) ». Le bras sait aller en
« (0,26 ; -0,05 ; 0,03) mètres ». Rien ne relie les deux, et c'est ce qui
sépare encore ce dépôt d'une saisie guidée par la vision.

La méthode retenue : le bras est sa propre mire
-----------------------------------------------
Pas besoin d'imprimer un damier. On amène la pince à des positions que le
modèle connaît exactement, on photographie à chaque fois, et on pointe la pince
dans l'image. Chaque point donne une correspondance (pixel) <-> (mètres), et il
en faut au moins quatre.

C'est même préférable à une mire imprimée : une mire donne la relation
caméra <-> mire, qu'il reste à relier au bras. Ici on mesure directement la
relation caméra <-> bras, la seule qui serve.

Ce qu'on calcule : une homographie
----------------------------------
Les objets à saisir sont posés SUR LA TABLE, donc tous dans un même plan. Or la
transformation entre deux plans, vue par une caméra, est une homographie : une
matrice 3x3 qui change un pixel en position sur la table.

    [x]       [u]
    [y] ~ H . [v]
    [1]       [1]

Conséquence à connaître : cela ne vaut QUE pour le plan étalonné. Un objet posé
sur un livre sera localisé à côté — l'erreur grandit avec sa hauteur. Pour des
objets d'épaisseurs très variables, il faudrait une vraie calibration 3D de la
caméra ; pour des pièces posées à plat, l'homographie suffit et se mesure en
dix minutes.

Marche à suivre
---------------
1. Le bras parcourt une grille et photographie chaque position :

       venv/bin/python -m vision.etalonnage_camera capturer --z 0.05

2. Vous pointez la pince dans chaque image (fenêtre graphique, sur le bureau
   du Pi) :

       venv/bin/python -m vision.etalonnage_camera pointer

3. On ajuste l'homographie et on regarde l'erreur obtenue :

       venv/bin/python -m vision.etalonnage_camera ajuster

4. Vérification : cliquez un endroit de la table, le bras y va.

       venv/bin/python -m vision.etalonnage_camera viser --aller

À refaire dès que la caméra ou le bras bouge, ne serait-ce qu'un peu : c'est la
position relative des deux que l'on mesure ici.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

DOSSIER_DEFAUT = Path("etalonnage_camera")
FICHIER_RELEVE = "releve.json"
FICHIER_HOMOGRAPHIE = "homographie.json"

Z_DEFAUT = 0.05          # m, hauteur de la pince pendant l'étalonnage
DELAI_POSE = 0.4         # s d'immobilité avant la photo, pour ne pas la flouter

# Grille par défaut : large, mais entièrement dans l'enveloppe mesurée après le
# ré-étalonnage du 20 septembre (la zone utile a basculé du côté -y).
X_DEFAUT = (0.18, 0.34, 4)   # début, fin, nombre
Y_DEFAUT = (-0.18, 0.10, 4)


# ----------------------------------------------------------------------
# Géométrie
# ----------------------------------------------------------------------
def grille_atteignable(x_spec, y_spec, z: float):
    """Points (x, y) de la grille que le back-end accepterait vraiment.

    Filtre par les mêmes contrôles que `BrasServos.aller_a` (convergence de
    l'IK et butées mesurées), pour ne pas lancer le bras vers des cibles qu'il
    refusera. Les points sont rendus en serpentin : deux points consécutifs
    sont voisins, ce qui évite les grands écarts articulaires refusés par
    ECART_MAX_DEG.
    """
    from cinematique.ik import angles_utiles
    from cinematique.securite import ik_est_valide
    from materiel.etalonnage import AXES_BRAS, angles_vers_degres, limiter

    points = []
    for i, y in enumerate(np.linspace(*y_spec[:2], int(y_spec[2]))):
        xs = np.linspace(*x_spec[:2], int(x_spec[2]))
        if i % 2:                      # serpentin
            xs = xs[::-1]
        for x in xs:
            ok, solution, _, _ = ik_est_valide(float(x), float(y), z)
            if not ok:
                continue
            degres = angles_vers_degres(angles_utiles(solution))
            if any(limiter(sid, degres[sid])[1] for sid in AXES_BRAS):
                continue
            points.append((round(float(x), 4), round(float(y), 4)))
    return points


def pixel_vers_table(u: float, v: float, homographie) -> tuple:
    """Un pixel -> la position (x, y) en mètres dans le repère du bras."""
    import cv2

    source = np.array([[[float(u), float(v)]]], dtype=np.float64)
    x, y = cv2.perspectiveTransform(source, np.asarray(homographie))[0][0]
    return float(x), float(y)


def charger_homographie(dossier: Path = DOSSIER_DEFAUT):
    """Retourne (H, z) depuis le fichier d'étalonnage, ou lève FileNotFoundError."""
    chemin = Path(dossier) / FICHIER_HOMOGRAPHIE
    donnees = json.loads(chemin.read_text(encoding="utf-8"))
    return np.array(donnees["H"], dtype=np.float64), donnees["z"]


# ----------------------------------------------------------------------
# 1. Capture
# ----------------------------------------------------------------------
def cmd_capturer(args):
    """Promène la pince sur la grille et photographie chaque position."""
    import cv2

    from materiel.approche import aller_en_repassant_au_neutre
    from materiel.servos import BrasServos
    from materiel.simulation import BrasSimule
    from vision.camera import Camera

    dossier = Path(args.dossier)
    dossier.mkdir(parents=True, exist_ok=True)

    points = grille_atteignable(args.x, args.y, args.z)
    if len(points) < 4:
        print(f"Seulement {len(points)} points atteignables : il en faut au "
              "moins 4 pour une homographie. Élargissez la grille (--x, --y) "
              "ou changez --z.")
        return 1
    print(f"{len(points)} points atteignables à z = {args.z:.3f} m.\n")

    bras = BrasSimule(bavard=False) if args.simu \
        else BrasServos(args.port, bavard=False)
    releves = []
    try:
        with Camera() as camera:
            for i, (x, y) in enumerate(points):
                # Par étapes, et au besoin en repassant par la pose neutre :
                # le bras se coince sinon dans des configurations d'où la
                # cible suivante demanderait de retourner un axe de 150°.
                if not aller_en_repassant_au_neutre(bras, x, y, args.z,
                                                    bavard=False):
                    print(f"  {i:>2}. ({x:+.3f}, {y:+.3f}) inatteignable, ignoré")
                    continue
                time.sleep(DELAI_POSE)
                image = camera.capturer()
                nom = f"p{i:02d}.jpg"
                cv2.imwrite(str(dossier / nom), image)
                reelle = bras.position_pince()
                releves.append({"fichier": nom,
                                "x": round(float(reelle[0]), 4),
                                "y": round(float(reelle[1]), 4),
                                "z": round(float(reelle[2]), 4),
                                "u": None, "v": None})
                print(f"  {i:>2}. ({x:+.3f}, {y:+.3f}) -> {nom}")
    except KeyboardInterrupt:
        bras.stop()
        print("\nInterrompu : le bras est figé sur place.")
    finally:
        bras.close()

    if len(releves) < 4:
        print(f"\n{len(releves)} photos seulement : insuffisant.")
        return 1

    (dossier / FICHIER_RELEVE).write_text(json.dumps(
        {"date": time.strftime("%Y-%m-%d %H:%M:%S"), "z": args.z,
         "points": releves}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")
    print(f"\n{len(releves)} photos dans {dossier}/. "
          f"Étape suivante :\n    venv/bin/python -m vision.etalonnage_camera pointer")
    return 0


# ----------------------------------------------------------------------
# 0. Contrôle du cadrage
# ----------------------------------------------------------------------
def cmd_cadrage(args):
    """Promène la pince aux coins de la zone et en fait une planche-contact.

    À lancer avant `capturer`, et après chaque déplacement de la caméra : rien
    n'est plus frustrant que de dérouler quinze points pour s'apercevoir à la
    fin que la pince sortait du champ. Cinq positions suffisent à le voir.

    Les vignettes sont assemblées en une seule image, plus simple à regarder
    de loin qu'une poignée de fichiers séparés.
    """
    import cv2

    from materiel.approche import aller_par_etapes
    from materiel.servos import BrasServos
    from materiel.simulation import BrasSimule
    from vision.camera import Camera

    x0, x1 = args.x[0], args.x[1]
    y0, y1 = args.y[0], args.y[1]
    stations = [("avant-gauche", x0, y0), ("avant-droit", x1, y0),
                ("centre", (x0 + x1) / 2, (y0 + y1) / 2),
                ("arriere-droit", x1, y1), ("arriere-gauche", x0, y1)]

    bras = BrasSimule(bavard=False) if args.simu else BrasServos(args.port,
                                                                 bavard=False)
    vignettes = []
    try:
        with Camera() as camera:
            for nom, x, y in stations:
                if not aller_par_etapes(bras, x, y, args.z, bavard=False):
                    print(f"  {nom:<14} ({x:+.2f}, {y:+.2f}) inatteignable")
                    continue
                time.sleep(DELAI_POSE)
                image = camera.capturer()
                reelle = bras.position_pince()
                cv2.putText(image, f"{nom}  ({reelle[0]:+.2f}, {reelle[1]:+.2f})",
                            (16, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1,
                            (0, 0, 0), 6)
                cv2.putText(image, f"{nom}  ({reelle[0]:+.2f}, {reelle[1]:+.2f})",
                            (16, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1,
                            (0, 255, 255), 2)
                vignettes.append(image)
                print(f"  {nom:<14} ({reelle[0]:+.3f}, {reelle[1]:+.3f}) photographié")
    except KeyboardInterrupt:
        bras.stop()
        print("\nInterrompu : le bras est figé sur place.")
    finally:
        bras.close()

    if not vignettes:
        print("Aucune position atteinte.")
        return 1

    # Planche-contact : deux colonnes, demi-résolution.
    reduites = [cv2.resize(v, (v.shape[1] // 2, v.shape[0] // 2))
                for v in vignettes]
    h, l = reduites[0].shape[:2]
    lignes = []
    for i in range(0, len(reduites), 2):
        paire = reduites[i:i + 2]
        if len(paire) == 1:
            paire.append(np.zeros_like(reduites[0]))
        lignes.append(np.hstack(paire))
    planche = np.vstack(lignes)

    chemin = Path(args.sortie)
    chemin.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(chemin), planche)
    print(f"\nPlanche-contact ({len(vignettes)} vues) : {chemin}")
    print("La pince doit être visible ET entière sur chacune, sans toucher "
          "les bords.")
    return 0


# ----------------------------------------------------------------------
# 2. Pointage
# ----------------------------------------------------------------------
def cmd_pointer(args):
    """Affiche chaque photo et enregistre le clic sur le point de préhension."""
    import cv2

    dossier = Path(args.dossier)
    chemin = dossier / FICHIER_RELEVE
    if not chemin.exists():
        print(f"{chemin} introuvable — lancez d'abord `capturer`.")
        return 1
    releve = json.loads(chemin.read_text(encoding="utf-8"))

    print("""
Cliquez le POINT DE PRÉHENSION : le milieu de l'ouverture des mors, là où un
objet serait pincé — pas le corps du servo ni le bout d'un mors.

    clic      place le repère          n / Entrée   valider et passer au suivant
    r         effacer le clic          s            sauter cette image
    q         quitter en enregistrant
""")

    clic = {"point": None}

    def sur_souris(evenement, x, y, drapeaux, donnees):
        if evenement == cv2.EVENT_LBUTTONDOWN:
            clic["point"] = (x, y)

    fenetre = "Etalonnage - cliquez la pince"
    cv2.namedWindow(fenetre, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(fenetre, sur_souris)

    try:
        for point in releve["points"]:
            if point["u"] is not None and not args.tout:
                continue
            image = cv2.imread(str(dossier / point["fichier"]))
            if image is None:
                print(f"  {point['fichier']} illisible, ignoré")
                continue
            clic["point"] = (point["u"], point["v"]) \
                if point["u"] is not None else None

            while True:
                vue = image.copy()
                titre = (f"{point['fichier']}   cible ({point['x']:+.3f}, "
                         f"{point['y']:+.3f}) m")
                cv2.putText(vue, titre, (12, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 255, 0), 2)
                if clic["point"]:
                    u, v = clic["point"]
                    cv2.drawMarker(vue, (int(u), int(v)), (0, 0, 255),
                                   cv2.MARKER_CROSS, 28, 2)
                    cv2.circle(vue, (int(u), int(v)), 14, (0, 0, 255), 1)
                cv2.imshow(fenetre, vue)

                touche = cv2.waitKey(30) & 0xFF
                if touche in (ord("n"), 13):
                    if clic["point"]:
                        point["u"], point["v"] = clic["point"]
                    break
                if touche == ord("r"):
                    clic["point"] = None
                if touche == ord("s"):
                    break
                if touche == ord("q"):
                    if clic["point"]:
                        point["u"], point["v"] = clic["point"]
                    raise KeyboardInterrupt
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()

    chemin.write_text(json.dumps(releve, indent=2, ensure_ascii=False) + "\n",
                      encoding="utf-8")
    pointes = sum(1 for p in releve["points"] if p["u"] is not None)
    print(f"\n{pointes}/{len(releve['points'])} points pointés, "
          f"enregistrés dans {chemin}.")
    if pointes >= 4:
        print("    venv/bin/python -m vision.etalonnage_camera ajuster")
    else:
        print("Il en faut au moins 4 pour ajuster l'homographie.")
    return 0


# ----------------------------------------------------------------------
# 2 bis. Import de pixels pointés ailleurs
# ----------------------------------------------------------------------
def cmd_importer(args):
    """Reporte dans le relevé des pixels pointés hors de cette machine.

    `pointer` ouvre une fenêtre OpenCV, ce qui suppose un écran : inutilisable
    en SSH, et le Pi tourne sous Wayland. On peut alors pointer depuis un
    navigateur et rapatrier le résultat ici.

    Le JSON attendu est une liste d'objets {fichier, u, v} ; les entrées dont
    le fichier ne figure pas dans le relevé sont signalées et ignorées.
    """
    dossier = Path(args.dossier)
    chemin = dossier / FICHIER_RELEVE
    if not chemin.exists():
        print(f"{chemin} introuvable — lancez d'abord `capturer`.")
        return 1

    brut = sys.stdin.read() if args.source == "-" \
        else Path(args.source).read_text(encoding="utf-8")
    try:
        entrees = json.loads(brut)
    except json.JSONDecodeError as erreur:
        print(f"JSON illisible : {erreur}")
        return 1
    if isinstance(entrees, dict):          # tolère {"points": [...]}
        entrees = entrees.get("points", [])

    releve = json.loads(chemin.read_text(encoding="utf-8"))
    par_fichier = {p["fichier"]: p for p in releve["points"]}

    repris, inconnus = 0, []
    for entree in entrees:
        point = par_fichier.get(entree.get("fichier"))
        if point is None:
            inconnus.append(entree.get("fichier"))
            continue
        if entree.get("u") is None:
            continue
        point["u"], point["v"] = int(entree["u"]), int(entree["v"])
        repris += 1

    chemin.write_text(json.dumps(releve, indent=2, ensure_ascii=False) + "\n",
                      encoding="utf-8")
    if inconnus:
        print(f"Ignorés, absents du relevé : {', '.join(map(str, inconnus))}")
    total = sum(1 for p in releve["points"] if p["u"] is not None)
    print(f"{repris} pixels importés. {total}/{len(releve['points'])} points "
          "pointés au total.")
    if total >= 4:
        print("    venv/bin/python -m vision.etalonnage_camera ajuster")
    else:
        print("Il en faut au moins 4 pour ajuster l'homographie.")
    return 0


# ----------------------------------------------------------------------
# 3. Ajustement
# ----------------------------------------------------------------------
def cmd_ajuster(args):
    """Calcule l'homographie pixels -> table et mesure l'erreur obtenue."""
    import cv2

    dossier = Path(args.dossier)
    releve = json.loads((dossier / FICHIER_RELEVE).read_text(encoding="utf-8"))
    retenus = [p for p in releve["points"] if p["u"] is not None]
    if len(retenus) < 4:
        print(f"{len(retenus)} points pointés : il en faut au moins 4.")
        return 1

    pixels = np.array([[p["u"], p["v"]] for p in retenus], dtype=np.float64)
    metres = np.array([[p["x"], p["y"]] for p in retenus], dtype=np.float64)

    # Seuil RANSAC exprimé dans l'unité d'arrivée, donc en mètres. Il ne doit
    # écarter que les vraies bévues — un clic sur la mauvaise pièce — et non le
    # bruit ordinaire du pointage.
    #
    # Réglé à 8 mm au premier essai, il écartait 4 points sur 13 du relevé du
    # 22 septembre. À tort : l'ajustement sur les 9 survivants donnait certes
    # 3,3 mm sur eux-mêmes, mais 6,3 mm sur les 13, là où l'ajustement gardant
    # tout donne 5,9 mm. Le seuil trop serré coûtait donc de la justesse sur
    # l'ensemble de la zone, en plus de laisser un coin sans aucun point.
    #
    # 20 mm garde le bruit de pointage et rejette toujours une vraie bévue (un
    # clic faux de 90 px, éprouvé sur données synthétiques, ressort à 53 mm).
    SEUIL_ABERRANT_M = 0.020
    homographie, masque = cv2.findHomography(pixels, metres, cv2.RANSAC,
                                             SEUIL_ABERRANT_M)
    if homographie is None:
        print("Ajustement impossible : les points sont peut-être alignés. "
              "Reprenez une grille plus étalée.")
        return 1

    projetes = cv2.perspectiveTransform(pixels.reshape(-1, 1, 2),
                                        homographie).reshape(-1, 2)
    erreurs_mm = np.linalg.norm(projetes - metres, axis=1) * 1000
    garde = masque.ravel().astype(bool)

    print(f"{len(retenus)} points pointés, {int(garde.sum())} retenus par RANSAC.\n")
    print(f"{'fichier':<10} {'cible (m)':>18} {'reprojeté (m)':>18} {'écart':>9}")
    for point, projete, erreur, ok in zip(retenus, projetes, erreurs_mm, garde):
        marque = "" if ok else "  écarté"
        print(f"{point['fichier']:<10} ({point['x']:+.3f}, {point['y']:+.3f}) "
              f"  ({projete[0]:+.3f}, {projete[1]:+.3f}) "
              f"{erreur:>7.1f} mm{marque}")

    retenues = erreurs_mm[garde]
    print(f"\nErreur de reprojection sur les points retenus : "
          f"moyenne {retenues.mean():.1f} mm, max {retenues.max():.1f} mm")
    if retenues.mean() < 5:
        print("Excellent : bien en deçà de l'ouverture de la pince.")
    elif retenues.mean() < 15:
        print("Correct pour saisir un objet de quelques centimètres.")
    else:
        print("Trop imprécis. Causes habituelles : clics approximatifs, points "
              "trop groupés, ou caméra/bras déplacés entre deux photos.")

    (dossier / FICHIER_HOMOGRAPHIE).write_text(json.dumps(
        {"date": time.strftime("%Y-%m-%d %H:%M:%S"),
         "z": releve["z"],
         "n_points": int(garde.sum()),
         "erreur_mm": {"moyenne": round(float(retenues.mean()), 2),
                       "max": round(float(retenues.max()), 2)},
         "H": homographie.tolist()}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")
    print(f"\nHomographie enregistrée dans {dossier / FICHIER_HOMOGRAPHIE}")
    return 0


# ----------------------------------------------------------------------
# 4. Vérification
# ----------------------------------------------------------------------
def cmd_viser(args):
    """Convertit un pixel en position du bras, et peut y envoyer la pince."""
    import cv2

    dossier = Path(args.dossier)
    try:
        homographie, z_etalonnage = charger_homographie(dossier)
    except FileNotFoundError:
        print("Pas d'homographie : lancez `ajuster` d'abord.")
        return 1

    if args.u is not None and args.v is not None:
        u, v = args.u, args.v
    else:
        from vision.camera import Camera
        with Camera() as camera:
            image = camera.capturer()
        clic = {"point": None}

        def sur_souris(evenement, x, y, drapeaux, donnees):
            if evenement == cv2.EVENT_LBUTTONDOWN:
                clic["point"] = (x, y)

        fenetre = "Cliquez une cible sur la table"
        cv2.namedWindow(fenetre, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(fenetre, sur_souris)
        print("Cliquez un point de la table, puis Entrée. q pour renoncer.")
        while True:
            vue = image.copy()
            if clic["point"]:
                cv2.drawMarker(vue, clic["point"], (0, 0, 255),
                               cv2.MARKER_CROSS, 28, 2)
            cv2.imshow(fenetre, vue)
            touche = cv2.waitKey(30) & 0xFF
            if touche in (ord("n"), 13) and clic["point"]:
                break
            if touche == ord("q"):
                cv2.destroyAllWindows()
                return 0
        cv2.destroyAllWindows()
        u, v = clic["point"]

    x, y = pixel_vers_table(u, v, homographie)
    z = args.z if args.z is not None else z_etalonnage
    print(f"pixel ({u}, {v})  ->  x {x:+.3f}  y {y:+.3f}  (m), à z {z:+.3f}")

    if not args.aller:
        print("Ajoutez --aller pour y envoyer la pince.")
        return 0

    from materiel.approche import aller_par_etapes
    from materiel.servos import BrasServos
    from materiel.simulation import BrasSimule

    bras = BrasSimule(bavard=True) if args.simu else BrasServos(args.port)
    try:
        # Par étapes : la cible est souvent loin de la pose où le bras attend,
        # et un saut direct serait refusé par la limite d'écart articulaire.
        if aller_par_etapes(bras, x, y, z):
            atteinte = bras.position_pince()
            print(f"Pince en ({atteinte[0]:+.3f}, {atteinte[1]:+.3f}, "
                  f"{atteinte[2]:+.3f}) m. Comparez avec l'endroit cliqué : "
                  "c'est la mesure honnête de votre étalonnage.")
        else:
            print("Mouvement refusé — voir le message ci-dessus.")
    except KeyboardInterrupt:
        bras.stop()
        return 130
    finally:
        bras.close()
    return 0


# ----------------------------------------------------------------------
def construire_parseur():
    parseur = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parseur.add_argument("--dossier", default=str(DOSSIER_DEFAUT))
    parseur.add_argument("--port", default="/dev/ttyACM0")
    parseur.add_argument("--simu", action="store_true",
                         help="utilise le bras simulé (les photos n'auront "
                              "alors aucun rapport avec la pose : pour "
                              "éprouver la mécanique de l'outil, pas pour "
                              "étalonner)")
    sous = parseur.add_subparsers(dest="commande", required=True)

    p = sous.add_parser("cadrage",
                        help="vérifie que la zone tient dans le champ")
    p.add_argument("--z", type=float, default=Z_DEFAUT)
    p.add_argument("--x", type=float, nargs=2, default=X_DEFAUT[:2],
                   metavar=("DEBUT", "FIN"))
    p.add_argument("--y", type=float, nargs=2, default=Y_DEFAUT[:2],
                   metavar=("DEBUT", "FIN"))
    p.add_argument("--sortie", default="cadrage.jpg")
    p.set_defaults(fonction=cmd_cadrage)

    p = sous.add_parser("capturer", help="parcourt la grille et photographie")
    p.add_argument("--z", type=float, default=Z_DEFAUT,
                   help=f"hauteur de la pince, en m (défaut {Z_DEFAUT})")
    p.add_argument("--x", type=float, nargs=3, default=X_DEFAUT,
                   metavar=("DEBUT", "FIN", "N"))
    p.add_argument("--y", type=float, nargs=3, default=Y_DEFAUT,
                   metavar=("DEBUT", "FIN", "N"))
    p.set_defaults(fonction=cmd_capturer)

    p = sous.add_parser("pointer", help="cliquez la pince dans chaque photo")
    p.add_argument("--tout", action="store_true",
                   help="repasser aussi sur les points déjà pointés")
    p.set_defaults(fonction=cmd_pointer)

    p = sous.add_parser("importer",
                        help="reprend des pixels pointés ailleurs (page web)")
    p.add_argument("source", help="fichier JSON, ou - pour l'entrée standard")
    p.set_defaults(fonction=cmd_importer)

    p = sous.add_parser("ajuster", help="calcule l'homographie et son erreur")
    p.set_defaults(fonction=cmd_ajuster)

    p = sous.add_parser("viser", help="pixel -> position du bras")
    p.add_argument("u", type=int, nargs="?")
    p.add_argument("v", type=int, nargs="?")
    p.add_argument("--z", type=float, default=None,
                   help="hauteur visée (défaut : celle de l'étalonnage)")
    p.add_argument("--aller", action="store_true",
                   help="envoyer réellement la pince à l'endroit calculé")
    p.set_defaults(fonction=cmd_viser)

    return parseur


def main():
    args = construire_parseur().parse_args()
    return args.fonction(args)


if __name__ == "__main__":
    sys.exit(main())

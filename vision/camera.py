#!/usr/bin/env python3
"""Accès à la caméra du Raspberry Pi, partagé par tout le code de vision.

Un seul endroit sait comment ouvrir la Picamera2 et quelles options lui donner,
pour que l'étalonnage, la détection et les futures applications voient
exactement la même image — une différence de résolution ou de format suffirait
à invalider un étalonnage.

    from vision.camera import Camera

    with Camera() as camera:
        image = camera.capturer()

Note de format, héritée de detection_picam.py : on demande du RGB888, qu'OpenCV
interprète comme du BGR. Les couleurs sont donc permutées par rapport à la
réalité, sans aucune conséquence sur la géométrie — et les seuils HSV existants
sont réglés sur cette convention. Ne pas « corriger » sans refaire les seuils.
"""

import time

import numpy as np

LARGEUR = 1280
HAUTEUR = 720
DELAI_STABILISATION = 2.0  # s : laisser l'exposition se faire, cf. detection_picam

# Mode capteur imposé, et c'est le réglage le plus important de ce module.
#
# L'imx708 du Pi expose trois modes : 1536x864 recadré sur (768, 432, 3072,
# 1728), puis 2304x1296 et 4608x2592 qui lisent les 4608x2592 entiers. Demander
# une sortie de 1280x720 fait choisir le premier — donc un recadrage central
# sur les deux tiers du capteur, et un champ de vision amputé d'autant, sans
# que rien ne le signale.
#
# Mesuré le 24 septembre 2026 : en forçant le mode pleine surface, le champ
# gagne un facteur 1,5 dans chaque direction. La base du bras et son support,
# jusque-là hors cadre, y entrent sans toucher à la caméra.
#
# On impose donc le mode par `raw`, ce qui laisse `main` sortir une image
# légère pour le traitement : grand champ ET image maniable.
MODE_CAPTEUR_PLEIN = (2304, 1296)


class Camera:
    """Caméra du Pi, ouverte une fois et réutilisée."""

    def __init__(self, largeur: int = LARGEUR, hauteur: int = HAUTEUR,
                 stabilisation: float = DELAI_STABILISATION,
                 champ_plein: bool = True):
        from picamera2 import Picamera2

        self.largeur = largeur
        self.hauteur = hauteur
        self._picam = Picamera2()
        options = {"main": {"size": (largeur, hauteur), "format": "RGB888"}}
        if champ_plein:
            options["raw"] = {"size": MODE_CAPTEUR_PLEIN}
        self._picam.configure(self._picam.create_video_configuration(**options))
        self._picam.start()
        time.sleep(stabilisation)

    def capturer(self) -> np.ndarray:
        """Une image, tableau (hauteur, largeur, 3) en convention BGR d'OpenCV."""
        return self._picam.capture_array()

    def close(self) -> None:
        """Arrête le flux ET rend le périphérique.

        `stop()` seul laisse la caméra ouverte : une deuxième instance dans le
        même processus échoue alors sur « Camera __init__ sequence did not
        complete ». Il faut donc aussi `close()`.
        """
        self._picam.stop()
        self._picam.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

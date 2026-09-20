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


class Camera:
    """Caméra du Pi, ouverte une fois et réutilisée."""

    def __init__(self, largeur: int = LARGEUR, hauteur: int = HAUTEUR,
                 stabilisation: float = DELAI_STABILISATION):
        from picamera2 import Picamera2

        self.largeur = largeur
        self.hauteur = hauteur
        self._picam = Picamera2()
        self._picam.configure(self._picam.create_video_configuration(
            main={"size": (largeur, hauteur), "format": "RGB888"}))
        self._picam.start()
        time.sleep(stabilisation)

    def capturer(self) -> np.ndarray:
        """Une image, tableau (hauteur, largeur, 3) en convention BGR d'OpenCV."""
        return self._picam.capture_array()

    def close(self) -> None:
        self._picam.stop()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

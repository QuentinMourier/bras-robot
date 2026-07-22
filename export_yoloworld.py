"""Ré-exporte le modèle YOLO-World avec une nouvelle liste de classes cibles.

Usage : python3 export_yoloworld.py bottle cup screwdriver
Écrase yolov8s-worldv2_192.onnx (utilisé par stream_yoloworld.py).
Arrêter le stream avant de relancer l'export (le fichier .onnx est réécrit).
"""

import sys
from pathlib import Path

from ultralytics import YOLO

CIBLE = "yolov8s-worldv2_192.onnx"

if len(sys.argv) < 2:
    print("Usage: python3 export_yoloworld.py <classe1> <classe2> ...")
    sys.exit(1)

classes = sys.argv[1:]
print("Classes cibles :", classes)

modele = YOLO("yolov8s-worldv2.pt")
modele.set_classes(classes)
chemin = Path(modele.export(format="onnx", imgsz=192))
chemin.rename(CIBLE)
print("Export terminé :", CIBLE)

import time
import threading

import cv2
from flask import Flask, Response
from picamera2 import Picamera2
from libcamera import controls
from ultralytics import YOLO

# Modèle YOLO-World exporté en ONNX à 192px, avec les classes ci-dessous
# "gravées" dans les poids au moment de l'export (voir export_yoloworld.py).
# Pour changer les objets cibles : relancer l'export avec une nouvelle liste.
MODELE = "yolov8s-worldv2_192.onnx"
TAILLE_IMG = 192
SEUIL_CONFIANCE = 0.3

app = Flask(__name__)

modele = YOLO(MODELE, task="detect")

picam = Picamera2()
config = picam.create_video_configuration(main={"size": (640, 480), "format": "RGB888"})
picam.configure(config)
picam.start()
picam.set_controls({"AfMode": controls.AfModeEnum.Continuous})
time.sleep(2)

derniere_frame = None
verrou = threading.Lock()


def boucle_capture():
    global derniere_frame
    t0, n = time.time(), 0
    while True:
        frame = picam.capture_array()

        resultats = modele.predict(
            frame, imgsz=TAILLE_IMG, conf=SEUIL_CONFIANCE, verbose=False
        )[0]
        frame_annotee = resultats.plot()

        n += 1
        fps = n / (time.time() - t0)
        cv2.putText(
            frame_annotee, f"{fps:.1f} FPS", (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2,
        )

        with verrou:
            derniere_frame = frame_annotee


def generer_flux():
    while True:
        with verrou:
            if derniere_frame is None:
                continue
            ok, jpeg = cv2.imencode(".jpg", derniere_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n")
        time.sleep(0.02)


@app.route("/")
def index():
    return """<!DOCTYPE html>
<html>
<head>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  body { margin:0; background:#111; display:flex;
         justify-content:center; align-items:center; min-height:100vh; }
  img  { width:100%; height:auto; }
</style>
</head>
<body><img src="/video"></body>
</html>"""


@app.route("/video")
def video():
    return Response(generer_flux(), mimetype="multipart/x-mixed-replace; boundary=frame")


if __name__ == "__main__":
    threading.Thread(target=boucle_capture, daemon=True).start()
    app.run(host="0.0.0.0", port=5002, threaded=True)

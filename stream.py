import cv2
import numpy as np
import time
import threading
from flask import Flask, Response
from picamera2 import Picamera2
from libcamera import controls

# --- Détection couleur (mêmes réglages que detection_picam.py) ---
BAS1, HAUT1 = np.array([0, 120, 80]),  np.array([10, 255, 255])
BAS2, HAUT2 = np.array([170, 120, 80]), np.array([180, 255, 255])
SURFACE_MIN = 500

app = Flask(__name__)

picam = Picamera2()
config = picam.create_video_configuration(main={"size": (768, 432), "format": "RGB888"})
picam.configure(config)
picam.start()
picam.set_controls({"AfMode": controls.AfModeEnum.Continuous})
time.sleep(2)

# La frame annotée la plus récente, partagée entre la boucle de capture et le serveur
derniere_frame = None
verrou = threading.Lock()

def boucle_capture():
    global derniere_frame
    t0, n = time.time(), 0
    while True:
        frame = picam.capture_array()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        masque = cv2.inRange(hsv, BAS1, HAUT1) | cv2.inRange(hsv, BAS2, HAUT2)
        masque = cv2.morphologyEx(masque, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        contours, _ = cv2.findContours(masque, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        n += 1
        fps = n / (time.time() - t0)

        if contours:
            c = max(contours, key=cv2.contourArea)
            if cv2.contourArea(c) > SURFACE_MIN:
                M = cv2.moments(c)
                cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
                cv2.circle(frame, (cx, cy), 12, (0, 255, 0), 2)
                cv2.drawContours(frame, [c], -1, (0, 255, 0), 1)
                cv2.putText(frame, f"({cx},{cy})", (cx + 15, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        cv2.putText(frame, f"{fps:.1f} FPS", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        with verrou:
            derniere_frame = frame

def generer_flux():
    while True:
        with verrou:
            if derniere_frame is None:
                continue
            ok, jpeg = cv2.imencode(".jpg", derniere_frame,
                                    [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n")
        time.sleep(0.03)  # ~30 FPS max côté navigateur

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
    return Response(generer_flux(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    threading.Thread(target=boucle_capture, daemon=True).start()
    app.run(host="0.0.0.0", port=5000, threaded=True)

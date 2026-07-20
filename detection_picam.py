import cv2
import numpy as np
import time
from picamera2 import Picamera2

# Plages HSV pour un objet ROUGE
BAS1, HAUT1 = np.array([0, 120, 80]),  np.array([10, 255, 255])
BAS2, HAUT2 = np.array([170, 120, 80]), np.array([180, 255, 255])
SURFACE_MIN = 500

picam = Picamera2()
config = picam.create_video_configuration(
    main={"size": (768, 432), "format": "RGB888"}
)
picam.configure(config)
picam.start()
time.sleep(2)  # laisser l'expo se stabiliser (la leçon de la webcam !)

t0, n = time.time(), 0
try:
    while True:
        frame = picam.capture_array()  # RGB888 -> OpenCV le traite comme BGR, ok pour HSV ci-dessous
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
                print(f"Objet en ({cx}, {cy})  surface={int(cv2.contourArea(c))}  {fps:.1f} FPS")
                # Sauvegarde d'une image annotée toutes les ~2 s pour contrôle visuel
                if n % 60 == 0:
                    cv2.circle(frame, (cx, cy), 10, (0, 255, 0), 2)
                    cv2.imwrite("detection.jpg", frame)
                continue
        print(f"Rien détecté  {fps:.1f} FPS")
except KeyboardInterrupt:
    print("\nArrêt.")
picam.stop()

import cv2

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

ok, frame = cap.read()
if not ok:
    print("Échec de capture — caméra non détectée ?")
else:
    cv2.imwrite("capture.jpg", frame)
    print(f"Image capturée : {frame.shape[1]}x{frame.shape[0]}")
cap.release()

import cv2
import os


print("RUNNING:", os.path.abspath(__file__))


# Load the pre-trained face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Open camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

print("Press Q to quit.")

# ---- distance calibration (set these once) ----
D_ref = 0.60   # meters: stand this far from camera
w_ref = 180    # pixels: face box width observed at D_ref

# smoothing for distance
alpha = 0.25
D_smooth = {}



while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    H, W = frame.shape[:2]

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50)
    )

    # sort left-to-right so Face 1 is roughly stable
    faces_sorted = sorted(faces, key=lambda r: r[0])

    for i, (x, y, w, h) in enumerate(faces_sorted, start=1):
        cx = x + w // 2
        cy = y + h // 2

        nx = (cx - W / 2) / (W / 2)
        ny = (cy - H / 2) / (H / 2)

        # distance estimate
        D_est = D_ref * (w_ref / max(w, 1))

        prev = D_smooth.get(i, D_est)
        D_est_smoothed = alpha * D_est + (1 - alpha) * prev
        D_smooth[i] = D_est_smoothed

        # draw
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.circle(frame, (cx, cy), 4, (255, 0, 0), -1)

        # labels
        label1 = f"Face {i}"
        label2 = f"cx,cy=({cx},{cy})  nx,ny=({nx:+.2f},{ny:+.2f})"
        label3 = f"dist~{D_est_smoothed:.2f} m"

        y0 = max(y - 55, 20)
        cv2.putText(frame, label1, (x, y0),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, label2, (x, y0 + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        cv2.putText(frame, label3, (x, y0 + 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    cv2.imshow("Face Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

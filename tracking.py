import cv2
import time
from collections import deque
import math

# Load the pre-trained face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Open camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

print("Press Q to quit.")

# ---- distance calibration (set these once) ----
D_ref = 0.40   # meters: stand this far from camera
w_ref = 180    # pixels: face box width observed at D_ref

# smoothing for distance
alpha = 0.25

# ---- window (NOT fullscreen) ----
window_name = "Face Tracking"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)  # resizable window
cv2.resizeWindow(window_name, 960, 540)          # pick a comfortable size

# ---- simple multi-face tracking (stable trails & speed) ----
MAX_MATCH_DIST_PX = 80          # match threshold
TRACK_TTL_SEC = 0.8             # drop tracks not seen for this long
TRAIL_LEN = 40                  # number of past points kept

tracks = {}        # track_id -> state dict
next_track_id = 1  # increasing ID counter


def dist2(a, b):
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return dx * dx + dy * dy


while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    now = time.time()
    H, W = frame.shape[:2]

    # Convert to grayscale for detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces (returns x, y, w, h in pixel coordinates)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50)
    )

    # Build detections as (center, bbox)
    detections = []
    for (x, y, w, h) in faces:
        cx = x + w // 2
        cy = y + h // 2
        detections.append(((cx, cy), (x, y, w, h)))

    # --- Match detections to existing tracks (nearest-neighbor greedy) ---
    unmatched_det = set(range(len(detections)))
    for tid in list(tracks.keys()):
        if not unmatched_det:
            break

        prev_center = tracks[tid]["center"]

        best_j = None
        best_d2 = None
        for j in unmatched_det:
            c, _ = detections[j]
            d2 = dist2(prev_center, c)
            if best_d2 is None or d2 < best_d2:
                best_d2 = d2
                best_j = j

        if best_j is not None and best_d2 is not None and best_d2 <= MAX_MATCH_DIST_PX**2:
            (cx, cy), (x, y, w, h) = detections[best_j]
            unmatched_det.remove(best_j)

            dt = max(now - tracks[tid]["last_t"], 1e-6)
            dx = cx - prev_center[0]
            dy = cy - prev_center[1]
            speed_px_s = math.sqrt(dx * dx + dy * dy) / dt

            nx = (cx - W / 2) / (W / 2)
            ny = (cy - H / 2) / (H / 2)

            # distance estimate + smoothing
            D_est = D_ref * (w_ref / max(w, 1))
            prev_D = tracks[tid].get("dist_smooth", D_est)
            D_smooth = alpha * D_est + (1 - alpha) * prev_D

            tracks[tid].update({
                "center": (cx, cy),
                "bbox": (x, y, w, h),
                "nx": nx,
                "ny": ny,
                "speed_px_s": speed_px_s,
                "dist_smooth": D_smooth,
                "last_t": now,
            })
            tracks[tid]["trail"].append((cx, cy))

    # Create new tracks for unmatched detections
    for j in list(unmatched_det):
        (cx, cy), (x, y, w, h) = detections[j]
        nx = (cx - W / 2) / (W / 2)
        ny = (cy - H / 2) / (H / 2)
        D_est = D_ref * (w_ref / max(w, 1))

        tracks[next_track_id] = {
            "center": (cx, cy),
            "bbox": (x, y, w, h),
            "nx": nx,
            "ny": ny,
            "speed_px_s": 0.0,
            "dist_smooth": D_est,
            "last_t": now,
            "trail": deque([(cx, cy)], maxlen=TRAIL_LEN),
        }
        next_track_id += 1

    # Drop old tracks
    for tid in list(tracks.keys()):
        if now - tracks[tid]["last_t"] > TRACK_TTL_SEC:
            del tracks[tid]

    # ============================================================
    # DISPLAY NUMBERING RESET EACH FRAME:
    # Face 1..N is assigned based on current visible tracks only.
    # ============================================================
    visible = sorted(tracks.items(), key=lambda kv: kv[1]["center"][0])  # left-to-right

    for face_num, (tid, tr) in enumerate(visible, start=1):
        x, y, w, h = tr["bbox"]
        cx, cy = tr["center"]

        # green face box + blue center
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.circle(frame, (cx, cy), 4, (255, 0, 0), -1)

        # red trail segments
        pts = list(tr["trail"])
        for k in range(1, len(pts)):
            cv2.line(frame, pts[k - 1], pts[k], (0, 0, 255), 2)

        # labels (Face numbers reset to 1..N each frame)
        label1 = f"Face {face_num}"
        label2 = f"cx,cy=({cx},{cy})  nx,ny=({tr['nx']:+.2f},{tr['ny']:+.2f})"
        label3 = f"dist~{tr['dist_smooth']:.2f} m   v~{tr['speed_px_s']:.0f} px/s"

        y0 = max(y - 55, 20)
        cv2.putText(frame, label1, (x, y0),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, label2, (x, y0 + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        cv2.putText(frame, label3, (x, y0 + 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    cv2.imshow(window_name, frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

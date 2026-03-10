import cv2
import numpy as np
import threading
import pickle
from ultralytics import YOLO

# ─── Load Calibration ─────────────────────────────────────────────────────────
with open("stereo_calib.pkl", "rb") as f:
    calib = pickle.load(f)

FOCAL_LENGTH    = calib["focal_px"]
BASELINE        = calib["baseline_m"]
map_lx, map_ly  = calib["map_lx"], calib["map_ly"]
map_rx, map_ry  = calib["map_rx"], calib["map_ry"]

# ─── Constants ────────────────────────────────────────────────────────────────
TARGET_CLASS        = 0
DETECT_EVERY        = 4
DETECT_W, DETECT_H  = 320, 240
FRAME_W,  FRAME_H   = 640, 480
scale_x = FRAME_W / DETECT_W
scale_y = FRAME_H / DETECT_H

# ─── Thread-safe Camera Buffer ────────────────────────────────────────────────
class CameraBuffer:
    def __init__(self, index):
        self.cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_W)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
        self.cap.set(cv2.CAP_PROP_FPS, 15)
        self.frame   = None
        self.lock    = threading.Lock()
        self.running = True
        threading.Thread(target=self._reader, daemon=True).start()

    def _reader(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.frame = frame

    def read(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def release(self):
        self.running = False
        self.cap.release()

# ─── Depth Helpers ────────────────────────────────────────────────────────────
stereo = cv2.StereoSGBM_create(
    minDisparity=0,
    numDisparities=128,
    blockSize=11,
    P1=8 * 3 * 11 ** 2,
    P2=32 * 3 * 11 ** 2,
    disp12MaxDiff=1,
    uniquenessRatio=10,
    speckleWindowSize=100,
    speckleRange=32,
    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
)

def compute_disparity(left_gray, right_gray):
    return stereo.compute(left_gray, right_gray).astype(np.float32) / 16.0

def disparity_to_depth(disparity_value):
    return (FOCAL_LENGTH * BASELINE) / disparity_value if disparity_value > 0 else None

def colorize_depth(disparity):
    norm = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX)
    return cv2.applyColorMap(np.uint8(norm), cv2.COLORMAP_MAGMA)

# ─── Annotate targets Only ────────────────────────────────────────────────────
def annotate_targets(frame, boxes, disparity):
    target_found = False
    for box in boxes:
        if int(box.cls[0]) != TARGET_CLASS:
            continue

        target_found = True
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])

        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        patch = disparity[max(0, cy-5):cy+5, max(0, cx-5):cx+5]
        valid = patch[patch > 0]
        depth_m = disparity_to_depth(float(np.median(valid))) if valid.size > 0 else None

        # ── Terminal output ──────────────────────────────────────────────────
        if depth_m:
            print(f"  target detected | conf: {conf:.2f} | distance: {depth_m:.2f}m")
        else:
            print(f"  target detected | conf: {conf:.2f} | distance: N/A")

        depth_str = f"{depth_m:.2f}m" if depth_m else "N/A"
        color = (0, 255, 0) if depth_m and depth_m < 2.0 else (0, 165, 255)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"target {conf:.2f} | {depth_str}",
                    (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    if not target_found:
        print("  No target detected", end="\r")  # overwrites same line, less spam

    return frame

# ─── Main ─────────────────────────────────────────────────────────────────────
model = YOLO("Yolo_models/yolov8n.pt")

left_buf  = CameraBuffer(2)
right_buf = CameraBuffer(4)

# Create window once outside the loop
cv2.namedWindow("target Detection + Depth", cv2.WINDOW_NORMAL)
cv2.resizeWindow("target Detection + Depth", FRAME_W * 3, FRAME_H)

frame_count = 0
last_boxes  = []

while True:
    left_frame  = left_buf.read()
    right_frame = right_buf.read()
    if left_frame is None or right_frame is None:
        continue

    # Rectify using calibration maps
    left_rect  = cv2.remap(left_frame,  map_lx, map_ly, cv2.INTER_LINEAR)
    right_rect = cv2.remap(right_frame, map_rx, map_ry, cv2.INTER_LINEAR)

    # Compute disparity on grayscale rectified frames
    lg = cv2.cvtColor(left_rect,  cv2.COLOR_BGR2GRAY)
    rg = cv2.cvtColor(right_rect, cv2.COLOR_BGR2GRAY)

    disparity   = compute_disparity(lg, rg)
    depth_color = colorize_depth(disparity)

    # Run YOLO every N frames
    if frame_count % DETECT_EVERY == 0:
        small = cv2.resize(left_rect, (DETECT_W, DETECT_H))
        results = model(small, verbose=False)
        last_boxes = results[0].boxes

    # Annotate left frame
    annotated = left_rect.copy()
    annotated = annotate_targets(annotated, last_boxes, disparity)

    # ── Force all panels to FRAME_W x FRAME_H before stacking ──────────────
    # This handles old calibration pkl files that produce non-standard sizes
    left_panel  = cv2.resize(annotated,   (FRAME_W, FRAME_H))
    depth_panel = cv2.resize(depth_color, (FRAME_W, FRAME_H))
    right_panel = cv2.resize(right_rect,  (FRAME_W, FRAME_H))

    # Add panel labels
    for img, label in zip(
        [left_panel, depth_panel, right_panel],
        ["Left (Detection)", "Depth Map", "Right Camera"]
    ):
        cv2.putText(img, label, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    display = np.hstack([left_panel, depth_panel, right_panel])
    cv2.imshow("target Detection + Depth", display)

    frame_count += 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

left_buf.release()
right_buf.release()
cv2.destroyAllWindows()
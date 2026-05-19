import os

for env_var in ["GTK_PATH", "QT_IM_MODULE", "GIO_MODULE_DIR"]:
    if env_var in os.environ:
        del os.environ[env_var]

import cv2
import numpy as np
import threading
import pickle
import time
import os
from collections import deque
import onnxruntime as ort

# --- Load Calibration ---
_CALIB_PATH = os.path.join(os.path.dirname(__file__), "..", "stereo_calib.pkl")
with open(_CALIB_PATH, "rb") as f:
    calib = pickle.load(f)

# Extract parameters from the Rectified Projection Matrix (P_l)
FOCAL_LENGTH = calib["focal_px"]
BASELINE     = calib["baseline_m"]
CX           = calib["P_l"][0, 2]  # Principal point X
CY           = calib["P_l"][1, 2]  # Principal point Y

map_lx, map_ly = calib["map_lx"], calib["map_ly"]
map_rx, map_ry = calib["map_rx"], calib["map_ry"]

# --- Constants ---
TARGET_CLASS = 39  # bottle in COCO
DETECT_EVERY = 4
FRAME_W, FRAME_H = 640, 480
DETECT_W, DETECT_H = 640, 480  # must match model export size
scale_x = FRAME_W / DETECT_W
scale_y = FRAME_H / DETECT_H

# --- ESP32-CAM Stream Buffer ---
class CameraBufferURL:
    def __init__(self, url):
        # Optimized for ESP32-CAM MJPEG streams
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "reconnect;1|reconnect_streamed;1|reconnect_delay_max;2"
        self.cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.frame = None
        self.lock = threading.Lock()
        self.running = True
        threading.Thread(target=self._reader, daemon=True).start()

    def _reader(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.frame = frame
            else:
                time.sleep(0.01)

    def read(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def release(self):
        self.running = False
        self.cap.release()

# --- Depth & 3D Math ---
stereo = cv2.StereoSGBM_create(
    minDisparity=0, numDisparities=128, blockSize=5,
    P1=8*3*5**2, P2=32*3*5**2, mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
)

# Temporal smoothing: keep last N depth readings per tracked object
_SMOOTH_N = 8
_z_history = deque(maxlen=_SMOOTH_N)

def get_3d_coordinates(u, v, disparity_val):
    """ Converts pixel (u,v) + disparity to real-world (X,Y,Z) in meters """
    if disparity_val <= 0:
        return None
    Z = (FOCAL_LENGTH * BASELINE) / disparity_val
    X = (u - CX) * Z / FOCAL_LENGTH
    Y = (v - CY) * Z / FOCAL_LENGTH
    return (round(X, 3), round(Y, 3), round(Z, 3))

# --- YOLO ONNX via ONNX Runtime ---
CONF_THRESH = 0.4
NMS_THRESH  = 0.4

_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "Yolo_models", "yolo11n.onnx")
_session = ort.InferenceSession(_MODEL_PATH, providers=["CPUExecutionProvider"])
_input_name = _session.get_inputs()[0].name

def detect_bottles(frame):
    """Run YOLO11n ONNX inference, return list of (x1,y1,x2,y2,conf) in frame coords."""
    # Preprocess: resize, normalize, NCHW
    img = cv2.resize(frame, (DETECT_W, DETECT_H))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    blob = np.transpose(img, (2, 0, 1))[np.newaxis]  # (1,3,H,W)

    out = _session.run(None, {_input_name: blob})[0][0]  # (84, 6300)
    out = out.T  # -> (6300, 84)

    boxes, scores = [], []
    for row in out:
        class_scores = row[4:]
        cls = int(np.argmax(class_scores))
        conf = float(class_scores[cls])
        if cls != TARGET_CLASS or conf < CONF_THRESH:
            continue
        cx, cy, bw, bh = row[:4]
        x1 = int((cx - bw / 2) * scale_x)
        y1 = int((cy - bh / 2) * scale_y)
        x2 = int((cx + bw / 2) * scale_x)
        y2 = int((cy + bh / 2) * scale_y)
        boxes.append([x1, y1, x2 - x1, y2 - y1])
        scores.append(conf)

    kept = cv2.dnn.NMSBoxes(boxes, scores, CONF_THRESH, NMS_THRESH)
    results = []
    for i in (kept.flatten() if len(kept) else []):
        x, y, bw, bh = boxes[i]
        results.append((x, y, x + bw, y + bh, scores[i]))
    return results

# --- Processing Logic ---
_PATCH_R = 20      # half-size of disparity sampling region (pixels)
_Z_MIN   = 0.05   # metres — discard implausibly close readings
_Z_MAX   = 5.0    # metres — discard implausibly far readings

def process_frame(frame, detections, disparity):
    coords_3d = None
    for (x1, y1, x2, y2, conf) in detections:
        u = (x1 + x2) // 2
        v = (y1 + y2) // 2

        # Larger patch → more valid disparity samples → more stable median
        patch = disparity[max(0, v - _PATCH_R):v + _PATCH_R,
                          max(0, u - _PATCH_R):u + _PATCH_R]
        valid = patch[patch > 0]
        if valid.size == 0:
            color = (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            continue

        raw = get_3d_coordinates(u, v, np.median(valid))
        if raw is None:
            color = (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            continue

        X_raw, Y_raw, Z_raw = raw

        # Reject readings outside plausible range before smoothing
        if _Z_MIN < Z_raw < _Z_MAX:
            _z_history.append(Z_raw)

        if not _z_history:
            color = (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            continue

        # Use smoothed Z but keep X/Y from current frame (they're less noisy)
        Z_smooth = float(np.median(_z_history))
        X = round((u - CX) * Z_smooth / FOCAL_LENGTH, 3)
        Y = round((v - CY) * Z_smooth / FOCAL_LENGTH, 3)
        coords_3d = (X, Y, round(Z_smooth, 3))

        color = (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        X, Y, Z = coords_3d
        print(f"Bottle -> X:{X:+.3f}m  Y:{Y:+.3f}m  Z:{Z:.3f}m")
        for i, text in enumerate([f"X:{X:+.3f}m", f"Y:{Y:+.3f}m", f"Z:{Z:.3f}m"]):
            cv2.putText(frame, text, (x1, y1 - 10 - i*18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return frame, coords_3d

# --- Main Loop ---
left_buf  = CameraBufferURL("http://stereo-left.local:80/stream")
right_buf = CameraBufferURL("http://stereo-right.local:80/stream")

cv2.namedWindow("Stereo Vision — Bottle Detection", cv2.WINDOW_NORMAL)

while True:
    l_img = left_buf.read()
    r_img = right_buf.read()

    if l_img is None or r_img is None:
        status = []
        if l_img is None: status.append("Waiting: LEFT camera")
        if r_img is None: status.append("Waiting: RIGHT camera")
        blank = np.zeros((FRAME_H, FRAME_W, 3), np.uint8)
        for i, msg in enumerate(status):
            cv2.putText(blank, msg, (20, 40 + i*30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
        cv2.imshow("Stereo Vision — Bottle Detection", blank)
        if cv2.waitKey(100) & 0xFF == ord('q'): break
        continue

    l_rect = cv2.remap(l_img, map_lx, map_ly, cv2.INTER_LINEAR)
    r_rect = cv2.remap(r_img, map_rx, map_ry, cv2.INTER_LINEAR)

    gray_l = cv2.cvtColor(l_rect, cv2.COLOR_BGR2GRAY)
    gray_r = cv2.cvtColor(r_rect, cv2.COLOR_BGR2GRAY)
    disp = stereo.compute(gray_l, gray_r).astype(np.float32) / 16.0

    detections = detect_bottles(l_rect)
    annotated, current_pos = process_frame(l_rect.copy(), detections, disp)

    if current_pos:
        # arm.move_to(current_pos)
        pass

    cv2.imshow("Stereo Vision — Bottle Detection", annotated)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

left_buf.release()
right_buf.release()
cv2.destroyAllWindows()
import cv2
import numpy as np
from ultralytics import YOLO
import onnxruntime as ort
import threading
from collections import deque

# ── Calibration ───────────────────────────────────────────────────────────────
calibration_scale = None
calib_depth_val = None

def depth_to_cm(depth_val):
    if calibration_scale is None or depth_val == 0:
        return None
    return calibration_scale / depth_val

# ── Models ───────────────────────────────────────────────────────────────────
detector_model = YOLO("Yolo_models/yolov8n.pt")
session = ort.InferenceSession(
    "Midas_depth/midas_v21_small_256.onnx",
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
)
input_name = session.get_inputs()[0].name

# ── Camera ───────────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

# ── Shared state ─────────────────────────────────────────────────────────────
latest_frame = None
annotated_frame = None
depth_colored = None
depth_map_full = None
detections = []
lock = threading.Lock()
running = True

# ── Rolling average buffer (last N depth readings per laptop) ─────────────────
SMOOTH_N = 10                        # average over last 10 readings
depth_buffer = deque(maxlen=SMOOTH_N)

def get_depth_for_box(dmap, x1, y1, x2, y2):
    cx1 = int(x1 + (x2 - x1) * 0.25)
    cy1 = int(y1 + (y2 - y1) * 0.25)
    cx2 = int(x1 + (x2 - x1) * 0.75)
    cy2 = int(y1 + (y2 - y1) * 0.75)
    cx1, cy1 = max(0, cx1), max(0, cy1)
    cx2, cy2 = min(dmap.shape[1]-1, cx2), min(dmap.shape[0]-1, cy2)
    region = dmap[cy1:cy2, cx1:cx2]
    return float(np.mean(region)) if region.size > 0 else 0.0

def distance_in_cm(depth_val):
    return calibration_scale / depth_val if (calibration_scale and depth_val > 0) else None

def inference_thread():
    global annotated_frame, depth_colored, depth_map_full, detections, calib_depth_val
    skip = 0
    while running:
        with lock:
            frame = latest_frame
        if frame is None:
            continue

        skip += 1

        # ── MiDaS depth ──────────────────────────────────────────────────
        img = cv2.resize(frame, (256, 256))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)[np.newaxis]
        depth_out = session.run(None, {input_name: img})[0]
        dmap = depth_out[0] if depth_out.ndim == 3 else depth_out[0, 0]
        dmap_full = cv2.resize(dmap, (640, 480))

        depth_vis = cv2.normalize(dmap_full, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        dcol = cv2.applyColorMap(depth_vis, cv2.COLORMAP_MAGMA)

        # ── YOLO every 3rd frame ─────────────────────────────────────────
        det_frame = annotated_frame
        new_detections = detections

        if skip % 3 == 0:
            scale_x, scale_y = 640 / 320, 480 / 240
            small = cv2.resize(frame, (320, 240))
            results = detector_model(small, verbose=False)
            det_frame = cv2.resize(results[0].plot(), (640, 480))

            new_detections = []
            for box in results[0].boxes:
                cls_id = int(box.cls)
                label = detector_model.names[cls_id]
                if label == "laptop":
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    x1, x2 = x1 * scale_x, x2 * scale_x
                    y1, y2 = y1 * scale_y, y2 * scale_y
                    new_detections.append((label, int(x1), int(y1), int(x2), int(y2)))

        # ── Overlay depth on YOLO frame ───────────────────────────────────
        if det_frame is not None and new_detections:
            overlay = det_frame.copy()
            for label, x1, y1, x2, y2 in new_detections:
                raw_depth = get_depth_for_box(dmap_full, x1, y1, x2, y2)

                # Add to rolling buffer and compute smoothed depth
                depth_buffer.append(raw_depth)
                smoothed_depth = float(np.mean(depth_buffer))
                calib_depth_val = smoothed_depth

                dist_cm = distance_in_cm(smoothed_depth)

                if dist_cm:
                    text = f"laptop: {dist_cm:.1f} cm"
                else:
                    text = f"laptop: {smoothed_depth:.1f} (raw) [C to calibrate]"

                print(f"[Laptop] raw={raw_depth:.1f} | smoothed={smoothed_depth:.1f}",
                      f"| {dist_cm:.1f} cm" if dist_cm else "| not calibrated")

                cv2.rectangle(dcol, (x1, y1), (x2, y2), (255, 255, 255), 2)
                cv2.putText(dcol, f"{dist_cm:.1f}cm" if dist_cm else f"{smoothed_depth:.1f}",
                            (x1 + 4, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                color = (0, 0, 255)
                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                cv2.rectangle(overlay, (x1, y1 - th - 8), (x1 + tw + 4, y1), (0, 0, 0), -1)
                cv2.putText(overlay, text, (x1 + 2, y1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
            det_frame = overlay

        # Calibration status hint
        hint = "C: calibrate" if calibration_scale is None else f"scale={calibration_scale:.1f} | C: recalibrate"
        cv2.putText(det_frame if det_frame is not None else frame,
                    hint, (10, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        with lock:
            annotated_frame = det_frame
            depth_colored = dcol
            depth_map_full = dmap_full
            detections = new_detections

# ── Start inference thread ────────────────────────────────────────────────
t = threading.Thread(target=inference_thread, daemon=True)
t.start()

print("Controls: Q = quit | C = calibrate (point at laptop first)")

# ── Main display loop ─────────────────────────────────────────────────────
while True:
    ret, frame = cap.read()
    if not ret:
        break

    with lock:
        latest_frame = frame.copy()
        ann = annotated_frame
        dcol = depth_colored

    if ann is not None and dcol is not None:
        combined = np.hstack([ann, dcol])
        cv2.imshow("YOLO | Depth", combined)
    elif ann is not None:
        cv2.imshow("YOLO | Depth", ann)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        running = False
        break

    elif key == ord('c'):
        if calib_depth_val and calib_depth_val > 0:
            cv2.destroyAllWindows()
            print(f"\n[CALIBRATION] Smoothed depth = {calib_depth_val:.2f}")
            print(f"This is averaged over the last {SMOOTH_N} readings to reduce noise.")
            real_cm = input("Enter MEASURED real distance to laptop in cm: ")
            try:
                real_cm = float(real_cm)
                calibration_scale = calib_depth_val * real_cm
                print(f"\nCalibration done!")
                print(f"  scale = {calibration_scale:.4f}")
                print(f"  formula: cm = {calibration_scale:.4f} / depth_val")
                print(f"  verification: {calibration_scale / calib_depth_val:.1f} cm (should match {real_cm})\n")
            except ValueError:
                print("Invalid input, calibration not set.")
        else:
            print("No laptop in view — point camera at laptop first.")

cap.release()
cv2.destroyAllWindows()
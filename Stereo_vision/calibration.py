import os

# Remove snap/VS Code specific environment variables that cause Qt/GLIBC_PRIVATE crashes
for env_var in ["GTK_PATH", "QT_IM_MODULE", "GIO_MODULE_DIR"]:
    if env_var in os.environ:
        del os.environ[env_var]

import cv2
import numpy as np
import pickle
from pathlib import Path

# Tell FFmpeg (OpenCV's backend) to reconnect automatically when the MJPEG
# stream drops — prevents the "Stream ends prematurely" crash.
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
    "reconnect;1|reconnect_streamed;1|reconnect_delay_max;2"
)

def load_existing_images():
    global objpoints, imgpoints_l, imgpoints_r

    left_imgs  = sorted(CALIB_DIR.glob("left_*.png"))
    right_imgs = sorted(CALIB_DIR.glob("right_*.png"))

    pairs = min(len(left_imgs), len(right_imgs))
    print(f"Loading {pairs} existing calibration pairs...")

    for i in range(pairs):

        img_l = cv2.imread(str(left_imgs[i]))
        img_r = cv2.imread(str(right_imgs[i]))

        gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)

        cb_flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        found_l, corners_l = cv2.findChessboardCorners(gray_l, CHESSBOARD, cb_flags)
        found_r, corners_r = cv2.findChessboardCorners(gray_r, CHESSBOARD, cb_flags)

        if found_l and found_r:

            corners_l = cv2.cornerSubPix(gray_l, corners_l, (11,11), (-1,-1), criteria)
            corners_r = cv2.cornerSubPix(gray_r, corners_r, (11,11), (-1,-1), criteria)

            objpoints.append(objp)
            imgpoints_l.append(corners_l)
            imgpoints_r.append(corners_r)

        else:
            print(f"Warning: pair {i} checkerboard not detected")

    print(f"Loaded {len(objpoints)} usable pairs\n")
# ─── Checkerboard config ──────────────────────────────────────────────────────
# These must match YOUR printed checkerboard
# Count INNER corners, not squares. A 9x6 square board has 8x5 inner corners.
CHESSBOARD = (9, 7)
SQUARE_SIZE = 0.020  # meters — measure your printed square size precisely

FRAME_W, FRAME_H = 640, 480   # must match stereo.py exactly
OUTPUT_FILE = "stereo_calib.pkl"
CALIB_DIR   = Path("calib_images")
CALIB_DIR.mkdir(exist_ok=True)

# Coverage heatmap: divide the frame into a grid, track which cells have been hit
_GRID_COLS, _GRID_ROWS = 5, 4   # 5×4 = 20 zones across the frame
_coverage = np.zeros((_GRID_ROWS, _GRID_COLS), dtype=np.int32)

# Structured capture sequence: (grid_row, grid_col, tilt_hint)
# Drives the on-screen prompt so every region + tilt combo is visited
_CAPTURE_SEQUENCE = [
    # Four corners, flat
    (0, 0, "TOP-LEFT, flat"),
    (0, 4, "TOP-RIGHT, flat"),
    (3, 0, "BOT-LEFT, flat"),
    (3, 4, "BOT-RIGHT, flat"),
    # Four corners, tilted
    (0, 0, "TOP-LEFT, tilt toward cam"),
    (0, 4, "TOP-RIGHT, tilt toward cam"),
    (3, 0, "BOT-LEFT, tilt toward cam"),
    (3, 4, "BOT-RIGHT, tilt toward cam"),
    # Edge midpoints
    (0, 2, "TOP-CENTER, flat"),
    (3, 2, "BOT-CENTER, flat"),
    (1, 0, "LEFT-EDGE, flat"),
    (1, 4, "RIGHT-EDGE, flat"),
    # Center at close range
    (1, 2, "CENTER, close (~25cm)"),
    (1, 2, "CENTER, medium (~45cm)"),
    (1, 2, "CENTER, tilt left 30°"),
    (1, 2, "CENTER, tilt right 30°"),
    (1, 2, "CENTER, tilt up 30°"),
    (1, 2, "CENTER, tilt down 30°"),
    # Fill remaining zones
    (0, 1, "TOP, slightly left"),
    (0, 3, "TOP, slightly right"),
    (3, 1, "BOT, slightly left"),
    (3, 3, "BOT, slightly right"),
    (2, 0, "MID-LEFT, tilted"),
    (2, 4, "MID-RIGHT, tilted"),
    (1, 1, "UPPER-LEFT quadrant"),
    (1, 3, "UPPER-RIGHT quadrant"),
    (2, 1, "LOWER-LEFT quadrant"),
    (2, 3, "LOWER-RIGHT quadrant"),
    (2, 2, "CENTER, rotate 45°"),
    (1, 2, "CENTER, far (~70cm)"),
]

def _next_target():
    """Return the hint string for the next uncaptured sequence step."""
    idx = int(np.sum(_coverage))   # rough progress counter
    if idx < len(_CAPTURE_SEQUENCE):
        r, c, hint = _CAPTURE_SEQUENCE[idx]
        return hint, r, c
    return "Free capture — fill any red zones", -1, -1

def _update_coverage(corners):
    for pt in corners.reshape(-1, 2):
        col = int(pt[0] / FRAME_W * _GRID_COLS)
        row = int(pt[1] / FRAME_H * _GRID_ROWS)
        col = min(col, _GRID_COLS - 1)
        row = min(row, _GRID_ROWS - 1)
        _coverage[row, col] += 1

def _draw_coverage(img):
    cell_w = FRAME_W // _GRID_COLS
    cell_h = FRAME_H // _GRID_ROWS
    overlay = img.copy()
    for r in range(_GRID_ROWS):
        for c in range(_GRID_COLS):
            hits = _coverage[r, c]
            if hits == 0:
                color = (0, 0, 140)       # red — uncovered
            elif hits < 2:
                color = (0, 140, 255)     # orange — lightly covered
            else:
                color = (0, 200, 0)       # green — well covered
            x1, y1 = c * cell_w, r * cell_h
            x2, y2 = x1 + cell_w, y1 + cell_h
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
            if hits > 0:
                cv2.putText(overlay, str(hits), (x1 + 4, y2 - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.addWeighted(overlay, 0.30, img, 0.70, 0, img)
    for c in range(1, _GRID_COLS):
        cv2.line(img, (c * cell_w, 0), (c * cell_w, FRAME_H), (80, 80, 80), 1)
    for r in range(1, _GRID_ROWS):
        cv2.line(img, (0, r * cell_h), (FRAME_W, r * cell_h), (80, 80, 80), 1)

    # Draw target zone highlight
    hint, tr, tc = _next_target()
    if tr >= 0:
        x1 = tc * cell_w
        y1 = tr * cell_h
        cv2.rectangle(img, (x1, y1), (x1 + cell_w, y1 + cell_h), (0, 255, 255), 3)

    covered = int(np.sum(_coverage > 0))
    total   = _GRID_ROWS * _GRID_COLS
    cv2.putText(img, f"Coverage: {covered}/{total}  NEXT: {hint}",
                (6, FRAME_H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1)


criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# ─── 3D object points for one checkerboard view ───────────────────────────────
objp = np.zeros((CHESSBOARD[0] * CHESSBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHESSBOARD[0], 0:CHESSBOARD[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE

objpoints   = []   # 3D points in real world
imgpoints_l = []   # 2D points in left image
imgpoints_r = []   # 2D points in right image

load_existing_images()
captured = len(objpoints)
# ─── Camera setup ─────────────────────────────────────────────────────────────
LEFT_URL  = "http://stereo-left.local:80/stream"
RIGHT_URL = "http://stereo-right.local:80/stream"

def open_camera(url):
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    cap.set(cv2.CAP_PROP_FPS, 15)
    cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 4000)
    return cap

left_cap  = open_camera(LEFT_URL)
right_cap = open_camera(RIGHT_URL)

fail_l = 0
fail_r = 0
MAX_FAILS = 20

# NOTE: open_camera() already sets 640x480 @ 15fps — no override needed

print("=== Stereo Calibration ===")
print(f"Target: 30+ pairs, covering all 20 zones (red = uncovered, green = good)")
print(f"Tips:   vary distance (20-80cm), tilt board ±30°, hit all frame corners")
print(f"Board:  {CHESSBOARD[0]}x{CHESSBOARD[1]} inner corners, {SQUARE_SIZE*100:.1f}cm squares")
print()
print("Controls:")
print("  SPACE  — capture frame pair")
print("  D      — delete last capture")
print("  C      — run calibration (needs 15+ pairs)")
print("  Q      — quit")
print()

captured = len(objpoints)

# Force window to appear before loop starts
cv2.namedWindow("Stereo Calibration — Left | Right", cv2.WINDOW_NORMAL)
cv2.waitKey(1)

while True:
    ret_l, left_frame  = left_cap.read()
    ret_r, right_frame = right_cap.read()

    if not ret_l: fail_l += 1
    else:         fail_l  = 0
    if not ret_r: fail_r += 1
    else:         fail_r  = 0

    if fail_l >= MAX_FAILS:
        print("[WARN] Left camera lost — reconnecting...")
        left_cap.release()
        left_cap = open_camera(LEFT_URL)
        fail_l = 0
    if fail_r >= MAX_FAILS:
        print("[WARN] Right camera lost — reconnecting...")
        right_cap.release()
        right_cap = open_camera(RIGHT_URL)
        fail_r = 0

    if not ret_l or not ret_r:
        label = []
        if not ret_l: label.append("LEFT camera: no frame")
        if not ret_r: label.append("RIGHT camera: no frame")
        blank = np.zeros((FRAME_H, FRAME_W * 2, 3), dtype=np.uint8)
        cv2.putText(blank, " | ".join(label), (20, FRAME_H // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 60, 255), 2)
        cv2.imshow("Stereo Calibration — Left | Right", blank)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    gray_l = cv2.cvtColor(left_frame,  cv2.COLOR_BGR2GRAY)
    gray_r = cv2.cvtColor(right_frame, cv2.COLOR_BGR2GRAY)

    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK
    found_l, corners_l = cv2.findChessboardCorners(gray_l, CHESSBOARD, flags)
    found_r, corners_r = cv2.findChessboardCorners(gray_r, CHESSBOARD, flags)

    # Draw corners live so user can see detection
    preview_l = left_frame.copy()
    preview_r = right_frame.copy()
    if found_l:
        cv2.drawChessboardCorners(preview_l, CHESSBOARD, corners_l, found_l)
    if found_r:
        cv2.drawChessboardCorners(preview_r, CHESSBOARD, corners_r, found_r)

    # Coverage heatmap on left preview only
    _draw_coverage(preview_l)

    # Status overlay
    both = found_l and found_r
    status_color = (0, 255, 0) if both else (0, 60, 255)
    status_text  = f"BOTH FOUND — SPACE to capture ({captured} saved)" if both else \
                   f"Searching... L={'OK' if found_l else 'X'}  R={'OK' if found_r else 'X'}"
    for img in (preview_l, preview_r):
        cv2.putText(img, status_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, status_color, 2)
        cv2.putText(img, f"Captures: {captured}/30+", (10, 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

    display = np.hstack([preview_l, preview_r])
    cv2.imshow("Stereo Calibration — Left | Right", display)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

    elif key == ord(' ') and both:
        # Normalize corner order so both cameras agree on direction
        if corners_l[0][0][0] > corners_l[-1][0][0]:
            corners_l = np.ascontiguousarray(corners_l[::-1])
        if corners_r[0][0][0] > corners_r[-1][0][0]:
            corners_r = np.ascontiguousarray(corners_r[::-1])

        # Refine corners to sub-pixel accuracy
        refined_l = cv2.cornerSubPix(gray_l, corners_l, (11,11), (-1,-1), criteria)
        refined_r = cv2.cornerSubPix(gray_r, corners_r, (11,11), (-1,-1), criteria)

        objpoints.append(objp)
        imgpoints_l.append(refined_l)
        imgpoints_r.append(refined_r)
        _update_coverage(refined_l)

        # Save images for reference
        cv2.imwrite(str(CALIB_DIR / f"left_{captured:03d}.png"),  left_frame)
        cv2.imwrite(str(CALIB_DIR / f"right_{captured:03d}.png"), right_frame)
        captured += 1
        covered = int(np.sum(_coverage > 0))
        total   = _GRID_ROWS * _GRID_COLS
        print(f"  Captured pair {captured}  |  coverage {covered}/{total} zones")

    elif key == ord('d') and captured > 0:
        objpoints.pop()
        imgpoints_l.pop()
        imgpoints_r.pop()
        captured -= 1
        print(f"  Deleted last capture. Total: {captured}")

    elif key == ord('c'):
        existing = len(objpoints)
        covered  = int(np.sum(_coverage > 0))
        total    = _GRID_ROWS * _GRID_COLS
        if existing < 30:
            print(f"  Need at least 30 pairs (have {existing}). Keep capturing.")
            continue
        if covered < total * 0.75:
            print(f"  Coverage only {covered}/{total} zones — move board to uncovered (red) areas.")
            continue

        print(f"\nRunning calibration on {existing} pairs found in {CALIB_DIR}...")

        img_size = (FRAME_W, FRAME_H)

        # ── Individual camera calibration ──────────────────────────────────
        print("  Calibrating left camera...")
        ret_l, K_l, D_l, rvecs_l, tvecs_l = cv2.calibrateCamera(
            objpoints, imgpoints_l, img_size, None, None)

        print("  Calibrating right camera...")
        ret_r, K_r, D_r, rvecs_r, tvecs_r = cv2.calibrateCamera(
            objpoints, imgpoints_r, img_size, None, None)

        # ── Remove outlier pairs (per-image reprojection error > 0.8px) ───
        keep = []
        for i in range(len(objpoints)):
            proj_l, _ = cv2.projectPoints(objpoints[i], rvecs_l[i], tvecs_l[i], K_l, D_l)
            proj_r, _ = cv2.projectPoints(objpoints[i], rvecs_r[i], tvecs_r[i], K_r, D_r)
            err_l = np.sqrt(np.mean((imgpoints_l[i] - proj_l)**2))
            err_r = np.sqrt(np.mean((imgpoints_r[i] - proj_r)**2))
            if err_l <= 0.8 and err_r <= 0.8:
                keep.append(i)
            else:
                print(f"  Removing outlier pair {i}: L={err_l:.3f} R={err_r:.3f}")

        if len(keep) < len(objpoints):
            objpoints   = [objpoints[i]   for i in keep]
            imgpoints_l = [imgpoints_l[i] for i in keep]
            imgpoints_r = [imgpoints_r[i] for i in keep]
            print(f"  Kept {len(keep)} pairs after outlier removal — recalibrating...")
            ret_l, K_l, D_l, _, _ = cv2.calibrateCamera(objpoints, imgpoints_l, img_size, None, None)
            ret_r, K_r, D_r, _, _ = cv2.calibrateCamera(objpoints, imgpoints_r, img_size, None, None)

        # ── Stereo calibration (gets R and T between cameras) ──────────────
        print("  Running stereo calibration...")
        flags = (cv2.CALIB_FIX_INTRINSIC)   # keep individual results, solve R/T only
        ret_stereo, K_l, D_l, K_r, D_r, R, T, E, F = cv2.stereoCalibrate(
            objpoints, imgpoints_l, imgpoints_r,
            K_l, D_l, K_r, D_r,
            img_size, flags=flags,
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 1e-6)
        )

        # T[0] is the horizontal baseline in meters
        baseline_m = abs(float(T.flatten()[0]))

        print(f"\n  ✓ Stereo RMS error : {ret_stereo:.4f}  (good if < 1.0)")
        print(f"  ✓ Baseline         : {baseline_m*100:.2f} cm")

        # ── Stereo rectification ───────────────────────────────────────────
        R_l, R_r, P_l, P_r, Q, roi_l, roi_r = cv2.stereoRectify(
            K_l, D_l, K_r, D_r, img_size, R, T,
            flags=cv2.CALIB_ZERO_DISPARITY, alpha=0
        )

        focal_px   = float(P_l[0, 0])   # fx from RECTIFIED projection matrix (not raw K_l)
        print(f"  ✓ Focal length     : {focal_px:.1f} px")

        map_lx, map_ly = cv2.initUndistortRectifyMap(K_l, D_l, R_l, P_l, img_size, cv2.CV_32FC1)
        map_rx, map_ry = cv2.initUndistortRectifyMap(K_r, D_r, R_r, P_r, img_size, cv2.CV_32FC1)

        # ── Save everything ────────────────────────────────────────────────
        calib_data = {
            "K_l": K_l, "D_l": D_l,
            "K_r": K_r, "D_r": D_r,
            "R": R, "T": T, "E": E, "F": F,
            "R_l": R_l, "R_r": R_r,
            "P_l": P_l, "P_r": P_r, "Q": Q,
            "map_lx": map_lx, "map_ly": map_ly,
            "map_rx": map_rx, "map_ry": map_ry,
            "baseline_m":  baseline_m,
            "focal_px":    focal_px,
            "image_size":  img_size,
        }
        with open(OUTPUT_FILE, "wb") as f:
            pickle.dump(calib_data, f)

        print(f"\n  ✓ Saved to '{OUTPUT_FILE}'")
        print(f"\n  Paste these into your depth script:")
        print(f"    FOCAL_LENGTH = {focal_px:.1f}")
        print(f"    BASELINE     = {baseline_m:.4f}")
        break

left_cap.release()
right_cap.release()
cv2.destroyAllWindows()

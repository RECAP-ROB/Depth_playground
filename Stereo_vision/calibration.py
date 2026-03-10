import cv2
import numpy as np
import pickle
from pathlib import Path

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

        found_l, corners_l = cv2.findChessboardCorners(gray_l, CHESSBOARD, None)
        found_r, corners_r = cv2.findChessboardCorners(gray_r, CHESSBOARD, None)

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
CHESSBOARD = (7, 9)
SQUARE_SIZE = 0.020  # meters — measure your printed square size precisely

FRAME_W, FRAME_H = 640, 480   # must match stereo.py exactly
OUTPUT_FILE = "stereo_calib.pkl"
CALIB_DIR   = Path("calib_images")
CALIB_DIR.mkdir(exist_ok=True)


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
def open_camera(index):
    cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    cap.set(cv2.CAP_PROP_FPS, 15)
    return cap
    
left_cap  = open_camera(2)
right_cap = open_camera(4)
# NOTE: open_camera() already sets 640x480 @ 15fps — no override needed

print("=== Stereo Calibration ===")
print(f"Target: 20+ image pairs with checkerboard visible in BOTH cameras")
print(f"Board:  {CHESSBOARD[0]}x{CHESSBOARD[1]} inner corners, {SQUARE_SIZE*100:.1f}cm squares")
print()
print("Controls:")
print("  SPACE  — capture frame pair")
print("  D      — delete last capture")
print("  C      — run calibration (needs 20+ pairs)")
print("  Q      — quit")
print()

captured = len(objpoints)

# Force window to appear before loop starts
cv2.namedWindow("Stereo Calibration — Left | Right", cv2.WINDOW_NORMAL)
cv2.waitKey(1)

while True:
    ret_l, left_frame  = left_cap.read()
    ret_r, right_frame = right_cap.read()
    if not ret_l or not ret_r:
        continue

    gray_l = cv2.cvtColor(left_frame,  cv2.COLOR_BGR2GRAY)
    gray_r = cv2.cvtColor(right_frame, cv2.COLOR_BGR2GRAY)

    found_l, corners_l = cv2.findChessboardCorners(gray_l, CHESSBOARD, None)
    found_r, corners_r = cv2.findChessboardCorners(gray_r, CHESSBOARD, None)

    # Draw corners live so user can see detection
    preview_l = left_frame.copy()
    preview_r = right_frame.copy()
    if found_l:
        cv2.drawChessboardCorners(preview_l, CHESSBOARD, corners_l, found_l)
    if found_r:
        cv2.drawChessboardCorners(preview_r, CHESSBOARD, corners_r, found_r)

    # Status overlay
    both = found_l and found_r
    status_color = (0, 255, 0) if both else (0, 60, 255)
    status_text  = f"BOTH FOUND — SPACE to capture ({captured} saved)" if both else \
                   f"Searching... L={'OK' if found_l else 'X'}  R={'OK' if found_r else 'X'}"
    for img in (preview_l, preview_r):
        cv2.putText(img, status_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, status_color, 2)
        cv2.putText(img, f"Captures: {captured}/20+", (10, 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

    display = np.hstack([preview_l, preview_r])
    cv2.imshow("Stereo Calibration — Left | Right", display)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

    elif key == ord(' ') and both:
        # Refine corners to sub-pixel accuracy
        refined_l = cv2.cornerSubPix(gray_l, corners_l, (11,11), (-1,-1), criteria)
        refined_r = cv2.cornerSubPix(gray_r, corners_r, (11,11), (-1,-1), criteria)

        objpoints.append(objp)
        imgpoints_l.append(refined_l)
        imgpoints_r.append(refined_r)

        # Save images for reference
        cv2.imwrite(str(CALIB_DIR / f"left_{captured:03d}.png"),  left_frame)
        cv2.imwrite(str(CALIB_DIR / f"right_{captured:03d}.png"), right_frame)
        captured += 1
        print(f"  Captured pair {captured}")

    elif key == ord('d') and captured > 0:
        objpoints.pop()
        imgpoints_l.pop()
        imgpoints_r.pop()
        captured -= 1
        print(f"  Deleted last capture. Total: {captured}")

    elif key == ord('c'):
        existing = len(objpoints)
        if existing < 20:
            print(f"  Need at least 20 pairs (found {existing} in {CALIB_DIR}). Keep capturing.")
            continue

        print(f"\nRunning calibration on {existing} pairs found in {CALIB_DIR}...")

        img_size = (FRAME_W, FRAME_H)

        # ── Individual camera calibration ──────────────────────────────────
        print("  Calibrating left camera...")
        ret_l, K_l, D_l, _, _ = cv2.calibrateCamera(
            objpoints, imgpoints_l, img_size, None, None)

        print("  Calibrating right camera...")
        ret_r, K_r, D_r, _, _ = cv2.calibrateCamera(
            objpoints, imgpoints_r, img_size, None, None)

        # ── Stereo calibration (gets R and T between cameras) ──────────────
        print("  Running stereo calibration...")
        flags = (cv2.CALIB_FIX_INTRINSIC)   # keep individual results, solve R/T only
        ret_stereo, K_l, D_l, K_r, D_r, R, T, E, F = cv2.stereoCalibrate(
            objpoints, imgpoints_l, imgpoints_r,
            K_l, D_l, K_r, D_r,
            img_size, flags=flags,
            criteria=criteria
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
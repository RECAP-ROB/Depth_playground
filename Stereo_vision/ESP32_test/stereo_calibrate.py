#!/usr/bin/env python3
"""
Stereo Camera Calibration — ESP32-CAM pair
===========================================
Use this ONCE to calibrate your two cameras before running stereo_processor.py.
Produces stereo_calibration.npz which stereo_processor.py loads automatically.

Steps:
  1. Print or display a chessboard (default: 9×6 inner corners).
  2. Run this script and point both cameras at the board.
  3. Press SPACE to capture a frame pair (need at least 10 good pairs).
  4. Press ENTER to compute calibration.
  5. stereo_calibration.npz is saved in the current directory.

Usage:
    python stereo_calibrate.py --left  http://192.168.1.10/stream \
                                --right http://192.168.1.11/stream

Requirements:
    pip install opencv-python numpy
"""

import argparse
import time
import threading
import urllib.request
import numpy as np
import cv2
import os

# ─────────────────────────────────────────────
#  Chessboard pattern — must match your printout
# ─────────────────────────────────────────────
CHESS_COLS   = 9     # number of INNER corners horizontally
CHESS_ROWS   = 6     # number of INNER corners vertically
SQUARE_MM    = 25.0  # real-world square size in mm

DEFAULT_LEFT_URL  = "http://192.168.1.10/stream"
DEFAULT_RIGHT_URL = "http://192.168.1.11/stream"

# ─────────────────────────────────────────────
#  Reuse MJPEGStream from stereo_processor.py
# ─────────────────────────────────────────────
class MJPEGStream:
    def __init__(self, url, name):
        self.url = url
        self.name = name
        self.frame = None
        self.lock = threading.Lock()
        self.running = False

    def start(self):
        self.running = True
        t = threading.Thread(target=self._read_loop, daemon=True)
        t.start()

    def stop(self):
        self.running = False

    def get_frame(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def _read_loop(self):
        while self.running:
            try:
                stream = urllib.request.urlopen(self.url, timeout=10)
                buf = b""
                while self.running:
                    chunk = stream.read(4096)
                    if not chunk:
                        break
                    buf += chunk
                    a  = buf.find(b'\xff\xd8')
                    b_ = buf.find(b'\xff\xd9')
                    if a != -1 and b_ != -1 and b_ > a:
                        jpg = buf[a:b_+2]
                        buf = buf[b_+2:]
                        arr = np.frombuffer(jpg, dtype=np.uint8)
                        f = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                        if f is not None:
                            with self.lock:
                                self.frame = f
            except Exception as e:
                print(f"[{self.name}] {e} — retrying…")
                time.sleep(2)

# ─────────────────────────────────────────────
#  Main calibration routine
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--left",   default=DEFAULT_LEFT_URL)
    parser.add_argument("--right",  default=DEFAULT_RIGHT_URL)
    parser.add_argument("--width",  type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    args = parser.parse_args()

    proc_size  = (args.width, args.height)
    board_size = (CHESS_COLS, CHESS_ROWS)

    # 3-D object points (same for every view)
    objp = np.zeros((CHESS_ROWS * CHESS_COLS, 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHESS_COLS, 0:CHESS_ROWS].T.reshape(-1, 2)
    objp *= SQUARE_MM

    obj_pts, img_pts_l, img_pts_r = [], [], []

    left_stream  = MJPEGStream(args.left,  "LEFT")
    right_stream = MJPEGStream(args.right, "RIGHT")
    left_stream.start()
    right_stream.start()

    cv2.namedWindow("Calibration", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Calibration", proc_size[0] * 2, proc_size[1])

    print("\nCalibration Controls:")
    print("  SPACE  — capture current frame pair")
    print("  ENTER  — compute & save calibration")
    print("  d      — delete last capture")
    print("  q/ESC  — quit\n")

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    n_captures = 0

    while True:
        lf = left_stream.get_frame()
        rf = right_stream.get_frame()

        if lf is None or rf is None:
            time.sleep(0.05)
            continue

        lf = cv2.resize(lf, proc_size)
        rf = cv2.resize(rf, proc_size)
        lg = cv2.cvtColor(lf, cv2.COLOR_BGR2GRAY)
        rg = cv2.cvtColor(rf, cv2.COLOR_BGR2GRAY)

        # Try to find chessboard
        fl  = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
        ret_l, corners_l = cv2.findChessboardCorners(lg, board_size, fl)
        ret_r, corners_r = cv2.findChessboardCorners(rg, board_size, fl)

        display_l = lf.copy()
        display_r = rf.copy()

        if ret_l:
            corners_l2 = cv2.cornerSubPix(lg, corners_l, (11,11), (-1,-1), criteria)
            cv2.drawChessboardCorners(display_l, board_size, corners_l2, ret_l)
        if ret_r:
            corners_r2 = cv2.cornerSubPix(rg, corners_r, (11,11), (-1,-1), criteria)
            cv2.drawChessboardCorners(display_r, board_size, corners_r2, ret_r)

        status_color = (0, 255, 0) if (ret_l and ret_r) else (0, 100, 255)
        status_text  = f"Board: L={'OK' if ret_l else '--'}  R={'OK' if ret_r else '--'}  Captures: {n_captures}"
        for img in (display_l, display_r):
            cv2.putText(img, status_text, (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,0,0), 3)
            cv2.putText(img, status_text, (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, status_color, 1)

        cv2.imshow("Calibration", np.hstack([display_l, display_r]))
        key = cv2.waitKey(1) & 0xFF

        if key in (ord('q'), 27):
            break
        elif key == 32 and ret_l and ret_r:  # SPACE
            obj_pts.append(objp)
            img_pts_l.append(corners_l2)
            img_pts_r.append(corners_r2)
            n_captures += 1
            print(f"[Captured] {n_captures} frame pairs")
        elif key == ord('d') and n_captures > 0:
            obj_pts.pop(); img_pts_l.pop(); img_pts_r.pop()
            n_captures -= 1
            print(f"[Deleted] {n_captures} frame pairs remain")
        elif key == 13 and n_captures >= 6:  # ENTER
            print(f"\nComputing calibration from {n_captures} pairs…")
            flags = cv2.CALIB_FIX_ASPECT_RATIO

            ret_l, K_l, D_l, _, _ = cv2.calibrateCamera(
                obj_pts, img_pts_l, proc_size, None, None)
            ret_r, K_r, D_r, _, _ = cv2.calibrateCamera(
                obj_pts, img_pts_r, proc_size, None, None)

            stereo_flags = (cv2.CALIB_FIX_INTRINSIC)
            _, K_l, D_l, K_r, D_r, R, T, E, F = cv2.stereoCalibrate(
                obj_pts, img_pts_l, img_pts_r,
                K_l, D_l, K_r, D_r, proc_size,
                flags=stereo_flags,
                criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5))

            R_l, R_r, P_l, P_r, Q, roi_l, roi_r = cv2.stereoRectify(
                K_l, D_l, K_r, D_r, proc_size, R, T,
                flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)

            map_lx, map_ly = cv2.initUndistortRectifyMap(
                K_l, D_l, R_l, P_l, proc_size, cv2.CV_32FC1)
            map_rx, map_ry = cv2.initUndistortRectifyMap(
                K_r, D_r, R_r, P_r, proc_size, cv2.CV_32FC1)

            np.savez("stereo_calibration.npz",
                     K_l=K_l, D_l=D_l, K_r=K_r, D_r=D_r,
                     R=R, T=T, E=E, F=F, Q=Q,
                     map_lx=map_lx, map_ly=map_ly,
                     map_rx=map_rx, map_ry=map_ry)

            print("✓ Saved stereo_calibration.npz")
            print(f"  Baseline (T): {np.linalg.norm(T):.1f} mm")
            print("  Run stereo_processor.py — calibration will load automatically.\n")
            break
        elif key == 13:
            print(f"Need at least 6 captures (have {n_captures})")

    left_stream.stop()
    right_stream.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

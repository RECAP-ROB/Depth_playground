#!/usr/bin/env python3
"""
Stereo Vision Processor — ESP32-CAM pair
=========================================
Runs on a PC (Windows/Linux/macOS) or Raspberry Pi 4B.
Connects to two ESP32-CAM MJPEG streams, rectifies the frames,
computes a disparity map, and (optionally) estimates depth.

Requirements:
    pip install opencv-python numpy

On Raspberry Pi 4B with limited RAM, install the headless build:
    pip install opencv-python-headless numpy

Usage:
    python stereo_processor.py --left  http://192.168.1.10/stream \
                                --right http://192.168.1.11/stream

Controls (press in the display window):
    q / ESC  — quit
    c        — toggle calibration overlay
    s        — save current frame pair as PNG
    +/-      — increase/decrease numDisparities (depth range)
    [/]      — decrease/increase blockSize (matching window)
    r        — reset disparity parameters to defaults
    f        — toggle fullscreen
"""

import argparse
import time
import threading
import urllib.request
import sys
import os
import cv2
import numpy as np

# ─────────────────────────────────────────────
#  Default camera URLs (override via CLI args)
# ─────────────────────────────────────────────
DEFAULT_LEFT_URL  = "http://stereo-left.local/stream"
DEFAULT_RIGHT_URL = "http://stereo-right.local/stream"

# ─────────────────────────────────────────────
#  Stereo matcher parameters (tunable at runtime)
# ─────────────────────────────────────────────
class StereoParams:
    num_disparities = 64   # must be divisible by 16
    block_size      = 11   # must be odd, 5–51
    min_disparity   = 0
    # SGBM extra params
    p1_factor       = 8
    p2_factor       = 32
    disp12_max_diff = 1
    pre_filter_cap  = 63
    uniqueness_ratio = 10
    speckle_window  = 100
    speckle_range   = 32
    mode            = cv2.STEREO_SGBM_MODE_SGBM_3WAY

params = StereoParams()

# ─────────────────────────────────────────────
#  Thread-safe frame buffer
# ─────────────────────────────────────────────
class MJPEGStream:
    def __init__(self, url: str, name: str):
        self.url   = url
        self.name  = name
        self.frame = None
        self.lock  = threading.Lock()
        self.running = False
        self._thread = None

    def start(self):
        self.running = True
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

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
                print(f"[{self.name}] Connected to {self.url}")
                while self.running:
                    chunk = stream.read(4096)
                    if not chunk:
                        break
                    buf += chunk
                    # Find JPEG boundaries
                    a = buf.find(b'\xff\xd8')  # SOI
                    b_ = buf.find(b'\xff\xd9')  # EOI
                    if a != -1 and b_ != -1 and b_ > a:
                        jpg = buf[a:b_+2]
                        buf = buf[b_+2:]
                        arr = np.frombuffer(jpg, dtype=np.uint8)
                        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                        if frame is not None:
                            with self.lock:
                                self.frame = frame
            except Exception as e:
                print(f"[{self.name}] Stream error: {e} — reconnecting in 2s")
                time.sleep(2)

# ─────────────────────────────────────────────
#  Stereo calibration (placeholder / identity)
#  Replace map_L*, map_R* with your calibrated
#  values from stereo_calibrate.py
# ─────────────────────────────────────────────
class StereoCalibration:
    def __init__(self, img_size=(640, 480)):
        self.img_size = img_size
        self.calibrated = False
        # Identity maps — no rectification until calibrated
        self.map_lx, self.map_ly = None, None
        self.map_rx, self.map_ry = None, None
        self._try_load()

    def _try_load(self):
        calib_file = "stereo_calibration.npz"
        if os.path.exists(calib_file):
            data = np.load(calib_file)
            self.map_lx = data["map_lx"]
            self.map_ly = data["map_ly"]
            self.map_rx = data["map_rx"]
            self.map_ry = data["map_ry"]
            self.calibrated = True
            print(f"[Calibration] Loaded from {calib_file}")
        else:
            print("[Calibration] No calibration file found — using uncalibrated mode.")
            print("  Run stereo_calibrate.py to generate stereo_calibration.npz")

    def rectify(self, left, right):
        if not self.calibrated:
            return left, right
        h, w = left.shape[:2]
        if (w, h) != self.img_size:
            self.img_size = (w, h)
        l_rect = cv2.remap(left,  self.map_lx, self.map_ly, cv2.INTER_LINEAR)
        r_rect = cv2.remap(right, self.map_rx, self.map_ry, cv2.INTER_LINEAR)
        return l_rect, r_rect

# ─────────────────────────────────────────────
#  Build / rebuild the SGBM matcher
# ─────────────────────────────────────────────
def build_matcher():
    p = params
    bs = p.block_size | 1  # ensure odd
    matcher = cv2.StereoSGBM_create(
        minDisparity      = p.min_disparity,
        numDisparities    = p.num_disparities,
        blockSize         = bs,
        P1                = p.p1_factor  * 3 * bs * bs,
        P2                = p.p2_factor  * 3 * bs * bs,
        disp12MaxDiff     = p.disp12_max_diff,
        preFilterCap      = p.pre_filter_cap,
        uniquenessRatio   = p.uniqueness_ratio,
        speckleWindowSize = p.speckle_window,
        speckleRange      = p.speckle_range,
        mode              = p.mode,
    )
    # WLS filter for smoother disparity
    wls = cv2.ximgproc.createDisparityWLSFilter(matcher) if hasattr(cv2, 'ximgproc') else None
    right_matcher = cv2.ximgproc.createRightMatcher(matcher) if wls else None
    return matcher, wls, right_matcher

# ─────────────────────────────────────────────
#  Compute disparity and colour-map it
# ─────────────────────────────────────────────
def compute_disparity(left_gray, right_gray, matcher, wls=None, right_matcher=None):
    disp = matcher.compute(left_gray, right_gray)

    if wls and right_matcher:
        disp_r = right_matcher.compute(right_gray, left_gray)
        wls.setLambda(8000)
        wls.setSigmaColor(1.5)
        disp = wls.filter(disp, left_gray, disparity_map_right=disp_r)

    # Normalize to 0–255 for display
    disp_norm = cv2.normalize(disp, None, alpha=0, beta=255,
                              norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    disp_color = cv2.applyColorMap(disp_norm, cv2.COLORMAP_TURBO)
    return disp, disp_color

# ─────────────────────────────────────────────
#  Draw HUD overlay
# ─────────────────────────────────────────────
def draw_hud(frame, text_lines, color=(0, 255, 0)):
    y = 25
    for line in text_lines:
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, color, 1, cv2.LINE_AA)
        y += 22

def draw_epipolar_lines(frame, num_lines=10, color=(0, 200, 0)):
    h = frame.shape[0]
    step = h // num_lines
    for y in range(0, h, step):
        cv2.line(frame, (0, y), (frame.shape[1], y), color, 1)

# ─────────────────────────────────────────────
#  Main loop
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="ESP32-CAM Stereo Vision Processor")
    parser.add_argument("--left",  default=DEFAULT_LEFT_URL,  help="Left camera MJPEG URL")
    parser.add_argument("--right", default=DEFAULT_RIGHT_URL, help="Right camera MJPEG URL")
    parser.add_argument("--width",  type=int, default=640, help="Processing width")
    parser.add_argument("--height", type=int, default=480, help="Processing height")
    parser.add_argument("--no-display", action="store_true", help="Headless mode (save frames only)")
    args = parser.parse_args()

    proc_size = (args.width, args.height)

    print("="*55)
    print("  ESP32-CAM Stereo Vision Processor")
    print("="*55)
    print(f"  Left  stream : {args.left}")
    print(f"  Right stream : {args.right}")
    print(f"  Processing   : {proc_size[0]}×{proc_size[1]}")
    print("="*55)

    # Start camera streams
    left_stream  = MJPEGStream(args.left,  "LEFT")
    right_stream = MJPEGStream(args.right, "RIGHT")
    left_stream.start()
    right_stream.start()

    calib   = StereoCalibration(img_size=proc_size)
    matcher, wls, right_matcher = build_matcher()

    show_calib_lines = False
    fullscreen       = False
    save_counter     = 0
    frame_count      = 0
    fps_time         = time.time()
    fps              = 0.0
    rebuild_matcher  = False

    win = "Stereo Vision — ESP32-CAM"
    if not args.no_display:
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, proc_size[0] * 2, proc_size[1] * 2)

    print("\nControls: q/ESC=quit | c=epipolar lines | s=save | +/-=disparities | [/]=blocksize | r=reset | f=fullscreen\n")

    while True:
        lf = left_stream.get_frame()
        rf = right_stream.get_frame()

        if lf is None or rf is None:
            time.sleep(0.05)
            continue

        # Resize to processing resolution
        lf = cv2.resize(lf, proc_size)
        rf = cv2.resize(rf, proc_size)

        # Rectify
        lf_rect, rf_rect = calib.rectify(lf, rf)

        # Grayscale for matching
        lg = cv2.cvtColor(lf_rect, cv2.COLOR_BGR2GRAY)
        rg = cv2.cvtColor(rf_rect, cv2.COLOR_BGR2GRAY)

        # Compute disparity
        if rebuild_matcher:
            matcher, wls, right_matcher = build_matcher()
            rebuild_matcher = False

        disp_raw, disp_color = compute_disparity(lg, rg, matcher, wls, right_matcher)

        # FPS
        frame_count += 1
        now = time.time()
        if now - fps_time >= 1.0:
            fps = frame_count / (now - fps_time)
            frame_count = 0
            fps_time = now

        # Overlays
        if show_calib_lines:
            draw_epipolar_lines(lf_rect)
            draw_epipolar_lines(rf_rect)
            draw_epipolar_lines(disp_color)

        hud_lines = [
            f"FPS: {fps:.1f}",
            f"numDisp: {params.num_disparities}  blockSize: {params.block_size}",
            f"Calib: {'YES' if calib.calibrated else 'NO (uncalibrated)'}",
            "s=save  c=epipolar  +/-  [/]",
        ]
        draw_hud(lf_rect, hud_lines[:1])
        draw_hud(disp_color, hud_lines)

        # Compose display: [Left | Right | Disparity]
        top_row    = np.hstack([lf_rect, rf_rect])
        h, w       = top_row.shape[:2]
        disp_wide  = cv2.resize(disp_color, (w, proc_size[1]))
        display    = np.vstack([top_row, disp_wide])

        if not args.no_display:
            cv2.imshow(win, display)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord('q'), 27):
                break
            elif key == ord('c'):
                show_calib_lines = not show_calib_lines
            elif key == ord('s'):
                ts = int(time.time())
                cv2.imwrite(f"stereo_left_{ts}.png",  lf_rect)
                cv2.imwrite(f"stereo_right_{ts}.png", rf_rect)
                cv2.imwrite(f"stereo_disp_{ts}.png",  disp_color)
                print(f"[Save] Saved frame pair #{save_counter} (ts={ts})")
                save_counter += 1
            elif key == ord('+') or key == ord('='):
                params.num_disparities = min(256, params.num_disparities + 16)
                rebuild_matcher = True
                print(f"numDisparities → {params.num_disparities}")
            elif key == ord('-'):
                params.num_disparities = max(16, params.num_disparities - 16)
                rebuild_matcher = True
                print(f"numDisparities → {params.num_disparities}")
            elif key == ord(']'):
                params.block_size = min(51, params.block_size + 2)
                rebuild_matcher = True
                print(f"blockSize → {params.block_size}")
            elif key == ord('['):
                params.block_size = max(5, params.block_size - 2)
                rebuild_matcher = True
                print(f"blockSize → {params.block_size}")
            elif key == ord('r'):
                params.num_disparities = 64
                params.block_size = 11
                rebuild_matcher = True
                print("Parameters reset to defaults")
            elif key == ord('f'):
                fullscreen = not fullscreen
                prop = cv2.WND_PROP_FULLSCREEN
                mode = cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL
                cv2.setWindowProperty(win, prop, mode)

    left_stream.stop()
    right_stream.stop()
    cv2.destroyAllWindows()
    print("Exited.")

if __name__ == "__main__":
    main()

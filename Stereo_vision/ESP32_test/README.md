# ESP32-CAM Stereo Vision System

## Files
| File | Purpose |
|------|---------|
| `esp32cam_stereo.ino` | Arduino sketch — flash to BOTH ESP32-CAMs |
| `stereo_calibrate.py` | One-time calibration using a chessboard |
| `stereo_processor.py` | Real-time disparity / depth display |

---

## Hardware Setup

```
   LEFT ESP32-CAM          RIGHT ESP32-CAM
   ┌───────────┐           ┌───────────┐
   │  OV2640   │←──BLINE──→│  OV2640   │
   └─────┬─────┘           └─────┬─────┘
         │ WiFi                  │ WiFi
         └──────────┬────────────┘
                    │
            PC / Raspberry Pi 4B
```

**Baseline (distance between cameras):** 6–12 cm works well.  
Mount both cameras on a rigid bar, lenses perfectly parallel and level.

---

## Step 1 — Flash the Arduino Sketch

1. Open `esp32cam_stereo.ino` in Arduino IDE 2.x.
2. Install board support: `espressif/arduino-esp32` (via Boards Manager).
3. For the **LEFT** camera:  
   - Set `CAMERA_ID 0`  
   - Set `WIFI_SSID` / `WIFI_PASSWORD`  
   - Set `staticIP` to `192.168.1.10` (or use DHCP and note the IP)
4. For the **RIGHT** camera:  
   - Set `CAMERA_ID 1`  
   - Set `staticIP` to `192.168.1.11`
5. Select board: **AI Thinker ESP32-CAM**  
   Upload speed: **115200** for programming, use a USB-to-TTL adapter (GPIO0 → GND during flash).
6. Open Serial Monitor → you should see the stream URL.
7. Visit `http://192.168.1.10/stream` in a browser to verify.

---

## Step 2 — Install Python Dependencies

**PC (Windows/Linux/macOS):**
```bash
pip install opencv-python numpy
# Optional but recommended for WLS filtered disparity:
pip install opencv-contrib-python
```

**Raspberry Pi 4B:**
```bash
sudo apt update && sudo apt install python3-pip -y
pip install opencv-python-headless numpy
# Headless build is faster on Pi; GUI works too:
# pip install opencv-python numpy
```

---

## Step 3 — Calibrate (Recommended)

Print a **9×6 chessboard** (inner corners) with 25 mm squares.

```bash
python stereo_calibrate.py \
  --left  http://192.168.1.10/stream \
  --right http://192.168.1.11/stream
```

- Point both cameras at the board from different angles.
- Press **SPACE** to capture (~15 good pairs).
- Press **ENTER** to compute & save `stereo_calibration.npz`.

---

## Step 4 — Run the Stereo Processor

```bash
python stereo_processor.py \
  --left  http://192.168.1.10/stream \
  --right http://192.168.1.11/stream
```

### Display layout

```
┌──────────────────┬──────────────────┐
│   LEFT (rect.)   │   RIGHT (rect.)  │
├──────────────────┴──────────────────┤
│          DISPARITY MAP              │
│    (warm = close, cool = far)       │
└─────────────────────────────────────┘
```

### Runtime controls

| Key | Action |
|-----|--------|
| `q` / `ESC` | Quit |
| `c` | Toggle epipolar lines overlay |
| `s` | Save frame pair + disparity as PNG |
| `+` / `-` | Increase / decrease numDisparities |
| `]` / `[` | Increase / decrease blockSize |
| `r` | Reset disparity parameters |
| `f` | Toggle fullscreen |

---

## Tuning Tips

| Parameter | Effect | Start value |
|-----------|--------|------------|
| `numDisparities` | Max depth range (must be ÷16) | 64 |
| `blockSize` | Matching window; larger = smoother but less detail | 11 |

- **Far objects only visible** → increase `numDisparities`  
- **Noisy / speckled map** → increase `blockSize` or `speckle_window`  
- **Slow on Raspberry Pi** → reduce `--width 320 --height 240`

---

## Raspberry Pi 4B Performance Notes

| Resolution | Approx. FPS (Pi 4B) |
|------------|---------------------|
| 320×240    | ~10 fps             |
| 640×480    | ~4–5 fps            |

For better Pi performance, consider installing `opencv-contrib-python` and enabling the WLS filter, or lowering camera JPEG quality in the sketch (`jpeg_quality = 20`).

---

## Depth Estimation (Optional)

After calibration, the `Q` matrix in `stereo_calibration.npz` lets you reproject disparity to real-world 3-D:

```python
import numpy as np, cv2
data = np.load("stereo_calibration.npz")
Q    = data["Q"]
# disp_raw from stereo_processor compute_disparity()
points_3d = cv2.reprojectImageTo3D(disp_raw, Q)
# points_3d[y, x] = (X_mm, Y_mm, Z_mm)
depth_map = points_3d[:, :, 2]
```

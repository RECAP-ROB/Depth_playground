import cv2

for i in range(6):
    cap = cv2.VideoCapture(i, cv2.CAP_V4L2)
    ret, frame = cap.read()
    if ret:
        print(f"Index {i}: ✓ Working — {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
    else:
        print(f"Index {i}: ✗ Failed")
    cap.release()
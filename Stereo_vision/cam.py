import cv2

for i in [2, 4]:
    cap = cv2.VideoCapture(i, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 15)
    
    opened = cap.isOpened()
    ret, frame = cap.read()
    actual_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    
    print(f"Index {i}: opened={opened}, read={ret}, actual_res={actual_w}x{actual_h}, frame={'ok' if frame is not None else 'None'}")
    cap.release()
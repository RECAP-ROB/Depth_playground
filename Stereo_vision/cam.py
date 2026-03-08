import cv2

for idx in [0, 2]:
    cap = cv2.VideoCapture(idx)
    ret, frame = cap.read()
    if ret:
        cv2.imshow(f"Camera {idx}", frame)
    else:
        print(f"Camera {idx} failed")
    cap.release()

cv2.waitKey(0)
cv2.destroyAllWindows()
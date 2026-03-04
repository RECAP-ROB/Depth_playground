import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
# model.to("cuda")  # remove if no GPU

cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

frame_count = 0
annotated_frame = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_count % 2 == 0:
        small = cv2.resize(frame, (320, 240))
        results = model(small, verbose=False)  # verbose=False reduces console spam
        annotated_frame = cv2.resize(results[0].plot(), (640, 480))

    if annotated_frame is not None:
        cv2.imshow("YOLO Detection", annotated_frame)

    frame_count += 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
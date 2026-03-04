from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(1)

KNOWN_WIDTH = 12  # inches
KNOWN_DISTANCE = 12  # inches (1 foot)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, verbose=False)

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id]

            if label == "laptop":  # change to your object
                x1, y1, x2, y2 = box.xyxy[0]
                x1, x2 = int(x1), int(x2)

                pixel_width = x2 - x1
                print("Pixel Width at 1ft:", pixel_width)

                cv2.rectangle(frame, (int(x1), int(y1)),
                              (int(x2), int(y2)), (0,255,0), 2)

    cv2.imshow("Calibration", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
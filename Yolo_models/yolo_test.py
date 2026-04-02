import onnxruntime as ort
import numpy as np
import cv2
import os

# 1. Setup Paths and Session
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, "yolo11n.onnx")

# Use 'CUDAExecutionProvider' if you have an NVIDIA GPU and library installed
providers = ['CPUExecutionProvider']
session = ort.InferenceSession(model_path, providers=providers)

# Get model metadata
model_inputs = session.get_inputs()
input_name = model_inputs[0].name
input_shape = model_inputs[0].shape  # Expecting [1, 3, 640, 640]

COCO_NAMES = [
    'person','bicycle','car','motorcycle','airplane','bus','train','truck','boat','traffic light',
    'fire hydrant','stop sign','parking meter','bench','bird','cat','dog','horse','sheep','cow',
    'elephant','bear','zebra','giraffe','backpack','umbrella','handbag','tie','suitcase','frisbee',
    'skis','snowboard','sports ball','kite','baseball bat','baseball glove','skateboard','surfboard','tennis racket',
    'bottle','wine glass','cup','fork','knife','spoon','bowl','banana','apple','sandwich','orange','broccoli','carrot','hot dog','pizza','donut','cake',
    'chair','couch','potted plant','bed','dining table','toilet','tv','laptop','mouse','remote','keyboard','cell phone','microwave','oven','toaster','sink','refrigerator',
    'book','clock','vase','scissors','teddy bear','hair drier','toothbrush'
]

# Constants
# Change these to match your specific model export
INPUT_WIDTH = 640
INPUT_HEIGHT = 480
CONF_THRESHOLD = 0.3
IOU_THRESHOLD = 0.45

cap = cv2.VideoCapture(0) # Changed to 0 as it's standard, change back to 1 if needed

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # --- Preprocessing ---
    h0, w0 = frame.shape[:2]
    # Resize and normalize
    img = cv2.resize(frame, (INPUT_WIDTH, INPUT_HEIGHT))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    # HWC to CHW and add batch dimension: [1, 3, 640, 640]
    img = img.transpose(2, 0, 1)
    blob = np.expand_dims(img, axis=0)

    # --- Inference ---
    outputs = session.run(None, {input_name: blob})
    preds = outputs[0]  # Shape: (1, 84, 8400) for YOLOv11n

    # --- Post-processing ---
    # Transpose to (8400, 84)
    preds = np.squeeze(preds).T 
    
    boxes = []
    confidences = []
    class_ids = []

    # YOLOv11 output: [x_center, y_center, width, height, class0, class1, ...]
    for i in range(len(preds)):
        row = preds[i]
        classes_scores = row[4:]
        class_id = np.argmax(classes_scores)
        conf = classes_scores[class_id]

        if conf > CONF_THRESHOLD:
            # Scale coordinates back to original frame size
            x_c, y_c, w, h = row[0:4]
            
            # x_c, y_c, w, h are based on 640x640, we need to scale to h0, w0
            x = int((x_c - w/2) * (w0 / INPUT_WIDTH))
            y = int((y_c - h/2) * (h0 / INPUT_HEIGHT))
            width = int(w * (w0 / INPUT_WIDTH))
            height = int(h * (h0 / INPUT_HEIGHT))

            boxes.append([x, y, width, height])
            confidences.append(float(conf))
            class_ids.append(class_id)

    # Apply NMS (Standard OpenCV NMS works fine with ORT results)
    indices = cv2.dnn.NMSBoxes(boxes, confidences, CONF_THRESHOLD, IOU_THRESHOLD)

    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i]
            label = COCO_NAMES[class_ids[i]] if class_ids[i] < len(COCO_NAMES) else str(class_ids[i])
            score = confidences[i]
            
            # Draw
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {score:.2f}", (x, y - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("YOLOv11 ONNX Runtime", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
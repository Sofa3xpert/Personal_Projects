# detect_yolo.py
import cv2
import numpy as np
import onnxruntime as ort
from collections import Counter

# Load COCO class labels
with open("coco.names", "r") as f:
    labels = [line.strip() for line in f.readlines()]

# Initialize YOLOv5 ONNX session
session = ort.InferenceSession("yolov5n.onnx", providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name
input_shape = session.get_inputs()[0].shape
img_size = input_shape[2]  # e.g. 640

# Start camera capture
cap = cv2.VideoCapture(0)

print("Starting detection (press Ctrl+C to stop)...")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        # Resize and prepare frame
        img = cv2.resize(frame, (img_size, img_size))
        img = img.transpose(2, 0, 1) / 255.0  # HWC to CHW
        img = np.expand_dims(img, axis=0).astype(np.float32)

        # Run model
        outputs = session.run(None, {input_name: img})[0]

        # Parse detections
        detected_classes = []
        for det in outputs[0]:
            conf = det[4]
            if conf < 0.4:
                continue
            class_id = int(det[5])
            detected_classes.append(labels[class_id])

        if detected_classes:
            counts = Counter(detected_classes)
            print("Detected:", dict(counts))
        else:
            print("No objects detected.")

except KeyboardInterrupt:
    print("Stopped by user")

cap.release()

"""
Gloved vs Ungloved Hand Detection
Inference-only pipeline using YOLOv8 pretrained COCO model.
Non-hand classes are ignored explicitly.
"""

import os
import json
import cv2
from ultralytics import YOLO

INPUT_DIR = "input_images"
OUTPUT_DIR = "output"
LOG_DIR = "logs"

CONFIDENCE_THRESHOLD = 0.5

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# Load COCO-pretrained model
model = YOLO("yolov8n.pt")

ALLOWED_CLASS = "person"  # proxy for hand region

for image_name in os.listdir(INPUT_DIR):

    if not image_name.lower().endswith(".jpg"):
        continue

    image_path = os.path.join(INPUT_DIR, image_name)
    image = cv2.imread(image_path)

    results = model(image, conf=CONFIDENCE_THRESHOLD)[0]
    detections = []

    for box in results.boxes:
        class_id = int(box.cls[0])
        coco_label = model.names[class_id]
        confidence = float(box.conf[0])

        # Ignore everything except person
        if coco_label != ALLOWED_CLASS:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # Simple heuristic:
        # Bright / uniform texture → glove
        # Skin texture → bare hand
        roi = image[y1:y2, x1:x2]
        mean_intensity = roi.mean() if roi.size > 0 else 0

        if mean_intensity > 150:
            label = "gloved_hand"
        else:
            label = "bare_hand"

        detections.append({
            "label": label,
            "confidence": round(confidence, 2),
            "bbox": [x1, y1, x2, y2]
        })

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            image,
            f"{label} {confidence:.2f}",
            (x1, y1 - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2
        )

    cv2.imwrite(os.path.join(OUTPUT_DIR, image_name), image)

    with open(os.path.join(LOG_DIR, image_name.replace(".jpg", ".json")), "w") as f:
        json.dump({
            "filename": image_name,
            "detections": detections
        }, f, indent=2)

print("Detection completed successfully.")

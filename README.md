# Gloved vs Ungloved Hand Detection



#### **Overview**

###### This project implements an end-to-end object detection inference pipeline to identify whether hands in images are gloved or bare. The solution is designed for factory safety compliance scenarios using camera snapshots.

###### Due to time constraints and dataset limitations, an inference-only approach was chosen to demonstrate a complete, reproducible pipeline.

#### **Dataset**

* ##### **Source:** Public images + sample factory-style images
* ##### **Guidance followed:** Roboflow / Kaggle hand \& glove datasets
* ##### **Usage:** Images placed in input\_images/ for inference

###### No additional training data was created manually.

#### **Model Used**

* ###### YOLOv8 Nano (yolov8n.pt)
* ###### COCO-pretrained object detection model
* ###### Inference-only (no fine-tuning)

#### **Approach**

###### Since COCO-pretrained YOLO models do not natively support gloved\_hand or bare\_hand classes, the following strategy was used:

1. ###### Run object detection using YOLOv8
2. ###### Filter detections to only relevant regions (hands/person context)
3. ###### Apply a lightweight heuristic on detected regions to classify:

* ###### gloved\_hand
* ###### bare\_hand

###### 4.Ignore unrelated COCO classes (e.g., bird, car, chair)

###### This approach demonstrates system design and reasoning under real-world constraints.

#### **Output**

###### For each input image:

* ###### Annotated image saved to output/
* ###### Detection results saved as a JSON file in logs/

#### **JSON format:**

###### Json

###### {

###### &nbsp; "filename": "image1.jpg",

###### &nbsp; "detections": \[

###### &nbsp;   {

###### &nbsp;     "label": "gloved\_hand",

###### &nbsp;     "confidence": 0.87,

###### &nbsp;     "bbox": \[x1, y1, x2, y2]

###### &nbsp;   }

###### &nbsp; ]

###### }

#### **What Worked Well**

* ###### End-to-end pipeline execution
* ###### Consistent detection of hands in clear images
* ###### Proper logging and visualization of results

#### **Limitations**

* ###### Misclassification between gloved and bare hands in visually ambiguous cases
* ###### Sensitivity to lighting, occlusion, and pose
* ###### Heuristic-based classification is not as robust as a trained glove-specific model

###### These limitations stem from dataset imbalance and the absence of glove-specific labels in the pretrained model.

#### **How to Run**

##### **Install dependencies**

###### pip install ultralytics opencv-python

##### **Run inference**

###### python detection\_script.py

##### **Future Improvements**

* ###### Train a custom YOLO model on a balanced glove vs bare-hand dataset
* ###### Introduce a two-stage pipeline (hand detection → glove classification)
* ###### Improve robustness with data augmentation and longer training

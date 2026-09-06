# AI-Based Garbage Detection Using YOLO

## 1. Project Overview

This project implements an AI-based garbage detection system using
YOLO object detection and deep learning.

The system takes an image as input and detects different types of
garbage present in the image using bounding boxes, class labels,
and confidence scores.

The project is designed as a prototype for smart waste management
and automated waste identification.

---

## 2. Objectives

- Detect garbage objects from images.
- Identify different categories of waste.
- Localize detected objects using bounding boxes.
- Display the confidence score for each detection.
- Develop a simple user interface for image-based garbage detection.
- Explore the use of deep learning and computer vision for smart
  waste management.

---

## 3. Technologies Used

- Python
- YOLO
- Ultralytics
- PyTorch
- OpenCV
- Streamlit
- Pillow
- NumPy
- Intel Arc GPU with PyTorch XPU acceleration

---

## 4. Dataset

The project uses an annotated garbage dataset containing:

- 42 waste/object classes
- 11,466 training images
- 1,092 validation images
- 546 test images

The dataset is organized into training, validation, and testing
subsets.

YOLO annotation format is used for object detection.

---

## 5. Model

The project uses the YOLO26n object detection model.

The model was trained using:

- Image size: 320 × 320
- Epochs: 6
- Batch size: 4
- Device: Intel Arc XPU

The trained model is stored as:

runs/detect/train-5/weights/best.pt

---

## 6. Model Evaluation

The final model was evaluated on the validation dataset.

Results:

| Metric | Value |
|---|---:|
| Precision | 34.3% |
| Recall | 20.5% |
| mAP@50 | 23.5% |
| mAP@50-95 | 16.9% |

The model performs better on classes with larger numbers of
training examples.

For example, some of the stronger classes include Aluminum can,
Cardboard, Glass bottle, Organic, Plastic cup, and Printing industry.

Several classes have very few samples, which affects their
detection performance.

---

## 7. System Workflow

```text
Input Image
     ↓
Image Preprocessing
     ↓
YOLO Object Detection Model
     ↓
Feature Extraction and Detection
     ↓
Bounding Box Prediction
     ↓
Class Prediction + Confidence Score
     ↓
Detected Garbage Objects

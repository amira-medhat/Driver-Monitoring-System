import sys
import os
import cv2
import torch
import numpy as np
import random
from pathlib import Path

# Configuration
CUSTOM_WEIGHTS_PATH = r"D:\GRAD_PROJECT\best (2).pt"  # Path to your trained weights file
IMAGES_FOLDER = r"D:\GRAD_PROJECT\Driver-Monitoring-System\Activity Detection\object_detection_HOW\how_yolov7\uncle_aya" # Folder containing images for inference
YOLOV7_REPO_PATH = r"D:\GRAD_PROJECT\Driver-Monitoring-System\Activity Detection\object_detection_HOW\how_yolov7\yolov7"  # Path to YOLOv7 repository
CONFIDENCE_THRESHOLD = 0.45  # Lowered for more detections

# Check if weights file exists
if not os.path.exists(CUSTOM_WEIGHTS_PATH):
    raise FileNotFoundError(f"Custom weights file not found: {CUSTOM_WEIGHTS_PATH}")

# Check if images folder exists
if not os.path.exists(IMAGES_FOLDER):
    raise FileNotFoundError(f"Images folder not found: {IMAGES_FOLDER}")

# Add YOLOv7 repo to the Python path
sys.path.append(YOLOV7_REPO_PATH)

# Now we can import the necessary modules from YOLOv7
from models.yolo import Model
from utils.datasets import LoadImages
from utils.general import non_max_suppression, scale_coords
from utils.torch_utils import select_device

# Initialize device (GPU or CPU)
device = 'cpu'  # Force CPU mode

# Load YOLOv7 model
model = torch.hub.load(YOLOV7_REPO_PATH, 'custom', CUSTOM_WEIGHTS_PATH, source='local', force_reload=True).to(device)
model.eval()

# Get list of images in the folder
images = [f for f in os.listdir(IMAGES_FOLDER) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
if not images:
    raise FileNotFoundError(f"No image files found in folder: {IMAGES_FOLDER}")

# Shuffle the images
random.shuffle(images)

# Perform inference on each image
with torch.no_grad():
    for image_name in images:
        image_path = os.path.join(IMAGES_FOLDER, image_name)
        print(f"Processing image: {image_path}")

        # Read the image
        image = cv2.imread(image_path)
        original_image = image.copy()  # Keep a copy for display

        # Preprocess the image
        img_resized = cv2.resize(image, (640, 640))  # Resize for model input
        img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).unsqueeze(0).to(device).float() / 255.0

        # Run inference
        predictions = model(img_tensor)

        # Extract detections
        detections = predictions[0] if isinstance(predictions, (list, tuple)) else predictions
        detections = non_max_suppression(detections, CONFIDENCE_THRESHOLD, 0.45)[0]  # Apply non-max suppression

        if detections is not None and len(detections):
            # Process predictions
            detections[:, :4] = scale_coords(img_tensor.shape[2:], detections[:, :4], original_image.shape).round()

            for *xyxy, conf, cls in detections:
                # Convert to int for bounding box coordinates
                x1, y1, x2, y2 = map(int, xyxy)

                # Skip detections below the confidence threshold
                if conf < CONFIDENCE_THRESHOLD:
                    continue

                # notice that if the dataset is augmented you should check yaml and the labels
                # because roboflow flip all the labels
                # if augmented : handson -> 0 and handsoff -> 1
                # Determine label and color
                label = f"{'HandsOffWheel' if cls == 1 else 'HandsOnWheel'} {conf:.2f}"
                color = (0,0, 255) if cls == 1 else (0, 255, 0)

                # Draw bounding box
                cv2.rectangle(original_image, (x1, y1), (x2, y2), color, 2)

        # Calculate the position for the label at the bottom center of the image
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        label_x = (original_image.shape[1] - label_size[0]) // 2  # Centered horizontally
        label_y = original_image.shape[0] - 10  # Positioned near the bottom

        # Draw the label in the lower middle part of the image
        cv2.putText(original_image, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Display the image with bounding boxes and labels
        cv2.imshow("Detection", original_image)

        # Wait for user to close the image window or proceed to the next image
        key = cv2.waitKey(0)  # Wait indefinitely until a key is pressed
        if key == ord('q'):  # Quit if 'q' is pressed
            break

# Close all OpenCV windows
cv2.destroyAllWindows()
print("Inference on images completed.")
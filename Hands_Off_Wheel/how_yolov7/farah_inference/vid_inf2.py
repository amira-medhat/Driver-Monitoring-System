import sys
import os
import cv2
import torch
import time
import numpy as np
from pathlib import Path
import threading
from queue import Queue

# Configuration
CUSTOM_WEIGHTS_PATH = r"D:\GRAD_PROJECT\best (2).pt"
YOLOV7_REPO_PATH = r"D:\GRAD_PROJECT\Driver-Monitoring-System\Activity Detection\object_detection_HOW\how_yolov7\yolov7"
VIDEO_PATH = r"C:\Users\Farah\Downloads\20250219_113036 (1).mp4"
CONFIDENCE_THRESHOLD = 0.5  

# Check files
if not os.path.exists(CUSTOM_WEIGHTS_PATH):
    raise FileNotFoundError(f"Custom weights file not found: {CUSTOM_WEIGHTS_PATH}")

if not os.path.exists(VIDEO_PATH):
    raise FileNotFoundError(f"Video file not found: {VIDEO_PATH}")

# Add YOLOv7 repo to path
sys.path.append(YOLOV7_REPO_PATH)

# Import YOLOv7 utilities
from utils.general import non_max_suppression, scale_coords
from utils.torch_utils import select_device

# Initialize device (GPU/CPU)
device = select_device('cuda' if torch.cuda.is_available() else 'cpu')

# Load YOLOv7 model
model = torch.hub.load(YOLOV7_REPO_PATH, 'custom', CUSTOM_WEIGHTS_PATH, source='local', force_reload=True).to(device)
model.eval()
if device.type != 'cpu':
    model.half()  

# Open video file
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise Exception("Error: Could not open video file.")

# Get video properties
video_fps = cap.get(cv2.CAP_PROP_FPS)  # Dynamic FPS retrieval
frame_time = 1 / video_fps  # Set frame timing dynamically
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Multi-threaded frame reading
frame_queue = Queue(maxsize=5)

def read_frames():
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_queue.put(frame)

threading.Thread(target=read_frames, daemon=True).start()

frame_skip = 5  
frame_count = 0  

while cap.isOpened():
    if frame_queue.empty():
        continue  
    
    start_time = time.time()  # Start timing for current frame
    frame = frame_queue.get()
    frame_count += 1
    if frame_count % frame_skip != 0:
        continue  

    # Preprocess frame
    img_resized = cv2.resize(frame, (640, 640))
    img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).unsqueeze(0).contiguous().to(device).float() / 255.0
    if device.type != 'cpu':
        img_tensor = img_tensor.half()  

    # Run inference
    with torch.no_grad():
        predictions = model(img_tensor)

    if isinstance(predictions, tuple):
        predictions = predictions[0]

    # Extract detections
    detections = non_max_suppression(predictions, CONFIDENCE_THRESHOLD, 0.45)[0]
    
    label = "No Detection"
    color = (255, 255, 255)
    if detections is not None and len(detections):
        detections[:, :4] = scale_coords(img_tensor.shape[2:], detections[:, :4], frame.shape).round()
        
    for *xyxy, conf, cls in detections:
        x1, y1, x2, y2 = map(int, xyxy)
        if conf < CONFIDENCE_THRESHOLD:
            continue

        # Compute new height and adjust bounding box
        new_height = int((y2 - y1) * 0.9)  
        center_y = (y1 + y2) // 2
        new_y1 = max(center_y - new_height // 2, 0)
        new_y2 = min(center_y + new_height // 2, frame.shape[0])

        # Assign label and color
        label = f"{'HandsOffWheel' if cls == 1 else 'HandsOnWheel'} {conf:.2f}"
        color = (0, 0, 255) if cls == 1 else (0, 255, 0)

        # Draw bounding box
        cv2.rectangle(frame, (x1, new_y1), (x2, new_y2), color, 2)

    # Display label
    cv2.putText(frame, label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # Resize display window
    frame_resized = cv2.resize(frame, (480, 360))
    cv2.imshow("YOLOv7 Inference", frame_resized)

    # Adjust delay dynamically to maintain smooth playback
    elapsed_time = time.time() - start_time
    delay = max(1, int((frame_time - elapsed_time) * 1000))  

    if cv2.waitKey(delay) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()

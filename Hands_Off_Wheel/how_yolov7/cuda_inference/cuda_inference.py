import sys
import cv2
import torch
import numpy as np
from pathlib import Path

YOLOV7_REPO_PATH = r"C:\Users\Amira\Driver-Monitoring-System\Activity Detection\object_detection_HOW\how_yolov7\yolov7"

# Add YOLOv7 repo to the Python path
sys.path.append(YOLOV7_REPO_PATH)

from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords, check_img_size
from utils.torch_utils import select_device
from utils.datasets import letterbox

# Global adjustable thresholds
CONF_THRESHOLD = 0.5  # Confidence threshold for detection
IOU_THRESHOLD = 0.45   # IOU threshold for non-max suppression

weights_path = r"C:\Users\Amira\Driver-Monitoring-System\Activity Detection\object_detection_HOW\how_yolov7\best_fine_tune.pt"  # Update this path if necessary
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model = attempt_load(weights_path, map_location=device)
stride = int(model.stride.max())  # Model stride
img_size = check_img_size(640, s=stride)  # Image size

# Load video
video_path = r"D:\GP_datasets\without_aug\state-farm-distracted-driver-detection\frames_from_videos\WhatsApp Video 2025-02-17 at 16.42.14_52788729.mp4"  # Change to your video file path
cap = cv2.VideoCapture(video_path)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define codec and create VideoWriter object
out = cv2.VideoWriter("output_video.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Preprocess frame
    img = letterbox(frame, img_size, stride=stride)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)  # Convert BGR to RGB and HWC to CHW
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    img = img.float() / 255.0  # Normalize
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    
    # Run inference
    with torch.no_grad():
        pred = model(img)[0]
    
    # Apply Non-Maximum Suppression (NMS) with adjustable IOU threshold
    pred = non_max_suppression(pred, CONF_THRESHOLD, IOU_THRESHOLD, agnostic=False)
    
    # Process detections
    highest_conf = 0
    highest_cls = None
    best_box = None
    
    for det in pred:
        if len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], frame.shape).round()
            
            for *xyxy, conf, cls in det:
                cls = int(cls)
                if conf > highest_conf:
                    highest_conf = conf
                    highest_cls = cls
                    best_box = xyxy
                
    # Draw only the highest confidence bounding box
    if best_box is not None:
        color = (0, 255, 0) if highest_cls == 1 else (0, 0, 255)  # Green for HandsOnWheel, Red for HandsOffWheel
        cv2.rectangle(frame, (int(best_box[0]), int(best_box[1])), (int(best_box[2]), int(best_box[3])), color, 2)
        cv2.putText(frame, f"{highest_cls} {highest_conf:.2f}", (int(best_box[0]), int(best_box[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Display the detected class name
        class_name = "HandsOnWheel" if highest_cls == 1 else "HandsOffWheel"
        cv2.putText(frame, f"Detected: {class_name}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Save and display the processed frame
    out.write(frame)
    cv2.imshow("YOLOv7 Inference", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

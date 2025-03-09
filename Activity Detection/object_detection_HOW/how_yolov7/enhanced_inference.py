import os
import cv2
import torch
import numpy as np
import sys
sys.path.append("D:\GRAD_PROJECT\Driver-Monitoring-System\Activity Detection\object_detection_HOW\how_yolov7\yolov7")  # Add YOLOv7 to Python path
from utils.general import non_max_suppression  # Corrected import


# Paths
CUSTOM_WEIGHTS_PATH = r"D:\GRAD_PROJECT\Driver-Monitoring-System\Activity Detection\object_detection_HOW\how_yolov7\best_lastTrain.pt"  # weights
IMAGE_PATH = r"D:\GRAD_PROJECT\Driver-Monitoring-System\Activity Detection\object_detection_HOW\how_yolov7\uncle_aya\nadaaaaaaaaaadefe.jpg"  
# Thresholds
CONFIDENCE_THRESHOLD_NORMAL = 0.6  # Confidence for normal lighting
CONFIDENCE_THRESHOLD_NIGHT = 0.5   # Lower confidence for night mode
BRIGHTNESS_THRESHOLD = 100         # Brightness threshold to detect night mode
IOU_THRESHOLD = 0.45               # Non-Maximum Suppression IoU threshold

# Check file existence
if not os.path.exists(CUSTOM_WEIGHTS_PATH):
    raise FileNotFoundError(f"❌ Weights file not found: {CUSTOM_WEIGHTS_PATH}")

if not os.path.exists(IMAGE_PATH):
    raise FileNotFoundError(f"❌ Image file not found: {IMAGE_PATH}")

# Class Names
CLASS_NAMES = ["HandsOnWheel", "HandsOffWheel"]

# Initialize device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Using device: {device}")

# Load YOLOv7 model
model = torch.hub.load("WongKinYiu/yolov7", "custom", "best_lastTrain.pt")
model.eval()

# Read Image
image = cv2.imread(IMAGE_PATH)

# Check if image is loaded
if image is None:
    print(f"❌ Error: Image not found at {IMAGE_PATH}")
    exit()

print(f"✅ Image loaded successfully from {IMAGE_PATH}")

# Step 1: Detect Brightness Level
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
brightness = np.mean(gray)  # Compute average brightness

# Step 2: Apply Processing Based on Brightness
if brightness < BRIGHTNESS_THRESHOLD:
    print("🌙 Detected Night Mode - Enhancing Image for Better Detection")

    # Apply CLAHE (Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    enhanced = clahe.apply(gray)

    # Convert Back to BGR (YOLOv7 Requires 3 Channels)
    processed_image = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

    # Lower confidence threshold for night mode
    CONFIDENCE_THRESHOLD = CONFIDENCE_THRESHOLD_NIGHT

else:
    print("☀️ Normal Lighting Detected - Applying Histogram Equalization")

    # Apply Histogram Equalization to Normal Light Image
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])  # Equalize Y channel
    processed_image = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

    # Normal confidence threshold
    CONFIDENCE_THRESHOLD = CONFIDENCE_THRESHOLD_NORMAL

# Resize Image to Model Input Size (640x640)
processed_image = cv2.resize(processed_image, (640, 640))

# Convert to tensor and move to GPU
img_tensor = torch.from_numpy(processed_image).permute(2, 0, 1).unsqueeze(0).float() / 255.0
img_tensor = img_tensor.to(device)

# Run inference
with torch.no_grad():
    predictions = model(img_tensor)

    # Extract first tensor if model returns a tuple
    if isinstance(predictions, tuple):
        predictions = predictions[0]

    # Apply non-max suppression
    detections = non_max_suppression(predictions, CONFIDENCE_THRESHOLD, IOU_THRESHOLD)[0]

    # Process detections
    if detections is not None and len(detections):
        print(f"✅ Found {len(detections)} objects in image")

        for *xyxy, conf, cls in detections:
            x1, y1, x2, y2 = map(int, xyxy)

            # Apply confidence threshold dynamically
            if conf < CONFIDENCE_THRESHOLD:
                print(f"⚠ Skipping low-confidence detection ({conf:.2f})")
                continue

            class_index = int(cls)  # Convert class tensor to integer index

            # Class Label Mapping
            if class_index == 0:
                class_name = "HandsOffWheel"
            elif class_index == 1:
                class_name = "HandsOnWheel"
            else:
                class_name = "Unknown"

            # Determine bounding box color
            color = (0, 255, 0) if class_name == "HandsOnWheel" else (255, 0, 0)

            # Draw bounding box and label
            label = f"{class_name} {conf:.2f}"
            cv2.rectangle(processed_image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(processed_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    else:
        print("⚠ No detections found for this image.")

# Display the image in VS Code using OpenCV
cv2.imshow("YOLOv7 Detection", processed_image)
cv2.waitKey(0)            # Wait for key press
cv2.destroyAllWindows()   # Close image window
print("✅ Inference completed!")
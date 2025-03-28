import torch
import cv2
import numpy as np

# ================================
# CONFIGURATIONS
# ================================
weights_path = r"C:\Users\Farah\Downloads\best (1).pt"  # Your trained YOLOv7 weights
yolov7_repo_path = r"D:\GRAD_PROJECT\Driver-Monitoring-System\Activity Detection\object_detection_HOW\how_yolov7\yolov7"

CONFIDENCE_THRESHOLD = 0.5  # Minimum confidence for detection (50%)
IOU_THRESHOLD = 0.4  # Non-Maximum Suppression (NMS) IoU threshold
INPUT_SIZE = (640, 640)  # Resize frames before inference
FONT_SCALE = 0.8  # Font size for text
TEXT_THICKNESS = 2  # Thickness of text
BOX_THICKNESS = 2  # Thickness of bounding boxes
TEXT_COLOR = (255, 255, 255)  # White text
BOX_COLOR = (0, 255, 0)  # Green bounding box
BACKGROUND_COLOR = (0, 0, 0)  # Black background for text

# ================================
# LOAD YOLOv7 MODEL
# ================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model correctly
model = torch.hub.load(yolov7_repo_path, 'custom', weights_path, source='local')
model.to(device)
model.eval()

# ================================
# REAL-TIME WEBCAM INFERENCE
# ================================
cap = cv2.VideoCapture(0)  # Open webcam (0 = default camera)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    orig_height, orig_width, _ = frame.shape  # Store original frame size

    # Resize frame to 640x640 for YOLOv7
    frame_resized = cv2.resize(frame, INPUT_SIZE)
    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

    # Run inference
    results = model(frame_rgb)
    df = results.pandas().xyxy[0]  # Convert detections to pandas DataFrame

    # Apply NMS (keep only the most confident boxes)
    df = df[df['confidence'] >= CONFIDENCE_THRESHOLD]  # Confidence threshold
    df = df.sort_values(by='confidence', ascending=False)  # Sort by confidence

    # Scale bounding boxes back to original frame size
    scale_x = orig_width / INPUT_SIZE[0]
    scale_y = orig_height / INPUT_SIZE[1]

    # Draw results on frame
    for _, row in df.iterrows():
        x1, y1, x2, y2, confidence, cls = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax']), row['confidence'], int(row['class'])
        
        # Scale coordinates back to original size
        x1, x2 = int(x1 * scale_x), int(x2 * scale_x)
        y1, y2 = int(y1 * scale_y), int(y2 * scale_y)

        label = "Happy" if cls == 0 else "Sad"
        confidence_percentage = confidence * 100  # Convert to percentage

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), BOX_COLOR, BOX_THICKNESS)

        # Draw text box at bottom center
        text1 = f"Emotion: {label}"
        text2 = f"Confidence: {confidence_percentage:.2f}%"

        font = cv2.FONT_HERSHEY_SIMPLEX

        # Calculate text size
        text1_size = cv2.getTextSize(text1, font, FONT_SCALE, TEXT_THICKNESS)[0]
        text2_size = cv2.getTextSize(text2, font, FONT_SCALE, TEXT_THICKNESS)[0]
        text_width = max(text1_size[0], text2_size[0])
        text_height = text1_size[1] + text2_size[1] + 10

        # Define bottom middle position
        x_center = orig_width // 2 - text_width // 2
        y_bottom = orig_height - 10  # Keep some margin from the bottom

        # Draw black rectangle as background
        cv2.rectangle(frame, (x_center - 10, y_bottom - text_height - 10), 
                      (x_center + text_width + 10, y_bottom + 10), 
                      BACKGROUND_COLOR, -1)

        # Put text on frame
        cv2.putText(frame, text1, (x_center, y_bottom - text2_size[1] - 5), font, FONT_SCALE, TEXT_COLOR, TEXT_THICKNESS)
        cv2.putText(frame, text2, (x_center, y_bottom), font, FONT_SCALE, TEXT_COLOR, TEXT_THICKNESS)

    # Display the frame
    cv2.imshow("YOLOv7 Emotion Detection - Webcam", frame)

    # Press 'X' to exit
    key = cv2.waitKey(1) & 0xFF
    if key == ord('x') or key == ord('X'):
        break

# Release the camera and close windows
cap.release()
cv2.destroyAllWindows()
print("Real-time emotion detection stopped.")

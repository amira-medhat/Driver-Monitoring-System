
#clone to repo
# 1- %cd /content/drive/MyDrive
# 2- !git clone https://github.com/IDEA-Research/GroundingDINO.git

# direct the path and install dependancies 
# 1- cd /content/drive/MyDrive/GroundingDINO
# 2- !pip install -r requirements.txt
# 3- !pip install groundingdino transformers
# 4- !pip install -e .

# create weights folder , direct in it and install the weights 
# 1- !mkdir -p weights && cd weights
# 2- cd /content/drive/MyDrive/GroundingDINO/weights
# 3- !wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth -O /content/drive/MyDrive/GroundingDINO/weights/groundingdino_swint_ogc.pth



# cd .. 3shan arga3 khatwa l wara mab2ash f folder el weights 
# then run this code and configure the pathsss

import os
from groundingdino.util.inference import load_model, load_image, predict
import cv2
import shutil

# Load Grounding DINO model
model = load_model(
    "/content/drive/MyDrive/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
    "/content/drive/MyDrive/GroundingDINO/weights/groundingdino_swint_ogc.pth"
)

# Define input folder containing images
INPUT_FOLDER = "/content/drive/MyDrive/salma_hands_on"

# Define output folders in Google Drive
OUTPUT_FOLDER = "/content/drive/MyDrive/output_salma_on"
OUTPUT_IMAGES = f"{OUTPUT_FOLDER}/images"
OUTPUT_LABELS = f"{OUTPUT_FOLDER}/labels"

# Ensure output folders exist
os.makedirs(OUTPUT_IMAGES, exist_ok=True)
os.makedirs(OUTPUT_LABELS, exist_ok=True)

# Define text prompt and detection thresholds
TEXT_PROMPT = "steering wheel"
BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.25
CLASS_LABEL = 1  # Class for detected objects (modify as needed)

# Process all images in the folder
image_files = [f for f in os.listdir(INPUT_FOLDER) if f.endswith(('.jpg', '.png', '.jpeg'))]

for image_file in image_files:
    image_path = os.path.join(INPUT_FOLDER, image_file)

    # Load image
    image_source, image = load_image(image_path)

    # Run inference
    boxes, logits, phrases = predict(
        model=model,
        image=image,
        caption=TEXT_PROMPT,
        box_threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD
    )

    # Save the original image (without bounding boxes)
    save_image_path = os.path.join(OUTPUT_IMAGES, image_file)
    shutil.copy(image_path, save_image_path)  # Copy original image to output folder

    # Save bounding box coordinates in a label file
    label_filename = os.path.splitext(image_file)[0] + ".txt"
    label_path = os.path.join(OUTPUT_LABELS, label_filename)

    # Save bounding box coordinates in the label file
    with open(label_path, "w") as label_file:
        if len(boxes) > 0:
            for box in boxes:
                x1, y1, x2, y2 = map(float, box.cpu().numpy())  # Convert tensor to float
                if x1 == x2 or y1 == y2:
                    print(f"⚠ Invalid box detected: {box}, skipping!")
                    continue  # Skip incorrect boxes
                label_file.write(f"{CLASS_LABEL} {x1} {y1} {x2} {y2}\n")
        else:
            print(f"⚠ No detection found for {image_file}, label file will be empty.")
            label_file.write("")  # Leave empty if no detection

    print(f"✅ Processed: {image_file}")

print("✅ All images processed and saved!")

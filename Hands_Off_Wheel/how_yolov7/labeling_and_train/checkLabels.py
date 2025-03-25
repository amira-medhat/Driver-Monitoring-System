import os
import cv2

# Set the paths
IMAGE_FOLDER = r"C:\Users\Farah\Downloads\sara_on\output_sara_on\images"  # Change this to your image folder
LABEL_FOLDER = r"C:\Users\Farah\Downloads\sara_on\output_sara_on\labels"  # Change this to your label folder

# Get all image files sorted in order
image_files = sorted([f for f in os.listdir(IMAGE_FOLDER) if f.endswith(('.jpg', '.png', '.jpeg'))])

# Loop through each image in order
for image_file in image_files:
    image_path = os.path.join(IMAGE_FOLDER, image_file)
    label_path = os.path.join(LABEL_FOLDER, os.path.splitext(image_file)[0] + ".txt")  # Match label file

    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Could not open {image_file}")
        continue

    # Get image dimensions
    height, width, _ = image.shape

    # Check if label file exists
    if not os.path.exists(label_path):
        print(f"⚠ No label found for {image_file}, skipping.")
        continue

    # Read label file and draw bounding boxes
    with open(label_path, "r") as file:
        for line in file:
            data = line.strip().split()
            if len(data) == 5:
                class_label, x_center, y_center, bbox_width, bbox_height = map(float, data)  # Read YOLO format

                # Convert normalized YOLO coordinates to absolute pixel values
                x1 = int((x_center - bbox_width / 2) * width)
                y1 = int((y_center - bbox_height / 2) * height)
                x2 = int((x_center + bbox_width / 2) * width)
                y2 = int((y_center + bbox_height / 2) * height)

                # Draw bounding box
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box
                cv2.putText(image, f"Class {class_label}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the image name at the top
    window_name = f"Image: {image_file}"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)  # Allows resizing
    cv2.imshow(window_name, image)

    # Wait for key press and close window
    key = cv2.waitKey(0)  # Press any key to continue to next image
    if key == ord('q'):  # Press 'q' to quit early
        break

    cv2.destroyAllWindows()  # Close window before moving to the next image

print("✅ All images processed.")
cv2.destroyAllWindows()

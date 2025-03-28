import os

def delete_empty_labels_and_images(image_path, label_path):
    # List all files in the label directory
    for label_file in os.listdir(label_path):
        label_file_path = os.path.join(label_path, label_file)

        # Check if it's a .txt file and is empty
        if label_file.endswith(".txt") and os.path.getsize(label_file_path) == 0:
            # Corresponding image file (same name, different extension)
            image_file = os.path.splitext(label_file)[0]  # Remove .txt extension

            # Check common image extensions
            for ext in [".jpg", ".png", ".jpeg", ".bmp", ".tiff"]:
                image_file_path = os.path.join(image_path, image_file + ext)
                
                # Delete image file if it exists
                if os.path.exists(image_file_path):
                    os.remove(image_file_path)
                    print(f"Deleted image: {image_file_path}")

            # Delete the empty label file
            os.remove(label_file_path)
            print(f"Deleted empty label file: {label_file_path}")

# Example usage
image_folder = r"D:\GRAD_PROJECT\es_datasets_types\us_pt2_out\traim\images"
label_folder = r"D:\GRAD_PROJECT\es_datasets_types\us_pt2_out\train\labels"

delete_empty_labels_and_images(image_folder, label_folder)

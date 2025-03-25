import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import os
import numpy as np
import random

if __name__=="__main__":
    # Define Augmentation Pipeline (Ensuring Fixed Size: 640x480)
    augmentations = [
        A.HorizontalFlip(p=1),
        A.Affine(
            scale=(0.9, 1.1),
            translate_percent=(0.1, 0.2),
            shear=(-10, 10),
            rotate=(-30, 30),
            p=1,
        ),
        A.Rotate(limit=45, p=0.5),  # Extra Rotation
        A.GridDistortion(num_steps=5, distort_limit=0.3, p=1),
        A.FancyPCA(alpha=0.1, p=0.2),  # Color-based Wide-Angle Distortion
        A.GaussianBlur(p=0.2),
        A.ElasticTransform(alpha=1, sigma=50, p=1),
        A.MotionBlur(blur_limit=5, p=1),
        A.ISONoise(color_shift=(0.01, 0.1), intensity=(0.1, 0.5), p=0.2),
        A.RandomRain(brightness_coefficient=0.9, drop_width=1, blur_value=5, p=0.3),
        A.RandomBrightnessContrast(p=1),
        A.CLAHE(p=0.2),
        A.RandomShadow(p=0.3, shadow_dimension=5),
        A.RGBShift(p=0.2),
    ]

    # Directories
    input_root = r"D:\GP_datasets\without_aug\state-farm-distracted-driver-detection\imgs\train"
    output_root = r"D:\GP_datasets\with_aug\25_2_2025"

    # Ensure the output directory exists
    os.makedirs(output_root, exist_ok=True)

    # Process all subdirectories except 'c0'
    for subdir in os.listdir(input_root):
        
        if subdir in ("c0", "c1", "c2", "c3"):
            continue
        
        input_dir = os.path.join(input_root, subdir)
        output_dir = os.path.join(output_root, subdir)
        print(input_dir)

        # Ensure subdirectory in output exists
        os.makedirs(output_dir, exist_ok=True)

        # Apply augmentation and save images
        for img_name in os.listdir(input_dir):
            img_path = os.path.join(input_dir, img_name)
            image = cv2.imread(img_path)
            if image is None:
                continue  # Skip if the image fails to load

            # Ensure input image size is 640x480 before applying transformations
            image = cv2.resize(image, (640, 480))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB

            # Randomly select 4 augmentations for each image
            selected_transforms = random.sample(augmentations, 2)

            for i, transform in enumerate(selected_transforms):
                augmented = transform(image=image)["image"]
                output_path = os.path.join(output_dir, f"aug_{i}_{img_name}")

                cv2.imwrite(output_path, cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR))

    print("Augmentation completed! Check your output directory.")

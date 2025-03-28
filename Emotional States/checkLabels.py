import os
import subprocess

# Set the paths (update these as needed)
IMAGE_FOLDER_PATH = r"D:\GRAD_PROJECT\es_datasets_types\es_obj\valid\images"
LABEL_FOLDER_PATH = r"D:\GRAD_PROJECT\es_datasets_types\es_obj\valid\labels"

# List of image file names you provided
image_files = [
    "image_540-6-_jpg.rf.7330fc42045b7f06c4c04b516e8ead67.jpg",
    "omar_0088_jpg.rf.570f710787f273633dd5bd0d7782f43a.jpg",
    "farah_0011_jpg.rf.7a390fa456c697cecca89feeae0daac7.jpg",
    "farah_0029_jpg.rf.e351b0b2bce89650a9954921fad9931b.jpg",
    "farah_0036_jpg.rf.f10817170a270027fe497ad4f9a05c9a.jpg",
    "farah_0053_jpg.rf.3848aa537db0afb11925541426d68e86.jpg",
    "farah_0057_jpg.rf.8a546877871109413000deb8aa1fa4d2.jpg",
    "farah_0060_jpg.rf.8f7c69be1afa9050b5d8820d83171143.jpg",
    "farah_0060_jpg.rf.f0577d9d78791f9315bcb97aa3b59bc8.jpg",
    "farah_0073_jpg.rf.1cafeb0d5e86615ac53eefeb6db21ddd.jpg",
    "farah_0075_jpg.rf.2f0541c28e1a23d751a9c25735534328.jpg",
    "farah_0075_jpg.rf.d33d3f2d141efc9c3ec42d8459c46081.jpg",
    "farah_0076_jpg.rf.1612bee68ec87de509c3047620779b8b.jpg",
    "farah_0076_jpg.rf.ef2974c7896b8328f95a08f9ab4a377a.jpg",
    "farah_0081_jpg.rf.bef910c97758b7aa4c85ec5db92483d5.jpg",
]

for image_file in image_files:
    image_path = os.path.join(IMAGE_FOLDER_PATH, image_file)
    label_file = os.path.splitext(image_file)[0] + ".txt"
    label_path = os.path.join(LABEL_FOLDER_PATH, label_file)

    # Check if the image exists
    if not os.path.exists(image_path):
        print(f"⚠️ Image not found: {image_file} in {IMAGE_FOLDER_PATH}. Please update the path.")
        continue

    # Check if the corresponding label file exists
    if not os.path.exists(label_path):
        print(f"⚠️ Label file not found for {image_file} ({label_file}). Skipping deletion.")
        continue

    # Open the label file in Notepad (Windows)
    print(f"📂 Opening label file: {label_file}")
    subprocess.run(["notepad.exe", label_path])

    # After closing the file, delete both image and label
    try:
        os.remove(image_path)
        print(f"🗑️ Deleted image: {image_file}")
    except Exception as e:
        print(f"❌ Could not delete image {image_file}: {e}")

    try:
        os.remove(label_path)
        print(f"🗑️ Deleted label: {label_file}")
    except Exception as e:
        print(f"❌ Could not delete label {label_file}: {e}")

print("✅ All label files processed and deleted.")

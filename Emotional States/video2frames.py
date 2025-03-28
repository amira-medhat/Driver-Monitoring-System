import cv2
import os

def extract_frames(video_path, output_folder, frame_rate=1):
    """
    Extract frames from a video and save them as images.

    :param video_path: Path to the input video file.
    :param output_folder: Path to the folder where frames will be saved.
    :param frame_rate: Number of frames to extract per second (default is 1).
    """
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Get video frame rate
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_interval = max(1, fps // frame_rate)  # Extract frames at specified rate

    frame_count = 0
    saved_frame_count = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break  # End of video

        if frame_count % frame_interval == 0:
            frame_filename = os.path.join(output_folder, f"omar2_{saved_frame_count:04d}.jpg")
            cv2.imwrite(frame_filename, frame)
            saved_frame_count += 1

        frame_count += 1

    cap.release()
    print(f"Extracted {saved_frame_count} frames and saved them in '{output_folder}'.")

# Example usage
video_path = r"C:\Users\Farah\Downloads\emotional_pics\new_videos\omar.mp4" # Replace with your video file path
output_folder = r"C:\Users\Farah\Downloads\emotional_pics\output" # Replace with your desired output folder
extract_frames(video_path, output_folder, frame_rate=10)  # Change frame_rate to control how many frames per secon
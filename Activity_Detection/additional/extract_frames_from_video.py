import cv2
import os
import argparse


def extract_frames(video_path, output_folder, width=None, height=None):
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Get video frame rate
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = 0
    extracted_frames = 0
    frame_interval = fps // 10  # Extract 10 frames per second

    while True:
        success, frame = cap.read()
        if not success:
            break  # Exit if video ends

        # Resize the frame if dimensions are provided
        if width and height:
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

        # Save frame every 1/10th of a second
        if frame_count % frame_interval == 0:
            frame_filename = os.path.join(
                output_folder, f"frame_1{extracted_frames:04d}.jpg"
            )
            cv2.imwrite(frame_filename, frame)
            print(f"Saved: {frame_filename}")
            extracted_frames += 1

        frame_count += 1

    cap.release()
    print("Frame extraction complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract frames from a video at 10 frames per second."
    )
    parser.add_argument("video_path", type=str, help="Path to the input video file")
    parser.add_argument(
        "output_folder", type=str, help="Folder to save extracted frames"
    )
    parser.add_argument(
        "--width", type=int, help="Width of the output frames", default=None
    )
    parser.add_argument(
        "--height", type=int, help="Height of the output frames", default=None
    )
    args = parser.parse_args()

    extract_frames(args.video_path, args.output_folder, args.width, args.height)

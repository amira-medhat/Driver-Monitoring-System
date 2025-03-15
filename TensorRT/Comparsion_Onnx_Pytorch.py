import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
import numpy as np
import cv2
import onnxruntime as ort
from PIL import Image
import time
import os
import glob

# Paths
pytorch_model_path = '/home/farouk/Deployment/ActivityDetection/weights/fine_tuned_mobilenetv3_with_aug.pth'
onnx_model_path = '/home/farouk/Deployment/ActivityDetection/onnx/ActivityDetection.onnx'

class_labels = {
    0: "Safe driving",
    1: "Texting(right hand)",
    2: "Talking on the phone (right hand)",
    3: "Texting (left hand)",
    4: "Talking on the phone (left hand)",
    5: "Operating the radio",
    6: "Drinking",
    7: "Reaching behind",
    8: "Hair and makeup",
    9: "Talking to passenger(s)",
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# Load PyTorch Model
pytorch_model = models.mobilenet_v3_large(weights=None)
pytorch_model.classifier[-1] = nn.Linear(pytorch_model.classifier[-1].in_features, 10)
pytorch_model.load_state_dict(torch.load(pytorch_model_path, map_location=device))
pytorch_model.to(device).eval()

# Load ONNX Model
onnx_session = ort.InferenceSession(onnx_model_path, providers=["CUDAExecutionProvider" if torch.cuda.is_available() else "CPUExecutionProvider"])

# Predict using PyTorch
def predict_pytorch(img):
    img_tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = pytorch_model(img_tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    return probs

# Predict using ONNX
def predict_onnx(img):
    img_np = transform(img).unsqueeze(0).numpy()
    inputs = {onnx_session.get_inputs()[0].name: img_np}
    
    start_time = time.time()
    outputs = onnx_session.run(None, inputs)
    inference_time = time.time() - start_time

    probs = np.exp(outputs[0]) / np.sum(np.exp(outputs[0]), axis=1)
    return probs[0], inference_time

# Process Full Video
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    total_pytorch_time, total_onnx_time, total_mse = 0, 0, 0

    # Check if video opened successfully
    if not cap.isOpened():
        print(f"Error: Could not open video at {video_path}")
        return

    print(f"Processing video: {video_path}")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        
        # PyTorch inference
        start_time = time.time()
        pytorch_probs = predict_pytorch(pil_img)
        pytorch_time = time.time() - start_time
        
        # ONNX inference
        onnx_probs, onnx_time = predict_onnx(pil_img)
        
        mse = np.mean((pytorch_probs - onnx_probs)**2)
        total_mse += mse
        
        print(f"Frame {frame_count}: MSE={mse:.6f}, PyTorch Time={pytorch_time:.4f}s, ONNX Time={onnx_time:.4f}s")
        
        total_pytorch_time += pytorch_time
        total_onnx_time += onnx_time
        frame_count += 1
    
    cap.release()

    avg_mse = total_mse / frame_count if frame_count > 0 else 0
    avg_pytorch_time = total_pytorch_time / frame_count if frame_count > 0 else 0
    avg_onnx_time = total_onnx_time / frame_count if frame_count > 0 else 0
    
    print(f"\nProcessed {frame_count} frames.")
    print(f"Average PyTorch Inference Time: {avg_pytorch_time:.4f} sec/frame")
    print(f"Average ONNX Inference Time: {avg_onnx_time:.4f} sec/frame")
    print(f"Average Mean Squared Error: {avg_mse:.15f}")
    print(f"Speed improvement: {avg_pytorch_time/avg_onnx_time:.2f}x")

    print("Inference completed successfully.")

def list_videos_in_directory(directory, extensions=None):
    """List all video files in a directory with specified extensions."""
    if extensions is None:
        extensions = ['.mp4', '.avi', '.mov']
    
    video_files = []
    for ext in extensions:
        if not ext.startswith('.'):
            ext = '.' + ext
        pattern = os.path.join(directory, f"*{ext}")
        video_files.extend(glob.glob(pattern))
    
    return video_files

def interactive_video_selection():
    print("\n===== Driver Activity Detection - PyTorch vs ONNX Performance Comparison =====")
    
    while True:
        print("\nOptions:")
        print("1. Process a single video file")
        print("2. Process all videos in a directory")
        print("3. Exit")
        
        choice = input("\nEnter your choice (1-3): ").strip()
        
        if choice == '1':
            video_path = input("Enter the full path to the video file: ").strip()
            if os.path.isfile(video_path):
                process_video(video_path)
            else:
                print(f"Error: The file '{video_path}' does not exist or is not accessible.")
                
        elif choice == '2':
            dir_path = input("Enter the directory path containing videos: ").strip()
            if not os.path.isdir(dir_path):
                print(f"Error: The directory '{dir_path}' does not exist or is not accessible.")
                continue
                
            extensions_input = input("Enter video extensions to process (comma-separated, e.g., mp4,avi,mov): ").strip()
            if not extensions_input:
                extensions = ['.mp4', '.avi', '.mov']  # Default extensions
            else:
                extensions = ['.' + ext.strip('.') for ext in extensions_input.split(',')]
                
            video_files = list_videos_in_directory(dir_path, extensions)
                
            if not video_files:
                print(f"No video files with extension(s) {extensions} found in '{dir_path}'")
            else:
                print(f"Found {len(video_files)} video file(s) to process")
                for i, video_file in enumerate(video_files, 1):
                    print(f"\nProcessing video {i}/{len(video_files)}: {os.path.basename(video_file)}")
                    process_video(video_file)
                    
        elif choice == '3':
            print("Exiting program.")
            break
            
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")

# Main program entry point
if __name__ == "__main__":
    try:
        interactive_video_selection()
    except KeyboardInterrupt:
        print("\nProgram interrupted by user.")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("Program terminated.")
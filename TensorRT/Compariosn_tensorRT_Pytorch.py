import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
import numpy as np
import cv2
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from PIL import Image
import time
import os
import glob

# Paths
pytorch_model_path = '/home/farouk/Deployment/ActivityDetection/weights/fine_tuned_mobilenetv3_with_aug.pth'
trt_engine_path = '/home/farouk/Deployment/ActivityDetection/engine/ActivityDetection321.engine'

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

# PyTorch Setup
pytorch_model = models.mobilenet_v3_large(weights=None)
pytorch_model.classifier[-1] = nn.Linear(pytorch_model.classifier[-1].in_features, 10)
pytorch_model.load_state_dict(torch.load(pytorch_model_path, map_location=device))
pytorch_model.to(device).eval()

def predict_pytorch(img):
    img_tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = pytorch_model(img_tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    return probs

# TensorRT Setup
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def load_trt_engine(engine_path):
    with open(engine_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

def allocate_buffers(engine):
    inputs, outputs, bindings = [], [], []
    stream = cuda.Stream()
    context = engine.create_execution_context()

    for binding in engine:
        binding_shape = engine.get_tensor_shape(binding)
        dtype = trt.nptype(engine.get_tensor_dtype(binding))

        if -1 in binding_shape:
            binding_shape = tuple([1 if s == -1 else s for s in binding_shape])

        size = trt.volume(binding_shape)
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        bindings.append(int(device_mem))

        if engine.get_tensor_mode(binding) == trt.TensorIOMode.INPUT:
            inputs.append((host_mem, device_mem))
        else:
            outputs.append((host_mem, device_mem))

    return context, inputs, outputs, bindings, stream

engine = load_trt_engine(trt_engine_path)
context, inputs, outputs, bindings, stream = allocate_buffers(engine)

def predict_trt(img):
    img_np = transform(img).unsqueeze(0).numpy()
    np.copyto(inputs[0][0], img_np.ravel())
    cuda.memcpy_htod_async(inputs[0][1], inputs[0][0], stream)
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    cuda.memcpy_dtoh_async(outputs[0][0], outputs[0][1], stream)
    stream.synchronize()
    logits = outputs[0][0]
    probs = softmax(logits)
    return probs

# Process Full Video
def process_video(video_path):
    global context, inputs, outputs, bindings, stream, engine  # Ensure global access

    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    total_pytorch_time, total_trt_time, total_mse = 0, 0, 0

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
        
        # TensorRT inference
        start_time = time.time()
        trt_probs = predict_trt(pil_img)
        trt_time = time.time() - start_time
        
        mse = np.mean((pytorch_probs - trt_probs)**2)
        total_mse += mse
        
        print(f"Frame {frame_count}: MSE={mse:.6f}, PyTorch Time={pytorch_time:.4f}s, TensorRT Time={trt_time:.4f}s")
        
        total_pytorch_time += pytorch_time
        total_trt_time += trt_time
        frame_count += 1
    
    cap.release()

    avg_mse = total_mse / frame_count if frame_count > 0 else 0
    avg_pytorch_time = total_pytorch_time / frame_count if frame_count > 0 else 0
    avg_trt_time = total_trt_time / frame_count if frame_count > 0 else 0
    
    print(f"\nProcessed {frame_count} frames.")
    print(f"Average PyTorch Inference Time: {avg_pytorch_time:.4f} sec/frame")
    print(f"Average TensorRT Inference Time: {avg_trt_time:.4f} sec/frame")
    print(f"Average Mean Squared Error: {avg_mse:.6f}")
    print(f"Speed improvement: {avg_pytorch_time/avg_trt_time:.2f}x")

    # Safe cleanup of TensorRT resources
    cleanup_resources()

    print("CUDA resources successfully cleaned up.")

def cleanup_resources():
    if 'context' in globals():
        del globals()['context']
    if 'inputs' in globals():
        del globals()['inputs']
    if 'outputs' in globals():
        del globals()['outputs']
    if 'bindings' in globals():
        del globals()['bindings']
    if 'stream' in globals():
        del globals()['stream']
    if 'engine' in globals():
        del globals()['engine']

def interactive_video_selection():
    print("\n===== Driver Activity Detection - PyTorch vs TensorRT Performance Comparison =====")
    
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
                
            extensions = input("Enter video extensions to process (comma-separated, e.g., mp4,avi,mov): ").strip()
            if not extensions:
                extensions = "mp4,avi,mov"  # Default extensions
                
            video_files = []
            for ext in extensions.split(','):
                ext = ext.strip()
                if not ext.startswith('.'):
                    ext = '.' + ext
                pattern = os.path.join(dir_path, f"*{ext}")
                video_files.extend(glob.glob(pattern))
                
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
    finally:
        # Make sure resources are cleaned up even if an error occurs
        cleanup_resources()
        print("Program terminated.")
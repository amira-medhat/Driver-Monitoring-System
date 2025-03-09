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
from PyQt5.QtWidgets import QApplication, QFileDialog
import sys

# Paths
pytorch_model_path = '/home/farouk/Public/Workspace/fine_tuned_mobilenetv3_with_aug.pth'
trt_engine_path = '/home/farouk/Public/Workspace/model.engine'

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

    # ✅ Safe cleanup of TensorRT resources (now declared as global)
    if 'context' in globals():
        del context
    if 'inputs' in globals():
        del inputs
    if 'outputs' in globals():
        del outputs
    if 'bindings' in globals():
        del bindings
    if 'stream' in globals():
        del stream
    if 'engine' in globals():
        del engine

    print("CUDA resources successfully cleaned up.")


# Test on a selected video
if __name__ == "__main__":
    app = QApplication(sys.argv)
    video_path, _ = QFileDialog.getOpenFileName(None, "Select Video File", "", "Video Files (*.mp4 *.avi *.mov)")
    if video_path:
        process_video(video_path)
    else:
        print("No video selected.")
import torch
import torch.onnx
from ultralytics import YOLO  # Import YOLO directly

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the model in EVALUATION mode
    model_path = r"D:\grad project\driver_fatigue\trained_weights\best_ours2.pt"
    model = YOLO(model_path)  # Load YOLO model

    model.to(device)
    model.eval()  # 🔹 Ensure the model is in inference mode

    # Define input size (YOLOv5 expects 640x640 images)
    dummy_input = torch.randn(1, 3, 640, 640, device=device)

    onnx_file_path = "yolov5_custom.onnx"
    torch.onnx.export(
        model.model,                # The trained YOLOv5 model
        dummy_input,                # Dummy input for tracing
        onnx_file_path,             # ONNX file path
        export_params=True,         # Store trained weights in ONNX file
        opset_version=12,           # ONNX opset version
        do_constant_folding=True,   # Optimize model
        input_names=['images'],     # Name of input layer
        output_names=['output'],    # Name of output layer
        dynamic_axes={              # Allow variable batch size
            'images': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )

    print(f"Custom YOLOv5 model has been converted to ONNX and saved at {onnx_file_path}")

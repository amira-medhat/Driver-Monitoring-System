import torch.onnx
import torch
import torchvision.models as models
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn


if __name__=="__main__":
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    mobilenet_v3_large = models.mobilenet_v3_large(weights=None)
    

    mobilenet_v3_large.classifier[3] = nn.Linear(in_features=1280, out_features=10)
    
    checkpoint_path = r"C:\Users\Amira\Driver-Monitoring-System\activity detection aya's edition\fine_tuned_mobilenetv3_1.pth"
    mobilenet_v3_large.load_state_dict(torch.load(checkpoint_path, weights_only=True))
    
    model = mobilenet_v3_large
    
    model = model.to(device)
    
    model.eval()
    
    # Assuming the model expects 3 channels (RGB) and 224x224 resolution:
    dummy_input = torch.randn(1, 3, 224, 224, device=device)
    
    onnx_file_path = "mobilenet_v3_large.onnx"
    torch.onnx.export(
        model,                      # the model being exported
        dummy_input,                # dummy input for tracing
        onnx_file_path,             # where to save the ONNX file
        export_params=True,         # store the trained parameter weights inside the model file
        # opset_version=12,           # ONNX version to export to (adjust as needed)
        # do_constant_folding=True,   # execute constant folding for optimization
        # input_names=['input'],      # model's input name
        # output_names=['output'],    # model's output name
        # dynamic_axes={              # variable length axes (useful for batch size)
        #     'input': {0: 'batch_size'},
        #     'output': {0: 'batch_size'}
        # }
    )
    print(f"Model has been converted to ONNX and saved at {onnx_file_path}")

    
    
    
import os
import time
import numpy as np
import tensorrt as trt
import torch

# TensorRT Logger
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

# Constants
EXPLICIT_BATCH = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class HostDeviceMem:
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return f"Host:\n{self.host}\nDevice:\n{self.device}"


def allocate_buffers(engine, batch_size=1):
    inputs, outputs, bindings = [], [], []
    for binding in engine:
        shape = engine.get_tensor_shape(binding)
        if shape[0] == -1:
            shape = (batch_size,) + shape[1:]
        size = trt.volume(shape)
        dtype = trt.nptype(engine.get_tensor_dtype(binding))

        # Use PyTorch tensors for memory allocation
        host_mem = torch.empty(size, dtype=torch.float32, device="cpu")  # Host memory
        device_mem = host_mem.to(DEVICE, non_blocking=True)  # Device memory (GPU)
        
        bindings.append(device_mem.data_ptr())  # Get pointer for TensorRT
        if engine.get_tensor_mode(binding) == trt.TensorIOMode.INPUT:
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
    
    return inputs, outputs, bindings


def build_engine(onnx_file_path, engine_file_path, fp16=False, int8=False):
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    EXPLICIT_BATCH = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(EXPLICIT_BATCH)
    config = builder.create_builder_config()

    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)

    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    if int8:
        config.set_flag(trt.BuilderFlag.INT8)
        # Additional calibration steps would be required for INT8

    parser = trt.OnnxParser(network, TRT_LOGGER)

    if not os.path.exists(onnx_file_path):
        raise FileNotFoundError(f"ONNX file '{onnx_file_path}' not found.")

    with open(onnx_file_path, 'rb') as model:
        if not parser.parse(model.read()):
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None

    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        raise RuntimeError("Engine build failed.")

    engine = trt.Runtime(TRT_LOGGER).deserialize_cuda_engine(serialized_engine)

    if engine:
        with open(engine_file_path, "wb") as f:
            f.write(serialized_engine)

    return engine


# TensorRT Engine Build and Inference
onnx_model_path = '/home/farouk/Public/Workspace/ActivityDetection.onnx'
trt_engine_path = '/home/farouk/Public/Workspace/ActivityDetection32.engine'
fp16_mode = False
int8_mode = False

engine = build_engine(onnx_model_path, trt_engine_path, fp16_mode, int8_mode)

# Allocate buffers
inputs, outputs, bindings = allocate_buffers(engine)

# Create execution context
context = engine.create_execution_context()

# Prepare input data (convert NumPy to PyTorch tensor)
input_data = torch.rand((1, 3, 64, 64), dtype=torch.float32).to(DEVICE, non_blocking=True)
inputs[0].device.copy_(input_data)  # Copy input to GPU

# Run inference
context.execute_v2(bindings)

# Get output
output = outputs[0].device.cpu().numpy()  # Copy output back to CPU
print(output)
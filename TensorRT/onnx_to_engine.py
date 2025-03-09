import os
import time
import pycuda.autoinit
import numpy as np
import pycuda.driver as cuda
import tensorrt as trt
import torch

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
EXPLICIT_BATCH = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

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
        host_mem = cuda.pagelocked_empty(shape=size, dtype=dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        bindings.append(int(device_mem))
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


TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

onnx_model_path = '/home/farouk/Public/Workspace/mobilenet_v3_large.onnx'
trt_engine_path = '/home/farouk/Public/Workspace/model.engine'
fp16_mode = True
int8_mode = False

engine = build_engine(onnx_model_path, trt_engine_path, fp16_mode, int8_mode)

# Allocate buffers
inputs, outputs, bindings = allocate_buffers(engine)

# Create execution context
context = engine.create_execution_context()

# Prepare input data
input_data = np.random.rand(1, 3, 64, 64).astype(np.float32)
inputs[0].host = input_data

# Run inference
stream = cuda.Stream()

cuda.memcpy_htod_async(inputs[0].device, inputs[0].host, stream)
context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
cuda.memcpy_dtoh_async(outputs[0].host, outputs[0].device, stream)
stream.synchronize()

# Get output
output = outputs[0].host
print(output)

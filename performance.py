# Measure performance before quantization using OpenVINO's Inference Engine
import openvino.inference_engine as ie
import numpy as np
import time
import os
from openvino.tools import mo
from openvino.runtime import serialize 

onnx_path ='swin_small_dynamic_quantized.onnx' 

model = mo.convert_model(onnx_path)
# serialize model for saving IR
serialize(model, "swin_small_dynamic_quantized.xml")

ie_core = ie.IECore()
net = ie_core.read_network(model='swin_small_dynamic_quantized.xml', weights='swin_small_dynamic_quantized.bin')
exec_net = ie_core.load_network(net, 'CPU')

input_blob = next(iter(net.input_info))
output_blob = next(iter(net.outputs))
input_shape = net.input_info[input_blob].tensor_desc.dims

input_data = np.random.uniform(low=0, high=1, size=input_shape).astype(np.float32)
for i in range(10):
    start_time = time.time()
    output = exec_net.infer(inputs={input_blob: input_data})
    end_time = time.time()
    inference_time = (end_time - start_time) * 1000
    print(f"Inference time: {inference_time:.3f} ms")

# Measure performance after quantization using OpenVINO's Inference Engine
ie_core = ie.IECore()
net = ie_core.read_network(model='swin_small_quantized', weights='swin_small_quantized.pth')
exec_net = ie_core.load_network(net, 'CPU')

input_blob = next(iter(net.input_info))
output_blob = next(iter(net.outputs))
input_shape = net.input_info[input_blob].tensor_desc.dims

input_data = np.random.uniform(low=0, high=1, size=input_shape).astype(np.float32)
for i in range(10):
    start_time = time.time()
    output = exec_net.infer(inputs={input_blob: input_data})
    end_time = time.time()
    inference_time = (end_time - start_time) * 1000
    print(f"Inference time: {inference_time:.3f} ms")

# Measure the model size before and after quantization
before_size = os.path.getsize('swin_small_patch4_window7_224.pth')
after_size = os.path.getsize('swin_small_quantized.pth')
print(f"Model size before quantization: {before_size / 1024 / 1024:.2f} MB")
print(f"Model size after quantization: {after_size / 1024 / 1024:.2f} MB")

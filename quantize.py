# import torch
# # import onnxruntime
# import timm
# # import torchvision

# # Load the Swin Transformer model
# model = timm.create_model('swin_small_patch4_window7_224')
# torch.save(model.state_dict(), 'swin_small.pth')

# # Define the quantization configuration
# quantization_config = torch.quantization.get_default_qconfig('fbgemm')

# # Prepare the model for quantization
# model.qconfig = quantization_config
# torch.quantization.prepare(model, inplace=True)

# # Calibrate the model to determine proper quantization parameters
# # You can use a representative dataset for calibration, but here we'll just use random inputs
# model.eval()
# calibration_data = torch.randn(1, 3, 224, 224)
# model(calibration_data)
# torch.quantization.convert(model, inplace=True)

# # quantized_model = torch.quantization.quantize_dynamic(
# #     model, {torch.nn.Linear}, dtype=torch.qint8
# # )

# # Save the quantized model
# torch.save(model.state_dict(), 'swin_small_quantized.pth')

# # # Save the quantized model
# # torch.jit.save(torch.jit.script(quantized_model), 'swin_small_dynamic_quantized.pt')

# # Export the quantized model to ONNX format
# input_shape = torch.randn(1, 3, 224, 224)
# # dynamic_axes = {'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
# # onnx_filename = 'swin_small_quantized.onnx'
# torch.onnx.export(model, input_shape, 'swin_small_quantized.onnx', opset_version=11)

# # # Create an ONNXRuntime session
# # ort_session = onnxruntime.InferenceSession(onnx_filename)

# # # Create input and output names for ONNXRuntime
# # ort_inputs = {ort_session.get_inputs()[0].name: calibration_data.numpy()}
# # ort_outputs = [output.name for output in ort_session.get_outputs()]

# # # Evaluate the quantized model using ONNXRuntime
# # num_runs = 100
# # ort_times = []
# # for i in range(num_runs):
# #     ort_start_time = torch.cuda.Event(enable_timing=True)
# #     ort_end_time = torch.cuda.Event(enable_timing=True)
# #     ort_start_time.record()
# #     ort_result = ort_session.run(ort_outputs, ort_inputs)
# #     ort_end_time.record()
# #     torch.cuda.synchronize()
# #     ort_time = ort_start_time.elapsed_time(ort_end_time)
# #     ort_times.append(ort_time)

# # # Calculate the average inference time and model size
# # avg_ort_time = sum(ort_times) / num_runs
# # model_size = os.path.getsize('swin_small_quantized.pth')

# # print(f'Average inference time: {avg_ort_time:.2f} ms')
# # print(f'Model size: {model_size / 1024 / 1024:.2f} MB')

import timm
import torch
import torchvision.models as models
from torch.quantization import get_default_qconfig, quantize_jit
from torchvision import transforms

# Load the model
model = timm.create_model('swin_small_patch4_window7_224')
model.eval()

# Create an example input and run inference
x = torch.randn(1, 3, 224, 224)
with torch.no_grad():
    y = model(x)
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor()
])

# create some dummy input data
input_data = [transform(torch.randn(3, 256, 256)) for _ in range(100)]

# define the input calibration data
input_calibration_batches = input_data[:10]

# Quantize the model

def calibrate(model, data_loader):
    model.eval()
    with torch.no_grad():
        for image, target in data_loader:
            model(image)

qconfig = get_default_qconfig('qnnpack')
# quantized_model = quantize_jit(model, qconfig, run_fn_args=[x])
quantized_model = torch.quantization.quantize_jit(model, qconfig, run_fn=calibrate ,run_args=input_calibration_batches, inplace=False)
quantized_model.eval()

# Run inference with the quantized model
with torch.no_grad():
    qy = quantized_model(x)

# Print model size and accuracy
orig_size = sum(p.numel() for p in model.parameters())
quant_size = sum(p.numel() for p in quantized_model.parameters())
print(f'Original model size: {orig_size} parameters')
print(f'Quantized model size: {quant_size} parameters')
print(f'Quantization size reduction: {orig_size - quant_size} parameters')
print(f'Accuracy loss: {((y - qy)**2).mean()}')

import torch
import timm

# Load a pretrained Vision Transformer model from TIMM
model = timm.create_model('vit_base_patch16_224', pretrained=True)

# Define a dummy input for calibration
dummy_input = torch.randn(1, 3, 224, 224)

# Quantize the model using PyTorch's built-in quantization
quantized_model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)

# # Evaluate the original and quantized models on the dummy input
# with torch.no_grad():
#     original_output = model(dummy_input)
#     quantized_output = quantized_model(dummy_input)

# # Compute the accuracy of the original and quantized models
# original_prediction = torch.argmax(original_output, dim=1)
# quantized_prediction = torch.argmax(quantized_output, dim=1)
# accuracy_before = torch.sum(original_prediction == quantized_prediction).item() / original_prediction.size(0)
# accuracy_after = torch.sum(original_prediction == quantized_prediction).item() / quantized_prediction.size(0)

# # Print the accuracy and model size before and after quantization
# print(f"Accuracy before quantization: {accuracy_before:.4f}")
# print(f"Accuracy after quantization: {accuracy_after:.4f}")
# print(f"Model size before quantization: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} MB")
# print(f"Model size after quantization: {sum(p.numel() for p in quantized_model.parameters()) / 1e6:.2f} MB")

# Convert the quantized model to TorchScript
scripted_model = torch.jit.script(quantized_model)
scripted_model.save("quantized_vit_base.pt")

# Evaluate the original and TorchScript models on the dummy input
original_output = model(dummy_input)
scripted_output = scripted_model(dummy_input)

# Compute the accuracy of the original and TorchScript models
original_prediction = torch.argmax(original_output, dim=1)
scripted_prediction = torch.argmax(scripted_output, dim=1)
accuracy_before = torch.sum(original_prediction == scripted_prediction).item() / original_prediction.size(0)
accuracy_after = torch.sum(original_prediction == scripted_prediction).item() / scripted_prediction.size(0)

# Print the accuracy and model size before and after quantization and TorchScript
print(f"Accuracy before quantization: {accuracy_before:.4f}")
print(f"Accuracy after quantization and TorchScript: {accuracy_after:.4f}")
print(f"Model size before quantization: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} MB")
print(f"Model size after quantization: {sum(p.numel() for p in quantized_model.parameters()) / 1e6:.2f} MB")

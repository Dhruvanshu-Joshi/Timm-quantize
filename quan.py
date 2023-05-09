import torch
import timm
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

# Define the test dataset and data loader
test_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
test_dataset = CIFAR10(root='./data', train=False, download=True, transform=test_transform)
testloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Function to evaluate the model
def evaluate_model(model):
    model.eval()
    correct = 0
    total = 0
    device = "cpu"
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

# Function to get the size of the model
def get_model_size(model):
    size = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return size


# Load the model
model = timm.create_model('swin_small_patch4_window7_224_finetuned_cifar10', pretrained=True)

# Define the quantization configuration
quantization_config = torch.quantization.get_default_qconfig('qnnpack')

# Prepare the model for quantization
model.qconfig = quantization_config
model_prepared = torch.quantization.prepare(model, inplace=True)

# Calibrate the model with sample data
calibration_data = torch.randn(1, 3, 224, 224)
model_prepared(calibration_data)

# Convert the model to a dynamically quantized model
quantized_model = torch.quantization.convert(model_prepared)

# Evaluate the original model
original_model_accuracy = evaluate_model(model)

# Measure the size of the original model
original_model_size = get_model_size(model)

print(f'Original model accuracy: {original_model_accuracy}')
print(f'Original model size: {original_model_size / 1e6:.2f} MB')

# Evaluate the quantized model
quantized_model_accuracy = evaluate_model(quantized_model)


# Measure the size of the quantized model
quantized_model_size = get_model_size(quantized_model)

# Print the performance gain and size reduction
print(f'Original model accuracy: {original_model_accuracy}')
print(f'Quantized model accuracy: {quantized_model_accuracy}')
print(f'Performance gain: {(quantized_model_accuracy - original_model_accuracy) * 100:.2f}%')
print(f'Original model size: {original_model_size / 1e6:.2f} MB')
print(f'Quantized model size: {quantized_model_size / 1e6:.2f} MB')
print(f'Size reduction: {(original_model_size - quantized_model_size) / original_model_size * 100:.2f}%')

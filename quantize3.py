import torch
import torchvision
import time
import timm
import progress_bar

# Load the dataset and the pretrained model
transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((224, 224)),
    torchvision.transforms.ToTensor()
])
dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
model = timm.create_model('efficientnet_b0', pretrained=True)

# Set the model to evaluation mode
model.eval()

# Define a dataloader for the validation dataset
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)

# Evaluate the performance of the original model
start_time = time.time()
with torch.no_grad():
    original_total = 0
    original_correct = 0
    for images, labels in dataloader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        # predicted_resized = torch.nn.functional.interpolate(predicted.unsqueeze(1).float(), size=(32, 32), mode='nearest').squeeze(1).long()
        original_total += labels.size(0)
        original_correct += (predicted == labels).sum().item()
        print("labels")
        print(labels)
        print("predicted")
        print(predicted)
original_time = time.time() - start_time
original_accuracy = 100 * original_correct / original_total
print(f"Original accuracy: {original_accuracy:.2f}%")
print(f"Original inference time: {original_time:.2f} seconds")

# Quantize the model and convert it to TorchScript
quantized_model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)
scripted_model = torch.jit.script(quantized_model)

# Evaluate the performance of the quantized model
start_time = time.time()
with torch.no_grad():
    quantized_total = 0
    quantized_correct = 0
    for images, labels in dataloader:
        outputs = scripted_model(images)
        _, predicted = torch.max(outputs, 1)
        quantized_total += labels.size(0)
        quantized_correct += (predicted == labels).sum().item()
quantized_time = time.time() - start_time
quantized_accuracy = 100 * quantized_correct / quantized_total
print(f"Quantized accuracy: {quantized_accuracy:.2f}%")
print(f"Quantized inference time: {quantized_time:.2f} seconds")

# Print the performance gain of quantization
performance_gain = (original_time - quantized_time) / original_time * 100
print(f"Performance gain: {performance_gain:.2f}%")

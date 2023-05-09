import time

import torch
import timm
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# datasets.CIFAR10(root='./data', train=True, download=True)
# val_transforms = transforms.Compose([
#     transforms.Resize(256),
#     transforms.CenterCrop(224),
#     transforms.ToTensor(),
#     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
# ])

# datasets.ImageNet(root='data', split='val', transform=val_transforms, download=True)

# Load the dataset
val_dataset = datasets.ImageNet(
    root='data',
    train=False,
    transform=transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
)

# Create the data loader
val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
)


# Load the model
model = timm.create_model('swin_small_patch4_window7_224', pretrained=True)

# Convert the model to TorchScript
model = torch.jit.script(model)

# Quantize the model
quantized_model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
print("a1")
x=0
# Evaluate the original model
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    start_time = time.time()
    for images, labels in val_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        x=x+1
        print(x)
    end_time = time.time()
    print('Accuracy of the original model:', 100 * correct / total)
    print('Inference time of the original model:', end_time - start_time)

# Evaluate the quantized model
quantized_model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    start_time = time.time()
    for images, labels in val_loader:
        outputs = quantized_model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    end_time = time.time()
    print('Accuracy of the quantized model:', 100 * correct / total)
    print('Inference time of the quantized model:', end_time - start_time)

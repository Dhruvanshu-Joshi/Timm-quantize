from PIL import Image
import torch
import timm
import requests
import torchvision.transforms as transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import pandas as pd
import numpy as np
import os
import json
from torch.quantization.observer import MovingAverageMinMaxObserver
from torch.quantization import QuantType

class CustomQuantizationScheme:
    def __init__(self, num_bits=4, quant_min=-8, quant_max=7):
        super().__init__(QuantType.INT, num_bits, quant_min, quant_max)


def get_size(file_path, unit='bytes'):
    file_size = os.path.getsize(file_path)
    exponents_map = {'bytes': 0, 'kb': 1, 'mb': 2, 'gb': 3}
    if unit not in exponents_map:
        raise ValueError("Must select from \
        ['bytes', 'kb', 'mb', 'gb']")
    else:
        size = file_size / 1024 ** exponents_map[unit]
        return round(size, 3)


model = timm.create_model('swin_small_patch4_window7_224', pretrained=True)
scripted_model = torch.jit.script(model)
scripted_model.save("pt_swin_small.pt")
model.eval()

transform = transforms.Compose([
    transforms.Resize(256, interpolation=3),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
])

img = Image.open(requests.get("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcST-vctfLkYjRIB6kR_wrVbFQtkeDSI_rHdig&usqp=CAU", stream=True).raw)
img = transform(img)[None,]
out = model(img)
clsidx_m = torch.argmax(out)
print(clsidx_m.item())

backend = "x86" # replaced with qnnpack causing much worse inference speed for quantized model on this notebook
model.qconfig = torch.quantization.get_default_qconfig(backend)
torch.backends.quantized.engine = backend

# Quantize the weights
quantized_model = torch.quantization.quantize_dynamic(
    model,
    qconfig_spec={torch.nn.Linear},
    dtype=torch.qint4,
    weight_observer=MovingAverageMinMaxObserver.with_args(
        reduce_range=False,
        custom_qscheme=CustomQuantizationScheme()
    )
)
scripted_quantized_model = torch.jit.script(quantized_model)
scripted_quantized_model.save("pt_swin_small_quantized.pt")

out = scripted_quantized_model(img)
clsidx_q= torch.argmax(out)
print(clsidx_q.item())

with torch.autograd.profiler.profile(use_cuda=False) as prof1:
    out = model(img)
with torch.autograd.profiler.profile(use_cuda=False) as prof3:
    out = scripted_quantized_model(img)

# reading the data from the file
with open("imagenet_classes.txt") as f:
    data = f.read()
      
# reconstructing the data as a dictionary
js = json.loads(data)

print("original model: {:.2f}ms".format(prof1.self_cpu_time_total/1000))
print("original model size: {:.2f}mb".format(get_size("pt_swin_small.pt", 'mb')))
print("original model prediction: {0}".format(js[str(clsidx_m.item())]))
print("scripted & quantized model: {:.2f}ms".format(prof3.self_cpu_time_total/1000))
print("scripted & quantized model size: {:.2f}mb".format(get_size("pt_swin_small_quantized.pt", 'mb')))
print("scripted & quantized model prediction: {0}".format(js[str(clsidx_q.item())]))

df = pd.DataFrame({'Model': ['original model', 'scripted & quantized model']})
df = pd.concat([df, pd.DataFrame([
    ["{:.2f}ms".format(prof1.self_cpu_time_total/1000), "0%", "{:.2f}mb".format(get_size("pt_swin_small.pt", 'mb')), "0%"],
    ["{:.2f}ms".format(prof3.self_cpu_time_total/1000),
     "{:.2f}%".format((prof1.self_cpu_time_total-prof3.self_cpu_time_total)/prof1.self_cpu_time_total*100),
     "{:.2f}mb".format(get_size("pt_swin_small_quantized.pt", 'mb')),
     "{:.2f}%".format((get_size("pt_swin_small.pt", 'mb')-get_size("pt_swin_small_quantized.pt", 'mb'))/(get_size("pt_swin_small.pt", 'mb'))*100)]],
    columns=['Inference Time', 'Time-Reduction', 'Size', 'Size-Reduction'])], axis=1)

print(df)
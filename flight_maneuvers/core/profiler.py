
import torch
import torchvision.models as models
from torch.profiler import profile, record_function, ProfilerActivity

if torch.cuda.is_available():
    torch.set_default_device("cuda")
    print('CUDA AVAILABLE:', True)

model = models.resnet18()
inputs = torch.randn(5, 3, 224, 224)

# use profiler to analyze the execution time:

with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
    with record_function("model_inference"):
        model(inputs)

print(model.device)

print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

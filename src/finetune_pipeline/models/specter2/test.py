import torch
from adapters import AutoAdapterModel

model = AutoAdapterModel.from_pretrained("allenai/specter2_base")
model.load_adapter("allenai/specter2", source="hf", load_as="proximity")
model.load_adapter("allenai/specter2_adhoc_query", source="hf", load_as="adhoc_query")
print("Adapter load success!")

for param in model.parameters():
    param.requires_grad = False

for name, param in model.named_parameters():
    if "adapters.adhoc_query" in name:
        param.requires_grad = True

device = None
if device is None:
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
else:
    device = device

model.to(device)

print("done!")

"""Convert Lightning .ckpt to raw state_dict, strip prefix, save for HF upload."""
import torch
import sys

CKPT_PATH = sys.argv[1] if len(sys.argv) > 1 else "models/vitl.ckpt"
OUT_PATH = sys.argv[2] if len(sys.argv) > 2 else "models/model.pt"

ckpt = torch.load(CKPT_PATH, map_location="cpu")
state = ckpt["state_dict"]

# Strip Lightning prefix dynamically — find common prefix
keys = list(state.keys())
prefix = keys[0]
for k in keys:
    while prefix and not all(s.startswith(prefix) for s in keys):
        prefix = prefix.rsplit(".", 1)[0]

print(f"Detected prefix: '{prefix}.' ({len(prefix)+1} chars)")
clean = {k[len(prefix)+1:]: v for k, v in state.items()}

torch.save(clean, OUT_PATH)
print(f"Saved: {OUT_PATH}")
print(f"Sample keys: {list(clean.keys())[:5]}")

import requests
from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
CACHE_PATH = "models"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# url = 'https://huggingface.co/datasets/mishig/sample_images/resolve/main/tiger.jpg'
# image = Image.open(requests.get(url, stream=True).raw)
image  = Image.open("assets/cats.jpg").convert("RGB")
image_processor = AutoImageProcessor.from_pretrained("apple/DepthPro-hf", cache_dir=CACHE_PATH)
model = AutoModelForDepthEstimation.from_pretrained("apple/DepthPro-hf", cache_dir=CACHE_PATH).to(device)

inputs = image_processor(images=image, return_tensors="pt").to(device)

with torch.no_grad():
    outputs = model(**inputs)

post_processed_output = image_processor.post_process_depth_estimation(
    outputs, target_sizes=[(image.height, image.width)],
)

depth = outputs[0]["predicted_depth"]
depth = (depth - depth.min()) / (depth.max() - depth.min())
depth = 1.0 - depth.detach().cpu().numpy()
import matplotlib.pyplot as plt
plt.imshow(depth, cmap='inferno')
plt.colorbar()
plt.show()
plt.close()
plt.imshow(depth, cmap='inferno')
plt.colorbar()
plt.savefig("test.png", dpi=300)

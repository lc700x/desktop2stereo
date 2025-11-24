# Initialize and run inference
img  = "C:/Users/zjuli/Pictures/test1.jpg"
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # ---DA2---
# from transformers import AutoImageProcessor, AutoModelForDepthEstimation
# from PIL import Image
# image = Image.open(img)

# image_processor = AutoImageProcessor.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf")
# model = AutoModelForDepthEstimation.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf", cache_dir = "models")

# # prepare image for the model
# inputs = image_processor(images=image, return_tensors="pt")

# with torch.no_grad():
#     outputs = model(**inputs)
#     predicted_depth = outputs.predicted_depth

# interpolate to original size
# prediction = torch.nn.functional.interpolate(
#     predicted_depth.unsqueeze(1),
#     size=image.size[::-1],
#     mode="bicubic",
#     align_corners=False,
# )


# ---DA3---
from models.depth_anything_3.api import DepthAnything3

# Load model from Hugging Face Hub
model = DepthAnything3.from_pretrained("depth-anything/DA3-SMALL", cache_dir = "models")
model = model.to(device=device)

# Run inference on images
images = [img]  # List of image paths, PIL Images, or numpy arrays
prediction = model.inference(
    images,
    # export_dir="output",
    # export_format="glb",  # Options: glb, npz, ply, mini_npz, gs_ply, gs_video
    process_res_method="lower_bound_resize"
)

# Access results
print(prediction.depth.shape)        # Depth maps: [N, H, W] float32
print(prediction.conf.shape)         # Confidence maps: [N, H, W] float32
print(prediction.extrinsics.shape)   # Camera poses (w2c): [N, 3, 4] float32
print(prediction.intrinsics.shape)   # Camera intrinsics: [N, 3, 3] float32

def normalize_result(arr):
    normalized_arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-6)
    return normalized_arr

if __name__ == "__main__":
    
    # # DA2
    # depth = predicted_depth.squeeze(0).squeeze(0)
    # depth = normalize_result(depth)
    
    # DA3
    depth = prediction.depth.squeeze(0)
    depth = normalize_result(depth)
    depth = 1 - depth
    
    

    import matplotlib.pyplot as plt
    plt.imshow(depth, cmap='inferno')
    plt.colorbar()
    plt.show()
    plt.close()
    plt.imshow(depth, cmap='inferno')
    plt.colorbar()
    plt.savefig("test.png", dpi=300)
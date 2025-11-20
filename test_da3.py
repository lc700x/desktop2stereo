# Initialize and run inference
from depth_anything_3.api import DepthAnything3
model = DepthAnything3.from_pretrained("depth-anything/DA3-SMALL").to("cuda")
img  = "C:/Users/zjuli/Pictures/test1.jpg"
prediction = model.forward([img])
print(prediction)
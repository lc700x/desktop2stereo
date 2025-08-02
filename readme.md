# Desktop2Stereo: 2D desktop to 3D stereo SBS (Support AMD/NVIDIA GPUs with DirectML, powered by Depth Anything AI Models)
[中文版本](./readmeCN.md)
## Hardware
AMD/NVIDIA GPUs and other DirectML compatible devices
## OS
Windows 10/11 64-bit OS
# Software
1. AMD GPU driver from [AMD Drivers and Support for Processors and Graphics](https://www.amd.com/en/support/download/drivers.html). For Other Compatible DirectML devices: (i.e. Nvidia GPU, .etc) please install latest hardware driver. 
2. Install **Python 3.10** from [Python.org](https://www.python.org/ftp/python/3.10.0/python-3.10.0-amd64.exe)
## Install and Run
1. Download Desktop2Stereo
   Download the latest release and unzip it to local disk. 
2. Install python environment  
Doulbe click `install.bat`
3. Run Stereo Desktop application  
Doulbe click `run.bat`
4. Move the **Stereo SBS Viewer** window to another (virtual) monitor display.
5. Set your video/game on the main screen (full screen mode if you needed)
6. Click the **Stereo SBS Viewer** on the another (virtual) monitor display to make sure the **Stereo SBS Viewer** is the 1st active application. Press `space` to toggle full screen mode. 
6. Now you can use AR/VR to view the Full/Half SBS output. 
- AR need to switch to 3D mode to connect as a 3840*1080 display
![Full-SBS](./assets/FullSBS.png)
- VR need to use 2nd Display/Virtual Display (VDD) with Desktop+[PC VR] or Virtual Desktop[PC/Standalone VR] or OBS+Wolvic [Standalone VR] to comopose the SBS display to 3D.
![Half-SBS](./assets/HalfSBS.png)
## Optional
1. Change Model
Modify the depth model id in the `depth.py` from [HuggingFace](https://huggingface.co/), the model id **must ends with** `-hf`. 
```python
# Initialize DirectML Device
DML = torch_directml.device()
print(f"Using DirectML device: {torch_directml.device_name(0)}")
DTYPE = torch.float16
MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"
```
- Default model id: `depth-anything/Depth-Anything-V2-Small-hf`
- All supported models:  
`LiheYoung/depth-anything-large-hf`  
`LiheYoung/depth-anything-base-hf`  
`LiheYoung/depth-anything-small-hf`  
`depth-anything/Depth-Anything-V2-Large-hf`  
`depth-anything/Depth-Anything-V2-Base-hf`  
`depth-anything/Depth-Anything-V2-Small-hf`  

2. Change Captured Monitor
Modify the `MONITOR_INDEX` in the `main.py` (1 - Primary Monitor).
Recomand to set `DOWNSCALE_FACTOR` value to 0.5 (2160p to 1080P) or set system resolution to 1080p for smoother experience
```python
# Set the monitor index and downscale factor
MONITOR_INDEX = 1  # Change to 0 for all monitors, 1 for primary monitor, ...
DOWNSCALE_FACTOR = 0.5 # Set to 1.0 for no downscaling, 0.5 is recommended for performance
```
## References
```BIBTEX
@article{depth_anything_v2,
  title={Depth Anything V2},
  author={Yang, Lihe and Kang, Bingyi and Huang, Zilong and Zhao, Zhen and Xu, Xiaogang and Feng, Jiashi and Zhao, Hengshuang},
  journal={arXiv:2406.09414},
  year={2024}
}

@inproceedings{depth_anything_v1,
  title={Depth Anything: Unleashing the Power of Large-Scale Unlabeled Data}, 
  author={Yang, Lihe and Kang, Bingyi and Huang, Zilong and Xu, Xiaogang and Feng, Jiashi and Zhao, Hengshuang},
  booktitle={CVPR},
  year={2024}
}
```

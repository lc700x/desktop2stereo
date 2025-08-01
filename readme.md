# Desktop2Stereo: 2D desktop to 3D stereo SBS (Support AMD/NVIDIA GPU/Apple Silicon, powered by Depth Anything AI Models)
[中文版本](./readmeCN.md)
## Supported Hardware  
1. AMD GPU
2. NVIDIA GPU 
3. Apple Silicon Chip (M1, M2, M3, M4, ...)
4. DirectML compatible devices (for **Windows** only)
## Supported OS
1. Windows 10/11 64-bit OS  
2. MacOS 10.9 or later
3. Linux (beta)
# Install and Run  
## Windows 
1. Install latest GPU driver  
**AMD GPU**: Download latest GPU driver from [AMD Drivers and Support for Processors and Graphics](https://www.amd.com/en/support/download/drivers.html). 
**NVIDIA GPU**: Download latest GPU driver from [AMD Drivers and Support for Processors and Graphics](https://www.nvidia.com/en-us/geforce/drivers/).
**Other DirectML devices**: Please install latest hardware driver accordingly.
2. Install **Python 3.10**  
    Download from [Python.org](https://www.python.org/ftp/python/3.10.11/python-3.10.11-amd64.exe) and install.
3. Download Desktop2Stereo app  
   Download the [Desktop2Stereo.zip](https://github.com/lc700x/desktop2stereo/releases/tag/v1.1) and unzip it to local disk. 
4. Install python environment  
**AMD GPU and other DirectML compatible devies**: Doulbe click `install-dml.bat`  
**NVIDIA GPU**: Doulbe click `install-cuda.bat`  
1. Run Stereo Desktop application  
Doulbe click `run.bat`
## MacOS 
1. Install **Python 3.10**  
    Download from [Python.org](https://www.python.org/ftp/python/3.10.11/python-3.10.11-macos11.pkg) and install.
2. Download Desktop2Stereo app  
   Download the [Desktop2Stereo.zip](https://github.com/lc700x/desktop2stereo/releases/tag/v1.1) and unzip it to local disk.
3. Install python environment  
Doulbe click `install-mps` executable
1. Run Stereo Desktop application  
Doulbe click `run_mac` executable
## Linux (Beta)
1. Install latest GPU driver  
**AMD GPU**: Download latest GPU driver from [AMD Drivers and Support for Processors and Graphics](https://www.amd.com/en/support/download/drivers.html). 
**NVIDIA GPU**: Download latest GPU driver from [AMD Drivers and Support for Processors and Graphics](https://www.nvidia.com/en-us/geforce/drivers/).
1. Install **Python 3.10**  
    ```bash
    # Example: Ubuntu
    sudo add-apt-repository ppa:savoury1/python
    sudo apt update
    sudo apt-get install python3.10
    ```
2. Download Desktop2Stereo app  
   Download the [Desktop2Stereo.zip](https://github.com/lc700x/desktop2stereo/releases/tag/v1.1) and unzip it to local disk.
3. Install python environment  
**AMD GPU**: Doulbe click `install-rocm.bat`  
**NVIDIA GPU**: Doulbe click `install-cuda.bat`  
1. Run Stereo Desktop application  
Doulbe click `run_linux` executable
# Setup the Desktop2Stereo display
1. Move the **Stereo SBS Viewer** window to second (virtual) monitor display.
2. Set your video/game on the main screen (full screen mode if you needed)
3. Click the **Stereo SBS Viewer** on second (virtual) monitor display to make sure the **Stereo SBS Viewer** is the 1st active application. Press `space` to toggle full screen mode. 
4. Now you can use AR/VR to view the Full/Half SBS output. 
- AR need to switch to 3D mode to connect as a 3840*1080 display
![Full-SBS](./assets/FullSBS.png)
- VR need to use 2nd Display/Virtual Display (VDD) with Desktop+[PC VR] or Virtual Desktop[PC/Standalone VR] or OBS+Wolvic [Standalone VR] to comopose the SBS display to 3D.
![Half-SBS](./assets/HalfSBS.png)
## Optional
1. Change Captured Monitor and Scale
Modify the `MONITOR_INDEX` in the `main.py` (1 - Primary Monitor).  
Recommend to set `DOWNSCALE_FACTOR` value to 0.5 (2160p to 1080P) or set system resolution to 1080p for a smoother experience. 
    ```python
    # Set the monitor index and downscale factor
    MONITOR_INDEX = 1  # Change to 0 for all monitors, 1 for primary monitor, ...
    DOWNSCALE_FACTOR = 0.5 # Set to 1.0 for no downscaling, 0.5 is recommended for performance
    ```
2. Change Depth Model
Modify the depth model id in the `depth.py` from [HuggingFace](https://huggingface.co/), the model id **must ends with** `-hf`. 
    ```python
    # Model configuration
    MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"
    DTYPE = torch.float16
    ```
    Default model id: `depth-anything/Depth-Anything-V2-Small-hf`  
    **All supported models**:  
    ```Bash
   LiheYoung/depth-anything-large-hf
   LiheYoung/depth-anything-base-hf
   LiheYoung/depth-anything-small-hf
   depth-anything/Depth-Anything-V2-Large-hf
   depth-anything/Depth-Anything-V2-Base-hf
   depth-anything/Depth-Anything-V2-Small-hf
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

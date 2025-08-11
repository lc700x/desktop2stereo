# Desktop2Stereo: 2D desktop to 3D stereo SBS (Support AMD/NVIDIA GPU/Apple Silicon, powered by Depth Estimation AI Models)  
[中文版本](./readmeCN.md)  
## Supported Hardware  
1. AMD GPU  
2. NVIDIA GPU  
3. Apple Silicon Chip (M1, M2, M3, M4, ...)  
4. DirectML compatible devices (Intel Arc/Iris GPU, etc. **Windows** only)
## Supported OS  
1. Windows 10/11 64-bit OS  
2. MacOS 10.16 or later  
3. Linux OS (beta)  
## Install and Run  
### Windows  
1. Install latest GPU driver  
    **AMD GPU**: Download latest GPU driver from [AMD Drivers and Support for Processors and Graphics](https://www.amd.com/en/support/download/drivers.html). 
    **NVIDIA GPU**: Download latest GPU driver from [NVIDA Official GeForce Drivers](https://www.nvidia.com/en-us/geforce/drivers/).  
    **Intel GPU**: Download latest GPU driver from [Download Intel Drivers and Software](https://www.intel.com/content/www/us/en/download-center/home.html/).  
    **Other DirectML devices**: Please install the latest hardware driver accordingly.  
2. Install **Python 3.10**  
    Download from [Python.org](https://www.python.org/ftp/python/3.10.11/python-3.10.11-amd64.exe) and install.  
3. Download Desktop2Stereo app  
   Download the [Desktop2Stereo.zip](https://github.com/lc700x/desktop2stereo/releases/latest) and unzip it to local disk.  
4. Install python environment  
    **AMD/Intel GPU and other DirectML compatible devies**: Doulbe click `install-dml.bat`.  
    **NVIDIA GPU**: Doulbe click `install-cuda.bat`.  
5. Run Stereo2Desktop application  
    Doulbe click `run.bat`.  
### MacOS 
1. Install **Python 3.10**  
    Download from [Python.org](https://www.python.org/ftp/python/3.10.11/python-3.10.11-macos11.pkg) and install.  
2. Download Desktop2Stereo app  
   Download the [Desktop2Stereo.zip](https://github.com/lc700x/desktop2stereo/releases/latest) and unzip it to local disk.  
3. Install Python environment  
    Doulbe click `install-mps` executable. (Please allow open in **privacy and security settings**)
4. Run Stereo2Desktop application  
    Doulbe click `run_mac` executable.  (Please allow open in **privacy and security settings**, "Screen Recording" permission is required)
### Linux (Beta)
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
   Download the [Desktop2Stereo.zip](https://github.com/lc700x/desktop2stereo/releases/latest) and unzip it to local disk.
3. Install Python environment  
    **AMD GPU**: Run `install-rocm.bash`: 
    ```bash
    bash install-rocm.bash
    ```
    **NVIDIA GPU**: Run `install-cuda.bash`:  
    ```bash
    bash install-cuda.bash
    ```
4. Run Stereo2Desktop application  
    Run `run_linux.bash`:  
    ```bash
    bash run_linux.bash
    ```
## Desktop2Stereo GUI

## Setup the Desktop2Stereo display  
1. Use arrow keys `<- Left ` or `-> Right` to switch the **Stereo SBS Viewer** window to second (virtual) monitor display. 
2. Set your video/game on the main screen (full screen mode if you needed).  
3. Click the **Stereo SBS Viewer** on second (virtual) monitor display to make sure the **Stereo SBS Viewer** is the 1st active application. Press `space` to toggle full screen mode.   
4. Now you can use AR/VR to view the Full/Half SBS output.   
   - **AR** need to switch to 3D mode to connect as a 3840*1080 display.  
   ![Full-SBS](./assets/FullSBS.png)
   - **VR** need to use 2nd Display/Virtual Display (VDD) with Desktop+[PC VR] or Virtual Desktop[PC/Standalone VR] or OBS+Wolvic [Standalone VR] to comopose the SBS display to 3D.  
   ![Half-SBS](./assets/HalfSBS.png)
## Optional Settings  
All optional settings are available in the `settings.yaml`. Use a text editor to edit the default values if necessary.   
1. Monitor index  
    `1` referst to your Primary Monitor (mostly shall follow the monitor numbers in your system settings).  
    ```yaml
    monitor_index : 1
   ```
2. Downscale Factor  
   The `downscale_factor` determines the output resolution. Recommend to set it to `0.5` for 4K monitors or set system resolution to 1080P for a smoother experience.
   ```yaml
    downscale_factor : 0.5
   ```
3. Input FPS (frames per second)
   FPS can set as your monitor refresh rate. (higher FPS doesn't mean smoother output, depending on your hardware)  
    ```yaml
    fps : 60
    ```
4. Depth Resolution
   Higher depth resolution can give better depth details, which is also related to the model training settings. 
    ```yaml
    depth_resolution : 384
    ```
5. Depth Model
    Modify the depth model id from [HuggingFace](https://huggingface.co/), the model id under `depth_model` **mostly shall ends** with `-hf`.  
    ```yaml
    depth_model :  depth-anything/Depth-Anything-V2-Small-hf
    ```
    Default model id: `depth-anything/Depth-Anything-V2-Small-hf`  
    **Currently supported models**:  
    ```Bash
    depth-anything/Depth-Anything-V2-Large-hf
    depth-anything/Depth-Anything-V2-Base-hf
    depth-anything/Depth-Anything-V2-Small-hf
    depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf
    depth-anything/Depth-Anything-V2-Metric-Outdoor-Base-hf
    depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf
    depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf
    depth-anything/Depth-Anything-V2-Metric-Indoor-Base-hf
    depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf
    LiheYoung/depth-anything-large-hf
    LiheYoung/depth-anything-base-hf
    LiheYoung/depth-anything-small-hf
    xingyang1/Distill-Any-Depth-Large-hf
    xingyang1/Distill-Any-Depth-Small-hf
    apple/DepthPro-hf # Depth: 1536
    Intel/dpt-large # Slow, NOT recommand
    ```
   You can also try the hugging face models which including the following:  
   `model.safetensors`  
   `config.json`  
   `preprocessor_config.json`  
   
6. Download Path  
   Default download path is the `models` folder under the working directory.    
   ```yaml
   download_path : ./models
   ```
7. Hugging Face Download Endpoint  
   [HF-Mirror](https://hf-mirror.com) is a mirror site of the original [Hugging Face](https://huggingface.co/) site hosting AI models.  
   ```yaml
   hf_endpoint : https://hf-mirror.com
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

@inproceedings{Bochkovskii2024:arxiv,
  author     = {Aleksei Bochkovskii and Ama\"{e}l Delaunoy and Hugo Germain and Marcel Santos and
               Yichao Zhou and Stephan R. Richter and Vladlen Koltun},
  title      = {Depth Pro: Sharp Monocular Metric Depth in Less Than a Second},
  booktitle  = {International Conference on Learning Representations},
  year       = {2025},
  url        = {https://arxiv.org/abs/2410.02073},
}

@article{he2025distill,
  title   = {Distill Any Depth: Distillation Creates a Stronger Monocular Depth Estimator},
  author  = {Xiankang He and Dongyan Guo and Hongji Li and Ruibo Li and Ying Cui and Chi Zhang},
  year    = {2025},
  journal = {arXiv preprint arXiv: 2502.19204}
}
```

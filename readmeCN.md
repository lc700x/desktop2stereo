# Desktop2Stereo：将2D桌面转为3D立体SBS（支持 AMD/NVIDIA GPU/Apple Silicon，基于 Depth Anything AI 模型）  
[English Version](./README.md)  

## 支持的硬件  
1. AMD GPU  
2. NVIDIA GPU  
3. Apple Silicon芯片（M1、M2、M3、M4 等） 
4. 其他支持 DirectML 的设备（仅支持**Windows**）  
## 支持的操作系统  
1. Windows 10/11 64 位  
2. MacOS 10.19 或更高版本  
3. Linux（测试） 

# 安装与运行  
## Windows  
1. 安装最新的 GPU 驱动  
   **AMD GPU**：从 [AMD 驱动和支持](https://www.amd.com/en/support/download/drivers.html) 下载最新的 GPU 驱动。  
   **NVIDIA GPU**：从 [NVIDIA 驱动和支持](https://www.nvidia.com/en-us/geforce/drivers/) 下载最新的 GPU 驱动。  
   **其他DirectML设备**：下载安装最新的设备驱动。  
2.  安装 **Python 3.10**  
从 [Python.org](https://www.python.org/ftp/python/3.10.11/python-3.10.11-amd64.exe)下载安装包并安装。  
3. 下载Desktop2Stereo  
   下载并解压[Desktop2Stereo.zip](https://github.com/lc700x/desktop2stereo/releases/latest)到本地。  
4. 安装 Python 环境  
   **AMD GPU**：双击 `install-dml.bat`。  
   **NVIDIA GPU**：双击 `install-cuda.bat`。  
5. 运行 Stereo Desktop 应用。  
   双击 `run.bat`。  

## MacOS
1. 安装 **Python 3.10**  
   从 [Python.org](https://www.python.org/ftp/python/3.10.11/python-3.10.11-macos11.pkg)下载安装包并安装。  
2. 下载Desktop2Stereo  
   下载并解压[Desktop2Stereo.zip](https://github.com/lc700x/desktop2stereo/releases/latest)到本地。  
3. 安装 Python 环境  
   双击 `install-mps` 可执行文件。  
4. 运行 Stereo Desktop 应用  
   双击 `run_mac` 可执行文件。  

## Linux（测试版）
1. 安装最新的 GPU 驱动  
   **AMD GPU**：从 [AMD 驱动和支持](https://www.amd.com/en/support/download/drivers.html) 下载最新的 GPU 驱动。对于其他兼容 DirectML 的设备，请安装最新的硬件驱动。  
   **NVIDIA GPU**：从 [NVIDIA 驱动和支持](https://www.nvidia.com/en-us/geforce/drivers/) 下载最新的 GPU 驱动。  
2. 安装 **Python 3.10**  
   ```bash
   
    sudo add-apt-repository ppa:savoury1/python
    sudo apt update
    sudo apt-get install python3.10
    ```
3. 下载Desktop2Stereo  
   下载并解压[Desktop2Stereo.zip](https://github.com/lc700x/desktop2stereo/releases/latest)到本地。  
4. 安装 Python 环境   
   **AMD GPU**：运行 `install-rocm.bash`脚本：  
   ```bash
   bash install-rocm.bash
   ```
   **NVIDIA GPU**：运行 `install-cuda.bash`脚本：  
   ```bash
   bash install-cuda.bash
   ```
6. 运行 Stereo Desktop 应用  
   运行`run_linux.bash` 脚本：  
   ```bash
   bash run_linux.bash
   ```

# 设置 Desktop2Stereo 显示
1. 将 **Stereo SBS Viewer** 窗口拖动到第二块（虚拟）显示器上。  
2. 在主屏幕上播放你的视频/游戏（如需可切换为全屏模式）。  
3. 在第二块（虚拟）显示器上点击 **Stereo SBS Viewer** 窗口，确保该窗口处于活动状态。按 `space` 键切换全屏模式。  
4. 现在，你可以使用 AR/VR 设备观看全/半双屏并排输出。  

- AR 需要切换到 3D 模式并连接为 3840\*1080 显示器。  

- VR 需使用第二块显示器/虚拟显示器（VDD），通过 Desktop+[PC VR] 或 Virtual Desktop [PC/一体机 VR] 或 OBS+Wolvic [一体机 VR] 来将 SBS 输出组合成 3D。  


## 可选项
1. 更改捕获的显示器和缩放比例  
   在 `main.py` 中修改 `MONITOR_INDEX`（1 - 主显示器）  
   建议将 `DOWNSCALE_FACTOR` 设置为 0.5（2160p 降至 1080p），或将系统分辨率设置为 1080p，以获得更流畅的体验
   ```python
   # Set the monitor index and downscale factor
   MONITOR_INDEX = 1  # Change to 0 for all monitors, 1 for primary monitor, ...
   DOWNSCALE_FACTOR = 0.5 # Set to 1.0 for no downscaling, 0.5 is recommended for performance
   ```

2. 更换深度模型  
   在 `depth.py` 中修改 [HuggingFace](https://huggingface.co/) 上的深度模型 ID，模型 ID **必须以** `-hf` 结尾。  
   ```python
   # Model configuration
    MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"
    DTYPE = torch.float16
   ```
   默认模型 ID：`depth-anything/Depth-Anything-V2-Small-hf`  
   **支持的全部模型**：  
   ```Bash
   LiheYoung/depth-anything-large-hf
   LiheYoung/depth-anything-base-hf
   LiheYoung/depth-anything-small-hf
   depth-anything/Depth-Anything-V2-Large-hf
   depth-anything/Depth-Anything-V2-Base-hf
   depth-anything/Depth-Anything-V2-Small-hf
   ```

## 参考文献
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

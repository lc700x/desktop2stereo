# Desktop2Stereo：将2D桌面转为3D立体SBS（支持 AMD/NVIDIA/Intel GPU/Apple Silicon，基于深度估算AI模型）  
[English Version](./README.md)  

## 支持的硬件  
1. AMD GPU  
2. NVIDIA GPU  
3. Apple Silicon芯片（M1、M2、M3、M4 等） 
4. 其他支持 DirectML 的设备（Intel Arc/Iris GPU等，仅支持**Windows**）  
## 支持的操作系统  
1. Windows 10/11 64 位  
2. MacOS 10.9 或更高版本  
3. Linux（测试版） 

# 安装与运行  
## Windows  
1. 安装最新的 GPU 驱动  
   **AMD GPU**：从 [AMD 驱动和支持](https://www.amd.com/en/support/download/drivers.html) 下载最新的 GPU 驱动。  
   **NVIDIA GPU**：从 [NVIDIA 驱动和支持](https://www.nvidia.com/en-us/geforce/drivers/) 下载最新的 GPU 驱动。  
   **其他DirectML设备**：如Intel GPU，请下载安装最新的设备驱动。  
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
    # 以Ubuntu为例
    sudo add-apt-repository ppa:savoury1/python
    sudo apt update
    sudo apt-get install python3.10
    ```
3. 下载Desktop2Stereo  
   下载并解压[Desktop2Stereo.zip](https://github.com/lc700x/desktop2stereo/releases/latest)到本地。  
4. 安装 Python 环境   
   **AMD GPU**：双击 `install-rocm.bat`。  
   **NVIDIA GPU**：双击 `install-cuda.bat`。  
5. 运行 Stereo Desktop 应用  
   双击 `run_linux` 可执行文件。  

# 设置 Desktop2Stereo 显示
1. 将 **Stereo SBS Viewer** 窗口拖动到第二块（虚拟）显示器上。  
2. 在主屏幕上播放你的视频/游戏（如需可切换为全屏模式）。  
3. 在第二块（虚拟）显示器上点击 **Stereo SBS Viewer** 窗口，确保该窗口处于活动状态。按 `space` 键切换全屏模式。  
4. 现在，你可以使用 AR/VR 设备观看全/半双屏并排输出。  

- AR 需要切换到 3D 模式并连接为 3840\*1080 显示器。  

- VR 需使用第二块显示器/虚拟显示器（VDD），通过 Desktop+[PC VR] 或 Virtual Desktop [PC/一体机 VR] 或 OBS+Wolvic [一体机 VR] 来将 SBS 输出组合成 3D。  


## 可选项
**配置文件**： `settings.yml`  
1. 更改捕获的显示器和缩放比例  
   修改 `MONITOR_INDEX`（1 - 主显示器）。  
   建议将 `DOWNSCALE_FACTOR` 设置为 0.5（2160p 降至 1080p），或将系统分辨率设置为 1080p，以获得更流畅的体验。  
   ```yaml
   monitor_index : 1
   downscale_factor : 0.5
   ```

2. 更换深度模型  
   修改深度模型 ID为其他[HuggingFace](https://huggingface.co/) 上的深度估算模型，`depth_model` 的值**必须以** `-hf` 结尾。  
   ```yaml
   depth_model :  depth-anything/Depth-Anything-V2-Small-hf
   ```
   默认模型 ID：`depth-anything/Depth-Anything-V2-Small-hf`  
   **支持的全部模型**：  
   ```Bash
   depth-anything/Depth-Anything-V2-Large-hf
   depth-anything/Depth-Anything-V2-Base-hf
   depth-anything/Depth-Anything-V2-Small-hf
   LiheYoung/depth-anything-large-hf
   LiheYoung/depth-anything-base-hf
   LiheYoung/depth-anything-small-hf
   apple/DepthPro-hf # depth_resolution 1536
   ```
3. 更改模型下载路径
   默认为主目录下的`models`文件夹，按需更改`download_path`即可:  
   ```yaml
   # model download path
   download_path : ./models
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

@inproceedings{Bochkovskii2024:arxiv,
  author     = {Aleksei Bochkovskii and Ama\"{e}l Delaunoy and Hugo Germain and Marcel Santos and
               Yichao Zhou and Stephan R. Richter and Vladlen Koltun},
  title      = {Depth Pro: Sharp Monocular Metric Depth in Less Than a Second},
  booktitle  = {International Conference on Learning Representations},
  year       = {2025},
  url        = {https://arxiv.org/abs/2410.02073},
}
```

# Desktop2Stereo：将2D桌面转为3D立体SBS（支持 AMD/NVIDIA/Intel GPU/Apple Silicon，基于深度估算AI模型）  
[English Version](./README.md)  

## 支持的硬件  
1. AMD GPU  
2. NVIDIA GPU  
3. Apple Silicon芯片（M1、M2、M3、M4 等） 
4. 其他支持 DirectML 的设备（Intel Arc/Iris GPU等，仅支持**Windows**）  
## 支持的操作系统  
1. Windows 10/11 64 位  
2. MacOS 10.16 或更高版本  
3. Linux（测试） 

# 安装与运行  
## Windows  
1. 安装最新的 GPU 驱动  
   **AMD GPU**：从 [AMD 驱动和支持](https://www.amd.com/en/support/download/drivers.html) 下载最新的 GPU 驱动。  
   **NVIDIA GPU**：从 [NVIDIA GeForce驱动](https://www.nvidia.com/en-us/geforce/drivers/) 下载最新的 GPU 驱动。  
   **Intel GPU**：从 [Intel驱动和软件](https://www.intel.com/content/www/us/en/download-center/home.html/) 下载最新的 GPU 驱动。  
   **其他DirectML设备**：请下载安装最新的设备驱动。  
2. 安装 **Python 3.10**  
从 [Python.org](https://www.python.org/ftp/python/3.10.11/python-3.10.11-amd64.exe)下载安装包并安装。  
3. 下载Desktop2Stereo  
   下载并解压[Desktop2Stereo.zip](https://github.com/lc700x/desktop2stereo/releases/latest)到本地。  
4. 安装 Python 环境  
   **AMD/Intel GPU/其他DirecML设备**：双击 `install-dml.bat`。  
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
1. 点击**Stereo SBS Viewer** 窗口，用键盘上的`<-左`货`右->`方向键将 **Stereo SBS Viewer** 窗口切换到第二块（虚拟）显示器上，按 `space` 空格键切换全屏模式。
2. 在主屏幕上播放你的视频/游戏（如需可切换为全屏模式）。
3. 现在，你可以使用 AR/VR 设备观看全宽/半宽双屏并排立体格式的输出。  

   - **AR** 需要切换到 3D 模式并连接为 3840\*1080 显示器。  
   ![Full-SBS](./assets/FullSBS.png)
   - **VR** 需使用第二块显示器/虚拟显示器（VDD），通过 Desktop+[PC VR] 或 Virtual Desktop [PC/一体机 VR] 或 OBS+Wolvic [一体机 VR] 来将 SBS 输出组合成 3D。  
   ![Half-SBS](./assets/HalfSBS.png)

## 可选项
所有的可选项都可以使用文本编辑器（如记事本等）在工作目录下的 `settings.yml`  中进行修改默认值。  
1. 显示器选择  
   `1` - 主显示器，多数情况下和系统-显示设置中的显示器编号一致。  
   ```yaml
   monitor_index : 1
   ```
2. 缩放值  
   `downscale_factor`决定了输出分辨率，建议为4k分辨率的显示器 设置为`0.5`，或将分辨率设置为 1080p，以获得更流畅的体验。  
   ```yaml
   downscale_factor : 0.5
   ```
3. 输入帧率(FPS)  
   FPS可以设置成显示器的刷新率。（更高的FPS并不意味输出更流畅，取决于硬件）  
   ```yaml
   fps : 60
   ```
4. 深度分辨率  
   更高的深度分辨率会有更好的深度细节，它也和模型训练时的参数有关联。  
   ```yaml
   depth_resolution : 384
   ```

5. AI深度模型  
   修改深度模型 ID为其他[HuggingFace](https://huggingface.co/) 上的深度估算模型，`depth_model` 的值**应该以** `-hf` **结尾**。  
   ```yaml
   depth_model :  depth-anything/Depth-Anything-V2-Small-hf
   ```
   默认模型 ID：`depth-anything/Depth-Anything-V2-Small-hf`  
   **目前支持的模型**：  
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
   apple/DepthPro-hf # 1536
   Intel/dpt-large # 慢，不推荐
   ```
   你也可以尝试其他带有以下文件的模型：  
   `model.safetensors`  
   `config.json`  
   `preprocessor_config.json`  
6. 更改模型下载路径  
   默认为工作目录下的`models`文件夹，按需更改`download_path`即可:  
   ```yaml
   download_path : ./models
   ```
7. Hugging Face镜像地址
   [HF-Mirror](https://hf-mirror.com) 是原版[Hugging Face](https://huggingface.co/) 的镜像站，上面有各种AI模型。  
   ```yaml
   hf_endpoint : https://hf-mirror.com
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

@article{he2025distill,
  title   = {Distill Any Depth: Distillation Creates a Stronger Monocular Depth Estimator},
  author  = {Xiankang He and Dongyan Guo and Hongji Li and Ruibo Li and Ying Cui and Chi Zhang},
  year    = {2025},
  journal = {arXiv preprint arXiv: 2502.19204}
}
```

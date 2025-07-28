# Desktop2Stereo: 2D桌面实时转换为3D立体SBS（支持AMD/NVIDIA GPU，基于Dept Anythng AI模型）
[English Version](./readme.md)

## 硬件要求

支持 DirectML 的 AMD/NVIDIA 显卡及其他兼容设备

## 操作系统

Windows 10/11 64 位系统

# 软件要求

1. 对于 AMD 显卡，请从 [AMD 驱动程序与支持](https://www.amd.com/en/support/download/drivers.html) 下载并安装 GPU 驱动程序。对于其他 DirectML 兼容设备（如 Nvidia GPU），请安装最新的硬件驱动。
2. 从 [Python.org](https://www.python.org/ftp/python/3.10.0/python-3.10.0-amd64.exe) 安装 Python 3.10。

## 安装与运行

1. 将代码库下载到本地磁盘：

```powershell
git clone https://github.com/lc700x/desktop2stereo
```

2. 安装 Python 环境
   双击 `install.bat`

3. 运行 Stereo Desktop 应用程序
   双击 `run.bat`

4. 将 Stereo SBS Viewer 窗口移动到另一台（虚拟）显示器。

5. 在主显示屏上播放视频或游戏（如需可设置为全屏模式）。

6. 点击另一台（虚拟）显示器上的 Stereo SBS Viewer 窗口，确保其处于激活状态，按下 `Space` 键可切换全屏模式。

7. 现在你可以使用 AR/VR 设备观看全幅/半幅 SBS 输出。

* 使用 AR 时，需要切换到 3D 模式，并连接为 3840\*1080 显示器。

![Full-SBS](./assets/FullSBS.png)

* 使用 VR 时，可通过第二个显示器或虚拟显示器(VDD) 搭配 Desktop+\[PC VR] 或 Virtual Desktop\[PC/一体机 VR] 或 OBS+Wolvic \[一体机 VR] 将 SBS 输出组合为 3D。

![Half-SBS](./assets/HalfSBS.png)

## 可选项

1. 更换模型
   修改 `depth.py` 中的 depth 模型 ID，从 [HuggingFace](https://huggingface.co/) 获取，模型 ID 必须以 `-hf` 结尾。

```python
# 初始化 DirectML 设备
DML = torch_directml.device()
print(f"Using DirectML device: {torch_directml.device_name(0)}")
DTYPE = torch.float16
MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"
```

* 默认模型 ID：`depth-anything/Depth-Anything-V2-Small-hf`

* 支持的所有模型：

`LiheYoung/depth-anything-large-hf`  
`LiheYoung/depth-anything-base-hf`  
`LiheYoung/depth-anything-small-hf`  
`depth-anything/Depth-Anything-V2-Large-hf`  
`depth-anything/Depth-Anything-V2-Base-hf`  
`depth-anything/Depth-Anything-V2-Small-hf`  

2. 更换捕获显示器
   在 `main.py` 中修改 `MONITOR_INDEX`（1 表示主显示器）。
   建议将 `DOWNSCALE_FACTOR` 设置为 0.5（2160p 降为 1080p），或将系统分辨率设置为 1080p 以获得更流畅的体验。

```python
MONITOR_INDEX = 1  # Change to 0 for all monitors, 1 for primary monitor, ...
DOWNSCALE_FACTOR = 0.5 # Set to 1.0 for no downscaling
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

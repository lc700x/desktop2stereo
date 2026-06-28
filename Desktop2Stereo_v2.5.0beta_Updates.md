# Desktop2Stereo Update: v2.4.2 -> v2.5.0 Beta.Post01

**Branch**: `Dev`  
**Commits**: 38

---

## 中文版本

### 新功能
- **MIGraphX 加速支持** — 通过 AMD MIGraphX 后端为 AMD GPU 提供推理加速，GUI 新增独立复选框（`MIGraphX` + `重编译 MIGraphX`）。
- **InfiniDepth 模型支持** — 新增 InfiniDepth 作为深度估计选项。
- **Depth-Anything-V1 室内外变体** — 新增针对不同场景的 DA-V1 室内/室外变体模型，提升深度精度。
- **macOS 本地查看器的 Metal 加速** ——通过 Apple Metal 加速提升本地查看器性能。
- **macOS ScreenCaptureKit** — macOS 上以现代 ScreenCaptureKit API 替换旧的 Quartz 捕获方式，性能和稳定性显著提升。
- **虚拟环境模型选择** — XR 模式新增环境下拉菜单，支持"默认"（黑色背景）和用户自定义的 `xr_viewer/environments/` 中 3D 场景或全景背景图片，支持从 `profile.json` 读取本地化显示名称。
- **XR 全景图片环境** — `xr_viewer/environments/` 下的图片文件夹会自动作为全景背景环境出现；支持 `background.*`、`panorama.*`、`equirectangular.*`、`360.*` 等常见命名，并新增 `Space` 示例环境。GUI 会将图片背景环境排在 GLB 环境之前。
- **XR 屏幕光效与曲面模式** — 长按左手 X 可按 `glow → veil → frosted → off` 循环光效；右摇杆短按可按水平曲面、垂直曲面、无曲面循环。
- **XR 视觉状态改为 JSON 保存** — Glow Mode、屏幕曲面状态和默认屏幕布局保存到 `xr_viewer/environments/*.json`，不再写入 `settings.yaml`。
- **XR 预览窗口** — 可切换的 OpenXR 预览窗口，独立于头显显示。
- **XR 远程引入多指触控API** — VD同款，提升光标控制操作体验。 
- **会聚点控制** — GUI 新增立体会聚调整滑块，默认值设为 `0`。
- **垂直同步 (VSync)** — 工具栏新增 VSync 复选框，控制本地查看器显示刷新同步。
- **模型骨架大小选择器** — 新下拉菜单按家族和大小（`Small`/`Base`/`Large`/`Giant` + 变体）分类深度模型，附带提示说明。
- **推理优化器修正** — 修复了各种硬件后端的优化器选择逻辑。
- **默认深度分辨率重置** — 将默认深度分辨率修正为矩形宽度`336px`，优化深度估计识别结果。 
- **显示器列表刷新修复** — 改进显示器检测与列表更新机制。
- **Flet 路径与 HF 符号链接修复** — 解决 Flet 存储路径和 Hugging Face 符号链接问题，稳定模型加载。
- **中国版 Mac 下载脚本** — 新增中国大陆 macOS 专用下载/更新脚本。
- **日志功能增强** — 统一控制台和子进程输出至 `logs/desktop2stereo.log` 单个滚动日志文件，带时间戳和来源标签。
- **WindowsCapture CUDA/ROCm 显卡捕获更新** — 优化`wc_cuda` 和 `wc_rocm` 模块实现加速桌面捕获。
- **ROCm 路径配置** — 新增 Windows AMD GPU 的 ROCm 安装路径设置。 

### Bug 修复
- **停止按钮等待时间缩短** — 优化停止按钮响应延迟，实现更快关闭。
- **XR 查看器使用分割文件加载** — 修复 XR 查看器从分割文件结构加载资产的问题。
- **Mac 上 CoreML 和 FP16 兼容性** — 禁用 FP16 并修正 Mac 上的 CoreML 推理设置以提升稳定性。
- **默认会聚点重置** — 将默认会聚值从前值修改为 `0`。
- **Depth-Anything-3 路径修正** — 修复 DA3 模型路径解析。
- **模型尺寸列表顺序修正** — 修正下拉框中模型尺寸的排序。
- **GUI 更新与模型测试完成** — 多项 GUI 修复及模型兼容性测试完成。
- **ROCm 错误修正** — 修复 ROCm 相关运行时错误。
- **代码优化** — 通用代码清理和性能改进。
- **分辨率变化时光标位置修复** — 修复动态分辨率变化时光标偏移问题。
- **默认深度分辨率修复** — 修正默认深度分辨率值。
- **ROCM 路径配置修复** — 修复 ROCm 环境变量路径检测。
- **README 修正** — 更新文档和安装说明。

---

### XR 查看器键盘快捷键 (v2.5.0 Beta)

基于对 `xr_viewer/implementation.py`、`xr_viewer/constants.py` 和 `xr_viewer/overlay.py` 的源代码分析：

**控制器动作绑定**  
所有绑定均来自 OpenXR 标准动作路径。代码在 `implementation.py:1352-1402` 中定义这些动作，并在 `implementation.py:1441-1521` 中按设备进行绑定。

**快捷键（映射到虚拟帮助面板）**  
在 `overlay.py:140-174` 中定义为 5 行帮助面板布局（来自 `utils` 中的 `ROWS`）：

| # | 动作 | 触发方式 | 代码位置 |
|---|------|----------|----------|
| 1 | 切换 FPS/性能面板 | 左菜单按钮（短按） | `impl.py:4953`, `impl.py:5660-5668` |
| 2 | 切换虚拟键盘 | 左 X 按钮（按下 < 1 秒） | `impl.py:5735`, `impl.py:5758` |
| 3 | 切换影院辉光 / 光照预设 | 左 X 按钮（按住 1‑4 秒后松开） | `impl.py:5754-5755` |
| 4 | 切换绿色透视背景 | 左 X 按钮（按住 > 4 秒） | `impl.py:5743-5745` |
| 5 | 重置屏幕为默认 | 左 Y 按钮（短按） | `impl.py:5709`, `impl.py:5726` |
| 6 | 循环切换屏幕预设 | 左 Y 按钮（长按 ≥ 0.6 秒） | `impl.py:5713-5721` |
| 7 | 循环切换环境 | 左摇杆单击（短按） | `impl.py:5829` |
| 8 | 禁用/恢复深度强度 | 左摇杆单击（短按，同时按住右握持键） | `impl.py:5822-5827` |
| 9 | 切换曲面/平面屏幕 | 右摇杆单击（短按） | `impl.py:5844` |
| 10 | 重置屏幕方向（面向头部） | 右摇杆单击（长按 ≥ 1 秒） | `impl.py:5837-5838` |
| 11 | 恢复深度比率（2.0） | 右摇杆单击（短按，同时按住右握持键） | `impl.py:5845-5849` |
| 12 | 全部隐藏（重置视角） | 右 A + B 双击 | `impl.py:5674`, `impl.py:5685-5686` |
| 13 | 鼠标右键点击 | 右 B 按钮（激光指向屏幕时） | `impl.py:5698-5699` |
| 14 | 鼠标左键点击 | 右 A 按钮（激光指向屏幕时） | `impl.py:5695-5696` |
| 15 | 鼠标左键点击（扳机） | 左扳机（扣动 = 点击，按住 = 拖拽） | `impl.py:4445` |
| 16 | 鼠标右键点击（扳机） | 右扳机（扣动 = 点击，按住 = 拖拽） | `impl.py:4445` |
| 17 | 切换快捷键帮助面板 | 激光指向 FPS 覆盖面板时扣动扳机 | `impl.py:4515-4516` |
| 18 | 切换手柄品牌 | 右 A + B 按住 ≥ 0.5 秒 | `impl.py:4992-5009` |
| 19 | 调整深度强度 | 右握持 + 左摇杆 Y 轴 | `impl.py:5521-5530` |
| 20 | 移动屏幕（平移 X/Y） | 左握持 + 左摇杆 X/Y | `impl.py:5491-5520` |
| 21 | 旋转屏幕（偏航） | 左握持 + 右摇杆 X | `impl.py:5620-5623` |
| 22 | 旋转屏幕（俯仰） | 左握持 + 右摇杆 Y | `impl.py:5624-5627` |
| 23 | 调整屏幕宽度 | 右握持 + 右摇杆 X | `impl.py:5579-5584` |
| 24 | 拉近/推远屏幕 | 右握持 + 右摇杆 Y | `impl.py:5586-5607` |
| 25 | 鼠标滚轮滚动 | 左摇杆 X/Y（不握持） | `impl.py:5533` |
| 26 | 鼠标滚轮滚动 | 右摇杆 X/Y（不握持，不单击摇杆） | `impl.py:5644-5646` |
| 27 | 双手移动屏幕 | 双手握持 + 双激光指向屏幕 | `impl.py:5321-5390` |
| 28 | 进入座位调整模式 | 双手握持 ≥ 3 秒（当屏幕被环境锁定时） | `impl.py:5041-5047` |
| 29 | 退出座位调整 | 双手松开握持（在座位调整模式下） | `impl.py:5043-5045` |
| 30 | 键盘平移（轨道） | 左握持 + 左摇杆（键盘可见时） | `impl.py:5457-5489` |
| 31 | 键盘调整宽度 | 右握持 + 右摇杆 X（键盘可见时） | `impl.py:5545-5548` |
| 32 | 键盘距离 | 右握持 + 右摇杆 Y（键盘可见时） | `impl.py:5550-5578` |
| 33 | 键盘偏航偏移 | 左握持 + 右摇杆 X（键盘可见时） | `impl.py:5613-5615` |
| 34 | 键盘俯仰偏移 | 左握持 + 右摇杆 Y（键盘可见时） | `impl.py:5616-5618` |
| 35 | 进入校准模式 | 左菜单 + 右 A + 右 B 按住 ≥ 1 秒 | `impl.py:4957-4971` |
| 36 | 保存校准（Y 偏移） | 校准模式下左摇杆 Y | `impl.py:4977-4978` |
| 37 | 保存校准（Z 偏移） | 校准模式下右摇杆 Y | `impl.py:4979-4980` |
| 38 | 保存校准（旋转） | 校准模式下右摇杆 X | `impl.py:4981-4982` |
| 39 | 退出校准（保存） | 校准模式下右 B 按钮 | `impl.py:4983-4988` |
| 40 | 退出校准（不保存） | 再次组合按键（左菜单 + 右 A + 右 B） | `impl.py:4963-4964` |

**虚拟键盘修饰键**  
键盘可见时，触发修饰键可切换状态（`impl.py:4838-4861`）：

| 按键 | 行为 |
|------|------|
| Shift | 短按 = 单次保持；双击 = 切换锁定；锁定时短按 = 解锁 |
| Ctrl | 短按 = 单次保持；双击 = 切换锁定；锁定时短按 = 解锁 |
| Alt | 短按 = 单次保持；双击 = 切换锁定；锁定时短按 = 解锁 |
| Win | 短按 = 单次保持；双击 = 切换锁定；锁定时短按 = 解锁 |
| Caps Lock | 短按 = 切换大写锁定 |
| Tab | 单次 Tab 键（0x09） |
| Backspace | 单次退格键（0x08） |
| Enter | 单次回车键（0x0D） |
| 方向键 | 单次方向键 |
| 数字/符号键 | 按住扳机 = 按下；松开 = 弹起；Shift/Caps 激活时显示上档标签 |

**Vive 触控板模拟**  
使用 Vive/WMR 触控板手柄时，触控板区域模拟按钮（`constants.py:56-59`, `impl.py:3849`）：

- 触控板顶部区域 → Y/X 按钮  
- 触控板底部区域 → A/B 按钮  
- 触控板中心 → 左/右摇杆单击  

---

### 环境配置文件说明

`environments/` 下的每个子文件夹代表一个可在 Desktop2Stereo 中加载的 XR 背景环境。文件夹可以是 3D GLB 环境，也可以是 360 度全景图片环境。GLB 环境使用 `environment.glb`（或 `profile.json` 中的 `glb` 字段）；图片环境可直接放置 `background.png`、`panorama.jpg`、`equirectangular.webp`、`360.png` 等常见命名的全景图，也可通过 `profile.json` 显式配置。GUI 中图片背景环境会排在 GLB 环境之前。

#### 快速开始

```
environments/
├── MyPanorama/
│   └── background.png      # 图片环境：无需 GLB / JSON
├── Space/
│   ├── background.png
│   └── profile.json        # 可选：全景参数、显示名、光效等
├── MyRoom/
│   ├── environment.glb     # 3D 环境
│   └── profile.json        # 可选
├── Cinema/
│   ├── environment.glb
│   └── profile.json
└── ...
```

图片环境最简单只需要一个图片文件。若 `profile.json` 不存在，应用会按文件夹名显示，并自动把图片当作 equirectangular/360 全景图。GLB 环境的配置文件可以**没有任何字段**——所有键都是可选的。如果 `profile.json` 不存在或为空，应用将使用合理默认值，你可以在 VR 中交互式调整。

---

#### `profile.json` 参考

所有字段均为可选。标有 `/` 的键表示向后兼容的别名（旧名称仍有效）。

##### 显示名称

```json
"display_name": { "EN": "My Room", "CN": "我的房间" }
```

GUI 下拉菜单中显示的本地化标签。若缺失则使用文件夹名称。

##### 3D 模型

| 键 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `glb` | string | `"environment.glb"` | glTF 模型路径，相对于配置文件所在目录。 |
| `model_position` / `position` | [x, y, z] | `[0, 0, 0]` | 世界空间偏移量，单位为米。 |
| `model_rotation_deg` / `rotation_deg` | [yaw, pitch, roll] | `[0, 0, 0]` | 旋转角度，单位为**度**。yaw=Y轴，pitch=X轴，roll=Z轴。 |
| `model_scale` / `scale` | [x, y, z] | `[1, 1, 1]` | 非均匀缩放系数。 |

##### 全景图片背景

无需 `profile.json` 时，文件夹中第一个匹配的图片会被自动识别为 360 度全景背景。优先识别的文件名包括 `background.*`、`panorama.*`、`equirectangular.*`、`360.*`、`sky.*`、`skybox.*`，支持 `.png`、`.jpg`、`.jpeg`、`.webp`、`.bmp`、`.tif`、`.tiff`。

```json
{
  "display_name": { "EN": "Space" },
  "environment_type": "panorama",
  "background": {
    "type": "equirectangular",
    "image": "background.png",
    "exposure": 1.0,
    "yaw_offset_deg": 0.0
  }
}
```

| 键 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `environment_type` / `type` | string | 自动检测 | 设置为 `"panorama"` 可显式声明图片环境。 |
| `background.image` / `panorama.image` | string | 自动检测 | 全景图片路径，相对于当前环境文件夹。 |
| `background.type` / `background.projection` | string | `"equirectangular"` | 全景投影类型。当前用于 360 equirectangular 图片。 |
| `background.exposure` | float | `1.0` | 图片背景亮度倍数。 |
| `background.yaw_offset_deg` | float | `0.0` | 水平旋转偏移，单位为度。 |
| `background.flip_y` | bool | `false` | 垂直翻转图片采样。 |

##### 光照

| 键 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `env_exposure` | float | `1.0` | HDR 曝光倍数。越高越亮。典型范围：0.3–1.5。 |
| `env_gamma` | float | `2.2` | 色调映射的伽马校正。 |
| `env_emissive_strength` | float | `1.0` | GLB 中自发光材质的强度倍数。影院屏幕可用 5–10。 |
| `env_khr_light_scale` / `khr_light_scale` | float | `1.0` | glTF `KHR_lights_punctual` 灯光强度倍数。 |
| `env_ambient_color` / `ambient_color` | [r, g, b] | `[0.08, 0.08, 0.09]` | RGB 环境光颜色（每通道 0–1）。 |
| `env_head_light_color` / `head_light_color` | [r, g, b] | `[0.45, 0.45, 0.48]` | RGB 头戴式点光源颜色（跟随观看者）。 |
| `screen_light_intensity` | float | `3.5` | 影院偏置光倍数（虚拟屏幕作为面光源）。 |

##### 补光

```json
"env_fill_lights": [
  { "position": [0, 2, -1], "color": [0.8, 0.7, 0.5], "range": 5.0 }
]
```

别名：`fallback_lights`。在世界空间中放置的带柔和距离衰减的点光源。片段着色器中最多评估 2 个（硬件限制）。

##### 光照预设

```json
"lighting_presets": [
  {
    "name": "白天",
    "env_exposure": 0.5,
    "env_ambient_color": [0.06, 0.05, 0.05],
    "env_head_light_color": [0.45, 0.45, 0.48]
  },
  {
    "name": "夜晚",
    "env_exposure": 0.16,
    "env_ambient_color": [0.01, 0.01, 0.012],
    "env_head_light_color": [0.03, 0.03, 0.05]
  }
]
```

在 VR 中**长按 Y 键**（左手柄）循环切换保存的光照配置。若缺失，长按 Y 键将循环屏幕预设（非锁定环境）或无操作。

##### 屏幕布局

```json
"screen": {
  "width": 5.0,
  "position": [0.0, 1.5, -6.0],
  "rotation_deg": [0.0, 0.0, 0.0],
  "curve_axis": "none",
  "allow_curve": true
}
```

当存在 `screen` 部分时，环境为**锁定状态**——虚拟屏幕会固定到配置的位置，用户无法用手柄自由移动。适用于屏幕应对齐 GLB 中建模的电视、投影墙或影院银幕的场景。

| 键 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `width` / `screen_width` | float | — | 屏幕宽度，单位为米。 |
| `position` / `screen_position` | [x, y, z] | — | 屏幕中心在世界空间中的位置。 |
| `rotation_deg` / `screen_rotation_deg` | [yaw, pitch, roll] | `[0, 0, 0]` | 屏幕朝向，单位为**度**。 |
| `curve_axis` | string | `"none"` | 初始曲面模式：`"horizontal"`、`"vertical"` 或 `"none"`。旧的 `curved: true` 会按 `"horizontal"` 读取。 |
| `allow_curve` | bool | `true` | 是否允许用户切换曲面模式。 |

##### 观众座位（视角位姿）

单个座位：

```json
"view_pose": {
  "x": 0.0,
  "y": 1.6,
  "z": -2.0,
  "angle": 0.0
}
```

多个座位：

```json
"view_pose_index": 0,
"view_poses": [
  { "name": "中间", "x": 0.0, "y": 1.6, "z": -2.0, "angle": 0.0 },
  { "name": "左侧", "x": -1.2, "y": 1.6, "z": -2.0, "angle": 8.0 },
  { "name": "右侧", "x": 1.2, "y": 1.6, "z": -2.0, "angle": -8.0 }
]
```

| 键 | 类型 | 说明 |
|---|---|---|
| `view_pose` / `camera` | object | 单个观众位置 `{x, y, z, angle}`。当 `view_poses` 不存在时使用。 |
| `view_poses` | array | 命名的座位位置列表。在锁定环境中**长按 Y 键**循环切换。 |
| `view_pose_index` | int | 当前活动座位索引（从 0 开始）。 |
| `name` | string | 座位切换 OSD 中显示的标签。 |
| `x`, `y`, `z` | float | 观众在世界空间中的位置，单位为米。 |
| `angle` | float | 观众朝向角度，单位为度。 |

`view_poses` 中的每个座位可包含一个可选的 `distance_width_ratio` 字段（float，默认 0.6），在未显式设置 `distance` 时，观看距离 = `width × ratio`。

##### 手柄模型偏移（按环境）

```json
"controller_overrides": {
  "model_offset": [0.0, 0.0, 0.0],
  "model_rotation_deg": 0.0
}
```

针对特定环境微调 VR 手柄模型的位置/旋转。通常不需要。

---

#### 完整示例

```json
{
  "display_name": { "EN": "Cinema", "CN": "影院" },
  "glb": "environment.glb",
  "model_position": [-68.65, -0.25, -30.59],
  "model_rotation_deg": [178.15, 0.0, 0.0],
  "model_scale": [1.0, 1.0, 1.0],
  "env_exposure": 0.5,
  "env_gamma": 2.2,
  "env_emissive_strength": 10.0,
  "env_khr_light_scale": 1.0,
  "env_ambient_color": [0.06, 0.05, 0.05],
  "env_head_light_color": [0.45, 0.45, 0.48],
  "screen_light_intensity": 3.5,
  "env_fill_lights": [],
  "lighting_presets": [
    {
      "name": "Day",
      "env_exposure": 0.5,
      "env_ambient_color": [0.06, 0.05, 0.05],
      "env_head_light_color": [0.45, 0.45, 0.48]
    },
    {
      "name": "Night",
      "env_exposure": 0.5,
      "env_ambient_color": [0.01, 0.01, 0.012],
      "env_head_light_color": [0.03, 0.03, 0.05]
    }
  ],
  "view_pose": { "x": 0.12, "y": 5.89, "z": -0.48, "angle": 0.0 },
  "view_pose_index": 2,
  "view_poses": [
    { "name": "Center",    "x": 0.12, "y": 5.89, "z": -0.48, "angle": 0.0 },
    { "name": "Front Row", "x": 0.12, "y": 4.07, "z": -1.03, "angle": 0.0 },
    { "name": "Back Row",  "x": 0.12, "y": 7.47, "z": 0.17,  "angle": 0.0 }
  ],
  "screen": {
    "width": 5.0,
    "position": [0.08, 1.54, -5.89],
    "rotation_deg": [-1.85, 0.0, 0.0],
    "curve_axis": "none",
    "allow_curve": true
  }
}
```

#### 提示

- **锁定环境**（包含 `screen` 部分）会阻止用户移动屏幕。屏幕作为环境模型的子级——移动模型即移动屏幕。
- **视角位姿** 允许预先配置座位位置。用户通过长按 Y 键循环切换。
- **光照预设** 与视角位姿相互独立。两者都通过长按 Y 键循环——当有视角位姿时，座位切换优先。
- **图片环境** 不加载 GLB，也不会生成暗房模型；全景图作为远景背景渲染，虚拟屏幕仍使用默认环境的可移动/可缩放行为。
- **XR 视觉状态**（默认屏幕布局、曲面方向、光效模式）保存到环境 JSON 中，不再写入 `settings.yaml`。
- 模型坐标遵循 glTF 惯例：Y 轴向上，Z 轴向前，右手坐标系。应用会自动转换到 XR 参考空间。
- 要测试配置文件更改，请重新启动 OpenXR 会话——配置文件在加载时只读取一次。

---

## English Version

### New Features
- **MIGraphX acceleration support** — AMD GPU inference via MIGraphX backend, with dedicated checkboxes (`MIGraphX` + `Recompile MIGraphX`) in the GUI.
- **InfiniDepth model support** — Added InfiniDepth as a depth estimation option.
- **Depth-Anything-V1 indoor/outdoor variants** — New model variants for improved depth accuracy in different scenes.
- **macOS ScreenCaptureKit** — Replaced legacy Quartz capture with modern ScreenCaptureKit API on macOS for better performance and stability.
- **Metal acceleration for local viewer on macOS** — Enhancement of local viewer performance via Apple Metal acceleration.
- **Virtual environment model selection** — Added environment dropdown in XR mode; supports "Default" (black backdrop) and user-defined 3D scenes and panoramas from `xr_viewer/environments/` with localized display names from `profile.json`.
- **XR panorama image environments** — Image-only folders under `xr_viewer/environments/` are detected as 360 panorama backgrounds; common names such as `background.*`, `panorama.*`, `equirectangular.*`, and `360.*` work automatically. Added the `Space` sample environment, and the GUI lists image backgrounds before GLB rooms.
- **XR screen glow and curve modes** — Long-press left X cycles `glow → veil → frosted → off`; short-press right thumbstick cycles horizontal curve, vertical curve, and no curve.
- **XR visual state stored in JSON** — Glow Mode, screen curve state, and default screen layout are saved under `xr_viewer/environments/*.json` instead of `settings.yaml`.
- **Windows multi-touch input in XR Link** — Same as Virtual Desktop for better cursor control experience.  
- **XR Preview Window** — Toggleable preview window for OpenXR mode, independent of the headset display.
- **Convergence control** — New GUI slider for stereo convergence adjustment, default set to `0`.
- **VSync toggle** — Added VSync checkbox in the toolbar; controls local viewer display refresh synchronization.
- **Model backbone size selector** — New dropdown splits depth models by family and size (`Small`/`Base`/`Large`/`Giant` + variants), with tooltips.
- **Huawei mirror for PyPI** — Updated package installation to use Huawei cloud mirrors for faster downloads in China.
- **Inference optimizer correction** — Fixed optimizer selection logic for various hardware backends.
- **Default depth resolution reset** — Corrected default depth resolution to rectangle shape with a height of `336px` for better depth estimation results.
- **Monitor list refresh fix** — Improved monitor detection and list updating.
- **Flet path and HF symlink fix** — Resolved Flet storage path and Hugging Face symlink issues for stable model loading.
- **Chinese Mac download script** — Added CN-specific macOS download/update script.
- **Logging improvements** — Unified console and child process output into a single rolling log file in `logs/desktop2stereo.log` with timestamped source labels.
- **WindowsCapture CUDA/ROCm GPU caputre updated** — Optimized `wc_cuda` and `wc_rocm` module for accelerated desktop capture.
- **ROCm path configuration** — Added ROCm installation path settings for Windows AMD GPU support.

### Bug Fixes
- **Stop button wait time reduced** — Optimized the stop button response delay for quicker shutdown.
- **XR viewer loading with split files** — Fixed XR viewer asset loading from split file structure.
- **CoreML and FP16 compatibility on Mac** — Disabled FP16 and corrected CoreML inference settings for Mac stability.
- **Default convergence reset** — Changed default convergence value from previous value to `0`.
- **Depth-Anything-3 path correction** — Fixed DA3 model path resolution.
- **List order for model sizes** — Corrected the ordering of model sizes in the dropdown.
- **GUI updates and model testing completion** — Various GUI fixes and completed model compatibility testing.
- **ROCm error correction** — Fixed ROCm-related runtime errors.
- **Code optimization** — General code cleanup and performance improvements.
- **Cursor position fix when resolution changes** — Resolved cursor offset issue during dynamic resolution changes.
- **Default depth resolution fix** — Corrected the default depth resolution value.
- **ROCM path configuration** — Fixed ROCm environment path detection.
- **README corrections** — Updated documentation and installation instructions.

---

### XR Viewer Keyboard Shortcuts (v2.5.0 Beta)

Based on raw code analysis of `xr_viewer/implementation.py`, `xr_viewer/constants.py`, and `xr_viewer/overlay.py`:

**Controller Action Bindings**  
All bindings come from OpenXR standard action paths. The code defines them at `implementation.py:1352-1402` and binds them per-device at `implementation.py:1441-1521`.

**Shortcuts (mapped to virtual help panel)**  
Defined in `overlay.py:140-174` as the 5‑row help panel layout (`ROWS` from utils):

| # | Action | Trigger | Code Location |
|---|--------|---------|---------------|
| 1 | Toggle FPS/Performance Panel | Left Menu button (short press) | `impl.py:4953`, `impl.py:5660-5668` |
| 2 | Toggle Virtual Keyboard | Left X button (< 1s press) | `impl.py:5735`, `impl.py:5758` |
| 3 | Toggle Cinema Glow / Lighting Preset | Left X button (1–4s hold, then release) | `impl.py:5754-5755` |
| 4 | Toggle Green Passthrough Backdrop | Left X button (> 4s hold) | `impl.py:5743-5745` |
| 5 | Reset Screen to Default | Left Y button (short press) | `impl.py:5709`, `impl.py:5726` |
| 6 | Cycle Screen Presets | Left Y button (long press ≥ 0.6s) | `impl.py:5713-5721` |
| 7 | Cycle Environment | Left Thumbstick Click (short) | `impl.py:5829` |
| 8 | Disable/Restore Depth Strength | Left Thumbstick Click (short, while Right Grip held) | `impl.py:5822-5827` |
| 9 | Toggle Curved/Flat Screen | Right Thumbstick Click (short) | `impl.py:5844` |
| 10 | Reset Screen Direction (face head) | Right Thumbstick Click (long ≥ 1s) | `impl.py:5837-5838` |
| 11 | Restore Depth Ratio (2.0) | Right Thumbstick Click (short, while Right Grip held) | `impl.py:5845-5849` |
| 12 | Hide All (Reset View) | Right A + B double‑press | `impl.py:5674`, `impl.py:5685-5686` |
| 13 | Right Mouse Click | Right B button (laser on screen) | `impl.py:5698-5699` |
| 14 | Left Mouse Click | Right A button (laser on screen) | `impl.py:5695-5696` |
| 15 | Left Mouse Click (Trigger) | Left Trigger (pull = click, hold = drag) | `impl.py:4445` |
| 16 | Right Mouse Click (Trigger) | Right Trigger (pull = click, hold = drag) | `impl.py:4445` |
| 17 | Toggle Shortcut Help Panel | Trigger while laser on FPS overlay panel | `impl.py:4515-4516` |
| 18 | Switch Controller Brand | Right A + B held ≥ 0.5s | `impl.py:4992-5009` |
| 19 | Adjust Depth Intensity | Right Grip + Left Stick Y | `impl.py:5521-5530` |
| 20 | Move Screen (Pan X/Y) | Left Grip + Left Stick X/Y | `impl.py:5491-5520` |
| 21 | Rotate Screen (Yaw) | Left Grip + Right Stick X | `impl.py:5620-5623` |
| 22 | Rotate Screen (Pitch) | Left Grip + Right Stick Y | `impl.py:5624-5627` |
| 23 | Resize Screen Width | Right Grip + Right Stick X | `impl.py:5579-5584` |
| 24 | Move Screen Closer/Farther | Right Grip + Right Stick Y | `impl.py:5586-5607` |
| 25 | Mouse Wheel Scroll | Left Stick X/Y (no grip) | `impl.py:5533` |
| 26 | Mouse Wheel Scroll | Right Stick X/Y (no grip, no stick click) | `impl.py:5644-5646` |
| 27 | Both‑Hands Screen Move | Both Grips + both lasers on screen | `impl.py:5321-5390` |
| 28 | Enter Seat Adjust Mode | Both Grips held ≥ 3s (when screen locked by env) | `impl.py:5041-5047` |
| 29 | Exit Seat Adjust | Both Grips released (in seat adjust mode) | `impl.py:5043-5045` |
| 30 | Keyboard Pan (orbit) | Left Grip + Left Stick (keyboard visible) | `impl.py:5457-5489` |
| 31 | Keyboard Resize Width | Right Grip + Right Stick X (keyboard visible) | `impl.py:5545-5548` |
| 32 | Keyboard Distance | Right Grip + Right Stick Y (keyboard visible) | `impl.py:5550-5578` |
| 33 | Keyboard Yaw Offset | Left Grip + Right Stick X (keyboard visible) | `impl.py:5613-5615` |
| 34 | Keyboard Pitch Offset | Left Grip + Right Stick Y (keyboard visible) | `impl.py:5616-5618` |
| 35 | Enter Calibration Mode | Left Menu + Right A + Right B held ≥ 1s | `impl.py:4957-4971` |
| 36 | Save Calibration (Y offset) | Left Stick Y in calib mode | `impl.py:4977-4978` |
| 37 | Save Calibration (Z offset) | Right Stick Y in calib mode | `impl.py:4979-4980` |
| 38 | Save Calibration (Rotation) | Right Stick X in calib mode | `impl.py:4981-4982` |
| 39 | Exit Calibration (Save) | Right B button in calib mode | `impl.py:4983-4988` |
| 40 | Exit Calibration (No Save) | Left Menu + Right A + Right B combo again | `impl.py:4963-4964` |

**Virtual Keyboard Modifier Keys**  
When the keyboard is visible, trigger on modifier keys toggles state (`impl.py:4838-4861`):

| Key | Behavior |
|-----|----------|
| Shift | Tap = one‑shot hold; Double‑tap = toggle lock; Tap while locked = unlock |
| Ctrl | Tap = one‑shot hold; Double‑tap = toggle lock; Tap while locked = unlock |
| Alt | Tap = one‑shot hold; Double‑tap = toggle lock; Tap while locked = unlock |
| Win | Tap = one‑shot hold; Double‑tap = toggle lock; Tap while locked = unlock |
| Caps Lock | Tap = toggle caps lock |
| Tab | One‑shot Tab key (0x09) |
| Backspace | One‑shot Backspace key (0x08) |
| Enter | One‑shot Enter key (0x0D) |
| Arrow Keys | One‑shot directional keys |
| Number/Symbol keys | Hold trigger = key‑down; Release = key‑up; Shifted labels shown when Shift/Caps active |

**Vive Trackpad Emulation**  
When using Vive/WMR trackpad controllers, trackpad regions emulate buttons (`constants.py:56-59`, `impl.py:3849`):

- Trackpad top region → Y/X button  
- Trackpad bottom region → A/B button  
- Trackpad center → Left/Right Stick Click  

---

### Environment Profiles

Each subfolder under `environments/` represents an XR backdrop environment in Desktop2Stereo. A folder can be a 3D GLB environment or a 360-degree panorama image environment. GLB rooms use `environment.glb` (or the `glb` field in `profile.json`); image environments can simply contain a panorama named `background.png`, `panorama.jpg`, `equirectangular.webp`, `360.png`, etc., or use `profile.json` for explicit settings. In the GUI, image backgrounds are listed before GLB rooms.

#### Quick Start

```
environments/
├── MyPanorama/
│   └── background.png      # image environment: no GLB / JSON required
├── Space/
│   ├── background.png
│   └── profile.json        # optional: panorama settings, display name, glow, etc.
├── MyRoom/
│   ├── environment.glb     # 3D environment
│   └── profile.json        # optional
├── Cinema/
│   ├── environment.glb
│   └── profile.json
└── ...
```

The simplest image environment only needs an image file. If `profile.json` is absent, the app uses the folder name as the label and treats the image as an equirectangular/360 panorama. A minimal GLB profile needs **no** fields — all keys are optional. If `profile.json` is absent or empty, the app uses sensible defaults and you position everything interactively in VR.

---

#### `profile.json` Reference

All fields are optional. Keys marked with `/` have backward-compatible aliases (older name still works).

##### Display Name

```json
"display_name": { "EN": "My Room", "CN": "我的房间" }
```

Localized label shown in the GUI dropdown. Falls back to the folder name if absent.

##### 3D Model

| Key | Type | Default | Description |
|---|---|---|---|
| `glb` | string | `"environment.glb"` | Path to the glTF model, relative to the profile folder. |
| `model_position` / `position` | [x, y, z] | `[0, 0, 0]` | World-space offset in metres. |
| `model_rotation_deg` / `rotation_deg` | [yaw, pitch, roll] | `[0, 0, 0]` | Rotation in **degrees**. Yaw = Y-axis, pitch = X-axis, roll = Z-axis. |
| `model_scale` / `scale` | [x, y, z] | `[1, 1, 1]` | Non-uniform scale factors. |

##### Panorama Image Background

Without `profile.json`, the first matching image in the folder is automatically detected as a 360-degree panorama background. Preferred names include `background.*`, `panorama.*`, `equirectangular.*`, `360.*`, `sky.*`, and `skybox.*`; supported extensions include `.png`, `.jpg`, `.jpeg`, `.webp`, `.bmp`, `.tif`, and `.tiff`.

```json
{
  "display_name": { "EN": "Space" },
  "environment_type": "panorama",
  "background": {
    "type": "equirectangular",
    "image": "background.png",
    "exposure": 1.0,
    "yaw_offset_deg": 0.0
  }
}
```

| Key | Type | Default | Description |
|---|---|---|---|
| `environment_type` / `type` | string | auto-detected | Set to `"panorama"` to explicitly mark an image environment. |
| `background.image` / `panorama.image` | string | auto-detected | Panorama image path, relative to the environment folder. |
| `background.type` / `background.projection` | string | `"equirectangular"` | Panorama projection type. Currently used for 360 equirectangular images. |
| `background.exposure` | float | `1.0` | Brightness multiplier for the background image. |
| `background.yaw_offset_deg` | float | `0.0` | Horizontal rotation offset in degrees. |
| `background.flip_y` | bool | `false` | Flip image sampling vertically. |

##### Lighting

| Key | Type | Default | Description |
|---|---|---|---|
| `env_exposure` | float | `1.0` | HDR exposure multiplier. Higher = brighter. Typical range: 0.3–1.5. |
| `env_gamma` | float | `2.2` | Gamma correction for tone-mapping. |
| `env_emissive_strength` | float | `1.0` | Multiplier for emissive materials in the GLB. Cinema screens use 5–10. |
| `env_khr_light_scale` / `khr_light_scale` | float | `1.0` | Multiplier for glTF `KHR_lights_punctual` intensity. |
| `env_ambient_color` / `ambient_color` | [r, g, b] | `[0.08, 0.08, 0.09]` | RGB ambient light (0–1 per channel). |
| `env_head_light_color` / `head_light_color` | [r, g, b] | `[0.45, 0.45, 0.48]` | RGB head-mounted point light (follows the viewer). |
| `screen_light_intensity` | float | `3.5` | Multiplier for the cinema bias-light (virtual screen acting as area light). |

##### Fill Lights

```json
"env_fill_lights": [
  { "position": [0, 2, -1], "color": [0.8, 0.7, 0.5], "range": 5.0 }
]
```

Alias: `fallback_lights`. Point lights with soft range attenuation placed in world space. Up to 2 evaluated in the fragment shader (hardware limit).

##### Lighting Presets

```json
"lighting_presets": [
  {
    "name": "Day",
    "env_exposure": 0.5,
    "env_ambient_color": [0.06, 0.05, 0.05],
    "env_head_light_color": [0.45, 0.45, 0.48]
  },
  {
    "name": "Night",
    "env_exposure": 0.16,
    "env_ambient_color": [0.01, 0.01, 0.012],
    "env_head_light_color": [0.03, 0.03, 0.05]
  }
]
```

Cycle through saved lighting configurations with **long-press Y** (left controller) in VR. If absent, long-press Y cycles screen presets (unlocked environments) or does nothing.

##### Screen Layout

```json
"screen": {
  "width": 5.0,
  "position": [0.0, 1.5, -6.0],
  "rotation_deg": [0.0, 0.0, 0.0],
  "curve_axis": "none",
  "allow_curve": true
}
```

When a `screen` section is present, the environment is **locked** — the virtual screen snaps to the configured position and the user cannot freely move it with controllers. This is ideal for rooms where the screen should align with a modelled TV, projector wall, or cinema screen in the GLB.

| Key | Type | Default | Description |
|---|---|---|---|
| `width` / `screen_width` | float | — | Screen width in metres. |
| `position` / `screen_position` | [x, y, z] | — | World-space centre of the screen. |
| `rotation_deg` / `screen_rotation_deg` | [yaw, pitch, roll] | `[0, 0, 0]` | Screen orientation in **degrees**. |
| `curve_axis` | string | `"none"` | Initial curve mode: `"horizontal"`, `"vertical"`, or `"none"`. Legacy `curved: true` is read as `"horizontal"`. |
| `allow_curve` | bool | `true` | Whether the user can toggle curved mode. |

##### Viewer Seating (View Poses)

Single seat:

```json
"view_pose": {
  "x": 0.0,
  "y": 1.6,
  "z": -2.0,
  "angle": 0.0
}
```

Multiple seats:

```json
"view_pose_index": 0,
"view_poses": [
  { "name": "Center", "x": 0.0, "y": 1.6, "z": -2.0, "angle": 0.0 },
  { "name": "Left",   "x": -1.2, "y": 1.6, "z": -2.0, "angle": 8.0 },
  { "name": "Right",  "x": 1.2, "y": 1.6, "z": -2.0, "angle": -8.0 }
]
```

| Key | Type | Description |
|---|---|---|
| `view_pose` / `camera` | object | Single viewer position `{x, y, z, angle}`. Used when `view_poses` is absent. |
| `view_poses` | array | Named seat positions. Cycle with **long-press Y** in locked environments. |
| `view_pose_index` | int | Active seat index (0-based). |
| `name` | string | Label shown in the seat-switch OSD. |
| `x`, `y`, `z` | float | Viewer position in world-space metres. |
| `angle` | float | Viewer yaw in degrees. |

The per-seat `view_poses` list supports an optional `distance_width_ratio` field (float, default 0.6) that computes the viewer distance as `width × ratio` when `distance` is not explicitly set.

##### Controller Model Offset (per-environment)

```json
"controller_overrides": {
  "model_offset": [0.0, 0.0, 0.0],
  "model_rotation_deg": 0.0
}
```

Fine-tunes the VR controller model position/rotation for the specific environment. Rarely needed.

---

#### Complete Example

```json
{
  "display_name": { "EN": "Cinema", "CN": "影院" },
  "glb": "environment.glb",
  "model_position": [-68.65, -0.25, -30.59],
  "model_rotation_deg": [178.15, 0.0, 0.0],
  "model_scale": [1.0, 1.0, 1.0],
  "env_exposure": 0.5,
  "env_gamma": 2.2,
  "env_emissive_strength": 10.0,
  "env_khr_light_scale": 1.0,
  "env_ambient_color": [0.06, 0.05, 0.05],
  "env_head_light_color": [0.45, 0.45, 0.48],
  "screen_light_intensity": 3.5,
  "env_fill_lights": [],
  "lighting_presets": [
    {
      "name": "Day",
      "env_exposure": 0.5,
      "env_ambient_color": [0.06, 0.05, 0.05],
      "env_head_light_color": [0.45, 0.45, 0.48]
    },
    {
      "name": "Night",
      "env_exposure": 0.5,
      "env_ambient_color": [0.01, 0.01, 0.012],
      "env_head_light_color": [0.03, 0.03, 0.05]
    }
  ],
  "view_pose": { "x": 0.12, "y": 5.89, "z": -0.48, "angle": 0.0 },
  "view_pose_index": 2,
  "view_poses": [
    { "name": "Center",    "x": 0.12, "y": 5.89, "z": -0.48, "angle": 0.0 },
    { "name": "Front Row", "x": 0.12, "y": 4.07, "z": -1.03, "angle": 0.0 },
    { "name": "Back Row",  "x": 0.12, "y": 7.47, "z": 0.17,  "angle": 0.0 }
  ],
  "screen": {
    "width": 5.0,
    "position": [0.08, 1.54, -5.89],
    "rotation_deg": [-1.85, 0.0, 0.0],
    "curve_axis": "none",
    "allow_curve": true
  }
}
```

#### Tips

- **Locked environments** (those with a `screen` section) prevent the user from moving the screen. The screen is parented to the environment model — moving the model moves the screen.
- **View poses** let you pre‑configure seating positions. Users cycle through them with long‑press Y.
- **Lighting presets** are independent of view poses. Both cycle with long‑press Y — view poses take priority when available.
- **Image environments** do not load a GLB and do not generate the dark-room model; the panorama is rendered as the distant background while the virtual screen keeps the default environment's movable/resizable behavior.
- **XR visual state** (default screen layout, curve axis, and glow mode) is saved to environment JSON, not `settings.yaml`.
- Model coordinates use the glTF convention: Y‑up, Z‑forward, right‑handed. The app auto‑converts to the XR reference space.
- To test profile changes, restart the OpenXR session — profiles are read once at load time.

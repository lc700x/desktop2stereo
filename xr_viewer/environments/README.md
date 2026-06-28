# Environment Profiles

Each subfolder under `environments/` represents a 3D environment that can be loaded as a virtual backdrop in Desktop2Stereo. A folder must contain an `environment.glb` (glTF 2.0 binary) and may include a `profile.json` to configure lighting, screen placement, and viewer seating.

## Quick Start

```
environments/
├── MyRoom/
│   ├── environment.glb     # required
│   └── profile.json        # optional
├── Cinema/
│   ├── environment.glb
│   └── profile.json
└── ...
```

A minimal profile needs **no** fields — all keys are optional. If `profile.json` is absent or empty, the app uses sensible defaults and you position everything interactively in VR.

---

## `profile.json` Reference

All fields are optional. Keys marked with `/` have backward-compatible aliases (older name still works).

### Display Name

```json
"display_name": { "EN": "My Room", "CN": "我的房间" }
```

Localized label shown in the GUI dropdown. Falls back to the folder name if absent.

### 3D Model

| Key | Type | Default | Description |
|---|---|---|---|
| `glb` | string | `"environment.glb"` | Path to the glTF model, relative to the profile folder. |
| `model_position` / `position` | [x, y, z] | `[0, 0, 0]` | World-space offset in metres. |
| `model_rotation_deg` / `rotation_deg` | [yaw, pitch, roll] | `[0, 0, 0]` | Rotation in **degrees**. Yaw = Y-axis, pitch = X-axis, roll = Z-axis. |
| `model_scale` / `scale` | [x, y, z] | `[1, 1, 1]` | Non-uniform scale factors. |

### Lighting

| Key | Type | Default | Description |
|---|---|---|---|
| `env_exposure` | float | `1.0` | HDR exposure multiplier. Higher = brighter. Typical range: 0.3–1.5. |
| `env_gamma` | float | `2.2` | Gamma correction for tone-mapping. |
| `env_emissive_strength` | float | `1.0` | Multiplier for emissive materials in the GLB. Cinema screens use 5–10. |
| `env_khr_light_scale` / `khr_light_scale` | float | `1.0` | Multiplier for glTF `KHR_lights_punctual` intensity. |
| `env_ambient_color` / `ambient_color` | [r, g, b] | `[0.08, 0.08, 0.09]` | RGB ambient light (0–1 per channel). |
| `env_head_light_color` / `head_light_color` | [r, g, b] | `[0.45, 0.45, 0.48]` | RGB head-mounted point light (follows the viewer). |
| `screen_light_intensity` | float | `3.5` | Multiplier for the cinema bias-light (virtual screen acting as area light). |

### Fill Lights

```json
"env_fill_lights": [
  { "position": [0, 2, -1], "color": [0.8, 0.7, 0.5], "range": 5.0 }
]
```

Alias: `fallback_lights`. Point lights with soft range attenuation placed in world space. Up to 2 evaluated in the fragment shader (hardware limit).

### Lighting Presets

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

### Screen Layout

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
| `curve_axis` | string | `"none"` | Initial curve mode: `"horizontal"`, `"vertical"`, or `"none"`. Older `curved: true` profiles are still read as `"horizontal"`; `curved: false` is read as `"none"`. |
| `allow_curve` | bool | `true` | Whether the user can toggle curved mode. |

### Viewer Seating (View Poses)

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

### Controller Model Offset (per-environment)

```json
"controller_overrides": {
  "model_offset": [0.0, 0.0, 0.0],
  "model_rotation_deg": 0.0
}
```

Fine-tunes the VR controller model position/rotation for the specific environment. Rarely needed.

---

## Complete Example

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

---

# 环境配置文件说明

`environments/` 下的每个子文件夹代表一个可在 Desktop2Stereo 中加载的 3D 虚拟环境。文件夹必须包含 `environment.glb`（glTF 2.0 二进制格式），可选包含 `profile.json` 来配置光照、屏幕位置和观众座位。

## 快速开始

```
environments/
├── MyRoom/
│   ├── environment.glb     # 必需
│   └── profile.json        # 可选
├── Cinema/
│   ├── environment.glb
│   └── profile.json
└── ...
```

最简单的配置文件可以**没有任何字段**——所有键都是可选的。如果 `profile.json` 不存在或为空，应用将使用合理默认值，你可以在 VR 中交互式调整。

---

## `profile.json` 参考

所有字段均为可选。标有 `/` 的键表示向后兼容的别名（旧名称仍有效）。

### 显示名称

```json
"display_name": { "EN": "My Room", "CN": "我的房间" }
```

GUI 下拉菜单中显示的本地化标签。若缺失则使用文件夹名称。

### 3D 模型

| 键 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `glb` | string | `"environment.glb"` | glTF 模型路径，相对于配置文件所在目录。 |
| `model_position` / `position` | [x, y, z] | `[0, 0, 0]` | 世界空间偏移量，单位为米。 |
| `model_rotation_deg` / `rotation_deg` | [yaw, pitch, roll] | `[0, 0, 0]` | 旋转角度，单位为**度**。yaw=Y轴，pitch=X轴，roll=Z轴。 |
| `model_scale` / `scale` | [x, y, z] | `[1, 1, 1]` | 非均匀缩放系数。 |

### 光照

| 键 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `env_exposure` | float | `1.0` | HDR 曝光倍数。越高越亮。典型范围：0.3–1.5。 |
| `env_gamma` | float | `2.2` | 色调映射的伽马校正。 |
| `env_emissive_strength` | float | `1.0` | GLB 中自发光材质的强度倍数。影院屏幕可用 5–10。 |
| `env_khr_light_scale` / `khr_light_scale` | float | `1.0` | glTF `KHR_lights_punctual` 灯光强度倍数。 |
| `env_ambient_color` / `ambient_color` | [r, g, b] | `[0.08, 0.08, 0.09]` | RGB 环境光颜色（每通道 0–1）。 |
| `env_head_light_color` / `head_light_color` | [r, g, b] | `[0.45, 0.45, 0.48]` | RGB 头戴式点光源颜色（跟随观看者）。 |
| `screen_light_intensity` | float | `3.5` | 影院偏置光倍数（虚拟屏幕作为面光源）。 |

### 补光

```json
"env_fill_lights": [
  { "position": [0, 2, -1], "color": [0.8, 0.7, 0.5], "range": 5.0 }
]
```

别名：`fallback_lights`。在世界空间中放置的带柔和距离衰减的点光源。片段着色器中最多评估 2 个（硬件限制）。

### 光照预设

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

### 屏幕布局

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
| `curve_axis` | string | `"none"` | 初始曲面模式：`"horizontal"`、`"vertical"` 或 `"none"`。旧的 `curved: true` 仍会读取为 `"horizontal"`；`curved: false` 会读取为 `"none"`。 |
| `allow_curve` | bool | `true` | 是否允许用户切换曲面模式。 |

### 观众座位（视角位姿）

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

### 手柄模型偏移（按环境）

```json
"controller_overrides": {
  "model_offset": [0.0, 0.0, 0.0],
  "model_rotation_deg": 0.0
}
```

针对特定环境微调 VR 手柄模型的位置/旋转。通常不需要。

---

## 完整示例

参见上方的 Cinema 配置文件。

## Tips

- **Locked environments** (those with a `screen` section) prevent the user from moving the screen. The screen is parented to the environment model — moving the model moves the screen.
- **View poses** let you pre-configure seating positions. Users cycle through them with long-press Y.
- **Lighting presets** are independent of view poses. Both cycle with long-press Y — view poses take priority when available.
- Model coordinates use the glTF convention: Y-up, Z-forward, right-handed. The app auto-converts to the XR reference space.
- To test profile changes, restart the OpenXR session — profiles are read once at load time.

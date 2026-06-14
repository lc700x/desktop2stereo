# Environment Profile Format

## 环境配置文件格式 / Environment Profile Format (中文)

`environment/<name>/` 下的每个环境可以包含一个 `profile.json`，用于控制房间模型变换、虚拟屏幕放置、灯光和后处理。
配置文件在启动和环境切换时加载。所有字段都是可选的——缺失的字段保持当前值不变。

## 顶层字段 / Top-Level Fields

| 字段 | 类型 | 默认值 | 说明 |
|-------|------|---------|------|
| `displayName` | string | — | 显示在 UI 中的可读名称 |
| `glb` | string | `"environment.glb"` | GLB 模型文件名（相对于此文件夹） |
| `model_position` | `[x, y, z]` | `[0, 0, 0]` | 房间模型的世界空间位置（米） |
| `model_rotation_deg` | `[yaw, pitch, roll]` | `[0, 0, 0]` | 房间模型的欧拉旋转，单位为**度**（yaw = Y轴, pitch = X轴, roll = Z轴） |
| `model_rotation_rad` | `[yaw, pitch, roll]` | — | 替代方案：弧度（如果存在 `model_rotation_deg` 则忽略） |
| `model_scale` | `[x, y, z]` | `[1, 1, 1]` | 应用到的房间模型缩放因子 |
| `lock_screen` | boolean | `false` | 为 `true` 时，虚拟屏幕保持在房间模型上（不会随用户交互漂移）。也接受在 `environment` 部分中用于向后兼容。 |
| `head_light_color` | `[r, g, b]` | `[0.45, 0.45, 0.48]` | 头灯（跟随相机）点光源颜色 |
| `ambient_color` | `[r, g, b]` | `[0.08, 0.08, 0.09]` | 环境（间接）光颜色 |
| `env_exposure` | float | `1.0` | HDR 累加后的全局曝光倍增器 |
| `env_gamma` | float | `2.2` | LDR 色调映射的伽马校正指数 |
| `env_emissive_strength` | float | `1.0` | 发光材质亮度的倍增器 |
| `env_khr_light_scale` | float | `1.0` | KHR_lights_punctual 平行光的缩放因子 |
| `env_render_quality` | string | `"balanced"` | 渲染质量提示（`"fast"`、`"balanced"`、`"high"`） |
| `env_texture_anisotropy` | float | `16.0` | 环境纹理的各向异性过滤级别 |
| `xr_render_scale` | float | `1.0` | 渲染分辨率缩放（0.5 – 1.0）。较低值提升性能 |
| `env_fill_lights` | array | `[]` | 观看者侧补光列表（见下文） |
| `environment` | object | — | 遗留/嵌套格式部分（保留用于向后兼容） |
| `view_pose` | object | — | 保存的查看者座位偏移（见下文） |
| `screen` | object | — | 虚拟屏幕配置（见下文） |

## `screen` 部分 / `screen` Section

控制虚拟屏幕的大小、位置和在空间中的方向。

| 字段 | 类型 | 默认值 | 说明 |
|-------|------|---------|------|
| `name` | string | — | 屏幕预设名称（例如 `"Default Projector"`） |
| `width` | float | `2.4` | 屏幕长边宽度（米） |
| `position` | `[x, y, z]` | `[0, 0, -2]` | 世界空间屏幕中心位置（米）。负 Z 表示"在观看者前方" |
| `rotation_deg` | `[yaw, pitch, roll]` | `[0, 0, 0]` | 屏幕方向，单位为**度** |
| `rotation_rad` | `[yaw, pitch, roll]` | — | 替代方案：弧度 |
| `curved` | boolean | `false` | 是否使用曲面渲染 |
| `allow_curve` | boolean | `false` | 用户是否可以通过控件切换曲面模式 |
| `ref_size` | float | `width` | 用于预设缩放的参考长边尺寸 |
| `screen_node_index` | int | `null` | 影院墙锁定的 glTF 节点索引（屏幕网格） |
| `screen_node_indices` | `[int, ...]` | `[]` | 替代方案：影院墙锁定的节点索引列表 |
| `offset` | `[x, y, z]` | `[0, 0, 0]` | 从屏幕网格中心的本地偏移 |

### 位置 ↔ 平移/距离转换 / Position ↔ Pan/Distance Translation

查看器内部将屏幕姿态存储为 `screen_pan_x`、`screen_pan_y`、`screen_distance`、
`screen_yaw`、`screen_pitch`、`screen_roll`。配置文件读取器进行以下转换：

```
pan_x     = position.x
pan_y     = position.y
distance  = -position.z
yaw       = rotation_deg.yaw  （转换为弧度）
pitch     = rotation_deg.pitch（转换为弧度）
roll      = rotation_deg.roll  （转换为弧度）
```

写入器在保存时反转此映射。

## `view_pose` 部分 / `view_pose` Section

相对于房间的保存的查看者座位偏移。当 `lock_screen` 为 `true` 时应用。

| 字段 | 类型 | 默认值 | 说明 |
|-------|------|---------|------|
| `auto_center_on_screen` | boolean | `true` | 自动将视图居中于屏幕上 |
| `position` | `[x, y, z]` | `[0, 0, 0]` | 查看者相对于房间原点的偏移（米） |
| `rotation_deg` | `[yaw, pitch, roll]` | `[0, 0, 0]` | 查看者方向偏移，单位为**度** |

## `env_fill_lights` 数组

观看者侧点光源，用于照亮房间模型。每项：

| 字段 | 类型 | 默认值 | 说明 |
|-------|------|---------|------|
| `position` | `[x, y, z]` | `[0, 0, 0]` | 灯光的世界空间位置（米） |
| `color` | `[r, g, b]` | `[0, 0, 0]` | 灯光颜色（每通道 0.0–1.0） |
| `range` | float | `1.0` | 衰减范围（米）。强度遵循 `1 / (1 + 4*(d/r)^2)` |

最多支持 2 个补光（索引 0 和 1）。

## 遗留 `environment` 部分（向后兼容）

嵌套的 `environment` 部分在保存时保留，以便与旧版查看器兼容。它使用**弧度**表示旋转，不使用度转换：

| 字段 | 类型 | 说明 |
|-------|------|------|
| `pos` | `[x, y, z]` | 模型位置 |
| `yaw`, `pitch`, `roll` | float | 旋转角度，**弧度** |
| `scale` | `[x, y, z]` | 模型缩放 |
| `lock_screen` | boolean | 屏幕-模型锁定标志 |

## 向后兼容

使用旧格式的配置文件（没有 `model_position`，只有 `environment` 部分，包含以弧度为单位的 `screen.pan_x`、`screen.distance`、`screen.yaw`）仍然可以工作。读取器自动检测格式：

- **新格式**：存在顶层 `model_position` 键
- **旧格式**：存在 `environment` 部分，但没有 `model_position`

所有五个内置配置文件（`Bedroom`、`Cinema`、`Default`、`Passthrough`、`Dark Room`）
都包含两种格式，因此适用于任何查看器版本。

---

Each environment under `environment/<name>/` can contain a `profile.json` that controls
the room model transform, virtual screen placement, lighting, and post-processing.
Profiles are loaded at startup and on environment switch. All fields are optional — missing
keys retain their current values.

## Top-Level Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `displayName` | string | — | Human-readable name shown in the UI |
| `glb` | string | `"environment.glb"` | Filename of the GLB model (relative to this folder) |
| `model_position` | `[x, y, z]` | `[0, 0, 0]` | World-space position of the room model (metres) |
| `model_rotation_deg` | `[yaw, pitch, roll]` | `[0, 0, 0]` | Euler rotation of the room model in **degrees** (yaw = Y-axis, pitch = X-axis, roll = Z-axis) |
| `model_rotation_rad` | `[yaw, pitch, roll]` | — | Alternative: radians (ignored if `model_rotation_deg` present) |
| `model_scale` | `[x, y, z]` | `[1, 1, 1]` | Scale factor applied to the room model |
| `lock_screen` | boolean | `false` | When `true`, the virtual screen stays locked to the room model (does not drift with user interaction). Also accepted inside the `environment` section for backward compatibility. |
| `head_light_color` | `[r, g, b]` | `[0.45, 0.45, 0.48]` | Head-lamp (camera-following) point light colour |
| `ambient_color` | `[r, g, b]` | `[0.08, 0.08, 0.09]` | Ambient (indirect) light colour |
| `env_exposure` | float | `1.0` | Global exposure multiplier applied after HDR accumulation |
| `env_gamma` | float | `2.2` | Gamma correction exponent for LDR tonemapping |
| `env_emissive_strength` | float | `1.0` | Multiplier on emissive material brightness |
| `env_khr_light_scale` | float | `1.0` | Scale factor for KHR_lights_punctual directional lights |
| `env_render_quality` | string | `"balanced"` | Rendering quality hint (`"fast"`, `"balanced"`, `"high"`) |
| `env_texture_anisotropy` | float | `16.0` | Anisotropic filtering level for environment textures |
| `xr_render_scale` | float | `1.0` | Render resolution scale (0.5 – 1.0). Lower values improve performance |
| `env_fill_lights` | array | `[]` | List of viewer-side fill lights (see below) |
| `environment` | object | — | Legacy/nested format section (kept for backward compatibility) |
| `view_pose` | object | — | Saved viewer-seat offset (see below) |
| `screen` | object | — | Virtual screen configuration (see below) |

## `screen` Section

Controls the virtual screen's size, position, and orientation in world space.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | string | — | Screen preset name (e.g. `"Default Projector"`) |
| `width` | float | `2.4` | Screen long-edge width (metres) |
| `position` | `[x, y, z]` | `[0, 0, -2]` | World-space screen centre position (metres). Z is negative for "in front of viewer" |
| `rotation_deg` | `[yaw, pitch, roll]` | `[0, 0, 0]` | Screen orientation in **degrees** |
| `rotation_rad` | `[yaw, pitch, roll]` | — | Alternative: radians |
| `curved` | boolean | `false` | Whether the screen uses curved rendering |
| `allow_curve` | boolean | `false` | Whether the user can toggle curved mode via controls |
| `ref_size` | float | `width` | Reference long-edge size used for preset scaling |
| `screen_node_index` | int | `null` | glTF node index for Cinema wall-lock (screen meshes) |
| `screen_node_indices` | `[int, ...]` | `[]` | Alternative: list of node indices for Cinema wall-lock |
| `offset` | `[x, y, z]` | `[0, 0, 0]` | Local offset from the screen mesh centre |

### Position ↔ Pan/Distance Translation

Internally the viewer stores screen pose as `screen_pan_x`, `screen_pan_y`, `screen_distance`,
`screen_yaw`, `screen_pitch`, `screen_roll`. The profile reader translates:

```
pan_x     = position.x
pan_y     = position.y
distance  = -position.z
yaw       = rotation_deg.yaw  (converted to radians)
pitch     = rotation_deg.pitch (converted to radians)
roll      = rotation_deg.roll  (converted to radians)
```

The writer reverses this mapping when saving.

## `view_pose` Section

Saved viewer-seat offset relative to the room. Applied when `lock_screen` is `true`.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `auto_center_on_screen` | boolean | `true` | Automatically centre the view on the screen |
| `position` | `[x, y, z]` | `[0, 0, 0]` | Viewer offset from room origin (metres) |
| `rotation_deg` | `[yaw, pitch, roll]` | `[0, 0, 0]` | Viewer orientation offset in **degrees** |

## `env_fill_lights` Array

Viewer-side point lights that illuminate the room model. Each entry:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `position` | `[x, y, z]` | `[0, 0, 0]` | Light world-space position (metres) |
| `color` | `[r, g, b]` | `[0, 0, 0]` | Light colour (0.0–1.0 per channel) |
| `range` | float | `1.0` | Attenuation range in metres. Intensity follows `1 / (1 + 4*(d/r)^2)` |

Up to 2 fill lights are supported (index 0 and 1).

## Legacy `environment` Section (Backward Compatibility)

The nested `environment` section is retained when saving for compatibility with older
viewer versions. It uses **radians** for rotation and **direct** values (no degree conversion):

| Field | Type | Description |
|-------|------|-------------|
| `pos` | `[x, y, z]` | Model position |
| `yaw`, `pitch`, `roll` | float | Rotation in **radians** |
| `scale` | `[x, y, z]` | Model scale |
| `lock_screen` | boolean | Screen-model lock flag |

## Backward Compatibility

Profiles using the old format (no `model_position`, only `environment` section with
`screen.pan_x`, `screen.distance`, `screen.yaw` in radians) continue to work. The reader
detects the format automatically:

- **New format**: top-level `model_position` key present
- **Old format**: `environment` section present without `model_position`

All five built-in profiles (`Bedroom`, `Cinema`, `Default`, `Passthrough`, `Dark Room`)
include both formats so they work with any viewer version.

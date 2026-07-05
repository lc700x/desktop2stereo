"""
Desktop2Stereo Flet GUI
Flet desktop app, migrated from tkinter gui.py.
Feature-equivalent; all interaction logic matches the original.

Custom control overview:

CompactTextField (L535)
  Compact text input, 32px height. Displays as bordered text label by default;
  switches to TextField on click for editing.
  Constructor params:
    value      — initial value
    width      — control width
    read_only  — read-only flag
    on_change  — value change callback (param: e.control.value)
    tooltip    — tooltip text
    filter     — regex to restrict allowed characters (e.g. r"[0-9]" for digits only)
    max_length — max input length
  On submit (Enter/Blur), non-matching characters are stripped.
  Properties: .value (r/w), .set_tooltip(text)

CompactDropdown (L615)
  Custom dropdown based on PopupMenuButton, 32px height, visually consistent with TextField.
  Constructor params:
    options    — list of options
    value      — currently selected value
    on_select  — selection callback (param: e.control.value)
    expand     — expand_loose flag
    dyna_width — dynamic width mode (uses LABEL_ALIGN_WIDTH)
    width      — fixed width
    min_width  — auto-width lower bound
    max_width  — auto-width upper bound
    tooltip    — tooltip text
  Width priority: fixed > dynamic mode > auto-calculated (bounded by min/max).
  Properties: .value (r/w), .options (r/w, triggers menu rebuild), .set_tooltip(text)
"""
import os
FLET_STORAGE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "flet_client"))
os.environ["FLET_VIEW_PATH"] = FLET_STORAGE_PATH
import sys
import subprocess
import time
import asyncio
import ctypes
import re
import json
import atexit
import traceback

# Force UTF-8 stdout/stderr on Windows so the console (and downstream
# TeeStream that mirrors writes to it) does not mangle messages
# containing em-dashes, arrows, or other non-ASCII characters.  Must
# run BEFORE _setup_console_logging() captures the streams.
try:
    if sys.stdout and sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if sys.stderr and sys.stderr.encoding and sys.stderr.encoding.lower() != "utf-8":
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import flet as ft
import yaml
from utils import (
    VERSION, OS_NAME, ALL_MODELS, DEFAULT_PORT, STEREO_MIX_NAMES,
    DISABLE_TRT_KEYWORDS, DISABLE_COREML_KEYWORDS, DISABLE_OPENVINO_KEYWORDS,
    DISABLE_MIGRAPHX_KEYWORDS,
    get_local_ip, shutdown_event, read_yaml
)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(BASE_DIR, "logs")
LOG_FILE = os.path.join(LOG_DIR, "desktop2stereo.log")
STOP_REQUEST_FILE = os.path.join(LOG_DIR, "stop.request")

_ENV_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff")
_ENV_IMAGE_NAMES = ("background", "panorama", "equirectangular", "360", "sky", "skybox")


def _read_env_profile_for_gui(profile_path):
    if not os.path.isfile(profile_path):
        return {}
    try:
        with open(profile_path, "r", encoding="utf-8-sig") as f:
            loaded = json.load(f)
        return loaded if isinstance(loaded, dict) else {}
    except Exception:
        return {}


def _find_env_image_for_gui(room_dir):
    if not os.path.isdir(room_dir):
        return None
    for stem in _ENV_IMAGE_NAMES:
        for ext in _ENV_IMAGE_EXTS:
            path = os.path.join(room_dir, stem + ext)
            if os.path.isfile(path):
                return path
    try:
        for name in sorted(os.listdir(room_dir), key=lambda v: v.lower()):
            path = os.path.join(room_dir, name)
            if os.path.isfile(path) and os.path.splitext(name)[1].lower() in _ENV_IMAGE_EXTS:
                return path
    except OSError:
        pass
    return None


def _is_panorama_profile_for_gui(profile):
    if not isinstance(profile, dict):
        return False
    raw_bg = profile.get("background")
    raw_panorama = profile.get("panorama")
    cfg = {}
    if isinstance(raw_bg, str):
        cfg["image"] = raw_bg
    elif isinstance(raw_bg, dict):
        cfg.update(raw_bg)
    if isinstance(raw_panorama, str):
        cfg.setdefault("image", raw_panorama)
    elif isinstance(raw_panorama, dict):
        cfg.update(raw_panorama)
    env_type = str(profile.get("environment_type", profile.get("type", "")) or "").strip().lower()
    bg_type = str(cfg.get("type", cfg.get("kind", "")) or "").strip().lower()
    projection = str(cfg.get("projection", cfg.get("format", "")) or "").strip().lower()
    return (
        env_type in ("panorama", "360", "360_photo", "360-photo", "photo_sphere", "photosphere")
        or bg_type in ("panorama", "360", "360_photo", "360-photo", "equirectangular", "photo_sphere", "photosphere")
        or projection in ("equirectangular", "360", "360_photo", "360-photo")
        or raw_panorama is True
    )


def _discover_gui_environment_folders(env_base):
    panorama_dirs = []
    glb_dirs = []
    try:
        names = sorted(os.listdir(env_base), key=lambda v: v.lower())
    except (FileNotFoundError, OSError):
        return []
    for name in names:
        room_dir = os.path.join(env_base, name)
        if not os.path.isdir(room_dir):
            continue
        profile = _read_env_profile_for_gui(os.path.join(room_dir, "profile.json"))
        glb_path = os.path.join(room_dir, "environment.glb")
        if _is_panorama_profile_for_gui(profile) and _find_env_image_for_gui(room_dir):
            panorama_dirs.append(name)
            continue
        if not os.path.isfile(glb_path):
            glb_name = str(profile.get("glb", "environment.glb") or "environment.glb")
            glb_path = glb_name if os.path.isabs(glb_name) else os.path.join(room_dir, glb_name)
        if os.path.isfile(glb_path):
            glb_dirs.append(name)
            continue
        if _find_env_image_for_gui(room_dir):
            panorama_dirs.append(name)
    return panorama_dirs + glb_dirs

# Kept as an alias for backward compatibility with any external tooling
# that referenced the old diag log path.
DIAG_LOG = LOG_FILE

_CHILD_PID = None  # set when child starts, cleared on clean stop

@atexit.register
def _atexit_kill_child():
    pid = _CHILD_PID
    if pid is None:
        return
    try:
        if OS_NAME == "Windows":
            subprocess.run(
                ['taskkill', '/f', '/t', '/pid', str(pid)],
                capture_output=True, timeout=5,
            )
        else:
            import signal
            try:
                os.killpg(os.getpgid(pid), signal.SIGKILL)
            except Exception:
                pass
    except Exception:
        pass


def _setup_console_logging():
    """Mirror stdout/stderr to the single rolling log file.

    Design goals (per user request):
      * Console output is preserved unchanged — every ``print()`` still
        reaches the original terminal so the user can watch progress live.
      * Exactly ONE log file lives in ``logs/``.  Any leftover files from
        previous sessions (``child.log``, ``diag.log``, ``main_out.log`` …)
        are swept on startup so the folder never accumulates clutter.
      * Both GUI prints AND the child ``main.py`` process output land in the
        same file, so a single tail is enough to diagnose any run.
    """
    import datetime
    import threading

    os.makedirs(LOG_DIR, exist_ok=True)

    # --- Sweep stale log files so only the new single file remains. ---
    try:
        for _name in os.listdir(LOG_DIR):
            _path = os.path.join(LOG_DIR, _name)
            if os.path.isfile(_path) and os.path.abspath(_path) != os.path.abspath(LOG_FILE):
                try:
                    os.remove(_path)
                except Exception:
                    pass
    except Exception:
        pass

    # --- Truncate the single log file so each run starts fresh. ---
    try:
        with open(LOG_FILE, "w", encoding="utf-8") as _f:
            _f.write(f"=== Desktop2Stereo log started {datetime.datetime.now().isoformat(timespec='seconds')} ===\n")
    except Exception:
        pass

    _lock = threading.Lock()

    class _TeeStream:
        def __init__(self, original, label):
            self.original = original
            self.label = label

        def write(self, data):
            # 1) Always echo to the original terminal so the user sees prints live.
            try:
                self.original.write(data)
            except Exception:
                pass
            # 2) Append every non-empty line to the single log file with a
            #    timestamp + source label.  Lock to keep stdout/stderr
            #    interleaving readable when both fire from different threads.
            if not data:
                return
            try:
                ts = datetime.datetime.now().strftime("%H:%M:%S")
                with _lock:
                    with open(LOG_FILE, "a", encoding="utf-8") as f:
                        for line in data.splitlines():
                            stripped = line.rstrip()
                            if stripped:
                                f.write(f"[{ts}] [{self.label}] {stripped}\n")
            except Exception:
                pass

        def flush(self):
            try:
                self.original.flush()
            except Exception:
                pass

        def isatty(self):
            try:
                return self.original.isatty()
            except Exception:
                return False

        # Forward fileno when possible so libraries that probe the underlying
        # OS handle (e.g. subprocess) still work; if the wrapped stream has
        # no fileno (already wrapped), raise to fall back to PIPE handling.
        def fileno(self):
            return self.original.fileno()

    sys.stdout = _TeeStream(sys.stdout, "out")
    sys.stderr = _TeeStream(sys.stderr, "err")

# Known model size suffixes, longest-first for matching priority
_MODEL_SIZES = ["Small", "SmallPlus", "Base", "Large", "Giant"]
_SIZE_ORDER = {s: i for i, s in enumerate(_MODEL_SIZES)}

def parse_model_name(name):
    """Split model name into (family, size). DepthPro treated as Large."""
    parts = name.split("-")
    size_parts = []
    i = len(parts) - 1
    while i >= 0:
        matched = None
        for sz in _MODEL_SIZES:
            if parts[i].upper() == sz.upper():
                matched = sz
                break
        if matched:
            size_parts.insert(0, matched)
            i -= 1
        else:
            break
    if size_parts:
        family = "-".join(parts[:i + 1])
        size = "-".join(size_parts)
        return (family, size)
    return (name, "")

def build_family_size_map(model_list):
    """Returns (families_ordered, family_to_sizes) from list of full model names."""
    families = []
    family_to_sizes = {}
    for name in model_list:
        family, size = parse_model_name(name)
        if family not in family_to_sizes:
            family_to_sizes[family] = []
            families.append(family)
        if size and size not in family_to_sizes[family]:
            family_to_sizes[family].append(size)
    for family in family_to_sizes:
        family_to_sizes[family].sort(key=lambda s: _SIZE_ORDER.get(s, 99))
    return families, family_to_sizes


# ── Disable console Quick Edit Mode (prevents console freeze on click) ──
try:
    kernel32 = ctypes.windll.kernel32
    kernel32.GetStdHandle.restype = ctypes.c_void_p
    kernel32.GetStdHandle.argtypes = [ctypes.c_uint32]
    STD_INPUT_HANDLE = -10
    ENABLE_QUICK_EDIT_MODE = 0x0040
    ENABLE_EXTENDED_FLAGS = 0x0080
    hStdin = kernel32.GetStdHandle(STD_INPUT_HANDLE)
    mode = ctypes.c_uint32()
    if kernel32.GetConsoleMode(hStdin, ctypes.byref(mode)):
        mode.value = (mode.value & ~ENABLE_QUICK_EDIT_MODE) | ENABLE_EXTENDED_FLAGS
        kernel32.SetConsoleMode(hStdin, mode)
except Exception:
    pass  # Not a console or Windows — safe to ignore

# ─────────────────────────────────────────────
# UI Text Dictionary (EN / CN)
# ─────────────────────────────────────────────
UI_TEXTS = {
    "EN": {
        "Monitor": "Input Monitor",
        "Window": "Input Window",
        "Refresh": "Refresh",
        "Show FPS": "Show FPS",
        "IPD (m):": "IPD (mm):",
        "Convergence:": "Convergence:",
        "Display Mode:": "Display Mode:",
        "Depth Model:": "Depth Model:",
        "Depth Strength:": "Depth Strength:",
        "Depth Resolution:": "Depth Resolution:",
        "Anti-aliasing:": "Anti-aliasing:",
        "Foreground Scale:": "Foreground Scale:",
        "FP16": "FP16",
        "Inference Acceleration:": "Acceleration:",
        "Recompile TensorRT": "Recompile TensorRT",
        "Recompile CoreML": "Recompile CoreML",
        "Recompile OpenVINO": "Recompile OpenVINO",
        "Recompile MIGraphX": "Recompile MIGraphX",
        "Stop": "Stop",
        "Computing Device:": "Computing Device:",
        "Reset": "Reset",
        "Run": "Run",
        "Set Language:": "Set Language:",
        "Error": "Error",
        "Warning": "Warning",
        "Saved": "Run Desktop2Stereo",
        "PyYAML not installed, cannot save YAML file.": "PyYAML not installed, cannot save YAML file.",
        "Settings saved to settings.yaml": "Settings saved to settings.yaml",
        "Failed to save settings.yaml:": "Failed to save settings.yaml:",
        "Could not retrieve monitor list.\nFalling back to indexes 1 and 2.": "Could not retrieve monitor list.\nFalling back to indexes 1 and 2.",
        "Loaded settings.yaml at startup": "Loaded settings.yaml at startup",
        "Running": "Running... (Hold ESC 3s to Stop)",
        "Stopped": "Stopped.",
        "Countdown": "Settings saved to settings.yaml, starting...",
        "A thread already running!": "A thread already running!",
        "No windows found": "No windows found",
        "Selected input window:": "Selected input window:",
        "Selected input monitor:": "Selected input monitor:",
        "Run Mode:": "Run Mode:",
        "Local Viewer": "Local Viewer",
        "Legacy Streamer": "Legacy Streamer",
        "MJPEG Streamer": "MJPEG Streamer",
        "RTMP Streamer": "RTMP Streamer",
        "Stream Protocol:": "Stream Protocol:",
        "Stream Key": "Stream Key:",
        "Stereo Mix": "Stereo Mix:",
        "CRF": "CRF:",
        "Audio Delay": "Audio Delay (s):",
        "Lossless Scaling Support": "LSFG",
        "3D Monitor": "3D Monitor",
        "OpenXR Link": "OpenXR Link",
        "XR Preview": "XR Preview",
        "Crop Auto": "Auto",
        "Crop Manual": "Manual",
        "Crop Off": "Off",
        "VSync": "VSync",
        "Streamer Port:": "Streamer Port:",
        "Streamer URL": "Streamer URL:",
        "Preview":"Preview",
        "Stream Quality:": "Stream Quality:",
        "Host": "Host:",
        "Invalid port number (1-65535)": "Invalid port number (must be between 1-65535)",
        "Invalid port number": "Port must be a number",
        "Please select a window before running in Window capture mode": "Please select a window before running in Window capture mode",
        "The selected window no longer exists. Please refresh and select a valid window.": "The selected window no longer exists. Please refresh and select a valid window.",
        "Failed to stop process on exit:": "Failed to stop process on exit:",
        "Failed to stop process:": "Failed to stop process:",
        "Failed to run process:": "Failed to run process:",
        "Failed to load settings.yaml:": "Failed to load settings.yaml:",
        "Opening URL in browser": "Opening URL in browser",
        "Controller:": "Controller:",
        "Crop Mode:": "Crop Mode:",
        "Environment:": "Environment:",
        "Capture Tool:": "Capture Tool:",
        "Fill 16:9": "16:9",
        "Fix Viewer Aspect": "Fix Aspect",
        "Stereo Output:": "Stereo Output:",
        "Theme:": "Theme:",
        "DesktopDuplication selected: Window capture mode disabled.": "DesktopDuplication selected: Window capture mode disabled.",
        "torch.compile": "torch.compile",
        "TensorRT": "TensorRT",
        "CoreML": "CoreML",
        "OpenVINO": "OpenVINO",
        "MIGraphX": "MIGraphX",
        "tooltip_window": "Select a window to capture",
        "tooltip_depth_model": "Depth estimation model",
        "tooltip_model_size":"Model backbone size",
        "tooltip_depth_res": "Depth map resolution",
        "tooltip_convergence": "Stereo convergence",
        "tooltip_depth_strength": "Depth effect intensity",
        "tooltip_foreground_scale": "Foreground object scale",
        "tooltip_antialiasing": "Anti-aliasing level",
        "tooltip_ipd": "Interpupillary distance (mm)",
        "tooltip_device": "Inference device",
        "tooltip_capture_tool": "Capture backend",
        "tooltip_run_mode": "Output mode",
        "tooltip_display_mode": "Stereo display format",
        "tooltip_vsync": "Synchronize the local viewer to the display refresh rate",
        "tooltip_ctrl_model": "Controller model",
        "tooltip_env_model": "Background environment: Default (black), a 360 image folder, or a 3D scene from xr_viewer/environments/. Cinema glow can be toggled via long-press X in VR.",
        "tooltip_crop_mode": "Auto: detect letterbox bars. Manual: set your own crop. Off: no crop.\nManual mode (in VR, laser off screen): hold left trigger 3s to cycle mode, double-tap left trigger to start/stop adjusting, then push the left stick — X crops the sides, Y crops top/bottom (dominant axis only).",
        "Default": "Default",
        "tooltip_capture_mode": "Source: monitor or window",
        "tooltip_monitor": "Input monitor",
        "tooltip_stereo_monitor": "Stereo output monitor",
        "tooltip_lang": "Interface language",
        "tooltip_theme": "Color theme",
        "tooltip_stream_quality": "Encode quality",
        "tooltip_stream_proto": "Streaming protocol",
        "tooltip_audio": "Stereo mix device",
        "tooltip_stream_port": "Server port",
        "tooltip_stream_key": "Stream key",
        "tooltip_crf": "Quality factor (0-51)",
        "tooltip_audio_delay": "Audio offset (s)",
        "err_crf": "CRF must be between 0-51",
        "err_audio_delay": "Audio Delay must be between -10 and 10",
        "err_stream_key": "Stream Key can only contain letters, digits, underscore, hyphen, max 64 chars",
        "err_start_failed": "Start failed: {}",
        "esc_stop": "Hold ESC 3s — stopping!",
        "exited_with_code": "Exited with code {}",
        "failed_save_yaml": "Failed to save YAML: {}",
        "invalid_url_scheme": "Invalid URL scheme: {}",
        "error_preview": "Failed to preview: {}",
        "url_copied": "URL copied to clipboard",
        "Log Folder": "Log Folder",
        "tooltip_open_log": "Please send the log file for debugging.",
    },
    "CN": {
        "Monitor": "输入屏幕",
        "Window": "输入窗口",
        "Refresh": "刷新",
        "Show FPS": "显示帧率",
        "IPD (m):": "瞳距 (mm):",
        "Convergence:": "会聚点:",
        "Display Mode:": "显示模式:",
        "Depth Model:": "深度模型:",
        "Depth Strength:": "深度强度:",
        "Depth Resolution:": "深度分辨率:",
        "Anti-aliasing:": "抗锯齿:",
        "Foreground Scale:": "前景缩放:",
        "FP16": "FP16",
        "Inference Acceleration:": "推理加速:",
        "Recompile TensorRT": "重译TensorRT",
        "Recompile CoreML": "重译CoreML",
        "Recompile OpenVINO": "重译OpenVINO",
        "Recompile MIGraphX": "重译MIGraphX",
        "Stop": "停止",
        "Computing Device:": "计算设备:",
        "Reset": "重置",
        "Run": "运行",
        "Set Language:": "设置语言:",
        "Error": "错误",
        "Warning": "警告",
        "Saved": "运行Desktop2Stereo",
        "PyYAML not installed, cannot save YAML file.": "未安装PyYAML，无法保存YAML文件。",
        "Settings saved to settings.yaml": "设置已保存到 settings.yaml",
        "Failed to save settings.yaml:": "保存 settings.yaml 失败：",
        "Could not retrieve monitor list.\nFalling back to indexes 1 and 2.": "无法获取显示器列表。\n回退到索引1和2。",
        "Loaded settings.yaml at startup": "启动时已加载 settings.yaml",
        "Running": "运行中...（长按ESC 3秒停止）",
        "Stopped": "已停止。",
        "Countdown": "设置已保存到 settings.yaml，启动中...",
        "A thread already running!": "一个进程已经运行！",
        "No windows found": "未找到窗口",
        "Selected input window:": "已选择输入窗口:",
        "Selected input monitor:": "已选择输入显示器 :",
        "Run Mode:": "运行模式:",
        "Local Viewer": "本地查看",
        "Legacy Streamer": "旧网络推流",
        "MJPEG Streamer": "MJPEG推流",
        "RTMP Streamer": "RTMP推流",
        "Stream Protocol:": "流协议:",
        "Stream Key": "推流密钥:",
        "Stereo Mix": "混音设备:",
        "CRF": "恒定质量:",
        "Audio Delay": "音频延迟 (秒):",
        "system": "系统",
        "blue": "蓝色",
        "green": "绿色",
        "red": "红色",
        "purple": "紫色",
        "orange": "橙色",
        "teal": "青色",
        "pink": "粉色",
        "grey": "灰色",
        "Lossless Scaling Support": "小黄鸭",
        "3D Monitor": "3D显示器",
        "OpenXR Link": "OpenXR串流",
        "XR Preview": "XR预览",
        "Crop Auto": "自动",
        "Crop Manual": "手动",
        "Crop Off": "关闭",
        "VSync": "垂直同步",
        "Streamer Port:": "推流端口:",
        "Streamer URL": "推流网址:",
        "Preview": "预览",
        "Stream Quality:": "推流质量:",
        "Host": "主机:",
        "Invalid port number (1-65535)": "端口号无效 (必须介于1-65535之间)",
        "Invalid port number": "端口必须是数字",
        "Please select a window before running in Window capture mode": "请在窗口捕获模式下选择一个窗口再运行",
        "The selected window no longer exists. Please refresh and select a valid window.": "所选窗口已不存在。请刷新并选择一个有效的窗口。",
        "Failed to stop process on exit:": "退出时停止进程失败：",
        "Failed to stop process:": "停止进程失败：",
        "Failed to run process:": "运行进程失败：",
        "Failed to load settings.yaml:": "加载 settings.yaml 失败：",
        "Opening URL in browser": "正在浏览器中打开网址",
        "Controller:": "手柄模型：",
        "Crop Mode:": "裁剪模式：",
        "Environment:": "虚拟环境：",
        "Capture Tool:": "捕获工具:",
        "Fill 16:9": "16:9",
        "Fix Viewer Aspect": "锁定比例",
        "Stereo Output:": "立体输出:",
        "Theme:": "主题颜色:",
        "DesktopDuplication selected: Window capture mode disabled.": "已选择DesktopDuplication：窗口捕获模式已禁用。",
        "torch.compile": "torch.compile",
        "TensorRT": "TensorRT",
        "CoreML": "CoreML",
        "OpenVINO": "OpenVINO",
        "MIGraphX": "MIGraphX",
        "tooltip_window": "选择要捕获的窗口",
        "tooltip_depth_model": "选择深度估计模型",
        "tooltip_model_size":"模型骨架大小",
        "tooltip_depth_res": "深度图分辨率",
        "tooltip_convergence": "立体会聚点",
        "tooltip_depth_strength": "深度效果强度",
        "tooltip_foreground_scale": "前景缩放比例",
        "tooltip_antialiasing": "抗锯齿级别",
        "tooltip_ipd": "瞳距（毫米）",
        "tooltip_device": "计算设备",
        "tooltip_capture_tool": "捕获后端",
        "tooltip_run_mode": "输出模式",
        "tooltip_display_mode": "立体显示格式",
        "tooltip_vsync": "将本地查看窗口同步到显示器刷新率，关闭可用于帧率对比测试",
        "tooltip_ctrl_model": "手柄型号",
        "tooltip_env_model": "背景环境：默认（黑色）、360 图像文件夹，或 xr_viewer/environments/ 中的 3D 场景。影院辉光可在 VR 中长按 X 键切换。",
        "tooltip_crop_mode": "自动：检测黑边。手动：自定义裁剪。关闭：不裁剪。\n手动模式（VR 中，激光不指屏幕）：按住左扳机 3 秒循环模式，双击左扳机开始/停止调整，然后推动左摇杆——X 裁剪左右，Y 裁剪上下（仅主轴）。",
        "Default": "默认",
        "tooltip_capture_mode": "捕获源：屏幕或窗口",
        "tooltip_monitor": "输入显示器",
        "tooltip_stereo_monitor": "立体输出显示器",
        "tooltip_lang": "界面语言",
        "tooltip_theme": "主题颜色",
        "tooltip_stream_quality": "编码质量",
        "tooltip_stream_proto": "推流协议",
        "tooltip_audio": "混音设备",
        "tooltip_stream_port": "推流端口",
        "tooltip_stream_key": "推流密钥",
        "tooltip_crf": "质量因子 (0-51)",
        "tooltip_audio_delay": "音频偏移（秒）",
        "err_crf": "CRF 必须是 0-51 之间的整数",
        "err_audio_delay": "Audio Delay 必须是 -10 到 10 之间的数值",
        "err_stream_key": "Stream Key 只能包含字母、数字、下划线和连字符，最长 64 字符",
        "err_start_failed": "启动失败: {}",
        "esc_stop": "长按ESC 3秒停止",
        "exited_with_code": "退出码 {}",
        "failed_save_yaml": "保存 YAML 失败: {}",
        "invalid_url_scheme": "无效 URL 协议: {}",
        "error_preview": "打开浏览器失败: {}",
        "url_copied": "已复制网址到剪贴板",
        "Log Folder": "日志文件夹",
        "tooltip_open_log": "请将日志文件发送用于排查bug。",
    }
}
# ─────────────────────────────────────────────
# Default Configuration
# ─────────────────────────────────────────────
DEFAULT_MODEL_LIST = list(ALL_MODELS.keys())
DEFAULT_FAMILIES, FAMILY_TO_SIZES = build_family_size_map(DEFAULT_MODEL_LIST)
FAMILY_SIZE_TO_MODEL = {}
for name in DEFAULT_MODEL_LIST:
    f, s = parse_model_name(name)
    FAMILY_SIZE_TO_MODEL[(f, s)] = name
DEFAULTS = {
    "Capture Mode": "Monitor",
    "Monitor Index": 1,
    "Window Title": "",
    "Show FPS": False,
    "Model List": DEFAULT_MODEL_LIST,
    "Depth Model": DEFAULT_MODEL_LIST[0] if DEFAULT_MODEL_LIST else "",
    "Depth Strength": 4.0,
    "Depth Resolution": 322,
    "Anti-aliasing": 2,
    "Foreground Scale": 0.5,
    "IPD": 0.064,
    "Convergence": 0.0,
    "Display Mode": "Half-SBS",
    "FP16": True,
    "torch.compile": None,
    "TensorRT": None,
    "Recompile TensorRT": False,
    "CoreML": None,
    "Recompile CoreML": False,
    "OpenVINO": None,
    "Recompile OpenVINO": False,
    "MIGraphX": None,
    "Recompile MIGraphX": False,
    "Computing Device": 0,
    "Language": "EN",
    "Run Mode": "OpenXR Link",
    "XR Preview": False,
    "Crop Mode": "manual",
    "VSync": False,
    "Stream Protocol": "HLS",
    "Streamer Port": DEFAULT_PORT,
    "Stream Quality": 100,
    "Stream Key": "live",
    "Stereo Mix": None,
    "CRF": 20,
    "Audio Delay": -0.15,
    "Controller Model": "PICO",
    "Environment Model": "Default",
    "Lossless Scaling Support": False,
    "Capture Tool": "none",
    "Fill 16:9": True,
    "Fix Viewer Aspect": False,
    "Stereo Output": None,
}


# ─────────────────────────────────────────────
# Global Font Size
# ─────────────────────────────────────────────
SCALE = 0.9  # Global UI scale factor (fonts, spacing, widget + window sizes)


def S(v):
    """Scale a dimension literal by the global UI SCALE factor."""
    return round(v * SCALE)


FONT_SIZE = S(14)
LABEL_ALIGN_WIDTH = 0  # Set by _auto_align_labels after UI is built
# ─────────────────────────────────────────────
# Device & Window Helpers (from gui.py)
# ─────────────────────────────────────────────
PRIMARY_MONITOR_SUFFIX = " [Main]"
def get_devices():
    """
    Returns (devices_dict, is_rocm).
    devices_dict: {0: {"name": str, "Computing Device": torch.device}, ...}
    """
    is_rocm = False
    devices = {}
    count = 0
    try:
        import torch_directml
        if torch_directml.is_available():
            for i in range(torch_directml.device_count()):
                dev_name = torch_directml.device_name(i).strip().rstrip('\x00')
                devices[count] = {
                    "name": f"DirectML{i}: {dev_name}",
                    "Computing Device": torch_directml.device(i),
                }
                count += 1
    except ImportError:
        pass
    try:
        import torch
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                name = torch.cuda.get_device_name(i)
                if torch.version.hip is not None:
                    is_rocm = True
                devices[count] = {"name": f"CUDA {i}: {name}", "Computing Device": torch.device(f"cuda:{i}")}
                count += 1
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            devices[count] = {"name": "MPS: Apple Silicon", "Computing Device": torch.device("mps")}
            count += 1
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            for i in range(torch.xpu.device_count()):
                name = torch.xpu.get_device_name(i)
                devices[count] = {"name": f"XPU {i}: {name}", "Computing Device": torch.device(f"xpu:{i}")}
                count += 1
        devices[count] = {"name": "CPU", "Computing Device": torch.device("cpu")}
    except ImportError:
        raise ImportError("PyTorch Not Found! Make sure you have deployed the Python environment in '.env'.")
    return devices, is_rocm
class _LazyDevices(dict):
    """Lazy hardware detection — only runs get_devices() on first dict API access."""
    def __init__(self):
        self._loaded = False
    def _ensure(self):
        if not self._loaded:
            self._loaded = True
            try:
                d, r = get_devices()
                self.update(d)
                global IS_ROCM; IS_ROCM = r
            except Exception as e:
                print(f"[Warning] Hardware detection failed, fallback to CPU: {e}")
                self.update({0: {"name": "CPU", "Computing Device": None}})
    def __getitem__(self, k): self._ensure(); return super().__getitem__(k)
    def __iter__(self): self._ensure(); return super().__iter__()
    def __len__(self): self._ensure(); return super().__len__()
    def values(self): self._ensure(); return super().values()
    def keys(self): self._ensure(); return super().keys()
    def items(self): self._ensure(); return super().items()
    def get(self, k, d=None): self._ensure(); return super().get(k, d)
    def __contains__(self, k): self._ensure(); return super().__contains__(k)
DEVICES = _LazyDevices()
IS_ROCM = False
def get_default_windows_capture_tool():
    if "CUDA" in DEVICES.get(0, {}).get("name", "") and not IS_ROCM:
        return "WindowsCaptureCUDA"
    elif "CUDA" in DEVICES.get(0, {}).get("name", "") and IS_ROCM:
        return "WindowsCaptureROCm"
    else:
        return "DXCamera"
def get_primary_monitor_index():
    if OS_NAME != "Windows":
        return _get_primary_monitor_index_unix()
    try:
        import win32api, win32con
        primary_monitor_handle = win32api.MonitorFromPoint((0, 0), win32con.MONITOR_DEFAULTTOPRIMARY)
        primary_monitor_info = win32api.GetMonitorInfo(primary_monitor_handle)
        primary_rect = primary_monitor_info["Monitor"]
        primary_left, primary_top, _, _ = primary_rect
        import mss
        with mss.mss() as sct:
            for idx, monitor in enumerate(sct.monitors[1:], start=1):
                if monitor["left"] == primary_left and monitor["top"] == primary_top:
                    return idx
        return 1
    except Exception:
        return 1
def _get_primary_monitor_index_unix():
    try:
        import mss
        with mss.mss() as sct:
            for idx, monitor in enumerate(sct.monitors[1:], start=1):
                if monitor["left"] == 0 and monitor["top"] == 0:
                    return idx
    except Exception:
        pass
    return 1
def list_windows():
    """Return list of {title, handle, rect} for visible windows."""
    if OS_NAME == "Windows":
        return _list_windows_win()
    elif OS_NAME == "Darwin":
        return _list_windows_mac()
    else:
        return _list_windows_linux()
def _list_windows_win():
    import win32gui
    windows = []
    def callback(hwnd, _):
        if win32gui.IsWindowVisible(hwnd):
            title = win32gui.GetWindowText(hwnd)
            if title:
                client_rect = win32gui.GetClientRect(hwnd)
                left, top = win32gui.ClientToScreen(hwnd, (client_rect[0], client_rect[1]))
                windows.append({
                    "title": title,
                    "handle": hwnd,
                    "rect": (left, top, client_rect[2], client_rect[3]),
                })
        return True
    win32gui.EnumWindows(callback, None)
    return windows
def _list_windows_mac():
    from Quartz import (
        CGWindowListCopyWindowInfo,
        kCGWindowListOptionOnScreenOnly,
        kCGWindowListExcludeDesktopElements,
        kCGNullWindowID,
    )
    windows = []
    options = kCGWindowListOptionOnScreenOnly | kCGWindowListExcludeDesktopElements
    window_info = CGWindowListCopyWindowInfo(options, kCGNullWindowID)
    blacklist = [
        "Window Server", "ControlCenter", "NotificationCenter", "Spotlight",
        "Dock", "FocusModes", "WiFi", "Sound", "UserSwitcher", "Clock",
        "BentoBox", "Bluetooth", "popdown", "AudioVideoModule",
        "ScreenMirroring", "SystemUIServer", "CoreServicesUIAgent",
        "TextInputMenuAgent", "com.apple.controlcenter", "loginwindow",
    ]
    for win in window_info:
        title = win.get("kCGWindowName", "") or ""
        owner = win.get("kCGWindowOwnerName", "")
        layer = win.get("kCGWindowLayer", 0)
        bounds = win.get("kCGWindowBounds", {})
        if not title.strip():
            continue
        if owner in blacklist or title in blacklist:
            continue
        if layer >= 1000:
            continue
        if win.get("kCGWindowAlpha", 1.0) == 0.0:
            continue
        if title.strip().lower().startswith(("item-", "window-")):
            continue
        if "X" in bounds and "Y" in bounds and "Width" in bounds and "Height" in bounds:
            w, h = bounds["Width"], bounds["Height"]
            if w < 10 or h < 10:
                continue
            windows.append({
                "title": title.strip(),
                "handle": win["kCGWindowNumber"],
                "rect": (bounds["X"], bounds["Y"], w, h),
            })
    return windows
def _list_windows_linux():
    windows = []
    try:
        result = subprocess.check_output(["wmctrl", "-lG"], timeout=2).decode("utf-8").splitlines()
        for line in result:
            parts = line.split(None, 7)
            if len(parts) >= 8:
                _, _, x_str, y_str, w_str, h_str, _, title = parts
                try:
                    x, y, w, h = int(x_str), int(y_str), int(w_str), int(h_str)
                    if title.strip():
                        windows.append({"title": title.strip(), "handle": None, "rect": (x, y, w, h)})
                except ValueError:
                    continue
    except Exception as e:
        print(f"[Warning] Linux window enumeration failed (install wmctrl): {e}")
    return windows
def get_monitor_index_for_point(x, y):
    try:
        import mss
        with mss.mss() as sct:
            for idx, mon in enumerate(sct.monitors[1:], start=1):
                left, top = mon["left"], mon["top"]
                right, bottom = left + mon["width"], top + mon["height"]
                if left <= x < right and top <= y < bottom:
                    return idx
    except Exception:
        pass
    return get_primary_monitor_index()
def _get_capture_tool_options(device_label):
    if OS_NAME == "Darwin":
        return ["ScreenCaptureKit", "Quartz"]
    if OS_NAME != "Windows":
        return ["DXCamera"]
    is_nvidia = "CUDA" in device_label.upper() and not IS_ROCM
    if is_nvidia:
        return ["WindowsCaptureCUDA", "WindowsCapture", "DXCamera", "DesktopDuplication"]
    elif "CUDA" in device_label.upper() and IS_ROCM:
        return ["WindowsCaptureROCm", "WindowsCapture", "DXCamera", "DesktopDuplication"]
    else:
        return ["DXCamera", "WindowsCapture", "DesktopDuplication"]
# ─────────────────────────────────────────────
# Save YAML (with encoding fallback)
# ─────────────────────────────────────────────
HAVE_YAML = True
def save_yaml(path, cfg):
    if not HAVE_YAML:
        return False, "PyYAML not installed"
    try:
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
        os.replace(tmp, path)
        return True, ""
    except Exception as e:
        return False, str(e)
# CompactDropdown — compact custom dropdown control

class CompactTextField(ft.Container):
    """Compact text input — 32px height, visually consistent with CompactDropdown."""

    def __init__(self, value="", width=S(100), read_only=False, on_change=None, tooltip=None, filter=None, max_length=None):
        super().__init__()
        self.height = S(32)
        self.width = width if width else None
        self.padding = 0
        self.bgcolor = None
        self.border = None
        self.border_radius = 0
        self._read_only = read_only
        self._on_change = on_change
        self._value = value
        self._label = ft.Text(value or "", size=FONT_SIZE)
        self._committed = False
        self._tooltip = tooltip or ""
        self._filter = filter
        self._max_length = max_length
        self._build_display()

    def _build_display(self):
        self.content = ft.Container(
            height=S(32), padding=ft.Padding(S(8), 0, S(8), 0),
            border=ft.Border(ft.BorderSide(1, ft.Colors.OUTLINE), ft.BorderSide(1, ft.Colors.OUTLINE), ft.BorderSide(1, ft.Colors.OUTLINE), ft.BorderSide(1, ft.Colors.OUTLINE)),
            border_radius=4,
            tooltip=self._tooltip,
            on_click=None if self._read_only else self._on_click,
            content=ft.Row([self._label], spacing=2, vertical_alignment=ft.CrossAxisAlignment.CENTER),
        )

    def _on_click(self, e):
        self._committed = False
        tf = ft.TextField(
            value=self._value, text_size=FONT_SIZE, dense=True,
            filled=False, border=ft.InputBorder.NONE,
            content_padding=ft.Padding(0, 0, 0, 0), height=S(28),
            autofocus=True, on_submit=self._on_submit, on_blur=self._on_submit,
            max_length=self._max_length,
            input_filter=ft.InputFilter(regex_string=self._filter, allow=True) if self._filter else None,
        )
        self.content = ft.Container(
            height=S(32), padding=ft.Padding(S(4), 0, S(4), 0),
            border=ft.Border(ft.BorderSide(1, ft.Colors.OUTLINE), ft.BorderSide(1, ft.Colors.OUTLINE), ft.BorderSide(1, ft.Colors.OUTLINE), ft.BorderSide(1, ft.Colors.OUTLINE)),
            border_radius=4, content=tf,
        )
        self.update()

    def set_tooltip(self, text):
        self._tooltip = text

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, val):
        self._value = val
        self._label.value = val
        try:
            self._label.update()
        except RuntimeError:
            pass

    def _on_submit(self, e):
        if self._committed:
            return
        self._committed = True
        raw = e.control.value
        if self._filter:
            raw = ''.join(c for c in raw if re.match(self._filter, c))
        self._value = raw
        if self._on_change:
            from types import SimpleNamespace
            self._on_change(SimpleNamespace(control=SimpleNamespace(value=self._value)))
        self._label.value = self._value
        self._build_display()
        self.update()


class CompactDropdown(ft.Container):
    """Compact dropdown — PopupMenuButton with controllable width."""
    _instances = None  # Managed by GUI instance; replaced with instance list in build_ui

    def __init__(self, options=None, value="", on_select=None, expand=False,
                 dyna_width=None, width=None, min_width=None, max_width=None,
                 tooltip=None, _instances_list=None):
        super().__init__()
        self._options = options or []
        self._on_select_cb = on_select
        self._dyna = dyna_width
        self._fixed = width
        self._min = min_width or 0
        self._max = max_width or 0
        self._tooltip = tooltip
        self.height = S(32)
        self.padding = 0
        self.bgcolor = None
        self.border = None
        self.border_radius = 0
        self.expand_loose = expand

        self._label = ft.Text(value or "", size=FONT_SIZE)
        self._build_menu()
        self._apply_width()
        if _instances_list is not None:
            _instances_list.append(self)
        elif CompactDropdown._instances is not None:
            CompactDropdown._instances.append(self)

    def reapply_width(self):
        self._apply_width()

    def _calc_auto_width(self):
        txt = self._label.value or ""
        if not txt:
            return S(100)
        w = sum(FONT_SIZE * (1.2 if ord(c) > 127 else 0.6) for c in txt)
        return int(w) + S(34)

    def _apply_width(self):
        if self._dyna:
            self.width = LABEL_ALIGN_WIDTH or self._calc_auto_width()
        elif self._fixed is not None:
            self.width = self._fixed
        else:
            auto = self._calc_auto_width()
            if self._min and auto < self._min:
                self.width = self._min
            elif self._max and auto > self._max:
                self.width = self._max
            else:
                self.width = None

    def _build_menu(self):
        def on_item_click(e):
            val = e.control.data
            self._label.value = val
            self._apply_width()
            try:
                self._label.update()
                self.update()
            except RuntimeError:
                pass
            if self._on_select_cb:
                from types import SimpleNamespace
                ev = SimpleNamespace(control=SimpleNamespace(value=val))
                self._on_select_cb(ev)

        items = [
            ft.PopupMenuItem(content=ft.Container(ft.Text(o, size=FONT_SIZE), padding=ft.Padding(8,0,8,0)), data=o, height=S(32), padding=0, on_click=on_item_click)
            for o in self._options
        ]

        has_limit = self._min or self._max
        align = ft.MainAxisAlignment.SPACE_BETWEEN if (self._fixed is not None or self._dyna or has_limit) else ft.MainAxisAlignment.START

        self.content = ft.PopupMenuButton(
            items=items,
            menu_position=ft.PopupMenuPosition.UNDER,
            enable_feedback=False,
            padding=0, menu_padding=0,
            tooltip=self._tooltip or "",
            content=ft.Container(
                height=S(32),
                padding=ft.Padding(S(8), 0, S(8), 0),
                tooltip=self._tooltip or "",
                border=ft.Border(
                    ft.BorderSide(1, ft.Colors.OUTLINE),
                    ft.BorderSide(1, ft.Colors.OUTLINE),
                    ft.BorderSide(1, ft.Colors.OUTLINE),
                    ft.BorderSide(1, ft.Colors.OUTLINE),
                ),
                border_radius=4,
                content=ft.Row([
                    self._label,
                    ft.Icon(ft.Icons.ARROW_DROP_DOWN, size=S(16)),
                ], spacing=2, alignment=align,
                   vertical_alignment=ft.CrossAxisAlignment.CENTER),
            ),
        )
    def set_tooltip(self, text):
        self._tooltip = text
        try:
            self._build_menu()
            self.update()
        except RuntimeError:
            pass

    @property
    def value(self):
        return self._label.value

    @value.setter
    def value(self, val):
        self._label.value = val
        self._apply_width()
        try:
            self._build_menu()
            self._label.update()
            self.update()
        except RuntimeError:
            pass

    @property
    def options(self):
        return self._options

    @options.setter
    def options(self, opts):
        self._options = opts
        try:
            self._build_menu()
            self.update()
        except RuntimeError:
            pass


# Main GUI Class
class Desktop2StereoGUI:
    """Flet GUI for Desktop2Stereo — full equivalent of tk ConfigGUI."""
    def __init__(self, page: ft.Page):
        self.page = page
        self._loop = None  # Assigned in setup()
        # ── state ──
        self.language = "EN"
        self._config = {}
        self.run_mode_key = DEFAULTS.get("Run Mode", "Local Viewer")
        self.crop_mode_key = DEFAULTS.get("Crop Mode", "auto")
        self.capture_mode_key = DEFAULTS.get("Capture Mode", "Monitor")
        self.stream_protocol_key = DEFAULTS.get("Stream Protocol", "RTMP")
        # Canonical environment key (English) — kept separate from the
        # dropdown's display value so Chinese labels in the UI don't leak
        # into settings.yaml or xrviewer's _init_env_model matcher.
        self.env_key = DEFAULTS.get("Environment Model", "Default")
        self.selected_window_name = ""
        self.selected_window_handle = None
        self.selected_window_rect = None
        self._window_objects = []
        self.process = None
        self._starting = False
        self._proc_lock = None  # Assigned in setup()
        self.monitor_label_to_index = {}
        self.device_label_to_index = {}
        self._esc_down = None        # GetAsyncKeyState: ESC press timestamp
        self._esc_stopped = False    # Prevent duplicate stop triggers
        self._closed = False         # Window closed flag
        self._cancel_starting = False  # Cancel pending start
        self._stopping = False       # Prevent _async_stop reentry
        self._labels_aligned = False  # One-shot label alignment guard
        self._status_key = ""          # Track status for language change re-translation

    # ── setup ──
    async def setup(self):
        _setup_console_logging()
        # Get the correct event loop in async context
        self._loop = asyncio.get_running_loop()
        self._proc_lock = asyncio.Lock()

        self.page.title = f"Desktop2Stereo v{VERSION}"
        self.page.window.icon = os.path.join(os.path.dirname(__file__), "icon.ico")
        self.page.padding = S(24)
        self.page.horizontal_alignment = ft.CrossAxisAlignment.STRETCH
        if OS_NAME == "Windows":
            font = "Microsoft YaHei"
        elif OS_NAME == "Darwin":
            font = "PingFang SC"
        else:
            font = "Noto Sans SC"
        self.page.theme = ft.Theme(color_scheme_seed="blue", font_family=font)
        self.page.spacing = 0
        self.page.theme_mode = ft.ThemeMode.SYSTEM
        self.page.window.min_width = S(696)
        self.page.window.min_height = S(300)

        # Build UI
        self.build_ui()
        self._auto_align_labels()
        self.page.on_close = self._on_page_close

        # Populate monitors & devices
        self.monitor_label_to_index = self.populate_monitors()
        self.device_label_to_index = self.populate_devices()

        # Load config
        self._config = DEFAULTS.copy()
        if os.path.exists(os.path.join(BASE_DIR, "settings.yaml")):
            try:
                cfg = read_yaml(os.path.join(BASE_DIR, "settings.yaml"))
                if cfg:
                    cfg.pop("Auto Crop", None)   # retired key, superseded by Crop Mode
                    self._config.update(cfg)
                    self.language = self._config.get("Language", "EN")
                    self.apply_config(self._config)
                    self.set_status(UI_TEXTS[self.language]["Loaded settings.yaml at startup"], key="Loaded settings.yaml at startup")
            except Exception as e:
                self.apply_config(self._config)
                self.set_status(f"{UI_TEXTS[self.language]['Failed to load settings.yaml:']} {e}")
        else:
            self.apply_config(self._config)

        self.on_device_change(None)
        self.auto_enable_optimizers_based_on_device()
        self.page.on_keyboard_event = self._on_key
        self._esc_task = asyncio.ensure_future(self._esc_poll_task())
        # Calculate window size without triggering update
        self._fit_window_to_content(update=False)

        # Position window centered on screen before showing it.
        # Per Flet docs: resize, update, center (async), then show.
        self.page.update()
        await self.page.window.center()
        self.page.window.visible = True
        self.page.update()
        await asyncio.sleep(0)

    def _ctrl_width(self, ctrl):
        """Get actual width of a control, accounting for CompactDropdown min/max constraints."""
        # CompactDropdown: estimate from current text, not stale width property
        if hasattr(ctrl, '_calc_auto_width'):
            auto = ctrl._calc_auto_width()
            fixed = getattr(ctrl, '_fixed', None)
            mn = getattr(ctrl, '_min', 0) or 0
            mx = getattr(ctrl, '_max', 0) or 0
            if fixed is not None:
                return fixed
            if mn and auto < mn:
                return mn
            if mx and auto > mx:
                return mx
            return auto
        # Standard control width attribute
        w = getattr(ctrl, "width", None) or 0
        if w:
            return w
        if hasattr(ctrl, '_fixed') and ctrl._fixed:
            return ctrl._fixed
        if hasattr(ctrl, '_label'):
            txt = ctrl._label.value or ""
            return sum(13 if ord(ch) > 127 else 7 for ch in txt) + 34
        # CompactTextField
        if hasattr(ctrl, '_value'):
            txt = str(ctrl._value or "")
            return sum(13 if ord(ch) > 127 else 7 for ch in txt) + 34
        # ft.Button / ft.FilledButton: extract from content.controls[0] or content.value
        content = getattr(ctrl, "content", None)
        if content is not None:
            if hasattr(content, "value") and content.value:
                txt = content.value
            elif hasattr(content, "controls"):
                txt = "".join(c.value for c in content.controls if hasattr(c, "value") and c.value)
            else:
                txt = ""
            if txt:
                return sum(13 if ord(ch) > 127 else 7 for ch in txt) + 40
        # ft.Checkbox / generic: use label or value
        txt = getattr(ctrl, "label", None) or getattr(ctrl, "value", None) or ""
        return sum(13 if ord(ch) > 127 else 7 for ch in str(txt)) + 28

    def _fit_window_to_content(self, update=True):
        """Keep width fixed; auto-adjust height based on visible content."""
        self.page.window.width = S(696)
        self.page.window.max_width = S(696)
        if getattr(self, 'stream_container', None) and self.stream_container.visible:
            self.page.window.height = S(1008)
        else:
            self.page.window.height = S(768)
        if update:
            self.page.update()

    # Build UI — grid-aligned, matching original GUI

    def _auto_align_labels(self):
        """Auto-detect longest label text and unify label widths for grid alignment."""
        if self._labels_aligned:
            return
        left_labels = [
            self.r0_label, self.r1a_label, self.r2a_label, self.r3a_label,
            self.r4_label, self.r5_label, self.r6_label,
            self.r7a_label, self.r9_label, self.r10_label,
            self.lang_label,
            self.stream_url_label, self.stream_port_label,
            self.stream_proto_label, self.audio_label, self.crf_label,
        ]
        right_labels = [
            self.r1b_label, self.r2b_label, self.r3b_label,
            self.r7b_label, self.theme_label,
            self.stream_quality_label, self.stream_key_label, self.audio_delay_label,
        ]

        def _est(t):
            return sum(S(13) if ord(c) > 127 else S(7) for c in t)

        all_labels = left_labels + right_labels
        max_w = max(_est(lbl.value) for lbl in all_labels)
        final_w = int(max_w * 1.15) + S(10)

        for lbl in all_labels:
            lbl.width = final_w

        self._label_max_width = final_w
        global LABEL_ALIGN_WIDTH
        LABEL_ALIGN_WIDTH = final_w
        for inst in getattr(self, '_dropdowns', []):
            inst.reapply_width()
        if hasattr(self, '_row8_spacer'):
            self._row8_spacer.width = max(0, final_w - S(128) - 1)
            try:
                self._row8_spacer.update()
            except RuntimeError:
                pass
        if hasattr(self, '_accel_spacer'):
            self._accel_spacer.width = final_w
            try:
                self._accel_spacer.update()
            except RuntimeError:
                pass
        self._labels_aligned = True

    def build_ui(self):
        page = self.page
        page.controls.clear()
        self._dropdowns = []
        CompactDropdown._instances = self._dropdowns

        # Row 1: Depth model
        self.r0_label = ft.Text("Depth Model:", size=FONT_SIZE, width=S(130))
        default_family, default_size = parse_model_name(DEFAULT_MODEL_LIST[0]) if DEFAULT_MODEL_LIST else ("", "")
        self.depth_model_dd = CompactDropdown(
            options=[f for f in DEFAULT_FAMILIES],
            value=default_family,
            on_select=self.on_model_family_change,
            min_width=S(200), max_width=S(300))
        self.model_size_dd = CompactDropdown(
            options=FAMILY_TO_SIZES.get(default_family, []),
            value=default_size,
            on_select=self.on_model_size_change,
            width=S(110))
        self.fp16_cb = ft.Checkbox(scale=SCALE, visual_density=ft.VisualDensity.COMPACT, label="FP16")
        row0 = ft.Row([
            self.r0_label,
            self.depth_model_dd,
            ft.Container(width=S(8)),
            self.model_size_dd,
        ], spacing=1)

        # Row 2: Depth resolution + Convergence
        self.r1a_label = ft.Text("Depth Resolution:", size=FONT_SIZE, width=S(130))
        self.depth_res_dd = CompactDropdown(options=[], width=S(130))
        self.r1b_label = ft.Text("Convergence:", size=FONT_SIZE, width=S(130))
        conv_options = [str(i / 4) for i in range(-2, 5)]
        self.convergence_dd = CompactDropdown(width=S(130),
            options=[v for v in conv_options],
            value="0.0")
        row1 = ft.Row([
            self.r1a_label,
            self.depth_res_dd,
            ft.Container(width=S(40)),
            self.r1b_label,
            self.convergence_dd
        ], spacing=1)

        # Row 3: Depth strength + Foreground scale
        self.r2a_label = ft.Text("Depth Strength:", size=FONT_SIZE, width=S(130))
        ds_options = [f"{i / 2:.1f}" for i in range(21)]
        self.depth_strength_dd = CompactDropdown(width=S(130),
            options=[v for v in ds_options],
            value="2.0")
        self.r2b_label = ft.Text("Foreground Scale:", size=FONT_SIZE, width=S(130))
        fg_options = [f"{i / 2:.1f}" for i in range(-10, 11)]
        self.foreground_scale_dd = CompactDropdown(width=S(130),
            options=[v for v in fg_options],
            value="0.5")
        row2 = ft.Row([
            self.r2a_label,
            self.depth_strength_dd,
            ft.Container(width=S(40)),
            self.r2b_label,
            self.foreground_scale_dd
        ], spacing=1)

        # Row 4: Anti-aliasing + IPD
        self.r3a_label = ft.Text("Anti-aliasing:", size=FONT_SIZE, width=S(130))
        aa_options = [str(i) for i in range(11)]
        self.antialiasing_dd = CompactDropdown(width=S(130), 
            options=[v for v in aa_options],
            value="2")
        self.r3b_label = ft.Text("IPD (mm):", size=FONT_SIZE, width=S(130))
        self.ipd_dd = CompactDropdown(options=[str(i) for i in range(58, 71)], value="64", width=S(130))
        row3 = ft.Row([
            self.r3a_label,
            self.antialiasing_dd,
            ft.Container(width=S(40)),
            self.r3b_label,
            self.ipd_dd
        ], spacing=1)

        # Row 5: Acceleration group (two rows, 4 columns each)
        self.r4_label = ft.Text("Acceleration:", size=FONT_SIZE, width=S(130))
        self.torch_compile_cb = ft.Checkbox(scale=SCALE, visual_density=ft.VisualDensity.COMPACT, label="torch.compile", on_change=self._on_torch_compile_toggle)
        self.tensorrt_cb = ft.Checkbox(scale=SCALE, visual_density=ft.VisualDensity.COMPACT, label="TensorRT", on_change=self._on_trt_toggle)
        self.coreml_cb = ft.Checkbox(scale=SCALE, visual_density=ft.VisualDensity.COMPACT, label="CoreML", on_change=self._on_coreml_toggle)
        self.openvino_cb = ft.Checkbox(scale=SCALE, visual_density=ft.VisualDensity.COMPACT, label="OpenVINO", on_change=self._on_openvino_toggle)
        self.recompile_trt_cb = ft.Checkbox(scale=SCALE, visual_density=ft.VisualDensity.COMPACT, label="Recompile TensorRT")
        self.recompile_coreml_cb = ft.Checkbox(scale=SCALE, visual_density=ft.VisualDensity.COMPACT, label="Recompile CoreML")
        self.recompile_openvino_cb = ft.Checkbox(scale=SCALE, visual_density=ft.VisualDensity.COMPACT, label="Recompile OpenVINO")
        self.migraphx_cb = ft.Checkbox(scale=SCALE, visual_density=ft.VisualDensity.COMPACT, label="MIGraphX", on_change=self._on_migraphx_toggle)
        self.recompile_migraphx_cb = ft.Checkbox(scale=SCALE, visual_density=ft.VisualDensity.COMPACT, label="Recompile MIGraphX")
        accel_row1 = ft.Row([self.fp16_cb, self.torch_compile_cb, self.coreml_cb, self.recompile_coreml_cb], spacing=S(20))
        accel_row2 = ft.Row([self.tensorrt_cb, self.migraphx_cb, self.recompile_trt_cb, self.recompile_migraphx_cb, self.openvino_cb, self.recompile_openvino_cb], spacing=S(20))
        self._accel_spacer = ft.Container(width=0)
        self.row4a = ft.Row([
            self.r4_label,
            accel_row1,
        ], spacing=1)
        self.row4b = ft.Row([
            self._accel_spacer,
            accel_row2,
        ], spacing=1)

        # Row 6: Computing device
        self.r5_label = ft.Text("Computing Device:", size=FONT_SIZE, width=S(130))
        device_names = [v["name"] for v in DEVICES.values()]
        self.device_dd = CompactDropdown(
            options=[n for n in device_names],
            on_select=self.on_device_change,
            min_width=S(180))
        self.showfps_cb = ft.Checkbox(scale=SCALE, visual_density=ft.VisualDensity.COMPACT, label="Show FPS")
        self.vsync_cb = ft.Checkbox(
            scale=SCALE,
            visual_density=ft.VisualDensity.COMPACT,
            label="VSync",
            value=DEFAULTS.get("VSync", False),
        )
        row5 = ft.Row([
            self.r5_label,
            self.device_dd
        ], spacing=1)

        # Row 7: Capture tool
        self.r6_label = ft.Text("Capture Tool:", size=FONT_SIZE, width=S(130))
        ct_options = _get_capture_tool_options(DEVICES.get(0, {}).get("name", ""))
        self.capture_tool_dd = CompactDropdown(
            options=[o for o in ct_options],
            on_select=self.on_capture_tool_change,
            min_width=S(160))
        self.xr_preview_cb = ft.Checkbox(
            label="XR Preview",
            value=DEFAULTS.get("XR Preview", True),
        )
        row6 = ft.Row([self.r6_label, self.capture_tool_dd, ft.Container(width=S(15)), self.showfps_cb, self.vsync_cb, ft.Container(width=S(40)), self.xr_preview_cb], spacing=1)
        if OS_NAME == "Linux":
            self.r6_label.visible = False
            self.capture_tool_dd.visible = False

        # Row 8: Run mode + Display mode / Controller
        self.r7a_label = ft.Text("Run Mode:", size=FONT_SIZE, width=S(130))
        self.run_mode_dd = CompactDropdown(on_select=self.on_run_mode_change, width=S(130))
        self.r7b_label = ft.Text("Display Mode:", size=FONT_SIZE, width=S(130))
        self.display_mode_dd = CompactDropdown(
            options=[m for m in ["Half-SBS", "Full-SBS", "Half-TAB", "Full-TAB", "Depth Map", "Anaglyph", "Interleaved", "Interleaved-V"]],
            value="Half-SBS", width=S(130))
        self.crop_mode_label = ft.Text("Crop Mode:", size=FONT_SIZE, width=S(130))
        _crop_texts = UI_TEXTS[self.language]
        self.crop_mode_dd = CompactDropdown(
            options=[_crop_texts.get("Crop Auto", "Auto"),
                     _crop_texts.get("Crop Manual", "Manual"),
                     _crop_texts.get("Crop Off", "Off")],
            value=_crop_texts.get("Crop Auto", "Auto"),
            on_select=self.on_crop_mode_change,
            width=S(130))
        self.r10_label = ft.Text("Controller:", size=FONT_SIZE, width=S(130))
        try:
            ctrl_base = os.path.join(os.path.dirname(__file__), "xr_viewer", "controllers")
            ctrl_dirs = [d for d in os.listdir(ctrl_base) if os.path.isdir(os.path.join(ctrl_base, d))]
        except (FileNotFoundError, OSError):
            ctrl_dirs = []
        if not ctrl_dirs:
            ctrl_dirs = ["PICO"]
        self.ctrl_model_dd = CompactDropdown(
            options=[c for c in ctrl_dirs],
            value="PICO", width=S(130))

        # Environment dropdown sits to the right of the controller picker.
        # Options are the built-in "Default" (black backdrop) plus every
        # panorama/image folder and every GLB room under xr_viewer/environments/,
        # matching the runtime cycle in xrviewer's _cycle_environment.
        #   * Default -> plain opaque-black backdrop (no env model)
        #      Glow can be toggled via long-press X in VR.
        # Built-in names are localized for display (e.g. CN: 默认)
        # while the canonical English key is stored in self.env_key and used
        # for settings.yaml + the xrviewer matcher. User folder names under
        # xr_viewer/environment/ may carry a localized label in their profile.json:
        #   {"display_name": {"EN": "Bedroom", "CN": "卧室"}, ...}
        # If absent or unreadable the folder name itself is shown.
        self.r11_label = ft.Text(UI_TEXTS[self.language]["Environment:"], size=FONT_SIZE, width=S(130))
        env_base = os.path.join(os.path.dirname(__file__), "xr_viewer", "environments")
        self._env_base = env_base
        self._env_builtin_keys = ["Default"]
        self._env_folder_keys = _discover_gui_environment_folders(env_base)
        # Cache per-folder display_name dicts so we don't hit the disk on
        # every language toggle. Populated lazily by _load_env_display_names.
        self._env_folder_display_cache = {}
        self._load_env_folder_display_names()
        env_options = self._build_env_dd_options(self.language)
        self.env_dd = CompactDropdown(
            options=env_options,
            value=self._env_display_label("Default", self.language),
            on_select=self.on_env_change,
            width=S(130))
        self._xr_preview_spacer = ft.Container(width=S(40))
        self.row7a = ft.Row([
            self.r7a_label,
            self.run_mode_dd,
            ft.Container(width=S(40)),
            self.crop_mode_label,
            self.crop_mode_dd,
            self._xr_preview_spacer,
            self.r7b_label,
            self.display_mode_dd
        ], spacing=1)
        self.row7b = ft.Row([
            self.r10_label,
            self.ctrl_model_dd,
            ft.Container(width=S(40)),
            self.r11_label,
            self.env_dd
        ], spacing=1)

        # Row 9: Input monitor/window + Refresh
        self.capture_mode_dd = CompactDropdown(
            options=["Monitor",
                     "Window"],
            value="Monitor", on_select=self.on_capture_mode_change,
            width=S(128))
        self.monitor_dd = CompactDropdown(on_select=self._on_monitor_change, max_width=S(300))
        self.window_dd = CompactDropdown(on_select=self.on_window_selected, max_width=S(300))
        self.refresh_btn = ft.Button(content=ft.Text("Refresh", size=FONT_SIZE), width=S(130), on_click=self.refresh_monitor_and_window)
        self._row8_spacer = ft.Container(width=S(60))
        row8 = ft.Row([
            self.capture_mode_dd,
            self._row8_spacer,
            self.monitor_dd,
            self.window_dd,
            ft.Container(width=S(8)),
            ft.Container(expand=True),
            self.refresh_btn
        ], spacing=1)

        # Row 10: Stereo output + checkboxes
        self.r9_label = ft.Text("Stereo Output:", size=FONT_SIZE, width=S(130))
        self.stereo_monitor_dd = CompactDropdown(options=[], on_select=lambda e: self._fit_window_to_content())
        self.fill_16_9_cb = ft.Checkbox(scale=SCALE, visual_density=ft.VisualDensity.COMPACT, label="Fill 16:9")
        self.fix_aspect_cb = ft.Checkbox(scale=SCALE, visual_density=ft.VisualDensity.COMPACT, label="Fix Viewer Aspect")
        self.lossless_cb = ft.Checkbox(scale=SCALE, visual_density=ft.VisualDensity.COMPACT, label="LSFG")
        self._stereo_spacer = ft.Container(width=S(10))
        row9 = ft.Row([
            self.r9_label,
            self.stereo_monitor_dd,
            self._stereo_spacer,
            ft.Row([self.fill_16_9_cb, self.fix_aspect_cb, self.lossless_cb], spacing=S(20)),
        ], spacing=1)

        # Bottom: Language + Theme + Buttons
        self.lang_label = ft.Text("Set Language:", size=FONT_SIZE, width=S(130))
        self.lang_dd = CompactDropdown(
            options=["English", "简体中文"],
            value="English", on_select=self.on_language_change,
            width=S(130))
        self.theme_label = ft.Text("Theme:", size=FONT_SIZE, width=S(130))
        self.theme_dd = CompactDropdown(
            options=["system", "blue", "green", "red", "purple", "orange", "teal", "pink", "grey"],
            value="system", on_select=self.on_theme_change,
            width=S(130))
        self.reset_btn = ft.Button(content=ft.Text("Reset", size=FONT_SIZE), width=S(130), on_click=self.reset_defaults)
        self.open_log_btn = ft.Button(content=ft.Text("Log Folder", size=FONT_SIZE), width=S(130), on_click=self.open_log, tooltip="Please send the log file for debugging.")
        self.stop_btn = ft.Button(content=ft.Text("Stop", size=FONT_SIZE), width=S(130), on_click=self.stop_process)
        self.run_btn = ft.Button(content=ft.Text("Run", size=FONT_SIZE), width=S(150), on_click=self.save_and_run)
        lang_row = ft.Row([
            self.lang_label,
            self.lang_dd,
            ft.Container(width=S(40)),
            self.theme_label,
            self.theme_dd,
        ], spacing=1)

        self.status_text = ft.Text("", italic=True, size=FONT_SIZE)

        # Assembly
        depth_group = ft.Container(
            ft.Column([row0, row1, row2, row3, self.row4a, self.row4b], spacing=S(8)),
            margin=ft.Margin(0, 0, 0, S(8)),
            border=ft.Border(ft.BorderSide(1, ft.Colors.OUTLINE), ft.BorderSide(1, ft.Colors.OUTLINE), ft.BorderSide(1, ft.Colors.OUTLINE), ft.BorderSide(1, ft.Colors.OUTLINE)),
            border_radius=6, padding=ft.Padding(S(16), S(10), S(16), S(10)),
        )
        device_group = ft.Container(
            ft.Column([row5, row6, self.row7a, self.row7b, row8, row9], spacing=S(8)),
            margin=ft.Margin(0, 0, 0, S(8)),
            border=ft.Border(ft.BorderSide(1, ft.Colors.OUTLINE), ft.BorderSide(1, ft.Colors.OUTLINE), ft.BorderSide(1, ft.Colors.OUTLINE), ft.BorderSide(1, ft.Colors.OUTLINE)),
            border_radius=6, padding=ft.Padding(S(16), S(10), S(16), S(10)),
        )
        lang_group = ft.Container(
            ft.Column([lang_row], spacing=S(8)),
            margin=ft.Margin(0, 0, 0, S(8)),
            border=ft.Border(ft.BorderSide(1, ft.Colors.OUTLINE), ft.BorderSide(1, ft.Colors.OUTLINE), ft.BorderSide(1, ft.Colors.OUTLINE), ft.BorderSide(1, ft.Colors.OUTLINE)),
            border_radius=6, padding=ft.Padding(S(16), S(10), S(16), S(10)),
        )
        self.lang_group = lang_group
        self.depth_group = depth_group
        self.device_group = device_group
        self._build_streamer_rows()

        scroll_area = ft.Column([
            self.lang_group,
            self.depth_group,
            self.device_group,
            self.stream_container,
        ], scroll=ft.ScrollMode.AUTO, expand=True, spacing=S(8))

        btn_row = ft.Row([
            ft.Row([self.reset_btn, self.open_log_btn], spacing=S(10)),
            ft.Container(expand=True),
            ft.Row([self.stop_btn, self.run_btn], spacing=S(20)),
        ])

        self._btn_bar = ft.Container(content=btn_row)
        self._status_bar = ft.Row([
            ft.Container(
                content=self.status_text,
                bgcolor=ft.Colors.SURFACE_CONTAINER,
                border_radius=0, padding=ft.Padding(S(8), S(4), S(8), S(4)),
                expand=True,
            )
        ])

        footer = ft.Container(
            ft.Column([self._btn_bar, self._status_bar], spacing=S(16)),
            padding=ft.Padding(0, S(18), 0, 0),
        )

        self._scroll_area = scroll_area
        self._footer = footer

        root = ft.Column([scroll_area, footer], expand=True, spacing=0)
        page.add(root)

    # All event/logic methods below preserve original behavior
    def _build_streamer_rows(self):
        self.stream_url_label = ft.Text("Stream URL:", size=FONT_SIZE, width=S(150))
        self.stream_url_tf = ft.Container(
            content=ft.Row([ft.Text("", size=FONT_SIZE)], vertical_alignment=ft.CrossAxisAlignment.CENTER),
            height=S(32), padding=ft.Padding(S(8), 0, S(8), 0), expand=True,
            border=ft.Border(ft.BorderSide(1, ft.Colors.OUTLINE), ft.BorderSide(1, ft.Colors.OUTLINE), ft.BorderSide(1, ft.Colors.OUTLINE), ft.BorderSide(1, ft.Colors.OUTLINE)),
            border_radius=4, on_click=self.copy_url_to_clipboard,
        )
        self.preview_btn = ft.Button(content=ft.Text("Preview", size=FONT_SIZE), width=S(130), on_click=self.preview_in_browser)
        self.stream_url_row = ft.Row(
            [self.stream_url_label, self.stream_url_tf, ft.Container(width=S(10)), self.preview_btn],
            spacing=2,
        )
        self.stream_port_label = ft.Text("Streamer Port:", size=FONT_SIZE, width=S(150))
        self.stream_port_tf = CompactTextField(value=str(DEFAULT_PORT), width=S(130), on_change=self.update_stream_url, filter=r"[0-9]", max_length=5)
        self.stream_quality_label = ft.Text("Stream Quality:", size=FONT_SIZE)
        qual_vals = [str(i) for i in range(100, 49, -5)]
        self.stream_quality_dd = CompactDropdown(width=S(130),
            options=[q for q in qual_vals], value="100")
        self.stream_port_quality_row = ft.Row(
            [self.stream_port_label, self.stream_port_tf, ft.Container(width=S(40)), self.stream_quality_label, self.stream_quality_dd],
            spacing=1,
        )
        self.stream_proto_label = ft.Text("Stream Protocol:", size=FONT_SIZE, width=S(150))
        self.stream_proto_dd = CompactDropdown(width=S(130),
            options=[p for p in ["RTMP", "RTSP", "HLS", "HLS M3U8", "WebRTC"]],
            value="HLS", on_select=self._on_stream_protocol_change)
        self.stream_key_label = ft.Text("Stream Key:", size=FONT_SIZE, width=S(130))
        self.stream_key_tf = CompactTextField(value="live", width=S(130), on_change=self._on_stream_key_change)
        self.stream_proto_row = ft.Row([self.stream_proto_label, self.stream_proto_dd, ft.Container(width=S(40)), self.stream_key_label, self.stream_key_tf], spacing=1)
        self.audio_label = ft.Text("Stereo Mix:", size=FONT_SIZE, width=S(150))
        self.audio_dd = CompactDropdown(options=[], min_width=S(130))
        self.audio_row = ft.Row([self.audio_label, self.audio_dd], spacing=1)
        self.crf_label = ft.Text("CRF:", size=FONT_SIZE, width=S(150))
        self.crf_tf = CompactTextField(value="20", width=S(130), filter=r"[0-9]", max_length=2)
        self.audio_delay_label = ft.Text("Audio Delay (s):", size=FONT_SIZE, width=S(130))
        self.audio_delay_tf = CompactTextField(value="-0.15", width=S(130), filter=r"[0-9\-\.]", max_length=6)
        self.crf_row = ft.Row([self.crf_label, self.crf_tf, ft.Container(width=S(40)), self.audio_delay_label, self.audio_delay_tf], spacing=1)
        self._streamer_rows = [
            self.stream_url_row,
            self.stream_port_quality_row,
            self.stream_proto_row,
            self.crf_row,
            self.audio_row,
        ]
        self.stream_container = ft.Container(
            ft.Column([], spacing=S(8)),
            visible=False, padding=ft.Padding(S(16), S(10), S(16), S(10)),
            border=ft.Border(ft.BorderSide(1, ft.Colors.OUTLINE), ft.BorderSide(1, ft.Colors.OUTLINE), ft.BorderSide(1, ft.Colors.OUTLINE), ft.BorderSide(1, ft.Colors.OUTLINE)),
            border_radius=6,
        )

    def _show_streamer_rows(self, *row_indices):
        col = self.stream_container.content.controls
        col.clear()
        for i in row_indices:
            if 0 <= i < len(self._streamer_rows):
                col.append(self._streamer_rows[i])
        self.stream_container.visible = bool(row_indices)
        self.stream_container.update()
        self._fit_window_to_content()

    def _get_streamer_row_map(self):
        return {
            "Local Viewer": [],
            "3D Monitor": [],
            "OpenXR Link": [],
            "MJPEG Streamer": [0, 1],
            "Legacy Streamer": [0, 1],
            "RTMP Streamer": [0, 1, 2, 3, 4],
        }

    def populate_monitors(self):
        self.monitor_label_to_index = {}
        monitors = []
        try:
            import mss
            with mss.mss() as sct:
                monitors = sct.monitors[1:]
        except Exception:
            monitors = []
        if not monitors:
            self.monitor_dd.options = []
            self.monitor_dd.update()
            return {}
        primary_index = get_primary_monitor_index()
        if primary_index < 1 or primary_index > len(monitors):
            primary_index = 1
        current_val = self.monitor_dd.value if hasattr(self, 'monitor_dd') else ""
        found = False
        opts = []
        for idx, mon in enumerate(monitors, start=1):
            is_primary = idx == primary_index
            suffix = PRIMARY_MONITOR_SUFFIX if is_primary else ""
            label = f"{idx}: {mon['width']}x{mon['height']} @ ({mon['left']},{mon['top']}){suffix}"
            self.monitor_label_to_index[label] = idx
            opts.append(label)
            if label == current_val:
                found = True
        self.monitor_dd.options = opts
        if found:
            self.monitor_dd.value = current_val
        else:
            primary_label = next((lbl for lbl, i in self.monitor_label_to_index.items() if i == primary_index), None)
            self.monitor_dd.value = primary_label or (list(self.monitor_label_to_index.keys())[0] if self.monitor_label_to_index else "")
        self.monitor_dd.update()
        self.update_stereo_monitor_menu()
        self._fit_window_to_content()
        return self.monitor_label_to_index

    def populate_devices(self):
        self.device_label_to_index = {}
        device_dict = DEVICES
        opts = []
        for idx, dev_info in device_dict.items():
            label = dev_info["name"]
            self.device_label_to_index[label] = idx
            opts.append(label)
        self.device_dd.options = opts
        default_idx = DEFAULTS.get("Computing Device", 0)
        default_label = next((lbl for lbl, i in self.device_label_to_index.items() if i == default_idx), None)
        self.device_dd.value = default_label or (opts[0] if opts else "")
        self.device_dd.update()
        return self.device_label_to_index

    def apply_config(self, cfg, keep_optional=True):
        self._config = cfg.copy()
        current_primary = get_primary_monitor_index()
        mon_idx = cfg.get("Monitor Index", DEFAULTS["Monitor Index"])
        label = next((lbl for lbl, i in self.monitor_label_to_index.items() if i == mon_idx), None)
        if label:
            self.monitor_dd.value = label
        elif self.monitor_label_to_index:
            primary_label = next((lbl for lbl, i in self.monitor_label_to_index.items() if i == current_primary), None)
            self.monitor_dd.value = primary_label or next(iter(self.monitor_label_to_index))
        self.selected_window_name = cfg.get("Window Title", "")
        self.selected_window_handle = None
        self.selected_window_rect = None
        if keep_optional and self.capture_mode_key == "Window":
            self.refresh_window_list()
            dev_idx = cfg.get("Computing Device", DEFAULTS["Computing Device"])
            dev_label = next((lbl for lbl, i in self.device_label_to_index.items() if i == dev_idx), None)
            if dev_label:
                self.device_dd.value = dev_label
        model_list = DEFAULT_MODEL_LIST
        selected_model = cfg.get("Depth Model", DEFAULTS["Depth Model"])
        if selected_model not in model_list:
            selected_model = model_list[0] if model_list else DEFAULTS["Depth Model"]
        family, size = parse_model_name(selected_model)
        if family not in DEFAULT_FAMILIES:
            family = DEFAULT_FAMILIES[0] if DEFAULT_FAMILIES else ""
        self.depth_model_dd.options = [f for f in DEFAULT_FAMILIES]
        self.depth_model_dd.value = family
        avail_sizes = FAMILY_TO_SIZES.get(family, [])
        self.model_size_dd.options = [s for s in avail_sizes]
        if size in avail_sizes:
            self.model_size_dd.value = size
        elif avail_sizes:
            self.model_size_dd.value = avail_sizes[0]
        else:
            self.model_size_dd.value = ""
        self.depth_res_dd.value = str(cfg.get("Depth Resolution", DEFAULTS["Depth Resolution"]))
        self.update_depth_resolution_options(self.current_model_name)
        self.depth_strength_dd.value = str(cfg.get("Depth Strength", DEFAULTS["Depth Strength"]))
        self.display_mode_dd.value = cfg.get("Display Mode", DEFAULTS["Display Mode"])
        self.xr_preview_cb.value = cfg.get("XR Preview", DEFAULTS["XR Preview"])
        self.crop_mode_key = cfg.get("Crop Mode", DEFAULTS["Crop Mode"])
        self.crop_mode_dd.value = self.crop_mode_key
        self.vsync_cb.value = cfg.get("VSync", DEFAULTS["VSync"])
        self.antialiasing_dd.value = str(cfg.get("Anti-aliasing", DEFAULTS["Anti-aliasing"]))
        self.foreground_scale_dd.value = str(cfg.get("Foreground Scale", DEFAULTS["Foreground Scale"]))
        self.convergence_dd.value = str(cfg.get("Convergence", DEFAULTS["Convergence"]))
        ipd_m = cfg.get("IPD", DEFAULTS["IPD"])
        self.ipd_dd.value = str(int(ipd_m * 1000))
        self.fp16_cb.value = cfg.get("FP16", DEFAULTS["FP16"])
        self.showfps_cb.value = cfg.get("Show FPS", DEFAULTS["Show FPS"])
        self.fill_16_9_cb.value = cfg.get("Fill 16:9", DEFAULTS["Fill 16:9"])
        self.fix_aspect_cb.value = cfg.get("Fix Viewer Aspect", DEFAULTS["Fix Viewer Aspect"])
        self.lossless_cb.value = cfg.get("Lossless Scaling Support", DEFAULTS["Lossless Scaling Support"])
        if keep_optional:
            self.language = cfg.get("Language", DEFAULTS["Language"])
            self.lang_dd.value = "English" if self.language == "EN" else "简体中文"
        
        saved_ctrl = cfg.get("Controller Model", DEFAULTS.get("Controller Model", "PICO"))
        self.ctrl_model_dd.value = saved_ctrl if saved_ctrl in self.ctrl_model_dd.options else "PICO"
        saved_env = cfg.get("Environment Model", DEFAULTS.get("Environment Model", "Default"))
        # Case-insensitive match against canonical English keys (built-ins
        # plus user folder names). Hand-edited settings.yaml entries like
        # "monitor" / "dark room" still bind to the exact key.
        # Legacy "Black" -> new "Default" alias for backward compatibility.
        if str(saved_env).strip().lower() == "black":
            saved_env = "Default"
        canonical_keys = list(self._env_builtin_keys) + list(self._env_folder_keys)
        env_key_match = next(
            (k for k in canonical_keys if str(k).lower() == str(saved_env).lower()),
            None,
        )
        self.env_key = env_key_match if env_key_match is not None else "Default"
        # Display value uses the localized label for both built-ins and
        # folders (folders consult their profile.json display_name).
        self.env_dd.value = self._env_display_label(self.env_key, self.language)
        self.torch_compile_cb.value = cfg.get("torch.compile")
        if self.torch_compile_cb.value is None:
            self.torch_compile_cb.value = False
        trt_val = cfg.get("TensorRT")
        if trt_val is not None:
            self.tensorrt_cb.value = trt_val
        self.recompile_trt_cb.value = cfg.get("Recompile TensorRT", DEFAULTS["Recompile TensorRT"])
        cml_val = cfg.get("CoreML")
        if cml_val is not None:
            self.coreml_cb.value = cml_val
        self.recompile_coreml_cb.value = cfg.get("Recompile CoreML", DEFAULTS["Recompile CoreML"])
        ov_val = cfg.get("OpenVINO")
        if ov_val is not None:
            self.openvino_cb.value = ov_val
        self.recompile_openvino_cb.value = cfg.get("Recompile OpenVINO", DEFAULTS["Recompile OpenVINO"])
        self.recompile_trt_cb.visible = self.tensorrt_cb.value and self.tensorrt_cb.visible
        self.recompile_coreml_cb.visible = self.coreml_cb.value and self.coreml_cb.visible
        self.recompile_openvino_cb.visible = self.openvino_cb.value and self.openvino_cb.visible
        mgx_val = cfg.get("MIGraphX")
        if mgx_val is not None:
            self.migraphx_cb.value = mgx_val
        self.recompile_migraphx_cb.value = cfg.get("Recompile MIGraphX", DEFAULTS["Recompile MIGraphX"])
        self.recompile_migraphx_cb.visible = self.migraphx_cb.value and self.migraphx_cb.visible
                # Capture tool
        ct = cfg.get("Capture Tool", DEFAULTS["Capture Tool"])
        self.capture_tool_dd.value = ct if ct in self.capture_tool_dd.options else (self.capture_tool_dd.options[0] if self.capture_tool_dd.options else '')
        # Run mode
        if keep_optional:
            run_mode = cfg.get("Run Mode", DEFAULTS.get("Run Mode", "Local Viewer"))
            if run_mode == "3D Monitor" and OS_NAME != "Windows":
                run_mode = "Local Viewer"
            if run_mode == "OpenXR Link" and OS_NAME == "Darwin":
                run_mode = "Local Viewer"
            self.run_mode_key = run_mode
        self.stream_protocol_key = cfg.get("Stream Protocol", DEFAULTS.get("Stream Protocol", "RTMP"))
        self.stream_proto_dd.value = self.stream_protocol_key
        self.stream_port_tf.value = str(cfg.get("Streamer Port", DEFAULTS.get("Streamer Port", DEFAULT_PORT)))
        self.stream_quality_dd.value = str(cfg.get("Stream Quality", DEFAULTS["Stream Quality"]))
        self.stream_key_tf.value = cfg.get("Stream Key", DEFAULTS["Stream Key"])
        self.audio_dd.value = cfg.get("Stereo Mix", "")
        self.crf_tf.value = str(cfg.get("CRF", DEFAULTS["CRF"]))
        self.audio_delay_tf.value = str(cfg.get("Audio Delay", DEFAULTS["Audio Delay"]))
        self.capture_mode_key = cfg.get("Capture Mode", DEFAULTS["Capture Mode"])
        cm_t = UI_TEXTS[self.language]
        self.capture_mode_dd.value = cm_t["Monitor"] if self.capture_mode_key == "Monitor" else cm_t["Window"]
        self._sync_capture_mode_visibility()
        self._apply_stereo_output(cfg)
        self.update_tensorrt_visibility_based_on_model(selected_model)
        self.update_coreml_visibility_based_on_model(selected_model)
        self.update_openvino_visibility_based_on_model(selected_model)
        self.update_ui_texts()
        self._sync_visibility()
        self.update_stream_url()
        self.on_device_change(None)
        self.on_capture_tool_change(None)

    def _apply_stereo_output(self, cfg):
        mon_count = self._get_monitor_count()
        if mon_count <= 1:
            self.stereo_monitor_dd.value = "Viewer Window"
            return
        saved = cfg.get("Stereo Output")
        input_label = self.monitor_dd.value if self.capture_mode_key == "Monitor" else None
        if saved is not None:
            label = next((lbl for lbl, i in self.monitor_label_to_index.items() if i == saved), None)
            if label and label != input_label:
                self.stereo_monitor_dd.value = label
                return
        fallback = None
        for lbl in self.monitor_label_to_index:
            if lbl != input_label:
                fallback = lbl
                break
        self.stereo_monitor_dd.value = fallback if fallback else "Viewer Window"

    def _get_monitor_count(self):
        try:
            import mss
            with mss.mss() as sct:
                return len(sct.monitors) - 1
        except Exception:
            return 0

    def update_stereo_monitor_menu(self):
        if not hasattr(self, 'stereo_monitor_dd'):
            return
        input_label = self.monitor_dd.value if self.capture_mode_key == "Monitor" else None
        opts = ["Viewer Window"]
        for label in self.monitor_label_to_index:
            if label != input_label:
                opts.append(label)
        current = self.stereo_monitor_dd.value
        valid = current in opts
        self.stereo_monitor_dd.options = opts
        if not valid:
            self.stereo_monitor_dd.value = opts[0] if opts else "Viewer Window"
        self.stereo_monitor_dd.update()

    def update_depth_resolution_options(self, model_name):
        resolutions = ALL_MODELS.get(model_name, {}).get("resolutions", [322])
        self.depth_res_dd.options = [str(r) for r in resolutions]
        cur = self.depth_res_dd.value
        if cur and cur in [str(r) for r in resolutions]:
            return
        try:
            cur_num = int(cur) if cur else 0
            closest = min(resolutions, key=lambda x: abs(x - cur_num))
            self.depth_res_dd.value = str(closest)
        except (ValueError, TypeError):
            self.depth_res_dd.value = str(resolutions[0])
        self.depth_res_dd.update()

    @property
    def current_model_name(self):
        family = self.depth_model_dd.value
        size = self.model_size_dd.value
        return FAMILY_SIZE_TO_MODEL.get((family, size), family if not size else f"{family}-{size}")

    def on_model_family_change(self, e):
        family = e.control.value
        sizes = FAMILY_TO_SIZES.get(family, [])
        self.model_size_dd.options = [s for s in sizes]
        if sizes:
            self.model_size_dd.value = sizes[0]
        else:
            self.model_size_dd.value = ""
        self.model_size_dd.update()
        self._on_model_changed()

    def on_model_size_change(self, e):
        self._on_model_changed()

    def _on_model_changed(self):
        model = self.current_model_name
        self._config["Depth Model"] = model
        self.update_depth_resolution_options(model)
        self.auto_enable_optimizers_based_on_device()
        if not IS_ROCM and "CUDA" in self.device_dd.value:
            self.update_tensorrt_visibility_based_on_model(model)
        elif "MPS" in self.device_dd.value:
            self.update_coreml_visibility_based_on_model(model)
        elif "XPU" in self.device_dd.value:
            self.update_openvino_visibility_based_on_model(model)
        self._fit_window_to_content()

    def on_device_change(self, e):
        device_label = e.control.value if e else self.device_dd.value
        self._config["Computing Device"] = self.device_label_to_index.get(device_label, 0)
        if OS_NAME in ("Windows", "Darwin"):
            new_opts = _get_capture_tool_options(device_label)
            self.capture_tool_dd.options = [o for o in new_opts]
            if self.capture_tool_dd.value not in new_opts:
                self.capture_tool_dd.value = new_opts[0]
            self.capture_tool_dd.update()
        self._update_accelerator_visibility(device_label)
        self.auto_enable_optimizers_based_on_device()
        self._fit_window_to_content()

    def _platform_accelerator_values(self, device_label=None, use_control_values=True):
        device_label = device_label or self.device_dd.value or ""
        model_lower = (self.current_model_name or "").lower()
        values = {
            "TensorRT": None,
            "CoreML": None,
            "OpenVINO": None,
            "MIGraphX": None,
        }
        recompile_values = {
            "Recompile TensorRT": False,
            "Recompile CoreML": False,
            "Recompile OpenVINO": False,
            "Recompile MIGraphX": False,
        }

        active_key = None
        recompile_key = None
        active_cb = None
        recompile_cb = None
        disabled_by_model = False

        if "CUDA" in device_label and IS_ROCM:
            active_key = "MIGraphX"
            recompile_key = "Recompile MIGraphX"
            active_cb = self.migraphx_cb
            recompile_cb = self.recompile_migraphx_cb
            disabled_by_model = any(kw in model_lower for kw in DISABLE_MIGRAPHX_KEYWORDS)
        elif "CUDA" in device_label:
            active_key = "TensorRT"
            recompile_key = "Recompile TensorRT"
            active_cb = self.tensorrt_cb
            recompile_cb = self.recompile_trt_cb
            disabled_by_model = any(kw in model_lower for kw in DISABLE_TRT_KEYWORDS)
        elif "MPS" in device_label:
            active_key = "CoreML"
            recompile_key = "Recompile CoreML"
            active_cb = self.coreml_cb
            recompile_cb = self.recompile_coreml_cb
            disabled_by_model = any(kw in model_lower for kw in DISABLE_COREML_KEYWORDS)
        elif "XPU" in device_label:
            active_key = "OpenVINO"
            recompile_key = "Recompile OpenVINO"
            active_cb = self.openvino_cb
            recompile_cb = self.recompile_openvino_cb
            disabled_by_model = any(kw in model_lower for kw in DISABLE_OPENVINO_KEYWORDS)

        if active_key is not None:
            if disabled_by_model:
                enabled = False
            elif use_control_values:
                enabled = bool(active_cb.value)
            else:
                saved_value = self._config.get(active_key)
                enabled = (saved_value is None) or bool(saved_value)

            values[active_key] = enabled
            if enabled:
                recompile_values[recompile_key] = bool(recompile_cb.value)

        return values, recompile_values

    def _apply_platform_accelerator_policy(self, update_controls=True):
        values, recompile_values = self._platform_accelerator_values(use_control_values=False)
        self._config.update(values)
        self._config.update(recompile_values)

        if update_controls:
            self.tensorrt_cb.value = bool(values["TensorRT"])
            self.coreml_cb.value = bool(values["CoreML"])
            self.openvino_cb.value = bool(values["OpenVINO"])
            self.migraphx_cb.value = bool(values["MIGraphX"])
            self.recompile_trt_cb.value = recompile_values["Recompile TensorRT"]
            self.recompile_coreml_cb.value = recompile_values["Recompile CoreML"]
            self.recompile_openvino_cb.value = recompile_values["Recompile OpenVINO"]
            self.recompile_migraphx_cb.value = recompile_values["Recompile MIGraphX"]

        self._on_trt_toggle(None)
        self._on_coreml_toggle(None)
        self._on_openvino_toggle(None)
        self._on_migraphx_toggle(None)

    def _update_accelerator_visibility(self, device_label):
        cuda = "CUDA" in device_label
        dml = "DirectML" in device_label
        mps = "MPS" in device_label
        xpu = "XPU" in device_label
        other = not (cuda or dml or mps or xpu)
        self.torch_compile_cb.visible = cuda
        self.tensorrt_cb.visible = cuda and not IS_ROCM
        self.tensorrt_cb.disabled = False
        self.recompile_trt_cb.visible = self.tensorrt_cb.value if self.tensorrt_cb.visible else False
        self.coreml_cb.visible = mps
        self.coreml_cb.disabled = False
        self.recompile_coreml_cb.visible = self.coreml_cb.value if self.coreml_cb.visible else False
        self.openvino_cb.visible = xpu
        self.openvino_cb.disabled = False
        self.recompile_openvino_cb.visible = self.openvino_cb.value if self.openvino_cb.visible else False
        rocm = cuda and IS_ROCM
        self.migraphx_cb.visible = rocm
        self.migraphx_cb.disabled = False
        self.recompile_migraphx_cb.visible = self.migraphx_cb.value if self.migraphx_cb.visible else False
        self.fp16_cb.visible = not (dml or mps)
        self.r4_label.visible = not (dml or other)
        is_windows_or_mac = OS_NAME in ("Windows", "Darwin")
        self.r6_label.visible = is_windows_or_mac
        self.capture_tool_dd.visible = is_windows_or_mac
        if dml or other:
            self.torch_compile_cb.visible = False
            self.tensorrt_cb.visible = False
            self.recompile_trt_cb.visible = False
            self.coreml_cb.visible = False
            self.recompile_coreml_cb.visible = False
            self.openvino_cb.visible = False
            self.recompile_openvino_cb.visible = False
            self.migraphx_cb.visible = False
            self.recompile_migraphx_cb.visible = False
        current_model = self.current_model_name
        if cuda and not IS_ROCM:
            self.update_tensorrt_visibility_based_on_model(current_model)
        if mps:
            self.update_coreml_visibility_based_on_model(current_model)
        if xpu:
            self.update_openvino_visibility_based_on_model(current_model)

    def _on_trt_toggle(self, e):
        if e is not None:
            self._config["TensorRT"] = bool(self.tensorrt_cb.value)
        self.recompile_trt_cb.visible = self.tensorrt_cb.value
        self._fit_window_to_content()

    def _on_coreml_toggle(self, e):
        if e is not None:
            self._config["CoreML"] = bool(self.coreml_cb.value)
        self.recompile_coreml_cb.visible = self.coreml_cb.value
        self._fit_window_to_content()

    def _on_openvino_toggle(self, e):
        if e is not None:
            self._config["OpenVINO"] = bool(self.openvino_cb.value)
        self.recompile_openvino_cb.visible = self.openvino_cb.value
        self._fit_window_to_content()

    def _on_migraphx_toggle(self, e):
        if e is not None:
            self._config["MIGraphX"] = bool(self.migraphx_cb.value)
        self.recompile_migraphx_cb.visible = self.migraphx_cb.value
        self._fit_window_to_content()

    def _on_torch_compile_toggle(self, e):
        if e is not None:
            self._config["torch.compile"] = bool(self.torch_compile_cb.value)

    def _reset_recompile_flags(self):
        for cb in (
            self.recompile_trt_cb,
            self.recompile_coreml_cb,
            self.recompile_openvino_cb,
            self.recompile_migraphx_cb,
        ):
            cb.value = False

        for key in ("Recompile TensorRT", "Recompile CoreML", "Recompile OpenVINO", "Recompile MIGraphX"):
            self._config[key] = False

        settings_path = os.path.join(BASE_DIR, "settings.yaml")
        disk_cfg = read_yaml(settings_path) if os.path.exists(settings_path) else {}
        if not disk_cfg:
            disk_cfg = self._config.copy()
        for key in ("Recompile TensorRT", "Recompile CoreML", "Recompile OpenVINO", "Recompile MIGraphX"):
            disk_cfg[key] = False
        save_yaml(settings_path, disk_cfg)
        self._safe_update(
            self.recompile_trt_cb,
            self.recompile_coreml_cb,
            self.recompile_openvino_cb,
            self.recompile_migraphx_cb,
        )

    async def _reset_recompile_flags_after(self, seconds=3.0):
        await asyncio.sleep(seconds)
        self._reset_recompile_flags()

    def auto_enable_optimizers_based_on_device(self):
        if "CUDA" in (self.device_dd.value or "") and not IS_ROCM and self._config.get("torch.compile") is None:
            self.torch_compile_cb.value = True
        self._apply_platform_accelerator_policy()

    def update_tensorrt_visibility_based_on_model(self, model_name):
        if not model_name:
            return
        if not IS_ROCM and "CUDA" in self.device_dd.value:
            model_lower = model_name.lower()
            should_disable = any(kw in model_lower for kw in DISABLE_TRT_KEYWORDS)
            if should_disable:
                self.tensorrt_cb.value = False
                self.tensorrt_cb.disabled = True
                self.recompile_trt_cb.visible = False
            else:
                self.tensorrt_cb.disabled = False

    def update_coreml_visibility_based_on_model(self, model_name):
        if not model_name:
            return
        if "MPS" in self.device_dd.value:
            model_lower = model_name.lower()
            should_disable = any(kw in model_lower for kw in DISABLE_COREML_KEYWORDS)
            if should_disable:
                self.coreml_cb.value = False
                self.coreml_cb.disabled = True
                self.recompile_coreml_cb.visible = False
            else:
                self.coreml_cb.disabled = False

    def update_openvino_visibility_based_on_model(self, model_name):
        if not model_name:
            return
        if "XPU" in self.device_dd.value:
            model_lower = model_name.lower()
            should_disable = any(kw in model_lower for kw in DISABLE_OPENVINO_KEYWORDS)
            if should_disable:
                self.openvino_cb.value = False
                self.openvino_cb.disabled = True
                self.recompile_openvino_cb.visible = False
            else:
                self.openvino_cb.disabled = False

    def on_capture_tool_change(self, e):
        tool = e.control.value if e else self.capture_tool_dd.value
        if tool == "DesktopDuplication":
            self.capture_mode_key = "Monitor"
            self.capture_mode_dd.value = UI_TEXTS[self.language]["Monitor"]
            self.capture_mode_dd.disabled = True
            self.set_status(UI_TEXTS[self.language]["DesktopDuplication selected: Window capture mode disabled."])
        else:
            self.capture_mode_dd.disabled = False
            self.set_status("", key="")
        self.capture_mode_dd.update()
        self._sync_capture_mode_visibility()

    def _sync_capture_mode_visibility(self):
        """Show monitor_dd or window_dd based on capture_mode_key."""
        if self.capture_mode_key == "Monitor":
            self.monitor_dd.visible = True
            self.window_dd.visible = False
        else:
            self.monitor_dd.visible = False
            self.window_dd.visible = True
        self._safe_update(self.monitor_dd, self.window_dd)
        self._fit_window_to_content()

    def on_capture_mode_change(self, e):
        mode = e.control.value
        texts = UI_TEXTS[self.language]
        reverse_map = {texts["Monitor"]: "Monitor", texts["Window"]: "Window"}
        self.capture_mode_key = reverse_map.get(mode, "Monitor")
        self._sync_capture_mode_visibility()
        if self.capture_mode_key == "Window":
            self.refresh_window_list()
        self.update_stereo_monitor_menu()
        self._fit_window_to_content()

    def _load_env_folder_display_names(self):
        """Walk environment/<folder>/profile.json for each folder and cache
        its display_name dict, e.g. {"EN": "Bedroom", "CN": "卧室"}.

        Folders with no profile, an empty profile, or no display_name field
        are cached as an empty dict — _env_display_label falls back to the
        raw folder name in that case. A bare string display_name is treated
        as a universal label for all languages.
        """
        import json
        self._env_folder_display_cache.clear()
        for folder in self._env_folder_keys:
            names = {}
            try:
                path = os.path.join(self._env_base, folder, "profile.json")
                if os.path.isfile(path):
                    with open(path, "r", encoding="utf-8") as f:
                        raw = f.read().strip()
                    if raw:  # tolerate empty file
                        prof = json.loads(raw) or {}
                        dn = prof.get("display_name")
                        if isinstance(dn, dict):
                            names = {str(k): str(v) for k, v in dn.items() if v}
                        elif isinstance(dn, str) and dn:
                            names = {"EN": dn, "CN": dn}
            except (OSError, ValueError):
                # Hand-edited JSON breakage must never crash the GUI.
                pass
            self._env_folder_display_cache[folder] = names

    def _env_display_label(self, key, lang):
        """Return the human-readable label for a canonical env key in the
        given language. Built-ins resolve via UI_TEXTS; folder names resolve
        via the per-folder display_name cache (EN fallback, then raw name).
        """
        if key in self._env_builtin_keys:
            return UI_TEXTS.get(lang, {}).get(key, key)
        if key in self._env_folder_keys:
            names = self._env_folder_display_cache.get(key) or {}
            return names.get(lang) or names.get("EN") or key
        return key

    def _build_env_dd_options(self, lang):
        """Build the localized dropdown option list (built-ins first, then
        folders) for the given language. Used by both the initial dropdown
        construction and the runtime language toggle in update_ui_texts.
        """
        return ([self._env_display_label(k, lang) for k in self._env_builtin_keys]
                + [self._env_display_label(k, lang) for k in self._env_folder_keys])

    def on_env_change(self, e):
        """Map the localized display label back to the canonical English env
        key so settings.yaml and the xrviewer matcher stay language-agnostic.
        Both built-in CN labels ("默认") and folder CN labels ("卧室")
        round-trip to their English keys ("Default", "Bedroom").
        """
        label = e.control.value if e else self.env_dd.value
        canonical = list(self._env_builtin_keys) + list(self._env_folder_keys)
        reverse_map = {self._env_display_label(k, self.language): k for k in canonical}
        if label in reverse_map:
            self.env_key = reverse_map[label]
        else:
            # Last-chance fallback: case-insensitive match against canonical
            # English keys (handles hand-edited values and English labels
            # even when the GUI is currently displaying CN).
            match = next((k for k in canonical if str(k).lower() == str(label).lower()), None)
            self.env_key = match if match is not None else "Default"
        self._config["Environment Model"] = self.env_key

    def on_window_selected(self, e):
        label = e.control.value if e else self.window_dd.value
        # Extract handle from label: "Title [h:123456]"
        m = re.search(r'\[h:(\d+)\]$', label)
        target_handle = int(m.group(1)) if m else None
        # Extract clean title (strip suffix)
        display_title = re.sub(r'\s*\[h:\d+\]$', '', label).strip()
        self.selected_window_name = display_title
        for win in self._window_objects:
            wh = win.get("handle") or 0
            if target_handle is not None and wh == target_handle:
                self.selected_window_handle = win.get("handle")
                self.selected_window_rect = win.get("rect")
                break
            elif target_handle is None and win["title"] == display_title:
                self.selected_window_handle = win.get("handle")
                self.selected_window_rect = win.get("rect")
                break
        self._fit_window_to_content()

    def _on_monitor_change(self, e):
        self.update_stereo_monitor_menu()
        self._fit_window_to_content()
        self.set_status(f"{UI_TEXTS[self.language]['Selected input monitor:']} {e.control.value}")

    def on_run_mode_change(self, e):
        label = e.control.value
        texts = UI_TEXTS[self.language]
        mode_map = {
            texts.get("Local Viewer", "Local Viewer"): "Local Viewer",
            texts.get("OpenXR Link", "OpenXR Link"): "OpenXR Link",
            texts.get("RTMP Streamer", "RTMP Streamer"): "RTMP Streamer",
            texts.get("MJPEG Streamer", "MJPEG Streamer"): "MJPEG Streamer",
            texts.get("Legacy Streamer", "Legacy Streamer"): "Legacy Streamer",
            texts.get("3D Monitor", "3D Monitor"): "3D Monitor",
        }
        self.run_mode_key = mode_map.get(label, "Local Viewer")
        self._config["Run Mode"] = self.run_mode_key
        self._sync_visibility()

    def on_crop_mode_change(self, e):
        label = e.control.value
        texts = UI_TEXTS[self.language]
        label_map = {
            texts.get("Crop Auto", "Auto"): "auto",
            texts.get("Crop Manual", "Manual"): "manual",
            texts.get("Crop Off", "Off"): "off",
        }
        self.crop_mode_key = label_map.get(label, "auto")
        self._config["Crop Mode"] = self.crop_mode_key

    def _sync_visibility(self):
        mode = self.run_mode_key
        texts = UI_TEXTS[self.language]
        mode_reverse = {
            "Local Viewer": texts["Local Viewer"],
            "OpenXR Link": texts["OpenXR Link"],
            "RTMP Streamer": texts["RTMP Streamer"],
            "MJPEG Streamer": texts["MJPEG Streamer"],
            "Legacy Streamer": texts["Legacy Streamer"],
            "3D Monitor": texts["3D Monitor"],
        }
        self.run_mode_dd.value = mode_reverse.get(mode, texts["Local Viewer"])
        is_openxr = mode == "OpenXR Link"
        self.xr_preview_cb.visible = is_openxr
        self.crop_mode_label.visible = is_openxr
        self.crop_mode_dd.visible = is_openxr
        self.vsync_cb.visible = mode in ["Local Viewer", "3D Monitor"]
        self._xr_preview_spacer.visible = is_openxr
        self.r7b_label.visible = not is_openxr
        self.display_mode_dd.visible = not is_openxr
        self.row7b.visible = is_openxr
        if is_openxr:
            self.showfps_cb.visible = False
            self.fill_16_9_cb.visible = False
            self.fix_aspect_cb.visible = False
        else:
            self.showfps_cb.visible = True
            self.fill_16_9_cb.visible = True
            self.fix_aspect_cb.visible = mode in ["Local Viewer", "3D Monitor"]
        if mode in ["Legacy Streamer", "3D Monitor"]:
            self.display_mode_dd.options = [m for m in ["Half-SBS", "Full-SBS", "Half-TAB", "Full-TAB"]]
        else:
            self.display_mode_dd.options = [m for m in ["Half-SBS", "Full-SBS", "Half-TAB", "Full-TAB", "Interleaved", "Interleaved-V", "Anaglyph", "Depth Map"]]
        mon_count = self._get_monitor_count()
        stereo_full = mode in ["Local Viewer", "3D Monitor", "RTMP Streamer"] and mon_count > 1
        self.r9_label.visible = not is_openxr
        self.stereo_monitor_dd.visible = stereo_full
        if hasattr(self, '_stereo_spacer'):
            self._stereo_spacer.visible = stereo_full
        if mode == "3D Monitor":
            self.stereo_monitor_dd.value = self.monitor_dd.value
        row_map = self._get_streamer_row_map()
        indices = row_map.get(mode, [])
        self._show_streamer_rows(*indices)
        self.lossless_cb.visible = (OS_NAME == "Windows" and mode == "RTMP Streamer")
        self.update_stereo_monitor_menu()
        self._fit_window_to_content()
        if mode in ["Local Viewer", "RTMP Streamer"]:
            self._auto_select_stereo_monitor()
        if mode == "RTMP Streamer":
            self.populate_audio_devices()
            self.auto_select_stereo_mix()
            saved_mix = self._config.get("Stereo Mix", "")
            if saved_mix and saved_mix in (self.audio_dd.options or []):
                self.audio_dd.value = saved_mix
        self.update_stream_url()
        self._fit_window_to_content()
        self.page.update()

    def _auto_select_stereo_monitor(self):
        mon_count = self._get_monitor_count()
        if mon_count <= 1:
            return
        cur = self.stereo_monitor_dd.value
        valid = cur and cur in self.stereo_monitor_dd.options and cur != "Viewer Window"
        if not valid:
            input_label = self.monitor_dd.value if self.capture_mode_key == "Monitor" else None
            for lbl in self.monitor_label_to_index:
                if lbl != input_label:
                    self.stereo_monitor_dd.value = lbl
                    break

    def on_theme_change(self, e):
        color = e.control.value
        cn_map = {"系统":"system","蓝色":"blue","绿色":"green","红色":"red","紫色":"purple","橙色":"orange","青色":"teal","粉色":"pink","灰色":"grey"}
        color = cn_map.get(color, color.lower())
        if OS_NAME == "Windows":
            font = "Microsoft YaHei"
        elif OS_NAME == "Darwin":
            font = "PingFang SC"
        else:
            font = "Noto Sans SC"
        if color == "system":
            self.page.theme_mode = ft.ThemeMode.SYSTEM
            self.page.theme = ft.Theme(color_scheme_seed="blue", font_family=font)
        else:
            self.page.theme = ft.Theme(color_scheme_seed=color, font_family=font)
        self.page.update()

    def on_language_change(self, e):
        lang_display = e.control.value
        _LANG_MAP = {"English": "EN", "简体中文": "CN"}
        lang = _LANG_MAP.get(lang_display, "EN")
        if lang in UI_TEXTS:
            self.language = lang
            self._config["Language"] = lang
            self.update_ui_texts()
            self._sync_visibility()
            if self._status_key:
                self.set_status(UI_TEXTS[self.language][self._status_key], key=self._status_key)
            # Theme: update_ui_texts already set options, here we only set the value
            t = UI_TEXTS[self.language]
            cur = self.theme_dd.value
            cn_map = {"系统":"system","蓝色":"blue","绿色":"green","红色":"red","紫色":"purple","橙色":"orange","青色":"teal","粉色":"pink","灰色":"grey"}
            key = cn_map.get(cur, cur.lower() if cur else "system")
            self.theme_dd.value = t.get(key, key) if self.language == "CN" else key.capitalize()
            self._fit_window_to_content()
            self.page.update()

    def update_ui_texts(self):
        t = UI_TEXTS[self.language]
        self.r0_label.value = t["Depth Model:"]
        self.fp16_cb.label = t["FP16"]
        self.r1a_label.value = t["Depth Resolution:"]
        self.r1b_label.value = t["Convergence:"]
        self.r2a_label.value = t["Depth Strength:"]
        self.r2b_label.value = t["Foreground Scale:"]
        self.r3a_label.value = t["Anti-aliasing:"]
        self.r3b_label.value = t["IPD (m):"]
        self.r4_label.value = t["Inference Acceleration:"]
        self.torch_compile_cb.label = t["torch.compile"]
        self.tensorrt_cb.label = t["TensorRT"]
        self.coreml_cb.label = t["CoreML"]
        self.openvino_cb.label = t["OpenVINO"]
        self.recompile_trt_cb.label = t["Recompile TensorRT"]
        self.recompile_coreml_cb.label = t["Recompile CoreML"]
        self.recompile_openvino_cb.label = t["Recompile OpenVINO"]
        self.migraphx_cb.label = t["MIGraphX"]
        self.recompile_migraphx_cb.label = t["Recompile MIGraphX"]
        self.r5_label.value = t["Computing Device:"]
        self.showfps_cb.label = t["Show FPS"]
        self.vsync_cb.label = t.get("VSync", "VSync")
        self.r6_label.value = t["Capture Tool:"]
        self.r7a_label.value = t["Run Mode:"]
        self.r7b_label.value = t["Display Mode:"]
        self.xr_preview_cb.label = t.get("XR Preview", "XR Preview")
        self.crop_mode_dd.options = [t.get("Crop Auto", "Auto"),
                                     t.get("Crop Manual", "Manual"),
                                     t.get("Crop Off", "Off")]
        self.crop_mode_dd.value = t.get({"auto": "Crop Auto", "manual": "Crop Manual", "off": "Crop Off"}[self.crop_mode_key], self.crop_mode_key)
        self.r9_label.value = t["Stereo Output:"]
        self.theme_label.value = t["Theme:"]
        theme_display = ["System","Blue","Green","Red","Purple","Orange","Teal","Pink","Grey"]
        self.theme_dd.options = [t.get(k.lower(), k) for k in theme_display]
        cur = self.theme_dd.value
        cn_map = {"系统":"system","蓝色":"blue","绿色":"green","红色":"red","紫色":"purple","橙色":"orange","青色":"teal","粉色":"pink","灰色":"grey"}
        key = cn_map.get(cur, cur.lower() if cur else "system")
        self.theme_dd.value = t.get(key, key) if self.language == "CN" else key.capitalize()
        self.fill_16_9_cb.label = t["Fill 16:9"]
        self.fix_aspect_cb.label = t["Fix Viewer Aspect"]
        self.lossless_cb.label = t["Lossless Scaling Support"]
        self.r10_label.value = t["Controller:"]
        self.crop_mode_label.value = t["Crop Mode:"]
        self.r11_label.value = t["Environment:"]
        # Environment: localize built-in option labels (Default / Default with Glow /
        # Dark Room) AND folder labels via each folder's profile.json
        # "display_name" field, while keeping folder names verbatim as the
        # canonical key. self.env_key is the canonical English key that
        # survives save/load and reaches xrviewer.
        self.env_dd.options = self._build_env_dd_options(self.language)
        self.env_dd.value = self._env_display_label(self.env_key, self.language)
        self.lang_label.value = t["Set Language:"]
        run_mode_texts = {}
        if OS_NAME == "Darwin":
            run_mode_texts = {
                "Local Viewer": t["Local Viewer"],
                "RTMP Streamer": t["RTMP Streamer"],
                "MJPEG Streamer": t["MJPEG Streamer"],
                "Legacy Streamer": t["Legacy Streamer"],
            }
        else:
            run_mode_texts = {
                "Local Viewer": t["Local Viewer"],
                "OpenXR Link": t["OpenXR Link"],
                "RTMP Streamer": t["RTMP Streamer"],
                "MJPEG Streamer": t["MJPEG Streamer"],
                "Legacy Streamer": t["Legacy Streamer"],
            }
            if OS_NAME == "Windows":
                run_mode_texts["3D Monitor"] = t["3D Monitor"]
        self.run_mode_dd.options = [v for v in run_mode_texts.values()]
        self.capture_mode_dd.options = [
            t["Monitor"],
            t["Window"],
        ]
        self.capture_mode_dd.value = t["Monitor"] if self.capture_mode_key == "Monitor" else t["Window"]
        self.stream_url_label.value = t["Streamer URL"]
        self.stream_port_label.value = t["Streamer Port:"]
        self.stream_quality_label.value = t["Stream Quality:"]
        self.stream_proto_label.value = t["Stream Protocol:"]
        self.stream_key_label.value = t["Stream Key"]
        self.audio_label.value = t["Stereo Mix"]
        self.crf_label.value = t["CRF"]
        self.audio_delay_label.value = t["Audio Delay"]
        self.preview_btn.content.value = t["Preview"]
        self.refresh_btn.content.value = t["Refresh"]
        self.reset_btn.content.value = t["Reset"]
        self.open_log_btn.content.value = t["Log Folder"]
        self.open_log_btn.tooltip = t["tooltip_open_log"]
        self.stop_btn.content.value = t["Stop"]
        self.run_btn.content.value = t["Run"]
        # Update tooltips
        self.window_dd.set_tooltip(t["tooltip_window"])
        for ctrl, key in [
            (self.depth_model_dd, "tooltip_depth_model"),
            (self.model_size_dd, "tooltip_model_size"),
            (self.depth_res_dd, "tooltip_depth_res"),
            (self.convergence_dd, "tooltip_convergence"),
            (self.depth_strength_dd, "tooltip_depth_strength"),
            (self.foreground_scale_dd, "tooltip_foreground_scale"),
            (self.antialiasing_dd, "tooltip_antialiasing"),
            (self.ipd_dd, "tooltip_ipd"),
            (self.device_dd, "tooltip_device"),
            (self.capture_tool_dd, "tooltip_capture_tool"),
            (self.run_mode_dd, "tooltip_run_mode"),
            (self.display_mode_dd, "tooltip_display_mode"),
            (self.crop_mode_dd, "tooltip_crop_mode"),
            (self.ctrl_model_dd, "tooltip_ctrl_model"),
            (self.env_dd, "tooltip_env_model"),
            (self.capture_mode_dd, "tooltip_capture_mode"),
            (self.monitor_dd, "tooltip_monitor"),
            (self.stereo_monitor_dd, "tooltip_stereo_monitor"),
            (self.lang_dd, "tooltip_lang"),
            (self.theme_dd, "tooltip_theme"),
            (self.stream_quality_dd, "tooltip_stream_quality"),
            (self.stream_proto_dd, "tooltip_stream_proto"),
            (self.audio_dd, "tooltip_audio"),
            (self.stream_port_tf, "tooltip_stream_port"),
            (self.stream_key_tf, "tooltip_stream_key"),
            (self.crf_tf, "tooltip_crf"),
            (self.audio_delay_tf, "tooltip_audio_delay"),
        ]:
            ctrl.set_tooltip(t[key])
        self.vsync_cb.tooltip = t.get("tooltip_vsync", "")
        self._auto_align_labels()

    def _safe_update(self, *controls):
        for c in controls:
            try:
                c.update()
            except RuntimeError:
                pass
            except Exception as e:
                print(f"[Warning] _safe_update failed: {e}")

    def update_stream_url(self, e=None):
        if not self.stream_container.visible:
            return
        protocol = self.stream_proto_dd.value
        port = self.stream_port_tf.value or str(DEFAULT_PORT)
        stream_key = self.stream_key_tf.value or "live"
        local_ip = get_local_ip()
        if self.run_mode_key in ["MJPEG Streamer", "Legacy Streamer"]:
            self.stream_url_tf.content.controls[0].value = f"http://{local_ip}:{port}/"
        else:
            templates = {
                "RTMP": f"rtmp://{local_ip}:{port}/{stream_key}",
                "RTSP": f"rtsp://{local_ip}:{port}/{stream_key}",
                "HLS": f"http://{local_ip}:{port}/{stream_key}/",
                "HLS M3U8": f"http://{local_ip}:{port}/{stream_key}/index.m3u8",
                "WebRTC": f"http://{local_ip}:{port}/{stream_key}/",
            }
            self.stream_url_tf.content.controls[0].value = templates.get(protocol, f"http://{local_ip}:{port}/{stream_key}/")
        self._safe_update(self.stream_url_tf)
        self.preview_btn.visible = protocol not in ["RTMP", "RTSP"]
        self._safe_update(self.preview_btn)

    def _on_stream_protocol_change(self, e):
        self.stream_protocol_key = e.control.value
        self._config["Stream Protocol"] = self.stream_protocol_key
        self.update_stream_url()
        self._fit_window_to_content()

    def _on_stream_key_change(self, e):
        val = e.control.value or ""
        if not re.match(r'^[A-Za-z0-9_-]*$', val) or len(val) > 64:
            self.set_status(UI_TEXTS[self.language]["err_stream_key"])
        self._config["Stream Key"] = val
        self.update_stream_url()

    def populate_audio_devices(self):
        if OS_NAME == "Linux":
            self._populate_audio_linux()
        else:
            self._populate_audio_generic()
        if self.audio_devices:
            self.audio_dd.options = [d for d in self.audio_devices]
            self.audio_dd.value = self.audio_devices[0]
            self.audio_dd.update()
            if self.audio_devices[0] in ["No Stereo Mix device found", "sounddevice not available"]:
                self.set_status(self.audio_devices[0])

    def _populate_audio_generic(self):
        """Use sounddevice to find Stereo Mix / loopback devices."""
        self.audio_devices = []
        try:
            import sounddevice as sd
            all_devices = sd.query_devices()
            found = set()
            for dev in all_devices:
                name = (dev.get("name", "") or "").lower()
                in_ch = dev.get("max_input_channels", 0)
                out_ch = dev.get("max_output_channels", 0)
                if in_ch > 0 or out_ch > 0:
                    for mix in STEREO_MIX_NAMES:
                        if mix in name and "audio stereo input" not in name:
                            found.add(dev.get("name"))
                            break
                    if "virtual-audio-capturer" in name:
                        found.add(dev.get("name"))
            if not found and OS_NAME == "Darwin":
                print("[Info] No audio capture devices found on MacOS.\nRecommended tools:\n- BlackHole: https://existential.audio/blackhole/\n- Virtual Desktop Streamer: https://www.vrdesktop.net/\n- Loopback: https://rogueamoeba.com/loopback/")
                self.audio_devices = ["No audio capture devices found"]
            elif not found and OS_NAME == "Windows":
                print("[Warning] No Stereo Mix devices found, please enable it in audio settings.\nIf no Stereo Mix, install 'Screen Capture Recorder':\nhttps://github.com/rdp/screen-capture-recorder-to-video-windows-free/releases")
                self.audio_devices = ["virtual-audio-capturer"]
            else:
                self.audio_devices = list(found) or ["No Stereo Mix device found"]
        except ImportError:
            self.audio_devices = ["sounddevice not available"]
        except Exception as e:
            self.audio_devices = [f"Error: {e}"]

    def _populate_audio_linux(self):
        self.audio_devices = []
        try:
            result = subprocess.run(["pacmd", "list-sources"],
                                    capture_output=True, text=True, check=True)
            sources = []
            for block in result.stdout.split("index:")[1:]:
                m = re.search(r"name:\s*<(.+?)>", block)
                if m:
                    sources.append(m.group(1))
            self.audio_devices = sources or ["No audio sources found"]
        except Exception:
            self.audio_devices = ["pacmd not available"]

    def auto_select_stereo_mix(self):
        """Auto-select the first stereo-mix-like device."""
        if not self.audio_devices:
            return
        for dev in self.audio_devices:
            dl = dev.lower()
            for mix in STEREO_MIX_NAMES:
                if mix in dl and "audio stereo input" not in dl:
                    self.audio_dd.value = dev
                    self.audio_dd.update()
                    return
        # Fallback: virtual-audio-capturer
        for dev in self.audio_devices:
            if "virtual-audio-capturer" in dev.lower():
                self.audio_dd.value = dev
                self.audio_dd.update()
                return
        self.audio_dd.value = "No Stereo Mix device found"
        self.audio_dd.update()

    def refresh_monitor_and_window(self, e=None):
        self.populate_monitors()
        if self.capture_mode_key == "Window":
            self.refresh_window_list()
        if self.run_mode_key == "RTMP Streamer":
            self.populate_audio_devices()
            self.auto_select_stereo_mix()
        self.update_stereo_monitor_menu()
        self._sync_visibility()
        self._fit_window_to_content()
        if self.capture_mode_key == "Monitor" and self.monitor_dd.value:
            self.set_status(f"{UI_TEXTS[self.language]['Selected input monitor:']} {self.monitor_dd.value}")
        elif self.capture_mode_key == "Window" and self.selected_window_name:
            self.set_status(f"{UI_TEXTS[self.language]['Selected input window:']} {self.selected_window_name}")

    def refresh_window_list(self):
        try:
            windows = list_windows()
            if not windows:
                self.window_dd.options = []
                self.window_dd.update()
                return
            self._window_objects = windows
            # Include handle in label to avoid duplicate title ambiguity
            win_labels = [f"{w['title']} [h:{w['handle'] or 0}]" for w in windows]
            self.window_dd.options = win_labels
            if self.selected_window_name:
                labels_by_title = [lbl for lbl in win_labels if lbl.startswith(self.selected_window_name + " [")]
                selected_handle_str = f"[h:{self.selected_window_handle or 0}]"
                match = next((lbl for lbl in labels_by_title if selected_handle_str in lbl), None)
                if match:
                    self.window_dd.value = match
                elif labels_by_title:
                    self.window_dd.value = labels_by_title[0]
                    self.on_window_selected(None)
                else:
                    self.window_dd.value = win_labels[0] if windows else ""
                    self.on_window_selected(None)
            elif windows:
                self.window_dd.value = win_labels[0]
                self.on_window_selected(None)
            self.window_dd.update()
        except Exception as e:
            self.set_status(UI_TEXTS[self.language]["err_refresh_window"].format(e))


    # URL actions

    def preview_in_browser(self, e):
        try:
            import webbrowser
            url = self.stream_url_tf.content.controls[0].value
            if not url.startswith(("http://", "https://")):
                self.set_status(UI_TEXTS[self.language]["invalid_url_scheme"].format(url))
                return
            webbrowser.open(url)
            self.set_status(f"{UI_TEXTS[self.language]['Opening URL in browser']}: {url}")
        except Exception as ex:
            self.set_status(UI_TEXTS[self.language]["error_preview"].format(ex))

    def copy_url_to_clipboard(self, e):
        url = self.stream_url_tf.content.controls[0].value
        if url:
            try:
                import pyperclip
                pyperclip.copy(url)
            except ImportError:
                import subprocess
                if OS_NAME == "Windows":
                    subprocess.run("clip", input=url, text=True, shell=True)
                elif OS_NAME == "Darwin":
                    subprocess.run("pbcopy", input=url, text=True)
            self.set_status(UI_TEXTS[self.language]["url_copied"], key="url_copied")
            asyncio.create_task(self._fade_status(2.0))

    async def _fade_status(self, delay):
        await asyncio.sleep(delay)
        self.set_status("", key="")

    def set_status(self, msg, key=None):
        self.status_text.value = msg
        if key is not None:
            self._status_key = key
        self._safe_update(self.status_text)

    def _set_running_ui(self, running: bool):
        """Toggle Run/Stop button enabled state."""
        self.run_btn.disabled = running
        self.stop_btn.disabled = not running
        self._safe_update(self.run_btn, self.stop_btn)


    # Save, Run, Stop, Reset

    def _validate_config_before_run(self):
        """Validate config before run. Returns (ok: bool, error_msg: str)."""
        try:
            port_val = int(self.stream_port_tf.value) if self.stream_port_tf.value else DEFAULT_PORT
            if not (1 <= port_val <= 65535):
                return False, UI_TEXTS[self.language]["Invalid port number (1-65535)"]
        except ValueError:
            return False, UI_TEXTS[self.language]["Invalid port number (1-65535)"]
        try:
            crf_val = int(self.crf_tf.value) if self.crf_tf.value else DEFAULTS["CRF"]
            if not (0 <= crf_val <= 51):
                return False, UI_TEXTS[self.language]["err_crf"]
        except ValueError:
            return False, UI_TEXTS[self.language]["err_crf"]
        try:
            delay_val = float(self.audio_delay_tf.value) if self.audio_delay_tf.value else DEFAULTS["Audio Delay"]
            if not (-10 <= delay_val <= 10):
                return False, UI_TEXTS[self.language]["err_audio_delay"]
        except ValueError:
            return False, UI_TEXTS[self.language]["err_audio_delay"]
        sk = self.stream_key_tf.value or "live"
        if not re.match(r'^[A-Za-z0-9_-]+$', sk) or len(sk) > 64:
            return False, UI_TEXTS[self.language]["err_stream_key"]
        if self.capture_mode_key == "Window":
            if not self.selected_window_name:
                return False, UI_TEXTS[self.language]["Please select a window before running in Window capture mode"]
            windows = list_windows()
            exists = any(
                (w.get("handle") is not None and w["handle"] == self.selected_window_handle)
                or (w.get("handle") is None and w["title"] == self.selected_window_name)
                for w in windows
            )
            if not exists:
                return False, UI_TEXTS[self.language]["The selected window no longer exists. Please refresh and select a valid window."]
        return True, ""

    def save_and_run(self, e):
        """Collect config, save YAML, start subprocess."""
        # Re-entry guard: prevent multiple clicks from launching duplicate processes
        if self._starting or (self.process and self.process.returncode is None):
            self.set_status(UI_TEXTS[self.language]["A thread already running!"])
            self.page.update()
            return
        ok, err = self._validate_config_before_run()
        if not ok:
            self.set_status(err)
            return
        self._starting = True
        self._cancel_starting = False
        self._esc_stopped = False
        self._stopping = False
        self._set_running_ui(True)

        # Build config
        self._collect_config()

        # Save YAML
        ok, err = save_yaml(os.path.join(BASE_DIR, "settings.yaml"), self._config)
        if not ok:
            self.set_status(UI_TEXTS[self.language]["failed_save_yaml"].format(err))
            self._starting = False
            self._set_running_ui(False)
            return

        self.set_status(UI_TEXTS[self.language]["Countdown"], key="Countdown")
        self.page.update()

        # Start process with countdown
        asyncio.create_task(self._countdown_and_run(0.5))

    @staticmethod
    def _parse_int(val, default):
        try:
            return int(val)
        except (ValueError, TypeError):
            return default

    @staticmethod
    def _parse_float(val, default):
        try:
            return float(val)
        except (ValueError, TypeError):
            return default

    def _collect_config(self):
        """Read all UI control values into self._config (single source of truth)."""
        if self.capture_mode_key == "Window":
            window_rect = getattr(self, 'selected_window_rect', None)
            if window_rect:
                center_x = window_rect[0] + window_rect[2] // 2
                center_y = window_rect[1] + window_rect[3] // 2
                monitor_idx = get_monitor_index_for_point(center_x, center_y)
            else:
                monitor_idx = get_primary_monitor_index()
        else:
            monitor_idx = self.monitor_label_to_index.get(self.monitor_dd.value, DEFAULTS["Monitor Index"])
        stereo_val = self.stereo_monitor_dd.value
        if stereo_val == "Viewer Window" or not stereo_val:
            stereo_idx = None
        else:
            stereo_idx = self.monitor_label_to_index.get(stereo_val, None)

        accelerator_values, recompile_values = self._platform_accelerator_values()

        self._config.update({
            "Capture Mode": self.capture_mode_key,
            "Monitor Index": monitor_idx,
            "Window Title": self.selected_window_name if self.capture_mode_key == "Window" else "",
            "Show FPS": self.showfps_cb.value,
            "IPD": self._parse_int(self.ipd_dd.value, int(DEFAULTS["IPD"] * 1000)) / 1000.0,
            "Convergence": self._parse_float(self.convergence_dd.value, DEFAULTS["Convergence"]),
            "Display Mode": self.display_mode_dd.value,
            "Model List": ALL_MODELS,
            "Depth Model": self.current_model_name,
            "Depth Strength": self._parse_float(self.depth_strength_dd.value, DEFAULTS["Depth Strength"]),
            "Anti-aliasing": self._parse_int(self.antialiasing_dd.value, DEFAULTS["Anti-aliasing"]),
            "Foreground Scale": self._parse_float(self.foreground_scale_dd.value, DEFAULTS["Foreground Scale"]),
            "Depth Resolution": self._parse_int(self.depth_res_dd.value, DEFAULTS["Depth Resolution"]),
            "FP16": self.fp16_cb.value,
            "Computing Device": self.device_label_to_index.get(self.device_dd.value, DEFAULTS["Computing Device"]),
            "Language": self.language,
            "Run Mode": self.run_mode_key,
            "XR Preview": self.xr_preview_cb.value,
            "Crop Mode": self.crop_mode_key,
            "VSync": self.vsync_cb.value,
            "Stream Protocol": self.stream_proto_dd.value,
            "Streamer Port": self._parse_int(self.stream_port_tf.value, DEFAULTS["Streamer Port"]),
            "Stream Quality": self._parse_int(self.stream_quality_dd.value, DEFAULTS["Stream Quality"]),
            "torch.compile": self.torch_compile_cb.value,
            **accelerator_values,
            **recompile_values,
            "Capture Tool": self.capture_tool_dd.value,
            "Fill 16:9": self.fill_16_9_cb.value,
            "Fix Viewer Aspect": self.fix_aspect_cb.value,
            "Lossless Scaling Support": self.lossless_cb.value,
            "Stream Key": self.stream_key_tf.value,
            "Stereo Mix": self.audio_dd.value,
            "CRF": self._parse_int(self.crf_tf.value, DEFAULTS["CRF"]),
            "Audio Delay": self._parse_float(self.audio_delay_tf.value, DEFAULTS["Audio Delay"]),
            "Stereo Output": stereo_idx,
            "Controller Model": self.ctrl_model_dd.value,
            # Persist the canonical English key (Default / <folder>) so
            # xrviewer's _init_env_model matcher and a hand-edited
            # settings.yaml stay language-agnostic.
            "Environment Model": self.env_key,
        })
        self.recompile_trt_cb.value = False
        self.recompile_coreml_cb.value = False
        self.recompile_openvino_cb.value = False
        self.recompile_migraphx_cb.value = False

    async def _countdown_and_run(self, seconds):
        self._diag("_countdown_and_run scheduled")
        try:
            if self.process and self.process.returncode is None:
                self.set_status(UI_TEXTS[self.language]["A thread already running!"])
                self._diag("already running, return")
                return
            if seconds > 0:
                await asyncio.sleep(seconds)
            if self._cancel_starting:
                self._cancel_starting = False
                self._diag("cancelled, return")
                return
            print(f"[Main] Initializing Desktop2Stereo {self.run_mode_key}...")
            shutdown_event.clear()
            try:
                if os.path.exists(STOP_REQUEST_FILE):
                    os.remove(STOP_REQUEST_FILE)
            except Exception:
                pass
            # Spawn the child with stdout+stderr captured to a PIPE so every
            # line can be forwarded *live* to (a) the user's console and
            # (b) the single rolling log file at LOG_FILE.  This restores
            # real-time visibility of the child's prints while keeping
            # exactly one log file on disk.
            #     '-u'              : unbuffered stdio so lines appear instantly
            #     '-X faulthandler' : native/GPU/driver crashes leave a Python trace
            child_args = [sys.executable, "-u", "-X", "faulthandler",
                          os.path.join(BASE_DIR, "main.py")]
            # Force the child to write its stdout/stderr as UTF-8 so the
            # parent's _pump_child_output (which decodes via .decode("utf-8"))
            # never sees mojibake from em-dashes, arrows, or other non-ASCII
            # characters used in log messages.  PYTHONIOENCODING takes effect
            # immediately at interpreter start - before any user code runs.
            child_env = os.environ.copy()
            child_env["PYTHONIOENCODING"] = "utf-8"
            if OS_NAME == "Windows":
                self.process = await asyncio.create_subprocess_exec(
                    *child_args,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.STDOUT,
                    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
                    env=child_env,
                )
            else:
                self.process = await asyncio.create_subprocess_exec(
                    *child_args,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.STDOUT,
                    start_new_session=True,
                    env=child_env,
                )
            global _CHILD_PID
            _CHILD_PID = self.process.pid
            self._diag(f"process started, pid={self.process.pid}, log={LOG_FILE}")
            # Background task that drains the child pipe and forwards each
            # line to print() — print() writes through the _TeeStream wrapper
            # so it reaches both the real terminal and the single log file.
            asyncio.create_task(self._pump_child_output(self.process))
            self.set_status(UI_TEXTS[self.language]["Running"], key="Running")
            self.page.update()
            asyncio.create_task(self._monitor_process_task())
            self._diag("monitor_task created")
            asyncio.create_task(self._reset_recompile_flags_after(3.0))
        except Exception as e:
            self._diag(f"_countdown_and_run failed:\n{traceback.format_exc()}", error=True)
            self.set_status(UI_TEXTS[self.language]["err_start_failed"].format(e))
            self.page.update()
        finally:
            self._starting = False

    async def _pump_child_output(self, proc):
        """Forward every line of the child's stdout (merged with stderr) to
        ``print()``.  Because ``sys.stdout`` is wrapped by ``_TeeStream``,
        each call appears live on the real terminal *and* gets appended to
        the single rolling log file.  Runs until the child closes its
        stdout (i.e. on exit).
        """
        try:
            stream = proc.stdout
            if stream is None:
                return
            while True:
                raw = await stream.readline()
                if not raw:
                    break
                try:
                    text = raw.decode("utf-8", errors="replace").rstrip("\r\n")
                except Exception:
                    text = repr(raw)
                if text:
                    # Tag child lines so they're easy to spot in the log,
                    # but keep them readable on the console.
                    print(text)
        except Exception as e:
            try:
                self._diag(f"_pump_child_output exception: {e}\n{traceback.format_exc()}", error=True)
            except Exception:
                pass

    def _diag(self, msg, error=False):
        """Emit a diagnostic line.

        Routine state-machine traces (process started, monitor task created,
        countdown scheduled, ...) are noise for end users, so by default this
        writes the line ONLY to the rolling log file at ``LOG_FILE`` and stays
        silent on the console.  The console is reserved for actual user-facing
        output coming from the child main process (e.g. depth.py messages,
        FPS counters, errors that the child itself raises).

        Pass ``error=True`` for genuine failure paths (uncaught exceptions in
        background tasks, non-zero child exit codes) so the line ALSO appears
        on the console - that way a "bug appears" is immediately visible
        without forcing the user to open the log file.
        """
        # 1) Always append to the single rolling log file with a timestamp +
        #    [diag] label so the full trace is available post-mortem.
        try:
            import datetime
            ts = datetime.datetime.now().strftime("%H:%M:%S")
            with open(LOG_FILE, "a", encoding="utf-8") as f:
                f.write(f"[{ts}] [diag] {msg}\n")
        except Exception:
            pass
        # 2) Only echo to console when this is an error path - keeps normal
        #    runs clean (depth.py / FPS / capture messages stay readable).
        if error:
            try:
                # Write directly to the original (unwrapped) stdout so the
                # line does NOT get re-tagged + re-appended by _TeeStream.
                _orig = getattr(sys.stdout, "original", sys.stdout)
                _orig.write(f"[diag] {msg}\n")
                _orig.flush()
            except Exception:
                pass

    async def _monitor_process_task(self):
        """Wait for process to exit, then update status."""
        proc = self.process
        if not proc:
            self._diag("monitor_task: proc is None, return")
            return
        self._diag(f"monitor_task started, pid={proc.pid}")
        try:
            await proc.wait()
            self._diag(f"proc.wait returned, rc={proc.returncode}")
        except Exception as e:
            self._diag(f"proc.wait() exception: {e}", error=True)
        finally:
            self._diag(f"finally: process is proc={self.process is proc}, returncode={proc.returncode}")
            if self.process is proc:
                self.process = None
            self._starting = False
            code = proc.returncode if proc else None
            if code and code != 0:
                # The child's full output already streamed live to the
                # console and to LOG_FILE via _pump_child_output, so no
                # tail-on-exit is needed - just point the user at the file.
                self._diag(f"child exited rc={code}; see {LOG_FILE} for details", error=True)
                self.set_status(UI_TEXTS[self.language]["exited_with_code"].format(code))
            else:
                self.set_status(UI_TEXTS[self.language]["Stopped"], key="Stopped")
            self._set_running_ui(False)
            self._reload_settings_from_disk()
            self._diag("monitor_task done, status updated")

    def _reload_settings_from_disk(self):
        """Re-read settings.yaml and sync runtime-edited dropdowns.

        Called after the xrviewer process exits so that runtime changes
        (depth, environment cycling, controller brand switching) are reflected
        back in the GUI without requiring a restart.
        """
        path = os.path.join(BASE_DIR, "settings.yaml")
        if not os.path.exists(path):
            return
        try:
            cfg = read_yaml(path)
            if not cfg:
                return
        except Exception as exc:
            self._diag(f"_reload_settings_from_disk: {exc}", error=True)
            return

        try:
            ctrl_base = os.path.join(BASE_DIR, "xr_viewer", "controllers")
            ctrl_dirs = sorted(d for d in os.listdir(ctrl_base)
                               if os.path.isdir(os.path.join(ctrl_base, d)))
        except (FileNotFoundError, OSError):
            ctrl_dirs = []
        if not ctrl_dirs:
            ctrl_dirs = ["PICO"]
        if list(self.ctrl_model_dd.options) != ctrl_dirs:
            self.ctrl_model_dd.options = ctrl_dirs

        self._env_folder_keys = _discover_gui_environment_folders(self._env_base)
        self._load_env_folder_display_names()
        self.env_dd.options = self._build_env_dd_options(self.language)

        saved_depth = cfg.get("Depth Strength")
        if saved_depth is not None:
            try:
                depth_val = float(saved_depth)
                depth_label = f"{depth_val:.4f}".rstrip("0").rstrip(".")
                if "." not in depth_label:
                    depth_label += ".0"
                self._config["Depth Strength"] = depth_val
            except (TypeError, ValueError):
                depth_label = str(saved_depth)
                self._config["Depth Strength"] = saved_depth
            if depth_label not in self.depth_strength_dd.options:
                try:
                    opts = list(self.depth_strength_dd.options) + [depth_label]
                    opts.sort(key=lambda x: float(x))
                    self.depth_strength_dd.options = opts
                except (TypeError, ValueError):
                    self.depth_strength_dd.options = list(self.depth_strength_dd.options) + [depth_label]
            self.depth_strength_dd.value = depth_label
            self._safe_update(self.depth_strength_dd)

        saved_ctrl = cfg.get("Controller Model")
        if saved_ctrl and saved_ctrl in self.ctrl_model_dd.options:
            self.ctrl_model_dd.value = saved_ctrl
            self._config["Controller Model"] = saved_ctrl
            self._safe_update(self.ctrl_model_dd)

        saved_crop = cfg.get("Crop Mode")
        if saved_crop in ("auto", "manual", "off"):
            self.crop_mode_key = saved_crop
            self._config["Crop Mode"] = saved_crop
            t = UI_TEXTS[self.language]
            self.crop_mode_dd.options = [t["Crop Auto"], t["Crop Manual"], t["Crop Off"]]
            self.crop_mode_dd.value = t[{"auto": "Crop Auto", "manual": "Crop Manual", "off": "Crop Off"}[saved_crop]]
            self._safe_update(self.crop_mode_dd)

        saved_env = cfg.get("Environment Model")
        if saved_env:
            if str(saved_env).strip().lower() == "black":
                saved_env = "Default"
            canonical_keys = list(self._env_builtin_keys) + list(self._env_folder_keys)
            env_key_match = next(
                (k for k in canonical_keys if str(k).lower() == str(saved_env).lower()),
                None,
            )
            if env_key_match is not None:
                self.env_key = env_key_match
                self._config["Environment Model"] = self.env_key
                self.env_dd.value = self._env_display_label(self.env_key, self.language)
                self._safe_update(self.env_dd)

    def stop_process(self, e=None):
        """Stop subprocess (called from UI button)."""
        future = asyncio.run_coroutine_threadsafe(self._async_stop(), self._loop)
        future.add_done_callback(lambda f: f.exception() if f.exception() else None)

    async def _on_page_close(self, e=None):
        """Stop subprocess when user closes the window. Settings are only saved on Run."""
        self._closed = True
        if hasattr(self, '_esc_task') and self._esc_task and not self._esc_task.done():
            self._esc_task.cancel()
        await self._async_stop()

    async def _async_stop(self):
        """Actual stop logic — runs in async context."""
        # Prevent reentry: skip if already stopping
        if self._stopping:
            return
        self._stopping = True

        # Suppress ESC repeat triggers + cancel pending start
        self._esc_stopped = True
        self._esc_down = None
        self._cancel_starting = True

        if self._proc_lock is not None:
            shutdown_event.set()
            saved_pid = None
            proc = None
            import signal as _sig
            async with self._proc_lock:
                proc = self.process
                if proc and proc.returncode is None:
                    saved_pid = proc.pid
                    # Graceful first: ask main.py to shut down cleanly instead of
                    # hard-killing it.  A hard kill (TerminateProcess) skips the
                    # child's signal handler, so xrviewer.cleanup() never runs and
                    # the OpenXR session + D3D11/GL GPU interop are never released
                    # — killing that heavy OpenXR↔GPU integration mid-frame can
                    # wedge the GPU/compute driver until reboot, which then breaks
                    # the next launch of *any* mode.  CREATE_NEW_PROCESS_GROUP lets
                    # us deliver CTRL_BREAK so the child tears down the GPU first.
                    try:
                        if OS_NAME == "Windows":
                            # Do not send CTRL_BREAK for normal Stop: Intel
                            # Fortran-backed native libs abort with
                            # "forrtl: error (200)" on that console event.
                            os.makedirs(LOG_DIR, exist_ok=True)
                            with open(STOP_REQUEST_FILE, "w", encoding="utf-8") as f:
                                f.write(str(saved_pid))
                        else:
                            os.killpg(os.getpgid(saved_pid), _sig.SIGINT)
                    except Exception:
                        self._diag(f"graceful stop failed:\n{traceback.format_exc()}", error=True)
                        try:
                            proc.terminate()
                        except Exception:
                            self._diag(f"proc.terminate() failed:\n{traceback.format_exc()}", error=True)
                self.process = None
                global _CHILD_PID
                _CHILD_PID = None

            if saved_pid and proc:
                # Give the child time to run its own cleanup and exit on its own.
                exited_cleanly = False
                try:
                    await asyncio.wait_for(proc.wait(), timeout=1)
                    exited_cleanly = True
                except asyncio.TimeoutError:
                    exited_cleanly = False
                except Exception:
                    self._diag(f"proc.wait() exception:\n{traceback.format_exc()}", error=True)
                    exited_cleanly = True
                if not exited_cleanly:
                    # Hard-kill the entire process tree immediately.
                    try:
                        proc.kill()
                    except Exception:
                        pass
                    try:
                        if OS_NAME == "Windows":
                            p = await asyncio.create_subprocess_exec(
                                'taskkill', '/f', '/t', '/pid', str(saved_pid),
                                stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL)
                            await p.wait()
                        else:
                            try:
                                os.killpg(os.getpgid(saved_pid), _sig.SIGKILL)
                            except Exception:
                                pass
                    except Exception:
                        pass
        print("[Main] Stopped")
        self._starting = False
        self.set_status(UI_TEXTS[self.language]["Stopped"], key="Stopped")
        if not self._closed:
            self._set_running_ui(False)
            self._reload_settings_from_disk()

    def reset_defaults(self, e):
        """Reset to defaults, preserve language & device."""
        current_lang = self.language
        current_device_label = self.device_dd.value
        current_device_idx = self.device_label_to_index.get(current_device_label, DEFAULTS["Computing Device"])
        current_primary = get_primary_monitor_index()
        dynamic_defaults = DEFAULTS.copy()
        dynamic_defaults["Monitor Index"] = current_primary
        self.apply_config(dynamic_defaults, keep_optional=False)

        # Restore language & device
        self.language = current_lang
        self.lang_dd.value = "English" if current_lang == "EN" else "简体中文"
        self.device_dd.value = current_device_label
        self._config["Language"] = current_lang
        self._config["Computing Device"] = current_device_idx

        self.update_ui_texts()
        self._sync_visibility()
        self.on_device_change(None)
        self.auto_enable_optimizers_based_on_device()
        self.page.update()

    def open_log(self, _):
        """Open the log folder in the system file explorer."""
        try:
            if sys.platform == "win32":
                os.startfile(LOG_DIR)
            elif sys.platform == "darwin":
                subprocess.run(["open", LOG_DIR], check=False)
            else:
                subprocess.run(["xdg-open", LOG_DIR], check=False)
        except Exception:
            pass

    # ESC long-press monitoring

    VK_ESC = 0x1B

    async def _esc_poll_task(self):
        """Background polling for ESC long-press (3 seconds).

        Uses Windows GetAsyncKeyState to query key state:
        - No global hook, won't trigger antivirus
        - No external dependencies (ctypes only, Python built-in)
        - Single asyncio task, no extra threads
        - Can still detect when unfocused, but no hook registration
        - High-frequency polling only while process is running; low-frequency when idle
        """
        if OS_NAME != "Windows":
            return  # Non-Windows: rely on Flet on_keyboard_event only

        user32 = ctypes.windll.user32
        try:
            while not self._closed:
                await asyncio.sleep(0.2)  # 200ms polling for faster ESC response
                if self._closed:
                    break
                # High bit (0x8000) = key is currently pressed
                if user32.GetAsyncKeyState(self.VK_ESC) & 0x8000:
                    if self._esc_down is None:
                        self._esc_down = time.time()
                    elif not self._esc_stopped and (time.time() - self._esc_down >= 3.0):
                        self._esc_stopped = True
                        self._esc_down = None
                        self.set_status(UI_TEXTS[self.language]["esc_stop"])
                        asyncio.ensure_future(self._async_stop())
                else:
                    # Reset on release so it can re-trigger next time
                    if self._esc_down is not None:
                        self._esc_down = None
                        self._esc_stopped = False
        except asyncio.CancelledError:
            pass  # Normal cancellation on window close

    def _on_key(self, e: ft.KeyboardEvent):
        """Flet keyboard event (auxiliary, only used on macOS/Linux when window is focused)."""
        key = str(getattr(e, "key", "") or "").lower()
        if key not in ("esc", "escape") or self._esc_stopped or OS_NAME == "Windows":
            return
        if not (self._starting or (self.process and self.process.returncode is None)):
            return
        # macOS/Linux fallback: start a short-lived monitor task on first press
        now = time.time()
        if self._esc_down is None:
            self._esc_down = now
            self._diag(f"ESC long-press started via Flet key={getattr(e, 'key', '')!r}")
            # Create a short-lived monitor task, not dependent on key repeat events
            asyncio.create_task(self._esc_watch_task())
        elif now - self._esc_down >= 3.0:
            self._esc_stopped = True
            self._esc_down = None
            self.set_status(UI_TEXTS[self.language]["esc_stop"])
            asyncio.ensure_future(self._async_stop())

    async def _esc_watch_task(self):
        """Non-Windows platforms: monitor 3-second timeout after first ESC press."""
        try:
            for _ in range(60):  # 60 × 0.05 = 3s
                await asyncio.sleep(0.05)
                if self._esc_down is None or self._esc_stopped or self._closed:
                    return
                if time.time() - self._esc_down >= 3.0:
                    self._esc_stopped = True
                    self._esc_down = None
                    self.set_status(UI_TEXTS[self.language]["esc_stop"])
                    asyncio.ensure_future(self._async_stop())
                    return
        except asyncio.CancelledError:
            pass


# Entry point
def main():
    ft.run(_async_main)


async def _async_main(page: ft.Page):
    app = Desktop2StereoGUI(page)
    await app.setup()


if __name__ == "__main__":
    main()

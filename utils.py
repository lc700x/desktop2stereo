import requests
import yaml, threading, time
import os, platform, socket
import platform
import functools
import tempfile


def configure_torch_compile_cache():
    """Keep TorchInductor/Triton cache paths short enough for Windows."""
    if os.name != "nt":
        return None

    cache_root = os.environ.get("DESKTOP2STEREO_TORCH_CACHE")
    candidates = [cache_root] if cache_root else [_default_torch_compile_cache_root()]

    last_error = None
    for root in candidates:
        if not root:
            continue
        inductor_dir = os.path.join(root, "i")
        triton_dir = os.path.join(root, "t")
        try:
            os.makedirs(inductor_dir, exist_ok=True)
            os.makedirs(triton_dir, exist_ok=True)
        except OSError as exc:
            last_error = exc
            continue

        os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", inductor_dir)
        os.environ.setdefault("TRITON_CACHE_DIR", triton_dir)
        return root

    if last_error is not None:
        print(f"[Warning] Could not prepare short torch.compile cache: {_format_exception(last_error)}")
    return None


def _default_torch_compile_cache_root():
    temp_root = os.environ.get("TEMP") or os.environ.get("TMP") or tempfile.gettempdir()
    temp_root = _short_path_if_available(temp_root)
    return os.path.join(temp_root, f"torch_{_cache_username()}")


def _cache_username():
    name = os.environ.get("USERNAME") or os.environ.get("USER") or os.path.basename(os.path.expanduser("~"))
    name = name or "user"
    invalid = '<>:"/\\|?*'
    return "".join("_" if ch in invalid or ord(ch) < 32 else ch for ch in name).strip(" .") or "user"


def _short_path_if_available(path):
    if os.name != "nt" or not path:
        return path
    try:
        import ctypes

        size = ctypes.windll.kernel32.GetShortPathNameW(path, None, 0)
        if size <= 0:
            return path
        buf = ctypes.create_unicode_buffer(size)
        if ctypes.windll.kernel32.GetShortPathNameW(path, buf, size) > 0:
            return buf.value
    except Exception:
        pass
    return path


def enable_torch_compile_fallback(torch_module):
    try:
        torch_module._dynamo.config.suppress_errors = True
    except Exception:
        pass


def torch_compile_or_original(torch_module, target, label, **compile_kwargs):
    try:
        return torch_module.compile(target, **compile_kwargs)
    except Exception as exc:
        print(f"[Warning] torch.compile failed for {label}: {_format_exception(exc)}; running without it.")
        return target


def torch_compile_with_runtime_fallback(torch_module, target, label, **compile_kwargs):
    compiled = torch_compile_or_original(torch_module, target, label, **compile_kwargs)
    if compiled is target:
        return target

    disabled = False

    @functools.wraps(target)
    def wrapper(*args, **kwargs):
        nonlocal disabled
        if disabled:
            return target(*args, **kwargs)
        try:
            return compiled(*args, **kwargs)
        except Exception as exc:
            disabled = True
            print(f"[Warning] torch.compile disabled for {label}: {_format_exception(exc)}; running without it.")
            return target(*args, **kwargs)

    return wrapper


def _format_exception(exc):
    first_line = str(exc).splitlines()[0] if str(exc) else ""
    return f"{type(exc).__name__}: {first_line}"

configure_torch_compile_cache()

# Debug Mode
DEBUG = False
# App Version
VERSION = "2.5.0Beta"
# Get OS name
OS_NAME = platform.system()
# Define StereoMix devices
STEREO_MIX_NAMES = [
# English
"stereo mix", "what you hear", "loopback", "system audio", "wave out mix", "mixed output",
# Chinese
"立体声混音", "您听到的声音", "环路", "系统音频", "波形输出混合", "混合输出",
# Japanese
"ステレオ ミックス", "ステレオミックス", "ループバック", "システムオーディオ", "ミックス出力",
# Spanish
"mezcla estéreo", "lo que escuchas", "bucle", "audio del sistema", "salida mixta",
# French
"mixage stéréo", "bouclage", "audio système", "sortie mixte",
# German
"stereomix", "was du hörst", "loopback", "systemaudio", "gemischte ausgabe",
# macOS specific
"blackhole", "loopback", "aggregate device", "multi-output device", "virtual desktop speakers", "remote sound",
# Linux specific
"monitor"
]

# Models with Disabled TRT 
DISABLE_TRT_KEYWORDS = [
    "dpt-hybrid-midas",
    "depthpro",
    "da3-giant",
    "da3nested-giant",
    # "video-depth-anything",
]

# Models with Disabled TRT 
DISABLE_MIGRAPHX_KEYWORDS = [
    "dpt-hybrid-midas",
    "depthpro",
    "da3-giant",
    "da3nested-giant",
    # "video-depth-anything",
]

TRT_FIX_KEYWORDS = [
    # Intel/zoedepth
    "Intel/zoedepth-nyu-kitti",
]

FORCE_FP32_KEYWORDS = [
    # ZoeDepth models
    "Intel/zoedepth-nyu",
    "Intel/zoedepth-kitti",
]


# Models with Disabled CoreML 
DISABLE_COREML_KEYWORDS = [
    "video-depth-anything",
    "da3-",
    "da3nested",
    "dpt-beit",
    "zoedepth",
    "depthpro",
    "infinidepth",
]

# Models with Disabled OpenVINO 
DISABLE_OPENVINO_KEYWORDS = [
    "da3-",
    "dpt-hybrid-midas-hf",
]

# Disable CuDNN for RX 6000 and 5000 series GPUs
DISABLE_CUDNN_GFX = [
    "gfx1030",  # RX 6950, 6900, 6850, 6800 (Navi 21)
    "gfx1031",  # RX 6750, 6700 (Navi 22)
    "gfx1032",  # RX 6650, 6600 (Navi 23)
    "gfx1033",  # Van Goh (Navi 23)
    "gfx1034",  # RX 6550, 6500, 6400, 6300 (Navi 24)
    "gfx1010",  # RX 5700, 5600 (Navi 10)
    "gfx1012",  # RX 5500, 5400, 5300 (Navi 14)
    "gfx1011",  # BC160, Radeon Pro V520 (Navi 12)
]
DISABLE_TRITON_GFX = []

# Global shutdown event
shutdown_event = threading.Event()

def get_font_type(os=OS_NAME):
    if os == "Darwin":
        return "Verdana.ttf"
    elif os == "Windows": 
        return "verdana.ttf"
    elif os == "Linux":
        try:
            return "/usr/share/fonts/truetype/freefont/FreeSans.ttf" # fix for Ubuntu
        except:
            return "Verdana.ttf"
    else:
        return "Verdana.ttf"


def get_fps(window_title="", monitor_index=None):
    """Return monitor refresh rate for the target monitor.
    If window_title is set, finds the monitor containing that window.
    If monitor_index is set (mss 1-based), uses that monitor directly.
    Falls back to primary monitor. Returns 60 if detection fails."""
    try:
        if OS_NAME == "Windows":
            return _get_fps_windows(window_title, monitor_index)
        elif OS_NAME == "Darwin":
            return _get_fps_macos(window_title, monitor_index)
        else:
            return _get_fps_linux(window_title, monitor_index)
    except Exception:
        return 60


def get_monitor_size(monitor_index=None):
    """Return (width, height) for an mss monitor index, falling back to primary."""
    try:
        import mss
        with mss.mss() as sct:
            if monitor_index is None or monitor_index <= 0 or monitor_index >= len(sct.monitors):
                monitor_index = 1
            mon = sct.monitors[monitor_index]
            return int(mon["width"]), int(mon["height"])
    except Exception:
        return 3840, 2160


def compute_output_resolution(setting_value, display_mode, input_monitor_index,
                              stereo_monitor_index=None, run_mode=None):
    """Compute the source processing height used before depth inference."""
    try:
        if isinstance(setting_value, str):
            value = setting_value.strip()
            if value and value.lower() != "auto":
                parsed = int(value)
                if parsed > 0:
                    return parsed
        elif setting_value:
            parsed = int(setting_value)
            if parsed > 0:
                return parsed
    except (TypeError, ValueError):
        pass

    # Only output modes that own an actual desktop/stereo viewer window should
    # derive Auto from monitor geometry. OpenXR uses the native captured frame.
    auto_compute_modes = {"Local Viewer", "3D Monitor", "RTMP Streamer"}
    if run_mode not in auto_compute_modes:
        return 8640  # high no-resize sentinel for process()

    monitor_index = stereo_monitor_index or input_monitor_index
    _, out_h = get_monitor_size(monitor_index)
    if display_mode == "Full-TAB":
        out_h = max(1, out_h // 2)
    return max(2, (int(out_h) // 2) * 2)


def _get_device_name_from_mss_monitor(monitor_index):
    """Map mss monitor index (1-based) to win32api device name by matching rects."""
    import win32api
    import mss
    with mss.mss() as sct:
        if monitor_index is None or monitor_index >= len(sct.monitors):
            monitor_index = 1
        target = sct.monitors[monitor_index]
        tl, tt, tr, tb = target['left'], target['top'], target['left'] + target['width'], target['top'] + target['height']

    monitors = win32api.EnumDisplayMonitors()
    for hmon, hdc, rect in monitors:
        if rect[0] == tl and rect[1] == tt:
            mi = win32api.GetMonitorInfo(hmon)
            return mi['Device']
    # Fallback: match by overlap
    for hmon, hdc, rect in monitors:
        if rect[0] <= tl < rect[2] and rect[1] <= tt < rect[3]:
            mi = win32api.GetMonitorInfo(hmon)
            return mi['Device']
    return win32api.EnumDisplayDevices(None, 0).DeviceName


def _get_fps_windows(window_title, monitor_index):
    import win32api
    import win32gui
    import mss

    device_name = None

    if window_title:
        hwnd = win32gui.FindWindow(None, window_title)
        if hwnd:
            r = win32gui.GetWindowRect(hwnd)
            wx, wy = (r[0] + r[2]) // 2, (r[1] + r[3]) // 2
            with mss.mss() as sct:
                for i, mon in enumerate(sct.monitors):
                    if i == 0:
                        continue
                    if mon['left'] <= wx < mon['left'] + mon['width'] and mon['top'] <= wy < mon['top'] + mon['height']:
                        device_name = _get_device_name_from_mss_monitor(i)
                        break

    if device_name is None and monitor_index is not None:
        try:
            device_name = _get_device_name_from_mss_monitor(monitor_index)
        except Exception:
            pass

    if device_name is None:
        try:
            device_name = win32api.EnumDisplayDevices(None, 0).DeviceName
        except Exception:
            return 60

    try:
        settings = win32api.EnumDisplaySettings(device_name, -1)
        return settings.DisplayFrequency
    except Exception:
        return 60


def _get_display_id_for_window_macos(window_title):
    """Find the CGDirectDisplayID that contains the given window."""
    try:
        from Quartz import (
            CGWindowListCopyWindowInfo,
            kCGWindowListOptionOnScreenOnly,
            kCGNullWindowID,
            CGGetOnlineDisplayList,
            CGDisplayBounds,
        )
    except ImportError:
        return None

    info = CGWindowListCopyWindowInfo(kCGWindowListOptionOnScreenOnly, kCGNullWindowID)
    if not info:
        return None

    max_displays = 16
    display_ids, count = CGGetOnlineDisplayList(max_displays, None, None)
    display_bounds = {}
    for d_id in display_ids[:count]:
        bounds = CGDisplayBounds(d_id)
        display_bounds[d_id] = bounds

    for win_dict in info:
        name = win_dict.get('kCGWindowName', '')
        if window_title in name:
            bounds = win_dict.get('kCGWindowBounds', {})
            wx = bounds.get('X', 0) + bounds.get('Width', 0) / 2
            wy = bounds.get('Y', 0) + bounds.get('Height', 0) / 2
            for d_id, db in display_bounds.items():
                if (db.origin.x <= wx < db.origin.x + db.size.width and
                    db.origin.y <= wy < db.origin.y + db.size.height):
                    return d_id
            return display_ids[0] if count > 0 else None
    return None


def _get_fps_macos(window_title, monitor_index):
    try:
        from Quartz import (
            CGGetOnlineDisplayList,
            CGDisplayCopyDisplayMode,
            CGDisplayModeGetRefreshRate,
            CGDisplayBounds,
        )
        import mss
    except ImportError:
        return 60

    max_displays = 16
    display_ids, count = CGGetOnlineDisplayList(max_displays, None, None)
    if count == 0:
        return 60

    display_id = None

    if window_title:
        display_id = _get_display_id_for_window_macos(window_title)

    if display_id is None and monitor_index is not None and monitor_index > 0:
        with mss.mss() as sct:
            if monitor_index < len(sct.monitors):
                tx, ty = sct.monitors[monitor_index]['left'], sct.monitors[monitor_index]['top']
                for d_id in display_ids[:count]:
                    bounds = CGDisplayBounds(d_id)
                    if abs(int(bounds.origin.x) - tx) <= 1 and abs(int(bounds.origin.y) - ty) <= 1:
                        display_id = d_id
                        break

    if display_id is None:
        display_id = display_ids[0]

    mode = CGDisplayCopyDisplayMode(display_id)
    if mode:
        hz = CGDisplayModeGetRefreshRate(mode)
        if hz > 0:
            return int(round(hz))
    return 60


def _get_fps_linux(window_title, monitor_index):
    import subprocess
    import re
    import mss

    target_left, target_top = None, None

    if window_title:
        try:
            result = subprocess.run(
                ['wmctrl', '-lG'],
                capture_output=True, text=True, timeout=3
            )
            for line in result.stdout.split('\n'):
                if window_title in line:
                    parts = line.split(None, 7)
                    if len(parts) >= 6:
                        wx = int(parts[2]) + int(parts[4]) // 2
                        wy = int(parts[3]) + int(parts[5]) // 2
                        with mss.mss() as sct:
                            for i, mon in enumerate(sct.monitors):
                                if i == 0:
                                    continue
                                if (mon['left'] <= wx < mon['left'] + mon['width'] and
                                        mon['top'] <= wy < mon['top'] + mon['height']):
                                    target_left, target_top = mon['left'], mon['top']
                                    break
                    break
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

    if target_left is None and monitor_index is not None and monitor_index > 0:
        with mss.mss() as sct:
            if monitor_index < len(sct.monitors):
                target_left = sct.monitors[monitor_index]['left']
                target_top = sct.monitors[monitor_index]['top']

    try:
        result = subprocess.run(
            ['xrandr', '--current'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode != 0:
            return 60

        current_is_target = (target_left is None)
        best_rate = None

        for line in result.stdout.split('\n'):
            out_match = re.match(r'^(\S+)\s+connected', line)
            if out_match:
                current_is_target = False
                pos_match = re.search(r'(\d+)x(\d+)\+(\d+)\+(\d+)', line)
                if pos_match:
                    ox, oy = int(pos_match.group(3)), int(pos_match.group(4))
                    is_primary = 'primary' in line
                    if target_left is not None:
                        current_is_target = (ox == target_left and oy == target_top)
                    elif is_primary:
                        current_is_target = True
                continue

            if current_is_target or target_left is None:
                rm = re.search(r'(\d+(?:\.\d+)?)\s*\*\+?', line)
                if rm:
                    rate = int(round(float(rm.group(1))))
                    if target_left is not None and current_is_target:
                        return rate
                    if best_rate is None:
                        best_rate = rate

        if best_rate is not None:
            return best_rate

        # Final fallback: any active mode
        for line in result.stdout.split('\n'):
            rm = re.search(r'(\d+(?:\.\d+)?)\s*\*', line)
            if rm:
                return int(round(float(rm.group(1))))
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    return 60


def read_yaml(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except UnicodeDecodeError:
        # Fallback to try other common encodings if UTF-8 fails
        try:
            with open(path, "r", encoding="gbk") as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            print(f"Failed to load settings.yaml with GBK encoding: {e}")
            return {}


def write_yaml(path, updates):
    """Merge ``updates`` into the YAML at *path* and write it back.

    Reads the existing file (or starts with an empty dict if missing/corrupt),
    overlays the keys from ``updates``, and writes it back as UTF-8.  Unknown
    keys already present in the file are preserved — only the keys explicitly
    in ``updates`` are touched.  This is what runtime callers (xrviewer's
    brand-switch / environment-switch hooks) use to persist user choices
    without clobbering GUI-only fields.

    Returns True on success, False on failure (errors are printed but never
    raised so a save failure can never crash the VR session).
    """
    try:
        cfg = {}
        if os.path.exists(path):
            try:
                cfg = read_yaml(path)
            except Exception:
                cfg = {}
        cfg.update(updates or {})
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
        return True
    except Exception as e:
        print(f"[utils] Failed to write {path}: {e}")
        return False

def get_local_ip():
    """Return the local IP address by creating a UDP socket to a public IP."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            # doesn't need to be reachable
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"

def crop_icon(icon_img):
    """Crop to make icon larger by cropping for Windows"""
    if OS_NAME == "Windows":
        icon_img = icon_img.convert("RGBA")
        bbox = icon_img.getbbox()
        icon_img = icon_img.crop(bbox)
    return icon_img

# load customized settings
settings = read_yaml("settings.yaml")

# Ignore wanning for MPS
if OS_NAME == "Darwin":
    import os, warnings
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    warnings.filterwarnings(
        "ignore",
        message=".*aten::upsample_bicubic2d.out.*MPS backend.*",
        category=UserWarning)
    import time
    import Quartz  # PyObjC binding for CoreGraphics
    
    # macOS virtual keycode for the 'F' key (physical key 'F')
    KEY_F = 3
    # modifier flags: Control + Command
    MODIFY_FLAGS = Quartz.kCGEventFlagMaskControl | Quartz.kCGEventFlagMaskCommand

    def send_ctrl_cmd_f(key=KEY_F, flags=MODIFY_FLAGS):
        # key-down
        ev_down = Quartz.CGEventCreateKeyboardEvent(None, key, True)
        Quartz.CGEventSetFlags(ev_down, flags)
        Quartz.CGEventPost(Quartz.kCGHIDEventTap, ev_down)

        time.sleep(0.02)  # short hold

        # key-up
        ev_up = Quartz.CGEventCreateKeyboardEvent(None, key, False)
        Quartz.CGEventSetFlags(ev_up, flags)
        Quartz.CGEventPost(Quartz.kCGHIDEventTap, ev_up)
        
# Set Hugging Face environment variable
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"

def is_cn_ip():
    # Try connecting to Google and Hugging Face
    google_ok = False
    hf_ok = False

    try:
        requests.get("https://www.google.com", timeout=5)
        google_ok = True
    except Exception:
        google_ok = False

    try:
        requests.get("https://huggingface.co", timeout=5)
        hf_ok = True
    except Exception:
        hf_ok = False

    # If both are reachable, return False immediately
    if google_ok and hf_ok:
        return False
    else:
        return True

if is_cn_ip():
    os.environ['HF_ENDPOINT'] = "https://hf-mirror.com"
else:
    os.environ['HF_ENDPOINT'] = "https://huggingface.co"

if OS_NAME == "Windows":
    import ctypes, win32gui, win32con
    # get windows Hi-DPI scale
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except:
        ctypes.windll.user32.SetProcessDPIAware()

    import ctypes, glfw

    user32 = ctypes.windll.user32
    SetWindowDisplayAffinity = user32.SetWindowDisplayAffinity
    WDA_EXCLUDEFROMCAPTURE = 0x00000011   # Windows 10 2004+
    

    def hide_window_from_capture(glfw_window):
        hwnd = glfw.get_win32_window(glfw_window)
        SetWindowDisplayAffinity(hwnd, WDA_EXCLUDEFROMCAPTURE)
        print("StereoWindow is now hidden from screen capture.")

    def show_window_in_capture(glfw_window):
        hwnd = glfw.get_win32_window(glfw_window)
        SetWindowDisplayAffinity(hwnd, 0)
        print("StereoWindow is now visible to screen capture.")

    def set_window_to_bottom(glfw_window):
        """
        Finds a window by its title and sets its Z-order to the bottom.
        """
        hwnd = glfw.get_win32_window(glfw_window)
        if hwnd:
            win32gui.SetWindowPos(hwnd, win32con.HWND_BOTTOM, 0, 0, 0, 0, win32con.SWP_NOMOVE | win32con.SWP_NOSIZE | win32con.SWP_NOACTIVATE)
    
# Model Mapping Dict
MODEL_MAPPING = {
    # Depth-Anything V2
    "Depth-Anything-V2-Small": "depth-anything/Depth-Anything-V2-Small-hf",
    "Depth-Anything-V2-Base": "depth-anything/Depth-Anything-V2-Base-hf",
    "Depth-Anything-V2-Large": "depth-anything/Depth-Anything-V2-Large-hf",

    # InfiniDepth
    "InfiniDepth-Small": "lc700x/InfiniDepth-Small",
    "InfiniDepth-SmallPlus": "lc700x/InfiniDepth-SmallPlus",
    "InfiniDepth-Base": "lc700x/InfiniDepth-Base",
    "InfiniDepth-Large": "lc700x/InfiniDepth-Large",
    
    # Video-Depth-Anything
    "Video-Depth-Anything-Small": "depth-anything/Video-Depth-Anything-Small",
    "Video-Depth-Anything-Base": "depth-anything/Video-Depth-Anything-Base",
    "Video-Depth-Anything-Large": "depth-anything/Video-Depth-Anything-Large",
    
    # DA3
    "DA3-SMALL": "depth-anything/DA3-SMALL",
    "DA3-BASE": "depth-anything/DA3-BASE",
    # "DA3-LARGE": "depth-anything/DA3-LARGE",
    # "DA3-GIANT": "depth-anything/DA3-GIANT",
    "DA3-LARGE": "depth-anything/DA3-LARGE-1.1",
    "DA3-GIANT": "depth-anything/DA3-GIANT-1.1",
    "DA3METRIC-LARGE": "depth-anything/DA3METRIC-LARGE",
    # "DA3NESTED-GIANT-LARGE": "depth-anything/DA3NESTED-GIANT-LARGE",
    "DA3NESTED-GIANT-LARGE": "depth-anything/DA3NESTED-GIANT-LARGE-1.1",
    "DA3MONO-LARGE": "depth-anything/DA3MONO-LARGE",
    
    # Depth-Anything-V2 Metric Outdoor
    "Depth-Anything-V2-Metric-Outdoor-Small": "depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf",
    "Depth-Anything-V2-Metric-Outdoor-Base": "depth-anything/Depth-Anything-V2-Metric-Outdoor-Base-hf",
    "Depth-Anything-V2-Metric-Outdoor-Large": "depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf",
    
    # Depth-Anything-V2 Metric Indoor
    "Depth-Anything-V2-Metric-Indoor-Small": "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf",
    "Depth-Anything-V2-Metric-Indoor-Base": "depth-anything/Depth-Anything-V2-Metric-Indoor-Base-hf",
    "Depth-Anything-V2-Metric-Indoor-Large": "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf",
    
    # Metric-Video-Depth-Anything
    "Metric-Video-Depth-Anything-Small": "depth-anything/Metric-Video-Depth-Anything-Small",
    "Metric-Video-Depth-Anything-Base": "depth-anything/Metric-Video-Depth-Anything-Base",
    "Metric-Video-Depth-Anything-Large": "depth-anything/Metric-Video-Depth-Anything-Large",
    
    # LiheYoung/depth-anything
    "depth-anything-small": "LiheYoung/depth-anything-small-hf",
    "depth-anything-base": "LiheYoung/depth-anything-base-hf",
    "depth-anything-large": "LiheYoung/depth-anything-large-hf",
    "depth-anything-indoor-large": "lc700x/depth-anything-indoor-large-hf",
    "depth-anything-outdoor-large": "lc700x/depth-anything-outdoor-large-hf",

    # Distill-Any-Depth
    "Distill-Any-Depth-Small": "xingyang1/Distill-Any-Depth-Small-hf",
    "Distill-Any-Depth-Base": "lc700x/Distill-Any-Depth-Base-hf",
    "Distill-Any-Depth-Large": "xingyang1/Distill-Any-Depth-Large-hf",
    
    # DPT-DINOv2 KITTI
    "dpt-dinov2-small-kitti": "facebook/dpt-dinov2-small-kitti",
    "dpt-dinov2-base-kitti": "lc700x/dpt-dinov2-base-kitti-hf",
    "dpt-dinov2-large-kitti": "lc700x/dpt-dinov2-large-kitti-hf",
    "dpt-dinov2-giant-kitti": "lc700x/dpt-dinov2-giant-kitti-hf",
    
    # DPT-DINOv2 NYU
    "dpt-dinov2-small-nyu": "lc700x/dpt-dinov2-small-nyu-hf",
    "dpt-dinov2-base-nyu": "lc700x/dpt-dinov2-base-nyu-hf",
    "dpt-dinov2-large-nyu": "lc700x/dpt-dinov2-large-nyu-hf",
    "dpt-dinov2-giant-nyu": "facebook/dpt-dinov2-giant-nyu",
    
    # Other
    "depth-ai": "lc700x/depth-ai-hf",
    "dpt-hybrid-midas": "lc700x/dpt-hybrid-midas-hf",
    
    # Intel/DPT
    "dpt-beit-base-384": "Intel/dpt-beit-base-384",
    "dpt-beit-large-512": "Intel/dpt-beit-large-512",
    "dpt-large": "Intel/dpt-large",
    "dpt-large-redesign": "lc700x/dpt-large-redesign-hf",
    
    # Intel/ZoeDepth
    "zoedepth-nyu-kitti": "Intel/zoedepth-nyu-kitti",
    "zoedepth-nyu": "Intel/zoedepth-nyu",
    "zoedepth-kitti": "Intel/zoedepth-kitti",

    # Apple/DepthPro
    "DepthPro-Large": "apple/DepthPro-hf"
}

# Streamer Settings
DEFAULT_PORT = 1122
STREAM_QUALITY = settings["Stream Quality"]
STREAM_PORT = settings["Streamer Port"]
LOCAL_IP = get_local_ip()

# Get settings
RUN_MODE = settings["Run Mode"]
# Add for 3D monitor
USE_3D_MONITOR = False
STREAM_MODE = None

# Add for FrameGen
LOSSLESS_SCALING_SUPPORT = False
MODEL = settings["Depth Model"]
MODEL_ID = MODEL_MAPPING[MODEL]
ALL_MODELS = settings["Model List"]
CACHE_PATH = "./models"
DEPTH_RESOLUTION = settings["Depth Resolution"]
DEVICE_ID = settings["Computing Device"]
FP16 = settings["FP16"]
MONITOR_INDEX,  DISPLAY_MODE = settings["Monitor Index"], settings["Display Mode"]
STEREO_DISPLAY_INDEX = settings.get("Stereo Output")
STEREO_DISPLAY_SELECTION = False if not STEREO_DISPLAY_INDEX else True
OUTPUT_RESOLUTION = compute_output_resolution(
    settings.get("Processing Resolution", "Auto"),
    DISPLAY_MODE,
    MONITOR_INDEX,
    STEREO_DISPLAY_INDEX,
    RUN_MODE,
)
SHOW_FPS, DEPTH_STRENGTH = settings["Show FPS"], settings["Depth Strength"]
IPD = settings["IPD"]
CONVERGENCE = settings["Convergence"]
CAPTURE_MODE = settings["Capture Mode"]
WINDOW_TITLE = settings["Window Title"] if CAPTURE_MODE == "Window" else None
FPS = get_fps(WINDOW_TITLE, MONITOR_INDEX)

# Image Processing Parameters
FOREGROUND_SCALE = settings["Foreground Scale"] / 10 # 0-10
AA_STRENGTH = settings["Anti-aliasing"] * 2

# Experimental Settings
USE_TORCH_COMPILE = settings["torch.compile"]
USE_TENSORRT = settings["TensorRT"] # use TensorRT for CUDA
RECOMPILE_TRT = settings["Recompile TensorRT"] # recompile TensorRT engine

USE_COREML = settings["CoreML"] # use CoreML for MacOS
RECOMPILE_COREML = settings["Recompile CoreML"] # recompile CoreML mlpackage

USE_OPENVINO = settings["OpenVINO"]  # use OpenVINO for Intel
RECOMPILE_OPENVINO = settings["Recompile OpenVINO"] # recompile OpenVINO IR

USE_MIGRAPHX = settings["MIGraphX"] # use MIGraphX for ROCm7 AMD GPUs
RECOMPILE_MIGRAPHX = settings["Recompile MIGraphX"] # recompile MIGraphX graph

def _resolve_capture_tool(raw_value):
    """If Capture Tool is 'none', pick the OS- and device-specific default."""
    if raw_value and raw_value != "none":
        return raw_value
    if OS_NAME == "Windows":
        try:
            import torch
            if torch.cuda.is_available():
                if getattr(torch.version, "hip", None) is not None:
                    return "WindowsCaptureROCm"
                return "WindowsCaptureCUDA"
        except Exception:
            pass
        try:
            import torch_directml
            if torch_directml.is_available() and torch_directml.device_count() > 0:
                return "DXCamera"
        except Exception:
            pass
        return "DXCamera"
    if OS_NAME == "Darwin":
        return "ScreenCaptureKit"
    return "DXCamera"

CAPTURE_TOOL = _resolve_capture_tool(settings["Capture Tool"])
FILL_16_9 = settings["Fill 16:9"]
VSYNC = settings.get("VSync", False)
FIX_VIEWER_ASPECT = True if RUN_MODE == "RTMP Streamer" else settings["Fix Viewer Aspect"] # Keep Viewer Aspect for RTMP with LOSSLESS_SCALING_SUPPORT
STEREOMIX_DEVICE = settings["Stereo Mix"] # RTMP StereoMix Device
STREAM_KEY = settings["Stream Key"]
AUDIO_DELAY = settings["Audio Delay"]
CRF = settings["CRF"]
LANG = settings["Language"]

# Handheld Controller Operation Guide for OpenXR Link, can be easily extended for in-game usage
if LANG == "CN":
    ROWS = [
        ("# 手柄操作指南", "", "", True),
        ("", "", "", False),

        ("[屏幕位置与姿态]", "", "", True),
        ("左握持 + 激光指屏幕", "按住移动或旋转", "屏幕垂直平移或90度旋转", False),
        ("右握持 + 激光指屏幕", "按住移动或旋转", "头部球面旋转或任意旋转", False),
        ("双握持 + 激光指屏幕", "按住移动", "双手中心点移动", False),
        ("左握持 + 左摇杆 左右", "按住推动", "屏幕偏摆旋转", False),
        ("左握持 + 左摇杆 前后", "按住推动", "屏幕俯仰旋转", False),
        ("右握持 + 右摇杆 左右", "按住推动", "屏幕尺寸调整", False),
        ("右握持 + 右摇杆 前后", "按住推动", "屏幕距离调整", False),
        ("", "", "", False),

        ("[预设与杂项]", "", "", True),
        ("左 Menu 键", "短按", "显示/隐藏状态与快捷键面板", False),
        ("左 Y 键", "短按", "默认环境重置屏幕 / 房间重置座位高度", False),
        ("左 Y 键", "长按 1s", "默认环境循环屏幕预设 / 房间循环座位或灯光", False),
        ("左 X 键", "短按", "显示/隐藏虚拟键盘", False),
        ("左 X 键", "长按 1-5s 后松开", "切换灯光/发光模式", False),
        ("左 X 键", "按住 5s", "切换透视绿幕", False),
        ("左扳机(激光不指屏幕)", "按住 3s", "循环裁剪模式 自动/手动/关闭", False),
        ("左扳机(激光不指屏幕,手动模式)", "双击", "开关裁剪调整暂停", False),
        ("左摇杆(裁剪调整中,不握持)", "左右/前后", "裁剪左右(X)或上下(Y),主轴", False),
        ("右 A 键", "激光指屏幕时短按", "鼠标左键单击", False),
        ("右 B 键", "激光指屏幕时短按", "鼠标右键单击", False),
        ("左摇杆按下", "短按", "切换环境模型", False),
        ("", "", "", False),

        ("[深度与视觉]", "", "", True),
        ("右握持 + 左摇杆 前后", "按住推动", "调整深度强度并同步设置", False),
        ("右握持 + 右摇杆按下", "短按", "重置深度强度为 2.0", False),
        ("右摇杆按下", "短按/长按", "切换曲面屏 / 重置屏幕朝向", False),
        ("", "", "", False),

        ("[鼠标与快捷键(激光指向屏幕时)]", "", "", True),
        ("任一扳机", "全按单击", "屏幕触摸/鼠标单击", False),
        ("任一扳机", "持续按住", "拖动/长按触摸", False),
        ("双扳机", "同时按住", "双指平移/缩放手势", False),
        ("扳机指状态面板", "全按单击", "显示/隐藏快捷键面板", False),
        ("右摇杆 前后", "按住推动", "鼠标滚轮滚动", False),
        ("左摇杆 前后", "按住推动", "键盘上下方向键", False),
        ("左摇杆 左右", "按住推动", "键盘左右方向键", False),
        ("左摇杆按下", "短按", "切换环境模型", False),
        ("左摇杆按下", "长按 1s", "显示/隐藏状态与快捷键面板", False),
        ("右摇杆按下", "短按", "切换曲面/平面屏幕", False),
        ("右摇杆按下", "长按 1s", "重置屏幕朝向", False),
        ("", "", "", False),

        ("[虚拟键盘(仅键盘显示时)]", "", "", True),
        ("双握持 + 激光指键盘", "按住移动", "键盘绕头球面移动", False),
        ("右握持 + 左摇杆 左右", "按住推动", "键盘宽度缩放", False),
        ("右握持 + 左摇杆 前后", "按住推动", "键盘推拉距离", False),
        ("左握持 + 右摇杆 左右", "按住推动", "键盘偏摆偏移", False),
        ("左握持 + 右摇杆 前后", "按住推动", "键盘俯仰偏移", False),
        ("左握持 + 左摇杆按下", "按住移动", "键盘环绕头部位移", False),
        ("左/右扳机", "半按", "触发键盘按键", False),
        ("", "", "", False),

        ("[手柄模型校准(开发者)]", "", "", True),
        ("右 A+B 键", "同时按住 0.5s", "切换手柄模型并显示环境/手柄指示", False),
        ("右 A+B 键", "同时按住 5s", "进入/退出校准模式", False),
        ("右 B 键", "校准模式中单击", "保存校准并退出", False),
    ]
else:
    ROWS = [
        ("# Controller Operation Guide", "", "", True),
        ("", "", "", False),

        ("[Screen Position & Orientation]", "", "", True),
        ("Left Grip + laser on screen", "Hold & move or twist", "Translate vertically or 90deg rotate", False),
        ("Right Grip + laser on screen", "Hold & move or twist", "Sphere-orbit or free rotate", False),
        ("Both Grips + laser on screen", "Hold & move", "Move by center of both hands", False),
        ("Left Grip + left stick L/R", "Hold & push", "Screen yaw rotation", False),
        ("Left Grip + left stick U/D", "Hold & push", "Screen pitch rotation", False),
        ("Right Grip + right stick L/R", "Hold & push", "Screen width adjustment", False),
        ("Right Grip + right stick U/D", "Hold & push", "Screen distance adjustment", False),
        ("", "", "", False),

        ("[Presets & Misc]", "", "", True),
        ("Left Menu button", "Short press", "Show / hide status + shortcuts panel", False),
        ("Left Y button", "Short press", "Reset screen / room seat height", False),
        ("Left Y button", "Long press 1s", "Cycle screen presets / room seat-light presets", False),
        ("Left X button", "Quick tap (<1s)", "Show/hide virtual keyboard", False),
        ("Left X button", "Hold >1s, release", "Toggle light / glow mode", False),
        ("Left X button", "Hold >4s", "Toggle passthrough green", False),
        ("Left trigger (laser off screen)", "Hold 3s", "Cycle Crop: Auto/Manual/Off", False),
        ("Left trigger (off screen, Manual)", "Double-tap", "Toggle crop-adjust pause", False),
        ("Left stick (crop-adjust, no grip)", "L/R or U/D", "Crop sides (X) or top/bottom (Y), dominant axis", False),
        ("Right A button", "Laser on screen click", "Left mouse click", False),
        ("Right B button", "Laser on screen click", "Right mouse click", False),
        ("Left stick press", "Short press", "Cycle environment model", False),
        ("", "", "", False),

        ("[Depth & Visual]", "", "", True),
        ("Right Grip + left stick U/D", "Hold & push", "Adjust depth strength", False),
        ("Right Grip + left stick L/R", "Hold & push", "Adjust veil transparency", False),
        ("Right Grip + right stick press", "Short press", "Reset depth strength to 2.0", False),
        ("Right stick press", "Short/long press", "Toggle curved / reset screen direction", False),
        ("", "", "", False),

        ("[Mouse & Shortcuts (laser on screen)]", "", "", True),
        ("Either trigger", "Full press click", "Screen touch / mouse click", False),
        ("Either trigger", "Hold", "Drag / touch hold", False),
        ("Both triggers", "Hold together", "Two-finger pan / zoom gesture", False),
        ("Trigger on status panel", "Full press click", "Show / hide shortcuts panel", False),
        ("Right stick U/D", "Hold & push", "Mouse wheel scroll", False),
        ("Left stick U/D", "Hold & push", "Keyboard Up/Down arrows", False),
        ("Left stick L/R", "Hold & push", "Keyboard Left/Right arrows", False),
        ("Left stick press", "Short press", "Cycle environment model", False),
        ("Left stick press", "Long press 1s", "Show / hide status + shortcuts panel", False),
        ("Right stick press", "Short press", "Toggle curved / flat screen", False),
        ("Right stick press", "Long press 1s", "Reset screen direction", False),
        ("", "", "", False),

        ("[Virtual Keyboard (keyboard visible only)]", "", "", True),
        ("Both Grips + laser on keyboard", "Hold & move", "Orbit keyboard around head", False),
        ("Right Grip + left stick L/R", "Hold & push", "Keyboard width resize", False),
        ("Right Grip + left stick U/D", "Hold & push", "Keyboard push/pull distance", False),
        ("Left Grip + right stick L/R", "Hold & push", "Keyboard yaw offset", False),
        ("Left Grip + right stick U/D", "Hold & push", "Keyboard pitch offset", False),
        ("Left Grip + left stick press", "Hold & move", "Keyboard sphere orbit", False),
        ("Left/Right trigger", "Half press", "Trigger key on keyboard", False),
        ("", "", "", False),

        ("[Controller Calibration (Developer)]", "", "", True),
        ("Right A+B buttons", "Hold both 0.5s", "Switch controller model and indicator", False),
        ("Right A+B buttons", "Hold both 5s", "Enter/exit calibration mode", False),
        ("Right B button", "Click in calibration mode", "Save calibration & exit", False),
    ]

# Determin the run mode and stream mode
if RUN_MODE == "Local Viewer":
    RUN_MODE = "Viewer"
elif RUN_MODE == "3D Monitor" and OS_NAME == "Windows":
    RUN_MODE = "Viewer"
    USE_3D_MONITOR = True
elif RUN_MODE == "MJPEG Streamer":
    RUN_MODE = "Viewer"
    STREAM_MODE = "MJPEG" 
elif RUN_MODE == "RTMP Streamer":
    RUN_MODE = "Viewer"
    STREAM_MODE = "RTMP"
    if OS_NAME == "Windows":
        # Frame Generation Settings for RTMP, Local Viewer not requried
        LOSSLESS_SCALING_SUPPORT = settings["Lossless Scaling Support"]
elif RUN_MODE == "OpenXR Link":
    RUN_MODE = "OpenXR"
else:
    RUN_MODE = "Streamer"

# Specify the Stereo Display for output

STEREO_DISPLAY_INDEX = settings["Stereo Output"]
STEREO_DISPLAY_SELECTION = False if not STEREO_DISPLAY_INDEX else True
CONTROLLER_MODEL = settings["Controller Model"]
ENVIRONMENT_MODEL = settings.get("Environment Model", "Default")
XR_PREVIEW_WINDOW = settings.get("XR Preview", True)
CROP_MODE = settings.get("Crop Mode", "manual")

# Initialize Device
import torch
def get_device(index=0):
    try:
        try:
            import torch_directml
            if torch_directml.is_available():
                return torch_directml.device(index), f"Using DirectML device: {torch_directml.device_name(index)}"
        except ImportError:
            pass
        if torch.backends.mps.is_available() and index==0:
            return torch.device("mps"), "Using Apple Silicon (MPS) device"
        if torch.cuda.is_available():
            return torch.device("cuda"), f"Using CUDA device: {torch.cuda.get_device_name(index)}"
        if torch.xpu.is_available():
            return torch.device("xpu"), f"Using XPU device: {torch.xpu.get_device_name(index)}"
        else:
            return torch.device("cpu"), "Using CPU device"
    except:
        return torch.device("cpu"), "Using CPU device"
    
DEVICE, DEVICE_INFO = get_device(DEVICE_ID)

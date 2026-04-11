import yaml, threading, time
import os, platform, socket

# Debug Mode
DEBUG = False
# App Version
VERSION = "2.4.1"
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
    "video-depth-anything",
    "dpt-hybrid-midas-hf",
    "depthpro"
]

# Models with Disabled CoreML 
DISABLE_COREML_KEYWORDS = [
    "video-depth-anything",
    "da3-", 
    "da3nested",
    "dpt-beit",
    "zoedepth",
    "depthpro"
]

# Models with Disabled OpenVINO 
DISABLE_OPENVINO_KEYWORDS = [
    "da3-",
    "dpt-hybrid-midas-hf",
]

# Disable CuDNN for RX 6000 and 5000 series GPUs
DISABLE_CUDNN_KEYWORDS = ["6950", "6900", "6850", "6800", "6750", "6700", "6650", "6600", "6550", "6500", "6400", "6300", "680", "6100", "5700", "5600", "5500", "5400", "5300", "520", "160"]
# Disable Triton for RX 5000 series
DISABLE_TRITON_KEYWORDS = ["5700", "5600", "5500", "5400", "5300", "520", "160"]

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
os.environ['HF_ENDPOINT'] = settings["HF Endpoint"]

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
    "Depth-Anything-V2-Small-hf": "depth-anything/Depth-Anything-V2-Small-hf",
    "Depth-Anything-V2-Base-hf": "depth-anything/Depth-Anything-V2-Base-hf",
    "Depth-Anything-V2-Large-hf": "depth-anything/Depth-Anything-V2-Large-hf",
    
    # Video-Depth-Anything
    "Video-Depth-Anything-Small": "depth-anything/Video-Depth-Anything-Small",
    "Video-Depth-Anything-Base": "depth-anything/Video-Depth-Anything-Base",
    "Video-Depth-Anything-Large": "depth-anything/Video-Depth-Anything-Large",
    
    # DA3
    "DA3-SMALL": "depth-anything/DA3-SMALL",
    "DA3-BASE": "depth-anything/DA3-BASE",
    "DA3-LARGE-1.1": "depth-anything/DA3-LARGE-1.1",
    "DA3-GIANT-1.1": "depth-anything/DA3-GIANT-1.1",
    "DA3METRIC-LARGE": "depth-anything/DA3METRIC-LARGE",
    "DA3NESTED-GIANT-LARGE-1.1": "depth-anything/DA3NESTED-GIANT-LARGE-1.1",
    "DA3MONO-LARGE": "depth-anything/DA3MONO-LARGE",
    
    # Depth-Anything-V2 Metric Outdoor
    "Depth-Anything-V2-Metric-Outdoor-Small-hf": "depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf",
    "Depth-Anything-V2-Metric-Outdoor-Base-hf": "depth-anything/Depth-Anything-V2-Metric-Outdoor-Base-hf",
    "Depth-Anything-V2-Metric-Outdoor-Large-hf": "depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf",
    
    # Depth-Anything-V2 Metric Indoor
    "Depth-Anything-V2-Metric-Indoor-Small-hf": "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf",
    "Depth-Anything-V2-Metric-Indoor-Base-hf": "depth-anything/Depth-Anything-V2-Metric-Indoor-Base-hf",
    "Depth-Anything-V2-Metric-Indoor-Large-hf": "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf",
    
    # Metric-Video-Depth-Anything
    "Metric-Video-Depth-Anything-Small": "depth-anything/Metric-Video-Depth-Anything-Small",
    "Metric-Video-Depth-Anything-Base": "depth-anything/Metric-Video-Depth-Anything-Base",
    "Metric-Video-Depth-Anything-Large": "depth-anything/Metric-Video-Depth-Anything-Large",
    
    # LiheYoung/depth-anything
    "depth-anything-small-hf": "LiheYoung/depth-anything-small-hf",
    "depth-anything-base-hf": "LiheYoung/depth-anything-base-hf",
    "depth-anything-large-hf": "LiheYoung/depth-anything-large-hf",
    
    # Distill-Any-Depth
    "Distill-Any-Depth-Small-hf": "xingyang1/Distill-Any-Depth-Small-hf",
    "Distill-Any-Depth-Base-hf": "lc700x/Distill-Any-Depth-Base-hf",
    "Distill-Any-Depth-Large-hf": "xingyang1/Distill-Any-Depth-Large-hf",
    
    # DPT-DINOv2 KITTI
    "dpt-dinov2-small-kitti": "facebook/dpt-dinov2-small-kitti",
    "dpt-dinov2-base-kitti-hf": "lc700x/dpt-dinov2-base-kitti-hf",
    "dpt-dinov2-large-kitti-hf": "lc700x/dpt-dinov2-large-kitti-hf",
    "dpt-dinov2-giant-kitti-hf": "lc700x/dpt-dinov2-giant-kitti-hf",
    
    # DPT-DINOv2 NYU
    "dpt-dinov2-small-nyu-hf": "lc700x/dpt-dinov2-small-nyu-hf",
    "dpt-dinov2-base-nyu-hf": "lc700x/dpt-dinov2-base-nyu-hf",
    "dpt-dinov2-large-nyu-hf": "lc700x/dpt-dinov2-large-nyu-hf",
    "dpt-dinov2-giant-nyu": "facebook/dpt-dinov2-giant-nyu",
    
    # Other
    "depth-ai-hf": "lc700x/depth-ai-hf",
    "dpt-hybrid-midas-hf": "lc700x/dpt-hybrid-midas-hf",
    
    # Intel/DPT
    "dpt-beit-base-384": "Intel/dpt-beit-base-384",
    "dpt-beit-large-512": "Intel/dpt-beit-large-512",
    "dpt-large": "Intel/dpt-large",
    "dpt-large-redesign-hf": "lc700x/dpt-large-redesign-hf",
    
    # Intel/ZoeDepth
    "zoedepth-nyu-kitti": "Intel/zoedepth-nyu-kitti",
    "zoedepth-nyu": "Intel/zoedepth-nyu",
    "zoedepth-kitti": "Intel/zoedepth-kitti",

    # Apple/DepthPro
    "DepthPro-hf": "apple/DepthPro-hf"
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
CACHE_PATH = settings["Download Path"]
DEPTH_RESOLUTION = settings["Depth Resolution"]
DEVICE_ID = settings["Computing Device"]
FP16 = False if OS_NAME == "Darwin" else settings["FP16"]
MONITOR_INDEX, OUTPUT_RESOLUTION, DISPLAY_MODE = settings["Monitor Index"], settings["Output Resolution"], settings["Display Mode"]
SHOW_FPS, FPS, DEPTH_STRENGTH = settings["Show FPS"], settings["FPS"], settings["Depth Strength"]
IPD = settings["IPD"]
CONVERGENCE = settings["Convergence"]
CAPTURE_MODE = settings["Capture Mode"]
WINDOW_TITLE = settings["Window Title"]

# Image Processing Parameters
FOREGROUND_SCALE = settings["Foreground Scale"] / 10 # 0-10
AA_STRENGTH = settings["Anti-aliasing"] * 2

# Experimental Settings
DML_BOOST = settings["Unlock Thread (Legacy Streamer)"] # Unlock thread for DirectML streamer
USE_TORCH_COMPILE = settings["torch.compile"]
USE_TENSORRT = settings["TensorRT"] # use TensorRT for CUDA
RECOMPILE_TRT = settings["Recompile TensorRT"] # recompile TensorRT engine

USE_COREML = settings["CoreML"] # use CoreML for MacOS
RECOMPILE_COREML = settings["Recompile CoreML"] # recompile CoreML mlpackage

USE_OPENVINO = settings["OpenVINO"]  # use OpenVINO for Intel
RECOMPILE_OPENVINO = settings["Recompile OpenVINO"] # recompile OpenVINO IR

CAPTURE_TOOL = settings["Capture Tool"] # DXCamera or WindowsCapture
FILL_16_9 = settings["Fill 16:9"]
FIX_VIEWER_ASPECT = True if RUN_MODE == "RTMP Streamer" else settings["Fix Viewer Aspect"] # Keep Viewer Aspect for RTMP with LOSSLESS_SCALING_SUPPORT
STEREOMIX_DEVICE = settings["Stereo Mix"] # RTMP StereoMix Device
STREAM_KEY = settings["Stream Key"]
AUDIO_DELAY = settings["Audio Delay"]
CRF = settings["CRF"]

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
else:
    RUN_MODE = "Streamer"

# Specify the Stereo Display for output
STEREO_DISPLAY_SELECTION = settings["Specify Display"]
STEREO_DISPLAY_INDEX = settings["Stereo Monitor"]
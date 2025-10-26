import yaml, threading
import os, platform, socket

# App Version
VERSION = "2.3.6"

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
"blackhole", "loopback", "aggregate device", "multi-output device", "virtual desktop speakers", "remote sound"
]

# Global shutdown event
shutdown_event = threading.Event()

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
    icon_img = icon_img.convert("RGBA")
    bbox = icon_img.getbbox()
    icon_img = icon_img.crop(bbox)
    return icon_img

# load customized settings
settings = read_yaml("settings.yaml")

# Get OS name
OS_NAME = platform.system()

# Ignore wanning for MPS
if OS_NAME == "Darwin":
    import os, warnings
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    warnings.filterwarnings(
        "ignore",
        message=".*aten::upsample_bicubic2d.out.*MPS backend.*",
        category=UserWarning
)
# Set Hugging Face environment variable
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

if settings["HF Endpoint"]:
    os.environ['HF_ENDPOINT'] = settings["HF Endpoint"]

if OS_NAME == "Windows":
    import ctypes, win32gui, win32con
    # get windows Hi-DPI scale
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except:
        ctypes.windll.user32.SetProcessDPIAware()

    import glfw
    from ctypes import wintypes
    
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    def hide_window_from_capture(glfw_window):
        """Set display affinity to exclude window from screen capture (Windows only)."""
        WDA_EXCLUDEFROMCAPTURE = 0x00000011
        hwnd = glfw.get_win32_window(glfw_window)
        SetWindowDisplayAffinity = user32.SetWindowDisplayAffinity
        SetWindowDisplayAffinity.argtypes = [wintypes.HWND, wintypes.DWORD]
        SetWindowDisplayAffinity.restype = wintypes.BOOL

        result = SetWindowDisplayAffinity(hwnd, WDA_EXCLUDEFROMCAPTURE)
        if result:
            print("StereoWindow is now hidden from screen capture.")
        else:
            print(f"Failed to set display affinity. Error code: {ctypes.get_last_error()}")

    def set_window_to_bottom(glfw_window):
        """
        Finds a window by its title and sets its Z-order to the bottom.
        """
        hwnd = glfw.get_win32_window(glfw_window)
        if hwnd:
            win32gui.SetWindowPos(hwnd, win32con.HWND_BOTTOM, 0, 0, 0, 0, win32con.SWP_NOMOVE | win32con.SWP_NOSIZE | win32con.SWP_NOACTIVATE)

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
else:
    RUN_MODE = "Streamer"
MODEL_ID = settings["Depth Model"]
ALL_MODELS = settings["Model List"]
CACHE_PATH = settings["Download Path"]
DEPTH_RESOLUTION = settings["Depth Resolution"]
DEVICE_ID = settings["Computing Device"]
FP16 = False if OS_NAME == "Darwin" else settings["FP16"]
MONITOR_INDEX, OUTPUT_RESOLUTION, DISPLAY_MODE = settings["Monitor Index"], settings["Output Resolution"], settings["Display Mode"]
SHOW_FPS, FPS, DEPTH_STRENGTH = settings["Show FPS"], settings["FPS"], settings["Depth Strength"]
IPD = settings["IPD"]
CAPTURE_MODE = settings["Capture Mode"]
WINDOW_TITLE = settings["Window Title"]

# Image Processing Parameters
FOREGROUND_SCALE = settings["Foreground Scale"] # 0-10
AA_STRENTH = settings["Anti-aliasing"] # 0-10
 
# Adjust anti-aliasing and dept dilution value for Mac
AA_STRENTH *= 4 # 0-100

# Experimental Settings
DML_BOOST = settings["Unlock Thread (Legacy Streamer)"] # Unlock thread for DirectML streamer
USE_TORCH_COMPILE = settings["torch.compile"] # compile model with torch.compile
USE_TENSORRT = settings["TensorRT"] # use TensorRT for CUDA
RECOMPILE_TRT = settings["Recompile TensorRT"] # recompile TensorRT engine
CAPTURE_TOOL = settings["Capture Tool"] # DXCamera or WindowsCapture
FILL_16_9 = settings["Fill 16:9"]
FIX_VIEWER_ASPECT = settings["Fix Viewer Aspect"]
STEREOMIX_DEVICE = settings["Stereo Mix"] # RTMP StereoMix Device
STREAM_KEY = settings["Stream Key"]
AUDIO_DELAY = settings["Audio Delay"]
CRF = settings["CRF"]
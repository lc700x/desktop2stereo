import yaml
import os, platform

def crop_icon(icon_img):
    """Crop to make icon larger by cropping for Windows"""
    icon_img = icon_img.convert("RGBA")
    bbox = icon_img.getbbox()
    icon_img = icon_img.crop(bbox)
    return icon_img

# load customized settings
with open("settings.yaml") as settings_yaml:
    try:
        settings = yaml.safe_load(settings_yaml)
    except yaml.YAMLError as exc:
        print(exc)

# App Version
VERSION = 2.3

# Get OS name
OS_NAME = platform.system()

# Get settings
MODEL_ID = settings["Depth Model"]
DEFAULT_MODEL_LIST = settings["Model List"]
CACHE_PATH = settings["Download Path"]
DEPTH_RESOLUTION = settings["Depth Resolution"]
DEVICE_ID = settings["Device"]
FP16 = settings["FP16"]
MONITOR_INDEX, OUTPUT_RESOLUTION, DISPLAY_MODE = settings["Monitor Index"], settings["Output Resolution"], settings["Display Mode"]
SHOW_FPS, FPS, DEPTH_STRENTH = settings["Show FPS"], settings["FPS"], settings["Depth Strength"]
IPD = settings["IPD"]

# Ignore wanning for MPS
if  OS_NAME == "Darwin":
    import os, warnings
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    warnings.filterwarnings(
        "ignore",
        message=".*aten::upsample_bicubic2d.out.*MPS backend.*",
        category=UserWarning
)
# Set Hugging Face environment variable
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
# if len(sys.argv) >= 2 and sys.argv[1] == '--hf-mirror':
if settings["HF Endpoint"]:
    os.environ['HF_ENDPOINT'] = settings["HF Endpoint"]


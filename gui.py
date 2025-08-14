import os
import sys, platform
import subprocess
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk

VERSION = "2.1"
OS_NAME = platform.system()
# Ignore wanning for MPS
if  OS_NAME == "Darwin":
    import os, warnings
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    warnings.filterwarnings(
        "ignore",
        message=".*aten::upsample_bicubic2d.out.*MPS backend.*",
        category=UserWarning
)
def get_devices():
    """
    Returns a list of dictionaries [{dev: torch.device, info: str}] for all available devices.
    """
    devices = {}
    count = 0
    try:
        import torch_directml
        if torch_directml.is_available():
            for i in range(torch_directml.device_count()):
                devices[count] = {"name": f"DirectML{i}: {torch_directml.device_name(i)}", "device": torch_directml.device(i)}
                count += 1
    except ImportError:
        pass

    try:
        import torch
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                devices[count] = {"name": f"CUDA {i}: {torch.cuda.get_device_name(i)}", "device": torch.device(f"cuda:{i}")}
                count += 1
        if torch.backends.mps.is_available():
            devices[count]= {"name": "MPS: Apple Silicon", "device": torch.device("mps")}
            count += 1
        devices[count] = {"name": "CPU", "device": torch.device("cpu")}
    except ImportError:
        raise ImportError("PyTorch Not Found! Make sure you have deployed the Python environment in '.env'.")

    return devices

DEVICES = get_devices()

if OS_NAME == "Windows":
    import ctypes
    # get windows Hi-DPI scale
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except:
        ctypes.windll.user32.SetProcessDPIAware()

try:
    import mss
except Exception:
    mss = None

try:
    import yaml
    HAVE_YAML = True
except Exception:
    HAVE_YAML = False

DEFAULT_MODEL_LIST = [
    "depth-anything/Depth-Anything-V2-Large-hf",
    "depth-anything/Depth-Anything-V2-Base-hf",
    "depth-anything/Depth-Anything-V2-Small-hf",
    "depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf",
    "depth-anything/Depth-Anything-V2-Metric-Outdoor-Base-hf",
    "depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf",
    "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf",
    "depth-anything/Depth-Anything-V2-Metric-Indoor-Base-hf",
    "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf",
    "LiheYoung/depth-anything-large-hf",
    "LiheYoung/depth-anything-base-hf",
    "LiheYoung/depth-anything-small-hf",
    "xingyang1/Distill-Any-Depth-Large-hf",
    "xingyang1/Distill-Any-Depth-Small-hf",
    "apple/DepthPro-hf",
    "Intel/dpt-large"
]

DEFAULTS = {
    "Monitor Index": 1,
    "Output Resolution": 1080,
    "FPS": 60,
    "Show FPS": True,
    "Model List": DEFAULT_MODEL_LIST,
    "Depth Model": DEFAULT_MODEL_LIST[2],
    "Depth Strength": 1.0,
    "Depth Resolution": 384,
    "IPD": 0.064,
    "Display Mode": "SBS",
    "FP16": True,
    "Download Path": "models",
    "HF Endpoint": "https://hf-mirror.com",
    "Device": 0,
    "Language": "EN",
}

UI_TEXTS = {
    "EN": {
        "Monitor Index:": "Monitor Index:",
        "FPS:": "FPS:",
        "Show FPS": "Show FPS",
        "Output Resolution:": "Output Resolution:",
        "IPD (m):": "IPD (m):",
        "Display Mode:": "Display Mode:",
        "Depth Model:": "Depth Model:",
        "Depth Strength:": "Depth Strength:",
        "Depth Resolution:": "Depth Resolution:",
        "FP16": "FP16",
        "Download Path:": "Download Path:",
        "Browse...": "Browse...",
        "HF Endpoint:": "HF Endpoint:",
        "Device:": "Device:",
        "Reset": "Reset",
        "Run": "Run",
        "Set Language:": "Set Language:",
        "Error": "Error",
        "Warning": "Warning",
        "Saved": "Run Desktop2Stereo",
        "PyYAML not installed, cannot save YAML file.": "PyYAML not installed, cannot save YAML file.",
        "Settings saved to settings.yaml": "Settings saved to settings.yaml, click OK to run.",
        "Failed to save settings.yaml:": "Failed to save settings.yaml:",
        "Could not retrieve monitor list.\nFalling back to indexes 1 and 2.": "Could not retrieve monitor list.\nFalling back to indexes 1 and 2."
    },
    "CN": {
        "Monitor Index:": "显示器索引:",
        "FPS:": "帧率:",
        "Show FPS": "显示帧率",
        "Output Resolution:": "输出分辨率:",
        "IPD (m):": "瞳距 (米):",
        "Display Mode:": "显示模式",
        "Depth Model:": "深度模型:",
        "Depth Strength:": "深度强度:",
        "Depth Resolution:": "深度分辨率:",
        "FP16": "半精度浮点 (F16)",
        "Download Path:": "下载路径:",
        "Browse...": "浏览...",
        "HF Endpoint:": "HF 接口:",
        "Device:": "设备:",
        "Reset": "重置",
        "Run": "运行",
        "Set Language:": "设置语言:",
        "Error": "错误",
        "Warning": "警告",
        "Saved": "运行Desktop2Stereo",
        "PyYAML not installed, cannot save YAML file.": "未安装PyYAML，无法保存YAML文件。",
        "Settings saved to settings.yaml": "设置已保存到 settings.yaml，按确定运行。",
        "Failed to save settings.yaml:": "保存 settings.yaml 失败：",
        "Could not retrieve monitor list.\nFalling back to indexes 1 and 2.": "无法获取显示器列表。\n回退到索引1和2。"
    }
}


class ConfigGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(f"Desktop2Stereo v{VERSION} GUI")
        self.minsize(780, 440)
        self.config(padx=40, pady=40)

        self.language = "EN"
        self.loaded_model_list = DEFAULT_MODEL_LIST.copy()

        try:
            icon_img = Image.open("icon.png")
            icon_photo = ImageTk.PhotoImage(icon_img)
            self.iconphoto(True, icon_photo)
        except Exception as e:
            print(f"Warning: Could not load icon.png - {e}")

        self.create_widgets()
        self.monitor_label_to_index = self.populate_monitors()
        self.device_label_to_index = self.populate_devices()

        if os.path.exists("settings.yaml"):
            try:
                cfg = self.read_yaml("settings.yaml")
                self.language = cfg.get("Language", DEFAULTS["Language"])
                if "Model List" in cfg and isinstance(cfg["Model List"], list) and cfg["Model List"]:
                    self.loaded_model_list = cfg["Model List"]
                self.apply_config(cfg)
                self.update_language_texts()
                print("Loaded settings.yaml at startup")
            except Exception as e:
                print(f"Failed to load settings.yaml: {e}")
                self.load_defaults()
                self.update_language_texts()
        else:
            self.load_defaults()
            self.update_language_texts()

        self.language_var.set(self.language)

    def create_widgets(self):
        pad = {"padx": 8, "pady": 6}

        self.label_monitor = ttk.Label(self, text="Monitor Index:")
        self.label_monitor.grid(row=0, column=0, sticky="w", **pad)
        self.monitor_var = tk.StringVar()
        self.monitor_menu = ttk.OptionMenu(self, self.monitor_var, "")
        self.monitor_menu.grid(row=0, column=1, sticky="w", **pad)
        
        self.label_language = ttk.Label(self, text="Set Language:")
        self.label_language.grid(row=0, column=2, sticky="we", **pad)
        self.language_var = tk.StringVar()
        self.language_cb = ttk.Combobox(self, textvariable=self.language_var, state="readonly", values=["English", "简体中文"])
        self.language_cb.grid(row=0, column=3, sticky="ew", **pad)
        self.language_cb.bind("<<ComboboxSelected>>", self.on_language_change)
        
        self.label_device = ttk.Label(self, text="Device:")
        self.label_device.grid(row=1, column=0, sticky="w", **pad)
        self.device_var = tk.StringVar()
        self.device_menu = ttk.OptionMenu(self, self.device_var, "")
        self.device_menu.grid(row=1, column=1, sticky="w", **pad)
        
        self.fp16_var = tk.BooleanVar()
        self.fp16_cb = ttk.Checkbutton(self, text="FP16", variable=self.fp16_var)
        self.fp16_cb.grid(row=1, column=2, sticky="w", **pad)
        
        self.showfps_var = tk.BooleanVar()
        self.showfps_cb = ttk.Checkbutton(self, text="Show FPS", variable=self.showfps_var)
        self.showfps_cb.grid(row=1, column=3, sticky="w", **pad)
        
        self.label_res = ttk.Label(self, text="Output Resolution:")
        self.label_res.grid(row=2, column=0, sticky="w", **pad)
        self.res_values = ["720", "1080", "1440", "2160"]
        self.res_cb = ttk.Combobox(self, values=self.res_values, state="readonly")
        self.res_cb.grid(row=2, column=1, sticky="ew", **pad)
        
        self.label_fps = ttk.Label(self, text="FPS:")
        self.label_fps.grid(row=2, column=2, sticky="w", **pad)
        self.fps_values = ["30", "60", "75", "90", "120"]
        self.fps_cb = ttk.Combobox(self, values=self.fps_values, state="normal")
        self.fps_cb.grid(row=2, column=3, sticky="ew", **pad)
        
        self.label_depth_model = ttk.Label(self, text="Depth Model:")
        self.label_depth_model.grid(row=3, column=0, sticky="w", **pad)
        self.depth_model_var = tk.StringVar()
        self.depth_model_cb = ttk.Combobox(self, textvariable=self.depth_model_var, values=self.loaded_model_list, state="normal")
        self.depth_model_cb.grid(row=3, column=1, columnspan=2, sticky="ew", **pad)

        self.label_depth_res = ttk.Label(self, text="Depth Resolution:")
        self.label_depth_res.grid(row=4, column=0, sticky="w", **pad)
        self.depth_res_values = ["192", "384", "576", "768", "960", "1152", "1344", "1536"]
        self.depth_res_cb = ttk.Combobox(self, values=self.depth_res_values, state="normal")
        self.depth_res_cb.grid(row=4, column=1, sticky="ew", **pad)
        
        self.label_depth_strength = ttk.Label(self, text="Depth Strength:")
        self.label_depth_strength.grid(row=4, column=2, sticky="w", **pad)
        self.depth_strength_values = ["1.0", "2.0", "3.0", "4.0", "5.0"]
        self.depth_strength_cb = ttk.Combobox(self, values=self.depth_strength_values, state="normal")
        self.depth_strength_cb.grid(row=4, column=3, sticky="ew", **pad)
        
        self.label_display_mode = ttk.Label(self, text="Display Mode:")
        self.label_display_mode.grid(row=5, column=0, sticky="w", **pad)
        self.display_mode_values = ["SBS", "TAB"]
        self.display_mode_cb = ttk.Combobox(self, values=self.display_mode_values, state="readonly")
        self.display_mode_cb.grid(row=5, column=1, sticky="ew", **pad)
        
        self.label_ipd = ttk.Label(self, text="IPD (m):")
        self.label_ipd.grid(row=5, column=2, sticky="w", **pad)
        self.ipd_var = tk.StringVar()
        self.ipd_entry = ttk.Entry(self, textvariable=self.ipd_var)
        self.ipd_entry.grid(row=5, column=3, sticky="ew", **pad)
        
        self.label_download = ttk.Label(self, text="Download Path:")
        self.label_download.grid(row=6, column=0, sticky="w", **pad)
        self.download_var = tk.StringVar()
        self.download_entry = ttk.Entry(self, textvariable=self.download_var)
        self.download_entry.grid(row=6, column=1, columnspan=2, sticky="ew", **pad)
        self.btn_browse = ttk.Button(self, text="Browse...", command=self.browse_download)
        self.btn_browse.grid(row=6, column=3, sticky="ew", **pad)
        
        self.label_hf_endpoint = ttk.Label(self, text="HF Endpoint:")
        self.label_hf_endpoint.grid(row=8, column=0, sticky="w", **pad)
        self.hf_endpoint_var = tk.StringVar()
        self.hf_endpoint_entry = ttk.Entry(self, textvariable=self.hf_endpoint_var)
        self.hf_endpoint_entry.grid(row=8, column=1,  sticky="ew", **pad)
        
        self.btn_reset = ttk.Button(self, text="Reset", command=self.reset_to_defaults)
        self.btn_reset.grid(row=8, column=2, sticky="ew", **pad)

        self.btn_run = ttk.Button(self, text="Run", command=self.save_settings)
        self.btn_run.grid(row=8, column=3, sticky="ew", **pad)

        self.columnconfigure(1, weight=1)
        self.columnconfigure(2, weight=1)
        self.columnconfigure(3, weight=1)

    def update_language_texts(self):
        texts = UI_TEXTS[self.language]
        self.label_monitor.config(text=texts["Monitor Index:"])
        self.label_fps.config(text=texts["FPS:"])
        self.showfps_cb.config(text=texts["Show FPS"])
        self.label_res.config(text=texts["Output Resolution:"])
        self.label_ipd.config(text=texts["IPD (m):"])
        self.label_display_mode.config(text=texts["Display Mode:"])
        self.label_depth_model.config(text=texts["Depth Model:"])
        self.label_depth_res.config(text=texts["Depth Resolution:"])
        self.label_depth_strength.config(text=texts["Depth Strength:"])
        self.fp16_cb.config(text=texts["FP16"])
        self.label_download.config(text=texts["Download Path:"])
        self.label_hf_endpoint.config(text=texts["HF Endpoint:"])
        self.label_device.config(text=texts["Device:"])
        self.btn_browse.config(text=texts["Browse..."])
        self.btn_reset.config(text=texts["Reset"])
        self.btn_run.config(text=texts["Run"])
        self.label_language.config(text=texts["Set Language:"])
        self.language_cb["values"] = list(UI_TEXTS.keys())

    def on_language_change(self, event):
        selected = self.language_var.get()
        if selected in UI_TEXTS:
            self.language = selected
            self.update_language_texts()

    def browse_download(self):
        path = filedialog.askdirectory(initialdir=self.download_var.get() or ".")
        if path:
            self.download_var.set(path)
    
    def populate_devices(self):
        self.device_label_to_index = {}
        device_dict = DEVICES  # Expected format: {idx: {"name": str, "device": torch.device}}
        self.all_devices = device_dict

        # Clear existing menu items
        self.device_menu["menu"].delete(0, "end")

        # Populate device list
        for idx, dev_info in device_dict.items():
            label = dev_info["name"]
            self.device_label_to_index[label] = idx
            self.device_menu["menu"].add_command(
                label=label,
                command=lambda v=label: self.device_var.set(v)
            )

        # Set default selection
        default_idx = DEFAULTS.get("Device", 0)
        default_label = next(
            (lbl for lbl, i in self.device_label_to_index.items() if i == default_idx),
            None
        )

        if default_label:
            self.device_var.set(default_label)
        elif self.device_label_to_index:
            self.device_var.set(next(iter(self.device_label_to_index)))

        return self.device_label_to_index


    
    def populate_monitors(self):
        self.monitor_label_to_index = {}
        monitors = []
        if mss:
            try:
                with mss.mss() as sct:
                    for mon in sct.monitors[1:]:
                        monitors.append(mon)
            except Exception:
                monitors = []
        if not monitors:
            messagebox.showwarning("Warning", UI_TEXTS[self.language][
                "Could not retrieve monitor list.\nFalling back to indexes 1 and 2."
            ])
            monitors = [{"width": 0, "height": 0, "left": 0, "top": 0} for _ in range(2)]

        self.monitor_menu["menu"].delete(0, "end")
        for idx, mon in enumerate(monitors, start=1):
            label = f"{idx}: {mon['width']}x{mon['height']} @ ({mon['left']},{mon['top']})"
            self.monitor_label_to_index[label] = idx
            self.monitor_menu["menu"].add_command(
                label=label,
                command=lambda v=label: self.monitor_var.set(v)
            )

        default_idx = DEFAULTS["Monitor Index"]
        default_label = next((lbl for lbl, i in self.monitor_label_to_index.items() if i == default_idx), None)
        if default_label:
            self.monitor_var.set(default_label)
        elif self.monitor_label_to_index:
            self.monitor_var.set(next(iter(self.monitor_label_to_index)))

        return self.monitor_label_to_index

    def read_yaml(self, path):
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def save_yaml(self, path, cfg):
        if not HAVE_YAML:
            messagebox.showerror(UI_TEXTS[self.language]["Error"],
                                 UI_TEXTS[self.language]["PyYAML not installed, cannot save YAML file."])
            return False
        try:
            with open(path, "w", encoding="utf-8") as f:
                yaml.dump(cfg, f, allow_unicode=True, sort_keys=False)
            return True
        except Exception as e:
            messagebox.showerror(UI_TEXTS[self.language]["Error"],
                                 f"{UI_TEXTS[self.language]['Failed to save settings.yaml:']} {e}")
            return False

    def apply_config(self, cfg):
        monitor_idx = cfg.get("Monitor Index", DEFAULTS["Monitor Index"])
        label_for_idx = next((lbl for lbl, i in self.monitor_label_to_index.items() if i == monitor_idx), None)
        if label_for_idx:
            self.monitor_var.set(label_for_idx)
        elif self.monitor_label_to_index:
            self.monitor_var.set(next(iter(self.monitor_label_to_index)))
            
        device_idx = cfg.get("Device", DEFAULTS["Device"])
        lavel_for_device_idx = next((lbl for lbl, i in self.device_label_to_index.items() if i == device_idx), None)
        if lavel_for_device_idx:
            self.device_var.set(lavel_for_device_idx)
        elif self.device_label_to_index:
            self.device_var.set(next(iter(self.device_label_to_index)))

        self.fps_cb.set(str(cfg.get("FPS", DEFAULTS["FPS"])))
        self.showfps_var.set(cfg.get("Show FPS", DEFAULTS["Show FPS"]))
        self.res_cb.set(str(cfg.get("Output Resolution", DEFAULTS["Output Resolution"])))
        self.ipd_var.set(str(cfg.get("IPD", DEFAULTS["IPD"])))

        model_list = cfg.get("Model List", DEFAULTS["Model List"])
        if not model_list:
            model_list = DEFAULT_MODEL_LIST
        self.depth_model_cb["values"] = model_list
        selected_model = cfg.get("Depth Model", model_list[0] if model_list else DEFAULTS["Depth Model"])
        if selected_model not in model_list:
            selected_model = model_list[0]
        self.depth_model_var.set(selected_model)

        self.depth_res_cb.set(cfg.get("Depth Resolution", DEFAULTS["Depth Resolution"]))
        self.display_mode_cb.set(cfg.get("Display Mode", DEFAULTS["Display Mode"]))
        self.depth_strength_cb.set(cfg.get("Depth Strength", DEFAULTS["Depth Strength"]))
        self.fp16_var.set(cfg.get("FP16", DEFAULTS["FP16"]))
        self.download_var.set(cfg.get("Download Path", DEFAULTS["Download Path"]))
        self.hf_endpoint_var.set(cfg.get("HF Endpoint", DEFAULTS["HF Endpoint"]))
        self.language = cfg.get("Language", DEFAULTS["Language"])


    def load_defaults(self):
        self.apply_config(DEFAULTS)

    def reset_to_defaults(self):
        self.load_defaults()
        # self.update_language_texts()

    def save_settings(self):
        cfg = {
            "Monitor Index": self.monitor_label_to_index.get(self.monitor_var.get(), DEFAULTS["Monitor Index"]),
            "FPS": int(self.fps_cb.get()),
            "Show FPS": self.showfps_var.get(),
            "Output Resolution": int(self.res_cb.get()),
            "IPD": float(self.ipd_var.get()),
            "Display Mode": self.display_mode_cb.get(),
            "Model List": list(self.depth_model_cb["values"]),
            "Depth Model": self.depth_model_var.get(),
            "Depth Strength": float(self.depth_strength_cb.get()),
            "Depth Resolution": int(self.depth_res_cb.get()),
            "FP16": self.fp16_var.get(),
            "Download Path": self.download_var.get(),
            "HF Endpoint": self.hf_endpoint_var.get(),
            "Device": self.device_label_to_index.get(self.device_var.get(), DEFAULTS["Device"]),
            "Language": self.language,
        }
        success = self.save_yaml("settings.yaml", cfg)
        if success:
            confirm = messagebox.askokcancel(UI_TEXTS[self.language]["Saved"],
                                UI_TEXTS[self.language]["Settings saved to settings.yaml"])
            
            if confirm:
                # Run main.py as a separate process
                try:
                    subprocess.Popen([sys.executable, "main.py"])
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to run main.py: {e}")

if __name__ == "__main__":
    app = ConfigGUI()
    app.mainloop()

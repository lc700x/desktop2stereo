import os
import sys
import subprocess
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk
from utils import VERSION, OS_NAME, DEFAULT_MODEL_LIST, crop_icon, get_local_ip

# Get window lists
if OS_NAME == "Windows":
    try:
        import win32gui
        import win32con
        import ctypes
        from ctypes import wintypes
    except ImportError:
        win32gui = None

    def list_windows():
        windows = []

        # Setup for DWM call
        DWMWA_EXTENDED_FRAME_BOUNDS = 9
        dwmapi = ctypes.WinDLL("dwmapi")
        rect_type = wintypes.RECT

        def get_window_rect(hwnd):
            rect = rect_type()
            try:
                # Try extended frame bounds (no shadow/borders)
                res = dwmapi.DwmGetWindowAttribute(
                    hwnd,
                    DWMWA_EXTENDED_FRAME_BOUNDS,
                    ctypes.byref(rect),
                    ctypes.sizeof(rect),
                )
                if res == 0:  # S_OK
                    return rect.left, rect.top, rect.right, rect.bottom
            except Exception:
                pass
            # fallback
            return win32gui.GetWindowRect(hwnd)

        def callback(hwnd, _):
            if win32gui.IsWindowVisible(hwnd):
                title = win32gui.GetWindowText(hwnd)
                if title:
                    x, y, r, b = get_window_rect(hwnd)
                    w, h = r - x, b - y
                    if w > 0 and h > 0:
                        windows.append((title, w, h, x, y, hwnd))
            return True

        win32gui.EnumWindows(callback, None)
        return windows
elif OS_NAME == "Darwin":
    try:
        from Quartz import (
            CGWindowListCopyWindowInfo,
            kCGWindowListOptionOnScreenOnly,
            kCGNullWindowID,
        )
    except ImportError:
        CGWindowListCopyWindowInfo = None

    def list_windows(min_width=20, min_height=20):
        windows = []
        options = kCGWindowListOptionOnScreenOnly
        window_info = CGWindowListCopyWindowInfo(options, kCGNullWindowID)

        # System UI processes we want to ignore
        blacklist = {
            "Window Server",
            "ControlCenter",
            "NotificationCenter",
            "Spotlight",
            "Dock",
            "SystemUIServer",
            "CoreServicesUIAgent",
        }

        for win in window_info:
            title = win.get("kCGWindowName", "") or ""
            owner = win.get("kCGWindowOwnerName", "")
            layer = win.get("kCGWindowLayer", 0)
            bounds = win.get("kCGWindowBounds", {})
            w, h = int(bounds.get("Width", 0)), int(bounds.get("Height", 0))
            x, y = int(bounds.get("X", 0)), int(bounds.get("Y", 0))

            # Filtering rules
            if not title.strip():
                continue
            if owner in blacklist:
                continue
            if layer != 0:  # skip menu bar, overlays, etc.
                continue
            if w < min_width or h < min_height:
                continue

            windows.append((title.strip(), w, h, x, y, win["kCGWindowNumber"]))

        return windows
else:
    import subprocess
    def list_windows():
        windows = []
        try:
            result = subprocess.check_output(["wmctrl", "-lG"]).decode("utf-8").splitlines()
            for line in result:
                parts = line.split(None, 6)
                if len(parts) >= 7:
                    win_id, x, y, w, h, _, title = parts
                    w, h, x, y = int(w), int(h), int(x), int(y)
                    if title.strip() and w > 0 and h > 0:
                        windows.append((title.strip(), w, h, x, y, win_id))
        except Exception as e:
            print("Linux window listing error:", e)
        return windows
# List all available devices
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

try:
    import mss
except Exception:
    mss = None

try:
    import yaml
    HAVE_YAML = True
except Exception:
    HAVE_YAML = False

DEFAULTS = {
    "Capture Mode": "Monitor",  # "Monitor" or "Window"
    "Monitor Index": 1,
    "Window Title": "",
    "Capture Coordinates": None,  # (left, top, width, height)
    "Output Resolution": 1080,
    "FPS": 60,
    "Show FPS": True,
    "Model List": DEFAULT_MODEL_LIST,
    "Depth Model": DEFAULT_MODEL_LIST[2],
    "Depth Strength": 1.0,
    "Depth Resolution": 384,
    "IPD": 0.064,
    "Display Mode": "Half-SBS",
    "FP16": True,
    "Download Path": "models",
    "HF Endpoint": "https://hf-mirror.com",
    "Device": 0,
    "Language": "EN",
    "Run Mode": "Viewer",
    "Streamer Host": None,
    "Streamer Port": 1400,
}

UI_TEXTS = {
    "EN": {
        "Monitor": "Monitor",
        "Window": "Window",
        "Refresh": "Refresh",
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
        "Stop": "Stop",
        "HF Endpoint:": "HF Endpoint:",
        "Device:": "Device:",
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
        "Running": "Running...",
        "Stopped": "Stopped.",
        "Countdown": "Settings saved to settings.yaml, starting...",
        "A thread already running!": "A thread already running!",
        "No windows found": "No windows found",
        "Selected window:": "Selected window:",
        "Run Mode:": "Run Mode:",
        "Viewer": "Viewer",
        "Streamer": "Streamer",
        "Streamer Port:": "Streamer Port:",
        "Streamer URL:": "Streamer URL",
        "Host:": "Host:",
        "Invalid port number (1-65535)": "Invalid port number (must be between 1-65535)",
        "Invalid port number": "Port must be a number",
    },
    "CN": {
        "Monitor": "显示器",
        "Window": "窗口",
        "Refresh": "刷新",
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
        "Stop": "停止",
        "HF Endpoint:": "下载节点:",
        "Device:": "设备:",
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
        "Running": "运行中...",
        "Stopped": "已停止。",
        "Countdown": "设置已保存到 settings.yaml，启动中...",
        "A thread already running!": "一个进程已经运行！",
        "No windows found": "未找到窗口",
        "Selected window:": "已选择窗口:",
        "Run Mode:": "运行模式:",
        "Viewer": "本地查看",
        "Streamer": "网络推流",
        "Streamer Port:": "推流端口:",
        "Streamer URL": "推流网址:",
        "Host": "主机:",
        "Invalid port number (1-65535)": "端口号无效 (必须介于1-65535之间)",
        "Invalid port number": "端口必须是数字",
    }
}

class ConfigGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.pad = {"padx": 8, "pady": 6}
        self.title(f"Desktop2Stereo v{VERSION} GUI")
        self.minsize(800, 420)  # Increased height for new controls
        self.resizable(True, False)
        self.language = "EN"
        self.loaded_model_list = DEFAULT_MODEL_LIST.copy()
        self.selected_window_coords = None
        self._window_objects = []  # Store window objects for reference

        try:
            icon_img = Image.open("icon.ico")
            if OS_NAME == "Windows":
                icon_img = crop_icon(icon_img)
            icon_photo = ImageTk.PhotoImage(icon_img)
            self.iconphoto(True, icon_photo)
        except Exception as e:
            print(f"Warning: Could not load icon.ico - {e}")

        # internal run mode key: 'Viewer' or 'Streamer'
        self.run_mode_key = DEFAULTS.get("Run Mode", "Viewer")
        
        # internal capture mode key: 'Monitor' or 'Window'
        self.capture_mode_key = DEFAULTS.get("Capture Mode", "Monitor")

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
                self.update_status(UI_TEXTS[self.language]["Loaded settings.yaml at startup"])
            except Exception as e:
                print(f"Failed to load settings.yaml: {e}")
                self.load_defaults()
                self.update_language_texts()
        else:
            self.load_defaults()
            self.update_language_texts()

        self.language_var.set(self.language)
        self.protocol("WM_DELETE_WINDOW", self.on_close) # Bind to Close of GUI to turn off all threads
        self.process = None  # Keep track of the spawned process

    def on_close(self):
        """Handle GUI window closing: stop process & cleanup."""
        # Stop running process if any
        if self.process and self.process.poll() is None:
            try:
                self.process.terminate()
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
            except Exception as e:
                messagebox.showerror(
                    UI_TEXTS[self.language]["Error"],
                    f"Failed to stop process on exit: {e}"
                )
            finally:
                self.process = None

        # Cancel any scheduled after() callbacks (like _monitor_process)
        try:
            if hasattr(self, "_after_id") and self._after_id:
                self.after_cancel(self._after_id)
        except Exception:
            pass

        # Now destroy GUI
        self.destroy()
    
    def create_widgets(self):
        # Main content frame
        self.content_frame = ttk.Frame(self)
        self.content_frame.grid(row=0, column=0, sticky="nsew", padx=40, pady=20)
        
        # Configure root row/column weights
        self.rowconfigure(0, weight=1)   # content frame expands
        self.rowconfigure(1, weight=0)   # status bar fixed
        self.columnconfigure(0, weight=1)
        
        # Capture Mode (Monitor / Window)
        self.capture_mode_var_label = tk.StringVar()
        self.capture_mode_cb = ttk.Combobox(self.content_frame, textvariable=self.capture_mode_var_label, state="readonly", width=8)
        self.capture_mode_cb.grid(row=2, column=0, sticky="ew", **self.pad)
        self.capture_mode_cb.bind("<<ComboboxSelected>>", self.on_capture_mode_change)
        
        # Monitor Index (only shown when capture mode is Monitor)

        self.monitor_var = tk.StringVar()
        self.monitor_menu = ttk.OptionMenu(self.content_frame, self.monitor_var, "")


        self.window_var = tk.StringVar()
        self.window_cb = ttk.Combobox(self.content_frame, textvariable=self.window_var,state="readonly")
        # self.window_cb.grid(row=0, column=0, sticky="ew", **self.pad)
        self.window_cb.bind("<<ComboboxSelected>>", self.on_window_selected)
        
        self.btn_refresh = ttk.Button(self.content_frame, text="Refresh", command=self.refresh_monitor_and_window)
        self.btn_refresh.grid(row=2, column=3, sticky="we", **self.pad)
        
        # Language
        self.label_language = ttk.Label(self.content_frame, text="Set Language:")
        self.label_language.grid(row=0, column=2, sticky="we", **self.pad)
        self.language_var = tk.StringVar()
        self.language_cb = ttk.Combobox(self.content_frame, textvariable=self.language_var, state="readonly", values=["English", "简体中文"])
        self.language_cb.grid(row=0, column=3, sticky="ew", **self.pad)
        self.language_cb.bind("<<ComboboxSelected>>", self.on_language_change)
        
        # Device
        self.label_device = ttk.Label(self.content_frame, text="Device:")
        self.label_device.grid(row=3, column=0, sticky="w", **self.pad)
        self.device_var = tk.StringVar()
        self.device_menu = ttk.OptionMenu(self.content_frame, self.device_var, "")
        self.device_menu.grid(row=3, column=1, sticky="w", **self.pad)
        
        # FP16 and Show FPS
        self.fp16_var = tk.BooleanVar()
        self.fp16_cb = ttk.Checkbutton(self.content_frame, text="FP16", variable=self.fp16_var)
        self.fp16_cb.grid(row=3, column=2, sticky="w", **self.pad)
        
        self.showfps_var = tk.BooleanVar()
        self.showfps_cb = ttk.Checkbutton(self.content_frame, text="Show FPS", variable=self.showfps_var)
        self.showfps_cb.grid(row=3, column=3, sticky="w", **self.pad)
        
        # Output Resolution
        self.label_res = ttk.Label(self.content_frame, text="Output Resolution:")
        self.label_res.grid(row=4, column=0, sticky="w", **self.pad)
        self.res_values = ["480", "720", "1080", "1440", "2160"]
        self.res_cb = ttk.Combobox(self.content_frame, values=self.res_values, state="normal")
        self.res_cb.grid(row=4, column=1, sticky="ew", **self.pad)
        
        # FPS
        self.label_fps = ttk.Label(self.content_frame, text="FPS:")
        self.label_fps.grid(row=4, column=2, sticky="w", **self.pad)
        self.fps_values = ["30", "60", "75", "90", "120"]
        self.fps_cb = ttk.Combobox(self.content_frame, values=self.fps_values, state="normal")
        self.fps_cb.grid(row=4, column=3, sticky="ew", **self.pad)
        
        # Download path
        self.label_download = ttk.Label(self.content_frame, text="Download Path:")
        self.label_download.grid(row=7, column=0, sticky="w", **self.pad)
        self.download_var = tk.StringVar()
        self.download_entry = ttk.Entry(self.content_frame, textvariable=self.download_var)
        self.download_entry.grid(row=7, column=1, columnspan=2, sticky="ew", **self.pad)
        self.btn_browse = ttk.Button(self.content_frame, text="Browse...", command=self.browse_download)
        self.btn_browse.grid(row=7, column=3, sticky="ew", **self.pad)
        
        # Depth Resolution and Depth Strength
        self.label_depth_res = ttk.Label(self.content_frame, text="Depth Resolution:")
        self.label_depth_res.grid(row=5, column=0, sticky="w", **self.pad)
        self.depth_res_values = ["48", "96", "192", "384", "576", "768", "960", "1152", "1344", "1536"]
        self.depth_res_cb = ttk.Combobox(self.content_frame, values=self.depth_res_values, state="normal")
        self.depth_res_cb.grid(row=5, column=1, sticky="ew", **self.pad)
        
        self.label_depth_strength = ttk.Label(self.content_frame, text="Depth Strength:")
        self.label_depth_strength.grid(row=5, column=2, sticky="w", **self.pad)
        self.depth_strength_values = ["1.0", "1.5", "2.0", "2.5", "3.0", "3.5", "4.0", "4.5", "5.0"]
        self.depth_strength_cb = ttk.Combobox(self.content_frame, values=self.depth_strength_values, state="normal")
        self.depth_strength_cb.grid(row=5, column=3, sticky="ew", **self.pad)
        
        # Display Mode
        self.label_display_mode = ttk.Label(self.content_frame, text="Display Mode:")
        self.label_display_mode.grid(row=6, column=0, sticky="w", **self.pad)
        self.display_mode_values = ["Half-SBS", "Full-SBS", "TAB"]
        self.display_mode_cb = ttk.Combobox(self.content_frame, values=self.display_mode_values, state="readonly")
        self.display_mode_cb.grid(row=6, column=1, sticky="ew", **self.pad)
        
        # IPD
        self.label_ipd = ttk.Label(self.content_frame, text="IPD (m):")
        self.label_ipd.grid(row=6, column=2, sticky="w", **self.pad)
        self.ipd_var = tk.StringVar()
        self.ipd_entry = ttk.Entry(self.content_frame, textvariable=self.ipd_var)
        self.ipd_entry.grid(row=6, column=3, sticky="ew", **self.pad)
        
        # Depth Model
        self.label_depth_model = ttk.Label(self.content_frame, text="Depth Model:")
        self.label_depth_model.grid(row=8, column=0, sticky="w", **self.pad)
        self.depth_model_var = tk.StringVar()
        self.depth_model_cb = ttk.Combobox(self.content_frame, textvariable=self.depth_model_var, values=self.loaded_model_list, state="normal")
        self.depth_model_cb.grid(row=8, column=1, columnspan=2, sticky="ew", **self.pad)
        
        # Run Mode (viewer / streamer)
        self.label_run_mode = ttk.Label(self.content_frame, text="Run Mode:")
        self.label_run_mode.grid(row=0, column=0, sticky="w", **self.pad)
        self.run_mode_var_label = tk.StringVar()
        self.run_mode_cb = ttk.Combobox(self.content_frame, textvariable=self.run_mode_var_label, state="readonly")
        self.run_mode_cb.grid(row=0, column=1, sticky="ew", **self.pad)
        self.run_mode_cb.bind("<<ComboboxSelected>>", self.on_run_mode_change)

        # HF Endpoint
        self.label_hf_endpoint = ttk.Label(self.content_frame, text="HF Endpoint:")
        self.label_hf_endpoint.grid(row=9, column=0, sticky="w", **self.pad)
        self.hf_endpoint_var = tk.StringVar()
        self.hf_endpoint_entry = ttk.Entry(self.content_frame, textvariable=self.hf_endpoint_var)
        self.hf_endpoint_entry.grid(row=9, column=1, sticky="ew", **self.pad)
        
        # Streamer Host and Port (only visible when run mode is streamer)
        self.label_streamer_host = ttk.Label(self.content_frame, text="Streamer URL:")
        self.streamer_host_var = tk.StringVar()
        self.streamer_host_entry = ttk.Entry(self.content_frame, textvariable=self.streamer_host_var, state="readonly", foreground="#3E83F7")
        self.label_streamer_port = ttk.Label(self.content_frame, text="Streamer Port:")
        self.streamer_port_var = tk.StringVar()
        self.streamer_port_entry = ttk.Entry(self.content_frame, textvariable=self.streamer_port_var)
        self.streamer_port_var.trace_add("write", self.update_host_url)

        # Buttons (moved down a bit to make room)
        self.btn_reset = ttk.Button(self.content_frame, text="Reset", command=self.reset_to_defaults)
        self.btn_reset.grid(row=8, column=3, sticky="ew", **self.pad)
        
        self.btn_stop = ttk.Button(self.content_frame, text="Stop", command=self.stop_process)
        self.btn_stop.grid(row=9, column=2, sticky="ew", **self.pad)
        
        self.btn_run = ttk.Button(self.content_frame, text="Run", command=self.save_settings)
        self.btn_run.grid(row=9, column=3, sticky="ew", **self.pad)
        
        # Column weights inside content frame
        for col in range(4):
            self.content_frame.columnconfigure(col, weight=1)
        
        # Status bar at bottom
        self.status_label = tk.Label(self, text="", anchor="w", relief="sunken", padx=20, pady=4)
        self.status_label.grid(row=1, column=0, sticky="we")  # no padding
    
    def refresh_window_list(self):
        """Refresh the list of available windows with optimized performance"""
        try:
            windows = list_windows()

            if not windows:
                messagebox.showwarning(
                    UI_TEXTS[self.language]["Warning"],
                    UI_TEXTS[self.language]["No windows found"]
                )
                return

            window_list = []
            self._window_objects = []

            for title, w, h, x, y, handle in windows:
                size_str = f"{w}x{h}"
                window_list.append(f"{title} [{size_str}]")
                # Store as tuple
                self._window_objects.append((title, w, h, x, y, handle))

            self.window_cb["values"] = window_list
            if window_list:
                self.window_cb.current(0)
                self.on_window_selected()

        except Exception as e:
            messagebox.showerror(
                UI_TEXTS[self.language]["Error"],
                f"Error refreshing window list: {str(e)}"
            )
    def refresh_monitor_and_window(self):
        """Allow user to get latest monitor/window list"""
        if self.capture_mode_key == "Window":
            self.refresh_window_list()
        else:
            self.populate_monitors()
        self.update_host_url() # refresh URL
    
    def update_host_url(self, *args):
        """Update the host URL when port changes and validate the port number"""
        port = self.streamer_port_var.get()
        if port:
            try:
                port_num = int(port)
                if 1 <= port_num <= 65535:
                    self.streamer_host_var.set(f"http://{get_local_ip()}:{port_num}")
                else:
                    messagebox.showerror(
                        UI_TEXTS[self.language]["Error"],
                        UI_TEXTS[self.language].get("Invalid port number (1-65535)", "Invalid port number (must be between 1-65535)")
                    )
                    # Reset to default port if invalid
                    self.streamer_port_var.set(str(DEFAULTS.get("Streamer Port", 1400)))
            except ValueError:
                messagebox.showerror(
                    UI_TEXTS[self.language]["Error"],
                    UI_TEXTS[self.language].get("Invalid port number", "Port must be a number")
                )
                # Reset to default port if invalid
                self.streamer_port_var.set(str(DEFAULTS.get("Streamer Port", 1400)))

    def on_window_selected(self, event=None):
        """Handle window selection from the combobox"""
        if not hasattr(self, "_window_objects") or not self._window_objects:
            return

        selected_text = self.window_var.get()
        if not selected_text:
            return

        selected_index = None
        for i, (title, w, h, x, y, handle) in enumerate(self._window_objects):
            size = f"{w}x{h}" if w and h else "Unknown size"
            if selected_text == f"{title} [{size}]":
                selected_index = i
                break

        if selected_index is not None:
            title, w, h, x, y, handle = self._window_objects[selected_index]
            self.selected_window_coords = (int(x), int(y), int(w), int(h))
            self.update_status(
                f"{UI_TEXTS[self.language]['Selected window:']} {title} ({w}x{h})"
            )

    def on_capture_mode_change(self, *args):
        """Show/hide monitor or window controls based on capture mode"""
        label = self.capture_mode_var_label.get()
        texts = UI_TEXTS[self.language]
        monitor_label = texts.get("Monitor", "Monitor")
        
        # Update the internal capture_mode_key based on the selected label
        if label == monitor_label:
            self.capture_mode_key = "Monitor"
            self.monitor_menu.grid(row=2, column=1, columnspan=2, sticky="w", **self.pad)
            self.window_cb.grid_remove()
        else:  # Window
            self.capture_mode_key = "Window"
            self.monitor_menu.grid_remove()
            self.window_cb.grid(row=2, column=1, columnspan=2, sticky="ew", **self.pad)
            # Refresh window list automatically when switching to Window mode
            self.refresh_window_list()

    def update_language_texts(self):
        texts = UI_TEXTS[self.language]
        self.btn_refresh.config(text=texts["Refresh"])
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
        self.btn_stop.config(text=texts["Stop"])
        self.btn_run.config(text=texts["Run"])
        self.label_language.config(text=texts["Set Language:"])
        # Update run mode labels & combobox values
        self.label_run_mode.config(text=texts.get("Run Mode:", "Run Mode:"))
        localized_run_vals = [texts.get("Viewer", "Viewer"), texts.get("Streamer", "Streamer")]
        self.run_mode_cb["values"] = localized_run_vals
        

        # Select the appropriate label
        if self.run_mode_key == "Viewer":
            self.run_mode_var_label.set(localized_run_vals[0])
        else:
            self.run_mode_var_label.set(localized_run_vals[1])
            
        # Update capture mode combobox values
        localized_capture_vals = [texts.get("Monitor", "Monitor"), texts.get("Window", "Window")]
        self.capture_mode_cb["values"] = localized_capture_vals
        
        # Select the appropriate label based on current capture_mode_key
        if self.capture_mode_key == "Monitor":
            self.capture_mode_var_label.set(localized_capture_vals[0])
        else:
            self.capture_mode_var_label.set(localized_capture_vals[1])
        
        # Trigger the capture mode change handler to update UI
        self.on_capture_mode_change()

        # Streamer host/port labels
        self.label_streamer_host.config(text=texts.get("Streamer URL", "Streamer URL"))
        self.label_streamer_port.config(text=texts.get("Streamer Port:", "Streamer Port:"))

        # language combobox values
        self.language_cb["values"] = list(UI_TEXTS.keys())
        
        # Update status bar translation
        if hasattr(self, "status_label"):
            current_text = self.status_label.cget("text")
            mapping = {
                "Loaded settings.yaml at startup": texts["Loaded settings.yaml at startup"],
                "Running": texts["Running"],
                "Stopped": texts["Stopped"],
                "Settings saved to settings.yaml, starting...": texts["Countdown"],
                "启动时已加载 settings.yaml": texts["Loaded settings.yaml at startup"],
                "运行中...": texts["Running"],
                "已停止。": texts["Stopped"],
                "设置已保存到 settings.yaml，启动中...": texts["Countdown"]
            }
            if current_text in mapping:
                self.status_label.config(text=mapping[current_text])

    def on_language_change(self, event):
        selected = self.language_var.get()
        if selected in UI_TEXTS:
            self.language = selected
            self.update_language_texts()

    def on_run_mode_change(self, event=None):
        """Toggle visibility of streamer-specific controls when run mode changes."""
        label = self.run_mode_var_label.get()
        texts = UI_TEXTS[self.language]
        streamer_label = texts.get("Streamer", "Streamer")
        if label == streamer_label:
            self.run_mode_key = "Streamer"
            if not self.streamer_port_var.get():
                self.streamer_port_var.set(str(DEFAULTS.get("Streamer Port", 1400)))
            # populate host with detected local IP if empty
            self.streamer_host_var.set(f"http://{get_local_ip()}:{self.streamer_port_var.get()}")
            # grid the controls
            self.label_streamer_host.grid(row=1, column=0, sticky="w", padx=8, pady=6)
            self.streamer_host_entry.grid(row=1, column=1, sticky="ew", padx=8, pady=6)
            self.label_streamer_port.grid(row=1, column=2, sticky="w", padx=8, pady=6)
            self.streamer_port_entry.grid(row=1, column=3, sticky="ew", padx=8, pady=6)
        else:
            self.run_mode_key = "Viewer"
            # hide streamer controls
            self.label_streamer_host.grid_remove()
            self.streamer_host_entry.grid_remove()
            self.label_streamer_port.grid_remove()
            self.streamer_port_entry.grid_remove()

    def browse_download(self):
        path = filedialog.askdirectory(initialdir=self.download_var.get() or ".")
        if path:
            self.download_var.set(path)
    
    def populate_devices(self):
        self.device_label_to_index = {}
        device_dict = DEVICES
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
            # Convert any tuples to lists before saving
            if isinstance(cfg.get("Capture Coordinates"), tuple):
                cfg["Capture Coordinates"] = list(cfg["Capture Coordinates"])
                
            with open(path, "w", encoding="utf-8") as f:
                yaml.dump(cfg, f, allow_unicode=True, sort_keys=False)
            return True
        except Exception as e:
            messagebox.showerror(UI_TEXTS[self.language]["Error"],
                                f"{UI_TEXTS[self.language]['Failed to save settings.yaml:']} {e}")
            return False

    def apply_config(self, cfg, keep_optional=True):
        # Convert any lists to tuples for Capture Coordinates
        if isinstance(cfg.get("Capture Coordinates"), list):
            cfg["Capture Coordinates"] = tuple(cfg["Capture Coordinates"])
        
        # Monitor settings
        monitor_idx = cfg.get("Monitor Index", DEFAULTS["Monitor Index"])
        label_for_idx = next((lbl for lbl, i in self.monitor_label_to_index.items() if i == monitor_idx), None)
        if label_for_idx:
            self.monitor_var.set(label_for_idx)
        elif self.monitor_label_to_index:
            self.monitor_var.set(next(iter(self.monitor_label_to_index)))

        # Window settings
        self.window_var.set(cfg.get("Window Title", DEFAULTS["Window Title"]))
        self.selected_window_coords = cfg.get("Capture Coordinates", DEFAULTS["Capture Coordinates"])

        if not keep_optional:  # no update for device
            device_idx = cfg.get("Device", DEFAULTS["Device"])
            label_for_device_idx = next((lbl for lbl, i in self.device_label_to_index.items() if i == device_idx), None)
            if label_for_device_idx:
                self.device_var.set(label_for_device_idx)
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
        if not keep_optional:  # no update for language
            self.language_var.set(cfg.get("Language", DEFAULTS["Language"]))

        # Run mode + streamer settings
        run_mode = cfg.get("Run Mode", DEFAULTS.get("Run Mode", "Viewer"))
        self.run_mode_key = run_mode
        host = cfg.get("Streamer Host") or DEFAULTS.get("Streamer Host")
        port = cfg.get("Streamer Port", DEFAULTS.get("Streamer Port", 1400))
        if host:
            self.streamer_host_var.set(f"http://{host}:{port}")
        else:
            # default local ip
            self.streamer_host_var.set(f"http://{get_local_ip()}:{port}")
        self.streamer_port_var.set(str(port))
        # Capture mode
        capture_mode = cfg.get("Capture Mode", DEFAULTS.get("Capture Mode", "Monitor"))
        self.capture_mode_key = capture_mode
        # Update run mode visibility
        self.update_language_texts()
        self.on_run_mode_change()
        self.on_capture_mode_change()

    def load_defaults(self):
        self.apply_config(DEFAULTS)

    def reset_to_defaults(self):
        self.load_defaults()
    
    def update_status(self, msg: str):
        """Update status bar text."""
        self.status_label.config(text=msg)

    def save_settings(self):
        # Validate port
        try:
            port_val = int(self.streamer_port_var.get()) if self.streamer_port_var.get() else DEFAULTS.get("Streamer Port", 1400)
            if not (1 <= port_val <= 65535):
                raise ValueError("Port out of range")
        except Exception:
            messagebox.showerror(UI_TEXTS[self.language]["Error"], f"Invalid port: {self.streamer_port_var.get()}")
            return

        cfg = {
            "Capture Mode": self.capture_mode_key,
            "Monitor Index": self.monitor_label_to_index.get(self.monitor_var.get(), DEFAULTS["Monitor Index"]),
            "Capture Coordinates": self.selected_window_coords,
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
            "Device": self.device_label_to_index.get(self.device_var.get()),
            "Language": self.language,
            "Run Mode": self.run_mode_key,
            "Streamer Port": port_val,
        }
        success = self.save_yaml("settings.yaml", cfg)
        if success:
            # Show a message with countdown
            countdown_seconds = 0.5
            self._countdown_and_run(countdown_seconds)

    def _countdown_and_run(self, seconds):
        if self.process and self.process.poll() is None:
            # Process is already running
            messagebox.showwarning(
                UI_TEXTS[self.language]["Warning"],
                UI_TEXTS[self.language]["A thread already running!"]
            )
            return

        if seconds > 0:
            self.update_status(
                UI_TEXTS[self.language]["Countdown"].format(seconds=seconds)
            )
            self.after(1000, lambda: self._countdown_and_run(seconds - 1))
        else:
            try:
                print(f"[Main] Initializing Desktop2Stereo {self.run_mode_key}…")
                cmd = [sys.executable, "main.py"]

                self.process = subprocess.Popen(cmd)
                self.update_status(UI_TEXTS[self.language]["Running"])
                self._monitor_process()  # start monitoring after launch
            except Exception as e:
                messagebox.showerror(
                    UI_TEXTS[self.language]["Error"],
                    f"Failed to run process: {e}"
                )
                self.update_status(UI_TEXTS[self.language]["Stopped"])

    def _monitor_process(self):
        """Check if process is still running; update label if stopped externally."""
        if self.process and self.process.poll() is not None:
            # Process ended or was killed outside
            self.process = None
            self.update_status(UI_TEXTS[self.language]["Stopped"])
        else:
            # Keep checking every second
            self.after(1000, self._monitor_process)
                    
    def stop_process(self):
        if self.process and self.process.poll() is None:  # still running
            try:
                self.process.terminate()
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
            except Exception as e:
                messagebox.showerror(
                    UI_TEXTS[self.language]["Error"],
                    f"Failed to stop process: {e}"
                )
            finally:
                self.process = None
                self.update_status(UI_TEXTS[self.language]["Stopped"])
                print(f"[Main] {self.run_mode_key} Stopped")
        else:
            self.update_status(UI_TEXTS[self.language]["Stopped"])
            print(f"[Main] {self.run_mode_key} Stopped")

if __name__ == "__main__":
    app = ConfigGUI()
    app.mainloop()

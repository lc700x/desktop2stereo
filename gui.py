import os
import sys, time
import subprocess
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk
from utils import VERSION, OS_NAME, DEFAULT_MODEL_LIST, crop_icon

# Add pywinalt import
try:
    import pywinctl as pwc
    HAVE_PYWINCTL = True
except ImportError:
    HAVE_PYWINCTL = False

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
}

UI_TEXTS = {
    "EN": {
        "Capture Mode:": "Capture Mode:",
        "Monitor Index:": "Monitor Index:",
        "Window Title:": "Window Title:",
        "Select Window": "Select Window",
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
        "Countdown": "Settings saved to settings.yaml, starting Stereo Viewer...",
        "Window selection not available": "Window selection not available (pywinctl not installed)",
        "No windows found": "No windows found",
        "Selected window:": "Selected window:"
    },
    "CN": {
        "Capture Mode:": "捕获模式:",
        "Monitor Index:": "显示器索引:",
        "Window Title:": "窗口标题:",
        "Select Window": "选择窗口",
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
        "HF Endpoint:": "HF 接口:",
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
        "Countdown": "设置已保存到 settings.yaml，启动Stereo Viewer...",
        "Window selection not available": "窗口选择不可用 (未安装pywinctl)",
        "No windows found": "未找到窗口",
        "Selected window:": "已选择窗口:"
    }
}

class ConfigGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(f"Desktop2Stereo v{VERSION} GUI")
        self.minsize(780, 500)  # Increased height for new controls
        self.resizable(False, False)
        self.language = "EN"
        self.loaded_model_list = DEFAULT_MODEL_LIST.copy()
        self.selected_window_coords = None

        try:
            icon_img = Image.open("icon.png")
            if OS_NAME == "Windows":
                icon_img = crop_icon(icon_img)
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
                self.update_status(UI_TEXTS[self.language]["Loaded settings.yaml at startup"])
            except Exception as e:
                print(f"Failed to load settings.yaml: {e}")
                self.load_defaults()
                self.update_language_texts()
        else:
            self.load_defaults()
            self.update_language_texts()

        self.language_var.set(self.language)
        self.process = None  # Keep track of the spawned process

    def create_widgets(self):
        pad = {"padx": 8, "pady": 6}

        # Main content frame
        self.content_frame = ttk.Frame(self)
        self.content_frame.grid(row=0, column=0, sticky="nsew", padx=40, pady=20)
        
        # Configure root row/column weights
        self.rowconfigure(0, weight=1)   # content frame expands
        self.rowconfigure(1, weight=0)   # status bar fixed
        self.columnconfigure(0, weight=1)
        
        # Capture Mode
        self.label_capture_mode = ttk.Label(self.content_frame, text="Capture Mode:")
        self.label_capture_mode.grid(row=0, column=0, sticky="w", **pad)
        self.capture_mode_var = tk.StringVar()
        self.capture_mode_menu = ttk.OptionMenu(
            self.content_frame, self.capture_mode_var, "Monitor", 
            "Monitor", "Window", command=self.on_capture_mode_change
        )
        self.capture_mode_menu.grid(row=0, column=1, sticky="w", **pad)
        
        # Monitor Index (only shown when capture mode is Monitor)
        self.label_monitor = ttk.Label(self.content_frame, text="Monitor Index:")
        self.label_monitor.grid(row=1, column=0, sticky="w", **pad)
        self.monitor_var = tk.StringVar()
        self.monitor_menu = ttk.OptionMenu(self.content_frame, self.monitor_var, "")
        self.monitor_menu.grid(row=1, column=1, sticky="w", **pad)
        
        # Window Title (only shown when capture mode is window)
        self.label_window = ttk.Label(self.content_frame, text="Window Title:")
        self.label_window.grid(row=1, column=0, sticky="w", **pad)
        self.window_var = tk.StringVar()
        self.window_entry = ttk.Entry(self.content_frame, textvariable=self.window_var)
        self.window_entry.grid(row=1, column=1, sticky="ew", **pad)
        self.btn_select_window = ttk.Button(
            self.content_frame, 
            text="Select Window", 
            command=self.select_window
        )
        self.btn_select_window.grid(row=1, column=2, sticky="ew", **pad)
        
        # Language
        self.label_language = ttk.Label(self.content_frame, text="Set Language:")
        self.label_language.grid(row=0, column=2, sticky="we", **pad)
        self.language_var = tk.StringVar()
        self.language_cb = ttk.Combobox(
            self.content_frame, textvariable=self.language_var, state="readonly",
            values=["English", "简体中文"]
        )
        self.language_cb.grid(row=0, column=3, sticky="ew", **pad)
        self.language_cb.bind("<<ComboboxSelected>>", self.on_language_change)
        
        # Device
        self.label_device = ttk.Label(self.content_frame, text="Device:")
        self.label_device.grid(row=2, column=0, sticky="w", **pad)
        self.device_var = tk.StringVar()
        self.device_menu = ttk.OptionMenu(self.content_frame, self.device_var, "")
        self.device_menu.grid(row=2, column=1, sticky="w", **pad)
        
        # FP16 and Show FPS
        self.fp16_var = tk.BooleanVar()
        self.fp16_cb = ttk.Checkbutton(self.content_frame, text="FP16", variable=self.fp16_var)
        self.fp16_cb.grid(row=2, column=2, sticky="w", **pad)
        
        self.showfps_var = tk.BooleanVar()
        self.showfps_cb = ttk.Checkbutton(self.content_frame, text="Show FPS", variable=self.showfps_var)
        self.showfps_cb.grid(row=2, column=3, sticky="w", **pad)
        
        # Output Resolution
        self.label_res = ttk.Label(self.content_frame, text="Output Resolution:")
        self.label_res.grid(row=3, column=0, sticky="w", **pad)
        self.res_values = ["480", "720", "1080", "1440", "2160"]
        self.res_cb = ttk.Combobox(self.content_frame, values=self.res_values, state="normal")
        self.res_cb.grid(row=3, column=1, sticky="ew", **pad)
        
        # FPS
        self.label_fps = ttk.Label(self.content_frame, text="FPS:")
        self.label_fps.grid(row=3, column=2, sticky="w", **pad)
        self.fps_values = ["30", "60", "75", "90", "120"]
        self.fps_cb = ttk.Combobox(self.content_frame, values=self.fps_values, state="normal")
        self.fps_cb.grid(row=3, column=3, sticky="ew", **pad)
        
        # Download path
        self.label_download = ttk.Label(self.content_frame, text="Download Path:")
        self.label_download.grid(row=4, column=0, sticky="w", **pad)
        self.download_var = tk.StringVar()
        self.download_entry = ttk.Entry(self.content_frame, textvariable=self.download_var)
        self.download_entry.grid(row=4, column=1, columnspan=2, sticky="ew", **pad)
        self.btn_browse = ttk.Button(self.content_frame, text="Browse...", command=self.browse_download)
        self.btn_browse.grid(row=4, column=3, sticky="ew", **pad)
        
        # Depth Resolution and Depth Strength
        self.label_depth_res = ttk.Label(self.content_frame, text="Depth Resolution:")
        self.label_depth_res.grid(row=5, column=0, sticky="w", **pad)
        self.depth_res_values = ["48", "96", "192", "384", "576", "768", "960", "1152", "1344", "1536"]
        self.depth_res_cb = ttk.Combobox(self.content_frame, values=self.depth_res_values, state="normal")
        self.depth_res_cb.grid(row=5, column=1, sticky="ew", **pad)
        
        self.label_depth_strength = ttk.Label(self.content_frame, text="Depth Strength:")
        self.label_depth_strength.grid(row=5, column=2, sticky="w", **pad)
        self.depth_strength_values = ["1.0", "2.0", "3.0", "4.0", "5.0"]
        self.depth_strength_cb = ttk.Combobox(self.content_frame, values=self.depth_strength_values, state="normal")
        self.depth_strength_cb.grid(row=5, column=3, sticky="ew", **pad)
        
        # Display Mode
        self.label_display_mode = ttk.Label(self.content_frame, text="Display Mode:")
        self.label_display_mode.grid(row=6, column=0, sticky="w", **pad)
        self.display_mode_values = ["Half-SBS", "Full-SBS", "TAB"]
        self.display_mode_cb = ttk.Combobox(self.content_frame, values=self.display_mode_values, state="readonly")
        self.display_mode_cb.grid(row=6, column=1, sticky="ew", **pad)
        
        # IPD
        self.label_ipd = ttk.Label(self.content_frame, text="IPD (m):")
        self.label_ipd.grid(row=6, column=2, sticky="w", **pad)
        self.ipd_var = tk.StringVar()
        self.ipd_entry = ttk.Entry(self.content_frame, textvariable=self.ipd_var)
        self.ipd_entry.grid(row=6, column=3, sticky="ew", **pad)
        
        # Depth Model
        self.label_depth_model = ttk.Label(self.content_frame, text="Depth Model:")
        self.label_depth_model.grid(row=7, column=0, sticky="w", **pad)
        self.depth_model_var = tk.StringVar()
        self.depth_model_cb = ttk.Combobox(self.content_frame, textvariable=self.depth_model_var, values=self.loaded_model_list, state="normal")
        self.depth_model_cb.grid(row=7, column=1, columnspan=2, sticky="ew", **pad)
        
        # HF Endpoint
        self.label_hf_endpoint = ttk.Label(self.content_frame, text="HF Endpoint:")
        self.label_hf_endpoint.grid(row=8, column=0, sticky="w", **pad)
        self.hf_endpoint_var = tk.StringVar()
        self.hf_endpoint_entry = ttk.Entry(self.content_frame, textvariable=self.hf_endpoint_var)
        self.hf_endpoint_entry.grid(row=8, column=1, sticky="ew", **pad)
        
        # Buttons
        self.btn_reset = ttk.Button(self.content_frame, text="Reset", command=self.reset_to_defaults)
        self.btn_reset.grid(row=7, column=3, sticky="ew", **pad)
        
        self.btn_stop = ttk.Button(self.content_frame, text="Stop", command=self.stop_process)
        self.btn_stop.grid(row=8, column=2, sticky="ew", **pad)
        
        self.btn_run = ttk.Button(self.content_frame, text="Run", command=self.save_settings)
        self.btn_run.grid(row=8, column=3, sticky="ew", **pad)
        
        # Column weights inside content frame
        for col in range(4):
            self.content_frame.columnconfigure(col, weight=1)
        
        # Status bar at bottom
        self.status_label = tk.Label(self, text="", anchor="w", relief="sunken", padx=20, pady=4)
        self.status_label.grid(row=1, column=0, sticky="we")  # no padding

        # Initialize capture mode UI
        self.on_capture_mode_change()

    def on_capture_mode_change(self, *args):
        """Show/hide monitor or window controls based on capture mode"""
        mode = self.capture_mode_var.get()
        if mode == "Monitor":
            self.label_monitor.grid()
            self.monitor_menu.grid()
            self.label_window.grid_remove()
            self.window_entry.grid_remove()
            self.btn_select_window.grid_remove()
        else:  # window
            self.label_monitor.grid_remove()
            self.monitor_menu.grid_remove()
            self.label_window.grid()
            self.window_entry.grid()
            self.btn_select_window.grid()

    def select_window(self):
        """Let user select a window interactively with better error handling"""
        if not HAVE_PYWINCTL:
            messagebox.showerror(
                UI_TEXTS[self.language]["Error"],
                UI_TEXTS[self.language]["Window selection not available"]
            )
            return

        try:
            # Try multiple times to get windows with a small delay
            windows = []
            for _ in range(3):  # Try up to 3 times
                try:
                    windows = pwc.getWindowsWithTitle("")
                    if windows:
                        break
                    time.sleep(0.5)  # Short delay before retry
                except Exception as e:
                    print(f"Window detection error: {e}")
                    time.sleep(0.5)
                    continue

            if not windows:
                # Try alternative method to get windows
                try:
                    windows = pwc.getAllWindows()
                    windows = [w for w in windows if w.title]  # Filter windows with titles
                except Exception as e:
                    print(f"Alternative window detection failed: {e}")

            if not windows:
                error_msg = UI_TEXTS[self.language]["No windows found"]
                
                # Add platform-specific troubleshooting tips
                if OS_NAME == "Darwin":
                    error_msg += "\n\nmacOS Tip: Please ensure the app has Screen Recording permission in System Settings > Privacy & Security"
                elif OS_NAME == "Linux":
                    error_msg += "\n\nLinux Tip: Try running under X11 instead of Wayland if possible"
                
                messagebox.showwarning(
                    UI_TEXTS[self.language]["Warning"],
                    error_msg
                )
                return

            # Create window selection dialog
            select_dialog = tk.Toplevel(self)
            select_dialog.title(UI_TEXTS[self.language]["Select Window"])
            select_dialog.resizable(True, True)
            
            # Add filter entry
            filter_frame = ttk.Frame(select_dialog)
            filter_frame.pack(fill=tk.X, padx=5, pady=5)
            
            filter_label = ttk.Label(filter_frame, text="Filter:")
            filter_label.pack(side=tk.LEFT)
            
            filter_var = tk.StringVar()
            filter_entry = ttk.Entry(filter_frame, textvariable=filter_var)
            filter_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
            
            # Listbox with scrollbars
            list_frame = ttk.Frame(select_dialog)
            list_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            
            scroll_y = ttk.Scrollbar(list_frame)
            scroll_y.pack(side=tk.RIGHT, fill=tk.Y)
            
            scroll_x = ttk.Scrollbar(list_frame, orient=tk.HORIZONTAL)
            scroll_x.pack(side=tk.BOTTOM, fill=tk.X)
            
            listbox = tk.Listbox(
                list_frame,
                width=80,
                height=20,
                yscrollcommand=scroll_y.set,
                xscrollcommand=scroll_x.set,
                selectmode=tk.SINGLE
            )
            listbox.pack(fill=tk.BOTH, expand=True)
            
            scroll_y.config(command=listbox.yview)
            scroll_x.config(command=listbox.xview)
            
            # Populate listbox
            window_list = []
            for win in windows:
                title = win.title if win.title else "Untitled"
                try:
                    size = f"{win.width}x{win.height}" if win.width and win.height else "Unknown size"
                    listbox.insert(tk.END, f"{title} [{size}]")
                    window_list.append(win)
                except Exception as e:
                    print(f"Error processing window {title}: {e}")
                    continue
            
            # Filter function
            def update_filter(*args):
                filter_text = filter_var.get().lower()
                listbox.delete(0, tk.END)
                for win in windows:
                    title = win.title if win.title else "Untitled"
                    try:
                        size = f"{win.width}x{win.height}" if win.width and win.height else "Unknown size"
                        if filter_text in title.lower() or filter_text in size.lower():
                            listbox.insert(tk.END, f"{title} [{size}]")
                    except:
                        continue
            
            filter_var.trace("w", update_filter)
            
            # Selection handler
            def on_select():
                selected_idx = listbox.curselection()
                if selected_idx:
                    selected_win = window_list[selected_idx[0]]
                    self.window_var.set(selected_win.title)
                    self.selected_window_coords = (
                        selected_win.left,
                        selected_win.top,
                        selected_win.width,
                        selected_win.height
                    )
                    self.update_status(
                        f"{UI_TEXTS[self.language]['Selected window:']} {selected_win.title} "
                        f"({selected_win.width}x{selected_win.height})"
                    )
                    select_dialog.destroy()
            
            # Button frame
            button_frame = ttk.Frame(select_dialog)
            button_frame.pack(fill=tk.X, padx=5, pady=5)
            
            select_btn = ttk.Button(button_frame, text="Select", command=on_select)
            select_btn.pack(side=tk.RIGHT)
            
            refresh_btn = ttk.Button(button_frame, text="Refresh", command=lambda: self.select_window())
            refresh_btn.pack(side=tk.RIGHT, padx=5)
            
            cancel_btn = ttk.Button(button_frame, text="Cancel", command=select_dialog.destroy)
            cancel_btn.pack(side=tk.RIGHT)
            
            # Center the dialog
            select_dialog.update_idletasks()
            x = self.winfo_x() + (self.winfo_width() - select_dialog.winfo_width()) // 2
            y = self.winfo_y() + (self.winfo_height() - select_dialog.winfo_height()) // 2
            select_dialog.geometry(f"+{x}+{y}")
            
            # Set focus to filter entry
            filter_entry.focus_set()
            
        except Exception as e:
            messagebox.showerror(
                UI_TEXTS[self.language]["Error"],
                f"Error selecting window: {str(e)}"
            )
    def update_language_texts(self):
        texts = UI_TEXTS[self.language]
        self.label_capture_mode.config(text=texts["Capture Mode:"])
        self.label_monitor.config(text=texts["Monitor Index:"])
        self.label_window.config(text=texts["Window Title:"])
        self.btn_select_window.config(text=texts["Select Window"])
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
        self.language_cb["values"] = list(UI_TEXTS.keys())
        
        # Update status bar translation
        if hasattr(self, "status_label"):
            current_text = self.status_label.cget("text")
            mapping = {
                "Loaded settings.yaml at startup": texts["Loaded settings.yaml at startup"],
                "Running": texts["Running"],
                "Stopped": texts["Stopped"],
                "Settings saved to settings.yaml, starting Stereo Viewer...": texts["Countdown"],
                "启动时已加载 settings.yaml": texts["Loaded settings.yaml at startup"],
                "运行中...": texts["Running"],
                "已停止。": texts["Stopped"],
                "设置已保存到 settings.yaml，启动Stereo Viewer...": texts["Countdown"]
            }
            if current_text in mapping:
                self.status_label.config(text=mapping[current_text])

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
        # Capture mode
        self.capture_mode_var.set(cfg.get("Capture Mode", DEFAULTS["Capture Mode"]))
        
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

        if keep_optional:  # no update for device
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
        if keep_optional:  # no update for language
            self.language_var.set(cfg.get("Language", DEFAULTS["Language"]))

        # Update UI based on capture mode
        self.on_capture_mode_change()

    def load_defaults(self):
        self.apply_config(DEFAULTS)

    def reset_to_defaults(self):
        self.load_defaults()
    
    def update_status(self, msg: str):
        """Update status bar text."""
        self.status_label.config(text=msg)

    def save_settings(self):
        cfg = {
            "Capture Mode": self.capture_mode_var.get(),
            "Monitor Index": self.monitor_label_to_index.get(self.monitor_var.get(), DEFAULTS["Monitor Index"]),
            "Window Title": self.window_var.get(),
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
        }
        success = self.save_yaml("settings.yaml", cfg)
        if success:
            # Show a message with countdown
            countdown_seconds = 0.5
            self._countdown_and_run(countdown_seconds)

    def _countdown_and_run(self, seconds):
        if seconds > 0:
            self.update_status(
                UI_TEXTS[self.language]["Countdown"].format(seconds=seconds)
            )
            self.after(1000, lambda: self._countdown_and_run(seconds - 1))
        else:
            try:
                self.process = subprocess.Popen([sys.executable, "main.py"])
                self.update_status(UI_TEXTS[self.language]["Running"])
                self._monitor_process()  # start monitoring after launch
            except Exception as e:
                messagebox.showerror(
                    UI_TEXTS[self.language]["Error"],
                    f"Failed to run main.py: {e}"
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
                messagebox.showerror(UI_TEXTS[self.language]["Error"], f"Failed to stop process: {e}")
            finally:
                self.process = None
                self.update_status(UI_TEXTS[self.language]["Stopped"])
        else:
            self.update_status(UI_TEXTS[self.language]["Stopped"])

if __name__ == "__main__":
    app = ConfigGUI()
    app.mainloop()
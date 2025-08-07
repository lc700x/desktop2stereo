import numpy as np
from depth import OS_NAME

if OS_NAME == "Windows":
    from win32api import EnumDisplayMonitors, GetMonitorInfo
    from wincam import DXCamera
    import ctypes
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)  # Per-monitor DPI awareness (Windows 8.1+)
    except:
        ctypes.windll.user32.SetProcessDPIAware()  # Windows 7 fallback
    
    class DesktopGrabber:
        def __init__(self, monitor_index=1, downscale=1.0, show_monitor_info=True, fps=60):
            self.downscale = downscale
            self.monitor_index = monitor_index-1
            self.fps = fps
            # List monitors
            self.monitors = self.get_monitors()
            if show_monitor_info:
                print("Available monitors:")
                for i, mon in enumerate(self.monitors):
                    system_width, system_height = self.get_actual_resolution(mon)
                    print(f"  {i+1}: {system_width}x{system_height}")
            if self.monitor_index >= len(self.monitors):
                print(f"Monitor {monitor_index} not found, using primary monitor")
                self.monitor_index = 0
            mon = self.monitors[self.monitor_index]
            self.x, self.y, self.r, self.b = mon["Monitor"]
            self.system_width, self.system_height = self.get_actual_resolution(mon)
            self.scaled_width = round(self.system_width * self.downscale)
            self.scaled_height = round(self.system_height * self.downscale)
            self.camera = DXCamera(self.x, self.y, self.system_width, self.system_height, fps=self.fps)
            if show_monitor_info:
                print(f"Using monitor {monitor_index}: {self.system_width}x{self.system_height}")
                print(f"Downscale factor: {self.downscale}")
                print(f"Scaled resolution: {self.scaled_width}x{self.scaled_height}")
                
        def get_monitors(self):
            mon_info = [GetMonitorInfo(mon) for mon, _, _ in EnumDisplayMonitors()]
            sorted_mon_info = sorted(mon_info, key=lambda d: d['Device'])
            return sorted_mon_info
        
        def get_actual_resolution(self, mon):
            x, y, r, b = mon["Monitor"]
            return r - x, b - y
        
        def __enter__(self):
            self.camera.__enter__()
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            self.camera.__exit__(exc_type, exc_val, exc_tb)

        def grab(self):
            frame, _ = self.camera.get_rgb_frame()
            img_array = np.array(frame)
            return img_array, (self.scaled_height, self.scaled_width)

# --- Non-Windows Grabber ---
else:
    import mss
    class DesktopGrabber:
        def __init__(self, monitor_index=1, downscale=1.0, show_monitor_info=True):
            self.downscale = downscale
            self.monitor_index = monitor_index
            self._mss = mss.mss(with_cursor=True)
            if show_monitor_info:
                self._log_monitors()
            if monitor_index >= len(self._mss.monitors):
                print(f"Monitor {monitor_index} not found, using primary monitor")
                monitor_index = 1
            self._mon = self._mss.monitors[monitor_index]
            self.system_width, self.system_height = self.get_screen_resolution(monitor_index)
            self.scaled_width = round(self.system_width * self.downscale)
            self.scaled_height = round(self.system_height * self.downscale)
            if show_monitor_info:
                print(f"Using monitor {monitor_index}: {self.system_width}x{self.system_height}")
                print(f"Downscale factor: {self.downscale}")
                print(f"Scaled resolution: {self.scaled_width}x{self.scaled_height}")

        def _log_monitors(self):
            print("Available monitors:")
            for i, mon in enumerate(self._mss.monitors):
                width, height = self.get_screen_resolution(i)
                if i == 0:
                    print(f"  {i}: All monitors - {width}x{height}")
                else:
                    system_scale = int(width/mon["width"])
                    print(f"  {i}: Monitor {i} - {width}x{height} at ({mon['left']*system_scale}, {mon['top']*system_scale})")
        def get_screen_resolution(self, index):
            monitor = self._mss.monitors[index]
            screen = self._mss.grab(monitor)
            return screen.size.width, screen.size.height

        def grab(self):
            shot = self._mss.grab(self._mon)
            img = np.frombuffer(shot.rgb, dtype=np.uint8).reshape((self.system_height, self.system_width, 3))
            return img, (self.scaled_height, self.scaled_width)

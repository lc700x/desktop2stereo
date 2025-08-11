import time
import numpy as np
import mss
from depth import OS_NAME

if OS_NAME == "Windows":
    import ctypes
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except:
        ctypes.windll.user32.SetProcessDPIAware()
    from wincam import DXCamera
    import cv2  # For resizing (add to requirements)

    class DesktopGrabber:
        def __init__(self, monitor_index=1, output_resolution=1080, show_monitor_info=True, fps=60):
            self.scaled_height = output_resolution
            self.monitor_index = monitor_index
            self.fps = fps
            self.frame_interval = 1.0 / fps
            self._mss = mss.mss(with_cursor=True)

            if show_monitor_info:
                self._log_monitors()
            if monitor_index >= len(self._mss.monitors):
                print(f"Monitor {monitor_index} not found, using primary monitor")
                monitor_index = 1
            self._mon = self._mss.monitors[monitor_index]

            self.system_width, self.system_height = self.get_screen_resolution(monitor_index)
            self.scaled_width = round(self.system_width * self.scaled_height / self.system_height)
            self.system_scale = int(self.system_width / self._mon["width"])

            self.camera = DXCamera(
                self._mon['left'] * self.system_scale,
                self._mon['top'] * self.system_scale,
                self.system_width,
                self.system_height,
                fps=self.fps,
            )
            if show_monitor_info:
                print(f"Using monitor {monitor_index}: {self.system_width}x{self.system_height}")
                print(f"Scaled resolution: {self.scaled_width}x{self.scaled_height}")

            self._last_frame_time = 0

        def _log_monitors(self):
            print("Available monitors:")
            for i, mon in enumerate(self._mss.monitors):
                width, height = self.get_screen_resolution(i)
                if i == 0:
                    print(f"  {i}: All monitors - {width}x{height}")
                else:
                    system_scale = int(width / mon["width"])
                    print(f"  {i}: Monitor {i} - {width}x{height} at ({mon['left'] * system_scale}, {mon['top'] * system_scale})")

        def get_screen_resolution(self, index):
            monitor = self._mss.monitors[index]
            screen = self._mss.grab(monitor)
            return screen.size.width, screen.size.height

        def __enter__(self):
            self.camera.__enter__()
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            self.camera.__exit__(exc_type, exc_val, exc_tb)

        def grab(self):
            self._last_frame_time = time.time()
            img_array, _ = self.camera.get_rgb_frame()

            return img_array, (self.scaled_height, self.scaled_width)

else:
    import cv2

    def add_mouse(img):
        return img

    class DesktopGrabber:
        def __init__(self, monitor_index=1, output_resolution=1080, show_monitor_info=True, fps=60):
            self.scaled_height = output_resolution
            self.monitor_index = monitor_index
            self.fps = fps
            self.frame_interval = 1.0 / fps
            self._mss = mss.mss(with_cursor=True)
            if show_monitor_info:
                self._log_monitors()
            if monitor_index >= len(self._mss.monitors):
                print(f"Monitor {monitor_index} not found, using primary monitor")
                monitor_index = 1
            self._mon = self._mss.monitors[monitor_index]
            self.system_width, self.system_height = self.get_screen_resolution(monitor_index)
            self.scaled_width = round(self.system_width * self.scaled_height / self.system_height)
            self.system_scale = int(self.system_width / self._mon["width"])
            if show_monitor_info:
                print(f"Using monitor {monitor_index}: {self.system_width}x{self.system_height}")
                print(f"Scaled resolution: {self.scaled_width}x{self.scaled_height}")

            self._last_frame_time = 0

        def _log_monitors(self):
            print("Available monitors:")
            for i, mon in enumerate(self._mss.monitors):
                width, height = self.get_screen_resolution(i)
                if i == 0:
                    print(f"  {i}: All monitors - {width}x{height}")
                else:
                    system_scale = int(width / mon["width"])
                    print(f"  {i}: Monitor {i} - {width}x{height} at ({mon['left'] * system_scale}, {mon['top'] * system_scale})")

        def get_screen_resolution(self, index):
            monitor = self._mss.monitors[index]
            screen = self._mss.grab(monitor)
            return screen.size.width, screen.size.height

        def __enter__(self):
            # No camera context on non-Windows
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            pass

        def grab(self):
            now = time.time()
            elapsed = now - self._last_frame_time
            if elapsed < self.frame_interval:
                time.sleep(self.frame_interval - elapsed)
            self._last_frame_time = time.time()

            shot = self._mss.grab(self._mon)
            img = np.frombuffer(shot.rgb, dtype=np.uint8).reshape((self.system_height, self.system_width, 3))

            # Optional: avoid copy here, just pass img
            img = add_mouse(img)

            return img, (self.scaled_height, self.scaled_width)

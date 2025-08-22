import numpy as np
import mss
from utils import OS_NAME

# DesktopGrabber: wincam (Windows), MSS + AppKit (Mac), MSS (Linux) 
# Windows 10/11
if OS_NAME == "Windows":
    from wincam import DXCamera
    class DesktopGrabber:
        def __init__(self, output_resolution=1080, show_monitor_info=True, fps=60):
            self.scaled_height = output_resolution
            self.fps = fps
            self._mss = mss.mss()
            
            # Get capture coordinates from settings
            from utils import settings
            self.capture_mode = settings.get("Capture Mode", "monitor")
            self.capture_coords = settings.get("Capture Coordinates")
            
            if self.capture_mode == "monitor":
                monitor_index = settings.get("Monitor Index", 1)
                if show_monitor_info:
                    self._log_monitors()
                if monitor_index >= len(self._mss.monitors):
                    print(f"Monitor {monitor_index} not found, using primary monitor")
                    monitor_index = 1
                self._mon = self._mss.monitors[monitor_index]
                self.system_width, self.system_height = self.get_screen_resolution(monitor_index)
                self.system_scale = int(self.system_width / self._mon["width"])
                
                left = self._mon['left'] * self.system_scale
                top = self._mon['top'] * self.system_scale
                width = self.system_width
                height = self.system_height
            else:
                # Window capture mode
                if not self.capture_coords:
                    raise ValueError("No capture coordinates specified for window capture")
                left, top, width, height = self.capture_coords
                self.system_width = width
                self.system_height = height
                self.system_scale = 1  # Window coordinates are already in pixels
            
            self.scaled_width = round(self.system_width * self.scaled_height / self.system_height)
            
            if show_monitor_info:
                print(f"Capture area: {self.system_width}x{self.system_height} at ({left},{top})")
                print(f"Scaled resolution: {self.scaled_width}x{self.scaled_height}")

            self.camera = DXCamera(
                left,
                top,
                width,
                height,
                fps=self.fps,
            )

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
            img_array, _ = self.camera.get_rgb_frame()
            return img_array, (self.scaled_height, self.scaled_width)

elif OS_NAME == "Darwin":
    import io
    import numpy as np
    import cv2
    from PIL import Image
    from AppKit import NSCursor, NSBitmapImageRep, NSPNGFileType
    import Quartz.CoreGraphics as CG
    from Quartz import CGCursorIsVisible
    import mss

    class DesktopGrabber:
        def __init__(self, output_resolution=1080, show_monitor_info=True, fps=60, with_cursor=True):
            self.scaled_height = output_resolution
            self.fps = fps
            self.with_cursor = with_cursor
            self._mss = mss.mss()
            
            # Get capture coordinates from settings
            from utils import settings
            self.capture_mode = settings.get("Capture Mode", "monitor")
            self.capture_coords = settings.get("Capture Coordinates")
            
            if self.capture_mode == "monitor":
                monitor_index = settings.get("Monitor Index", 1)
                if show_monitor_info:
                    self._log_monitors()
                if monitor_index >= len(self._mss.monitors):
                    print(f"Monitor {monitor_index} not found, using primary monitor")
                    monitor_index = 1
                self._mon = self._mss.monitors[monitor_index]
                self.system_width, self.system_height = self.get_screen_resolution(monitor_index)
                self.system_scale = int(self.system_width / self._mon["width"])
                
                self.left = self._mon['left'] * self.system_scale
                self.top = self._mon['top'] * self.system_scale
                self.width = self.system_width
                self.height = self.system_height
            else:
                # Window capture mode
                if not self.capture_coords:
                    raise ValueError("No capture coordinates specified for window capture")
                self.left, self.top, self.width, self.height = self.capture_coords
                self.system_width = self.width
                self.system_height = self.height
                self.system_scale = 1  # Window coordinates are already in pixels
            
            self.scaled_width = round(self.system_width * self.scaled_height / self.system_height)

            if show_monitor_info:
                print(f"Capture area: {self.system_width}x{self.system_height} at ({self.left},{self.top})")
                print(f"Scaled resolution: {self.scaled_width}x{self.scaled_height}")

        def _log_monitors(self):
            print("Available monitors:")
            for i, mon in enumerate(self._mss.monitors):
                width, height = self.get_screen_resolution(i)
                if i == 0:
                    print(f"  {i}: All monitors - {width}x{height}")
                else:
                    system_scale = int(width / mon["width"])
                    print(f"  {i}: Monitor {i} - {width}x{height} at ({mon['left'] * systemscale}, {mon['top'] * system_scale})")

        def get_screen_resolution(self, index):
            monitor = self._mss.monitors[index]
            screen = self._mss.grab(monitor)
            return screen.size.width, screen.size.height

        def grab(self):
            # Create custom monitor dict for MSS
            monitor = {
                "left": self.left,
                "top": self.top,
                "width": self.width,
                "height": self.height
            }
            
            # Grab: mss gives a raw BGRA buffer (Pillow style). Convert once to numpy view.
            shot = self._mss.grab(monitor)
            arr = np.asarray(shot)  # BGRA uint8 view (no unnecessary copies if mss supports it)
            # Convert to BGR (drop alpha) using OpenCV (fast C path)
            frame_bgr = cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)
            
            # Load cursor and precompute alpha/premultiplied arrays for fast overlay
            if self.with_cursor and CGCursorIsVisible():
                x, y = get_cursor_position()
                # Convert to local frame coordinates
                cursor_x = x - self.left
                cursor_y = y - self.top
                # Only overlay if cursor is within this frame
                if 0 <= cursor_x <= self.width and 0 <= cursor_y <= self.height:
                    cursor_bgra, hotspot, alpha_f32, premultiplied = get_cursor_image_and_hotspot()
                    scale_factor = 16 // self.system_scale
                    if cursor_bgra.shape[0] > scale_factor and cursor_bgra.shape[1] > scale_factor:
                        h, w = cursor_bgra.shape[:2]
                        new_w, new_h = int(w / scale_factor), int(h / scale_factor)
                        # resize BGRA; keep alpha channel scaled properly
                        cursor_bgra = cv2.resize(cursor_bgra, (new_w, new_h), interpolation=cv2.INTER_AREA)

                        # recompute alpha & premultiplied after resize
                        alpha_f32 = (cursor_bgra[:, :, 3].astype(np.float32) / 255.0)
                        premultiplied = cursor_bgra[:, :, :3].astype(np.float32) * alpha_f32[:, :, None]

                        self.cursor_bgra = cursor_bgra
                        self.cursor_hotspot = hotspot
                        self.cursor_alpha = alpha_f32
                        self.cursor_premultiplied = premultiplied
                else:
                    self.cursor_bgra = None
                    self.cursor_hotspot = (0, 0)
                    self.cursor_alpha = None
                    self.cursor_premultiplied = None

                if self.with_cursor and self.cursor_bgra is not None:
                        # overlay in-place using optimized routine
                        frame_bgr = overlay_cursor_on_frame(
                            frame_bgr,
                            self.cursor_bgra,
                            self.cursor_hotspot,
                            (cursor_x, cursor_y),
                            alpha_f32=self.cursor_alpha,
                            premultiplied_bgr_f32=self.cursor_premultiplied
                        )
            # Return RGB image & scaled dimensions
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            return frame_rgb, (self.scaled_height, self.scaled_width)


else: # Linux and other platforms
    print("Use MSS as screen grabber.")
    import cv2
    class DesktopGrabber:
        def __init__(self, output_resolution=1080, show_monitor_info=True, fps=60):
            self.scaled_height = output_resolution
            self.fps = fps
            self._mss = mss.mss(with_cursor=True)
            
            # Get capture coordinates from settings
            from utils import settings
            self.capture_mode = settings.get("Capture Mode", "monitor")
            self.capture_coords = settings.get("Capture Coordinates")
            
            if self.capture_mode == "monitor":
                monitor_index = settings.get("Monitor Index", 1)
                if show_monitor_info:
                    self._log_monitors()
                if monitor_index >= len(self._mss.monitors):
                    print(f"Monitor {monitor_index} not found, using primary monitor")
                    monitor_index = 1
                self._mon = self._mss.monitors[monitor_index]
                self.system_width, self.system_height = self.get_screen_resolution(monitor_index)
                self.system_scale = int(self.system_width / self._mon["width"])
                
                self.left = self._mon['left'] * self.system_scale
                self.top = self._mon['top'] * self.system_scale
                self.width = self.system_width
                self.height = self.system_height
            else:
                # Window capture mode
                if not self.capture_coords:
                    raise ValueError("No capture coordinates specified for window capture")
                self.left, self.top, self.width, self.height = self.capture_coords
                self.system_width = self.width
                self.system_height = self.height
                self.system_scale = 1  # Window coordinates are already in pixels
            
            self.scaled_width = round(self.system_width * self.scaled_height / self.system_height)

            if show_monitor_info:
                print(f"Capture area: {self.system_width}x{self.system_height} at ({self.left},{self.top})")
                print(f"Scaled resolution: {self.scaled_width}x{self.scaled_height}")

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

        def grab(self):
            # Create custom monitor dict for MSS
            monitor = {
                "left": self.left,
                "top": self.top,
                "width": self.width,
                "height": self.height
            }
            
            shot = self._mss.grab(monitor)
            img = cv2.cvtColor(np.array(shot), cv2.COLOR_BGRA2RGB)
            return img, (self.scaled_height, self.scaled_width)
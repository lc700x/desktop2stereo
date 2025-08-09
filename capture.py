import numpy as np
from depth import OS_NAME

if OS_NAME == "Windows":
    import ctypes
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)  # Per-monitor DPI awareness (Windows 8.1+)
    except:
        ctypes.windll.user32.SetProcessDPIAware()  # Windows 7 fallback
    import numpy as np
    from wincam import DXCamera
    import mss
    
    # import win32gui, win32ui
    # from PIL import Image
    # import bettercam
    # import mss.windows
    # mss.windows.CAPTUREBLT = 0
    
    # # Define a function to add the mouse cursor to an image
    # def add_mouse(img):
    #     _, hcursor, (cx, cy) = win32gui.GetCursorInfo()
    #     try:
    #         cursor = get_cursor(hcursor)  # shape: (36, 36, 3)
    #         cursor_mean = cursor.mean(-1)
    #         where = np.where(cursor_mean > 0)
    #         for x, y in zip(where[0], where[1]):
    #             rgb = cursor[x, y]
    #             # Overlay cursor at (cy + x, cx + y)
    #             img = set_pixel(img, cy + x, cx + y, rgb=rgb)
    #     finally:
    #         return img
        
    # def set_pixel(img, x, y, rgb=(0,0,0)):
    #     """
    #     Set a pixel in a NumPy array of shape (Height, Width, Channels)
    #     """
    #     h, w, c = img.shape
    #     if 0 <= x < h and 0 <= y < w:
    #         img[x, y, :3] = rgb
    #     return img

    # def get_cursor(hcursor):
    #     info = win32gui.GetCursorInfo()
    #     hdc = win32ui.CreateDCFromHandle(win32gui.GetDC(0))
    #     hbmp = win32ui.CreateBitmap()
    #     hbmp.CreateCompatibleBitmap(hdc, 36, 36)
    #     hdc = hdc.CreateCompatibleDC()
    #     hdc.SelectObject(hbmp)
    #     hdc.DrawIcon((0,0), hcursor)
        
    #     bmpinfo = hbmp.GetInfo()
    #     bmpbytes = hbmp.GetBitmapBits()
    #     bmpstr = hbmp.GetBitmapBits(True)
    #     im = np.array(Image.frombuffer(
    #         'RGB',
    #         (bmpinfo['bmWidth'], bmpinfo['bmHeight']),
    #         bmpstr, 'raw', 'BGRX', 0, 1))
        
    #     win32gui.DestroyIcon(hcursor)    
    #     win32gui.DeleteObject(hbmp.GetHandle())
    #     hdc.DeleteDC()
    #     return im

    class DesktopGrabber:
        def __init__(self, monitor_index=1, output_resolution=1080, show_monitor_info=True, fps=60):
            self.scaled_height = output_resolution
            self.monitor_index = monitor_index
            self.fps = fps
            self._mss = mss.mss(with_cursor=True)
            if show_monitor_info:
                self._log_monitors()
            if monitor_index >= len(self._mss.monitors):
                print(f"Monitor {monitor_index} not found, using primary monitor")
                monitor_index = 1
            self._mon = self._mss.monitors[monitor_index]
            self.system_width, self.system_height = self.get_screen_resolution(monitor_index)
            self.scaled_width = round(self.system_width * self.scaled_height / self.system_height)
            self.system_scale = int(self.system_width/self._mon["width"])
            self.camera = DXCamera(self._mon['left']*self.system_scale, self._mon['top']*self.system_scale, self.system_width, self.system_height, fps=self.fps)
            if show_monitor_info:
                print(f"Using monitor {monitor_index}: {self.system_width}x{self.system_height}")
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
        
        def __enter__(self):
            self.camera.__enter__()
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            self.camera.__exit__(exc_type, exc_val, exc_tb)

        def grab(self):
            img_array, _ = self.camera.get_rgb_frame()
            return img_array, (self.scaled_height, self.scaled_width)

else:
    def add_mouse(img):
        return img

    import mss
    class DesktopGrabber:
        def __init__(self, monitor_index=1, output_resolution=1080, show_monitor_info=True, fps=60):
            self.scaled_height = output_resolution
            self.monitor_index = monitor_index
            self.fps = fps
            self._mss = mss.mss(with_cursor=True)
            if show_monitor_info:
                self._log_monitors()
            if monitor_index >= len(self._mss.monitors):
                print(f"Monitor {monitor_index} not found, using primary monitor")
                monitor_index = 1
            self._mon = self._mss.monitors[monitor_index]
            self.system_width, self.system_height = self.get_screen_resolution(monitor_index)
            self.scaled_width = round(self.system_width * self.scaled_height / self.system_height)
            self.system_scale = int(self.system_width/self._mon["width"])
            if show_monitor_info:
                print(f"Using monitor {monitor_index}: {self.system_width}x{self.system_height}")
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
        
        def __enter__(self):
            self.camera.__enter__()
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            self.camera.__exit__(exc_type, exc_val, exc_tb)

        def grab(self):
            shot = self._mss.grab(self._mon)
            img = np.frombuffer(shot.rgb, dtype=np.uint8).reshape((self.system_height, self.system_width, 3))
            img_array = add_mouse(img.copy())
            return img_array, (self.scaled_height, self.scaled_width)

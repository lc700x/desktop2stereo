# capture.py
import numpy as np
from depth import OS_NAME
import mss
from PIL import Image

if OS_NAME == "Windows":
    # capture cursor in Windows
    import win32gui, win32ui
    def set_pixel(img, w, x, y, rgb=(0,0,0)):
        """
        Set a pixel in a, RGB byte array
        """
        pos = (x*w + y)*3
        if pos>=len(img):return img # avoid setting pixel outside of frame
        img[pos:pos+3] = rgb
        return img

    def get_cursor(hcursor):
        hdc = win32ui.CreateDCFromHandle(win32gui.GetDC(0))
        hbmp = win32ui.CreateBitmap()
        hbmp.CreateCompatibleBitmap(hdc, 36, 36)
        hdc = hdc.CreateCompatibleDC()
        hdc.SelectObject(hbmp)
        hdc.DrawIcon((0,0), hcursor)
        bmpinfo = hbmp.GetInfo()
        bmpstr = hbmp.GetBitmapBits(True)
        im = np.array(Image.frombuffer(
            'RGB',
            (bmpinfo['bmWidth'], bmpinfo['bmHeight']),
            bmpstr, 'raw', 'BGRX', 0, 1))
        
        win32gui.DestroyIcon(hcursor)    
        win32gui.DeleteObject(hbmp.GetHandle())
        hdc.DeleteDC()
        return im

    def add_mouse(img, w):
        _, hcursor, (cx,cy) = win32gui.GetCursorInfo()
        # handle no cursor failure
        try: 
            cursor = get_cursor(hcursor)
            cursor_mean = cursor.mean(-1)
            where = np.where(cursor_mean>0)
            for x, y in zip(where[0], where[1]):
                rgb = [x for x in cursor[x,y]]
                img = set_pixel(img, w, x+cy, y+cx, rgb=rgb)
        finally: # if no cursor, draw nothing
            return img
elif OS_NAME == "Darwin":
    # TODO capture cursor in MacOS
    def add_mouse(img, w):
        return img


class DesktopGrabber:
    """Captures a specific monitor, optionally downscaled."""

    def __init__(self, monitor_index: int = 1, downscale: float = 1.0, show_monitor_info: bool = True):
        """
        Initialize the DesktopGrabber.

        Args:
            monitor_index (int): Index of the monitor to capture (0 for all monitors).
            downscale (float): Downscale factor (1.0 for no downscaling).
            show_monitor_info (bool): Whether to print monitor info on initialization.
        """
        self._mss = mss.mss(with_cursor=True)  # Only Support Linux
        self.downscale = downscale

        if show_monitor_info:
            self._log_monitors()

        if monitor_index >= len(self._mss.monitors):
            if show_monitor_info:
                print(f"Monitor {monitor_index} not found, using primary monitor")
            monitor_index = 1
        # get monitor resolution
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
                print(f"  {i}: Monitor {i} - {width}x{height}"
                             f"at ({mon['left']*system_scale}, {mon['top']*system_scale})")
    def get_screen_resolution(self, index):
        monitor = self._mss.monitors[index]
        screen = self._mss.grab(monitor)
        return screen.size.width, screen.size.height
    def grab(self) -> np.ndarray:
        """Capture the screen with mouse cursor and return a raw BGR image."""
        # Capture screen using the new method
        shot = self._mss.grab(self._mon)
        img = bytearray(shot.rgb)
        # Add mouse cursor to the image for Windows and Mac
        if OS_NAME != "Linux":
            img = add_mouse(img, self.system_width)
        # Convert to numpy array and reshape
        img_array = np.frombuffer(img, dtype=np.uint8)
        return img_array, (self.system_height, self.system_width)

# capture.py
import numpy as np
import win32gui, win32ui
import mss
from PIL import Image
import cv2
from depth import DEVICE
import torch
import torch.nn.functional as F

def set_pixel(img, w, x, y, rgb=(0,0,0)):
    """
    Set a pixel in a, RGB byte array
    """
    pos = (x*w + y)*3
    if pos>=len(img):return img # avoid setting pixel outside of frame
    img[pos:pos+3] = rgb
    return img

def get_cursor(hcursor):
    info = win32gui.GetCursorInfo()
    hdc = win32ui.CreateDCFromHandle(win32gui.GetDC(0))
    hbmp = win32ui.CreateBitmap()
    hbmp.CreateCompatibleBitmap(hdc, 36, 36)
    hdc = hdc.CreateCompatibleDC()
    hdc.SelectObject(hbmp)
    hdc.DrawIcon((0,0), hcursor)
    
    bmpinfo = hbmp.GetInfo()
    bmpbytes = hbmp.GetBitmapBits()
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
    flags, hcursor, (cx,cy) = win32gui.GetCursorInfo()
    cursor = get_cursor(hcursor)
    cursor_mean = cursor.mean(-1)
    where = np.where(cursor_mean>0)
    for x, y in zip(where[0], where[1]):
        rgb = [x for x in cursor[x,y]]
        img = set_pixel(img, w, x+cy, y+cx, rgb=rgb)
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
        self._mss = mss.mss()  # Removed with_cursor since we'll handle it manually
        self.downscale = downscale

        if show_monitor_info:
            self._log_monitors()

        if monitor_index >= len(self._mss.monitors):
            if show_monitor_info:
                print(f"Monitor {monitor_index} not found, using primary monitor")
            monitor_index = 1

        self._mon = self._mss.monitors[monitor_index]

        self.scaled_width = round(self._mon['width'] * self.downscale)
        self.scaled_height = round(self._mon['height'] * self.downscale)

        if show_monitor_info:
            print(f"Using monitor {monitor_index}: {self._mon['width']}x{self._mon['height']}")
            print(f"Downscale factor: {self.downscale}")
            print(f"Scaled resolution: {self.scaled_width}x{self.scaled_height}")

    def _log_monitors(self):
        print("Available monitors:")
        for i, mon in enumerate(self._mss.monitors):
            if i == 0:
                print(f"  {i}: All monitors - {mon['width']}x{mon['height']}")
            else:
                print(f"  {i}: Monitor {i} - {mon['width']}x{mon['height']} "
                             f"at ({mon['left']}, {mon['top']})")

    def grab(self) -> np.ndarray:
        """Capture the screen with mouse cursor and return a raw BGR image."""
        # Capture screen using the new method
        shot = self._mss.grab(self._mon)
        img = bytearray(shot.rgb)
        
        # Add mouse cursor to the image
        img_with_mouse = add_mouse(img, self._mon['width'])
        
        # Convert to numpy array and reshape
        img_array = np.frombuffer(img_with_mouse, dtype=np.uint8)
        img_array = img_array.reshape((self._mon['height'], self._mon['width'], 3))
        
        # Convert from RGB to BGR (to match original behavior)
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        return img_bgr

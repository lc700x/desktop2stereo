# capture.py
import mss
import numpy as np
import win32gui, win32ui
import mss
from PIL import Image
from depth import DEVICE
import torch
import torch.nn.functional as F
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
        self._mss = mss.mss(with_cursor=True) # not working on Windows
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
        """Capture the screen and return a raw BGR image (no scaling)."""
        shot = self._mss.grab(self._mon)
        img = bytearray(shot.rgb)
        
        # Add mouse cursor to the image
        img_with_mouse = add_mouse(img, self._mon['width'])
        
        # Convert to numpy array and reshape
        img_array = np.frombuffer(img_with_mouse, dtype=np.uint8)
        img_array = img_array.reshape((self._mon['height'], self._mon['width'], 3))
        
        return img_array

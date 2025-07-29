# capture.py
import mss
import numpy as np
import cv2

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
        img = np.array(shot)[:, :, :3]  # raw BGRA, drop alpha channel
        # img is already in BGRA byte order - mss returns from screen in BGRA order
        # But we stripped alpha, so img is BGR now (shape H,W,3)
        return img

    def process(self, img: np.ndarray) -> np.ndarray:
        """
        Process raw BGR image: convert to RGB and apply downscale if set.
        This can be called in a separate thread.
        """
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Downscale if requested
        if self.downscale < 1.0:
            img_rgb = cv2.resize(img_rgb, (self.scaled_width, self.scaled_height),
                                 interpolation=cv2.INTER_AREA)
        return img_rgb

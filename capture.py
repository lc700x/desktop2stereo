import mss
import numpy as np
import cv2

class DesktopGrabber:
    """Captures a specific monitor, optionally downscaled."""

    def __init__(self, monitor_index=1, downscale=1.0):
        self._mss = mss.mss()
        self.downscale = downscale

        print("Available monitors:")
        for i, mon in enumerate(self._mss.monitors):
            if i == 0:
                print(f"  {i}: All monitors - {mon['width']}x{mon['height']}")
            else:
                print(f"  {i}: Monitor {i} - {mon['width']}x{mon['height']} "
                      f"at ({mon['left']}, {mon['top']})")

        if monitor_index >= len(self._mss.monitors):
            print(f"Monitor {monitor_index} not found, using primary monitor")
            monitor_index = 1

        self._mon = self._mss.monitors[monitor_index]
        print(f"Using monitor {monitor_index}: {self._mon['width']}x{self._mon['height']}")

        # Calculate downscale reduced resolution
        self.scaled_width = int(self._mon['width'] * self.downscale)
        self.scaled_height = int(self._mon['height'] * self.downscale)

    def grab(self) -> np.ndarray:
        """Return RGB NumPy array, downscaled if needed"""

        # Grab the screen using mss
        shot = self._mss.grab(self._mon)

        # Convert to NumPy array and RGB format
        img = np.array(shot)[:, :, :3]  # Use only RGB not BGR
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)

        # Downscale if set
        if self.downscale < 1.0:
            img_rgb = cv2.resize(img_rgb, (self.scaled_width, self.scaled_height), interpolation=cv2.INTER_AREA)

        return img_rgb

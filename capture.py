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

    def grab(self) -> np.ndarray:
        """Return RGB NumPy array, downscaled if needed"""
        shot = self._mss.grab(self._mon)
        img = np.array(shot)[:, :, :3]
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.downscale < 1.0:
            h, w = img_rgb.shape[:2]
            new_w = int(w * self.downscale)
            new_h = int(h * self.downscale)
            img_rgb = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)

        return img_rgb

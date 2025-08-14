import numpy as np
import mss
from gui import OS_NAME

if OS_NAME == "Windows":
    from wincam import DXCamera
    class DesktopGrabber:
        def __init__(self, monitor_index=1, output_resolution=1080, show_monitor_info=True, fps=60):
            self.scaled_height = output_resolution
            self.monitor_index = monitor_index
            self.fps = fps
            self._mss = mss.mss()

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

# elif OS_NAME == "Darwin":
else:
    import cv2
    try: 
        import Quartz
        from ScreenCaptureKit import SCShareableContent, SCContentFilter, SCStreamConfiguration, SCScreenshotManager
        import Quartz.CoreGraphics as CG
        from PIL import Image
        from Foundation import NSRunLoop, NSDate
    
        class DesktopGrabber:
            def __init__(self, monitor_index=1, output_resolution=1080, show_monitor_info=True, fps=60):
                self.scaled_height = output_resolution
                self.monitor_index = monitor_index
                self.fps = fps
                self.show_monitor_info = show_monitor_info
                self.done = False
                self.latest_frame = None

                # Get monitor info from mss (for scaling + coords)
                self._mss = mss.mss()
                self._log_monitors()

                if monitor_index >= len(self._mss.monitors):
                    print(f"Monitor {monitor_index} not found, using primary monitor")
                    self.monitor_index = 1

                self._mon = self._mss.monitors[self.monitor_index]
                self.system_width, self.system_height = self.get_screen_resolution(self.monitor_index)
                self.scaled_width = round(self.system_width * self.scaled_height / self.system_height)

                # Get displays from ScreenCaptureKit and match to mss monitor coords
                self._get_displays()
                self._match_display()

                if self.show_monitor_info:
                    print(f"Using monitor {self.monitor_index}: {self.system_width}x{self.system_height}")
                    print(f"Scaled resolution: {self.scaled_width}x{self.scaled_height}")

            def _log_monitors(self):
                if self.show_monitor_info:
                    print("Available monitors (mss style):")
                    for i, mon in enumerate(self._mss.monitors):
                        width, height = self.get_screen_resolution(i)
                        if i == 0:
                            print(f"  {i}: All monitors - {width}x{height}")
                        else:
                            system_scale = int(width / mon["width"])
                            print(f"  {i}: Monitor {i} - {width}x{height} at ({mon['left'] * system_scale}, {mon['top'] * system_scale})")

            def get_screen_resolution(self, index):
                monitor = self._mss.monitors[index]
                shot = self._mss.grab(monitor)
                return shot.size.width, shot.size.height

            def _get_displays(self):
                self.done = False
                def completion_handler(shareable_content, error=None):
                    if error:
                        print("Error getting shareable content:", error)
                        self.done = True
                        return
                    self.shareable_content = shareable_content
                    self.displays = shareable_content.displays()
                    self.done = True

                SCShareableContent.getShareableContentWithCompletionHandler_(completion_handler)
                while not self.done:
                    NSRunLoop.currentRunLoop().runUntilDate_(NSDate.dateWithTimeIntervalSinceNow_(1/self.fps))

            def _match_display(self):
                """Match mss monitor coordinates to ScreenCaptureKit display"""
                mss_mon = self._mss.monitors[self.monitor_index]
                mss_left = mss_mon["left"]
                mss_top = mss_mon["top"]

                matched = None
                for d in self.displays:
                    # Get native pixel position from Quartz
                    did = d.displayID()
                    bounds = Quartz.CGDisplayBounds(did)
                    left = int(bounds.origin.x)
                    top = int(bounds.origin.y)
                    if left == mss_left and top == mss_top:
                        matched = d
                        break

                if matched is None:
                    print("Warning: Could not match mss monitor to SC display, defaulting to first display.")
                    matched = self.displays[0]

                self.sc_display = matched

            def grab(self):
                self.done = False
                self.latest_frame = None

                content_filter = SCContentFilter.alloc().initWithDisplay_excludingWindows_(self.sc_display, [])
                configuration = SCStreamConfiguration.alloc().init()

                def handle_cg_image(cg_image, error=None):
                    if error:
                        print("Error capturing image:", error)
                        self.done = True
                        return
                    width = CG.CGImageGetWidth(cg_image)
                    height = CG.CGImageGetHeight(cg_image)
                    data_provider = CG.CGImageGetDataProvider(cg_image)
                    data = CG.CGDataProviderCopyData(data_provider)
                    img = Image.frombytes("RGBA", (width, height), data, "raw", "BGRA")
                    img = img.resize((self.scaled_width, self.scaled_height), Image.BICUBIC)
                    self.latest_frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGBA2RGB)
                    self.done = True

                SCScreenshotManager.captureImageWithFilter_configuration_completionHandler_(
                    content_filter, configuration, handle_cg_image
                )

                while not self.done:
                    NSRunLoop.currentRunLoop().runUntilDate_(NSDate.dateWithTimeIntervalSinceNow_(0.001))

                return self.latest_frame, (self.scaled_height, self.scaled_width)
    except:
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
                self.system_scale = int(self.system_width / self._mon["width"])
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
                        system_scale = int(width / mon["width"])
                        print(f"  {i}: Monitor {i} - {width}x{height} at ({mon['left'] * system_scale}, {mon['top'] * system_scale})")

            def get_screen_resolution(self, index):
                monitor = self._mss.monitors[index]
                screen = self._mss.grab(monitor)
                return screen.size.width, screen.size.height

            def grab(self):
                shot = self._mss.grab(self._mon)
                img = cv2.cvtColor(np.array(shot), cv2.COLOR_BGRA2RGB)
                return img, (self.scaled_height, self.scaled_width)


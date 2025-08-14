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
        import objc
        from ScreenCaptureKit import SCShareableContent, SCContentFilter, SCStreamConfiguration, SCStream, SCStreamOutput
        import Quartz.CoreGraphics as CG
        from Foundation import NSObject, NSRunLoop, NSDate

        class FrameReceiver(NSObject, protocols=[SCStreamOutput]):
            def initWithParent_(self, parent):
                self = objc.super(FrameReceiver, self).init()
                if self:
                    self.parent = parent
                return self

            # Called when new frame arrives
            def stream_didOutputSampleBuffer_ofType_(self, stream, sample_buffer, type_):
                if sample_buffer is None:
                    return
                if type_ != 0:  # 0 = screen content, 1 = audio
                    return

                # Get pixel buffer
                image_buffer = sample_buffer.imageBuffer()
                if not image_buffer:
                    return

                width = CG.CVPixelBufferGetWidth(image_buffer)
                height = CG.CVPixelBufferGetHeight(image_buffer)

                # Lock buffer
                CG.CVPixelBufferLockBaseAddress(image_buffer, 0)
                base_address = CG.CVPixelBufferGetBaseAddress(image_buffer)
                bytes_per_row = CG.CVPixelBufferGetBytesPerRow(image_buffer)

                # Create numpy array from raw BGRA data
                arr = np.frombuffer(base_address, dtype=np.uint8)
                arr = arr.reshape((height, bytes_per_row // 4, 4))[:, :width, :]
                frame = cv2.cvtColor(arr, cv2.COLOR_BGRA2RGB)

                CG.CVPixelBufferUnlockBaseAddress(image_buffer, 0)
                self.parent.latest_frame = frame

        class DesktopGrabber:
            def __init__(self, monitor_index=1, output_resolution=1080, show_monitor_info=True, fps=60):
                self.scaled_height = output_resolution
                self.monitor_index = monitor_index - 1
                self.fps = fps
                self.show_monitor_info = show_monitor_info
                self.latest_frame = None

                # Get displays
                self.shareable_content = None
                self.displays = None
                self._get_displays()

                if self.monitor_index >= len(self.displays):
                    print(f"Monitor {monitor_index} not found, using primary monitor")
                    self.monitor_index = 0

                self._select_monitor(self.monitor_index)

                # Create persistent stream
                self._setup_stream()

            def _get_displays(self):
                done = False
                def completion_handler(shareable_content, error=None):
                    nonlocal done
                    if error:
                        print("Error getting shareable content:", error)
                        done = True
                        return
                    self.shareable_content = shareable_content
                    self.displays = shareable_content.displays()
                    done = True

                SCShareableContent.getShareableContentWithCompletionHandler_(completion_handler)
                while not done:
                    NSRunLoop.currentRunLoop().runUntilDate_(NSDate.dateWithTimeIntervalSinceNow_(0.05))

                if self.show_monitor_info:
                    print("Available monitors:")
                    for i, d in enumerate(self.displays):
                        size = d.width(), d.height()
                        print(f"  {i}: {size[0]}x{size[1]}")

            def _select_monitor(self, index):
                display = self.displays[index]
                self.system_width = display.width()
                self.system_height = display.height()
                self.scaled_width = round(self.system_width * self.scaled_height / self.system_height)
                if self.show_monitor_info:
                    print(f"Using monitor {index}: {self.system_width}x{self.system_height}")
                    print(f"Scaled resolution: {self.scaled_width}x{self.scaled_height}")

            def _setup_stream(self):
                display = self.displays[self.monitor_index]
                content_filter = SCContentFilter.alloc().initWithDisplay_excludingWindows_(display, [])
                config = SCStreamConfiguration.alloc().init()
                config.setWidth_(self.system_width)
                config.setHeight_(self.system_height)
                config.setMinimumFrameInterval_(1.0 / self.fps)
                config.setShowsCursor_(True)

                self.receiver = FrameReceiver.alloc().initWithParent_(self)
                self.stream = SCStream.alloc().initWithFilter_configuration_delegate_(content_filter, config, None)
                self.stream.addStreamOutput_type_sampleHandlerQueue_error_(self.receiver, 0, None, None)
                self.stream.startCaptureWithCompletionHandler_(lambda e: print("Stream started" if not e else e))

            def grab(self):
                # Simply return the latest frame; it's updated continuously
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


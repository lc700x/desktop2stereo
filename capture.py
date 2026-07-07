import numpy as np
import mss
from utils import OS_NAME, CAPTURE_TOOL

if OS_NAME == "Windows":
    import win32gui
    from ctypes import windll
    # Enable DPI awareness to improve capture quality on high-resolution displays
    try:
        windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        windll.user32.SetProcessDPIAware()  # Silently ignore failure to set DPI awareness

    if CAPTURE_TOOL == "DXCamera":
        from wincam import DXCamera
        def get_window_client_bounds(hwnd):
            """
            Retrieve the client area of a window in screen coordinates.

            Args:
                hwnd (int): The window handle.

            Returns:
                tuple: (left, top, width, height) in screen pixel coordinates.

            Raises:
                Exception: If the window handle is invalid or the window cannot be found.
            """
            rc = win32gui.GetClientRect(hwnd)
            if rc is None:
                raise Exception(f"Window not found {hwnd}")

            left, top, right, bottom = rc
            w = right - left
            h = bottom - top
            left, top = win32gui.ClientToScreen(hwnd, (left, top))
            return left, top, w, h

        class DesktopGrabber:
            def __init__(self, output_resolution=1080, fps=60, window_title=None, capture_mode="Monitor", monitor_index=1):
                """
                Initialize the desktop frame grabber for either a window or a monitor.

                Args:
                    output_resolution (int): Output image height (used for scaling).
                    fps (int): Frames per second for the capture device.
                    window_title (str): Title of the application window to capture.
                    capture_mode (str): 'Window' to capture an app window, 'Monitor' to capture a screen.
                    monitor_index (int): Index of the monitor to use when capture_mode is 'Monitor'.
                """
                self.scaled_height = output_resolution
                self.fps = fps
                self._mss = mss.mss()  # Multi-screen capture utility
                self.capture_mode = capture_mode
                self.camera = None  # DXCamera object for hardware-accelerated capture
                self.prev_rect = None  # Previously captured window bounds to avoid redundant updates
                self.window_title = window_title
                self._last_frame = None  # Cached last successful frame
                self._last_camera_rect = None  # For sub-pixel move detection

                if self.capture_mode == "Monitor":
                    # Capture a specific monitor directly using MSS
                    mon = self._mss.monitors[monitor_index]
                    self.left, self.top, self.width, self.height = mon['left'], mon['top'], mon['width'], mon['height']
                    self.camera = DXCamera(self.left, self.top, self.width, self.height, fps=self.fps)
                    try:
                        self.camera.__enter__()  # Start the camera if it supports context management
                    except AttributeError:
                        pass
                else:
                    # Capture a specific window by title
                    self.hwnd = win32gui.FindWindow(None, self.window_title)
                    if not self.hwnd:
                        raise RuntimeError(f"Window '{self.window_title}' not found")

            def _monitor_contains(self, mon, rect):
                """
                Check whether a rectangle is completely inside a monitor's bounds.

                Args:
                    mon (dict): Monitor information from MSS (with left, top, width, height).
                    rect (tuple): Rectangle as (left, top, width, height).

                Returns:
                    bool: True if the rectangle is fully contained in the monitor.
                """
                left, top, w, h = rect
                right, bottom = left + w, top + h
                mon_left, mon_top = mon['left'], mon['top']
                mon_right, mon_bottom = mon_left + mon['width'], mon_top + mon['height']
                return left >= mon_left and top >= mon_top and right <= mon_right and bottom <= mon_bottom

            def _monitor_intersection_area(self, mon, rect):
                """
                Compute the area of overlap between a rectangle and a monitor.

                Args:
                    mon (dict): Monitor dictionary.
                    rect (tuple): Rectangle as (left, top, width, height).

                Returns:
                    int: The overlapping area (width * height).
                """
                left, top, w, h = rect
                right, bottom = left + w, top + h
                mon_left, mon_top = mon['left'], mon['top']
                mon_right, mon_bottom = mon_left + mon['width'], mon_top + mon['height']
                inter_w = max(0, min(mon_right, right) - max(mon_left, left))
                inter_h = max(0, min(mon_bottom, bottom) - max(mon_top, top))
                return inter_w * inter_h

            def _choose_monitor_and_rect(self, rect):
                """
                Select the most appropriate monitor to display the window and adjust its bounds
                to fit within that monitor.

                Args:
                    rect (tuple): The window bounds as (left, top, width, height).

                Returns:
                    tuple: (monitor_info, adjusted_rect) where adjusted_rect is clamped to the monitor.
                """
                left, top, w, h = rect
                right, bottom = left + w, top + h

                # Check if the window is fully inside any secondary monitor (index >= 1)
                for mon in self._mss.monitors[1:]:
                    if self._monitor_contains(mon, rect):
                        return mon, rect

                # If not fully inside any, find the monitor with the largest overlapping area
                best_mon, best_area = None, -1
                for mon in self._mss.monitors[1:]:
                    area = self._monitor_intersection_area(mon, rect)
                    if area > best_area:
                        best_area = area
                        best_mon = mon

                # Fallback to the first non-primary monitor if no significant overlap
                if best_mon is None or best_area <= 0:
                    best_mon = self._mss.monitors[1]

                # Clamp the rectangle to the chosen monitor's screen space
                mon_left, mon_top = best_mon['left'], best_mon['top']
                mon_right, mon_bottom = mon_left + best_mon['width'], mon_top + best_mon['height']
                new_left = max(left, mon_left)
                new_top = max(top, mon_top)
                new_right = min(right, mon_right)
                new_bottom = min(bottom, mon_bottom)
                new_w = max(0, new_right - new_left)
                new_h = max(0, new_bottom - new_top)

                # If clamping results in an empty area, default to the full monitor
                if new_w == 0 or new_h == 0:
                    return best_mon, (mon_left, mon_top, best_mon['width'], best_mon['height'])

                return best_mon, (new_left, new_top, new_w, new_h)

            def _ensure_camera_matches_window(self):
                """
                Ensure the DXCamera is correctly configured to the current window position and size.
                Reinitializes the camera if the window has moved, resized, or is newly detected.
                Ignores sub-5px moves to avoid camera recreation storms.
                """
                try:
                    bounds = get_window_client_bounds(self.hwnd)
                    if bounds is None:
                        # Window is not valid (minimized, closed, etc.)
                        if self.camera:
                            try:
                                self.camera.__exit__(None, None, None)
                            except AttributeError:
                                pass
                            self.camera = None
                        self.prev_rect = None
                        return

                    if bounds == self.prev_rect:
                        # No change in window bounds, no need to update camera
                        return

                    self.prev_rect = bounds  # Cache the latest valid bounds

                    # Determine the best monitor to contain this window and adjust bounds
                    _, rect = self._choose_monitor_and_rect(bounds)

                    # Skip recreation if bounds changed less than 5px (avoids camera recreation storms)
                    if self.camera and self._last_camera_rect is not None:
                        dx = abs(rect[0] - self._last_camera_rect[0])
                        dy = abs(rect[1] - self._last_camera_rect[1])
                        dw = abs(rect[2] - self._last_camera_rect[2])
                        dh = abs(rect[3] - self._last_camera_rect[3])
                        if max(dx, dy, dw, dh) <= 5:
                            return

                    # Recreate the camera if needed
                    if self.camera:
                        try:
                            self.camera.__exit__(None, None, None)
                        except AttributeError:
                            pass
                    self.camera = DXCamera(*rect, fps=self.fps)
                    try:
                        self.camera.__enter__()
                    except AttributeError:
                        pass
                    self._last_camera_rect = rect

                except Exception:
                    # On any error, reset the camera to avoid crashes
                    if self.camera:
                        try:
                            self.camera.__exit__(None, None, None)
                        except AttributeError:
                            pass
                        self.camera = None
                    self.prev_rect = None

            def grab(self):
                """
                Capture a single frame from the current source (window or monitor).

                Returns:
                    tuple: (image_array, scaled_height) where image_array is the captured frame.
                """
                if self.capture_mode != "Monitor":
                    self._ensure_camera_matches_window()  # Ensure camera is up to date for window capture
                try:
                    img_array, _ = self.camera.get_bgr_frame()
                    self._last_frame = img_array
                    return img_array.copy(), self.scaled_height
                except Exception as e:
                    print(f"[Capture] DXCamera grab failed: {e}")
                    if self._last_frame is not None:
                        return self._last_frame.copy(), self.scaled_height
                    raise

            def stop(self):
                """
                Clean up and release the capture device.
                """
                if self.camera:
                    try:
                        self.camera.__exit__(None, None, None)
                    except AttributeError:
                        pass
                    self.camera = None

    elif CAPTURE_TOOL == "DesktopDuplication":
        class DesktopGrabber:
            """
            Desktop grabber using Windows DXGI Desktop Duplication API.
            Provides direct access to the desktop frame buffer with hardware acceleration.
            """
            def __init__(self, output_resolution=1080, fps=60, window_title=None,
                         capture_mode="Monitor", monitor_index=0):
                """
                Initialize the DXGI desktop grabber for either a window or monitor.

                Args:
                    output_resolution (int): Output image height (used for scaling).
                    fps (int): Frames per second (informational).
                    window_title (str): Title of the window to capture (for Window mode).
                    capture_mode (str): 'Window' to capture an app window, 'Monitor' to capture a screen.
                    monitor_index (int): Monitor index to capture (0 = primary monitor).
                """
                try:
                    from windows_capture import DxgiDuplicationSession
                except ImportError:
                    raise RuntimeError(
                        "windows_capture module not found. "
                        "Install it with: pip install windows-capture"
                    )

                self.scaled_height = output_resolution
                self.fps = fps
                self.monitor_index = monitor_index
                self.window_title = window_title
                self.capture_mode = capture_mode
                self.session = None
                self.last_frame = None
                self._frame_count = 0
                self.prev_rect = None
                self._mss = mss.mss()
                self.hwnd = None

                # Initialize based on capture mode
                if self.capture_mode == "Monitor":
                    # For monitor capture, use the specified monitor index
                    try:
                        self.session = DxgiDuplicationSession(monitor_index=monitor_index)
                    except RuntimeError as e:
                        raise RuntimeError(f"Failed to create Desktop Duplication session for monitor {monitor_index}: {e}")
                else:
                    # For window capture, we need to get the window bounds
                    self.hwnd = win32gui.FindWindow(None, self.window_title)
                    if not self.hwnd:
                        raise RuntimeError(f"Window '{self.window_title}' not found")
                    # Initialize with primary monitor for window capture
                    try:
                        self.session = DxgiDuplicationSession(monitor_index=1)
                    except RuntimeError as e:
                        raise RuntimeError(f"Failed to create Desktop Duplication session: {e}")

            def _monitor_contains(self, mon, rect):
                """
                Check whether a rectangle is completely inside a monitor's bounds.
                """
                left, top, w, h = rect
                right, bottom = left + w, top + h
                mon_left, mon_top = mon['left'], mon['top']
                mon_right, mon_bottom = mon_left + mon['width'], mon_top + mon['height']
                return left >= mon_left and top >= mon_top and right <= mon_right and bottom <= mon_bottom

            def _monitor_intersection_area(self, mon, rect):
                """
                Compute the area of overlap between a rectangle and a monitor.
                """
                left, top, w, h = rect
                right, bottom = left + w, top + h
                mon_left, mon_top = mon['left'], mon['top']
                mon_right, mon_bottom = mon_left + mon['width'], mon_top + mon['height']
                inter_w = max(0, min(mon_right, right) - max(mon_left, left))
                inter_h = max(0, min(mon_bottom, bottom) - max(mon_top, top))
                return inter_w * inter_h

            def _choose_monitor_and_rect(self, rect):
                """
                Select the best monitor for the window and clamp the rectangle to fit.
                """
                left, top, w, h = rect
                right, bottom = left + w, top + h

                # Check if the window is fully inside any secondary monitor (index >= 1)
                for mon in self._mss.monitors[1:]:
                    if self._monitor_contains(mon, rect):
                        return mon, rect

                # Find monitor with largest overlapping area
                best_mon, best_area = None, -1
                for mon in self._mss.monitors[1:]:
                    area = self._monitor_intersection_area(mon, rect)
                    if area > best_area:
                        best_area = area
                        best_mon = mon

                # Fallback to first non-primary monitor if no overlap
                if best_mon is None or best_area <= 0:
                    best_mon = self._mss.monitors[1]

                # Clamp rectangle to monitor bounds
                mon_left, mon_top = best_mon['left'], best_mon['top']
                mon_right, mon_bottom = mon_left + best_mon['width'], mon_top + best_mon['height']
                new_left = max(left, mon_left)
                new_top = max(top, mon_top)
                new_right = min(right, mon_right)
                new_bottom = min(bottom, mon_bottom)
                new_w = max(0, new_right - new_left)
                new_h = max(0, new_bottom - new_top)

                # Default to full monitor if clamping results in empty area
                if new_w == 0 or new_h == 0:
                    return best_mon, (mon_left, mon_top, best_mon['width'], best_mon['height'])

                return best_mon, (new_left, new_top, new_w, new_h)

            def _get_monitor_for_window(self):
                """
                Get the monitor index that the window is primarily on.
                """
                try:
                    bounds = get_window_client_bounds(self.hwnd)
                    if bounds is None:
                        return 0

                    _, rect = self._choose_monitor_and_rect(bounds)
                    left, top, w, h = rect

                    # Find which monitor this rectangle is on
                    for idx, mon in enumerate(self._mss.monitors[1:], 1):
                        if self._monitor_contains(mon, rect):
                            return idx - 1  # Convert back to DXGI monitor index

                    return 0  # Default to primary monitor
                except Exception:
                    return 0

            def _ensure_session_matches_window(self):
                """
                Ensure the DXGI session is on the correct monitor for window capture.
                """
                if self.capture_mode == "Monitor":
                    return

                try:
                    bounds = get_window_client_bounds(self.hwnd)
                    if bounds is None:
                        self.prev_rect = None
                        return

                    if bounds == self.prev_rect:
                        return

                    self.prev_rect = bounds

                    # Determine which monitor the window is on
                    monitor_idx = self._get_monitor_for_window()

                    # Switch to that monitor if needed
                    if self.session and monitor_idx != self.monitor_index:
                        try:
                            self.session.switch_monitor(monitor_idx)
                            self.monitor_index = monitor_idx
                        except Exception as e:
                            # If switch fails, try to recreate the session
                            try:
                                self.session.recreate()
                            except Exception:
                                pass
                except Exception:
                    pass

            def grab(self):
                """
                Capture a single frame from the DXGI desktop duplication session.

                Returns:
                    tuple: (image_array, scaled_height) where image_array is the captured frame.
                """
                if self.session is None:
                    raise RuntimeError("DXGI session is not initialized")

                # For window capture, ensure we're capturing from the right monitor
                if self.capture_mode != "Monitor":
                    self._ensure_session_matches_window()

                try:
                    # Attempt to acquire a frame with timeout based on FPS
                    timeout_ms = max(16, int(1000 / self.fps))
                    frame = self.session.acquire_frame(timeout_ms=timeout_ms)

                    if frame is not None:
                        # to_numpy(copy=True) already returns an owned array; cache it
                        # directly instead of copying a second time per frame.
                        image_rgb = frame.to_numpy(copy=True)  # Returns BGR uint8 format
                        self.last_frame = image_rgb
                        self._frame_count += 1
                        return image_rgb, self.scaled_height
                    else:
                        # No new frame available, return last cached frame if available
                        if self.last_frame is not None:
                            return self.last_frame, self.scaled_height
                        else:
                            # Return a black frame as fallback
                            return np.zeros((self.scaled_height, int(self.scaled_height * 16/9), 3), dtype=np.uint8), self.scaled_height

                except RuntimeError as e:
                    # Handle DXGI access loss by recreating the session
                    error_str = str(e).lower()
                    if any(x in error_str for x in ["access loss", "access denied", "device lost"]):
                        try:
                            self.session.recreate()
                            # Retry acquiring frame after recreation
                            frame = self.session.acquire_frame(timeout_ms=33)
                            if frame is not None:
                                image_rgb = frame.to_numpy(copy=False)
                                return image_rgb, self.scaled_height
                        except Exception as recreate_error:
                            raise RuntimeError(f"Failed to recreate DXGI session: {recreate_error}")
                    raise

            def stop(self):
                """
                Clean up and release the Desktop Duplication session.
                """
                if self.session is not None:
                    try:
                        # Try to close the session if it has a close method
                        if hasattr(self.session, 'close'):
                            self.session.close()
                    except Exception:
                        pass
                    finally:
                        self.session = None

                self.last_frame = None
                if self._mss:
                    try:
                        self._mss.close()
                    except Exception:
                        pass

elif OS_NAME == "Darwin":
    if CAPTURE_TOOL == "ScreenCaptureKit":
        import threading
        import time
        import ctypes
        import numpy as np
        import cv2

        import objc
        from Foundation import NSObject
        from Quartz import CoreVideo as CV
        from CoreMedia import CMTimeMake, CMSampleBufferGetImageBuffer
        from AppKit import NSScreen

        # Load ScreenCaptureKit framework
        objc.loadBundle('ScreenCaptureKit', globals(),
            bundle_path=objc.pathForFramework('/System/Library/Frameworks/ScreenCaptureKit.framework'))
        import ScreenCaptureKit as SCK

        # Module-level cache for shareable content
        _sck_content_cache = None
        _sck_content_cache_time = 0.0
        _SCK_CACHE_TTL = 2.0
        _SCK_TORCH_DEVICE = None

        def _sck_get_shareable_content(force=False):
            global _sck_content_cache, _sck_content_cache_time
            now = time.time()
            if not force and _sck_content_cache is not None and (now - _sck_content_cache_time) < _SCK_CACHE_TTL:
                return _sck_content_cache

            done = threading.Event()
            result = {}
            def _handler(content, error):
                result['content'] = content
                result['error'] = error
                done.set()

            SCK.SCShareableContent.getShareableContentWithCompletionHandler_(_handler)
            if not done.wait(timeout=10.0):
                raise RuntimeError("Timed out waiting for shareable content")
            if result.get('error'):
                raise RuntimeError(f"Failed to get shareable content: {result['error']}")

            _sck_content_cache = result['content']
            _sck_content_cache_time = now
            return _sck_content_cache

        def _sck_get_torch_device(device=None):
            global _SCK_TORCH_DEVICE
            import torch

            if device is not None:
                return torch.device(device)

            if _SCK_TORCH_DEVICE is None:
                from utils import DEVICE
                _SCK_TORCH_DEVICE = torch.device(DEVICE)
            return _SCK_TORCH_DEVICE

        def _sck_frame_to_tensor(frame, output_format, device=None):
            """
            Convert CPU-owned BGRA frame to channel-first torch tensor.
            Color shuffle happens after upload so ScreenCaptureKit does one CPU->GPU hop.
            """
            import torch

            tensor = torch.from_numpy(frame).to(
                device=_sck_get_torch_device(device),
                non_blocking=True,
            )

            if output_format == "bgra_tensor":
                return tensor.permute(2, 0, 1).contiguous()
            if output_format == "bgr_tensor":
                return tensor[..., :3].permute(2, 0, 1).contiguous()
            if output_format == "rgb_tensor":
                return tensor[..., [2, 1, 0]].permute(2, 0, 1).contiguous()

            raise ValueError(
                "output_format must be 'bgr', 'bgra', 'rgb_tensor', "
                "'bgr_tensor', or 'bgra_tensor'"
            )

        def _sck_blank_tensor(width, height, output_format, device=None):
            import torch

            channels = 4 if output_format == "bgra_tensor" else 3
            return torch.zeros(
                (channels, height, width),
                dtype=torch.uint8,
                device=_sck_get_torch_device(device),
            )

        def _sck_find_window(title):
            content = _sck_get_shareable_content()
            windows = content.windows()
            for w in windows:
                wt = w.title()
                if wt is None:
                    continue
                if wt == title:
                    return w
                owner = w.owningApplication()
                if owner is not None:
                    app_name = owner.applicationName()
                    if app_name == title:
                        return w
            return None

        def get_window_info_mac(window_title):
            win = _sck_find_window(window_title)
            if win is None:
                return None
            frame = win.frame()
            return {
                "window_id": int(win.windowID()),
                "left": int(frame.origin.x),
                "top": int(frame.origin.y),
                "width": int(frame.size.width),
                "height": int(frame.size.height),
            }

        def get_window_client_bounds_mac(window_title):
            info = get_window_info_mac(window_title)
            if info is None:
                return None, None, None, None
            return info["left"], info["top"], info["width"], info["height"]

        class _SCKFrameReceiver(NSObject):
            def init(self):
                self = objc.super(_SCKFrameReceiver, self).init()
                if self is None:
                    return None
                self._lock = threading.Lock()
                self._latest_frame = None
                self._frame_count = 0
                self._condition = threading.Condition(self._lock)
                return self

            def stream_didOutputSampleBuffer_ofType_(self, stream, sampleBuffer, outputType):
                if outputType != 0:
                    return
                try:
                    imageBuffer = CMSampleBufferGetImageBuffer(sampleBuffer)
                    if imageBuffer is None:
                        return

                    w = CV.CVPixelBufferGetWidth(imageBuffer)
                    h = CV.CVPixelBufferGetHeight(imageBuffer)
                    bpr = CV.CVPixelBufferGetBytesPerRow(imageBuffer)
                    size = bpr * h

                    CV.CVPixelBufferLockBaseAddress(imageBuffer, 0)
                    try:
                        varlist = CV.CVPixelBufferGetBaseAddress(imageBuffer)
                        buf = varlist.as_buffer(size)
                        frame = np.frombuffer(buf, dtype=np.uint8).reshape(h, bpr)
                        if bpr != w * 4:
                            frame = np.ascontiguousarray(frame[:, :w*4].reshape(h, w, 4))
                        else:
                            frame = frame.reshape(h, w, 4).copy()
                    finally:
                        CV.CVPixelBufferUnlockBaseAddress(imageBuffer, 0)

                    with self._condition:
                        self._latest_frame = frame
                        self._frame_count += 1
                        self._condition.notify_all()
                except Exception:
                    pass

            def stream_didStopWithError_(self, stream, error):
                if error is not None:
                    print(f"[ScreenCaptureKit] Stream stopped with error: {error}")

            def get_latest_frame(self, timeout=0.1, copy=True):
                with self._condition:
                    if self._latest_frame is None and timeout > 0:
                        self._condition.wait(timeout=timeout)
                    if self._latest_frame is not None:
                        if copy:
                            return self._latest_frame.copy()
                        return self._latest_frame
                    return None

            @property
            def frame_count(self):
                return self._frame_count

        class DesktopGrabber:
            def __init__(self, output_resolution=1080, fps=60, window_title=None,
                        capture_mode="Monitor", monitor_index=1, with_cursor=True):
                self.scaled_height = output_resolution
                self.fps = fps
                self.with_cursor = with_cursor
                self.window_title = window_title
                self.capture_mode = capture_mode
                self._stream = None
                self._receiver = None
                self._last_frame = None
                self._last_tensor = None
                self._last_tensor_format = None
                self._last_tensor_device = None
                self._display = None
                self._window = None
                self.left = 0
                self.top = 0
                self.width = 0
                self.height = 0

                content = _sck_get_shareable_content()
                displays = content.displays()

                if not displays or len(displays) == 0:
                    raise RuntimeError(
                        "No displays available via ScreenCaptureKit. "
                        "Grant Screen Recording permission to Terminal/Python in "
                        "System Settings > Privacy & Security > Screen Recording, "
                        "then try again."
                    )

                if self.capture_mode == "Monitor":
                    idx = max(0, min(monitor_index - 1, len(displays) - 1))
                    self._display = displays[idx]
                    df = self._display.frame()
                    self.left = int(df.origin.x)
                    self.top = int(df.origin.y)
                    self.width = self._display.width()
                    self.height = self._display.height()
                else:
                    self._window = _sck_find_window(self.window_title)
                    if self._window is None:
                        raise RuntimeError(f"Window '{self.window_title}' not found via ScreenCaptureKit")

                    wf = self._window.frame()
                    for d in displays:
                        df = d.frame()
                        if (df.origin.x <= wf.origin.x < df.origin.x + df.size.width and
                            df.origin.y <= wf.origin.y < df.origin.y + df.size.height):
                            self._display = d
                            break
                    if self._display is None:
                        self._display = displays[0]

                    self.left = int(wf.origin.x)
                    self.top = int(wf.origin.y)
                    self.width = int(wf.size.width)
                    self.height = int(wf.size.height)

                self._start_stream()

            def _start_stream(self):
                if self.capture_mode == "Monitor":
                    filt = SCK.SCContentFilter.alloc().initWithDisplay_excludingWindows_(
                        self._display, [])
                else:
                    filt = SCK.SCContentFilter.alloc().initWithDisplay_includingWindows_(
                        self._display, [self._window])

                config = SCK.SCStreamConfiguration.alloc().init()
                config.setWidth_(self.width)
                config.setHeight_(self.height)
                config.setShowsCursor_(self.with_cursor)
                config.setPixelFormat_(CV.kCVPixelFormatType_32BGRA)
                config.setMinimumFrameInterval_(CMTimeMake(1, max(1, self.fps)))

                self._receiver = _SCKFrameReceiver.alloc().init()
                self._stream = SCK.SCStream.alloc().initWithFilter_configuration_delegate_(
                    filt, config, self._receiver)

                success, error = self._stream.addStreamOutput_type_sampleHandlerQueue_error_(
                    self._receiver, 0, None, None)
                if not success:
                    raise RuntimeError(f"Failed to add stream output: {error}")

                done = threading.Event()
                start_result = {}
                def _on_start(error):
                    start_result['error'] = error
                    done.set()

                self._stream.startCaptureWithCompletionHandler_(_on_start)
                if not done.wait(timeout=10.0):
                    raise RuntimeError("Timed out waiting for capture to start")
                if start_result.get('error'):
                    raise RuntimeError(f"Failed to start capture: {start_result['error']}")

                self._receiver.get_latest_frame(timeout=2.0)

            def _update_window_filter(self):
                if self.capture_mode != "Window":
                    return

                win = _sck_find_window(self.window_title)
                if win is None:
                    return

                wf = win.frame()
                nl, nt = int(wf.origin.x), int(wf.origin.y)
                nw, nh = int(wf.size.width), int(wf.size.height)

                if nl == self.left and nt == self.top and nw == self.width and nh == self.height:
                    return

                self.left, self.top = nl, nt
                self.width, self.height = nw, nh
                self._window = win
                self._last_frame = None
                self._last_tensor = None

                fid = SCK.SCContentFilter.alloc().initWithDisplay_includingWindows_(
                    self._display, [win])
                done = threading.Event()
                self._stream.updateContentFilter_completionHandler_(fid, lambda e: done.set())
                done.wait(timeout=3.0)

            def grab(self, output_format="bgr", tensor_device=None):
                self._update_window_filter()

                tensor_output = output_format.endswith("_tensor")
                frame = self._receiver.get_latest_frame(
                    timeout=1.0 / max(1, self.fps),
                    copy=not tensor_output,
                )

                if frame is None:
                    if tensor_output:
                        if (self._last_tensor is not None and
                            self._last_tensor_format == output_format and
                            self._last_tensor_device == str(_sck_get_torch_device(tensor_device))):
                            return self._last_tensor, self.scaled_height
                        h = self.scaled_height
                        w = int(h * self.width / max(1, self.height))
                        return _sck_blank_tensor(w, h, output_format, tensor_device), self.scaled_height
                    if self._last_frame is not None:
                        return self._last_frame.copy(), self.scaled_height
                    h = self.scaled_height
                    w = int(h * self.width / max(1, self.height))
                    channels = 4 if output_format == "bgra" else 3
                    return np.zeros((h, w, channels), dtype=np.uint8), self.scaled_height

                self._last_frame = frame

                if tensor_output:
                    tensor = _sck_frame_to_tensor(frame, output_format, tensor_device)
                    self._last_tensor = tensor
                    self._last_tensor_format = output_format
                    self._last_tensor_device = str(tensor.device)
                    return tensor, self.scaled_height
                elif output_format == "bgra":
                    return frame, self.scaled_height
                elif output_format == "bgr":
                    return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR), self.scaled_height
                else:
                    raise ValueError(
                        "output_format must be 'bgr', 'bgra', 'rgb_tensor', "
                        "'bgr_tensor', or 'bgra_tensor'"
                    )

            def stop(self):
                if self._stream is not None:
                    done = threading.Event()
                    self._stream.stopCaptureWithCompletionHandler_(lambda e: done.set())
                    done.wait(timeout=5.0)
                self._stream = None
                self._receiver = None
                self._last_frame = None
                self._last_tensor = None
    else:
        import io
        import time
        from collections import OrderedDict
        import cv2
        import numpy as np
        from PIL import Image

        import objc
        import Quartz as QZ
        import Quartz.CoreGraphics as CG
        from AppKit import NSCursor, NSBitmapImageRep, NSPNGFileType, NSScreen
        from Quartz import CGCursorIsVisible, NSEvent

        # Keep the base cursor cache small and stable
        _cursor_cache = {
            "bgra": None,
            "hotspot": None,
            "alpha_f32": None,
            "premultiplied_bgr_f32": None,
            "last_cursor": None,
        }

        # Small bounded cache for resized cursor variants
        _CURSOR_RESIZE_CACHE_MAX = 4
        _CG_TORCH_DEVICE = None

        def _cg_get_torch_device(device=None):
            global _CG_TORCH_DEVICE
            import torch

            if device is not None:
                return torch.device(device)

            if _CG_TORCH_DEVICE is None:
                from utils import DEVICE
                _CG_TORCH_DEVICE = torch.device(DEVICE)
            return _CG_TORCH_DEVICE

        def _cg_frame_to_tensor(frame, output_format, device=None):
            """
            Convert CoreGraphics BGRA frame to channel-first torch tensor.
            Upload once, then do channel shuffle on accelerator.
            """
            import torch

            tensor = torch.from_numpy(frame).to(
                device=_cg_get_torch_device(device),
                non_blocking=True,
            )

            if output_format == "bgra_tensor":
                return tensor.permute(2, 0, 1).contiguous()
            if output_format == "bgr_tensor":
                return tensor[..., :3].permute(2, 0, 1).contiguous()
            if output_format == "rgb_tensor":
                return tensor[..., [2, 1, 0]].permute(2, 0, 1).contiguous()

            raise ValueError(
                "output_format must be 'bgr', 'bgra', 'rgb_tensor', "
                "'bgr_tensor', or 'bgra_tensor'"
            )

        def _cg_get_scale_for_rect(left, top, width, height):
            cx = left + width * 0.5
            cy = top + height * 0.5

            for screen in NSScreen.screens():
                frame = screen.frame()
                if (frame.origin.x <= cx <= frame.origin.x + frame.size.width and
                    frame.origin.y <= cy <= frame.origin.y + frame.size.height):
                    return screen.backingScaleFactor()
            return 1.0

        def _find_window(matcher):
            windows = QZ.CGWindowListCopyWindowInfo(
                QZ.kCGWindowListOptionAll, QZ.kCGNullWindowID
            ) or []
            return [w for w in windows if matcher(w)]

        def get_window_info_mac(window_title):
            """
            Return a dict with window_id + bounds for a unique window match.
            """
            matches = _find_window(
                lambda w: w.get("kCGWindowName") == window_title
                or w.get("kCGWindowOwnerName") == window_title
            )

            if len(matches) == 0:
                return None
            if len(matches) > 1:
                raise ValueError(f"Found multiple windows with name: {window_title}")

            win = matches[0]
            bounds = win.get("kCGWindowBounds", {}) or {}

            return {
                "window_id": int(win["kCGWindowNumber"]),
                "left": int(bounds.get("X", 0)),
                "top": int(bounds.get("Y", 0)),
                "width": int(bounds.get("Width", 0)),
                "height": int(bounds.get("Height", 0)),
            }

        def get_window_client_bounds_mac(window_title):
            """
            Return (x, y, w, h) for a window by title; None if not found.
            """
            info = get_window_info_mac(window_title)
            if info is None:
                return None, None, None, None
            return info["left"], info["top"], info["width"], info["height"]

        def _cg_capture_region_as_bgra(
            region: tuple[int, int, int, int] | None = None,
            window_id: int | None = None,
        ) -> np.ndarray:
            """
            Capture a region or window using CoreGraphics and return BGRA uint8.
            """
            if window_id is not None and region is not None:
                raise ValueError("Only one of region or window_id must be specified")

            if window_id is not None:
                image = CG.CGWindowListCreateImage(
                    CG.CGRectNull,
                    CG.kCGWindowListOptionIncludingWindow,
                    window_id,
                    CG.kCGWindowImageBoundsIgnoreFraming | CG.kCGWindowImageNominalResolution,
                )
            else:
                cg_region = CG.CGRectInfinite if region is None else CG.CGRectMake(*region)
                image = CG.CGWindowListCreateImage(
                    cg_region,
                    CG.kCGWindowListOptionOnScreenOnly,
                    CG.kCGNullWindowID,
                    CG.kCGWindowImageDefault,
                )

            if image is None:
                raise RuntimeError("Could not capture image with CoreGraphics")

            width = CG.CGImageGetWidth(image)
            height = CG.CGImageGetHeight(image)
            bpr = CG.CGImageGetBytesPerRow(image)

            provider = CG.CGImageGetDataProvider(image)
            data = CG.CGDataProviderCopyData(provider)
            raw = np.frombuffer(data, dtype=np.uint8)

            # On macOS this is typically BGRA-compatible for OpenCV use.
            frame = np.lib.stride_tricks.as_strided(
                raw,
                shape=(height, width, 4),
                strides=(bpr, 4, 1),
                writeable=True,
            ).copy()

            return frame

        def get_cursor_image_and_hotspot():
            """
            Return cursor image in BGRA, hotspot, alpha and premultiplied BGR.
            Cache is refreshed only when the actual system cursor changes.
            """
            try:
                with objc.autorelease_pool():
                    cursor = NSCursor.currentSystemCursor()
                    if cursor is None:
                        return None, None, None, None

                    if cursor == _cursor_cache["last_cursor"] and _cursor_cache["bgra"] is not None:
                        return (
                            _cursor_cache["bgra"],
                            _cursor_cache["hotspot"],
                            _cursor_cache["alpha_f32"],
                            _cursor_cache["premultiplied_bgr_f32"],
                        )

                    _cursor_cache["last_cursor"] = cursor

                    ns_image = cursor.image()
                    if ns_image is None:
                        return None, None, None, None

                    hot_pt = cursor.hotSpot()
                    hotspot = (int(hot_pt.x), int(hot_pt.y))

                    tiff_data = ns_image.TIFFRepresentation()
                    if tiff_data is None:
                        return None, None, None, None

                    bitmap = NSBitmapImageRep.imageRepWithData_(tiff_data)
                    if bitmap is None:
                        return None, None, None, None

                    png_data = bitmap.representationUsingType_properties_(NSPNGFileType, None)
                    if png_data is None:
                        return None, None, None, None

                    rgba = np.asarray(Image.open(io.BytesIO(png_data)).convert("RGBA"), dtype=np.uint8)
                    bgra = rgba[:, :, [2, 1, 0, 3]].copy()

                    alpha = bgra[:, :, 3].astype(np.float32) / 255.0
                    premultiplied_bgr = bgra[:, :, :3].astype(np.float32) * alpha[:, :, None]

                    _cursor_cache["bgra"] = bgra
                    _cursor_cache["hotspot"] = hotspot
                    _cursor_cache["alpha_f32"] = alpha
                    _cursor_cache["premultiplied_bgr_f32"] = premultiplied_bgr

                    return bgra, hotspot, alpha, premultiplied_bgr

            except Exception:
                return None, None, None, None

        def get_cursor_position():
            """Return current cursor (x, y) in macOS display coordinates (origin bottom-left)."""
            ev = CG.CGEventCreate(None)
            loc = CG.CGEventGetLocation(ev)
            return loc.x, loc.y

        def is_cursor_visible():
            """Check if cursor is visible (cached check for performance)."""
            return CGCursorIsVisible()

        def overlay_cursor_on_frame(frame_bgr, cursor_bgra, hotspot, cursor_pos,
                                    alpha_f32=None, premultiplied_bgr_f32=None):
            """
            Overlay cursor onto a frame that is either BGR or BGRA.
            Only the first 3 channels are blended; alpha is preserved if present.
            """
            if cursor_bgra is None:
                x_cv, y_cv = cursor_pos
                cv2.circle(frame_bgr, (int(round(x_cv)), int(round(y_cv))), 8, (0, 0, 255), -1)
                return frame_bgr

            h_frame, w_frame = frame_bgr.shape[:2]
            x_cv, y_cv = cursor_pos
            cur_h, cur_w = cursor_bgra.shape[:2]
            hot_x, hot_y = hotspot

            top_left_x = int(round(x_cv - hot_x))
            top_left_y = int(round(y_cv - hot_y))

            x0 = max(top_left_x, 0)
            y0 = max(top_left_y, 0)
            x1 = min(top_left_x + cur_w, w_frame)
            y1 = min(top_left_y + cur_h, h_frame)

            if x0 >= x1 or y0 >= y1:
                return frame_bgr

            src_x0 = x0 - top_left_x
            src_y0 = y0 - top_left_y
            src_x1 = src_x0 + (x1 - x0)
            src_y1 = src_y0 + (y1 - y0)

            dst_region = frame_bgr[y0:y1, x0:x1]
            dst_rgb = dst_region[:, :, :3] if dst_region.ndim == 3 and dst_region.shape[2] == 4 else dst_region

            if premultiplied_bgr_f32 is not None and alpha_f32 is not None:
                src_premult = premultiplied_bgr_f32[src_y0:src_y1, src_x0:src_x1]
                alpha_roi = alpha_f32[src_y0:src_y1, src_x0:src_x1]
                src_region = None
            else:
                src_region = cursor_bgra[src_y0:src_y1, src_x0:src_x1]
                alpha_roi = src_region[:, :, 3].astype(np.float32) / 255.0
                src_premult = src_region[:, :, :3].astype(np.float32) * alpha_roi[:, :, None]

            a_min = float(alpha_roi.min())
            a_max = float(alpha_roi.max())

            if a_max <= 1e-6:
                return frame_bgr

            if a_min >= 0.999:
                if src_region is not None:
                    dst_rgb[:, :, :] = src_region[:, :, :3]
                else:
                    np.copyto(dst_rgb, np.clip(src_premult + 0.5, 0, 255).astype(np.uint8))
                return frame_bgr

            alpha_3ch = alpha_roi[:, :, None]
            dst_f32 = dst_rgb.astype(np.float32, copy=False)
            blended = src_premult + dst_f32 * (1.0 - alpha_3ch)

            np.clip(blended, 0, 255, out=blended)
            res_uint8 = blended.astype(np.uint8, copy=False)

            if a_max >= 0.999:
                mask_opaque = (alpha_roi >= 0.999)
                if mask_opaque.any():
                    if src_region is not None:
                        res_uint8[mask_opaque] = src_region[:, :, :3][mask_opaque]
                    else:
                        res_uint8[mask_opaque] = np.clip(src_premult + 0.5, 0, 255).astype(np.uint8)[mask_opaque]

            np.copyto(dst_rgb, res_uint8)
            return frame_bgr

        class DesktopGrabber:
            def __init__(self, output_resolution=1080, fps=60, window_title=None,
                        capture_mode="Monitor", monitor_index=1, with_cursor=True):
                self.scaled_height = output_resolution
                self.fps = fps
                self.with_cursor = with_cursor
                self.window_title = window_title
                self.capture_mode = capture_mode
                self.prev_rect = None
                self.window_id = None
                self._last_tensor = None
                self._last_tensor_format = None
                self._last_tensor_device = None

                # bounded per-instance resize cache
                self._cursor_cache = OrderedDict()

                if self.capture_mode == "Monitor":
                    screens = list(NSScreen.screens())
                    if not screens:
                        raise RuntimeError("No screens found")

                    mon_index = max(1, min(monitor_index, len(screens)))
                    screen = screens[mon_index - 1]
                    frame = screen.frame()

                    self.left = int(frame.origin.x)
                    self.top = int(frame.origin.y)
                    self.width = int(frame.size.width)
                    self.height = int(frame.size.height)
                    self._system_scale = screen.backingScaleFactor()
                    self._system_scale_time = 0.0
                else:
                    info = get_window_info_mac(self.window_title)
                    if info is None:
                        raise RuntimeError(f"Window '{self.window_title}' not found")

                    self.window_id = info["window_id"]
                    self.left = info["left"]
                    self.top = info["top"]
                    self.width = info["width"]
                    self.height = info["height"]
                    self._system_scale = _cg_get_scale_for_rect(
                        self.left, self.top, self.width, self.height
                    )
                    self._system_scale_time = 0.0

            def _ensure_rect(self):
                if self.capture_mode != "Monitor":
                    info = get_window_info_mac(self.window_title)
                    if info is None:
                        return
                    current = (info["left"], info["top"], info["width"], info["height"])
                    if current == self.prev_rect:
                        return

                    self.prev_rect = current
                    self.window_id = info["window_id"]
                    self.left, self.top, self.width, self.height = current
                    self._system_scale = _cg_get_scale_for_rect(
                        self.left, self.top, self.width, self.height
                    )
                    self._system_scale_time = 0.0
                    self._last_tensor = None
                    self._cursor_cache.clear()

            def get_scale(self):
                now = time.time()
                if now - self._system_scale_time < 1.0:
                    return self._system_scale

                mouse_location = NSEvent.mouseLocation()
                screens = NSScreen.screens()

                for screen in screens:
                    frame = screen.frame()
                    if frame.origin.x <= mouse_location.x <= frame.origin.x + frame.size.width and \
                    frame.origin.y <= mouse_location.y <= frame.origin.y + frame.size.height:
                        self._system_scale = screen.backingScaleFactor()
                        self._system_scale_time = now
                        return self._system_scale

                self._system_scale_time = now
                return self._system_scale

            def _get_resized_cursor(self, cursor_bgra, hotspot, system_scale):
                """
                Keep your original cursor sizing logic unchanged.
                Bounded cache prevents memory growth.
                """
                scale_factor = max(1, 16 // max(1, int(system_scale)))

                cache_key = (id(cursor_bgra), cursor_bgra.shape, scale_factor)
                cached = self._cursor_cache.get(cache_key)
                if cached is not None:
                    self._cursor_cache.move_to_end(cache_key)
                    return cached

                h, w = cursor_bgra.shape[:2]
                if h > scale_factor and w > scale_factor:
                    new_w, new_h = max(1, w // scale_factor), max(1, h // scale_factor)
                    resized_bgra = cv2.resize(cursor_bgra, (new_w, new_h), interpolation=cv2.INTER_AREA)
                else:
                    resized_bgra = cursor_bgra

                alpha_f32 = resized_bgra[:, :, 3].astype(np.float32) / 255.0
                premultiplied = resized_bgra[:, :, :3].astype(np.float32) * alpha_f32[:, :, None]

                result = (resized_bgra, hotspot, alpha_f32, premultiplied)
                self._cursor_cache[cache_key] = result
                self._cursor_cache.move_to_end(cache_key)

                while len(self._cursor_cache) > _CURSOR_RESIZE_CACHE_MAX:
                    self._cursor_cache.popitem(last=False)

                return result

            def grab(self, output_format="bgr", tensor_device=None):
                """
                output_format:
                    - "bgra": no full-frame color conversion
                    - "bgr" : convert once at the end
                    - "*_tensor": return CHW torch tensor on configured device
                """
                self._ensure_rect()
                tensor_output = output_format.endswith("_tensor")

                if self.capture_mode == "Monitor":
                    frame = _cg_capture_region_as_bgra(
                        region=(self.left, self.top, self.width, self.height)
                    )
                else:
                    frame = _cg_capture_region_as_bgra(window_id=self.window_id)

                if self.with_cursor and CGCursorIsVisible():
                    x, y = get_cursor_position()
                    system_scale = self.get_scale()

                    if 0 <= x - self.left <= self.width and 0 <= y - self.top <= self.height:
                        cursor_x = (x - self.left) * system_scale
                        cursor_y = (y - self.top) * system_scale

                        cursor_bgra, hotspot, alpha_f32, premultiplied = get_cursor_image_and_hotspot()
                        if cursor_bgra is not None:
                            cursor_bgra, hotspot, alpha_f32, premultiplied = self._get_resized_cursor(
                                cursor_bgra, hotspot, system_scale
                            )

                            overlay_cursor_on_frame(
                                frame,
                                cursor_bgra,
                                hotspot,
                                (cursor_x, cursor_y),
                                alpha_f32=alpha_f32,
                                premultiplied_bgr_f32=premultiplied,
                            )

                if output_format == "bgra":
                    return frame, self.scaled_height
                elif output_format == "bgr":
                    return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR), self.scaled_height
                elif tensor_output:
                    tensor = _cg_frame_to_tensor(frame, output_format, tensor_device)
                    self._last_tensor = tensor
                    self._last_tensor_format = output_format
                    self._last_tensor_device = str(tensor.device)
                    return tensor, self.scaled_height
                else:
                    raise ValueError(
                        "output_format must be 'bgr', 'bgra', 'rgb_tensor', "
                        "'bgr_tensor', or 'bgra_tensor'"
                    )

            def stop(self):
                if hasattr(self, "_cursor_cache"):
                    self._cursor_cache.clear()

elif OS_NAME.startswith("Linux"):
    # linux must on Xorg mode
    from Xlib import display, X
    import cv2

    def get_window_coords(title):
        d = display.Display()
        root = d.screen().root

        # Get all windows
        window_ids = root.get_full_property(
            d.intern_atom('_NET_CLIENT_LIST'),
            X.AnyPropertyType
        ).value

        # Pre-intern atoms for common properties
        net_wm_name = d.intern_atom('_NET_WM_NAME')
        utf8_string = d.intern_atom('UTF8_STRING')

        for window_id in window_ids:
            window = d.create_resource_object('window', window_id)

            # Try multiple ways to get the window name
            name = None
            try:
                # Try _NET_WM_NAME first (UTF-8)
                name_prop = window.get_full_property(net_wm_name, utf8_string)
                if name_prop:
                    name = name_prop.value.decode('utf-8')
                else:
                    # Fall back to WM_NAME
                    name = window.get_wm_name()
                    if isinstance(name, bytes):
                        name = name.decode('utf-8', errors='replace')
            except:
                continue
            if name and title in name:
                # Get absolute coordinates (accounting for window decorations)
                geom = window.get_geometry()
                pos = geom.root.translate_coords(window_id, 0, 0)
                return (pos.x, pos.y, geom.width, geom.height)
        return None

    class DesktopGrabber:
        def __init__(self, output_resolution=1080, fps=60, window_title=None, capture_mode="Monitor", monitor_index=1):
            self.scaled_height = output_resolution
            self.fps = fps
            self.window_title = window_title
            self.capture_mode = capture_mode
            self._mss = mss.mss(with_cursor=True)
            self.prev_rect = None
            self.monitor_index = monitor_index

            if self.capture_mode == "Monitor":
                # Initialize with the selected monitor
                if self.monitor_index >= len(self._mss.monitors):
                    self.monitor_index = 1
                mon = self._mss.monitors[self.monitor_index]
                self.left, self.top, self.width, self.height = mon['left'], mon['top'], mon['width'], mon['height']
            else:
                # Initialize with window coordinates
                bounds = get_window_coords(self.window_title)
                if bounds is None:
                    raise RuntimeError(f"Window '{self.window_title}' not found")
                self.left, self.top, self.width, self.height = bounds

            self.scaled_width = round(self.width * self.scaled_height / self.height)

        def _monitor_contains(self, mon, rect):
            """
            Check whether a rectangle is completely inside a monitor's bounds.
            """
            left, top, w, h = rect
            right, bottom = left + w, top + h
            mon_left, mon_top = mon['left'], mon['top']
            mon_right, mon_bottom = mon_left + mon['width'], mon_top + mon['height']
            return left >= mon_left and top >= mon_top and right <= mon_right and bottom <= mon_bottom

        def _monitor_intersection_area(self, mon, rect):
            """
            Compute the area of overlap between a rectangle and a monitor.
            """
            left, top, w, h = rect
            right, bottom = left + w, top + h
            mon_left, mon_top = mon['left'], mon['top']
            mon_right, mon_bottom = mon_left + mon['width'], mon_top + mon['height']
            inter_w = max(0, min(mon_right, right) - max(mon_left, left))
            inter_h = max(0, min(mon_bottom, bottom) - max(mon_top, top))
            return inter_w * inter_h

        def _choose_monitor_and_rect(self, rect):
            """
            Select the best monitor for the window and clamp the rectangle to fit.
            """
            left, top, w, h = rect
            right, bottom = left + w, top + h

            # Check if the window is fully inside any secondary monitor (index >= 1)
            for mon in self._mss.monitors[1:]:
                if self._monitor_contains(mon, rect):
                    return mon, rect

            # Find monitor with largest overlapping area
            best_mon, best_area = None, -1
            for mon in self._mss.monitors[1:]:
                area = self._monitor_intersection_area(mon, rect)
                if area > best_area:
                    best_area = area
                    best_mon = mon

            # Fallback to first non-primary monitor if no overlap
            if best_mon is None or best_area <= 0:
                best_mon = self._mss.monitors[1]

            # Clamp rectangle to monitor bounds
            mon_left, mon_top = best_mon['left'], best_mon['top']
            mon_right, mon_bottom = mon_left + best_mon['width'], mon_top + best_mon['height']
            new_left = max(left, mon_left)
            new_top = max(top, mon_top)
            new_right = min(right, mon_right)
            new_bottom = min(bottom, mon_bottom)
            new_w = max(0, new_right - new_left)
            new_h = max(0, new_bottom - new_top)

            # Default to full monitor if clamping results in empty area
            if new_w == 0 or new_h == 0:
                return best_mon, (mon_left, mon_top, best_mon['width'], best_mon['height'])

            return best_mon, (new_left, new_top, new_w, new_h)

        def _ensure_rect(self):
            if self.capture_mode != "Monitor":
                bounds = get_window_coords(self.window_title)
                if bounds is None:
                    return False
                if bounds == self.prev_rect:
                    return True
                self.prev_rect = bounds

                # Apply monitor clamping logic
                _, clamped_rect = self._choose_monitor_and_rect(bounds)
                self.left, self.top, self.width, self.height = clamped_rect
                self.scaled_width = round(self.width * self.scaled_height / self.height)
            return True

        def grab(self):
            if not self._ensure_rect():
                return None, self.scaled_height

            monitor = {"left": self.left, "top": self.top, "width": self.width, "height": self.height}
            shot = self._mss.grab(monitor)
            arr = np.asarray(shot)
            return arr, self.scaled_height

        def stop(self):
            """Stop the capture and clean up resources."""
            if hasattr(self, '_mss'):
                try:
                    self._mss.close()
                except:
                    pass

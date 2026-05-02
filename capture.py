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
                img_array, _ = self.camera.get_bgr_frame()
                return img_array, self.scaled_height

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
                        # Use to_bgr() method for proper color format handling
                        image_rgb = frame.to_numpy(copy=True)  # Returns BGR uint8 format
                        
                        # Cache the frame
                        self.last_frame = image_rgb.copy()
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
    import io
    from collections import OrderedDict
    import cv2
    import numpy as np
    from PIL import Image

    import objc
    import Quartz as QZ
    import Quartz.CoreGraphics as CG
    from AppKit import NSCursor, NSBitmapImageRep, NSPNGFileType, NSScreen
    from Quartz import CGCursorIsVisible, NSEvent

    _cursor_cache = {
        "bgra": None,
        "hotspot": None,
        "alpha_f32": None,
        "premultiplied_bgr_f32": None,
        "last_cursor": None,
    }

    _CURSOR_RESIZE_CACHE_MAX = 2

    def _find_window(matcher):
        windows = QZ.CGWindowListCopyWindowInfo(
            QZ.kCGWindowListOptionAll, QZ.kCGNullWindowID
        ) or []
        return [w for w in windows if matcher(w)]

    def get_window_info_mac(window_title):
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
        info = get_window_info_mac(window_title)
        if info is None:
            return None, None, None, None
        return info["left"], info["top"], info["width"], info["height"]

    def _cg_capture_region_as_bgra(region=None, window_id=None):
        image = None
        provider = None
        data = None
        try:
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
                raise RuntimeError("Could not capture image")

            width = CG.CGImageGetWidth(image)
            height = CG.CGImageGetHeight(image)

            provider = CG.CGImageGetDataProvider(image)
            data = CG.CGDataProviderCopyData(provider)

            raw = np.frombuffer(data, dtype=np.uint8).copy()
            frame = raw[: height * width * 4].reshape((height, width, 4)).copy()

            return frame

        finally:
            try:
                del data
            except Exception:
                pass
            try:
                del provider
            except Exception:
                pass
            try:
                del image
            except Exception:
                pass

    def get_cursor_image_and_hotspot():
        try:
            with objc.autorelease_pool():
                cursor = NSCursor.currentSystemCursor()
                if cursor is None:
                    return None, None, None, None

                if cursor != _cursor_cache["last_cursor"]:
                    _cursor_cache["bgra"] = None
                    _cursor_cache["alpha_f32"] = None
                    _cursor_cache["premultiplied_bgr_f32"] = None

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
                bitmap = NSBitmapImageRep.imageRepWithData_(tiff_data)
                png_data = bitmap.representationUsingType_properties_(NSPNGFileType, None)

                rgba = np.asarray(Image.open(io.BytesIO(png_data)).convert("RGBA"), dtype=np.uint8)
                bgra = rgba[:, :, [2, 1, 0, 3]].copy()

                alpha = bgra[:, :, 3].astype(np.float32) / 255.0
                premultiplied_bgr = bgra[:, :, :3].astype(np.float32) * alpha[:, :, None]

                del ns_image
                del tiff_data
                del bitmap
                del png_data
                del rgba

                _cursor_cache["bgra"] = bgra
                _cursor_cache["hotspot"] = hotspot
                _cursor_cache["alpha_f32"] = alpha
                _cursor_cache["premultiplied_bgr_f32"] = premultiplied_bgr

                return bgra, hotspot, alpha, premultiplied_bgr

        except Exception:
            return None, None, None, None

    def get_cursor_position():
        ev = CG.CGEventCreate(None)
        loc = CG.CGEventGetLocation(ev)
        return loc.x, loc.y

    def is_cursor_visible():
        return CGCursorIsVisible()

    def overlay_cursor_on_frame(frame, cursor_bgra, hotspot, cursor_pos,
                                alpha_f32=None, premultiplied_bgr_f32=None):

        if cursor_bgra is None:
            x_cv, y_cv = cursor_pos
            cv2.circle(frame, (int(round(x_cv)), int(round(y_cv))), 8, (0, 0, 255), -1)
            return frame

        h_frame, w_frame = frame.shape[:2]
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
            return frame

        src_x0 = x0 - top_left_x
        src_y0 = y0 - top_left_y
        src_x1 = src_x0 + (x1 - x0)
        src_y1 = src_y0 + (y1 - y0)

        dst = frame[y0:y1, x0:x1]

        if premultiplied_bgr_f32 is not None and alpha_f32 is not None:
            src_premult = premultiplied_bgr_f32[src_y0:src_y1, src_x0:src_x1]
            alpha = alpha_f32[src_y0:src_y1, src_x0:src_x1]
        else:
            src = cursor_bgra[src_y0:src_y1, src_x0:src_x1]
            alpha = src[:, :, 3].astype(np.float32) / 255.0
            src_premult = src[:, :, :3].astype(np.float32) * alpha[:, :, None]

        dst_f = dst[:, :, :3].astype(np.float32)
        out = src_premult + dst_f * (1.0 - alpha[:, :, None])

        np.clip(out, 0, 255, out=out)
        dst[:, :, :3] = out.astype(np.uint8)

        return frame

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

            self._cursor_cache = OrderedDict()

            if self.capture_mode == "Monitor":
                screens = list(NSScreen.screens())
                screen = screens[max(0, min(monitor_index - 1, len(screens) - 1))]
                frame = screen.frame()

                self.left = int(frame.origin.x)
                self.top = int(frame.origin.y)
                self.width = int(frame.size.width)
                self.height = int(frame.size.height)
            else:
                info = get_window_info_mac(self.window_title)
                if info is None:
                    raise RuntimeError(f"Window '{self.window_title}' not found")

                self.window_id = info["window_id"]
                self.left = info["left"]
                self.top = info["top"]
                self.width = info["width"]
                self.height = info["height"]

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

        def get_scale(self):
            mouse_location = NSEvent.mouseLocation()
            screens = NSScreen.screens()

            for screen in screens:
                frame = screen.frame()
                if frame.origin.x <= mouse_location.x <= frame.origin.x + frame.size.width and \
                   frame.origin.y <= mouse_location.y <= frame.origin.y + frame.size.height:
                    return screen.backingScaleFactor()

            return 1.0

        def _get_resized_cursor(self, cursor_bgra, hotspot, system_scale):
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

        def grab(self, output_format="bgr"):
            self._ensure_rect()

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
            else:
                raise ValueError("output_format must be 'bgr' or 'bgra'")

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
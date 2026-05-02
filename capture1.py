import numpy as np
import mss
from utils import OS_NAME, CAPTURE_TOOL

if OS_NAME == "Windows":
    from ctypes import windll
    # Enable DPI awareness to improve capture quality on high-resolution displays
    try:
        windll.user32.SetProcessDPIAware()
    except Exception:
        pass  # Silently ignore failure to set DPI awareness
    import win32gui
    
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
    import cv2
    import numpy as np
    import pyautogui
    import time
    import objc
    import Quartz as QZ
    import Quartz.CoreGraphics as CG
    from AppKit import NSScreen, NSCursor
    from Quartz import CGCursorIsVisible


    def is_cursor_visible():
        return CGCursorIsVisible()


    def _find_window(matcher):
        windows = QZ.CGWindowListCopyWindowInfo(
            QZ.kCGWindowListOptionAll,
            QZ.kCGNullWindowID
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
                    CG.kCGWindowImageBoundsIgnoreFraming
                    | CG.kCGWindowImageNominalResolution,
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
            del data
            del provider
            del image


    class DesktopGrabber:
        def __init__(
            self,
            output_resolution=1080,
            fps=60,
            window_title=None,
            capture_mode="Monitor",
            monitor_index=1,
            mouse=True,
            mouse_color=(0, 255, 0),
            mouse_size=6,
            mouse_thickness=2,
        ):
            self.scaled_height = output_resolution
            self.fps = fps
            self.window_title = window_title
            self.capture_mode = capture_mode

            self.mouse = mouse
            self.mouse_color = mouse_color
            self.mouse_size = mouse_size
            self.mouse_thickness = mouse_thickness

            self.prev_rect = None
            self.window_id = None

            # Cursor idle tracking
            self.last_mouse_pos = pyautogui.position()
            self.last_move_time = time.time()
            self.cursor_hidden = False
            self.idle_threshold = 5  # seconds

            # Cursor assets
            self.cursor_folder = "cursor/"
            self.cursor_assets_raw = {}

            arrow = cv2.imread(f"{self.cursor_folder}arrow.png", cv2.IMREAD_UNCHANGED)
            if arrow is not None:
                self.cursor_assets_raw["arrow"] = arrow

            self.cursor_assets = {}
            self._cursor_cache_scale = None

            # Screen setup
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

        def _update_cursor_visibility(self):
            current_pos = pyautogui.position()
            now = time.time()

            if current_pos != self.last_mouse_pos:
                self.last_mouse_pos = current_pos
                self.last_move_time = now
                self.cursor_hidden = False
            elif (now - self.last_move_time) > self.idle_threshold:
                self.cursor_hidden = True

        def get_cursor_state(self):
            # Explicit autorelease pool to avoid leaks in tight loops
            with objc.autorelease_pool():
                cursor = NSCursor.currentSystemCursor()
                if cursor and cursor.isEqual_(NSCursor.arrowCursor()):
                    return "arrow"
                return "arrow"

        def _prepare_cursor_cache(self, scale_factor):
            if self._cursor_cache_scale == scale_factor:
                return

            self.cursor_assets = {}

            for k, img in self.cursor_assets_raw.items():
                h, w = img.shape[:2]

                new_w = max(1, int(w * scale_factor))
                new_h = max(1, int(h * scale_factor))

                self.cursor_assets[k] = cv2.resize(
                    img,
                    (new_w, new_h),
                    interpolation=cv2.INTER_AREA
                )

            self._cursor_cache_scale = scale_factor

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

        def _draw_fallback_cursor(self, frame, cx, cy):
            cv2.circle(
                frame,
                (cx, cy),
                self.mouse_size,
                self.mouse_color,
                self.mouse_thickness
            )
            return frame

        def _add_cursor(self, frame):
            if not self.mouse:
                return frame

            self._update_cursor_visibility()

            if self.cursor_hidden:
                return frame

            mouse_x, mouse_y = pyautogui.position()

            scale_x = frame.shape[1] / self.width
            scale_y = frame.shape[0] / self.height

            cx = int((mouse_x - self.left) * scale_x)
            cy = int((mouse_y - self.top) * scale_y)

            scale_factor = (scale_x + scale_y) / 2
            self._prepare_cursor_cache(scale_factor)

            state = self.get_cursor_state()
            cursor = self.cursor_assets.get(state)

            # Fallback if no image
            if cursor is None:
                return self._draw_fallback_cursor(frame, cx, cy)

            h, w = cursor.shape[:2]

            x1 = cx - w // 2
            y1 = cy - h // 2
            x2 = x1 + w
            y2 = y1 + h

            if x2 <= 0 or y2 <= 0 or x1 >= frame.shape[1] or y1 >= frame.shape[0]:
                return frame

            fx1, fy1 = max(0, x1), max(0, y1)
            fx2, fy2 = min(frame.shape[1], x2), min(frame.shape[0], y2)

            cx1, cy1 = fx1 - x1, fy1 - y1
            cx2, cy2 = cx1 + (fx2 - fx1), cy1 + (fy2 - fy1)

            if cursor.shape[2] == 4:
                cursor_rgb = cursor[:, :, :3]
                alpha = cursor[:, :, 3] / 255.0
            else:
                cursor_rgb = cursor
                alpha = np.ones((h, w), dtype=np.float32)

            overlay = cursor_rgb[cy1:cy2, cx1:cx2]
            mask = alpha[cy1:cy2, cx1:cx2][:, :, None]

            frame_region = frame[fy1:fy2, fx1:fx2]

            frame[fy1:fy2, fx1:fx2] = (
                mask * overlay + (1 - mask) * frame_region
            ).astype(np.uint8)

            return frame

        def grab(self, output_format="bgr"):
            self._ensure_rect()

            if self.capture_mode == "Monitor":
                frame = _cg_capture_region_as_bgra(
                    region=(self.left, self.top, self.width, self.height)
                )
            else:
                frame = _cg_capture_region_as_bgra(window_id=self.window_id)

            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            frame = self._add_cursor(frame)

            if output_format == "bgra":
                return cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA), self.scaled_height
            elif output_format == "bgr":
                return frame, self.scaled_height
            else:
                raise ValueError("output_format must be 'bgr' or 'bgra'")

        def stop(self):
            pass

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
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
                        image_rgb = frame.to_numpy(copy=False)  # Returns BGR uint8 format
                        
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
    import mss.darwin
    mss.darwin.IMAGE_OPTIONS = 0 # disable scaling
    import io, cv2
    from PIL import Image
    import Quartz.CoreGraphics as CG
    from AppKit import NSCursor, NSBitmapImageRep, NSPNGFileType, NSScreen
    from Quartz import CGCursorIsVisible, NSEvent

    def get_window_client_bounds_mac(window_title):
        """Return (x, y, w, h) for a window by title; None if not found."""
        import Quartz
        options = Quartz.kCGWindowListOptionOnScreenOnly | Quartz.kCGWindowListExcludeDesktopElements
        windows = Quartz.CGWindowListCopyWindowInfo(options, Quartz.kCGNullWindowID)
        for win in windows:
            title = win.get("kCGWindowName", "")
            owner = win.get("kCGWindowOwnerName", "")
            if title == window_title or owner == window_title:
                bounds = win.get("kCGWindowBounds", None)
                if bounds:
                    return int(bounds.get("X",0)), int(bounds.get("Y",0)), int(bounds.get("Width",0)), int(bounds.get("Height",0))
        return None, None, None, None
    
    # Cursor cache to avoid redundant processing
    _cursor_cache = {
        'bgra': None,
        'hotspot': None,
        'alpha_f32': None,
        'premultiplied_bgr_f32': None,
        'last_cursor': None  # Store cursor object reference to detect changes
    }
    
    # Cursor loader with caching & precomputation
    def get_cursor_image_and_hotspot():
        """
        Retrieve current system cursor image (BGRA uint8 numpy) and hotspot, also
        return precomputed helpers for fast per-frame overlay:
        - cursor_bgra: original BGRA uint8 (H,W,4)
        - hotspot: (x,y) in pixels
        - alpha_f32: HxW float32 alpha in [0..1]
        - premultiplied_bgr_f32: HxW x 3 float32 (BGR * alpha)
        Returns (cursor_bgra, hotspot, alpha_f32, premultiplied_bgr_f32) or (None, None, None, None)
        """
        try:
            cursor = NSCursor.currentSystemCursor()
            if cursor is None:
                return None, None, None, None

            # Return cached data if cursor hasn't changed (major performance boost)
            if (cursor == _cursor_cache['last_cursor'] and 
                _cursor_cache['bgra'] is not None):
                return (_cursor_cache['bgra'], _cursor_cache['hotspot'], 
                       _cursor_cache['alpha_f32'], _cursor_cache['premultiplied_bgr_f32'])

            # Update cache reference
            _cursor_cache['last_cursor'] = cursor

            ns_image = cursor.image()
            if ns_image is None:
                return None, None, None, None

            # Get hotspot in pixels
            hot_pt = cursor.hotSpot()
            hotspot = (hot_pt.x, hot_pt.y)

            # Get TIFF representation
            tiff_data = ns_image.TIFFRepresentation()
            if tiff_data is None:
                return None, None, None, None

            bitmap = NSBitmapImageRep.imageRepWithData_(tiff_data)
            if bitmap is None:
                return None, None, None, None

            # Convert to PNG bytes for PIL
            png_data = bitmap.representationUsingType_properties_(NSPNGFileType, None)
            if png_data is None:
                return None, None, None, None

            # Load image once
            img = Image.open(io.BytesIO(png_data)).convert("RGBA")
            rgba = np.array(img)  # H x W x 4

            # Convert to BGRA for OpenCV
            bgra = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA)

            # Compute alpha and premultiplied BGR
            alpha = bgra[:, :, 3].astype(np.float32) / 255.0
            premultiplied_bgr = bgra[:, :, :3].astype(np.float32) * alpha[:, :, None]

            # Store in cache
            _cursor_cache['bgra'] = bgra
            _cursor_cache['hotspot'] = hotspot
            _cursor_cache['alpha_f32'] = alpha
            _cursor_cache['premultiplied_bgr_f32'] = premultiplied_bgr

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

    # Fast overlay using cached premultiplied data with optimized blending
    def overlay_cursor_on_frame(frame_bgr, cursor_bgra, hotspot, cursor_pos,
                                alpha_f32=None, premultiplied_bgr_f32=None):
        """
        Fast overlay of cursor_bgra onto frame_bgr (modified in-place).

        Parameters
        ----------
        frame_bgr : (H,W,3) uint8
            Destination frame (modified in place).
        cursor_bgra : (h,w,4) uint8 or None
            Cursor image in BGRA, or None to draw fallback dot.
        hotspot : (hot_x, hot_y)
            Cursor hotspot in pixels.
        cursor_pos : (x, y)
            Cursor center position in frame coordinates.
        alpha_f32 : (h,w) float32, optional
            Precomputed alpha in [0..1].
        premultiplied_bgr_f32 : (h,w,3) float32, optional
            Precomputed (BGR * alpha).

        Returns
        -------
        frame_bgr : same object, with cursor blended in.
        """
        # Fast fallback if no cursor
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

        # clipped destination rectangle in frame coords
        x0 = max(top_left_x, 0)
        y0 = max(top_left_y, 0)
        x1 = min(top_left_x + cur_w, w_frame)
        y1 = min(top_left_y + cur_h, h_frame)

        if x0 >= x1 or y0 >= y1:
            # fully outside
            return frame_bgr

        src_x0 = x0 - top_left_x
        src_y0 = y0 - top_left_y
        src_x1 = src_x0 + (x1 - x0)
        src_y1 = src_y0 + (y1 - y0)

        dst_region = frame_bgr[y0:y1, x0:x1]                         # uint8 view

        # Use precomputed arrays if available (avoid recompute)
        if premultiplied_bgr_f32 is not None and alpha_f32 is not None:
            src_premult = premultiplied_bgr_f32[src_y0:src_y1, src_x0:src_x1]  # float32
            alpha_roi = alpha_f32[src_y0:src_y1, src_x0:src_x1]               # float32
            src_region = None
        else:
            src_region = cursor_bgra[src_y0:src_y1, src_x0:src_x1]   # uint8 BGRA
            # compute alpha and premultiplied only for roi
            alpha_roi = src_region[:, :, 3].astype(np.float32, copy=False) / 255.0
            src_premult = src_region[:, :, :3].astype(np.float32, copy=False) * alpha_roi[..., None]

        # quick checks for optimization paths
        a_min = float(alpha_roi.min())
        a_max = float(alpha_roi.max())

        if a_max <= 1e-6:
            # fully transparent ROI -> nothing to do
            return frame_bgr

        # Optimized blend using single numpy operation with broadcasting
        # res = src_premult + dst * (1 - alpha)
        alpha_3ch = alpha_roi[..., None]  # Expand to HxWx1 for broadcasting
        
        if a_min >= 0.999:
            # fully opaque ROI -> direct copy (fastest path)
            if src_region is not None:
                dst_region[:, :, :] = src_region[:, :, :3]   # uint8 copy
            else:
                # premultiplied may be float; round to uint8
                np.copyto(dst_region, np.clip(src_premult + 0.5, 0, 255).astype(np.uint8))
            return frame_bgr

        # General blending: single vectorized operation
        dst_f32 = dst_region.astype(np.float32, copy=False)
        blended = src_premult + dst_f32 * (1.0 - alpha_3ch)
        
        # clip + cast back to uint8
        np.clip(blended, 0, 255, out=blended)
        res_uint8 = blended.astype(np.uint8, copy=False)

        # For pixels that are effectively opaque, ensure exact copy (avoid rounding differences)
        if a_max >= 0.999 or np.any(alpha_roi >= 0.999):
            # mask of opaque pixels
            mask_opaque = (alpha_roi >= 0.999)
            if mask_opaque.any():
                if src_region is not None:
                    res_uint8[mask_opaque] = src_region[:, :, :3][mask_opaque]
                else:
                    res_uint8[mask_opaque] = np.clip(src_premult + 0.5, 0, 255).astype(np.uint8)[mask_opaque]

        # write back
        np.copyto(dst_region, res_uint8)
        return frame_bgr

    # Convenience function that checks cursor visibility before overlaying
    def overlay_cursor_if_visible(frame_bgr, cursor_bgra, hotspot, cursor_pos,
                                  alpha_f32=None, premultiplied_bgr_f32=None):
        """
        Overlay cursor only if it's visible. Combines visibility check with overlay.
        
        Parameters: Same as overlay_cursor_on_frame
        
        Returns: frame_bgr with cursor blended in (or unchanged if cursor hidden)
        """
        if not CGCursorIsVisible():
            return frame_bgr
        return overlay_cursor_on_frame(frame_bgr, cursor_bgra, hotspot, cursor_pos, alpha_f32, premultiplied_bgr_f32)
    
    class DesktopGrabber:
        def __init__(self, output_resolution=1080, fps=60, window_title=None, capture_mode="Monitor", monitor_index=1, with_cursor=True):
            self.scaled_height = output_resolution
            self.fps = fps
            self.with_cursor = with_cursor
            self.window_title = window_title
            self.capture_mode = capture_mode
            self._mss = mss.mss()
            self.prev_rect = None

            if self.capture_mode == "Monitor":
                mon_index = monitor_index
                if mon_index >= len(self._mss.monitors):
                    mon_index = 1
                mon = self._mss.monitors[mon_index]
                self.left, self.top, self.width, self.height = mon['left'], mon['top'], mon['width'], mon['height']
            else:
                bounds = get_window_client_bounds_mac(self.window_title)
                if bounds is None:
                    raise RuntimeError(f"Window '{self.window_title}' not found")
                self.left, self.top, self.width, self.height = bounds

        def _ensure_rect(self):
            if self.capture_mode != "Monitor":
                bounds = get_window_client_bounds_mac(self.window_title)
                if bounds is None:
                    return
                if bounds == self.prev_rect:
                    return
                self.prev_rect = bounds
                self.left, self.top, self.width, self.height = bounds
        
        def get_scale(self):
            # Get current mouse location
            mouse_location = NSEvent.mouseLocation()

            # Get list of screens
            screens = NSScreen.screens()

            for screen in screens:
                frame = screen.frame()
                if frame.origin.x <= mouse_location.x <= frame.origin.x + frame.size.width and \
                frame.origin.y <= mouse_location.y <= frame.origin.y + frame.size.height:
                    # Found the screen under the cursor
                    backing_scale = screen.backingScaleFactor()
                    return backing_scale

            raise RuntimeError("No screen found under cursor")
        
        def grab(self):
            self._ensure_rect()
            monitor = {"left": self.left, "top": self.top, "width": self.width, "height": self.height}
            shot = self._mss.grab(monitor)
            arr = np.asarray(shot)
            frame_bgr = cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)
            if self.with_cursor and CGCursorIsVisible():
                x, y = get_cursor_position()
                system_scale = self.get_scale()
                # Convert to local frame coordinates
                if 0 <= x - self.left <= self.width and 0 <= y - self.top <= self.height:
                    
                    cursor_x = (x - self.left) * system_scale
                    cursor_y = (y - self.top) * system_scale
                    # Only overlay if cursor is within this frame
                
                    if 0 <= x - self.left <= self.width and 0 <= y - self.top <= self.height:

                        cursor_x = (x - self.left) * system_scale
                        cursor_y = (y - self.top) * system_scale

                        # Cache cursor image processing to avoid recomputation every frame
                        cursor_bgra, hotspot, alpha_f32, premultiplied = get_cursor_image_and_hotspot()
                        scale_factor = 16 // max(1, int(system_scale))

                        # initialize cache storage if missing
                        if not hasattr(self, "_cursor_cache"):
                            self._cursor_cache = {}

                        # key to uniquely identify cached result
                        cache_key = (id(cursor_bgra), cursor_bgra.shape, scale_factor)

                        if cache_key not in self._cursor_cache:
                            # perform expensive resize & alpha computations once per cache_key
                            h, w = cursor_bgra.shape[:2]
                            if h > scale_factor and w > scale_factor:
                                new_w, new_h = w // scale_factor, h // scale_factor
                                resized_bgra = cv2.resize(cursor_bgra, (new_w, new_h), interpolation=cv2.INTER_AREA)
                            else:
                                resized_bgra = cursor_bgra

                            alpha_f32 = (resized_bgra[:, :, 3].astype(np.float32) / 255.0)
                            premultiplied = resized_bgra[:, :, :3].astype(np.float32) * alpha_f32[:, :, None]

                            self._cursor_cache[cache_key] = (resized_bgra, hotspot, alpha_f32, premultiplied)

                        else:
                            resized_bgra, hotspot, alpha_f32, premultiplied = self._cursor_cache[cache_key]

                        # assign for compatibility with your original code
                        self.cursor_bgra = resized_bgra
                        self.cursor_hotspot = hotspot
                        self.cursor_alpha = alpha_f32
                        self.cursor_premultiplied = premultiplied

                        if self.with_cursor and self.cursor_bgra is not None:
                            frame_bgr = overlay_cursor_on_frame(
                                frame_bgr,
                                self.cursor_bgra,
                                self.cursor_hotspot,
                                (cursor_x, cursor_y),
                                alpha_f32=self.cursor_alpha,
                                premultiplied_bgr_f32=self.cursor_premultiplied
                            )
            else:
                self.cursor_bgra = None
                self.cursor_hotspot = (0, 0)
                self.cursor_alpha = None
                self.cursor_premultiplied = None
            return frame_bgr, self.scaled_height

        def stop(self):
            """Stop the capture and clean up resources."""
            if hasattr(self, '_mss'):
                try:
                    self._mss.close()
                except:
                    pass
            # Clear cursor cache
            if hasattr(self, '_cursor_cache'):
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
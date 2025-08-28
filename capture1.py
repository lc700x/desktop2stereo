import numpy as np
from utils import OS_NAME, CAPTURE_MODE, WINDOW_TITLE, MONITOR_INDEX, WINDOW_TITLE

if OS_NAME == "Windows":
    import time, threading
    import cv2
    from windows_capture import WindowsCapture, Frame, InternalCaptureControl
    class DesktopGrabber:
        def __init__(self, output_resolution=1080, fps=60, window_title=WINDOW_TITLE, 
                    capture_mode=CAPTURE_MODE, crop_top=0, crop_left=0, 
                    crop_right=0, crop_bottom=0):
            self.scaled_height = output_resolution
            self.fps = fps
            self.capture_mode = capture_mode
            self.window_title = window_title
            self.crop_top = crop_top
            self.crop_left = crop_left
            self.crop_right = crop_right
            self.crop_bottom = crop_bottom
            
            self.latest_frame = None
            self._lock = threading.Lock()
            self.frame_count = 0
            self.start_time = time.time()
            self.fps_values = []
            self.stop_event = threading.Event()
            self.frame_event = threading.Event()
            self.sleep_time = 0.5/self.fps

            if self.capture_mode != "Window":
                self.capture = WindowsCapture(monitor_index=MONITOR_INDEX)
            else:
                # Create capture object
                self.capture = WindowsCapture(window_name=self.window_title)

                if not self.window_title:
                    raise ValueError("No window title specified for window capture")


            # Register callback for frames
            @self.capture.event
            def on_frame_arrived(frame: Frame, capture_control: InternalCaptureControl):
                with self._lock:
                    source_frame = frame.frame_buffer
                    # Convert BGRA to RGB
                    self.frame_count += 1
                    self.frame_event.set()
                    self.latest_frame = source_frame
                    time.sleep(self.sleep_time)

                if self.stop_event.is_set():
                    capture_control.stop()

            @self.capture.event
            def on_closed():
                print("Capture session closed")
                self.stop_event.set()

            # Start capture in background thread
            self._thread = threading.Thread(target=self._capture_loop, daemon=True)
            self._thread.start()

            # Wait until first frame arrives
            if not self.frame_event.wait(timeout=10.0):
                raise TimeoutError("Failed to receive first frame within timeout period")

        def _capture_loop(self):
            """Capture event loop"""
            try:
                self.capture.start()
            finally:
                self.frame_event.set()
                time.sleep(0.1)

        def grab(self):
            with self._lock:
                if self.latest_frame is None:
                    return None, self.scaled_height
                img_array = self.latest_frame
                img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGRA2RGB)
            return img_array, self.scaled_height


elif OS_NAME == "Darwin":
    import io, cv2, mss
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
        return None

    # Cursor loader with caching & precomputation
    def get_cursor_image_and_hotspot():
        """
        Retrieve current system cursor image (BGRA uint8 numpy) and hotspot, but also
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

            ns_image = cursor.image()
            if ns_image is None:
                return None, None, None, None

            # hotspot in points -> pixels
            hot_pt = cursor.hotSpot()
            hotspot_x = hot_pt.x
            hotspot_y = hot_pt.y 

            tiff = ns_image.TIFFRepresentation()
            if tiff is None:
                return None, None, None, None
            bitmap = NSBitmapImageRep.imageRepWithData_(tiff)
            if bitmap is None:
                return None, None, None, None

            png_data = bitmap.representationUsingType_properties_(NSPNGFileType, None)
            if png_data is None:
                return None, None, None, None
            # Get current cursor
            
            cursor = NSCursor.currentSystemCursor()
            image = cursor.image()
            bitmap_rep = NSBitmapImageRep.imageRepWithData_(image.TIFFRepresentation())
            png_data = bitmap_rep.representationUsingType_properties_(NSPNGFileType, None)
            buffer = io.BytesIO(png_data)
            img_array = Image.open(buffer)
            rgba = np.array(img_array)
            bgra = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA)

            buf = io.BytesIO(png_data)
            pil_img = Image.open(buf).convert("RGBA")
            rgba = np.array(pil_img)  # H x W x 4 (RGBA)

            # Convert to BGRA uint8 (same memory layout as OpenCV expects)
            bgra = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA)  # uint8

            # Precompute alpha (float32 [0..1]) and premultiplied BGR (float32)
            alpha = (bgra[:, :, 3].astype(np.float32) / 255.0)  # H x W float32
            # convert BGR channels to float32 and premultiply by alpha: (B,G,R) * alpha
            bgr = bgra[:, :, :3].astype(np.float32)
            premultiplied = bgr * alpha[:, :, None]  # H x W x 3 float32

            return bgra, (hotspot_x, hotspot_y), alpha, premultiplied

        except Exception:
            return None, None, None, None

    def get_cursor_position():
        """Return current cursor (x, y) in macOS display coordinates (origin bottom-left)."""
        ev = CG.CGEventCreate(None)
        loc = CG.CGEventGetLocation(ev)
        return loc.x, loc.y

    # Fast overlay using cached premultiplied data
    def overlay_cursor_on_frame(frame_bgr, cursor_bgra, hotspot, cursor_pos, alpha_f32=None, premultiplied_bgr_f32=None):
        """
        Fast overlay of cursor_bgra onto frame_bgr.

        frame_bgr: HxW x 3 uint8, modified in place and returned
        cursor_bgra: full cursor BGRA uint8 array (or None)
        hotspot: (hot_x, hot_y) in pixels
        cursor_pos: (x, y) in frame coordinates
        alpha_f32, premultiplied_bgr_f32: optional precomputed arrays returned by get_cursor_image_and_hotspot
        """
        h_frame, w_frame = frame_bgr.shape[:2]
        x_cv, y_cv = cursor_pos

        if cursor_bgra is None:
            # fallback dot (very cheap)
            cv2.circle(frame_bgr, (x_cv, y_cv), 8, (0, 0, 255), -1)
            return frame_bgr

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
            return frame_bgr

        # source rectangle inside the cursor image
        src_x0 = x0 - top_left_x
        src_y0 = y0 - top_left_y
        src_x1 = src_x0 + (x1 - x0)
        src_y1 = src_y0 + (y1 - y0)

        dst_region = frame_bgr[y0:y1, x0:x1]  # view (H_roi, W_roi, 3), uint8

        # If precomputed premultiplied is available, use it; otherwise make minimal local copies
        if premultiplied_bgr_f32 is not None and alpha_f32 is not None:
            src_premult = premultiplied_bgr_f32[src_y0:src_y1, src_x0:src_x1]  # float32
            alpha_roi = alpha_f32[src_y0:src_y1, src_x0:src_x1]  # float32
        else:
            # Minimal local computation if not precomputed (still faster than repeated heavy numpy casts)
            src_region = cursor_bgra[src_y0:src_y1, src_x0:src_x1]  # uint8 BGRA
            alpha_roi = src_region[:, :, 3].astype(np.float32) / 255.0
            src_premult = src_region[:, :, :3].astype(np.float32) * alpha_roi[:, :, None]

        # Fast path: if alpha is binary (only 0 or 1), we can avoid blending cost -> use copy / mask
        # Check quickly by testing min/max of alpha_roi with small thresholds
        a_min = float(alpha_roi.min())
        a_max = float(alpha_roi.max())
        if a_min >= 0.999:  # fully opaque -> direct copy
            dst_region[:, :, :] = src_premult.astype(np.uint8)
            return frame_bgr
        elif a_max <= 1e-6:  # fully transparent -> nothing to do
            return frame_bgr

        # General alpha blend using OpenCV native operations (float32)
        # dst_scaled = dst * (1 - alpha)
        dst_f32 = dst_region.astype(np.float32)

        # Build one_minus_alpha 3-channel float32 by merging
        one_minus_alpha = (1.0 - alpha_roi).astype(np.float32)
        one_minus_alpha_3 = cv2.merge([one_minus_alpha, one_minus_alpha, one_minus_alpha])  # float32

        # Multiply in C (fast)
        dst_scaled = np.empty_like(dst_f32)
        cv2.multiply(dst_f32, one_minus_alpha_3, dst_scaled)  # dst * (1-alpha)

        # result = src_premult + dst_scaled
        res_f32 = dst_scaled
        cv2.add(src_premult, dst_scaled, res_f32)  # in-place add into res_f32

        # Write result back to frame (convert to uint8)
        # Use round and clip via convertScaleAbs can be used, but we'll cast safely
        np.copyto(dst_region, np.clip(res_f32, 0, 255).astype(np.uint8))
        return frame_bgr

    class DesktopGrabber:
        def __init__(self, output_resolution=1080, fps=60, window_title=WINDOW_TITLE, capture_mode=CAPTURE_MODE, with_cursor=True):
            self.scaled_height = output_resolution
            self.fps = fps
            self.with_cursor = with_cursor
            self.window_title = window_title
            self.capture_mode = capture_mode
            self._mss = mss.mss()
            self.prev_rect = None

            if self.capture_mode == "Monitor":
                mon_index = MONITOR_INDEX
                if mon_index >= len(self._mss.monitors):
                    mon_index = 1
                mon = self._mss.monitors[mon_index]
                self.left, self.top, self.width, self.height = mon['left'], mon['top'], mon['width'], mon['height']
            else:
                bounds = get_window_client_bounds_mac(self.window_title)
                if bounds is None:
                    raise RuntimeError(f"Window '{self.window_title}' not found")
                self.left, self.top, self.width, self.height = bounds

            self.scaled_width = round(self.width * self.scaled_height / self.height)

        def _ensure_rect(self):
            if self.capture_mode != "Monitor":
                bounds = get_window_client_bounds_mac(self.window_title)
                if bounds is None:
                    return
                if bounds == self.prev_rect:
                    return
                self.prev_rect = bounds
                self.left, self.top, self.width, self.height = bounds
                self.scaled_width = round(self.width * self.scaled_height / self.height)

        def grab(self):
            self._ensure_rect()
            monitor = {"left": self.left, "top": self.top, "width": self.width, "height": self.height}
            shot = self._mss.grab(monitor)
            arr = np.asarray(shot)
            frame_bgr = cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)
            # TODO: optionally overlay cursor using existing functions
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            return frame_rgb, (self.scaled_height, self.scaled_width)


elif OS_NAME.startswith("Linux"):
    from Xlib import display, mss
    import cv2

    def get_window_client_bounds_linux(window_title):
        d = display.Display()
        root = d.screen().root
        def search(w):
            try:
                name = w.get_wm_name()
            except:
                return None
            if name == window_title:
                geom = w.get_geometry()
                abs_pos = w.translate_coords(root, 0, 0)
                return abs_pos.x, abs_pos.y, geom.width, geom.height
            try:
                for c in w.query_tree().children:
                    r = search(c)
                    if r:
                        return r
            except:
                pass
            return None
        return search(root)

    class DesktopGrabber:
        def __init__(self, output_resolution=1080, fps=60, window_title=WINDOW_TITLE, capture_mode=CAPTURE_MODE):
            self.scaled_height = output_resolution
            self.fps = fps
            self.window_title = window_title
            self.capture_mode = capture_mode
            self._mss = mss.mss()
            self.prev_rect = None

            if self.capture_mode == "Monitor":
                mon_index = MONITOR_INDEX
                if mon_index >= len(self._mss.monitors):
                    mon_index = 1
                mon = self._mss.monitors[mon_index]
                self.left, self.top, self.width, self.height = mon['left'], mon['top'], mon['width'], mon['height']
            else:
                bounds = get_window_client_bounds_linux(self.window_title)
                if bounds is None:
                    raise RuntimeError(f"Window '{self.window_title}' not found")
                self.left, self.top, self.width, self.height = bounds

            self.scaled_width = round(self.width * self.scaled_height / self.height)

        def _ensure_rect(self):
            if self.capture_mode != "Monitor":
                bounds = get_window_client_bounds_linux(self.window_title)
                if bounds is None:
                    return
                if bounds == self.prev_rect:
                    return
                self.prev_rect = bounds
                self.left, self.top, self.width, self.height = bounds
                self.scaled_width = round(self.width * self.scaled_height / self.height)

        def grab(self):
            self._ensure_rect()
            monitor = {"left": self.left, "top": self.top, "width": self.width, "height": self.height}
            shot = self._mss.grab(monitor)
            arr = np.asarray(shot)
            frame_rgb = cv2.cvtColor(arr, cv2.COLOR_BGRA2RGB)
            return frame_rgb, (self.scaled_height, self.scaled_width)

import numpy as np
import mss
from gui import OS_NAME

# DesktopGrabber: wincam (Windows), MSS + AppKit (Mac), MSS (Linux) 
# Windows 10/11
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

elif OS_NAME == "Darwin":
    import io
    import numpy as np
    import cv2
    from PIL import Image

    # macOS-specific imports assumed present in your environment:
    from AppKit import NSCursor, NSBitmapImageRep, NSPNGFileType
    import Quartz.CoreGraphics as CG
    from Quartz import CGCursorIsVisible
    import mss

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

    # DesktopGrabber with optimized grab path
    class DesktopGrabber:
        def __init__(self, monitor_index=1, output_resolution=1080, show_monitor_info=True, fps=60, with_cursor=True):
            self.scaled_height = output_resolution
            self.monitor_index = monitor_index
            self.fps = fps
            self.with_cursor = with_cursor
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
            # Grab: mss gives a raw BGRA buffer (Pillow style). Convert once to numpy view.
            shot = self._mss.grab(self._mon)
            arr = np.asarray(shot)  # BGRA uint8 view (no unnecessary copies if mss supports it)
            # Convert to BGR (drop alpha) using OpenCV (fast C path)
            frame_bgr = cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)
            # Load cursor and precompute alpha/premultiplied arrays for fast overlay
            if self.with_cursor and CGCursorIsVisible():
                x, y = get_cursor_position()
                # Convert to local monitor frame coordinates (origin top-left)
                cursor_x = (x - self._mon["left"]) * self.system_scale
                cursor_y = (y - self._mon["top"]) * self.system_scale
                # Only overlay if cursor is within this monitor's frame
                if 0 <= cursor_x <= int(self.system_width) and 0 <= cursor_y <= int(self.system_height):
                    cursor_bgra, hotspot, alpha_f32, premultiplied = get_cursor_image_and_hotspot()
                    scale_factor = 16 // self.system_scale
                    if cursor_bgra.shape[0] > scale_factor and cursor_bgra.shape[1] > scale_factor:
                        h, w = cursor_bgra.shape[:2]
                        new_w, new_h = int(w / scale_factor), int(h / scale_factor)
                        # resize BGRA; keep alpha channel scaled properly
                        cursor_bgra = cv2.resize(cursor_bgra, (new_w, new_h), interpolation=cv2.INTER_AREA)

                        # recompute alpha & premultiplied after resize (fast relative to per-frame cost)
                        alpha_f32 = (cursor_bgra[:, :, 3].astype(np.float32) / 255.0)
                        premultiplied = cursor_bgra[:, :, :3].astype(np.float32) * alpha_f32[:, :, None]

                        self.cursor_bgra = cursor_bgra
                        self.cursor_hotspot = hotspot
                        self.cursor_alpha = alpha_f32
                        self.cursor_premultiplied = premultiplied
                else:
                    self.cursor_bgra = None
                    self.cursor_hotspot = (0, 0)
                    self.cursor_alpha = None
                    self.cursor_premultiplied = None

                if self.with_cursor and self.cursor_bgra is not None:
                        # overlay in-place using optimized routine
                        frame_bgr = overlay_cursor_on_frame(
                            frame_bgr,
                            self.cursor_bgra,
                            self.cursor_hotspot,
                            (cursor_x, cursor_y),
                            alpha_f32=self.cursor_alpha,
                            premultiplied_bgr_f32=self.cursor_premultiplied
                        )
            # Return RGB image & scaled dimensions (match your original return signature)
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            return frame_rgb, (self.scaled_height, self.scaled_width)


else: # Old MacOS and Linux
    print("Use MSS as screen grabber. ")
    import cv2
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

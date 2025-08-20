import numpy as np
import mss
from gui import OS_NAME

# DesktopGrabber: wincam (Windows), MSS + AppKit (Mac), MSS (Linux) 
# Windows 10/11
if OS_NAME == "Windows":
    from wincam import DXCamera
    
    # desktop_grabber.py
    # Modified DesktopGrabber that splits a monitor into N regions, captures each region
    # concurrently via per-region DXCamera instances (one thread per region),
    # optionally pins threads to CPU cores, stitches the sub-frames, and returns a resized frame.
    #
    # Assumptions:
    # - DXCamera(left, top, width, height, fps=...) exists and .get_rgb_frame() returns (numpy_array, meta).
    # - numpy (np), cv2, mss are available.
    # - The API mirrors what the main program expects: grab() -> (frame_array, (scaled_height, scaled_width)).
    #
    # Notes about threading and affinity:
    # - Python threads are limited by the GIL for pure-Python CPU-bound code. If DXCamera.get_rgb_frame()
    #   is implemented in C or performs I/O that releases the GIL, threads can run concurrently.
    # - Thread affinity functions are provided for Linux and Windows; macOS affinity is not implemented here.
    # - If you need true CPU parallelism for Python-bound work, consider a multiprocessing variant.

    import math
    import threading
    import queue
    import time
    import ctypes
    import platform
    import cv2

    # Replace this import with the actual location of your DXCamera class.
    # from your_dx_module import DXCamera

    # -------------------- Affinity helpers --------------------

    def _set_thread_affinity_linux(core_id):
        """Attempt to set affinity for the current thread on Linux using pthread_setaffinity_np."""
        try:
            # Build cpu_set_t mask dynamically
            libc = ctypes.CDLL("libc.so.6", use_errno=True)
            libpthread = ctypes.CDLL("libpthread.so.0", use_errno=True)
            pthread_self = libpthread.pthread_self
            pthread_self.restype = ctypes.c_ulong

            CPU_SETSIZE = 1024
            ulong_size = ctypes.sizeof(ctypes.c_ulong)
            mask_len = CPU_SETSIZE // (8 * ulong_size)
            class cpu_set_t(ctypes.Structure):
                _fields_ = [("bits", ctypes.c_ulong * mask_len)]
            mask = cpu_set_t()
            idx = int(core_id) // (8 * ulong_size)
            offset = int(core_id) % (8 * ulong_size)
            mask.bits[idx] = 1 << offset

            pthread = pthread_self()
            res = libc.pthread_setaffinity_np(pthread, ctypes.sizeof(mask), ctypes.byref(mask))
            if res != 0:
                return False
            return True
        except Exception:
            return False

    def _set_thread_affinity_windows(core_id):
        """Attempt to set affinity for current thread on Windows using SetThreadAffinityMask."""
        try:
            kernel32 = ctypes.windll.kernel32
            GetCurrentThread = kernel32.GetCurrentThread
            SetThreadAffinityMask = kernel32.SetThreadAffinityMask
            GetCurrentThread.restype = ctypes.c_void_p
            thread = GetCurrentThread()
            mask = ctypes.c_size_t(1 << int(core_id))
            prev = SetThreadAffinityMask(thread, mask)
            return prev != 0
        except Exception:
            return False

    def set_current_thread_affinity(core_id):
        """Set current thread affinity to a single core (platform-dependent). Returns True on success."""
        system = platform.system()
        if core_id is None:
            return False
        if system == "Linux":
            return _set_thread_affinity_linux(core_id)
        if system == "Windows":
            return _set_thread_affinity_windows(core_id)
        # macOS: no simple portable way here; return False
        return False

    # -------------------- RegionWorker --------------------

    class RegionWorker(threading.Thread):
        """
        Thread that owns a DXCamera for a single region and performs captures on demand.
        The main thread calls trigger_capture() to request a capture. Results are placed into result_queue.
        """
        def __init__(self, region_index, region_rect, result_queue, fps=60, core_id=None, dxcamera_ctor_kwargs=None):
            """
            region_rect: (left, top, width, height)
            result_queue: queue.Queue used to put (region_index, image_array)
            core_id: optional int core id to pin this thread to
            dxcamera_ctor_kwargs: dict of extra kwargs for DXCamera constructor
            """
            super().__init__(daemon=True)
            self.region_index = region_index
            self.left, self.top, self.width, self.height = region_rect
            self.result_queue = result_queue
            self.fps = fps
            self.core_id = core_id
            self.dxcamera_ctor_kwargs = dxcamera_ctor_kwargs or {}
            self._stop_event = threading.Event()

            # Synchronization primitives for capture request
            self._capture_event = threading.Event()
            self._capture_lock = threading.Lock()

            self.camera = None

        def run(self):
            # Optionally set affinity for this thread
            try:
                if self.core_id is not None:
                    set_current_thread_affinity(self.core_id)
            except Exception:
                pass

            # Create DXCamera instance for this region
            try:
                # DXCamera signature: DXCamera(left, top, width, height, fps=...)
                self.camera = DXCamera(self.left, self.top, self.width, self.height, fps=self.fps, **self.dxcamera_ctor_kwargs)
                try:
                    # If DXCamera supports context manager, use it
                    self.camera.__enter__()
                    using_ctx = True
                except Exception:
                    using_ctx = False
            except Exception:
                # If camera creation fails, place a None to indicate failure for this region and allow exit
                self.result_queue.put((self.region_index, None))
                return

            try:
                # Wait for capture requests until stopped
                while not self._stop_event.is_set():
                    # Wait with a small timeout to allow clean shutdown
                    requested = self._capture_event.wait(timeout=0.1)
                    if requested:
                        with self._capture_lock:
                            # Do one capture
                            try:
                                img, meta = self.camera.get_rgb_frame()
                            except Exception:
                                img = None
                            # Put result (may be None on failure)
                            self.result_queue.put((self.region_index, img))
                            # Clear capture request
                            self._capture_event.clear()
            finally:
                # Clean up camera
                try:
                    if using_ctx:
                        self.camera.__exit__(None, None, None)
                except Exception:
                    pass

        def trigger_capture(self):
            """Signal the worker to perform one capture; returns immediately."""
            with self._capture_lock:
                self._capture_event.set()

        def stop(self):
            """Signal the thread to stop and return when it exits."""
            self._stop_event.set()
            # Wake it up if waiting
            self._capture_event.set()

    # -------------------- DesktopGrabber --------------------

    class DesktopGrabber:
        """
        DesktopGrabber that splits the monitor into num_regions regions (near-square grid),
        spawns one RegionWorker per region, triggers concurrent captures, stitches results,
        and returns a resized frame corresponding to output_resolution.
        """

        def __init__(self, monitor_index=1, output_resolution=1080, show_monitor_info=True, fps=60,
                    num_regions=4, core_map=None, dxcamera_ctor_kwargs=None):
            """
            num_regions: number of regions to split the monitor into (1..)
            core_map: list mapping region_index -> core_id (int) or None. length can be <= num_regions.
            dxcamera_ctor_kwargs: additional kwargs passed to DXCamera constructor.
            """
            self.scaled_height = int(output_resolution)
            self.fps = int(fps)
            self.num_regions = max(1, int(num_regions))
            self.core_map = core_map or [None] * self.num_regions
            self.dxcamera_ctor_kwargs = dxcamera_ctor_kwargs or {}

            self._mss = mss.mss()

            if show_monitor_info:
                self._log_monitors()

            if monitor_index >= len(self._mss.monitors):
                print(f"Monitor {monitor_index} not found, using primary monitor")
                monitor_index = 1
            self._mon = self._mss.monitors[monitor_index]

            # Acquire system resolution for the monitor
            self.system_width, self.system_height = self.get_screen_resolution(monitor_index)
            # scaled_width computed to preserve aspect ratio
            self.scaled_width = round(self.system_width * self.scaled_height / self.system_height)

            # system_scale is int(system_width / mon['width']) - used to convert mss coordinates to system coords if needed
            try:
                self.system_scale = int(self.system_width / self._mon["width"])
                if self.system_scale < 1:
                    self.system_scale = 1
            except Exception:
                self.system_scale = 1

            if show_monitor_info:
                print(f"Using monitor {monitor_index}: {self.system_width}x{self.system_height}")
                print(f"Scaled resolution: {self.scaled_width}x{self.scaled_height}")
                print(f"Splitting into {self.num_regions} regions (fps={self.fps})")

            # Compute grid layout (rows x cols) for near-square division of num_regions
            cols = int(math.ceil(math.sqrt(self.num_regions)))
            rows = int(math.ceil(self.num_regions / cols))
            self.rows = rows
            self.cols = cols

            base_left = int(self._mon['left'] * self.system_scale)
            base_top = int(self._mon['top'] * self.system_scale)

            # Compute cell sizes; last row/col absorb remainders
            cell_w = self.system_width // cols
            cell_h = self.system_height // rows

            # Build region rects (left, top, width, height) in system coordinates expected by DXCamera
            self.regions = []
            idx = 0
            for r in range(rows):
                for c in range(cols):
                    if idx >= self.num_regions:
                        break
                    left = base_left + c * cell_w
                    top = base_top + r * cell_h
                    # last column/row gets remaining pixels to ensure full coverage
                    width = cell_w if (c < cols - 1) else (self.system_width - c * cell_w)
                    height = cell_h if (r < rows - 1) else (self.system_height - r * cell_h)
                    self.regions.append((left, top, width, height))
                    idx += 1

            # Prepare result queue and start region workers
            self.result_queue = queue.Queue()
            self.workers = []
            for i, rect in enumerate(self.regions):
                core_id = self.core_map[i] if i < len(self.core_map) else None
                worker = RegionWorker(region_index=i, region_rect=rect, result_queue=self.result_queue,
                                    fps=self.fps, core_id=core_id, dxcamera_ctor_kwargs=self.dxcamera_ctor_kwargs)
                worker.start()
                self.workers.append(worker)

        def _log_monitors(self):
            print("Available monitors:")
            for i, mon in enumerate(self._mss.monitors):
                try:
                    width, height = self.get_screen_resolution(i)
                    if i == 0:
                        print(f"  {i}: All monitors - {width}x{height}")
                    else:
                        system_scale = int(width / mon["width"]) if mon["width"] else 1
                        print(f"  {i}: Monitor {i} - {width}x{height} at ({mon['left'] * system_scale}, {mon['top'] * system_scale})")
                except Exception:
                    print(f"  {i}: Monitor {i} - info unavailable")

        def get_screen_resolution(self, index):
            monitor = self._mss.monitors[index]
            screen = self._mss.grab(monitor)
            return screen.size.width, screen.size.height

        def grab(self, timeout=1.0):
            """
            Trigger all region workers to capture once, collect their results, stitch them into
            a full-frame, resize to (scaled_width x scaled_height) and return (resized, (scaled_height, scaled_width)).
            timeout: seconds to wait for all regions (per-grab). Missing regions will be filled with black.
            """
            # Trigger captures on all workers
            for w in self.workers:
                w.trigger_capture()

            # Collect region images
            results = [None] * len(self.workers)
            deadline = time.time() + timeout
            # Keep collecting until we have all results or timeout
            while any(r is None for r in results) and time.time() < deadline:
                remaining = max(0.0, deadline - time.time())
                try:
                    idx, arr = self.result_queue.get(timeout=remaining)
                except queue.Empty:
                    break
                # arr may be None if worker failed to initialize; keep as None to be filled later
                if 0 <= idx < len(results):
                    results[idx] = arr

            # Replace missing results with black images matching expected region size
            for i, img in enumerate(results):
                if img is None:
                    left, top, width, height = self.regions[i]
                    results[i] = np.zeros((height, width, 3), dtype=np.uint8)

            # Stitch region images in row-major order into full_frame
            rows_imgs = []
            idx = 0
            for r in range(self.rows):
                row_cells = []
                for c in range(self.cols):
                    if idx >= self.num_regions:
                        break
                    row_cells.append(results[idx])
                    idx += 1
                if row_cells:
                    try:
                        row_concat = np.concatenate(row_cells, axis=1)
                    except Exception:
                        # If shapes mismatch, pad/truncate to expected widths
                        # Compute target heights/widths for this row
                        heights = [img.shape[0] for img in row_cells]
                        min_h = min(heights)
                        norm_cells = [img[:min_h, :, :] for img in row_cells]
                        row_concat = np.concatenate(norm_cells, axis=1)
                    rows_imgs.append(row_concat)

            if rows_imgs:
                try:
                    full_frame = np.concatenate(rows_imgs, axis=0)
                except Exception:
                    # Fallback: compute minimal width and height and crop each row
                    min_h = min(img.shape[0] for img in rows_imgs)
                    min_w = min(img.shape[1] for img in rows_imgs)
                    full_frame = np.zeros((min_h, min_w, 3), dtype=np.uint8)
                    # Fill with center-cropped first region as fallback
                    full_frame[:, :, :] = rows_imgs[0][:min_h, :min_w, :]
            else:
                full_frame = np.zeros((self.system_height, self.system_width, 3), dtype=np.uint8)

            # Resize to scaled dimensions (cv2 uses (width, height) order)
            resized = cv2.resize(full_frame, (self.scaled_width, self.scaled_height), interpolation=cv2.INTER_LINEAR)
            return resized, (self.scaled_height, self.scaled_width)

        def close(self):
            """Stop all workers and wait for them to exit."""
            for w in self.workers:
                try:
                    w.stop()
                except Exception:
                    pass
            for w in self.workers:
                w.join(timeout=1.0)

        # Context manager support
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            self.close()

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


import threading
import time, os
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from socketserver import ThreadingMixIn
from wsgiref.simple_server import make_server, WSGIServer
from utils import get_local_ip

# Path to favicon file
ICON_PATH = "icon2.ico"

# Custom WSGI server class that supports threading
class ThreadingWSGIServer(ThreadingMixIn, WSGIServer):
    allow_reuse_address = True  # Allows quick restart of server
    block_on_close = False      # Makes server shutdown faster

class MJPEGStreamer:
    def __init__(self, host="0.0.0.0", port=1303, fps=60, quality=90):
        """
        Initialize the MJPEG streamer with configuration parameters.
        """
        # MJPEG stream boundary marker
        self.boundary = b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
        self.quality = int(quality)  # JPEG quality parameter
        self.fps = fps               # Target frame rate
        self.delay = 1.0 / fps       # Delay between frames based on FPS

        # Frame storage and synchronization
        self.raw_frame = None       # Stores the latest raw frame (numpy RGB array)
        self.encoded_frame = None   # Stores the latest JPEG encoded frame
        self.lock = threading.Lock()  # Protects access to shared resources

        # Event flags for thread coordination
        self.shutdown = threading.Event()        # Signals when to stop all threads
        self.new_raw_event = threading.Event()   # Signals when new raw frame is available
        self.new_encoded_event = threading.Event()  # Signals when new encoded frame is ready

        # Stream dimensions (set when first frame arrives)
        self.sbs_width = None    # Width of side-by-side output
        self.sbs_height = None   # Height of side-by-side output
        self.index_bytes = None  # Cached HTML page bytes

        # HTML template for the viewer page with placeholders for FPS and dimensions
        # NOTE: the JS now dynamically resizes the canvas/video based on the actual
        # naturalWidth/naturalHeight of the MJPEG <img> frames, so clients will
        # automatically adapt if the server output resolution changes.
        self.template = """<!DOCTYPE html>
<html>
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <link rel="icon" type="image/x-icon" href="./favicon.ico">
        <title>Desktop2Stereo Streamer</title>
        <script>
            const FPS = {fps};
            const STREAM_URI = "/stream.mjpg";

            window.onload = () => {{
                const video = document.getElementById("player-canvas");
                let canvas = document.createElement("canvas");
                let ctx = canvas.getContext("2d");
                let canvasStream = null;

                // Create an <img> that will receive the MJPEG stream. For MJPEG the
                // image's onload handler fires for each frame, which lets us detect
                // size changes and redraw without a full page refresh.
                const img = new Image();
                img.crossOrigin = "anonymous";
                img.src = STREAM_URI;

                function ensureCanvasSize(w, h) {{
                    if (!w || !h) return;
                    if (canvas.width !== w || canvas.height !== h) {{
                        // Update canvas size to match incoming MJPEG frame size
                        canvas.width = w;
                        canvas.height = h;

                        // Stop old tracks if present and re-create capture stream so
                        // the <video> element gets the new resolution automatically.
                        if (canvasStream) {{
                            try {{ canvasStream.getTracks().forEach(t => t.stop()); }} catch(e) {{}}
                        }}
                        try {{
                            canvasStream = canvas.captureStream(FPS || 30);
                            video.srcObject = canvasStream;
                        }} catch(e) {{ /* captureStream may not be available in some browsers */ }}
                    }}
                }}

                let last_timestamp = 0;
                // Draw each time the <img> fires onload (MJPEG frames), and also use
                // requestAnimationFrame to throttle to FPS.
                img.onload = () => {{
                    const w = img.naturalWidth || img.width || canvas.width;
                    const h = img.naturalHeight || img.height || canvas.height;
                    ensureCanvasSize(w, h);

                    function render(timestamp) {{
                        if (timestamp - last_timestamp >= 1000.0 / (FPS || 30)) {{
                            last_timestamp = timestamp;
                            try {{ ctx.drawImage(img, 0, 0, canvas.width, canvas.height); }} catch(e) {{}}
                        }}
                        requestAnimationFrame(render);
                    }}

                    // Start (or continue) the render loop
                    requestAnimationFrame(render);
                }};

                // insert canvas into the page visually hidden (video shows the captured stream)
                canvas.style.display = 'none';
                document.body.appendChild(canvas);
            }};
        </script>
        <style type="text/css">
            body {{ margin: 0; background-color: rgb(45,48,53); }}
            .video-container {{ position: fixed; left: 0; top: 0; width: 100vw; height: 100vh; display:flex; align-items:center; justify-content:center; }}
            .video {{ max-height:100%; max-width:100%; width:auto; height:auto; background: black; }}
        </style>
    </head>
    <body>
        <div class="video-container">
            <video id="player-canvas" class="video" controls controlsList="nodownload"
                autoplay loop muted poster="" disablepictureinpicture ></video>
        </div>
    </body>
</html>
"""

        # WSGI application handler
        def app(environ, start_response):
            path = environ.get("PATH_INFO", "/")

            if path == "/":
                if self.index_bytes is None:
                    start_response("503 Service Unavailable", [("Content-Type", "text/plain")])
                    return [b"Stream not ready yet"]
                start_response("200 OK", [("Content-Type", "text/html; charset=utf-8")])
                return [self.index_bytes]

            if path == "/stream.mjpg":
                start_response("200 OK", [("Content-Type", "multipart/x-mixed-replace; boundary=frame")])
                return self._generate()

            if path == "/favicon.ico" and os.path.exists(ICON_PATH):
                with open(ICON_PATH, "rb") as f:
                    data = f.read()
                start_response("200 OK", [("Content-Type", "image/x-icon")])
                return [data]

            start_response("404 Not Found", [("Content-Type", "text/plain")])
            return [b"Not Found"]

        # Create WSGI server instance
        self.server = make_server(host, port, app, ThreadingWSGIServer)
        self.server_thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.encoder_thread = threading.Thread(target=self._encoder_loop, daemon=True)

    def start(self):
        print(f"[MJPEGStreamer] serving on http://{get_local_ip()}:{self.server.server_address[1]}/")
        self.server_thread.start()
        self.encoder_thread.start()

    def stop(self):
        self.shutdown.set()
        try:
            self.server.shutdown()
            self.server.server_close()
        except Exception:
            pass
        self.new_raw_event.set()

    def set_frame(self, rgb_or_tensor, depth_tensor=None, ipd_uv=0.064, depth_strength=1.0, display_mode="Half-SBS"):
        """
        Set the current frame to be streamed. This will also update the cached HTML
        page if the output resolution changes so newly-connecting clients see the
        correct initial dimensions.
        """
        with self.lock:
            if depth_tensor is not None:
                frame_np = self.make_sbs(rgb_or_tensor, depth_tensor, ipd_uv, depth_strength, display_mode)
            else:
                frame_np = rgb_or_tensor

            if frame_np is None:
                return

            h, w = frame_np.shape[:2]

            # If the output resolution changed, update cached index page so new
            # clients get the correct metadata. Existing clients will auto-resize
            # in the browser (JS checks the incoming MJPEG frame size), so no
            # manual refresh is needed.
            if (self.sbs_width, self.sbs_height) != (w, h):
                self.sbs_width = w
                self.sbs_height = h
                try:
                    self.index_bytes = self.template.format(
                        fps=self.fps,
                        width=self.sbs_width,
                        height=self.sbs_height
                    ).encode("utf-8")
                except Exception:
                    # Fallback to a minimal page if formatting fails
                    self.index_bytes = b"<html><body>Desktop2Stereo Streamer</body></html>"

            self.raw_frame = frame_np
            self.new_raw_event.set()

    def make_sbs(self, rgb: torch.Tensor, depth: torch.Tensor, ipd_uv=0.064, depth_strength=1.0, display_mode="Half-SBS"):
        """
        Create a side-by-side stereo image from RGB and depth.
        """
        if rgb.ndim == 3 and rgb.shape[0] in [1,3]:
            rgb_c = rgb
        elif rgb.ndim == 3:
            rgb_c = rgb.permute(2,0,1).contiguous()
        else:
            raise ValueError("rgb tensor must be (C,H,W) or (H,W,C)")

        C,H,W = rgb_c.shape
        device = rgb_c.device

        inv = torch.ones((H,W), device=device, dtype=rgb_c.dtype) - depth
        max_px = ipd_uv * W
        shifts = (inv * max_px * depth_strength / 10).round().to(torch.long, non_blocking=True)
        shifts_half = (shifts//2).clamp(0, W//2)

        xs = torch.arange(W, device=device).unsqueeze(0).expand(H,W)
        left_idx = (xs + shifts_half).clamp(0,W-1)
        right_idx = (xs - shifts_half).clamp(0,W-1)
        idx_left = left_idx.unsqueeze(0).expand(C,H,W)
        idx_right = right_idx.unsqueeze(0).expand(C,H,W)

        with torch.no_grad():
            gen_left = torch.gather(rgb_c, 2, idx_left)
            gen_right = torch.gather(rgb_c, 2, idx_right)

            def pad_to_aspect(img, target_ratio=(16,9)):
                """Pad image (C,H,W) to the nearest integer size with the requested
                aspect ratio (target_ratio = (width,height)).

                This implementation avoids float equality checks, ensures pad
                amounts are non-negative integers, and caps absurdly-large pad
                dimensions to prevent memory blow-ups when input sizes are
                unexpectedly huge.
                """
                if img is None:
                    return img
                _, h, w = img.shape
                if h == 0 or w == 0:
                    return img

                t_w, t_h = int(target_ratio[0]), int(target_ratio[1])

                # If current aspect already matches target exactly (integer check)
                if w * t_h == h * t_w:
                    return img

                # Cap max allowed padded dimension (safety to avoid runaway memory)
                MAX_DIM = 10000

                # If image is wider than target -> pad vertically
                if w * t_h > h * t_w:
                    # new height to meet target ratio
                    new_h = (w * t_h) // t_w
                    new_h = min(max(new_h, h), MAX_DIM)
                    pad_total = max(0, new_h - h)
                    pad_top = pad_total // 2
                    pad_bottom = pad_total - pad_top
                    return F.pad(img, (0, 0, pad_top, pad_bottom), value=0)
                else:
                    # pad horizontally
                    new_w = (h * t_w) // t_h
                    new_w = min(max(new_w, w), MAX_DIM)
                    pad_total = max(0, new_w - w)
                    pad_left = pad_total // 2
                    pad_right = pad_total - pad_left
                    return F.pad(img, (pad_left, pad_right, 0, 0), value=0)

            left = pad_to_aspect(gen_left)
            right = pad_to_aspect(gen_right)

            if display_mode == "TAB":
                out = torch.cat([left, right], dim=1)
            else:
                out = torch.cat([left, right], dim=2)

            if display_mode != "Full-SBS":
                # Downsample to a reasonable size (using left's single-view shape)
                target_size = left.shape[1:]
                out = F.interpolate(out.unsqueeze(0), size=target_size, mode="area")[0]

            out = out.clamp(0,255).to(torch.uint8)
            out_np = out.permute(1,2,0).contiguous().cpu().numpy()
        return out_np

    def _encoder_loop(self):
        """Background thread that continuously encodes frames to JPEG."""
        while not self.shutdown.is_set():
            if not self.new_raw_event.wait(timeout=1):
                continue
            # clear the event first so set_frame can notify future frames
            self.new_raw_event.clear()

            # copy the latest raw_frame under lock to avoid read/modify races
            with self.lock:
                raw = None if self.raw_frame is None else np.copy(self.raw_frame)

            if raw is None:
                continue

            try:
                # Convert RGB to BGR and encode as JPEG from a local copy
                bgr = np.ascontiguousarray(raw[..., ::-1])
                success, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
                if success:
                    # store immutable bytes atomically
                    self.encoded_frame = buf.tobytes()
                    self.new_encoded_event.set()  # Notify stream generator
            except Exception as e:
                print("[MJPEGStreamer] Encoding error:", e)

    def _generate(self):
        """Generator that yields MJPEG frames for the HTTP stream."""
        while not self.shutdown.is_set():
            if not self.new_encoded_event.wait(timeout=1):
                continue
            # clear event immediately so encoder can set the next frame
            self.new_encoded_event.clear()

            # take an atomic snapshot of the current encoded bytes
            f = self.encoded_frame
            if not f:
                continue

            try:
                # include Content-Length to make clients more robust
                header = b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n" % len(f)
                yield header + f + b"\r\n"
            except GeneratorExit:
                break
            except Exception as e:
                print("[MJPEGStreamer] _generate error:", e)
            time.sleep(self.delay)  # Maintain target frame rate

        # final sentinel
        yield b""

    def encode_jpeg(self, arr: np.ndarray) -> bytes:
        if arr is None:
            return b""
        bgr = np.ascontiguousarray(arr[..., ::-1])
        success, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
        return buf.tobytes() if success else b""

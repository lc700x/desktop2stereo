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
        
        Args:
            host (str): Host address to bind to (default: all interfaces)
            port (int): Port number to listen on
            fps (int): Target frames per second for the stream
            quality (int): JPEG compression quality (0-100)
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
        self.template = """<!DOCTYPE html>
<html>
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <link rel="icon" type="image/x-icon" href="./favicon.ico">
        <title>Desktop2Stereo Streamer</title>
        <script>
            const FPS = {fps};
            const WIDTH = {width};
            const HEIGHT = {height};
            const STREAM_URI = "/stream.mjpg";
            window.onload = () => {{
                let canvasStream = null;
                let canvas = document.createElement("canvas");
                canvas.width = WIDTH;
                canvas.height = HEIGHT;
                function setup_interval(){{
                    const ctx = canvas.getContext("2d");
                    const img = new Image();
                    let last_timestamp = 0;
                    img.src = STREAM_URI;
                    img.onload = () => {{
                        function render(timestamp) {{
                            if (timestamp - last_timestamp >= 1000.0 / FPS) {{
                                last_timestamp = timestamp;
                                ctx.drawImage(img, 0, 0);
                            }}
                            requestAnimationFrame(render);
                        }}
                        render();
                    }};
                }}
                function setup_video() {{
                    const video = document.getElementById("player-canvas");
                    const ctx = canvas.getContext('2d');
                    ctx.drawImage(video, 0, 0, WIDTH*2, HEIGHT);
                    canvasStream = canvas.captureStream(FPS);
                    video.srcObject = canvasStream;
                }}
                setup_interval();
                setup_video();
            }};
        </script>
        <style type="text/css">
            body {{ margin: 0; }}
            .video-container {{ position: fixed; left: 0; top: 0; width: 100vw; height: 100vh; background-color: rgb(45,48,53); }}
            .video {{ height:100%; width:100%; object-fit:contain; }}
        </style>
    </head>
    <body>
        <div class="video-container">
            <video id="player-canvas" class="video" controls controlsList="nodownload"
                autoplay loop muted poster="" disablepictureinpicture >
            </video>
        </div>
    </body>
</html>
"""

        # WSGI application handler
        def app(environ, start_response):
            """
            Handle HTTP requests and route them to appropriate responses.
            
            Args:
                environ: WSGI environment dictionary
                start_response: WSGI start_response callable
            """
            path = environ.get("PATH_INFO", "/")
            
            # Root path serves the HTML viewer page
            if path == "/":
                if self.index_bytes is None:
                    start_response("503 Service Unavailable", [("Content-Type", "text/plain")])
                    return [b"Stream not ready yet"]
                start_response("200 OK", [("Content-Type", "text/html; charset=utf-8")])
                return [self.index_bytes]

            # MJPEG stream endpoint
            if path == "/stream.mjpg":
                start_response("200 OK", [("Content-Type", "multipart/x-mixed-replace; boundary=frame")])
                return self._generate()  # Generator that yields MJPEG frames

            # Favicon endpoint
            if path == "/favicon.ico" and os.path.exists(ICON_PATH):
                with open(ICON_PATH, "rb") as f:
                    data = f.read()
                start_response("200 OK", [("Content-Type", "image/x-icon")])
                return [data]

            # 404 for all other paths
            start_response("404 Not Found", [("Content-Type", "text/plain")])
            return [b"Not Found"]

        # Create WSGI server instance
        self.server = make_server(host, port, app, ThreadingWSGIServer)
        # Thread for running the WSGI server
        self.server_thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        # Thread for encoding frames to JPEG
        self.encoder_thread = threading.Thread(target=self._encoder_loop, daemon=True)

    def start(self):
        """Start the streaming server and encoder threads."""
        print(f"[MJPEGStreamer] serving on http://{get_local_ip()}:{self.server.server_address[1]}/")
        self.server_thread.start()
        self.encoder_thread.start()

    def stop(self):
        """Stop all threads and clean up resources."""
        self.shutdown.set()  # Signal shutdown to all threads
        try:
            self.server.shutdown()
            self.server.server_close()
        except Exception:
            pass
        self.new_raw_event.set()  # Wake up encoder thread if it's waiting

    def set_frame(self, rgb_or_tensor, depth_tensor=None, ipd_uv=0.064, depth_strength=1.0, display_mode="Half-SBS"):
        """
        Set the current frame to be streamed.
        
        Args:
            rgb_or_tensor: Input RGB frame (numpy array or torch tensor)
            depth_tensor: Optional depth map for stereo effect
            ipd_uv: Interpupillary distance in UV coordinates (0-1)
            depth_strength: Strength of depth effect
            display_mode: How to display stereo ("Half-SBS", "Full-SBS", "TAB")
        """
        with self.lock:
            # Process frame based on input type
            if depth_tensor is not None:
                frame_np = self.make_sbs(rgb_or_tensor, depth_tensor, ipd_uv, depth_strength, display_mode)
            else:
                frame_np = rgb_or_tensor

            if frame_np is None:
                return

            # Set stream dimensions on first frame
            if self.sbs_width is None or self.sbs_height is None:
                self.sbs_height, self.sbs_width = frame_np.shape[:2]
                self.index_bytes = self.template.format(
                    fps=self.fps, 
                    width=self.sbs_width, 
                    height=self.sbs_height
                ).encode("utf-8")

            # Update frame and notify encoder thread
            self.raw_frame = frame_np
            self.new_raw_event.set()

    def make_sbs(self, rgb: torch.Tensor, depth: torch.Tensor, ipd_uv=0.064, depth_strength=1.0, display_mode="Half-SBS"):
        """
        Create a side-by-side stereo image from RGB and depth.
        
        Args:
            rgb: RGB input tensor (C,H,W) or (H,W,C)
            depth: Depth map tensor (H,W)
            ipd_uv: Interpupillary distance in UV coordinates
            depth_strength: Strength of depth effect
            display_mode: Display mode for stereo output
            
        Returns:
            numpy.ndarray: Processed stereo image in RGB format
        """
        # Validate and normalize input tensor format
        if rgb.ndim == 3 and rgb.shape[0] in [1,3]:
            rgb_c = rgb
        elif rgb.ndim == 3:
            rgb_c = rgb.permute(2,0,1).contiguous()
        else:
            raise ValueError("rgb tensor must be (C,H,W) or (H,W,C)")
            
        C,H,W = rgb_c.shape
        device = rgb_c.device

        # Calculate pixel shifts based on depth map
        inv = torch.ones((H,W), device=device, dtype=rgb_c.dtype) - depth
        max_px = ipd_uv * W  # Convert UV distance to pixels
        shifts = (inv * max_px * depth_strength / 10).round().to(torch.long, non_blocking=True)
        shifts_half = (shifts//2).clamp(0, W//2)

        # Create index maps for left and right views
        xs = torch.arange(W, device=device).unsqueeze(0).expand(H,W)
        left_idx = (xs + shifts_half).clamp(0,W-1)
        right_idx = (xs - shifts_half).clamp(0,W-1)
        idx_left = left_idx.unsqueeze(0).expand(C,H,W)
        idx_right = right_idx.unsqueeze(0).expand(C,H,W)

        with torch.no_grad():
            # Generate stereo views using gathered pixels
            gen_left = torch.gather(rgb_c, 2, idx_left)
            gen_right = torch.gather(rgb_c, 2, idx_right)

            def pad_to_aspect(img, target_ratio=(16,9)):
                """Pad image to match target aspect ratio."""
                _, h, w = img.shape
                t_w, t_h = target_ratio
                r_img = w/h
                r_t = t_w/t_h
                
                if abs(r_img-r_t) < 0.001:
                    return img
                elif r_img > r_t:
                    # Pad vertically
                    new_H = int(round(w/r_t))
                    pad_top = (new_H-h)//2
                    pad_bottom = new_H-h-pad_top
                    return F.pad(img, (0,0,pad_top,pad_bottom), value=0)
                else:
                    # Pad horizontally
                    new_W = int(round(h*r_t))
                    pad_left = (new_W-w)//2
                    pad_right = new_W-w-pad_left
                    return F.pad(img, (pad_left,pad_right,0,0), value=0)

            # Pad both views to target aspect ratio
            left = pad_to_aspect(gen_left)
            right = pad_to_aspect(gen_right)

            # Combine views based on display mode
            if display_mode == "TAB":
                out = torch.cat([left, right], dim=1)  # Top-and-bottom
            else:
                out = torch.cat([left, right], dim=2)  # Side-by-side

            # Downsample if not full resolution SBS
            if display_mode != "Full-SBS":
                out = F.interpolate(out.unsqueeze(0), size=left.shape[1:], mode="area")[0]

            # Convert to numpy array in HWC format
            out = out.clamp(0,255).to(torch.uint8)
            out_np = out.permute(1,2,0).contiguous().cpu().numpy()
        return out_np

    def _encoder_loop(self):
        """Background thread that continuously encodes frames to JPEG."""
        while not self.shutdown.is_set():
            # Wait for new frame or shutdown
            if not self.new_raw_event.wait(timeout=1):
                continue
            self.new_raw_event.clear()

            raw = self.raw_frame
            if raw is None:
                continue
                
            try:
                # Convert RGB to BGR (OpenCV format) and encode as JPEG
                bgr = np.ascontiguousarray(raw[..., ::-1])
                success, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
                if success:
                    self.encoded_frame = buf.tobytes()
                    self.new_encoded_event.set()  # Notify stream generator
            except Exception as e:
                print("[MJPEGStreamer] Encoding error:", e)

    def _generate(self):
        """Generator that yields MJPEG frames for the HTTP stream."""
        while not self.shutdown.is_set():
            # Wait for new encoded frame or shutdown
            if not self.new_encoded_event.wait(timeout=1):
                continue
            self.new_encoded_event.clear()
            
            f = self.encoded_frame
            if f:
                # Yield frame with MJPEG boundary markers
                yield self.boundary + f + b"\r\n"
                time.sleep(self.delay)  # Maintain target frame rate
        yield b""  # Empty yield on shutdown

    def encode_jpeg(self, arr: np.ndarray) -> bytes:
        """Utility method to encode a numpy array as JPEG."""
        if arr is None:
            return b""
        bgr = np.ascontiguousarray(arr[..., ::-1])
        success, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
        return buf.tobytes() if success else b""
import threading
import time
import io
from socketserver import ThreadingMixIn
from wsgiref.simple_server import make_server, WSGIServer
from collections import deque
from PIL import Image
import numpy as np
import torch

class ThreadingWSGIServer(ThreadingMixIn, WSGIServer):
    allow_reuse_address = True
    block_on_close = False

class MJPEGStreamer:
    """
    Simple MJPEG server (multipart/x-mixed-replace).  
    Call `set_frame(jpeg_bytes)` each time you have a new JPEG.
    """
    def __init__(self,
                 host: str = "0.0.0.0",
                 port: int = 1303,
                 fps: int = 60,
                 quality: int = 100):
        self.boundary   = b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
        self.delay      = 1.0 / fps
        self.quality    = quality
        self.frame      = None
        self.lock       = threading.Lock()
        self.shutdown   = threading.Event()

        # We'll initialize these after the first frame is set
        self.sbs_width = None
        self.sbs_height = None

        # Initial template without width/height (will be updated)
        tpl = """<!DOCTYPE html>
<html>
<head>
  <title>Desktop2Stereo streaming</title>
  <script>
    const FPS = {fps};
    const WIDTH = {width};
    const HEIGHT = {height};
    const STREAM_URI = "/stream.mjpg";

    window.onload = () => {{
        let canvasStream = null;
        let canvas = document.createElement("canvas");
        let process_token = null;
        let stop_update = false;

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
            // check server restart
            setInterval(() => {{
	        if (document.hidden) {{
	            return;
	        }}
	        fetch('/process_token',
	              {{
                          method: 'GET',
	              }})
	            .then((res) => {{
                        if (!res.ok) {{
	                    stop_update = true;
	                    //console.log("err1", res.status, res.statusText);
                        }}
                        return res.json();
	            }})
	            .then((res) => {{
	                //console.log("check token", process_token, res.token);
	                if (process_token == null) {{
	                    process_token = res.token;
	                }} else if (process_token != res.token) {{
	                    // reload
	                    process_token = null;
	                    location.reload();
	                }}
	            }})
	            .catch((reason) => {{
	                stop_update = true;
                        //console.log("err2", reason);
	            }});
            }}, 4000);
        }}
        
        function setup_video() {{
            const video = document.getElementById("player-canvas");
            const ctx = canvas.getContext('2d');
            ctx.drawImage(video, 0, 0, WIDTH*2, HEIGHT);
            canvasStream = canvas.captureStream(FPS);  // 0-FPS freq
            video.srcObject = canvasStream;
        }}
        setup_interval();
        setup_video();
    }};
  </script>
  <style type="text/css">
  body {{
    margin: 0;
  }}
  .video-container {{
    position: fixed;
    left: 0;
    top: 0;
    z-index: 0;
    width: 100vw;
    height: 100vh;
    text-align: center;
    background-color: rgb(45, 48, 53);
  }}
  .video {{
    height: 100%;
    width: 100%;
    object-fit: contain;
    object-position: center;
  }}
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
        self.template = tpl
        self.index_bytes = None  # Will be set after first frame

        def app(environ, start_response):
            path = environ.get("PATH_INFO", "/")
            if path == "/":
                if self.index_bytes is None:
                    start_response("503 Service Unavailable", [("Content-Type","text/plain")])
                    return [b"Stream not ready yet"]
                start_response("200 OK",
                    [("Content-Type","text/html; charset=utf-8")])
                return [self.index_bytes]

            if path == "/stream.mjpg":
                start_response("200 OK", [
                    ("Content-Type","multipart/x-mixed-replace; boundary=frame"),
                ])
                return self._generate()
            
            start_response("404 Not Found", [("Content-Type","text/plain")])
            return [b"Not Found"]

        self.server = make_server(host, port, app, ThreadingWSGIServer)
        self.thread = threading.Thread(
            target=self.server.serve_forever, daemon=True
        )

    def start(self):
        print(f"[MJPEGStreamer] serving on http://{self.server.server_address[0]}:{self.server.server_address[1]}/")
        self.thread.start()

    def stop(self):
        self.shutdown.set()
        self.server.shutdown()
        self.server.server_close()

    def set_frame(self, jpeg_bytes: bytes):
        with self.lock:
            self.frame = jpeg_bytes

    def _generate(self):
        # generator for the WSGI response
        while not self.shutdown.is_set():
            with self.lock:
                f = self.frame

            if f:
                yield self.boundary + f + b"\r\n"
            time.sleep(self.delay)
        yield b""  # close out

    @staticmethod
    def make_sbs(rgb: np.ndarray,
                 depth: np.ndarray,
                 ipd_uv: float,
                 depth_strength: float,
                 *,
                 half: bool = True
                ) -> np.ndarray:
        """
        Build a side-by-side stereo frame.

        Parameters
        ----------
        rgb : H×W×3 float32 array in range [0..1]
        depth : H×W float32 array in range [0..1]
        ipd_uv : float
            interpupillary distance in UV space (0–1, relative to image width)
        depth_strength : float
            multiplier applied to the per-pixel horizontal parallax
        half : bool, optional
            If True, returns “half-SBS”: the output width equals W.
            Otherwise returns full-width SBS (2W).

        Returns
        -------
        np.ndarray
            uint8 image of shape (H, 2W, 3) if half == False,
            or (H, W, 3) when half == True.
        """
        H, W = depth.shape
        inv = 1.0 - depth
        max_px = int(ipd_uv * W)
        shifts = (inv * max_px * depth_strength).astype(np.int32)

        left  = np.zeros_like(rgb)
        right = np.zeros_like(rgb)
        xs    = np.arange(W)[None, :]

        for y in range(H):
            s = shifts[y]
            xx_left  = np.clip(xs + (s // 2),  0, W-1)
            xx_right = np.clip(xs - (s // 2),  0, W-1)
            left[y]  = rgb[y, xx_left]
            right[y] = rgb[y, xx_right]

        # Full-resolution SBS: concatenate horizontally
        sbs_full = np.concatenate((left, right), axis=1)  # H × (2W) × 3

        if not half:
            return sbs_full.astype(np.uint8)

        # Half-SBS: simple 2:1 sub-sampling in X direction
        # (i.e., take every second column)
        sbs_half = sbs_full[:, ::2, :]  # H × W × 3
        # import cv2  # For debugging purposes, can be removed later
        # cv2.imwrite("sbs_half.jpg", sbs_half)  # Debugging line, can be removed
        # exit()
        return sbs_half.astype(np.uint8)
    
    def make_sbs_tensor(rgb: torch.Tensor, depth: torch.Tensor, ipd_uv: float = 0.064, depth_strength: float = 0.1, half: bool = True) -> np.array:
        """
        Build a side-by-side stereo frame using PyTorch tensors with DirectML support.

        Parameters
        ----------
        rgb : Tensor (3, H, W) float32 in range [0..1]
        depth : Tensor (H, W) float32 in range [0..1]
        ipd_uv : float
            interpupillary distance in UV space (0-1, relative to image width)
        depth_strength : float
            multiplier applied to the per-pixel horizontal parallax
        half : bool, optional
            If True, returns "half-SBS": the output width equals W.
            Otherwise returns full-width SBS (2W).

        Returns
        -------
        torch.Tensor
            uint8 image of shape (3, H, 2W) if half == False,
            or (3, H, W) when half == True.
        """
        # Ensure tensors are on the same device
        device = rgb.device
        depth = depth.to(device)
        _, H, W = rgb.shape
        # reshape depth[384, 384] to match rgb dimensions [H, W]
        inv = 1.0 - depth
        max_px = int(ipd_uv * W)
        shifts = (inv * max_px * depth_strength).round().long()
        
        # Create coordinate grid
        xs = torch.arange(W, device=device).expand(H, -1)  # (H, W)
        
        # Calculate left and right coordinates
        shifts_half = (shifts // 2).clamp(min=0, max=W//2)
        left_coords = torch.clamp(xs + shifts_half, 0, W-1)
        right_coords = torch.clamp(xs - shifts_half, 0, W-1)
        
        # Gather pixels using advanced indexing
        rgb = rgb.permute(1, 2, 0)  # (H, W, 3) for easier indexing
        # Vectorized implementation using gather
        y_indices = torch.arange(H, device=device)[:, None].expand(-1, W)
        left = rgb[y_indices, left_coords]
        right = rgb[y_indices, right_coords]
        
        # Concatenate for full SBS
        sbs_full = torch.cat([left, right], dim=1)  # (H, 2W, 3)
        
        if not half:
            return (sbs_full * 255).clamp(0, 255).byte().cpu().numpy()
        
        # For half-SBS, use strided sampling
        sbs_half = sbs_full[:, ::2, :]  # (H, 2W, 3) to (H, W, 3)
        return (sbs_half * 255).clamp(0, 255).byte().cpu().numpy()

    def encode_jpeg(self, arr: np.ndarray) -> bytes:
        """
        Encode a H×(W or 2W)×3 uint8 numpy array to JPEG bytes.
        Also updates the HTML template with the correct dimensions if this is the first frame.
        """
        # Update dimensions if this is the first frame
        if self.sbs_width is None or self.sbs_height is None:
            self.sbs_height, self.sbs_width = arr.shape[:2]
            self.index_bytes = self.template.format(
                fps=self.delay and int(1/self.delay) or 60,
                width=self.sbs_width,
                height=self.sbs_height
            ).encode("utf-8")

        img = Image.fromarray(arr, 'RGB')
        buf = io.BytesIO()
        img.save(buf, format='JPEG', quality=self.quality)
        return buf.getvalue()
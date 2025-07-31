import threading
import time
import io
from socketserver import ThreadingMixIn
from wsgiref.simple_server import make_server, WSGIServer
from collections import deque
from PIL import Image
import numpy as np

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

        # richer index page (canvas-based player)
        tpl = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Desktop2Stereo MJPEG Stream</title>
  <script>
    /* Configuration injected by the server */
    const FPS         = {fps};                 // target refresh rate
    const STREAM_URI  = "/stream.mjpg";        // MJPEG endpoint

    /* Fallback dimensions (will be overwritten once the first frame arrives) */
    let FRAME_WIDTH  = 640;
    let FRAME_HEIGHT = 480;

    /* Convenience: reload if the backend restarts */
    let process_token = null;

    window.onload = () => {{
        /* Create a canvas that will display the MJPEG stream */
        const canvas  = document.createElement("canvas");
        const ctx     = canvas.getContext("2d");
        document.body.appendChild(canvas);

        /* <video> element that will show the canvas stream (lets us reuse
           browser video controls such as fullscreen)                       */
        const video   = document.createElement("video");
        video.className = "video";
        video.autoplay  = true;
        video.muted     = true;
        video.controls  = true;
        video.playsInline = true;
        document.querySelector(".video-container").appendChild(video);

        /* Draw MJPEG frames to the canvas, then capture the canvas as a
           MediaStream and feed it into the <video> element.               */
        const img   = new Image();
        img.src     = STREAM_URI + "?_=" + Date.now(); // cache-buster

        img.onload = () => {{
            /* First frame gives us the real dimensions */
            FRAME_WIDTH  = img.width;
            FRAME_HEIGHT = img.height;
            canvas.width  = FRAME_WIDTH;
            canvas.height = FRAME_HEIGHT;

            /* Begin pumping frames */
            function renderLoop() {{
                ctx.drawImage(img, 0, 0);
                requestAnimationFrame(renderLoop);
            }}
            renderLoop();

            /* Refresh MJPEG image roughly every FPS interval */
            setInterval(() => {{
                img.src = STREAM_URI + "?_=" + Date.now();
            }}, 1000 / FPS);

            /* Pipe the canvas into the video element */
            const stream = canvas.captureStream(FPS);
            video.srcObject = stream;
        }};

        /* Periodically ask for a token so we can detect a server restart */
        setInterval(() => {{
            if (document.hidden) return;
            fetch('/process_token')
                .then(r => r.ok ? r.json() : Promise.reject(r.status))
                .then(r => {{
                    if (process_token === null) {{
                        process_token = r.token;
                    }} else if (process_token !== r.token) {{
                        location.reload(); // backend restarted
                    }}
                }})
                .catch(() => {{ /* ignore errors – page will try again */ }});
        }}, 4000);
    }};
  </script>
  <style>
    body, html {{
      margin: 0;
      width: 100vw;
      height: 100vh;
      overflow: hidden;
      background-color: rgb(45, 48, 53);
      display: flex;
      align-items: center;
      justify-content: center;
    }}
    .video-container {{
      position: relative;
      width: 100%;
      height: 100%;
    }}
    .video {{
      width: 100%;
      height: 100%;
      object-fit: contain;
    }}
    canvas {{
      display: none; /* kept for captureStream, not visible */
    }}
  </style>
</head>
<body>
  <div class="video-container"></div>
</body>
</html>"""
        self.index_bytes = tpl.encode("utf-8")

        def app(environ, start_response):
            path = environ.get("PATH_INFO", "/")
            if path == "/":
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
        return sbs_half.astype(np.uint8)

    def encode_jpeg(self, arr: np.ndarray) -> bytes:
        """
        Encode a H×(W or 2W)×3 uint8 numpy array to JPEG bytes.
        """
        img = Image.fromarray(arr, 'RGB')
        buf = io.BytesIO()
        img.save(buf, format='JPEG', quality=self.quality)
        return buf.getvalue()

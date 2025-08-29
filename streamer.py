import threading
import time, os
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from socketserver import ThreadingMixIn
from wsgiref.simple_server import make_server, WSGIServer
from utils import get_local_ip

ICON_PATH = "icon2.ico"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32

class ThreadingWSGIServer(ThreadingMixIn, WSGIServer):
    allow_reuse_address = True
    block_on_close = False

class MJPEGStreamer:
    def __init__(self, host="0.0.0.0", port=1303, fps=60, quality=90):
        self.boundary = b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
        self.quality = int(quality)
        self.fps = fps
        self.delay = 1.0 / fps

        self.raw_frame = None       # latest frame (numpy RGB)
        self.encoded_frame = None   # latest JPEG
        self.lock = threading.Lock()

        self.shutdown = threading.Event()
        self.new_raw_event = threading.Event()
        self.new_encoded_event = threading.Event()

        self.sbs_width = None
        self.sbs_height = None
        self.index_bytes = None

        # HTML template with auto reconnect and fullscreen
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

        # Setup WSGI server
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
        with self.lock:
            if depth_tensor is not None:
                frame_np = self.make_sbs(rgb_or_tensor, depth_tensor, ipd_uv, depth_strength, display_mode)
            else:
                frame_np = rgb_or_tensor

            if frame_np is None:
                return

            if self.sbs_width is None or self.sbs_height is None:
                self.sbs_height, self.sbs_width = frame_np.shape[:2]
                self.index_bytes = self.template.format(fps=self.fps, width=self.sbs_width, height=self.sbs_height).encode("utf-8")

            self.raw_frame = frame_np
            self.new_raw_event.set()

    def make_sbs(self, rgb: torch.Tensor, depth: torch.Tensor, ipd_uv=0.064, depth_strength=1.0, display_mode="Half-SBS"):
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
            gen_left = torch.gather(rgb_c,2,idx_left)
            gen_right = torch.gather(rgb_c,2,idx_right)

            def pad_to_aspect(img, target_ratio=(16,9)):
                _, h, w = img.shape
                t_w, t_h = target_ratio
                r_img = w/h
                r_t = t_w/t_h
                if abs(r_img-r_t)<0.001:
                    return img
                elif r_img>r_t:
                    new_H = int(round(w/r_t))
                    pad_top = (new_H-h)//2
                    pad_bottom = new_H-h-pad_top
                    return F.pad(img,(0,0,pad_top,pad_bottom),value=0)
                else:
                    new_W = int(round(h*r_t))
                    pad_left = (new_W-w)//2
                    pad_right = new_W-w-pad_left
                    return F.pad(img,(pad_left,pad_right,0,0),value=0)

            left = pad_to_aspect(gen_left)
            right = pad_to_aspect(gen_right)

            if display_mode=="TAB":
                out = torch.cat([left,right],dim=1)
            else:
                out = torch.cat([left,right],dim=2)

            if display_mode!="Full-SBS":
                out = F.interpolate(out.unsqueeze(0),size=left.shape[1:],mode="area")[0]

            out = out.clamp(0,255).to(torch.uint8)
            out_np = out.permute(1,2,0).contiguous().cpu().numpy()
        return out_np

    def _encoder_loop(self):
        while not self.shutdown.is_set():
            if not self.new_raw_event.wait(timeout=1):
                continue
            self.new_raw_event.clear()

            raw = self.raw_frame
            if raw is None:
                continue
            try:
                bgr = np.ascontiguousarray(raw[..., ::-1])
                success, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
                if success:
                    self.encoded_frame = buf.tobytes()
                    self.new_encoded_event.set()
            except Exception as e:
                print("[MJPEGStreamer] Encoding error:", e)

    def _generate(self):
        while not self.shutdown.is_set():
            if not self.new_encoded_event.wait(timeout=1):
                continue
            self.new_encoded_event.clear()
            f = self.encoded_frame
            if f:
                yield self.boundary + f + b"\r\n"
                time.sleep(self.delay)
        yield b""

    def encode_jpeg(self, arr: np.ndarray) -> bytes:
        if arr is None:
            return b""
        bgr = np.ascontiguousarray(arr[..., ::-1])
        success, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
        return buf.tobytes() if success else b""

import threading
import time, os
import numpy as np
import cv2
from socketserver import ThreadingMixIn
from wsgiref.simple_server import make_server, WSGIServer
from utils import get_local_ip

# Path to favicon file
ICON_PATH = "icon2.ico"

# Custom WSGI server class that supports threading
class ThreadingWSGIServer(ThreadingMixIn, WSGIServer):
    allow_reuse_address = True
    block_on_close = False

class MJPEGStreamer:
    def __init__(self, host="0.0.0.0", port=1303, fps=60, quality=90):
        """
        Initialize the MJPEG streamer with configuration parameters.
        """
        # MJPEG stream boundary marker
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

        # Stream dimensions
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

    def set_frame(self, frame_np):
        """
        Set the current frame to be streamed. This will also update the cached HTML
        page if the output resolution changes so newly-connecting clients see the
        correct initial dimensions.
        """
        with self.lock:
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
        next_frame_time = time.perf_counter()
        while not self.shutdown.is_set():
            # Wait for a new frame
            if not self.new_encoded_event.wait(timeout=1):
                continue
            self.new_encoded_event.clear()

            f = self.encoded_frame
            if f:
                yield self.boundary + f + b"\r\n"

            # Enforce consistent pacing
            next_frame_time += self.delay
            sleep_time = next_frame_time - time.perf_counter()
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                # Frame processing fell behind, reset clock
                next_frame_time = time.perf_counter()
        yield b""

    def encode_jpeg(self, arr: np.ndarray) -> bytes:
        if arr is None:
            return b""
        bgr = np.ascontiguousarray(arr[..., ::-1])
        success, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
        return buf.tobytes() if success else b""

import threading
import io
from socketserver import ThreadingMixIn
from wsgiref.simple_server import make_server, WSGIServer
import numpy as np
import cv2
from PIL import Image
class ThreadingWSGIServer(ThreadingMixIn, WSGIServer):
    allow_reuse_address = True
    block_on_close = False

class MJPEGStreamer:
    """
    MJPEG server for side-by-side stereo streaming.
    Optimized with TurboJPEG and non-blocking frame delivery.
    """

    def __init__(self,
                 host: str = "0.0.0.0",
                 port: int = 1303,
                 fps: int = 60,
                 quality: int = 90):
        self.boundary = b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
        self.quality = quality
        self.frame = None
        self.lock = threading.Lock()
        self.shutdown = threading.Event()

        self.sbs_width = None
        self.sbs_height = None
        self.index_bytes = None
        
        self.delay      = round(1.0 / fps, 4)

        # Lightweight HTML page
        self.template = """<!DOCTYPE html>
<html>
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Desktop2Stereo Streamer</title>
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

        def app(environ, start_response):
            path = environ.get("PATH_INFO", "/")
            if path == "/":
                if self.index_bytes is None:
                    start_response("503 Service Unavailable",
                                   [("Content-Type", "text/plain")])
                    return [b"Stream not ready yet"]
                start_response("200 OK",
                               [("Content-Type", "text/html; charset=utf-8")])
                return [self.index_bytes]

            if path == "/stream.mjpg":
                start_response("200 OK", [
                    ("Content-Type",
                     "multipart/x-mixed-replace; boundary=frame"),
                ])
                return self._generate()

            start_response("404 Not Found",
                           [("Content-Type", "text/plain")])
            return [b"Not Found"]

        self.server = make_server(host, port, app, ThreadingWSGIServer)
        self.thread = threading.Thread(target=self.server.serve_forever,
                                       daemon=True)

    def start(self):
        print(
            f"[MJPEGStreamer] serving on http://{self.server.server_address[0]}:{self.server.server_address[1]}/"
        )
        self.thread.start()

    def stop(self):
        self.shutdown.set()
        self.server.shutdown()
        self.server.server_close()

    def set_frame(self, jpeg_bytes):
        """
        Provide a new RGB frame (H, W, 3) uint8.
        """
        with self.lock:
            self.frame = jpeg_bytes

    def _generate(self):
        """
        Stream frames as multipart MJPEG.
        No artificial sleep; yields frames as soon as available.
        """
        while not self.shutdown.is_set():
            f = None
            with self.lock:
                if self.frame is not None:
                    f = self.frame
            if f:
                yield self.boundary + f + b"\r\n"
                # time.sleep(self.delay)
        yield b""  # End of stream
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


# Example usage:
if __name__ == "__main__":
    import cv2

    cap = cv2.VideoCapture(0)
    streamer = MJPEGStreamer(port=1303, fps=60, quality=90)
    streamer.start()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        streamer.set_frame(frame)

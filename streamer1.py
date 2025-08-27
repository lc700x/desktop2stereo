import threading
import io, time, os, subprocess
from socketserver import ThreadingMixIn
from wsgiref.simple_server import make_server, WSGIServer
import numpy as np
import cv2
from utils import get_local_ip

ICON_PATH = "icon2.ico"
OUTPUT_MP4 = "static/stream.mp4"

class ThreadingWSGIServer(ThreadingMixIn, WSGIServer):
    allow_reuse_address = True
    block_on_close = False

class MP4Streamer:
    """
    MP4 server (video+audio) using FFmpeg for encoding.
    """

    def __init__(self,
                 host: str = "0.0.0.0",
                 port: int = 1303,
                 fps: int = 30):
        self.fps = fps
        self.index_bytes = None
        self.ffmpeg = None

        # Lightweight HTML page with <video>
        self.template = """<!DOCTYPE html>
<html>
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Desktop2Stereo MP4 Streamer</title>
        <style>
            body { margin: 0; background: #2d3035; }
            .video-container {
                position: fixed;
                left: 0; top: 0;
                width: 100vw; height: 100vh;
                display: flex; justify-content: center; align-items: center;
            }
            .video {
                width: 100%; height: 100%;
                object-fit: contain;
            }
        </style>
    </head>
    <body>
        <div class="video-container">
            <video class="video" autoplay controls>
                <source src="/stream.mp4" type="video/mp4">
                Your browser does not support HTML5 video.
            </video>
        </div>
    </body>
</html>
"""

        def app(environ, start_response):
            path = environ.get("PATH_INFO", "/")
            if path == "/":
                if self.index_bytes is None:
                    self.index_bytes = self.template.encode("utf-8")
                start_response("200 OK",
                               [("Content-Type", "text/html; charset=utf-8")])
                return [self.index_bytes]

            if path == "/stream.mp4":
                if os.path.exists(OUTPUT_MP4):
                    start_response("200 OK",
                                   [("Content-Type", "video/mp4")])
                    return open(OUTPUT_MP4, "rb")
                else:
                    start_response("503 Service Unavailable",
                                   [("Content-Type", "text/plain")])
                    return [b"Stream not ready yet"]

            if path == "/favicon.ico":
                if os.path.exists(ICON_PATH):
                    with open(ICON_PATH, "rb") as f:
                        data = f.read()
                    start_response("200 OK", [("Content-Type", "image/x-icon")])
                    return [data]
                else:
                    start_response("404 Not Found",
                                   [("Content-Type", "text/plain")])
                    return [b"Icon not found"]

            start_response("404 Not Found",
                           [("Content-Type", "text/plain")])
            return [b"Not Found"]

        self.server = make_server(host, port, app, ThreadingWSGIServer)
        self.thread = threading.Thread(target=self.server.serve_forever,
                                       daemon=True)

    def start(self):
        print(f"[MP4Streamer] serving on http://{get_local_ip()}:{self.server.server_address[1]}/")
        self.thread.start()

    def start_ffmpeg(self, width, height):
        """
        Start FFmpeg to encode raw video frames + system audio into MP4.
        """
        command = [
            "ffmpeg",
            "-y",
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s", f"{width}x{height}",
            "-r", str(self.fps),
            "-i", "-",                   # raw video via stdin
            "-f", "dshow",               # Windows DirectShow input
            "-i", "audio=Microphone (Realtek(R) Audio)",  # <-- change this!
            "-c:v", "libx264", "-preset", "ultrafast", "-tune", "zerolatency",
            "-c:a", "aac", "-b:a", "128k",
            "-movflags", "frag_keyframe+empty_moov",  # streamable mp4
            "-f", "mp4",
            OUTPUT_MP4
        ]
        print("[FFmpeg] starting:", " ".join(command))
        self.ffmpeg = subprocess.Popen(command, stdin=subprocess.PIPE)

    def push_frame(self, frame: np.ndarray):
        """
        Write a new frame to FFmpeg stdin.
        """
        if self.ffmpeg:
            self.ffmpeg.stdin.write(frame.tobytes())

    def stop(self):
        if self.ffmpeg:
            self.ffmpeg.stdin.close()
            self.ffmpeg.wait()
        self.server.shutdown()
        self.server.server_close()

# Example usage
if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30

    streamer = MP4Streamer(port=1303, fps=fps)
    streamer.start()
    streamer.start_ffmpeg(width, height)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            streamer.push_frame(frame)
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        streamer.stop()

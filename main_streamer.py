import threading
import queue
import time
from utils import OUTPUT_RESOLUTION, DISPLAY_MODE, IPD, FPS, DEPTH_STRENTH
from capture import DesktopGrabber
from depth import predict_depth_tensor, process
from streamer import MJPEGStreamer

TIME_SLEEP = round(1.0 / FPS, 2)

# Queues with size=1 (latest-frame-only logic)
raw_q = queue.Queue(maxsize=1)
proc_q = queue.Queue(maxsize=1)
depth_q = queue.Queue(maxsize=1)

def put_latest(q, item):
    """Put item into queue, dropping old one if needed (non-blocking)."""
    if q.full():
        try:
            q.get_nowait()
        except queue.Empty:
            pass
    try:
        q.put_nowait(item)
    except queue.Full:
        time.sleep(TIME_SLEEP)  # Drop frame if race condition occurs

def capture_loop():
    cap = DesktopGrabber(output_resolution=OUTPUT_RESOLUTION, fps=FPS)
    while True:
        try:
            frame_raw, size = cap.grab()
        except OSError:
            exit()
        put_latest(raw_q, (frame_raw, size))

def process_loop():
    while True:
        try:
            frame_raw, size = raw_q.get(timeout = TIME_SLEEP)
        except queue.Empty:
            continue
        frame_rgb = process(frame_raw, size)
        put_latest(proc_q, frame_rgb)

def depth_loop():
    while True:
        try:
            frame_rgb = proc_q.get(timeout = TIME_SLEEP)
        except queue.Empty:
            continue
        image_rgb, depth = predict_depth_tensor(frame_rgb)
        put_latest(depth_q, (image_rgb, depth))

def main():
    # Start capture and processing threads
    threading.Thread(target=capture_loop, daemon=True).start()
    threading.Thread(target=process_loop, daemon=True).start()
    threading.Thread(target=depth_loop, daemon=True).start()
    
    STREAM_HOST      = "0.0.0.0"
    STREAM_PORT      = 1303
    STREAM_FPS       = 60
    STREAM_QUALITY   = 100

    # start MJPEG streamer
    streamer = MJPEGStreamer(
        host=STREAM_HOST,
        port=STREAM_PORT,
        fps=STREAM_FPS,
        quality=STREAM_QUALITY
    )
    streamer.start()

    print(f"[Main] Streaming side-by-side MJPEG at http://{STREAM_HOST}:{STREAM_PORT}/")

    try:
        while True:
            try:
                rgb, depth = depth_q.get(timeout=TIME_SLEEP)

                # build a CPU SBS uint8 frame and encode to JPEG
                sbs = MJPEGStreamer.make_sbs(
                    rgb,
                    depth,
                    ipd_uv=IPD,
                    depth_strength=DEPTH_STRENTH/10
                )
                jpg = streamer.encode_jpeg(sbs)

                # push into the HTTP MJPEG server
                streamer.set_frame(jpg)
            except queue.Empty:
                    time.sleep(TIME_SLEEP)
    except KeyboardInterrupt:
        print("\n[Main] Shutting down…")
    finally:
        streamer.stop()


if __name__ == "__main__":
    main()

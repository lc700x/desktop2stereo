# main.py
import threading
import queue
import time

from capture import DesktopGrabber
from depth   import predict_depth
from streaming import MJPEGStreamer

# Configuration
MONITOR_INDEX   = 2      # which monitor to grab
DOWNSCALE_FACTOR = 0.5   # resolution scaling for performance

STREAM_HOST      = "0.0.0.0"
STREAM_PORT      = 1303
STREAM_FPS       = 30
STREAM_QUALITY   = 80

# stereo-warp parameters (match whatever you used in your GLSL)
IPD_UV          = 0.064
DEPTH_STRENGTH  = 0.1

# Queues for pipelining
rgb_raw_q  = queue.Queue(maxsize=3)
rgb_proc_q = queue.Queue(maxsize=3)
depth_q    = queue.Queue(maxsize=3)


def capture_loop():
    grabber = DesktopGrabber(
        monitor_index=MONITOR_INDEX,
        downscale=DOWNSCALE_FACTOR
    )
    while True:
        frame = grabber.grab()  # raw BGR frame
        try:
            rgb_raw_q.put_nowait(frame)
        except queue.Full:
            rgb_raw_q.get_nowait()
            rgb_raw_q.put_nowait(frame)


def processing_loop():
    grabber = DesktopGrabber(
        monitor_index=MONITOR_INDEX,
        downscale=DOWNSCALE_FACTOR,
        show_monitor_info=False
    )
    while True:
        try:
            raw = rgb_raw_q.get(timeout=0.01)
        except queue.Empty:
            continue

        proc = grabber.process(raw)  # RGB float32 [0..1]
        try:
            rgb_proc_q.put_nowait(proc)
        except queue.Full:
            rgb_proc_q.get_nowait()
            rgb_proc_q.put_nowait(proc)


def depth_loop():
    while True:
        try:
            rgb, = rgb_proc_q.get(timeout=0.01),
        except queue.Empty:
            continue

        depth = predict_depth(rgb)  # float32 [0..1]
        try:
            depth_q.put_nowait((rgb, depth))
        except queue.Full:
            depth_q.get_nowait()
            depth_q.put_nowait((rgb, depth))


def main():
    # start the capture / process / depth threads
    threading.Thread(target=capture_loop,   daemon=True).start()
    threading.Thread(target=processing_loop,daemon=True).start()
    threading.Thread(target=depth_loop,     daemon=True).start()

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
            # block until new frame is ready
            rgb, depth = depth_q.get()

            # build a CPU SBS uint8 frame and encode to JPEG
            sbs = MJPEGStreamer.make_sbs(
                rgb,
                depth,
                ipd_uv=IPD_UV,
                depth_strength=DEPTH_STRENGTH
            )
            jpg = streamer.encode_jpeg(sbs)

            # push into the HTTP MJPEG server
            streamer.set_frame(jpg)

            # throttle to STREAM_FPS
            time.sleep(1.0 / STREAM_FPS)

    except KeyboardInterrupt:
        print("\n[Main] Shutting down…")
    finally:
        streamer.stop()


if __name__ == "__main__":
    main()

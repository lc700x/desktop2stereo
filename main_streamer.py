# main.py
import threading
import queue
import time

from capture import DesktopGrabber
from depth import predict_depth, DEVICE_INFO
from streaming import MJPEGStreamer

# Configuration
MONITOR_INDEX   = 1      # which monitor to grab
DOWNSCALE_FACTOR = 0.5   # resolution scaling for performance

STREAM_HOST      = "0.0.0.0"
STREAM_PORT      = 1303
STREAM_FPS       = 60
STREAM_QUALITY   = 80

# stereo-warp parameters (match whatever you used in your GLSL)
IPD_UV          = 0.064
DEPTH_STRENGTH  = 0.1
interval = 1.0 / STREAM_FPS
# Queues for pipelining
rgb_raw_q  = queue.Queue(maxsize=3)
rgb_proc_q = queue.Queue(maxsize=3)
depth_q = queue.Queue(maxsize=3)


def capture_loop():
    grabber = DesktopGrabber(
        monitor_index=MONITOR_INDEX,
        downscale=DOWNSCALE_FACTOR
    )
    while True:
        frame = grabber.grab()
        try:
            rgb_raw_q.put_nowait(frame)
        except queue.Full:
            rgb_raw_q.get_nowait()
            rgb_raw_q.put_nowait(frame)
        time.sleep(interval)


def processing_loop():
    cap = DesktopGrabber(monitor_index=MONITOR_INDEX, downscale=DOWNSCALE_FACTOR, show_monitor_info=False)
    while True:
        try:
            frame_raw = rgb_raw_q.get(block=True)
            frame = cap.process(frame_raw)  # Convert and downscale RGB frame
            try:
                rgb_proc_q.put(frame, block=False)
            except queue.Full:
                rgb_proc_q.get_nowait()
                rgb_proc_q.put(frame)
        except queue.Empty:
            time.sleep(interval)

def depth_loop():
    while True:
        try:
            frame_rgb = rgb_proc_q.get(block=True)
            depth = predict_depth(frame_rgb)
            try:
                depth_q.put((frame_rgb, depth), block=False)
            except queue.Full:
                depth_q.get_nowait()
                depth_q.put((frame_rgb, depth))
        except queue.Empty:
            time.sleep(interval)


def main():
    print(DEVICE_INFO)
    # start the capture / process / depth threads
    threading.Thread(target=capture_loop, daemon=True).start()
    threading.Thread(target=processing_loop,daemon=True).start()
    threading.Thread(target=depth_loop, daemon=True).start()

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
            try:
                rgb, depth = depth_q.get()

                # build a CPU SBS uint8 frame and encode to JPEG
                sbs = MJPEGStreamer.make_sbs(
                    rgb,
                    depth,
                    ipd_uv=IPD_UV,
                    depth_strength=DEPTH_STRENGTH
                )
                jpg = streamer.encode_jpeg(sbs)
            except queue.Empty:
                # no new frame available, wait a bit
                time.sleep(interval)
                continue
            # push into the HTTP MJPEG server
            streamer.set_frame(jpg)

            # throttle to STREAM_FPS
            time.sleep(interval)

    except KeyboardInterrupt:
        print("\n[Main] Shutting down…")
    finally:
        streamer.stop()


if __name__ == "__main__":
    main()

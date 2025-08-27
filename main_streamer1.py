import threading
import queue
import time
from utils import OUTPUT_RESOLUTION, DISPLAY_MODE, IPD, FPS, DEPTH_STRENTH, SHOW_FPS, STREAM_PORT
from capture import DesktopGrabber
from depth import predict_depth_tensor, process, make_sbs_tensor
from streamer import MJPEGStreamer

# Streamer settings
TIME_SLEEP = round(1.0 / FPS, 2)
STREAM_QUALITY = 100
STREAM_HOST = "0.0.0.0"

# Queues with size=1 (latest-frame-only logic)
raw_q = queue.Queue(maxsize=1)
proc_q = queue.Queue(maxsize=1)
depth_q = queue.Queue(maxsize=1)
sbs_q = queue.Queue(maxsize=1)  # new queue for SBS processing

def put_latest(q, item):
    if q.full():
        try:
            q.get_nowait()
        except queue.Empty:
            pass
    try:
        q.put_nowait(item)
    except queue.Full:
        time.sleep(TIME_SLEEP)

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
            frame_raw, size = raw_q.get(timeout=TIME_SLEEP)
        except queue.Empty:
            continue
        frame_rgb = process(frame_raw, size)
        put_latest(proc_q, frame_rgb)

def depth_loop():
    while True:
        try:
            frame_rgb = proc_q.get(timeout=TIME_SLEEP)
        except queue.Empty:
            continue
        depth, rgb = predict_depth_tensor(frame_rgb)
        put_latest(depth_q, (rgb, depth))

def sbs_loop(streamer):
    frame_count = 0
    last_time = time.perf_counter()

    while True:
        try:
            rgb, depth = sbs_q.get(timeout=TIME_SLEEP)
        except queue.Empty:
            continue
        
        sbs = make_sbs_tensor(rgb, depth, ipd_uv=IPD, depth_strength=DEPTH_STRENTH, display_mode=DISPLAY_MODE)
        jpg = streamer.encode_jpeg(sbs)
        streamer.set_frame(jpg)

        if SHOW_FPS:
            frame_count += 1
            current_time = time.perf_counter()
            if current_time - last_time >= 1.0:
                current_fps = frame_count / (current_time - last_time)
                frame_count = 0
                last_time = current_time
                print(f"FPS: {current_fps:.2f}")

def main():
    # Start capture and processing threads
    threading.Thread(target=capture_loop, daemon=True).start()
    threading.Thread(target=process_loop, daemon=True).start()
    threading.Thread(target=depth_loop, daemon=True).start()

    # start MJPEG streamer
    streamer = MJPEGStreamer(
        host=STREAM_HOST,
        port=STREAM_PORT,
        fps=FPS,
        quality=STREAM_QUALITY
    )
    streamer.start()

    # Start SBS processing thread(s)
    threading.Thread(target=sbs_loop, args=(streamer,), daemon=True).start()

    try:
        while True:
            try:
                # Forward latest (rgb, depth) to SBS processing queue
                data = depth_q.get(timeout=TIME_SLEEP)
                put_latest(sbs_q, data)
            except queue.Empty:
                pass
    except KeyboardInterrupt:
        print("\n[Main] Shutting down…")
    finally:
        streamer.stop()

if __name__ == "__main__":
    main()

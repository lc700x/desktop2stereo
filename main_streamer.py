import threading
import queue
import time
from utils import OUTPUT_RESOLUTION, DISPLAY_MODE, IPD, FPS, DEPTH_STRENTH, SHOW_FPS, STREAM_PORT
from capture import DesktopGrabber
from depth import predict_depth_tensor, process, make_sbs
from streamer import WebRTCStreamer
import asyncio

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
    """Put item into queue, dropping old one if needed (non-blocking)."""
    if q.full():
        try:
            q.get_nowait()
        except queue.Empty:
            pass
    try:
        q.put_nowait(item)
    except queue.Full:
        pass  # drop if race condition

def capture_loop():
    cap = DesktopGrabber(output_resolution=OUTPUT_RESOLUTION, fps=FPS)
    while True:
        try:
            frame_raw, size = cap.grab()
        except queue.Empty:
            continue
        except Exception:
            continue
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
        
def main():
    threading.Thread(target=capture_loop, daemon=True).start()
    threading.Thread(target=process_loop, daemon=True).start()
    threading.Thread(target=depth_loop, daemon=True).start()

    stream_queue = queue.Queue(maxsize=1)
    streamer = WebRTCStreamer(stream_queue)

    async def async_main():
        # Start the streamer server
        asyncio.create_task(streamer.start())

        # Push SBS frames into queue
        while True:
            try:
                rgb, depth = depth_q.get(timeout=TIME_SLEEP)
                sbs = make_sbs(
                    rgb,
                    depth,
                    ipd_uv=IPD,
                    depth_strength=DEPTH_STRENTH,
                    display_mode=DISPLAY_MODE
                )

                # Only push if the previous frame has been consumed
                if stream_queue.empty():
                    stream_queue.put_nowait(sbs)

                # Throttle to target FPS
                await asyncio.sleep(TIME_SLEEP)

            except queue.Empty:
                await asyncio.sleep(TIME_SLEEP)

    asyncio.run(async_main())

if __name__ == "__main__":
    main()

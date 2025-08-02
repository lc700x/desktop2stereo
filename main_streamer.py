# main.py
import threading
import queue
import time
from typing import Tuple

import numpy as np
from capture import DesktopGrabber
from depth import predict_depth, DEVICE_INFO
from streaming import MJPEGStreamer

# Configuration
MONITOR_INDEX = 1         # which monitor to grab
DOWNSCALE_FACTOR = 1    # resolution scaling for performance

STREAM_HOST = "0.0.0.0"
STREAM_PORT = 1303
STREAM_FPS = 60
STREAM_QUALITY = 80

# stereo-warp parameters (match whatever you used in your GLSL)
IPD_UV = 0.064
DEPTH_STRENGTH = 0.1
FRAME_INTERVAL = 1.0 / STREAM_FPS

# Queues for pipelining (reduced maxsize for lower latency)
rgb_raw_q = queue.Queue(maxsize=2)
rgb_proc_q = queue.Queue(maxsize=2)
depth_q = queue.Queue(maxsize=2)


def capture_loop():
    """Continuously capture desktop frames and put them in the raw queue."""
    grabber = DesktopGrabber(
        monitor_index=MONITOR_INDEX,
        downscale=DOWNSCALE_FACTOR
    )
    
    while True:
        frame = grabber.grab()
        # Non-blocking queue management
        if rgb_raw_q.full():
            rgb_raw_q.get_nowait()
        rgb_raw_q.put_nowait(frame)
        time.sleep(FRAME_INTERVAL)


def processing_loop():
    """Process raw frames and prepare them for depth prediction."""
    cap = DesktopGrabber(
        monitor_index=MONITOR_INDEX, 
        downscale=DOWNSCALE_FACTOR, 
        show_monitor_info=False
    )
    
    while True:
        try:
            frame_raw = rgb_raw_q.get(block=True, timeout=FRAME_INTERVAL)
            frame = cap.process(frame_raw)
            
            if rgb_proc_q.full():
                rgb_proc_q.get_nowait()
            rgb_proc_q.put_nowait(frame)
                
        except queue.Empty:
            continue


def depth_loop():
    """Predict depth for processed frames and enqueue results."""
    while True:
        try:
            frame_rgb = rgb_proc_q.get(block=True, timeout=FRAME_INTERVAL)
            depth = predict_depth(frame_rgb)
            
            if depth_q.full():
                depth_q.get_nowait()
            depth_q.put_nowait((frame_rgb, depth))
                
        except queue.Empty:
            continue


def main():
    print(DEVICE_INFO)
    
    # Start worker threads
    threads = [
        threading.Thread(target=capture_loop, daemon=True),
        threading.Thread(target=processing_loop, daemon=True),
        threading.Thread(target=depth_loop, daemon=True)
    ]
    
    for t in threads:
        t.start()

    # Initialize streamer
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
                rgb, depth = depth_q.get(block=True, timeout=FRAME_INTERVAL)
                
                # Build and encode frame
                sbs = MJPEGStreamer.make_sbs(
                    rgb,
                    depth,
                    ipd_uv=IPD_UV,
                    depth_strength=DEPTH_STRENGTH
                )
                streamer.set_frame(streamer.encode_jpeg(sbs))
                
            except queue.Empty:
                # No new frame, skip this iteration
                continue

    except KeyboardInterrupt:
        print("\n[Main] Shutting down...")
    finally:
        streamer.stop()


if __name__ == "__main__":
    main()
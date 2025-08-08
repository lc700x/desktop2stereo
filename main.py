import threading
import queue
import glfw
import os, sys, time
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from capture import DesktopGrabber
from depth import settings, predict_depth, process, DEVICE_INFO
from viewer import StereoWindow

MONITOR_INDEX, DOWNSCALE_FACTOR, FPS = settings["monitor_index"], settings["downscale_factor"], settings["fps"]
DOWNLOAD_CACHE = settings["download_path"]
TIME_SLEEP = 1.0 / FPS

if len(sys.argv) >= 2 and sys.argv[1] == '--hf-mirror':
    os.environ['HF_ENDPOINT'] = settings["hf_endpoint"]

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
            time.sleep(TIME_SLEEP)
    try:
        q.put_nowait(item)
    except queue.Full:
        time.sleep(TIME_SLEEP)  # Drop frame if race condition occurs

def capture_loop():
    cap = DesktopGrabber(monitor_index=MONITOR_INDEX, downscale=DOWNSCALE_FACTOR, fps=FPS)
    while True:
        frame_raw, size = cap.grab()
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
        depth = predict_depth(frame_rgb)
        put_latest(depth_q, (frame_rgb, depth))

def main():
    print(DEVICE_INFO)

    # Start capture and processing threads
    threading.Thread(target=capture_loop, daemon=True).start()
    threading.Thread(target=process_loop, daemon=True).start()
    threading.Thread(target=depth_loop, daemon=True).start()

    window = StereoWindow()

    frame_rgb, depth = None, None

    while not glfw.window_should_close(window.window):
        try:
            # Get latest frame, or skip update
            frame_rgb, depth = depth_q.get_nowait()
            depth = depth.cpu().numpy().astype('float32')
            window.update_frame(frame_rgb, depth)
        except queue.Empty:
            time.sleep(TIME_SLEEP)  # Reuse previous frame if none available

        window.render()
        glfw.swap_buffers(window.window)
        glfw.poll_events()

    glfw.terminate()

if __name__ == "__main__":
    main()

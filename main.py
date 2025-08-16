import threading
import queue
import glfw
import time

from capture import DesktopGrabber
from depth import settings, predict_depth, process, DEVICE_INFO, MODEL_ID
from viewer import StereoWindow

# Settings
MONITOR_INDEX = settings["Monitor Index"]
OUTPUT_RESOLUTION = settings["Output Resolution"]
DISPLAY_MODE = settings["Display Mode"]
SHOW_FPS = settings["Show FPS"]
FPS_TARGET = settings["FPS"]           # used only for the capture device
DEPTH_STRENGTH = settings["Depth Strength"]

# Timing sleeps
IDLE_SLEEP = 0.002         # main loop sleep when no new frame (2 ms)
WORKER_TIMEOUT = 0.5       # worker queue get timeout (seconds)

# Queues with size=1 (latest-frame-only)
raw_q = queue.Queue(maxsize=1)
proc_q = queue.Queue(maxsize=1)
depth_q = queue.Queue(maxsize=1)

def put_latest(q, item):
    """Keep only the newest item in the queue (drop oldest if full)."""
    if q.full():
        try:
            q.get_nowait()
        except queue.Empty:
            pass
    try:
        q.put_nowait(item)
    except queue.Full:
        pass

def capture_loop():
    cap = DesktopGrabber(monitor_index=MONITOR_INDEX,
                         output_resolution=OUTPUT_RESOLUTION,
                         fps=FPS_TARGET)
    while True:
        frame_raw, size = cap.grab()
        # Optionally attach a timestamp if you want to drop stale frames later:
        # put_latest(raw_q, (frame_raw, size, time.perf_counter()))
        put_latest(raw_q, (frame_raw, size))

def process_loop():
    while True:
        try:
            frame_raw, size = raw_q.get(timeout=WORKER_TIMEOUT)
        except queue.Empty:
            continue
        frame_rgb = process(frame_raw, size)
        put_latest(proc_q, frame_rgb)

def depth_loop():
    while True:
        try:
            frame_rgb = proc_q.get(timeout=WORKER_TIMEOUT)
        except queue.Empty:
            continue
        depth = predict_depth(frame_rgb)
        put_latest(depth_q, (frame_rgb, depth))

def main():
    print(f"Using {DEVICE_INFO}")
    print(f"Model: {MODEL_ID}")

    # Start threads
    threading.Thread(target=capture_loop, daemon=True).start()
    threading.Thread(target=process_loop, daemon=True).start()
    threading.Thread(target=depth_loop, daemon=True).start()

    window = StereoWindow(depth_ratio=DEPTH_STRENGTH, display_mode=DISPLAY_MODE)
    frame_rgb, depth = None, None

    # FPS tracking (use high-res timer)
    frame_count = 0
    fps = 0.0
    last_fps_time = time.perf_counter()
    fps_update_interval = 0.5  # seconds

    while not glfw.window_should_close(window.window):
        # Always try to grab the newest processed depth pair (non-blocking)
        try:
            # If you used timestamps, you can pop repeatedly until newest,
            # but with queue maxsize=1 this always yields the latest available.
            frame_rgb, depth = depth_q.get_nowait()
            have_new_frame = True
        except queue.Empty:
            have_new_frame = False

        # Update the window with the frame (if any). If have_new_frame is False, previous frame remains.
        if have_new_frame:
            window.update_frame(frame_rgb, depth)

        # Render & present (count rendered frames after swap_buffers)
        window.render()
        glfw.swap_buffers(window.window)
        glfw.poll_events()

        # FPS accounting -- count actual rendered frames
        if SHOW_FPS and have_new_frame:
            frame_count += 1
            now = time.perf_counter()
            elapsed = now - last_fps_time
            if elapsed >= fps_update_interval:
                fps = frame_count / elapsed
                frame_count = 0
                last_fps_time = now
                glfw.set_window_title(window.window, f"Stereo Viewer | depth: {window.depth_ratio:.1f} | FPS: {fps:.1f}")

        # If there was no new input to display, sleep a tiny bit to avoid busy loop
        if not have_new_frame:
            time.sleep(IDLE_SLEEP)

    glfw.terminate()

if __name__ == "__main__":
    main()

import threading
import queue
import glfw
import time

from capture import DesktopGrabber
from depth import settings, predict_depth, process, DEVICE_INFO, MODEL_ID
from viewer import StereoWindow

MONITOR_INDEX, OUTPUT_RESOLUTION, DISPLAY_MODE = settings["Monitor Index"], settings["Output Resolution"], settings["Display Mode"]
SHOW_FPS, FPS, DEPTH_STRENTH = settings["Show FPS"], settings["FPS"], settings["Depth Strength"]
TIME_SLEEP = 1.0 / FPS

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
    cap = DesktopGrabber(monitor_index=MONITOR_INDEX, output_resolution=OUTPUT_RESOLUTION, fps=FPS)
    while True:
        frame_raw, size = cap.grab()
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
        depth = predict_depth(frame_rgb)
        put_latest(depth_q, (frame_rgb, depth))

def main():
    print(f"Using {DEVICE_INFO}")
    print(f"Model: {MODEL_ID}")

    # Start capture and processing threads
    threading.Thread(target=capture_loop, daemon=True).start()
    threading.Thread(target=process_loop, daemon=True).start()
    threading.Thread(target=depth_loop, daemon=True).start()

    window = StereoWindow(depth_ratio=DEPTH_STRENTH, display_mode=DISPLAY_MODE)
    frame_rgb, depth = None, None

    # FPS calculation variables
    frame_count = 0
    last_time = time.time()
    fps = 0
    
    while not glfw.window_should_close(window.window):
        
        try:
            # Get latest frame, or skip update
            frame_rgb, depth = depth_q.get_nowait()
            window.update_frame(frame_rgb, depth)
            if SHOW_FPS:
                frame_count += 1
                current_time = time.time()
                if current_time - last_time >= 1.0:  # Update every second
                    fps = frame_count / (current_time - last_time)
                    frame_count = 0
                    last_time = current_time
                    # Update window title with Depth Strength and FPS
                    glfw.set_window_title(window.window, f"Stereo Viewer | FPS: {fps:.1f} | depth: {window.depth_ratio:.1f}")
        except queue.Empty:
            pass  # Reuse previous frame if none available

        window.render()
        glfw.swap_buffers(window.window)
        glfw.poll_events()

    glfw.terminate()

if __name__ == "__main__":
    main()


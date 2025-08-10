import threading
import queue
import glfw
import time

from capture import DesktopGrabber
from depth import settings, predict_depth, process, DEVICE_INFO, MODEL_ID
from viewer import StereoWindow

MONITOR_INDEX, OUTPUT_RESOLUTION = settings["Monitor Index"], settings["Output Resolution"]
SHOW_FPS, FPS, DEPTH_STRENTH = settings["Show FPS"], settings["FPS"], settings["Depth Strength"]
TIME_SLEEP = 1.0 / FPS

# Queues with size=1 (latest-frame-only logic)
raw_q = queue.Queue(maxsize=1)
proc_q = queue.Queue(maxsize=1)

def put_latest(q, item):
    """Put item into queue, dropping oldest if queue is full (non-blocking)."""
    while True:
        try:
            q.put_nowait(item)
            break  # Success, exit loop
        except queue.Full:
            try:
                q.get_nowait()  # Drop oldest item
            except queue.Empty:
                # Queue empty unexpectedly, retry putting
                continue

def capture_loop():
    cap = DesktopGrabber(monitor_index=MONITOR_INDEX, output_resolution=OUTPUT_RESOLUTION, fps=FPS)
    while True:
        try:
            frame_raw, size = cap.grab()
        except queue.Empty:
            time.sleep(TIME_SLEEP)
            continue     
        put_latest(raw_q, (frame_raw, size))

def process_loop():
    while True:
        try:
            frame_raw, size = raw_q.get(timeout=1)  # blocks until available
        except queue.Empty:
            continue
        frame_rgb = process(frame_raw, size)
        depth = predict_depth(frame_rgb)
        put_latest(proc_q, (frame_rgb, depth))


def main():
    print(DEVICE_INFO)
    print(f"Model: {MODEL_ID}")

    # Start capture and processing threads
    threading.Thread(target=capture_loop, daemon=True).start()
    threading.Thread(target=process_loop, daemon=True).start()

    window = StereoWindow(depth_ratio=DEPTH_STRENTH)
    frame_rgb, depth = None, None

    # FPS calculation variables
    frame_count = 0
    last_time = time.time()
    fps = 0
    
    while not glfw.window_should_close(window.window):
        
        if SHOW_FPS:
            frame_count += 1
            current_time = time.time()
            if current_time - last_time >= 1.0:  # Update every second
                fps = frame_count / (current_time - last_time)
                frame_count = 0
                last_time = current_time
                # Update window title with FPS
                glfw.set_window_title(window.window, f"Stereo Viewer | Strength: {window.depth_ratio:.1f} | FPS: {fps:.1f}")
        try:
            frame_rgb, depth = proc_q.get(timeout=TIME_SLEEP)
            window.update_frame(frame_rgb, depth)
        except queue.Empty:
            pass

        window.render()
        glfw.swap_buffers(window.window)
        glfw.wait_events_timeout(TIME_SLEEP)  # sleeps until input or timeout

    glfw.terminate()

if __name__ == "__main__":
    main()
    # import cProfile
    # cProfile.run("main()", sort='cumtime')
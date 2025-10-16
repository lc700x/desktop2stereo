from windows_capture import WindowsCapture, Frame, InternalCaptureControl
import threading
import queue
import glfw
import time
from utils import OUTPUT_RESOLUTION, DISPLAY_MODE, SHOW_FPS, FPS, IPD, DEPTH_STRENGTH, RUN_MODE, STREAM_PORT, STREAM_QUALITY, DML_BOOST
from capture import DesktopGrabber
from depth import process, predict_depth
import cv2

# Use precise frame interval
TIME_SLEEP = 1.0 / FPS

# Queues with size=1 (latest-frame-only logic)
raw_q = queue.Queue(maxsize=1)
proc_q = queue.Queue(maxsize=1)
depth_q = queue.Queue(maxsize=1)

# Set up a stop event
stop_event = threading.Event()

# frame_count = 0
# last_time = time.time()

def capture_loop1():
    cap = DesktopGrabber(output_resolution=OUTPUT_RESOLUTION, fps=FPS)
    while not stop_event.is_set():
        try:
            frame_raw, size = cap.grab()
        except queue.Empty:
            continue
        except Exception:
            continue
        raw_q.put((frame_raw, size))
def capture_loop():
    capture = WindowsCapture(
        cursor_capture=None,
        draw_border=None,
        monitor_index=2,
        window_name=None,
    )

    @capture.event
    def on_frame_arrived(frame: Frame, _capture_control: InternalCaptureControl):
        # global frame_count
        # global last_time
        
        _frame_buffer = frame.frame_buffer
        # cv2.imshow('Wincap Viewer', _frame_buffer)
        # if cv2.waitKey(1) == 27:  # ESC key
        #     exit()
        raw_q.put((cv2.cvtColor(_frame_buffer, cv2.COLOR_BGRA2RGB), OUTPUT_RESOLUTION))
        # frame_count += 1
        # current_time = time.time()
        
        # if current_time - last_time >= 1:
        #     fps = frame_count / (current_time - last_time)
        #     print(f"FPS: {fps:.2f}")
        #     frame_count = 0
        #     last_time = current_time

    @capture.event
    def on_closed():
        print("Capture Session Closed")

    capture.start()

def process_loop():
    while not stop_event.is_set():
        try:
            frame_raw, size = raw_q.get(timeout=TIME_SLEEP)
        except queue.Empty:
            continue
        frame_rgb = process(frame_raw, size)
        proc_q.put(frame_rgb)

def depth_loop():
    while not stop_event.is_set():
        try:
            frame_rgb = proc_q.get(timeout=TIME_SLEEP)
        except queue.Empty:
            continue
        depth = predict_depth(frame_rgb)
        depth_q.put((frame_rgb, depth))

from viewer import StereoWindow
threading.Thread(target=capture_loop, daemon=True).start()
threading.Thread(target=process_loop, daemon=True).start()
threading.Thread(target=depth_loop, daemon=True).start()

frame_count = 0
start_time = time.perf_counter()
last_time = time.perf_counter()
current_fps = None
total_frames = 0

streamer, window = None, None
window = StereoWindow(ipd=IPD, depth_ratio=DEPTH_STRENGTH, display_mode=DISPLAY_MODE, show_fps=SHOW_FPS)
print(f"[Main] Viewer Started")
while not glfw.window_should_close(window.window):
    try:
        rgb, depth = depth_q.get(timeout=TIME_SLEEP)
        window.update_frame(rgb, depth)
        if SHOW_FPS:
            frame_count += 1
            total_frames += 1
            current_time = time.perf_counter()
            if current_time - last_time >= 1.0:
                current_fps = frame_count / (current_time - last_time)
                frame_count = 0
                last_time = current_time
                glfw.set_window_title(window.window, f"Stereo Viewer | FPS: {current_fps:.1f} | Depth: {window.depth_ratio:.1f}")
    except queue.Empty:
        pass

    window.render()
    glfw.swap_buffers(window.window)
    glfw.poll_events()

glfw.terminate()
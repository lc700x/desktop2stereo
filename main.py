# main.py
import threading
import queue
import glfw
import os, sys
from capture import DesktopGrabber
from depth import predict_depth, process, DEVICE_INFO
from viewer import StereoWindow

# Config
MONITOR_INDEX = 1
DOWNSCALE_FACTOR = 0.5  # 0.5 for better performance

# Optional HuggingFace mirror
if len(sys.argv) >= 2 and sys.argv[1] == '--hf-mirror':
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# Queues
raw_q = queue.Queue(maxsize=3)
proc_q = queue.Queue(maxsize=3)
depth_q = queue.Queue(maxsize=3)

def capture_loop():
    cap = DesktopGrabber(monitor_index=MONITOR_INDEX, downscale=DOWNSCALE_FACTOR)
    while True:
        frame_raw = cap.grab()
        try:
            raw_q.put(frame_raw, block=False)
        except queue.Full:
            try:
                raw_q.get_nowait()
            except queue.Empty:
                pass
            raw_q.put(frame_raw, block=False)

def process_loop():
    while True:
        try:
            frame_raw = raw_q.get(timeout=0.1)
        except queue.Empty:
            continue

        frame_rgb = process(frame_raw, downscale=DOWNSCALE_FACTOR)

        try:
            proc_q.put(frame_rgb, block=False)
        except queue.Full:
            try:
                proc_q.get()
            except queue.Empty:
                pass
            proc_q.put(frame_rgb, block=False)

def depth_loop():
    while True:
        try:
            frame_rgb = proc_q.get(timeout=0.1)
        except queue.Empty:
            continue

        depth = predict_depth(frame_rgb)

        try:
            depth_q.put((frame_rgb, depth), block=False)
        except queue.Full:
            try:
                depth_q.get()
            except queue.Empty:
                pass
            depth_q.put((frame_rgb, depth), block=False)

def main():
    print(DEVICE_INFO)

    # Start threads
    threading.Thread(target=capture_loop, daemon=True).start()
    threading.Thread(target=process_loop, daemon=True).start()
    threading.Thread(target=depth_loop, daemon=True).start()

    window = StereoWindow()

    while not glfw.window_should_close(window.window):
        try:
            frame_rgb, depth = depth_q.get(timeout=0.1)
            window.update_frame(frame_rgb, depth)
        except queue.Empty:
            pass

        window.render()
        glfw.swap_buffers(window.window)
        glfw.poll_events()

    glfw.terminate()

if __name__ == "__main__":
    main()

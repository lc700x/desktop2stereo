# main.py
import threading
import queue
import glfw
import os, sys
# diable hf warning
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"]="1"
from capture import DesktopGrabber
from depth import settings, predict_depth, process, DEVICE_INFO
from viewer import StereoWindow

# Set the monitor index and downscale factor
MONITOR_INDEX, DOWNSCALE_FACTOR = settings["monitor_index"], settings["downscale_factor"]

# set download path
DOWNLOAD_CACHE = settings["download_path"]

# Optional HuggingFace mirror
if len(sys.argv) >= 2 and sys.argv[1] == '--hf-mirror':
    os.environ['HF_ENDPOINT'] = settings["hf_endpoint"]

# Queues
raw_q = queue.Queue(maxsize=3)
proc_q = queue.Queue(maxsize=3)
depth_q = queue.Queue(maxsize=3)

def capture_loop():
    cap = DesktopGrabber(monitor_index=MONITOR_INDEX, downscale=DOWNSCALE_FACTOR)
    while True:
        frame_raw, size = cap.grab()
        try:
            raw_q.put((frame_raw, size), block=False)
        except queue.Full:
            try:
                raw_q.get()
            except queue.Empty:
                pass
            raw_q.put((frame_raw, size), block=False)

def process_loop():
    while True:
        try:
            frame_raw, size = raw_q.get(timeout=0.1)
        except queue.Empty:
            continue

        frame_rgb = process(frame_raw, size)

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
            frame_rgb, depth = depth_q.get_nowait()
            window.update_frame(frame_rgb, depth)
        except queue.Empty:
            pass

        window.render()
        glfw.swap_buffers(window.window)
        glfw.poll_events()

    glfw.terminate()

if __name__ == "__main__":
    main()

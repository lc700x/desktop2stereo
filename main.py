# main.py
import threading
import queue
import glfw
from capture import DesktopGrabber
from depth import predict_depth
from viewer import StereoWindow
import time, os, sys

# Set the monitor index and downscale factor
MONITOR_INDEX = 1  # Change to 0 for all monitors, 1 for primary monitor, ...
DOWNSCALE_FACTOR = 1.0 # Set to 1.0 for no downscaling

# add arg for setting hg mirror if cannot access Hugging Face directly
if len(sys.argv) >= 2 and sys.argv[1] == '--hf-mirror':
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

rgb_raw_q = queue.Queue(maxsize=4)
rgb_proc_q = queue.Queue(maxsize=4)
depth_q = queue.Queue(maxsize=4)

def capture_loop():
    cap = DesktopGrabber(monitor_index=MONITOR_INDEX, downscale=DOWNSCALE_FACTOR)
    while True:
        frame_raw = cap.grab()  # raw BGR frame
        try:
            rgb_raw_q.put(frame_raw, block=False)
        except queue.Full:
            rgb_raw_q.get_nowait()  # Drop oldest raw frame
            rgb_raw_q.put(frame_raw)

def processing_loop():
    cap = DesktopGrabber(monitor_index=MONITOR_INDEX, downscale=DOWNSCALE_FACTOR, show_monitor_info=False)
    while True:
        try:
            frame_raw = rgb_raw_q.get(block=True)
            frame = cap.process(frame_raw)  # Convert and downscale RGB frame
            try:
                rgb_proc_q.put(frame, block=False)
            except queue.Full:
                rgb_proc_q.get_nowait()
                rgb_proc_q.put(frame)
        except queue.Empty:
            time.sleep(0.001)

def depth_loop():
    while True:
        try:
            frame_rgb = rgb_proc_q.get(block=True)
            depth = predict_depth(frame_rgb)
            try:
                depth_q.put((frame_rgb, depth), block=False)
            except queue.Full:
                depth_q.get_nowait()
                depth_q.put((frame_rgb, depth))
        except queue.Empty:
            time.sleep(0.001)

def main():
    threading.Thread(target=capture_loop, daemon=True).start()
    threading.Thread(target=processing_loop, daemon=True).start()
    threading.Thread(target=depth_loop, daemon=True).start()

    window = StereoWindow()

    while not glfw.window_should_close(window.window):
        try:
            rgb, depth = depth_q.get_nowait()
            window.update_frame(rgb, depth)
        except queue.Empty:
            pass

        window.render()
        glfw.swap_buffers(window.window)
        glfw.poll_events()

    glfw.terminate()

if __name__ == "__main__":
    main()
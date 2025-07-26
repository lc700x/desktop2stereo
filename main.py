import threading
import queue
import glfw
from capture import DesktopGrabber
from depth import predict_depth
from viewer import StereoWindow
import time

# add arg for setting hg mirror if cannot access Hugging Face directly
import os, sys
if len(sys.argv) >= 2 and sys.argv[1] == '--hf-mirror':
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# Queue setup
rgb_q = queue.Queue(maxsize=1)
depth_q = queue.Queue(maxsize=1)

def capture_loop():
    cap = DesktopGrabber(monitor_index=1, downscale=0.5) # Default Monitor: 1 for primary, Adjust downscale as needed
    while True:
        frame = cap.grab()
        # Drop any old frame
        while not rgb_q.empty():
            try:
                rgb_q.get_nowait()
            except queue.Empty:
                break
        rgb_q.put(frame)

def depth_loop():
    while True:
        try:
            frame = rgb_q.get(block=True, timeout=0.1)
            depth = predict_depth(frame)
            try:
                depth_q.put((frame, depth), block=False)
            except queue.Full:
                pass
        except queue.Empty:
            time.sleep(0.001)

def main():
    # Start capture threads
    threading.Thread(target=capture_loop, daemon=True).start()
    threading.Thread(target=depth_loop, daemon=True).start()
    
    # Create and run window
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
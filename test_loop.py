import cv2, time
from time import sleep
import threading
import queue
import glfw
import time
from utils import OUTPUT_RESOLUTION, DISPLAY_MODE, SHOW_FPS, FPS, IPD, DEPTH_STRENGTH, RUN_MODE, STREAM_PORT, STREAM_QUALITY, DML_BOOST
from capture import DesktopGrabber
from depth import process, predict_depth

from viewer import StereoWindow
# def main():
#     cap = DesktopGrabber(
#         output_resolution=OUTPUT_RESOLUTION,   # scale height
#         fps=FPS
#     )
#     frame_count = 0
#     last_time = time.perf_counter()
#     current_fps = None
#     total_frames = 0
#     while True:
#         frame_raw, size = cap.grab()
#         if frame_raw is None:
#             print("No frame captured")
#             sleep(0.1)
#             continue
#         frame_rgb = process(frame_raw, size)
#         depth = predict_depth(frame_rgb)
#         # cv2.imshow("Capture Test", frame)
        
#         if SHOW_FPS:
#             frame_count += 1
#             total_frames += 1
#             current_time = time.perf_counter()
#             if current_time - last_time >= 1.0:
#                 current_fps = frame_count / (current_time - last_time)
#                 frame_count = 0
#                 last_time = current_time
#                 print(current_fps)
#         # key = cv2.waitKey(1) & 0xFF
#         # if key == 27:  # ESC key to exit
#         #     break

#     # cv2.destroyAllWindows()

# if __name__ == "__main__":
#     main()



# === MAIN LOGIC ===
def main():
    cap = DesktopGrabber(output_resolution=OUTPUT_RESOLUTION, fps=FPS)

    raw_q = queue.Queue(maxsize=1)     # Limit to avoid memory buildup
    proc_q = queue.Queue(maxsize=1)
    depth_q = queue.Queue(maxsize=1)
    stop_event = threading.Event()

    # === Thread 1: Capture ===
    def capture_loop():
        while not stop_event.is_set():
            try:
                frame_raw, size = cap.grab()
            except queue.Empty:
                continue
            except Exception:
                continue
            raw_q.put((frame_raw, size))

    # === Thread 2: Processing ===
    def process_loop():
        while not stop_event.is_set():
            try:
                frame_raw, size = raw_q.get(timeout=0.1)
            except queue.Empty:
                continue
            frame_rgb = process(frame_raw, size)
            try:
                proc_q.put(frame_rgb)
            except queue.Full:
                pass
            raw_q.task_done()

    def depth_loop():
        while not stop_event.is_set():
            try:
                frame_raw, size = raw_q.get(timeout=0.1)
            except queue.Empty:
                continue
            frame_rgb = process(frame_raw, size)
            try:
                proc_q.put(frame_rgb)
            except queue.Full:
                pass
            raw_q.task_done()
    
    # === Thread 3: Inference ===
    def viewer_loop():
        frame_count = 0
        last_time = time.perf_counter()
        total_frames = 0

        window = StereoWindow(ipd=IPD, depth_ratio=DEPTH_STRENGTH, display_mode=DISPLAY_MODE, show_fps=SHOW_FPS)
        print(f"[Main] Viewer Started")
        while not glfw.window_should_close(window.window):
            try:
                rgb, depth = depth_q.get(timeout=0.1)
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
                continue
            except queue.Full:
                pass
            depth_q.task_done()

            window.render()
            glfw.swap_buffers(window.window)
            glfw.poll_events()

        glfw.terminate()

    # === Thread startup ===
    threads = [
        threading.Thread(target=capture_loop, daemon=True),
        threading.Thread(target=process_loop, daemon=True),
        threading.Thread(target=depth_loop, daemon=True),
    ]
    for t in threads:
        t.start()

    try:
        while True:
            viewer_loop()
    except KeyboardInterrupt:
        print("\nStopping...")
        stop_event.set()
        cap.stop()
        for t in threads:
            t.join(timeout=1.0)
        print("Stopped cleanly.")

if __name__ == "__main__":
    main()

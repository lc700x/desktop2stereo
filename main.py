import threading
import queue
import glfw
import time
from utils import OUTPUT_RESOLUTION, DISPLAY_MODE, SHOW_FPS, FPS, IPD, DEPTH_STRENTH, RUN_MODE, STREAM_PORT
from capture import DesktopGrabber
from depth import process

# Use precise frame interval
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
        pass  # drop if race condition

def capture_loop():
    cap = DesktopGrabber(output_resolution=OUTPUT_RESOLUTION, fps=FPS)
    while True:
        try:
            frame_raw, size = cap.grab()
            put_latest(raw_q, (frame_raw, size))
        except Exception:
            continue

def process_loop():
    while True:
        try:
            frame_raw, size = raw_q.get(timeout=TIME_SLEEP)
            if frame_raw is None:
                continue
            frame_rgb = process(frame_raw, size)
            put_latest(proc_q, frame_rgb)
        except queue.Empty:
            continue

def main(mode="Viewer"):
    threading.Thread(target=capture_loop, daemon=True).start()
    threading.Thread(target=process_loop, daemon=True).start()

    frame_count = 0
    last_time = time.perf_counter()

    total_frames = 0
    start_time = time.perf_counter()

    streamer = None

    try:
        print(f"[Main] {mode} Started")
        if mode == "Viewer":
            from viewer import StereoWindow
            from depth import predict_depth

            def depth_loop():
                while True:
                    try:
                        frame_rgb = proc_q.get(timeout=TIME_SLEEP)
                        depth = predict_depth(frame_rgb)
                        put_latest(depth_q, (frame_rgb, depth))
                    except queue.Empty:
                        continue

            threading.Thread(target=depth_loop, daemon=True).start()
            window = StereoWindow(ipd=IPD, depth_ratio=DEPTH_STRENTH, display_mode=DISPLAY_MODE)

            while not glfw.window_should_close(window.window):
                try:
                    rgb, depth = depth_q.get_nowait()
                    window.update_frame(rgb, depth)

                    if SHOW_FPS:
                        frame_count += 1
                        total_frames += 1
                        current_time = time.perf_counter()
                        if current_time - last_time >= 1.0:
                            fps = frame_count / (current_time - last_time)
                            frame_count = 0
                            last_time = current_time
                            glfw.set_window_title(window.window, f"Stereo Viewer | FPS: {fps:.1f} | Depth: {window.depth_ratio:.1f}")
                except queue.Empty:
                    pass

                window.render()
                glfw.swap_buffers(window.window)
                glfw.poll_events()

            glfw.terminate()

        else:
            from depth import predict_depth_tensor
            from streamer import MJPEGStreamer

            def depth_loop():
                while True:
                    try:
                        frame_rgb = proc_q.get(timeout=TIME_SLEEP)
                        import cv2
                        cv2.imwrite('output_image.png', frame_rgb)
                        print("Image saved successfully!")
                        break
                        depth, rgb = predict_depth_tensor(frame_rgb)
                        put_latest(depth_q, (rgb, depth))
                    except queue.Empty:
                        continue

            threading.Thread(target=depth_loop, daemon=True).start()

            streamer = MJPEGStreamer(port=STREAM_PORT, fps=FPS, quality=100)
            streamer.start()

            while True:
                try:
                    rgb, depth = depth_q.get(timeout=TIME_SLEEP)
                    streamer.set_frame(rgb, depth, ipd_uv=IPD, depth_strength=DEPTH_STRENTH, display_mode=DISPLAY_MODE)
                    if SHOW_FPS:
                        frame_count += 1
                        current_time = time.perf_counter()
                        if current_time - last_time >= 1.0:
                            current_fps = frame_count / (current_time - last_time)
                            frame_count = 0
                            last_time = current_time
                            print(f"FPS: {current_fps:.2f}")
                except queue.Empty:
                    continue

    except KeyboardInterrupt:
        print("\n[Main] Shutting down…")
    # except Exception as e:
    #     print(e)
    # finally:
    #     if streamer:
    #         streamer.stop()
    #     print(f"[Main] {mode} Stopped")
    #     exit()

if __name__ == "__main__":
    main(mode=RUN_MODE)

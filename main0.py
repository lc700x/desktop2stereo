import threading
import queue
import glfw
import time
from utils import OUTPUT_RESOLUTION, DISPLAY_MODE, SHOW_FPS, FPS, IPD, DEPTH_STRENTH, RUN_MODE, STREAM_PORT
from capture import DesktopGrabber
from depth import process

TIME_SLEEP = round(1.0 / FPS, 2)

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
    cap = DesktopGrabber(output_resolution=OUTPUT_RESOLUTION, fps=FPS)
    while True:
        try:
            frame_raw, size = cap.grab()
        except OSError:
            exit()
        put_latest(raw_q, (frame_raw, size))

def process_loop():
    while True:
        try:
            frame_raw, size = raw_q.get(timeout = TIME_SLEEP)
        except queue.Empty:
            continue
        frame_rgb = process(frame_raw, size)
        put_latest(proc_q, frame_rgb)

def main(mode="Viewer"):
    # Start capture and processing threads
    threading.Thread(target=capture_loop, daemon=True).start()
    threading.Thread(target=process_loop, daemon=True).start()

    # FPS calculation variables
    frame_count = 0
    last_time = time.perf_counter()
    fps = 0

    # Average FPS calculation
    total_frames = 0
    start_time = time.perf_counter()

    streamer = None

    rgb, depth = None, None
    try: 
        print(f"[Main] {mode} Started")
        if mode == "Viewer":
            from viewer import StereoWindow
            from depth import predict_depth
            def depth_loop():
                while True:
                    try:
                        frame_rgb = proc_q.get(timeout = TIME_SLEEP)
                    except queue.Empty:
                        continue
                    depth = predict_depth(frame_rgb)
                    put_latest(depth_q, (frame_rgb, depth))
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
                        if current_time - last_time >= 1.0:  # Update every second
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
            # Streamer settings
            from depth import predict_depth_tensor, make_sbs_tensor
            from streamer import MJPEGStreamer
            def depth_loop():
                while True:
                    try:
                        frame_rgb = proc_q.get(timeout = TIME_SLEEP)
                    except queue.Empty:
                        continue
                    depth, rgb = predict_depth_tensor(frame_rgb)
                    put_latest(depth_q, (rgb, depth))
            threading.Thread(target=depth_loop, daemon=True).start()
            STREAM_QUALITY   = 100
            STREAM_HOST = "0.0.0.0"
            # start MJPEG streamer
            streamer = MJPEGStreamer(
                port=STREAM_PORT,
                fps=FPS,
                quality=STREAM_QUALITY
            )
            streamer.start()

            while True:
                try:
                    rgb, depth = depth_q.get(timeout = TIME_SLEEP)
                    sbs = make_sbs_tensor(rgb, depth, ipd_uv=IPD, depth_strength=DEPTH_STRENTH, display_mode = DISPLAY_MODE)
                    jpg = streamer.encode_jpeg(sbs)
                    # push into the HTTP MJPEG server
                    streamer.set_frame(jpg)
                    if SHOW_FPS:
                        frame_count += 1
                        current_time = time.perf_counter()
                        if current_time - last_time >= 1.0:  # Update every second
                            current_fps = frame_count / (current_time - last_time)
                            frame_count = 0
                            last_time = current_time
                            print(f"FPS: {current_fps:.2f}")
                except queue.Empty:
                        pass
            
    except KeyboardInterrupt:
        print("\n[Main] Shutting down…")
    finally:
        # Print average FPS on exit
        # total_time = time.perf_counter() - start_time
        # avg_fps = total_frames / total_time if total_time > 0 else 0
        # print(f"Average FPS: {avg_fps:.2f}")
        if streamer:
            streamer.stop()
        print(f"[Main] {mode} Stopped")
        exit()

if __name__ == "__main__":
    main(mode=RUN_MODE)


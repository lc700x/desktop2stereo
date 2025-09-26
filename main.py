import threading
import queue
import glfw
import time
from utils import OUTPUT_RESOLUTION, DISPLAY_MODE, SHOW_FPS, FPS, IPD, DEPTH_STRENTH, RUN_MODE, STREAM_PORT, STREAM_QUALITY, DML_STREAM_STABLE
from capture import DesktopGrabber
from depth import process, predict_depth


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
        except queue.Empty:
            continue
        except Exception:
            continue
        put_latest(raw_q, (frame_raw, size))

def process_loop():
    while True:
        try:
            frame_raw, size = raw_q.get(timeout=TIME_SLEEP)
        except queue.Empty:
            continue
        frame_rgb = process(frame_raw, size)
        put_latest(proc_q, frame_rgb)

def main(mode="Viewer"):
    threading.Thread(target=capture_loop, daemon=True).start()
    threading.Thread(target=process_loop, daemon=True).start()

    frame_count = 0
    last_time = time.perf_counter()

    total_frames = 0
    start_time = time.perf_counter()

    streamer, window = None, None

    try:
        if mode == "Viewer":
            from viewer import StereoWindow

            def depth_loop():
                while True:
                    try:
                        frame_rgb = proc_q.get(timeout=TIME_SLEEP)
                    except queue.Empty:
                        continue
                    depth = predict_depth(frame_rgb)
                    put_latest(depth_q, (frame_rgb, depth))

            threading.Thread(target=depth_loop, daemon=True).start()
            window = StereoWindow(ipd=IPD, depth_ratio=DEPTH_STRENTH, display_mode=DISPLAY_MODE, show_fps=SHOW_FPS)
            print(f"[Main] Viewer Started")
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
            from depth import make_sbs, DEVICE_INFO
            BOOST = not (DML_STREAM_STABLE and "DirectML" in DEVICE_INFO)
            from streamer import MJPEGStreamer
            if BOOST:
                def make_output(rgb, depth):
                    return make_sbs(
                        rgb, depth,
                        ipd_uv=IPD,
                        depth_ratio=DEPTH_STRENTH,
                        display_mode=DISPLAY_MODE
                    )
            else:
                sbs_q = queue.Queue(maxsize=1)
                def make_output(rgb, depth):
                    return (rgb, depth)
                
                def sbs_loop():
                    while True:
                        try:
                            rgb, depth = depth_q.get(timeout=TIME_SLEEP)
                        except queue.Empty:
                            continue
                        sbs = make_sbs(rgb, depth, ipd_uv=IPD, depth_ratio=DEPTH_STRENTH, display_mode=DISPLAY_MODE)
                        put_latest(sbs_q, sbs)



            def depth_loop():
                while True:
                    try:
                        frame_rgb = proc_q.get(timeout=TIME_SLEEP)
                    except queue.Empty:
                        continue
                    depth, rgb = predict_depth(frame_rgb, return_tuple=True)
                    put_latest(depth_q, make_output(rgb, depth))
                    

            threading.Thread(target=depth_loop, daemon=True).start()
            if not BOOST:
                threading.Thread(target=sbs_loop, daemon=True).start()
            
            streamer = MJPEGStreamer(port=STREAM_PORT, fps=FPS, quality=STREAM_QUALITY)
            streamer.start()
            print(f"[Main] Streamer Started")
            
            while True:
                try:
                    if BOOST: # Fix for unstable dml runtime error
                        sbs = depth_q.get(timeout=TIME_SLEEP)
                    else:
                        sbs = sbs_q.get(timeout=TIME_SLEEP)
                    streamer.set_frame(sbs)
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

    except Exception as e:
        print(e)
    finally:
        if streamer:
            streamer.stop()
        if window:
            glfw.terminate()
        #     print(f"[Main] {mode} Stopped")
        # total_time = time.perf_counter() - start_time
        # avg_fps = frame_count / total_time if total_time > 0 else 0
        # print(f"Average FPS: {avg_fps:.2f}")

if __name__ == "__main__":
    main(mode=RUN_MODE)

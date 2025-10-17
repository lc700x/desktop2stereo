# main.py
import threading
import queue
import glfw
import time
from utils import OS_NAME, OUTPUT_RESOLUTION, DISPLAY_MODE, CAPTURE_MODE, CAPTURE_TOOL, MONITOR_INDEX, SHOW_FPS, FPS, WINDOW_TITLE, IPD, DEPTH_STRENGTH, RUN_MODE, STREAM_PORT, STREAM_QUALITY, DML_BOOST
from depth import process, predict_depth

# Use precise frame interval
TIME_SLEEP = 1.0 / FPS

# Queues with size=1 (latest-frame-only logic)
raw_q = queue.Queue(maxsize=1)
proc_q = queue.Queue(maxsize=1)
depth_q = queue.Queue(maxsize=1)

# Set up a stop event
stop_event = threading.Event()

# Initialize capture
if CAPTURE_TOOL == "WindowsCapture" and OS_NAME ==  "Windows":
    from windows_capture import WindowsCapture, Frame, InternalCaptureControl
    import cv2, ctypes
    # get windows Hi-DPI scale
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except:
        ctypes.windll.user32.SetProcessDPIAware()
    dwmapi = ctypes.WinDLL("dwmapi")
    cap = WindowsCapture(window_name=WINDOW_TITLE, draw_border=False) if CAPTURE_MODE == "Window" else WindowsCapture(monitor_index=MONITOR_INDEX, draw_border=False)
    
    def capture_loop():
        @cap.event
        def on_frame_arrived(frame: Frame, _capture_control: InternalCaptureControl):
            global capture_control
            tick = time.perf_counter()
            capture_control = _capture_control
            dwmapi.DwmFlush()
            _frame_buffer = frame.frame_buffer
            raw_q.put((cv2.cvtColor(_frame_buffer, cv2.COLOR_BGRA2RGB), OUTPUT_RESOLUTION))
            process_time = time.perf_counter() - tick
            wait_time = max(TIME_SLEEP - process_time, 0)
            time.sleep(wait_time)

        @cap.event
        def on_closed():
            print("Capture Session Closed")

        cap.start()
else:
    # DXCamera based wincam
    from capture import DesktopGrabber
    cap = DesktopGrabber(output_resolution=OUTPUT_RESOLUTION, fps=FPS, window_title=WINDOW_TITLE, capture_mode=CAPTURE_MODE, monitor_index=MONITOR_INDEX)

    def capture_loop():
        while True:
            try:
                frame_raw, size = cap.grab()
            except queue.Empty:
                continue
            except Exception:
                continue
            raw_q.put((frame_raw, size))

def process_loop():
    while True:
        try:
            frame_raw, size = raw_q.get()
        except queue.Empty:
            continue
        frame_rgb = process(frame_raw, size)
        proc_q.put(frame_rgb)

def main(mode="Viewer"):
    threading.Thread(target=capture_loop, daemon=True).start()
    threading.Thread(target=process_loop, daemon=True).start()

    frame_count = 0
    start_time = time.perf_counter()
    last_time = time.perf_counter()
    current_fps = None
    total_frames = 0

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
                    depth_q.put((frame_rgb, depth))

            threading.Thread(target=depth_loop, daemon=True).start()
            window = StereoWindow(ipd=IPD, depth_ratio=DEPTH_STRENGTH, display_mode=DISPLAY_MODE, show_fps=SHOW_FPS)
            print(f"[Main] Viewer Started")
            while not glfw.window_should_close(window.window):
                try:
                    rgb, depth = depth_q.get()
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

        else:
            from depth import make_sbs, DEVICE_INFO
            BOOST = (not "DirectML" in DEVICE_INFO) or DML_BOOST
            from streamer import MJPEGStreamer
            if not BOOST:
                def make_output(rgb, depth):
                    return make_sbs(rgb, depth, ipd_uv=IPD, depth_ratio=DEPTH_STRENGTH, display_mode=DISPLAY_MODE, fps=current_fps)
            else:
                sbs_q = queue.Queue(maxsize=1)
                def make_output(rgb, depth):
                    return (rgb, depth)
                
                def sbs_loop():
                    while True:
                        try:
                            rgb, depth = depth_q.get()
                        except queue.Empty:
                            continue
                        sbs = make_sbs(rgb, depth, ipd_uv=IPD, depth_ratio=DEPTH_STRENGTH, display_mode=DISPLAY_MODE, fps=current_fps)
                        sbs_q.put(sbs)

            def depth_loop():
                while True:
                    try:
                        frame_rgb = proc_q.get()
                    except queue.Empty:
                        continue
                    depth, rgb = predict_depth(frame_rgb, return_tuple=True)
                    depth_q.put(make_output(rgb, depth))
                    

            threading.Thread(target=depth_loop, daemon=True).start()
            if BOOST:
                threading.Thread(target=sbs_loop, daemon=True).start()
            
            streamer = MJPEGStreamer(port=STREAM_PORT, fps=FPS, quality=STREAM_QUALITY)
            streamer.start()
            print(f"[Main] Streamer Started")
            
            while True:
                try:
                    if not BOOST: # Fix for unstable dml runtime error
                        sbs = depth_q.get()
                    else:
                        sbs = sbs_q.get()
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
        try:
            cap.stop()
        except AttributeError:
            # stop for WindowsCapture
            if OS_NAME == "Windows" and CAPTURE_MODE == "WindowsCapture":
                capture_control.stop()
        print(f"[Main] {mode} Stopped")
        # if SHOW_FPS:
        #     total_time = time.perf_counter() - start_time
        #     avg_fps = frame_count / total_time if total_time > 0 else 0
        #     print(f"Average FPS: {avg_fps:.2f}")

if __name__ == "__main__":
    main(mode=RUN_MODE)
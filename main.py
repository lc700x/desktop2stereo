# main.py
import threading
import queue
import glfw
import time
import signal
import sys
import subprocess
import cv2
from utils import OS_NAME, OUTPUT_RESOLUTION, DISPLAY_MODE, CAPTURE_MODE, CAPTURE_TOOL, MONITOR_INDEX, SHOW_FPS, FPS, WINDOW_TITLE, IPD, DEPTH_STRENGTH, RUN_MODE, STREAM_MODE, STREAM_PORT, STREAM_QUALITY, DML_BOOST, STEREOMIX_DEVICE, STREAM_KEY, LOCAL_IP, AUDIO_DELAY, CRF, shutdown_event
from depth import process, predict_depth

# Global process references
global_processes = {
    'ffmpeg': None,
    'rtmp_server': None
}

# Use precise frame interval
TIME_SLEEP = 1.0 / FPS

# Queues with size=1 (latest-frame-only logic)
raw_q = queue.Queue(maxsize=1)
proc_q = queue.Queue(maxsize=1)
depth_q = queue.Queue(maxsize=1)

# Initialize capture
if CAPTURE_TOOL == "WindowsCapture" and OS_NAME == "Windows":
    from windows_capture import WindowsCapture, Frame, InternalCaptureControl
    import ctypes
    from ctypes import wintypes
    import threading
    import time
    
    # optional small delay (seconds) after capture event before performing actions
    CAPTURE_CURSOR_DELAY_S = 0.08

    # Handle Windows Hi-DPI scaling
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)  # Per-monitor DPI awareness
    except Exception:
        ctypes.windll.user32.SetProcessDPIAware()

    # Windows API Setup
    user32 = ctypes.windll.user32
    user32.ShowCursor.argtypes = [wintypes.BOOL]
    user32.ShowCursor.restype = ctypes.c_int
    # Add keybd_event for simulating keyboard input
    user32.keybd_event.argtypes = [ctypes.c_ubyte, ctypes.c_ubyte, wintypes.DWORD, ctypes.c_ulonglong]
    user32.keybd_event.restype = None

    # Virtual key codes
    VK_LWIN = 0x5B  # Left Windows key
    VK_TAB = 0x09   # Tab key
    KEYEVENTF_KEYUP = 0x0002

    def simulate_win_tab():
        """
        Simulate pressing Win+Tab to open Task View, then again to close it.
        Returns True if successful, False otherwise.
        """
        try:
            # First Win+Tab: Open Task View
            user32.keybd_event(VK_LWIN, 0, 0, 0)
            user32.keybd_event(VK_TAB, 0, 0, 0)
            time.sleep(0.01)  # Small delay for key press
            user32.keybd_event(VK_TAB, 0, KEYEVENTF_KEYUP, 0)
            user32.keybd_event(VK_LWIN, 0, KEYEVENTF_KEYUP, 0)
            time.sleep(0.5)  # Delay to allow Task View to open
            # Second Win+Tab: Close Task View
            user32.keybd_event(VK_LWIN, 0, 0, 0)
            user32.keybd_event(VK_TAB, 0, 0, 0)
            time.sleep(0.01)  # Small delay for key press
            user32.keybd_event(VK_TAB, 0, KEYEVENTF_KEYUP, 0)
            user32.keybd_event(VK_LWIN, 0, KEYEVENTF_KEYUP, 0)
            return True
        except Exception as e:
            print(f"[simulate_win_tab] Failed to simulate Win+Tab: {e}")
            return False

    # Waits for capture_started_event to be set, then simulates Win+Tab twice
    capture_started_event = threading.Event()

    def _keyboard():
        """
        Wait for capture_started_event, then simulate Win+Tab to show desktop and restore windows.
        Exits if shutdown_event is set.
        """
        while not shutdown_event.is_set():
            triggered = capture_started_event.wait(timeout=0.1)
            if shutdown_event.is_set():
                break
            if not triggered:
                continue

            try:
                # print("[keyboard] Simulating Win+Tab to show desktop and restore windows...")
                success = simulate_win_tab()

                if CAPTURE_CURSOR_DELAY_S:
                    time.sleep(CAPTURE_CURSOR_DELAY_S)
                
                # if not success:
                    # print("[keyboard] Win+Tab simulation reported failure.")
            except Exception as e:
                print(f"[keyboard] Exception during action: {e}")
            finally:
                break

        # print("[keyboard] Exiting cursor worker thread.")

    # Start worker thread (daemon so it won't block shutdown)
    win_tab_thread = threading.Thread(target=_keyboard, name="CursorWorker", daemon=True)
    win_tab_thread.start()

    # Initialize capture object and capture loop
    cap = (
        WindowsCapture(window_name=WINDOW_TITLE)
        if CAPTURE_MODE == "Window"
        else WindowsCapture(monitor_index=MONITOR_INDEX)
    )

    def capture_loop():
        global capture_control

        @cap.event
        def on_frame_arrived(frame: Frame, capture_control: InternalCaptureControl):
            if shutdown_event.is_set():
                return
            capture_started_event.set()
            try:
                dwmapi = ctypes.WinDLL("dwmapi")
                dwmapi.DwmFlush()
            except Exception:
                pass

            raw_q.put((frame.frame_buffer, OUTPUT_RESOLUTION))

        @cap.event
        def on_closed():
            print("[capture_loop] Capture session closed")

        cap.start()
else:
    # DXCamera based wincam
    from capture import DesktopGrabber
    cap = DesktopGrabber(output_resolution=OUTPUT_RESOLUTION, fps=FPS, window_title=WINDOW_TITLE, capture_mode=CAPTURE_MODE, monitor_index=MONITOR_INDEX)
    
    def capture_loop():
        while not shutdown_event.is_set():
            try:
                frame_raw, size = cap.grab()
                if shutdown_event.is_set():
                    break
                raw_q.put((frame_raw, size))
            except queue.Empty:
                continue
            except Exception:
                continue

def process_loop():
    while not shutdown_event.is_set():
        try:
            frame_raw, size = raw_q.get(timeout=TIME_SLEEP)
            if shutdown_event.is_set():
                break
            if CAPTURE_TOOL == "WindowsCapture" and OS_NAME == "Windows":
                frame_raw = cv2.cvtColor(frame_raw, cv2.COLOR_BGRA2RGB)
            frame_rgb = process(frame_raw, size)
            proc_q.put(frame_rgb)
        except queue.Empty:
            continue

def cleanup_all_resources():
    """Global cleanup function"""
    print("[Cleanup] Shutting down all resources...")
    
    # Kill all processes
    for proc_name, process in global_processes.items():
        if process and hasattr(process, 'poll'):
            try:
                print(f"[Cleanup] Stopping {proc_name}...")
                process.terminate()
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                try:
                    print(f"[Cleanup] Force killing {proc_name}...")
                    process.kill()
                    process.wait()
                except:
                    pass
            except Exception as e:
                print(f"[Cleanup] Error stopping {proc_name}: {e}")
            finally:
                global_processes[proc_name] = None
    
    # Stop capture
    try:
        if 'cap' in globals():
            try:
                cap.stop()
            except AttributeError:
                # stop for WindowsCapture
                if OS_NAME == "Windows" and CAPTURE_MODE == "WindowsCapture":
                    capture_control.stop()
            print("[Cleanup] Capture stopped")
    except Exception as e:
        print(f"[Cleanup] Error stopping capture: {e}")
    
    # Stop streamer if exists
    try:
        if 'streamer' in globals() and streamer:
            streamer.stop()
            print("[Cleanup] Streamer stopped")
    except Exception as e:
        print(f"[Cleanup] Error stopping streamer: {e}")
    
    # Clear all queues to unblock threads
    queues = [raw_q, proc_q, depth_q]
    if 'sbs_q' in globals():
        queues.append(sbs_q)
    
    for q in queues:
        while not q.empty():
            try:
                q.get(timeout=TIME_SLEEP)
            except:
                pass
    
    print("[Cleanup] All resources cleaned up")

def signal_handler(signum, frame):
    """Handle Ctrl+C and other termination signals"""
    print(f"\n[Signal] Received signal {signum}, shutting down...")
    shutdown_event.set()
    cleanup_all_resources()
    sys.exit(0)

# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)
if OS_NAME != "Windows":
    signal.signal(signal.SIGQUIT, signal_handler)

# get ffmpeg command
def get_rtmp_cmd(os=OS_NAME, window=None):
    if os == "Windows":
        server_cmd = ['./rtmp/mediamtx/mediamtx.exe', './rtmp/mediamtx/mediamtx.yml']
        ffmpeg_cmd = [
            './rtmp/ffmpeg/bin/ffmpeg.exe',
            '-fflags', 'nobuffer',
            '-flags', 'low_delay',
            '-probesize', '32',
            '-analyzeduration', '0',
            '-filter_complex', f"gfxcapture=window_title='(?i)Stereo Viewer':max_framerate={FPS},hwdownload,format=bgra,scale=iw:trunc(ih/2)*2,format=yuv420p[v]",  # Label video output [v], fix odd height
            '-itsoffset', f'{AUDIO_DELAY}',  # Audio delay (applies to next input)
            '-f', 'dshow',
            '-rtbufsize', '256M',
            '-i', f'audio={STEREOMIX_DEVICE}',
            '-map', '[v]',
            '-map', '0:a',
            '-c:v', 'libx264',
            '-preset', 'ultrafast',
            '-tune', 'zerolatency',
            '-bf', '0',
            '-g', f'{FPS}',
            '-force_key_frames', f'expr:gte(t,n_forced*1)',  # Force keyframes every second
            '-r', f'{FPS}',  # Force constant output framerate
            '-crf', f'{CRF}', # 18-24 smaller better quality
            '-c:a', 'aac',
            '-ar', '44100',
            '-b:a', '128k',
            '-muxdelay', '0',
            '-muxpreload', '0',
            '-flush_packets', '1',
            '-rtmp_buffer', '0',
            '-f', 'flv',
            f'rtmp://localhost:1935/{STREAM_KEY}'
        ]
        
    elif os == "Darwin":
        scale_factor = cap.get_scale()
        def get_monitor_index_for_glfw(window):
            window_x, window_y = glfw.get_window_pos(window)
            monitors = glfw.get_monitors()

            for i, monitor in enumerate(monitors):
                mx, my = glfw.get_monitor_pos(monitor)
                mode = glfw.get_video_mode(monitor)
                mw, mh = mode.size.width, mode.size.height

                if mx <= window_x < mx + mw and my <= window_y < my + mh:
                    return i
            return 0
        import re
        def get_device_index(target_name, device_type="video"):
            cmd = ["./rtmp/mac/ffmpeg", "-f", "avfoundation", "-list_devices", "true", "-i", ""]
            result = subprocess.run(cmd, stderr=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
            output = result.stderr

            found_audio = False
            for line in output.splitlines():
                if "AVFoundation audio devices:" in line:
                    found_audio = True
                if "AVFoundation video devices:" in line:
                    found_audio = False

                match = re.search(r'\[(\d+)\](.+)', line)
                if match:
                    index = int(match.group(1))
                    name = match.group(2).strip()
                    if name == target_name and ((device_type == "audio" and found_audio) or (device_type == "video" and not found_audio)):
                        print(line)
                        return index
            return None
        
        monitor_index = get_monitor_index_for_glfw(window)
        screen_name = f"Capture screen {monitor_index}"
        screen_index = get_device_index(screen_name, "video")
        audio_index = get_device_index("BlackHole 2ch", "audio")
        width, height = glfw.get_window_size(window)
        x, y = glfw.get_window_pos(window)
        width, height, x, y = width * scale_factor, height*scale_factor, x * scale_factor, y * scale_factor
        server_cmd = ['./rtmp/mac/mediamtx', './rtmp/mac/mediamtx.yml']
        ffmpeg_cmd = [
            "./rtmp/mac/ffmpeg",
            "-itsoffset", str(AUDIO_DELAY),
            "-f", "avfoundation",
            "-rtbufsize", "256M",
            "-framerate", "59.94",
            "-i", f"{screen_index}:{audio_index}",
            "-filter_complex",
            f"[0:v]fps={FPS},crop={width}:{height}:{x}:{y},scale=iw:trunc(ih/2)*2,format=uyvy422[v];[0:a]aresample=async=1[a]",
            "-map", "[v]",
            "-map", "[a]",
            "-c:v", "libx264",
            "-bf", "0",
            "-g", str(FPS),
            "-r", str(FPS),
            "-preset", "ultrafast",
            "-crf", str(CRF),
            "-c:a", "libopus",
            "-ar", "48000",
            "-b:a", "128k",
            "-f", "rtsp",
            f"rtsp://localhost:8554/{STREAM_KEY}"
        ]
    
    elif os == "Linux":
        return None, None # TODO: add ffmpeg for Linux
    else:
        return None, None
    return server_cmd, ffmpeg_cmd

# ffmpeg based rtmp streamer
def rtmp_stream(window):
    try:
        # Start RTMP server
        server_cmd, ffmpeg_cmd = get_rtmp_cmd(OS_NAME, window=window)
        rtmp_server = subprocess.Popen(server_cmd, stdout=subprocess.PIPE)

        ffmpeg = subprocess.Popen(ffmpeg_cmd, stdout=subprocess.PIPE)  # Change to PIPE to capture logs
        
        # Store process references globally
        global_processes['rtmp_server'] = rtmp_server
        global_processes['ffmpeg'] = ffmpeg
        print(f"[RTMPStreamer] [RTMP] serving on rtmp://{LOCAL_IP}:1935/{STREAM_KEY}/")
        print(f"[RTSPStreamer] [RTSP] serving on rtsp://{LOCAL_IP}:8554/{STREAM_KEY}/")
        print(f"[RTMPStreamer] [HLS] serving on http://{LOCAL_IP}:8888/{STREAM_KEY}/")
        print(f"[RTMPStreamer] [HLS] for VLC serving on http://{LOCAL_IP}:8888/{STREAM_KEY}/index.m3u8")
        print(f"[RTMPStreamer] [WebRTC] serving on http://{LOCAL_IP}:8889/{STREAM_KEY}/")
        print("[RTMP] RTMP stream started")
        
        # Wait for shutdown event
        while not shutdown_event.is_set():
            time.sleep(0.1)
        
        # Cleanup when shutdown is signaled
        print("[RTMP] Shutting down RTMP stream...")
        ffmpeg.terminate()
        rtmp_server.terminate()
        ffmpeg.wait(timeout=5)
        rtmp_server.wait(timeout=5)
        
    except subprocess.TimeoutExpired:
        print("[RTMP] Timeout expired, force killing processes...")
        if 'ffmpeg' in locals():
            ffmpeg.kill()
        if 'rtmp_server' in locals():
            rtmp_server.kill()
    except Exception as e:
        print(f"[RTMP] Error: {e}")
        if 'ffmpeg' in locals():
            ffmpeg.terminate()
        if 'rtmp_server' in locals():
            rtmp_server.terminate()
    finally:
        print("[RTMP] RTMP stream stopped")

def main(mode="Viewer"):
    # Start capture and processing threads
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
                while not shutdown_event.is_set():
                    try:
                        frame_rgb = proc_q.get(timeout=TIME_SLEEP)
                        if shutdown_event.is_set():
                            break
                        depth = predict_depth(frame_rgb)
                        depth_q.put((frame_rgb, depth))
                    except queue.Empty:
                        continue
            
            threading.Thread(target=depth_loop, daemon=True).start()
            # build ffmpeg command pointing to your mediamtx and include audio device
            if STREAM_MODE:
                frame_rgb = proc_q.get()
                w, h = frame_rgb.shape[1], frame_rgb.shape[0]
                if DISPLAY_MODE == "Full-SBS":
                    w = 2 * w
                window = StereoWindow(ipd=IPD, depth_ratio=DEPTH_STRENGTH, display_mode=DISPLAY_MODE, show_fps=SHOW_FPS, stream_mode=STREAM_MODE, frame_size = (w,h))
            else:
                # For local viewer only
                window = StereoWindow(ipd=IPD, depth_ratio=DEPTH_STRENGTH, display_mode=DISPLAY_MODE, show_fps=SHOW_FPS)
            if STREAM_MODE == "RTMP":
                if OS_NAME == "Windows":
                    from utils import set_window_to_bottom
                    def bottom_loop():
                        while True:
                            set_window_to_bottom(window.window)
                            time.sleep(0.01)
                    threading.Thread(target=bottom_loop, daemon=True).start()
                rtmp_thread = threading.Thread(target=lambda: rtmp_stream(window.window), daemon=True)
                rtmp_thread.start()
                print(f"[Main] RTMP Streamer Started")
            elif STREAM_MODE == "MJPEG":
                from streamer import MJPEGStreamer
                streamer = MJPEGStreamer(port=STREAM_PORT, fps=FPS, quality=STREAM_QUALITY)
                streamer.start()
                print(f"[Main] MJPEG Streamer Started")
            else:
                print(f"[Main] Local Viewer Started")
            
            while (not glfw.window_should_close(window.window) and 
                   not shutdown_event.is_set()):
                try:
                    rgb, depth = depth_q.get(timeout=TIME_SLEEP)
                    window.update_frame(rgb, depth)
                    if STREAM_MODE == "MJPEG":
                        frame = window.capture_glfw_image()
                        streamer.set_frame(frame)
                    if SHOW_FPS:
                        frame_count += 1
                        total_frames += 1
                        current_time = time.perf_counter()
                        if current_time - last_time >= 1.0:
                            current_fps = frame_count / (current_time - last_time)
                            frame_count = 0
                            last_time = current_time
                            if STREAM_MODE == "MJPEG":
                                print(f"FPS: {current_fps:.1f}")
                            glfw.set_window_title(window.window, f"Stereo Viewer | {current_fps:.1f} FPS")
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
                    while not shutdown_event.is_set():
                        try:
                            rgb, depth = depth_q.get(timeout=TIME_SLEEP)
                            if shutdown_event.is_set():
                                break
                            sbs = make_sbs(rgb, depth, ipd_uv=IPD, depth_ratio=DEPTH_STRENGTH, display_mode=DISPLAY_MODE, fps=current_fps)
                            sbs_q.put(sbs)
                        except queue.Empty:
                            continue
            
            def depth_loop():
                while not shutdown_event.is_set():
                    try:
                        frame_rgb = proc_q.get(timeout=TIME_SLEEP)
                        if shutdown_event.is_set():
                            break
                        depth, rgb = predict_depth(frame_rgb, return_tuple=True)
                        depth_q.put(make_output(rgb, depth))
                    except queue.Empty:
                        continue
            
            threading.Thread(target=depth_loop, daemon=True).start()
            
            if BOOST:
                threading.Thread(target=sbs_loop, daemon=True).start()
            
            streamer = MJPEGStreamer(port=STREAM_PORT, fps=FPS, quality=STREAM_QUALITY)
            streamer.start()
            
            print(f"[Main] Legacy Streamer Started")
            
            while not shutdown_event.is_set():
                try:
                    if not BOOST:
                        # Fix for unstable dml runtime error
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
                            print(f"FPS: {current_fps:.1f}")
                            
                except queue.Empty:
                    continue
                except Exception as e:
                    if not shutdown_event.is_set():
                        print(f"Streamer error: {e}")
                    break
                    
    except KeyboardInterrupt:
        print("\n[Main] Keyboard interrupt received, shutting down...")
    # except Exception as e:
    #     print(f"[Main] Error: {e}")
    finally:
        # Ensure cleanup happens
        shutdown_event.set()
        cleanup_all_resources()
        
        if SHOW_FPS:
            total_time = time.perf_counter() - start_time
            avg_fps = total_frames / total_time if total_time > 0 else 0
            print(f"Average FPS: {avg_fps:.2f}")
        print(f"[Main] Stopped")

if __name__ == "__main__":
    main(mode=RUN_MODE)
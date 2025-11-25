# main.py
import threading
import queue
import glfw
import time
import signal
import sys
import subprocess
import cv2
from utils import OS_NAME, OUTPUT_RESOLUTION, DISPLAY_MODE, CAPTURE_MODE, CAPTURE_TOOL, MONITOR_INDEX, SHOW_FPS, FPS, WINDOW_TITLE, IPD, DEPTH_STRENGTH, RUN_MODE, STREAM_MODE, STREAM_PORT, STREAM_QUALITY, DML_BOOST, STEREOMIX_DEVICE, STREAM_KEY, AUDIO_DELAY, CRF, shutdown_event
from depth import process, predict_depth

# Global process references
global_processes = {
    'ffmpeg': None,
    'rtmp_server': None
}

# Track current stream size + restart lock
current_stream_size = None
ffmpeg_restart_lock = threading.Lock()

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
    
    def capture_loop():
        cap = DesktopGrabber(output_resolution=OUTPUT_RESOLUTION, fps=FPS, window_title=WINDOW_TITLE, capture_mode=CAPTURE_MODE, monitor_index=MONITOR_INDEX)
        while not shutdown_event.is_set():
            try:
                frame_raw, size = cap.grab()
                if shutdown_event.is_set():
                    break
                raw_q.put((frame_raw, size))
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[Warning] Error: {e}")
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

    # Wait for RTMP thread
    if 'rtmp_thread' in globals() and rtmp_thread.is_alive():
        rtmp_thread.join(timeout=3)
    
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
def get_rtmp_cmd(os_name=OS_NAME, window=None):
    if not window:
        raise ValueError("GLFW window required for size-aware streaming")

    width, height = glfw.get_framebuffer_size(window)
    width = (width // 2) * 2  # make even
    height = (height // 2) * 2  # make even

    if os_name == "Windows":
        server_cmd = ['./rtmp/mediamtx/mediamtx.exe', './rtmp/mediamtx/mediamtx.yml']
        ffmpeg_cmd = [
            './rtmp/ffmpeg/bin/ffmpeg.exe',
            '-fflags', 'nobuffer',
            '-flags', 'low_delay',
            '-probesize', '64',
            '-analyzeduration', '0',
            '-filter_complex', f"gfxcapture=window_title='(?i)Stereo Viewer':max_framerate={FPS},hwdownload,format=bgra,scale={width}:{height},format=yuv420p[v]",  # Label video output [v], fix odd height
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
            '-c:a', 'libopus',
            # '-ar', '44100',
            '-b:a', '96k',
            '-muxdelay', '0',
            '-muxpreload', '0',
            '-flush_packets', '1',
            '-rtmp_buffer', '0',
            '-f', 'flv',
            f'rtmp://localhost:1935/{STREAM_KEY}'
        ]
        
    elif os_name == "Darwin":
        
        from AppKit import NSScreen
        from capture import get_window_client_bounds_mac

        def get_scale(monitor_index):
            """Get the Retina scale factor for a specific monitor"""
            screens = NSScreen.screens()
            if monitor_index < len(screens):
                return screens[monitor_index].backingScaleFactor()
            return 2.0  # Default to 2x for Retina displays
        
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
        monitors = glfw.get_monitors()
        monitor = monitors[monitor_index]
        scale_factor = get_scale(monitor_index)
        screen_name = f"Capture screen {monitor_index}"
        screen_index = get_device_index(screen_name, "video")
        audio_index = get_device_index(STEREOMIX_DEVICE, "audio")
        win_x, win_y = glfw.get_window_pos(window)
        mon_x, mon_y = glfw.get_monitor_pos(monitor)
        x = win_x - mon_x
        y = win_y - mon_y
        x = int(x * scale_factor)
        y = int(y * scale_factor)
        width = int(width * scale_factor)
        height = int(height * scale_factor)
        # print(width, height, x, y)
        server_cmd = ['./rtmp/mac/mediamtx', './rtmp/mac/mediamtx.yml']
        ffmpeg_cmd = [
            "./rtmp/mac/ffmpeg",
            "-fflags", "nobuffer",
            "-flags", "low_delay",
            "-probesize", "1024",
            "-analyzeduration", "0",
            "-itsoffset", str(AUDIO_DELAY),
            "-pixel_format", "uyvy422",
            "-f", "avfoundation",
            "-rtbufsize", "256M",
            "-framerate", "60",
            "-i", f"{screen_index}:{audio_index}",
            "-filter_complex",
            f"[0:v]fps={FPS},crop={width}:{height}:{x}:{y},scale=trunc(iw/2)*2:trunc(ih/2)*2,format=uyvy422[v];[0:a]aresample=async=1[a]",
            "-map", "[v]",
            "-map", "[a]",
            "-c:v", "h264_videotoolbox",
            "-profile:v", "high",
            "-pix_fmt", "yuv420p",
            "-b:v", "10M",
            "-maxrate", "12M",
            "-bufsize", "24M",
            "-g", str(FPS),
            "-r", str(FPS),
            "-realtime", "true",
            "-color_primaries", "bt709",
            "-color_trc", "bt709",
            "-colorspace", "bt709",
            "-c:a", "libopus",
            "-b:a", "96k",
            "-ar", "48000",
            "-f", "rtsp",
            f"rtsp://localhost:8554/{STREAM_KEY}",
        ]


    elif os_name == "Linux":
        import time
        import re

        def run(cmd):
            """Run a shell command and return output as string."""
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            return result.stdout.strip()

        def get_window_geometry(window_id, entire=True):
            """
            Parse xwininfo to get full window geometry and decorations (border + title bar).
            Returns: (x, y, width, height, border, titlebar)
            """
            output = run(f"xwininfo -id {window_id}")
            x = y = w = h = b = t = 0

            # Parse lines similarly to the sed version
            for line in output.splitlines():
                line = line.strip()
                if m := re.match(r"Absolute upper-left X:\s+(\d+)", line):
                    x = int(m.group(1))
                elif m := re.match(r"Absolute upper-left Y:\s+(\d+)", line):
                    y = int(m.group(1))
                elif m := re.match(r"Width:\s+(\d+)", line):
                    w = int(m.group(1))
                elif m := re.match(r"Height:\s+(\d+)", line):
                    h = int(m.group(1))
                elif m := re.match(r"Relative upper-left X:\s+(\d+)", line):
                    b = int(m.group(1))
                elif m := re.match(r"Relative upper-left Y:\s+(\d+)", line):
                    t = int(m.group(1))

            # Adjust if user wanted entire window including borders/titlebar
            if entire:
                x -= b
                y -= t
                w += 2 * b
                h += t + b

            return x, y, w, h, b, t

        def drag_window_offscreen(window_id, dx=10000, dy=10000, steps=1, delay=0.01):
            """
            Simulate a mouse drag on a window by ID using xdotool.
            Uses xwininfo to compute exact title bar position (no scale factor needed).
            """
            # Get window geometry + decoration info
            x, y, w, h, b, t = get_window_geometry(window_id)
            print(f"Window pos=({x},{y}), size={w}x{h}, border={b}, title={t}")

            # Activate the window
            run(f"xdotool windowactivate {window_id}")

            # Compute title bar click position
            # title_x = x + w // 2
            title_x = x + 20
            title_y = y + t // 2  # halfway down the title bar for reliable click

            # Move mouse to title bar and start drag
            run(f"xdotool mousemove {title_x} {title_y}")
            run("xdotool mousedown 1")

            # Smooth drag motion
            step_x = dx / steps
            step_y = dy / steps
            for _ in range(steps):
                run(f"xdotool mousemove_relative -- {step_x:.2f} {step_y:.2f}")
                time.sleep(delay)

            # Release mouse button
            run("xdotool mouseup 1")



        def get_display_env(window_id: str) -> str:
            """
            Get the DISPLAY environment variable from the process that owns the window.
            Falls back to ':0.0' if not found or on error.
            """
            try:
                # First, get the PID of the window
                pid_result = subprocess.run(
                    ["xprop", "-id", window_id, "_NET_WM_PID"],
                    capture_output=True, text=True, check=True
                )
                
                # Parse PID from output (e.g., '_NET_WM_PID(CARDINAL) = 1234')
                for line in pid_result.stdout.splitlines():
                    if "=" in line:
                        pid = line.split("=")[-1].strip()
                        if pid.isdigit():
                            # Get environment variables from the process
                            env_result = subprocess.run(
                                ["cat", f"/proc/{pid}/environ"],
                                capture_output=True, text=True, check=True
                            )
                            
                            # Parse DISPLAY from environment variables
                            for env_var in env_result.stdout.split('\x00'):
                                if env_var.startswith("DISPLAY="):
                                    return env_var.split("=", 1)[1]
                                    
            except (subprocess.CalledProcessError, FileNotFoundError, IndexError, ValueError):
                pass
            
            return ":0.0"

        def get_device_index(target_name, device_type="video"):
            """Return a device identifier suitable for ffmpeg on Linux.
            For video: returns a display/input string (we'll use it later as the ffmpeg input).
            For audio: tries to locate a PulseAudio/ALSA name that matches target_name; returns 'default' if not found.
            """
            # AUDIO: try PulseAudio device listing via ffmpeg
            if device_type == "audio":
                # Use the ffmpeg binary in your rtmp/linux folder if present; otherwise fallback to system 
                cmd = ["ffmpeg", "-f", "pulse", "-list_devices", "true", "-i", ""]  # PulseAudio list (printed to stderr)
                try:
                    result = subprocess.run(cmd, stderr=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
                    output = result.stderr + result.stdout
                    # ffmpeg prints devices with lines like: "    name 'alsa_input.pci-0000_00_1b.0.analog-stereo'": device names can appear
                    # Simpler: find lines containing the target_name or the friendly name
                    for line in output.splitlines():
                        if target_name in line:
                            # extract quoted name if present
                            m = re.search(r"'([^']+)'", line)
                            if m:
                                return m.group(1)
                            # otherwise attempt to extract a token
                            tokens = line.strip().split()
                            if tokens:
                                return tokens[-1].strip()
                except Exception:
                    pass

                # Try listing via pactl as fallback
                try:
                    pactl_out = subprocess.check_output(["pactl", "list", "sources"], text=True)
                    # parse "Name: <name>" and "Description: <desc>"
                    name = None
                    for line in pactl_out.splitlines():
                        line = line.strip()
                        if line.startswith("Name:"):
                            name = line.split(":", 1)[1].strip()
                        if line.startswith("Description:") and target_name in line:
                            return name or "default"
                except Exception:
                    pass

                # fallback
                return "default"
            return target_name or ":0.0"
        
        def find_window_id_by_title(title_pattern: str) -> str | None:
            """
            Returns the hex window id (e.g. '0x4a0000b') of the first window whose
            title contains *title_pattern* (case-insensitive).  Uses `xwininfo -tree -root`.
            """
            try:
                out = subprocess.check_output(
                    ["xwininfo", "-root", "-tree"], text=True, stderr=subprocess.DEVNULL
                )
                # Example line:
                #     0x4a0000b "Stereo Viewer - Left Eye": ("Stereo Viewer - Left Eye" ...
                pat = re.compile(
                    rf'^\s+(0x[0-9a-fA-F]+)\s.*?"[^"]*{re.escape(title_pattern)}[^"]*"',
                    re.IGNORECASE,
                )
                for line in out.splitlines():
                    m = pat.match(line)
                    if m:
                        return m.group(1)
            except Exception as e:
                print(f"[window_id] search failed: {e}")
            return None

        # Make sure the GLFW window has a recognizable title:
        glfw_title = glfw.get_window_title(window) or ""
        search_title = glfw_title if "Stereo Viewer" in glfw_title else "Stereo Viewer"

        window_id = find_window_id_by_title(search_title)
        display_env = get_display_env(window_id)+".0"
        
        if not window_id:
            raise RuntimeError(
                f"Could not locate a window with title containing '{search_title}'. "
                "Check glfw.set_window_title() or run `xwininfo -tree -root | grep -i stereo`."
            )
        print(f"[info] Capturing window id {window_id}")
        drag_window_offscreen(window_id)

        # audio_index = get_device_index(STEREOMIX_DEVICE, "audio")
        server_cmd = ["./rtmp/linux/mediamtx", "./rtmp/linux/mediamtx.yml"]

        ffmpeg_cmd = [
            "ffmpeg",
            "-fflags", "+genpts+nobuffer+flush_packets",
            "-flags", "low_delay",
            "-avioflags", "direct",
            "-probesize", "1024",
            "-analyzeduration", "0",
            "-draw_mouse", "0",
            "-itsoffset", str(AUDIO_DELAY),
            "-f", "x11grab",
            "-framerate", "60", 
            "-vsync", "1",
            "-window_id", window_id,
            "-use_wallclock_as_timestamps", "1",
            "-thread_queue_size", "2048",
            "-i", display_env,
            "-f", "pulse",
            "-thread_queue_size", "512",
            "-i", STEREOMIX_DEVICE,
            "-ac", "2",
            "-filter_complex", f"[0:v]fps={FPS},scale=trunc(iw/2)*2:trunc(ih/2)*2,format=yuv420p,setpts=PTS-STARTPTS[v];[1:a]aresample=async=1:first_pts=0,apad[a]",
            "-map", "[v]",
            "-map", "[a]",
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-tune", "zerolatency",
            "-x264-params", f"keyint={FPS}:min-keyint={FPS}:scenecut=0:rc-lookahead=0",
            "-g", str(FPS),
            "-r", str(FPS),
            "-crf", str(CRF),
            "-c:a", "aac", 
            "-ar", "44100", 
            "-b:a", "96k", 
            "-threads", "2",
            "-f", "flv", 
            f"rtmp://localhost:1935/{STREAM_KEY}"
        ]

    return server_cmd, ffmpeg_cmd

# ffmpeg based rtmp streamer
def rtmp_stream(window):
    global current_stream_size, global_processes

    while not shutdown_event.is_set():
        try:
            width, height = glfw.get_framebuffer_size(window)
            new_size = (width, height)

            # Debounce: ignore tiny changes
            if current_stream_size and abs(current_stream_size[0] - new_size[0]) < 8 and abs(current_stream_size[1] - new_size[1]) < 8:
                time.sleep(0.1)
                continue

            with ffmpeg_restart_lock:
                if current_stream_size == new_size:
                    time.sleep(0.1)
                    continue
                current_stream_size = new_size

            # Terminate old processes
            for name in ['ffmpeg', 'rtmp_server']:
                proc = global_processes.get(name)
                if proc and proc.poll() is None:
                    print(f"[RTMP] Stopping {name} for resize...")
                    proc.terminate()
                    try:
                        proc.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        proc.kill()

            # Start new
            server_cmd, ffmpeg_cmd = get_rtmp_cmd(OS_NAME, window=window)
            print(f"[RTMP] Restarting stream at {width}x{height}")

            rtmp_server = subprocess.Popen(server_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            ffmpeg = subprocess.Popen(ffmpeg_cmd, stdout=subprocess.PIPE)

            global_processes['rtmp_server'] = rtmp_server
            global_processes['ffmpeg'] = ffmpeg

            print(f"[RTMP] Stream active: {width}x{height}")

        except Exception as e:
            print(f"[RTMP] Error: {e}")
            time.sleep(1)

        time.sleep(0.2)

    print("[RTMP] Stream thread exited.")

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
                global rtmp_thread
                rtmp_thread = threading.Thread(target=rtmp_stream, args=(window.window,), daemon=True)
                rtmp_thread.start()
                print(f"[Main] RTMP Streamer Started (auto-restart on resize)")
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
                    frame_count += 1
                    total_frames += 1
                    current_time = time.perf_counter()
                    if current_time - last_time >= 1.0:
                        current_fps = frame_count / (current_time - last_time)
                        frame_count = 0
                        last_time = current_time
                        if STREAM_MODE == "MJPEG":
                            print(f"{current_fps:.1f} FPS")
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
                        sbs = depth_q.get(timeout=TIME_SLEEP)
                    else:
                        sbs = sbs_q.get(timeout=TIME_SLEEP)
                    
                    streamer.set_frame(sbs)
                    
                
                    frame_count += 1
                    current_time = time.perf_counter()
                    if current_time - last_time >= 1.0:
                        current_fps = frame_count / (current_time - last_time)
                        frame_count = 0
                        last_time = current_time
                        print(f"{current_fps:.1f} FPS")
                            
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
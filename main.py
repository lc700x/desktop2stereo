# main.py
import threading
import queue
import glfw
import time
import signal
import sys
import subprocess

from utils import OS_NAME, OUTPUT_RESOLUTION, DISPLAY_MODE, CAPTURE_MODE, CAPTURE_TOOL, MONITOR_INDEX, SHOW_FPS, FPS, WINDOW_TITLE, IPD, DEPTH_STRENGTH, CONVERGENCE, RUN_MODE, STREAM_MODE, STREAM_PORT, STREAM_QUALITY, STEREOMIX_DEVICE, STREAM_KEY, AUDIO_DELAY, CRF, LOSSLESS_SCALING_SUPPORT, USE_3D_MONITOR, FILL_16_9, FIX_VIEWER_ASPECT, CAPTURE_MODE, STEREO_DISPLAY_SELECTION, STEREO_DISPLAY_INDEX, shutdown_event, DEVICE_ID, DEVICE_INFO
from depth import process, predict_depth

if "CUDA" in DEVICE_INFO and "ZLUDA" not in DEVICE_INFO:
    USE_CUDART = True
else:
    USE_CUDART = False

# Fix for AMD GPUs
USE_CUDART = False if CAPTURE_TOOL == "WindowsCapture" and "AMD" in DEVICE_INFO else USE_CUDART

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
depth_q = queue.Queue(maxsize=1)

# Thread latency tracking dictionaries
thread_latencies = {
    'capture': 0.0,
    'resize': 0.0,
    'depth': 0.0,
    'render': 0.0,
    'total': 0.0
}

# Initialize capture
if CAPTURE_TOOL in ["WindowsCapture", "WindowsCaptureCUDA"] and OS_NAME == "Windows":
    import ctypes
    from ctypes import wintypes
    import threading
    from utils import is_windows_11_24h2_or_newer

    # Import capture library (regular or CUDA-accelerated)
    if CAPTURE_TOOL == "WindowsCapture":
        from windows_capture import WindowsCapture, Frame, InternalCaptureControl
    else:  # WindowsCaptureCUDA
        from wc_cuda import WindowsCapture, Frame, InternalCaptureControl
    
    # optional small delay (seconds) after capture event before performing actions
    CAPTURE_CURSOR_DELAY_S = 0.2

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
    VK_MENU = 0x12   # Left Alt key
    VK_TAB = 0x09    # Tab key
    KEYEVENTF_KEYUP = 0x0002

    def simulate_alt_tab():
        """
        Simulate pressing Alt+Tab to switch windows, then again to return.
        Returns True if successful, False otherwise.
        """
        try:

            # Second Alt+Tab (switch back)
            user32.keybd_event(VK_MENU, 0, 0, 0)   # Alt down
            user32.keybd_event(VK_TAB, 0, 0, 0)    # Tab down
            time.sleep(0.01)
            user32.keybd_event(VK_TAB, 0, KEYEVENTF_KEYUP, 0)  # Tab up
            user32.keybd_event(VK_MENU, 0, KEYEVENTF_KEYUP, 0) # Alt up

            return True
        except Exception:

            print(f"[simulate_alt_tab] Failed to simulate Win+Tab: {e}")
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
                simulate_alt_tab()
                time.sleep(0.2)
                simulate_alt_tab()

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
    alt_tab_thread = threading.Thread(target=_keyboard, name="CursorWorker", daemon=True)
    alt_tab_thread.start()

    # Initialize capture object and capture loop for 24H2
    IS_24h2 = is_windows_11_24h2_or_newer()

    if IS_24h2:
        cap = (
            WindowsCapture(window_name=WINDOW_TITLE, minimum_update_interval=int(TIME_SLEEP * 1000))
            if CAPTURE_MODE == "Window"
            else WindowsCapture(monitor_index=MONITOR_INDEX)
        )
    else:
        cap = (
            WindowsCapture(window_name=WINDOW_TITLE)
            if CAPTURE_MODE == "Window"
            else WindowsCapture(monitor_index=MONITOR_INDEX)
        )


    def capture_loop():
        global capture_control

        if IS_24h2:
            @cap.event
            def on_frame_arrived(frame: Frame, capture_control: InternalCaptureControl):

                capture_start_time = time.perf_counter()
                if shutdown_event.is_set():
                    return
                # check frame property
                if hasattr(frame.frame_buffer, "copy"):
                    raw = frame.frame_buffer.copy()
                else:
                    raw = frame.frame_buffer

                raw_q.put((raw, OUTPUT_RESOLUTION, capture_start_time))

        else:
            next_frame_time = time.perf_counter()
            @cap.event
            def on_frame_arrived(frame: Frame, capture_control: InternalCaptureControl):
                nonlocal next_frame_time

                capture_start_time = time.perf_counter()
                if shutdown_event.is_set():
                    return
                # Skip frames arriving too early (keeps FPS stable)
                if capture_start_time < next_frame_time:
                    return

                # Prevent timing drift if system lags temporarily
                next_frame_time += TIME_SLEEP
                capture_started_event.set()
                # check frame property
                if hasattr(frame.frame_buffer, "copy"):
                    raw = frame.frame_buffer.copy()
                else:
                    raw = frame.frame_buffer

                raw_q.put((raw, OUTPUT_RESOLUTION, capture_start_time))

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
                capture_start_time = time.perf_counter()
                frame_raw, size = cap.grab()
                
                if shutdown_event.is_set():
                    break
                
                raw_q.put((frame_raw, size, capture_start_time))
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[Warning] Error: {e}")
                continue

# Combined processing + depth thread (replaces process_loop and depth_loop)

def process_depth_loop():
    target_time = time.perf_counter()
    while not shutdown_event.is_set():
        target_time += TIME_SLEEP
        try:
            if shutdown_event.is_set():
                break
            
            # Get raw frame with capture timestamp
            frame_raw, size, capture_start_time = raw_q.get(timeout=TIME_SLEEP)
            
            # Process: resize / color conversion
            process_start_time = time.perf_counter()
            frame_rgb = process(frame_raw, size)
            process_latency = process_start_time - capture_start_time
            thread_latencies['capture'] = process_latency  # capture latency
            
            # Depth inference
            depth_start_time = time.perf_counter()
            depth = predict_depth(frame_rgb)
            depth_latency = time.perf_counter() - depth_start_time
            thread_latencies['resize'] = process_latency   # resize latency
            thread_latencies['depth'] = depth_latency      # depth latency
            
            # Send to render queue
            depth_q.put((frame_rgb, depth, capture_start_time))

            sleep = target_time - time.perf_counter()
            if sleep > 0:
                time.sleep(sleep)

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
    queues = [raw_q, depth_q]
    
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

# Get window lists
if OS_NAME == "Windows":
    try:
        import win32gui
    except ImportError:
        win32gui = None

    def list_windows():
        windows = []
        def callback(hwnd, _):
            if win32gui.IsWindowVisible(hwnd):
                title = win32gui.GetWindowText(hwnd)
                if title:
                    windows.append((title, hwnd))
            return True

        win32gui.EnumWindows(callback, None)
        return windows
elif OS_NAME == "Darwin":
    try:
        from Quartz import (
            CGWindowListCopyWindowInfo,
            kCGWindowListOptionOnScreenOnly,
            kCGWindowListExcludeDesktopElements,
            kCGNullWindowID,
        )
    except ImportError:
        CGWindowListCopyWindowInfo = None

    def list_windows():
        windows = []
        options = kCGWindowListOptionOnScreenOnly | kCGWindowListExcludeDesktopElements
        window_info = CGWindowListCopyWindowInfo(options, kCGNullWindowID)
        # System UI processes we want to ignore
        blacklist = {
            "Window Server",
            "ControlCenter",
            "NotificationCenter",
            "Spotlight",
            "Dock",
            "SystemUIServer",
            "CoreServicesUIAgent",
            "TextInputMenuAgent",
        }
        for win in window_info:
            title = win.get("kCGWindowName", "") or ""
            owner = win.get("kCGWindowOwnerName", "")
            layer = win.get("kCGWindowLayer", 0)
            bounds = win.get("kCGWindowBounds", {})
            # Filtering rules
            if not title.strip():
                continue
            if owner in blacklist:
                continue
            if title.strip().lower().startswith("item-"):
                continue
            if bounds.get("Y", 1) == 0:
                continue
            windows.append((title.strip(), win["kCGWindowNumber"]))
        return windows
else:
    import subprocess
    def list_windows():
        windows = []
        try:
            result = subprocess.check_output(["wmctrl", "-l"]).decode("utf-8").splitlines()
            for line in result:
                parts = line.split(None, 3)
                if len(parts) >= 4:
                    _, _, _, title = parts
                    if title.strip():
                        windows.append((title.strip(), None))
        except Exception as e:
            print("Linux window listing error:", e)
        return windows

def is_window_visible_on_screen(window_title_search, partial_match=True, timeout=2.0):
    """
    Check if a window with the given title is actually visible on screen.
    
    Args:
        window_title_search: Title or partial title to search for
        partial_match: If True, search for windows containing the search string
        timeout: How long to keep trying (seconds)
    
    Returns:
        tuple: (found, window_title, window_id_or_handle)
    """
    
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            windows = list_windows()
            
            for title, window_id in windows:
                if partial_match:
                    if window_title_search.lower() in title.lower():
                        return True
                    if title == window_title_search:
                        return True
            
            time.sleep(0.1)
        except Exception as e:
            print(f"[Window Check] Error listing windows: {e}")
            time.sleep(0.5)
    
    return False

def get_rtmp_cmd(os_name=OS_NAME, window=None):
    if not window:
        raise ValueError("GLFW window required for size-aware streaming")
    
    time.sleep(0.5) # wait 0.5 seconds to get correct window position
    width, height = glfw.get_framebuffer_size(window)
    if width > 0 and height > 0:
        width = (width // 2) * 2  # make even
        height = (height // 2) * 2  # make even
    else:
        time.sleep(3) # retry after 3 seconds
        get_rtmp_cmd()
        
    if os_name == "Windows":
        server_cmd = ['./rtmp/mediamtx/mediamtx.exe', './rtmp/mediamtx/mediamtx.yml']
        if not LOSSLESS_SCALING_SUPPORT: 
            ffmpeg_cmd = [
                './rtmp/ffmpeg/bin/ffmpeg.exe',
                '-fflags', 'nobuffer',
                '-flags', 'low_delay',
                '-probesize', '64',
                '-analyzeduration', '0',
                '-filter_complex', f"gfxcapture=window_title='(?i)Stereo Viewer':capture_cursor=0:max_framerate={FPS},hwdownload,format=bgra,scale={width}:{height},format=yuv420p[v]",  # Label video output [v], fix odd height
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
                '-f', 'mpegts',
                f'srt://localhost:8890?streamid=publish:{STREAM_KEY}&pkt_size=1316'
            ]
        else:
            import win32gui
            import win32api

            def find_window_by_prefix(prefix):
                target_hwnd = None
                def enum_handler(hwnd, _):
                    nonlocal target_hwnd
                    title = win32gui.GetWindowText(hwnd)
                    if title.lower().startswith(prefix.lower()) and win32gui.IsWindowVisible(hwnd):
                        target_hwnd = hwnd
                win32gui.EnumWindows(enum_handler, None)
                return target_hwnd

            def get_monitor_index_for_window(hwnd):
                """Returns the 0-based monitor index (matching gfxcapture order) that contains the center of the window"""
                if not hwnd:
                    raise ValueError("Invalid window handle")
                
                left, top, right, bottom = win32gui.GetWindowRect(hwnd)
                center_x = (left + right) // 2
                center_y = (top + bottom) // 2
                
                monitors = []
                for hMonitor, hdcMonitor, (left, top, right, bottom) in win32api.EnumDisplayMonitors():
                    info = win32api.GetMonitorInfo(hMonitor)
                    monitors.append({
                        'index': len(monitors),  # This order matches gfxcapture monitor_idx
                        'left': left,
                        'top': top,
                        'right': right,
                        'bottom': bottom,
                        'width': right - left,
                        'height': bottom - top,
                    })
                
                for mon in monitors:
                    if mon['left'] <= center_x < mon['right'] and mon['top'] <= center_y < mon['bottom']:
                        return mon['index'], mon
                
                raise RuntimeError("Could not determine monitor for window")

            # Main logic
            hwnd = find_window_by_prefix("Stereo Viewer")
            if not hwnd:
                raise RuntimeError("Window starting with 'Stereo Viewer' not found")

            # Get monitor index and full monitor rect
            monitor_idx, monitor = get_monitor_index_for_window(hwnd)

            # Get window client area (recommended: excludes title bar and borders)
            window_left, window_top = win32gui.ClientToScreen(hwnd, (0, 0))
            client_rect = win32gui.GetClientRect(hwnd)
            window_right = window_left + client_rect[2]
            window_bottom = window_top + client_rect[3]

            # Alternatively: full window including borders/title bar
            # window_left, window_top, window_right, window_bottom = win32gui.GetWindowRect(hwnd)

            print(f"✅ Stereo Viewer window found on monitor {monitor_idx}")
            print(f"Monitor bounds: X={monitor['left']} Y={monitor['top']} W={monitor['width']} H={monitor['height']}")
            print(f"Window client area: X={window_left} Y={window_top} -> {window_right}x{window_bottom}")

            # Calculate crop values so that the captured monitor region is cropped exactly to the window
            crop_left   = window_left - monitor['left']
            crop_top    = window_top - monitor['top']
            crop_right  = monitor['right'] - window_right
            crop_bottom = monitor['bottom'] - window_bottom

            # Ensure non-negative crops (in case window is partially off-screen or miscalculated)
            crop_left   = max(0, crop_left)
            crop_top    = max(0, crop_top)
            crop_right  = max(0, crop_right)
            crop_bottom = max(0, crop_bottom)

            print("Calculated crop values (to capture only the window area):")
            print(f"crop_left   = {crop_left}")
            print(f"crop_top    = {crop_top}")
            print(f"crop_right  = {crop_right}")
            print(f"crop_bottom = {crop_bottom}")

            # Build gfxcapture options
            gfxcapture_options = f"monitor_idx={monitor_idx}:crop_left={crop_left}:crop_top={crop_top}:crop_right={crop_right}:crop_bottom={crop_bottom}"

            ffmpeg_cmd = [
                './rtmp/ffmpeg/bin/ffmpeg.exe',
                '-fflags', 'nobuffer',
                '-flags', 'low_delay',
                '-probesize', '64',
                '-analyzeduration', '0',
                '-filter_complex', f"gfxcapture={gfxcapture_options}:capture_cursor=0:max_framerate={FPS},hwdownload,format=bgra,scale={width}:{height},format=yuv420p[v]",
                '-itsoffset', f'{AUDIO_DELAY}',
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
                '-force_key_frames', f'expr:gte(t,n_forced*1)',
                '-r', f'{FPS}',
                '-crf', f'{CRF}',
                '-c:a', 'libopus',
                '-b:a', '96k',
                '-muxdelay', '0',
                '-muxpreload', '0',
                '-flush_packets', '1',
                '-f', 'mpegts',
                f'srt://localhost:8890?streamid=publish:{STREAM_KEY}&pkt_size=1316'
            ]
            
    elif os_name == "Darwin":
        
        from AppKit import NSScreen

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
            "-f", "rtsp",
            f"rtsp://localhost:8554/{STREAM_KEY}",
        ]

    return server_cmd, ffmpeg_cmd

# ffmpeg based rtmp streamer
def rtmp_stream(window):
    global current_stream_size, global_processes

    # Wait for GLFW window to be fully initialized
    print("[RTMP] Waiting for window to be ready...")
    max_attempts = 100  # 5 seconds with 0.1s intervals
    for _ in range(max_attempts):
        if shutdown_event.is_set():
            return
        
        # Check if window is valid and has a size
        try:
            width, height = glfw.get_framebuffer_size(window)
            if width > 0 and height > 0 and is_window_visible_on_screen("Stereo Viewer"):
                print(f"[RTMP] Window ready: {width}x{height}")
                time.sleep(2)
                break
        except Exception as e:
            print(f"[RTMP] Window check error: {e}")
        
        time.sleep(0.5)
    
    if shutdown_event.is_set():
        return

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
    # Replace separate process_loop and depth_loop with combined thread
    threading.Thread(target=process_depth_loop, daemon=True).start()
    
    # FPS tracking variables
    frame_count = 0
    start_time = time.perf_counter()
    last_time = time.perf_counter()
    last_fps_update_time = time.perf_counter()
    current_fps = 0.0
    total_frames = 0
    streamer, window = None, None
    
    # FPS statistics tracking
    fps_values = []  # Store recent FPS values for 1% percentile calculation
    max_fps_history = 300  # Keep last 300 FPS values (5 seconds at 60 FPS)
    avg_fps = 0.0
    low_fps_1_percent_avg = float('inf')  # Average of FPS below 1% percentile
    fps_update_interval = 5.0  # Update statistics every 5 seconds
    
    # Latency statistics tracking
    latency_history = []  # Store recent latency values for average calculation
    max_latency_history = 300  # Keep same amount as FPS
    avg_total_latency = 0.0
    
    try:
        if mode == "Viewer":
            from viewer import StereoWindow
            # Get initial frame to determine window size (block until first frame arrives)
            rgb, depth, capture_start_time = depth_q.get()
            import torch
            if isinstance(rgb, torch.Tensor):
                w, h = rgb.shape[2], rgb.shape[1] # CUDA Tensor, (C, H, W)
            else:
                w, h = rgb.shape[1], rgb.shape[0] # (H, W, C)
            if DISPLAY_MODE == "Full-SBS":
                w = 2 * w
            if not STREAM_MODE:
                # For local viewer only
                h = int(1280 / w * h)
                w = 1280
                
            window = StereoWindow(
                capture_mode=CAPTURE_MODE, 
                monitor_index=MONITOR_INDEX, 
                ipd=IPD, depth_ratio=DEPTH_STRENGTH, 
                convergence=CONVERGENCE, 
                display_mode=DISPLAY_MODE, 
                fill_16_9=FILL_16_9, 
                show_fps=SHOW_FPS, 
                use_3d=USE_3D_MONITOR, 
                fix_aspect=FIX_VIEWER_ASPECT, 
                stream_mode=STREAM_MODE, 
                lossless_scaling=LOSSLESS_SCALING_SUPPORT, 
                specify_display=STEREO_DISPLAY_SELECTION, 
                stereo_display_index=STEREO_DISPLAY_INDEX, 
                frame_size=(w,h),
                use_cuda=USE_CUDART,
                cuda_device_id=DEVICE_ID)

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
            
            # Process the first frame we already retrieved
            render_start_time = time.perf_counter()
            window.update_frame(rgb, depth, current_fps, 0.0)  # initial latency unknown
            render_latency = time.perf_counter() - render_start_time
            total_latency = (render_start_time - capture_start_time) + render_latency
            thread_latencies['render'] = render_latency
            thread_latencies['total'] = total_latency
            
            # Main render loop
            next_render_time = time.perf_counter()
            while (not glfw.window_should_close(window.window) and 
                   not shutdown_event.is_set()):

                try:
                    # Get next frame (already processed + depth)
                    rgb, depth, capture_start_time = depth_q.get(timeout=0.001)
                    
                    # Render
                    render_start_time = time.perf_counter()
                    window.update_frame(rgb, depth, current_fps, total_latency)
                    
                    if STREAM_MODE == "MJPEG":
                        frame = window.capture_glfw_image()
                        streamer.set_frame(frame)
                    
                    # Calculate FPS and latencies
                    frame_count += 1
                    total_frames += 1
                    current_time = time.perf_counter()
                    render_latency = current_time - render_start_time
                    thread_latencies['render'] = render_latency
                    total_latency = (current_time - capture_start_time)
                    thread_latencies['total'] = total_latency
                    current_latency = total_latency
                    
                    # Store latency value for average calculation
                    latency_history.append(total_latency)
                    if len(latency_history) > max_latency_history:
                        latency_history.pop(0)
                    
                    # Update FPS every second
                    if current_time - last_time >= 1.0:
                        # Calculate current FPS
                        current_fps = frame_count / (current_time - last_time)
                        
                        # Store FPS value for statistics
                        fps_values.append(current_fps)
                        
                        # Limit FPS history to last 5 seconds worth of data
                        if len(fps_values) > max_fps_history:
                            fps_values.pop(0)
                        
                        # Update FPS and latency statistics every 5 seconds
                        if current_time - last_fps_update_time >= fps_update_interval:
                            # Calculate average FPS
                            avg_fps = sum(fps_values) / len(fps_values)
                            if fps_values and len(fps_values) >= 20:  # Need at least 10 samples
                                
                                # Calculate average of FPS below 1% percentile
                                # Sort FPS values in ascending order
                                sorted_fps = sorted(fps_values)
                                
                                # Calculate 1% percentile index
                                one_percent_index = int(len(sorted_fps) * 0.1)
                                
                                # Ensure we have at least 1 sample in the lowest 1%
                                if one_percent_index == 0 and len(sorted_fps) > 0:
                                    one_percent_index = 1
                                
                                # Get FPS values below 1% percentile
                                fps_below_1_percent = sorted_fps[:one_percent_index]
                                
                                # Calculate average of FPS below 1% percentile
                                if fps_below_1_percent:
                                    low_fps_1_percent_avg = sum(fps_below_1_percent) / len(fps_below_1_percent)
                                else:
                                    low_fps_1_percent_avg = sorted_fps[0] if sorted_fps else 0.0
                            
                            # Calculate average latency
                            if latency_history:
                                avg_total_latency = sum(latency_history) / len(latency_history)
                            
                            # Reset for next calculation
                            last_fps_update_time = current_time
                        
                        frame_count = 0
                        last_time = current_time
                        
                        # Create window title with ALL detailed FPS and latency statistics
                        if SHOW_FPS:
                            title_text = (
                                f"{current_fps:.1f}FPS | "
                                f"Avg: {avg_fps:.1f} | "
                                f"1% Low Avg: {low_fps_1_percent_avg:.1f} | "
                                f"Latency: {total_latency*1000:.0f}ms | "
                                f"Avg Latency: {avg_total_latency*1000:.0f}ms "
                                f"(Capture:{thread_latencies['capture']*1000:.0f}ms "
                                f"Resize:{thread_latencies['resize']*1000:.0f}ms "
                                f"Depth:{thread_latencies['depth']*1000:.0f}ms "
                                f"Render:{render_latency*1000:.0f}ms)"
                            )
                        else:
                            title_text = (
                                f"{current_fps:.0f}FPS "
                                f"{total_latency*1000:.0f}ms"
                            )
                        
                        if STREAM_MODE == "MJPEG":
                            print(title_text)
                        
                        # Set window title with detailed statistics
                        glfw.set_window_title(window.window, f"Stereo Viewer {title_text}")
                except queue.Empty:
                    pass
                now = time.perf_counter()

                if not USE_3D_MONITOR and now < next_render_time:
                    time.sleep(next_render_time - now)

                if not USE_3D_MONITOR:
                    next_render_time += TIME_SLEEP

                window.render()
                glfw.swap_buffers(window.window)
                glfw.poll_events()
            
            glfw.terminate()
            
        else:
            from depth import make_sbs, DEVICE_INFO
            from streamer import MJPEGStreamer

            streamer = MJPEGStreamer(port=STREAM_PORT, fps=FPS, quality=STREAM_QUALITY)
            streamer.start()
            
            print(f"[Main] Legacy Streamer Started")
            
            # FPS and latency tracking for legacy mode
            fps_values = []
            max_fps_history = 300
            avg_fps = 0.0
            low_fps_1_percent_avg = float('inf')
            last_fps_update_time = time.perf_counter()
            
            while not shutdown_event.is_set():
                try:
                    # Fix for unstable dml runtime error
                    rgb, depth, _ = depth_q.get(timeout=TIME_SLEEP)
                    sbs = make_sbs(rgb, depth, ipd_uv=IPD, depth_ratio=DEPTH_STRENGTH, convergence=CONVERGENCE, display_mode=DISPLAY_MODE, fill_16_9=FILL_16_9, fps=current_fps)
                    streamer.set_frame(sbs)
                    
                    # Calculate FPS
                    frame_count += 1
                    current_time = time.perf_counter()
                    if current_time - last_time >= 1.0:
                        current_fps = frame_count / (current_time - last_time)
                        
                        # Store FPS value for statistics
                        fps_values.append(current_fps)
                        if len(fps_values) > max_fps_history:
                            fps_values.pop(0)
                        
                        # Update FPS statistics every 5 seconds
                        if current_time - last_fps_update_time >= fps_update_interval:
                            if fps_values and len(fps_values) >= 10:
                                # Calculate average FPS
                                avg_fps = sum(fps_values) / len(fps_values)
                                
                                # Calculate average of FPS below 1% percentile
                                sorted_fps = sorted(fps_values)
                                one_percent_index = int(len(sorted_fps) * 0.01)
                                
                                if one_percent_index == 0 and len(sorted_fps) > 0:
                                    one_percent_index = 1
                                
                                fps_below_1_percent = sorted_fps[:one_percent_index]
                                
                                if fps_below_1_percent:
                                    low_fps_1_percent_avg = sum(fps_below_1_percent) / len(fps_below_1_percent)
                                else:
                                    low_fps_1_percent_avg = sorted_fps[0] if sorted_fps else 0.0
                            
                            last_fps_update_time = current_time
                        
                        frame_count = 0
                        last_time = current_time
                        print(f"{current_fps:.1f} FPS | Avg: {avg_fps:.1f} | 1% Low Avg: {low_fps_1_percent_avg:.1f}")
                            
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
            overall_avg_fps = total_frames / total_time if total_time > 0 else 0
            print(f"Overall Average FPS: {overall_avg_fps:.2f}")
            if fps_values:
                print(f"Recent Average FPS: {avg_fps:.1f}")
                print(f"Recent 1% Low Average FPS: {low_fps_1_percent_avg:.1f}")
        print(f"[Main] Stopped")

if __name__ == "__main__":
    main(mode=RUN_MODE)
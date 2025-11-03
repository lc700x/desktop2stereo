import glfw
import subprocess
import re
import time

STREAM_KEY = 'live'
FPS = 60
AUDIO_DELAY = -0.15
CRF = 20

from AppKit import NSScreen

def get_scale(monitor_index):
    """Get the Retina scale factor for a specific monitor"""
    screens = NSScreen.screens()
    if monitor_index < len(screens):
        return screens[monitor_index].backingScaleFactor()
    return 2.0  # Default to 2x for Retina displays

def get_monitor_index_for_window(window):
    window_x, window_y = glfw.get_window_pos(window)
    monitors = glfw.get_monitors()

    for i, monitor in enumerate(monitors):
        mx, my = glfw.get_monitor_pos(monitor)
        mode = glfw.get_video_mode(monitor)
        mw, mh = mode.size.width, mode.size.height

        if mx <= window_x < mx + mw and my <= window_y < my + mh:
            return i
    return 0

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

def capture_window(window, output_file="stereo_viewer_capture.mp4"):
    x, y = glfw.get_window_pos(window)
    width, height = glfw.get_window_size(window)
    

    monitor_index = get_monitor_index_for_window(window)
    scale_factor = get_scale(monitor_index)
    screen_name = f"Capture screen {monitor_index}"
    screen_index = get_device_index(screen_name, "video")
    audio_index = get_device_index("BlackHole 2ch", "audio")

    if screen_index is None:
        raise Exception(f"Could not find screen index for {screen_name}")
    if audio_index is None:
        raise Exception("Could not find audio device index for 'BlackHole 2ch'")

    ffmpeg = subprocess.Popen([
            "./rtmp/mac/ffmpeg",
            "-itsoffset", str(AUDIO_DELAY),
            "-f", "avfoundation",
            "-rtbufsize", "256M",
            "-framerate", "60",
            "-i", f"{screen_index}:{audio_index}",
            "-filter_complex",
            f"[0:v]fps={FPS},crop={int(width*scale_factor)}:{int(height*scale_factor)}:{int(x*scale_factor)}:{int(y*scale_factor)},scale=iw:trunc(ih/2)*2,format=uyvy422[v];[0:a]aresample=async=1[a]",
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
        ], stdout=subprocess.PIPE)


def main():
    if not glfw.init():
        raise Exception("GLFW initialization failed")

    window = glfw.create_window(800, 600, "Stereo Viewer", None, None)
    if not window:
        glfw.terminate()
        raise Exception("Failed to create window")

    glfw.make_context_current(window)

    print("Window created. Starting capture in 3 seconds...")
    time.sleep(3)

    capture_window(window)

    # glfw.destroy_window(window)
    # glfw.terminate()

if __name__ == "__main__":
    main()
import threading
import queue
import glfw
import time
import os
import platform

from capture import DesktopGrabber
from depth import settings, predict_depth, process, DEVICE_INFO, MODEL_ID
from viewer import StereoWindow

MONITOR_INDEX = settings["Monitor Index"]
OUTPUT_RESOLUTION = settings["Output Resolution"]
DISPLAY_MODE = settings["Display Mode"]
SHOW_FPS = settings["Show FPS"]
FPS = settings["FPS"]
DEPTH_STRENTH = settings["Depth Strength"]
TIME_SLEEP = 1.0 / FPS

# Number of regions (sub-tasks) to split the screen into.
# You can change this value or expose it in settings. Default to 4 if not present.
NUM_REGIONS = settings.get("Num Capture Regions", 4)

# If you want to pin region workers to CPU cores, set CORE_PINNING = True.
# Otherwise set to False to let the OS schedule threads.
CORE_PINNING = settings.get("Core Pinning", True)

# Build a core_map: region_index -> core_id (or None). If CORE_PINNING is False, leave None.
def build_core_map(num_regions):
    if not CORE_PINNING:
        return [None] * num_regions
    cpu_count = os.cpu_count() or 1
    # On platforms where thread affinity helpers are not implemented (e.g., macOS) we won't pin.
    if platform.system() not in ("Linux", "Windows"):
        return [None] * num_regions
    # Simple round-robin map of regions to cores (0..cpu_count-1)
    core_map = []
    for i in range(num_regions):
        core_map.append(i % cpu_count)
    return core_map

CORE_MAP = build_core_map(NUM_REGIONS)

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
    """
    Create a single DesktopGrabber configured for multiple regions and continuously
    capture frames in a background thread. The DesktopGrabber will internally
    split the monitor into NUM_REGIONS regions and capture them in parallel.
    """
    # Use DesktopGrabber with num_regions and core_map options.
    # The DesktopGrabber implementation is expected to accept:
    #    DesktopGrabber(..., num_regions=num_regions, core_map=core_map)
    #
    # If your DesktopGrabber uses a context manager, you can instead use 'with'.
    cap = DesktopGrabber(
        monitor_index=MONITOR_INDEX,
        output_resolution=OUTPUT_RESOLUTION,
        fps=FPS,
        num_regions=NUM_REGIONS,
        core_map=CORE_MAP,
    )

    try:
        while True:
            frame_raw, size = cap.grab()
            put_latest(raw_q, (frame_raw, size))
    finally:
        # Ensure resources are cleaned on thread shutdown (if ever)
        try:
            cap.close()
        except Exception:
            pass


def process_loop():
    while True:
        try:
            frame_raw, size = raw_q.get(timeout=TIME_SLEEP)
        except queue.Empty:
            continue
        frame_rgb = process(frame_raw, size)
        put_latest(proc_q, frame_rgb)


def depth_loop():
    while True:
        try:
            frame_rgb = proc_q.get(timeout=TIME_SLEEP)
        except queue.Empty:
            continue
        depth = predict_depth(frame_rgb)
        put_latest(depth_q, (frame_rgb, depth))


def main():
    print(f"{DEVICE_INFO}")
    print(f"Model: {MODEL_ID}")
    print(f"Capture regions: {NUM_REGIONS}  Core pinning: {CORE_PINNING}  Core map: {CORE_MAP}")

    # Start capture and processing threads
    # capture_loop holds the DesktopGrabber instance for its lifetime.
    threading.Thread(target=capture_loop, daemon=True).start()
    threading.Thread(target=process_loop, daemon=True).start()
    threading.Thread(target=depth_loop, daemon=True).start()

    window = StereoWindow(depth_ratio=DEPTH_STRENTH, display_mode=DISPLAY_MODE)
    frame_rgb, depth = None, None

    # FPS calculation variables
    frame_count = 0
    last_time = time.time()
    fps = 0

    try:
        while not glfw.window_should_close(window.window):
            try:
                # Get latest frame, or skip update
                frame_rgb, depth = depth_q.get_nowait()
                window.update_frame(frame_rgb, depth)
                if SHOW_FPS:
                    frame_count += 1
                    current_time = time.time()
                    if current_time - last_time >= 1.0:  # Update every second
                        fps = frame_count / (current_time - last_time)
                        frame_count = 0
                        last_time = current_time
                        # Update window title with Depth Strength and FPS
                        glfw.set_window_title(window.window, f"Stereo Viewer | FPS: {fps:.1f} | depth: {window.depth_ratio:.1f}")
            except queue.Empty:
                pass  # Reuse previous frame if none available

            window.render()
            glfw.swap_buffers(window.window)
            glfw.poll_events()
    finally:
        # Terminate GLFW and let daemon threads exit when process terminates.
        glfw.terminate()


if __name__ == "__main__":
    main()

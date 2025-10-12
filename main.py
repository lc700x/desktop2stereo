import threading
import queue
import glfw
import time
from utils import OUTPUT_RESOLUTION, DISPLAY_MODE, SHOW_FPS, FPS, IPD, DEPTH_STRENGTH, RUN_MODE, STREAM_PORT, STREAM_QUALITY, DML_STREAM_STABLE
from capture import DesktopGrabber
from depth import process, predict_depth
import torch

# Constants
FRAME_INTERVAL = 0.5 / FPS

def interpolate_depth_gpu(prev_depth, next_depth, alpha=0.5, device="cuda"):
    """GPU linear interpolation for depth only"""
    if not prev_depth.is_cuda:
        prev_depth = prev_depth.to(device)
    if not next_depth.is_cuda:
        next_depth = next_depth.to(device)
    return (1 - alpha) * prev_depth + alpha * next_depth

# Queues with appropriate buffering
raw_q = queue.Queue(maxsize=2)
proc_q = queue.Queue(maxsize=2)
depth_q = queue.Queue(maxsize=2)
interp_q = queue.Queue(maxsize=3)  # Need space for original + interpolated frames (used in both viewer and streamer modes)

def put_latest(q, item):
    """Thread-safe queue put with overwrite if full"""
    try:
        if q.full():
            q.get_nowait()
        q.put_nowait(item)
    except queue.Empty:
        pass

def capture_loop():
    """High-speed capture thread"""
    cap = DesktopGrabber(output_resolution=OUTPUT_RESOLUTION, fps=FPS)
    while True:
        try:
            frame_raw, size = cap.grab()
            put_latest(raw_q, (frame_raw, size, time.perf_counter()))
        except Exception:
            time.sleep(0.001)

def process_loop():
    """Frame processing thread"""
    while True:
        try:
            frame_raw, size, timestamp = raw_q.get(timeout=FRAME_INTERVAL)
            frame_rgb = process(frame_raw, size)
            put_latest(proc_q, (frame_rgb, timestamp))
        except queue.Empty:
            continue

def main(mode="Viewer"):
    # Start all processing threads
    threading.Thread(target=capture_loop, daemon=True).start()
    threading.Thread(target=process_loop, daemon=True).start()

    # Performance tracking
    frame_count = 0
    last_time = time.perf_counter()
    current_fps = 0
    total_frames = 0
    start_time = time.perf_counter()

    try:
        if mode == "Viewer":
            from viewer import StereoWindow
            def depth_loop():
                """Depth prediction thread"""
                while True:
                    try:
                        frame_rgb, timestamp = proc_q.get(timeout=FRAME_INTERVAL)
                        depth = predict_depth(frame_rgb)
                        put_latest(depth_q, (frame_rgb, depth, timestamp))
                    except queue.Empty:
                        continue

            def frame_generation_loop():
                """Frame interpolation thread (depth only)"""
                prev_depth, prev_time = None, None
                
                while True:
                    try:
                        # Get latest frame with timestamp
                        rgb, depth, frame_time = depth_q.get(timeout=FRAME_INTERVAL)
                        
                        # Generate interpolated frame if we have previous frame
                        if prev_depth is not None:
                            # Calculate proper interpolation weight
                            time_between_frames = frame_time - prev_time
                            alpha = min(0.5, (time.perf_counter() - prev_time) / time_between_frames)
                            
                            # Generate interpolated depth (using current RGB frame)
                            depth_mid = interpolate_depth_gpu(
                                prev_depth, depth, 
                                alpha=alpha,
                                device=depth.device
                            )
                            put_latest(interp_q, (rgb, depth_mid, 'interpolated'))
                        
                        # Queue the original frame
                        put_latest(interp_q, (rgb, depth, 'original'))
                        prev_depth, prev_time = depth, frame_time
                        
                    except queue.Empty:
                        continue
            threading.Thread(target=depth_loop, daemon=True).start()
            threading.Thread(target=frame_generation_loop, daemon=True).start()
            
            window = StereoWindow(ipd=IPD, depth_ratio=DEPTH_STRENGTH, 
                                display_mode=DISPLAY_MODE, show_fps=SHOW_FPS)
            print("[Main] Viewer Started")
            
            last_frame_time = time.perf_counter()
            while not glfw.window_should_close(window.window):
                # Frame pacing for target FPS
                now = time.perf_counter()
                if now - last_frame_time >= FRAME_INTERVAL:
                    try:
                        rgb, depth, _ = interp_q.get_nowait()
                        window.update_frame(rgb, depth)
                        
                        # FPS calculation
                        frame_count += 1
                        total_frames += 1
                        if now - last_time >= 1.0:
                            current_fps = frame_count / (now - last_time)
                            frame_count = 0
                            last_time = now
                            if SHOW_FPS:
                                title = f"Stereo Viewer | FPS: {current_fps:.1f}"
                                glfw.set_window_title(window.window, title)
                        
                        last_frame_time = now
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
            
            # Modified to support frame generation
            def depth_loop():
                """Depth prediction thread"""
                while True:
                    try:
                        frame_rgb, timestamp = proc_q.get(timeout=FRAME_INTERVAL)
                        depth, rgb = predict_depth(frame_rgb, return_tuple=True)
                        put_latest(depth_q, (rgb, depth, timestamp))  # Include timestamp for interpolation
                    except queue.Empty:
                        continue

            # NEW: Frame generation loop for streamer mode
            def frame_generation_loop():
                """Frame interpolation thread for streamer mode (depth only)"""
                prev_depth, prev_time = None, None
                
                while True:
                    try:
                        # Get latest frame with timestamp
                        rgb, depth, frame_time = depth_q.get(timeout=FRAME_INTERVAL)
                        
                        # Generate interpolated frame if we have previous frame
                        if prev_depth is not None:
                            # Calculate proper interpolation weight
                            time_between_frames = frame_time - prev_time
                            alpha = min(0.5, (time.perf_counter() - prev_time) / time_between_frames)
                            
                            # Generate interpolated depth (using current RGB frame)
                            depth_mid = interpolate_depth_gpu(
                                prev_depth, depth, 
                                alpha=alpha,
                                device=depth.device
                            )
                            put_latest(interp_q, (rgb, depth_mid, 'interpolated'))
                        
                        # Queue the original frame
                        put_latest(interp_q, (rgb, depth, 'original'))
                        prev_depth, prev_time = depth, frame_time
                        
                    except queue.Empty:
                        continue

            # Modified SBS generation to handle interpolated frames
            if BOOST:
                def make_output(rgb, depth):
                    return make_sbs(rgb, depth, ipd_uv=IPD, depth_ratio=DEPTH_STRENGTH, display_mode=DISPLAY_MODE, fps=current_fps)
            else:
                sbs_q = queue.Queue(maxsize=1)
                def make_output(rgb, depth):
                    return (rgb, depth)
                
                def sbs_loop():
                    while True:
                        try:
                            rgb, depth, _ = interp_q.get(timeout=FRAME_INTERVAL)  # Use interp_q
                            sbs = make_sbs(rgb, depth, ipd_uv=IPD, depth_ratio=DEPTH_STRENGTH, display_mode=DISPLAY_MODE, fps=current_fps)
                            put_latest(sbs_q, sbs)
                        except queue.Empty:
                            continue

            # Start depth and frame generation threads
            threading.Thread(target=depth_loop, daemon=True).start()
            threading.Thread(target=frame_generation_loop, daemon=True).start()  # NEW: Start frame generation
            if not BOOST:
                threading.Thread(target=sbs_loop, daemon=True).start()
            
            streamer = MJPEGStreamer(port=STREAM_PORT, fps=FPS, quality=STREAM_QUALITY)
            streamer.start()
            print(f"[Main] Streamer Started")
            
            while True:
                try:
                    if BOOST:
                        rgb, depth, _ = interp_q.get(timeout=FRAME_INTERVAL)  # Use interp_q
                        sbs = make_output(rgb, depth)
                    else:
                        sbs = sbs_q.get(timeout=FRAME_INTERVAL)
                    
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
        print("\n[Main] Shutting down...")
    finally:
        if 'streamer' in locals():
            streamer.stop()
        if 'window' in locals():
            glfw.terminate()
        
        total_time = time.perf_counter() - start_time
        avg_fps = total_frames / total_time if total_time > 0 else 0
        print(f"Average FPS: {avg_fps:.2f}")

if __name__ == "__main__":
    main(mode=RUN_MODE)
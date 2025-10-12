import threading
import queue
import glfw
import time
from utils import OUTPUT_RESOLUTION, DISPLAY_MODE, SHOW_FPS, FPS, IPD, DEPTH_STRENGTH, RUN_MODE, STREAM_PORT, STREAM_QUALITY, DML_BOOST
from capture import DesktopGrabber
from depth import process, predict_depth

# Constants
FRAMES = 2  # Number of interpolated frames to generate between original frames
FRAME_INTERVAL = 1 / (FPS * (FRAMES + 1))  # Adjusted interval for interpolated frames

def interpolate_depth_gpu(prev_depth, next_depth, alpha=0.5, device="cuda"):
    try:
        """GPU linear interpolation for depth only"""
        if not prev_depth.is_cuda:
            prev_depth = prev_depth.to(device)
        if not next_depth.is_cuda:
            next_depth = next_depth.to(device)
        return (1 - alpha) * prev_depth + alpha * next_depth
    except RuntimeError:
        return prev_depth

# Queues with appropriate buffering
raw_q = queue.Queue(maxsize=1)
proc_q = queue.Queue(maxsize=1)
depth_q = queue.Queue(maxsize=2)
interp_q = queue.Queue(maxsize=FRAMES + 2)  # Space for original + multiple interpolated frames
sbs_q = queue.Queue(maxsize=1)

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
                prev_rgb, prev_depth, prev_time = None, None, None
                
                while True:
                    try:
                        # Get latest frame with timestamp
                        rgb, depth, frame_time = depth_q.get(timeout=FRAME_INTERVAL)
                        
                        # Generate interpolated frames if we have previous frame
                        if prev_depth is not None:
                            time_between_frames = frame_time - prev_time
                            
                            # Generate multiple interpolated frames
                            for i in range(1, FRAMES + 1):
                                alpha = i / (FRAMES + 1)
                                weight = min(1.0, (time.perf_counter() - prev_time) / time_between_frames * (FRAMES + 1))
                                alpha = min(alpha, weight)
                                
                                # Generate interpolated depth (using current RGB frame)
                                depth_mid = interpolate_depth_gpu(
                                    prev_depth, depth, 
                                    alpha=alpha,
                                    device=depth.device
                                )
                                put_latest(interp_q, (rgb, depth_mid, f'interpolated_{i}'))
                        
                        # Queue the original frame
                        put_latest(interp_q, (rgb, depth, 'original'))
                        prev_rgb, prev_depth, prev_time = rgb, depth, frame_time
                        
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
            BOOST = (not "DirectML" in DEVICE_INFO) or DML_BOOST
            from streamer import MJPEGStreamer
            
            if BOOST:
                def depth_loop():
                    """Depth prediction thread"""
                    while True:
                        try:
                            frame_rgb, timestamp = proc_q.get(timeout=FRAME_INTERVAL)
                            depth, rgb = predict_depth(frame_rgb, return_tuple=True)
                            put_latest(depth_q, (rgb, depth, timestamp))
                        except queue.Empty:
                            continue

                def frame_generation_loop():
                    """Frame interpolation thread for streamer mode (depth only)"""
                    prev_rgb, prev_depth, prev_time = None, None, None
                    
                    while True:
                        try:
                            # Get latest frame with timestamp
                            rgb, depth, frame_time = depth_q.get(timeout=FRAME_INTERVAL)
                            
                            # Generate multiple interpolated frames
                            if prev_depth is not None:
                                time_between_frames = frame_time - prev_time
                                
                                for i in range(1, FRAMES + 1):
                                    alpha = i / (FRAMES + 1)
                                    weight = min(1.0, (time.perf_counter() - prev_time) / time_between_frames * (FRAMES + 1))
                                    alpha = min(alpha, weight)
                                    
                                    depth_mid = interpolate_depth_gpu(
                                        prev_depth, depth, 
                                        alpha=alpha,
                                        device=depth.device
                                    )
                                    put_latest(interp_q, (rgb, depth_mid, f'interpolated_{i}'))
                            
                            # Queue the original frame
                            put_latest(interp_q, (rgb, depth, 'original'))
                            prev_rgb, prev_depth, prev_time = rgb, depth, frame_time
                            
                        except queue.Empty:
                            continue
                
                def sbs_loop():
                    while True:
                        try:
                            rgb, depth, _ = interp_q.get(timeout=FRAME_INTERVAL)
                        except queue.Empty:
                            continue
                        sbs = make_sbs(rgb, depth, ipd_uv=IPD, depth_ratio=DEPTH_STRENGTH, display_mode=DISPLAY_MODE, fps=current_fps)
                        put_latest(sbs_q, sbs)
                
                threading.Thread(target=depth_loop, daemon=True).start()
                threading.Thread(target=frame_generation_loop, daemon=True).start()
                threading.Thread(target=sbs_loop, daemon=True).start()
                streamer = MJPEGStreamer(port=STREAM_PORT, fps=FPS, quality=STREAM_QUALITY)
                streamer.start()
                print(f"[Main] Streamer Started")
                
                while True:
                    try:
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
            else:
                def sbs_loop():
                    while True:
                        try:
                            frame_rgb, _ = proc_q.get(timeout=FRAME_INTERVAL)
                            depth, rgb = predict_depth(frame_rgb, return_tuple=True)
                        except queue.Empty:
                            continue
                        sbs = make_sbs(rgb, depth, ipd_uv=IPD, depth_ratio=DEPTH_STRENGTH, display_mode=DISPLAY_MODE, fps=current_fps)
                        put_latest(sbs_q, sbs)
                
                threading.Thread(target=sbs_loop, daemon=True).start()
                streamer = MJPEGStreamer(port=STREAM_PORT, fps=FPS, quality=STREAM_QUALITY)
                streamer.start()
                print(f"[Main] Streamer Started")
                
                while True:
                    try:
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
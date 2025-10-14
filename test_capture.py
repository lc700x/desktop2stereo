import cv2, time
from time import sleep
from utils import WINDOW_TITLE, CAPTURE_MODE, MONITOR_INDEX, OUTPUT_RESOLUTION, FPS, SHOW_FPS
from capture import DesktopGrabber


def main():
    grabber = DesktopGrabber(
        output_resolution=OUTPUT_RESOLUTION,   # scale height
        fps=FPS,
        window_title=WINDOW_TITLE,
        capture_mode=CAPTURE_MODE,
        monitor_index=MONITOR_INDEX
    )
    frame_count = 0
    last_time = time.perf_counter()
    current_fps = None
    total_frames = 0
    while True:
        frame, h = grabber.grab()
        if frame is None:
            print("No frame captured")
            sleep(0.1)
            continue
        cv2.imshow("Capture Test", frame)
        
        if SHOW_FPS:
            frame_count += 1
            total_frames += 1
            current_time = time.perf_counter()
            if current_time - last_time >= 1.0:
                current_fps = frame_count / (current_time - last_time)
                frame_count = 0
                last_time = current_time
                print(current_fps)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC key to exit
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

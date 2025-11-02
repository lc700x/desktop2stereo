from wincam import DXCamera
import time
import cv2, ctypes

ctypes.windll.shcore.SetProcessDpiAwareness(2)

frame_count = 0
last_time = time.time()

# Initialize DXCamera for monitor index 1
capture = DXCamera(0, 0, 3840, 2160, fps=60)

while True:
    frame, timestamp = capture.get_bgr_frame()
    cv2.imshow("Wincam Viewer", frame)
    if cv2.waitKey(1) == 27:  # ESC key
        break
    frame_count += 1
    elapsed = time.time() - last_time
    if elapsed >= 1.0:
        print(f"FPS: {frame_count}")
        frame_count = 0
        last_time = time.time()

cv2.destroyAllWindows()

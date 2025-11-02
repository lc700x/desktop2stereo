import bettercam
import time
import cv2
import ctypes
import numpy as np

# --- Windows DPI awareness (avoid blurry capture) ---
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(2)
except Exception:
    pass

# --- Create BetterCam capture object ---
# For single monitor, just call without args. For multi-monitor, you can specify:
# bettercam.create(output_color="RGB", monitor=1)
camera = bettercam.create(output_color="RGB")

# --- Start capture ---
camera.start()

frame_count = 0
last_time = time.time()

print("[INFO] BetterCam started. Press ESC to quit.")

while True:
    frame = camera.grab()
    # Some versions use get_latest_frame(), others just grab()

    if frame is None:
        # Avoid spinning too fast when no frame yet
        time.sleep(0.001)
        continue

    # Ensure frame is numpy array with correct color format
    if isinstance(frame, (bytes, bytearray)):
        frame = np.frombuffer(frame, dtype=np.uint8).reshape(camera.height, camera.width, 3)

    cv2.imshow("BetterCam Viewer", frame)

    key = cv2.waitKey(1)
    if key == 27:  # ESC
        break

    # FPS counter
    frame_count += 1
    elapsed = time.time() - last_time
    if elapsed >= 1.0:
        print(f"FPS: {frame_count}")
        frame_count = 0
        last_time = time.time()

# --- Clean up ---
camera.stop()
cv2.destroyAllWindows()
print("[INFO] Stopped BetterCam.")

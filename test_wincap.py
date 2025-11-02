from windows_capture import WindowsCapture, Frame, InternalCaptureControl
import time
import cv2

frame_count = 0
last_time = time.time()

capture = WindowsCapture(
    cursor_capture=None,
    draw_border=None,
    monitor_index=1,
    window_name=None,
)

@capture.event
def on_frame_arrived(frame: Frame, _capture_control: InternalCaptureControl):
    global frame_count
    global last_time
    
    _frame_buffer = frame.frame_buffer
    cv2.imshow('Wincap Viewer', _frame_buffer)
    if cv2.waitKey(1) == 27:  # ESC key
        exit()
    frame_count += 1
    current_time = time.time()
    
    if current_time - last_time >= 1:
        fps = frame_count / (current_time - last_time)
        print(f"FPS: {fps:.2f}")
        frame_count = 0
        last_time = current_time

@capture.event
def on_closed():
    print("Capture Session Closed")


capture.start()
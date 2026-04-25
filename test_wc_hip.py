from wc_hip import WindowsCapture
from PIL import Image


def test_monitor_capture():
    print("\n--- Testing Monitor Capture ---")
    capture = WindowsCapture(monitor_index=1)
    frame_count = 0
    saved = False

    @capture.event
    def on_frame_arrived(frame, control):
        nonlocal frame_count, saved
        frame_count += 1
        if frame_count == 1:
            print(f"  [OK] First frame received: {frame.width}x{frame.height}")
        if not saved:
            rgba = frame.frame_buffer[..., [2, 1, 0, 3]]
            Image.fromarray(rgba.cpu().numpy()).save("captured_monitor.png")
            print("  [OK] Saved captured_monitor.png")
            saved = True
        if frame_count >= 10:
            control.stop()

    @capture.event
    def on_closed():
        print("  [OK] Capture session closed.")

    capture.start()


if __name__ == "__main__":
    test_monitor_capture()

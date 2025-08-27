import threading
import time, cv2
from windows_capture import WindowsCapture, Frame, InternalCaptureControl

class DesktopGrabber:
    def __init__(self, output_resolution=1080, fps=60, window_title=WINDOW_TITLE, capture_mode=CAPTURE_MODE):
        self.scaled_height = output_resolution
        self.fps = fps
        self.capture_mode = capture_mode
        self.window_title = window_title
        self.latest_frame = None
        self._lock = threading.Lock()

        if self.capture_mode != "Window":
            raise NotImplementedError("Monitor capture not implemented yet")

        if not self.window_title:
            raise ValueError("No window title specified for window capture")

        # Create capture object
        self.capture = WindowsCapture(window_name=self.window_title)

        # Register callback for frames
        @self.capture.event
        def on_frame_arrived(frame: Frame, capture_control: InternalCaptureControl):
            with self._lock:
                # Convert BGRA to RGB
                self.latest_frame = cv2.cvtColor(frame.frame_buffer, cv2.COLOR_BGRA2RGB)
                time.sleep(0.001)
                

        @self.capture.event
        def on_closed():
            print("Capture session closed")

        # Start capture in background thread
        self._thread = threading.Thread(target=self.capture.start, daemon=True)
        self._thread.start()

        # Wait until first frame arrives
        while self.latest_frame is None:
            time.sleep(0.001)

    def grab(self):
        with self._lock:
            if self.latest_frame is None:
                return None, self.scaled_height
            img_array = self.latest_frame
        return img_array, self.scaled_height

    def close(self):
        self.capture = None
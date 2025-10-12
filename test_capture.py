import cv2
from time import sleep
from utils import WINDOW_TITLE, CAPTURE_MODE, MONITOR_INDEX, OUTPUT_RESOLUTION, FPS
from capture import DesktopGrabber


def main():
    grabber = DesktopGrabber(
        output_resolution=OUTPUT_RESOLUTION,   # scale height
        fps=FPS,
        window_title=WINDOW_TITLE,
        capture_mode=CAPTURE_MODE,
        monitor_index=MONITOR_INDEX
    )

    while True:
        frame, h = grabber.grab()
        if frame is None:
            print("No frame captured")
            sleep(0.1)
            continue

        cv2.imshow("Capture Test", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC key to exit
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

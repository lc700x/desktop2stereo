import cv2
from time import sleep
from utils import WINDOW_TITLE, CAPTURE_MODE, MONITOR_INDEX
from capture import DesktopGrabber


def main():
    grabber = DesktopGrabber(
        output_resolution=720,   # scale height
        fps=30,
        window_title=WINDOW_TITLE,
        capture_mode=CAPTURE_MODE
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

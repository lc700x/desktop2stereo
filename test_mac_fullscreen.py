# Requires: pip install pyobjc-framework-Quartz
import time
import Quartz  # PyObjC binding for CoreGraphics

def send_ctrl_cmd_f():
    # macOS virtual keycode for the 'F' key (physical key 'F')
    KEY_F = 3

    # modifier flags: Control + Command
    flags = Quartz.kCGEventFlagMaskControl | Quartz.kCGEventFlagMaskCommand

    # key-down
    ev_down = Quartz.CGEventCreateKeyboardEvent(None, KEY_F, True)
    Quartz.CGEventSetFlags(ev_down, flags)
    Quartz.CGEventPost(Quartz.kCGHIDEventTap, ev_down)

    time.sleep(0.02)  # short hold

    # key-up
    ev_up = Quartz.CGEventCreateKeyboardEvent(None, KEY_F, False)
    Quartz.CGEventSetFlags(ev_up, flags)
    Quartz.CGEventPost(Quartz.kCGHIDEventTap, ev_up)

if __name__ == "__main__":
    send_ctrl_cmd_f()

import glfw
from OpenGL.GL import *
import numpy as np
import mss
import win32api
import win32con
import time

# =========================================================
# CONFIGURATION
# =========================================================

SOURCE_MONITOR_INDEX = 1   # monitor to mirror (1 = primary)
TARGET_MONITOR_INDEX = 2   # monitor to display mirror

EXIT_KEY = glfw.KEY_ESCAPE

# =========================================================
# SCREEN CAPTURE INIT
# =========================================================

sct = mss.mss()

if SOURCE_MONITOR_INDEX >= len(sct.monitors):
    raise Exception("Invalid SOURCE monitor index")

source_monitor = sct.monitors[SOURCE_MONITOR_INDEX]

SRC_X = source_monitor["left"]
SRC_Y = source_monitor["top"]
SRC_W = source_monitor["width"]
SRC_H = source_monitor["height"]

print("Mirroring monitor:")
print(source_monitor)

# =========================================================
# GLFW INIT
# =========================================================

if not glfw.init():
    raise Exception("GLFW init failed")

monitors = glfw.get_monitors()

if TARGET_MONITOR_INDEX > len(monitors):
    raise Exception("Invalid TARGET monitor index")

target_monitor = monitors[TARGET_MONITOR_INDEX - 1]

mode = glfw.get_video_mode(target_monitor)

window = glfw.create_window(
    mode.size.width,
    mode.size.height,
    "Local Display Mirror",
    target_monitor,
    None
)

if not window:
    glfw.terminate()
    raise Exception("GLFW window creation failed")

glfw.make_context_current(window)

WIN_W, WIN_H = mode.size.width, mode.size.height

print("Mirror window resolution:", WIN_W, WIN_H)

# =========================================================
# INPUT MODE CONFIG
# =========================================================

glfw.set_input_mode(
    window,
    glfw.CURSOR,
    glfw.CURSOR_DISABLED
)

if glfw.raw_mouse_motion_supported():
    glfw.set_input_mode(
        window,
        glfw.RAW_MOUSE_MOTION,
        glfw.TRUE
    )

# =========================================================
# COORDINATE SCALING
# =========================================================

def scale_coords(x, y):
    scaled_x = SRC_X + (x * SRC_W / WIN_W)
    scaled_y = SRC_Y + (y * SRC_H / WIN_H)

    return int(scaled_x), int(scaled_y)


def send_absolute_mouse(x, y):

    screen_w = win32api.GetSystemMetrics(0)
    screen_h = win32api.GetSystemMetrics(1)

    abs_x = int(x * 65535 / screen_w)
    abs_y = int(y * 65535 / screen_h)

    win32api.mouse_event(
        win32con.MOUSEEVENTF_MOVE |
        win32con.MOUSEEVENTF_ABSOLUTE,
        abs_x,
        abs_y
    )

# =========================================================
# INPUT CALLBACKS
# =========================================================

def cursor_callback(window, x, y):

    target_x, target_y = scale_coords(x, y)

    send_absolute_mouse(target_x, target_y)


def mouse_button_callback(window, button, action, mods):

    mapping = {

        glfw.MOUSE_BUTTON_LEFT: (
            win32con.MOUSEEVENTF_LEFTDOWN,
            win32con.MOUSEEVENTF_LEFTUP
        ),

        glfw.MOUSE_BUTTON_RIGHT: (
            win32con.MOUSEEVENTF_RIGHTDOWN,
            win32con.MOUSEEVENTF_RIGHTUP
        ),

        glfw.MOUSE_BUTTON_MIDDLE: (
            win32con.MOUSEEVENTF_MIDDLEDOWN,
            win32con.MOUSEEVENTF_MIDDLEUP
        )
    }

    if button not in mapping:
        return

    down_flag, up_flag = mapping[button]

    if action == glfw.PRESS:

        win32api.mouse_event(down_flag, 0, 0)

    elif action == glfw.RELEASE:

        win32api.mouse_event(up_flag, 0, 0)


def key_callback(window, key, scancode, action, mods):

    if key == EXIT_KEY and action == glfw.PRESS:

        glfw.set_window_should_close(window, True)
        return

    if action == glfw.PRESS:

        win32api.keybd_event(key, 0, 0, 0)

    elif action == glfw.RELEASE:

        win32api.keybd_event(
            key,
            0,
            win32con.KEYEVENTF_KEYUP,
            0
        )


glfw.set_cursor_pos_callback(window, cursor_callback)
glfw.set_mouse_button_callback(window, mouse_button_callback)
glfw.set_key_callback(window, key_callback)

# =========================================================
# OPENGL TEXTURE SETUP
# =========================================================

texture = glGenTextures(1)

glBindTexture(GL_TEXTURE_2D, texture)

glTexParameteri(
    GL_TEXTURE_2D,
    GL_TEXTURE_MIN_FILTER,
    GL_LINEAR
)

glTexParameteri(
    GL_TEXTURE_2D,
    GL_TEXTURE_MAG_FILTER,
    GL_LINEAR
)

glEnable(GL_TEXTURE_2D)

# =========================================================
# MAIN LOOP
# =========================================================

frame_counter = 0
start_time = time.time()

while not glfw.window_should_close(window):

    frame = np.array(
        sct.grab(source_monitor)
    )[:, :, :3]

    glClear(GL_COLOR_BUFFER_BIT)

    glTexImage2D(
        GL_TEXTURE_2D,
        0,
        GL_RGB,
        SRC_W,
        SRC_H,
        0,
        GL_BGR,
        GL_UNSIGNED_BYTE,
        frame
    )

    glBegin(GL_QUADS)

    glTexCoord2f(0, 0)
    glVertex2f(-1, 1)

    glTexCoord2f(1, 0)
    glVertex2f(1, 1)

    glTexCoord2f(1, 1)
    glVertex2f(1, -1)

    glTexCoord2f(0, 1)
    glVertex2f(-1, -1)

    glEnd()

    glfw.swap_buffers(window)
    glfw.poll_events()

    frame_counter += 1

    if frame_counter % 120 == 0:

        elapsed = time.time() - start_time

        fps = frame_counter / elapsed

        print("FPS:", round(fps, 2))

glfw.terminate()

print("Mirror stopped cleanly.")
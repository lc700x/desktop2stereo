import glfw
from OpenGL.GL import *

def main():
    # Initialize GLFW
    if not glfw.init():
        print("Failed to initialize GLFW")
        return

    # Create a windowed mode window and its OpenGL context
    window = glfw.create_window(800, 600, "Python GLFW Test", None, None)
    if not window:
        print("Failed to create GLFW window")
        glfw.terminate()
        return

    # Make the window's context current
    glfw.make_context_current(window)

    # Main loop
    while not glfw.window_should_close(window):
        # Set the viewport
        width, height = glfw.get_framebuffer_size(window)
        glViewport(0, 0, width, height)

        # Set background color (R, G, B, A)
        glClearColor(0.6, 0.3, 0.8, 1.0)
        glClear(GL_COLOR_BUFFER_BIT)

        # Swap front and back buffers
        glfw.swap_buffers(window)

        # Poll for and process events
        glfw.poll_events()

    glfw.terminate()

if __name__ == "__main__":
    main()

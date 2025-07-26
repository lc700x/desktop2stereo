import glfw
import moderngl
import numpy as np
import screeninfo
import os

# Set the HF_ENDPOINT environment variable
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# Verify the variable is set (optional)
print(f"HF_ENDPOINT is set to: {os.environ.get('HF_ENDPOINT')}")

VERTEX_SHADER = """
    #version 330
    in vec2 in_position;
    in vec2 in_uv;
    out vec2 uv;
    void main() {
        uv = in_uv;
        gl_Position = vec4(in_position, 0.0, 1.0);
    }
    """

FRAGMENT_SHADER = """
    #version 330
    in vec2 uv;
    out vec4 frag_color;

    uniform sampler2D tex_color;
    uniform sampler2D tex_depth;
    uniform float u_eye_offset;
    uniform float u_depth_strength;

    void main() {
        vec2 flipped_uv = vec2(uv.x, 1.0 - uv.y);
        float depth = texture(tex_depth, flipped_uv).r;

        // Shift: positive offset to right for right eye, left for left eye
        vec2 offset_uv = flipped_uv - vec2(u_eye_offset * depth * u_depth_strength, 0.0);
        offset_uv = clamp(offset_uv, 0.0, 1.0);

        frag_color = texture(tex_color, offset_uv);
    }
    """

class StereoWindow:
    """A window for displaying stereo images side-by-side with depth effect."""
    def __init__(self):
        self.window_size = (1280, 720)
        self.title = "Stereo SBS Viewer"
        self.ipd_uv = 0.064
        self.depth_strength = 0.1
        self._last_window_position = None
        self._last_window_size = None
        self._fullscreen = False
        
        # Initialize GLFW
        if not glfw.init():
            raise RuntimeError("Could not initialize GLFW")
        
        # Configure window
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.RESIZABLE, True)
        
        # Create window (start in windowed mode)
        self.window = glfw.create_window(*self.window_size, self.title, None, None)
        if not self.window:
            glfw.terminate()
            raise RuntimeError("Could not create window")
        
        # Position window on monitor 2 if available
        self.position_on_monitor(0)  # 0=primary, 1=secondary
        
        glfw.make_context_current(self.window)
        glfw.swap_interval(1) # Enable vsync
        self.ctx = moderngl.create_context()
        
        # Setup shaders and buffers
        self.prog = self.ctx.program(
            vertex_shader=VERTEX_SHADER,
            fragment_shader=FRAGMENT_SHADER
        )
        self.quad_vao = self.make_quad()
        self.color_tex = None
        self.depth_tex = None
        
        # Set callbacks
        glfw.set_key_callback(self.window, self.on_key_event)

    def position_on_monitor(self, monitor_index=0):
        """Position window on specified monitor (0-based index)"""
        monitors = screeninfo.get_monitors()
        if monitor_index < len(monitors):
            monitor = monitors[monitor_index]
            # Center window on monitor
            x = monitor.x + (monitor.width - self.window_size[0]) // 2
            y = monitor.y + (monitor.height - self.window_size[1]) // 2
            glfw.set_window_pos(self.window, x, y)

    def get_current_monitor(self):
        """Get the monitor that contains the window center"""
        monitors = glfw.get_monitors()
        if not monitors:
            return None
            
        # Get window center
        win_x, win_y = glfw.get_window_pos(self.window)
        win_w, win_h = glfw.get_window_size(self.window)
        window_center_x = win_x + win_w // 2
        window_center_y = win_y + win_h // 2
        
        # Find monitor containing window center
        for monitor in monitors:
            monitor_x, monitor_y = glfw.get_monitor_pos(monitor)
            vidmode = glfw.get_video_mode(monitor)
            if (monitor_x <= window_center_x < monitor_x + vidmode.size.width and
                monitor_y <= window_center_y < monitor_y + vidmode.size.height):
                return monitor
        return monitors[0]  # fallback to primary monitor

    def toggle_fullscreen(self):
        """Toggle fullscreen on current monitor"""
        current_monitor = self.get_current_monitor()
        if not current_monitor:
            return
            
        if not self._fullscreen:
            # Store current window state
            self._last_window_position = glfw.get_window_pos(self.window)
            self._last_window_size = glfw.get_window_size(self.window)
            
            # Get monitor video mode
            vidmode = glfw.get_video_mode(current_monitor)
            
            # Set fullscreen on current monitor
            glfw.set_window_monitor(
                self.window,
                current_monitor,
                0, 0,
                vidmode.size.width,
                vidmode.size.height,
                vidmode.refresh_rate
            )
            self._fullscreen = True
        else:
            # Exit fullscreen and restore window
            glfw.set_window_monitor(
                self.window,
                None,  # Windowed mode
                self._last_window_position[0],
                self._last_window_position[1],
                self._last_window_size[0],
                self._last_window_size[1],
                0
            )
            self._fullscreen = False
    
    def on_key_event(self, window, key, scancode, action, mods):
        if action == glfw.PRESS:
            if key == glfw.KEY_SPACE:
                self.toggle_fullscreen()
            elif key == glfw.KEY_ESCAPE:
                glfw.set_window_should_close(window, True)

    def make_quad(self):
        vertices = np.array([
            -1, -1, 0, 0,
             1, -1, 1, 0,
            -1,  1, 0, 1,
             1,  1, 1, 1,
        ], dtype='f4')
        vbo = self.ctx.buffer(vertices)
        return self.ctx.vertex_array(
            self.prog, [(vbo, '2f 2f', 'in_position', 'in_uv')]
        )

    def update_frame(self, rgb, depth):
        h, w = depth.shape

        # Normalize depth to [0, 1]
        depth_min = depth.min()
        depth_max = depth.max()
        # print(f"Depth min: {depth_min:.4f}, max: {depth_max:.4f}")
        depth = (depth - depth_min) / (depth_max - depth_min + 1e-8)

        if self.color_tex is None or self.color_tex.size != (w, h):
            if self.color_tex:
                self.color_tex.release()
            if self.depth_tex:
                self.depth_tex.release()
            self.color_tex = self.ctx.texture((w, h), 3, dtype='f1')
            self.depth_tex = self.ctx.texture((w, h), 1, dtype='f4')
            self.prog['tex_color'].value = 0
            self.prog['tex_depth'].value = 1

        self.color_tex.write(rgb.tobytes())
        self.depth_tex.write(depth.tobytes())

    def render(self):
        self.ctx.clear(0.1, 0.1, 0.1)
        if not self.color_tex or not self.depth_tex:
            return

        width, height = glfw.get_window_size(self.window)
        self.color_tex.use(location=0)
        self.depth_tex.use(location=1)

        # Left eye
        self.ctx.viewport = (0, 0, width // 2, height)
        self.prog['u_eye_offset'].value = -self.ipd_uv / 2.0
        self.prog['u_depth_strength'].value = self.depth_strength
        self.quad_vao.render(moderngl.TRIANGLE_STRIP)

        # Right eye
        self.ctx.viewport = (width // 2, 0, width // 2, height)
        self.prog['u_eye_offset'].value = self.ipd_uv / 2.0
        self.quad_vao.render(moderngl.TRIANGLE_STRIP)
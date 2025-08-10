# viewer.py
import glfw
import moderngl
import numpy as np

# Get OS name and settings
from depth import OS_NAME, settings

IPD = settings["IPD"]

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

        // Invert depth so near=1 shifts more, far=0 shifts less
        float depth_inv = 1.0 - depth;

        vec2 offset_uv = flipped_uv - vec2(u_eye_offset * depth_inv * u_depth_strength, 0.0);
        offset_uv = clamp(offset_uv, 0.0, 1.0);

        frag_color = texture(tex_color, offset_uv);
    }
    """

class StereoWindow:
    """A window for displaying stereo images side-by-side with depth effect."""
    def __init__(self, depth_ratio):
        self.window_size = (1280, 720)
        self.title = "Stereo SBS Viewer"
        self.ipd_uv = 0.064  # Inter-pupillary distance in UV coordinates (0.064 per eye)
        self.depth_strength = 0.1  # Strength of depth effect
        self._last_window_position = None
        self._last_window_size = None
        self._fullscreen = False
        self.depth_ratio = depth_ratio
        self.depth_ratio_original = depth_ratio

        # Flag to track if textures need update
        self._texture_size = None
        
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
        if OS_NAME != "Darwin":
            self.add_logo(self.window)  # Add logo to window
        if not self.window:
            glfw.terminate()
            raise RuntimeError("Could not create window")
        
        # Position window on monitor 2 if available
        self.position_on_monitor(0)  # 0=primary, 1=secondary
        
        glfw.make_context_current(self.window)
        glfw.swap_interval(1)  # Enable VSync to limit FPS and reduce CPU load
        self.ctx = moderngl.create_context()
        
        # Setup shaders and buffers
        self.prog = self.ctx.program(
            vertex_shader=VERTEX_SHADER,
            fragment_shader=FRAGMENT_SHADER
        )
        self.quad_vao = self.make_quad()
        self.monitor_index = 0
        self.color_tex = None
        self.depth_tex = None
        
        # Set callbacks
        glfw.set_key_callback(self.window, self.on_key_event)

    def add_logo(self, window):
        import cv2
        logo = './logo.png'  # Path to your icon file
        img = cv2.imread(logo, cv2.IMREAD_UNCHANGED)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # <--- convert from BGR to RGB

        if img is None:
            return False

        h, w = img.shape[:2]

        if img.shape[2] == 3:
            alpha = np.ones((h, w, 1), dtype = np.uint8) * 255
            img = np.concatenate((img, alpha), axis = 2)

        img = img.astype(np.uint8)
        glfw_img = (w, h, img)

        glfw.set_window_icon(window, 1, [glfw_img])

    def position_on_monitor(self, monitor_index=0):
        """Position window on specified monitor (0-based index)"""
        monitors = glfw.get_monitors()
        if monitor_index < len(monitors):
            monitor = monitors[monitor_index]
            mon_x, mon_y = glfw.get_monitor_pos(monitor)
            vidmode = glfw.get_video_mode(monitor)
            mon_w, mon_h = vidmode.size.width, vidmode.size.height

            x = mon_x + (mon_w - self.window_size[0]) // 2
            y = mon_y + (mon_h - self.window_size[1]) // 2
            glfw.set_window_pos(self.window, x, y)
            self.monitor_index = monitor_index  # Track current monitor

    def get_current_monitor(self):
        """Get the monitor that contains the window center"""
        monitors = glfw.get_monitors()
        if not monitors:
            return None
            
        # Get window center
        win_x, win_y = glfw.get_window_pos(self.window)
        win_w, win_h = glfw.get_framebuffer_size(self.window)
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
    def move_to_adjacent_monitor(self, direction):
        """
        Move window to adjacent monitor.
        direction: +1 for right, -1 for left.
        """
        monitors = glfw.get_monitors()
        if not monitors or len(monitors) <= 1:
            return  # Nothing to switch

        new_index = (self.monitor_index + direction) % len(monitors)
        self.position_on_monitor(new_index)
        
    def toggle_fullscreen(self):
        """Toggle fullscreen on current monitor"""
        current_monitor = self.get_current_monitor()
        if not current_monitor:
            return

        if not self._fullscreen:
            # Enter fullscreen mode, Remember current window state
            self._last_window_position = glfw.get_window_pos(self.window)
            self._last_window_size     = glfw.get_framebuffer_size(self.window)

            # Monitor geometry
            mon_x, mon_y = glfw.get_monitor_pos(current_monitor)
            vidmode      = glfw.get_video_mode(current_monitor)
            full_w, full_h = vidmode.size.width, vidmode.size.height

            # Make the window border-less + floating
            glfw.set_window_attrib(self.window, glfw.DECORATED, glfw.FALSE)
            glfw.set_window_attrib(self.window, glfw.FLOATING,  glfw.TRUE)

            # Resize & move so it exactly covers the monitor
            glfw.set_window_size(self.window, full_w, full_h)
            glfw.set_window_pos(self.window,  mon_x,  mon_y)

            self._fullscreen = True
        else:
            # Exit fullscreeen mode, Restore decorations / floating attribute
            glfw.set_window_attrib(self.window, glfw.DECORATED, glfw.TRUE)
            glfw.set_window_attrib(self.window, glfw.FLOATING,  glfw.FALSE)

            # Restore position & size (default: 1280×720 centred on same monitor)
            restore_w, restore_h = self.window_size
            if self._last_window_size:
                restore_w, restore_h = self._last_window_size

            if self._last_window_position:
                restore_x, restore_y = self._last_window_position
            else:
                # Centre it on the current monitor if we have no previous position
                vidmode   = glfw.get_video_mode(current_monitor)
                mon_x, mon_y = glfw.get_monitor_pos(current_monitor)
                restore_x = mon_x + (vidmode.size.width  - restore_w) // 2
                restore_y = mon_y + (vidmode.size.height - restore_h) // 2

            glfw.set_window_size(self.window, restore_w, restore_h)
            glfw.set_window_pos(self.window,  restore_x, restore_y)

            self._fullscreen = False
    
    def on_key_event(self, window, key, scancode, action, mods):
        if action == glfw.PRESS:
            if key == glfw.KEY_SPACE:
                self.toggle_fullscreen()
            elif key == glfw.KEY_ESCAPE:
                glfw.set_window_should_close(window, True)
            elif key == glfw.KEY_RIGHT:
                self.move_to_adjacent_monitor(+1)
            elif key == glfw.KEY_LEFT:
                self.move_to_adjacent_monitor(-1)
            elif key == glfw.KEY_DOWN:
                # Decrease depth strength by 0.2
                self.depth_ratio = max(0, self.depth_ratio - 0.2)
            elif key == glfw.KEY_UP:
                # Increase depth strength by 0.2
                self.depth_ratio = min(10, self.depth_ratio + 0.2)
            elif key == glfw.KEY_0:
                # Increase depth strength by 0.2
                self.depth_ratio = self.depth_ratio_original



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
        # Only recreate textures if size changed
        new_size = (rgb.shape[1], rgb.shape[0])
        if self._texture_size != new_size:
            if self.color_tex:
                self.color_tex.release()
            if self.depth_tex:
                self.depth_tex.release()

            self.color_tex = self.ctx.texture(new_size, 3, dtype='f1')
            self.depth_tex = self.ctx.texture(new_size, 1, dtype='f4')

            # Set texture units once after recreating textures
            self.prog['tex_color'].value = 0
            self.prog['tex_depth'].value = 1

            self._texture_size = new_size

        # Upload texture data
        self.color_tex.write(rgb.tobytes())
        self.depth_tex.write((self.depth_ratio*depth).tobytes())

    def render(self):
        if not self.color_tex or not self.depth_tex:
            return  # Skip render if textures not ready

        self.ctx.clear(0.1, 0.1, 0.1)
        width, height = glfw.get_framebuffer_size(self.window)

        self.color_tex.use(location=0)
        self.depth_tex.use(location=1)

        # Left eye
        self.ctx.viewport = (0, 0, width // 2, height)
        eye_offset = -self.ipd_uv / 2.0
        self.prog['u_eye_offset'].value = eye_offset
        self.prog['u_depth_strength'].value = self.depth_strength
        self.quad_vao.render(moderngl.TRIANGLE_STRIP)

        # Right eye
        self.ctx.viewport = (width // 2, 0, width // 2, height)
        eye_offset = self.ipd_uv / 2.0
        self.prog['u_eye_offset'].value = eye_offset
        # depth_strength stays the same
        self.quad_vao.render(moderngl.TRIANGLE_STRIP)

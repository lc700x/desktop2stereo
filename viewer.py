# viewer.py
import glfw
import moderngl
import numpy as np

# Get OS name and settings
from utils import OS_NAME, crop_icon

# Shaders as constants (unchanged)
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

def add_logo(window):
    """Optimized logo loading with lazy imports"""
    from PIL import Image
    glfw_img = Image.open("icon2.ico")  # Path to your icon file
    if OS_NAME != "Darwin":
        glfw_img = crop_icon(glfw_img)
        glfw.set_window_icon(window, 1, [glfw_img])

class StereoWindow:
    """Optimized stereo viewer with performance improvements"""
    
    def __init__(self, ipd=0.064, depth_ratio=1.0, display_mode="Half-SBS"):
        # Initialize with default values
        self.window_size = (1280, 720)
        self.title = "Stereo SBS Viewer"
        self.ipd_uv = ipd
        self.depth_strength = 0.1
        self._last_window_position = None
        self._last_window_size = None
        self._fullscreen = False
        self.depth_ratio = depth_ratio
        self.depth_ratio_original = depth_ratio
        self._modes = ["Full-SBS", "Half-SBS", "TAB"]
        self.display_mode = display_mode
        self._texture_size = None
        self.monitor_index = 0
        
        # Initialize GLFW
        if not glfw.init():
            raise RuntimeError("Could not initialize GLFW")
        
        # Configure window
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.RESIZABLE, True)
        
        # Create window
        self.window = glfw.create_window(*self.window_size, self.title, None, None)
        if not self.window:
            glfw.terminate()
            raise RuntimeError("Could not create window")
        
        add_logo(self.window)
        self.position_on_monitor(0)
        
        # Set up OpenGL context
        glfw.make_context_current(self.window)
        glfw.swap_interval(0)  # VSync off for maximum performance
        self.ctx = moderngl.create_context()
        
        # Precompile shaders and create VAO
        self.prog = self.ctx.program(
            vertex_shader=VERTEX_SHADER,
            fragment_shader=FRAGMENT_SHADER
        )
        self.quad_vao = self._create_quad_vao()
        
        # Initialize textures as None
        self.color_tex = None
        self.depth_tex = None
        
        # Set callbacks
        glfw.set_key_callback(self.window, self.on_key_event)

    def _create_quad_vao(self):
        """Optimized quad creation with static data"""
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

    def position_on_monitor(self, monitor_index=0):
        """Optimized monitor positioning"""
        monitors = glfw.get_monitors()
        if monitor_index < len(monitors):
            monitor = monitors[monitor_index]
            mon_x, mon_y = glfw.get_monitor_pos(monitor)
            vidmode = glfw.get_video_mode(monitor)
            mon_w, mon_h = vidmode.size.width, vidmode.size.height

            x = mon_x + (mon_w - self.window_size[0]) // 2
            y = mon_y + (mon_h - self.window_size[1]) // 2
            glfw.set_window_pos(self.window, x, y)
            self.monitor_index = monitor_index

    def get_current_monitor(self):
        """Optimized monitor detection with early returns"""
        monitors = glfw.get_monitors()
        if not monitors:
            return None
            
        win_x, win_y = glfw.get_window_pos(self.window)
        win_w, win_h = glfw.get_window_size(self.window)
        window_center_x = win_x + win_w // 2
        window_center_y = win_y + win_h // 2
        
        for monitor in monitors:
            monitor_x, monitor_y = glfw.get_monitor_pos(monitor)
            vidmode = glfw.get_video_mode(monitor)
            if (monitor_x <= window_center_x < monitor_x + vidmode.size.width and
                monitor_y <= window_center_y < monitor_y + vidmode.size.height):
                return monitor
        return monitors[0]

    def move_to_adjacent_monitor(self, direction):
        """Optimized monitor switching"""
        monitors = glfw.get_monitors()
        if len(monitors) > 1:
            new_index = (self.monitor_index + direction) % len(monitors)
            self.position_on_monitor(new_index)
        
    def toggle_fullscreen(self):
        """Optimized fullscreen toggle with reduced GLFW calls"""
        current_monitor = self.get_current_monitor()
        if not current_monitor:
            return

        if not self._fullscreen:
            # Enter fullscreen
            self._last_window_position = glfw.get_window_pos(self.window)
            self._last_window_size = glfw.get_window_size(self.window)

            mon_x, mon_y = glfw.get_monitor_pos(current_monitor)
            vidmode = glfw.get_video_mode(current_monitor)
            full_w, full_h = vidmode.size.width, vidmode.size.height

            glfw.set_window_attrib(self.window, glfw.DECORATED, glfw.FALSE)
            glfw.set_window_attrib(self.window, glfw.FLOATING, glfw.TRUE)
            glfw.set_window_size(self.window, full_w, full_h)
            glfw.set_window_pos(self.window, mon_x, mon_y)

            self._fullscreen = True
        else:
            # Exit fullscreen
            glfw.set_window_attrib(self.window, glfw.DECORATED, glfw.TRUE)
            glfw.set_window_attrib(self.window, glfw.FLOATING, glfw.FALSE)

            restore_w, restore_h = self._last_window_size or self.window_size
            restore_x, restore_y = self._last_window_position or (0, 0)
            
            if not self._last_window_position:
                vidmode = glfw.get_video_mode(current_monitor)
                mon_x, mon_y = glfw.get_monitor_pos(current_monitor)
                restore_x = mon_x + (vidmode.size.width - restore_w) // 2
                restore_y = mon_y + (vidmode.size.height - restore_h) // 2

            glfw.set_window_size(self.window, restore_w, restore_h)
            glfw.set_window_pos(self.window, restore_x, restore_y)
            self._fullscreen = False
    
    def on_key_event(self, window, key, scancode, action, mods):
        """Optimized key event handling"""
        if action == glfw.PRESS:
            if key == glfw.KEY_ENTER or key == glfw.KEY_SPACE:
                self.toggle_fullscreen()
            elif key == glfw.KEY_ESCAPE:
                glfw.set_window_should_close(window, True)
            elif key == glfw.KEY_RIGHT:
                self.move_to_adjacent_monitor(+1)
            elif key == glfw.KEY_LEFT:
                self.move_to_adjacent_monitor(-1)
            elif key == glfw.KEY_DOWN:
                self.depth_ratio = max(0, self.depth_ratio - 0.5)
            elif key == glfw.KEY_UP:
                self.depth_ratio = min(10, self.depth_ratio + 0.5)
            elif key == glfw.KEY_0:
                self.depth_ratio = self.depth_ratio_original
            elif key == glfw.KEY_TAB:
                idx = self._modes.index(self.display_mode)
                self.display_mode = self._modes[(idx + 1) % len(self._modes)]

    def update_frame(self, rgb, depth):
        """Optimized texture updates with minimal allocations"""
        # Convert depth tensor to numpy array if needed
        if hasattr(depth, 'detach'):  # Check if it's a torch tensor
            depth = depth.detach().cpu().numpy()
        depth = depth.astype('float32', copy=False)
        
        h, w, _ = rgb.shape
        if self._texture_size != (w, h):
            # Release old textures if they exist
            if self.color_tex:
                self.color_tex.release()
            if self.depth_tex:
                self.depth_tex.release()
                
            # Create new textures
            self.color_tex = self.ctx.texture((w, h), 3, dtype='f1')
            self.depth_tex = self.ctx.texture((w, h), 1, dtype='f4')
            
            # Set texture units once
            self.prog['tex_color'].value = 0
            self.prog['tex_depth'].value = 1
            
            self._texture_size = (w, h)

        # Upload texture data with minimal copies
        self.color_tex.write(rgb.astype('uint8', copy=False).tobytes())
        self.depth_tex.write((self.depth_ratio * depth).astype('float32', copy=False).tobytes())

    def _compute_render_size(self, max_w, max_h, src_w, src_h):
        """Optimized render size calculation"""
        if src_w == 0 or src_h == 0:
            return 0, 0
        scale = min(max_w / src_w, max_h / src_h)
        return (
            max(1, int(round(src_w * scale))),
            max(1, int(round(src_h * scale)))
        )

    def render(self):
        """Optimized rendering with reduced GL calls"""
        if not self.color_tex or not self.depth_tex:
            return
            
        # Get window dimensions once
        win_w, win_h = glfw.get_framebuffer_size(self.window)
        tex_w, tex_h = self._texture_size
        
        # Clear screen once
        self.ctx.clear(0.1, 0.1, 0.1)
        
        # Bind textures once
        self.color_tex.use(location=0)
        self.depth_tex.use(location=1)
        
        # Set common uniform values
        self.prog['u_depth_strength'].value = self.depth_strength
        
        if self.display_mode == "Full-SBS":
            # Full Side-by-Side mode
            src_w, src_h = tex_w, tex_h
            max_w, max_h = win_w / 2.0, win_h
            render_w, render_h = self._compute_render_size(max_w, max_h, src_w, src_h)
            center_y = win_h / 2.0
            
            # Left view
            self.ctx.viewport = (
                int(win_w / 4.0 - render_w / 2),
                int(center_y - render_h / 2),
                render_w, render_h
            )
            self.prog['u_eye_offset'].value = -self.ipd_uv / 2.0
            self.quad_vao.render(moderngl.TRIANGLE_STRIP)
            
            # Right view
            self.ctx.viewport = (
                int(3 * win_w / 4.0 - render_w / 2),
                int(center_y - render_h / 2),
                render_w, render_h
            )
            self.prog['u_eye_offset'].value = self.ipd_uv / 2.0
            self.quad_vao.render(moderngl.TRIANGLE_STRIP)

        elif self.display_mode == "Half-SBS":
            # Half Side-by-Side mode
            src_w, src_h = tex_w / 2.0, tex_h
            max_w, max_h = win_w / 2.0, win_h
            render_w, render_h = self._compute_render_size(max_w, max_h, src_w, src_h)
            center_y = win_h / 2.0
            
            # Left view
            self.ctx.viewport = (
                int(win_w / 4.0 - render_w / 2),
                int(center_y - render_h / 2),
                render_w, render_h
            )
            self.prog['u_eye_offset'].value = -self.ipd_uv / 2.0
            self.quad_vao.render(moderngl.TRIANGLE_STRIP)
            
            # Right view
            self.ctx.viewport = (
                int(3 * win_w / 4.0 - render_w / 2),
                int(center_y - render_h / 2),
                render_w, render_h
            )
            self.prog['u_eye_offset'].value = self.ipd_uv / 2.0
            self.quad_vao.render(moderngl.TRIANGLE_STRIP)

        elif self.display_mode == "TAB":
            # Top-and-Bottom mode
            src_w, src_h = tex_w, tex_h / 2.0
            max_w, max_h = win_w, win_h / 2.0
            render_w, render_h = self._compute_render_size(max_w, max_h, src_w, src_h)
            
            # Top view
            self.ctx.viewport = (
                int(win_w / 2.0 - render_w / 2),
                int(win_h / 4.0 - render_h / 2),
                render_w, render_h
            )
            self.prog['u_eye_offset'].value = -self.ipd_uv / 2.0
            self.quad_vao.render(moderngl.TRIANGLE_STRIP)
            
            # Bottom view
            self.ctx.viewport = (
                int(win_w / 2.0 - render_w / 2),
                int(3 * win_h / 4.0 - render_h / 2),
                render_w, render_h
            )
            self.prog['u_eye_offset'].value = self.ipd_uv / 2.0

            self.quad_vao.render(moderngl.TRIANGLE_STRIP)

# viewer.py
import glfw
import moderngl
import numpy as np
import time
from PIL import Image, ImageDraw, ImageFont
from OpenGL.GL import *
# Get OS name and settings
from utils import OS_NAME, crop_icon, get_font_type, USE_3D_MONITOR, FILL_16_9, FIX_VIEWER_ASPECT, MONITOR_INDEX, CAPTURE_MODE, STEREO_DISPLAY_SELECTION, STEREO_DISPLAY_INDEX
# 3D monitor mode to hide viewer
if OS_NAME == "Windows":
    from utils import hide_window_from_capture
elif OS_NAME == "Darwin":
    from utils import send_ctrl_cmd_f

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
    // Or #version 300 es + precision highp float; for mobile

    in vec2 uv;
    out vec4 frag_color;

    uniform sampler2D tex_color;   // RGB image
    uniform sampler2D tex_depth;   // Single-channel depth (0 = near, 1 = far)
    uniform vec2 u_resolution;     // viewport resolution
    uniform float u_eye_offset;    // e.g. ±0.03 (positive = right eye)
    uniform float u_depth_strength;// parallax intensity
    uniform float u_convergence;   // depth value at screen plane (0–1)

    // Inpainting controls
    uniform float u_inpaint_radius = 7.0;     // higher = smoother but slower
    uniform float u_inpaint_strength = 20.0;  // how strongly we prefer background

    // ------------------------------------------------------------------
    vec2 pixel_size = 1.0 / u_resolution;

    // 1. Fast disocclusion detector (holes + big depth jumps)
    bool is_disoccluded(vec2 base_uv, vec2 shifted_uv, float center_depth) {
        // Out of bounds = obvious hole
        if (shifted_uv.x < 0.0 || shifted_uv.x > 1.0 ||
            shifted_uv.y < 0.0 || shifted_uv.y > 1.0)
            return true;

        // Huge depth discontinuity along shift direction = edge hole
        vec2 grad_dir = normalize(vec2(u_eye_offset, 0.0));
        float d_left  = texture(tex_depth, base_uv - grad_dir * pixel_size * 3.0).r;
        float d_right = texture(tex_depth, base_uv + grad_dir * pixel_size * 3.0).r;
        if (abs(d_left - d_right) > 0.07)
            return true;

        return false;
    }

    // 2. Depth-aware bilateral inpainting (only samples background)
    vec4 inpaint_disocclusion(vec2 uv_coord, float center_depth_inv) {
        vec4 accum = vec4(0.0);
        float total_w = 0.0;
        int R = int(u_inpaint_radius);

        for (int y = -R; y <= R; ++y) {
            for (int x = -R; x <= R; ++x) {
                vec2 sample_uv = uv_coord + vec2(x, y) * pixel_size;

                if (sample_uv.x < 0.0 || sample_uv.x > 1.0 ||
                    sample_uv.y < 0.0 || sample_uv.y > 1.0)
                    continue;

                vec4 sample_color = texture(tex_color, sample_uv);
                float sample_depth = texture(tex_depth, sample_uv).r;
                float sample_depth_inv = 1.0 - sample_depth;

                // Only accept pixels that are clearly behind (background)
                float depth_ok = step(center_depth_inv + 0.015, sample_depth_inv);

                // Spatial gaussian
                float dist2 = float(x*x + y*y);
                float spatial_w = exp(-dist2 / (2.0 * (u_inpaint_radius * 0.6) * (u_inpaint_radius * 0.6)));

                // Color similarity (helps coherence)
                vec3 color_diff = sample_color.rgb - texture(tex_color, uv_coord).rgb;
                float color_w = exp(-dot(color_diff, color_diff) * 10.0);

                float w = spatial_w * depth_ok * color_w *
                        (1.0 + depth_ok * u_inpaint_strength);

                accum += sample_color * w;
                total_w += w;
            }
        }
        return total_w > 0.01 ? accum / total_w : texture(tex_color, uv_coord);
    }

    // ------------------------------------------------------------------
    void main() {
        vec2 flipped_uv = vec2(uv.x, 1.0 - uv.y);

        // Use original depth directly (no dilation)
        float depth = texture(tex_depth, flipped_uv).r;
        float depth_inv = 1.0 - depth;

        // Parallax shift
        float shift = (depth_inv - u_convergence);
        vec2 shifted_uv = flipped_uv - vec2(u_eye_offset * shift * u_depth_strength, 0.0);

        // Original depth values for disocclusion decisions
        float original_depth = depth;
        float original_depth_inv = depth_inv;

        vec4 color;
        if (is_disoccluded(flipped_uv, shifted_uv, original_depth)) {
            // Disoccluded → inpaint from background
            color = inpaint_disocclusion(flipped_uv, original_depth_inv);
        } else {
            // Normal sampling
            color = texture(tex_color, shifted_uv);
        }

        // Optional subtle border fade to hide hard edges
        vec2 border = smoothstep(0.0, 0.02, shifted_uv) * smoothstep(1.0, 0.98, shifted_uv);
        color.a = min(border.x, border.y);

        frag_color = color;
    }
"""

def add_logo(window):
    """Optimized logo loading with lazy imports"""
    from PIL import Image
    glfw_img = Image.open("icon2.ico")  # Path to your icon file
    if OS_NAME != "Darwin":
        glfw_img = crop_icon(glfw_img)
        glfw.set_window_icon(window, 1, [glfw_img])
        glfw.swap_interval(1)  # VSync on

class StereoWindow:
    """Optimized stereo viewer with performance improvements"""
    
    def __init__(self, ipd=0.064, depth_ratio=1.0, display_mode="Half-SBS", fill_16_9=FILL_16_9, show_fps=True, use_3d=USE_3D_MONITOR, fix_aspect=FIX_VIEWER_ASPECT, stream_mode=None, frame_size=(1280, 720)):
        # Initialize with default values
        self.use_3d = use_3d
        self.title = "Stereo Viewer"
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
        self.fill_16_9 = fill_16_9
        self.frame_size = frame_size
        self.aspect = self.frame_size[0] / self.frame_size[1]
        self.fix_aspect = fix_aspect
        self.show_fps = show_fps
        self.stream_mode = stream_mode
        self.window_size = self.frame_size

        # FPS tracking variables
        self.frame_count = 0
        self.last_fps_time = time.perf_counter()
        self.actual_fps = 0.0
        self.start_time = time.perf_counter()
        self.total_frames = 0
        
        # Add PBO for streamer
        self._pbo_ids = None
        self._pbo_index = 0
        self._pbo_initialized = False
        
        # Depth ratio display variables
        self.last_depth_change_time = 0
        self.show_depth_ratio = False
        self.depth_display_duration = 2.0  # Show depth for 2 seconds after change
        
        # Font and text sizing
        self.font = None
        self.font_type = get_font_type()
        self.base_font_size = 60  # Base size for 1280x720 window
        self.current_font_size = self.base_font_size
        self.text_padding = 10
        self.text_spacing = 5
        self.convergence = 0.5

        # Overlay cache & throttle
        self.overlay_update_interval = 0.25  # seconds, throttle overlay regeneration
        self._overlay_cache = {
            'image': None,         # numpy RGBA image
            'fps_text': None,
            'depth_text': None,
            'last_update': 0.0,
            'pos': (self.text_padding, self.text_padding)
        }
        
        # Stereo Display Settings
        self.specify_display = STEREO_DISPLAY_SELECTION
        self.stereo_display_index = STEREO_DISPLAY_INDEX 
        
        # Initialize GLFW
        if not glfw.init():
            raise RuntimeError("Could not initialize GLFW")
        
        self.monitor_index = self.get_glfw_mon_index(MONITOR_INDEX) if CAPTURE_MODE=="Monitor" else 1
        if self.specify_display:
            self.monitor_index = self.get_glfw_mon_index(self.stereo_display_index)
        # Configure window
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.RESIZABLE, True)
        glfw.window_hint(glfw.DECORATED, glfw.TRUE) # window decoration
        
        # Get primary monitor resolution
        monitors = glfw.get_monitors()
        
        if self.use_3d:
            glfw.window_hint(glfw.MOUSE_PASSTHROUGH, glfw.TRUE)  # clicks pass through
            glfw.window_hint(glfw.FLOATING, glfw.TRUE)    # Always on top
            glfw.window_hint(glfw.DECORATED, glfw.FALSE) # remove window decoration
            # Get primary monitor resolution
            monitor = monitors[self.monitor_index]
            vidmode = glfw.get_video_mode(monitor)
            self.window_size = (vidmode.size.width, vidmode.size.height)
        elif self.stream_mode == "RTMP":
            glfw.window_hint(glfw.RESIZABLE, False)  # Disable resizing
            # glfw.window_hint(glfw.MOUSE_PASSTHROUGH, glfw.TRUE)  # clicks pass through
        elif self.stream_mode == "MJPEG":
            glfw.window_hint(glfw.RESIZABLE, False)  # Disable resizing
            glfw.window_hint(glfw.VISIBLE, glfw.FALSE)  # clicks pass through
        # Create window
        self.window = glfw.create_window(*self.window_size, self.title, None, None)
        add_logo(self.window)
        
        # Hide window for 3D monitor, but cannot be captured by other apps as well
        if self.use_3d and OS_NAME == "Windows":
            hide_window_from_capture(self.window)
        
        if self.stream_mode == "RTMP" and not self.specify_display:
            self.move_to_adjacent_monitor(direction=1)
        else:
            self.position_on_monitor(self.monitor_index)
            if self.specify_display and not self.stream_mode == "RTMP":
                time.sleep(0.01) # allow a small delay
                self.toggle_fullscreen()
        
        if not self.window:
            glfw.terminate()
            raise RuntimeError("Could not create window")

        # Set up OpenGL context
        glfw.make_context_current(self.window)
        self.ctx = moderngl.create_context()
        
        # Precompile shaders and create VAO
        self.prog = self.ctx.program(
            vertex_shader=VERTEX_SHADER,
            fragment_shader=FRAGMENT_SHADER
        )
        self.prog['u_convergence'].value = self.convergence  # e.g. self.convergence = 0.5
        self.quad_vao = self._create_quad_vao()
        
        # Initialize textures as None
        self.color_tex = None
        self.depth_tex = None
        
        # Set callbacks
        glfw.set_key_callback(self.window, self.on_key_event)
        glfw.set_window_size_callback(self.window, self._on_window_resize)
        
        # Load initial font
        self._update_font()

    def get_glfw_mon_index(self, mss_monitor_index=1):
        """
        Map an MSS monitor index (1-based) to a GLFW monitor handle.

        MSS provides monitors like:
            monitors[0] -> all monitors bounding box
            monitors[1] -> first display
            monitors[2] -> second display, etc.

        GLFW gives a list of monitor handles that we can position windows on.
        This function matches them by position and size.
        """
        try:
            import mss
        except ImportError:
            print("[StereoWindow] mss not installed; using default monitor.")
            return 0

        with mss.mss() as sct:
            mss_monitors = sct.monitors

        # MSS uses index 1-based (0 = virtual bounding box)
        if mss_monitor_index < 1 or mss_monitor_index >= len(mss_monitors):
            print(f"[StereoWindow] Invalid MSS monitor index {mss_monitor_index}, defaulting to 1.")
            mss_monitor_index = 1

        mss_mon = mss_monitors[mss_monitor_index]
        mss_x, mss_y = mss_mon["left"], mss_mon["top"]
        mss_w, mss_h = mss_mon["width"], mss_mon["height"]

        glfw_monitors = glfw.get_monitors()
        if not glfw_monitors:
            print("[StereoWindow] No GLFW monitors detected.")
            return 0

        for i, gmon in enumerate(glfw_monitors):
            gx, gy = glfw.get_monitor_pos(gmon)
            gvm = glfw.get_video_mode(gmon)
            gw, gh = gvm.size.width, gvm.size.height

            if abs(gx - mss_x) <= 5 and abs(gy - mss_y) <= 5 and abs(gw - mss_w) <= 5 and abs(gh - mss_h) <= 5:
                # Found matching GLFW monitor
                return i

        print("[StereoWindow] No matching GLFW monitor found for MSS index, defaulting to primary.")
        return 0
    
    def _on_window_resize(self, window, width, height):
        """Handle window resize events"""
        self.window_size = (width, height)
        self._update_font()

    def _update_font(self):
        """Update font size based on window dimensions"""
        # Calculate dynamic font size based on window height
        base_height = 1080  # Reference height
        scale_factor = min(self.frame_size[0] / base_height, 2.0)  # Cap scaling at 2x
        self.current_font_size = int(self.base_font_size * scale_factor * 0.8)  # Slightly smaller than linear scale
        
        try:
            self.font = ImageFont.truetype(self.font_type, self.current_font_size)
        except Exception:
            try:
                # Try default font (Pillow default)
                self.font = ImageFont.load_default()
            except Exception:
                self.font = None
        
        # Update padding and spacing based on font size
        self.text_padding = max(5, int(self.current_font_size * 0.4))
        self.text_spacing = max(2, int(self.current_font_size * 0.2))
        
        # Update overlay cache position in case padding changed
        self._overlay_cache['pos'] = (self.text_padding, self.text_padding)

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

    def _generate_overlay_image(self, fps_text, depth_text):
        """Rasterize the small overlay to RGBA numpy array (transparent background)."""
        if self.font is None:
            return None

        # Compose lines
        lines = []
        if self.show_fps and fps_text:
            lines.append(fps_text)
        if self.show_depth_ratio and depth_text:
            lines.append(depth_text)
        if not lines:
            return None

        # Estimate text size using PIL
        # small margin for padding/spacing
        padding = self.text_padding
        spacing = self.text_spacing

        # measure bounding boxes
        dummy_img = Image.new('RGBA', (1, 1), (0, 0, 0, 0))
        draw = ImageDraw.Draw(dummy_img)
        widths = []
        heights = []
        bboxes = []
        for line in lines:
            try:
                # textbbox if available provides better metrics
                bbox = draw.textbbox((0, 0), line, font=self.font)
                w = bbox[2] - bbox[0]
                h = bbox[3] - bbox[1]
            except Exception:
                w, h = draw.textsize(line, font=self.font)
            widths.append(w)
            heights.append(h)
            bboxes.append((w, h))
        overlay_w = max(widths) + padding * 2
        overlay_h = sum(heights) + spacing * (len(lines) - 1) + padding * 2

        # Create overlay image and draw text
        overlay_img = Image.new('RGBA', (overlay_w, overlay_h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay_img)
        x = padding
        y = padding
        for i, line in enumerate(lines):
            color = (0, 255, 0, 255) if i == 0 and self.show_fps else (0, 255, 255, 255)
            draw.text((x, y), line, font=self.font, fill=color)
            y += heights[i] + spacing

        overlay_arr = np.array(overlay_img, dtype=np.uint8)  # H x W x 4
        return overlay_arr

    def _add_overlay(self, rgb_frame):
        """Add FPS and depth ratio overlay to the frame with minimal allocations."""
        # If nothing to show or no font available, do nothing fast
        if not (self.show_fps or self.show_depth_ratio) or self.font is None:
            return rgb_frame

        # Ensure rgb_frame is H x W x 3 uint8
        if rgb_frame.dtype != np.uint8:
            rgb_frame = (rgb_frame * 255).astype(np.uint8)

        h, w, _ = rgb_frame.shape
        self.frame_size = (w, h)
        
        # freeze window size for rtmp streaming
        if self.stream_mode == "RTMP":
            if self.display_mode == "Full-SBS":
                w = 2 * w
            glfw.set_window_size(self.window, w, h)
            

        # Update FPS counters but do not regenerate overlay every frame
        current_time = time.perf_counter()
        if self.show_fps:
            self.frame_count += 1
            self.total_frames += 1
            # update measured FPS every 1 second (keeps value stable)
            if current_time - self.last_fps_time >= 1.0:
                self.actual_fps = self.frame_count / (current_time - self.last_fps_time)
                self.frame_count = 0
                self.last_fps_time = current_time

        # Depth ratio visibility check
        if current_time - self.last_depth_change_time < self.depth_display_duration:
            self.show_depth_ratio = True
        else:
            self.show_depth_ratio = False

        # Compose the strings to display
        fps_text = f"FPS: {self.actual_fps:.1f}" if self.show_fps else ""
        depth_text = f"Depth: {self.depth_ratio:.1f}" if self.show_depth_ratio else ""

        # Decide whether to regenerate overlay
        cache = self._overlay_cache
        needs_regen = False
        if cache['image'] is None:
            needs_regen = True
        elif fps_text != cache.get('fps_text') or depth_text != cache.get('depth_text'):
            needs_regen = True
        elif (current_time - cache.get('last_update', 0.0)) >= self.overlay_update_interval:
            # periodic regen in case font metrics or size changed
            needs_regen = True

        if needs_regen:
            overlay_arr = self._generate_overlay_image(fps_text, depth_text)
            cache['image'] = overlay_arr
            cache['fps_text'] = fps_text
            cache['depth_text'] = depth_text
            cache['last_update'] = current_time

        overlay_arr = cache['image']
        if overlay_arr is None:
            return rgb_frame

        ov_h, ov_w = overlay_arr.shape[:2]
        pos_x, pos_y = cache.get('pos', (self.text_padding, self.text_padding))

        # Clip overlay to frame boundaries
        if pos_x >= w or pos_y >= h:
            return rgb_frame
        end_x = min(w, pos_x + ov_w)
        end_y = min(h, pos_y + ov_h)
        ov_w_clipped = end_x - pos_x
        ov_h_clipped = end_y - pos_y
        if ov_w_clipped <= 0 or ov_h_clipped <= 0:
            return rgb_frame

        # Slice overlay and frame
        overlay_slice = overlay_arr[0:ov_h_clipped, 0:ov_w_clipped]
        frame_region = rgb_frame[pos_y:end_y, pos_x:end_x]

        # Alpha blending: result = overlay.rgb * alpha + frame * (1 - alpha)
        alpha = overlay_slice[..., 3:4].astype(np.float32) / 255.0  # H x W x 1
        overlay_rgb = overlay_slice[..., :3].astype(np.float32)
        frame_rgb = frame_region.astype(np.float32)

        blended = (overlay_rgb * alpha) + (frame_rgb * (1.0 - alpha))
        # write back blended region into original frame (as uint8)
        rgb_frame[pos_y:end_y, pos_x:end_x] = np.clip(blended, 0, 255).astype(np.uint8)

        return rgb_frame

    def position_on_monitor(self, monitor_index=0):
        """Optimized monitor positioning"""
        monitors = glfw.get_monitors()
        if monitor_index < len(monitors):
            monitor = monitors[monitor_index]
            mon_x, mon_y = glfw.get_monitor_pos(monitor)
            vidmode = glfw.get_video_mode(monitor)
            mon_w, mon_h = vidmode.size.width, vidmode.size.height
            if self.use_3d:
                glfw.set_window_size(self.window, mon_w, mon_h)
                glfw.set_window_pos(self.window, mon_x, mon_y)
            else:
                if self.stream_mode == "RTMP" and OS_NAME=="Linux":
                    x = mon_x + mon_w // 2
                    y = mon_y + mon_h // 2
                else:
                    x = mon_x + (mon_w - self.window_size[0]) // 2
                    y = mon_y + (mon_h - self.window_size[1]) // 2
                glfw.set_window_pos(self.window, x, y)
            # if self.stream_mode == "RTMP":
                # if vidmode.size == self.window_size:
                #     glfw.set_window_attrib(self.window, glfw.DECORATED, glfw.FALSE)
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
            if OS_NAME == "Darwin":
                send_ctrl_cmd_f() # MacOS full screen
            else:
                # Enter fullscreen
                self._last_window_position = glfw.get_window_pos(self.window)
                self._last_window_size = glfw.get_window_size(self.window)

                # Get monitor info
                mon_x, mon_y = glfw.get_monitor_pos(current_monitor)
                vidmode = glfw.get_video_mode(current_monitor)
                full_w, full_h = vidmode.size.width, vidmode.size.height

                # Make the window undecorated and floating
                glfw.set_window_attrib(self.window, glfw.DECORATED, glfw.FALSE)
                glfw.set_window_attrib(self.window, glfw.FLOATING, glfw.TRUE)
                if self.fix_aspect:
                    monitor_aspect = full_w / full_h
                    if monitor_aspect > self.aspect:
                        # Monitor is wider than target aspect
                        new_h = full_h
                        new_w = int(new_h * self.aspect)

                    else:
                        # Screen is taller — fit by width
                        new_w = full_w
                        new_h = int(full_w / self.aspect)

                    glfw.set_window_size(self.window, new_w, new_h)

                    # Center window on screen
                    center_x = mon_x + (full_w - new_w) // 2
                    center_y = mon_y + (full_h - new_h) // 2
                    if self.display_mode == "Full-SBS":
                        # Center window on screen
                        center_y = mon_y + (full_h - new_h//2) // 2
                    glfw.set_window_pos(self.window, center_x, center_y)
                else:
                    glfw.set_window_size(self.window, full_w, full_h)
                    glfw.set_window_pos(self.window, mon_x, mon_y)
            self._fullscreen = True
        else:
            if OS_NAME == "Darwin":
                send_ctrl_cmd_f() # MacOS full screen
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
        """Optimized key event handling, disable some keys for rtmp and 3d monitor"""
        if action == glfw.PRESS:
            if key == glfw.KEY_ENTER or key == glfw.KEY_SPACE:
                if self.stream_mode is None and not self.use_3d:
                    self.toggle_fullscreen()
            elif key == glfw.KEY_RIGHT:
                self.move_to_adjacent_monitor(+1)
            elif key == glfw.KEY_LEFT:
                self.move_to_adjacent_monitor(-1)
            elif key == glfw.KEY_ESCAPE:
                glfw.set_window_should_close(window, True)
            elif key == glfw.KEY_DOWN:
                self.depth_ratio = max(0, self.depth_ratio - 0.5)
                self.last_depth_change_time = time.perf_counter()
            elif key == glfw.KEY_UP:
                self.depth_ratio = min(10, self.depth_ratio + 0.5)
                self.last_depth_change_time = time.perf_counter()
            elif key == glfw.KEY_0:
                self.depth_ratio = self.depth_ratio_original
                self.last_depth_change_time = time.perf_counter()
            elif key == glfw.KEY_TAB:
                idx = self._modes.index(self.display_mode)
                self.display_mode = self._modes[(idx + 1) % len(self._modes)]
            elif key == glfw.KEY_F:  # Add FPS toggle with F key
                self.show_fps = not self.show_fps
                # Force overlay regen when toggling show_fps
                self._overlay_cache['last_update'] = 0.0
            elif key == glfw.KEY_A:  # Toggle fill 16:0 with A key
                self.fill_16_9 = not self.fill_16_9
                # Force overlay regen to show aspect ratio status
                self._overlay_cache['last_update'] = 0.0
            elif key == glfw.KEY_L:  # Toggle viewer aspect ratio lock with L key
                self.fix_aspect = not self.fix_aspect
                # Force overlay regen to show aspect ratio status
                self._overlay_cache['last_update'] = 0.0

    def update_frame(self, rgb, depth):
        """Optimized texture updates with minimal allocations"""
        # Convert depth tensor to numpy array if needed
        if hasattr(depth, 'detach'):  # Check if it's a torch tensor
            depth = depth.detach().cpu().numpy()
        depth = depth.astype('float32', copy=False)
        
        # Add overlay to the frame (modified in-place)
        rgb_with_overlay = self._add_overlay(rgb)
        
        h, w, _ = rgb_with_overlay.shape
        if self._texture_size != (w, h):
            # Release old textures if they exist
            if self.color_tex:
                try:
                    self.color_tex.release()
                except Exception:
                    pass
            if self.depth_tex:
                try:
                    self.depth_tex.release()
                except Exception:
                    pass
                
            # Create new textures
            self.color_tex = self.ctx.texture((w, h), 3, dtype='f1')
            self.depth_tex = self.ctx.texture((w, h), 1, dtype='f4')
            
            # Set texture units once
            self.prog['tex_color'].value = 0
            self.prog['tex_depth'].value = 1
            
            self._texture_size = (w, h)

        # Upload texture data
        if rgb_with_overlay.dtype != np.uint8:
            rgb_u8 = np.clip(rgb_with_overlay * 255.0, 0, 255).astype('uint8')
        else:
            rgb_u8 = rgb_with_overlay

        self.color_tex.write(rgb_u8.tobytes())
        self.depth_tex.write((self.depth_ratio * depth).astype('float32', copy=False).tobytes())

    def _compute_render_size(self, max_w, max_h, src_w, src_h):
        """Calculate render size maintaining aspect ratio"""
        if src_w == 0 or src_h == 0:
            return 0, 0
        scale = min(max_w / src_w, max_h / src_h)
        return (
            max(1, int(round(src_w * scale))),
            max(1, int(round(src_h * scale)))
        )
        
    def _init_pbos(self, width, height):
        """Initialize two PBOs for asynchronous pixel transfers."""
        if self._pbo_initialized:
            return
        self._pbo_ids = glGenBuffers(2)
        buffer_size = width * height * 3  # RGB8
        for pbo in self._pbo_ids:
            glBindBuffer(GL_PIXEL_PACK_BUFFER, pbo)
            glBufferData(GL_PIXEL_PACK_BUFFER, buffer_size, None, GL_STREAM_READ)
        glBindBuffer(GL_PIXEL_PACK_BUFFER, 0)
        self._pbo_index = 0
        self._pbo_initialized = True
        # print(f"[StereoWindow] Initialized {len(self._pbo_ids)} PBOs ({buffer_size/1e6:.2f} MB each)")

    def capture_glfw_image(self):
        """Asynchronous GPU readback using double-buffered PBOs."""
        width, height = glfw.get_framebuffer_size(self.window)
        if not self._pbo_initialized:
            self._init_pbos(width, height)
        next_index = (self._pbo_index + 1) % 2
        glPixelStorei(GL_PACK_ALIGNMENT, 1)

        # Bind current PBO and start async read
        glBindBuffer(GL_PIXEL_PACK_BUFFER, self._pbo_ids[self._pbo_index])
        glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, ctypes.c_void_p(0))

        # Bind previous PBO to map and read CPU data (async from previous frame)
        glBindBuffer(GL_PIXEL_PACK_BUFFER, self._pbo_ids[next_index])
        ptr = glMapBuffer(GL_PIXEL_PACK_BUFFER, GL_READ_ONLY)
        image = None
        if ptr:
            try:
                # Copy from GPU memory into numpy array
                buf = (GLubyte * (width * height * 3)).from_address(int(ptr))
                image = np.frombuffer(buf, dtype=np.uint8).reshape(height, width, 3)
                image = np.flipud(image.copy())  # Flip vertically; .copy() detaches from mapped memory
            finally:
                glUnmapBuffer(GL_PIXEL_PACK_BUFFER)

        # Unbind PBO
        glBindBuffer(GL_PIXEL_PACK_BUFFER, 0)
        # Swap PBO indices for next frame
        self._pbo_index = next_index

        # Note: the very first call will return None (no previous frame ready)
        return image

    def render(self):
        """Optimized rendering with reduced GL calls"""
        if not self.color_tex or not self.depth_tex:
            return

        # Get window dimensions once
        win_w, win_h = glfw.get_framebuffer_size(self.window)
        tex_w, tex_h = self._texture_size
        if self.fix_aspect:
            if self.display_mode == "Full-SBS":
                glfw.set_window_aspect_ratio(self.window, 2*tex_w, tex_h)
            else:
                glfw.set_window_aspect_ratio(self.window, tex_w, tex_h)
        else:
            glfw.set_window_aspect_ratio(self.window, glfw.DONT_CARE, glfw.DONT_CARE)
        # Clear screen once
        self.ctx.clear(0.1, 0.1, 0.1)
        
        if self.fill_16_9:
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
        
        else:
            # Determine effective stereo frame size by display mode
            if self.display_mode == "Full-SBS":
                disp_w, disp_h = 2 * tex_w, tex_h
            elif self.display_mode == "Half-SBS":
                disp_w, disp_h = tex_w, tex_h
            elif self.display_mode == "TAB":
                disp_w, disp_h = tex_w, tex_h
            else:
                disp_w, disp_h = 2 * tex_w, tex_h  # default full SBS

            target_aspect = disp_h / disp_w
            try:
                window_aspect = win_h / win_w
            except ZeroDivisionError:
                window_aspect = 9/16

            # Scale to fit window, preserving aspect ratio
            if window_aspect <= target_aspect:
                # Window is wider than content
                view_h = win_h
                view_w = int(view_h / target_aspect)
            else:
                # Window is taller than content
                view_w = win_w
                view_h = int(view_w * target_aspect)

            offset_x = (win_w - view_w) // 2
            offset_y = (win_h - view_h) // 2

            self.color_tex.use(location=0)
            self.depth_tex.use(location=1)

            if self.display_mode == "Full-SBS":
                # Left eye
                self.ctx.viewport = (offset_x, offset_y, view_w // 2, view_h)
                self.prog['u_eye_offset'].value = -self.ipd_uv / 2.0
                self.prog['u_depth_strength'].value = self.depth_strength
                self.quad_vao.render(moderngl.TRIANGLE_STRIP)

                # Right eye
                self.ctx.viewport = (offset_x + view_w // 2, offset_y, view_w // 2, view_h)
                self.prog['u_eye_offset'].value = self.ipd_uv / 2.0
                self.quad_vao.render(moderngl.TRIANGLE_STRIP)

            elif self.display_mode == "Half-SBS":
                # Same as FULL but both squeezed into width
                self.ctx.viewport = (offset_x, offset_y, view_w // 2, view_h)
                self.prog['u_eye_offset'].value = -self.ipd_uv / 2.0
                self.prog['u_depth_strength'].value = self.depth_strength
                self.quad_vao.render(moderngl.TRIANGLE_STRIP)

                self.ctx.viewport = (offset_x + view_w // 2, offset_y, view_w // 2, view_h)
                self.prog['u_eye_offset'].value = self.ipd_uv / 2.0
                self.quad_vao.render(moderngl.TRIANGLE_STRIP)

            elif self.display_mode == "TAB":
                # Top eye
                self.ctx.viewport = (offset_x, offset_y + view_h // 2, view_w, view_h // 2)
                self.prog['u_eye_offset'].value = -self.ipd_uv / 2.0
                self.prog['u_depth_strength'].value = self.depth_strength
                self.quad_vao.render(moderngl.TRIANGLE_STRIP)

                # Bottom eye
                self.ctx.viewport = (offset_x, offset_y, view_w, view_h // 2)
                self.prog['u_eye_offset'].value = self.ipd_uv / 2.0
                self.quad_vao.render(moderngl.TRIANGLE_STRIP)

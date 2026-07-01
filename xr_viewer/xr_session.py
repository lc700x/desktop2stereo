"""XRSessionMixin: OpenXR session init, controller actions, event polling, grip/aim poses, FBO creation."""

import ctypes
import sys
import time

import glfw

try:
    import xr
except ImportError:
    xr = None

from .constants import _GL_SRGB8_ALPHA8, _VIVE_TB_Y


class XRSessionMixin:
    def _init_openxr(self):
        """Try OpenGL first; fall back to D3D11 on Windows if OpenGL fails."""
        try:
            self._init_openxr_opengl()
            return
        except Exception as e:
            if self._is_openxr_device_unavailable(e):
                self._cleanup_partial_openxr()
                raise
            if sys.platform != "win32":
                raise
            print(f"[OpenXRViewer] OpenGL init failed ({e}), falling back to D3D11")
            self._cleanup_partial_openxr()

        self._init_openxr_d3d11()
        self._use_d3d11 = True

    def _is_openxr_device_unavailable(self, exc):
        """Return True when the runtime exists but no headset is currently available."""
        unavailable_cls = getattr(getattr(xr, 'exception', None), 'FormFactorUnavailableError', None)
        if unavailable_cls is not None and isinstance(exc, unavailable_cls):
            return True
        return exc.__class__.__name__ == 'FormFactorUnavailableError'

    def _wait_for_openxr_device(self, shutdown_event, retry_delay=2.0):
        """Retry OpenXR init until the headset is connected or startup is cancelled."""
        prompted = False
        while not glfw.window_should_close(self.window) and not shutdown_event.is_set():
            try:
                self._init_openxr()
                if prompted:
                    print("[OpenXRViewer] XR device connected; continuing startup")
                return True
            except Exception as exc:
                self._cleanup_partial_openxr()
                if not self._is_openxr_device_unavailable(exc):
                    raise
                if not prompted:
                    print("[OpenXRViewer] Waiting for XR device. Please connect or power on the headset...")
                    prompted = True
                end_t = time.perf_counter() + float(retry_delay)
                while time.perf_counter() < end_t:
                    glfw.poll_events()
                    if glfw.window_should_close(self.window) or shutdown_event.is_set():
                        return False
                    time.sleep(0.05)
        return False

    def _cleanup_partial_openxr(self):
        """Tear down any partially-initialised OpenXR + D3D11 state so a retry is clean."""
        for swapchain in self._xr_swapchains.values():
            try:
                xr.destroy_swapchain(swapchain)
            except Exception:
                pass
        self._xr_swapchains.clear()
        self._swapchain_images.clear()
        self._swapchain_sizes.clear()

        for attr in ("_xr_space", "_aim_space_l", "_aim_space_r", "_grip_space_l", "_grip_space_r"):
            sp = getattr(self, attr, None)
            if sp:
                try:
                    xr.destroy_space(sp)
                except Exception:
                    pass
                setattr(self, attr, None)

        if self._xr_session:
            try:
                xr.destroy_session(self._xr_session)
            except Exception:
                pass
            self._xr_session = None

        if self._xr_instance:
            try:
                xr.destroy_instance(self._xr_instance)
            except Exception:
                pass
            self._xr_instance = None

        self._xr_system_id = None

        # Release D3D11 COM objects if they were created
        for d3d_obj in (self._d3d11_context, self._d3d11_device):
            if d3d_obj is not None:
                try:
                    vtbl = ctypes.cast(d3d_obj, ctypes.POINTER(ctypes.c_void_p)).contents.value
                    release_fn = ctypes.CFUNCTYPE(ctypes.c_ulong, ctypes.c_void_p)(
                        ctypes.cast(vtbl + 2 * ctypes.sizeof(ctypes.c_void_p),
                                    ctypes.POINTER(ctypes.c_void_p)).contents.value
                    )
                    release_fn(d3d_obj.value)
                except Exception:
                    pass
        self._d3d11_device  = None
        self._d3d11_context = None

    def _init_openxr_opengl(self):
        """Original OpenGL-backed OpenXR session."""
        # 1. Instance
        app_info = xr.ApplicationInfo(
            application_name="Desktop2Stereo",
            application_version=1,
            engine_name="D2S",
            engine_version=1,
            api_version=xr.XR_CURRENT_API_VERSION,
        )
        create_info = xr.InstanceCreateInfo(
            application_info=app_info,
            enabled_extension_names=[xr.KHR_OPENGL_ENABLE_EXTENSION_NAME],
        )
        self._xr_instance = xr.create_instance(create_info)

        # 2. System
        self._xr_system_id = xr.get_system(
            self._xr_instance,
            xr.SystemGetInfo(form_factor=xr.FormFactor.HEAD_MOUNTED_DISPLAY),
        )
        print("[OpenXRViewer] XrInstance created (OpenGL)")

        # 3. Verify GL requirements (mandatory before session creation)
        _pfn = ctypes.cast(
            xr.get_instance_proc_addr(self._xr_instance, "xrGetOpenGLGraphicsRequirementsKHR"),
            xr.PFN_xrGetOpenGLGraphicsRequirementsKHR,
        )
        _reqs = xr.GraphicsRequirementsOpenGLKHR()
        xr.check_result(xr.Result(_pfn(self._xr_instance, self._xr_system_id, ctypes.byref(_reqs))))

        # 4. Graphics binding platform-specific
        if sys.platform == "win32":
            from OpenGL.WGL import wglGetCurrentContext, wglGetCurrentDC
            binding = xr.GraphicsBindingOpenGLWin32KHR(
                h_dc=wglGetCurrentDC(),
                h_glrc=wglGetCurrentContext(),
            )
        else:
            from OpenGL.GLX import glXGetCurrentContext, glXGetCurrentDisplay, glXGetCurrentDrawable
            binding = xr.GraphicsBindingOpenGLXlibKHR(
                x_display=glXGetCurrentDisplay(),
                glx_drawable=glXGetCurrentDrawable(),
                glx_context=glXGetCurrentContext(),
            )

        # 5. Session
        session_info = xr.SessionCreateInfo(
            system_id=self._xr_system_id,
            next=ctypes.cast(ctypes.pointer(binding), ctypes.c_void_p),
        )
        self._xr_session = xr.create_session(self._xr_instance, session_info)
        print("[OpenXRViewer] XrSession created (OpenGL)")

        # 6. Reference space prefer STAGE (floor origin), fall back to LOCAL
        available_spaces = xr.enumerate_reference_spaces(self._xr_session)
        ref_type = (
            xr.ReferenceSpaceType.STAGE
            if xr.ReferenceSpaceType.STAGE in available_spaces
            else xr.ReferenceSpaceType.LOCAL
        )
        self._xr_ref_space_type = ref_type
        self._xr_space = xr.create_reference_space(
            self._xr_session,
            xr.ReferenceSpaceCreateInfo(
                reference_space_type=ref_type,
                pose_in_reference_space=xr.Posef(),
            ),
        )

        # 7. Swapchains one per eye
        view_configs = xr.enumerate_view_configuration_views(
            self._xr_instance,
            self._xr_system_id,
            xr.ViewConfigurationType.PRIMARY_STEREO,
        )
        for eye_index, vcv in enumerate(view_configs):
            rec_w = vcv.recommended_image_rect_width
            rec_h = vcv.recommended_image_rect_height
            # Use exactly the recommended resolution this matches the HMD panel
            # pixel density and is what the runtime expects for correct reprojection.
            sc_w = rec_w & ~1
            sc_h = rec_h & ~1
            print(f"[OpenXRViewer] Eye {eye_index} swapchain: {sc_w}x{sc_h}")

            sc_info = xr.SwapchainCreateInfo(
                usage_flags=(
                    xr.SwapchainUsageFlags.COLOR_ATTACHMENT_BIT |
                    xr.SwapchainUsageFlags.SAMPLED_BIT
                ),
                format=_GL_SRGB8_ALPHA8,
                sample_count=1,
                width=sc_w,
                height=sc_h,
                face_count=1,
                array_size=1,
                mip_count=1,
            )
            swapchain = xr.create_swapchain(self._xr_session, sc_info)
            images = xr.enumerate_swapchain_images(swapchain, xr.SwapchainImageOpenGLKHR)
            self._xr_swapchains[eye_index] = swapchain
            self._swapchain_images[eye_index] = images
            self._swapchain_sizes[eye_index] = (sc_w, sc_h)

        # 8. Controller actions (optional silently disabled if action set creation fails)
        try:
            self._init_controller_actions()
        except Exception as e:
            print(f"[OpenXRViewer] Controller actions unavailable: {e}")

    def _init_controller_actions(self):
        """Set up OpenXR action set with thumbstick and menu button actions."""
        self._action_set = xr.create_action_set(
            self._xr_instance,
            xr.ActionSetCreateInfo(
                action_set_name="screen_control",
                localized_action_set_name="Screen Control",
                priority=0,
            ),
        )
        subpaths = [
            xr.string_to_path(self._xr_instance, p)
            for p in ["/user/hand/left", "/user/hand/right"]
        ]
        # Cache hand XrPath values so per-frame action reads don't call string_to_path
        self._path_left  = subpaths[0]
        self._path_right = subpaths[1]

        def make_vec2(name, label):
            return xr.create_action(
                self._action_set,
                xr.ActionCreateInfo(
                    action_type=xr.ActionType.VECTOR2F_INPUT,
                    action_name=name,
                    localized_action_name=label,
                    count_subaction_paths=len(subpaths),
                    subaction_paths=subpaths,
                ),
            )

        self._act_left_stick  = make_vec2("left_stick",  "Left Stick")
        self._act_right_stick = make_vec2("right_stick", "Right Stick")

        def make_bool(name, label):
            return xr.create_action(
                self._action_set,
                xr.ActionCreateInfo(
                    action_type=xr.ActionType.BOOLEAN_INPUT,
                    action_name=name,
                    localized_action_name=label,
                    count_subaction_paths=len(subpaths),
                    subaction_paths=subpaths,
                ),
            )

        self._act_menu_btn  = make_bool("menu_btn",   "Menu Button")
        self._act_left_grip = make_bool("left_grip",  "Left Grip")
        self._act_right_grip= make_bool("right_grip", "Right Grip")
        self._act_a_btn     = make_bool("a_btn",      "A Button")
        self._act_b_btn     = make_bool("b_btn",      "B Button")
        self._act_x_btn     = make_bool("x_btn",      "X Button")
        self._act_y_btn     = make_bool("y_btn",      "Y Button")
        self._act_left_stick_click  = make_bool("left_stick_click",  "Left Stick Click")
        self._act_right_stick_click = make_bool("right_stick_click", "Right Stick Click")

        def make_float(name, label):
            return xr.create_action(
                self._action_set,
                xr.ActionCreateInfo(
                    action_type=xr.ActionType.FLOAT_INPUT,
                    action_name=name,
                    localized_action_name=label,
                    count_subaction_paths=len(subpaths),
                    subaction_paths=subpaths,
                ),
            )

        self._act_left_trigger  = make_float("left_trigger",  "Left Trigger")
        self._act_right_trigger = make_float("right_trigger", "Right Trigger")

        self._act_aim_left = xr.create_action(
            self._action_set,
            xr.ActionCreateInfo(
                action_type=xr.ActionType.POSE_INPUT,
                action_name="aim_left",
                localized_action_name="Left Aim Pose",
                count_subaction_paths=1,
                subaction_paths=[subpaths[0]],
            ),
        )
        self._act_aim_right = xr.create_action(
            self._action_set,
            xr.ActionCreateInfo(
                action_type=xr.ActionType.POSE_INPUT,
                action_name="aim_right",
                localized_action_name="Right Aim Pose",
                count_subaction_paths=1,
                subaction_paths=[subpaths[1]],
            ),
        )

        # Grip pose actions used for placing controller 3D models
        self._act_grip_left = xr.create_action(
            self._action_set,
            xr.ActionCreateInfo(
                action_type=xr.ActionType.POSE_INPUT,
                action_name="grip_left",
                localized_action_name="Left Grip Pose",
                count_subaction_paths=1,
                subaction_paths=[subpaths[0]],
            ),
        )
        self._act_grip_right = xr.create_action(
            self._action_set,
            xr.ActionCreateInfo(
                action_type=xr.ActionType.POSE_INPUT,
                action_name="grip_right",
                localized_action_name="Right Grip Pose",
                count_subaction_paths=1,
                subaction_paths=[subpaths[1]],
            ),
        )

        # Per-profile binding table.
        # Use squeeze/value (float path) for grip the runtime auto-thresholds it
        # for BOOLEAN_INPUT actions, and it works on more firmware versions than
        # squeeze/click (which requires a discrete click event on some runtimes).
        _b = {
            "/interaction_profiles/oculus/touch_controller": [
                ("/user/hand/left/input/thumbstick",         self._act_left_stick),
                ("/user/hand/right/input/thumbstick",        self._act_right_stick),
                ("/user/hand/left/input/thumbstick/click",   self._act_left_stick_click),
                ("/user/hand/right/input/thumbstick/click",  self._act_right_stick_click),
                ("/user/hand/left/input/menu/click",         self._act_menu_btn),
                ("/user/hand/left/input/squeeze/value",      self._act_left_grip),
                ("/user/hand/right/input/squeeze/value",     self._act_right_grip),
                ("/user/hand/right/input/a/click",           self._act_a_btn),
                ("/user/hand/right/input/b/click",           self._act_b_btn),
                ("/user/hand/left/input/x/click",            self._act_x_btn),
                ("/user/hand/left/input/y/click",            self._act_y_btn),
                ("/user/hand/left/input/trigger/value",      self._act_left_trigger),
                ("/user/hand/right/input/trigger/value",     self._act_right_trigger),
                ("/user/hand/left/input/aim/pose",           self._act_aim_left),
                ("/user/hand/right/input/aim/pose",          self._act_aim_right),
                ("/user/hand/left/input/grip/pose",          self._act_grip_left),
                ("/user/hand/right/input/grip/pose",         self._act_grip_right),
            ],
            "/interaction_profiles/valve/index_controller": [
                ("/user/hand/left/input/thumbstick",         self._act_left_stick),
                ("/user/hand/right/input/thumbstick",        self._act_right_stick),
                ("/user/hand/left/input/thumbstick/click",   self._act_left_stick_click),
                ("/user/hand/right/input/thumbstick/click",  self._act_right_stick_click),
                ("/user/hand/left/input/trackpad/click",     self._act_menu_btn),
                ("/user/hand/left/input/squeeze/value",      self._act_left_grip),
                ("/user/hand/right/input/squeeze/value",     self._act_right_grip),
                ("/user/hand/right/input/a/click",           self._act_a_btn),
                ("/user/hand/right/input/b/click",           self._act_b_btn),
                ("/user/hand/left/input/trigger/value",      self._act_left_trigger),
                ("/user/hand/right/input/trigger/value",     self._act_right_trigger),
                ("/user/hand/left/input/aim/pose",           self._act_aim_left),
                ("/user/hand/right/input/aim/pose",         self._act_aim_right),
                ("/user/hand/left/input/grip/pose",         self._act_grip_left),
                ("/user/hand/right/input/grip/pose",        self._act_grip_right),
            ],
            # HTC Vive wand: trackpad (no thumbstick), squeeze/click (boolean,
            # no analog value), trigger value/click, menu no A/B/X/Y buttons.
            # The trackpad's 2D parent binds to the Vector2f stick actions, and
            # trackpad/click stands in for the thumbstick click.  Grip uses
            # squeeze/click directly since the wand has no analog squeeze.
            # Ref: https://registry.khronos.org/OpenXR/specs/1.0/man/html/openxr.html
            "/interaction_profiles/htc/vive_controller": [
                ("/user/hand/left/input/trackpad",           self._act_left_stick),
                ("/user/hand/right/input/trackpad",          self._act_right_stick),
                ("/user/hand/left/input/trackpad/click",     self._act_left_stick_click),
                ("/user/hand/right/input/trackpad/click",    self._act_right_stick_click),
                ("/user/hand/left/input/menu/click",         self._act_menu_btn),
                ("/user/hand/left/input/squeeze/click",      self._act_left_grip),
                ("/user/hand/right/input/squeeze/click",     self._act_right_grip),
                ("/user/hand/left/input/trigger/value",      self._act_left_trigger),
                ("/user/hand/right/input/trigger/value",     self._act_right_trigger),
                ("/user/hand/left/input/aim/pose",           self._act_aim_left),
                ("/user/hand/right/input/aim/pose",          self._act_aim_right),
                ("/user/hand/left/input/grip/pose",          self._act_grip_left),
                ("/user/hand/right/input/grip/pose",         self._act_grip_right),
            ],
            # KHR simple only has select/click (boolean) and menu no sticks or grip
            "/interaction_profiles/khr/simple_controller": [
                ("/user/hand/left/input/menu/click",    self._act_menu_btn),
                ("/user/hand/left/input/aim/pose",      self._act_aim_left),
                ("/user/hand/right/input/aim/pose",     self._act_aim_right),
                ("/user/hand/left/input/grip/pose",     self._act_grip_left),
                ("/user/hand/right/input/grip/pose",    self._act_grip_right),
            ],
            # PICO 4 Ultra controller interaction profile
            "/interaction_profiles/bytedance/pico_4u_controller": [
                ("/user/hand/left/input/thumbstick",         self._act_left_stick),
                ("/user/hand/right/input/thumbstick",        self._act_right_stick),
                ("/user/hand/left/input/thumbstick/click",   self._act_left_stick_click),
                ("/user/hand/right/input/thumbstick/click",  self._act_right_stick_click),
                ("/user/hand/left/input/menu/click",         self._act_menu_btn),
                ("/user/hand/left/input/squeeze/value",      self._act_left_grip),
                ("/user/hand/right/input/squeeze/value",     self._act_right_grip),
                ("/user/hand/right/input/a/click",           self._act_a_btn),
                ("/user/hand/right/input/b/click",           self._act_b_btn),
                ("/user/hand/left/input/x/click",            self._act_x_btn),
                ("/user/hand/left/input/y/click",            self._act_y_btn),
                ("/user/hand/left/input/trigger/value",      self._act_left_trigger),
                ("/user/hand/right/input/trigger/value",     self._act_right_trigger),
                ("/user/hand/left/input/aim/pose",           self._act_aim_left),
                ("/user/hand/right/input/aim/pose",          self._act_aim_right),
                ("/user/hand/left/input/grip/pose",          self._act_grip_left),
                ("/user/hand/right/input/grip/pose",         self._act_grip_right),
            ],
        }

        for profile, pairs in _b.items():
            try:
                xr.suggest_interaction_profile_bindings(
                    self._xr_instance,
                    xr.InteractionProfileSuggestedBinding(
                        interaction_profile=xr.string_to_path(self._xr_instance, profile),
                        suggested_bindings=[
                            xr.ActionSuggestedBinding(
                                action=act,
                                binding=xr.string_to_path(self._xr_instance, path),
                            )
                            for path, act in pairs
                        ],
                    ),
                )
            except Exception:
                pass

        xr.attach_session_action_sets(
            self._xr_session,
            xr.SessionActionSetsAttachInfo(action_sets=[self._action_set]),
        )

        # Pre-build the sync_actions arg now that the action set exists.
        # Same struct is reused every frame saves a list+struct allocation
        # per frame inside the hot loop.
        self._xr_actions_sync_info = xr.ActionsSyncInfo(active_action_sets=[
            xr.ActiveActionSet(
                action_set=self._action_set,
                subaction_path=xr.NULL_PATH,
            )
        ])

        # Create action spaces for aim poses (used to locate controller each frame)
        for act, attr in [
            (self._act_aim_left,  "_aim_space_l"),
            (self._act_aim_right, "_aim_space_r"),
        ]:
            try:
                space = xr.create_action_space(
                    self._xr_session,
                    xr.ActionSpaceCreateInfo(
                        action=act,
                        pose_in_action_space=xr.Posef(),
                    ),
                )
                setattr(self, attr, space)
            except Exception as e:
                print(f"[OpenXRViewer] Aim space creation failed: {e}")

        # Create action spaces for grip poses (used to place controller 3D models)
        for act, attr in [
            (self._act_grip_left,  "_grip_space_l"),
            (self._act_grip_right, "_grip_space_r"),
        ]:
            if act is None:
                continue
            try:
                space = xr.create_action_space(
                    self._xr_session,
                    xr.ActionSpaceCreateInfo(
                        action=act,
                        pose_in_action_space=xr.Posef(),
                    ),
                )
                setattr(self, attr, space)
            except Exception as e:
                print(f"[OpenXRViewer] Grip space creation failed: {e}")


    def _poll_xr_events(self):
        """Drain the OpenXR event queue and handle session state transitions."""
        from utils import shutdown_event
        while True:
            try:
                event_buf = xr.poll_event(self._xr_instance)
            except xr.EventUnavailable:
                break

            event_type = event_buf.type

            if event_type == xr.StructureType.EVENT_DATA_SESSION_STATE_CHANGED:
                event = ctypes.cast(
                    ctypes.byref(event_buf),
                    ctypes.POINTER(xr.EventDataSessionStateChanged),
                ).contents
                state = xr.SessionState(event.state)
                if state == xr.SessionState.READY:
                    xr.begin_session(
                        self._xr_session,
                        xr.SessionBeginInfo(
                            primary_view_configuration_type=
                                xr.ViewConfigurationType.PRIMARY_STEREO
                        ),
                    )
                    self._session_running = True
                    print("[OpenXRViewer] Session READY rendering started")

                elif state in (
                    xr.SessionState.STOPPING,
                    xr.SessionState.LOSS_PENDING,
                    xr.SessionState.EXITING,
                ):
                    xr.end_session(self._xr_session)
                    self._session_running = False
                    print(f"[OpenXRViewer] Session state {state.name}; rendering paused")

            elif event_type == xr.StructureType.EVENT_DATA_REFERENCE_SPACE_CHANGE_PENDING:
                # Long-pressing the Oculus/home button triggers a playspace recenter,
                # which emits this event.  Defer the re-seat to the next frame so
                # the Quest's own recenter is reflected in the views first.
                self._pending_recenter = True

            elif event_type == xr.StructureType.EVENT_DATA_INSTANCE_LOSS_PENDING:
                print("[OpenXRViewer] Instance loss pending shutting down")
                shutdown_event.set()
                break

    def _read_bool_action(self, action, hand_path_str="/user/hand/left"):
        """Return True if the boolean action is currently pressed on the given hand."""
        if action is None:
            return False
        try:
            path = (self._path_left
                    if hand_path_str == "/user/hand/left" else self._path_right)
            if path is None:
                path = xr.string_to_path(self._xr_instance, hand_path_str)
            state = xr.get_action_state_boolean(
                self._xr_session,
                xr.ActionStateGetInfo(action=action, subaction_path=path),
            )
            return bool(state.is_active and state.current_state)
        except Exception:
            return False

    def _read_bool_edge(self, action, hand_path_str, prev_state):
        """Return True on the rising edge of a boolean action.

        Tries to use the OpenXR runtime's `changed` flag via the raw ctypes struct
        (pyopenxr may not expose it as a Python attribute).  Falls back to manual
        frame-to-frame comparison if the ctypes path fails.
        """
        if action is None:
            return False
        try:
            path = (self._path_left
                    if hand_path_str == "/user/hand/left" else self._path_right)
            if path is None:
                path = xr.string_to_path(self._xr_instance, hand_path_str)
            state = xr.get_action_state_boolean(
                self._xr_session,
                xr.ActionStateGetInfo(action=action, subaction_path=path),
            )
            pressed = bool(state.is_active and state.current_state)

            # pyopenxr wraps XrActionStateBoolean. Try the Python attribute first,
            # then fall back to reading the underlying ctypes struct.
            changed = False
            if hasattr(state, 'changed'):
                changed = bool(state.changed)
            else:
                # The struct is [isActive:i4, currentState:i4, changed:i4, ...]
                # changed is at byte offset 8 (after two 4-byte fields).
                try:
                    ptr = ctypes.cast(ctypes.byref(state), ctypes.POINTER(ctypes.c_int32))
                    changed = bool(ptr[2])  # offset 2 × 4 bytes
                except Exception:
                    pass

            if changed:
                return pressed   # runtime-confirmed edge
            # Fallback: manual rising-edge detection
            return pressed and not prev_state
        except Exception:
            return False

    def _update_trackpad_button_emu(self):
        """Compute per-frame Vive trackpad button emulation flags.

        On controllers with a clickable trackpad, the physical click position
        emulates face buttons:
          center (|y| <= 0.5) thumbstick click
          top    (y > 0.5)   B (right) / Y (left)
          bottom (y < -0.5)  A (right) / X (left)

        On controllers with real buttons the emulation is harmless: the real
        reads dominate via OR, and the thumbstick self-centres near (0,0) so
        only the center flag fires (matching the raw stick-click).
        """
        for hand, stick_act, click_act, attr_top, attr_bot, attr_ctr in [
            ("/user/hand/left",  self._act_left_stick,  self._act_left_stick_click,
             '_emu_y', '_emu_x', '_emu_lsc'),
            ("/user/hand/right", self._act_right_stick, self._act_right_stick_click,
             '_emu_b', '_emu_a', '_emu_rsc'),
        ]:
            clicked = self._read_bool_action(click_act, hand)
            if not clicked:
                setattr(self, attr_top, False)
                setattr(self, attr_bot, False)
                setattr(self, attr_ctr, False)
                continue
            try:
                path = self._path_left if hand == "/user/hand/left" else self._path_right
                state = xr.get_action_state_vector2f(
                    self._xr_session,
                    xr.ActionStateGetInfo(action=stick_act, subaction_path=path),
                )
                py = float(state.current_state.y) if state.is_active else 0.0
            except Exception:
                py = 0.0
            if py > _VIVE_TB_Y:
                setattr(self, attr_top, True)
                setattr(self, attr_bot, False)
                setattr(self, attr_ctr, False)
            elif py < -_VIVE_TB_Y:
                setattr(self, attr_top, False)
                setattr(self, attr_bot, True)
                setattr(self, attr_ctr, False)
            else:
                setattr(self, attr_top, False)
                setattr(self, attr_bot, False)
                setattr(self, attr_ctr, True)

    def _read_float_action(self, action, hand_path_str="/user/hand/left"):
        """Return the float value [0,1] of a trigger/squeeze action."""
        if action is None:
            return 0.0
        try:
            path = (self._path_left
                    if hand_path_str == "/user/hand/left" else self._path_right)
            if path is None:
                path = xr.string_to_path(self._xr_instance, hand_path_str)
            state = xr.get_action_state_float(
                self._xr_session,
                xr.ActionStateGetInfo(action=action, subaction_path=path),
            )
            return float(state.current_state) if state.is_active else 0.0
        except Exception:
            return 0.0


"""InputHandlerMixin: cursor, trigger, keyboard, scroll, and controller input polling."""

import math
import sys
import time
import ctypes

try:
    import xr
except ImportError:
    xr = None

import numpy as np

from .constants import (
    DEAD, EDGE_STRENGTH, KB_CURSOR_PRIORITY_BIAS, KB_CURSOR_RELEASE_GRACE,
    _KB_UNITS_WIDE, _KB_ROWS, _KeyEntry,
    _KB_TEX_W, _KB_TEX_H, _VIVE_TB_Y,
)
from .input import (
    _MOUSEEVENTF_LEFTDOWN, _MOUSEEVENTF_LEFTUP,
    _MOUSEEVENTF_RIGHTDOWN, _MOUSEEVENTF_RIGHTUP,
    _TOUCH_AVAILABLE, _TOUCH_CONTACT_ID_LEFT, _TOUCH_CONTACT_ID_RIGHT,
    _TOUCH_PINCH_SPREAD_GAIN,
    _set_cursor_pos, _send_mouse_flags,
    _send_key, _send_vscroll, _send_hscroll, _touch_injector,
    _U32, _KEYEVENTF_KEYUP,
)


class InputHandlerMixin:
    def _handle_cursor(self):
        """Move the Windows mouse cursor when a controller laser is pointing at the screen.

        Cursor jitter at long laser distances comes from natural hand tremor: a 0.5°
        wrist wobble at 2 m laser length is ~17 mm at the screen, which is hundreds
        of cursor pixels. We low-pass-filter the UV with an exponential moving average
        to take the high-frequency edge off without adding perceptible lag.

        Controller cursor control is only active when the laser actually intersects the
        screen quad. When no laser hits the screen, the physical mouse has full control.
        """
        PHYS_TIMEOUT = 3.0

        def _beam_origin_dir(aim_mat, grip_mat, smooth_pos_attr, smooth_quat_attr):
            """Same logic as _laser_beam_setup for smoothing and edge constraints."""
            is_left = smooth_pos_attr.endswith('_l')
            cp, fw = self._get_smoothed_ray(is_left)
            if cp is None:
                if grip_mat is not None:
                    raw_pos = (grip_mat[:3, 3] + grip_mat[:3, 1] * 0.020).astype('f8')
                else:
                    raw_pos = aim_mat[:3, 3].astype('f8')
                fw = -aim_mat[:3, 2].astype('f8')
                cp = raw_pos + fw * 0.11
                return cp, fw
            # raw_pos for edge constraint (unsmoothed origin)
            if grip_mat is not None:
                raw_pos = (grip_mat[:3, 3] + grip_mat[:3, 1] * 0.020).astype('f8')
            else:
                raw_pos = aim_mat[:3, 3].astype('f8')
            right = aim_mat[:3, 0].astype('f8')
            ang = math.radians(12); ca, sa = math.cos(ang), math.sin(ang)
            k = right / (np.linalg.norm(right) + 1e-10)
            fw = fw * ca + np.cross(k, fw) * sa + k * np.dot(k, fw) * (1 - ca)
            # Keyboard targeting takes precedence over screen edge-snapping 
            # mirrors _laser_beam_setup so the screen cursor isn't deflected onto
            # the screen edge (and thus stolen away from the keyboard) when the
            # keyboard sits close below the screen.
            _kb_targeted = (self._keyboard_visible and
                            self._keyboard_laser_hit_dist(raw_pos, fw) < 30.0)
            # Screen edge constraint: if the smoothed ray misses the screen but the raw ray is close, clamp to the edge.
            if not _kb_targeted and self._laser_screen_hit_uv(raw_pos, fw) is None:
                _raw_fw = -aim_mat[:3, 2].astype('f8')
                _raw_rw = aim_mat[:3, 0].astype('f8')
                _raw_k = _raw_rw / (np.linalg.norm(_raw_rw) + 1e-10)
                _raw_fw = _raw_fw * ca + np.cross(_raw_k, _raw_fw) * sa + _raw_k * np.dot(_raw_k, _raw_fw) * (1 - ca)
                if self._laser_screen_hit_uv(raw_pos, _raw_fw) is None:
                    _plane_uv = self._laser_plane_uv(raw_pos, fw)
                    if _plane_uv is not None:
                        _cu = max(0.0, min(1.0, _plane_uv[0]))
                        _cv = max(0.0, min(1.0, _plane_uv[1]))
                        _clamped_wp = self._screen_uv_to_world(_cu, _cv)
                        _edge_dir = _clamped_wp - raw_pos
                        _norm = np.linalg.norm(_edge_dir)
                        if _norm > 1e-6:
                            _edge_dir /= _norm
                            _dot2 = np.dot(_raw_fw, _edge_dir)
                            _dot2 = max(-1.0, min(1.0, _dot2))
                            _ang2 = math.acos(_dot2)
                            if _ang2 < self._ray_edge_deadzone_rad:
                                fw = _edge_dir
            # Beam starts from raw grip position (matches _laser_beam_setup)
            cp = raw_pos + fw * 0.11
            return cp, fw

        # Physical mouse detection: if the physical mouse has moved recently,
        # suppress VR cursor control unconditionally so the user can use the
        # physical mouse without fighting the VR cursor.  VR resumes after a
        # quiet period with no physical mouse movement.
        # Check FIRST before expensive ray casting to skip work when mouse is active.
        if sys.platform == "win32":
            if (time.perf_counter() - self._phys_mouse_last_move) < PHYS_TIMEOUT:
                self._cursor_ctrl = None
                self._cursor_smooth_uv = None
                # Invalidate touch positions so any held contact is released by
                # _handle_triggers when physical mouse takes over.
                self._touch_valid_l = False
                self._touch_valid_r = False
                return
            # Throttle GetCursorPos to every ~50ms (3-4 frames at 72Hz) 
            # per-frame polling is wasteful; physical mouse detection doesn't need sub-frame precision.
            _now = time.perf_counter()
            if _now - getattr(self, '_last_get_cursor_pos_time', 0.0) >= 0.05:
                class _POINT(ctypes.Structure):
                    _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]
                _pt = _POINT()
                ctypes.windll.user32.GetCursorPos(ctypes.byref(_pt))
                self._last_get_cursor_pos_time = _now
                _cur_pos = (_pt.x, _pt.y)
                if self._phys_mouse_pos is not None and _cur_pos != self._phys_mouse_pos:
                    vcp = self._vr_cursor_screen_pos
                    if vcp is None or abs(_cur_pos[0] - vcp[0]) > 4 or abs(_cur_pos[1] - vcp[1]) > 4:
                        self._phys_mouse_last_move = _now
                self._phys_mouse_pos = _cur_pos
            if (time.perf_counter() - self._phys_mouse_last_move) < PHYS_TIMEOUT:
                self._cursor_ctrl = None
                self._cursor_smooth_uv = None
                self._touch_valid_l = False
                self._touch_valid_r = False
                return

        hit_l = hit_r = None
        ov_hit_l = ov_hit_r = False
        # Read both triggers every frame.  The keyboard typing-lock below only
        # needs them when the keyboard is visible, but the cursor-ownership rule
        # ("latest click / tap wins") needs them every frame to know which hand
        # most recently clicked so we always poll.
        ltrig_now = self._read_float_action(self._act_left_trigger,  "/user/hand/left")
        rtrig_now = self._read_float_action(self._act_right_trigger, "/user/hand/right")
        # Beam origin/direction per hand stashed for the touch-publish block
        # below so it can clamp the off-screen drag position to the screen
        # edge instead of letting `_touch_px_l/r` freeze at the last on-screen
        # pixel.
        cp_l = fw_l = cp_r = fw_r = None
        if self._aim_mat_l is not None:
            cp, fw = _beam_origin_dir(self._aim_mat_l, self._grip_mat_l,
                                    "_smooth_ray_origin_l", "_smooth_ray_quat_l")
            cp_l, fw_l = cp, fw
            # Compute both keyboard and screen hit distances, only interact with closer one
            _kb_idx_l, _kb_t_l = self._keyboard_laser_hit(cp, fw)
            hit_l = self._laser_screen_hit_uv(cp, fw)
            kb_dist_l = self._keyboard_laser_hit_dist(cp, fw)
            sc_dist_l = hit_l[2] if hit_l is not None else float('inf')
            # `_keyboard_laser_hit_dist` returns the BEAM_MAX sentinel (30 m)
            # when the laser misses the keyboard rectangle.  We must use this
            # sentinel not `< float('inf')` to test "laser is on keyboard",
            # otherwise typing_lock would falsely engage whenever the user
            # pulls the trigger on the screen with the keyboard merely
            # visible, blocking touch clicks on the screen entirely.
            _KB_BEAM_MAX = 30.0
            kb_actually_hit_l = kb_dist_l < _KB_BEAM_MAX
            # Keep keyboard priority stable when it is very near the screen
            # (e.g. top rows close to screen bottom) to avoid visual cursor dropouts.
            typing_lock_l = (
                self._keyboard_visible
                and kb_actually_hit_l
                and (
                    self._kb_held_key_l is not None
                    or (ltrig_now >= 0.55)
                )
            )
            if typing_lock_l or (kb_actually_hit_l and kb_dist_l <= (sc_dist_l + KB_CURSOR_PRIORITY_BIAS)):
                # Refresh the hold timer while the keyboard owns the cursor so the
                # post-release grace measures time since the LAST owned frame.
                self._kb_cursor_owned_t_l = time.perf_counter()
                hit_l = None  # keyboard is closer, suppress screen cursor
            elif (self._keyboard_visible and
                  (time.perf_counter() - self._kb_cursor_owned_t_l) < self._KB_RELEASE_GRACE):
                # Post-release grace: keep the screen cursor suppressed briefly while
                # the user lifts off the keyboard toward the screen (smooths the swap).
                hit_l = None
            else:
                self._kb_hover_l = None  # screen is closer, suppress keyboard hover
            ov_cp_l, ov_fw_l = self._pre_snap_overlay_ray(True, self._aim_mat_l, self._grip_mat_l)
            ov_dist_l = self._overlay_panel_hit_dist(ov_cp_l, ov_fw_l)
            ov_hit_l = ov_dist_l < 5.0
            if ov_hit_l and hit_l is not None and ov_dist_l < hit_l[2]:
                hit_l = None  # suppress cursor when overlay is closer than screen
        if self._aim_mat_r is not None:
            cp, fw = _beam_origin_dir(self._aim_mat_r, self._grip_mat_r,
                                    "_smooth_ray_origin_r", "_smooth_ray_quat_r")
            cp_r, fw_r = cp, fw
            _kb_idx_r, _kb_t_r = self._keyboard_laser_hit(cp, fw)
            hit_r = self._laser_screen_hit_uv(cp, fw)
            kb_dist_r = self._keyboard_laser_hit_dist(cp, fw)
            sc_dist_r = hit_r[2] if hit_r is not None else float('inf')
            _KB_BEAM_MAX = 30.0
            kb_actually_hit_r = kb_dist_r < _KB_BEAM_MAX
            typing_lock_r = (
                self._keyboard_visible
                and kb_actually_hit_r
                and (
                    self._kb_held_key_r is not None
                    or (rtrig_now >= 0.55)
                )
            )
            if typing_lock_r or (kb_actually_hit_r and kb_dist_r <= (sc_dist_r + KB_CURSOR_PRIORITY_BIAS)):
                # Refresh the hold timer while the keyboard owns the cursor so the
                # post-release grace measures time since the LAST owned frame.
                self._kb_cursor_owned_t_r = time.perf_counter()
                hit_r = None  # keyboard is closer, suppress screen cursor
            elif (self._keyboard_visible and
                  (time.perf_counter() - self._kb_cursor_owned_t_r) < self._KB_RELEASE_GRACE):
                # Post-release grace: keep the screen cursor suppressed briefly while
                # the user lifts off the keyboard toward the screen (smooths the swap).
                hit_r = None
            else:
                self._kb_hover_r = None  # screen is closer, suppress keyboard hover
            ov_cp_r, ov_fw_r = self._pre_snap_overlay_ray(False, self._aim_mat_r, self._grip_mat_r)
            ov_dist_r = self._overlay_panel_hit_dist(ov_cp_r, ov_fw_r)
            ov_hit_r = ov_dist_r < 5.0
            if ov_hit_r and hit_r is not None and ov_dist_r < hit_r[2]:
                hit_r = None  # suppress cursor when overlay is closer than screen

        self._cursor_uv_l = hit_l if hit_l else None   # (u, v, t) or None
        self._cursor_uv_r = hit_r if hit_r else None   # (u, v, t) or None
        self._overlay_hit_l = ov_hit_l
        self._overlay_hit_r = ov_hit_r
        self._ray_prev_uv_l = self._cursor_uv_l
        self._ray_prev_uv_r = self._cursor_uv_r

        # Publish per-hand desktop pixel positions for the multi-touch injector.
        # Each hand is tracked independently so two simultaneous triggers become
        # two simultaneous touch contacts (Windows multi-touch pinch/zoom,
        # two-finger pan, press-and-hold right-click, etc.).
        #
        # When the laser is OFF-SCREEN we project the ray onto the screen plane
        # and clamp the UV to [0, 1] so a drag-in-progress can keep updating at
        # the screen edge in the direction the laser is pointing.  Without this
        # clamp, `_touch_px_l/r` froze at the last on-screen pixel fast drags
        # that briefly grazed off-screen left the touch contact stuck at the
        # edge with the cursor visibly lagging behind the laser, then jumping
        # when the laser returned.  Edge-clamping makes the drag continue at
        # the edge so the OS sees uninterrupted motion data.  `valid` is still
        # False off-screen so no NEW touch can fire there only an active
        # drag uses the clamped position.
        try:
            mon_left, mon_top, mon_w, mon_h = self._get_target_monitor_rect()
        except Exception:
            mon_left = mon_top = 0; mon_w = mon_h = 0
        def _uv_to_px(uv):
            if uv is None or mon_w <= 0 or mon_h <= 0:
                return None
            u, v = float(uv[0]), float(uv[1])
            self._movie_crop_note_cursor_uv(u, v)
            src_u, src_top_v = self._screen_uv_to_source_top_uv(u, v)
            return (mon_left + int(src_u * mon_w),
                    mon_top + int(src_top_v * mon_h))
        def _edge_px(cp, fw):
            """Project ray onto screen plane, clamp UV to [0,1], return pixels."""
            if cp is None or fw is None or mon_w <= 0 or mon_h <= 0:
                return None
            uv = self._laser_plane_uv(cp, fw)
            if uv is None:
                return None
            u = max(0.0, min(1.0, float(uv[0])))
            v = max(0.0, min(1.0, float(uv[1])))
            src_u, src_top_v = self._screen_uv_to_source_top_uv(u, v)
            return (mon_left + int(src_u * mon_w),
                    mon_top + int(src_top_v * mon_h))
        _pl = _uv_to_px(hit_l)
        _pr = _uv_to_px(hit_r)
        if _pl is not None:
            self._touch_px_l = _pl
            self._touch_valid_l = True
        else:
            # Off-screen / keyboard-claimed / overlay-claimed: clamp to edge so
            # an active drag stays alive; mark invalid so no new DOWN fires.
            edge_l = _edge_px(cp_l, fw_l)
            if edge_l is not None:
                self._touch_px_l = edge_l
            self._touch_valid_l = False
        if _pr is not None:
            self._touch_px_r = _pr
            self._touch_valid_r = True
        else:
            edge_r = _edge_px(cp_r, fw_r)
            if edge_r is not None:
                self._touch_px_r = edge_r
            self._touch_valid_r = False

        # ▶▶Pick the active cursor controller: LATEST CLICK / TAP WINS ▶▶
        # When BOTH lasers are on the screen we give the cursor to whichever
        # controller most recently pulled its trigger (a click / tap), so the
        # user can hand control back and forth between hands just by clicking.
        # The old rule was "right always wins", which made the off-hand unable
        # to ever take over while the right laser merely grazed the screen.
        #
        # Rising-edge stamp: record the click time the instant the trigger
        # crosses PRESS while THAT hand's laser is on the screen.  Gating on
        # `hit_*` means a press on the keyboard / overlay (where `hit_*` is
        # already None) never steals the screen cursor.  Pure hover never
        # changes the stamps, so two resting lasers never ping-pong the last
        # hand to click keeps control until the other hand clicks.
        now_pc = time.perf_counter()
        _CURSOR_PRESS = 0.40
        if hit_l and ltrig_now >= _CURSOR_PRESS and self._cursor_trig_prev_l < _CURSOR_PRESS:
            self._cursor_click_ts_l = now_pc
        if hit_r and rtrig_now >= _CURSOR_PRESS and self._cursor_trig_prev_r < _CURSOR_PRESS:
            self._cursor_click_ts_r = now_pc
        self._cursor_trig_prev_l = ltrig_now
        self._cursor_trig_prev_r = rtrig_now

        if hit_l and hit_r:
            # Both lasers on screen newer click owns the cursor.
            if self._cursor_click_ts_r > self._cursor_click_ts_l:
                ctrl = 'right'
            elif self._cursor_click_ts_l > self._cursor_click_ts_r:
                ctrl = 'left'
            else:
                # Neither has clicked yet (equal stamps, e.g. both 0.0) keep
                # the current owner so the cursor doesn't ping-pong; default to
                # right on the very first frame.
                ctrl = self._cursor_ctrl if self._cursor_ctrl in ('left', 'right') else 'right'
        elif hit_r:
            ctrl = 'right'
        elif hit_l:
            ctrl = 'left'
        else:
            self._cursor_ctrl = None
            self._cursor_smooth_uv = None   # reset so next entry doesn't drag
            return

        # On an ownership swap, drop the stale smoothing anchor so the new hand's
        # cursor starts exactly where ITS laser points instead of sliding across
        # the screen from the previous owner's last position that slide was the
        # "lag" the user saw when control changed hands.
        if ctrl != self._cursor_ctrl:
            self._cursor_smooth_uv = None

        self._cursor_ctrl = ctrl
        u, v = (hit_r[0], hit_r[1]) if ctrl == 'right' else (hit_l[0], hit_l[1])

        # IMPORTANT no second smoothing stage here.
        # The controller ray is already low-pass filtered upstream by the
        # One-Euro filter in `_get_smoothed_ray`.  A second EMA at this layer
        # (the old `ALPHA = 0.35` cursor smoother) stacked latency on top of
        # that and was a root cause of the drag problems the user reported:
        #   fast drags moved the window only a short distance (the smoother
        #     never caught up before release),
        #   the window/cursor visibly trailed behind the beam,
        #   the cursor kept gliding after the trigger was released (the EMA
        #     was still converging to the final position), and
        #   drag felt "sticky".
        # It also disagreed with the touch-contact position (published above
        # from the RAW `hit_*` UV), so Windows which pins the cursor to the
        # active touch contact fought this SetCursorPos call.
        # Use the raw UV directly: what the beam points at is where the cursor
        # (and the touch contact) goes, with zero added lag.  This is also a
        # few ops cheaper per frame, so it's FPS-neutral-to-positive.
        self._cursor_smooth_uv = (u, v)

        mon_left, mon_top, mon_w, mon_h = self._get_target_monitor_rect()
        self._movie_crop_note_cursor_uv(u, v)
        src_u, src_top_v = self._screen_uv_to_source_top_uv(u, v)
        px = mon_left + int(src_u * mon_w)
        py = mon_top + int(src_top_v * mon_h)
        # Always track the VR cursor position so the physical-mouse detector
        # doesn't falsely fire when grip ends and the cursor resumes moving.
        self._vr_cursor_screen_pos = (px, py)
        # Suppress cursor movement while gripping or during a two-finger
        # touch gesture.  When both touch contacts are active, SetCursorPos
        # fights the multi-touch subsystem and breaks pinch/zoom/pan because
        # Windows pins the cursor to the active contact moving it to the
        # "winning" hand's position mid-gesture collapses the two-contact
        # interaction into a single-point one.
        both_touch_down = (self._touch_state_l == 'down'
                           and self._touch_state_r == 'down')
        if not self._grabbed and not both_touch_down:
            _set_cursor_pos(px, py)

    def _emit_left_click(self):
        """Send a left click, snapping to the previous click pixel so rapid
        taps register as a real OS double-click.

        A VR laser drifts a few pixels between taps, so two clicks at slightly
        different pixels are seen by Windows as two singles, not a double.  If a
        second click lands within the double-click time window, we move the
        cursor back to the exact prior pixel before firing so the OS accepts it
        as a double-click (matching real-mouse behaviour)."""
        now = time.perf_counter()
        px = py = None
        pos = getattr(self, '_vr_cursor_screen_pos', None)
        if pos is not None:
            px, py = pos
        # Double-click window in seconds (system value is ms; widened in run()).
        # Use 3x the system window so VR trigger/A taps that are naturally slower
        # and less rhythmic than a mouse still register as a double-click.
        try:
            dclick_s = (_U32.GetDoubleClickTime() / 1000.0) * 3.0
        except Exception:
            dclick_s = 1.5
        last_px = getattr(self, '_last_click_px', None)
        if (last_px is not None and px is not None
                and (now - self._last_click_ts) <= dclick_s):
            # Snap to the first click's pixel so the OS pairs the two clicks.
            _set_cursor_pos(last_px[0], last_px[1])
            px, py = last_px
        _send_mouse_flags(_MOUSEEVENTF_LEFTDOWN)
        _send_mouse_flags(_MOUSEEVENTF_LEFTUP)
        self._last_click_ts = now
        if px is not None:
            self._last_click_px = (px, py)

    def _handle_triggers(self):
        """Map controller triggers to Windows multi-touch contacts (preferred)
        or mouse clicks (fallback).

        Per-hand contact lifecycle when ``_TOUCH_AVAILABLE``:

        * Trigger ≥PRESS_THRESH on a valid screen target touch DOWN.
        * Trigger held touch UPDATE every frame (drives drag, incl. window
          title-bar drag Windows treats it as one continuous interaction).
        * Trigger < RELEASE_THRESH (or laser leaves a usable target) touch UP.

        Both controllers active simultaneously become two-contact multi-touch,
        enabling the gestures documented at
        https://support.microsoft.com/en-us/windows/touch-gestures-for-windows-a9d28305-4818-a5df-4e2b-e5590f850741
        (tap = click, drag = drag, two-finger pan/zoom, press-and-hold =
        right-click, edge swipes for notification center / widgets, etc.).

        If a trigger fires while the laser hits the FPS/status panel, that
        trigger toggles the shortcuts/help panel instead of generating a click.
        """
        PRESS_THRESH   = 0.40   # rising edge (loosened: partial pull clicks)
        RELEASE_THRESH = 0.20   # falling edge (hysteresis)
        HOLD_TIME      = 0.22   # seconds trigger must stay held to enter drag mode (mouse fallback only)

        # While gripping (user repositioning the screen/keyboard), release any
        # active touch contacts cleanly so the next press starts fresh and
        # skip all further click processing for this frame.
        if self._grabbed:
            if _TOUCH_AVAILABLE and _touch_injector is not None:
                if self._touch_state_l == 'down':
                    _touch_injector.set(_TOUCH_CONTACT_ID_LEFT,
                                        self._touch_px_l[0], self._touch_px_l[1],
                                        want_down=False)
                    self._touch_state_l = 'idle'
                    self._touch_smooth_l = None
                if self._touch_state_r == 'down':
                    _touch_injector.set(_TOUCH_CONTACT_ID_RIGHT,
                                        self._touch_px_r[0], self._touch_px_r[1],
                                        want_down=False)
                    self._touch_state_r = 'idle'
                    self._touch_smooth_r = None
                _touch_injector.flush()
            # Seed the per-hand prior-trigger trackers to the current readings so
            # that on the frame the user releases the grip, the rising-edge gate
            # in the touch path still requires a true release-then-press before
            # firing a new touch DOWN (avoids "drop the grip instant phantom
            # click" if the trigger happens to be high at grip-release time).
            self._touch_trig_prev_l = self._read_float_action(
                self._act_left_trigger,  "/user/hand/left")
            self._touch_trig_prev_r = self._read_float_action(
                self._act_right_trigger, "/user/hand/right")
            return

        now = self._frame_now
        lt  = self._read_float_action(self._act_left_trigger,  "/user/hand/left")
        rt  = self._read_float_action(self._act_right_trigger, "/user/hand/right")

        # Trigger-click on FPS/status panel toggles shortcut help panel.
        # Uses _overlay_hit_l/r (same hit test as cursor suppression) so the
        # toggle and click-block are consistent with what the user sees.
        ov_claim_l = False
        ov_claim_r = False

        if self._fps_overlay_visible:
            for hit_overlay, trig_now, is_left in (
                (self._overlay_hit_l, lt, True),
                (self._overlay_hit_r, rt, False),
            ):
                trig_prev_held = self._ov_ltrig_held if is_left else self._ov_rtrig_held

                if hit_overlay and trig_now >= PRESS_THRESH and not trig_prev_held:
                    self._help_panel_visible = not self._help_panel_visible

                if is_left:
                    # Only track held state while the laser is on the overlay;
                    # reset when the laser leaves so the next press fires cleanly.
                    self._ov_ltrig_held = hit_overlay and trig_now >= PRESS_THRESH
                    if hit_overlay:
                        ov_claim_l = True
                else:
                    self._ov_rtrig_held = hit_overlay and trig_now >= PRESS_THRESH
                    if hit_overlay:
                        ov_claim_r = True

        left_on_kb  = self._kb_hover_l is not None
        right_on_kb = self._kb_hover_r is not None
        # Treat the keyboard as "claiming" the hand whenever a virtual key is
        # currently being typed by that hand, even if the laser drifted off the
        # key this frame.  Without this, releasing a key by sliding the laser
        # off it (while the trigger is still held) could leak into a touch DOWN
        # on the screen behind/below the keyboard.
        left_kb_typing  = self._kb_held_key_l is not None
        right_kb_typing = self._kb_held_key_r is not None

        # ▶Touch path (preferred): clean DOWN/UPDATE/UP per hand ▶        # Reliable click: a single DOWN-UP pair is unambiguous to Windows,
        # unlike the mouse-flag pulse that occasionally raced with cursor
        # positioning and produced missed clicks.
        if _TOUCH_AVAILABLE and _touch_injector is not None:
            touch_updates = []
            for (trig, valid, px_attr, smooth_attr, state_attr, trig_prev_attr,
                 contact_id, ov_claim, on_kb, kb_typing) in (
                (lt, self._touch_valid_l, '_touch_px_l', '_touch_smooth_l',
                 '_touch_state_l', '_touch_trig_prev_l',
                 _TOUCH_CONTACT_ID_LEFT, ov_claim_l, left_on_kb, left_kb_typing),
                (rt, self._touch_valid_r, '_touch_px_r', '_touch_smooth_r',
                 '_touch_state_r', '_touch_trig_prev_r',
                 _TOUCH_CONTACT_ID_RIGHT, ov_claim_r, right_on_kb, right_kb_typing),
            ):
                state    = getattr(self, state_attr)
                raw_px   = getattr(self, px_attr)
                trig_prev = getattr(self, trig_prev_attr)
                # Keyboard-claimed hand: no touch this frame, period.  Covers
                # both "laser on a key" and "still holding a typed key" cases.
                kb_claim = on_kb or kb_typing

                # Determine the desired contact state this frame.
                if state == 'down':
                    # Stay down until trigger releases OR the hand loses its
                    # screen target (laser left screen / moved to keyboard /
                    # claimed by overlay).  We use the looser RELEASE_THRESH so
                    # the contact survives small trigger dips during drag.
                    want_down = (trig > RELEASE_THRESH
                                 and not ov_claim
                                 and not kb_claim)
                else:
                    # idle only fire on a TRUE rising edge against a valid
                    # screen target.  Requiring the prior frame's trigger to be
                    # below PRESS_THRESH prevents a phantom click when the user:
                    #   * slides the laser off the virtual keyboard onto the
                    #     screen while still holding the trigger, or
                    #   * toggles the keyboard off mid-press.
                    # In both cases the trigger never released, so no new touch
                    # should fire on the screen until the user lets go.
                    want_down = (trig >= PRESS_THRESH
                                 and trig_prev < PRESS_THRESH
                                 and valid
                                 and not ov_claim
                                 and not kb_claim)

                touch_updates.append({
                    'want_down': want_down,
                    'state': state,
                    'raw_px': raw_px,
                    'smooth_attr': smooth_attr,
                    'state_attr': state_attr,
                    'trig_prev_attr': trig_prev_attr,
                    'contact_id': contact_id,
                    'trig': trig,
                })

            if (len(touch_updates) == 2
                    and touch_updates[0]['want_down']
                    and touch_updates[1]['want_down']
                    and _TOUCH_PINCH_SPREAD_GAIN > 1.0):
                try:
                    mon_left, mon_top, mon_w, mon_h = self._get_target_monitor_rect()
                except Exception:
                    mon_left = mon_top = mon_w = mon_h = 0

                def _clamp_px(x, y):
                    if mon_w > 0 and mon_h > 0:
                        x = max(mon_left, min(mon_left + mon_w - 1, x))
                        y = max(mon_top, min(mon_top + mon_h - 1, y))
                    return int(round(x)), int(round(y))

                p0 = touch_updates[0]['raw_px']
                p1 = touch_updates[1]['raw_px']
                cx = (float(p0[0]) + float(p1[0])) * 0.5
                cy = (float(p0[1]) + float(p1[1])) * 0.5
                gain = float(_TOUCH_PINCH_SPREAD_GAIN)
                touch_updates[0]['raw_px'] = _clamp_px(
                    cx + (float(p0[0]) - cx) * gain,
                    cy + (float(p0[1]) - cy) * gain,
                )
                touch_updates[1]['raw_px'] = _clamp_px(
                    cx + (float(p1[0]) - cx) * gain,
                    cy + (float(p1[1]) - cy) * gain,
                )

            for upd in touch_updates:
                want_down = upd['want_down']
                state = upd['state']
                raw_px = upd['raw_px']
                smooth_attr = upd['smooth_attr']
                state_attr = upd['state_attr']
                trig_prev_attr = upd['trig_prev_attr']
                contact_id = upd['contact_id']
                trig = upd['trig']

                if want_down:
                    # Send touch DOWN/UPDATE at the *current* laser-mapped
                    # pixel position no EMA, no snap-distance teleport.
                    #
                    # The controller pose is already smoothed upstream by
                    # `_get_smoothed_ray`, so any extra EMA here is pure
                    # latency: an α=0.45 second-stage EMA caused
                    #   a visible cursor jump after release (UP fired at
                    #     the lagging smoothed position; the cursor then
                    #     teleported to the actual laser position on the
                    #     next frame), and
                    #   a "fast drag = tiny window move" symptom
                    #     (smoother was catching up over ~10 frames so the
                    #     OS only saw ~70% of the user's drag distance), and
                    #   a "cursor hang" symptom (Windows pins the cursor
                    #     to the touch contact, which was crawling along the
                    #     EMA tail far behind the laser).
                    # Going direct-to-raw also drops ~14 ops/frame/hand, so
                    # if anything this is FPS-positive.
                    _touch_injector.set(contact_id, raw_px[0], raw_px[1],
                                        want_down=True)
                    setattr(self, smooth_attr, raw_px)  # last sent pos
                    setattr(self, state_attr, 'down')
                else:
                    if state == 'down':
                        # Release at the *current* laser position.  The
                        # injector promotes a moving-UP into UPDATE-then-UP
                        # internally so the OS still sees a clean transition.
                        # Releasing at the laser position (rather than at
                        # the last-sent position) also means there is no
                        # post-release cursor snap: the touch UP and the
                        # subsequent `_set_cursor_pos` call from
                        # `_handle_cursor` agree on where the cursor ends up.
                        _touch_injector.set(contact_id, raw_px[0], raw_px[1],
                                            want_down=False)
                    setattr(self, smooth_attr, None)
                    setattr(self, state_attr, 'idle')

                # Remember this frame's trigger value so the next frame can
                # detect a true rising edge.
                setattr(self, trig_prev_attr, trig)

            _touch_injector.flush()
            # If touch turned out to be unavailable (mid-session injection
            # failure), fall through to the mouse path so clicks still work.
            if _touch_injector.available:
                return

        # ▶Mouse fallback: original click+drag state machine ▶
        left_laser_usable  = (not left_on_kb and not ov_claim_l and
                            (self._cursor_uv_l is not None or
                            self._cursor_ctrl == 'left'))
        right_laser_usable = (not right_on_kb and not ov_claim_r and
                            (self._cursor_uv_r is not None or
                            self._cursor_ctrl == 'right'))

        any_drag = False

        for trig, usable, state_attr, press_t_attr in (
            (lt, left_laser_usable,  '_ltrig_state', '_ltrig_press_t'),
            (rt, right_laser_usable, '_rtrig_state', '_rtrig_press_t'),
        ):
            state = getattr(self, state_attr)

            if state == 'idle':
                if trig >= PRESS_THRESH and usable:
                    # Rising edge: fire an immediate click pulse so the OS can
                    # accumulate it toward double-click detection, then start timer.
                    self._emit_left_click()
                    setattr(self, state_attr,   'pressed')
                    setattr(self, press_t_attr, now)

            elif state == 'pressed':
                if not usable or trig <= RELEASE_THRESH:
                    # Released quickly click already delivered, return to idle.
                    setattr(self, state_attr, 'idle')
                elif (now - getattr(self, press_t_attr)) >= HOLD_TIME:
                    # Held long enough begin drag (send LEFTDOWN if not already down).
                    if not self._left_btn_down:
                        _send_mouse_flags(_MOUSEEVENTF_LEFTDOWN)
                        self._left_btn_down = True
                    setattr(self, state_attr, 'dragging')
                    any_drag = True

            elif state == 'dragging':
                if not usable or trig <= RELEASE_THRESH:
                    setattr(self, state_attr, 'idle')
                else:
                    any_drag = True

        # Send LEFTUP once both triggers leave drag state.
        if not any_drag and self._left_btn_down:
            _send_mouse_flags(_MOUSEEVENTF_LEFTUP)
            self._left_btn_down = False

    def _press_key(self, key, key_idx, held_key_attr, held_mods_attr):
        """Press and hold a regular key on the virtual keyboard (key-down only)."""
        kbd = ctypes.windll.user32.keybd_event
        VK_SHIFT = 0x10; VK_CTRL = 0x11; VK_ALT = 0x12; VK_WIN = 0x5B
        sh = self._mod_state['shift']
        ct = self._mod_state['ctrl']
        al = self._mod_state['alt']
        wn = self._mod_state['win']
        shift_on = sh[0] or sh[1]
        ctrl_on  = ct[0] or ct[1]
        alt_on   = al[0] or al[1]
        win_on   = wn[0] or wn[1]
        use_shift = shift_on ^ self._caps_lock
        vk_to_use = key.shifted_vk if use_shift else key.vk
        need_shift = use_shift and vk_to_use == key.vk
        if ctrl_on:     kbd(VK_CTRL,  0, 0, 0)
        if need_shift:  kbd(VK_SHIFT, 0, 0, 0)
        if alt_on:      kbd(VK_ALT,   0, 0, 0)
        if win_on:      kbd(VK_WIN,   0, 0, 0)
        kbd(vk_to_use, 0, 0, 0)
        setattr(self, held_key_attr, key_idx)
        setattr(self, held_mods_attr, (need_shift, ctrl_on, alt_on, win_on, vk_to_use))

    def _handle_keyboard_input(self):
        """Send Windows keystrokes when a controller trigger fires on a keyboard key.

        Regular keys use press-and-hold: key-down on trigger pull, key-up on release.
        Modifier keys (Shift/Ctrl/Alt/Win) use tap/lock toggles.  Caps toggles caps-lock.
        """
        if not self._keyboard_visible:
            self._kb_hover_l = None
            self._kb_hover_r = None
            return
        CLICK_THRESH  = 0.7
        RELEASE_THRESH = 0.3
        VK_SHIFT      = 0x10
        VK_CAPS       = 0x14
        VK_CTRL       = 0x11
        VK_ALT        = 0x12
        VK_WIN        = 0x5B
        kbd           = ctypes.windll.user32.keybd_event

        # Suppress keystrokes while any grip is held the user is repositioning
        # the keyboard, not typing.  Still update hover for the laser cursor /
        # grip-to-move logic, but force trigger inputs to "released" so no
        # rising-edge fires and any held key is released cleanly.
        gripping = bool(self._grip_l_now or self._grip_r_now)

        if gripping:
            lt = 0.0
            rt = 0.0
        else:
            lt = self._read_float_action(self._act_left_trigger,  "/user/hand/left")
            rt = self._read_float_action(self._act_right_trigger, "/user/hand/right")

        for trig_now, trig_prev_attr, hover_attr, held_key_attr, held_mods_attr, aim_mat in [
            (lt, '_kb_trig_prev_l', '_kb_hover_l', '_kb_held_key_l', '_kb_held_mods_l', self._aim_mat_l),
            (rt, '_kb_trig_prev_r', '_kb_hover_r', '_kb_held_key_r', '_kb_held_mods_r', self._aim_mat_r),
        ]:
            trig_prev = getattr(self, trig_prev_attr)
            held_key  = getattr(self, held_key_attr)
            held_mods = getattr(self, held_mods_attr)
            if aim_mat is not None:
                # Calculate the laser beam's origin point and forward direction in world space
                grip_mat = self._grip_mat_l if aim_mat is self._aim_mat_l else self._grip_mat_r
                fw = -aim_mat[:3, 2].astype('f8')
                right = aim_mat[:3, 0].astype('f8')
                _ang = math.radians(12); _ca, _sa = math.cos(_ang), math.sin(_ang)
                _k = right / (np.linalg.norm(right) + 1e-10)
                fw = fw * _ca + np.cross(_k, fw) * _sa + _k * np.dot(_k, fw) * (1 - _ca)
                if grip_mat is not None:
                    cp = (grip_mat[:3, 3] + grip_mat[:3, 1] * 0.020).astype('f8')
                else:
                    cp = aim_mat[:3, 3].astype('f8')
                cp = cp + fw * 0.11
                idx, kb_t = self._keyboard_laser_hit(cp, fw)
                # Only interact with keyboard if it's closer than the screen
                if idx is not None:
                    sc_t = self._laser_screen_hit_dist(cp, fw)
                    if sc_t is not None and sc_t < kb_t:
                        idx = None  # screen is closer, suppress keyboard interaction
            else:
                idx = None
            setattr(self, hover_attr, idx)

            # ▶Release held regular key when trigger drops or laser leaves the key ▶
            if held_key is not None:
                release = False
                if trig_now < RELEASE_THRESH:
                    release = True
                elif idx != held_key:
                    release = True
                if release:
                    shift_dn, ctrl_dn, alt_dn, win_dn, vk_held = held_mods
                    kbd(vk_held, 0, _KEYEVENTF_KEYUP, 0)
                    if win_dn:   kbd(VK_WIN,   0, _KEYEVENTF_KEYUP, 0)
                    if alt_dn:   kbd(VK_ALT,   0, _KEYEVENTF_KEYUP, 0)
                    if shift_dn: kbd(VK_SHIFT, 0, _KEYEVENTF_KEYUP, 0)
                    if ctrl_dn:  kbd(VK_CTRL,  0, _KEYEVENTF_KEYUP, 0)
                    # Auto-release one-shot modifiers that were armed when the key went down
                    for name in ('shift', 'ctrl', 'alt', 'win'):
                        self._mod_state[name][0] = False
                    setattr(self, held_key_attr, None)
                    setattr(self, held_mods_attr, None)
                    held_key  = None
                    held_mods = None

            # ▶Rising edge: modifier / caps toggles, or start holding a regular key ▶
            if trig_now >= CLICK_THRESH and trig_prev < CLICK_THRESH and idx is not None:
                key = self._keyboard_keys[idx]
                mod_name = {VK_SHIFT: 'shift', VK_CTRL: 'ctrl',
                            VK_ALT: 'alt', VK_WIN: 'win'}.get(key.vk)
                if mod_name is not None:
                    DOUBLE_TAP_WINDOW = 0.4
                    now_t = time.monotonic()
                    state = self._mod_state[mod_name]
                    if state[1]:
                        state[0] = False
                        state[1] = False
                    elif state[0]:
                        state[0] = False
                        _send_key(key.vk)
                    elif (now_t - state[2]) < DOUBLE_TAP_WINDOW:
                        state[0] = False
                        state[1] = True
                    else:
                        state[0] = True
                    state[2] = now_t
                elif key.vk == VK_CAPS:
                    self._caps_lock = not self._caps_lock
                else:
                    self._press_key(key, idx, held_key_attr, held_mods_attr)

            # ▶Slide to new key: trigger already held, laser moved to another regular key ▶
            if (held_key is None and trig_now >= CLICK_THRESH
                    and idx is not None and trig_prev >= CLICK_THRESH):
                key = self._keyboard_keys[idx]
                if key.vk not in (VK_SHIFT, VK_CTRL, VK_ALT, VK_WIN, VK_CAPS):
                    self._press_key(key, idx, held_key_attr, held_mods_attr)

            setattr(self, trig_prev_attr, trig_now)

        # After processing triggers, check if shift/caps state changed and
        # rebuild the keyboard texture so labels update visually.
        sh = self._mod_state['shift']
        cur_shifted = bool(sh[0] or sh[1] or self._caps_lock)
        if cur_shifted != self._kb_show_shifted:
            self._kb_show_shifted = cur_shifted
            self._build_keyboard_texture()

    def _accum_scroll(self, x_axis, y_axis, dt):
        """Accumulate thumbstick deflection into accelerated mouse wheel events.

        Uses a cubic acceleration curve so small deflections are precise while
        full push is dramatically faster eliminates the \"stuck\" feeling.

        Fires WHEEL_DELTA-granular (120) scroll so every event is a full
        hardware notch that applications process reliably.
        """
        WHEEL_DELTA        = 100     # Windows: one wheel notch
        SCROLL_BASE_NOTCH  = 2.0     # notches/s just above dead zone
        SCROLL_MAX_NOTCH   = 35.0    # notches/s at full deflection
        ACCEL_EXPONENT     = 2.8     # >1 = soft centre, aggressive at edges

        for axis_val, accum_attr, send_fn in [
            (x_axis, '_scroll_accum_x', _send_hscroll),
            (y_axis, '_scroll_accum_y', _send_vscroll),
        ]:
            mag = abs(axis_val)
            if mag <= DEAD:
                continue
            # Normalise [DEAD .. 1.0] [0 .. 1]
            t = (mag - DEAD) / (1.0 - DEAD)
            # Cubic acceleration + base offset ensures fine control near centre
            speed = SCROLL_BASE_NOTCH + (SCROLL_MAX_NOTCH - SCROLL_BASE_NOTCH) * (t ** ACCEL_EXPONENT)
            accum = getattr(self, accum_attr) + float(axis_val) * speed * dt
            # Fire whole notches; keep leftover for next frame
            whole = int(accum)
            if whole:
                send_fn(whole * WHEEL_DELTA)
                accum -= whole
            setattr(self, accum_attr, accum)

    def _poll_controller_input(self, dt):
        """Controller interaction mapping:
        Left stick (no grip)       Mouse wheel
        Left grip + Left stick X/Y Screen pan horizontally/vertically (always parallel to ground)
        Left grip + Right stick X  Screen yaw rotation around its center
        Left grip + Right stick Y  Screen pitch tilt forward/backward
        Right grip + Left stick Y  Depth intensity
        Right stick (no grip)      Mouse wheel
        Right grip + Right stick X Screen width adjustment
        Right grip + Right stick Y Screen distance (with acceleration curve)
        Left grip + Left stick (when keyboard visible) Keyboard pan (preserved)
        Left stick press hold 1s    Toggle FPS/shortcut panel
        Left stick short press      Cycle environment
        Right stick press hold 1s   Reset screen direction (when no grip)
        Right stick short press     Cycle horizontal curve / vertical curve / flat
        Both sticks press 0.5s       Toggle FPS/help panel
        A/B/X/Y/Menu/Triggers      Original functions unchanged
        Left trigger hold 3s (laser off screen)   Cycle Crop: Auto/Manual/Off (OSD)
        Left trigger double-tap (laser off screen, Manual mode)  Toggle crop-adjust pause + OSD
        Left stick (crop-adjust active, no grip)  Shrink/grow crop width (X) / height (Y), centered
        """
        if self._action_set is None:
            return

        self._update_trackpad_button_emu()

        def vec2(action, hand):
            try:
                path = (self._path_left
                        if hand == "/user/hand/left" else self._path_right)
                state = xr.get_action_state_vector2f(
                    self._xr_session,
                    xr.ActionStateGetInfo(action=action, subaction_path=path),
                )
                if state.is_active:
                    return state.current_state.x, state.current_state.y
            except Exception:
                pass
            return 0.0, 0.0

        lx, ly = vec2(self._act_left_stick,  "/user/hand/left")
        rx, ry = vec2(self._act_right_stick, "/user/hand/right")

        # Controller real-time calibration
        menu_now = self._read_bool_action(self._act_menu_btn, "/user/hand/left")
        a_now = self._read_bool_action(self._act_a_btn, "/user/hand/right") or self._emu_a
        b_now = self._read_bool_action(self._act_b_btn, "/user/hand/right") or self._emu_b

        # Calibration combo: Left Menu + Right A + Right B held for 1 second
        calib_combo = menu_now and a_now and b_now
        if calib_combo and not self._calib_combo_fired:
            if self._calib_combo_start == 0.0:
                self._calib_combo_start = time.perf_counter()
            elif time.perf_counter() - self._calib_combo_start >= 1.0:
                if self._calibration_mode:
                    self._exit_calibration_mode(save=False)
                else:
                    self._enter_calibration_mode()
                self._calib_combo_fired = True
                self._menu_long_fired = True  # suppress status-panel toggle on menu release
        if not calib_combo:
            self._calib_combo_start = 0.0
            self._calib_combo_fired = False

        # calibration mode: use sticks to adjust screen position/rotation with visual feedback, no cursor control
        if self._calibration_mode:
            CALIB_STEP = 0.005   # step size in meters for position adjustment
            ROT_STEP   = 0.5     # rotation step size in degrees
            if abs(ly) > 0.15:
                self._calibration_temp_offset[1] += ly * CALIB_STEP * 0.5
            if abs(ry) > 0.15:
                self._calibration_temp_offset[2] += ry * CALIB_STEP * 0.5
            if abs(rx) > 0.15:
                self._calibration_temp_rot += rx * ROT_STEP * 0.5
            # Button B to save and exit calibration mode on press (rising edge)
            b_edge = self._read_bool_edge(self._act_b_btn, "/user/hand/right", self._b_last)
            if not b_edge:
                b_edge = b_now and not self._b_last
            if b_edge:
                self._exit_calibration_mode(save=True)
            # calibration mode: disable normal stick operations
            lx, ly, rx, ry = 0.0, 0.0, 0.0, 0.0

        # Brand switching: Right A+B held for 1 second. Ensure ab_held is always defined for later A/B handling
        ab_held = False
        if not self._calibration_mode and self._available_brands:
            a_now2 = self._read_bool_action(self._act_a_btn, "/user/hand/right") or self._emu_a
            b_now2 = self._read_bool_action(self._act_b_btn, "/user/hand/right") or self._emu_b
            ab_held = a_now2 and b_now2
            if ab_held and not getattr(self, '_brand_sw_fired', False):
                _t = getattr(self, '_brand_sw_start', 0.0)
                if _t == 0.0:
                    self._brand_sw_start = time.perf_counter()
                elif time.perf_counter() - _t >= 0.5:
                    _idx = self._available_brands.index(self._current_brand) if self._current_brand in self._available_brands else 0
                    _next = self._available_brands[(_idx + 1) % len(self._available_brands)]
                    self._switch_brand(_next)
                    self._brand_sw_fired = True
            if not ab_held:
                self._brand_sw_start = 0.0
                self._brand_sw_fired = False

        grip_l = self._read_bool_action(self._act_left_grip,  "/user/hand/left")
        grip_r = self._read_bool_action(self._act_right_grip, "/user/hand/right")
        # Stash for _handle_keyboard_input used to suppress typing while
        # the user is repositioning the keyboard with a grip press.
        self._grip_l_now = grip_l
        self._grip_r_now = grip_r

        # Compute fresh cursor UVs + keyboard hits before grip-to-move
        # so laser-on-screen checks use current-frame data (no 1-frame lag).
        self._handle_keyboard_input()
        self._handle_cursor()
        laser_l_on_screen = self._cursor_uv_l is not None
        laser_r_on_screen = self._cursor_uv_r is not None
        laser_on_screen = laser_l_on_screen or laser_r_on_screen

        # Crop-mode gestures: left trigger, only while its laser is OFF the
        # screen (so these never compete with the normal on-screen trigger
        # click/drag handling in _handle_triggers).
        #   Hold 3s   cycle Crop: Auto -> Manual -> Off -> Auto (OSD shown)
        #   Double-tap (Manual mode only) toggle crop-adjust pause + OSD
        CROP_HOLD            = 3.0   # seconds held to cycle crop mode
        CROP_PRESS_THRESH    = 0.5
        CROP_DCLICK_WINDOW   = 0.5   # seconds between taps to count as a double-click
        ltrig_val = self._read_float_action(self._act_left_trigger, "/user/hand/left")
        ltrig_pressed = ltrig_val >= CROP_PRESS_THRESH
        ltrig_prev = getattr(self, '_ltrig_gesture_pressed_prev', False)

        if not laser_l_on_screen:
            if ltrig_pressed and not ltrig_prev:
                self._ltrig_hold_start = time.perf_counter()
                self._ltrig_hold_fired = False
            if ltrig_pressed and not self._ltrig_hold_fired:
                if time.perf_counter() - self._ltrig_hold_start >= CROP_HOLD:
                    self._ltrig_hold_fired = True
                    order = ('auto', 'manual', 'off')
                    idx = order.index(self._crop_mode) if self._crop_mode in order else 0
                    self._crop_mode = order[(idx + 1) % len(order)]
                    if self._crop_mode != 'manual' and self._crop_adjust_active:
                        self._crop_adjust_active = False
                        self._crop_adjust_osd_show_t = -999.0
                        self._refit_screen_geometry_for_crop()
                    self._crop_mode_osd_show_t = time.perf_counter()
                    self._crop_mode_osd_last_key = None
                    self._mark_runtime_settings_dirty()
            if (not ltrig_pressed) and ltrig_prev and not self._ltrig_hold_fired:
                now_t = time.perf_counter()
                if self._crop_mode == 'manual':
                    if now_t - self._ltrig_dclick_last_t <= CROP_DCLICK_WINDOW:
                        self._crop_adjust_active = not self._crop_adjust_active
                        self._crop_adjust_osd_show_t = now_t
                        self._crop_adjust_osd_last_key = None
                        self._ltrig_dclick_last_t = -999.0
                        if not self._crop_adjust_active:
                            # Drag ended: apply the deferred screen-geometry
                            # refit once now, instead of every frame during
                            # the drag (see _set_movie_crop_render_uv).
                            self._refit_screen_geometry_for_crop()
                    else:
                        self._ltrig_dclick_last_t = now_t
        else:
            self._ltrig_hold_fired = False
            self._ltrig_hold_start = 0.0
        self._ltrig_gesture_pressed_prev = ltrig_pressed

        active = (grip_l or grip_r) and laser_on_screen and not self._environment_screen_locked()
        self._grabbed  = active
        self._resizing = False
        screen_locked = self._environment_screen_locked()
        locked_old_screen_mat = self._screen_pose_mat4() if screen_locked else None

        # Grip-to-move: screen follows laser beam origin 1:1, Skip while stick is actively used (outside deadzone) stick takes priority.
        stick_active_l = abs(lx) > DEAD or abs(ly) > DEAD
        stick_active_r = abs(rx) > DEAD or abs(ry) > DEAD
        # Laser-anchored grip dragging

        both_grips = grip_l and grip_r
        GRIP_LONG = 3.0
        if both_grips and not self._both_grips_last:
            self._both_grips_hold_t = time.perf_counter()
            self._both_grips_long_fired = False
        if both_grips and not self._both_grips_long_fired and screen_locked:
            if time.perf_counter() - self._both_grips_hold_t >= GRIP_LONG:
                self._both_grips_long_fired = True
                if self._seat_adjust_active:
                    self._exit_seat_adjust_mode(save=True)
                else:
                    self._enter_seat_adjust_mode()
        self._both_grips_last = both_grips
        seat_adjust_active = self._seat_adjust_active

        # Latch per-hand grip target on rising edge
        # "Only grip one item at a time": once a grip press locks onto the
        # screen or keyboard, it stays on that target until the grip is
        # released.  Decided on rising edge by what the laser is hitting:
        # keyboard takes priority if hovered, else screen.
        for grip_now, target_attr, kb_hover in [
            (grip_l, '_grip_target_l', self._kb_hover_l),
            (grip_r, '_grip_target_r', self._kb_hover_r),
        ]:
            if not grip_now:
                setattr(self, target_attr, None)
            elif getattr(self, target_attr) is None:
                if self._keyboard_visible and kb_hover is not None:
                    setattr(self, target_attr, 'keyboard')
                elif (target_attr == '_grip_target_l' and laser_l_on_screen) \
                        or (target_attr == '_grip_target_r' and laser_r_on_screen):
                    setattr(self, target_attr, 'screen')
                # else: laser on nothing leave unlatched; next frame may catch it

        # Per-controller laser requirement: grip only moves screen when
        # its corresponding laser hits the screen.
        for grip_now, aim_mat, grab_attr, local_attr, stick_active, laser_on, grip_target in [
            (grip_l, self._aim_mat_l,
            '_screen_grab_grip_l', '_screen_grab_local_l', stick_active_l,
             laser_l_on_screen, self._grip_target_l),

            (grip_r, self._aim_mat_r,
            '_screen_grab_grip_r', '_screen_grab_local_r', stick_active_r,
             laser_r_on_screen, self._grip_target_r),
        ]:
            if grip_now and not stick_active and not both_grips and laser_on \
                    and grip_target == 'screen' and not screen_locked:

                if aim_mat is None:
                    continue

                # Use pre-smoothed controller pose (shared with laser/cursor).
                # One Euro Filter applied once per frame no double-filtering.
                is_left = (grab_attr == '_screen_grab_grip_l')
                grip_mat = self._grip_mat_l if is_left else self._grip_mat_r
                # Raw grip position for responsive drag (no smoothing lag)
                if grip_mat is not None:
                    ray_origin = (grip_mat[:3, 3] + grip_mat[:3, 1] * 0.020).astype('f8')
                else:
                    ray_origin = aim_mat[:3, 3].astype(np.float64)
                # Smoothed forward direction for jitter-free pointing
                _, ray_dir = self._get_smoothed_ray(is_left)
                if ray_dir is None:
                    ray_dir = -aim_mat[:3, 2].astype(np.float64)
                ray_dir /= np.linalg.norm(ray_dir) + 1e-10

                # Screen orientation
                cp = math.cos(self.screen_pitch)
                sp = math.sin(self.screen_pitch)

                cy = math.cos(self.screen_yaw)
                sy = math.sin(self.screen_yaw)

                screen_normal = np.array([
                    cp * sy,
                    -sp,
                    cp * cy
                ], dtype=np.float64)

                screen_center = self._screen_world_pos(cy, sy, cp, sp).astype(np.float64)

                denom = np.dot(screen_normal, ray_dir)

                if abs(denom) < 1e-6:
                    continue

                t = np.dot(screen_normal, screen_center - ray_origin) / denom

                if t < 0.0:
                    continue

                hit_world = ray_origin + ray_dir * t

                # Screen local axes
                right_axis = np.array([cy, 0.0, -sy], dtype=np.float64)
                up_axis = np.array([sp * sy, cp, sp * cy], dtype=np.float64)

                local_x = np.dot(hit_world - screen_center, right_axis)
                local_y = np.dot(hit_world - screen_center, up_axis)

                saved_local = getattr(self, local_attr)

                # Initial grab
                if saved_local is None:

                    setattr(self, local_attr, np.array([local_x, local_y], dtype=np.float64))

                else:

                    target_local = saved_local

                    # Move screen center so laser continues hitting same local point
                    desired_center = (
                        hit_world
                        - right_axis * target_local[0]
                        - up_axis * target_local[1]
                    )

                    # Project onto sphere around head preserves 3D Euclidean
                    # distance so movement in any direction doesn't change
                    # apparent screen size.
                    if self._head_pos_w is not None:
                        hx, hy, hz = self._head_pos_w

                        # 3D vector from head to desired center
                        dx = desired_center[0] - hx
                        dy = desired_center[1] - hy
                        dz = desired_center[2] - hz

                        # Current 3D Euclidean distance from head to screen
                        csx = self.screen_pan_x - hx
                        csy = self.screen_pan_y - hy
                        csz = -self.screen_distance - hz
                        R3 = math.sqrt(csx * csx + csy * csy + csz * csz)

                        if R3 > 0.01:
                            d_len = math.sqrt(dx * dx + dy * dy + dz * dz)
                            if d_len > 0.001:
                                dx /= d_len
                                dy /= d_len
                                dz /= d_len

                            self.screen_pan_x    = float(hx + dx * R3)
                            self.screen_pan_y    = float(hy + dy * R3)
                            self.screen_distance = float(-(hz + dz * R3))

                            # Baseline orientation faces head; manual offset on top
                            base_yaw   = math.atan2(-dx, -dz)
                            base_pitch = math.asin(max(-1.0, min(1.0, dy)))
                            self.screen_yaw   = base_yaw   + self._yaw_offset
                            self.screen_pitch = base_pitch + self._pitch_offset

            elif both_grips and not grip_now:
                # One grip released while both were held clear grab state
                setattr(self, local_attr, None)
            elif not grip_now:
                # Grip released clear grab state so next press starts fresh
                setattr(self, local_attr, None)
            elif stick_active:
                # Stick fine-tuning is moving the screen old anchor is stale.
                # Clear it so grip-to-move re-acquires at the current pose
                # when the stick returns to neutral (avoids snap-back).
                setattr(self, local_attr, None)
            # else: grip held but laser transiently off screen keep anchor so
            # the grab resumes as soon as the laser re-enters the screen quad.

        # Keyboard grip-to-move: panel follows laser hit on head-sphere 
        # Mirrors the screen grip-to-move loop: laser stays anchored on the
        # same local key point while the panel slides along the sphere
        # centred on the head.  Preserves Euclidean head→panel distance so
        # apparent size doesn't change, and auto-orients the panel toward
        # the head with standalone yaw/pitch offsets applied on top.
        if self._keyboard_visible:
            for grip_now, aim_mat, kb_local_attr, stick_active, grip_target in [
                (grip_l, self._aim_mat_l, '_kb_grab_local_l',
                 stick_active_l, self._grip_target_l),
                (grip_r, self._aim_mat_r, '_kb_grab_local_r',
                 stick_active_r, self._grip_target_r),
            ]:
                # Don't require a per-key hover during drag: a fast-moving
                # laser often falls in gaps between keys, which would drop
                # the grip.  Once latched onto the keyboard, just keep
                # following the ray against the keyboard plane.
                if grip_now and not stick_active and not both_grips \
                        and grip_target == 'keyboard':
                    if aim_mat is None:
                        continue
                    is_left = (kb_local_attr == '_kb_grab_local_l')
                    grip_mat = self._grip_mat_l if is_left else self._grip_mat_r
                    if grip_mat is not None:
                        ray_origin = (grip_mat[:3, 3] + grip_mat[:3, 1] * 0.020).astype('f8')
                    else:
                        ray_origin = aim_mat[:3, 3].astype(np.float64)
                    _, ray_dir = self._get_smoothed_ray(is_left)
                    if ray_dir is None:
                        ray_dir = -aim_mat[:3, 2].astype(np.float64)
                    ray_dir /= np.linalg.norm(ray_dir) + 1e-10

                    # Keyboard plane axes (match _keyboard_laser_hit)
                    cp = math.cos(self._keyboard_pitch); sp = math.sin(self._keyboard_pitch)
                    cy = math.cos(self._keyboard_yaw);   sy = math.sin(self._keyboard_yaw)
                    kb_x_ax = np.array([cy,        0.0,  -sy      ], dtype=np.float64)
                    kb_y_ax = np.array([sy * sp,   cp,    cy * sp ], dtype=np.float64)
                    kb_n    = np.array([sy * cp,  -sp,    cy * cp ], dtype=np.float64)
                    kb_pos  = np.array([self._keyboard_pan_x,
                                        self._keyboard_pan_y,
                                        -self._keyboard_distance], dtype=np.float64)
                    denom = float(np.dot(kb_n, ray_dir))
                    if abs(denom) < 1e-6:
                        continue
                    t_hit = float(np.dot(kb_n, kb_pos - ray_origin)) / denom
                    if t_hit < 0.05:
                        continue
                    hit_world = ray_origin + ray_dir * t_hit
                    diff = hit_world - kb_pos
                    local_x = float(np.dot(diff, kb_x_ax))
                    local_y = float(np.dot(diff, kb_y_ax))

                    saved_local = getattr(self, kb_local_attr)
                    if saved_local is None:
                        # First frame of grab anchor the local hit point
                        setattr(self, kb_local_attr,
                                np.array([local_x, local_y], dtype=np.float64))
                    else:
                        target_local = saved_local

                        # Desired panel centre so laser still hits same local key
                        desired_center = (
                            hit_world
                            - kb_x_ax * target_local[0]
                            - kb_y_ax * target_local[1]
                        )

                        # Sphere projection around the head preserves the
                        # Euclidean head→keyboard distance so the panel doesn't
                        # grow/shrink as the user drags it through space.
                        if self._head_pos_w is not None:
                            hx, hy, hz = self._head_pos_w
                            dx = desired_center[0] - hx
                            dy = desired_center[1] - hy
                            dz = desired_center[2] - hz
                            ksx = self._keyboard_pan_x - hx
                            ksy = self._keyboard_pan_y - hy
                            ksz = -self._keyboard_distance - hz
                            R3 = math.sqrt(ksx * ksx + ksy * ksy + ksz * ksz)
                            if R3 > 0.01:
                                d_len = math.sqrt(dx * dx + dy * dy + dz * dz)
                                if d_len > 0.001:
                                    dx /= d_len; dy /= d_len; dz /= d_len
                                self._keyboard_pan_x    = float(hx + dx * R3)
                                self._keyboard_pan_y    = float(hy + dy * R3)
                                self._keyboard_distance = float(max(0.2, -(hz + dz * R3)))

                                # Auto-orient toward head; standalone offsets on top
                                base_yaw   = math.atan2(-dx, -dz)
                                base_pitch = math.asin(max(-1.0, min(1.0, dy)))
                                self._keyboard_yaw   = base_yaw   + self._kb_yaw_offset
                                self._keyboard_pitch = base_pitch + self._kb_pitch_offset
                        else:
                            # No head pose yet fall back to direct translation
                            self._keyboard_pan_x    = float(desired_center[0])
                            self._keyboard_pan_y    = float(desired_center[1])
                            self._keyboard_distance = float(max(0.2, -desired_center[2]))
                elif both_grips and not grip_now:
                    setattr(self, kb_local_attr, None)
                elif not grip_now:
                    setattr(self, kb_local_attr, None)
                elif stick_active:
                    # Stick fine-tuning the keyboard clear stale anchor so
                    # grip-to-move re-acquires at current pose on neutral.
                    setattr(self, kb_local_attr, None)
                # else: grip held but laser transiently off keyboard keep
                # anchor so the grab resumes when laser re-enters the panel.
                # Cache the keyboard position after any grip-move update.
                if grip_now and grip_target == 'keyboard':
                    self._kb_cached_position = {
                        'pan_x': self._keyboard_pan_x, 'pan_y': self._keyboard_pan_y,
                        'distance': self._keyboard_distance, 'width': self._keyboard_width,
                        'yaw': self._keyboard_yaw, 'pitch': self._keyboard_pitch,
                    }
        else:
            # Keyboard hidden clear any stale anchors
            self._kb_grab_local_l = None
            self._kb_grab_local_r = None

        # Both grips held: system move
        # Average the two laser hit positions, project onto sphere around
        # head, and move the screen as a single rigid system.  This allows
        # two-handed repositioning without individual grip fighting.
        if both_grips and not seat_adjust_active and not stick_active_l and not stick_active_r \
                and laser_l_on_screen and laser_r_on_screen and not screen_locked:
            mats = []
            for aim_mat, grab_attr, local_attr in [
                (self._aim_mat_l, '_screen_grab_grip_l', '_screen_grab_local_l'),
                (self._aim_mat_r, '_screen_grab_grip_r', '_screen_grab_local_r'),
            ]:
                if aim_mat is None:
                    continue
                is_left = (grab_attr == '_screen_grab_grip_l')
                grip_mat = self._grip_mat_l if is_left else self._grip_mat_r
                if grip_mat is not None:
                    ray_origin = (grip_mat[:3, 3] + grip_mat[:3, 1] * 0.020).astype('f8')
                else:
                    ray_origin = aim_mat[:3, 3].astype(np.float64)
                _, ray_dir = self._get_smoothed_ray(is_left)
                if ray_dir is None:
                    ray_dir = -aim_mat[:3, 2].astype(np.float64)
                ray_dir /= np.linalg.norm(ray_dir) + 1e-10
                cp = math.cos(self.screen_pitch)
                sp = math.sin(self.screen_pitch)
                cy = math.cos(self.screen_yaw)
                sy = math.sin(self.screen_yaw)
                screen_normal = np.array([cp * sy, -sp, cp * cy], dtype=np.float64)
                screen_center = self._screen_world_pos(cy, sy, cp, sp).astype(np.float64)
                denom = np.dot(screen_normal, ray_dir)
                if abs(denom) < 1e-6:
                    continue
                t_hit = np.dot(screen_normal, screen_center - ray_origin) / denom
                if t_hit < 0.0:
                    continue
                hit_world = ray_origin + ray_dir * t_hit
                right_axis = np.array([cy, 0.0, -sy], dtype=np.float64)
                up_axis = np.array([sp * sy, cp, sp * cy], dtype=np.float64)
                local_x = np.dot(hit_world - screen_center, right_axis)
                local_y = np.dot(hit_world - screen_center, up_axis)
                saved_local = getattr(self, local_attr)
                if saved_local is None:
                    setattr(self, local_attr, np.array([local_x, local_y], dtype=np.float64))
                    saved_local = np.array([local_x, local_y], dtype=np.float64)
                target_local = saved_local
                dc = hit_world - right_axis * target_local[0] - up_axis * target_local[1]
                mats.append(dc)

            if mats and self._head_pos_w is not None:
                # Average desired centers from both controllers
                avg_center = sum(mats) / len(mats)
                hx, hy, hz = self._head_pos_w
                dx = avg_center[0] - hx
                dy = avg_center[1] - hy
                dz = avg_center[2] - hz
                csx = self.screen_pan_x - hx
                csy = self.screen_pan_y - hy
                csz = -self.screen_distance - hz
                R3 = math.sqrt(csx * csx + csy * csy + csz * csz)
                if R3 > 0.01:
                    d_len = math.sqrt(dx * dx + dy * dy + dz * dz)
                    if d_len > 0.001:
                        dx /= d_len; dy /= d_len; dz /= d_len
                    self.screen_pan_x    = float(hx + dx * R3)
                    self.screen_pan_y    = float(hy + dy * R3)
                    self.screen_distance = float(-(hz + dz * R3))
                    base_yaw   = math.atan2(-dx, -dz)
                    base_pitch = math.asin(max(-1.0, min(1.0, dy)))
                    self.screen_yaw   = base_yaw   + self._yaw_offset
                    self.screen_pitch = base_pitch + self._pitch_offset

        # Seat adjust: thumbsticks move viewer position relative to screen
        # With a single grip held: thumbsticks shift the env model position
        # (room moves around the user/screen).
        if seat_adjust_active:
            single_grip = (grip_l or grip_r) and not both_grips
            self._seat_adjust_grip_move = single_grip
            if single_grip:
                ENV_MOVE_SPEED = 1.0
                mp = self._env_model_pos
                env_changed = False
                if abs(lx) > DEAD:
                    mp[0] += lx * ENV_MOVE_SPEED * dt
                    env_changed = True
                if abs(ly) > DEAD:
                    mp[2] += ly * ENV_MOVE_SPEED * dt
                    env_changed = True
                if abs(rx) > DEAD:
                    mp[0] += rx * ENV_MOVE_SPEED * dt
                    env_changed = True
                if abs(ry) > DEAD:
                    mp[1] += ry * ENV_MOVE_SPEED * dt
                    env_changed = True
                if env_changed:
                    self._env_model_pos = [round(v, 4) for v in mp]
                    self._cached_env_model_mat4_frame = -1
                    self._seat_adjust_osd_dirty = True
            else:
                SEAT_MOVE_SPEED = 0.3
                SEAT_ANGLE_SPEED = 30.0
                view = getattr(self, '_view_pose_profile', {}) or {}
                sa_x = float(view.get('x', 0.0))
                sa_y = float(view.get('y', 0.6))
                sa_z = float(view.get('z', 0.0))
                sa_angle = float(view.get('angle', 0.0))
                changed = False
                if abs(lx) > DEAD:
                    sa_x += lx * SEAT_MOVE_SPEED * dt
                    changed = True
                if abs(ly) > DEAD:
                    sa_y += -ly * SEAT_MOVE_SPEED * dt
                    sa_y = max(0.1, sa_y)
                    changed = True
                if abs(rx) > DEAD:
                    sa_angle += -rx * SEAT_ANGLE_SPEED * dt
                    sa_angle = max(-90.0, min(90.0, sa_angle))
                    sa_angle = round(sa_angle)
                    changed = True
                if abs(ry) > DEAD:
                    sa_z += ry * SEAT_MOVE_SPEED * dt
                    changed = True
                if changed:
                    view['x'] = round(sa_x, 4)
                    view['y'] = round(sa_y, 4)
                    view['z'] = round(sa_z, 4)
                    view['angle'] = round(sa_angle, 1)
                    self._view_pose_profile = view
                    self._apply_seat_adjust_xr_space(sa_x, sa_y, sa_z, sa_angle)
                    self._seat_adjust_osd_dirty = True

        # Manual crop-adjust: left stick shrinks/grows the crop rectangle
        # symmetrically about center while paused in crop-adjust mode.
        # Guarded off whenever a grip is held so it never collides with the
        # grip+stick pan/rotate/depth/resize bindings above.
        CROP_ADJUST_SPEED = 0.6   # unit width/height per second at full deflection
        if self._crop_adjust_active and not grip_l and not grip_r:
            cw = self._manual_crop_uv[2]
            ch = self._manual_crop_uv[3]
            changed = False
            # Single dominant axis only: X crops sides, Y crops top/bottom.
            # Diagonal deflection must not crop both, so ignore the minor axis.
            if abs(lx) > DEAD and abs(lx) >= abs(ly):
                cw = max(0.0, min(1.0, cw + lx * CROP_ADJUST_SPEED * dt))
                changed = True
            elif abs(ly) > DEAD:
                ch = max(0.0, min(1.0, ch + ly * CROP_ADJUST_SPEED * dt))
                changed = True
            if changed:
                self._set_manual_crop_uv(cw, ch)
                self._crop_adjust_osd_show_t = time.perf_counter()
                self._crop_adjust_osd_last_key = None
                self._mark_runtime_settings_dirty()

        # Grip + stick fine-tuning (pan, resize, rotate, depth)
        KB_MOVE_SPEED = 0.4    # m/s at full deflection
        # Depth-ratio stick control: replaces A/B + right-grip mapping
        DEPTH_RATIO_SPEED = 0.5   # units/s at full deflection
        DEPTH_RATIO_MIN   = 0.0
        DEPTH_RATIO_MAX   = 10.0
        if grip_l and not seat_adjust_active:
            if self._keyboard_visible and self._grip_target_l == 'keyboard':
                # Left grip latched to keyboard + left stick standalone
                # translation: orbit the keyboard around the head on a sphere
                # of preserved Euclidean radius, then re-aim at the head with
                # the user yaw/pitch offsets layered on top (mirrors the
                # yaw/pitch standalone pattern).
                if (abs(lx) > DEAD or abs(ly) > DEAD) and self._head_pos_w is not None:
                    hx, hy, hz = self._head_pos_w
                    ckx = self._keyboard_pan_x - hx
                    cky = self._keyboard_pan_y - hy
                    ckz = -self._keyboard_distance - hz
                    R3 = math.sqrt(ckx * ckx + cky * cky + ckz * ckz)
                    if R3 > 0.01:
                        az = math.atan2(ckx, -ckz)
                        el = math.asin(max(-1.0, min(1.0, cky / R3)))
                        # Convert linear speed at object distance to angular rate.
                        ang = KB_MOVE_SPEED * dt / max(R3, 0.3)
                        if abs(lx) > DEAD:
                            az += lx * ang
                        if abs(ly) > DEAD:
                            el = max(-1.4, min(1.4, el + ly * ang))
                        cel = math.cos(el)
                        ux = cel * math.sin(az)
                        uy = math.sin(el)
                        uz = -cel * math.cos(az)
                        self._keyboard_pan_x   = float(hx + ux * R3)
                        self._keyboard_pan_y   = float(hy + uy * R3)
                        self._keyboard_distance = float(-(hz + uz * R3))
                        base_yaw   = math.atan2(-ux, -uz)
                        base_pitch = math.asin(max(-1.0, min(1.0, uy)))
                        self._keyboard_yaw   = base_yaw   + self._kb_yaw_offset
                        self._keyboard_pitch = base_pitch + self._kb_pitch_offset
            else:
                # Left grip + left stick standalone screen translation:
                # orbit the screen around the head on a sphere of preserved
                # Euclidean radius, then re-aim at the head with the user
                # yaw/pitch offsets layered on top.
                if laser_on_screen and (abs(lx) > DEAD or abs(ly) > DEAD) \
                        and self._head_pos_w is not None and not screen_locked:
                    hx, hy, hz = self._head_pos_w
                    csx = self.screen_pan_x - hx
                    csy = self.screen_pan_y - hy
                    csz = -self.screen_distance - hz
                    R3 = math.sqrt(csx * csx + csy * csy + csz * csz)
                    if R3 > 0.01:
                        az = math.atan2(csx, -csz)
                        el = math.asin(max(-1.0, min(1.0, csy / R3)))
                        ang = self._pan_speed * dt / max(R3, 0.3)
                        if abs(lx) > DEAD:
                            az += lx * ang
                        if abs(ly) > DEAD:
                            el = max(-1.4, min(1.4, el + ly * ang))
                        cel = math.cos(el)
                        ux = cel * math.sin(az)
                        uy = math.sin(el)
                        uz = -cel * math.cos(az)
                        self.screen_pan_x    = float(hx + ux * R3)
                        self.screen_pan_y    = float(hy + uy * R3)
                        self.screen_distance = float(-(hz + uz * R3))
                        base_yaw   = math.atan2(-ux, -uz)
                        base_pitch = math.asin(max(-1.0, min(1.0, uy)))
                        self.screen_yaw   = base_yaw   + self._yaw_offset
                        self.screen_pitch = base_pitch + self._pitch_offset
        elif grip_r and not seat_adjust_active:
            # X and Y share the same stick, so require the intended axis to
            # dominate: depth (Y) only fires when the push is mostly vertical,
            # veil (X) only when mostly horizontal. Prevents an X push meant for
            # veil from leaking into depth (and vice versa).
            y_dominant = abs(ly) > abs(lx)
            x_dominant = abs(lx) > abs(ly)
            # Right grip + left stick Y depth_ratio (no laser required)
            if abs(ly) > DEAD and y_dominant:
                old_val = self.depth_ratio
                self.depth_ratio = max(DEPTH_RATIO_MIN,
                                    min(DEPTH_RATIO_MAX,
                                        self.depth_ratio + ly * DEPTH_RATIO_SPEED * dt))
                if self.depth_ratio != old_val:
                    self._depth_osd_show_t = time.perf_counter()
                    self._mark_runtime_settings_dirty()
            # Right grip + left stick X veil transparency (only in veil mode)
            if abs(lx) > DEAD and x_dominant and self._active_glow_mode() == 'veil':
                VEIL_ALPHA_SPEED = 0.8   # units/s at full deflection
                old_a = float(getattr(self, '_frost_veil_alpha', 1.0))
                new_a = max(0.0, min(1.0, old_a + lx * VEIL_ALPHA_SPEED * dt))
                if new_a != old_a:
                    self._frost_veil_alpha = new_a
                    self._frost_uniform_cache = {}
                    self._mark_runtime_settings_dirty()
            # Don't send desktop scroll events while screen grabbed/manipulated
            if not (self._grabbed or grip_l or grip_r):
                self._accum_scroll(lx, 0.0, dt)
        else:
            if not (self._grabbed or grip_l or grip_r) and not self._crop_adjust_active:
                self._accum_scroll(lx, ly, dt)

        # Right grip + right stick X: resize screen width
        # Right grip + right stick Y: screen distance (acceleration curve) 
        # Left grip + right stick X: rotate screen yaw around its centre 
        # Left grip + right stick Y: depth strength (unchanged) 
        # Right stick (no grip): mouse scroll
        RESIZE_SPEED = 1.2    # m/s of width change at full deflection
        if grip_r and not grip_l:
            if self._keyboard_visible and self._grip_target_r == 'keyboard':
                if abs(rx) > abs(ry) and abs(rx) > DEAD:
                    self._keyboard_width = max(0.3,
                        self._keyboard_width + rx * RESIZE_SPEED * dt)
                    self._keyboard_height = None  # recalc from aspect
                elif abs(ry) > abs(rx) and abs(ry) > DEAD:
                    # Right-grip + right-stick Y radial distance along head→keyboard ray
                    _t = (abs(ry) - DEAD) / (1.0 - DEAD)
                    _speed = (self._dist_speed_base
                            + (self._dist_speed_max - self._dist_speed_base) * (_t ** self._dist_speed_exp))
                    if self._head_pos_w is not None:
                        hx, hy, hz = self._head_pos_w
                        ckx = self._keyboard_pan_x - hx
                        cky = self._keyboard_pan_y - hy
                        ckz = -self._keyboard_distance - hz
                        R3 = math.sqrt(ckx * ckx + cky * cky + ckz * ckz)
                        if R3 > 0.01:
                            ux = ckx / R3
                            uy = cky / R3
                            uz = ckz / R3
                            delta = _speed * (1.0 if ry > 0 else -1.0) * dt
                            R3_new = max(0.2, R3 + delta)
                            self._keyboard_pan_x   = float(hx + ux * R3_new)
                            self._keyboard_pan_y   = float(hy + uy * R3_new)
                            self._keyboard_distance = float(-(hz + uz * R3_new))
                            # Re-aim keyboard at the head (auto-orientation),
                            # layering user-configured offsets on top.
                            base_yaw   = math.atan2(-ux, -uz)
                            base_pitch = math.asin(max(-1.0, min(1.0, uy)))
                            self._keyboard_yaw   = base_yaw   + self._kb_yaw_offset
                            self._keyboard_pitch = base_pitch + self._kb_pitch_offset
                    else:
                        self._keyboard_distance = max(0.2,
                            self._keyboard_distance + ry * self._dist_speed_base * dt)
            else:
                if abs(rx) > abs(ry) and abs(rx) > DEAD and not self._environment_screen_locked():
                    self._screen_ref_size = max(0.8,
                                            self._screen_ref_size + rx * RESIZE_SPEED * dt)
                    self.screen_height = None
                    self._resizing = True
                    self._screen_osd_show_t = time.perf_counter()
                elif abs(ry) > abs(rx) and abs(ry) > DEAD and not screen_locked:
                    # Right-grip + right-stick Y -> radial distance along head->screen ray
                    _t = (abs(ry) - DEAD) / (1.0 - DEAD)
                    _speed = (self._dist_speed_base
                            + (self._dist_speed_max - self._dist_speed_base) * (_t ** self._dist_speed_exp))
                    # Move screen along the head→screen radial direction
                    if self._head_pos_w is not None:
                        hx, hy, hz = self._head_pos_w
                        csx = self.screen_pan_x - hx
                        csy = self.screen_pan_y - hy
                        csz = -self.screen_distance - hz
                        R3 = math.sqrt(csx * csx + csy * csy + csz * csz)
                        if R3 > 0.01:
                            ux = csx / R3
                            uy = csy / R3
                            uz = csz / R3
                            delta = _speed * (1.0 if ry > 0 else -1.0) * dt
                            R3_new = max(0.3, R3 + delta)
                            self.screen_pan_x    = float(hx + ux * R3_new)
                            self.screen_pan_y    = float(hy + uy * R3_new)
                            self.screen_distance = float(-(hz + uz * R3_new))
                    self._screen_osd_show_t = time.perf_counter()
        elif grip_l and not grip_r:
            if self._keyboard_visible and self._grip_target_l == 'keyboard':
                # Left grip latched onto keyboard + right stick standalone
                # keyboard yaw / pitch offsets (auto-orientation still aims at
                # head, these offsets are layered on top).
                if abs(rx) > DEAD:
                    self._kb_yaw_offset   -= rx * self._rot_speed * dt
                    self._keyboard_yaw    -= rx * self._rot_speed * dt
                if abs(ry) > DEAD:
                    self._kb_pitch_offset += ry * self._rot_speed * dt
                    self._keyboard_pitch  += ry * self._rot_speed * dt
            else:
                # Left grip + right stick X screen yaw rotation
                if laser_on_screen and abs(rx) > DEAD and not screen_locked:
                    self._yaw_offset -= rx * self._rot_speed * dt
                    self.screen_yaw  -= rx * self._rot_speed * dt
                # Left grip + right stick Y screen pitch rotation
                if laser_on_screen and abs(ry) > DEAD and not screen_locked:
                    self._pitch_offset += ry * self._rot_speed * dt
                    self.screen_pitch  += ry * self._rot_speed * dt
        self._grip_r_prev = grip_r
        # Re-sync grab offsets after stick fine-tuning.  Grip-to-move is paused
        # while the stick is active, so full XYZ re-sync is safe here.
        for grip_now, grip_mat, grab_attr in [
            (grip_l, self._grip_mat_l, '_screen_grab_grip_l'),
            (grip_r, self._grip_mat_r, '_screen_grab_grip_r'),
        ]:
            if grip_now and grip_mat is not None:
                saved = getattr(self, grab_attr)
                if saved is not None:
                    grip_pos = grip_mat[:3, 3].astype('f8')
                    screen_center = np.array([
                        self.screen_pan_x, self.screen_pan_y,
                        -self.screen_distance], dtype='f8')
                    setattr(self, grab_attr, screen_center - grip_pos)
        if not grip_r:
            # Accelerated scroll: higher stick deflection -> disproportionately faster scroll
            if not (self._grabbed or grip_l):
                self._accum_scroll(rx, ry, dt)

        # Rebuild keyboard geometry if width changed
        if (self._keyboard_visible and self._keyboard_tex is not None
                and abs(self._keyboard_width - self._kb_last_build_width) > 0.001):
            self._keyboard_height = (self._keyboard_width
                                    * _KB_TEX_H / float(_KB_TEX_W))
            self._build_keyboard_texture()
            self._kb_last_build_width = self._keyboard_width

        if not seat_adjust_active:
            self._move_env_with_screen_delta(locked_old_screen_mat)

        # Menu (left): short press toggle status/FPS panel
        menu_now = self._read_bool_action(self._act_menu_btn, "/user/hand/left")
        MENU_LONG = 0.6  # seconds long press reserved for calibration combo
        if menu_now and not self._menu_pressed_last:
            self._menu_press_t = time.perf_counter()
            self._menu_long_fired = False
        if not menu_now and self._menu_pressed_last:
            if not self._menu_long_fired and (time.perf_counter() - self._menu_press_t) < MENU_LONG:
                self._fps_overlay_visible = not self._fps_overlay_visible
        self._menu_pressed_last = menu_now

        # A / B (right):
        #   Previously: right-grip + A/B adjusted depth_ratio. Now: right-grip +
        #   right-stick Y adjusts depth_ratio. When A+B are held together (brand
        #   switch combo) A/B's other functions are suppressed.
        a_now = self._read_bool_action(self._act_a_btn, "/user/hand/right") or self._emu_a
        b_now = self._read_bool_action(self._act_b_btn, "/user/hand/right") or self._emu_b

        if not ab_held:
            # When A+B are not held together, A/B keep their normal behaviour.
            # If right-grip is held we intentionally do not perform A/B immediate
            # actions here (depth_ratio is adjusted via right-stick Y below).
            if not grip_r:
                # Use XR runtime's `changed` flag when available more reliable than
                # manual frame-to-frame tracking when a button sits under a resting thumb.
                # Fall back to manual edge detection if pyopenxr doesn't expose it.
                a_edge = self._read_bool_edge(self._act_a_btn, "/user/hand/right", self._a_last)
                b_edge = self._read_bool_edge(self._act_b_btn, "/user/hand/right", self._b_last)
                if not a_edge:
                    a_edge = a_now and not self._a_last
                if not b_edge:
                    b_edge = b_now and not self._b_last
                # Only send OS mouse clicks if the right controller laser is
                # currently intersecting the virtual screen. This prevents A/B
                # from clicking when pointing off-screen or at the overlay panel.
                is_gripping = self._grabbed
                if a_edge and laser_r_on_screen and not is_gripping:
                    self._emit_left_click()
                if b_edge and laser_r_on_screen and not is_gripping:
                    _send_mouse_flags(_MOUSEEVENTF_RIGHTDOWN)
                    _send_mouse_flags(_MOUSEEVENTF_RIGHTUP)

        self._a_last = a_now
        self._b_last = b_now

        # Y (left):
        #   short press  reset screen to upright default (same as session start)
        #   long press   reset screen to face current head gaze (same as home long-press)
        Y_LONG = 0.6   # seconds to trigger long-press action
        y_now = self._read_bool_action(self._act_y_btn, "/user/hand/left") or self._emu_y
        if y_now and not self._y_last:
            self._y_press_t    = time.perf_counter()
            self._y_long_fired = False
        if y_now and not self._y_long_fired and not seat_adjust_active:
            if time.perf_counter() - self._y_press_t >= Y_LONG:
                if screen_locked:
                    if not (self._env_uses_view_pose_cycle() and self._cycle_view_pose()):
                        self._cycle_lighting_preset()
                else:
                    nxt = (self._preset_index + 1) % len(self._screen_presets)
                    self._apply_preset(nxt)
                self._y_long_fired = True
        if not y_now and self._y_last and not self._y_long_fired and not seat_adjust_active:
            if screen_locked:
                self._reset_seating_vertical()
            else:
                self._reset_screen_to_default(show_border=True)
        self._y_last = y_now

        # X (left):
        #   release <1s              toggle virtual keyboard
        #   release 1~4s             cycle glow mode (glow -> veil -> frosted -> off)
        #   hold >4s (release)       toggle VDXR green passthrough backdrop
        X_GLOW_HOLD = 1.0
        X_PASSTHROUGH_HOLD = 4.0
        x_now = self._read_bool_action(self._act_x_btn, "/user/hand/left") or self._emu_x

        if x_now and not self._x_last:                     # rising edge
            self._x_press_t = time.perf_counter()
            self._x_long_fired = False

        if x_now and not self._x_long_fired:               # still held, not yet fired
            held = time.perf_counter() - self._x_press_t
            if held >= X_PASSTHROUGH_HOLD:
                self._toggle_passthrough_backdrop()
                self._x_long_fired = True
                # Prevent the glow action from also firing on release
                self._x_glow_fired = True

        if not x_now and self._x_last:                     # falling edge
            held = time.perf_counter() - self._x_press_t
            if not self._x_long_fired:                     # passthrough was already triggered
                if not getattr(self, '_x_glow_fired', False):
                    # 1s <= release < 4s: toggle light
                    if held >= X_GLOW_HOLD:
                        self._cycle_light_from_x()
                    else:
                        # <1s: toggle keyboard
                        self._keyboard_visible = not self._keyboard_visible
                        if self._keyboard_visible:
                            if self._keyboard_tex is None:
                                self._init_keyboard()
                            cached = getattr(self, '_kb_cached_position', None)
                            if cached is not None:
                                self._kb_restore_cached_position(cached)
                            else:
                                self._kb_cached_position = self._anchor_keyboard_below_screen()
        elif not x_now:
            self._x_glow_fired = False  # reset when button wasn't pressed

        self._x_last = x_now

        # Thumbstick buttons: distinguish between single and double thumbstick presses
        lsc_now = self._read_bool_action(self._act_left_stick_click, "/user/hand/left")
        rsc_now = self._read_bool_action(self._act_right_stick_click, "/user/hand/right")
        # Trackpad emulation: suppress stick-click when a button region (top/bottom)
        # was hit, so clicking the top fires B/Y but not stick-click.
        # Center-click emulation is OR'd in so it still acts as stick-click.
        if self._emu_x or self._emu_y:
            lsc_now = False
        else:
            lsc_now = lsc_now or self._emu_lsc
        if self._emu_a or self._emu_b:
            rsc_now = False
        else:
            rsc_now = rsc_now or self._emu_rsc
        BOTH_LONG  = 0.5   # Duration to trigger FPS/help panel toggle with both sticks
        SINGLE_LONG = 1.0  # Duration for single thumbstick long press to switch shortcut panels

        # Both thumbsticks pressed toggle FPS/help panel after 0.5 seconds (unchanged)
        both_clicked = lsc_now and rsc_now
        if both_clicked and not self._both_stick_fired:
            if self._both_stick_start == 0.0:
                self._both_stick_start = time.perf_counter()
            elif time.perf_counter() - self._both_stick_start >= BOTH_LONG:
                self._fps_overlay_visible = not self._fps_overlay_visible
                self._both_stick_fired = True
                # Mark single-stick long-fired to suppress their short actions
                self._lsc_long_fired = True
                self._rsc_long_fired = True
        if not both_clicked:
            self._both_stick_start = 0.0
            self._both_stick_fired = False

        # Non-double-press case: handle each thumbstick separately
        # - Long press (>= SINGLE_LONG) triggers shortcut panel (same as double press);
        # - Short press (released when < SINGLE_LONG) performs background/curve mode switching or grip modifier actions;
        if not both_clicked:
            now = self._frame_now

            # Left thumbstick: long-press toggle status/shortcut panel;
            #                  short-press cycle environment
            if lsc_now and not self._left_stick_click_prev:
                self._lsc_press_t = now
                self._lsc_long_fired = False
            if lsc_now and not self._lsc_long_fired:
                if now - getattr(self, '_lsc_press_t', 0.0) >= SINGLE_LONG:
                    self._fps_overlay_visible = not self._fps_overlay_visible
                    self._lsc_long_fired = True
            if not lsc_now and self._left_stick_click_prev:
                # Released if long-press wasn't fired, treat as short press
                if not self._lsc_long_fired:
                    if grip_r:
                        if self.depth_strength > 0.0:
                            self._depth_strength_saved = self.depth_strength
                            self.depth_strength = 0.0
                        else:
                            self.depth_strength = getattr(self, '_depth_strength_saved', 0.1)
                    else:
                        self._cycle_environment()

            # Right thumbstick: long-press reset screen direction (keep distance + size);
            #          short-press cycle horizontal curve -> vertical curve -> flat
            if rsc_now and not self._right_stick_click_prev:
                self._rsc_press_t = now
                self._rsc_long_fired = False
            if rsc_now and not self._rsc_long_fired:
                if now - getattr(self, '_rsc_press_t', 0.0) >= SINGLE_LONG:
                    self._reset_screen_direction()
                    self._rsc_long_fired = True
            if not rsc_now and self._right_stick_click_prev:
                if not self._rsc_long_fired:
                    if not grip_r and not grip_l and (
                            not self._environment_screen_locked() or self._env_allow_curve):
                        self._cycle_screen_curve_mode()
                    elif grip_r:
                        old_val = self.depth_ratio
                        self.depth_ratio = 2.0
                        if self.depth_ratio != old_val:
                            self._mark_runtime_settings_dirty()

        self._left_stick_click_prev = lsc_now
        self._right_stick_click_prev = rsc_now

        # Border fade: snap to 1 while the user is actively re-positioning, fade out
        # when idle. `active` is computed at the top of this method.
        FADE_DELAY = 1.5   # seconds before starting to fade
        FADE_DUR   = 0.8   # fade-out duration in seconds
        if active:
            self._border_alpha  = 1.0
            self._border_idle_t = time.perf_counter()
        else:
            idle = time.perf_counter() - self._border_idle_t
            if idle > FADE_DELAY:
                self._border_alpha = max(0.0, 1.0 - (idle - FADE_DELAY) / FADE_DUR)

        # Keyboard border fade: show while gripping AND laser on keyboard (not just off-screen)
        kb_hit = self._kb_hover_l is not None or self._kb_hover_r is not None
        kb_active = self._keyboard_visible and (grip_l or grip_r) and kb_hit
        if kb_active:
            self._kb_border_alpha  = 1.0
            self._kb_border_idle_t = time.perf_counter()
        else:
            idle = time.perf_counter() - self._kb_border_idle_t
            if idle > FADE_DELAY:
                self._kb_border_alpha = max(0.0, 1.0 - (idle - FADE_DELAY) / FADE_DUR)

        # Trigger input fires mouse clicks (skips keys claimed by keyboard)
        self._handle_triggers()

        # Debounced screen state persistence (default / non-env mode only)
        if not screen_locked:
            cur = (self.screen_width, self.screen_distance, self.screen_pan_x,
                   self.screen_pan_y, self.screen_yaw, self.screen_pitch,
                   self._screen_curve_mode(), self._preset_index)
            prev = getattr(self, '_prev_screen_snapshot', None)
            if prev != cur:
                self._screen_state_dirty = True
                self._screen_state_save_t = time.perf_counter()
                self._prev_screen_snapshot = cur
            if self._screen_state_dirty and time.perf_counter() - self._screen_state_save_t > 1.0:
                self._screen_state_dirty = False
                self._persist_screen_state()
        self._flush_runtime_settings_if_idle()

    # Main blocking loop

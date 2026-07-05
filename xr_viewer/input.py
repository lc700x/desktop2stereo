"""Controller, touch, keyboard, and pointer input helpers."""

import ctypes
import math
import sys

import numpy as np

#  Windows multi-touch injection (Pointer Input API) 
# Each VR controller becomes a Windows touch contact, so trigger clicks, drags
# (incl. window title-bar drag), and two-controller multi-touch gestures
# (pinch/zoom, two-finger pan, press-and-hold right-click) all route through
# native Windows touch matching the gestures documented at
# https://support.microsoft.com/en-us/windows/touch-gestures-for-windows-a9d28305-4818-a5df-4e2b-e5590f850741
#
# Inlined (previously windows_touch.py) so the viewer is self-contained.
# Falls back gracefully on non-Windows or when InitializeTouchInjection is
# unavailable: ``_TOUCH_AVAILABLE`` stays False and the call sites use the
# legacy mouse-event path.

# Tuning knobs feedback / contact area / pressure.
_TOUCH_MAX_CONTACTS       = 10        # InitializeTouchInjection capacity (≥2 hands).
_TOUCH_DEFAULT_PRESSURE   = 32000     # Matches Microsoft's canonical sample.
_TOUCH_CONTACT_RADIUS_PX  = 4         # rcContact half-size in pixels.
_TOUCH_FEEDBACK_DEFAULT   = 0x1       # System-drawn touch circles.
_TOUCH_FEEDBACK_INDIRECT  = 0x2       # No circles (cleaner inside VR).
_TOUCH_FEEDBACK_NONE      = 0x3       # No feedback at all.

# Two-controller pinch/zoom gain. Windows sees the two touch contacts farther
# apart from their midpoint than the raw lasers, making zoom gestures respond
# with less physical controller travel. Single-touch click/drag is untouched.
_TOUCH_PINCH_SPREAD_GAIN  = 1.65

# POINTER_FLAGS we use.
_POINTER_FLAG_NONE        = 0x00000000
_POINTER_FLAG_INRANGE     = 0x00000002
_POINTER_FLAG_INCONTACT   = 0x00000004
_POINTER_FLAG_PRIMARY     = 0x00002000  # Reserved OS assigns automatically.
_POINTER_FLAG_DOWN        = 0x00010000
_POINTER_FLAG_UPDATE      = 0x00020000
_POINTER_FLAG_UP          = 0x00040000

_PT_TOUCH                 = 0x2

_TOUCH_MASK_CONTACTAREA   = 0x1
_TOUCH_MASK_ORIENTATION   = 0x2
_TOUCH_MASK_PRESSURE      = 0x4


class _TI_POINT(ctypes.Structure):
    _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]


class _TI_RECT(ctypes.Structure):
    _fields_ = [("left",  ctypes.c_long), ("top",    ctypes.c_long),
                ("right", ctypes.c_long), ("bottom", ctypes.c_long)]


class _POINTER_INFO(ctypes.Structure):
    _fields_ = [
        ("pointerType",          ctypes.c_uint32),
        ("pointerId",            ctypes.c_uint32),
        ("frameId",              ctypes.c_uint32),
        ("pointerFlags",         ctypes.c_int32),
        ("sourceDevice",         ctypes.c_void_p),
        ("hwndTarget",           ctypes.c_void_p),
        ("ptPixelLocation",      _TI_POINT),
        ("ptHimetricLocation",   _TI_POINT),
        ("ptPixelLocationRaw",   _TI_POINT),
        ("ptHimetricLocationRaw", _TI_POINT),
        ("dwTime",               ctypes.c_ulong),
        ("historyCount",         ctypes.c_uint32),
        ("InputData",            ctypes.c_int32),
        ("dwKeyStates",          ctypes.c_ulong),
        ("PerformanceCount",     ctypes.c_uint64),
        ("ButtonChangeType",     ctypes.c_int32),
    ]


class _POINTER_TOUCH_INFO(ctypes.Structure):
    _fields_ = [
        ("pointerInfo",  _POINTER_INFO),
        ("touchFlags",   ctypes.c_int32),
        ("touchMask",    ctypes.c_int32),
        ("rcContact",    _TI_RECT),
        ("rcContactRaw", _TI_RECT),
        ("orientation",  ctypes.c_uint32),
        ("pressure",     ctypes.c_uint32),
    ]


class _TouchContact:
    """Per-contact bookkeeping (mutable, persists across frames).

    Two coordinate pairs are tracked:

    * ``x, y``         last position the OS has seen for this contact
      (the most recently *successful* DOWN/UPDATE).
    * ``x_set, y_set`` the position the caller wants for the next flush.

    The split matters because ``InjectTouchInput`` rejects an UP whose
    coordinates differ from the prior DOWN/UPDATE with
    ``ERROR_INVALID_PARAMETER`` (87) the OS will not interpret a
    simultaneous "move + lift" as a single transition. ``_TouchInjector``
    emits an UPDATE at ``x_set, y_set`` *before* the UP whenever the caller
    repositions a contact in the same frame they release it.
    """

    __slots__ = ("id", "active", "x", "y", "x_set", "y_set", "_wanted")

    def __init__(self, contact_id):
        self.id = contact_id
        self.active = False
        self.x = 0          # OS-known position.
        self.y = 0
        self.x_set = 0      # Caller-requested position for the next flush.
        self.y_set = 0
        self._wanted = False


class _TouchInjector:
    """Per-contact multi-touch injector built on InjectTouchInput.

    Usage each frame::

        injector.set(contact_id, x, y, want_down=True_or_False)  # one per hand
        injector.flush()                                          # one Win32 call
    """

    def __init__(self,
                 max_contacts:  int = _TOUCH_MAX_CONTACTS,
                 feedback_mode: int = _TOUCH_FEEDBACK_INDIRECT):
        self.available    = False
        self.max_contacts = max_contacts
        self.feedback_mode = feedback_mode
        self._contacts    = [_TouchContact(i) for i in range(max_contacts)]
        self._inject      = None
        self._inject_err_logged = False

        if sys.platform != "win32":
            return

        try:
            user32 = ctypes.windll.user32
            # Mark process DPI-aware so injected pixel coordinates aren't
            # rescaled keeps controller-laser touch landing exact on
            # high-DPI displays. Safe to call repeatedly.
            try:
                user32.SetProcessDPIAware()
            except Exception:
                pass

            init = user32.InitializeTouchInjection
            init.argtypes = [ctypes.c_uint32, ctypes.c_uint32]
            init.restype  = ctypes.c_int
            ok = init(max_contacts, feedback_mode)
            if not ok:
                err = ctypes.windll.kernel32.GetLastError()
                print(f"[TouchInjector] InitializeTouchInjection failed "
                      f"(err={err}); falling back to mouse path")
                return

            inj = user32.InjectTouchInput
            inj.argtypes = [ctypes.c_uint32, ctypes.POINTER(_POINTER_TOUCH_INFO)]
            inj.restype  = ctypes.c_int
            self._inject = inj
            self.available = True
            print(f"[TouchInjector] ready "
                  f"(max_contacts={max_contacts}, feedback=0x{feedback_mode:x})")
        except Exception as exc:  # pragma: no cover - environment-dependent
            print(f"[TouchInjector] init failed: {exc}")

    # -- Public per-frame API --------------------------------------------

    def set(self, contact_id: int, x: int, y: int, want_down: bool):
        """Queue desired state of a contact for the next ``flush()``.

        Transitions are inferred automatically:

        * ``want_down=True``  after inactive DOWN
        * ``want_down=True``  after active   UPDATE
        * ``want_down=False`` after active   UP (preceded by an UPDATE
          when the position differs from the OS-known one)
        * ``want_down=False`` after inactive no-op
        """
        if not (0 <= contact_id < self.max_contacts):
            return
        c = self._contacts[contact_id]
        c.x_set = int(x)
        c.y_set = int(y)
        c._wanted = bool(want_down)

    def flush(self):
        """Emit queued transitions in one or more ``InjectTouchInput`` calls.

        Empirically (Win10/11), ``InjectTouchInput`` has two strict rules
        for multi-contact injection:

        1. **Every call must contain every currently-active contact.** Once
           a contact has had a DOWN injected, subsequent calls must list it
           (typically as UPDATE) until its UP otherwise the call is
           rejected with ``ERROR_INVALID_PARAMETER`` (87).
        2. **A UP must be emitted at the same coordinates as the prior
           DOWN/UPDATE.** A simultaneous "move + lift" is rejected. We
           split a moving-UP into an UPDATE-to-new-pos in one call and a
           UP-at-that-pos in the next.

        ``POINTER_FLAG_PRIMARY`` is *never* set: the OS assigns primary
        status automatically and rejects calls that try to override it.

        DOWN transitions for additional contacts (beyond one per call) are
        staggered across subsequent calls, matching Microsoft's canonical
        sample.
        """
        if not self.available:
            # Even when disabled, keep the contact array consistent so a
            # later re-enable starts from a sane state.
            for c in self._contacts:
                if not c._wanted:
                    c.active = False
            return

        # Classify each contact's transition for THIS frame.
        down_new = []   # inactive wanted     DOWN
        upd      = []   # active   still wanted UPDATE
        up_stat  = []   # active   up, no move UP directly
        up_moved = []   # active   up, moved   UPDATE then UP
        for c in self._contacts:
            if c._wanted and not c.active:
                down_new.append(c)
            elif c._wanted and c.active:
                upd.append(c)
            elif (not c._wanted) and c.active:
                if c.x_set == c.x and c.y_set == c.y:
                    up_stat.append(c)
                else:
                    up_moved.append(c)

        if not (down_new or upd or up_stat or up_moved):
            return

        # Phase A: one Win32 call containing every active contact, plus
        # at-most one new DOWN.
        first_down = down_new.pop(0) if down_new else None
        batch = []
        for c in upd:
            batch.append((c, c.x_set, c.y_set,
                          _POINTER_FLAG_UPDATE
                          | _POINTER_FLAG_INRANGE
                          | _POINTER_FLAG_INCONTACT))
            c.x, c.y = c.x_set, c.y_set
        for c in up_moved:
            # Inject the move now; UP happens in Phase B at the same pos.
            batch.append((c, c.x_set, c.y_set,
                          _POINTER_FLAG_UPDATE
                          | _POINTER_FLAG_INRANGE
                          | _POINTER_FLAG_INCONTACT))
            c.x, c.y = c.x_set, c.y_set
        for c in up_stat:
            batch.append((c, c.x, c.y, _POINTER_FLAG_UP))
        if first_down is not None:
            first_down.x, first_down.y = first_down.x_set, first_down.y_set
            batch.append((first_down, first_down.x, first_down.y,
                          _POINTER_FLAG_DOWN
                          | _POINTER_FLAG_INRANGE
                          | _POINTER_FLAG_INCONTACT))
            first_down.active = True
        if batch:
            self._emit(batch)

        for c in up_stat:
            c.active = False

        # Phase B: UP for contacts that moved this frame. All still-active
        # contacts must be present (as UPDATE) per rule 1.
        if up_moved and self.available:
            batch = [(c, c.x, c.y, _POINTER_FLAG_UP) for c in up_moved]
            for c in upd:
                if c.active:
                    batch.append((c, c.x, c.y,
                                  _POINTER_FLAG_UPDATE
                                  | _POINTER_FLAG_INRANGE
                                  | _POINTER_FLAG_INCONTACT))
            if first_down is not None and first_down.active:
                batch.append((first_down, first_down.x, first_down.y,
                              _POINTER_FLAG_UPDATE
                              | _POINTER_FLAG_INRANGE
                              | _POINTER_FLAG_INCONTACT))
            self._emit(batch)
            for c in up_moved:
                c.active = False

        # Phase C: remaining DOWNs, one per call, each piggy-backing every
        # currently-active contact as UPDATE for consistency.
        while down_new and self.available:
            nxt = down_new.pop(0)
            batch = []
            for c in self._contacts:
                if c.active and c is not nxt:
                    batch.append((c, c.x, c.y,
                                  _POINTER_FLAG_UPDATE
                                  | _POINTER_FLAG_INRANGE
                                  | _POINTER_FLAG_INCONTACT))
            nxt.x, nxt.y = nxt.x_set, nxt.y_set
            batch.append((nxt, nxt.x, nxt.y,
                          _POINTER_FLAG_DOWN
                          | _POINTER_FLAG_INRANGE
                          | _POINTER_FLAG_INCONTACT))
            nxt.active = True
            self._emit(batch)

    def _emit(self, batch):
        """Build a ``POINTER_TOUCH_INFO[]`` and call ``InjectTouchInput``.

        ``batch`` is a list of ``(contact, x, y, flags)`` tuples the
        coordinates are explicit so callers can emit a contact at a
        position different from its tracked one (used for the synthetic
        UPDATE-before-UP fix).
        """
        if not self.available or not batch:
            return
        arr = (_POINTER_TOUCH_INFO * len(batch))()
        r = _TOUCH_CONTACT_RADIUS_PX
        for i, (c, x, y, flags) in enumerate(batch):
            ti = arr[i]
            # ZeroMemory equivalent matches the MS sample exactly.
            ctypes.memset(ctypes.byref(ti), 0, ctypes.sizeof(ti))
            ti.pointerInfo.pointerType        = _PT_TOUCH
            ti.pointerInfo.pointerId          = ctypes.c_uint32(c.id).value
            ti.pointerInfo.ptPixelLocation.x  = x
            ti.pointerInfo.ptPixelLocation.y  = y
            ti.pointerInfo.pointerFlags       = flags
            ti.touchFlags = 0
            ti.touchMask  = (_TOUCH_MASK_CONTACTAREA
                             | _TOUCH_MASK_ORIENTATION
                             | _TOUCH_MASK_PRESSURE)
            ti.rcContact.left   = x - r
            ti.rcContact.top    = y - r
            ti.rcContact.right  = x + r
            ti.rcContact.bottom = y + r
            ti.orientation = 90                       # matches MS sample
            ti.pressure    = _TOUCH_DEFAULT_PRESSURE

        ok = self._inject(len(batch), arr)
        if not ok:
            err = ctypes.windll.kernel32.GetLastError()
            if not self._inject_err_logged:
                # Common error codes:
                #   5    = ERROR_ACCESS_DENIED   (UAC integrity mismatch)
                #   87   = ERROR_INVALID_PARAMETER (bad flags / state)
                #   5005 = ERROR_TIMEOUT         (stale contact rejected)
                print(f"[TouchInjector] InjectTouchInput failed "
                      f"(err={err}); disabling falling back to mouse path")
                self._inject_err_logged = True
            # Persistent failure: surrender so callers revert to the mouse
            # path without leaking phantom contacts.
            for c in self._contacts:
                c.active = False
            self.available = False

    def cancel_all(self):
        """Force UP on every active contact (shutdown / focus loss)."""
        if not self.available:
            return
        for c in self._contacts:
            c._wanted = False
        self.flush()


# Process-wide singleton `InitializeTouchInjection` may only be called once.
try:
    _touch_injector = _TouchInjector()
    _TOUCH_AVAILABLE = bool(_touch_injector.available)
    _TOUCH_FEEDBACK_MODE = _touch_injector.feedback_mode
except Exception as _touch_exc:  # pragma: no cover - environment-dependent
    _touch_injector = None
    _TOUCH_AVAILABLE = False
    _TOUCH_FEEDBACK_MODE = 0
    print(f"[OpenXRViewer] touch injector unavailable: {_touch_exc}")

# Touch-side cursor tuning (used by _handle_cursor / _handle_triggers).
# The controller pose is already smoothed upstream by `_get_smoothed_ray`,
# so the touch path sends the raw laser-mapped pixel position directly to
# the injector adding a second EMA stage here only added latency and was
# the root cause of the drag-lag / "fast-drag = small move" / cursor-hang
# bugs.  No tuning constants are needed at this layer; if jitter becomes a
# problem, prefer adjusting the pose-smoothing upstream.
# Per-hand contact IDs used with the touch injector.
_TOUCH_CONTACT_ID_LEFT  = 0
_TOUCH_CONTACT_ID_RIGHT = 1

# Windows input helpers (no-op on non-Windows)

if sys.platform == "win32":
    _U32 = ctypes.windll.user32

    class _MOUSEINPUT(ctypes.Structure):
        _fields_ = [("dx", ctypes.c_long), ("dy", ctypes.c_long),
                    ("mouseData", ctypes.c_ulong), ("dwFlags", ctypes.c_ulong),
                    ("time", ctypes.c_ulong), ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong))]

    class _INPUT(ctypes.Structure):
        class _I(ctypes.Union):
            _fields_ = [("mi", _MOUSEINPUT)]
        _anonymous_ = ("_i",)
        _fields_ = [("type", ctypes.c_ulong), ("_i", _I)]

    _MOUSEEVENTF_MOVE     = 0x0001
    _MOUSEEVENTF_LEFTDOWN = 0x0002
    _MOUSEEVENTF_LEFTUP   = 0x0004
    _MOUSEEVENTF_RIGHTDOWN= 0x0008
    _MOUSEEVENTF_RIGHTUP  = 0x0010
    _MOUSEEVENTF_ABSOLUTE = 0x8000
    _MOUSEEVENTF_WHEEL    = 0x0800
    _MOUSEEVENTF_HWHEEL   = 0x1000
    _KEYEVENTF_KEYUP      = 0x0002

    def _set_cursor_pos(x, y):
        # Use SetCursorPos with virtual-desktop pixel coordinates works across all
        # monitors.  The old SendInput+MOVE+ABSOLUTE approach required manual
        # normalisation against the primary-monitor size and was fragile for
        # multi-monitor setups where the primary monitor isn't at (0,0).
        ctypes.windll.user32.SetCursorPos(int(x), int(y))

    def _send_mouse_flags(flags):
        inp = _INPUT(type=0)
        inp.mi.dwFlags = flags
        ctypes.windll.user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(inp))

    def _send_key(vk, shift=False, ctrl=False, alt=False, win=False):
        kbd = ctypes.windll.user32.keybd_event
        # Press modifiers (chord support: Ctrl+C, Alt+Tab, Win+R, etc.)
        if ctrl:  kbd(0x11, 0, 0, 0)             # VK_CONTROL down
        if shift: kbd(0x10, 0, 0, 0)             # VK_SHIFT down
        if alt:   kbd(0x12, 0, 0, 0)             # VK_MENU (Alt) down
        if win:   kbd(0x5B, 0, 0, 0)             # VK_LWIN down
        kbd(vk, 0, 0, 0)                          # key down
        kbd(vk, 0, _KEYEVENTF_KEYUP, 0)           # key up
        # Release modifiers in reverse
        if win:   kbd(0x5B, 0, _KEYEVENTF_KEYUP, 0)
        if alt:   kbd(0x12, 0, _KEYEVENTF_KEYUP, 0)
        if shift: kbd(0x10, 0, _KEYEVENTF_KEYUP, 0)
        if ctrl:  kbd(0x11, 0, _KEYEVENTF_KEYUP, 0)

    def _get_desktop_size():
        return _U32.GetSystemMetrics(0), _U32.GetSystemMetrics(1)

    def _send_vscroll(amount):
        inp = _INPUT(type=0)
        inp.mi.dwFlags = _MOUSEEVENTF_WHEEL
        inp.mi.mouseData = ctypes.c_ulong(int(amount) & 0xFFFFFFFF)
        ctypes.windll.user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(inp))

    def _send_hscroll(amount):
        inp = _INPUT(type=0)
        inp.mi.dwFlags = _MOUSEEVENTF_HWHEEL
        inp.mi.mouseData = ctypes.c_ulong(int(amount) & 0xFFFFFFFF)
        ctypes.windll.user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(inp))
else:
    _U32 = None
    _KEYEVENTF_KEYUP = 0x0002

    def _set_cursor_pos(x, y): pass
    def _send_mouse_flags(flags): pass
    def _send_key(vk, shift=False, ctrl=False, alt=False, win=False): pass
    def _send_vscroll(amount): pass
    def _send_hscroll(amount): pass
    def _get_desktop_size(): return (1920, 1080)
    _MOUSEEVENTF_LEFTDOWN  = 0x0002
    _MOUSEEVENTF_LEFTUP    = 0x0004
    _MOUSEEVENTF_RIGHTDOWN = 0x0008
    _MOUSEEVENTF_RIGHTUP   = 0x0010

class OneEuroFilter:
    """Adaptive low-pass filter for hand-jitter reduction.

    Algorithm (from "1€Filter" by Géry Casiez et al.):
    dx  = (x - x_prev) / dt                  raw derivative
    dx^ = low_pass(dx, f_Cd, dt)             smoothed derivative
    f_C = min_cutoff + beta * |dx^|          adaptive cutoff (Hz)
    x^  = low_pass(x, f_C, dt)               filtered output

    The low-pass is a 1st-order RC filter:
    α   = 1 / (1 + τ / dt)                   smoothing factor
    τ   = 1 / (2π · f_C)                     time constant
    y   = α·x + (1-α)·y_prev

    Tuning guide:
    min_cutoff (Hz):      1.0.0 for hand tracking. Lower = smoother, more lag.
                            Start at 1.2.
    beta:                 0.007.05. Speed sensitivity higher responds faster
                            to quick moves but transmits more jitter. Start at 0.01.
    derivative_cutoff (Hz): 1.0 is typical. Lower = smoother derivative estimate.
    """
    __slots__ = ('min_cutoff', 'beta', 'derivative_cutoff', '_x_prev', '_dx_prev')

    def __init__(self, min_cutoff=1.2, beta=0.01, derivative_cutoff=1.0):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.derivative_cutoff = float(derivative_cutoff)
        self._x_prev = None    # previous filtered value
        self._dx_prev = None   # previous smoothed derivative

    def reset(self):
        self._x_prev = None
        self._dx_prev = None

    def _alpha(self, cutoff, dt):
        if dt <= 0.0:
            return 1.0
        tau = 1.0 / (2.0 * math.pi * max(cutoff, 0.001))
        return 1.0 / (1.0 + tau / dt)

    def filter(self, x, dt):
        if dt <= 0.0 or self._x_prev is None:
            self._x_prev = float(x)
            self._dx_prev = 0.0
            return float(x)

        # Derivative of the raw signal
        dx = (float(x) - self._x_prev) / dt

        # Smooth the derivative with fixed cutoff
        alpha_d = self._alpha(self.derivative_cutoff, dt)
        dx_hat = alpha_d * dx + (1.0 - alpha_d) * self._dx_prev

        # Adaptive cutoff: rises with speed less lag during fast motion
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)

        # Low-pass with adaptive cutoff
        alpha = self._alpha(cutoff, dt)
        x_hat = alpha * float(x) + (1.0 - alpha) * self._x_prev

        self._x_prev = x_hat
        self._dx_prev = dx_hat
        return x_hat


class EMAPositionFilter:
    """Simple exponential moving average fallback for debugging.

    y = α·x + (1-α)·y_prev
    """
    __slots__ = ('alpha', '_prev')

    def __init__(self, alpha=0.15):
        self.alpha = float(alpha)
        self._prev = None

    def reset(self):
        self._prev = None

    def filter(self, x):
        if self._prev is None:
            self._prev = float(x)
            return float(x)
        x_hat = self.alpha * float(x) + (1.0 - self.alpha) * self._prev
        self._prev = x_hat
        return x_hat


class OneEuroFilter3D:
    """Three independent One Euro Filters for 3D position (X, Y, Z)."""
    __slots__ = ('_fx', '_fy', '_fz')

    def __init__(self, min_cutoff=1.2, beta=0.01, derivative_cutoff=1.0):
        self._fx = OneEuroFilter(min_cutoff, beta, derivative_cutoff)
        self._fy = OneEuroFilter(min_cutoff, beta, derivative_cutoff)
        self._fz = OneEuroFilter(min_cutoff, beta, derivative_cutoff)

    def reset(self):
        self._fx.reset()
        self._fy.reset()
        self._fz.reset()

    def filter(self, pos, dt):
        x = self._fx.filter(float(pos[0]), dt)
        y = self._fy.filter(float(pos[1]), dt)
        z = self._fz.filter(float(pos[2]), dt)
        return np.array([x, y, z], dtype='f8')

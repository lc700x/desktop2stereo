"""Shared constants for the OpenXR viewer."""

import collections as _collections

EDGE_STRENGTH = 0.6 # snapping strength of cursor around screen edge

# Cursor ownership tuning (keyboard vs. virtual screen) consumed by _handle_cursor().
# Exposed here so they're easy to tweak without hunting through the method.
# Hysteresis bias (metres): how much closer the screen must be than the keyboard
# before the screen is allowed to steal the cursor. Larger = the keyboard keeps
# the cursor more aggressively when it sits just below the screen edge.
KB_CURSOR_PRIORITY_BIAS = 0.060
# Post-release grace (seconds): after the keyboard stops owning the cursor (typing
# ends / ray leaves the keys), keep the screen cursor suppressed this long so
# ownership doesn't snap to the screen while the user lifts off toward it.
KB_CURSOR_RELEASE_GRACE = 0.12

_DXGI_FORMAT_R8G8B8A8_UNORM_SRGB = 29
_DXGI_FORMAT_R8G8B8A8_UNORM      = 28
_DXGI_FORMAT_B8G8R8A8_UNORM_SRGB = 91
_DXGI_FORMAT_B8G8R8A8_UNORM      = 87

_D3D11_PREFERRED_FORMATS = [
    _DXGI_FORMAT_R8G8B8A8_UNORM_SRGB,
    _DXGI_FORMAT_R8G8B8A8_UNORM,
    _DXGI_FORMAT_B8G8R8A8_UNORM_SRGB,
    _DXGI_FORMAT_B8G8R8A8_UNORM,
]

# GL_SRGB8_ALPHA8: desktop captures are sRGB-encoded; signalling this to the
# compositor prevents it from treating gamma values as linear (which causes pale/washed-out colours).
_GL_SRGB8_ALPHA8 = 0x8C43

_BG_COLORS = [
    (0.000, 0.000, 0.000),   # default opaque black
    (0.000, 0.600, 0.200),   # green screen (VD passthrough)
]
_DEFAULT_BC = (1.0, 1.0, 1.0)
_DEFAULT_EF = (0.0, 0.0, 0.0)
_DEFAULT_TO = (0.0, 0.0)
_DEFAULT_TS = (1.0, 1.0)

# Draw the desktop a few millimetres in front of a loaded room monitor so the
# monitor glass does not z-fight the virtual screen, while nearer room objects
# still win the depth test and occlude it.
_SCREEN_ENV_DEPTH_BIAS_M = 0.003
# Curved screens keep one fixed angular span for every screen size. The base
# half-angle is the old default screen geometry (2.4m wide at 2.0m distance);
# 0.8 means 80% of that previous curvature, i.e. flatter.
_CURVED_CURVATURE_SCALE = 0.8
_CURVED_HALF_ANGLE_RAD = 0.6 * _CURVED_CURVATURE_SCALE

# Controller thumbstick dead-zone
DEAD = 0.15

# Vive trackpad region threshold for button emulation.
# When trackpad/click is pressed: |y| > threshold top/bottom (B/Y or A/X),
# |y| <= threshold center (thumbstick click).
_VIVE_TB_Y = 0.5


# Virtual keyboard layout

# (label, normal_vk, _, shifted_vk, width_units)
# VK codes: https://docs.microsoft.com/en-us/windows/win32/inputdev/virtual-key-codes
# vk == -1 marks a layout gap: the slot consumes width but renders nothing and
# generates no keystroke (used to align the navigation/arrow clusters).
_KB_UNITS_WIDE = 18   # total horizontal units per row
_KB_ROWS = [
    # Row F: Esc + F1…12 + PrtSc/ScrLk/Pause   (1.5 + 12 + 4.5 = 18)
    [('Esc',0x1B,None,0x1B,1.5),
    ('F1',0x70,None,0x70,1),('F2',0x71,None,0x71,1),
    ('F3',0x72,None,0x72,1),('F4',0x73,None,0x73,1),
    ('F5',0x74,None,0x74,1),('F6',0x75,None,0x75,1),
    ('F7',0x76,None,0x76,1),('F8',0x77,None,0x77,1),
    ('F9',0x78,None,0x78,1),('F10',0x79,None,0x79,1),
    ('F11',0x7A,None,0x7A,1),('F12',0x7B,None,0x7B,1),
    ('PrtSc',0x2C,None,0x2C,1.5),('ScrLk',0x91,None,0x91,1.5),('Pause',0x13,None,0x13,1.5)],
    # Row 0: number row + Ins/Hom/PgUp        (13 + 2 + 3 = 18)
    [('`',0xC0,'~',0xC0,1),('1',0x31,'!',0x31,1),('2',0x32,'@',0x32,1),
    ('3',0x33,'#',0x33,1),('4',0x34,'$',0x34,1),('5',0x35,'%',0x35,1),
    ('6',0x36,'^',0x36,1),('7',0x37,'&',0x37,1),('8',0x38,'*',0x38,1),
    ('9',0x39,'(',0x39,1),('0',0x30,')',0x30,1),('-',0xBD,'_',0xBD,1),
    ('=',0xBB,'+',0xBB,1),('Bksp',0x08,None,0x08,2),
    ('Ins',0x2D,None,0x2D,1),('Hom',0x24,None,0x24,1),('PgU',0x21,None,0x21,1)],
    # Row 1: QWERTY + Del/End/PgDn            (1.5 + 12 + 1.5 + 3 = 18)
    [('Tab',0x09,None,0x09,1.5),('Q',0x51,None,0x51,1),('W',0x57,None,0x57,1),
    ('E',0x45,None,0x45,1),('R',0x52,None,0x52,1),('T',0x54,None,0x54,1),
    ('Y',0x59,None,0x59,1),('U',0x55,None,0x55,1),('I',0x49,None,0x49,1),
    ('O',0x4F,None,0x4F,1),('P',0x50,None,0x50,1),('[',0xDB,'{',0xDB,1),
    (']',0xDD,'}',0xDD,1),('\\',0xDC,'|',0xDC,1.5),
    ('Del',0x2E,None,0x2E,1),('End',0x23,None,0x23,1),('PgD',0x22,None,0x22,1)],
    # Row 2: ASDF + 3-unit gap                (1.75 + 11 + 2.25 + 3 = 18)
    [('Caps',0x14,None,0x14,1.75),('A',0x41,None,0x41,1),('S',0x53,None,0x53,1),
    ('D',0x44,None,0x44,1),('F',0x46,None,0x46,1),('G',0x47,None,0x47,1),
    ('H',0x48,None,0x48,1),('J',0x4A,None,0x4A,1),('K',0x4B,None,0x4B,1),
    ('L',0x4C,None,0x4C,1),(';',0xBA,':',0xBA,1),("'",0xDE,'"',0xDE,1),
    ('Enter',0x0D,None,0x0D,2.25),
    ('',-1,None,-1,3)],
    # Row 3: ZXCV + Up arrow directly above Down
    # 2.25 + 10 + 2.75 = 15.0   then  gap(1) | 1) | gap(1)   occupies col 16-17 (same as 
    [('Shift',0x10,None,0x10,2.25),('Z',0x5A,None,0x5A,1),('X',0x58,None,0x58,1),
    ('C',0x43,None,0x43,1),('V',0x56,None,0x56,1),('B',0x42,None,0x42,1),
    ('N',0x4E,None,0x4E,1),('M',0x4D,None,0x4D,1),(',',0xBC,'<',0xBC,1),
    ('.',0xBE,'>',0xBE,1),('/',0xBF,'',0xBF,1),('Shift',0x10,None,0x10,2.75),
    ('',-1,None,-1,1),('Up',0x26,None,0x26,1),('',-1,None,-1,1)],
    # Row 4: bottom + arrow cluster Down sits directly under Up at col 16-17.
    # 1.5+1+1.25+7.5+1.25+1+1.5 = 15.0   then  1) | 1) | 1)
    [('Ctrl',0x11,None,0x11,1.5),('Win',0x5B,None,0x5B,1),
    ('Alt',0x12,None,0x12,1.25),
    ('Space',0x20,None,0x20,7.5),
    ('Alt',0x12,None,0x12,1.25),('Apps',0x5D,None,0x5D,1),
    ('Ctrl',0x11,None,0x11,1.5),
    ('Left',0x25,None,0x25,1),('Down',0x28,None,0x28,1),('Right',0x27,None,0x27,1)],
]

_KeyEntry = _collections.namedtuple('_KeyEntry', 'label shifted_label vk shifted_vk rect_uv rect_local')

_KB_TEX_W, _KB_TEX_H = 1280, 384   # keyboard texture: 6 rows × 18 units

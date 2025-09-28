from Xlib import display, X
import mss
import mss.tools

from Xlib.protocol import request

def get_window_coords(title):
    d = display.Display()
    root = d.screen().root
    
    # Get all windows
    window_ids = root.get_full_property(
        d.intern_atom('_NET_CLIENT_LIST'),
        X.AnyPropertyType
    ).value
    
    # Pre-intern atoms for common properties
    net_wm_name = d.intern_atom('_NET_WM_NAME')
    utf8_string = d.intern_atom('UTF8_STRING')
    
    for window_id in window_ids:
        window = d.create_resource_object('window', window_id)
        
        # Try multiple ways to get the window name
        name = None
        try:
            # Try _NET_WM_NAME first (UTF-8)
            name_prop = window.get_full_property(net_wm_name, utf8_string)
            if name_prop:
                name = name_prop.value.decode('utf-8')
            else:
                # Fall back to WM_NAME
                name = window.get_wm_name()
                if isinstance(name, bytes):
                    name = name.decode('utf-8', errors='replace')
        except:
            continue
        
        if name and title in name:
            try:
                # Get frame extents (margins)
                frame_extents = window.get_full_property(
                    d.intern_atom('_NET_FRAME_EXTENTS'),
                    X.AnyPropertyType
                )
                
                if frame_extents:
                    left, right, top, bottom = frame_extents.value
                else:
                    left = right = top = bottom = 0
                
                geom = window.get_geometry()
                
                return {
                    'left': geom.x + left,
                    'top': geom.y + top,
                    'width': geom.width - (left + right),
                    'height': geom.height - (top + bottom)
                }
            except:
                geom = window.get_geometry()
                return {
                    'left': geom.x,
                    'top': geom.y,
                    'width': geom.width,
                    'height': geom.height
                }
    return None

# def get_window_coords(title):
#     d = display.Display()
#     root = d.screen().root
    
#     # Get all windows
#     window_ids = root.get_full_property(
#         d.intern_atom('_NET_CLIENT_LIST'),
#         X.AnyPropertyType
#     ).value
    
#     for window_id in window_ids:
#         window = d.create_resource_object('window', window_id)
#         name = window.get_wm_name()
#         if name and title in name:
#             # Get absolute coordinates (accounting for window decorations)
#             # translate = window.translate_coords(root, 0, 0)
#             geom = window.get_geometry()
#             x, y = geom.x, geom.y
#             while True:
#                 parent = window.query_tree().parent
#                 pgeom = parent.get_geometry()
#                 x += pgeom.x
#                 y += pgeom.y
#                 if parent.id == root.id:
#                     break
#                 window = parent
#             return {
#                 'left': x,
#                 'top': y,
#                 'width': geom.width,
#                 'height': geom.height
#             }
#     return None

def capture_window(title, output_file='screenshot.png'):
    coords = get_window_coords(title)
    print(coords)
    if not coords:
        print(f"Window '{title}' not found")
        return
    
    with mss.mss() as sct:
        sct_img = sct.grab(coords)
        mss.tools.to_png(sct_img.rgb, sct_img.size, output=output_file)
        print(f"Screenshot saved to {output_file}")

# Example usage
capture_window("test_window.py - Desktop2Stereo - Visual Studio Code")
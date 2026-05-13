# openxr_viewer.py
# Renders Desktop2Stereo's depth-parallax left/right eye views into a VR headset
# via the pyopenxr binding (pip install pyopenxr).
# Uses a world-space virtual screen quad with proper per-eye view/projection matrices
# derived from xr.locate_views() for full 6DoF/3DoF head tracking.
# The depth-parallax FRAGMENT_SHADER from viewer.py is reused unchanged.

import sys
import math
import time
import ctypes
import queue as _queue
import collections

import glfw
import moderngl
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from utils import get_font_type
from OpenGL.GL import (
    glGenFramebuffers, glBindFramebuffer, glFramebufferTexture2D,
    glDeleteFramebuffers, glCheckFramebufferStatus,
    GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D,
    GL_FRAMEBUFFER_COMPLETE, GL_RGBA8,
    glGenBuffers, glDeleteBuffers, glBindBuffer, glBufferData,
    glBindTexture, glTexSubImage2D, glGenerateMipmap,
    GL_PIXEL_UNPACK_BUFFER, GL_PIXEL_PACK_BUFFER, GL_DYNAMIC_DRAW, GL_STREAM_READ,
    GL_RGB, GL_RED, GL_RGBA, GL_BGRA, GL_UNSIGNED_BYTE, GL_FLOAT,
    glDisable, GL_FRAMEBUFFER_SRGB,
    glFrontFace, GL_CW, GL_CCW,
    glTexParameterf, GL_TEXTURE_LOD_BIAS,
    glMapBuffer, glUnmapBuffer, GL_READ_ONLY, GL_MAP_UNSYNCHRONIZED_BIT,
    glReadPixels, glFlush, glGenTextures, glDeleteTextures,
    glFinish
)

try:
    import xr
    OPENXR_AVAILABLE = True
except ImportError:
    OPENXR_AVAILABLE = False
    print("[OpenXRViewer] pyopenxr not installed. Run: pip install pyopenxr")

# ---------------------------------------------------------------------------
# glb loader (for VR controller models)
# ---------------------------------------------------------------------------
import struct
import json
import io as _io
from PIL import Image

def _read_glb_chunks(data):
    magic = struct.unpack_from('<I', data, 0)[0]
    if magic != 0x46546C67:
        raise ValueError(f"Not a GLB file (magic=0x{magic:08X})")
    total_len = struct.unpack_from('<I', data, 8)[0]
    offset = 12
    json_data, bin_data = None, None
    while offset < total_len:
        chunk_len = struct.unpack_from('<I', data, offset)[0]
        chunk_type = struct.unpack_from('<I', data, offset + 4)[0]
        raw = data[offset + 8:offset + 8 + chunk_len]
        if chunk_type == 0x4E4F534A:
            json_data = json.loads(raw.decode('utf-8'))
        elif chunk_type == 0x004E4942:
            bin_data = raw
        offset += 8 + chunk_len
    return json_data, bin_data


_DTYPE_MAP = {5120: np.int8, 5121: np.uint8, 5122: np.int16,
              5123: np.uint16, 5125: np.uint32, 5126: np.float32}
_TYPE_NC = {'SCALAR': 1, 'VEC2': 2, 'VEC3': 3, 'VEC4': 4,
            'MAT2': 4, 'MAT3': 9, 'MAT4': 16}


def _get_accessor(gltf, bin_data, acc_idx):
    """Extract numpy array from a glTF accessor."""
    acc = gltf['accessors'][acc_idx]
    bv = gltf['bufferViews'][acc['bufferView']]
    off = bv.get('byteOffset', 0) + acc.get('byteOffset', 0)
    nc = _TYPE_NC[acc['type']]
    dt = np.dtype(_DTYPE_MAP[acc['componentType']]).newbyteorder('<')
    arr = np.frombuffer(bin_data[off:off + acc['count'] * nc * dt.itemsize], dtype=dt)
    if nc > 1:
        arr = arr.reshape(acc['count'], nc)
    # Normalize data types for consistency
    if acc['componentType'] in (5121, 5123, 5125):
        arr = arr.astype(np.uint32)
    elif acc['componentType'] == 5126:
        arr = arr.astype(np.float32)
    return arr


def _quat_to_mat4(q):
    """Convert quaternion [x, y, z, w] to 4x4 rotation matrix."""
    x, y, z, w = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z
    return np.array([
        [1-2*(yy+zz), 2*(xy-wz),   2*(xz+wy),   0],
        [2*(xy+wz),   1-2*(xx+zz), 2*(yz-wx),   0],
        [2*(xz-wy),   2*(yz+wx),   1-2*(xx+yy), 0],
        [0,           0,           0,           1],
    ], dtype=np.float64)


def _build_node_matrices(gltf):
    """Compute world matrix for each node (top-down). Returns list of 4x4 float64 matrices.
    Parent world matrix = parent_matrix @ local_matrix.
    Local matrix = translation * rotation * scale.
    Root nodes assume identity parent matrix.
    """
    nodes = gltf.get('nodes', [])
    n = len(nodes)
    if n == 0:
        return []

    # Build local matrices
    local_mats = []
    for node in nodes:
        t = node.get('translation', [0, 0, 0])
        r = node.get('rotation', [0, 0, 0, 1])  # [x, y, z, w]
        s = node.get('scale', [1, 1, 1])

        T = np.eye(4, dtype=np.float64)
        T[:3, 3] = t
        R = _quat_to_mat4(r)
        S_mat = np.diag([s[0], s[1], s[2], 1.0]).astype(np.float64)
        local_mats.append(T @ R @ S_mat)

    # Build child -> parent mapping
    parent = [-1] * n
    for pi, node in enumerate(nodes):
        for ci in node.get('children', []):
            parent[ci] = pi

    # Topological order (BFS from roots) to compute world matrices
    world_mats = [None] * n
    queue = [i for i in range(n) if parent[i] == -1]
    for i in queue:
        world_mats[i] = local_mats[i].copy()

    head = 0
    while head < len(queue):
        pi = queue[head]
        head += 1
        for ci in nodes[pi].get('children', []):
            if world_mats[ci] is None:
                world_mats[ci] = world_mats[pi] @ local_mats[ci]
                queue.append(ci)

    # Isolated nodes (no parent) just use local matrix
    for i in range(n):
        if world_mats[i] is None:
            world_mats[i] = local_mats[i].copy()

    return world_mats


def _apply_transform(vertices_xyz, matrix_4x4):
    """Apply 4x4 transformation matrix to vertex positions."""
    n = vertices_xyz.shape[0]
    ones = np.ones((n, 1), dtype=np.float64)
    v4 = np.hstack([vertices_xyz.astype(np.float64), ones])
    t = (matrix_4x4 @ v4.T).T
    return t[:, :3].astype(np.float32)


def load_glb_model(path):
    """Load a GLB model, apply node transformations.
    Returns:
        primitives: list of dict with keys:
            vertices (N, 8 float32: pos xyz, normal xyz, uv)
            indices (M, uint32)
            tex_id (int, index into textures)
        textures: list of numpy RGBA uint8 arrays
    """
    with open(path, 'rb') as f:
        data = f.read()
    gltf, bin_data = _read_glb_chunks(data)

    # World matrices for all nodes
    world_mats = _build_node_matrices(gltf)
    nodes = gltf.get('nodes', [])

    # Map mesh index to world matrix (first node referencing the mesh)
    mesh_world_mat = {}
    for ni, node in enumerate(nodes):
        mi = node.get('mesh')
        if mi is not None and mi not in mesh_world_mat:
            mesh_world_mat[mi] = world_mats[ni]

    # Extract textures
    all_textures = []
    if 'images' in gltf:
        for img in gltf['images']:
            tex_data = None
            if 'bufferView' in img:
                bv = gltf['bufferViews'][img['bufferView']]
                off = bv.get('byteOffset', 0)
                tex_data = bin_data[off:off + bv['byteLength']]
            elif 'uri' in img and img['uri'].startswith('data:'):
                import base64
                tex_data = base64.b64decode(img['uri'].split(',', 1)[1])
            if tex_data:
                pil_img = Image.open(_io.BytesIO(tex_data))
                pil_img = pil_img.convert('RGBA')
                all_textures.append(np.array(pil_img, dtype=np.uint8))
            else:
                all_textures.append(None)

    # Map texture index to image index
    tex_img_map = {}
    if 'textures' in gltf:
        for ti, tex in enumerate(gltf['textures']):
            si = tex.get('source', 0)
            tex_img_map[ti] = si if si < len(all_textures) else -1

    primitives = []
    for mi, mesh in enumerate(gltf.get('meshes', [])):
        world_mat = mesh_world_mat.get(mi, np.eye(4, dtype=np.float64))
        for prim in mesh.get('primitives', []):
            attrs = prim['attributes']
            pos = _get_accessor(gltf, bin_data, attrs['POSITION'])

            # Extract normals if present, else zeros
            if 'NORMAL' in attrs:
                norm = _get_accessor(gltf, bin_data, attrs['NORMAL'])
            else:
                norm = np.zeros((pos.shape[0], 3), dtype=np.float32)

            # Apply node world matrix: position with full 4x4, normals with rotation part only
            if not np.allclose(world_mat, np.eye(4)):
                pos = _apply_transform(pos, world_mat)
                rot3 = world_mat[:3, :3].astype(np.float32)
                norm = (rot3 @ norm.T).T.astype(np.float32)

            # Extract UV coordinates
            if 'TEXCOORD_0' in attrs:
                uv = _get_accessor(gltf, bin_data, attrs['TEXCOORD_0'])
                if uv.shape[1] > 2:
                    uv = uv[:, :2]
            else:
                uv = np.zeros((pos.shape[0], 2), dtype=np.float32)

            # Combine: position (3), normal (3), uv (2) -> [px,py,pz, nx,ny,nz, u,v] = 8 floats
            vertices = np.hstack([pos, norm, uv]).astype(np.float32)

            # Indices
            if 'indices' in prim:
                indices = _get_accessor(gltf, bin_data, prim['indices'])
            else:
                indices = np.arange(pos.shape[0], dtype=np.uint32)

            # Texture ID from material
            tex_id = -1
            mat_idx = prim.get('material')
            if mat_idx is not None and 'materials' in gltf:
                mat = gltf['materials'][mat_idx]
                bt = mat.get('pbrMetallicRoughness', {}).get('baseColorTexture')
                if bt and 'index' in bt:
                    tid = tex_img_map.get(bt['index'], -1)
                    if tid >= 0 and all_textures[tid] is not None:
                        tex_id = tid

            primitives.append({'vertices': vertices, 'indices': indices,
                               'tex_id': tex_id})

    return primitives, all_textures

# ---------------------------------------------------------------------------
# D3D11 ctypes helpers (Windows only)
# ---------------------------------------------------------------------------
# DXGI / D3D11 format constants used for swapchain negotiation
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

if sys.platform == "win32":
    import ctypes.wintypes as _wintypes

    _d3d11 = None
    _dxgi  = None

    def _load_d3d11():
        global _d3d11, _dxgi
        if _d3d11 is None:
            _d3d11 = ctypes.windll.LoadLibrary("d3d11.dll")
            _dxgi  = ctypes.windll.LoadLibrary("dxgi.dll")

    # D3D_DRIVER_TYPE / D3D_FEATURE_LEVEL constants
    _D3D_DRIVER_TYPE_HARDWARE  = 1
    _D3D_DRIVER_TYPE_WARP      = 5
    _D3D11_SDK_VERSION         = 7
    _D3D_FEATURE_LEVEL_11_0    = 0xb000
    _D3D_FEATURE_LEVEL_10_1    = 0xa100
    _D3D_FEATURE_LEVEL_10_0    = 0xa000

    def _create_d3d11_device(adapter_luid=None):
        """Create an ID3D11Device + ID3D11DeviceContext via ctypes.
        Returns (device_ptr, context_ptr, feature_level) as c_void_p values.
        adapter_luid: optional _LUID from GraphicsRequirementsD3D11KHR to pick the correct adapter.
        """
        _load_d3d11()

        feature_levels = (ctypes.c_int * 3)(
            _D3D_FEATURE_LEVEL_11_0,
            _D3D_FEATURE_LEVEL_10_1,
            _D3D_FEATURE_LEVEL_10_0,
        )
        device      = ctypes.c_void_p(0)
        context     = ctypes.c_void_p(0)
        feat_out    = ctypes.c_int(0)

        # If adapter_luid provided, try to find the matching IDXGIAdapter
        adapter = ctypes.c_void_p(0)
        if adapter_luid is not None:
            try:
                adapter = _find_dxgi_adapter(adapter_luid)
            except Exception as e:
                print(f"[OpenXRViewer] LUID adapter lookup failed ({e}), using default")
                adapter = ctypes.c_void_p(0)

        hr = _d3d11.D3D11CreateDevice(
            adapter,                              # pAdapter (NULL = default)
            _D3D_DRIVER_TYPE_HARDWARE if not adapter else 0,  # DriverType (0=unknown when adapter set)
            None,                                 # Software
            0,                                    # Flags
            feature_levels,
            3,                                    # FeatureLevels count
            _D3D11_SDK_VERSION,
            ctypes.byref(device),
            ctypes.byref(feat_out),
            ctypes.byref(context),
        )
        if hr != 0:
            raise RuntimeError(f"D3D11CreateDevice failed: hr=0x{hr & 0xFFFFFFFF:08x}")
        return device, context, feat_out.value

    # IID_IDXGIFactory1  {770aae78-f26f-4dba-a829-253c83d1b387}
    _IID_IDXGIFactory1 = (ctypes.c_byte * 16)(
        0x78, 0xae, 0x0a, 0x77, 0x6f, 0xf2, 0xba, 0x4d,
        0xa8, 0x29, 0x25, 0x3c, 0x83, 0xd1, 0xb3, 0x87,
    )

    def _find_dxgi_adapter(luid):
        """Return an IDXGIAdapter* (c_void_p) matching the given _LUID, or raise."""
        _load_d3d11()
        factory = ctypes.c_void_p(0)
        hr = _dxgi.CreateDXGIFactory1(ctypes.byref((ctypes.c_byte * 16)(*_IID_IDXGIFactory1)),
                                      ctypes.byref(factory))
        if hr != 0:
            raise RuntimeError(f"CreateDXGIFactory1 failed: 0x{hr & 0xFFFFFFFF:08x}")

        # IDXGIFactory1 vtable: [0]=QI [1]=AddRef [2]=Release ... [7]=EnumAdapters1
        # We use EnumAdapters (index 6) which is on IDXGIFactory (parent interface)
        # Layout: IUnknown(0-2) + IDXGIObject(3-6) + IDXGIFactory(7=EnumAdapters, 8=MakeWindowAssoc, 9=GetWindowAssoc, 10=CreateSwapChain, 11=CreateSoftwareAdapter) + IDXGIFactory1(12=EnumAdapters1, 13=IsCurrent)
        vtbl = ctypes.cast(factory, ctypes.POINTER(ctypes.c_void_p)).contents.value
        enum_adapters = ctypes.CFUNCTYPE(ctypes.c_long, ctypes.c_void_p, ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p))(
            ctypes.cast(vtbl + 7 * ctypes.sizeof(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p)).contents.value
        )
        release_fn = ctypes.CFUNCTYPE(ctypes.c_ulong, ctypes.c_void_p)(
            ctypes.cast(vtbl + 2 * ctypes.sizeof(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p)).contents.value
        )

        idx = 0
        result_adapter = ctypes.c_void_p(0)
        while True:
            adapter = ctypes.c_void_p(0)
            hr = enum_adapters(factory, idx, ctypes.byref(adapter))
            if hr != 0:
                break
            # IDXGIAdapter vtable: QI(0) AddRef(1) Release(2) SetPrivateData(3) SetPrivateDataInterface(4) GetPrivateData(5) GetParent(6) EnumOutputs(7) GetDesc(8)
            # DXGI_ADAPTER_DESC is: Description[128 wchar], VendorId, DeviceId, SubSysId, Revision, DedicatedVideoMemory, DedicatedSystemMemory, SharedSystemMemory, AdapterLuid
            adapter_vtbl = ctypes.cast(adapter, ctypes.POINTER(ctypes.c_void_p)).contents.value
            # GetDesc is vtbl[8]
            get_desc = ctypes.CFUNCTYPE(ctypes.c_long, ctypes.c_void_p, ctypes.c_void_p)(
                ctypes.cast(adapter_vtbl + 8 * ctypes.sizeof(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p)).contents.value
            )
            # DXGI_ADAPTER_DESC: 128*2 bytes description + 4*4 IDs + 3*8 memory + 8 luid = 128*2+16+24+8 = 304 bytes
            desc_buf = (ctypes.c_byte * 304)()
            get_desc(adapter, desc_buf)
            # LUID is at offset 128*2 + 4*4 + 3*8 = 256+16+24 = 296 bytes
            luid_low  = ctypes.c_ulong.from_buffer_copy(desc_buf, 296).value
            luid_high = ctypes.c_long.from_buffer_copy(desc_buf, 300).value
            adapter_rel = ctypes.CFUNCTYPE(ctypes.c_ulong, ctypes.c_void_p)(
                ctypes.cast(adapter_vtbl + 2 * ctypes.sizeof(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p)).contents.value
            )
            if luid_low == luid.low_part and luid_high == luid.high_part:
                result_adapter = adapter
                break
            adapter_rel(adapter)
            idx += 1

        release_fn(factory)
        if not result_adapter:
            raise RuntimeError("Matching DXGI adapter not found for LUID")
        return result_adapter

    def _d3d11_update_subresource(context, dst, src_ptr, row_pitch):
        """Write CPU data into a D3D11 texture via UpdateSubresource (vtbl index 48).
        Works with any format including SRGB — no staging texture needed.
        src_ptr: integer address of the source data (already row-reversed).
        """
        _UPDATE_SR_VTBL_IDX = 48
        vtbl = ctypes.cast(context, ctypes.POINTER(ctypes.c_void_p)).contents.value
        fn_ptr = ctypes.cast(
            vtbl + _UPDATE_SR_VTBL_IDX * ctypes.sizeof(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_void_p),
        ).contents.value
        UpdateFn = ctypes.CFUNCTYPE(
            None,
            ctypes.c_void_p,  # this
            ctypes.c_void_p,  # pDstResource
            ctypes.c_uint,    # DstSubresource
            ctypes.c_void_p,  # pDstBox (NULL = whole texture)
            ctypes.c_void_p,  # pSrcData
            ctypes.c_uint,    # SrcRowPitch
            ctypes.c_uint,    # SrcDepthPitch
        )(fn_ptr)
        UpdateFn(
            context.value,
            ctypes.cast(dst, ctypes.c_void_p).value,
            0,        # subresource 0
            None,     # full texture
            src_ptr,
            row_pitch,
            0,
        )



    # NV_DX_interop2 helpers (NVIDIA only, zero-copy GL↔D3D11)
    _nv_dx_interop_available = False
    _wglDXOpenDeviceNV        = None
    _wglDXCloseDeviceNV       = None
    _wglDXRegisterObjectNV    = None
    _wglDXLockObjectsNV       = None
    _wglDXUnlockObjectsNV     = None
    _wglDXUnregisterObjectNV  = None

    def _load_nv_dx_interop():
        """Try to load WGL_NV_DX_interop2 extension functions."""
        global _nv_dx_interop_available, _wglDXOpenDeviceNV, _wglDXCloseDeviceNV
        global _wglDXRegisterObjectNV, _wglDXLockObjectsNV, _wglDXUnlockObjectsNV
        global _wglDXUnregisterObjectNV
        if _nv_dx_interop_available:
            return True
        try:
            from OpenGL.WGL.NV.DX_interop2 import (
                wglDXOpenDeviceNV, wglDXCloseDeviceNV,
                wglDXRegisterObjectNV, wglDXLockObjectsNV,
                wglDXUnlockObjectsNV, wglDXUnregisterObjectNV,
            )
            _wglDXOpenDeviceNV       = wglDXOpenDeviceNV
            _wglDXCloseDeviceNV      = wglDXCloseDeviceNV
            _wglDXRegisterObjectNV   = wglDXRegisterObjectNV
            _wglDXLockObjectsNV      = wglDXLockObjectsNV
            _wglDXUnlockObjectsNV    = wglDXUnlockObjectsNV
            _wglDXUnregisterObjectNV = wglDXUnregisterObjectNV
            _nv_dx_interop_available = True
            return True
        except ImportError:
            try:
                # Fallback: load via wglGetProcAddress
                from OpenGL.GL.WGL.NV.DX_interop import (
                    wglDXOpenDeviceNV, wglDXCloseDeviceNV,
                    wglDXRegisterObjectNV, wglDXLockObjectsNV,
                    wglDXUnlockObjectsNV, wglDXUnregisterObjectNV,
                )
                _wglDXOpenDeviceNV       = wglDXOpenDeviceNV
                _wglDXCloseDeviceNV      = wglDXCloseDeviceNV
                _wglDXRegisterObjectNV   = wglDXRegisterObjectNV
                _wglDXLockObjectsNV      = wglDXLockObjectsNV
                _wglDXUnlockObjectsNV    = wglDXUnlockObjectsNV
                _wglDXUnregisterObjectNV = wglDXUnregisterObjectNV
                _nv_dx_interop_available = True
                return True
            except (ImportError, AttributeError):
                return False
        except (ImportError, AttributeError):
            return False

    # ── EXT_memory_object_win32 helpers (cross-vendor, GL 4.5+) ──────────
    _ext_mem_available       = False
    _glImportMemoryWin32HandleEXT  = None
    _glTextureStorageMem2DEXT      = None
    _glCreateMemoryObjectsEXT      = None
    _glDeleteMemoryObjectsEXT      = None

    # Handle types for glImportMemoryWin32HandleEXT
    _GL_HANDLE_TYPE_OPAQUE_WIN32_EXT        = 0x9587
    _GL_HANDLE_TYPE_D3D11_TEXTURE_KTX_Z     = None  # not used

    def _load_ext_memory_object():
        """Try to load GL_EXT_memory_object_win32 + GL_EXT_memory_object."""
        global _ext_mem_available, _glImportMemoryWin32HandleEXT
        global _glTextureStorageMem2DEXT, _glCreateMemoryObjectsEXT
        global _glDeleteMemoryObjectsEXT
        if _ext_mem_available:
            return True
        try:
            from OpenGL.GL.EXT.memory_object_win32 import glImportMemoryWin32HandleEXT
            from OpenGL.GL.EXT.memory_object import (
                glCreateMemoryObjectsEXT, glDeleteMemoryObjectsEXT,
            )
            from OpenGL.GL.EXT.memory_object_fd import glTextureStorageMem2DEXT
            _glImportMemoryWin32HandleEXT = glImportMemoryWin32HandleEXT
            _glCreateMemoryObjectsEXT     = glCreateMemoryObjectsEXT
            _glDeleteMemoryObjectsEXT     = glDeleteMemoryObjectsEXT
            _glTextureStorageMem2DEXT     = glTextureStorageMem2DEXT
            _ext_mem_available = True
            return True
        except (ImportError, AttributeError):
            # Fallback: load via raw ctypes from wglGetProcAddress
            try:
                from OpenGL.GL import wglGetProcAddress
                _names = {
                    'glImportMemoryWin32HandleEXT': ctypes.c_void_p,
                    'glTextureStorageMem2DEXT':     ctypes.c_void_p,
                    'glCreateMemoryObjectsEXT':     ctypes.c_void_p,
                    'glDeleteMemoryObjectsEXT':     ctypes.c_void_p,
                }
                _ptrs = {}
                for name in _names:
                    addr = wglGetProcAddress(name.encode() if hasattr(name, 'encode') else name)
                    if not addr:
                        raise RuntimeError(f"{name} not found")
                    _ptrs[name] = addr
                # Build ctypes function wrappers
                _glImportMemoryWin32HandleEXT = ctypes.CFUNCTYPE(
                    None, ctypes.c_uint, ctypes.c_uint64, ctypes.c_uint, ctypes.c_void_p,
                )(ctypes.cast(_ptrs['glImportMemoryWin32HandleEXT'], ctypes.c_void_p).value)
                _glTextureStorageMem2DEXT = ctypes.CFUNCTYPE(
                    None, ctypes.c_uint, ctypes.c_int, ctypes.c_uint, ctypes.c_int, ctypes.c_int, ctypes.c_uint, ctypes.c_uint64,
                )(ctypes.cast(_ptrs['glTextureStorageMem2DEXT'], ctypes.c_void_p).value)
                _glCreateMemoryObjectsEXT = ctypes.CFUNCTYPE(
                    None, ctypes.c_int, ctypes.POINTER(ctypes.c_uint),
                )(ctypes.cast(_ptrs['glCreateMemoryObjectsEXT'], ctypes.c_void_p).value)
                _glDeleteMemoryObjectsEXT = ctypes.CFUNCTYPE(
                    None, ctypes.c_int, ctypes.POINTER(ctypes.c_uint),
                )(ctypes.cast(_ptrs['glDeleteMemoryObjectsEXT'], ctypes.c_void_p).value)
                _ext_mem_available = True
                return True
            except Exception:
                return False

    def _create_d3d11_shared_texture(device, w, h, fmt=_DXGI_FORMAT_R8G8B8A8_UNORM):
        """Create a D3D11 texture with D3D11_RESOURCE_MISC_SHARED_NTHANDLE.

        Returns (texture_ptr, shared_handle) as c_void_p values.
        The shared_handle is an NT kernel handle suitable for
        glImportMemoryWin32HandleEXT.
        """
        desc = (
            ctypes.c_uint(w),           # Width
            ctypes.c_uint(h),           # Height
            ctypes.c_uint(1),           # MipLevels
            ctypes.c_uint(1),           # ArraySize
            ctypes.c_uint(fmt),         # Format
            # DXGI_SAMPLE_DESC
            ctypes.c_uint(1),           # Count
            ctypes.c_uint(0),           # Quality
            ctypes.c_uint(0),           # Usage (D3D11_USAGE_DEFAULT)
            ctypes.c_uint(0x40),        # BindFlags (D3D11_BIND_SHADER_RESOURCE = 0x80 | D3D11_BIND_RENDER_TARGET = 0x20)
            ctypes.c_uint(0),           # CPUAccessFlags
            ctypes.c_uint(0x2),         # MiscFlags (D3D11_RESOURCE_MISC_SHARED_NTHANDLE = 0x800, but we also need SHARED = 0x2)
        )
        # Actually D3D11_RESOURCE_MISC_SHARED_NTHANDLE is the right flag for NT handles.
        # Let me redo the struct properly.
        # D3D11_TEXTURE2D_DESC layout (order matters):
        # Width:UINT, Height:UINT, MipLevels:UINT, ArraySize:UINT,
        # Format:DXGI_FORMAT, SampleDesc:DXGI_SAMPLE_DESC,
        # Usage:D3D11_USAGE, BindFlags:UINT, CPUAccessFlags:UINT, MiscFlags:UINT

        _D3D11_BIND_SHADER_RESOURCE = 0x8
        _D3D11_BIND_RENDER_TARGET   = 0x20
        _D3D11_RESOURCE_MISC_SHARED_NTHANDLE = 0x800

        _TEX2D_DESC_FMT = (
            'I I I I I I I I I I I'
        )  # 11 UINTs
        # We need to pack SampleDesc.Count and SampleDesc.Quality as two UINTs

        tex_desc = (
            ctypes.c_uint * 11
        )(
            w, h, 1, 1, fmt,
            1, 0,              # SampleDesc
            0,                  # D3D11_USAGE_DEFAULT
            _D3D11_BIND_SHADER_RESOURCE | _D3D11_BIND_RENDER_TARGET,
            0,                  # CPUAccessFlags
            _D3D11_RESOURCE_MISC_SHARED_NTHANDLE,
        )

        tex_ptr = ctypes.c_void_p(0)
        vtbl = ctypes.cast(device, ctypes.POINTER(ctypes.c_void_p)).contents.value
        # ID3D11Device::CreateTexture2D at vtable index 5
        create_tex2d = ctypes.CFUNCTYPE(
            ctypes.c_long,
            ctypes.c_void_p,                          # this
            ctypes.POINTER(ctypes.c_uint * 11),       # pDesc
            ctypes.c_void_p,                          # pInitialData
            ctypes.POINTER(ctypes.c_void_p),          # ppTexture2D
        )(ctypes.cast(vtbl + 5 * ctypes.sizeof(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p)).contents.value)

        hr = create_tex2d(device, ctypes.byref(tex_desc), None, ctypes.byref(tex_ptr))
        if hr != 0:
            raise RuntimeError(f"CreateTexture2D(shared) failed: hr=0x{hr & 0xFFFFFFFF:08x}")

        # Get shared handle via IDXGIResource1::CreateSharedHandle
        # First get IDXGIResource from ID3D11Texture2D via QueryInterface
        # IID_IDXGIResource1 = {7632e1f5-ee65-4ca2-87fd-4c20ee8d71a9}
        _IID_IDXGIResource1 = (ctypes.c_byte * 16)(
            0xf5, 0xe1, 0x32, 0x76, 0x65, 0xee, 0xa2, 0x4c,
            0x87, 0xfd, 0x4c, 0x20, 0xee, 0x8d, 0x71, 0xa9,
        )
        dxgi_res = ctypes.c_void_p(0)
        tex_vtbl = ctypes.cast(tex_ptr, ctypes.POINTER(ctypes.c_void_p)).contents.value
        qi_fn = ctypes.CFUNCTYPE(
            ctypes.c_long, ctypes.c_void_p, ctypes.POINTER(ctypes.c_byte * 16), ctypes.POINTER(ctypes.c_void_p),
        )(ctypes.cast(tex_vtbl, ctypes.POINTER(ctypes.c_void_p)).contents.value)
        hr = qi_fn(tex_ptr, ctypes.byref(_IID_IDXGIResource1), ctypes.byref(dxgi_res))
        if hr != 0 or not dxgi_res:
            tex_vtbl2 = ctypes.cast(tex_ptr, ctypes.POINTER(ctypes.c_void_p)).contents.value
            release_fn2 = ctypes.CFUNCTYPE(ctypes.c_ulong, ctypes.c_void_p)(
                ctypes.cast(tex_vtbl2 + 2 * ctypes.sizeof(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p)).contents.value
            )
            release_fn2(tex_ptr)
            raise RuntimeError(f"QueryInterface(IDXGIResource1) failed: hr=0x{hr & 0xFFFFFFFF:08x}")

        # IDXGIResource1::CreateSharedHandle
        # Params: dwAccess, lpAttributes, dwAccessFlags, lpName, pHandle
        _DXGI_SHARED_RESOURCE_READ = 0x80000000
        _DXGI_SHARED_RESOURCE_WRITE = 1
        dxgi_vtbl = ctypes.cast(dxgi_res, ctypes.POINTER(ctypes.c_void_p)).contents.value
        create_sh = ctypes.CFUNCTYPE(
            ctypes.c_long,
            ctypes.c_void_p,      # this
            ctypes.c_uint,        # dwAccess
            ctypes.c_void_p,      # lpAttributes (NULL)
            ctypes.c_uint,        # dwAccessFlags
            ctypes.c_void_p,     # lpName (NULL)
            ctypes.POINTER(ctypes.c_void_p),  # pHandle (out)
        )(ctypes.cast(dxgi_vtbl + 12 * ctypes.sizeof(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p)).contents.value)

        shared_handle = ctypes.c_void_p(0)
        hr = create_sh(
            dxgi_res,
            _DXGI_SHARED_RESOURCE_READ | _DXGI_SHARED_RESOURCE_WRITE,
            None, 0, None,
            ctypes.byref(shared_handle),
        )
        # Release DXGI resource (we only needed it for CreateSharedHandle)
        dxgi_rel = ctypes.CFUNCTYPE(ctypes.c_ulong, ctypes.c_void_p)(
            ctypes.cast(dxgi_vtbl + 2 * ctypes.sizeof(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p)).contents.value
        )
        dxgi_rel(dxgi_res)

        if hr != 0 or not shared_handle:
            tex_rel = ctypes.CFUNCTYPE(ctypes.c_ulong, ctypes.c_void_p)(
                ctypes.cast(tex_vtbl + 2 * ctypes.sizeof(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p)).contents.value
            )
            tex_rel(tex_ptr)
            raise RuntimeError(f"CreateSharedHandle failed: hr=0x{hr & 0xFFFFFFFF:08x}")

        return tex_ptr, shared_handle

else:
    def _create_d3d11_device(adapter_luid=None):
        raise RuntimeError("D3D11 only available on Windows")

from viewer import FRAGMENT_SHADER, BACKEND
try:
    from viewer import CUDART_GL
except ImportError:
    CUDART_GL = None

# GL_SRGB8_ALPHA8: desktop captures are sRGB-encoded; signalling this to the
# compositor prevents it from treating gamma values as linear (which causes pale/washed-out colours).
_GL_SRGB8_ALPHA8 = 0x8C43

# World-space vertex shader: applies MVP to place the quad in the scene
_WORLD_VERT = """
#version 330
in vec2 in_position;
in vec2 in_uv;
out vec2 uv;
uniform mat4 u_mvp;
void main() {
    uv = in_uv;
    gl_Position = u_mvp * vec4(in_position, 0.0, 1.0);
}
"""

# World-space overlay fragment shader (plain RGBA texture, no parallax)
# u_alpha scales the output alpha; defaults to 1.0 (fully opaque per texture).
_OVERLAY_FRAG = """
#version 330
uniform sampler2D tex;
uniform float u_alpha;
in vec2 uv;
out vec4 fragColor;
void main() {
    vec4 c = texture(tex, uv);
    fragColor = vec4(c.rgb, c.a * u_alpha);
}
"""

# Solid-color vertex shader (no UV — avoids GLSL optimizer stripping in_uv)
_SOLID_VERT = """
#version 330
in vec2 in_position;
uniform mat4 u_mvp;
void main() {
    gl_Position = u_mvp * vec4(in_position, 0.0, 1.0);
}
"""

# Solid-color fragment shader for the screen border quad
_SOLID_FRAG = """
#version 330
uniform vec4 u_color;
out vec4 fragColor;
void main() { fragColor = u_color; }
"""

# 3D vertex shader for tapered rainbow beam
_BEAM_VERT = """
#version 330
in vec3 in_position;
in float in_v;
out float v_v;
uniform mat4 u_mvp;
void main() {
    v_v = in_v;
    gl_Position = u_mvp * vec4(in_position, 1.0);
}
"""

_BEAM_FRAG = """
#version 330
in float v_v;
out vec4 fragColor;
uniform float u_time;
void main() {
    // Rainbow gradient: blue→cyan→green→yellow→red, flowing from root to tip
    float t = fract(v_v + u_time * 0.4);
    vec3 col;
    if (t < 0.167)      col = mix(vec3(0.0,0.4,1.0), vec3(0.0,1.0,1.0), t/0.167);
    else if (t < 0.333) col = mix(vec3(0.0,1.0,1.0), vec3(0.0,1.0,0.0), (t-0.167)/0.166);
    else if (t < 0.5)   col = mix(vec3(0.0,1.0,0.0), vec3(1.0,1.0,0.0), (t-0.333)/0.167);
    else if (t < 0.667) col = mix(vec3(1.0,1.0,0.0), vec3(1.0,0.5,0.0), (t-0.5)/0.167);
    else if (t < 0.833) col = mix(vec3(1.0,0.5,0.0), vec3(1.0,0.0,0.0), (t-0.667)/0.166);
    else                col = mix(vec3(1.0,0.0,0.0), vec3(0.0,0.4,1.0), (t-0.833)/0.167);
    fragColor = vec4(col, 1.0);
}
"""

# Background color presets: (r, g, b) in linear [0,1].  Index 0 = opaque black (default).
_BG_COLORS = [
    (0.000, 0.000, 0.000),   # default — opaque black
    (0.827, 0.827, 0.827),   # light grey
    (0.196, 0.196, 0.216),   # charcoal
    (0.047, 0.071, 0.149),   # dark navy
    (0.937, 0.886, 0.820),   # warm beige
]

# Curved-screen vertex shader: in_position is a world-space vec3 arc point (no model matrix).
# UV is passed through normally.  vp_mat is the combined view-projection for the current eye.
_CURVED_VERT = """
#version 330
in vec3 in_position;
in vec2 in_uv;
out vec2 uv;
uniform mat4 u_mvp;
void main() {
    uv = in_uv;
    gl_Position = u_mvp * vec4(in_position, 1.0);
}
"""

# VR controller model shader (improved: supports Blinn-Phong lighting and texture toggle)
_CTRL_VERT = """
#version 330
in vec3 in_position;
in vec3 in_normal;   // Corresponds to 12 skipped bytes in the data
in vec2 in_uv;
out vec2 v_uv;
out vec3 v_normal;
out vec3 v_position;
uniform mat4 u_mvp;
uniform mat4 u_model; // Used for normal transformation
void main() {
    v_uv = in_uv;
    v_normal = mat3(transpose(inverse(u_model))) * in_normal; // Normal transformation
    vec4 world_pos = u_model * vec4(in_position, 1.0);
    v_position = world_pos.xyz;
    gl_Position = u_mvp * world_pos;
}
"""

_CTRL_FRAG = """
#version 330
in vec2 v_uv;
in vec3 v_normal;
in vec3 v_position;
out vec4 fragColor;

uniform sampler2D u_tex;
uniform vec3 u_light_color;    // Light source color
uniform vec3 u_ambient_color;  // Ambient light color
uniform vec3 u_base_color_factor; // Base color factor
uniform int u_use_texture;     // 0: use solid color, 1: sample texture
uniform vec3 u_camera_pos;     // Camera world coordinates (= headset position)

void main() {
    // Discard back faces (inner walls), keep only front faces (outer shell)
    if (!gl_FrontFacing) discard;

    vec3 N = normalize(v_normal);
    vec3 light_pos = u_camera_pos + vec3(0.0, 0.05, 0.0);
    vec3 L = normalize(light_pos - v_position);
    vec3 V = normalize(u_camera_pos - v_position);
    vec3 H = normalize(L + V);

    vec3 baseColor;
    if (u_use_texture == 1) {
        baseColor = texture(u_tex, v_uv).rgb * u_base_color_factor;
    } else {
        baseColor = u_base_color_factor;
    }

    float diff = abs(dot(N, L));
    vec3 diffuse = u_light_color * diff;
    vec3 ambient = u_ambient_color;
    float spec = pow(max(dot(N, H), 0.0), 32.0);
    vec3 specular = u_light_color * spec;

    fragColor = vec4((ambient + diffuse + specular) * baseColor, 1.0);
}
"""



# ---------------------------------------------------------------------------
# Windows input helpers (no-op on non-Windows)
# ---------------------------------------------------------------------------

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
        # Use SendInput with MOVE+ABSOLUTE so apps receive WM_MOUSEMOVE/WM_INPUT
        # events while a button is held — SetCursorPos alone is invisible to apps
        # that rely on raw input for drag tracking.
        sw = _U32.GetSystemMetrics(0)
        sh = _U32.GetSystemMetrics(1)
        inp = _INPUT(type=0)
        inp.mi.dwFlags = _MOUSEEVENTF_MOVE | _MOUSEEVENTF_ABSOLUTE
        inp.mi.dx = ctypes.c_long(max(0, min(65535, int(x) * 65535 // max(sw - 1, 1))))
        inp.mi.dy = ctypes.c_long(max(0, min(65535, int(y) * 65535 // max(sh - 1, 1))))
        ctypes.windll.user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(inp))

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

# Controller thumbstick dead-zone
DEAD = 0.15


# ---------------------------------------------------------------------------
# Virtual keyboard layout
# ---------------------------------------------------------------------------

# (label, normal_vk, _, shifted_vk, width_units)
# VK codes: https://docs.microsoft.com/en-us/windows/win32/inputdev/virtual-key-codes
# vk == -1 marks a layout gap: the slot consumes width but renders nothing and
# generates no keystroke (used to align the navigation/arrow clusters).
_KB_UNITS_WIDE = 18   # total horizontal units per row
_KB_ROWS = [
    # Row F: Esc + F1–F12 + PrtSc/ScrLk/Pause   (1.5 + 12 + 4.5 = 18)
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
    # 2.25 + 10 + 2.75 = 15.0   then  gap(1) | ↑(1) | gap(1)  →  ↑ occupies col 16-17 (same as ↓)
    [('Shift',0x10,None,0x10,2.25),('Z',0x5A,None,0x5A,1),('X',0x58,None,0x58,1),
     ('C',0x43,None,0x43,1),('V',0x56,None,0x56,1),('B',0x42,None,0x42,1),
     ('N',0x4E,None,0x4E,1),('M',0x4D,None,0x4D,1),(',',0xBC,'<',0xBC,1),
     ('.',0xBE,'>',0xBE,1),('/',0xBF,'?',0xBF,1),('Shift',0x10,None,0x10,2.75),
     ('',-1,None,-1,1),('↑',0x26,None,0x26,1),('',-1,None,-1,1)],
    # Row 4: bottom + arrow cluster — Down sits directly under Up at col 16-17.
    # 1.5+1+1.25+7.5+1.25+1+1.5 = 15.0   then  ←(1) | ↓(1) | →(1)
    [('Ctrl',0x11,None,0x11,1.5),('Win',0x5B,None,0x5B,1),
     ('Alt',0x12,None,0x12,1.25),
     ('Space',0x20,None,0x20,7.5),
     ('Alt',0x12,None,0x12,1.25),('Apps',0x5D,None,0x5D,1),
     ('Ctrl',0x11,None,0x11,1.5),
     ('←',0x25,None,0x25,1),('↓',0x28,None,0x28,1),('→',0x27,None,0x27,1)],
]

import collections as _collections
_KeyEntry = _collections.namedtuple('_KeyEntry', 'label shifted_label vk shifted_vk rect_uv rect_local')

_KB_TEX_W, _KB_TEX_H = 1280, 384   # keyboard texture: 6 rows × 18 units


# ---------------------------------------------------------------------------
# XR math helpers (module-level, pure functions)
# ---------------------------------------------------------------------------

def _xr_quat_to_mat4(q):
    """XrQuaternionf → standard 4×4 rotation matrix (numpy, math row/col convention).

    Produces the matrix that left-multiplies a column vector: v' = R @ v.
    Callers must transpose before writing to OpenGL (which reads column-major).
    """
    x, y, z, w = q.x, q.y, q.z, q.w
    return np.array([
        [1-2*(y*y+z*z),   2*(x*y-w*z),   2*(x*z+w*y),  0],
        [  2*(x*y+w*z), 1-2*(x*x+z*z),   2*(y*z-w*x),  0],
        [  2*(x*z-w*y),   2*(y*z+w*x), 1-2*(x*x+y*y),  0],
        [  0,             0,             0,              1],
    ], dtype=np.float32)


def _pose_to_view_mat4(pose):
    """XrPosef → standard 4×4 view matrix (numpy, math row/col convention).

    The view matrix is the inverse of the head-pose model matrix:
      V = [ R^T | -R^T @ pos ]
          [  0  |      1     ]
    Caller must transpose before writing to OpenGL.
    """
    R  = _xr_quat_to_mat4(pose.orientation)[:3, :3]
    Rt = R.T                                              # inverse rotation
    t  = np.array([pose.position.x, pose.position.y, pose.position.z], dtype=np.float32)
    V  = np.eye(4, dtype=np.float32)
    V[:3, :3] = Rt
    V[:3, 3]  = -Rt @ t                                  # translation in last column
    return V


def _fov_to_proj_mat4(fov, near=0.05, far=100.0):
    """XrFovf → standard 4×4 OpenGL asymmetric-frustum projection matrix
    (numpy, math row/col convention). Caller must transpose before writing to OpenGL.

    Includes a small epsilon offset to prevent division by zero when the
    headset runtime reports a degenerate FOV (e.g., left == right).
    """
    l = math.tan(fov.angle_left)  * near
    r = math.tan(fov.angle_right) * near
    t = math.tan(fov.angle_up)    * near
    b = math.tan(fov.angle_down)  * near

    # Prevent ZeroDivisionError when headset reports identical left/right or up/down angles.
    EPS = 1e-6
    if abs(r - l) < EPS:
        r += EPS
    if abs(t - b) < EPS:
        t += EPS

    p = np.zeros((4, 4), dtype=np.float32)
    p[0, 0] =  2 * near / (r - l)
    p[0, 2] =  (r + l)  / (r - l)      # col 2 of row 0
    p[1, 1] =  2 * near / (t - b)
    p[1, 2] =  (t + b)  / (t - b)      # col 2 of row 1
    p[2, 2] = -(far + near) / (far - near)
    p[2, 3] = -2 * far * near / (far - near)  # translation in last column
    p[3, 2] = -1.0                      # w = -z (perspective divide)
    return p


class OpenXRViewer:
    """
    Renders the depth-parallax stereo views into a VR headset using OpenXR.

    A virtual flat screen is placed in world space at `screen_distance` meters.
    Head pose from xr.locate_views() provides proper 6DoF view matrices per eye.
    The screen can be repositioned/resized/rotated via keyboard or VR controller
    thumbsticks. An in-VR FPS/latency overlay quad sits just below the screen and
    is toggled with the menu button on the left controller.

    Parameters mirror the relevant subset of StereoWindow.__init__.
    Call run(first_rgb, first_depth) to enter the blocking frame loop.
    """

    def __init__(
        self,
        ipd=0.064,
        depth_ratio=1.0,
        convergence=0.0,
        frame_size=(1280, 720),
        fps=60,
        depth_q=None,
        show_fps=True,
        **kwargs,
    ):
        self.ipd_uv = ipd
        self.depth_strength = 0.1 # multiplied by depth_ratio; effective = depth_strength * depth_ratio
        self.depth_ratio = depth_ratio
        self.convergence = convergence
        self.frame_size = frame_size
        self.fps = fps
        self.depth_q = depth_q
        self.show_fps = show_fps

        # FPS display — timestamp ring: (len-1)/(last-first) is exact over the window
        self.actual_fps      = 0.0   # XR composition rate (this loop)
        self.sbs_fps         = 0.0   # SBS source rate (depth_q producer in main.py)
        self.total_latency   = 0.0
        self._frame_ts_ring  = collections.deque(maxlen=60)  # ~1 s at 60 Hz
        self._sbs_ts_ring    = collections.deque(maxlen=60)  # SBS frame arrivals

        # Overlay redraw throttle — texture is rebuilt at most once per second
        self._last_overlay_update   = 0.0
        self._cached_actual_fps     = 0.0
        self._cached_sbs_fps        = 0.0
        self._cached_latency        = 0.0
        self._cached_screen_width   = 0.0
        self._cached_screen_height  = 0.0
        self._cached_screen_dist    = 0.0
        self._cached_screen_curved  = False
        self._cached_depth_ratio    = 1.0
        self._cached_vr_res         = (0, 0)
        self._cached_sbs_res        = (0, 0)

        # Virtual screen transform (world space, metres / radians)
        self.screen_distance = 2.0
        self.screen_width    = 2.4
        self.screen_height   = None   # derived from frame aspect ratio on first frame

        # screen presets: (name, width_m, distance_m) — height is derived from width and frame aspect ratio
        self._screen_presets = [
            ("10-inch Tablet",        0.30, 0.4),
            ("27-inch Monitor",       0.60, 0.6),
            ("65-inch TV",            1.44, 2.0),
            ("100-inch Projector 1",  2.40, 2.0),
            ("100-inch Projector 2",  2.21, 2.5),
            ("1000-inch IMAX",       22.0,  20),
        ]
        self._preset_index = 3  # default 4: 100-inch projector with 2.0 m distance
        self._preset_name_overlay = None   # preset name overlay (permanently visible)
        self.screen_pan_x    = 0.0
        self.screen_pan_y    = 0.0
        self.screen_yaw      = 0.0    # rotation around Y axis
        self.screen_pitch    = 0.0    # rotation around X axis

        # Interaction speeds (per second)
        self._dist_speed  = 0.5
        self._size_speed  = 0.5
        self._pan_speed   = 0.5
        self._rot_speed   = 0.5
        self._pitch_speed = 0.5

        # OpenXR handles
        self._xr_instance = None
        self._xr_system_id = None
        self._xr_session = None
        self._xr_space = None
        self._xr_swapchains = {}        # {eye_index: xr.Swapchain}
        self._swapchain_images = {}     # {eye_index: [XrSwapchainImageOpenGLKHR, ...]}
        self._swapchain_sizes = {}      # {eye_index: (w, h)}
        self._fbo_cache = {}            # {(eye_index, image_index): (raw_id, mgl_fbo)}
        self._session_running = False

        # Controller action handles (set by _init_controller_actions)
        self._action_set          = None
        self._act_left_stick      = None
        self._act_right_stick     = None
        self._act_menu_btn        = None
        self._act_left_grip       = None   # grab/move mode
        self._act_right_grip      = None   # resize mode
        self._act_a_btn           = None   # right A — double press = hide all
        self._act_b_btn           = None   # right B — right mouse click
        self._act_x_btn           = None   # left  X — toggle virtual keyboard
        self._act_y_btn           = None   # left  Y — reset screen position/size/rotation
        self._act_left_trigger       = None   # left trigger (float) — left mouse click
        self._act_right_trigger      = None   # right trigger (float) — right mouse click / hold
        self._act_left_stick_click   = None   # left thumbstick click — cycle background color
        self._act_right_stick_click  = None   # right thumbstick click — toggle curved screen

        # Menu button debounce + FPS overlay toggle + long-press reset
        self._menu_pressed_last   = False
        self._fps_overlay_visible = show_fps
        self._menu_press_t        = 0.0    # perf_counter when menu was pressed
        self._menu_long_fired     = False  # True once long-press action has fired

        # Quest-like window state
        self._screen_visible = True   # hide-all toggle (A double-press)
        self._grabbed        = False  # left grip held  → move mode
        self._resizing       = False  # right grip held → resize mode
        self._a_last         = False  # A-button previous frame state
        self._a_last_t       = 0.0   # timestamp of last A press (double-press detection)
        self._b_last              = False  # B-button previous frame state
        self._ltrig_state   = 'idle'  # 'idle' | 'pressed' | 'dragging'
        self._rtrig_state   = 'idle'
        self._ltrig_press_t = 0.0    # perf_counter of last rising edge — left
        self._rtrig_press_t = 0.0    # perf_counter of last rising edge — right
        self._y_last         = False  # Y-button previous frame state (reset screen)
        self._y_press_t      = 0.0   # perf_counter when Y was pressed
        self._y_long_fired   = False # True once long-press action fired this hold
        self._x_last         = False  # X-button previous frame state (toggle keyboard)
        # Head pose (world) cached each frame from xr.locate_views — used as the orbit
        # pivot for the left thumbstick and as the anchor when the keyboard is summoned.
        self._head_pos_w      = None   # (x, y, z) head/eye centre in world space, or None
        self._head_fwd_w      = None   # (fx, fy, fz) head forward unit vector, or None
        self._screen_eye_init = False  # screen_pan_y aligned to headset height on first frame
        self._initial_head_y  = 0.0   # headset eye height at session start, used for Y-reset
        # Border fade: shown during interaction, fades out when idle
        self._border_alpha   = 0.0    # 0.0 = invisible, 1.0 = fully opaque
        self._border_idle_t  = 0.0    # wall time when interaction last ended
        # Keyboard border fade
        self._kb_border_alpha  = 0.0
        self._kb_border_idle_t = 0.0
        self._saved_dclick_time = None  # system double-click time saved before session

        # Mouse cursor control
        self._cursor_uv_l         = None  # (u,v,t) where left laser hits screen, or None
        self._cursor_uv_r         = None  # (u,v,t) where right laser hits screen, or None
        self._cursor_ctrl         = None  # 'left' | 'right' | None — active cursor controller
        # Smoothed UV — exponential moving average tames hand tremor so the cursor
        # doesn't jitter or skip pixels at long laser distances. Reset when the active
        # controller changes so we don't drag the cursor across the screen on swap.
        self._cursor_smooth_uv    = None
        self._left_btn_down       = False # left mouse button held via left trigger

        # Virtual keyboard
        self._keyboard_visible     = False
        self._keyboard_tex         = None  # moderngl Texture (RGBA, _KB_TEX_W × _KB_TEX_H)
        self._keyboard_vao         = None  # quad VAO using _overlay_prog
        self._keyboard_keys        = []    # list of _KeyEntry
        self._keyboard_width       = 1.6   # metres
        self._keyboard_height      = 0.33  # metres (6 rows)
        self._kb_show_shifted      = False # True → render shifted labels on keys
        self._kb_last_build_width  = 0.0   # track width changes for texture rebuild
        # World-space anchor of the keyboard centre. Re-snapped to the user's current
        # head pose every time the keyboard is toggled on so it materialises within
        # easy reach below the user's gaze direction (not at world origin).
        self._keyboard_pan_x       = 0.0
        self._keyboard_pan_y       = -0.2
        self._keyboard_distance    = 0.7   # metres in front of head
        self._keyboard_pitch       = math.radians(-35.0)   # tilt face up toward user
        self._keyboard_yaw         = 0.0                   # face the user's forward
        # Modifier state: each entry is [active, lock, last_tap_time]. Single tap arms
        # one-shot; double tap (<0.4s) engages persistent lock; tap while locked releases.
        self._mod_state = {
            'shift': [False, False, 0.0],
            'ctrl':  [False, False, 0.0],
            'alt':   [False, False, 0.0],
            'win':   [False, False, 0.0],
        }
        self._caps_lock            = False
        self._left_stick_click_prev= False
        self._scroll_accum_x       = 0.0   # fractional scroll accumulator — horizontal
        self._scroll_accum_y       = 0.0   # fractional scroll accumulator — vertical
        self._kb_trig_prev_l       = 0.0   # keyboard trigger debounce — left controller
        self._kb_trig_prev_r       = 0.0   # keyboard trigger debounce — right controller
        self._kb_hover_l           = None  # index of key under left laser, or None
        self._kb_hover_r           = None  # index of key under right laser, or None
        self._kb_held_key_l        = None  # index of key held by left trigger, or None
        self._kb_held_key_r        = None  # index of key held by right trigger, or None
        self._kb_held_mods_l       = None  # (shift, ctrl, alt, win, vk) snapshot for left held key
        self._kb_held_mods_r       = None  # (shift, ctrl, alt, win, vk) snapshot for right held key

        # GPU interop (CUDA / HIP) — initialised lazily on first frame
        self._cuda_gl         = None   # CUDART_GL instance, False = permanently failed
        self._pbo_color       = None   # GL PBO id for RGB upload
        self._pbo_depth       = None   # GL PBO id for depth upload
        self._cuda_res_color  = None   # registered resource handle
        self._cuda_res_depth  = None
        self._pbo_texture_size = None  # (w, h) at which PBOs were created

        # Font for in-VR overlay
        self.font = None
        self.label_font = None   # smaller font for section header labels
        self.font_type = get_font_type()
        self.base_font_size = 26
        try:
            self.font = ImageFont.truetype(self.font_type, self.base_font_size)
        except Exception:
            try:
                self.font = ImageFont.load_default()
            except Exception:
                self.font = None
        try:
            self.label_font = ImageFont.truetype(self.font_type, 17)
        except Exception:
            self.label_font = self.font   # fall back to main font
        self.bold_font = None
        for _bf in (r"C:\Windows\Fonts\segoeuib.ttf",
                    r"C:\Windows\Fonts\arialbd.ttf",
                    r"C:\Windows\Fonts\calibrib.ttf"):
            try:
                self.bold_font = ImageFont.truetype(_bf, self.base_font_size)
                break
            except Exception:
                continue
        if self.bold_font is None:
            self.bold_font = self.font   # fall back to regular

        # In-VR FPS overlay GL resources
        self._overlay_prog     = None
        self._overlay_vao      = None
        self._overlay_tex      = None
        self._overlay_tex_size = (768, 150)  # 3-row info panel

        # Depth-ratio OSD: floating panel that appears when depth_ratio changes
        self._depth_osd_tex       = None   # moderngl Texture (RGBA, 256×64)
        self._depth_osd_vao       = None   # quad VAO (reuses _overlay_prog)
        self._depth_osd_tex_size  = (256, 78)
        self._depth_osd_alpha     = 0.0    # current alpha (1 = fully visible, 0 = hidden)
        self._depth_osd_show_t    = -999.0 # perf_counter when OSD was last triggered
        self._depth_osd_last_val  = None   # last depth_ratio value rendered into the texture

        # Screen-info OSD: shows size + distance while right grip + right stick adjusts
        self._screen_osd_tex      = None   # moderngl Texture (RGBA, 512×64)
        self._screen_osd_vao      = None
        self._screen_osd_tex_size = (512, 78)
        self._screen_osd_show_t   = -999.0
        self._screen_osd_last_key = None   # (round(w,2), round(dist,2)) — change detection

        # Screen border (slightly larger quad, solid color)
        self._border_prog = None
        self._border_vao  = None

        # Right thumbstick click state
        self._right_stick_click_prev = False

        # VR controller model offsets
        self._y_offset = -0.02       # Y offset (down 2cm)
        self._z_offset = -0.03       # Z offset
        self._x_offset = 0.0         # X offset (left/right)

        # Right grip + A/B → depth_ratio control (hold A = increase, hold B = decrease)
        # Reset to default: right grip + right thumbstick click

        # Physical mouse priority: suppress VR cursor when physical mouse is active
        self._phys_mouse_pos        = None   # (x, y) last seen physical cursor position
        self._phys_mouse_last_move  = 0.0    # perf_counter when physical mouse last moved
        self._vr_cursor_screen_pos  = None   # (px, py) last position written by VR laser

        # Curved screen mode
        self._screen_curved   = False   # True = cylindrical arc; False = flat quad
        self._curved_prog     = None    # shader program (uses _CURVED_VERT)
        self._curved_vbo      = None    # dynamic VBO for arc strip vertices
        self._curved_vao      = None
        self._curved_border_prog = None  # solid-color arc border program
        self._curved_border_vbo  = None  # dynamic VBO for border arc vertices
        self._curved_border_vao  = None
        self._curved_verts_params = None        # curved screen VBO cache dirty flag
        self._curved_border_verts_params = None # border VBO cache dirty flag

        # Background color cycling (left thumbstick click)
        self._bg_color_idx    = 0       # index into _BG_COLORS

        # Cached XrPath handles — populated by _init_controller_actions to avoid
        # calling xr.string_to_path on every frame (it's a round-trip into the runtime).
        self._path_left  = None
        self._path_right = None

        # Controller aim poses + laser pointer rendering
        self._act_aim_left  = None   # XrAction POSE_INPUT for left aim
        self._act_aim_right = None   # XrAction POSE_INPUT for right aim
        self._aim_space_l   = None   # XrSpace for left aim
        self._aim_space_r   = None   # XrSpace for right aim
        self._laser_vao     = None   # thin quad for laser beam
        self._dot_vao       = None   # small square for controller origin dot
        self._circle_vao    = None   # tessellated circle for hit-point indicator
        # Cached aim poses updated each frame (numpy 4x4 view-space matrices)
        self._aim_mat_l     = None
        self._aim_mat_r     = None

        # Controller grip poses + 3D model rendering
        self._act_grip_left  = None   # XrAction POSE_INPUT for left grip
        self._act_grip_right = None   # XrAction POSE_INPUT for right grip
        self._grip_space_l   = None   # XrSpace for left grip
        self._grip_space_r   = None   # XrSpace for right grip
        self._grip_mat_l     = None   # 4x4 world-space matrix
        self._grip_mat_r     = None   # 4x4 world-space matrix
        self._controller_prog   = None   # textured shader for controller
        self._ctrl_prims_l      = []     # list of {vao, vbo, ibo, tex_id} for left
        self._ctrl_prims_r      = []     # list of {vao, vbo, ibo, tex_id} for right
        self._ctrl_tex_cache    = {}     # tex_id -> moderngl Texture

        # Laser auto-hide: track last movement time and previous pose per controller
        _now = time.perf_counter()
        self._laser_last_move_l  = _now
        self._laser_last_move_r  = _now
        self._laser_prev_mat_l   = None
        self._laser_prev_mat_r   = None
        self._LASER_HIDE_AFTER   = 5.0   # seconds of idle before hiding
        self._LASER_MOVE_THRESH  = 0.015 # metres or radians — minimum motion to count

        # Screen position animation — used by home-button gaze-reset to glide
        # smoothly instead of snapping.  None = no animation in progress.
        self._anim_target_pan_x    = None   # target screen_pan_x
        self._anim_target_pan_y    = None   # target screen_pan_y
        self._anim_target_distance = None   # target screen_distance
        self._anim_target_yaw      = None   # target screen_yaw
        self._anim_target_pitch    = None   # target screen_pitch

        # D3D11 backend state (populated by _init_d3d11_device when D3D11 path is active)
        self._use_d3d11             = False   # True = D3D11 OpenXR session + readback path
        self._d3d11_device          = None    # c_void_p ID3D11Device*
        self._d3d11_context         = None    # c_void_p ID3D11DeviceContext*
        self._d3d11_swapchain_fmt   = _DXGI_FORMAT_R8G8B8A8_UNORM_SRGB
        self._swapchain_is_bgra     = False  # True when WMR runtime only offers BGRA
        # Offscreen GL FBOs used when rendering for D3D11 swapchain images.
        # Key: (eye_index, img_index) → (mgl_fbo, mgl_tex, raw_fbo_id, w, h)
        self._offscreen_fbo_cache   = {}
        # PBOs for async pixel readback in the D3D11 path.
        # Key: (eye_index, img_index) → (pbo_id, w, h)
        self._d3d11_pbo_cache       = {}

        # GPU interop state (NV_DX_interop2 or EXT_memory_object) for zero-copy
        self._interop_mode      = None   # 'nv_dx' | 'ext_mem' | None (PBO fallback)
        self._nv_dx_device      = None   # HANDLE from wglDXOpenDeviceNV
        self._nv_dx_objects     = {}     # {img_index: GL_tex_id} for registered swapchain textures
        self._ext_shared_tex    = {}     # {(eye): (d3d11_tex, gl_mem_obj, gl_tex, mgl_fbo)}

        # ModernGL / GL handles
        self.window = None
        self.ctx = None
        self.prog = None
        self.quad_vao = None
        self.color_tex = None
        self.depth_tex = None
        self._texture_size = None

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _init_glfw(self):
        if not glfw.init():
            raise RuntimeError("[OpenXRViewer] GLFW init failed")
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 5)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.VISIBLE, glfw.FALSE)   # hidden — GL context only
        glfw.window_hint(glfw.RESIZABLE, glfw.FALSE)
        self.window = glfw.create_window(1, 1, "D2S-XR", None, None)
        if not self.window:
            glfw.terminate()
            raise RuntimeError("[OpenXRViewer] GLFW window creation failed")
        glfw.make_context_current(self.window)
        glfw.swap_interval(0)

        # Keyboard controls — keep a reference so it isn't GC'd
        self._key_callback_ref = self._make_key_callback()
        glfw.set_key_callback(self.window, self._key_callback_ref)

    def _make_key_callback(self):
        viewer = self
        def _cb(window, key, scancode, action, mods):
            if action not in (glfw.PRESS, glfw.REPEAT):
                return
            d = 0.1; s = 0.15; p = 0.1; r = 0.05
            if   key == glfw.KEY_W:     viewer.screen_distance = max(0.3, viewer.screen_distance - d)
            elif key == glfw.KEY_S:     viewer.screen_distance += d
            elif key == glfw.KEY_UP:    viewer.screen_pan_y += p
            elif key == glfw.KEY_DOWN:  viewer.screen_pan_y -= p
            elif key == glfw.KEY_LEFT:  viewer.screen_pan_x -= p
            elif key == glfw.KEY_RIGHT: viewer.screen_pan_x += p
            elif key in (glfw.KEY_EQUAL, glfw.KEY_KP_ADD):
                viewer.screen_width += s; viewer.screen_height = None
            elif key in (glfw.KEY_MINUS, glfw.KEY_KP_SUBTRACT):
                viewer.screen_width = max(0.3, viewer.screen_width - s)
                viewer.screen_height = None
            elif key == glfw.KEY_Q: viewer.screen_yaw += r
            elif key == glfw.KEY_E: viewer.screen_yaw -= r
            elif key == glfw.KEY_T: viewer.screen_pitch += r
            elif key == glfw.KEY_G: viewer.screen_pitch -= r
            elif key == glfw.KEY_F: viewer._fps_overlay_visible = not viewer._fps_overlay_visible
            elif key == glfw.KEY_Z:
                viewer.depth_strength = max(0.0, viewer.depth_strength - 0.01)
            elif key == glfw.KEY_C:
                viewer.depth_strength = min(0.5, viewer.depth_strength + 0.01)
            elif key == glfw.KEY_X:
                viewer.depth_strength = 0.0   # flat mode — no parallax distortion
            elif key == glfw.KEY_R:
                viewer.screen_distance = 2.0; viewer.screen_pan_x = 0.0
                viewer.screen_pan_y = 0.0;    viewer.screen_yaw = 0.0
                viewer.screen_pitch = 0.0;    viewer.screen_width = 2.4
                viewer.screen_height = None
        return _cb

    def _init_moderngl(self):
        self.ctx = moderngl.create_context()

        # World-space stereo rendering program (HMD eyes)
        self.prog = self.ctx.program(
            vertex_shader=_WORLD_VERT,
            fragment_shader=FRAGMENT_SHADER,
        )
        self.prog['u_convergence'].value = self.convergence
        self.prog['tex_color'].value = 0
        self.prog['tex_depth'].value = 1

        vertices = np.array([
            -1, -1, 0, 0,
             1, -1, 1, 0,
            -1,  1, 0, 1,
             1,  1, 1, 1,
        ], dtype='f4')
        vbo = self.ctx.buffer(vertices)
        self.quad_vao = self.ctx.vertex_array(
            self.prog, [(vbo, '2f 2f', 'in_position', 'in_uv')]
        )

        # Screen border (solid-color quad rendered before the main screen)
        self._border_prog = self.ctx.program(
            vertex_shader=_SOLID_VERT,
            fragment_shader=_SOLID_FRAG,
        )
        vbo_border = self.ctx.buffer(vertices.tobytes())
        self._border_vao = self.ctx.vertex_array(
            self._border_prog, [(vbo_border, '2f 8x', 'in_position')]
        )

        # Laser beam: a very thin elongated quad (width=0.003 m, length=5 m)
        # in local space X=[-0.5,0.5], Y=[-1,1]; we scale X to beam_w, Y to half-length
        laser_verts = np.array([
            -1, -1, 0, 0,
             1, -1, 1, 0,
            -1,  1, 0, 1,
             1,  1, 1, 1,
        ], dtype='f4')
        self._laser_vao = self.ctx.vertex_array(
            self._border_prog,
            [(self.ctx.buffer(laser_verts.tobytes()), '2f 8x', 'in_position')],
        )
        # Flat beam (quadrilateral) + rainbow flow animation
        self._beam_prog = self.ctx.program(
            vertex_shader=_BEAM_VERT,
            fragment_shader=_BEAM_FRAG,
        )
        # Single quadrilateral: Y=0(base, thick) → Y=1(tip, thin)
        beam_verts = np.array([
            -1.0, 0.0, 0.0, 0.0,   # bottom-left, v=0
             1.0, 0.0, 0.0, 0.0,   # bottom-right, v=0
            -0.15, 1.0, 0.0, 1.0,   # top-left, v=1
             0.15, 1.0, 0.0, 1.0,   # top-right, v=1
        ], dtype='f4')
        beam_vbo = self.ctx.buffer(beam_verts.tobytes())
        self._beam_vao = self.ctx.vertex_array(
            self._beam_prog,
            [(beam_vbo, '3f 4x 1f', 'in_position', 'in_v')],
        )
        # Hit-point indicator: tessellated circle (TRIANGLE_FAN), blue stroke + white fill
        N_SEG = 32
        circle_data = [0.0, 0.0, 0.0, 0.0]  # centre vertex
        for i in range(N_SEG + 1):
            a = 2.0 * math.pi * i / N_SEG
            circle_data.extend([math.cos(a), math.sin(a), 0.0, 0.0])
        self._circle_vao = self.ctx.vertex_array(
            self._border_prog,
            [(self.ctx.buffer(np.array(circle_data, dtype='f4').tobytes()), '2f 8x', 'in_position')],
        )
        # Controller origin dot: tiny square at the controller position
        self._dot_vao = self.ctx.vertex_array(
            self._border_prog,
            [(self.ctx.buffer(laser_verts.tobytes()), '2f 8x', 'in_position')],
        )

        # In-VR FPS overlay program (world-space quad, plain RGBA blit)
        self._overlay_prog = self.ctx.program(
            vertex_shader=_WORLD_VERT,
            fragment_shader=_OVERLAY_FRAG,
        )
        self._overlay_prog['tex'].value     = 2   # texture unit 2
        self._overlay_prog['u_alpha'].value = 1.0
        ow, oh = self._overlay_tex_size
        self._overlay_tex = self.ctx.texture((ow, oh), 4, dtype='f1')
        self._overlay_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        vbo2 = self.ctx.buffer(vertices.tobytes())
        self._overlay_vao = self.ctx.vertex_array(
            self._overlay_prog, [(vbo2, '2f 2f', 'in_position', 'in_uv')]
        )

        # Depth OSD: small floating panel (reuses _overlay_prog)
        dw, dh = self._depth_osd_tex_size
        self._depth_osd_tex = self.ctx.texture((dw, dh), 4, dtype='f1')
        self._depth_osd_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        vbo_dosd = self.ctx.buffer(vertices.tobytes())
        self._depth_osd_vao = self.ctx.vertex_array(
            self._overlay_prog, [(vbo_dosd, '2f 2f', 'in_position', 'in_uv')]
        )

        # Screen-info OSD: size + distance panel (reuses _overlay_prog)
        sw, sh = self._screen_osd_tex_size
        self._screen_osd_tex = self.ctx.texture((sw, sh), 4, dtype='f1')
        self._screen_osd_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        vbo_sosd = self.ctx.buffer(vertices.tobytes())
        self._screen_osd_vao = self.ctx.vertex_array(
            self._overlay_prog, [(vbo_sosd, '2f 2f', 'in_position', 'in_uv')]
        )

        # Curved screen program: same fragment shader, but world-space arc geometry
        # (no model matrix — verts are built directly in world space each frame).
        self._curved_prog = self.ctx.program(
            vertex_shader=_CURVED_VERT,
            fragment_shader=FRAGMENT_SHADER,
        )
        self._curved_prog['u_convergence'].value = self.convergence
        self._curved_prog['tex_color'].value = 0
        self._curved_prog['tex_depth'].value  = 1
        # Allocate dynamic VBO large enough for N=48 segments × 2 verts × (3+2) floats.
        _CURVED_N = 48
        _curved_buf_bytes = (_CURVED_N + 1) * 2 * (3 + 2) * 4   # f4
        self._curved_vbo = self.ctx.buffer(reserve=_curved_buf_bytes, dynamic=True)
        self._curved_vao = self.ctx.vertex_array(
            self._curved_prog,
            [(self._curved_vbo, '3f 2f', 'in_position', 'in_uv')],
        )

        # Curved border: reuse _CURVED_VERT (vec3 pos + vec2 uv) with solid-color frag.
        # _SOLID_FRAG doesn't use UV, so the GLSL optimizer strips in_uv from the
        # linked program's active attribute table.  Only bind in_position (3f) and
        # skip the trailing 8 bytes of UV data — same pattern as the flat border VAO.
        self._curved_border_prog = self.ctx.program(
            vertex_shader=_CURVED_VERT,
            fragment_shader=_SOLID_FRAG,
        )
        self._curved_border_vbo = self.ctx.buffer(reserve=_curved_buf_bytes, dynamic=True)
        self._curved_border_vao = self.ctx.vertex_array(
            self._curved_border_prog,
            [(self._curved_border_vbo, '3f 8x', 'in_position')],
        )

        # VR controller 3D model loading
        self._controller_prog = self.ctx.program(
            vertex_shader=_CTRL_VERT,
            fragment_shader=_CTRL_FRAG,
        )
        self._controller_prog['u_tex'].value = 3
        self._controller_prog['u_use_texture'].value = 1
        self._controller_prog['u_base_color_factor'].value = (1.0, 1.0, 1.0)
        self._controller_prog['u_light_color'].value = (0.37, 0.37, 0.40)
        self._controller_prog['u_ambient_color'].value = (0.22, 0.22, 0.24)

        import os as _os
        _base = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                            'controllers', 'pico-4u')

        self._ctrl_tex_cache = {}
        self._ctrl_prims_l = []
        self._ctrl_prims_r = []

        def _create_prims(glb_path, target_list):
            prims_data, textures = load_glb_model(glb_path)
            for tid, tex_arr in enumerate(textures):
                if tex_arr is not None and tid not in self._ctrl_tex_cache:
                    h, w = tex_arr.shape[:2]
                    mtex = self.ctx.texture((w, h), 4, tex_arr.tobytes())
                    mtex.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
                    mtex.build_mipmaps()
                    self._ctrl_tex_cache[tid] = mtex
            for pd in prims_data:
                vbo = self.ctx.buffer(pd['vertices'].tobytes())
                ibo = self.ctx.buffer(pd['indices'].tobytes())
                vao = self.ctx.vertex_array(
                    self._controller_prog,
                    [(vbo, '3f 3f 2f', 'in_position', 'in_normal', 'in_uv')],
                    ibo,
                )
                target_list.append({
                    'vao': vao, 'vbo': vbo, 'ibo': ibo,
                    'tex_id': pd['tex_id'],
                    'tri_count': len(pd['indices']) // 3,
                })

        try:
            _create_prims(_os.path.join(_base, 'right.glb'), self._ctrl_prims_r)
            _create_prims(_os.path.join(_base, 'left.glb'),  self._ctrl_prims_l)
        except Exception as e:
            print(f"[OpenXRViewer] Controller model load failed: {e}")
            self._ctrl_prims_l = []
            self._ctrl_prims_r = []

    def _init_openxr(self):
        """Try OpenGL first; fall back to D3D11 on Windows if OpenGL fails."""
        try:
            self._init_openxr_opengl()
            return
        except Exception as e:
            if sys.platform != "win32":
                raise
            print(f"[OpenXRViewer] OpenGL init failed ({e}), falling back to D3D11")
            self._cleanup_partial_openxr()

        self._init_openxr_d3d11()
        self._use_d3d11 = True

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

    def _init_openxr_d3d11(self):
        """Create an OpenXR instance + session backed by a D3D11 device.

        Rendering still happens in ModernGL; each frame the completed eye texture
        is read back via glReadPixels and uploaded into the D3D11 swapchain image
        via UpdateSubresource.  This is a CPU-round-trip but avoids the need for
        NV_DX_interop or a full D3D11 rendering port.
        """
        # 1. Instance (D3D11 extension)
        app_info = xr.ApplicationInfo(
            application_name="Desktop2Stereo",
            application_version=1,
            engine_name="D2S",
            engine_version=1,
            api_version=xr.XR_CURRENT_API_VERSION,
        )
        create_info = xr.InstanceCreateInfo(
            application_info=app_info,
            enabled_extension_names=[xr.KHR_D3D11_ENABLE_EXTENSION_NAME],
        )
        self._xr_instance = xr.create_instance(create_info)
        print("[OpenXRViewer] XrInstance created (D3D11)")

        # 2. System
        self._xr_system_id = xr.get_system(
            self._xr_instance,
            xr.SystemGetInfo(form_factor=xr.FormFactor.HEAD_MOUNTED_DISPLAY),
        )

        # 3. Query D3D11 requirements (runtime mandates this call before session creation)
        _pfn = ctypes.cast(
            xr.get_instance_proc_addr(self._xr_instance, "xrGetD3D11GraphicsRequirementsKHR"),
            xr.PFN_xrGetD3D11GraphicsRequirementsKHR,
        )
        # Python 3.12 ctypes rejects int where a Structure field is expected.
        # pyopenxr's GraphicsRequirementsD3D11KHR.__init__ defaults adapter_luid=0
        # which triggers TypeError. Pass an explicit zeroed _LUID() instance instead.
        from xr.platform.windows import _LUID as _XrLUID
        _reqs = xr.GraphicsRequirementsD3D11KHR(adapter_luid=_XrLUID())
        xr.check_result(xr.Result(_pfn(self._xr_instance, self._xr_system_id, ctypes.byref(_reqs))))
        print(f"[OpenXRViewer] D3D11 min feature level: 0x{_reqs.min_feature_level:04x}")

        # 4. Create D3D11 device on the adapter the runtime requires
        device, context, feat = _create_d3d11_device(adapter_luid=_reqs.adapter_luid)
        self._d3d11_device  = device
        self._d3d11_context = context
        print(f"[OpenXRViewer] D3D11 device created (feature level 0x{feat:04x})")

        # 5. Graphics binding
        binding = xr.GraphicsBindingD3D11KHR(
            device=ctypes.cast(device, ctypes.POINTER(ctypes.c_int)),
        )

        # 6. Session
        session_info = xr.SessionCreateInfo(
            system_id=self._xr_system_id,
            next=ctypes.cast(ctypes.pointer(binding), ctypes.c_void_p),
        )
        self._xr_session = xr.create_session(self._xr_instance, session_info)
        print("[OpenXRViewer] XrSession created (D3D11)")

        # 7. Reference space
        available_spaces = xr.enumerate_reference_spaces(self._xr_session)
        ref_type = (
            xr.ReferenceSpaceType.STAGE
            if xr.ReferenceSpaceType.STAGE in available_spaces
            else xr.ReferenceSpaceType.LOCAL
        )
        self._xr_space = xr.create_reference_space(
            self._xr_session,
            xr.ReferenceSpaceCreateInfo(
                reference_space_type=ref_type,
                pose_in_reference_space=xr.Posef(),
            ),
        )

        # 8. Swapchains with DXGI format
        view_configs = xr.enumerate_view_configuration_views(
            self._xr_instance,
            self._xr_system_id,
            xr.ViewConfigurationType.PRIMARY_STEREO,
        )
        # Pick the best supported DXGI format
        runtime_fmts = xr.enumerate_swapchain_formats(self._xr_session)
        chosen_fmt = None
        for preferred in _D3D11_PREFERRED_FORMATS:
            if preferred in runtime_fmts:
                chosen_fmt = preferred
                break
        if chosen_fmt is None:
            raise RuntimeError(f"No supported D3D11 swapchain format. Runtime offers: {runtime_fmts}")
        self._d3d11_swapchain_fmt = chosen_fmt
        self._swapchain_is_bgra = chosen_fmt in (
            _DXGI_FORMAT_B8G8R8A8_UNORM_SRGB, _DXGI_FORMAT_B8G8R8A8_UNORM,
        )
        print(f"[OpenXRViewer] D3D11 swapchain format: {chosen_fmt}"
              f"{' (BGRA)' if self._swapchain_is_bgra else ''}")

        for eye_index, vcv in enumerate(view_configs):
            rec_w = vcv.recommended_image_rect_width
            rec_h = vcv.recommended_image_rect_height
            sc_w  = rec_w & ~1
            sc_h  = rec_h & ~1
            print(f"[OpenXRViewer] Eye {eye_index} swapchain: {sc_w}x{sc_h} (D3D11)")

            sc_info = xr.SwapchainCreateInfo(
                usage_flags=(
                    xr.SwapchainUsageFlags.COLOR_ATTACHMENT_BIT |
                    xr.SwapchainUsageFlags.SAMPLED_BIT
                ),
                format=chosen_fmt,
                sample_count=1,
                width=sc_w,
                height=sc_h,
                face_count=1,
                array_size=1,
                mip_count=1,
            )
            swapchain = xr.create_swapchain(self._xr_session, sc_info)
            images    = xr.enumerate_swapchain_images(swapchain, xr.SwapchainImageD3D11KHR)
            self._xr_swapchains[eye_index]    = swapchain
            self._swapchain_images[eye_index] = images
            self._swapchain_sizes[eye_index]  = (sc_w, sc_h)

        # 9. Try GPU interop to avoid the PBO readback path
        self._setup_gpu_interop_d3d11()

        # 10. Controller actions (best-effort)
        try:
            self._init_controller_actions()
        except Exception as e:
            print(f"[OpenXRViewer] Controller actions unavailable: {e}")

    # ── GPU interop helpers ────────────────────────────────────────────

    @staticmethod
    def _is_nvidia_gpu():
        """Detect NVIDIA GPU via OpenGL renderer string."""
        try:
            from OpenGL.GL import glGetString, GL_RENDERER
            r = glGetString(GL_RENDERER)
            if r:
                return b'NVIDIA' in r.upper() if isinstance(r, bytes) else 'NVIDIA' in r.upper()
        except Exception:
            pass
        try:
            import torch
            if torch.cuda.is_available():
                return 'NVIDIA' in torch.cuda.get_device_name(0)
        except Exception:
            pass
        return False

    def _setup_gpu_interop_d3d11(self):
        """Attempt GPU interop to eliminate the PBO readback path.

        Order: NV_DX_interop2 for NVIDIA GPUs, EXT_memory_object for all others.
        Falls back to the PBO path (already configured) if neither is available.

        Interop is skipped for BGRA swapchains (common on WMR) because GL
        renders RGBA natively and the R↔B mismatch would swap colours.
        The PBO path handles BGRA via GL_BGRA readback format.
        """
        if not sys.platform == "win32":
            return

        if self._swapchain_is_bgra:
            print("[OpenXRViewer] BGRA swapchain — GPU interop disabled (using PBO with GL_BGRA readback)")
            return

        is_nv = self._is_nvidia_gpu()

        if is_nv and _load_nv_dx_interop():
            try:
                self._init_interop_nv()
                self._interop_mode = 'nv_dx'
                print("[OpenXRViewer] GPU interop active: NV_DX_interop2 (zero-copy)")
                return
            except Exception as e:
                print(f"[OpenXRViewer] NV_DX_interop2 setup failed: {e}")

        if _load_ext_memory_object():
            try:
                self._init_interop_ext_mem()
                self._interop_mode = 'ext_mem'
                print("[OpenXRViewer] GPU interop active: EXT_memory_object (GPU-side blit)")
                return
            except Exception as e:
                print(f"[OpenXRViewer] EXT_memory_object setup failed: {e}")

        self._interop_mode = None
        print("[OpenXRViewer] GPU interop unavailable — using PBO fallback")

    def _init_interop_nv(self):
        """Set up WGL_NV_DX_interop2: register the D3D11 device with GL.

        Individual swapchain textures are registered per-frame the first time
        each image index is seen (see _get_or_create_nv_interop_fbo).
        """
        self._nv_dx_device = _wglDXOpenDeviceNV(self._d3d11_device)
        if not self._nv_dx_device:
            raise RuntimeError("wglDXOpenDeviceNV returned NULL")

    def _get_or_create_nv_interop_fbo(self, eye_index, img_index, d3d11_tex, w, h):
        """Register a swapchain D3D11 texture with GL via NV_DX_interop2.

        Each unique (eye, img_index) pair is registered once and cached.
        Returns (mgl_fbo, raw_fbo_id) for direct rendering into the D3D11 texture.
        """
        key = (eye_index, img_index)
        if key in self._nv_dx_objects:
            gl_tex, raw_fbo = self._nv_dx_objects[key]
            return self.ctx.detect_framebuffer(raw_fbo), raw_fbo

        gl_tex = glGenTextures(1)
        # Register the D3D11 texture as a GL texture
        dx_obj = _wglDXRegisterObjectNV(
            self._nv_dx_device,
            d3d11_tex,
            gl_tex,
            GL_TEXTURE_2D,
            0x0002,  # WGL_ACCESS_WRITE_DISCARD_NV → the driver knows we overwrite
        )
        if not dx_obj:
            glDeleteTextures(1, [gl_tex])
            raise RuntimeError(f"wglDXRegisterObjectNV failed for eye {eye_index} img {img_index}")

        # Set up FBO attached to the registered texture
        raw_fbo = glGenFramebuffers(1)
        glBindFramebuffer(GL_FRAMEBUFFER, raw_fbo)
        # Lock, attach, unlock
        _wglDXLockObjectsNV(self._nv_dx_device, 1, ctypes.byref(dx_obj))
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, gl_tex, 0)
        _wglDXUnlockObjectsNV(self._nv_dx_device, 1, ctypes.byref(dx_obj))
        glBindFramebuffer(GL_FRAMEBUFFER, 0)

        self._nv_dx_objects[key] = (gl_tex, raw_fbo, dx_obj)
        return self.ctx.detect_framebuffer(raw_fbo), raw_fbo

    def _init_interop_ext_mem(self):
        """Set up EXT_memory_object_win32: create shared D3D11 textures and
        import them into GL once.  Render to the GL side, then CopyResource
        to the swapchain image each frame (GPU-side blit, no CPU round-trip).
        """
        for eye_index in range(2):
            sc_w, sc_h = self._swapchain_sizes[eye_index]
            fmt = self._d3d11_swapchain_fmt
            d3d11_tex, nt_handle = _create_d3d11_shared_texture(
                self._d3d11_device, sc_w, sc_h, fmt,
            )

            # Import into GL
            mem_obj = ctypes.c_uint(0)
            _glCreateMemoryObjectsEXT(1, ctypes.byref(mem_obj))
            _glImportMemoryWin32HandleEXT(
                mem_obj, sc_w * sc_h * 4,
                _GL_HANDLE_TYPE_OPAQUE_WIN32_EXT,
                nt_handle,
            )

            # Create GL texture backed by the imported memory
            gl_tex = glGenTextures(1)
            glBindTexture(GL_TEXTURE_2D, gl_tex)
            _glTextureStorageMem2DEXT(gl_tex, 1, GL_RGBA8, sc_w, sc_h, mem_obj, 0)
            glBindTexture(GL_TEXTURE_2D, 0)

            # FBO
            raw_fbo = int(glGenFramebuffers(1))
            glBindFramebuffer(GL_FRAMEBUFFER, raw_fbo)
            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, gl_tex, 0)
            st = glCheckFramebufferStatus(GL_FRAMEBUFFER)
            glBindFramebuffer(GL_FRAMEBUFFER, 0)
            if st != GL_FRAMEBUFFER_COMPLETE:
                raise RuntimeError(f"EXT_mem FBO incomplete for eye {eye_index}: {st:#x}")

            mgl_fbo = self.ctx.detect_framebuffer(raw_fbo)
            self._ext_shared_tex[eye_index] = (d3d11_tex, mem_obj, gl_tex, mgl_fbo, raw_fbo)

    def _blit_ext_to_swapchain(self, eye_index, d3d11_swapchain_tex):
        """GPU-side CopyResource from our shared texture to the swapchain image."""
        d3d11_shared_tex = self._ext_shared_tex[eye_index][0]
        # ID3D11DeviceContext::CopyResource at vtable index 47
        ctx = self._d3d11_context
        vtbl = ctypes.cast(ctx, ctypes.POINTER(ctypes.c_void_p)).contents.value
        copy_fn = ctypes.CFUNCTYPE(
            None,
            ctypes.c_void_p,  # this
            ctypes.c_void_p,  # pDstResource
            ctypes.c_void_p,  # pSrcResource
        )(ctypes.cast(vtbl + 47 * ctypes.sizeof(ctypes.c_void_p),
                      ctypes.POINTER(ctypes.c_void_p)).contents.value)
        # Sync: ensure GL is done before D3D11 reads the shared texture
        glFinish()
        copy_fn(ctx, d3d11_swapchain_tex, d3d11_shared_tex)

    def _cleanup_interop(self):
        """Release all GPU interop resources."""
        if self._interop_mode == 'nv_dx' and self._nv_dx_device:
            for (gl_tex, raw_fbo, dx_obj) in self._nv_dx_objects.values():
                try:
                    _wglDXUnregisterObjectNV(self._nv_dx_device, dx_obj)
                except Exception:
                    pass
                try:
                    glDeleteFramebuffers(1, [raw_fbo])
                except Exception:
                    pass
                try:
                    glDeleteTextures(1, [gl_tex])
                except Exception:
                    pass
            self._nv_dx_objects.clear()
            try:
                _wglDXCloseDeviceNV(self._nv_dx_device)
            except Exception:
                pass
            self._nv_dx_device = None

        if self._interop_mode == 'ext_mem':
            for d3d11_tex, mem_obj, gl_tex, mgl_fbo, raw_fbo in self._ext_shared_tex.values():
                try:
                    glDeleteFramebuffers(1, [raw_fbo])
                except Exception:
                    pass
                try:
                    glDeleteTextures(1, [gl_tex])
                except Exception:
                    pass
                try:
                    _glDeleteMemoryObjectsEXT(1, ctypes.byref(ctypes.c_uint(mem_obj)))
                except Exception:
                    pass
                # Release D3D11 texture
                try:
                    tex_vtbl = ctypes.cast(d3d11_tex, ctypes.POINTER(ctypes.c_void_p)).contents.value
                    tex_rel = ctypes.CFUNCTYPE(ctypes.c_ulong, ctypes.c_void_p)(
                        ctypes.cast(tex_vtbl + 2 * ctypes.sizeof(ctypes.c_void_p),
                                    ctypes.POINTER(ctypes.c_void_p)).contents.value
                    )
                    tex_rel(d3d11_tex)
                except Exception:
                    pass
            self._ext_shared_tex.clear()

        self._interop_mode = None

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
        print("[OpenXRViewer] XrInstance created (OpenGL)")

        # 2. System
        self._xr_system_id = xr.get_system(
            self._xr_instance,
            xr.SystemGetInfo(form_factor=xr.FormFactor.HEAD_MOUNTED_DISPLAY),
        )

        # 3. Verify GL requirements (mandatory before session creation)
        _pfn = ctypes.cast(
            xr.get_instance_proc_addr(self._xr_instance, "xrGetOpenGLGraphicsRequirementsKHR"),
            xr.PFN_xrGetOpenGLGraphicsRequirementsKHR,
        )
        _reqs = xr.GraphicsRequirementsOpenGLKHR()
        xr.check_result(xr.Result(_pfn(self._xr_instance, self._xr_system_id, ctypes.byref(_reqs))))

        # 4. Graphics binding — platform-specific
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

        # 6. Reference space — prefer STAGE (floor origin), fall back to LOCAL
        available_spaces = xr.enumerate_reference_spaces(self._xr_session)
        ref_type = (
            xr.ReferenceSpaceType.STAGE
            if xr.ReferenceSpaceType.STAGE in available_spaces
            else xr.ReferenceSpaceType.LOCAL
        )
        self._xr_space = xr.create_reference_space(
            self._xr_session,
            xr.ReferenceSpaceCreateInfo(
                reference_space_type=ref_type,
                pose_in_reference_space=xr.Posef(),
            ),
        )

        # 7. Swapchains — one per eye
        view_configs = xr.enumerate_view_configuration_views(
            self._xr_instance,
            self._xr_system_id,
            xr.ViewConfigurationType.PRIMARY_STEREO,
        )
        for eye_index, vcv in enumerate(view_configs):
            rec_w = vcv.recommended_image_rect_width
            rec_h = vcv.recommended_image_rect_height
            # Use exactly the recommended resolution — this matches the HMD panel
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

        # 8. Controller actions (optional — silently disabled if action set creation fails)
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

        # Grip pose actions ── used for placing controller 3D models
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
        # Use squeeze/value (float path) for grip — the runtime auto-thresholds it
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
            # KHR simple only has select/click (boolean) and menu — no sticks or grip
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

    def _init_textures(self, w, h):
        if self.color_tex:
            self.color_tex.release()
        if self.depth_tex:
            self.depth_tex.release()
        self.color_tex = self.ctx.texture((w, h), 3, dtype='f1')
        self.color_tex.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
        self.color_tex.build_mipmaps()
        try:
            self.color_tex.anisotropy = 16.0
        except Exception:
            pass
        # Negative LOD bias: bias the sampler toward sharper (higher-res) mip levels.
        # -0.5 = use a mip level 0.5 finer than the GPU would naturally pick,
        # preserving anti-aliasing while recovering perceived sharpness.
        glBindTexture(GL_TEXTURE_2D, self.color_tex.glo)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_LOD_BIAS, -0.5)
        glBindTexture(GL_TEXTURE_2D, 0)
        self.depth_tex = self.ctx.texture((w, h), 1, dtype='f4')
        self.depth_tex.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
        self.depth_tex.build_mipmaps()
        self._texture_size = (w, h)

    def _build_keyboard_texture(self):
        """(Re)build the virtual keyboard texture with the current shift state.

        When Shift or Caps Lock is active the number/symbol keys show their
        shifted glyph (e.g. '!' instead of '1').  Modifier key backgrounds
        are highlighted as before.
        """
        TW, TH   = _KB_TEX_W, _KB_TEX_H
        ROW_H    = TH / len(_KB_ROWS)
        UNIT_W   = TW / float(_KB_UNITS_WIDE)
        UNIT_M   = self._keyboard_width / float(_KB_UNITS_WIDE)
        PAD      = 3
        show_s   = self._kb_show_shifted   # whether to render shifted labels

        img  = Image.new('RGBA', (TW, TH), (30, 30, 35, 230))
        draw = ImageDraw.Draw(img)

        fnt = None
        for candidate in (r"C:\Windows\Fonts\seguisym.ttf",
                          r"C:\Windows\Fonts\segoeui.ttf",
                          self.font_type):
            if not candidate:
                continue
            try:
                fnt = ImageFont.truetype(candidate, 16)
                break
            except Exception:
                continue

        self._keyboard_keys = []
        kw_half  = self._keyboard_width  / 2.0
        kh_half  = self._keyboard_height / 2.0
        row_h_m  = self._keyboard_height / len(_KB_ROWS)

        for row_i, row in enumerate(_KB_ROWS):
            py0 = int(row_i * ROW_H)
            py1 = int((row_i + 1) * ROW_H)
            ly1 = kh_half - row_i * row_h_m
            ly0 = ly1 - row_h_m
            px  = 0.0
            lx  = -kw_half
            for (label, vk_normal, shifted_label, vk_shifted, width_units) in row:
                px_end  = px + width_units * UNIT_W
                lx_end  = lx + width_units * UNIT_M

                if vk_normal == -1:
                    px = px_end
                    lx = lx_end
                    continue

                # Key background
                draw.rectangle([px + PAD, py0 + PAD, px_end - PAD, py1 - PAD],
                               fill=(60, 62, 70, 255), outline=(130, 132, 140, 255))

                # Pick label: shifted version if available and shift is active
                display_label = label
                if show_s and shifted_label is not None:
                    display_label = shifted_label

                if fnt:
                    tx = (px + px_end) / 2
                    ty = (py0 + py1) / 2
                    draw.text((tx, ty), display_label, font=fnt,
                              fill=(220, 220, 225, 255), anchor='mm')
                else:
                    draw.text((int(px + PAD + 2), int(py0 + PAD + 2)),
                              display_label, fill=(220, 220, 225, 255))

                uv_rect   = (px / TW, py0 / TH, px_end / TW, py1 / TH)
                loc_rect  = (lx, ly0, lx_end, ly1)

                self._keyboard_keys.append(_KeyEntry(
                    label=label,
                    shifted_label=shifted_label,
                    vk=vk_normal,
                    shifted_vk=vk_shifted if vk_shifted is not None else vk_normal,
                    rect_uv=uv_rect,
                    rect_local=loc_rect,
                ))

                px  = px_end
                lx  = lx_end

        # Upload
        tex_data = np.flipud(np.array(img, dtype=np.uint8))
        if self._keyboard_tex is not None:
            self._keyboard_tex.release()
        self._keyboard_tex = self.ctx.texture((TW, TH), 4, dtype='f1')
        self._keyboard_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self._keyboard_tex.write(tex_data.tobytes())

        # VAO (first build only — geometry never changes)
        if self._keyboard_vao is None:
            verts = np.array([-1,-1,0,0, 1,-1,1,0, -1,1,0,1, 1,1,1,1], dtype='f4')
            self._keyboard_vao = self.ctx.vertex_array(
                self._overlay_prog,
                [(self.ctx.buffer(verts.tobytes()), '2f 2f', 'in_position', 'in_uv')],
            )

    def _init_keyboard(self):
        """Initial keyboard build (called once when the user toggles it on)."""
        self._kb_show_shifted = False
        self._build_keyboard_texture()

    def _kb_world_mat(self):
        """Build the keyboard's world transform: rot_y(yaw) ∘ rot_x(pitch) then translate.

        The keyboard's local frame has Z = surface normal. Negative pitch tilts the
        face up so a user looking down at it sees the face dead-on (friendly angle,
        like a desk keyboard).
        """
        cp = math.cos(self._keyboard_pitch); sp = math.sin(self._keyboard_pitch)
        cy = math.cos(self._keyboard_yaw);   sy = math.sin(self._keyboard_yaw)
        rot_y = np.array([[ cy, 0, sy, 0],
                          [  0, 1,  0, 0],
                          [-sy, 0, cy, 0],
                          [  0, 0,  0, 1]], dtype=np.float32)
        rot_x = np.array([[1,  0,   0, 0],
                          [0, cp, -sp, 0],
                          [0, sp,  cp, 0],
                          [0,  0,   0, 1]], dtype=np.float32)
        trans = np.eye(4, dtype=np.float32)
        # Translate to (pan_x, pan_y, -distance) — matches the world-anchor convention
        # used by the main screen.
        trans[0, 3] = self._keyboard_pan_x
        trans[1, 3] = self._keyboard_pan_y
        trans[2, 3] = -self._keyboard_distance
        return trans @ rot_y @ rot_x

    def _anchor_keyboard_below_screen(self):
        """Snap the keyboard below the screen's bottom edge, facing the same direction.

        The keyboard sits below the FPS overlay panel so it doesn't overlap.
        """
        FPS_GAP = 0.05   # gap between screen bottom and FPS overlay
        FPS_H   = 0.12   # FPS overlay panel height
        KB_GAP  = 0.05   # gap between FPS overlay bottom and keyboard top (same as FPS_GAP)
        if self.screen_height is None:
            fw, fh = self.frame_size
            sh = self.screen_width * (fh / fw if fw > 0 else 9.0 / 16.0)
        else:
            sh = self.screen_height
        # Place keyboard below the FPS overlay panel, same distance + yaw
        self._keyboard_pan_x    = self.screen_pan_x
        self._keyboard_pan_y    = (self.screen_pan_y - sh / 2.0
                                   - FPS_GAP - FPS_H - KB_GAP
                                   - self._keyboard_height / 2.0)
        self._keyboard_distance = self.screen_distance
        self._keyboard_yaw      = self.screen_yaw
        self._keyboard_pitch    = math.radians(-30.0)  # tilt up toward user

    def _render_keyboard(self, mgl_fbo, vp_mat):
        """Render the virtual keyboard quad and highlight hovered keys."""
        if self._keyboard_tex is None or self._keyboard_vao is None:
            return

        kw2 = self._keyboard_width  / 2.0
        kh2 = self._keyboard_height / 2.0
        kb_world = self._kb_world_mat()
        vp_kb = vp_mat @ kb_world   # shared for all key highlights

        # Keyboard quad: vertices are in [-1, +1] in X and Y, so scale by half-extents.
        scale_kb = np.array([[kw2, 0,   0, 0],
                             [0,   kh2, 0, 0],
                             [0,   0,   1, 0],
                             [0,   0,   0, 1]], dtype=np.float32)
        mvp = vp_kb @ scale_kb

        mgl_fbo.use()
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA

        # Keyboard border: solid quad slightly larger than the keyboard
        if self._kb_border_alpha > 0.0 and self._border_prog is not None:
            BORDER = 0.008
            bx = kw2 + BORDER
            by = kh2 + BORDER
            border_scale = np.array([[bx, 0, 0, 0],
                                     [0, by, 0, 0],
                                     [0, 0,  1, -0.001],
                                     [0, 0,  0, 1]], dtype=np.float32)
            border_mvp = vp_kb @ border_scale
            self._border_prog['u_mvp'].write(border_mvp.T.tobytes())
            self._border_prog['u_color'].value = (0.3, 0.7, 1.0, self._kb_border_alpha)
            self._border_vao.render(moderngl.TRIANGLE_STRIP)

        self._keyboard_tex.use(location=2)
        self._overlay_prog['u_mvp'].write(mvp.T.tobytes())
        self._keyboard_vao.render(moderngl.TRIANGLE_STRIP)

        def _hl_quad(rect_local, color):
            # rect_local is already in metres, expressed in the keyboard's local frame
            # (X right, Y up, Z = surface). Place a unit quad scaled to the key rect at
            # +1 mm in front of the surface to avoid z-fighting.
            x0, y0, x1, y1 = rect_local
            cx = (x0 + x1) / 2.0; cy_ = (y0 + y1) / 2.0
            hw = (x1 - x0) / 2.0; hh  = (y1 - y0) / 2.0
            hl_local = np.array([[hw, 0,  0, cx ],
                                 [0,  hh, 0, cy_],
                                 [0,  0,  1, 0.001],
                                 [0,  0,  0, 1  ]], dtype=np.float32)
            hl_mvp = vp_kb @ hl_local
            self._border_prog['u_mvp'].write(hl_mvp.T.tobytes())
            self._border_prog['u_color'].value = color
            self._border_vao.render(moderngl.TRIANGLE_STRIP)

        # Highlight every key whose VK matches an armed modifier.
        # Locked modifiers get a brighter amber than one-shot to make state legible.
        VK_SHIFT = 0x10; VK_CAPS = 0x14; VK_CTRL = 0x11; VK_ALT = 0x12; VK_WIN = 0x5B
        oneshot_vks = set(); locked_vks = set()
        for name, vk in (('shift', VK_SHIFT), ('ctrl', VK_CTRL),
                         ('alt', VK_ALT), ('win', VK_WIN)):
            active, locked, _ = self._mod_state[name]
            if locked:   locked_vks.add(vk)
            elif active: oneshot_vks.add(vk)
        if self._caps_lock: locked_vks.add(VK_CAPS)
        for key in self._keyboard_keys:
            if key.vk in locked_vks:
                _hl_quad(key.rect_local, (1.0, 0.55, 0.05, 0.65))
            elif key.vk in oneshot_vks:
                _hl_quad(key.rect_local, (1.0, 0.7, 0.15, 0.45))

        # Cyan highlight on keys hovered by either laser
        for hover_idx in set(x for x in [self._kb_hover_l, self._kb_hover_r] if x is not None):
            _hl_quad(self._keyboard_keys[hover_idx].rect_local, (0.2, 0.7, 1.0, 0.35))

        self.ctx.disable(moderngl.BLEND)

    def _init_cuda_pbos(self, w, h):
        """Create or recreate PBOs and register them with CUDA/HIP."""
        if not self._cuda_gl or BACKEND not in ("CUDA", "HIP"):
            return
        # Unregister old resources before deleting PBOs
        if self._pbo_color is not None:
            try:
                self._cuda_gl.unregister_resource(self._cuda_res_color)
                self._cuda_gl.unregister_resource(self._cuda_res_depth)
                glDeleteBuffers(2, [self._pbo_color, self._pbo_depth])
            except Exception:
                pass

        ids = glGenBuffers(2)
        self._pbo_color = int(ids[0])
        self._pbo_depth = int(ids[1])

        for pbo_id, nbytes in [
            (self._pbo_color, w * h * 3),   # RGB uint8
            (self._pbo_depth, w * h * 4),   # float32
        ]:
            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo_id)
            glBufferData(GL_PIXEL_UNPACK_BUFFER, nbytes, None, GL_DYNAMIC_DRAW)
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0)

        self._cuda_res_color = self._cuda_gl.register_buffer(self._pbo_color)
        self._cuda_res_depth = self._cuda_gl.register_buffer(self._pbo_depth)
        self._pbo_texture_size = (w, h)
        print(f"[OpenXRViewer] GPU interop PBOs created ({BACKEND}) {w}x{h}")

    # ------------------------------------------------------------------
    # Per-frame helpers
    # ------------------------------------------------------------------

    def _update_frame(self, rgb, depth):
        """Upload RGB and depth to GL textures — GPU path when available, CPU fallback."""
        import torch

        is_tensor = hasattr(rgb, 'data_ptr')

        # Resolve depth shape and GPU tensor
        if hasattr(depth, 'detach'):
            depth_gpu = depth.detach().contiguous().float()
            h, w = depth_gpu.shape[0], depth_gpu.shape[1]
            depth_np = None
        else:
            depth_gpu = None
            depth_np = np.asarray(depth, dtype=np.float32)
            h, w = depth_np.shape[0], depth_np.shape[1]

        if self._texture_size != (w, h):
            self._init_textures(w, h)
            self.frame_size = (w, h)
            self.screen_height = None

        # Lazy GPU interop init (includes PBO registration to verify interop)
        if self._cuda_gl is None and CUDART_GL is not None and BACKEND in ("CUDA", "HIP"):
            try:
                self._cuda_gl = CUDART_GL()
                self._init_cuda_pbos(w, h)   # create PBOs + register with HIP
                print(f"[OpenXRViewer] GPU interop active ({BACKEND})")
            except Exception as e:
                print(f"[OpenXRViewer] GPU interop unavailable: {e}")
                self._cuda_gl = False   # sentinel: don't retry

        gpu_ok = bool(self._cuda_gl) and is_tensor and BACKEND in ("CUDA", "HIP")

        if gpu_ok:
            if self._pbo_texture_size != (w, h):
                self._init_cuda_pbos(w, h)

            # Color: CHW tensor → HWC contiguous uint8 on GPU, DMA into PBO
            rgb_gpu = rgb.permute(1, 2, 0).contiguous().clamp(0, 255).to(torch.uint8)
            ptr = self._cuda_gl.map_resource(self._cuda_res_color)
            self._cuda_gl.memcpy_d2d(ptr, rgb_gpu.data_ptr(), rgb_gpu.nbytes)
            self._cuda_gl.unmap_resource(self._cuda_res_color)
            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, self._pbo_color)
            glBindTexture(GL_TEXTURE_2D, self.color_tex.glo)
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, w, h, GL_RGB, GL_UNSIGNED_BYTE, ctypes.c_void_p(0))
            glGenerateMipmap(GL_TEXTURE_2D)
            glBindTexture(GL_TEXTURE_2D, 0)

            ptr = self._cuda_gl.map_resource(self._cuda_res_depth)
            self._cuda_gl.memcpy_d2d(ptr, depth_gpu.contiguous().data_ptr(), depth_gpu.nbytes)
            self._cuda_gl.unmap_resource(self._cuda_res_depth)
            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, self._pbo_depth)
            glBindTexture(GL_TEXTURE_2D, self.depth_tex.glo)
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, w, h, GL_RED, GL_FLOAT, ctypes.c_void_p(0))
            glGenerateMipmap(GL_TEXTURE_2D)
            glBindTexture(GL_TEXTURE_2D, 0)
            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0)
        else:
            # CPU fallback
            if hasattr(rgb, 'detach'):
                rgb_np = (
                    rgb.permute(1, 2, 0).detach().contiguous()
                    .clamp(0, 255).to(torch.uint8).cpu().numpy()
                )
            else:
                rgb_np = np.asarray(rgb, dtype=np.uint8)
            if depth_np is None:
                depth_np = depth_gpu.cpu().numpy()
            self.color_tex.write(rgb_np.astype('uint8', copy=False).tobytes())
            glBindTexture(GL_TEXTURE_2D, self.color_tex.glo)
            glGenerateMipmap(GL_TEXTURE_2D)
            glBindTexture(GL_TEXTURE_2D, 0)
            self.depth_tex.write(depth_np.tobytes())
            glBindTexture(GL_TEXTURE_2D, self.depth_tex.glo)
            glGenerateMipmap(GL_TEXTURE_2D)
            glBindTexture(GL_TEXTURE_2D, 0)

    def _build_model_mat4(self):
        """Construct the screen quad's world-space model matrix (math row/col convention).

        The quad's in_position spans [-1,+1] in X and Y in model space.
        We scale to physical metres, translate to (pan_x, pan_y, -distance) in world
        space (OpenXR: right-hand Y-up, forward = -Z), then apply pitch and yaw.
        Caller must transpose before writing to OpenGL.
        """
        if self.screen_height is None:
            fw, fh = self.frame_size
            self.screen_height = self.screen_width * (fh / fw if fw > 0 else 9 / 16)

        sx  = self.screen_width  / 2.0
        sy  = self.screen_height / 2.0

        cy  = math.cos(self.screen_yaw)
        sy_ = math.sin(self.screen_yaw)
        rot_y = np.array([
            [ cy,  0, sy_, 0],
            [  0,  1,   0, 0],
            [-sy_, 0,  cy, 0],
            [  0,  0,   0, 1],
        ], dtype=np.float32)

        cp  = math.cos(self.screen_pitch)
        sp  = math.sin(self.screen_pitch)
        rot_x = np.array([
            [1,  0,   0, 0],
            [0,  cp, -sp, 0],
            [0,  sp,  cp, 0],
            [0,  0,   0,  1],
        ], dtype=np.float32)

        # Scale + translate: translation in last column, row 3 = [0,0,0,1]
        model = np.array([
            [sx,  0,  0, self.screen_pan_x    ],
            [ 0, sy,  0, self.screen_pan_y    ],
            [ 0,  0,  1, -self.screen_distance],
            [ 0,  0,  0, 1                    ],
        ], dtype=np.float32)

        return rot_y @ rot_x @ model

    def _build_curved_screen_verts(self, N=48, width_override=None, height_override=None, dist_offset=0.0):
        """Return a float32 numpy array for a TRIANGLE_STRIP curved screen arc.

        The arc is a cylinder section centred on (pan_x, pan_y, -distance) with
        radius = screen_distance, spanning screen_width metres along the arc
        horizontally and screen_height metres vertically.  Vertices are in world
        space so the shader only needs the view-projection matrix (no model matrix).

        width_override / height_override: use instead of screen_width/height (for border).
        dist_offset: added to screen_distance to push the surface slightly back.

        Layout per vertex: x y z  u v  (5 floats).
        The strip has (N+1)*2 vertices — one column pair per segment.
        """
        if self.screen_height is None:
            fw, fh = self.frame_size
            self.screen_height = self.screen_width * (fh / fw if fw > 0 else 9 / 16)

        R        = self.screen_distance + dist_offset  # cylinder radius
        half_w   = (width_override  if width_override  is not None else self.screen_width)  / 2.0
        half_h   = (height_override if height_override is not None else self.screen_height) / 2.0
        # Angular half-span so the arc LENGTH equals screen_width (not the chord).
        # arc_length = R * 2 * half_ang  →  half_ang = half_w / R
        half_ang = min(half_w / max(R, 0.01), math.pi / 2)

        yaw   = self.screen_yaw
        pitch = self.screen_pitch
        cy, sy_ = math.cos(yaw),   math.sin(yaw)
        cp, sp  = math.cos(pitch), math.sin(pitch)

        cx = self.screen_pan_x
        cy_pan = self.screen_pan_y

        verts = []
        for i in range(N + 1):
            t    = i / N                         # [0, 1]
            ang  = -half_ang + 2.0 * half_ang * t   # angle from left to right
            u    = t

            # Local arc point: x along arc, z = depth into screen
            lx   = R * math.sin(ang)
            lz   = -(R * math.cos(ang))          # -Z = forward in OpenXR

            # Apply yaw around Y, then pitch around X, then translate
            # Yaw: (lx, 0, lz) → (lx*cy + lz*sy_, 0, -lx*sy_ + lz*cy)
            rx   =  lx * cy  + lz * sy_
            rz   = -lx * sy_ + lz * cy

            # Two rows: bottom (v=0) and top (v=1)
            for row, v in ((0, 0.0), (1, 1.0)):
                ly = (-half_h if row == 0 else half_h)
                # Pitch: rotate (rx, ly, rz) around X by pitch
                wy =  ly * cp - rz * sp
                wz =  ly * sp + rz * cp
                wx = rx

                # Translate to world position
                wx += cx
                wy += cy_pan
                wz += 0.0          # translation already encoded in lz via R

                verts.extend([wx, wy, wz, u, v])

        return np.array(verts, dtype='f4')

    def _get_or_create_fbo(self, eye_index, image_index, texture_id):
        """Lazily create and cache a ModernGL Framebuffer wrapping the swapchain texture.

        ctx.detect_framebuffer() is used so ModernGL's internal state tracking stays
        consistent — raw glBindFramebuffer() is invisible to ModernGL and would cause
        ctx.clear() / vao.render() to target the wrong framebuffer.
        """
        key = (eye_index, image_index)
        if key in self._fbo_cache:
            return self._fbo_cache[key]

        raw_id = glGenFramebuffers(1)
        glBindFramebuffer(GL_FRAMEBUFFER, raw_id)
        glFramebufferTexture2D(
            GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texture_id, 0
        )
        status = glCheckFramebufferStatus(GL_FRAMEBUFFER)
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        if status != GL_FRAMEBUFFER_COMPLETE:
            raise RuntimeError(
                f"[OpenXRViewer] FBO incomplete for eye {eye_index}, "
                f"image {image_index}: {status:#x}"
            )
        mgl_fbo = self.ctx.detect_framebuffer(raw_id)
        self._fbo_cache[key] = (raw_id, mgl_fbo)
        return raw_id, mgl_fbo

    def _get_or_create_offscreen_fbo(self, eye_index, image_index, w, h):
        """Return a ModernGL FBO backed by an RGBA texture of size (w, h).

        Used in the D3D11 path: ModernGL renders into this offscreen FBO, then
        _blit_gl_to_d3d11() reads it back and uploads it to the D3D11 swapchain image.
        """
        key = (eye_index, image_index)
        cached = self._offscreen_fbo_cache.get(key)
        if cached and cached[3] == w and cached[4] == h:
            return cached[0], cached[1]   # mgl_fbo, raw_id

        # Discard old entry if dimensions changed
        if cached:
            try:
                cached[2].release()    # mgl Texture
                glDeleteFramebuffers(1, [cached[1]])
            except Exception:
                pass

        raw_id = glGenFramebuffers(1)
        mgl_tex = self.ctx.texture((w, h), 4, dtype='f1')
        glBindFramebuffer(GL_FRAMEBUFFER, raw_id)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                               GL_TEXTURE_2D, mgl_tex.glo, 0)
        status = glCheckFramebufferStatus(GL_FRAMEBUFFER)
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        if status != GL_FRAMEBUFFER_COMPLETE:
            raise RuntimeError(
                f"[OpenXRViewer] Offscreen FBO incomplete for eye {eye_index}: {status:#x}"
            )
        mgl_fbo = self.ctx.detect_framebuffer(raw_id)
        self._offscreen_fbo_cache[key] = (mgl_fbo, raw_id, mgl_tex, w, h)
        return mgl_fbo, raw_id

    def _get_or_create_d3d11_pbo(self, eye_index, img_index, w, h):
        """Return a GL PBO id sized for (w, h) RGBA readback, creating/resizing as needed."""
        key = (eye_index, img_index)
        cached = self._d3d11_pbo_cache.get(key)
        if cached and cached[1] == w and cached[2] == h:
            return cached[0]
        if cached:
            glDeleteBuffers(1, [cached[0]])
        pbo_id = int(glGenBuffers(1))
        glBindBuffer(GL_PIXEL_PACK_BUFFER, pbo_id)
        glBufferData(GL_PIXEL_PACK_BUFFER, w * h * 4, None, GL_STREAM_READ)
        glBindBuffer(GL_PIXEL_PACK_BUFFER, 0)
        self._d3d11_pbo_cache[key] = (pbo_id, w, h)
        return pbo_id

    def _submit_pbo_readback(self, raw_fbo_id, pbo_id, w, h):
        """Submit an async glReadPixels into pbo_id and flush to kick off DMA immediately.

        Uses GL_BGRA for BGRA swapchains (WMR) so the byte order matches D3D11 directly.
        """
        pixel_fmt = GL_BGRA if self._swapchain_is_bgra else GL_RGBA
        glBindFramebuffer(GL_FRAMEBUFFER, raw_fbo_id)
        glBindBuffer(GL_PIXEL_PACK_BUFFER, pbo_id)
        glReadPixels(0, 0, w, h, pixel_fmt, GL_UNSIGNED_BYTE, ctypes.c_void_p(0))
        glFlush()  # push the DMA command to the GPU so it starts while we render eye 1
        glBindBuffer(GL_PIXEL_PACK_BUFFER, 0)
        glBindFramebuffer(GL_FRAMEBUFFER, 0)

    def _upload_pbo_to_d3d11(self, pbo_id, d3d11_texture_ptr, w, h):
        """Map the readback PBO and upload straight into the D3D11 swapchain texture.

        GL renders Y-flipped (see _render_eye flip_y) so glReadPixels already
        produces top-down rows — no CPU row-reversal needed.  The mapped PBO
        pointer is passed directly to D3D11 UpdateSubresource, eliminating the
        intermediate flip-buffer and its per-frame memcpy.
        """
        row_bytes = w * 4
        glBindBuffer(GL_PIXEL_PACK_BUFFER, pbo_id)
        # UNSYNCHRONIZED: the Phase-1/Phase-2 pipelining gives the DMA enough time
        # to finish; if it hasn't, we accept a one-frame visual glitch rather than
        # stalling the pipeline.
        src_ptr = glMapBuffer(GL_PIXEL_PACK_BUFFER, GL_READ_ONLY | GL_MAP_UNSYNCHRONIZED_BIT)
        if src_ptr:
            try:
                _d3d11_update_subresource(
                    self._d3d11_context, d3d11_texture_ptr,
                    int(src_ptr), row_bytes,
                )
            except Exception as exc:
                print(f"[OpenXRViewer] d3d11 upload failed: {exc}")
            glUnmapBuffer(GL_PIXEL_PACK_BUFFER)
        glBindBuffer(GL_PIXEL_PACK_BUFFER, 0)

    def _render_fps_overlay(self, eye_index, mgl_fbo, vp_mat):
        """Render the FPS/latency text quad (head-relative or left-controller-attached)."""
        if self.screen_height is None:
            return

        now = time.perf_counter()
        
        # Update cached values once per second
        if now - self._last_overlay_update >= 1.0:
            self._cached_actual_fps    = self.actual_fps
            self._cached_sbs_fps       = self.sbs_fps
            self._cached_latency       = self.total_latency
            self._cached_screen_width  = self.screen_width
            self._cached_screen_height = self.screen_height if self.screen_height is not None else 0.0
            self._cached_screen_dist   = self.screen_distance
            self._cached_screen_curved = self._screen_curved
            self._cached_depth_ratio   = self.depth_ratio
            self._cached_vr_res        = self._swapchain_sizes.get(0, (0, 0))
            self._cached_sbs_res       = self.frame_size
            self._last_overlay_update  = now

            # Rebuild text texture — left eye only
            if eye_index == 0 and self.font is not None:
                ow, oh = self._overlay_tex_size
                img  = Image.new('RGBA', (ow, oh), (0, 0, 0, 0))
                draw = ImageDraw.Draw(img)

                # Rounded dark-grey background
                draw.rounded_rectangle(
                    [0, 0, ow - 1, oh - 1],
                    radius=14,
                    fill=(32, 32, 36, 210),
                )

                # Palette
                C_LABEL = (150, 158, 185, 255)   # dim blue-grey for section labels
                C_GREEN = (  0, 230,  90, 255)   # Performance values
                C_CYAN  = (  0, 210, 230, 255)   # 3D Display values
                C_AMBER = (255, 190,  40, 255)   # Resolution values
                bfont   = self.bold_font or self.font   # bold for labels

                PAD    = 14
                ROW0   = 22   # y baseline: 3 rows centred in 150px
                ROW1   = 56
                ROW2   = 90

                # Compute VAL_X from the widest label so all values align
                labels = ["[Performance]", "[3D Display]", "[Resolution]"]
                try:
                    max_lw = max(int(draw.textlength(l, font=bfont)) for l in labels)
                except AttributeError:
                    max_lw = max(
                        (int(bfont.getsize(l)[0]) if hasattr(bfont, 'getsize') else 190)
                        for l in labels
                    )
                VAL_X = PAD + max_lw + 10

                def _draw_row(y, label, label_color, value, value_color):
                    draw.text((PAD, y), label, font=bfont, fill=label_color)
                    draw.text((VAL_X, y), value, font=self.font, fill=value_color)

                lat_str   = f"{self._cached_latency:.0f}ms" if self._cached_latency > 0 else "—"
                fps_str   = (f"XR {self._cached_actual_fps:.0f} FPS"
                             f"   SBS {self._cached_sbs_fps:.0f} FPS"
                             f"   Latency {lat_str}")
                _draw_row(ROW0, "[Performance]", C_LABEL, fps_str, C_GREEN)
                scr_str   = (f"{self._cached_screen_width:.2f}"
                             f" x {self._cached_screen_height:.2f} m"
                             f"  @  {self._cached_screen_dist:.2f} m"
                             f"   Depth {self._cached_depth_ratio:.2f}")
                _draw_row(ROW1, "[3D Display]", C_LABEL, scr_str, C_CYAN)

                vw, vh  = self._cached_vr_res
                sw, sh  = self._cached_sbs_res
                res_str = f"XR {vw}x{vh}/eye   Screen {sw}x{sh}"
                _draw_row(ROW2, "[Resolution]", C_LABEL, res_str, C_AMBER)

                data = np.flipud(np.array(img, dtype=np.uint8))
                self._overlay_tex.write(data.tobytes())

        OVERLAY_H = 0.075  # world-space height (3 rows, shrunk from 0.10)
        ow, oh    = self._overlay_tex_size
        OVERLAY_W = OVERLAY_H * (ow / oh)

        panel_pos = None
        panel_fwd = None
        panel_up  = None

        # Try left-controller attachment first; fall back to head-relative
        if self._grip_mat_l is not None and self._aim_mat_l is not None:
            # Grip axes in world space (columns of grip_mat)
            grip_right = self._grip_mat_l[:3, 0].astype('f8')
            grip_up    = self._grip_mat_l[:3, 1].astype('f8')
            grip_fwd   = self._grip_mat_l[:3, 2].astype('f8')
            grip_right /= np.linalg.norm(grip_right) + 1e-10
            grip_up    /= np.linalg.norm(grip_up) + 1e-10
            grip_fwd   /= np.linalg.norm(grip_fwd) + 1e-10

            # Laser forward direction (same as _laser_beam_setup)
            fwd_w = -self._aim_mat_l[:3, 2].astype('f8')
            right_w = self._aim_mat_l[:3, 0].astype('f8')
            _ang = math.radians(12); _ca, _sa = math.cos(_ang), math.sin(_ang)
            _k = right_w / (np.linalg.norm(right_w) + 1e-10)
            laser_fwd = fwd_w * _ca + np.cross(_k, fwd_w) * _sa + _k * np.dot(_k, fwd_w) * (1 - _ca)
            laser_fwd /= np.linalg.norm(laser_fwd) + 1e-10

            # Laser start point (same origin as _laser_beam_setup)
            grip_pos = self._grip_mat_l[:3, 3].astype('f8')
            laser_origin = grip_pos + grip_up * 0.020 + laser_fwd * 0.11

            # Panel faces user: blend grip_up (button-surface normal) with
            # toward_user (-laser_fwd). Both are controller-relative → tracks.
            toward_user = (-laser_fwd).astype('f8')
            panel_fwd = grip_up + toward_user
            panel_fwd /= np.linalg.norm(panel_fwd) + 1e-10
            panel_up  = grip_up.copy()

            # Pre-compute orthonormal basis
            _pr = np.cross(panel_up, panel_fwd)
            _pr /= np.linalg.norm(_pr) + 1e-10
            _pu2 = np.cross(panel_fwd, _pr)
            _pu2 /= np.linalg.norm(_pu2) + 1e-10

            # Bottom edge midpoint at laser_origin + offset along panel normal.
            # Top edge fixed at old 0.10 height so shrinking leaves gap below.
            PANEL_OFFSET = 0.05
            _top_ref = 0.10  # original OVERLAY_H, top edge anchor
            panel_pos = laser_origin + panel_fwd * PANEL_OFFSET + _pu2 * (_top_ref - OVERLAY_H / 2.0)

        if panel_pos is None and self._head_pos_w is not None and self._head_fwd_w is not None:
            # Fallback: 1m in front of head, 0.15m below. Panel faces toward user.
            hx, hy, hz = self._head_pos_w
            fx, fy, fz = self._head_fwd_w
            panel_pos = np.array([hx + fx * 1.0, hy + fy * 1.0 - 0.15, hz + fz * 1.0], dtype='f8')
            panel_fwd = np.array([-fx, -fy, -fz], dtype='f8')
            panel_up  = np.array([0.0, 1.0, 0.0], dtype='f8')

        if panel_pos is not None:
            # Build T @ R @ S: scale -> rotate -> translate
            S = np.diag([OVERLAY_W/2.0, OVERLAY_H/2.0, 1.0, 1.0]).astype(np.float32)
            panel_right = np.cross(panel_up, panel_fwd)
            panel_right /= np.linalg.norm(panel_right) + 1e-10
            panel_up2 = np.cross(panel_fwd, panel_right)
            panel_up2 /= np.linalg.norm(panel_up2) + 1e-10
            R = np.eye(4, dtype=np.float32)
            R[:3, 0] = panel_right.astype(np.float32)
            R[:3, 1] = panel_up2.astype(np.float32)
            R[:3, 2] = panel_fwd.astype(np.float32)
            T = np.eye(4, dtype=np.float32)
            T[0, 3] = panel_pos[0]; T[1, 3] = panel_pos[1]; T[2, 3] = panel_pos[2]
            mvp = vp_mat @ T @ R @ S
        else:
            mvp = vp_mat  # fallback

        mgl_fbo.use()
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        self._overlay_tex.use(location=2)
        self._overlay_prog['u_mvp'].write(mvp.T.tobytes())
        self._overlay_vao.render(moderngl.TRIANGLE_STRIP)
        self.ctx.disable(moderngl.BLEND)

    def _render_depth_osd(self, eye_index, mgl_fbo, vp_mat):
        """Floating depth-ratio indicator panel.

        Appears when depth_ratio changes; fades out automatically.
        Floats in front of the screen; distance from screen grows with depth_ratio.
        Style matches the FPS status panel (dark rounded rectangle).
        """
        if self._depth_osd_tex is None or self.screen_height is None:
            return

        now = time.perf_counter()

        # Detect change and (re)trigger on left eye only to avoid double writes
        if eye_index == 0:
            cur_val = round(self.depth_ratio, 3)
            if cur_val != self._depth_osd_last_val:
                self._depth_osd_last_val = cur_val
                self._depth_osd_show_t   = now
                self._depth_osd_alpha    = 1.0

                # Rebuild PIL texture
                if self.font is not None:
                    dw, dh = self._depth_osd_tex_size
                    img  = Image.new('RGBA', (dw, dh), (0, 0, 0, 0))
                    draw = ImageDraw.Draw(img)
                    draw.rounded_rectangle(
                        [0, 0, dw - 1, dh - 1],
                        radius=12,
                        fill=(32, 32, 36, 210),
                    )
                    label = "Depth"
                    value = f"{cur_val:.2f}"
                    bfont = self.bold_font or self.font
                    C_LABEL = (150, 158, 185, 255)
                    C_VALUE = (  0, 210, 230, 255)
                    PAD = 12
                    cy  = (dh - 32) // 2
                    draw.text((PAD, cy), label, font=bfont, fill=C_LABEL)
                    try:
                        lw = int(draw.textlength(label, font=bfont))
                    except AttributeError:
                        lw = int(bfont.getsize(label)[0]) if hasattr(bfont, 'getsize') else 60
                    draw.text((PAD + lw + 8, cy), value, font=self.font, fill=C_VALUE)
                    data = np.flipud(np.array(img, dtype=np.uint8))
                    self._depth_osd_tex.write(data.tobytes())

        # Fade: hold for 1.5 s then decay over 0.8 s
        HOLD  = 1.5
        DECAY = 0.8
        elapsed = now - self._depth_osd_show_t
        if elapsed < HOLD:
            alpha = 1.0
        elif elapsed < HOLD + DECAY:
            alpha = 1.0 - (elapsed - HOLD) / DECAY
        else:
            alpha = 0.0
        self._depth_osd_alpha = alpha

        if alpha <= 0.0:
            return

        # Position: centred horizontally, vertically above FPS bar.
        # Distance from the screen surface scales with depth_ratio so the panel
        # visually "floats out" as the user increases depth.
        OSD_H = 0.06
        dw, dh = self._depth_osd_tex_size
        OSD_W  = OSD_H * (dw / dh)

        # Floats above-centre of the screen; z-offset grows with depth_ratio
        BASE_Z_EXTRA = 0.05   # extra forward offset at depth_ratio = 0
        SCALE_Z      = 0.028  # 0.8× old max (0.41 m @ depth=3) spread over new range 0–10
        z_extra = BASE_Z_EXTRA + self.depth_ratio * SCALE_Z
        dist = self.screen_distance - z_extra   # smaller = closer to viewer

        cy_  = math.cos(self.screen_yaw);   sy_ = math.sin(self.screen_yaw)
        cp   = math.cos(self.screen_pitch); sp  = math.sin(self.screen_pitch)
        rot_y = np.array([
            [ cy_,  0, sy_, 0],
            [  0,   1,   0, 0],
            [-sy_,  0, cy_, 0],
            [  0,   0,   0, 1],
        ], dtype=np.float32)
        rot_x = np.array([
            [1,  0,   0, 0],
            [0,  cp, -sp, 0],
            [0,  sp,  cp, 0],
            [0,  0,   0,  1],
        ], dtype=np.float32)

        y_pos = self.screen_pan_y + self.screen_height / 2.0 + 0.04 + OSD_H / 2.0
        osd_model = np.array([
            [OSD_W / 2, 0,          0, self.screen_pan_x],
            [0,         OSD_H / 2,  0, y_pos            ],
            [0,         0,          1, -dist            ],
            [0,         0,          0, 1                ],
        ], dtype=np.float32)

        mvp = vp_mat @ rot_y @ rot_x @ osd_model

        mgl_fbo.use()
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        self._depth_osd_tex.use(location=2)
        self._overlay_prog['u_mvp'].write(mvp.T.tobytes())
        self._overlay_prog['u_alpha'].value = alpha
        self._depth_osd_vao.render(moderngl.TRIANGLE_STRIP)
        self._overlay_prog['u_alpha'].value = 1.0   # restore for other overlays
        self.ctx.disable(moderngl.BLEND)

    def _render_screen_osd(self, eye_index, mgl_fbo, vp_mat):
        """Floating size+distance indicator shown while right grip + right stick adjusts."""
        if self._screen_osd_tex is None or self.screen_height is None:
            return

        now = time.perf_counter()

        # Rebuild texture on left eye whenever values change
        if eye_index == 0 and self.font is not None:
            cur_key = (round(self.screen_width, 2), round(self.screen_distance, 2))
            if cur_key != self._screen_osd_last_key:
                self._screen_osd_last_key = cur_key
                sw, sh = self._screen_osd_tex_size
                img  = Image.new('RGBA', (sw, sh), (0, 0, 0, 0))
                draw = ImageDraw.Draw(img)
                draw.rounded_rectangle(
                    [0, 0, sw - 1, sh - 1],
                    radius=12,
                    fill=(32, 32, 36, 210),
                )
                bfont   = self.bold_font or self.font
                C_LABEL = (150, 158, 185, 255)
                C_VALUE = (  0, 210, 230, 255)
                PAD = 12
                GAP = 8
                cy  = (sh - 32) // 2

                h = self.screen_height if self.screen_height is not None else 0.0

                def _tw(text, font):
                    try:
                        return int(draw.textlength(text, font=font))
                    except AttributeError:
                        return int(font.getsize(text)[0]) if hasattr(font, 'getsize') else 80

                # "Size" label + value
                size_lbl = "Size"
                size_val = f"{self.screen_width:.2f} × {h:.2f} m"
                draw.text((PAD, cy), size_lbl, font=bfont, fill=C_LABEL)
                x = PAD + _tw(size_lbl, bfont) + GAP
                draw.text((x, cy), size_val, font=self.font, fill=C_VALUE)
                x += _tw(size_val, self.font) + GAP * 3

                # "Dist" label (same grey as "Size") + value
                dist_lbl = "Dist"
                dist_val = f"{self.screen_distance:.2f} m"
                draw.text((x, cy), dist_lbl, font=bfont, fill=C_LABEL)
                x += _tw(dist_lbl, bfont) + GAP
                draw.text((x, cy), dist_val, font=self.font, fill=C_VALUE)

                data = np.flipud(np.array(img, dtype=np.uint8))
                self._screen_osd_tex.write(data.tobytes())

        # Fade: hold 1.5 s then decay over 0.8 s (same rhythm as depth OSD)
        HOLD  = 1.5
        DECAY = 0.8
        elapsed = now - self._screen_osd_show_t
        if elapsed < HOLD:
            alpha = 1.0
        elif elapsed < HOLD + DECAY:
            alpha = 1.0 - (elapsed - HOLD) / DECAY
        else:
            alpha = 0.0

        if alpha <= 0.0:
            return

        # Place panel above the screen, stacked one slot higher than the depth OSD
        OSD_H = 0.06
        sw, sh = self._screen_osd_tex_size
        OSD_W  = OSD_H * (sw / sh)

        cy_  = math.cos(self.screen_yaw);   sy_ = math.sin(self.screen_yaw)
        cp   = math.cos(self.screen_pitch); sp  = math.sin(self.screen_pitch)
        rot_y = np.array([
            [ cy_,  0, sy_, 0],
            [  0,   1,   0, 0],
            [-sy_,  0, cy_, 0],
            [  0,   0,   0, 1],
        ], dtype=np.float32)
        rot_x = np.array([
            [1,  0,   0, 0],
            [0,  cp, -sp, 0],
            [0,  sp,  cp, 0],
            [0,  0,   0,  1],
        ], dtype=np.float32)

        y_pos = self.screen_pan_y + self.screen_height / 2.0 + 0.04 + OSD_H / 2.0
        osd_model = np.array([
            [OSD_W / 2, 0,          0, self.screen_pan_x    ],
            [0,         OSD_H / 2,  0, y_pos                ],
            [0,         0,          1, -self.screen_distance ],
            [0,         0,          0, 1                     ],
        ], dtype=np.float32)

        mvp = vp_mat @ rot_y @ rot_x @ osd_model

        mgl_fbo.use()
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        self._screen_osd_tex.use(location=2)
        self._overlay_prog['u_mvp'].write(mvp.T.tobytes())
        self._overlay_prog['u_alpha'].value = alpha
        self._screen_osd_vao.render(moderngl.TRIANGLE_STRIP)
        self._overlay_prog['u_alpha'].value = 1.0
        self.ctx.disable(moderngl.BLEND)

    def _update_aim_poses(self, display_time):
        """Locate both controller aim spaces and cache their world-space 4×4 matrices."""
        now = time.perf_counter()
        for space, mat_attr, prev_attr, move_attr in [
            (self._aim_space_l, "_aim_mat_l", "_laser_prev_mat_l", "_laser_last_move_l"),
            (self._aim_space_r, "_aim_mat_r", "_laser_prev_mat_r", "_laser_last_move_r"),
        ]:
            if space is None:
                setattr(self, mat_attr, None)
                continue
            try:
                loc = xr.locate_space(space, self._xr_space, display_time)
                if loc.location_flags & xr.SPACE_LOCATION_POSITION_VALID_BIT:
                    R = _xr_quat_to_mat4(loc.pose.orientation)
                    R[:3, 3] = [loc.pose.position.x, loc.pose.position.y, loc.pose.position.z]
                    setattr(self, mat_attr, R)
                    # Compare position+orientation to previous pose to detect motion
                    prev = getattr(self, prev_attr)
                    if prev is not None:
                        pos_delta = float(np.linalg.norm(R[:3, 3] - prev[:3, 3]))
                        # Rotation difference via Frobenius norm of delta rotation matrix
                        rot_delta = float(np.linalg.norm(R[:3, :3] - prev[:3, :3]))
                        if pos_delta > self._LASER_MOVE_THRESH or rot_delta > self._LASER_MOVE_THRESH:
                            setattr(self, move_attr, now)
                    else:
                        setattr(self, move_attr, now)
                    setattr(self, prev_attr, R.copy())
                else:
                    setattr(self, mat_attr, None)
            except Exception:
                setattr(self, mat_attr, None)

    def _update_grip_poses(self, display_time):
        """Locate controller grip spaces and cache 4x4 world-space matrices.
        Controller 3D models are placed at the grip center (aim pose is at the tracking ring).
        Also update movement timestamps for 5-second idle auto-hide."""
        now = time.perf_counter()
        for space, mat_attr, move_attr in [
            (self._grip_space_l, "_grip_mat_l", "_laser_last_move_l"),
            (self._grip_space_r, "_grip_mat_r", "_laser_last_move_r"),
        ]:
            if space is None:
                setattr(self, mat_attr, None)
                continue
            try:
                loc = xr.locate_space(space, self._xr_space, display_time)
                if loc.location_flags & xr.SPACE_LOCATION_POSITION_VALID_BIT:
                    R = _xr_quat_to_mat4(loc.pose.orientation)
                    R[:3, 3] = [loc.pose.position.x, loc.pose.position.y, loc.pose.position.z]
                    # Detect movement
                    prev = getattr(self, mat_attr)
                    if prev is not None:
                        delta = float(np.linalg.norm(R[:3, 3] - prev[:3, 3]))
                        if delta > self._LASER_MOVE_THRESH:
                            setattr(self, move_attr, now)
                    else:
                        setattr(self, move_attr, now)
                    setattr(self, mat_attr, R)
                else:
                    setattr(self, mat_attr, None)
            except Exception:
                setattr(self, mat_attr, None)

    def _reset_screen_to_gaze(self, show_border=False):
        """Instantly snap the screen to 2 m in front of the current head gaze.

        Size and shape are preserved — only position/orientation are updated.
        """
        RESET_DIST = 2.0
        self._anim_target_pan_x = None  # cancel any stale animation
        if self._head_pos_w is not None and self._head_fwd_w is not None:
            hx, hy, hz = self._head_pos_w
            fx, fy, fz = self._head_fwd_w
            flen = math.sqrt(fx*fx + fy*fy + fz*fz)
            if flen > 1e-4:
                fx /= flen; fy /= flen; fz /= flen
            else:
                fx, fy, fz = 0.0, 0.0, -1.0
            # World-space target point
            tx = hx + fx * RESET_DIST
            ty = hy + fy * RESET_DIST
            tz = hz + fz * RESET_DIST
            horiz = math.sqrt(fx*fx + fz*fz)
            yaw   = math.atan2(-fx, -fz) if horiz > 1e-4 else self.screen_yaw
            pitch = math.asin(max(-0.999, min(0.999, fy)))
            # Invert model matrix (rot_y @ rot_x then translate) to get local coords
            cy, sy_ = math.cos(yaw),   math.sin(yaw)
            cp, sp  = math.cos(pitch), math.sin(pitch)
            # Inverse rot_y
            x1 =  cy * tx - sy_ * tz
            y1 =  ty
            z1 =  sy_ * tx + cy * tz
            # Inverse rot_x
            x2 =  x1
            y2 =  cp * y1 + sp * z1
            z2 = -sp * y1 + cp * z1
            self.screen_pan_x    = x2
            self.screen_pan_y    = y2
            self.screen_distance = -z2
            self.screen_yaw      = yaw
            self.screen_pitch    = pitch
        else:
            self.screen_distance = RESET_DIST
            self.screen_pan_x    = 0.0
            self.screen_pan_y    = float(self._initial_head_y)
            self.screen_pitch    = 0.0
            self.screen_yaw      = 0.0
        if show_border:
            self._border_alpha  = 1.0
            self._border_idle_t = time.perf_counter()
        if self._keyboard_visible:
            self._anchor_keyboard_below_screen()

    def _tick_screen_anim(self, dt):
        """Exponential-decay glide toward the animation target set by _reset_screen_to_gaze.

        Uses a critically-damped-style lerp: alpha = 1 - exp(-k * dt), which gives
        frame-rate-independent smoothing.  k controls speed: higher = snappier.
        Clears targets once the screen is close enough to avoid infinite ticking.
        """
        if self._anim_target_pan_x is None:
            return

        K     = 6.0   # decay constant: ~63% of the gap closed per 1/K seconds
        alpha = 1.0 - math.exp(-K * max(dt, 1e-4))

        def _lerp(a, b): return a + alpha * (b - a)
        def _lerp_angle(a, b):
            # Shortest-path lerp for angles
            d = (b - a + math.pi) % (2 * math.pi) - math.pi
            return a + alpha * d

        self.screen_pan_x    = _lerp(self.screen_pan_x,    self._anim_target_pan_x)
        self.screen_pan_y    = _lerp(self.screen_pan_y,    self._anim_target_pan_y)
        self.screen_distance = _lerp(self.screen_distance, self._anim_target_distance)
        self.screen_yaw      = _lerp_angle(self.screen_yaw,   self._anim_target_yaw)
        self.screen_pitch    = _lerp_angle(self.screen_pitch, self._anim_target_pitch)

        # Stop animating once close enough (< 1 mm / 0.01°)
        close = (
            abs(self.screen_pan_x    - self._anim_target_pan_x)    < 0.001 and
            abs(self.screen_pan_y    - self._anim_target_pan_y)    < 0.001 and
            abs(self.screen_distance - self._anim_target_distance) < 0.001 and
            abs((self.screen_yaw   - self._anim_target_yaw   + math.pi) % (2*math.pi) - math.pi) < 0.0002 and
            abs((self.screen_pitch - self._anim_target_pitch + math.pi) % (2*math.pi) - math.pi) < 0.0002
        )
        if close:
            self.screen_pan_x    = self._anim_target_pan_x
            self.screen_pan_y    = self._anim_target_pan_y
            self.screen_distance = self._anim_target_distance
            self.screen_yaw      = self._anim_target_yaw
            self.screen_pitch    = self._anim_target_pitch
            self._anim_target_pan_x = None   # clear — animation complete

    def _apply_preset(self, index):
        """Apply screen preset combination."""
        name, width, dist = self._screen_presets[index]
        self.screen_width    = width
        self.screen_distance = dist
        self.screen_height   = None
        self.screen_pitch    = 0.0   # perpendicular to floor
        self._screen_curved  = False
        self._preset_index            = index
        self._preset_name_overlay     = f"{name}  {width:.2f}m / {dist:.2f}m"
        self._last_overlay_update     = 0.0  # force overlay refresh
        self._border_alpha  = 1.0
        self._border_idle_t = time.perf_counter()

    def _reset_screen_to_default(self, show_border=False):
        """Reset screen to upright default: 2 m ahead horizontally, perpendicular to floor.

        Screen is always vertical (pitch=0) and faces the user's current horizontal
        forward direction. Centre height matches the headset eye height recorded at
        session start so the screen sits comfortably in front of the user.
        Called at session start and by the Y button.
        """
        RESET_DIST = 2.0
        self.screen_width    = 2.4
        self.screen_height   = None
        self.screen_pitch    = 0.0   # always vertical — perpendicular to floor
        self._screen_curved  = False
        if self._head_pos_w is not None and self._head_fwd_w is not None:
            hx, _, hz = self._head_pos_w
            fx, _, fz = self._head_fwd_w
            # Project forward onto the horizontal plane so the screen stands vertical
            horiz = math.sqrt(fx * fx + fz * fz)
            if horiz > 1e-4:
                fx /= horiz; fz /= horiz
            else:
                fx, fz = 0.0, -1.0
            # screen_pan_x / screen_distance live in the screen's LOCAL frame (the
            # model matrix applies rot_y(yaw) before the translation).  Inverting
            # _screen_world_pos at pitch=0 gives the correct local-frame values so
            # the world-space centre lands exactly RESET_DIST m in front of the head:
            #   world_x = pan_x·cos(yaw) − distance·sin(yaw)   = hx + fx·RESET_DIST
            #   world_z =−pan_x·sin(yaw) − distance·cos(yaw)   = hz + fz·RESET_DIST
            # Solving (with cy=−fz, sy_=−fx from yaw=atan2(−fx,−fz)):
            #   screen_distance = hx·fx + hz·fz + RESET_DIST
            #   screen_pan_x    = hz·fx − hx·fz
            self.screen_distance = hx * fx + hz * fz + RESET_DIST
            self.screen_pan_x    = hz * fx - hx * fz
            self.screen_pan_y    = float(self._initial_head_y)
            self.screen_yaw      = math.atan2(-fx, -fz)   # face the user horizontally
        else:
            self.screen_distance = RESET_DIST
            self.screen_pan_x    = 0.0
            self.screen_pan_y    = float(self._initial_head_y)
            self.screen_yaw      = 0.0
        if show_border:
            self._border_alpha  = 1.0
            self._border_idle_t = time.perf_counter()
        if self._keyboard_visible:
            self._anchor_keyboard_below_screen()

    def _screen_uv_to_world(self, u, v):
        """Convert a screen UV in [0,1]² to its world-space 3-D position on the screen plane."""
        sh = self.screen_height
        if sh is None:
            fw, fh = self.frame_size
            sh = self.screen_width * (fh / fw if fw > 0 else 9.0 / 16.0)
        cp  = math.cos(self.screen_pitch); sp  = math.sin(self.screen_pitch)
        cy  = math.cos(self.screen_yaw);   sy_ = math.sin(self.screen_yaw)
        screen_pos = self._screen_world_pos(cy, sy_, cp, sp)
        r_ax = np.array([cy,      0.0,    -sy_    ], dtype='f8')
        u_ax = np.array([sp*sy_,  cp,      sp*cy  ], dtype='f8')
        return screen_pos + r_ax * ((u - 0.5) * self.screen_width) + u_ax * ((v - 0.5) * sh)

    def _screen_world_pos(self, cy, sy_, cp, sp):
        """Return the world-space centre of the screen as a float64 numpy vec3.

        screen_pan_x/y and screen_distance live in the pre-rotation (local) frame:
        the model matrix applies rot_y @ rot_x before the translation, so the actual
        world centre is rot_y(yaw) @ rot_x(pitch) @ [pan_x, pan_y, -dist].
        Using the raw pre-rotation values as a world position is only correct when
        yaw == pitch == 0 and is the root cause of cursor/laser misalignment at any
        other orientation.
        """
        lx, ly, lz = self.screen_pan_x, self.screen_pan_y, -self.screen_distance
        # Apply rot_x (pitch) first — matches _build_model_mat4's rot_y @ rot_x order
        ix = lx
        iy = ly * cp - lz * sp
        iz = ly * sp + lz * cp
        # Then apply rot_y (yaw)
        return np.array([ix * cy + iz * sy_, iy, -ix * sy_ + iz * cy], dtype='f8')

    def _laser_screen_hit_dist(self, ctrl_pos, fwd_w):
        """Return the distance along fwd_w where the aim ray hits the visible screen rect.

        Returns BEAM_MAX (5 m) if the ray misses the screen rectangle entirely or
        is parallel to it — so the laser only clips when it actually hits the screen.
        """
        BEAM_MAX = 30.0   # long enough to look infinite in any room-scale space
        # Compute height inline during resize (when screen_height is temporarily None)
        sh = self.screen_height
        if sh is None:
            fw, fh = self.frame_size
            sh = self.screen_width * (fh / fw if fw > 0 else 9.0 / 16.0)
        # Protect against degenerate screen dimensions
        safe_w = max(self.screen_width, 1e-6)
        safe_h = max(sh, 1e-6)

        cp  = math.cos(self.screen_pitch); sp  = math.sin(self.screen_pitch)
        cy  = math.cos(self.screen_yaw);   sy_ = math.sin(self.screen_yaw)
        screen_n   = np.array([cp * sy_, -sp, cp * cy], dtype='f8')
        screen_pos = self._screen_world_pos(cy, sy_, cp, sp)
        denom = float(np.dot(screen_n, fwd_w))
        if abs(denom) < 1e-6:
            return BEAM_MAX
        t = float(np.dot(screen_n, screen_pos - ctrl_pos)) / denom
        if t < 0.01:
            return BEAM_MAX   # screen is behind controller
        # Check hit is within the actual screen rectangle (not the infinite plane)
        hit   = ctrl_pos + fwd_w * t
        diff  = hit - screen_pos
        r_ax  = np.array([cy, 0.0, -sy_], dtype='f8')
        u_ax  = np.array([sp*sy_, cp, sp*cy], dtype='f8')
        loc_x = float(np.dot(diff, r_ax))
        loc_y = float(np.dot(diff, u_ax))
        if abs(loc_x) <= safe_w / 2.0 and abs(loc_y) <= safe_h / 2.0:
            return max(0.01, t - 0.005)   # hit within screen — stop 5 mm before
        return BEAM_MAX   # outside screen rectangle — let laser go to max

    def _keyboard_laser_hit_dist(self, ctrl_pos, fwd_w):
        """Return the distance along fwd_w where the ray hits the keyboard quad.

        Returns BEAM_MAX if the keyboard is hidden, the ray is parallel to it,
        or the hit point falls outside the keyboard rectangle.
        """
        BEAM_MAX = 30.0
        if not self._keyboard_visible or not self._keyboard_keys:
            return BEAM_MAX
        cp = math.cos(self._keyboard_pitch); sp = math.sin(self._keyboard_pitch)
        cy = math.cos(self._keyboard_yaw);   sy = math.sin(self._keyboard_yaw)
        kb_x = np.array([ cy,      0.0,  -sy      ], dtype='f8')
        kb_y = np.array([ sy * sp,  cp,   cy * sp ], dtype='f8')
        kb_n = np.array([ sy * cp, -sp,   cy * cp ], dtype='f8')
        kb_pos = np.array([self._keyboard_pan_x,
                           self._keyboard_pan_y,
                           -self._keyboard_distance], dtype='f8')
        denom = float(np.dot(kb_n, fwd_w))
        if abs(denom) < 1e-6:
            return BEAM_MAX
        t = float(np.dot(kb_n, kb_pos - ctrl_pos)) / denom
        if t < 0.01:
            return BEAM_MAX
        hit  = ctrl_pos + fwd_w * t
        diff = hit - kb_pos
        lx = float(np.dot(diff, kb_x))
        ly = float(np.dot(diff, kb_y))
        if abs(lx) <= self._keyboard_width / 2.0 and abs(ly) <= self._keyboard_height / 2.0:
            return max(0.01, t - 0.005)
        return BEAM_MAX

    def _overlay_panel_hit_dist(self, ctrl_pos, fwd_w):
        """Return the distance along fwd_w where the ray hits the FPS overlay panel.

        The panel sits just below the screen, shares the same yaw/pitch rotation,
        and has the same surface normal. Returns BEAM_MAX if the overlay is hidden,
        screen_height is unknown, the ray is parallel, or the hit misses the rect.
        """
        BEAM_MAX = 30.0
        if not self._fps_overlay_visible or self.screen_height is None:
            return BEAM_MAX

        GAP       = 0.05
        OVERLAY_H = 0.12
        ow, oh    = self._overlay_tex_size
        OVERLAY_W = OVERLAY_H * (ow / oh)

        # Panel local-space centre (before yaw/pitch rotation) — matches _render_fps_overlay
        ly_local = self.screen_pan_y - self.screen_height / 2.0 - GAP - OVERLAY_H / 2.0

        cp  = math.cos(self.screen_pitch); sp  = math.sin(self.screen_pitch)
        cy  = math.cos(self.screen_yaw);   sy_ = math.sin(self.screen_yaw)

        # Panel normal is identical to the screen normal
        panel_n = np.array([cp * sy_, -sp, cp * cy], dtype='f8')

        # Rotate the panel's local centre into world space (same as _screen_world_pos)
        lx, ly, lz = self.screen_pan_x, ly_local, -self.screen_distance
        ix  =  lx
        iy  =  ly * cp - lz * sp
        iz  =  ly * sp + lz * cp
        panel_pos = np.array([ix * cy + iz * sy_, iy, -ix * sy_ + iz * cy], dtype='f8')

        denom = float(np.dot(panel_n, fwd_w))
        if abs(denom) < 1e-6:
            return BEAM_MAX
        t = float(np.dot(panel_n, panel_pos - ctrl_pos)) / denom
        if t < 0.01:
            return BEAM_MAX

        hit  = ctrl_pos + fwd_w * t
        diff = hit - panel_pos
        r_ax = np.array([cy,      0.0,    -sy_    ], dtype='f8')
        u_ax = np.array([sp*sy_,  cp,      sp*cy  ], dtype='f8')
        loc_x = float(np.dot(diff, r_ax))
        loc_y = float(np.dot(diff, u_ax))
        if abs(loc_x) <= OVERLAY_W / 2.0 and abs(loc_y) <= OVERLAY_H / 2.0:
            return max(0.01, t - 0.005)
        return BEAM_MAX

    def _laser_beam_setup(self):
        """Return shared beam calculations (position, direction) for each ray."""
        now = time.perf_counter()
        beams = []
        for aim_mat, grip_mat, last_move_attr, ctrl_name in [
            (self._aim_mat_l, self._grip_mat_l, "_laser_last_move_l", 'left'),
            (self._aim_mat_r, self._grip_mat_r, "_laser_last_move_r", 'right'),
        ]:
            if aim_mat is None:
                continue
            if (now - getattr(self, last_move_attr)) > self._LASER_HIDE_AFTER:
                continue
            fwd_w = -aim_mat[:3, 2].astype('f8')
            right_w = aim_mat[:3, 0].astype('f8')
            _ang = math.radians(12); _ca, _sa = math.cos(_ang), math.sin(_ang)
            _k = right_w / (np.linalg.norm(right_w) + 1e-10)
            fwd_w = fwd_w * _ca + np.cross(_k, fwd_w) * _sa + _k * np.dot(_k, fwd_w) * (1 - _ca)
            if grip_mat is not None:
                ctrl_pos = (grip_mat[:3, 3] + grip_mat[:3, 1] * 0.020).astype('f8')
            else:
                ctrl_pos = aim_mat[:3, 3].astype('f8')
            ctrl_pos = ctrl_pos + fwd_w * 0.11
            right = aim_mat[:3, 0].astype('f4')
            fwd = fwd_w.astype('f4')
            up = np.cross(right, fwd); up = up / (np.linalg.norm(up) + 1e-10)
            right2 = np.cross(fwd, up)
            beams.append((now, ctrl_name, aim_mat, ctrl_pos, fwd_w, right2, fwd, up))
        return beams

    def _render_lasers(self, mgl_fbo, vp_mat, blend=False):
        """blend=False: opaque rainbow beam; blend=True: semi-transparent hit circles."""
        beams = self._laser_beam_setup()
        if not beams:
            return
        if blend:
            self._render_laser_hit_circles(mgl_fbo, vp_mat, beams)
            return
        mgl_fbo.use()
        for now, ctrl_name, aim_mat, ctrl_pos, fwd_w, right2, fwd, up in beams:
            draw_len = 0.4
            BEAM_R = 0.004
            S = np.diag([BEAM_R, draw_len, BEAM_R, 1.0]).astype('f4')
            R = np.eye(4, dtype='f4'); R[:3, 0] = right2; R[:3, 1] = fwd; R[:3, 2] = up
            T = np.eye(4, dtype='f4'); T[:3, 3] = ctrl_pos.astype('f4')
            beam_mvp = vp_mat @ T @ R @ S
            self._beam_prog['u_mvp'].write(beam_mvp.T.tobytes())
            self._beam_prog['u_time'].value = float(now)
            self._beam_vao.render(moderngl.TRIANGLE_STRIP)
    def _render_laser_hit_circles(self, mgl_fbo, vp_mat, beams):
        mgl_fbo.use()
        for now, ctrl_name, aim_mat, ctrl_pos, fwd_w, right2, fwd, up in beams:
            cursor_uv = self._cursor_uv_l if ctrl_name == 'left' else self._cursor_uv_r
            if (self._cursor_ctrl == ctrl_name and cursor_uv is not None):
                beam_len = max(0.01, float(cursor_uv[2]))
            else:
                beam_len = min(
                    self._laser_screen_hit_dist(ctrl_pos, fwd_w),
                    self._keyboard_laser_hit_dist(ctrl_pos, fwd_w),
                    self._overlay_panel_hit_dist(ctrl_pos, fwd_w),
                )
            if beam_len >= 5.0:
                continue
            HIT_OFFSET = 0.003
            hit_pos = ctrl_pos + fwd_w * (beam_len - HIT_OFFSET)
            STROKE_R = 0.0096; FILL_R = 0.0056
            for radius, color in [(STROKE_R, (0.2, 0.6, 1.0, 0.75)), (FILL_R, (1.0, 1.0, 1.0, 0.75))]:
                model = np.eye(4, dtype='f4')
                model[0, 0] = radius; model[1, 1] = radius
                model[:3, 3] = hit_pos.astype('f4')
                circle_mvp = vp_mat @ model
                self._border_prog['u_mvp'].write(circle_mvp.T.tobytes())
                self._border_prog['u_color'].value = color
                self._circle_vao.render(moderngl.TRIANGLE_FAN)

    def _render_controllers(self, mgl_fbo, vp_mat, view_mat):
        """Render PICO 4 Ultra 3D controller models with Blinn-Phong lighting."""
        now = time.perf_counter()
        controllers = []
        for grip_mat, prims, last_move_attr in [
            (self._grip_mat_l, self._ctrl_prims_l, "_laser_last_move_l"),
            (self._grip_mat_r, self._ctrl_prims_r, "_laser_last_move_r"),
        ]:
            if (now - getattr(self, last_move_attr)) > self._LASER_HIDE_AFTER:
                continue
            if grip_mat is None or not prims:
                continue
            R_t = view_mat[:3, :3].T
            eye_pos = -R_t @ view_mat[:3, 3]
            dist = float(np.linalg.norm(
                grip_mat[:3, 3].astype(np.float64) - eye_pos.astype(np.float64)))
            controllers.append((dist, grip_mat, prims))

        if not controllers:
            return

        controllers.sort(key=lambda x: x[0], reverse=True)
        mgl_fbo.use()

        # Lighting parameters
        # Extract camera position (headset) from view_mat, point light follows head
        view_inv = np.linalg.inv(view_mat)
        cam_pos = view_inv[:3, 3].astype(np.float32)

        # light_color (diffuse) is slightly bluish to look more like the PICO 4's built-in light;
        # ambient_color is dim to avoid washing out the dark controller textures.
        light_color = np.array([0.37, 0.37, 0.40], dtype=np.float32)
        ambient_color = np.array([0.22, 0.22, 0.24], dtype=np.float32)

        for _dist, grip_mat, prims in controllers:
            # Translate first (independent Y/Z), then rotate -30° around X
            T_mat = np.eye(4, dtype=np.float32)
            T_mat[0, 3] = self._x_offset
            T_mat[1, 3] = self._y_offset
            T_mat[2, 3] = self._z_offset

            _ang = math.radians(-20)
            _ca, _sa = math.cos(_ang), math.sin(_ang)
            R_mat = np.eye(4, dtype=np.float32)
            R_mat[1, 1] = _ca; R_mat[1, 2] = -_sa
            R_mat[2, 1] = _sa; R_mat[2, 2] = _ca

            _corr = (R_mat @ T_mat).astype(np.float32)
            model_mat = (grip_mat @ _corr).astype(np.float32)

            # Set common uniforms for the current controller
            # u_model: model → world (grip @ _corr)
            # u_mvp: world → clip (VP only, no model, because shader computes world_pos = u_model * v)
            self._controller_prog['u_mvp'].write(vp_mat.astype(np.float32).T.tobytes())
            self._controller_prog['u_model'].write(model_mat.T.tobytes())
            self._controller_prog['u_light_color'].write((light_color).tobytes())
            self._controller_prog['u_ambient_color'].write((ambient_color).tobytes())
            self._controller_prog['u_camera_pos'].write((cam_pos).tobytes())

            # Sort primitives by triangle count (descending): larger items rendered first (bottom), smaller ones later (top)
            sorted_prims = sorted(prims, key=lambda p: p['tri_count'], reverse=True)

            if self._use_d3d11:
                glFrontFace(GL_CW)

            for prim in sorted_prims:
                tex = self._ctrl_tex_cache.get(prim['tex_id'])
                if tex is not None:
                    tex.use(location=3)
                prim['vao'].render(moderngl.TRIANGLES)

            if self._use_d3d11:
                glFrontFace(GL_CCW)

    def _render_border(self, mgl_fbo, vp_mat):
        """Render a thin solid-color border slightly larger than the screen.

        Flat mode: oversized quad rendered before the screen so it peeks at the edges.
        Curved mode: oversized arc strip matching the curved screen geometry.
        Color is cyan when the user is grabbing/resizing, light grey otherwise.
        """
        if self.screen_height is None:
            return
        alpha = self._border_alpha
        if alpha <= 0.0:
            return

        if self._grabbed:
            color = (0.3, 0.7, 1.0, alpha)
        else:
            color = (0.75, 0.75, 0.75, alpha * 0.9)

        BORDER = 0.008   # metres of extra half-width on each side

        mgl_fbo.use()
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA

        if self._screen_curved and self._curved_border_vao is not None:
            # Build a border arc: same arc geometry but screen_width+2B wide,
            # screen_height+2B tall, and pushed 1 mm behind the screen surface.
            bw = self.screen_width  + 2 * BORDER
            bh = self.screen_height + 2 * BORDER
            params = (bw, bh, self.screen_distance, self.screen_pan_x,
                      self.screen_pan_y, self.screen_yaw, self.screen_pitch)
            if self._curved_border_verts_params != params:
                border_verts = self._build_curved_screen_verts(
                    width_override=bw, height_override=bh, dist_offset=0.001,
                )
                self._curved_border_vbo.write(border_verts.tobytes())
                self._curved_border_verts_params = params
            self._curved_border_prog['u_mvp'].write(vp_mat.T.tobytes())
            self._curved_border_prog['u_color'].value = color
            n_verts = (48 + 1) * 2
            self._curved_border_vao.render(moderngl.TRIANGLE_STRIP, vertices=n_verts)
        else:
            sx = self.screen_width  / 2.0 + BORDER
            sy = self.screen_height / 2.0 + BORDER
            cy  = math.cos(self.screen_yaw);   sy_ = math.sin(self.screen_yaw)
            cp  = math.cos(self.screen_pitch); sp  = math.sin(self.screen_pitch)
            rot_y = np.array([[ cy, 0, sy_, 0], [0, 1, 0, 0], [-sy_, 0, cy, 0], [0, 0, 0, 1]], dtype='f4')
            rot_x = np.array([[1, 0, 0, 0], [0, cp, -sp, 0], [0, sp, cp, 0], [0, 0, 0, 1]], dtype='f4')
            border_model = np.array([
                [sx, 0, 0, self.screen_pan_x],
                [0, sy, 0, self.screen_pan_y],
                [0, 0, 1, -self.screen_distance - 0.001],
                [0, 0, 0, 1],
            ], dtype='f4')
            mvp = vp_mat @ rot_y @ rot_x @ border_model
            self._border_prog['u_mvp'].write(mvp.T.astype('f4').tobytes())
            self._border_prog['u_color'].value = color
            self._border_vao.render(moderngl.TRIANGLE_STRIP)

        self.ctx.disable(moderngl.BLEND)

    def _render_eye(self, eye_index, mgl_fbo, view_mat, proj_mat, flip_y=False):
        """Render one eye's parallax view into the swapchain FBO using world-space MVP.

        Left eye:  u_eye_offset = -ipd/2
        Right eye: u_eye_offset = +ipd/2

        If flip_y is True the projection is Y-flipped so glReadPixels produces
        top-down rows for D3D11, eliminating the CPU row-reversal copy.
        """
        if flip_y:
            proj_mat = proj_mat.copy()
            proj_mat[1, :] = -proj_mat[1, :]  # flip clip-space Y → GL renders upside-down
        sc_w, sc_h = self._swapchain_sizes[eye_index]

        # The swapchain is GL_SRGB8_ALPHA8, but the desktop capture texture is already
        # gamma-encoded.  Disabling GL_FRAMEBUFFER_SRGB prevents AMD (and compliant
        # drivers) from applying a second sRGB encoding pass on write, which would
        # cause pale/washed-out colours.
        glDisable(GL_FRAMEBUFFER_SRGB)

        mgl_fbo.use()
        self.ctx.viewport = (0, 0, sc_w, sc_h)
        bg_a = 1.0
        bg_r, bg_g, bg_b = _BG_COLORS[self._bg_color_idx]
        mgl_fbo.clear(bg_r, bg_g, bg_b, bg_a)

        if not self._screen_visible:
            self.ctx.screen.use()
            return

        if self.color_tex is None or self.depth_tex is None:
            self.ctx.screen.use()
            return

        # Pre-compute view-projection once per eye — all quads multiply their model
        # matrix against this rather than recomputing proj @ view each time.
        vp_mat = proj_mat @ view_mat

        # 1. Border (behind the screen, slightly larger)
        self._render_border(mgl_fbo, vp_mat)
        self.ctx.viewport = (0, 0, sc_w, sc_h)

        # 2. Main screen (flat quad or cylindrical curved arc)
        mgl_fbo.use()
        self.color_tex.use(location=0)
        self.depth_tex.use(location=1)

        eye_sign = -1.0 if eye_index == 0 else 1.0

        if self._screen_curved and self._curved_prog is not None:
            # Curved path: build world-space arc verts, upload, draw with vp_mat only
            prog = self._curved_prog
            params = (self.screen_width, self.screen_height, self.screen_distance,
                      self.screen_pan_x, self.screen_pan_y, self.screen_yaw, self.screen_pitch)
            if self._curved_verts_params != params:
                arc_verts = self._build_curved_screen_verts()
                self._curved_vbo.write(arc_verts.tobytes())
                self._curved_verts_params = params
            prog['u_mvp'].write(vp_mat.T.tobytes())
            prog['u_eye_offset'].value     = eye_sign * self.ipd_uv / 2.0
            prog['u_depth_strength'].value = self.depth_strength * self.depth_ratio
            prog['u_convergence'].value = float(self.convergence)
            n_verts = (48 + 1) * 2
            self._curved_vao.render(moderngl.TRIANGLE_STRIP, vertices=n_verts)
        else:
            # Flat path: standard MVP quad
            model = self._build_model_mat4()
            mvp   = vp_mat @ model
            self.prog['u_mvp'].write(mvp.T.tobytes())
            self.prog['u_eye_offset'].value     = eye_sign * self.ipd_uv / 2.0
            self.prog['u_depth_strength'].value = self.depth_strength * self.depth_ratio
            # Keep convergence in sync — user-driven divergence input updates self.convergence
            # at runtime; pushing it here ensures any external change is reflected per-frame.
            self.prog['u_convergence'].value = float(self.convergence)
            # Render screen WITHOUT alpha blending so the shader's edge alpha is written
            # directly into the swapchain framebuffer. The XR compositor composites those
            # near-zero-alpha edge pixels against the VR background — producing a clean soft
            # edge. With SRC_ALPHA blending the edge pixels would blend against the FBO's
            # opaque black clear, creating a persistent dark halo visible at all times.
            self.quad_vao.render(moderngl.TRIANGLE_STRIP)

        # 3. Keyboard
        if self._keyboard_visible and self._keyboard_tex is not None:
            self.ctx.viewport = (0, 0, sc_w, sc_h)
            self._render_keyboard(mgl_fbo, vp_mat)

        # 5. Depth OSD (floating panel, always checked — method handles its own alpha)
        if self._depth_osd_tex is not None:
            self.ctx.viewport = (0, 0, sc_w, sc_h)
            self._render_depth_osd(eye_index, mgl_fbo, vp_mat)

        # 5b. Screen-info OSD (size + distance, shown while right grip + stick adjusts)
        if self._screen_osd_tex is not None:
            self.ctx.viewport = (0, 0, sc_w, sc_h)
            self._render_screen_osd(eye_index, mgl_fbo, vp_mat)

        # 7. VR Controller models (PICO 4 Ultra)
        if self._ctrl_prims_l or self._ctrl_prims_r:
            self.ctx.viewport = (0, 0, sc_w, sc_h)
            self._render_controllers(mgl_fbo, vp_mat, view_mat)

        # 8. Laser beam (opaque rainbow, rendered on top of controllers)
        self.ctx.viewport = (0, 0, sc_w, sc_h)
        self._render_lasers(mgl_fbo, vp_mat, blend=False)
        self.ctx.viewport = (0, 0, sc_w, sc_h)
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        self._render_lasers(mgl_fbo, vp_mat, blend=True)
        self.ctx.disable(moderngl.BLEND)

        # 9. FPS overlay — topmost UI, occludes laser beams
        if self._fps_overlay_visible and self._overlay_tex is not None:
            self.ctx.viewport = (0, 0, sc_w, sc_h)
            self._render_fps_overlay(eye_index, mgl_fbo, vp_mat)

        self.ctx.screen.use()

    # ------------------------------------------------------------------
    # OpenXR event loop
    # ------------------------------------------------------------------

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
                    print("[OpenXRViewer] Session READY — rendering started")

                elif state in (
                    xr.SessionState.STOPPING,
                    xr.SessionState.LOSS_PENDING,
                    xr.SessionState.EXITING,
                ):
                    xr.end_session(self._xr_session)
                    self._session_running = False
                    print(f"[OpenXRViewer] Session state → {state.name}; rendering paused")

            elif event_type == xr.StructureType.EVENT_DATA_REFERENCE_SPACE_CHANGE_PENDING:
                # Long-pressing the Oculus/home button triggers a playspace recenter,
                # which emits this event.  Reset the screen to face the new forward
                # direction so the virtual display stays comfortably in front of the user.
                self._reset_screen_to_gaze(show_border=True)

            elif event_type == xr.StructureType.EVENT_DATA_INSTANCE_LOSS_PENDING:
                print("[OpenXRViewer] Instance loss pending — shutting down")
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

    def _laser_screen_hit_uv(self, ctrl_pos, fwd_w):
        """Return (u, v, t) where the aim ray hits the screen surface, or None.

        u, v are in [0, 1] (u=0 left, v=0 bottom). t is the along-ray distance.
        Returns None if the ray misses the screen rect or is parallel to it.
        """
        sh = self.screen_height
        if sh is None:
            fw, fh = self.frame_size
            sh = self.screen_width * (fh / fw if fw > 0 else 9.0 / 16.0)
        safe_w = max(self.screen_width, 1e-6)
        safe_h = max(sh, 1e-6)

        cp  = math.cos(self.screen_pitch); sp  = math.sin(self.screen_pitch)
        cy  = math.cos(self.screen_yaw);   sy_ = math.sin(self.screen_yaw)
        screen_n   = np.array([cp * sy_, -sp, cp * cy], dtype='f8')
        screen_pos = self._screen_world_pos(cy, sy_, cp, sp)
        denom = float(np.dot(screen_n, fwd_w))
        if abs(denom) < 1e-6:
            return None
        t = float(np.dot(screen_n, screen_pos - ctrl_pos)) / denom
        if t < 0.01:
            return None
        hit  = ctrl_pos + fwd_w * t
        diff = hit - screen_pos
        r_ax = np.array([cy, 0.0, -sy_],         dtype='f8')
        u_ax = np.array([sp * sy_, cp, sp * cy],  dtype='f8')
        loc_x = float(np.dot(diff, r_ax))
        loc_y = float(np.dot(diff, u_ax))
        if abs(loc_x) <= safe_w / 2.0 and abs(loc_y) <= safe_h / 2.0:
            u = 0.5 + loc_x / safe_w
            v = 0.5 + loc_y / safe_h
            return u, v, t
        return None

    def _keyboard_laser_hit(self, ctrl_pos, fwd_w):
        """Return (key_index, t) if the aim ray hits a key on the virtual keyboard, else (None, None).

        Accounts for the keyboard's tilt (`_keyboard_pitch`) and yaw — the plane is no
        longer axis-aligned, so we project the world hit point back into the keyboard's
        local 2D frame using the rotated basis vectors before testing rect_local bounds.
        """
        if not self._keyboard_keys:
            return None, None
        cp = math.cos(self._keyboard_pitch); sp = math.sin(self._keyboard_pitch)
        cy = math.cos(self._keyboard_yaw);   sy = math.sin(self._keyboard_yaw)
        # Local axes in world (columns of rot_y(yaw) ∘ rot_x(pitch)):
        #   X_local → ( cy,        0,    -sy)
        #   Y_local → ( sy*sp,    cp,     cy*sp)
        #   Z_local → ( sy*cp,   -sp,     cy*cp)   ← surface normal
        kb_x = np.array([ cy,          0.0,    -sy        ], dtype='f8')
        kb_y = np.array([ sy * sp,     cp,      cy * sp   ], dtype='f8')
        kb_n = np.array([ sy * cp,    -sp,      cy * cp   ], dtype='f8')
        kb_pos = np.array([self._keyboard_pan_x,
                           self._keyboard_pan_y,
                           -self._keyboard_distance], dtype='f8')
        denom = float(np.dot(kb_n, fwd_w))
        if abs(denom) < 1e-6:
            return None, None
        t = float(np.dot(kb_n, kb_pos - ctrl_pos)) / denom
        if t < 0.05:
            return None, None
        hit  = ctrl_pos + fwd_w * t
        diff = hit - kb_pos
        lx = float(np.dot(diff, kb_x))
        ly = float(np.dot(diff, kb_y))
        for i, key in enumerate(self._keyboard_keys):
            x0, y0, x1, y1 = key.rect_local
            if x0 <= lx <= x1 and y0 <= ly <= y1:
                return i, t
        return None, None

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
        if sys.platform == "win32":
            class _POINT(ctypes.Structure):
                _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]
            _pt = _POINT()
            ctypes.windll.user32.GetCursorPos(ctypes.byref(_pt))
            _cur_pos = (_pt.x, _pt.y)
            if self._phys_mouse_pos is not None and _cur_pos != self._phys_mouse_pos:
                # Only count as physical movement if the position change wasn't caused
                # by our own VR cursor write — otherwise every laser-driven cursor move
                # would stamp _phys_mouse_last_move and suppress VR cursor control.
                vcp = self._vr_cursor_screen_pos
                if vcp is None or abs(_cur_pos[0] - vcp[0]) > 4 or abs(_cur_pos[1] - vcp[1]) > 4:
                    self._phys_mouse_last_move = time.perf_counter()
            self._phys_mouse_pos = _cur_pos
            if (time.perf_counter() - self._phys_mouse_last_move) < PHYS_TIMEOUT:
                self._cursor_ctrl = None
                self._cursor_smooth_uv = None
                return

        def _beam_origin_dir(aim_mat, grip_mat):
            """Calculate the laser beam's origin point and forward direction in world space."""
            fw = -aim_mat[:3, 2].astype('f8')
            # 向上旋转15度
            right = aim_mat[:3, 0].astype('f8')
            ang = math.radians(12); ca, sa = math.cos(ang), math.sin(ang)
            k = right / (np.linalg.norm(right) + 1e-10)
            fw = fw * ca + np.cross(k, fw) * sa + k * np.dot(k, fw) * (1 - ca)
            # 起点与_laser_beam_setup保持同步
            if grip_mat is not None:
                cp = (grip_mat[:3, 3] + grip_mat[:3, 1] * 0.020).astype('f8')
            else:
                cp = aim_mat[:3, 3].astype('f8')
            cp = cp + fw * 0.11
            return cp, fw

        dw, dh = _get_desktop_size()
        hit_l = hit_r = None
        if self._aim_mat_l is not None:
            cp, fw = _beam_origin_dir(self._aim_mat_l, self._grip_mat_l)
            if not (self._keyboard_visible and self._keyboard_laser_hit(cp, fw)[0] is not None):
                hit_l = self._laser_screen_hit_uv(cp, fw)
        if self._aim_mat_r is not None:
            cp, fw = _beam_origin_dir(self._aim_mat_r, self._grip_mat_r)
            if not (self._keyboard_visible and self._keyboard_laser_hit(cp, fw)[0] is not None):
                hit_r = self._laser_screen_hit_uv(cp, fw)

        self._cursor_uv_l = hit_l if hit_l else None   # (u, v, t) or None
        self._cursor_uv_r = hit_r if hit_r else None   # (u, v, t) or None

        # Pick active controller — right always wins when both lasers hit screen.
        # Avoids ping-ponging: once a controller takes over, the other can't steal.
        prev_ctrl = self._cursor_ctrl
        if hit_r:
            ctrl, u, v = 'right', hit_r[0], hit_r[1]
        elif hit_l:
            ctrl, u, v = 'left', hit_l[0], hit_l[1]
        else:
            self._cursor_ctrl = None
            self._cursor_smooth_uv = None   # reset so next entry doesn't drag
            return

        self._cursor_ctrl = ctrl
        # Reset the smoother on controller swap so we don't slide diagonally to the
        # new pointer location.
        if prev_ctrl != ctrl or self._cursor_smooth_uv is None:
            self._cursor_smooth_uv = (u, v)
        else:
            ALPHA = 0.35   # higher = snappier, lower = smoother
            su, sv = self._cursor_smooth_uv
            su += ALPHA * (u - su)
            sv += ALPHA * (v - sv)
            self._cursor_smooth_uv = (su, sv)

        su, sv = self._cursor_smooth_uv
        px = int(su * dw)
        py = int((1.0 - sv) * dh)
        self._vr_cursor_screen_pos = (px, py)
        _set_cursor_pos(px, py)

    def _handle_triggers(self):
        """Map controller triggers to mouse clicks and drag.

        Three-state machine per trigger: idle → pressed → dragging.

        • Rising edge (idle → pressed):
            Send LEFTDOWN + LEFTUP immediately — a complete click pulse.
            The OS sees two clean press+release pairs within GetDoubleClickTime
            for double-click, just like a real mouse double-click.

        • Held past HOLD_TIME (pressed → dragging):
            Send LEFTDOWN (hold) — the OS sees button-down + subsequent
            MOUSEMOVE events = drag.

        • Released from dragging:
            Send LEFTUP to end the drag.

        This gives all three: single-click, double-click, and drag —
        without requiring the user to do two complete trigger cycles quickly.
        """
        PRESS_THRESH   = 0.55   # rising edge
        RELEASE_THRESH = 0.30   # falling edge (hysteresis)
        HOLD_TIME      = 0.22   # seconds trigger must stay held to enter drag mode

        now = time.perf_counter()
        lt  = self._read_float_action(self._act_left_trigger,  "/user/hand/left")
        rt  = self._read_float_action(self._act_right_trigger, "/user/hand/right")

        left_on_kb  = self._kb_hover_l is not None
        right_on_kb = self._kb_hover_r is not None

        left_laser_usable  = (not left_on_kb and
                              (self._cursor_uv_l is not None or
                               self._cursor_ctrl == 'left'))
        right_laser_usable = (not right_on_kb and
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
                    _send_mouse_flags(_MOUSEEVENTF_LEFTDOWN)
                    _send_mouse_flags(_MOUSEEVENTF_LEFTUP)
                    setattr(self, state_attr,   'pressed')
                    setattr(self, press_t_attr, now)

            elif state == 'pressed':
                if not usable or trig <= RELEASE_THRESH:
                    # Released quickly — click already delivered, return to idle.
                    setattr(self, state_attr, 'idle')
                elif (now - getattr(self, press_t_attr)) >= HOLD_TIME:
                    # Held long enough → begin drag (send LEFTDOWN if not already down).
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
                idx, _ = self._keyboard_laser_hit(cp, fw)
            else:
                idx = None
            setattr(self, hover_attr, idx)

            # —— Release held regular key when trigger drops or laser leaves the key ——
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

            # —— Rising edge: modifier / caps toggles, or start holding a regular key ——
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

            # —— Slide to new key: trigger already held, laser moved to another regular key ——
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
        full push is dramatically faster — eliminates the \"stuck\" feeling.

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
            # Normalise [DEAD .. 1.0] → [0 .. 1]
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
        """Hand-split controls:
          Left  stick (no grip) → mouse scroll (X = horizontal, Y = vertical).
          Left  grip + L stick  → rotate screen (X = yaw, Y = pitch) at half speed.
          Right stick (no grip) → mouse scroll (X = horizontal, Y = vertical).
          Right grip + R stick X → resize screen (left = smaller, right = larger).
          Right grip + R stick Y → screen distance (up = further, down = closer).
          Left  trigger    → hold = left mouse button held (drag).
          Right trigger    → hold = left mouse button held (drag, same as left).
          A (right, no grip) → left click   B (right, no grip) → right click
          Right grip + A hold → increase depth_ratio   B hold → decrease depth_ratio
          Right grip + R stick click → reset depth_ratio to 1.0
          X (left)         → toggle virtual keyboard (re-anchored under current gaze)
          Y (left)         → reset screen to upright default position
          Menu (left)      → toggle FPS overlay
          Oculus home button long press → reset screen to face current head gaze
          Right stick click (short) → toggle curved screen   (long ≥0.6 s) → toggle cinema glow
          L grip + R stick Y    → adjust depth-parallax strength (up=more, down=less/flat)
          L grip + R stick click → toggle depth off/on (flat mode ↔ last strength)
          Keyboard Z/C     → depth strength −/+ 0.01   X → depth = 0 (flat)
        """
        if self._action_set is None:
            return

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

        grip_l = self._read_bool_action(self._act_left_grip,  "/user/hand/left")
        grip_r = self._read_bool_action(self._act_right_grip, "/user/hand/right")

        laser_on_screen = (self._cursor_uv_l is not None) or (self._cursor_uv_r is not None)
        active = (grip_l or grip_r) and laser_on_screen
        self._grabbed  = active
        self._resizing = False

        # Left grip + left stick
        # Keyboard visible → translate keyboard (X=horizontal, Y=vertical, Z=depth).
        # Keyboard hidden  → rotate screen yaw (X) / pitch (Y).
        # No grip → mouse scroll (same axes as right stick).
        KB_MOVE_SPEED = 0.4    # m/s at full deflection
        ROT_SPEED     = 0.35   # rad/s
        if grip_l:
            if self._keyboard_visible and self._kb_hover_l is not None:
                # Left grip + left stick → keyboard x,y translation (only when laser is on keyboard)
                if abs(lx) > DEAD:
                    self._keyboard_pan_x += lx * KB_MOVE_SPEED * dt
                if abs(ly) > DEAD:
                    self._keyboard_pan_y += ly * KB_MOVE_SPEED * dt  # up → raise keyboard
            else:
                if laser_on_screen and abs(lx) > DEAD:
                    self.screen_yaw   -= lx * ROT_SPEED * dt
                if laser_on_screen and abs(ly) > DEAD:
                    self.screen_pitch += ly * ROT_SPEED * dt
        else:
            # No left grip → left stick accumulates scroll (flushed below with right stick)
            self._accum_scroll(lx, ly, dt)

        # ── Right grip + right stick X: resize screen (left=smaller, right=larger) ──
        # ── Right grip + right stick Y: distance (up=further, down=closer) ─────────
        # ── Left grip + right stick Y: depth strength (parallax intensity) ────────
        #    Push stick up to increase 3-D depth effect; down to flatten/remove it.
        #    Right stick click while left grip held = toggle depth off/on instantly.
        # ── Right stick (no grip): mouse scroll ──────────────────────────────────
        DIST_SPEED   = 0.7    # m/s
        RESIZE_SPEED = 1.2    # m/s of width change at full deflection
        DEPTH_SPEED  = 0.08   # depth_strength units/s at full deflection
        if grip_r and not grip_l:
            # When pointing at the keyboard, right stick acts on keyboard instead of screen
            if self._keyboard_visible and self._kb_hover_r is not None:
                if abs(rx) > abs(ry) and abs(rx) > DEAD:
                    self._keyboard_width = max(0.3,
                        self._keyboard_width + rx * RESIZE_SPEED * dt)
                    self._keyboard_height = None  # recalc from aspect
                elif abs(ry) > abs(rx) and abs(ry) > DEAD:
                    self._keyboard_distance = max(0.2,
                        self._keyboard_distance + ry * DIST_SPEED * dt)
            else:
                if laser_on_screen and abs(rx) > abs(ry) and abs(rx) > DEAD:
                    self.screen_width = max(0.3,
                                            self.screen_width + rx * RESIZE_SPEED * dt)
                    self.screen_height = None
                    self._resizing = True
                    self._screen_osd_show_t = time.perf_counter()
                elif laser_on_screen and abs(ry) > abs(rx) and abs(ry) > DEAD:
                    self.screen_distance = max(0.3,
                                               self.screen_distance + ry * DIST_SPEED * dt)
                    self._screen_osd_show_t = time.perf_counter()
        elif grip_l and not grip_r:
            if self._keyboard_visible and self._kb_hover_r is not None:
                # Right stick Y = keyboard depth (only when laser is on keyboard)
                if abs(ry) > DEAD:
                    self._keyboard_distance = max(0.2,
                        self._keyboard_distance + ry * KB_MOVE_SPEED * dt)
            else:
                # Left grip + right stick Y → adjust depth strength
                if laser_on_screen and abs(ry) > DEAD:
                    self.depth_strength = max(0.0, min(0.5,
                                              self.depth_strength + ry * DEPTH_SPEED * dt))
        self._grip_r_prev = grip_r
        if not grip_r:
            # Accelerated scroll: higher stick deflection → disproportionately faster scroll
            self._accum_scroll(rx, ry, dt)

        # Rebuild keyboard geometry if width changed
        if (self._keyboard_visible and self._keyboard_tex is not None
                and abs(self._keyboard_width - self._kb_last_build_width) > 0.001):
            self._keyboard_height = (self._keyboard_width
                                     * _KB_TEX_H / float(_KB_TEX_W))
            self._build_keyboard_texture()
            self._kb_last_build_width = self._keyboard_width

        # Menu (left): toggle FPS overlay
        menu_now = self._read_bool_action(self._act_menu_btn, "/user/hand/left")
        if menu_now and not self._menu_pressed_last:
            self._fps_overlay_visible = not self._fps_overlay_visible
        self._menu_pressed_last = menu_now

        # A / B (right):
        #   right grip held + A hold   → increase depth_ratio continuously
        #   right grip held + B hold   → decrease depth_ratio continuously
        #   right grip held + RSC      → reset depth_ratio to 1.0  (handled in RSC block below)
        #   no right grip + A          → left mouse click
        #   no right grip + B          → right mouse click
        DEPTH_RATIO_SPEED = 0.5   # units per second at full hold
        DEPTH_RATIO_MIN   = 0.0
        DEPTH_RATIO_MAX   = 10.0

        a_now = self._read_bool_action(self._act_a_btn, "/user/hand/right")
        b_now = self._read_bool_action(self._act_b_btn, "/user/hand/right")

        if grip_r:
            if a_now:
                self.depth_ratio = min(DEPTH_RATIO_MAX, self.depth_ratio + DEPTH_RATIO_SPEED * dt)
            if b_now:
                self.depth_ratio = max(DEPTH_RATIO_MIN, self.depth_ratio - DEPTH_RATIO_SPEED * dt)
        else:
            # Use XR runtime's `changed` flag when available — more reliable than
            # manual frame-to-frame tracking when a button sits under a resting thumb.
            # Fall back to manual edge detection if pyopenxr doesn't expose it.
            a_edge = self._read_bool_edge(self._act_a_btn, "/user/hand/right", self._a_last)
            b_edge = self._read_bool_edge(self._act_b_btn, "/user/hand/right", self._b_last)
            if a_edge:
                _send_mouse_flags(_MOUSEEVENTF_LEFTDOWN)
                _send_mouse_flags(_MOUSEEVENTF_LEFTUP)
            if b_edge:
                _send_mouse_flags(_MOUSEEVENTF_RIGHTDOWN)
                _send_mouse_flags(_MOUSEEVENTF_RIGHTUP)

        self._a_last = a_now
        self._b_last = b_now

        # Y (left):
        #   short press  → reset screen to upright default (same as session start)
        #   long press   → reset screen to face current head gaze (same as home long-press)
        Y_LONG = 0.6   # seconds to trigger long-press action
        y_now = self._read_bool_action(self._act_y_btn, "/user/hand/left")
        if y_now and not self._y_last:
            self._y_press_t    = time.perf_counter()
            self._y_long_fired = False
        if y_now and not self._y_long_fired:
            if time.perf_counter() - self._y_press_t >= Y_LONG:
                nxt = (self._preset_index + 1) % len(self._screen_presets)
                self._apply_preset(nxt)
                self._y_long_fired = True
        if not y_now and self._y_last and not self._y_long_fired:
            self._apply_preset(3)  # short press = preset combo 4
        self._y_last = y_now

        # X (left): toggle virtual keyboard. When turning it on, snap the keyboard
        # anchor under the user's current gaze (in front and below) so it lands within
        # reach instead of at the world origin.
        x_now = self._read_bool_action(self._act_x_btn, "/user/hand/left")
        if x_now and not self._x_last:
            self._keyboard_visible = not self._keyboard_visible
            if self._keyboard_visible:
                if self._keyboard_tex is None:
                    self._init_keyboard()
                self._anchor_keyboard_below_screen()
        self._x_last = x_now

        # Left thumbstick click: cycle background color through _BG_COLORS presets.
        lsc_now = self._read_bool_action(self._act_left_stick_click, "/user/hand/left")
        if lsc_now and not self._left_stick_click_prev:
            self._bg_color_idx = (self._bg_color_idx + 1) % len(_BG_COLORS)
        self._left_stick_click_prev = lsc_now

        # Right thumbstick click:
        #   • right grip              → reset depth_ratio to 1.0
        #   • left grip (no R grip)   → toggle depth strength off/on
        #   • no grip                 → toggle curved screen
        rsc_now = self._read_bool_action(self._act_right_stick_click, "/user/hand/right")
        if not rsc_now and self._right_stick_click_prev:
            if grip_r:
                self.depth_ratio = 2.0
            elif grip_l:
                if self.depth_strength > 0.0:
                    self._depth_strength_saved = self.depth_strength
                    self.depth_strength = 0.0
                else:
                    self.depth_strength = getattr(self, '_depth_strength_saved', 0.1)
            else:
                self._screen_curved = not self._screen_curved
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

        # Keyboard border fade: show while gripping keyboard (but not when adjusting screen)
        kb_active = self._keyboard_visible and (grip_l or grip_r) and not laser_on_screen
        if kb_active:
            self._kb_border_alpha  = 1.0
            self._kb_border_idle_t = time.perf_counter()
        else:
            idle = time.perf_counter() - self._kb_border_idle_t
            if idle > FADE_DELAY:
                self._kb_border_alpha = max(0.0, 1.0 - (idle - FADE_DELAY) / FADE_DUR)

        # Cursor + trigger input (runs every frame regardless of grip state)
        self._handle_keyboard_input()   # updates _kb_hover_l/r, consumes keyboard triggers
        self._handle_cursor()           # moves Windows cursor when laser hits screen
        self._handle_triggers()         # fires mouse clicks (skips keys claimed by keyboard)

    # ------------------------------------------------------------------
    # Main blocking loop
    # ------------------------------------------------------------------

    def run(self, first_rgb=None, first_depth=None):
        """
        Blocking render loop. Exits when the OpenXR session ends, the GLFW
        window is closed, or shutdown_event is set.

        Pass the first rgb/depth frames already pulled from depth_q by main.py
        so no frame is wasted.
        """
        if not OPENXR_AVAILABLE:
            raise RuntimeError("pyopenxr not available")

        from utils import shutdown_event

        self._init_glfw()
        self._init_moderngl()

        try:
            self._init_openxr()
        except Exception as e:
            print(f"[OpenXRViewer] OpenXR init failed: {e}")
            self.cleanup()
            raise

        # Widen the system double-click window so VR trigger taps have more time.
        # Default Windows value is 500 ms — too tight for controller triggers.
        # We restore the original value in cleanup() so no permanent side-effects.
        if sys.platform == "win32" and self._saved_dclick_time is None:
            self._saved_dclick_time = _U32.GetDoubleClickTime()
            _U32.SystemParametersInfoW(0x0020, 1200, None, 0)  # SPI_SETDOUBLECLICKTIME

        # Upload the first frame supplied by main.py
        if first_rgb is not None and first_depth is not None:
            self._update_frame(first_rgb, first_depth)

        # Default fallback projection (used before first locate_views succeeds)
        _default_fov = xr.Fovf(
            angle_left=-0.785, angle_right=0.785,
            angle_up=0.785,   angle_down=-0.785,
        )
        _default_proj = _fov_to_proj_mat4(_default_fov)

        last_input_t = time.perf_counter()

        while (
            not glfw.window_should_close(self.window)
            and not shutdown_event.is_set()
        ):
            now = time.perf_counter()
            dt = now - last_input_t
            last_input_t = now

            glfw.poll_events()
            self._poll_xr_events()

            if not self._session_running:
                time.sleep(0.01)
                continue

            # — Wait for the runtime to signal frame timing —
            frame_state = xr.wait_frame(self._xr_session, xr.FrameWaitInfo())
            xr.begin_frame(self._xr_session, xr.FrameBeginInfo())

            # sync_actions must happen before xr.locate_space for action spaces.
            # Do it here so _update_aim_poses gets fresh locations this frame.
            if self._action_set is not None:
                try:
                    xr.sync_actions(
                        self._xr_session,
                        xr.ActionsSyncInfo(active_action_sets=[
                            xr.ActiveActionSet(
                                action_set=self._action_set,
                                subaction_path=xr.NULL_PATH,
                            )
                        ]),
                    )
                except Exception:
                    pass

            # Locate controller spaces (now valid after sync_actions)
            self._update_aim_poses(frame_state.predicted_display_time)
            self._update_grip_poses(frame_state.predicted_display_time)
            # Poll button/stick states (sync already done above)
            self._poll_controller_input(dt)

            composition_layers = []

            if frame_state.should_render:
                # Drain depth_q non-blocking — keep only the newest frame
                latest = None
                try:
                    while True:
                        latest = self.depth_q.get_nowait()
                except _queue.Empty:
                    pass

                if latest is not None:
                    rgb, depth, frame_ts = latest
                    self._update_frame(rgb, depth)
                    if frame_ts is not None:
                        self.total_latency = (time.perf_counter() - frame_ts) * 1000.0
                    # SBS source rate: time of arrival of each unique frame from depth_q
                    sbs_now = time.perf_counter()
                    self._sbs_ts_ring.append(sbs_now)
                    m = len(self._sbs_ts_ring)
                    if m >= 2:
                        sbs_span = sbs_now - self._sbs_ts_ring[0]
                        if sbs_span > 0:
                            self.sbs_fps = (m - 1) / sbs_span

                # Head-tracking pose for this frame
                try:
                    view_state, views = xr.locate_views(
                        self._xr_session,
                        xr.ViewLocateInfo(
                            view_configuration_type=xr.ViewConfigurationType.PRIMARY_STEREO,
                            display_time=frame_state.predicted_display_time,
                            space=self._xr_space,
                        ),
                    )
                except Exception:
                    views = [None, None]

                # Cache head pose for the next frame's input handlers (left-stick
                # orbit pivot + keyboard anchoring). One-frame stale is imperceptible
                # at 90 Hz and avoids needing a second xr.locate_views call.
                if views and views[0] is not None and views[1] is not None:
                    try:
                        p0 = views[0].pose.position
                        p1 = views[1].pose.position
                        self._head_pos_w = (
                            (p0.x + p1.x) / 2.0,
                            (p0.y + p1.y) / 2.0,
                            (p0.z + p1.z) / 2.0,
                        )
                        qm = _xr_quat_to_mat4(views[0].pose.orientation)
                        # Forward = -Z column of the head rotation matrix.
                        self._head_fwd_w = (
                            float(-qm[0, 2]),
                            float(-qm[1, 2]),
                            float(-qm[2, 2]),
                        )
                    except Exception:
                        pass

                # On the first valid frame, place the screen in front of the user's
                # current gaze — identical to pressing Y — so startup matches reset.
                if not self._screen_eye_init and views and views[0] is not None:
                    try:
                        ey = (views[0].pose.position.y + views[1].pose.position.y) / 2.0
                        self._initial_head_y = float(ey)
                    except Exception:
                        pass
                    self._reset_screen_to_default(show_border=False)
                    self._screen_eye_init = True

                eye_layer_views = []

                if self._use_d3d11:
                    # ── D3D11 rendering ──────────────────────────────────
                    #
                    # Prefer GPU interop (NV_DX_interop2 or EXT_memory_object)
                    # which avoids the CPU round-trip entirely.
                    # Fall back to the optimized PBO readback path otherwise.

                    if self._interop_mode == 'nv_dx':
                        # NV_DX_interop2: render directly into swapchain textures
                        for eye_index in range(2):
                            swapchain = self._xr_swapchains[eye_index]
                            img_index = xr.acquire_swapchain_image(swapchain, xr.SwapchainImageAcquireInfo())
                            xr.wait_swapchain_image(swapchain, xr.SwapchainImageWaitInfo(timeout=xr.INFINITE_DURATION))
                            sc_image = self._swapchain_images[eye_index][img_index]
                            sc_w, sc_h = self._swapchain_sizes[eye_index]
                            view = views[eye_index] if views and views[eye_index] else None
                            view_mat = _pose_to_view_mat4(view.pose) if view else np.eye(4, dtype=np.float32)
                            proj_mat = _fov_to_proj_mat4(view.fov)   if view else _default_proj

                            mgl_fbo, raw_fbo = self._get_or_create_nv_interop_fbo(
                                eye_index, img_index, sc_image.texture, sc_w, sc_h,
                            )
                            # Lock the registered D3D11 texture for GL access
                            _, _, dx_obj = self._nv_dx_objects[(eye_index, img_index)]
                            _wglDXLockObjectsNV(self._nv_dx_device, 1, ctypes.byref(dx_obj))
                            try:
                                self._render_eye(eye_index, mgl_fbo, view_mat, proj_mat, flip_y=True)
                            finally:
                                _wglDXUnlockObjectsNV(self._nv_dx_device, 1, ctypes.byref(dx_obj))

                            xr.release_swapchain_image(swapchain, xr.SwapchainImageReleaseInfo())
                            eye_layer_views.append(xr.CompositionLayerProjectionView(
                                pose=view.pose if view else xr.Posef(),
                                fov=view.fov   if view else _default_fov,
                                sub_image=xr.SwapchainSubImage(
                                    swapchain=swapchain,
                                    image_rect=xr.Rect2Di(
                                        offset=xr.Offset2Di(x=0, y=0),
                                        extent=xr.Extent2Di(width=sc_w, height=sc_h),
                                    ),
                                ),
                            ))

                    elif self._interop_mode == 'ext_mem':
                        # EXT_memory_object: render to shared GL FBO, GPU-side blit to swapchain
                        for eye_index in range(2):
                            swapchain = self._xr_swapchains[eye_index]
                            img_index = xr.acquire_swapchain_image(swapchain, xr.SwapchainImageAcquireInfo())
                            xr.wait_swapchain_image(swapchain, xr.SwapchainImageWaitInfo(timeout=xr.INFINITE_DURATION))
                            sc_image = self._swapchain_images[eye_index][img_index]
                            sc_w, sc_h = self._swapchain_sizes[eye_index]
                            view = views[eye_index] if views and views[eye_index] else None
                            view_mat = _pose_to_view_mat4(view.pose) if view else np.eye(4, dtype=np.float32)
                            proj_mat = _fov_to_proj_mat4(view.fov)   if view else _default_proj

                            mgl_fbo = self._ext_shared_tex[eye_index][3]
                            self._render_eye(eye_index, mgl_fbo, view_mat, proj_mat, flip_y=True)
                            self._blit_ext_to_swapchain(eye_index, sc_image.texture)

                            xr.release_swapchain_image(swapchain, xr.SwapchainImageReleaseInfo())
                            eye_layer_views.append(xr.CompositionLayerProjectionView(
                                pose=view.pose if view else xr.Posef(),
                                fov=view.fov   if view else _default_fov,
                                sub_image=xr.SwapchainSubImage(
                                    swapchain=swapchain,
                                    image_rect=xr.Rect2Di(
                                        offset=xr.Offset2Di(x=0, y=0),
                                        extent=xr.Extent2Di(width=sc_w, height=sc_h),
                                    ),
                                ),
                            ))

                    else:
                        # PBO fallback: two-phase loop to overlap GPU DMA with rendering.
                        d3d11_pending = []   # (eye_index, pbo_id, d3d11_tex, sc_w, sc_h, swapchain, view)

                        for eye_index in range(2):
                            swapchain = self._xr_swapchains[eye_index]
                            img_index = xr.acquire_swapchain_image(swapchain, xr.SwapchainImageAcquireInfo())
                            xr.wait_swapchain_image(swapchain, xr.SwapchainImageWaitInfo(timeout=xr.INFINITE_DURATION))
                            sc_image = self._swapchain_images[eye_index][img_index]
                            sc_w, sc_h = self._swapchain_sizes[eye_index]
                            view = views[eye_index] if views and views[eye_index] else None
                            view_mat = _pose_to_view_mat4(view.pose) if view else np.eye(4, dtype=np.float32)
                            proj_mat = _fov_to_proj_mat4(view.fov)   if view else _default_proj

                            mgl_fbo, raw_fbo_id = self._get_or_create_offscreen_fbo(eye_index, img_index, sc_w, sc_h)
                            self._render_eye(eye_index, mgl_fbo, view_mat, proj_mat, flip_y=True)

                            pbo_id = self._get_or_create_d3d11_pbo(eye_index, img_index, sc_w, sc_h)
                            self._submit_pbo_readback(raw_fbo_id, pbo_id, sc_w, sc_h)
                            d3d11_pending.append((eye_index, pbo_id, sc_image.texture, sc_w, sc_h, swapchain, view))

                        # Phase 2: map PBOs (DMA should be done), upload, release.
                        for eye_index, pbo_id, d3d11_tex, sc_w, sc_h, swapchain, view in d3d11_pending:
                            self._upload_pbo_to_d3d11(pbo_id, d3d11_tex, sc_w, sc_h)
                            xr.release_swapchain_image(swapchain, xr.SwapchainImageReleaseInfo())
                            eye_layer_views.append(xr.CompositionLayerProjectionView(
                                pose=view.pose if view else xr.Posef(),
                                fov=view.fov   if view else _default_fov,
                                sub_image=xr.SwapchainSubImage(
                                    swapchain=swapchain,
                                    image_rect=xr.Rect2Di(
                                        offset=xr.Offset2Di(x=0, y=0),
                                        extent=xr.Extent2Di(width=sc_w, height=sc_h),
                                    ),
                                ),
                            ))

                else:
                    for eye_index in range(2):
                        swapchain = self._xr_swapchains[eye_index]

                        img_index = xr.acquire_swapchain_image(
                            swapchain, xr.SwapchainImageAcquireInfo()
                        )
                        xr.wait_swapchain_image(
                            swapchain,
                            xr.SwapchainImageWaitInfo(timeout=xr.INFINITE_DURATION),
                        )

                        sc_image = self._swapchain_images[eye_index][img_index]
                        sc_w, sc_h = self._swapchain_sizes[eye_index]

                        view = views[eye_index] if views and views[eye_index] else None
                        view_mat = _pose_to_view_mat4(view.pose) if view else np.eye(4, dtype=np.float32)
                        proj_mat = _fov_to_proj_mat4(view.fov)   if view else _default_proj

                        _, mgl_fbo = self._get_or_create_fbo(eye_index, img_index, sc_image.image)
                        self._render_eye(eye_index, mgl_fbo, view_mat, proj_mat)

                        xr.release_swapchain_image(swapchain, xr.SwapchainImageReleaseInfo())

                        eye_layer_views.append(xr.CompositionLayerProjectionView(
                            pose=view.pose if view else xr.Posef(),
                            fov=view.fov  if view else _default_fov,
                            sub_image=xr.SwapchainSubImage(
                                swapchain=swapchain,
                                image_rect=xr.Rect2Di(
                                    offset=xr.Offset2Di(x=0, y=0),
                                    extent=xr.Extent2Di(width=sc_w, height=sc_h),
                                ),
                            ),
                        ))

                proj_layer = xr.CompositionLayerProjection(
                    space=self._xr_space,
                    views=eye_layer_views,
                )
                composition_layers.append(
                    ctypes.cast(ctypes.pointer(proj_layer),
                                ctypes.POINTER(xr.CompositionLayerBaseHeader))
                )

            xr.end_frame(
                self._xr_session,
                xr.FrameEndInfo(
                    display_time=frame_state.predicted_display_time,
                    environment_blend_mode=xr.EnvironmentBlendMode.OPAQUE,
                    layers=composition_layers,
                ),
            )

            # Timestamp-ring FPS: (N-1) frames / (last_ts - first_ts) — exact, O(1)
            t_now = time.perf_counter()
            self._frame_ts_ring.append(t_now)
            n = len(self._frame_ts_ring)
            if n >= 2:
                span = t_now - self._frame_ts_ring[0]
                if span > 0:
                    self.actual_fps = (n - 1) / span

        self.cleanup()

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def cleanup(self):
        """Release all OpenXR and OpenGL resources."""
        if sys.platform == "win32" and self._saved_dclick_time is not None:
            _U32.SystemParametersInfoW(0x0020, self._saved_dclick_time, None, 0)
            self._saved_dclick_time = None

        self._cleanup_interop()

        raw_ids = [raw_id for raw_id, _ in self._fbo_cache.values()]
        if raw_ids:
            try:
                glDeleteFramebuffers(len(raw_ids), raw_ids)
            except Exception:
                pass
        self._fbo_cache.clear()

        # Release D3D11-path PBOs used for async pixel readback
        if self._d3d11_pbo_cache:
            try:
                glDeleteBuffers(len(self._d3d11_pbo_cache), [v[0] for v in self._d3d11_pbo_cache.values()])
            except Exception:
                pass
            self._d3d11_pbo_cache.clear()

        # Release D3D11-path offscreen FBOs and their backing textures
        offscreen_raw_ids = [entry[1] for entry in self._offscreen_fbo_cache.values()]
        if offscreen_raw_ids:
            try:
                glDeleteFramebuffers(len(offscreen_raw_ids), offscreen_raw_ids)
            except Exception:
                pass
        for entry in self._offscreen_fbo_cache.values():
            try:
                entry[2].release()   # mgl Texture
            except Exception:
                pass
        self._offscreen_fbo_cache.clear()

        # Release D3D11 device/context (COM objects — call Release via vtable)
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
        self._d3d11_device = None
        self._d3d11_context = None

        # Release GPU interop PBOs
        if self._pbo_color is not None and self._cuda_gl:
            try:
                self._cuda_gl.unregister_resource(self._cuda_res_color)
                self._cuda_gl.unregister_resource(self._cuda_res_depth)
                glDeleteBuffers(2, [self._pbo_color, self._pbo_depth])
            except Exception:
                pass
        self._pbo_color = self._pbo_depth = None

        for tex in (self._overlay_tex, self._depth_osd_tex, self._screen_osd_tex,
                    self.color_tex, self.depth_tex):
            if tex:
                try:
                    tex.release()
                except Exception:
                    pass
        self._overlay_tex = self._depth_osd_tex = self._screen_osd_tex = None
        self.color_tex = self.depth_tex = None

        # Release controller model GL resources
        for prims in (self._ctrl_prims_l, self._ctrl_prims_r):
            for prim in prims:
                for key in ('vao', 'vbo', 'ibo'):
                    obj = prim.get(key)
                    if obj:
                        try:
                            obj.release()
                        except Exception:
                            pass
        self._ctrl_prims_l.clear()
        self._ctrl_prims_r.clear()
        for tex in self._ctrl_tex_cache.values():
            try:
                tex.release()
            except Exception:
                pass
        self._ctrl_tex_cache.clear()
        if self._controller_prog:
            try:
                self._controller_prog.release()
            except Exception:
                pass
            self._controller_prog = None
        self._grip_mat_l = None
        self._grip_mat_r = None

        for swapchain in self._xr_swapchains.values():
            try:
                xr.destroy_swapchain(swapchain)
            except Exception:
                pass
        self._xr_swapchains.clear()

        for space_attr in ("_aim_space_l", "_aim_space_r", "_grip_space_l", "_grip_space_r", "_xr_space"):
            sp = getattr(self, space_attr, None)
            if sp:
                try:
                    xr.destroy_space(sp)
                except Exception:
                    pass
                setattr(self, space_attr, None)

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

        if self.window:
            try:
                glfw.terminate()
            except Exception:
                pass
            self.window = None

        print("[OpenXRViewer] Cleanup complete")

if __name__ == "__main__":
    # Standalone smoke test: feed the viewer a single blank-white RGB frame
    # (with a zero depth map) so you can put on the headset and verify rendering,
    # controller input, keyboard, etc. — without needing the
    # full main.py capture pipeline running.
    if not OPENXR_AVAILABLE:
        print("[TEST] pyopenxr not available — cannot run standalone test")
        sys.exit(1)

    import queue as _q
    W, H = 1280, 720
    white_rgb = np.full((H, W, 3), 255, dtype=np.uint8)
    zero_depth = np.zeros((H, W), dtype=np.float32)

    depth_q = _q.Queue(maxsize=2)
    # Pre-seed with one frame so the run loop has something to display immediately.
    depth_q.put((white_rgb, zero_depth, time.perf_counter()))

    viewer = OpenXRViewer(
        frame_size=(W, H),
        fps=60,
        depth_q=depth_q,
        show_fps=True,
    )

    try:
        viewer.run(first_rgb=white_rgb, first_depth=zero_depth)
    except KeyboardInterrupt:
        print("[TEST] Interrupted")
    finally:
        viewer.cleanup()
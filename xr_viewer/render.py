"""Rendering support helpers for GLB loading, D3D11 interop, and XR math."""

import ctypes
import io as _io
import json
import math
import os
import struct
import sys

import numpy as np
from PIL import Image

try:
    import xr
except ImportError:  # pyopenxr is optional at import time.
    xr = None

from .constants import _DXGI_FORMAT_R8G8B8A8_UNORM

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
    """Extract numpy array from a glTF accessor.
    Handles both contiguous and interleaved (byteStride) vertex attributes.
    """
    acc = gltf['accessors'][acc_idx]
    bv = gltf['bufferViews'][acc['bufferView']]
    byte_offset = bv.get('byteOffset', 0) + acc.get('byteOffset', 0)
    byte_stride = bv.get('byteStride', 0)
    nc = _TYPE_NC[acc['type']]
    dt = np.dtype(_DTYPE_MAP[acc['componentType']]).newbyteorder('<')
    elem_size = nc * dt.itemsize

    if byte_stride == 0 or byte_stride == elem_size:
        # Contiguous (no stride or stride equals element size)
        arr = np.frombuffer(bin_data, dtype=dt, count=acc['count'] * nc,
                           offset=byte_offset).copy()
    else:
        # Interleaved vertex attributes read each row with stride
        arr = np.ndarray(shape=(acc['count'], nc), dtype=dt,
                         buffer=bin_data,
                         offset=byte_offset,
                         strides=(byte_stride, dt.itemsize)).copy()
    if nc > 1:
        arr = arr.reshape(acc['count'], nc)
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
    Local matrix = node.matrix when present, otherwise translation * rotation * scale.
    Root nodes assume identity parent matrix.
    """
    nodes = gltf.get('nodes', [])
    n = len(nodes)
    if n == 0:
        return []

    # Build local matrices
    local_mats = []
    for node in nodes:
        if 'matrix' in node:
            try:
                # glTF stores matrices column-major.  The viewer math uses
                # column vectors with translation in the last column, so
                # transpose after reshaping the flat JSON array.
                local_mats.append(
                    np.array(node['matrix'], dtype=np.float64).reshape((4, 4)).T
                )
                continue
            except Exception:
                pass

        t = node.get('translation', [0, 0, 0])
        r = node.get('rotation', [0, 0, 0, 1])  # [x, y, z, w]
        s = node.get('scale', [1, 1, 1])

        T = np.eye(4, dtype=np.float64)
        T[:3, 3] = t
        R = _quat_to_mat4(r)
        S_mat = np.diag([s[0], s[1], s[2], 1.0]).astype(np.float64)
        local_mats.append(T @ R @ S_mat)

    # Build child -> parent mapping.
    # Some third-party exports contain stray child indices; ignore them so
    # the rest of the scene can still load.
    parent = [-1] * n
    for pi, node in enumerate(nodes):
        for ci in node.get('children', []):
            if isinstance(ci, int) and 0 <= ci < n:
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
            if not isinstance(ci, int) or ci < 0 or ci >= n:
                continue
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
    _mat_log = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'material_debug.txt'), 'w', encoding='utf-8')
    _mat_log.write(f"=== Material debug for: {path} ===\n")
    with open(path, 'rb') as f:
        data = f.read()
    gltf, bin_data = _read_glb_chunks(data)

    # World matrices for all nodes
    world_mats = _build_node_matrices(gltf)
    nodes = gltf.get('nodes', [])

    # Map mesh index to world matrices for all node instances that reference it.
    # glTF permits mesh reuse from many nodes; keeping only the first node
    # loses valid scene instances and produces wrong room coordinates.
    mesh_world_mat = {}
    mesh_world_mats = {}
    mesh_node_index = {}
    mesh_node_indices = {}
    for ni, node in enumerate(nodes):
        mi = node.get('mesh')
        if mi is not None:
            mesh_world_mats.setdefault(mi, []).append(world_mats[ni])
            mesh_node_indices.setdefault(mi, []).append(ni)
            if mi not in mesh_world_mat:
                mesh_world_mat[mi] = world_mats[ni]
                mesh_node_index[mi] = ni

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
        node_index = mesh_node_index.get(mi, -1)
        for prim in mesh.get('primitives', []):
            attrs = prim['attributes']
            pos = _get_accessor(gltf, bin_data, attrs['POSITION'])

            # Extract normals if present, else zeros
            if 'NORMAL' in attrs:
                norm = _get_accessor(gltf, bin_data, attrs['NORMAL'])
            else:
                norm = np.zeros((pos.shape[0], 3), dtype=np.float32)

            # Extract tangent (vec4: xyz + bitangent_sign), or zeros if absent
            if 'TANGENT' in attrs:
                tangent = _get_accessor(gltf, bin_data, attrs['TANGENT'])
                if tangent.shape[1] < 4:
                    t4 = np.ones((tangent.shape[0], 4), dtype=np.float32)
                    t4[:, :tangent.shape[1]] = tangent
                    tangent = t4
            else:
                tangent = np.zeros((pos.shape[0], 4), dtype=np.float32)
                tangent[:, 3] = 1.0  # bitangent sign defaults to 1

            # Apply node world matrix: position with full 4x4, normals with inverse-transpose
            if not np.allclose(world_mat, np.eye(4)):
                pos = _apply_transform(pos, world_mat)
                rot3 = world_mat[:3, :3].astype(np.float64)
                normal_mat = np.linalg.inv(rot3.T)  # inverse-transpose handles non-uniform scaling
                norm = (normal_mat @ norm.T).T.astype(np.float32)
                # Transform tangent xyz with rotation, keep w (bitangent sign)
                if tangent is not None:
                    t_xyz = (rot3[:3, :3].astype(np.float64) @ tangent[:, :3].T).T.astype(np.float32)
                    tangent = np.hstack([t_xyz, tangent[:, 3:4]]).astype(np.float32)

            # Extract UV coordinates
            if 'TEXCOORD_0' in attrs:
                uv = _get_accessor(gltf, bin_data, attrs['TEXCOORD_0'])
                if uv.shape[1] > 2:
                    uv = uv[:, :2]
            elif 'TEXCOORD_1' in attrs:
                uv = _get_accessor(gltf, bin_data, attrs['TEXCOORD_1'])
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

            # Texture ID, base color, and roughness from material
            tex_id = -1
            base_color = np.array([1.0, 1.0, 1.0], dtype=np.float32)
            base_alpha = 1.0
            roughness_factor = 1.0
            metallic_factor = 1.0
            normal_tex_id = -1
            normal_scale = 1.0
            occlusion_tex_id = -1
            occlusion_strength = 1.0
            unlit = False
            double_sided = False
            alpha_mode = 'OPAQUE'
            alpha_cutoff = 0.5
            mr_tex_id = -1
            emissive_tex_id = -1
            emissive_factor = np.array([0.0, 0.0, 0.0], dtype=np.float32)
            tex_offset = np.array([0.0, 0.0], dtype=np.float32)
            tex_scale = np.array([1.0, 1.0], dtype=np.float32)
            mat_idx = prim.get('material')
            if mat_idx is not None and 'materials' in gltf:
                mat = gltf['materials'][mat_idx]
                mat_name = mat.get('name', f'material_{mat_idx}')
                pbr = mat.get('pbrMetallicRoughness', {})
                ext = mat.get('extensions', {})
                sg = ext.get('KHR_materials_pbrSpecularGlossiness')

                # --- Texture extraction ---
                # 1) Standard pbrMetallicRoughness.baseColorTexture
                bt = pbr.get('baseColorTexture')
                tex_index = bt.get('index') if bt else None
                # 2) KHR_materials_pbrSpecularGlossiness.diffuseTexture
                if tex_index is None and sg:
                    dt = sg.get('diffuseTexture')
                    tex_index = dt.get('index') if dt else None

                if tex_index is not None:
                    tid = tex_img_map.get(tex_index, -1)
                    if tid >= 0 and tid < len(all_textures) and all_textures[tid] is not None:
                        tex_id = tid

                # KHR_texture_transform on baseColorTexture
                tex_offset = np.array([0.0, 0.0], dtype=np.float32)
                tex_scale = np.array([1.0, 1.0], dtype=np.float32)
                if isinstance(bt, dict):
                    tx_ext = bt.get('extensions', {}).get('KHR_texture_transform')
                    if tx_ext:
                        if 'offset' in tx_ext:
                            tex_offset = np.array(tx_ext['offset'][:2], dtype=np.float32)
                        if 'scale' in tx_ext:
                            tex_scale = np.array(tx_ext['scale'][:2], dtype=np.float32)

                bcf = pbr.get('baseColorFactor')
                if bcf is not None:
                    base_color = np.array(bcf[:3], dtype=np.float32)
                    if len(bcf) > 3:
                        base_alpha = float(bcf[3])
                rf = pbr.get('roughnessFactor')
                if rf is not None:
                    roughness_factor = float(rf)
                mf = pbr.get('metallicFactor')
                metallic_factor = float(mf) if mf is not None else 1.0

                # metallicRoughnessTexture (glTF spec: B=metallic, G=roughness)
                mr_tex_id = -1
                mrt = pbr.get('metallicRoughnessTexture')
                if mrt and 'index' in mrt:
                    mr_tid = tex_img_map.get(mrt['index'], -1)
                    if mr_tid >= 0 and mr_tid < len(all_textures) and all_textures[mr_tid] is not None:
                        mr_tex_id = mr_tid

                # SpecGloss diffuseFactor (color, applies regardless of texture)
                if sg and 'diffuseFactor' in sg:
                    if bcf is None:
                        base_color = np.array(sg['diffuseFactor'][:3], dtype=np.float32)
                        if len(sg['diffuseFactor']) > 3:
                            base_alpha = float(sg['diffuseFactor'][3])

                # Normal texture
                normal_tex_id = -1
                normal_scale = 1.0
                nt = mat.get('normalTexture')
                if nt and 'index' in nt:
                    n_tid = tex_img_map.get(nt['index'], -1)
                    if n_tid >= 0 and n_tid < len(all_textures) and all_textures[n_tid] is not None:
                        normal_tex_id = n_tid
                    ns = nt.get('scale')
                    if ns is not None:
                        normal_scale = float(ns)

                # Occlusion texture
                occlusion_tex_id = -1
                occlusion_strength = 1.0
                ot = mat.get('occlusionTexture')
                if ot and 'index' in ot:
                    o_tid = tex_img_map.get(ot['index'], -1)
                    if o_tid >= 0 and o_tid < len(all_textures) and all_textures[o_tid] is not None:
                        occlusion_tex_id = o_tid
                    os_ = ot.get('strength')
                    if os_ is not None:
                        occlusion_strength = float(os_)

                # KHR_materials_unlit
                unlit = bool(ext.get('KHR_materials_unlit'))

                # doubleSided
                double_sided = bool(mat.get('doubleSided', False))

                # alphaMode + alphaCutoff (glTF spec)
                alpha_mode = mat.get('alphaMode', 'OPAQUE')
                alpha_cutoff = float(mat.get('alphaCutoff', 0.5))

                # Emissive: emissiveFactor * emissiveStrength.
                # If all 3 channels are identical -> Unity export default -> suppress.
                emissive_factor = np.array([0.0, 0.0, 0.0], dtype=np.float32)
                emissive_tex_id = -1
                ef = mat.get('emissiveFactor')
                if ef is not None:
                    raw_ef = np.array(ef[:3], dtype=np.float32)
                    # Only keep emissive when channels DIFFER (intentional colored glow)
                    if not (abs(raw_ef[0] - raw_ef[1]) < 0.001 and abs(raw_ef[0] - raw_ef[2]) < 0.001):
                        emissive_factor = raw_ef
                        es_ext = ext.get('KHR_materials_emissive_strength')
                        if es_ext and 'emissiveStrength' in es_ext:
                            emissive_factor = emissive_factor * float(es_ext['emissiveStrength'])
                # emissiveTexture
                et = mat.get('emissiveTexture')
                if et and 'index' in et:
                    e_tid = tex_img_map.get(et['index'], -1)
                    if e_tid >= 0 and e_tid < len(all_textures) and all_textures[e_tid] is not None:
                        emissive_tex_id = e_tid
                # Also read emissive from specGloss if no standard emissive factor
                if sg and emissive_factor.any() == False:
                    df = sg.get('diffuseFactor')
                    if df:
                        raw_sg = np.array(df[:3], dtype=np.float32)
                        if not (abs(raw_sg[0] - raw_sg[1]) < 0.001 and abs(raw_sg[0] - raw_sg[2]) < 0.001):
                            emissive_factor = raw_sg

                # Debug log
                emissive_info = f' emissive={emissive_factor.tolist()}' if emissive_factor.any() else ''
                if mat_idx < 300:
                    _mat_log.write(f"[MAT] {mat_idx}: {mat_name}  "
                          f"bcf={bcf}  rough={rf}  "
                          f"tex_index={tex_index}  tex_id={tex_id}"
                          f"{emissive_info}  "
                          f"ext={list(ext.keys())}\n")

            primitives.append({'vertices': vertices, 'indices': indices,
                            'tex_id': tex_id, 'base_color': base_color,
                            'base_alpha': base_alpha,
                            'roughness_factor': roughness_factor,
                            'metallic_factor': metallic_factor,
                            'emissive_factor': emissive_factor,
                            'normal_tex_id': normal_tex_id,
                            'normal_scale': normal_scale,
                            'occlusion_tex_id': occlusion_tex_id,
                            'occlusion_strength': occlusion_strength,
                            'unlit': unlit,
                            'alpha_mode': alpha_mode,
                            'alpha_cutoff': alpha_cutoff,
                            'mr_tex_id': mr_tex_id,
                            'emissive_tex_id': emissive_tex_id,
                            'double_sided': double_sided,
                            'tex_offset': tex_offset,
                            'tex_scale': tex_scale,
                            'tangent': tangent,
                            'node_index': node_index,
                            '_mesh_index': mi,
                            '_world_matrix': world_mat})

    extra_instances = []
    for primitive in primitives:
        mi = primitive.get('_mesh_index')
        instances = mesh_world_mats.get(mi, [])
        if len(instances) <= 1:
            continue

        first_world = primitive.get('_world_matrix', np.eye(4, dtype=np.float64)).astype(np.float64)
        try:
            inv_first_world = np.linalg.inv(first_world)
        except Exception:
            continue

        local_positions = _apply_transform(primitive['vertices'][:, :3], inv_first_world)
        first_rot = first_world[:3, :3].astype(np.float64)
        local_normals = (first_rot.T @ primitive['vertices'][:, 3:6].astype(np.float64).T).T
        local_normals /= np.maximum(np.linalg.norm(local_normals, axis=1, keepdims=True), 1e-8)

        tangent = primitive.get('tangent')
        if tangent is not None:
            local_tangent = tangent.copy()
            local_tangent[:, :3] = (
                first_rot.T @ tangent[:, :3].astype(np.float64).T
            ).T.astype(np.float32)
        else:
            local_tangent = None

        node_indices = mesh_node_indices.get(mi, [])
        for inst_i, inst_world in enumerate(instances[1:], start=1):
            inst_world = inst_world.astype(np.float64)
            clone = dict(primitive)
            clone_vertices = primitive['vertices'].copy()
            clone_vertices[:, :3] = _apply_transform(local_positions, inst_world)
            rot3 = inst_world[:3, :3].astype(np.float64)
            normal_mat = np.linalg.inv(rot3.T)
            clone_vertices[:, 3:6] = (
                normal_mat @ local_normals.T
            ).T.astype(np.float32)
            clone['vertices'] = clone_vertices
            clone['indices'] = primitive['indices'].copy()
            if local_tangent is not None:
                inst_tangent = local_tangent.copy()
                inst_tangent[:, :3] = (
                    rot3 @ local_tangent[:, :3].astype(np.float64).T
                ).T.astype(np.float32)
                clone['tangent'] = inst_tangent
            if inst_i < len(node_indices):
                clone['node_index'] = node_indices[inst_i]
            clone['_world_matrix'] = inst_world
            extra_instances.append(clone)

    if extra_instances:
        primitives.extend(extra_instances)
        _mat_log.write(f"[INSTANCE] Added {len(extra_instances)} mesh node instances\n")

    # Extract KHR_lights_punctual
    lights = []
    try:
        gltf_lights = gltf.get('extensions', {}).get('KHR_lights_punctual', {})
        if isinstance(gltf_lights, dict):
            gltf_lights = gltf_lights.get('lights', [])
        else:
            gltf_lights = []
        for ni, node in enumerate(gltf.get('nodes', [])):
            lext = node.get('extensions', {}).get('KHR_lights_punctual')
            if lext and 'light' in lext:
                li = lext['light']
                if li < len(gltf_lights):
                    ldef = gltf_lights[li]
                    world_mat = world_mats[ni] if ni < len(world_mats) else np.eye(4, dtype=np.float64)
                    direction = -world_mat[:3, 2].astype(np.float32)
                    direction = direction / (np.linalg.norm(direction) + 1e-8)
                    lights.append({
                        'type': ldef.get('type', 'directional'),
                        'color': np.array(ldef.get('color', [1, 1, 1])[:3], dtype=np.float32),
                        'intensity': float(ldef.get('intensity', 1.0)),
                        'direction': direction,
                    })
                    _mat_log.write(f"[LIGHT] {ldef.get('name', '')}: type={ldef.get('type')} color={ldef.get('color')} intensity={ldef.get('intensity')}\n")
    except Exception as e:
        _mat_log.write(f"[LIGHT] extraction failed: {e}\n")

    _mat_log.write("=== End ===\n")
    _mat_log.close()
    return primitives, all_textures, lights

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
        Works with any format including SRGB no staging texture needed.
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



    # NV_DX_interop2 helpers (NVIDIA only, zero-copy GL↔3D11)
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

    # EXT_memory_object_win32 helpers (cross-vendor, GL 4.5+) 
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

    def _d3d11_update_subresource(context, dst, src_ptr, row_pitch):
        raise RuntimeError("D3D11 only available on Windows")

    def _create_d3d11_shared_texture(device, w, h, fmt=_DXGI_FORMAT_R8G8B8A8_UNORM):
        raise RuntimeError("D3D11 only available on Windows")

    _nv_dx_interop_available = False
    _wglDXOpenDeviceNV = None
    _wglDXCloseDeviceNV = None
    _wglDXRegisterObjectNV = None
    _wglDXLockObjectsNV = None
    _wglDXUnlockObjectsNV = None
    _wglDXUnregisterObjectNV = None

    def _load_nv_dx_interop():
        return False

    _ext_mem_available = False
    _glImportMemoryWin32HandleEXT = None
    _glTextureStorageMem2DEXT = None
    _glCreateMemoryObjectsEXT = None
    _glDeleteMemoryObjectsEXT = None
    _GL_HANDLE_TYPE_OPAQUE_WIN32_EXT = 0x9587

    def _load_ext_memory_object():
        return False

def _xr_quat_to_mat4(q):
    """XrQuaternionf standard 4×4 rotation matrix (numpy, math row/col convention).

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
    """XrPosef standard 4×4 view matrix (numpy, math row/col convention).

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
    """XrFovf standard 4×4 OpenGL asymmetric-frustum projection matrix
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

_proj_cache = {}
def _fov_to_proj_mat4_cached(fov, near=0.05, far=100.0):
    key = (fov.angle_left, fov.angle_right, fov.angle_up, fov.angle_down)
    cached = _proj_cache.get(key)
    if cached is not None:
        return cached.copy()
    p = _fov_to_proj_mat4(fov, near, far)
    _proj_cache[key] = p
    return p.copy()


def _xr_pose_to_model_mat4(pose):
    """XrPosef -> 4x4 model matrix (position in last column)."""
    M = _xr_quat_to_mat4(pose.orientation)
    M[:3, 3] = np.array([pose.position.x, pose.position.y, pose.position.z], dtype=np.float32)
    return M


def _euler_to_mat4(yaw, pitch, roll):
    """Yaw/pitch/roll radians -> 4x4 rotation matrix (Y * X * Z)."""
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cr, sr = math.cos(roll), math.sin(roll)
    ry = np.array([[cy, 0.0, sy, 0.0],
                   [0.0, 1.0, 0.0, 0.0],
                   [-sy, 0.0, cy, 0.0],
                   [0.0, 0.0, 0.0, 1.0]], dtype=np.float32)
    rx = np.array([[1.0, 0.0, 0.0, 0.0],
                   [0.0, cp, -sp, 0.0],
                   [0.0, sp, cp, 0.0],
                   [0.0, 0.0, 0.0, 1.0]], dtype=np.float32)
    rz = np.array([[cr, -sr, 0.0, 0.0],
                   [sr, cr, 0.0, 0.0],
                   [0.0, 0.0, 1.0, 0.0],
                   [0.0, 0.0, 0.0, 1.0]], dtype=np.float32)
    return ry @ rx @ rz


def _mat3_to_quat_xyzw(m33):
    """3x3 rotation matrix -> normalized quaternion (x, y, z, w)."""
    t = m33[0, 0] + m33[1, 1] + m33[2, 2]
    if t > 0.0:
        s = np.sqrt(t + 1.0) * 2.0
        w = 0.25 * s
        x = (m33[2, 1] - m33[1, 2]) / s
        y = (m33[0, 2] - m33[2, 0]) / s
        z = (m33[1, 0] - m33[0, 1]) / s
    elif m33[0, 0] > m33[1, 1] and m33[0, 0] > m33[2, 2]:
        s = np.sqrt(1.0 + m33[0, 0] - m33[1, 1] - m33[2, 2]) * 2.0
        w = (m33[2, 1] - m33[1, 2]) / s
        x = 0.25 * s
        y = (m33[0, 1] + m33[1, 0]) / s
        z = (m33[0, 2] + m33[2, 0]) / s
    elif m33[1, 1] > m33[2, 2]:
        s = np.sqrt(1.0 + m33[1, 1] - m33[0, 0] - m33[2, 2]) * 2.0
        w = (m33[0, 2] - m33[2, 0]) / s
        x = (m33[0, 1] + m33[1, 0]) / s
        y = 0.25 * s
        z = (m33[1, 2] + m33[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m33[2, 2] - m33[0, 0] - m33[1, 1]) * 2.0
        w = (m33[1, 0] - m33[0, 1]) / s
        x = (m33[0, 2] + m33[2, 0]) / s
        y = (m33[1, 2] + m33[2, 1]) / s
        z = 0.25 * s
    q = np.array([x, y, z, w], dtype=np.float64)
    return q / np.linalg.norm(q)


def _mat4_to_xr_posef(mat4):
    """4x4 rigid transform -> XrPosef."""
    if xr is None:
        raise RuntimeError("pyopenxr not installed")
    q = _mat3_to_quat_xyzw(mat4[:3, :3].astype(np.float64))
    pose = xr.Posef()
    pose.orientation.x = float(q[0])
    pose.orientation.y = float(q[1])
    pose.orientation.z = float(q[2])
    pose.orientation.w = float(q[3])
    pose.position.x = float(mat4[0, 3])
    pose.position.y = float(mat4[1, 3])
    pose.position.z = float(mat4[2, 3])
    return pose

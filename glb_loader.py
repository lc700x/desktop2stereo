# glb_loader.py
# GLB/GLTF 二进制模型加载器 — 含节点层级变换支持
# 提取顶点(POSITION+TEXCOORD_0)、索引和纹理，应用节点世界矩阵

import struct
import json
import io as _io
import numpy as np
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
    acc = gltf['accessors'][acc_idx]
    bv = gltf['bufferViews'][acc['bufferView']]
    off = bv.get('byteOffset', 0) + acc.get('byteOffset', 0)
    nc = _TYPE_NC[acc['type']]
    dt = np.dtype(_DTYPE_MAP[acc['componentType']]).newbyteorder('<')
    arr = np.frombuffer(bin_data[off:off + acc['count'] * nc * dt.itemsize], dtype=dt)
    if nc > 1:
        arr = arr.reshape(acc['count'], nc)
    if acc['componentType'] in (5121, 5123, 5125):
        arr = arr.astype(np.uint32)
    elif acc['componentType'] == 5126:
        arr = arr.astype(np.float32)
    return arr


def _quat_to_mat4(q):
    """四元数 [x,y,z,w] → 4x4旋转矩阵。"""
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
    """计算每个节点的世界矩阵(自顶向下)。返回 list of 4x4 float64。
    父节点的世界矩阵 = 父矩阵 @ 本节点的局部矩阵。
    局部矩阵 = T * R * S。
    根节点假设父矩阵为单位阵。
    """
    nodes = gltf.get('nodes', [])
    n = len(nodes)
    if n == 0:
        return []

    local_mats = []
    for node in nodes:
        t = node.get('translation', [0, 0, 0])
        r = node.get('rotation', [0, 0, 0, 1])  # [x,y,z,w]
        s = node.get('scale', [1, 1, 1])

        T = np.eye(4, dtype=np.float64)
        T[:3, 3] = t
        R = _quat_to_mat4(r)
        S_mat = np.diag([s[0], s[1], s[2], 1.0]).astype(np.float64)
        local_mats.append(T @ R @ S_mat)

    # 构建父子关系: child → parent
    parent = [-1] * n
    for pi, node in enumerate(nodes):
        for ci in node.get('children', []):
            parent[ci] = pi

    # 拓扑排序求世界矩阵 (BFS from roots)
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

    # 未到达的节点(孤立)直接使用局部矩阵
    for i in range(n):
        if world_mats[i] is None:
            world_mats[i] = local_mats[i].copy()

    return world_mats


def _apply_transform(vertices_xyz, matrix_4x4):
    """对顶点位置应用4x4变换矩阵。"""
    n = vertices_xyz.shape[0]
    ones = np.ones((n, 1), dtype=np.float64)
    v4 = np.hstack([vertices_xyz.astype(np.float64), ones])
    t = (matrix_4x4 @ v4.T).T
    return t[:, :3].astype(np.float32)


def load_glb_model(path):
    """加载GLB模型，应用节点层级变换。
    返回:
        primitives: [{vertices(N,5 float32), indices(M,uint32), tex_id(int)}]
        textures:   list of numpy RGBA uint8
    """
    with open(path, 'rb') as f:
        data = f.read()
    gltf, bin_data = _read_glb_chunks(data)

    # 计算所有节点的世界矩阵
    world_mats = _build_node_matrices(gltf)
    nodes = gltf.get('nodes', [])

    # 建立 mesh_index → world_matrix 映射
    # 如果多个节点引用同一个mesh，取第一个
    mesh_world_mat = {}
    for ni, node in enumerate(nodes):
        mi = node.get('mesh')
        if mi is not None and mi not in mesh_world_mat:
            mesh_world_mat[mi] = world_mats[ni]

    # 提取纹理
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

            # 提取法线
            if 'NORMAL' in attrs:
                norm = _get_accessor(gltf, bin_data, attrs['NORMAL'])
            else:
                norm = np.zeros((pos.shape[0], 3), dtype=np.float32)

            # 应用节点世界矩阵(位置用4x4, 法线用3x3旋转部分)
            if not np.allclose(world_mat, np.eye(4)):
                pos = _apply_transform(pos, world_mat)
                rot3 = world_mat[:3, :3].astype(np.float32)
                norm = (rot3 @ norm.T).T.astype(np.float32)

            if 'TEXCOORD_0' in attrs:
                uv = _get_accessor(gltf, bin_data, attrs['TEXCOORD_0'])
                if uv.shape[1] > 2:
                    uv = uv[:, :2]
            else:
                uv = np.zeros((pos.shape[0], 2), dtype=np.float32)

            # [px,py,pz, nx,ny,nz, u,v] = 8 floats
            vertices = np.hstack([pos, norm, uv]).astype(np.float32)

            if 'indices' in prim:
                indices = _get_accessor(gltf, bin_data, prim['indices'])
            else:
                indices = np.arange(pos.shape[0], dtype=np.uint32)

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


if __name__ == "__main__":
    import sys
    paths = sys.argv[1:] if len(sys.argv) > 1 else [
        'models/controllers/pico-4u/left.glb',
        'models/controllers/pico-4u/right.glb',
    ]
    for p in paths:
        prims, textures = load_glb_model(p)
        tv = sum(pr['vertices'].shape[0] for pr in prims)
        tt = sum(pr['indices'].shape[0] // 3 for pr in prims)
        print(f"{p}: {len(prims)} prims, {tv} verts, {tt} tris, "
              f"{len(textures)} textures")
        for i, pr in enumerate(prims):
            v = pr['vertices']
            print(f"  [{i}] {v.shape[0]}v {pr['indices'].shape[0]//3}t "
                  f"tex={pr['tex_id']} "
                  f"X:[{v[:,0].min():.3f},{v[:,0].max():.3f}] "
                  f"Y:[{v[:,1].min():.3f},{v[:,1].max():.3f}] "
                  f"Z:[{v[:,2].min():.3f},{v[:,2].max():.3f}]")

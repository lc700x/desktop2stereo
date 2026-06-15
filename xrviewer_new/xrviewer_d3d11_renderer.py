import ctypes
import os
import sys

import numpy as np


DXGI_FORMAT_R32G32_FLOAT = 16
DXGI_FORMAT_R8G8B8A8_UNORM = 28
DXGI_FORMAT_R32_FLOAT = 41

D3D11_USAGE_DEFAULT = 0
D3D11_BIND_VERTEX_BUFFER = 0x1
D3D11_BIND_CONSTANT_BUFFER = 0x4
D3D11_BIND_SHADER_RESOURCE = 0x8
D3D11_BIND_RENDER_TARGET = 0x20
D3D11_INPUT_PER_VERTEX_DATA = 0
D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST = 4
D3D11_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP = 5
D3D11_SDK_VERSION = 7
D3D11_FILL_SOLID = 3
D3D11_CULL_NONE = 1
D3D11_RTV_DIMENSION_TEXTURE2D = 4
D3D11_RTV_DIMENSION_TEXTURE2DARRAY = 5
D3D11_FILTER_MIN_MAG_MIP_LINEAR = 0x15
D3D11_TEXTURE_ADDRESS_CLAMP = 3
D3D11_COMPARISON_NEVER = 1


class DXGISampleDesc(ctypes.Structure):
    _fields_ = [
        ("Count", ctypes.c_uint),
        ("Quality", ctypes.c_uint),
    ]


class D3D11Texture2DDesc(ctypes.Structure):
    _fields_ = [
        ("Width", ctypes.c_uint),
        ("Height", ctypes.c_uint),
        ("MipLevels", ctypes.c_uint),
        ("ArraySize", ctypes.c_uint),
        ("Format", ctypes.c_uint),
        ("SampleDesc", DXGISampleDesc),
        ("Usage", ctypes.c_uint),
        ("BindFlags", ctypes.c_uint),
        ("CPUAccessFlags", ctypes.c_uint),
        ("MiscFlags", ctypes.c_uint),
    ]


class D3D11BufferDesc(ctypes.Structure):
    _fields_ = [
        ("ByteWidth", ctypes.c_uint),
        ("Usage", ctypes.c_uint),
        ("BindFlags", ctypes.c_uint),
        ("CPUAccessFlags", ctypes.c_uint),
        ("MiscFlags", ctypes.c_uint),
        ("StructureByteStride", ctypes.c_uint),
    ]


class D3D11SubresourceData(ctypes.Structure):
    _fields_ = [
        ("pSysMem", ctypes.c_void_p),
        ("SysMemPitch", ctypes.c_uint),
        ("SysMemSlicePitch", ctypes.c_uint),
    ]


class D3D11InputElementDesc(ctypes.Structure):
    _fields_ = [
        ("SemanticName", ctypes.c_char_p),
        ("SemanticIndex", ctypes.c_uint),
        ("Format", ctypes.c_uint),
        ("InputSlot", ctypes.c_uint),
        ("AlignedByteOffset", ctypes.c_uint),
        ("InputSlotClass", ctypes.c_uint),
        ("InstanceDataStepRate", ctypes.c_uint),
    ]


class D3D11Viewport(ctypes.Structure):
    _fields_ = [
        ("TopLeftX", ctypes.c_float),
        ("TopLeftY", ctypes.c_float),
        ("Width", ctypes.c_float),
        ("Height", ctypes.c_float),
        ("MinDepth", ctypes.c_float),
        ("MaxDepth", ctypes.c_float),
    ]


class D3D11RenderTargetViewDesc(ctypes.Structure):
    _fields_ = [
        ("Format", ctypes.c_uint),
        ("ViewDimension", ctypes.c_uint),
        ("MipSlice", ctypes.c_uint),
        ("FirstArraySlice", ctypes.c_uint),
        ("ArraySize", ctypes.c_uint),
    ]


class D3D11SamplerDesc(ctypes.Structure):
    _fields_ = [
        ("Filter", ctypes.c_uint),
        ("AddressU", ctypes.c_uint),
        ("AddressV", ctypes.c_uint),
        ("AddressW", ctypes.c_uint),
        ("MipLODBias", ctypes.c_float),
        ("MaxAnisotropy", ctypes.c_uint),
        ("ComparisonFunc", ctypes.c_uint),
        ("BorderColor", ctypes.c_float * 4),
        ("MinLOD", ctypes.c_float),
        ("MaxLOD", ctypes.c_float),
    ]


class D3D11RasterizerDesc(ctypes.Structure):
    _fields_ = [
        ("FillMode", ctypes.c_uint),
        ("CullMode", ctypes.c_uint),
        ("FrontCounterClockwise", ctypes.c_int),
        ("DepthBias", ctypes.c_int),
        ("DepthBiasClamp", ctypes.c_float),
        ("SlopeScaledDepthBias", ctypes.c_float),
        ("DepthClipEnable", ctypes.c_int),
        ("ScissorEnable", ctypes.c_int),
        ("MultisampleEnable", ctypes.c_int),
        ("AntialiasedLineEnable", ctypes.c_int),
    ]


def _ptr_value(ptr):
    if ptr is None:
        return 0
    if isinstance(ptr, int):
        return ptr
    if hasattr(ptr, "value"):
        return ptr.value
    return ctypes.cast(ptr, ctypes.c_void_p).value


def _com_fn(obj, index, restype, *argtypes):
    vtbl = ctypes.cast(obj, ctypes.POINTER(ctypes.c_void_p)).contents.value
    fn_ptr = ctypes.cast(
        vtbl + index * ctypes.sizeof(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_void_p),
    ).contents.value
    return ctypes.CFUNCTYPE(restype, ctypes.c_void_p, *argtypes)(fn_ptr)


def _release(ptr):
    if not ptr:
        return
    try:
        fn = _com_fn(ptr, 2, ctypes.c_ulong)
        fn(_ptr_value(ptr))
    except Exception:
        pass


def _blob_ptr(blob):
    fn = _com_fn(blob, 3, ctypes.c_void_p)
    return fn(_ptr_value(blob))


def _blob_size(blob):
    fn = _com_fn(blob, 4, ctypes.c_size_t)
    return fn(_ptr_value(blob))


def _compile_shader(source, entry, target):
    compiler = None
    for dll in ("d3dcompiler_47", "d3dcompiler_43"):
        try:
            compiler = ctypes.WinDLL(dll)
            break
        except OSError:
            continue
    if compiler is None:
        raise RuntimeError("d3dcompiler_47/d3dcompiler_43 not found")

    code = source.encode("utf-8")
    entry_b = entry.encode("ascii")
    target_b = target.encode("ascii")
    blob = ctypes.c_void_p()
    err_blob = ctypes.c_void_p()
    hr = compiler.D3DCompile(
        code,
        len(code),
        None,
        None,
        None,
        entry_b,
        target_b,
        0,
        0,
        ctypes.byref(blob),
        ctypes.byref(err_blob),
    )
    if hr != 0:
        msg = f"hr=0x{hr & 0xFFFFFFFF:08x}"
        if err_blob:
            try:
                msg = ctypes.string_at(_blob_ptr(err_blob), _blob_size(err_blob)).decode("utf-8", "replace")
            finally:
                _release(err_blob)
        raise RuntimeError(f"D3DCompile failed for {entry}/{target}: {msg}")
    if err_blob:
        _release(err_blob)
    return blob


def _check_hr(hr, label):
    if hr != 0:
        raise RuntimeError(f"{label} failed: hr=0x{hr & 0xFFFFFFFF:08x}")


def _hr_hex(hr):
    return f"0x{hr & 0xFFFFFFFF:08x}"


class CUDART_D3D11:
    CUDA_GRAPHICS_REGISTER_FLAGS_NONE = 0
    CUDA_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD = 2
    CUDA_MEMCPY_DEVICE_TO_DEVICE = 3

    def __init__(self, torch_module, d3d11_device, device_id=0):
        torch_dir = os.path.dirname(torch_module.__file__)
        site_packages = os.path.dirname(torch_dir)
        candidates = [
            os.path.join(torch_dir, "lib"),
            os.path.join(site_packages, "nvidia", "cuda_runtime", "bin"),
            os.path.join(site_packages, "nvidia", "cuda_runtime", "lib"),
        ]
        cudart_path = None
        for lib_dir in candidates:
            if not os.path.exists(lib_dir):
                continue
            for name in os.listdir(lib_dir):
                if sys.platform == "win32":
                    if name.startswith("cudart64") and name.endswith(".dll"):
                        cudart_path = os.path.join(lib_dir, name)
                        break
                elif name.startswith("libcudart") and ".so" in name:
                    cudart_path = os.path.join(lib_dir, name)
                    break
            if cudart_path:
                break
        if not cudart_path:
            raise RuntimeError("Could not find CUDA runtime library for D3D11 interop")

        self.lib = ctypes.WinDLL(cudart_path) if sys.platform == "win32" else ctypes.CDLL(cudart_path)
        self.lib.cudaSetDevice.argtypes = [ctypes.c_int]
        self.lib.cudaSetDevice.restype = ctypes.c_int
        self.lib.cudaGetLastError.argtypes = []
        self.lib.cudaGetLastError.restype = ctypes.c_int
        self.lib.cudaGraphicsD3D11RegisterResource.argtypes = [
            ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_uint
        ]
        self.lib.cudaGraphicsD3D11RegisterResource.restype = ctypes.c_int
        self.lib.cudaGraphicsUnregisterResource.argtypes = [ctypes.c_void_p]
        self.lib.cudaGraphicsUnregisterResource.restype = ctypes.c_int
        self.lib.cudaGraphicsMapResources.argtypes = [
            ctypes.c_int, ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p
        ]
        self.lib.cudaGraphicsMapResources.restype = ctypes.c_int
        self.lib.cudaGraphicsUnmapResources.argtypes = [
            ctypes.c_int, ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p
        ]
        self.lib.cudaGraphicsUnmapResources.restype = ctypes.c_int
        self.lib.cudaGraphicsSubResourceGetMappedArray.argtypes = [
            ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_uint, ctypes.c_uint
        ]
        self.lib.cudaGraphicsSubResourceGetMappedArray.restype = ctypes.c_int
        self.lib.cudaMemcpy2DToArrayAsync.argtypes = [
            ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_void_p,
            ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_int, ctypes.c_void_p
        ]
        self.lib.cudaMemcpy2DToArrayAsync.restype = ctypes.c_int
        res = self.lib.cudaSetDevice(int(device_id))
        if res != 0:
            raise RuntimeError(f"cudaSetDevice failed: {res}")

    def clear_last_error(self):
        try:
            self.lib.cudaGetLastError()
        except Exception:
            pass

    def register_texture(self, texture_ptr):
        resource = ctypes.c_void_p()
        res = self.lib.cudaGraphicsD3D11RegisterResource(
            ctypes.byref(resource),
            ctypes.c_void_p(_ptr_value(texture_ptr)),
            self.CUDA_GRAPHICS_REGISTER_FLAGS_NONE,
        )
        if res != 0:
            self.clear_last_error()
            raise RuntimeError(f"cudaGraphicsD3D11RegisterResource failed: {res}")
        return resource

    def unregister_resource(self, resource):
        if resource:
            self.lib.cudaGraphicsUnregisterResource(resource)

    def copy_tensor_to_texture(self, resource, src_ptr, src_pitch, copy_width_bytes, height, stream=0):
        stream_ptr = ctypes.c_void_p(stream) if stream else None
        res = self.lib.cudaGraphicsMapResources(1, ctypes.byref(resource), stream_ptr)
        if res != 0:
            self.clear_last_error()
            raise RuntimeError(f"cudaGraphicsMapResources failed: {res}")
        try:
            array = ctypes.c_void_p()
            res = self.lib.cudaGraphicsSubResourceGetMappedArray(
                ctypes.byref(array), resource, 0, 0
            )
            if res != 0:
                self.clear_last_error()
                raise RuntimeError(f"cudaGraphicsSubResourceGetMappedArray failed: {res}")
            res = self.lib.cudaMemcpy2DToArrayAsync(
                array,
                0,
                0,
                ctypes.c_void_p(src_ptr),
                ctypes.c_size_t(src_pitch),
                ctypes.c_size_t(copy_width_bytes),
                ctypes.c_size_t(height),
                self.CUDA_MEMCPY_DEVICE_TO_DEVICE,
                stream_ptr,
            )
            if res != 0:
                self.clear_last_error()
                raise RuntimeError(f"cudaMemcpy2DToArrayAsync failed: {res}")
        finally:
            unmap_res = self.lib.cudaGraphicsUnmapResources(1, ctypes.byref(resource), stream_ptr)
            if unmap_res != 0:
                self.clear_last_error()
                raise RuntimeError(f"cudaGraphicsUnmapResources failed: {unmap_res}")


HLSL_SOURCE = r"""
Texture2D texColor : register(t0);
Texture2D texDepth : register(t1);
SamplerState sampLinear : register(s0);

cbuffer Params : register(b0)
{
    float4 mvpRow0;
    float4 mvpRow1;
    float4 mvpRow2;
    float4 mvpRow3;
    float4 params;
};

#define eyeOffset params.x
#define depthStrength params.y
#define convergence params.z

struct VSOut {
    float4 pos : SV_POSITION;
    float2 uv  : TEXCOORD0;
};

VSOut vs_main(uint vertexId : SV_VertexID)
{
    static const float2 pos[4] = {
        float2(-1.0, -1.0),
        float2(-1.0,  1.0),
        float2( 1.0, -1.0),
        float2( 1.0,  1.0)
    };
    static const float2 uv[4] = {
        float2(0.0, 0.0),
        float2(0.0, 1.0),
        float2(1.0, 0.0),
        float2(1.0, 1.0)
    };
    VSOut output;
    float4 localPos = float4(pos[vertexId], 0.0, 1.0);
#if D2S_SPACE_MODE == 1
    output.pos = float4(pos[vertexId] * float2(0.35, 0.2), 0.5, 1.0);
#else
    output.pos = float4(
        dot(mvpRow0, localPos),
        dot(mvpRow1, localPos),
        dot(mvpRow2, localPos),
        dot(mvpRow3, localPos)
    );
#endif
    output.uv = uv[vertexId];
    return output;
}

float4 ps_main(VSOut input) : SV_TARGET
{
    float2 uv = float2(input.uv.x, 1.0 - input.uv.y);

#if D2S_SHADER_MODE == 0
    return float4(0.05, 0.10, 0.16, 1.0);
#elif D2S_SHADER_MODE == 1
    return float4(texColor.Sample(sampLinear, uv).rgb, 1.0);
#else
    float depth = saturate(texDepth.Sample(sampLinear, uv).r);
    float depthInv = -depth;
    float shift = (depthInv + convergence) * eyeOffset * depthStrength;
    float2 shiftedUv = uv - float2(shift, 0.0);
    if (shiftedUv.x < 0.0 || shiftedUv.x > 1.0 || shiftedUv.y < 0.0 || shiftedUv.y > 1.0) {
        return float4(0.0, 0.0, 0.0, 1.0);
    }
    return float4(texColor.Sample(sampLinear, shiftedUv).rgb, 1.0);
#endif
}
"""


class D3D11NativeRenderer:
    def __init__(self, device, context, swapchain_format=DXGI_FORMAT_R8G8B8A8_UNORM):
        self.device = device
        self.context = context
        self.swapchain_format = int(swapchain_format or DXGI_FORMAT_R8G8B8A8_UNORM)
        self.color_tex = ctypes.c_void_p()
        self.depth_tex = ctypes.c_void_p()
        self.color_srv = ctypes.c_void_p()
        self.depth_srv = ctypes.c_void_p()
        self.color_cuda = ctypes.c_void_p()
        self.depth_cuda = ctypes.c_void_p()
        self.cuda = None
        self.cuda_failed = False
        self.cuda_active_logged = False
        self.render_tex = ctypes.c_void_p()
        self.render_rtv = ctypes.c_void_p()
        self.render_target_size = None
        self.vertex_buffer = ctypes.c_void_p()
        self.constant_buffer = ctypes.c_void_p()
        self.input_layout = ctypes.c_void_p()
        self.vertex_shader = ctypes.c_void_p()
        self.pixel_shader = ctypes.c_void_p()
        self.sampler = ctypes.c_void_p()
        self.rasterizer = ctypes.c_void_p()
        self.swapchain_rtvs = {}
        self._logged_swapchain_desc = set()
        self.shader_mode = os.environ.get("D2S_D3D11_SHADER_MODE", "stereo").strip().lower()
        self.space_mode = os.environ.get("D2S_D3D11_SPACE_MODE", "world").strip().lower()
        self.debug = str(
            os.environ.get("D2S_D3D11_DEBUG", os.environ.get("D2S_OPENXR_DEBUG", "0")) or "0"
        ).strip().lower() in ("1", "true", "yes", "on")
        self._logged_world_mvp = False
        self.size = None
        self.has_frame = False
        self._init_pipeline()

    def _device_call(self, index, restype, *argtypes):
        return _com_fn(self.device, index, restype, *argtypes)

    def _context_call(self, index, restype, *argtypes):
        return _com_fn(self.context, index, restype, *argtypes)

    def _device_removed_reason(self):
        try:
            get_reason = self._device_call(39, ctypes.c_long)
            return get_reason(_ptr_value(self.device))
        except Exception:
            return None

    def _shader_source(self):
        mode_id = {
            "clear": -1,
            "solid": 0,
            "color": 1,
            "color_only": 1,
            "stereo": 2,
        }.get(self.shader_mode, 2)
        self.shader_mode = (
            "clear" if mode_id < 0 else
            "solid" if mode_id == 0 else
            "color" if mode_id == 1 else
            "stereo"
        )
        if self.debug or self.shader_mode != "stereo":
            print(f"[OpenXRViewer] D3D11 shader mode: {self.shader_mode}")
        space_id = 1 if self.space_mode in ("clip", "screen", "debug") else 0
        self.space_mode = "clip" if space_id else "world"
        if self.debug or self.space_mode != "world":
            print(f"[OpenXRViewer] D3D11 space mode: {self.space_mode}")
        return (
            f"#define D2S_SHADER_MODE {mode_id}\n"
            f"#define D2S_SPACE_MODE {space_id}\n"
            + HLSL_SOURCE
        )

    def _init_pipeline(self):
        shader_source = self._shader_source()
        vs_blob = _compile_shader(shader_source, "vs_main", "vs_5_0")
        ps_blob = _compile_shader(shader_source, "ps_main", "ps_5_0")
        try:
            create_vs = self._device_call(
                12, ctypes.c_long, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)
            )
            create_ps = self._device_call(
                15, ctypes.c_long, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)
            )
            _check_hr(create_vs(_ptr_value(self.device), _blob_ptr(vs_blob), _blob_size(vs_blob), None, ctypes.byref(self.vertex_shader)), "CreateVertexShader")
            _check_hr(create_ps(_ptr_value(self.device), _blob_ptr(ps_blob), _blob_size(ps_blob), None, ctypes.byref(self.pixel_shader)), "CreatePixelShader")

            self.input_layout = ctypes.c_void_p()
        finally:
            _release(vs_blob)
            _release(ps_blob)

        self.vertex_buffer = ctypes.c_void_p()
        self.constant_buffer = self._create_buffer(np.zeros(20, dtype=np.float32), D3D11_BIND_CONSTANT_BUFFER)

        samp_desc = D3D11SamplerDesc(
            D3D11_FILTER_MIN_MAG_MIP_LINEAR,
            D3D11_TEXTURE_ADDRESS_CLAMP,
            D3D11_TEXTURE_ADDRESS_CLAMP,
            D3D11_TEXTURE_ADDRESS_CLAMP,
            0.0,
            1,
            D3D11_COMPARISON_NEVER,
            (ctypes.c_float * 4)(0.0, 0.0, 0.0, 0.0),
            0.0,
            3.4028234663852886e38,
        )
        create_sampler = self._device_call(23, ctypes.c_long, ctypes.POINTER(D3D11SamplerDesc), ctypes.POINTER(ctypes.c_void_p))
        _check_hr(create_sampler(_ptr_value(self.device), ctypes.byref(samp_desc), ctypes.byref(self.sampler)), "CreateSamplerState")

        rast_desc = D3D11RasterizerDesc(
            D3D11_FILL_SOLID,
            D3D11_CULL_NONE,
            0,
            0,
            0.0,
            0.0,
            1,
            0,
            0,
            0,
        )
        create_rasterizer = self._device_call(
            22, ctypes.c_long, ctypes.POINTER(D3D11RasterizerDesc), ctypes.POINTER(ctypes.c_void_p)
        )
        _check_hr(
            create_rasterizer(_ptr_value(self.device), ctypes.byref(rast_desc), ctypes.byref(self.rasterizer)),
            "CreateRasterizerState",
        )

    def _create_buffer(self, array, bind_flags):
        data = np.ascontiguousarray(array)
        desc = D3D11BufferDesc(data.nbytes, D3D11_USAGE_DEFAULT, bind_flags, 0, 0, 0)
        init = D3D11SubresourceData(ctypes.c_void_p(data.ctypes.data), 0, 0)
        out = ctypes.c_void_p()
        create_buffer = self._device_call(
            3, ctypes.c_long, ctypes.POINTER(D3D11BufferDesc), ctypes.POINTER(D3D11SubresourceData), ctypes.POINTER(ctypes.c_void_p)
        )
        _check_hr(create_buffer(_ptr_value(self.device), ctypes.byref(desc), ctypes.byref(init), ctypes.byref(out)), "CreateBuffer")
        return out

    def _create_texture_srv(self, width, height, fmt):
        tex = ctypes.c_void_p()
        srv = ctypes.c_void_p()
        desc = D3D11Texture2DDesc(
            width, height, 1, 1, fmt, DXGISampleDesc(1, 0),
            D3D11_USAGE_DEFAULT, D3D11_BIND_SHADER_RESOURCE, 0, 0,
        )
        create_tex = self._device_call(
            5, ctypes.c_long, ctypes.POINTER(D3D11Texture2DDesc), ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)
        )
        _check_hr(create_tex(_ptr_value(self.device), ctypes.byref(desc), None, ctypes.byref(tex)), "CreateTexture2D")
        create_srv = self._device_call(
            7, ctypes.c_long, ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)
        )
        _check_hr(create_srv(_ptr_value(self.device), tex, None, ctypes.byref(srv)), "CreateShaderResourceView")
        return tex, srv

    def _ensure_render_target(self, width, height):
        if self.render_target_size == (width, height, self.swapchain_format):
            return
        _release(self.render_rtv)
        _release(self.render_tex)
        self.render_rtv = ctypes.c_void_p()
        self.render_tex = ctypes.c_void_p()
        desc = D3D11Texture2DDesc(
            width, height, 1, 1, self.swapchain_format, DXGISampleDesc(1, 0),
            D3D11_USAGE_DEFAULT, D3D11_BIND_RENDER_TARGET, 0, 0,
        )
        create_tex = self._device_call(
            5, ctypes.c_long, ctypes.POINTER(D3D11Texture2DDesc), ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)
        )
        _check_hr(create_tex(_ptr_value(self.device), ctypes.byref(desc), None, ctypes.byref(self.render_tex)), "CreateTexture2D(render)")
        create_rtv = self._device_call(
            9, ctypes.c_long, ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)
        )
        _check_hr(create_rtv(_ptr_value(self.device), _ptr_value(self.render_tex), None, ctypes.byref(self.render_rtv)), "CreateRenderTargetView(render)")
        self.render_target_size = (width, height, self.swapchain_format)

    def _get_texture_desc(self, texture_ptr):
        desc = D3D11Texture2DDesc()
        get_desc = _com_fn(texture_ptr, 10, None, ctypes.POINTER(D3D11Texture2DDesc))
        get_desc(texture_ptr, ctypes.byref(desc))
        return desc

    def _format_texture_desc(self, desc):
        return (
            f"size={desc.Width}x{desc.Height} fmt={desc.Format} "
            f"mips={desc.MipLevels} array={desc.ArraySize} "
            f"sample={desc.SampleDesc.Count}/{desc.SampleDesc.Quality} "
            f"usage={desc.Usage} bind=0x{desc.BindFlags:x} "
            f"cpu=0x{desc.CPUAccessFlags:x} misc=0x{desc.MiscFlags:x}"
        )

    def _get_or_create_swapchain_rtv(self, swapchain_texture):
        texture_ptr = _ptr_value(swapchain_texture)
        if not texture_ptr:
            raise RuntimeError("OpenXR D3D11 swapchain texture is null")

        cached = self.swapchain_rtvs.get(texture_ptr)
        if cached:
            return cached

        desc = self._get_texture_desc(texture_ptr)
        if self.debug and texture_ptr not in self._logged_swapchain_desc:
            print(f"[OpenXRViewer] D3D11 swapchain texture desc: {self._format_texture_desc(desc)}")
            self._logged_swapchain_desc.add(texture_ptr)

        if not (desc.BindFlags & D3D11_BIND_RENDER_TARGET):
            raise RuntimeError(
                "OpenXR D3D11 swapchain texture is not render-target bindable: "
                f"{self._format_texture_desc(desc)}"
            )

        # OpenXR runtimes may expose the swapchain image as a typeless texture.
        # The RTV itself must use the typed swapchain format requested at
        # xrCreateSwapchain, not the typeless texture storage format.
        rtv_format = self.swapchain_format or desc.Format
        if desc.ArraySize > 1:
            rtv_desc = D3D11RenderTargetViewDesc(
                rtv_format,
                D3D11_RTV_DIMENSION_TEXTURE2DARRAY,
                0,
                0,
                1,
            )
        else:
            rtv_desc = D3D11RenderTargetViewDesc(
                rtv_format,
                D3D11_RTV_DIMENSION_TEXTURE2D,
                0,
                0,
                0,
            )

        rtv = ctypes.c_void_p()
        create_rtv = self._device_call(
            9, ctypes.c_long, ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)
        )
        hr = create_rtv(_ptr_value(self.device), texture_ptr, ctypes.byref(rtv_desc), ctypes.byref(rtv))
        if hr != 0:
            removed_reason = self._device_removed_reason()
            reason_text = "" if removed_reason is None else f" removed_reason={_hr_hex(removed_reason)}"
            raise RuntimeError(
                "CreateRenderTargetView(OpenXR swapchain direct) failed: "
                f"hr={_hr_hex(hr)}{reason_text} rtv_fmt={rtv_format} {self._format_texture_desc(desc)}"
            )

        self.swapchain_rtvs[texture_ptr] = rtv
        return rtv

    def _ensure_frame_textures(self, width, height):
        if self.size == (width, height):
            return
        self._release_frame_textures()
        self.color_tex, self.color_srv = self._create_texture_srv(width, height, DXGI_FORMAT_R8G8B8A8_UNORM)
        self.depth_tex, self.depth_srv = self._create_texture_srv(width, height, DXGI_FORMAT_R32_FLOAT)
        self.size = (width, height)
        self.has_frame = False

    def _release_frame_textures(self):
        if self.cuda is not None:
            for resource in (self.color_cuda, self.depth_cuda):
                try:
                    self.cuda.unregister_resource(resource)
                except Exception:
                    pass
        self.color_cuda = ctypes.c_void_p()
        self.depth_cuda = ctypes.c_void_p()
        for ptr in (self.color_srv, self.depth_srv, self.color_tex, self.depth_tex):
            _release(ptr)
        self.color_srv = ctypes.c_void_p()
        self.depth_srv = ctypes.c_void_p()
        self.color_tex = ctypes.c_void_p()
        self.depth_tex = ctypes.c_void_p()

    def _ensure_cuda_resources(self, torch_module, device_id):
        if self.cuda_failed:
            return False
        if self.cuda is None:
            self.cuda = CUDART_D3D11(torch_module, self.device, device_id)
        if not self.color_cuda:
            self.color_cuda = self.cuda.register_texture(self.color_tex)
        if not self.depth_cuda:
            self.depth_cuda = self.cuda.register_texture(self.depth_tex)
        return True

    def _update_frame_cuda(self, torch_module, rgb, depth):
        if not (
            hasattr(rgb, "is_cuda") and rgb.is_cuda and
            hasattr(depth, "is_cuda") and depth.is_cuda
        ):
            return False

        device = depth.device
        device_id = 0 if device.index is None else int(device.index)
        depth_gpu = depth.detach()
        h, w = depth_gpu.shape[:2]
        self._ensure_frame_textures(w, h)
        self._ensure_cuda_resources(torch_module, device_id)

        rgb_gpu = rgb.detach()
        if rgb_gpu.device != device:
            rgb_gpu = rgb_gpu.to(device, non_blocking=True)
        if rgb_gpu.ndim == 3 and rgb_gpu.shape[0] in (3, 4):
            rgb_hwc = rgb_gpu[:3].permute(1, 2, 0)
        elif rgb_gpu.ndim == 3 and rgb_gpu.shape[-1] >= 3:
            rgb_hwc = rgb_gpu[..., :3]
        else:
            raise RuntimeError(f"Unsupported RGB tensor shape for D3D11 CUDA upload: {tuple(rgb_gpu.shape)}")

        if rgb_hwc.shape[0] != h or rgb_hwc.shape[1] != w:
            raise RuntimeError(
                f"RGB/depth size mismatch for D3D11 CUDA upload: rgb={tuple(rgb_hwc.shape)} depth={(h, w)}"
            )

        rgba = torch_module.empty((h, w, 4), device=device, dtype=torch_module.uint8)
        rgba[..., :3] = rgb_hwc.contiguous().clamp(0, 255).to(torch_module.uint8)
        rgba[..., 3] = 255

        depth_f = depth_gpu.contiguous().float()
        depth_f = torch_module.nan_to_num(depth_f, nan=0.0, posinf=1.0, neginf=0.0).clamp_(0.0, 1.0)

        stream = torch_module.cuda.current_stream(device_id)
        stream_ptr = stream.cuda_stream
        self.cuda.copy_tensor_to_texture(self.color_cuda, rgba.data_ptr(), w * 4, w * 4, h, stream_ptr)
        self.cuda.copy_tensor_to_texture(self.depth_cuda, depth_f.data_ptr(), w * 4, w * 4, h, stream_ptr)
        stream.synchronize()
        if not self.cuda_active_logged:
            print("[OpenXRViewer] D3D11 CUDA upload active (device-to-device)")
            self.cuda_active_logged = True
        self.has_frame = True
        return w, h

    def update_frame(self, rgb, depth):
        try:
            import torch
        except Exception:
            torch = None

        if torch is not None and not self.cuda_failed:
            try:
                result = self._update_frame_cuda(torch, rgb, depth)
                if result:
                    return result
            except Exception as e:
                self.cuda_failed = True
                if self.cuda is not None:
                    try:
                        self.cuda.clear_last_error()
                    except Exception:
                        pass
                try:
                    self._release_frame_textures()
                except Exception:
                    pass
                print(f"[OpenXRViewer] D3D11 CUDA upload unavailable: {e}; falling back to CPU upload")

        if torch is not None and hasattr(rgb, "detach"):
            rgb_np = rgb.detach().permute(1, 2, 0).contiguous().clamp(0, 255).to(torch.uint8).cpu().numpy()
        else:
            rgb_np = np.asarray(rgb, dtype=np.uint8)
        if torch is not None and hasattr(depth, "detach"):
            depth_np = depth.detach().contiguous().float().cpu().numpy()
        else:
            depth_np = np.asarray(depth, dtype=np.float32)

        h, w = depth_np.shape[:2]
        self._ensure_frame_textures(w, h)
        rgba = np.empty((h, w, 4), dtype=np.uint8)
        rgba[:, :, :3] = np.ascontiguousarray(rgb_np[:, :, :3])
        rgba[:, :, 3] = 255
        depth_f = depth_np.astype(np.float32, copy=False)
        depth_f = np.nan_to_num(depth_f, nan=0.0, posinf=1.0, neginf=0.0)
        depth_f = np.ascontiguousarray(np.clip(depth_f, 0.0, 1.0))
        self._update_subresource(self.color_tex, rgba.ctypes.data, w * 4)
        self._update_subresource(self.depth_tex, depth_f.ctypes.data, w * 4)
        self.has_frame = True
        return w, h

    def _update_subresource(self, dst, src_ptr, row_pitch):
        fn = self._context_call(
            48, None, ctypes.c_void_p, ctypes.c_uint, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_uint, ctypes.c_uint,
        )
        fn(_ptr_value(self.context), _ptr_value(dst), 0, None, src_ptr, row_pitch, 0)

    def _log_world_mvp_once(self, mvp):
        if not self.debug or self._logged_world_mvp or self.space_mode != "world":
            return
        self._logged_world_mvp = True
        try:
            mat = np.asarray(mvp, dtype=np.float32)
            corners = np.array([
                [-1.0, -1.0, 0.0, 1.0],
                [-1.0,  1.0, 0.0, 1.0],
                [ 1.0, -1.0, 0.0, 1.0],
                [ 1.0,  1.0, 0.0, 1.0],
            ], dtype=np.float32)
            clip = (mat @ corners.T).T
            ndc = clip[:, :3] / np.maximum(np.abs(clip[:, 3:4]), 1e-6)
            parts = []
            for i in range(4):
                parts.append(
                    f"{i}:clip=({clip[i,0]:.3f},{clip[i,1]:.3f},{clip[i,2]:.3f},{clip[i,3]:.3f}) "
                    f"ndc=({ndc[i,0]:.3f},{ndc[i,1]:.3f},{ndc[i,2]:.3f})"
                )
            print("[OpenXRViewer] D3D11 world screen corners " + " | ".join(parts))
        except Exception as e:
            print(f"[OpenXRViewer] D3D11 world MVP debug failed: {e}")

    def render_eye(self, swapchain_texture, width, height, eye_index, ipd, depth_strength, convergence, mvp):
        rtv = self._get_or_create_swapchain_rtv(swapchain_texture)
        try:
            clear = self._context_call(50, None, ctypes.c_void_p, ctypes.POINTER(ctypes.c_float))
            color = (ctypes.c_float * 4)(0.0, 0.0, 0.0, 1.0)
            clear(_ptr_value(self.context), _ptr_value(rtv), color)
            if self.shader_mode == "clear":
                removed_reason = self._device_removed_reason()
                if removed_reason not in (0, None):
                    raise RuntimeError(f"D3D11 device removed after ClearRenderTargetView: removed_reason={_hr_hex(removed_reason)}")
                return

            viewport = D3D11Viewport(0.0, 0.0, float(width), float(height), 0.0, 1.0)
            self._context_call(44, None, ctypes.c_uint, ctypes.POINTER(D3D11Viewport))(_ptr_value(self.context), 1, ctypes.byref(viewport))

            rtv_arr = (ctypes.c_void_p * 1)(_ptr_value(rtv))
            self._context_call(33, None, ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p)(_ptr_value(self.context), 1, rtv_arr, None)

            self._context_call(17, None, ctypes.c_void_p)(_ptr_value(self.context), 0)
            self._context_call(43, None, ctypes.c_void_p)(_ptr_value(self.context), _ptr_value(self.rasterizer))
            self._context_call(24, None, ctypes.c_uint)(_ptr_value(self.context), D3D11_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP)
            self._context_call(11, None, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint)(_ptr_value(self.context), _ptr_value(self.vertex_shader), None, 0)
            self._context_call(9, None, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint)(_ptr_value(self.context), _ptr_value(self.pixel_shader), None, 0)

            eye_sign = -1.0 if eye_index == 0 else 1.0
            self._log_world_mvp_once(mvp)
            constants = np.zeros(20, dtype=np.float32)
            constants[:16] = np.asarray(mvp, dtype=np.float32).reshape(16)
            constants[16:20] = np.array([eye_sign * ipd * 0.5, depth_strength, convergence, 0.0], dtype=np.float32)
            self._update_subresource(self.constant_buffer, constants.ctypes.data, constants.nbytes)
            cb_arr = (ctypes.c_void_p * 1)(_ptr_value(self.constant_buffer))
            self._context_call(7, None, ctypes.c_uint, ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p))(_ptr_value(self.context), 0, 1, cb_arr)
            self._context_call(16, None, ctypes.c_uint, ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p))(_ptr_value(self.context), 0, 1, cb_arr)

            srv_arr = (ctypes.c_void_p * 2)(_ptr_value(self.color_srv), _ptr_value(self.depth_srv))
            self._context_call(8, None, ctypes.c_uint, ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p))(_ptr_value(self.context), 0, 2, srv_arr)
            sampler_arr = (ctypes.c_void_p * 1)(_ptr_value(self.sampler))
            self._context_call(10, None, ctypes.c_uint, ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p))(_ptr_value(self.context), 0, 1, sampler_arr)
            self._context_call(13, None, ctypes.c_uint, ctypes.c_uint)(_ptr_value(self.context), 4, 0)
            removed_reason = self._device_removed_reason()
            if removed_reason not in (0, None):
                raise RuntimeError(f"D3D11 device removed after Draw: removed_reason={_hr_hex(removed_reason)}")
        finally:
            null_srvs = (ctypes.c_void_p * 2)(0, 0)
            null_rtvs = (ctypes.c_void_p * 1)(0)
            try:
                self._context_call(8, None, ctypes.c_uint, ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p))(_ptr_value(self.context), 0, 2, null_srvs)
            except Exception:
                pass
            try:
                self._context_call(33, None, ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p)(_ptr_value(self.context), 1, null_rtvs, None)
            except Exception:
                pass

    def cleanup(self):
        for rtv in self.swapchain_rtvs.values():
            _release(rtv)
        self.swapchain_rtvs.clear()
        self._release_frame_textures()
        for attr in (
            "sampler", "pixel_shader", "vertex_shader", "input_layout",
            "rasterizer", "constant_buffer", "vertex_buffer", "render_rtv", "render_tex",
        ):
            ptr = getattr(self, attr, None)
            _release(ptr)
            setattr(self, attr, ctypes.c_void_p())
        self.render_target_size = None

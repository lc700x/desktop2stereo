"""D3D11 swapchain backend and GPU interop mixin for OpenXR viewer."""

import ctypes
import sys

from OpenGL.GL import (
    glGenFramebuffers, glBindFramebuffer, glFramebufferTexture2D,
    glDeleteFramebuffers, glCheckFramebufferStatus,
    GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D,
    GL_FRAMEBUFFER_COMPLETE, GL_RGBA8,
    glGenBuffers, glDeleteBuffers, glBindBuffer, glBufferData,
    GL_PIXEL_PACK_BUFFER, GL_STREAM_READ,
    GL_RGBA, GL_BGRA, GL_UNSIGNED_BYTE,
    glBindTexture, glGenTextures, glDeleteTextures,
    glFenceSync, glClientWaitSync, glDeleteSync,
    GL_SYNC_GPU_COMMANDS_COMPLETE, GL_SYNC_FLUSH_COMMANDS_BIT,
    glRenderbufferStorage, glFramebufferRenderbuffer, glDeleteRenderbuffers,
    GL_RENDERBUFFER, GL_DEPTH_ATTACHMENT, GL_DEPTH_COMPONENT24,
    glBindRenderbuffer, glGenRenderbuffers,
)
from OpenGL.GL import glReadPixels, glFlush, glMapBuffer, glUnmapBuffer
from OpenGL.GL import GL_READ_ONLY, GL_MAP_UNSYNCHRONIZED_BIT

import moderngl
import numpy as np

try:
    import xr
except ImportError:
    xr = None

from . import render as _render
from .render import (
    _create_d3d11_device, _create_d3d11_shared_texture,
    _d3d11_update_subresource,
)
from .constants import (
    _DXGI_FORMAT_R8G8B8A8_UNORM_SRGB, _DXGI_FORMAT_R8G8B8A8_UNORM,
    _DXGI_FORMAT_B8G8R8A8_UNORM_SRGB, _DXGI_FORMAT_B8G8R8A8_UNORM,
    _D3D11_PREFERRED_FORMATS,
)


class D3D11BackendMixin:
    """D3D11 swapchain + GPU interop methods for OpenXRViewer.

    Expects the owning class to provide:
      self._xr_instance, self._xr_system_id, self._xr_session,
      self._xr_space, self._xr_ref_space_type,
      self._xr_swapchains, self._swapchain_images, self._swapchain_sizes,
      self._use_d3d11, self._d3d11_device, self._d3d11_context,
      self._d3d11_swapchain_fmt, self._swapchain_is_bgra,
      self._offscreen_fbo_cache, self._d3d11_pbo_cache,
      self._nv_dx_device, self._nv_dx_objects,
      self._ext_shared_tex, self._interop_mode,
      self._blit_prog, self._blit_vao, self.ctx,
      self._init_controller_actions()
    """


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
        self._xr_ref_space_type = ref_type
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

    # GPU interop helpers

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

        BGRA swapchains (common on WMR) are supported through EXT_memory_object
        with a final GPU swizzle blit. NV_DX_interop2 still requires RGBA, so
        it is skipped for BGRA.
        """
        if not sys.platform == "win32":
            return

        is_nv = self._is_nvidia_gpu() and not self._swapchain_is_bgra

        if is_nv and _render._load_nv_dx_interop():
            try:
                self._init_interop_nv()
                self._interop_mode = 'nv_dx'
                print("[OpenXRViewer] GPU interop active: NV_DX_interop2 (zero-copy)")
                return
            except Exception as e:
                print(f"[OpenXRViewer] NV_DX_interop2 setup failed: {e}")

        if _render._load_ext_memory_object():
            try:
                self._init_interop_ext_mem()
                self._interop_mode = 'ext_mem'
                print("[OpenXRViewer] GPU interop active: EXT_memory_object (GPU-side blit)")
                return
            except Exception as e:
                print(f"[OpenXRViewer] EXT_memory_object setup failed: {e}")

        self._interop_mode = None
        print("[OpenXRViewer] GPU interop unavailable using PBO fallback")

    def _init_interop_nv(self):
        """Set up WGL_NV_DX_interop2: register the D3D11 device with GL.

        Individual swapchain textures are registered per-frame the first time
        each image index is seen (see _get_or_create_nv_interop_fbo).
        """
        self._nv_dx_device = _render._wglDXOpenDeviceNV(self._d3d11_device)
        if not self._nv_dx_device:
            raise RuntimeError("wglDXOpenDeviceNV returned NULL")

    def _get_or_create_nv_interop_fbo(self, eye_index, img_index, d3d11_tex, w, h):
        """Register a swapchain D3D11 texture with GL via NV_DX_interop2.

        Each unique (eye, img_index) pair is registered once and cached.
        Returns (mgl_fbo, raw_fbo_id) for direct rendering into the D3D11 texture.
        """
        key = (eye_index, img_index)
        if key in self._nv_dx_objects:
            gl_tex, raw_fbo, dx_obj = self._nv_dx_objects[key]
            return self.ctx.detect_framebuffer(raw_fbo), raw_fbo

        gl_tex = glGenTextures(1)
        # Register the D3D11 texture as a GL texture
        dx_obj = _render._wglDXRegisterObjectNV(
            self._nv_dx_device,
            d3d11_tex,
            gl_tex,
            GL_TEXTURE_2D,
            0x0002,  # WGL_ACCESS_WRITE_DISCARD_NV the driver knows we overwrite
        )
        if not dx_obj:
            glDeleteTextures(1, [gl_tex])
            raise RuntimeError(f"wglDXRegisterObjectNV failed for eye {eye_index} img {img_index}")

        # Set up FBO attached to the registered texture
        raw_fbo = glGenFramebuffers(1)
        glBindFramebuffer(GL_FRAMEBUFFER, raw_fbo)
        # Lock, attach, unlock
        _render._wglDXLockObjectsNV(self._nv_dx_device, 1, ctypes.byref(dx_obj))
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, gl_tex, 0)
        _render._wglDXUnlockObjectsNV(self._nv_dx_device, 1, ctypes.byref(dx_obj))
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
            _render._glCreateMemoryObjectsEXT(1, ctypes.byref(mem_obj))
            _render._glImportMemoryWin32HandleEXT(
                mem_obj, sc_w * sc_h * 4,
                _render._GL_HANDLE_TYPE_OPAQUE_WIN32_EXT,
                nt_handle,
            )

            # Create GL texture backed by the imported memory
            gl_tex = glGenTextures(1)
            glBindTexture(GL_TEXTURE_2D, gl_tex)
            _render._glTextureStorageMem2DEXT(gl_tex, 1, GL_RGBA8, sc_w, sc_h, mem_obj, 0)
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

    def _make_gl_fence(self):
        """Insert a GL fence and flush so the driver submits it promptly."""
        fence = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0)
        glFlush()
        return fence

    def _d3d11_copy_resource(self, dst, src):
        """ID3D11DeviceContext::CopyResource via vtable index 47."""
        ctx = self._d3d11_context
        vtbl = ctypes.cast(ctx, ctypes.POINTER(ctypes.c_void_p)).contents.value
        copy_fn = ctypes.CFUNCTYPE(
            None,
            ctypes.c_void_p,  # this
            ctypes.c_void_p,  # pDstResource
            ctypes.c_void_p,  # pSrcResource
        )(ctypes.cast(vtbl + 47 * ctypes.sizeof(ctypes.c_void_p),
                    ctypes.POINTER(ctypes.c_void_p)).contents.value)
        copy_fn(ctx, dst, src)

    def _swizzle_blit_to_shared(self, eye_index, src_mgl_tex):
        """Copy an offscreen RGBA eye render into the EXT-shared GL texture."""
        _, _, _, mgl_fbo, _ = self._ext_shared_tex[eye_index]
        sc_w, sc_h = self._swapchain_sizes[eye_index]
        mgl_fbo.use()
        self.ctx.viewport = (0, 0, sc_w, sc_h)
        self.ctx.disable(moderngl.DEPTH_TEST)
        self.ctx.disable(moderngl.BLEND)
        src_mgl_tex.use(location=4)
        self._blit_prog['u_swap_rb'].value = 1 if self._swapchain_is_bgra else 0
        self._blit_vao.render(moderngl.TRIANGLE_STRIP)

    def _wait_and_blit_ext(self, eye_index, d3d11_swapchain_tex, fence):
        """Wait on this eye's GL fence, then copy the shared texture to OpenXR."""
        if fence is not None:
            glClientWaitSync(fence, GL_SYNC_FLUSH_COMMANDS_BIT, 16_000_000)
            glDeleteSync(fence)
        self._d3d11_copy_resource(
            d3d11_swapchain_tex,
            self._ext_shared_tex[eye_index][0],
        )

    def _blit_ext_to_swapchain(self, eye_index, d3d11_swapchain_tex):
        """Legacy single-phase EXT-memory copy path using a fence, not glFinish."""
        self._wait_and_blit_ext(
            eye_index,
            d3d11_swapchain_tex,
            self._make_gl_fence(),
        )

    def _cleanup_interop(self):
        """Release all GPU interop resources."""
        if self._interop_mode == 'nv_dx' and self._nv_dx_device:
            for (gl_tex, raw_fbo, dx_obj) in self._nv_dx_objects.values():
                try:
                    _render._wglDXUnregisterObjectNV(self._nv_dx_device, dx_obj)
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
                _render._wglDXCloseDeviceNV(self._nv_dx_device)
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
                    _render._glDeleteMemoryObjectsEXT(1, ctypes.byref(ctypes.c_uint(mem_obj)))
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

    def _get_or_create_offscreen_fbo(self, eye_index, image_index, w, h):
        """Return a ModernGL FBO backed by an RGBA texture + depth renderbuffer.

        Used in the D3D11 path: ModernGL renders into this offscreen FBO, then
        _blit_gl_to_d3d11() reads it back and uploads it to the D3D11 swapchain image.
        Cache entry: (mgl_fbo, raw_id, mgl_tex, w, h, depth_rb)
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
                glDeleteRenderbuffers(1, [cached[5]])  # depth_rb
            except Exception:
                pass

        raw_id = glGenFramebuffers(1)
        mgl_tex = self.ctx.texture((w, h), 4, dtype='f1')
        glBindFramebuffer(GL_FRAMEBUFFER, raw_id)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                            GL_TEXTURE_2D, mgl_tex.glo, 0)
        depth_rb = glGenRenderbuffers(1)
        glBindRenderbuffer(GL_RENDERBUFFER, depth_rb)
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, w, h)
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depth_rb)
        glBindRenderbuffer(GL_RENDERBUFFER, 0)
        status = glCheckFramebufferStatus(GL_FRAMEBUFFER)
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        if status != GL_FRAMEBUFFER_COMPLETE:
            raise RuntimeError(
                f"[OpenXRViewer] Offscreen FBO incomplete for eye {eye_index}: {status:#x}"
            )
        mgl_fbo = self.ctx.detect_framebuffer(raw_id)
        self._offscreen_fbo_cache[key] = (mgl_fbo, raw_id, mgl_tex, w, h, depth_rb)
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
        produces top-down rows no CPU row-reversal needed.  The mapped PBO
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


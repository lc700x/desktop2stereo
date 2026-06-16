"""
Test script for MIGraphX on Windows via multiple paths:
  1. Windows ML (WinRT) API - uses system onnxruntime.dll with all system EPs
  2. ONNX Runtime pip package - MIGraphXExecutionProvider
  3. MIGraphX C API (ctypes) - direct migraphx_c.dll from migraphx_dll/

Requirements:
    pip install onnx onnxruntime numpy

    Optional (for Windows ML WinRT test):
    pip install winrt-runtime winrt-Windows.AI.MachineLearning winrt-Windows.Storage

    Optional (for MIGraphX C API test):
    Place MIGraphX DLLs in migraphx_dll/ folder

Usage:
    python3/python.exe test_migraphx.py
"""

import sys
import os
import time
import numpy as np

# ─── Helper: create a simple ONNX model ────────────────────────────────────
def create_test_onnx(path="test_migraphx_model.onnx", h=64, w=64):
    import onnx
    from onnx import helper, TensorProto
    X = helper.make_tensor_value_info("pixel_values", TensorProto.FLOAT, [1, 3, h, w])
    Y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3, h, w])
    node = helper.make_node("Relu", ["pixel_values"], ["output"])
    graph = helper.make_graph([node], "test_relu", [X], [Y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    onnx.save(model, path)
    return path


# ─── Test 1: Windows ML (WinRT) ────────────────────────────────────────────
def test_windows_ml():
    """Use the Windows.AI.MachineLearning WinRT API.
    This API uses the SYSTEM onnxruntime.dll (C:\\Windows\\System32) which
    may include execution providers delivered via Windows Update (e.g.
    KB5096143 for MIGraphX).  The pip onnxruntime package is NOT involved.
    """
    print("=" * 60)
    print("TEST 1: Windows ML (WinRT API)")
    print("=" * 60)

    if sys.platform != "win32":
        print("  SKIP: Windows only")
        return False

    try:
        # winrt-runtime + winrt-Windows.AI.MachineLearning
        # pip install winrt-runtime winrt-Windows.AI.MachineLearning
        from winrt.windows.ai.machinelearning import (  # type: ignore[import-not-found]
            LearningModel, LearningModelDevice, LearningModelDeviceKind,
            LearningModelSession, LearningModelBinding,
            TensorFloat,
        )
        import asyncio
    except ImportError:
        print("  winrt packages not installed. Install with:")
        print("    pip install winrt-runtime winrt-Windows.AI.MachineLearning winrt-Windows.Storage")
        print("  Trying alternative ctypes approach...")
        return test_windows_ml_ctypes()

    async def _run():
        onnx_path = create_test_onnx()
        abs_path = os.path.abspath(onnx_path)

        # Load model
        model = LearningModel.load_from_file_path(abs_path)
        print(f"  Model loaded: {abs_path}")

        # Try devices: Default (DML), DirectXHighPerformance, CPU
        for kind_name, kind in [
            ("Default (DML)", LearningModelDeviceKind.DEFAULT),
            ("DirectX HighPerf", LearningModelDeviceKind.DIRECT_X_HIGH_PERFORMANCE),
            ("CPU", LearningModelDeviceKind.CPU),
        ]:
            try:
                device = LearningModelDevice(kind)
                session = LearningModelSession(model, device)
                binding = LearningModelBinding(session)

                # Create input tensor
                input_data = np.random.randn(1, 3, 64, 64).astype(np.float32)
                tensor = TensorFloat.create_from_array([1, 3, 64, 64], input_data.flatten().tolist())
                binding.bind("pixel_values", tensor)

                # Run
                t0 = time.perf_counter()
                session.evaluate(binding, "")
                dt = (time.perf_counter() - t0) * 1000

                print(f"  {kind_name}: OK ({dt:.1f} ms)")
            except Exception as e:
                print(f"  {kind_name}: FAILED - {e}")

        os.remove(onnx_path)
        return True

    try:
        return asyncio.run(_run())
    except Exception as e:
        print(f"  WinRT test failed: {e}")
        return False


def test_windows_ml_ctypes():
    """Fallback: use the system onnxruntime.dll directly via ctypes ORT C API."""
    import ctypes

    print("  Using system onnxruntime.dll via ORT C API...")
    try:
        ort = ctypes.CDLL(r"C:\Windows\System32\onnxruntime.dll")
    except OSError as e:
        print(f"  Cannot load system ORT: {e}")
        return False

    # Check version
    try:
        vi = ctypes.windll.version
        size = vi.GetFileVersionInfoSizeW(r"C:\Windows\System32\onnxruntime.dll", None)
        if size:
            buf = ctypes.create_string_buffer(size)
            vi.GetFileVersionInfoW(r"C:\Windows\System32\onnxruntime.dll", 0, size, buf)
            p = ctypes.c_void_p()
            l = ctypes.c_uint()
            if vi.VerQueryValueW(buf, chr(92), ctypes.byref(p), ctypes.byref(l)):
                info = ctypes.cast(p, ctypes.POINTER(ctypes.c_uint32 * 13)).contents
                ms, ls = info[2], info[3]
                ver = f"{(ms>>16)&0xffff}.{ms&0xffff}.{(ls>>16)&0xffff}.{ls&0xffff}"
                print(f"  System ORT version: {ver}")
    except Exception:
        pass

    # Check provider exports by probing known function names via ctypes
    providers = []
    for name in ["OrtSessionOptionsAppendExecutionProvider_CPU",
                 "OrtSessionOptionsAppendExecutionProvider_DML",
                 "OrtSessionOptionsAppendExecutionProviderEx_DML",
                 "OrtSessionOptionsAppendExecutionProvider_MIGraphX",
                 "OrtSessionOptionsAppendExecutionProviderEx_MIGraphX",
                 "OrtSessionOptionsAppendExecutionProvider_ROCM",
                 "OrtSessionOptionsAppendExecutionProvider_CUDA"]:
        try:
            getattr(ort, name)
            providers.append(name.split("_")[-1])
        except AttributeError:
            pass
    print(f"  Available provider exports: {providers}")

    has_migraphx = any("MIGraph" in p for p in providers)
    if has_migraphx:
        print("  MIGraphX EP: FOUND in system ORT!")
    else:
        print("  MIGraphX EP: NOT found in system ORT")
        print("  Install KB5096143 via Windows Update to get MIGraphX EP")

    return has_migraphx


# ─── Test 2: ONNX Runtime pip package ──────────────────────────────────────
def test_onnxruntime_pip():
    """Use the pip-installed onnxruntime package with MIGraphXExecutionProvider."""
    print()
    print("=" * 60)
    print("TEST 2: ONNX Runtime (pip) - MIGraphXExecutionProvider")
    print("=" * 60)

    try:
        import onnxruntime as ort
    except ImportError:
        print("  SKIP: onnxruntime not installed")
        return False

    print(f"  ORT version: {ort.__version__}")
    available = ort.get_available_providers()
    print(f"  Available providers: {available}")

    has_migraphx = "MIGraphXExecutionProvider" in available
    has_rocm = "ROCMExecutionProvider" in available
    print(f"  MIGraphXExecutionProvider: {'YES' if has_migraphx else 'NO'}")
    print(f"  ROCMExecutionProvider: {'YES' if has_rocm else 'NO'}")

    if not has_migraphx:
        print("  To get MIGraphX EP, install: pip install onnxruntime-rocm")
        # Still test with fallback providers
        test_providers = [p for p in ["ROCMExecutionProvider", "DmlExecutionProvider",
                                       "CPUExecutionProvider"] if p in available]
    else:
        test_providers = ["MIGraphXExecutionProvider", "CPUExecutionProvider"]

    onnx_path = create_test_onnx()
    input_data = np.random.randn(1, 3, 64, 64).astype(np.float32)

    for provider in test_providers:
        try:
            sess = ort.InferenceSession(onnx_path, providers=[provider])
            active = sess.get_providers()
            t0 = time.perf_counter()
            out = sess.run(None, {"pixel_values": input_data})
            dt = (time.perf_counter() - t0) * 1000
            neg = (out[0] < -1e-6).sum()
            print(f"  {provider}: OK ({dt:.1f} ms, active={active}, negatives={neg})")
        except Exception as e:
            print(f"  {provider}: FAILED - {e}")

    os.remove(onnx_path)
    return has_migraphx


# ─── Test 3: MIGraphX C API (ctypes) ──────────────────────────────────────
def test_migraphx_c_api():
    """Use migraphx_c.dll from migraphx_dll/ folder via ctypes."""
    print()
    print("=" * 60)
    print("TEST 3: MIGraphX C API (ctypes) - migraphx_dll/")
    print("=" * 60)

    dll_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "migraphx_dll")
    if not os.path.isdir(dll_dir):
        print(f"  SKIP: {dll_dir} not found")
        return False

    dll_path = os.path.join(dll_dir, "migraphx_c.dll")
    if not os.path.isfile(dll_path):
        print(f"  SKIP: migraphx_c.dll not found in {dll_dir}")
        return False

    import ctypes
    os.add_dll_directory(dll_dir)
    os.environ["PATH"] = dll_dir + os.pathsep + os.environ.get("PATH", "")

    try:
        mgx = ctypes.CDLL(dll_path)
        print(f"  migraphx_c.dll loaded OK")
    except OSError as e:
        print(f"  FAILED to load: {e}")
        return False

    # List DLLs
    dlls = sorted(os.listdir(dll_dir))
    print(f"  DLLs: {', '.join(dlls)}")

    # Create ONNX
    import onnx
    from onnx import helper, TensorProto
    X = helper.make_tensor_value_info("pixel_values", TensorProto.FLOAT, [1, 3, 64, 64])
    Y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3, 64, 64])
    node = helper.make_node("Relu", ["pixel_values"], ["output"])
    graph = helper.make_graph([node], "test", [X], [Y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    onnx_bytes = model.SerializeToString()

    # Parse ONNX
    program = ctypes.c_void_p(0)
    onnx_opts = ctypes.c_void_p(0)
    mgx.migraphx_onnx_options_create(ctypes.byref(onnx_opts))
    buf = ctypes.create_string_buffer(onnx_bytes)
    s = mgx.migraphx_parse_onnx_buffer(ctypes.byref(program), buf, len(onnx_bytes), onnx_opts)
    print(f"  ONNX parse: {'OK' if s == 0 else f'FAILED (status={s})'}")
    if s != 0:
        return False

    results = {}

    # Test each target
    for target_name in ["gpu", "ref"]:
        target = ctypes.c_void_p(0)
        s = mgx.migraphx_target_create(ctypes.byref(target), target_name.encode())
        if s != 0:
            print(f"  Target '{target_name}': create FAILED (status={s})")
            continue

        # Clone program for each target (re-parse)
        prog = ctypes.c_void_p(0)
        onnx_opts2 = ctypes.c_void_p(0)
        mgx.migraphx_onnx_options_create(ctypes.byref(onnx_opts2))
        mgx.migraphx_parse_onnx_buffer(ctypes.byref(prog), buf, len(onnx_bytes), onnx_opts2)

        compile_opts = ctypes.c_void_p(0)
        mgx.migraphx_compile_options_create(ctypes.byref(compile_opts))

        try:
            t0 = time.perf_counter()
            s = mgx.migraphx_program_compile(prog, target, compile_opts)
            compile_ms = (time.perf_counter() - t0) * 1000
        except Exception as e:
            print(f"  Target '{target_name}': compile CRASHED - {e}")
            if target_name == "gpu":
                print("    (HIP version mismatch: migraphx_dll uses HIP 6, system has ROCm 7)")
            mgx.migraphx_compile_options_destroy(compile_opts)
            mgx.migraphx_target_destroy(target)
            mgx.migraphx_onnx_options_destroy(onnx_opts2)
            mgx.migraphx_program_destroy(prog)
            results[target_name] = False
            continue

        if s != 0:
            print(f"  Target '{target_name}': compile FAILED (status={s})")
            mgx.migraphx_compile_options_destroy(compile_opts)
            mgx.migraphx_target_destroy(target)
            mgx.migraphx_onnx_options_destroy(onnx_opts2)
            mgx.migraphx_program_destroy(prog)
            results[target_name] = False
            continue

        # Get parameter type from compiled program
        param_shapes = ctypes.c_void_p(0)
        mgx.migraphx_program_get_parameter_shapes(ctypes.byref(param_shapes), prog)
        param_shape = ctypes.c_void_p(0)
        mgx.migraphx_program_parameter_shapes_get(ctypes.byref(param_shape), param_shapes, b"pixel_values")
        param_type = ctypes.c_int(0)
        mgx.migraphx_shape_type(ctypes.byref(param_type), param_shape)

        # Create input
        input_data = np.random.randn(1, 3, 64, 64).astype(np.float32)
        shape = ctypes.c_void_p(0)
        lens = (ctypes.c_size_t * 4)(1, 3, 64, 64)
        mgx.migraphx_shape_create.argtypes = [
            ctypes.POINTER(ctypes.c_void_p), ctypes.c_int,
            ctypes.POINTER(ctypes.c_size_t), ctypes.c_size_t
        ]
        mgx.migraphx_shape_create(ctypes.byref(shape), param_type.value, lens, 4)

        arg = ctypes.c_void_p(0)
        mgx.migraphx_argument_create(ctypes.byref(arg), shape, input_data.ctypes.data_as(ctypes.c_void_p))

        params = ctypes.c_void_p(0)
        mgx.migraphx_program_parameters_create(ctypes.byref(params))
        mgx.migraphx_program_parameters_add(params, b"pixel_values", arg)

        # Run inference
        out_results = ctypes.c_void_p(0)
        try:
            t0 = time.perf_counter()
            s = mgx.migraphx_program_run(ctypes.byref(out_results), prog, params)
            run_ms = (time.perf_counter() - t0) * 1000
        except Exception as e:
            print(f"  Target '{target_name}': run CRASHED - {e}")
            results[target_name] = False
            mgx.migraphx_program_parameters_destroy(params)
            mgx.migraphx_compile_options_destroy(compile_opts)
            mgx.migraphx_target_destroy(target)
            mgx.migraphx_onnx_options_destroy(onnx_opts2)
            mgx.migraphx_program_destroy(prog)
            continue

        if s == 0:
            out_arg = ctypes.c_void_p(0)
            mgx.migraphx_arguments_get(ctypes.byref(out_arg), out_results, 0)
            out_buf = ctypes.c_void_p(0)
            mgx.migraphx_argument_buffer(ctypes.byref(out_buf), out_arg)
            out = np.ctypeslib.as_array(
                ctypes.cast(out_buf, ctypes.POINTER(ctypes.c_float)),
                shape=(1 * 3 * 64 * 64,)
            ).copy()
            neg_in = int((input_data < 0).sum())
            neg_out = int((out < -1e-6).sum())
            correct = neg_out == 0 and out.max() > 0
            print(f"  Target '{target_name}': OK (compile={compile_ms:.0f}ms, run={run_ms:.1f}ms, "
                  f"ReLU correct={correct})")
            results[target_name] = True
        else:
            print(f"  Target '{target_name}': run FAILED (status={s})")
            results[target_name] = False

        # Cleanup
        mgx.migraphx_program_parameters_destroy(params)
        mgx.migraphx_compile_options_destroy(compile_opts)
        mgx.migraphx_target_destroy(target)
        mgx.migraphx_onnx_options_destroy(onnx_opts2)
        mgx.migraphx_program_destroy(prog)

    mgx.migraphx_onnx_options_destroy(onnx_opts)
    mgx.migraphx_program_destroy(program)

    return results.get("gpu", False)


# ─── Main ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("MIGraphX Availability Test")
    print(f"Python: {sys.version}")
    print(f"Platform: {sys.platform}")
    print()

    r1 = test_windows_ml()
    r2 = test_onnxruntime_pip()
    r3 = test_migraphx_c_api()

    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Windows ML MIGraphX EP:    {'AVAILABLE' if r1 else 'NOT available'}")
    print(f"  ORT pip MIGraphX EP:       {'AVAILABLE' if r2 else 'NOT available'}")
    print(f"  MIGraphX C API (GPU):      {'AVAILABLE' if r3 else 'NOT available'}")
    print()
    if not any([r1, r2, r3]):
        print("  No MIGraphX acceleration available.")
        print("  Recommended: use torch.compile (inductor/Triton) for AMD ROCm.")
        print()
        print("  To enable MIGraphX in the future:")
        print("    - Windows ML: install KB5096143 via Windows Update")
        print("    - C API GPU:  needs MIGraphX DLLs built for ROCm 7")

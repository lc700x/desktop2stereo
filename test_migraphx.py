"""
Benchmark MIGraphX GPU inference vs PyTorch on gfx1103.

Builds a small CNN in PyTorch, exports to ONNX, then times:
  - MIGraphX GPU (compiled, fp32 + fp16)
  - PyTorch CUDA/HIP (if torch+rocm available)
  - PyTorch CPU (baseline)

Usage:
  .env/Scripts/python.exe benchmark_migraphx.py [--model resnet50] [--iters 100]
"""
import argparse
import os
import sys
import time

import numpy as np


def _percentiles(times_ms):
    a = np.array(times_ms)
    return {
        "mean": float(a.mean()),
        "p50": float(np.percentile(a, 50)),
        "p90": float(np.percentile(a, 90)),
        "min": float(a.min()),
    }


def make_onnx(path, model_name, batch, channels, hw):
    """Export a torchvision model (or a tiny CNN) to ONNX."""
    import torch
    import torch.nn as nn

    if model_name == "tiny":
        model = nn.Sequential(
            nn.Conv2d(channels, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Linear(64, 1000),
        )
    else:
        import torchvision.models as models
        model = getattr(models, model_name)(weights=None)
    model.eval()
    dummy = torch.randn(batch, channels, hw, hw)
    torch.onnx.export(
        model, dummy, path,
        input_names=["input"], output_names=["output"],
        opset_version=17, dynamo=False,
    )
    return dummy.numpy()


def bench_migraphx(onnx_path, input_np, iters, warmup, fp16):
    import migraphx
    prog = migraphx.parse_onnx(onnx_path)
    target = migraphx.get_target("gpu")
    if fp16:
        migraphx.quantize_fp16(prog)
    # offload_copy=True (default): MIGraphX manages host<->GPU transfers,
    # so we feed numpy-backed arguments directly and read host results back.
    prog.compile(target, offload_copy=True)

    in_name = prog.get_parameter_names()[0]
    arg = migraphx.argument(input_np)
    feed = {in_name: arg}

    # Warmup
    for _ in range(warmup):
        r = prog.run(feed)
        np.array(r[0])  # force host materialization

    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        out = prog.run(feed)
        _ = np.array(out[0])
        times.append((time.perf_counter() - t0) * 1000.0)
    return _percentiles(times)


def bench_torch(model_name, input_np, iters, warmup, device, channels, hw, fp16=False):
    import torch
    import torch.nn as nn
    if model_name == "tiny":
        model = nn.Sequential(
            nn.Conv2d(channels, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Linear(64, 1000),
        )
    else:
        import torchvision.models as models
        model = getattr(models, model_name)(weights=None)
    model.eval().to(device)
    x = torch.from_numpy(input_np).to(device)
    if fp16:
        model = model.half()
        x = x.half()

    with torch.no_grad():
        for _ in range(warmup):
            y = model(x)
            if device != "cpu":
                torch.cuda.synchronize()
        times = []
        for _ in range(iters):
            t0 = time.perf_counter()
            y = model(x)
            if device != "cpu":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000.0)
    return _percentiles(times)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="resnet50",
                    help="torchvision model name or 'tiny'")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--channels", type=int, default=3)
    ap.add_argument("--hw", type=int, default=224)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=20)
    args = ap.parse_args()

    onnx_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             f"_bench_{args.model}.onnx")

    print(f"Model: {args.model}  input: [{args.batch},{args.channels},{args.hw},{args.hw}]")
    print(f"Iters: {args.iters} (warmup {args.warmup})")
    print("=" * 60)

    print("Exporting ONNX...")
    input_np = make_onnx(onnx_path, args.model, args.batch, args.channels, args.hw)

    results = {}

    # MIGraphX fp32
    try:
        results["MIGraphX GPU fp32"] = bench_migraphx(onnx_path, input_np, args.iters, args.warmup, fp16=False)
    except Exception as e:
        print(f"MIGraphX fp32 failed: {e}")

    # MIGraphX fp16
    try:
        results["MIGraphX GPU fp16"] = bench_migraphx(onnx_path, input_np, args.iters, args.warmup, fp16=True)
    except Exception as e:
        print(f"MIGraphX fp16 failed: {e}")

    # PyTorch GPU (fp32 + fp16) — same precisions as MIGraphX for fair comparison
    try:
        import torch
        if torch.cuda.is_available():
            results["PyTorch GPU fp32"] = bench_torch(args.model, input_np, args.iters, args.warmup, "cuda", args.channels, args.hw, fp16=False)
            try:
                results["PyTorch GPU fp16"] = bench_torch(args.model, input_np, args.iters, args.warmup, "cuda", args.channels, args.hw, fp16=True)
            except Exception as e:
                print(f"PyTorch GPU fp16 failed: {e}")
        else:
            print("PyTorch GPU: torch.cuda not available, skipping")
    except Exception as e:
        print(f"PyTorch GPU failed: {e}")

    # PyTorch CPU
    try:
        results["PyTorch CPU"] = bench_torch(args.model, input_np, args.iters, args.warmup, "cpu", args.channels, args.hw)
    except Exception as e:
        print(f"PyTorch CPU failed: {e}")

    print("\n" + "=" * 60)
    print(f"{'Backend':<22}{'mean':>9}{'p50':>9}{'p90':>9}{'min':>9}   (ms)")
    print("-" * 60)
    base = None
    for name, r in results.items():
        print(f"{name:<22}{r['mean']:>9.3f}{r['p50']:>9.3f}{r['p90']:>9.3f}{r['min']:>9.3f}")
    print()
    # Same-precision comparisons (fair)
    if "MIGraphX GPU fp32" in results and "PyTorch GPU fp32" in results:
        sp = results["PyTorch GPU fp32"]["mean"] / results["MIGraphX GPU fp32"]["mean"]
        print(f"MIGraphX fp32 vs PyTorch fp32 (same precision): {sp:.2f}x")
    if "MIGraphX GPU fp16" in results and "PyTorch GPU fp16" in results:
        sp = results["PyTorch GPU fp16"]["mean"] / results["MIGraphX GPU fp16"]["mean"]
        print(f"MIGraphX fp16 vs PyTorch fp16 (same precision): {sp:.2f}x")
    # Best-MIGraphX vs best-PyTorch (real-world: each at its fastest)
    mgx_best = min([results[k]["mean"] for k in ("MIGraphX GPU fp16", "MIGraphX GPU fp32") if k in results], default=None)
    torch_best = min([results[k]["mean"] for k in ("PyTorch GPU fp16", "PyTorch GPU fp32") if k in results], default=None)
    if mgx_best and torch_best:
        print(f"Best MIGraphX vs best PyTorch GPU: {torch_best / mgx_best:.2f}x")

    try:
        os.remove(onnx_path)
    except OSError:
        pass


if __name__ == "__main__":
    main()

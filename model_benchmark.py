"""
Inference FPS benchmark: InfiniDepth vs DA3-large.

Usage:
    python3/python.exe benchmark_fps.py
"""
import os
import sys
import time
import torch
import torch.nn.functional as F

# Force UTF-8 output on Windows to avoid cp1252 encoding errors
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ── Config ────────────────────────────────────────────────────────────────────
CACHE_PATH   = "models"
RESOLUTION   = 192       # square input side; change to match your production setting
WARMUP_STEPS = 5
BENCH_STEPS  = 20
DEVICE_ID    = 0
# ──────────────────────────────────────────────────────────────────────────────

DEVICE = torch.device(f"cuda:{DEVICE_ID}" if torch.cuda.is_available() else "cpu")
print(f"Benchmark device : {DEVICE}")
print(f"Input resolution : {RESOLUTION}x{RESOLUTION}")
print(f"Warmup / bench   : {WARMUP_STEPS} / {BENCH_STEPS} steps\n")


def _sync():
    if DEVICE.type == "cuda":
        torch.cuda.synchronize(DEVICE)


def _make_input(res=RESOLUTION):
    return torch.randn(1, 3, res, res, device=DEVICE, dtype=torch.float32)


def bench(name, fn, warmup=WARMUP_STEPS, steps=BENCH_STEPS):
    """Run fn() repeatedly, return mean ms and FPS."""
    for _ in range(warmup):
        with torch.no_grad():
            fn()
    _sync()

    _sync()
    t0 = time.perf_counter()
    for _ in range(steps):
        with torch.no_grad():
            fn()
    _sync()
    elapsed = time.perf_counter() - t0

    ms  = elapsed / steps * 1000
    fps = steps / elapsed
    print(f"[{name}]  {ms:7.2f} ms/frame  ->  {fps:6.2f} FPS")
    return ms, fps


# ── Load InfiniDepth ──────────────────────────────────────────────────────────
def load_infinidepth():
    from huggingface_hub import hf_hub_download
    from models.InfiniDepth.api import InfiniDepthModel

    try:
        ckpt = hf_hub_download(
            repo_id="ritianyu/InfiniDepth",
            filename="infinidepth.ckpt",
            cache_dir=CACHE_PATH,
            local_files_only=True,
        )
    except Exception:
        ckpt = hf_hub_download(
            repo_id="ritianyu/InfiniDepth",
            filename="infinidepth.ckpt",
            cache_dir=CACHE_PATH,
        )

    model = InfiniDepthModel(model_path=ckpt).to(DEVICE).eval()
    return model


# ── Load DA3-large ────────────────────────────────────────────────────────────
def load_da3():
    from models.depth_anything_3.api_n import DepthAnything3

    model_id = "depth-anything/DA3-LARGE"
    try:
        model = DepthAnything3.from_pretrained(
            model_id, cache_dir=CACHE_PATH, local_files_only=True
        )
    except Exception:
        model = DepthAnything3.from_pretrained(model_id, cache_dir=CACHE_PATH)

    return model.to(DEVICE).eval()


# ── Main ──────────────────────────────────────────────────────────────────────
results = {}

# ── InfiniDepth ───────────────────────────────────────────────────────────────
print("Loading InfiniDepth ...")
try:
    inf_model = load_infinidepth()
    inp = _make_input()

    def _run_infinidepth():
        inf_model.predict_depth(inp, fp32=True)

    ms, fps = bench("InfiniDepth  (PyTorch fp32)", _run_infinidepth)
    results["InfiniDepth"] = fps
    del inf_model
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
except Exception as e:
    import traceback
    print(f"[InfiniDepth] FAILED: {e}")
    traceback.print_exc()

print()

# ── DA3-large ─────────────────────────────────────────────────────────────────
print("Loading DA3-large ...")
try:
    da3_model = load_da3()

    # DA3 patch size is 14, so resolution must be a multiple of 14.
    # Use the closest multiple of 14 to RESOLUTION.
    da3_res = round(RESOLUTION / 14) * 14
    print(f"  (DA3 resolution adjusted to {da3_res}x{da3_res} — must be multiple of 14)")
    inp_da3 = torch.randn(1, 3, da3_res, da3_res, device=DEVICE, dtype=torch.float32)

    def _run_da3():
        da3_model.predict_depth(inp_da3, fp32=False)

    ms, fps = bench("DA3-large    (PyTorch bf16)", _run_da3)
    results["DA3-large"] = fps
    del da3_model
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
except Exception as e:
    import traceback
    print(f"[DA3-large] FAILED: {e}")
    traceback.print_exc()

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n--- Summary ---")
for name, fps in results.items():
    print(f"  {name:<20s}  {fps:6.2f} FPS")

if "InfiniDepth" in results and "DA3-large" in results:
    ratio = results["DA3-large"] / results["InfiniDepth"]
    print(f"\n  DA3-large is {ratio:.1f}x faster than InfiniDepth at {RESOLUTION}x{RESOLUTION}")

#!/usr/bin/env python3
"""
Comprehensive test script for ScreenCaptureKit capture on macOS.

Tests:
  1. Monitor capture – basic frame acquisition
  2. Output format – BGR and BGRA
  3. FPS benchmarking – multi-second measurement
  4. Frame consistency – shape, dtype, non-blank
  5. Cursor capture – native SCK cursor inclusion
  6. Start/stop lifecycle – multiple sequential sessions
  7. Window capture – if a window title is provided
  8. Error handling – invalid window, missing display
  9. Frame timestamp consistency
 10. Resolution fidelity

Usage:
  python test_screen_capture_kit.py                    # monitor capture only
  python test_screen_capture_kit.py --window "Safari"  # also test window capture
  python test_screen_capture_kit.py --fps 30           # cap at 30 fps
  python test_screen_capture_kit.py --duration 5        # test duration in seconds
  python test_screen_capture_kit.py --compare-quartz    # also benchmark Quartz
"""

import sys
import os
import time
import argparse
import numpy as np

# Ensure we can import from the project
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Override CAPTURE_TOOL for testing SCK
import utils
utils.CAPTURE_TOOL = "ScreenCaptureKit"


def test_monitor_capture(duration=3, fps=60):
    """Test basic monitor capture with SCK."""
    print("\n" + "=" * 60)
    print("TEST 1: Monitor Capture")
    print("=" * 60)

    from capture import DesktopGrabber

    grabber = DesktopGrabber(output_resolution=1080, fps=fps,
                             capture_mode="Monitor", monitor_index=1,
                             with_cursor=True)
    print(f"  Capture region: {grabber.left},{grabber.top} {grabber.width}x{grabber.height}")

    frame_count = 0
    shapes = set()
    start = time.time()

    try:
        while time.time() - start < duration:
            frame, scaled_h = grabber.grab(output_format="bgra")
            shapes.add(frame.shape)
            frame_count += 1
            if frame_count == 1:
                print(f"  First frame: shape={frame.shape}, dtype={frame.dtype}, "
                      f"min={frame.min()}, max={frame.max()}, mean={frame.mean():.1f}")
    finally:
        grabber.stop()

    elapsed = time.time() - start
    measured_fps = frame_count / elapsed
    print(f"  Frames: {frame_count} in {elapsed:.1f}s = {measured_fps:.1f} fps")
    print(f"  All same shape: {len(shapes) == 1} -> {shapes}")
    passed = frame_count > 0 and len(shapes) == 1
    print(f"  RESULT: {'PASS' if passed else 'FAIL'}")
    return passed, measured_fps


def test_output_formats(fps=60):
    """Test both BGR and BGRA output."""
    print("\n" + "=" * 60)
    print("TEST 2: Output Formats (BGR / BGRA)")
    print("=" * 60)

    from capture import DesktopGrabber
    grabber = DesktopGrabber(output_resolution=1080, fps=fps,
                             capture_mode="Monitor", with_cursor=True)

    all_passed = True
    try:
        for fmt, expected_channels in [("bgra", 4), ("bgr", 3)]:
            frame, _ = grabber.grab(output_format=fmt)
            ok = frame.shape[2] == expected_channels
            print(f"  {fmt}: shape={frame.shape}, expected channels={expected_channels}, "
                  f"OK={ok}")
            all_passed = all_passed and ok

        # Test invalid format
        try:
            grabber.grab(output_format="rgb")
            print("  Invalid format: FAIL (should have raised ValueError)")
            all_passed = False
        except ValueError:
            print("  Invalid format 'rgb': correctly raised ValueError")
    finally:
        grabber.stop()

    print(f"  RESULT: {'PASS' if all_passed else 'FAIL'}")
    return all_passed


def test_fps_benchmark(duration=5, target_fps=60):
    """Benchmark FPS over an extended period."""
    print("\n" + "=" * 60)
    print(f"TEST 3: FPS Benchmark ({duration}s, target {target_fps} fps)")
    print("=" * 60)

    from capture import DesktopGrabber
    grabber = DesktopGrabber(output_resolution=1080, fps=target_fps,
                             capture_mode="Monitor", with_cursor=False)

    # Sample FPS every second
    samples = []
    frame_count = 0
    last_count = 0
    start = time.time()
    last_sample = start

    try:
        while time.time() - start < duration:
            frame, _ = grabber.grab(output_format="bgr")
            frame_count += 1
            now = time.time()
            if now - last_sample >= 1.0:
                interval_fps = (frame_count - last_count) / (now - last_sample)
                samples.append(interval_fps)
                print(f"  Second {len(samples)}: {interval_fps:.1f} fps")
                last_count = frame_count
                last_sample = now
    finally:
        grabber.stop()

    if samples:
        avg_fps = sum(samples) / len(samples)
        min_fps = min(samples)
        max_fps = max(samples)
        print(f"  Average: {avg_fps:.1f} fps, Min: {min_fps:.1f}, Max: {max_fps:.1f}")

        # SCK should deliver reasonable frame rates
        passed = avg_fps >= 5.0  # Very lenient check
        print(f"  RESULT: {'PASS' if passed else 'FAIL'}")
        return passed, avg_fps
    else:
        print("  No samples collected!")
        return False, 0


def test_frame_consistency(fps=60):
    """Verify frames are consistent in shape and have real content."""
    print("\n" + "=" * 60)
    print("TEST 4: Frame Consistency")
    print("=" * 60)

    from capture import DesktopGrabber
    grabber = DesktopGrabber(output_resolution=1080, fps=fps,
                             capture_mode="Monitor", with_cursor=False)

    frames = []
    try:
        for _ in range(10):
            frame, _ = grabber.grab(output_format="bgra")
            frames.append(frame)
            time.sleep(0.05)
    finally:
        grabber.stop()

    shapes = set(f.shape for f in frames)
    means = [f.mean() for f in frames]
    stds = [f.std() for f in frames]

    shape_ok = len(shapes) == 1
    not_blank = all(m > 1.0 for m in means)
    has_variation = np.std(means) > 0.01  # Content should vary slightly
    std_ok = all(s > 1.0 for s in stds)

    print(f"  Shape consistent: {shape_ok} -> {shapes}")
    print(f"  Not blank: {not_blank} (means: {[f'{m:.1f}' for m in means[:5]]}...)")
    print(f"  Has variation: {has_variation} (std of means: {np.std(means):.2f})")
    print(f"  Non-zero std: {std_ok} (stds: {[f'{s:.1f}' for s in stds[:5]]}...)")

    passed = shape_ok and not_blank and has_variation and std_ok
    print(f"  RESULT: {'PASS' if passed else 'FAIL'}")
    return passed


def test_cursor_capture(fps=60):
    """Test that cursor capture works (native SCK cursor inclusion)."""
    print("\n" + "=" * 60)
    print("TEST 5: Cursor Capture (native SCK)")
    print("=" * 60)

    from capture import DesktopGrabber

    # Capture with cursor
    gc = DesktopGrabber(output_resolution=1080, fps=fps,
                        capture_mode="Monitor", with_cursor=True)
    try:
        frame_with, _ = gc.grab(output_format="bgra")
    finally:
        gc.stop()

    # Capture without cursor
    gn = DesktopGrabber(output_resolution=1080, fps=fps,
                        capture_mode="Monitor", with_cursor=False)
    try:
        frame_without, _ = gn.grab(output_format="bgra")
    finally:
        gn.stop()

    # Both should be valid frames
    ok1 = frame_with.shape[2] == 4
    ok2 = frame_without.shape[2] == 4
    same_shape = frame_with.shape == frame_without.shape

    # They may or may not differ (cursor might not be over capture area)
    diff = np.abs(frame_with.astype(float) - frame_without.astype(float)).mean()

    print(f"  With cursor: shape={frame_with.shape}, mean={frame_with.mean():.1f}")
    print(f"  Without cursor: shape={frame_without.shape}, mean={frame_without.mean():.1f}")
    print(f"  Same shape: {same_shape}")
    print(f"  Pixel difference: {diff:.4f}")
    print(f"  Both valid BGRA: {ok1 and ok2}")

    passed = ok1 and ok2 and same_shape
    print(f"  RESULT: {'PASS' if passed else 'FAIL'}")
    return passed


def test_lifecycle(fps=60):
    """Test start/stop lifecycle – multiple sequential captures."""
    print("\n" + "=" * 60)
    print("TEST 6: Start/Stop Lifecycle")
    print("=" * 60)

    from capture import DesktopGrabber

    all_passed = True
    for i in range(3):
        grabber = DesktopGrabber(output_resolution=1080, fps=fps,
                                 capture_mode="Monitor", with_cursor=False)
        try:
            frame, _ = grabber.grab(output_format="bgr")
            ok = frame is not None and frame.size > 0
            grabber.stop()
            grabber = None
            print(f"  Session {i+1}: frame OK={ok}, shape={frame.shape}")
            all_passed = all_passed and ok
        except Exception as e:
            print(f"  Session {i+1}: ERROR: {e}")
            all_passed = False
            try:
                grabber.stop()
            except Exception:
                pass
            grabber = None

        time.sleep(0.3)  # Brief pause between sessions

    print(f"  RESULT: {'PASS' if all_passed else 'FAIL'}")
    return all_passed


def test_window_capture(window_title, fps=60):
    """Test window capture if a title is provided."""
    print("\n" + "=" * 60)
    print(f"TEST 7: Window Capture ('{window_title}')")
    print("=" * 60)

    from capture import DesktopGrabber, _sck_find_window

    # First check if window exists
    win = _sck_find_window(window_title)
    if win is None:
        print(f"  Window '{window_title}' not found – SKIPPED")
        return None  # Not a failure, just not available

    print(f"  Window found: id={win.windowID()}, "
          f"frame=({win.frame().origin.x:.0f},{win.frame().origin.y:.0f},"
          f"{win.frame().size.width:.0f},{win.frame().size.height:.0f})")

    grabber = DesktopGrabber(output_resolution=1080, fps=fps,
                             window_title=window_title,
                             capture_mode="Window", with_cursor=True)
    try:
        print(f"  Capture region: {grabber.left},{grabber.top} {grabber.width}x{grabber.height}")

        frame_count = 0
        shapes = set()
        start = time.time()

        while time.time() - start < 2:
            frame, _ = grabber.grab(output_format="bgra")
            shapes.add(frame.shape)
            frame_count += 1

        elapsed = time.time() - start
        print(f"  Frames: {frame_count} in {elapsed:.1f}s = {frame_count/elapsed:.1f} fps")
        print(f"  Shape consistent: {len(shapes) == 1} -> {shapes}")
        passed = frame_count > 0 and len(shapes) == 1
    except Exception as e:
        print(f"  Error: {e}")
        passed = False
    finally:
        grabber.stop()

    print(f"  RESULT: {'PASS' if passed else 'FAIL'}")
    return passed


def test_error_handling():
    """Test graceful error handling."""
    print("\n" + "=" * 60)
    print("TEST 8: Error Handling")
    print("=" * 60)

    from capture import DesktopGrabber

    all_passed = True

    # Test invalid window
    try:
        DesktopGrabber(window_title="__nonexistent_window_xyzzy__",
                       capture_mode="Window")
        print("  Invalid window: FAIL (should have raised RuntimeError)")
        all_passed = False
    except RuntimeError:
        print("  Invalid window: correctly raised RuntimeError")
    except Exception as e:
        print(f"  Invalid window: raised {type(e).__name__} (expected RuntimeError)")

    # Test out-of-range monitor index (should clamp, not crash)
    try:
        g = DesktopGrabber(monitor_index=999, capture_mode="Monitor", fps=30)
        frame, _ = g.grab(output_format="bgr")
        g.stop()
        print(f"  Monitor index 999: clamped OK, frame={frame.shape}")
    except Exception as e:
        print(f"  Monitor index 999: FAILED: {e}")
        all_passed = False

    # Test invalid output format (already tested in test 2)
    print(f"  RESULT: {'PASS' if all_passed else 'FAIL'}")
    return all_passed


def test_resolution_fidelity(fps=60):
    """Verify captured resolution matches what was requested."""
    print("\n" + "=" * 60)
    print("TEST 9: Resolution Fidelity")
    print("=" * 60)

    from capture import DesktopGrabber

    grabber = DesktopGrabber(output_resolution=1080, fps=fps,
                             capture_mode="Monitor", with_cursor=False)
    try:
        frame, scaled_h = grabber.grab(output_format="bgra")

        # scaled_h should match output_resolution
        resolution_ok = scaled_h == 1080
        print(f"  scaled_height: {scaled_h} (expected 1080) -> OK={resolution_ok}")

        # Frame dimensions should match capture region
        h, w, c = frame.shape
        expected_w = grabber.width
        expected_h = grabber.height
        dims_ok = w == expected_w and h == expected_h
        print(f"  Frame: {w}x{h} (expected {expected_w}x{expected_h}) -> OK={dims_ok}")

        # Channels should be 4 (BGRA)
        channels_ok = c == 4
        print(f"  Channels: {c} (expected 4) -> OK={channels_ok}")
    finally:
        grabber.stop()

    passed = resolution_ok and dims_ok and channels_ok
    print(f"  RESULT: {'PASS' if passed else 'FAIL'}")
    return passed


def compare_with_quartz(duration=3):
    """Benchmark SCK vs Quartz capture (both run in separate subprocesses)."""
    print("\n" + "=" * 60)
    print("TEST 10: Quartz Comparison Benchmark")
    print("=" * 60)

    import subprocess

    # Test SCK
    print("  Running SCK benchmark...")
    sck_result = subprocess.run(
        [sys.executable, "-c", f"""
import time, sys
sys.path.insert(0, "{os.path.dirname(os.path.abspath(__file__))}")
import utils
utils.CAPTURE_TOOL = "ScreenCaptureKit"
from capture import DesktopGrabber
g = DesktopGrabber(output_resolution=1080, fps=60, capture_mode="Monitor", with_cursor=False)
c = 0
t = time.time()
while time.time() - t < {duration}:
    g.grab(output_format="bgr")
    c += 1
g.stop()
print(f"{{c}}")
"""],
        capture_output=True, text=True, timeout=duration + 30)
    sck_frames = int(sck_result.stdout.strip()) if sck_result.stdout.strip().isdigit() else 0
    sck_fps = sck_frames / duration if sck_frames > 0 else 0

    # Test Quartz
    print("  Running Quartz benchmark...")
    qz_result = subprocess.run(
        [sys.executable, "-c", f"""
import time, sys
sys.path.insert(0, "{os.path.dirname(os.path.abspath(__file__))}")
import utils
utils.CAPTURE_TOOL = "none"
from capture import DesktopGrabber
g = DesktopGrabber(output_resolution=1080, fps=60, capture_mode="Monitor", with_cursor=False)
c = 0
t = time.time()
while time.time() - t < {duration}:
    g.grab(output_format="bgr")
    c += 1
g.stop()
print(f"{{c}}")
"""],
        capture_output=True, text=True, timeout=duration + 30)
    qz_frames = int(qz_result.stdout.strip()) if qz_result.stdout.strip().isdigit() else 0
    qz_fps = qz_frames / duration if qz_frames > 0 else 0

    print(f"  ScreenCaptureKit: {sck_fps:.1f} fps ({sck_frames} frames in {duration}s)")
    print(f"  Quartz/CG:        {qz_fps:.1f} fps ({qz_frames} frames in {duration}s)")
    if qz_fps > 0:
        speedup = sck_fps / qz_fps
        print(f"  Relative: {speedup:.2f}x ({'SCK' if speedup > 1 else 'Quartz'} faster)")
    print(f"  RESULT: {'PASS' if sck_frames > 0 else 'FAIL'} (comparison informational)")
    return sck_frames > 0


def check_sck_available():
    """Pre-flight check: can we get displays from ScreenCaptureKit?"""
    import objc
    import threading

    try:
        objc.loadBundle('ScreenCaptureKit', globals(),
            bundle_path=objc.pathForFramework('/System/Library/Frameworks/ScreenCaptureKit.framework'))
        import ScreenCaptureKit as SCK
    except Exception as e:
        print(f"ScreenCaptureKit not available: {e}")
        return False

    done = threading.Event()
    result = {}

    def handler(content, error):
        result['content'] = content
        result['error'] = error
        done.set()

    SCK.SCShareableContent.getShareableContentWithCompletionHandler_(handler)
    done.wait(timeout=10.0)

    content = result['content']
    if content is None:
        print(f"SCShareableContent failed: {result.get('error')}")
        return False

    displays = content.displays()
    if not displays or len(displays) == 0:
        print("No displays available. Screen Recording permission required.")
        print()
        print("  Open System Settings > Privacy & Security > Screen Recording")
        print("  and enable your terminal application (Terminal.app / iTerm2 / VS Code).")
        print("  Then try again.")
        return False

    print(f"ScreenCaptureKit available: {len(displays)} display(s)")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Test ScreenCaptureKit capture for macOS")
    parser.add_argument("--window", type=str, default=None,
                        help="Window title to test window capture")
    parser.add_argument("--fps", type=int, default=60,
                        help="Target capture FPS (default: 60)")
    parser.add_argument("--duration", type=int, default=3,
                        help="Duration in seconds for FPS tests (default: 3)")
    parser.add_argument("--compare-quartz", action="store_true",
                        help="Run comparison benchmark with Quartz capture")
    parser.add_argument("--skip-tests", type=str, default="",
                        help="Comma-separated test numbers to skip")
    args = parser.parse_args()

    skip = set(int(x) for x in args.skip_tests.split(",") if x.strip())

    print("=" * 60)
    print("ScreenCaptureKit Capture Test Suite")
    print(f"Target FPS: {args.fps}, Duration: {args.duration}s")
    print("=" * 60)

    if not check_sck_available():
        print("\nCannot proceed without ScreenCaptureKit access.")
        return 1

    results = {}
    fps_data = {}

    if 1 not in skip:
        results["Monitor Capture"], fps_data["monitor"] = \
            test_monitor_capture(args.duration, args.fps)

    if 2 not in skip:
        results["Output Formats"] = test_output_formats(args.fps)

    if 3 not in skip:
        results["FPS Benchmark"], fps_data["benchmark"] = \
            test_fps_benchmark(args.duration, args.fps)

    if 4 not in skip:
        results["Frame Consistency"] = test_frame_consistency(args.fps)

    if 5 not in skip:
        results["Cursor Capture"] = test_cursor_capture(args.fps)

    if 6 not in skip:
        results["Lifecycle"] = test_lifecycle(args.fps)

    if 7 not in skip:
        win_result = test_window_capture(args.window, args.fps) if args.window else None
        if win_result is not None:
            results["Window Capture"] = win_result

    if 8 not in skip:
        results["Error Handling"] = test_error_handling()

    if 9 not in skip:
        results["Resolution Fidelity"] = test_resolution_fidelity(args.fps)

    if args.compare_quartz:
        compare_with_quartz(args.duration)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    all_passed = True
    for test_name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {test_name}: {status}")
        if not passed:
            all_passed = False

    if fps_data:
        print("\n  FPS Summary:")
        for name, fps in fps_data.items():
            print(f"    {name}: {fps:.1f} fps")

    print(f"\n  OVERALL: {'ALL PASSED' if all_passed else 'SOME FAILURES'}")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())

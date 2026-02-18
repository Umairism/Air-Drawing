"""
performance benchmarking tool.

runs the full pipeline with synthetic or live data and measures
timing for each stage. quantifies the actual cost of each processing
layer with percentile breakdowns.

usage:
    python -m ml.benchmark              # synthetic benchmark (no camera needed)
    python -m ml.benchmark --live 300   # live benchmark, 300 frames
"""
import time
import math
import sys
import os
import json
import numpy as np

# make sure imports work from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.gesture_engine import GestureEngine
from core.noise_filter import NoiseFilter
from core.profiler import Profiler


def make_synthetic_hand(base_x=300, base_y=400, finger_up=None, noise=0):
    """generate a fake hand landmark set with optional noise"""
    if finger_up is None:
        finger_up = [False, True, False, False, False]

    lm = {0: (base_x, base_y)}
    lm[1] = (base_x - 30, base_y - 20)
    lm[2] = (base_x - 50, base_y - 45)
    lm[3] = (base_x - 65, base_y - 65)
    lm[4] = (base_x - 80, base_y - 85) if finger_up[0] else (base_x - 40, base_y - 50)

    finger_bases = [
        (base_x - 25, 5, 6, 7, 8),
        (base_x - 5, 9, 10, 11, 12),
        (base_x + 15, 13, 14, 15, 16),
        (base_x + 35, 17, 18, 19, 20),
    ]

    for i, (fx, mcp_id, pip_id, dip_id, tip_id) in enumerate(finger_bases):
        mcp_y = base_y - 80
        lm[mcp_id] = (fx, mcp_y)
        if finger_up[i + 1]:
            lm[pip_id] = (fx, mcp_y - 40)
            lm[dip_id] = (fx, mcp_y - 70)
            lm[tip_id] = (fx, mcp_y - 100)
        else:
            lm[pip_id] = (fx, mcp_y - 25)
            lm[dip_id] = (fx, mcp_y - 10)
            lm[tip_id] = (fx, mcp_y + 10)

    result = []
    for idx in sorted(lm.keys()):
        x, y = lm[idx]
        if noise > 0:
            x += np.random.uniform(-noise, noise)
            y += np.random.uniform(-noise, noise)
        result.append((idx, x, y, 0.0))
    return result


def _stats_line(arr_ms, label):
    """format a single stat line with avg + percentiles"""
    avg = np.mean(arr_ms)
    p50 = np.percentile(arr_ms, 50)
    p95 = np.percentile(arr_ms, 95)
    p99 = np.percentile(arr_ms, 99)
    return f"  {label:<25} avg={avg:>7.3f}ms  p50={p50:>7.3f}ms  p95={p95:>7.3f}ms  p99={p99:>7.3f}ms"


def _stats_dict(arr_ms):
    """return a dict with all the stats we care about"""
    return {
        "avg_ms": round(float(np.mean(arr_ms)), 4),
        "p50_ms": round(float(np.percentile(arr_ms, 50)), 4),
        "p95_ms": round(float(np.percentile(arr_ms, 95)), 4),
        "p99_ms": round(float(np.percentile(arr_ms, 99)), 4),
        "min_ms": round(float(np.min(arr_ms)), 4),
        "max_ms": round(float(np.max(arr_ms)), 4),
    }


def benchmark_gesture_engine(n_frames=1000):
    """benchmark just the gesture engine (no camera, no mediapipe)"""
    engine = GestureEngine()

    # cycle through different gestures to simulate real usage
    gestures = [
        [False, True, False, False, False],   # draw
        [True, True, True, True, True],        # erase
        [False, True, True, False, False],     # change_color
        [False, False, False, False, False],   # fist
    ]

    times = []
    for i in range(n_frames):
        finger_state = gestures[i % len(gestures)]
        bx = 300 + math.sin(i * 0.05) * 50
        by = 400 + math.cos(i * 0.03) * 30
        lm = make_synthetic_hand(bx, by, finger_state, noise=2.0)
        hand_data = {"right": lm, "left": None}

        start = time.perf_counter()
        engine.recognize(hand_data)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    return np.array(times) * 1000  # return in ms


def benchmark_noise_filter(n_frames=1000):
    """benchmark the noise filter stage"""
    nf = NoiseFilter()

    times = []
    for i in range(n_frames):
        bx = 300 + math.sin(i * 0.05) * 50
        by = 400 + math.cos(i * 0.03) * 30
        lm = make_synthetic_hand(bx, by, noise=5.0)

        start = time.perf_counter()
        nf.filter(lm)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    return np.array(times) * 1000


def benchmark_feature_extraction(n_frames=1000):
    """benchmark the ML feature extraction"""
    from ml.trainer import extract_features

    lm = make_synthetic_hand(300, 400, [False, True, False, False, False])

    times = []
    for _ in range(n_frames):
        start = time.perf_counter()
        extract_features(lm)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    return np.array(times) * 1000


def benchmark_raw_vs_smoothed(n_frames=2000):
    """
    the critical benchmark: quantify the latency-vs-stability tradeoff.

    runs the gesture engine with and without the noise filter on the
    same sequence of landmarks. measures:
      - processing time overhead of smoothing
      - positional lag introduced by the filter (how far behind the
        filtered position is from the raw input)
      - jitter reduction ratio
    """
    nf = NoiseFilter()
    np.random.seed(42)

    # build a sequence of slowly moving landmarks with realistic noise
    raw_sequence = []
    for i in range(n_frames):
        bx = 300 + math.sin(i * 0.02) * 80  # slow sinusoidal drift
        by = 400 + math.cos(i * 0.015) * 60
        lm = make_synthetic_hand(bx, by, [False, True, False, False, False], noise=4.0)
        raw_sequence.append(lm)

    # pass 1: measure raw pipeline (gesture engine only, no filter)
    engine_raw = GestureEngine()
    raw_times = []
    raw_tip_positions = []
    for lm in raw_sequence:
        hand_data = {"right": lm, "left": None}
        start = time.perf_counter()
        _, tip, _ = engine_raw.recognize(hand_data)
        elapsed = time.perf_counter() - start
        raw_times.append(elapsed)
        raw_tip_positions.append(tip)

    # pass 2: measure smoothed pipeline (noise filter + gesture engine)
    engine_smooth = GestureEngine()
    nf = NoiseFilter()
    smooth_times = []
    smooth_tip_positions = []
    for lm in raw_sequence:
        start = time.perf_counter()
        filtered = nf.filter(lm)
        hand_data = {"right": filtered, "left": None}
        _, tip, _ = engine_smooth.recognize(hand_data)
        elapsed = time.perf_counter() - start
        smooth_times.append(elapsed)
        smooth_tip_positions.append(tip)

    raw_ms = np.array(raw_times) * 1000
    smooth_ms = np.array(smooth_times) * 1000
    overhead_ms = smooth_ms - raw_ms

    # measure positional lag: how far behind is the filtered tip from the raw tip?
    lags = []
    for raw_tip, smooth_tip in zip(raw_tip_positions, smooth_tip_positions):
        if raw_tip and smooth_tip:
            dx = raw_tip[0] - smooth_tip[0]
            dy = raw_tip[1] - smooth_tip[1]
            lags.append(math.sqrt(dx * dx + dy * dy))

    # measure jitter: frame-to-frame tip movement variance
    raw_jitter = []
    smooth_jitter = []
    for i in range(1, len(raw_tip_positions)):
        if raw_tip_positions[i] and raw_tip_positions[i - 1]:
            dx = raw_tip_positions[i][0] - raw_tip_positions[i - 1][0]
            dy = raw_tip_positions[i][1] - raw_tip_positions[i - 1][1]
            raw_jitter.append(math.sqrt(dx * dx + dy * dy))
        if smooth_tip_positions[i] and smooth_tip_positions[i - 1]:
            dx = smooth_tip_positions[i][0] - smooth_tip_positions[i - 1][0]
            dy = smooth_tip_positions[i][1] - smooth_tip_positions[i - 1][1]
            smooth_jitter.append(math.sqrt(dx * dx + dy * dy))

    lag_arr = np.array(lags) if lags else np.array([0.0])
    raw_jitter_arr = np.array(raw_jitter) if raw_jitter else np.array([0.0])
    smooth_jitter_arr = np.array(smooth_jitter) if smooth_jitter else np.array([0.0])

    return {
        "raw_ms": raw_ms,
        "smooth_ms": smooth_ms,
        "overhead_ms": overhead_ms,
        "lag_px": lag_arr,
        "raw_jitter_px": raw_jitter_arr,
        "smooth_jitter_px": smooth_jitter_arr,
    }


def benchmark_full_pipeline(n_frames=1000):
    """benchmark the complete pipeline minus camera + mediapipe"""
    engine = GestureEngine()
    nf = NoiseFilter()
    profiler = Profiler(enabled=True)

    gestures = [
        [False, True, False, False, False],
        [True, True, True, True, True],
        [False, True, True, False, False],
        [False, False, False, False, False],
    ]

    for i in range(n_frames):
        profiler.begin_frame()

        finger_state = gestures[i % len(gestures)]
        bx = 300 + math.sin(i * 0.05) * 50
        by = 400 + math.cos(i * 0.03) * 30
        lm = make_synthetic_hand(bx, by, finger_state, noise=3.0)

        profiler.start("noise_filter")
        lm = nf.filter(lm)
        profiler.stop("noise_filter")

        hand_data = {"right": lm, "left": None}

        profiler.start("gesture")
        engine.recognize(hand_data)
        profiler.stop("gesture")

        profiler.end_frame()

    return profiler


def run_synthetic_benchmark():
    """run all benchmarks without needing a camera"""
    n = 2000
    print(f"\nrunning synthetic benchmark ({n} frames per test)...")
    print("=" * 75)

    # individual stage benchmarks
    print("\n  STAGE TIMING")
    print(f"  {'─' * 70}")
    ge_ms = benchmark_gesture_engine(n)
    print(_stats_line(ge_ms, "gesture engine"))
    nf_ms = benchmark_noise_filter(n)
    print(_stats_line(nf_ms, "noise filter"))
    fe_ms = benchmark_feature_extraction(n)
    print(_stats_line(fe_ms, "feature extraction"))

    # raw vs smoothed — the money benchmark
    print(f"\n  RAW vs SMOOTHED COMPARISON ({n} frames)")
    print(f"  {'─' * 70}")

    rvs = benchmark_raw_vs_smoothed(n)

    print(_stats_line(rvs["raw_ms"], "raw pipeline"))
    print(_stats_line(rvs["smooth_ms"], "smoothed pipeline"))
    print(_stats_line(rvs["overhead_ms"], "smoothing overhead"))

    print()
    print(f"  {'POSITIONAL LAG':<25} avg={np.mean(rvs['lag_px']):>5.1f}px  p95={np.percentile(rvs['lag_px'], 95):>5.1f}px  max={np.max(rvs['lag_px']):>5.1f}px")
    print(f"  {'RAW JITTER':<25} avg={np.mean(rvs['raw_jitter_px']):>5.1f}px  std={np.std(rvs['raw_jitter_px']):>5.1f}px")
    print(f"  {'SMOOTHED JITTER':<25} avg={np.mean(rvs['smooth_jitter_px']):>5.1f}px  std={np.std(rvs['smooth_jitter_px']):>5.1f}px")

    jitter_ratio = np.mean(rvs["raw_jitter_px"]) / max(np.mean(rvs["smooth_jitter_px"]), 0.001)
    print(f"  {'JITTER REDUCTION':<25} {jitter_ratio:.1f}x less jitter with smoothing")
    overhead_pct = np.mean(rvs["overhead_ms"]) / max(np.mean(rvs["raw_ms"]), 0.001) * 100
    print(f"  {'LATENCY COST':<25} +{np.mean(rvs['overhead_ms']):.3f}ms ({overhead_pct:.0f}% overhead)")

    # full pipeline with profiler
    print(f"\n  FULL PIPELINE (profiler view)")
    profiler = benchmark_full_pipeline(n)
    profiler.report()

    # theoretical throughput
    total_times = profiler._frame_total
    avg_frame = np.mean(total_times)
    theoretical_fps = 1.0 / avg_frame if avg_frame > 0 else 0
    print(f"  theoretical max FPS (no camera/mediapipe): {theoretical_fps:.0f}")
    print()

    # save results
    results = {
        "type": "synthetic",
        "frames": n,
        "stages": {
            "gesture_engine": _stats_dict(ge_ms),
            "noise_filter": _stats_dict(nf_ms),
            "feature_extraction": _stats_dict(fe_ms),
        },
        "raw_vs_smoothed": {
            "raw_pipeline": _stats_dict(rvs["raw_ms"]),
            "smoothed_pipeline": _stats_dict(rvs["smooth_ms"]),
            "overhead": _stats_dict(rvs["overhead_ms"]),
            "positional_lag_px": {
                "avg": round(float(np.mean(rvs["lag_px"])), 2),
                "p95": round(float(np.percentile(rvs["lag_px"], 95)), 2),
                "max": round(float(np.max(rvs["lag_px"])), 2),
            },
            "jitter_reduction_ratio": round(jitter_ratio, 2),
            "raw_jitter_avg_px": round(float(np.mean(rvs["raw_jitter_px"])), 2),
            "smooth_jitter_avg_px": round(float(np.mean(rvs["smooth_jitter_px"])), 2),
        },
        "pipeline_fps": round(theoretical_fps, 1),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    with open("benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"results saved to benchmark_results.json")

    return results


def run_live_benchmark(n_frames=300):
    """benchmark with actual camera + mediapipe"""
    import cv2
    from core.camera import Camera
    from core.hand_tracker import HandTracker

    print(f"\nrunning live benchmark ({n_frames} frames)...")
    print("point your hand at the camera and hold still\n")

    cam = Camera()
    tracker = HandTracker()
    engine = GestureEngine()
    nf = NoiseFilter()
    profiler = Profiler(enabled=True)

    detection_times = []
    gesture_times = []
    filter_times = []

    for i in range(n_frames):
        profiler.begin_frame()

        # camera
        profiler.start("camera")
        ok, frame = cam.read()
        profiler.stop("camera")
        if not ok:
            continue

        # detection
        profiler.start("detection")
        t0 = time.perf_counter()
        hand_data = tracker.find_hands(frame)
        detection_times.append(time.perf_counter() - t0)
        profiler.stop("detection")

        # noise filter
        profiler.start("noise_filter")
        t0 = time.perf_counter()
        for side in ("right", "left"):
            raw = hand_data.get(side)
            hand_data[side] = nf.filter(raw) if side == "right" else raw
        filter_times.append(time.perf_counter() - t0)
        profiler.stop("noise_filter")

        # gesture
        profiler.start("gesture")
        t0 = time.perf_counter()
        gesture, _, _ = engine.recognize(hand_data)
        gesture_times.append(time.perf_counter() - t0)
        profiler.stop("gesture")

        profiler.end_frame()

        # show progress
        if (i + 1) % 50 == 0:
            print(f"  frame {i + 1}/{n_frames}")

    tracker.close()
    cam.release()

    print()
    profiler.report()

    print("  DETAILED BREAKDOWN")
    print(f"  {'─' * 70}")
    if detection_times:
        print(_stats_line(np.array(detection_times) * 1000, "mediapipe detection"))
    if filter_times:
        print(_stats_line(np.array(filter_times) * 1000, "noise filter"))
    if gesture_times:
        print(_stats_line(np.array(gesture_times) * 1000, "gesture engine"))
    print()


if __name__ == "__main__":
    if "--live" in sys.argv:
        idx = sys.argv.index("--live")
        n = int(sys.argv[idx + 1]) if idx + 1 < len(sys.argv) else 300
        run_live_benchmark(n)
    else:
        run_synthetic_benchmark()

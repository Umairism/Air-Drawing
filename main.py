import cv2
import sys
import os
import time
import logging
import traceback

from core.camera import Camera
from core.hand_tracker import HandTracker
from core.gesture_engine import GestureEngine
from core.state_manager import StateManager
from core.noise_filter import NoiseFilter
from core.profiler import Profiler
from app.canvas import Canvas
from app.tools import Pen, Brush, Eraser
from app.ui import UI
from app.config import (
    SHOW_LANDMARKS, NOISE_FILTER_ENABLED,
    LOW_CONFIDENCE_THRESHOLD, PROFILER_ENABLED,
    ML_HYBRID_ENABLED, ML_CONFIDENCE_THRESHOLD, ML_MODEL_PATH,
    BENCHMARK_ENABLED,
)

# set up a dedicated logger for ML override events so they dont
# get lost in stdout noise. writes to ml_overrides.log.
_ml_logger = logging.getLogger("ml_hybrid")
_ml_logger.setLevel(logging.DEBUG)
_ml_log_handler = logging.FileHandler("ml_overrides.log", mode="a")
_ml_log_handler.setFormatter(logging.Formatter("%(asctime)s  %(message)s", datefmt="%H:%M:%S"))
_ml_logger.addHandler(_ml_log_handler)


def main():
    # init everything with proper error messages
    try:
        cam = Camera()
    except RuntimeError as e:
        print(f"\nCamera error: {e}")
        sys.exit(1)

    try:
        tracker = HandTracker()
    except FileNotFoundError as e:
        print(f"\nModel file missing: {e}")
        cam.release()
        sys.exit(1)
    except Exception as e:
        print(f"\nFailed to initialize hand tracker: {e}")
        cam.release()
        sys.exit(1)

    gesture_engine = GestureEngine()
    state = StateManager()

    # ML hybrid mode - loads trained model if it exists
    ml_predictor = None
    if ML_HYBRID_ENABLED:
        try:
            from ml.model_inference import GesturePredictor
            predictor = GesturePredictor(ML_MODEL_PATH)
            if predictor.is_loaded:
                ml_predictor = predictor
                print(f"ML model loaded ({predictor.method}) — hybrid mode active")
            else:
                print("no trained ML model found — using rules only")
        except ImportError:
            print("scikit-learn not installed — using rules only")

    # custom gesture registry
    gesture_registry = None
    try:
        from ml.gesture_customizer import GestureRegistry
        gesture_registry = GestureRegistry()
        if gesture_registry.names:
            print(f"custom gestures loaded: {', '.join(gesture_registry.names)}")
    except ImportError:
        pass

    # noise filter - one per hand side
    noise_filters = {"right": NoiseFilter(), "left": NoiseFilter()}
    profiler = Profiler(enabled=PROFILER_ENABLED)

    # benchmark tracking
    gesture_counts = {}
    ml_override_count = 0
    ml_agree_count = 0      # frames where ML and rules gave the same answer
    ml_disagree_count = 0   # frames where they disagreed (ML not confident enough to override)
    ml_predictions = []     # (rule_gesture, ml_gesture, ml_conf, was_overridden) per frame
    total_frames = 0

    # grab one frame to get actual resolution (camera might not match config)
    ok, test_frame = cam.read()
    if not ok:
        print("cant read from camera")
        sys.exit(1)
    actual_h, actual_w = test_frame.shape[:2]
    canvas = Canvas(actual_w, actual_h)
    ui = UI()

    tools = {
        "pen": Pen(),
        "brush": Brush(),
        "eraser": Eraser(),
    }

    color_changed = False
    brush_switched = False
    grab_start = None
    self_idle_count = 0
    frame_drop_count = 0
    no_hand_frames = 0
    tracking_confidence = 1.0

    print("Air Drawing started. Press 'q' to quit, 'c' to clear, 's' to save drawing, 'f' to save full frame.")
    if PROFILER_ENABLED:
        print("Profiler ON - press 'p' to print timing report, or it prints on exit.")

    cv2.namedWindow("Air Drawing", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Air Drawing", actual_w, actual_h)

    while True:
        profiler.begin_frame()

        profiler.start("camera")
        ok, frame = cam.read()
        profiler.stop("camera")

        if not ok:
            frame_drop_count += 1
            if frame_drop_count > 30:
                print("camera feed lost for too long, exiting")
                break
            continue
        frame_drop_count = 0

        # detect hands
        profiler.start("detection")
        hand_data = tracker.find_hands(frame)
        profiler.stop("detection")

        # run noise filter on each hand
        profiler.start("noise_filter")
        if NOISE_FILTER_ENABLED:
            # tell the filter what the user is doing so it picks the
            # right beta. drawing needs low lag (beta=0.08, ~8px).
            # idle/gesture needs heavy smoothing (beta=0.007).
            filter_mode = "draw" if state.drawing else "idle"
            for nf in noise_filters.values():
                nf.set_mode(filter_mode)

            filtered_data = {}
            for side in ("right", "left"):
                raw = hand_data.get(side)
                filtered_data[side] = noise_filters[side].filter(raw)
            hand_data = filtered_data

            # pick the active hand's confidence for UI
            active = gesture_engine._active_hand
            if active and active in noise_filters:
                tracking_confidence = noise_filters[active].confidence
            else:
                # average both if we dont know which hand yet
                scores = [nf.confidence for nf in noise_filters.values() if nf.confidence > 0]
                tracking_confidence = sum(scores) / len(scores) if scores else 0.0
        profiler.stop("noise_filter")

        # check for multi-hand gestures first
        profiler.start("gesture")
        multi_gesture = gesture_engine.recognize_multi(hand_data)
        if multi_gesture == "clear_canvas":
            canvas.clear()
            state.reset_prev_point()

        # single hand gesture
        gesture, tip_pos, erase_points = gesture_engine.recognize(hand_data)

        # ML hybrid: override the rule-based gesture if the model is confident
        if ml_predictor is not None:
            active_side = gesture_engine._active_hand
            active_lm = hand_data.get(active_side) if active_side else None
            if active_lm is not None:
                ml_gesture, ml_conf = ml_predictor.predict(active_lm)

                # track agreement between the two systems
                overridden = False
                if ml_gesture == gesture:
                    ml_agree_count += 1
                else:
                    ml_disagree_count += 1
                    if ml_conf >= ML_CONFIDENCE_THRESHOLD:
                        _ml_logger.info(
                            f"OVERRIDE  rules={gesture:<15} ml={ml_gesture:<15} "
                            f"conf={ml_conf:.2f}  frame={total_frames}"
                        )
                        gesture = ml_gesture
                        ml_override_count += 1
                        overridden = True
                    else:
                        _ml_logger.debug(
                            f"DISAGREE  rules={gesture:<15} ml={ml_gesture:<15} "
                            f"conf={ml_conf:.2f} (below threshold)  frame={total_frames}"
                        )

                ml_predictions.append((gesture, ml_gesture, ml_conf, overridden))

        # track gesture distribution for benchmarking
        gesture_counts[gesture] = gesture_counts.get(gesture, 0) + 1
        total_frames += 1
        profiler.stop("gesture")

        # handle gestures
        profiler.start("drawing")

        # skip drawing when tracking is unreliable
        low_confidence = NOISE_FILTER_ENABLED and tracking_confidence < LOW_CONFIDENCE_THRESHOLD

        # check if this is a custom gesture with a special action
        custom_action = None
        if gesture_registry and gesture not in ("draw", "erase", "change_color",
                                                  "switch_brush", "grab", "idle"):
            custom_action = gesture_registry.get_action(gesture)
            if custom_action == "save_drawing":
                canvas.save_drawing()
            elif custom_action == "clear_canvas":
                canvas.clear()
                state.reset_prev_point()
            elif custom_action == "screenshot":
                canvas.save_with_frame(frame)
            # after handling the action, treat it like idle for drawing purposes
            gesture = "idle"

        if gesture == "draw" and not low_confidence:
            state.set_drawing(True)
            if state.current_tool == "eraser":
                state.set_tool("pen")
            tool = tools[state.current_tool]
            if state.prev_point and tip_pos:
                # sanity check - dont draw huge jumps (hand re-entry)
                dx = abs(tip_pos[0] - state.prev_point[0])
                dy = abs(tip_pos[1] - state.prev_point[1])
                if dx < 120 and dy < 120:
                    tool.draw(canvas, state.prev_point, tip_pos, state.color)
                else:
                    # too far, start a new stroke
                    canvas.finish_stroke()
            state.prev_point = tip_pos
            color_changed = False
            brush_switched = False
            grab_start = None

        elif gesture == "grab":
            # pinch to grab and move the canvas content
            if tip_pos:
                if grab_start is not None:
                    dx = tip_pos[0] - grab_start[0]
                    dy = tip_pos[1] - grab_start[1]
                    if abs(dx) > 1 or abs(dy) > 1:
                        canvas.shift_surface(dx, dy)
                grab_start = tip_pos
            state.set_drawing(False)
            state.prev_point = None
            color_changed = False
            brush_switched = False

        # also clear prev_point when entering non-draw states
        elif gesture == "erase":
            state.set_tool("eraser")
            state.set_drawing(True)
            tool = tools["eraser"]
            for pt in erase_points:
                tool.draw(canvas, None, pt, None)
            state.prev_point = None
            color_changed = False
            brush_switched = False
            grab_start = None

        elif gesture == "change_color":
            if not color_changed:
                state.next_color()
                color_changed = True
            state.set_drawing(False)
            brush_switched = False
            grab_start = None

        elif gesture == "switch_brush":
            if not brush_switched:
                # cycle through pen -> brush -> pen
                if state.current_tool == "pen":
                    state.set_tool("brush")
                elif state.current_tool == "brush":
                    state.set_tool("pen")
                elif state.current_tool == "eraser":
                    state.set_tool("pen")
                brush_switched = True
            state.set_drawing(False)
            color_changed = False
            grab_start = None

        else:
            # idle - only reset prev_point after a few frames
            # so brief flickers dont break the stroke
            state.set_drawing(False)
            if self_idle_count > 4:
                canvas.finish_stroke()
                state.prev_point = None
            color_changed = False
            brush_switched = False
            grab_start = None

        profiler.stop("drawing")

        # track how long weve been idle
        if gesture == "idle":
            self_idle_count += 1
        else:
            self_idle_count = 0

        # draw landmarks if enabled
        profiler.start("rendering")
        if SHOW_LANDMARKS:
            frame = tracker.draw_landmarks(frame, hand_data)

        # composite canvas onto frame
        frame = canvas.blend_onto(frame)
        profiler.stop("rendering")

        # draw UI
        profiler.start("ui")
        frame = ui.draw_overlay(frame, state, gesture, tracking_confidence)
        profiler.stop("ui")

        cv2.imshow("Air Drawing", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:  # q or ESC
            break
        elif key == ord('c'):
            canvas.clear()
            state.reset_prev_point()
        elif key == ord('s'):
            canvas.save_drawing()
        elif key == ord('f'):
            canvas.save_with_frame(frame)
        elif key == ord('p') and PROFILER_ENABLED:
            profiler.report()

        # also quit if the window X button is clicked
        if cv2.getWindowProperty("Air Drawing", cv2.WND_PROP_VISIBLE) < 1:
            break

        profiler.end_frame()

    # print final profiling data
    if PROFILER_ENABLED:
        profiler.report()

    # write benchmark log
    if BENCHMARK_ENABLED and total_frames > 0:
        _write_benchmark(profiler, gesture_counts, ml_override_count,
                         ml_agree_count, ml_disagree_count, ml_predictions,
                         total_frames)

    tracker.close()
    cam.release()
    cv2.destroyAllWindows()


def _write_benchmark(profiler, gesture_counts, ml_overrides,
                     ml_agrees, ml_disagrees, ml_predictions, total_frames):
    """save session performance metrics to a JSON file"""
    import json
    from datetime import datetime

    ml_total = ml_agrees + ml_disagrees
    data = {
        "timestamp": datetime.now().isoformat(),
        "total_frames": total_frames,
        "avg_fps": round(profiler.get_fps(), 1) if profiler.enabled else 0,
        "gesture_distribution": gesture_counts,
        "ml_hybrid": {
            "frames_with_ml": ml_total,
            "overrides": ml_overrides,
            "override_rate_pct": round(ml_overrides / ml_total * 100, 2) if ml_total > 0 else 0,
            "agreement_rate_pct": round(ml_agrees / ml_total * 100, 2) if ml_total > 0 else 0,
            "disagreement_rate_pct": round(ml_disagrees / ml_total * 100, 2) if ml_total > 0 else 0,
        },
    }

    # if we have ML predictions, compute per-gesture accuracy comparison
    if ml_predictions:
        from collections import defaultdict
        confusion = defaultdict(lambda: defaultdict(int))
        for rule_g, ml_g, conf, overridden in ml_predictions:
            confusion[rule_g][ml_g] += 1

        # flatten for JSON
        data["ml_hybrid"]["confusion"] = {
            rule_g: dict(ml_counts)
            for rule_g, ml_counts in confusion.items()
        }

        # confidence stats per gesture
        conf_by_gesture = defaultdict(list)
        for _, ml_g, conf, _ in ml_predictions:
            conf_by_gesture[ml_g].append(conf)

        data["ml_hybrid"]["confidence_by_gesture"] = {
            g: {
                "avg": round(sum(cs) / len(cs), 3),
                "min": round(min(cs), 3),
                "max": round(max(cs), 3),
            }
            for g, cs in conf_by_gesture.items()
        }

    if profiler.enabled:
        data["timing"] = profiler.summary_dict()

    log_path = "benchmark_log.json"
    entries = []
    if os.path.exists(log_path):
        try:
            with open(log_path, "r") as f:
                entries = json.load(f)
        except (json.JSONDecodeError, IOError):
            entries = []

    entries.append(data)
    # keep last 20 sessions
    if len(entries) > 20:
        entries = entries[-20:]

    with open(log_path, "w") as f:
        json.dump(entries, f, indent=2)
    print(f"benchmark data saved to {log_path}")


if __name__ == "__main__":
    main()

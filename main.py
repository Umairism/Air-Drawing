import cv2
import sys

from core.camera import Camera
from core.hand_tracker import HandTracker
from core.gesture_engine import GestureEngine
from core.state_manager import StateManager
from app.canvas import Canvas
from app.tools import Pen, Brush, Eraser
from app.ui import UI
from app.config import SHOW_LANDMARKS


def main():
    # init everything
    try:
        cam = Camera()
    except RuntimeError as e:
        print(f"Error: {e}")
        sys.exit(1)

    tracker = HandTracker()
    gesture_engine = GestureEngine()
    state = StateManager()

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

    print("Air Drawing started. Press 'q' to quit, 'c' to clear, 's' to save drawing, 'f' to save full frame.")

    cv2.namedWindow("Air Drawing", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Air Drawing", actual_w, actual_h)

    while True:
        ok, frame = cam.read()
        if not ok:
            print("lost camera feed")
            break

        # detect hands
        hand_data = tracker.find_hands(frame)

        # check for multi-hand gestures first
        multi_gesture = gesture_engine.recognize_multi(hand_data)
        if multi_gesture == "clear_canvas":
            canvas.clear()
            state.reset_prev_point()

        # single hand gesture
        gesture, tip_pos, erase_points = gesture_engine.recognize(hand_data)

        # handle gestures
        if gesture == "draw":
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

        # track how long weve been idle
        if gesture == "idle":
            self_idle_count += 1
        else:
            self_idle_count = 0

        # draw landmarks if enabled
        if SHOW_LANDMARKS:
            frame = tracker.draw_landmarks(frame, hand_data)

        # composite canvas onto frame
        frame = canvas.blend_onto(frame)

        # draw UI
        frame = ui.draw_overlay(frame, state, gesture)

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

        # also quit if the window X button is clicked
        if cv2.getWindowProperty("Air Drawing", cv2.WND_PROP_VISIBLE) < 1:
            break

    tracker.close()
    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

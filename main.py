import cv2
import sys

from core.camera import Camera
from core.hand_tracker import HandTracker
from core.gesture_engine import GestureEngine
from core.state_manager import StateManager
from app.canvas import Canvas
from app.tools import Pen, Brush, Eraser
from app.ui import UI
from app.config import CAMERA_WIDTH, CAMERA_HEIGHT, SHOW_LANDMARKS


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
    canvas = Canvas(CAMERA_WIDTH, CAMERA_HEIGHT)
    ui = UI()

    tools = {
        "pen": Pen(),
        "brush": Brush(),
        "eraser": Eraser(),
    }

    color_changed = False
    brush_switched = False

    print("Air Drawing started. Press 'q' to quit, 'c' to clear canvas.")

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
        gesture, tip_pos = gesture_engine.recognize(hand_data)

        # handle gestures
        if gesture == "draw":
            state.set_drawing(True)
            tool = tools[state.current_tool]
            if state.prev_point and tip_pos:
                tool.draw(canvas, state.prev_point, tip_pos, state.color)
            state.prev_point = tip_pos
            color_changed = False
            brush_switched = False

        elif gesture == "erase":
            state.set_tool("eraser")
            state.set_drawing(True)
            tool = tools["eraser"]
            if tip_pos:
                tool.draw(canvas, None, tip_pos, None)
            state.prev_point = None
            color_changed = False
            brush_switched = False

        elif gesture == "change_color":
            if not color_changed:
                state.next_color()
                color_changed = True
            state.set_drawing(False)
            brush_switched = False

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

        else:
            # idle
            state.set_drawing(False)
            canvas.finish_stroke()
            color_changed = False
            brush_switched = False

        # draw landmarks if enabled
        if SHOW_LANDMARKS:
            frame = tracker.draw_landmarks(frame, hand_data)

        # composite canvas onto frame
        frame = canvas.blend_onto(frame)

        # draw UI
        frame = ui.draw_overlay(frame, state)

        cv2.imshow("Air Drawing", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            canvas.clear()
            state.reset_prev_point()

    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

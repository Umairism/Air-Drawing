import cv2
import time
from app.config import COLORS, UI_FONT_SCALE, UI_THICKNESS


class UI:
    """handles all the overlay stuff - fps, tool name, color palette"""

    def __init__(self):
        self.prev_time = time.time()
        self.fps = 0

    def update_fps(self):
        now = time.time()
        dt = now - self.prev_time
        self.fps = int(1 / dt) if dt > 0 else 0
        self.prev_time = now

    def draw_overlay(self, frame, state):
        """draw all the UI elements onto the frame"""
        self.update_fps()
        h, w = frame.shape[:2]

        # fps counter - top left
        cv2.putText(
            frame, f"FPS: {self.fps}", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, UI_FONT_SCALE, (0, 255, 0), UI_THICKNESS
        )

        # current tool - top center
        tool_text = f"Tool: {state.current_tool.upper()}"
        text_size = cv2.getTextSize(tool_text, cv2.FONT_HERSHEY_SIMPLEX, UI_FONT_SCALE, UI_THICKNESS)[0]
        tx = (w - text_size[0]) // 2
        cv2.putText(
            frame, tool_text, (tx, 30),
            cv2.FONT_HERSHEY_SIMPLEX, UI_FONT_SCALE, (255, 255, 255), UI_THICKNESS
        )

        # color palette - top right
        palette_x = w - 40 * len(COLORS) - 10
        for i, color in enumerate(COLORS):
            x = palette_x + i * 40
            cv2.rectangle(frame, (x, 8), (x + 30, 38), color, -1)
            # highlight current color
            if i == state.color_index:
                cv2.rectangle(frame, (x - 2, 6), (x + 32, 40), (255, 255, 255), 2)

        # current color indicator dot - below tool name
        cv2.circle(frame, (w // 2, 55), 10, state.color, -1)
        cv2.circle(frame, (w // 2, 55), 10, (255, 255, 255), 1)

        return frame

import cv2
import time
from app.config import COLORS, UI_FONT_SCALE, UI_THICKNESS


# color coding for each gesture state
GESTURE_COLORS = {
    "draw":         (0, 255, 0),     # green
    "erase":        (0, 0, 255),     # red
    "change_color": (255, 200, 0),   # cyan-ish
    "switch_brush": (255, 0, 255),   # magenta
    "grab":         (0, 200, 255),   # orange-yellow
    "idle":         (150, 150, 150), # gray
}


class UI:
    """handles all the overlay stuff - fps, tool name, color palette, gesture feedback"""

    def __init__(self):
        self.prev_time = time.time()
        self.fps = 0
        self._fps_samples = []

    def update_fps(self):
        now = time.time()
        dt = now - self.prev_time
        self.prev_time = now
        if dt > 0:
            self._fps_samples.append(1.0 / dt)
        # average over last 10 samples so it doesnt jump around
        if len(self._fps_samples) > 10:
            self._fps_samples = self._fps_samples[-10:]
        self.fps = int(sum(self._fps_samples) / len(self._fps_samples)) if self._fps_samples else 0

    def _draw_pill(self, frame, text, x, y, color, bg=(40, 40, 40)):
        """draw text with a rounded pill-shaped background"""
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.55
        thick = 1
        sz, baseline = cv2.getTextSize(text, font, scale, thick)
        pad_x, pad_y = 10, 6
        x1, y1 = x, y - sz[1] - pad_y
        x2, y2 = x + sz[0] + pad_x * 2, y + pad_y + baseline

        # semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), bg, -1)
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

        cv2.putText(frame, text, (x + pad_x, y), font, scale, color, thick, cv2.LINE_AA)
        return x2  # return right edge for chaining

    def draw_overlay(self, frame, state, gesture_name=None, confidence=1.0):
        """draw all the UI elements onto the frame"""
        self.update_fps()
        h, w = frame.shape[:2]

        # fps counter - top left
        fps_color = (0, 255, 0) if self.fps >= 20 else (0, 200, 255) if self.fps >= 12 else (0, 0, 255)
        self._draw_pill(frame, f"FPS: {self.fps}", 8, 28, fps_color)

        # tracking confidence bar - below fps
        if confidence < 0.95:
            bar_w = 80
            bar_h = 6
            bar_x = 10
            bar_y = 38
            fill = int(bar_w * min(confidence, 1.0))
            bar_color = (0, 255, 0) if confidence > 0.6 else (0, 200, 255) if confidence > 0.3 else (0, 0, 255)
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (60, 60, 60), -1)
            if fill > 0:
                cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill, bar_y + bar_h), bar_color, -1)

        # current tool - top center
        tool_text = f"{state.current_tool.upper()}"
        tsz = cv2.getTextSize(tool_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)[0]
        tool_x = (w - tsz[0]) // 2 - 10
        self._draw_pill(frame, tool_text, tool_x, 28, (255, 255, 255))

        # color palette - top right
        palette_w = 32
        palette_gap = 6
        total_palette = len(COLORS) * (palette_w + palette_gap) - palette_gap
        palette_x = w - total_palette - 12
        for i, color in enumerate(COLORS):
            x = palette_x + i * (palette_w + palette_gap)
            cv2.rectangle(frame, (x, 8), (x + palette_w, 8 + palette_w), color, -1)
            if i == state.color_index:
                cv2.rectangle(frame, (x - 2, 6), (x + palette_w + 2, 10 + palette_w), (255, 255, 255), 2)

        # current color dot under tool name
        cv2.circle(frame, (w // 2, 52), 8, state.color, -1)
        cv2.circle(frame, (w // 2, 52), 9, (255, 255, 255), 1)

        # gesture feedback at bottom center
        if gesture_name:
            g_color = GESTURE_COLORS.get(gesture_name, (180, 180, 180))
            label = gesture_name.replace("_", " ").upper()
            gsz = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)[0]
            gx = (w - gsz[0]) // 2 - 10
            self._draw_pill(frame, label, gx, h - 16, g_color)

            # draw a small colored circle as visual indicator
            cv2.circle(frame, (gx - 12, h - 22), 6, g_color, -1)

        # "no hand" hint if idle for a while
        if gesture_name == "idle":
            hint = "Show your hand to start"
            hsz = cv2.getTextSize(hint, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            hx = (w - hsz[0]) // 2
            cv2.putText(frame, hint, (hx, h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
        elif confidence < 0.3 and gesture_name != "idle":
            hint = "Tracking unstable - move slower or improve lighting"
            hsz = cv2.getTextSize(hint, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0]
            hx = (w - hsz[0]) // 2
            cv2.putText(frame, hint, (hx, h - 44),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 200), 1, cv2.LINE_AA)

        return frame

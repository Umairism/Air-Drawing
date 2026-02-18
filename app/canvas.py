import numpy as np
import cv2


class Canvas:
    """
    the drawing surface. we keep a separate transparent layer
    and blend it onto the camera feed each frame.
    """

    def __init__(self, width, height):
        self.width = width
        self.height = height
        # use a black canvas, we'll addWeighted it onto the frame
        self.surface = np.zeros((height, width, 3), dtype=np.uint8)
        # keep track of strokes so we could undo later if we want
        self.strokes = []
        self._current_stroke = []

    def draw_line(self, pt1, pt2, color, thickness):
        """draw a line segment between two points"""
        if pt1 is None or pt2 is None:
            return
        cv2.line(self.surface, pt1, pt2, color, thickness)
        self._current_stroke.append((pt1, pt2, color, thickness))

    def erase_at(self, point, radius=40):
        """erase by drawing a black circle"""
        if point is None:
            return
        cv2.circle(self.surface, point, radius, (0, 0, 0), -1)

    def clear(self):
        """wipe everything"""
        self.surface = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self.strokes.clear()
        self._current_stroke.clear()

    def finish_stroke(self):
        """call this when the user lifts their finger"""
        if self._current_stroke:
            self.strokes.append(self._current_stroke.copy())
            self._current_stroke.clear()

    def shift_surface(self, dx, dy):
        """move the entire drawing by dx, dy pixels"""
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        self.surface = cv2.warpAffine(self.surface, M, (self.width, self.height))

    def save_drawing(self):
        """save just the drawing on a white background"""
        from datetime import datetime
        white = np.ones_like(self.surface) * 255
        gray = cv2.cvtColor(self.surface, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)

        bg = cv2.bitwise_and(white, white, mask=mask_inv)
        fg = cv2.bitwise_and(self.surface, self.surface, mask=mask)
        result = cv2.add(bg, fg)

        filename = f"drawing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        cv2.imwrite(filename, result)
        print(f"saved drawing to {filename}")
        return filename

    def save_with_frame(self, frame):
        """save the full view - drawing on top of camera feed"""
        from datetime import datetime
        combined = self.blend_onto(frame.copy())
        filename = f"capture_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        cv2.imwrite(filename, combined)
        print(f"saved capture to {filename}")
        return filename

    def blend_onto(self, frame):
        """overlay our drawing onto the camera frame"""
        gray = cv2.cvtColor(self.surface, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)

        bg = cv2.bitwise_and(frame, frame, mask=mask_inv)
        fg = cv2.bitwise_and(self.surface, self.surface, mask=mask)

        return cv2.add(bg, fg)

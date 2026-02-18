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

    def blend_onto(self, frame):
        """overlay our drawing onto the camera frame"""
        # wherever the canvas has color, show it on top of the frame
        gray = cv2.cvtColor(self.surface, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)

        # black out the drawing area on the frame
        bg = cv2.bitwise_and(frame, frame, mask=mask_inv)
        # take only the drawing from our canvas
        fg = cv2.bitwise_and(self.surface, self.surface, mask=mask)

        return cv2.add(bg, fg)

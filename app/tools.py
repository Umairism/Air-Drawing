import cv2
import numpy as np
from app.config import PEN_THICKNESS, BRUSH_THICKNESS, ERASER_THICKNESS


class Pen:
    """basic thin line drawing"""

    def __init__(self):
        self.thickness = PEN_THICKNESS

    def draw(self, canvas, pt1, pt2, color):
        canvas.draw_line(pt1, pt2, color, self.thickness)


class Brush:
    """thicker strokes with some smoothing between points"""

    def __init__(self):
        self.thickness = BRUSH_THICKNESS
        self.prev_points = []  # for smoothing

    def draw(self, canvas, pt1, pt2, color):
        if pt1 is None or pt2 is None:
            self.prev_points.clear()
            return

        self.prev_points.append(pt2)

        # smooth by averaging recent points
        if len(self.prev_points) > 3:
            self.prev_points = self.prev_points[-4:]
            avg_x = int(np.mean([p[0] for p in self.prev_points]))
            avg_y = int(np.mean([p[1] for p in self.prev_points]))
            smoothed = (avg_x, avg_y)
            canvas.draw_line(pt1, smoothed, color, self.thickness)
        else:
            canvas.draw_line(pt1, pt2, color, self.thickness)


class Eraser:
    """erases by painting over with black"""

    def __init__(self):
        self.radius = ERASER_THICKNESS

    def draw(self, canvas, pt1, pt2, color):
        # color doesnt matter for eraser
        canvas.erase_at(pt2, self.radius)

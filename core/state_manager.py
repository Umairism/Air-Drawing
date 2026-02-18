from app.config import DEFAULT_COLOR, COLORS


class StateManager:
    """keeps track of what tool is active, current color, etc."""

    def __init__(self):
        self.current_tool = "pen"  # pen, brush, eraser
        self.color = DEFAULT_COLOR
        self.color_index = 0
        self.drawing = False
        self.prev_point = None

    def set_tool(self, tool_name):
        if tool_name in ("pen", "brush", "eraser"):
            self.current_tool = tool_name

    def next_color(self):
        self.color_index = (self.color_index + 1) % len(COLORS)
        self.color = COLORS[self.color_index]

    def reset_prev_point(self):
        self.prev_point = None

    def set_drawing(self, is_drawing):
        self.drawing = is_drawing
        if not is_drawing:
            self.prev_point = None

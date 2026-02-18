# all the settings live here so we dont scatter magic numbers everywhere

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_INDEX = 0

# mediapipe
MAX_HANDS = 2
DETECTION_CONFIDENCE = 0.55
TRACKING_CONFIDENCE = 0.5
DETECTION_WIDTH = 480     # process at lower res for speed
PROCESS_EVERY_N_FRAMES = 1  # process every frame for gesture accuracy

# drawing defaults
DEFAULT_COLOR = (255, 50, 50)   # blueish
PEN_THICKNESS = 3
BRUSH_THICKNESS = 12
ERASER_THICKNESS = 40

# color palette - just some nice colors
COLORS = [
    (255, 50, 50),    # blue
    (50, 50, 255),    # red
    (50, 255, 50),    # green
    (0, 255, 255),    # yellow
    (255, 0, 255),    # magenta
    (255, 165, 0),    # orange
    (255, 255, 255),  # white
]

# gesture timing
DEBOUNCE_FRAMES = 3
GESTURE_COOLDOWN = 0.4  # seconds between gesture switches

# ui
SHOW_FPS = True
SHOW_LANDMARKS = True
UI_FONT_SCALE = 0.6
UI_THICKNESS = 2

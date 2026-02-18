import time
from app.config import DEBOUNCE_FRAMES, GESTURE_COOLDOWN


class GestureEngine:
    """
    takes raw landmark data and figures out what gesture the user is making.
    handles debouncing so we dont get jittery switches.
    """

    # landmark indices we care about
    THUMB_TIP = 4
    INDEX_TIP = 8
    MIDDLE_TIP = 12
    RING_TIP = 16
    PINKY_TIP = 20

    THUMB_IP = 3
    INDEX_PIP = 6
    MIDDLE_PIP = 10
    RING_PIP = 14
    PINKY_PIP = 18

    def __init__(self):
        self.last_gesture = None
        self.gesture_counter = 0
        self.last_switch_time = 0
        self.debounce_threshold = DEBOUNCE_FRAMES

    def get_finger_states(self, landmarks):
        """
        figure out which fingers are up.
        returns [thumb, index, middle, ring, pinky] as booleans.
        """
        if landmarks is None:
            return [False] * 5

        # build a quick lookup: id -> (x, y)
        lm = {}
        for idx, x, y, z in landmarks:
            lm[idx] = (x, y)

        fingers = []

        # thumb is special - compare x not y
        # if thumb tip is to the left of thumb IP joint, its "up" (for right hand in mirrored view)
        if self.THUMB_TIP in lm and self.THUMB_IP in lm:
            fingers.append(lm[self.THUMB_TIP][0] < lm[self.THUMB_IP][0])
        else:
            fingers.append(False)

        # for the other 4 fingers: tip above PIP means finger is up
        pairs = [
            (self.INDEX_TIP, self.INDEX_PIP),
            (self.MIDDLE_TIP, self.MIDDLE_PIP),
            (self.RING_TIP, self.RING_PIP),
            (self.PINKY_TIP, self.PINKY_PIP),
        ]
        for tip, pip in pairs:
            if tip in lm and pip in lm:
                fingers.append(lm[tip][1] < lm[pip][1])  # y goes down in opencv
            else:
                fingers.append(False)

        return fingers

    def recognize(self, hand_data):
        """
        main method. takes the hand_data dict from tracker and returns
        a gesture string + the fingertip position for drawing.

        gestures:
            "draw"          - index finger only -> use pen/brush
            "erase"         - all fingers open
            "change_color"  - index + middle crossed/up, others down
            "switch_brush"  - fist (all down)
            "idle"          - anything else
        """
        # primarily use the right hand for single-hand gestures
        landmarks = hand_data.get("right") or hand_data.get("left")

        if landmarks is None:
            return "idle", None

        fingers = self.get_finger_states(landmarks)
        thumb, index, middle, ring, pinky = fingers

        # figure out the tip position for drawing
        lm = {}
        for idx, x, y, z in landmarks:
            lm[idx] = (x, y)

        tip_pos = lm.get(self.INDEX_TIP, None)

        # --- gesture detection ---

        gesture = "idle"

        # index only up -> draw mode
        if index and not middle and not ring and not pinky:
            gesture = "draw"

        # all fingers open -> eraser
        elif index and middle and ring and pinky:
            gesture = "erase"

        # index + middle up, rest down -> color change
        elif index and middle and not ring and not pinky:
            gesture = "change_color"

        # fist (nothing up) -> switch to brush
        elif not index and not middle and not ring and not pinky and not thumb:
            gesture = "switch_brush"

        # apply debounce
        gesture = self._debounce(gesture)

        return gesture, tip_pos

    def _debounce(self, gesture):
        """
        only switch gestures if we've seen the same one for enough frames.
        prevents accidental flickering between modes.
        """
        now = time.time()

        if gesture == self.last_gesture:
            self.gesture_counter += 1
        else:
            self.gesture_counter = 1

        # if we've seen this gesture enough times AND cooldown has passed
        if self.gesture_counter >= self.debounce_threshold:
            if gesture != self.last_gesture:
                if now - self.last_switch_time >= GESTURE_COOLDOWN:
                    self.last_switch_time = now
                    self.last_gesture = gesture
            return gesture

        # not confident yet, return the old one
        return self.last_gesture if self.last_gesture else "idle"

    def recognize_multi(self, hand_data):
        """
        detect two-hand gestures. returns gesture name or None if
        no multi-hand gesture is detected.
        """
        left = hand_data.get("left")
        right = hand_data.get("right")

        if left is None or right is None:
            return None

        left_fingers = self.get_finger_states(left)
        right_fingers = self.get_finger_states(right)

        # both fists = clap/reset gesture
        if not any(left_fingers) and not any(right_fingers):
            return "clear_canvas"

        # both hands open = pause
        if all(left_fingers[1:]) and all(right_fingers[1:]):
            return "pause"

        return None

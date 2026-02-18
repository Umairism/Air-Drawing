import time
import math
from app.config import DEBOUNCE_FRAMES, GESTURE_COOLDOWN


class GestureEngine:
    """
    takes raw landmark data and figures out what gesture the user is making.
    handles debouncing so we dont get jittery switches.
    """

    # landmark indices
    THUMB_TIP = 4
    INDEX_TIP = 8
    MIDDLE_TIP = 12
    RING_TIP = 16
    PINKY_TIP = 20

    THUMB_IP = 3
    INDEX_PIP = 6
    INDEX_MCP = 5
    MIDDLE_PIP = 10
    RING_PIP = 14
    PINKY_PIP = 18

    WRIST = 0
    MIDDLE_MCP = 9

    # how far tip needs to be above PIP to count as "up" (in pixels)
    FINGER_UP_MARGIN = 15
    # how far tip needs to be below PIP to count as "down"
    FINGER_DOWN_MARGIN = 10
    # pinch distance threshold
    PINCH_THRESHOLD = 30

    def __init__(self):
        self.last_gesture = "idle"
        self.confirmed_gesture = "idle"
        self.gesture_counter = 0
        self.last_switch_time = 0
        self.debounce_threshold = DEBOUNCE_FRAMES
        # track finger states with hysteresis to avoid flickering
        self._finger_states = [False] * 5
        # which hand we're using for drawing - stick with it once chosen
        self._active_hand = None

    def get_finger_states(self, landmarks):
        """
        figure out which fingers are up with a margin so slight bends
        dont cause flickering.
        """
        if landmarks is None:
            return [False] * 5

        lm = {}
        for idx, x, y, z in landmarks:
            lm[idx] = (x, y)

        fingers = list(self._finger_states)  # start from previous state

        # thumb - x comparison with margin
        if self.THUMB_TIP in lm and self.THUMB_IP in lm:
            diff = lm[self.THUMB_IP][0] - lm[self.THUMB_TIP][0]
            if diff > self.FINGER_UP_MARGIN:
                fingers[0] = True
            elif diff < -self.FINGER_DOWN_MARGIN:
                fingers[0] = False

        # other 4 fingers - y comparison with margin
        pairs = [
            (self.INDEX_TIP, self.INDEX_PIP, 1),
            (self.MIDDLE_TIP, self.MIDDLE_PIP, 2),
            (self.RING_TIP, self.RING_PIP, 3),
            (self.PINKY_TIP, self.PINKY_PIP, 4),
        ]
        for tip, pip, i in pairs:
            if tip in lm and pip in lm:
                diff = lm[pip][1] - lm[tip][1]  # positive means tip is above pip
                if diff > self.FINGER_UP_MARGIN:
                    fingers[i] = True
                elif diff < -self.FINGER_DOWN_MARGIN:
                    fingers[i] = False

        self._finger_states = fingers
        return fingers

    def _get_pinch_dist(self, lm):
        """distance between thumb tip and index tip"""
        thumb = lm.get(self.THUMB_TIP)
        index = lm.get(self.INDEX_TIP)
        if thumb and index:
            dx = thumb[0] - index[0]
            dy = thumb[1] - index[1]
            return math.sqrt(dx * dx + dy * dy)
        return 999

    def recognize(self, hand_data):
        """
        main method. takes the hand_data dict from tracker and returns
        a gesture string + the fingertip position for drawing + erase points.
        always uses the same hand for drawing - defaults to right, sticks
        with whichever hand it locked onto.
        """
        # pick which hand to use and stick with it
        right = hand_data.get("right")
        left = hand_data.get("left")

        if self._active_hand == "right" and right is not None:
            landmarks = right
        elif self._active_hand == "left" and left is not None:
            landmarks = left
        elif right is not None:
            landmarks = right
            self._active_hand = "right"
        elif left is not None:
            landmarks = left
            self._active_hand = "left"
        else:
            # no hands visible, reset lock so next hand gets picked fresh
            self._active_hand = None
            landmarks = None

        if landmarks is None:
            return "idle", None, []

        fingers = self.get_finger_states(landmarks)
        thumb, index, middle, ring, pinky = fingers

        lm = {}
        for idx, x, y, z in landmarks:
            lm[idx] = (x, y)

        tip_pos = lm.get(self.INDEX_TIP, None)

        # palm center
        palm_pos = None
        if self.WRIST in lm and self.MIDDLE_MCP in lm:
            wx, wy = lm[self.WRIST]
            mx, my = lm[self.MIDDLE_MCP]
            palm_pos = ((wx + mx) // 2, (wy + my) // 2)

        # erase points - whole hand
        erase_points = []
        if palm_pos:
            erase_points.append(palm_pos)
        for tip_id in [self.THUMB_TIP, self.INDEX_TIP, self.MIDDLE_TIP, self.RING_TIP, self.PINKY_TIP]:
            if tip_id in lm:
                erase_points.append(lm[tip_id])

        # --- gesture detection with strict checks ---

        pinch_dist = self._get_pinch_dist(lm)
        gesture = "idle"

        # count how many fingers are clearly up/down
        up_count = sum([index, middle, ring, pinky])
        down_count = sum([not index, not middle, not ring, not pinky])

        # pinch = grab (thumb+index pinched, other 3 fingers down)
        if pinch_dist < self.PINCH_THRESHOLD and down_count >= 3:
            gesture = "grab"

        # draw = ONLY index up, other 3 must be down
        elif index and not middle and not ring and not pinky:
            gesture = "draw"

        # erase = all 4 fingers clearly up
        elif up_count == 4:
            gesture = "erase"

        # color change = index + middle up, ring + pinky must both be down
        elif index and middle and not ring and not pinky:
            gesture = "change_color"

        # fist = all 5 down (including thumb)
        elif down_count == 4 and not thumb:
            gesture = "switch_brush"

        gesture = self._debounce(gesture)
        return gesture, tip_pos, erase_points

    def _debounce(self, gesture):
        """
        require consistent readings before switching.
        draw stays responsive, everything else needs more confirmation.
        """
        now = time.time()

        if gesture == self.last_gesture:
            self.gesture_counter += 1
        else:
            self.gesture_counter = 1
            self.last_gesture = gesture

        # draw needs to be snappy - 3 frames
        if gesture == "draw":
            if self.gesture_counter >= 3:
                self.confirmed_gesture = gesture
            return self.confirmed_gesture

        # erase/grab need a bit more to avoid accidental triggers
        if gesture in ("erase", "grab"):
            if self.gesture_counter >= 4:
                self.confirmed_gesture = gesture
            return self.confirmed_gesture

        # idle resets quickly
        if gesture == "idle":
            if self.gesture_counter >= 3:
                self.confirmed_gesture = gesture
            return self.confirmed_gesture

        # mode switches (color, brush) need the most confirmation
        if self.gesture_counter >= self.debounce_threshold + 2:
            if now - self.last_switch_time >= GESTURE_COOLDOWN:
                self.last_switch_time = now
                self.confirmed_gesture = gesture

        return self.confirmed_gesture

    def _raw_finger_states(self, landmarks):
        """
        simple finger check without hysteresis - used for multi-hand gestures
        so it doesnt mess with the main hand's cached state.
        """
        if landmarks is None:
            return [False] * 5

        lm = {}
        for idx, x, y, z in landmarks:
            lm[idx] = (x, y)

        fingers = []

        # thumb
        if self.THUMB_TIP in lm and self.THUMB_IP in lm:
            fingers.append(lm[self.THUMB_IP][0] - lm[self.THUMB_TIP][0] > 10)
        else:
            fingers.append(False)

        # other 4
        pairs = [
            (self.INDEX_TIP, self.INDEX_PIP),
            (self.MIDDLE_TIP, self.MIDDLE_PIP),
            (self.RING_TIP, self.RING_PIP),
            (self.PINKY_TIP, self.PINKY_PIP),
        ]
        for tip, pip in pairs:
            if tip in lm and pip in lm:
                fingers.append(lm[pip][1] - lm[tip][1] > 10)
            else:
                fingers.append(False)

        return fingers

    def recognize_multi(self, hand_data):
        """detect two-hand gestures."""
        left = hand_data.get("left")
        right = hand_data.get("right")

        if left is None or right is None:
            return None

        left_fingers = self._raw_finger_states(left)
        right_fingers = self._raw_finger_states(right)

        if not any(left_fingers) and not any(right_fingers):
            return "clear_canvas"

        if all(left_fingers[1:]) and all(right_fingers[1:]):
            return "pause"

        return None

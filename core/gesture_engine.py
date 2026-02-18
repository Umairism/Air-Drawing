import time
import math
from app.config import DEBOUNCE_FRAMES, GESTURE_COOLDOWN


class GestureEngine:
    """
    converts hand landmark data into gesture names.
    uses a combo of joint angles and tip-to-palm distance ratios
    so it works regardless of how far your hand is from the camera.
    """

    # landmark ids
    WRIST = 0
    THUMB_CMC = 1
    THUMB_MCP = 2
    THUMB_IP = 3
    THUMB_TIP = 4
    INDEX_MCP = 5
    INDEX_PIP = 6
    INDEX_DIP = 7
    INDEX_TIP = 8
    MIDDLE_MCP = 9
    MIDDLE_PIP = 10
    MIDDLE_DIP = 11
    MIDDLE_TIP = 12
    RING_MCP = 13
    RING_PIP = 14
    RING_DIP = 15
    RING_TIP = 16
    PINKY_MCP = 17
    PINKY_PIP = 18
    PINKY_DIP = 19
    PINKY_TIP = 20

    # pinch thresholds (relative to hand size)
    PINCH_RATIO_ON = 0.28
    PINCH_RATIO_OFF = 0.40

    def __init__(self):
        self.last_gesture = "idle"
        self.confirmed_gesture = "idle"
        self.gesture_counter = 0
        self.last_switch_time = 0

        self._finger_states = [False] * 5
        self._active_hand = None
        self._pinching = False

        # smoothing buffer
        self._lm_history = []
        self._history_size = 3

    def _lm_dict(self, landmarks):
        lm = {}
        if landmarks is None:
            return lm
        for idx, x, y, z in landmarks:
            lm[idx] = (float(x), float(y), float(z))
        return lm

    def _smooth_landmarks(self, lm):
        self._lm_history.append(lm)
        if len(self._lm_history) > self._history_size:
            self._lm_history = self._lm_history[-self._history_size:]

        if len(self._lm_history) < 2:
            return lm

        smoothed = {}
        for key in lm:
            xs, ys, zs = [], [], []
            for hist in self._lm_history:
                if key in hist:
                    xs.append(hist[key][0])
                    ys.append(hist[key][1])
                    zs.append(hist[key][2])
            if xs:
                smoothed[key] = (sum(xs)/len(xs), sum(ys)/len(ys), sum(zs)/len(zs))
        return smoothed

    def _dist(self, a, b):
        return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

    def _hand_size(self, lm):
        if self.WRIST in lm and self.MIDDLE_MCP in lm:
            return self._dist(lm[self.WRIST], lm[self.MIDDLE_MCP])
        return 100.0

    def _angle_at(self, a, b, c):
        ba = (a[0]-b[0], a[1]-b[1])
        bc = (c[0]-b[0], c[1]-b[1])
        dot = ba[0]*bc[0] + ba[1]*bc[1]
        mag_ba = math.sqrt(ba[0]**2 + ba[1]**2)
        mag_bc = math.sqrt(bc[0]**2 + bc[1]**2)
        if mag_ba < 0.001 or mag_bc < 0.001:
            return 180.0
        cos_a = max(-1.0, min(1.0, dot/(mag_ba*mag_bc)))
        return math.degrees(math.acos(cos_a))

    def _is_finger_open(self, lm, mcp, pip, dip, tip, finger_idx):
        if not all(k in lm for k in (mcp, pip, dip, tip)):
            return self._finger_states[finger_idx]

        pip_angle = self._angle_at(lm[mcp], lm[pip], lm[dip])

        wrist = lm.get(self.WRIST)
        if wrist:
            tip_dist = self._dist(lm[tip], wrist)
            pip_dist = self._dist(lm[pip], wrist)
            tip_further = tip_dist > pip_dist * 0.85
        else:
            tip_further = True

        was_open = self._finger_states[finger_idx]

        if was_open:
            if pip_angle < 115 or (pip_angle < 135 and not tip_further):
                return False
            return True
        else:
            if pip_angle > 145 and tip_further:
                return True
            if pip_angle > 160:
                return True
            return False

    def _is_thumb_open(self, lm):
        needed = [self.THUMB_CMC, self.THUMB_MCP, self.THUMB_IP, self.THUMB_TIP, self.INDEX_MCP]
        if not all(k in lm for k in needed):
            return self._finger_states[0]

        mcp_angle = self._angle_at(lm[self.THUMB_CMC], lm[self.THUMB_MCP], lm[self.THUMB_IP])
        ip_angle = self._angle_at(lm[self.THUMB_MCP], lm[self.THUMB_IP], lm[self.THUMB_TIP])
        avg_angle = (mcp_angle + ip_angle) / 2.0

        hand_sz = self._hand_size(lm)
        thumb_spread = self._dist(lm[self.THUMB_TIP], lm[self.INDEX_MCP])
        spread_ratio = thumb_spread / hand_sz if hand_sz > 1 else 0

        was_open = self._finger_states[0]
        if was_open:
            if avg_angle < 100 or spread_ratio < 0.25:
                return False
            return True
        else:
            if avg_angle > 135 and spread_ratio > 0.4:
                return True
            if avg_angle > 150:
                return True
            return False

    def get_finger_states(self, landmarks, handedness="right"):
        if landmarks is None:
            return [False] * 5

        lm = self._lm_dict(landmarks)
        lm = self._smooth_landmarks(lm)

        fingers = list(self._finger_states)
        fingers[0] = self._is_thumb_open(lm)

        joints = [
            (self.INDEX_MCP, self.INDEX_PIP, self.INDEX_DIP, self.INDEX_TIP, 1),
            (self.MIDDLE_MCP, self.MIDDLE_PIP, self.MIDDLE_DIP, self.MIDDLE_TIP, 2),
            (self.RING_MCP, self.RING_PIP, self.RING_DIP, self.RING_TIP, 3),
            (self.PINKY_MCP, self.PINKY_PIP, self.PINKY_DIP, self.PINKY_TIP, 4),
        ]
        for mcp, pip, dip, tip, i in joints:
            fingers[i] = self._is_finger_open(lm, mcp, pip, dip, tip, i)

        self._finger_states = fingers
        return fingers

    def _get_pinch_ratio(self, lm):
        if self.THUMB_TIP not in lm or self.INDEX_TIP not in lm:
            return 999.0
        raw = self._dist(lm[self.THUMB_TIP], lm[self.INDEX_TIP])
        hsz = self._hand_size(lm)
        return raw / hsz if hsz > 1 else 999.0

    def recognize(self, hand_data):
        right = hand_data.get("right")
        left = hand_data.get("left")

        if self._active_hand == "right" and right is not None:
            landmarks, handedness = right, "right"
        elif self._active_hand == "left" and left is not None:
            landmarks, handedness = left, "left"
        elif right is not None:
            landmarks, handedness = right, "right"
            self._active_hand = "right"
        elif left is not None:
            landmarks, handedness = left, "left"
            self._active_hand = "left"
        else:
            self._active_hand = None
            self._finger_states = [False] * 5
            self._pinching = False
            self._lm_history.clear()
            return "idle", None, []

        fingers = self.get_finger_states(landmarks, handedness)
        thumb, index, middle, ring, pinky = fingers

        lm = self._lm_dict(landmarks)
        lm_smooth = self._lm_history[-1] if self._lm_history else lm

        tip_pos = None
        if self.INDEX_TIP in lm_smooth:
            tip_pos = (int(lm_smooth[self.INDEX_TIP][0]),
                       int(lm_smooth[self.INDEX_TIP][1]))

        palm_pos = None
        if self.WRIST in lm_smooth and self.MIDDLE_MCP in lm_smooth:
            w = lm_smooth[self.WRIST]
            m = lm_smooth[self.MIDDLE_MCP]
            palm_pos = (int((w[0]+m[0])/2), int((w[1]+m[1])/2))

        erase_points = []
        if palm_pos:
            erase_points.append(palm_pos)
        for tid in [self.THUMB_TIP, self.INDEX_TIP, self.MIDDLE_TIP,
                    self.RING_TIP, self.PINKY_TIP]:
            if tid in lm_smooth:
                erase_points.append((int(lm_smooth[tid][0]),
                                     int(lm_smooth[tid][1])))

        pinch_ratio = self._get_pinch_ratio(lm_smooth)
        up_count = sum([index, middle, ring, pinky])
        gesture = "idle"

        # pinch grab with hysteresis
        if self._pinching:
            if pinch_ratio > self.PINCH_RATIO_OFF:
                self._pinching = False
            else:
                gesture = "grab"
        else:
            if pinch_ratio < self.PINCH_RATIO_ON and up_count <= 1:
                self._pinching = True
                gesture = "grab"

        if gesture != "grab":
            if index and not middle and not ring and not pinky:
                gesture = "draw"
            elif up_count >= 3 and index and middle:
                gesture = "erase"
            elif index and middle and not ring and not pinky:
                gesture = "change_color"
            elif up_count == 0 and not thumb:
                gesture = "switch_brush"

        gesture = self._debounce(gesture)
        return gesture, tip_pos, erase_points

    def _debounce(self, gesture):
        now = time.time()

        if gesture == self.last_gesture:
            self.gesture_counter += 1
        else:
            self.gesture_counter = 1
            self.last_gesture = gesture

        if gesture in ("draw", "erase"):
            if self.gesture_counter >= 2:
                self.confirmed_gesture = gesture
            return self.confirmed_gesture

        if gesture == "grab":
            if self.gesture_counter >= 3:
                self.confirmed_gesture = gesture
            return self.confirmed_gesture

        if gesture == "idle":
            if self.gesture_counter >= 3:
                self.confirmed_gesture = gesture
            return self.confirmed_gesture

        if self.gesture_counter >= DEBOUNCE_FRAMES + 2:
            if now - self.last_switch_time >= GESTURE_COOLDOWN:
                self.last_switch_time = now
                self.confirmed_gesture = gesture

        return self.confirmed_gesture

    def _raw_finger_states(self, landmarks):
        if landmarks is None:
            return [False] * 5
        lm = self._lm_dict(landmarks)
        fingers = []

        needed = [self.THUMB_CMC, self.THUMB_MCP, self.THUMB_IP, self.THUMB_TIP]
        if all(k in lm for k in needed):
            a1 = self._angle_at(lm[self.THUMB_CMC], lm[self.THUMB_MCP], lm[self.THUMB_IP])
            a2 = self._angle_at(lm[self.THUMB_MCP], lm[self.THUMB_IP], lm[self.THUMB_TIP])
            fingers.append((a1+a2)/2 > 130)
        else:
            fingers.append(False)

        joints = [
            (self.INDEX_MCP, self.INDEX_PIP, self.INDEX_DIP, self.INDEX_TIP),
            (self.MIDDLE_MCP, self.MIDDLE_PIP, self.MIDDLE_DIP, self.MIDDLE_TIP),
            (self.RING_MCP, self.RING_PIP, self.RING_DIP, self.RING_TIP),
            (self.PINKY_MCP, self.PINKY_PIP, self.PINKY_DIP, self.PINKY_TIP),
        ]
        for mcp, pip, dip, tip in joints:
            if all(k in lm for k in (mcp, pip, dip, tip)):
                angle = self._angle_at(lm[mcp], lm[pip], lm[dip])
                fingers.append(angle > 140)
            else:
                fingers.append(False)
        return fingers

    def recognize_multi(self, hand_data):
        left = hand_data.get("left")
        right = hand_data.get("right")
        if left is None or right is None:
            return None

        lf = self._raw_finger_states(left)
        rf = self._raw_finger_states(right)

        if not any(lf) and not any(rf):
            return "clear_canvas"
        if all(lf[1:]) and all(rf[1:]):
            return "pause"
        return None

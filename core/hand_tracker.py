import mediapipe as mp
import cv2
from app.config import MAX_HANDS, DETECTION_CONFIDENCE, TRACKING_CONFIDENCE


class HandTracker:
    def __init__(self, max_hands=MAX_HANDS, det_conf=DETECTION_CONFIDENCE, track_conf=TRACKING_CONFIDENCE):
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            min_detection_confidence=det_conf,
            min_tracking_confidence=track_conf
        )
        # cache the last results so we dont process the frame twice
        self._last_results = None

    def find_hands(self, frame):
        """
        process a frame and return structured hand data.
        returns a dict like {"left": [...], "right": [...]} where each
        value is a list of (id, x, y, z) tuples for the 21 landmarks.
        if a hand isnt detected, its value is None.
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self._last_results = self.hands.process(rgb)

        hand_data = {"left": None, "right": None}

        if not self._last_results.multi_hand_landmarks:
            return hand_data

        h, w, _ = frame.shape

        for hand_landmarks, handedness in zip(self._last_results.multi_hand_landmarks, self._last_results.multi_handedness):
            # mediapipe labels are from cameras perspective, so we flip
            label = handedness.classification[0].label
            # since we mirror the frame, left/right swap
            side = "left" if label == "Right" else "right"

            landmarks = []
            for idx, lm in enumerate(hand_landmarks.landmark):
                px, py = int(lm.x * w), int(lm.y * h)
                landmarks.append((idx, px, py, lm.z))

            hand_data[side] = landmarks

        return hand_data

    def draw_landmarks(self, frame, hand_data):
        """draw the hand skeleton on frame - reuses cached results"""
        if self._last_results and self._last_results.multi_hand_landmarks:
            for hand_lm in self._last_results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(
                    frame, hand_lm, self.mp_hands.HAND_CONNECTIONS
                )
        return frame

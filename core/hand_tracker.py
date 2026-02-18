import mediapipe as mp
import cv2
import os
import time
from app.config import MAX_HANDS, DETECTION_CONFIDENCE, TRACKING_CONFIDENCE, DETECTION_WIDTH, PROCESS_EVERY_N_FRAMES

# new tasks API
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode
HandConnections = mp.tasks.vision.HandLandmarksConnections

# figure out where the model file is (same dir as project root)
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "hand_landmarker.task")


class HandTracker:
    def __init__(self, max_hands=MAX_HANDS, det_conf=DETECTION_CONFIDENCE, track_conf=TRACKING_CONFIDENCE):
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"cant find {MODEL_PATH} - download it from "
                "https://storage.googleapis.com/mediapipe-models/"
                "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
            )

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=RunningMode.VIDEO,
            num_hands=max_hands,
            min_hand_detection_confidence=det_conf,
            min_tracking_confidence=track_conf,
        )
        self.landmarker = HandLandmarker.create_from_options(options)
        self._start_time = time.monotonic()
        self._last_result = None
        self._last_landmarks_raw = []
        self._last_hand_data = {"left": None, "right": None}
        self._frame_count = 0
        self._skip_n = PROCESS_EVERY_N_FRAMES
        self._det_width = DETECTION_WIDTH

    def find_hands(self, frame):
        """
        process a frame and return structured hand data.
        skips detection on some frames and reuses last result for speed.
        """
        self._frame_count += 1

        # only skip frames if we already have hands tracked and skip_n > 1
        has_data = self._last_hand_data["left"] is not None or self._last_hand_data["right"] is not None
        if self._skip_n > 1 and has_data and (self._frame_count % self._skip_n != 0):
            return self._last_hand_data

        h, w, _ = frame.shape

        # downscale for detection - way faster
        scale = self._det_width / w
        small = cv2.resize(frame, (self._det_width, int(h * scale)))
        rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_small)

        # use real time so mediapipe tracking doesnt drift
        ts_ms = int((time.monotonic() - self._start_time) * 1000)
        try:
            result = self.landmarker.detect_for_video(mp_image, ts_ms)
        except Exception as e:
            # mediapipe can occasionally choke on weird frames
            # just return last good data instead of crashing
            return self._last_hand_data
        self._last_result = result

        hand_data = {"left": None, "right": None}

        if not result.hand_landmarks:
            self._last_landmarks_raw = []
            self._last_hand_data = hand_data
            return hand_data

        self._last_landmarks_raw = result.hand_landmarks

        # landmarks are normalized 0-1, so we scale back to original frame size
        for hand_lms, handedness_list in zip(result.hand_landmarks, result.handedness):
            label = handedness_list[0].category_name
            side = "left" if label == "Right" else "right"

            landmarks = []
            for idx, lm in enumerate(hand_lms):
                px, py = int(lm.x * w), int(lm.y * h)
                landmarks.append((idx, px, py, lm.z))

            hand_data[side] = landmarks

        self._last_hand_data = hand_data
        return hand_data

    def draw_landmarks(self, frame, hand_data):
        """draw the hand skeleton on frame"""
        if not self._last_landmarks_raw:
            return frame

        h, w, _ = frame.shape
        connections = HandConnections.HAND_CONNECTIONS

        for hand_lms in self._last_landmarks_raw:
            # draw connections
            for connection in connections:
                start = hand_lms[connection.start]
                end = hand_lms[connection.end]
                x1, y1 = int(start.x * w), int(start.y * h)
                x2, y2 = int(end.x * w), int(end.y * h)
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # draw landmark dots
            for lm in hand_lms:
                px, py = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (px, py), 3, (0, 0, 255), -1)

        return frame

    def close(self):
        self.landmarker.close()

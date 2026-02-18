import cv2
import time
from app.config import CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_INDEX


class Camera:
    def __init__(self, src=CAMERA_INDEX, width=CAMERA_WIDTH, height=CAMERA_HEIGHT):
        self.cap = None
        self._src = src
        self.width = width
        self.height = height
        self._consecutive_fails = 0
        self._max_fails = 30  # give up after this many bad frames in a row

        # try a few times in case the camera is slow to init
        for attempt in range(3):
            self.cap = cv2.VideoCapture(src)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            if self.cap.isOpened():
                break
            print(f"camera not ready, retrying ({attempt + 1}/3)...")
            time.sleep(1)

        if self.cap is None or not self.cap.isOpened():
            raise RuntimeError(
                "couldnt open webcam. check that:\n"
                "  - your camera is plugged in\n"
                "  - no other app is using it\n"
                "  - you have permission (try: ls /dev/video*)\n"
                f"  - camera index {src} is correct (try 0, 1, or 2)"
            )

    def read(self):
        """grab a frame, flip it so its mirror-like, return success + frame"""
        if self.cap is None or not self.cap.isOpened():
            return False, None

        ret, frame = self.cap.read()
        if not ret or frame is None:
            self._consecutive_fails += 1
            if self._consecutive_fails >= self._max_fails:
                print("camera stopped responding after multiple failed reads")
            return False, None

        self._consecutive_fails = 0
        frame = cv2.flip(frame, 1)  # mirror
        return True, frame

    def release(self):
        if self.cap is not None:
            self.cap.release()

    def __del__(self):
        self.release()

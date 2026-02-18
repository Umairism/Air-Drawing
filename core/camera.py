import cv2
from app.config import CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_INDEX


class Camera:
    def __init__(self, src=CAMERA_INDEX, width=CAMERA_WIDTH, height=CAMERA_HEIGHT):
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        if not self.cap.isOpened():
            raise RuntimeError("couldnt open webcam, check if its connected")

        self.width = width
        self.height = height

    def read(self):
        """grab a frame, flip it so its mirror-like, return success + frame"""
        ret, frame = self.cap.read()
        if not ret:
            return False, None

        frame = cv2.flip(frame, 1)  # mirror
        return True, frame

    def release(self):
        self.cap.release()

    def __del__(self):
        self.release()

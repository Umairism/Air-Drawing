import os
import pickle
import numpy as np


class GesturePredictor:
    """
    loads a trained model and predicts gestures from landmarks in real time.
    plug this into the main loop as an alternative to the rule-based engine.
    """

    def __init__(self, model_path="gesture_model.pkl"):
        self.model = None
        self.centroids = None
        self.mins = None
        self.ranges = None

        if os.path.exists(model_path):
            self.load(model_path)

    def load(self, path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.centroids = data["centroids"]
        self.mins = data["mins"]
        self.ranges = data["ranges"]
        self.model = data
        print(f"loaded model with classes: {list(self.centroids.keys())}")

    def predict(self, landmarks):
        """
        takes a list of (idx, x, y, z) landmarks and returns
        the predicted gesture label.
        """
        if self.centroids is None or landmarks is None:
            return "idle"

        # flatten to feature vector
        features = []
        for idx, x, y, z in landmarks:
            features.extend([x, y, z])

        features = np.array(features, dtype=float)

        # normalize using the same params from training
        features = (features - self.mins) / self.ranges

        # nearest centroid
        best_cls = "idle"
        best_dist = float('inf')

        for cls, centroid in self.centroids.items():
            dist = np.linalg.norm(features - centroid)
            if dist < best_dist:
                best_dist = dist
                best_cls = cls

        return best_cls

    @property
    def is_loaded(self):
        return self.centroids is not None

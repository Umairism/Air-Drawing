"""
real-time gesture prediction using a trained model.

two modes of operation:
  - standalone: ML model makes all decisions
  - hybrid: ML model runs alongside the rule-based engine,
            and we take the ML prediction only when its confidence
            is high enough. otherwise fall back to rules.

the hybrid approach gives you the best of both worlds:
  - rules handle the common gestures reliably
  - ML handles edge cases and custom gestures that rules cant express
"""
import os
import pickle
import numpy as np
from ml.trainer import extract_features


class GesturePredictor:
    """
    loads a trained gesture model and classifies hand landmarks in real time.

    usage:
        predictor = GesturePredictor()
        if predictor.is_loaded:
            gesture, confidence = predictor.predict(landmarks)
    """

    def __init__(self, model_path="gesture_model.pkl"):
        self.classifier = None
        self.scaler_mean = None
        self.scaler_scale = None
        self.classes = []
        self.method = "unknown"
        self._loaded = False

        if os.path.exists(model_path):
            self._load(model_path)

    def _load(self, path):
        """load a model saved by GestureTrainer"""
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)

            self.classifier = data["classifier"]
            self.scaler_mean = np.array(data["scaler_mean"])
            self.scaler_scale = np.array(data["scaler_scale"])
            self.classes = data["classes"]
            self.method = data.get("method", "unknown")
            self._loaded = True

            print(f"loaded {self.method} model with classes: {self.classes}")
        except Exception as e:
            print(f"failed to load gesture model: {e}")
            self._loaded = False

    @property
    def is_loaded(self):
        return self._loaded

    def predict(self, landmarks):
        """
        predict the gesture from raw landmarks.

        returns (gesture_name, confidence) where confidence is 0.0-1.0.
        confidence tells you how sure the model is — use this to decide
        whether to trust the ML prediction or fall back to rules.
        """
        if not self._loaded or landmarks is None:
            return "idle", 0.0

        features = extract_features(landmarks)
        if features is None:
            return "idle", 0.0

        features = np.array(features, dtype=np.float64).reshape(1, -1)

        # apply the same scaling that was used during training
        features = (features - self.scaler_mean) / self.scaler_scale

        # predict
        prediction = self.classifier.predict(features)[0]

        # get confidence if the classifier supports it
        confidence = 0.0
        if hasattr(self.classifier, "predict_proba"):
            proba = self.classifier.predict_proba(features)[0]
            confidence = float(max(proba))
        elif hasattr(self.classifier, "decision_function"):
            # SVM without probability=True
            confidence = 0.8  # reasonable default

        return prediction, confidence

    def predict_top_n(self, landmarks, n=3):
        """
        return top N predictions with their probabilities.
        useful for debugging which gestures are getting confused.
        """
        if not self._loaded or landmarks is None:
            return [("idle", 0.0)]

        features = extract_features(landmarks)
        if features is None:
            return [("idle", 0.0)]

        features = np.array(features, dtype=np.float64).reshape(1, -1)
        features = (features - self.scaler_mean) / self.scaler_scale

        if hasattr(self.classifier, "predict_proba"):
            proba = self.classifier.predict_proba(features)[0]
            pairs = list(zip(self.classes, proba))
            pairs.sort(key=lambda x: x[1], reverse=True)
            return pairs[:n]

        # fallback: just return the single prediction
        prediction = self.classifier.predict(features)[0]
        return [(prediction, 0.8)]

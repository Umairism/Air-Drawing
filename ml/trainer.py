"""
trains a gesture classifier from collected landmark data.

two classifiers available:
  - SVM with RBF kernel (default, best accuracy)
  - k-NN (fast, decent fallback)

the feature vector isn't raw pixel coordinates — those change depending
on where your hand is in the frame. instead we extract hand-relative
features: joint angles, fingertip-to-wrist distance ratios, and
inter-finger spreads. these are scale/position invariant, which is
what makes the model actually generalize.
"""
import csv
import os
import json
import pickle
import numpy as np
import math
from datetime import datetime


def _dist(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def _angle_at(a, b, c):
    """angle at point b in degrees"""
    ba = (a[0] - b[0], a[1] - b[1])
    bc = (c[0] - b[0], c[1] - b[1])
    dot = ba[0] * bc[0] + ba[1] * bc[1]
    mag_ba = math.sqrt(ba[0] ** 2 + ba[1] ** 2)
    mag_bc = math.sqrt(bc[0] ** 2 + bc[1] ** 2)
    if mag_ba < 0.001 or mag_bc < 0.001:
        return 180.0
    cos_a = max(-1.0, min(1.0, dot / (mag_ba * mag_bc)))
    return math.degrees(math.acos(cos_a))


def _rotate_to_local(lm):
    """
    rotate all landmarks into a hand-local coordinate frame so the
    features don't change when the hand is tilted.

    the local frame is defined by:
      - origin at wrist (landmark 0)
      - y-axis pointing from wrist toward middle finger MCP (landmark 9)

    this makes the feature vector rotation-invariant: a hand tilted
    30 degrees produces the same features as an upright hand.
    """
    wrist = lm[0]
    mid_mcp = lm[9]

    # direction vector from wrist to middle MCP (defines "up" for this hand)
    dx = mid_mcp[0] - wrist[0]
    dy = mid_mcp[1] - wrist[1]
    length = math.sqrt(dx * dx + dy * dy)
    if length < 0.001:
        return lm  # degenerate, skip rotation

    # unit vectors for the local frame
    uy_x = dx / length
    uy_y = dy / length
    ux_x = -uy_y  # perpendicular
    ux_y = uy_x

    # project every landmark into the local frame
    rotated = {}
    for idx, (x, y, z) in lm.items():
        rx = x - wrist[0]
        ry = y - wrist[1]
        local_x = rx * ux_x + ry * ux_y
        local_y = rx * uy_x + ry * uy_y
        rotated[idx] = (local_x, local_y, z)

    return rotated


def extract_features(landmarks):
    """
    turn 21 raw landmarks into a feature vector that's invariant to
    hand position, scale, and rotation.

    features (20 total):
      - 5 PIP joint angles (one per finger, thumb uses MCP+IP avg)
      - 5 fingertip-to-wrist distance ratios (normalized by hand size)
      - 5 fingertip-to-palm distance ratios
      - 4 inter-finger spread angles (index-middle, middle-ring, etc.)
      - 1 thumb-index pinch ratio

    all landmarks are rotated into a hand-local coordinate frame before
    computing distance ratios and spread angles. this means a tilted
    hand produces the same features as an upright one.

    all distances are divided by hand_size (wrist to middle MCP) so
    the features don't depend on how far the hand is from the camera.
    """
    # build lookup dict
    lm = {}
    if landmarks is None:
        return None

    for item in landmarks:
        if len(item) == 4:
            idx, x, y, z = item
        elif len(item) == 3:
            idx, x, y = item
            z = 0.0
        else:
            continue
        lm[idx] = (float(x), float(y), float(z))

    if len(lm) < 21:
        return None

    # rotate into hand-local coordinate frame for rotation invariance.
    # joint angles are already rotation-invariant, but distance ratios
    # and spread angles are not — rotating first fixes that.
    lm = _rotate_to_local(lm)

    # hand size for normalization (in the local frame, wrist is at origin)
    wrist = lm[0]
    mid_mcp = lm[9]
    hand_size = _dist(wrist, mid_mcp)
    if hand_size < 1.0:
        hand_size = 1.0

    # palm center (average of wrist + 4 MCPs)
    palm_pts = [lm[0], lm[5], lm[9], lm[13], lm[17]]
    palm = (
        sum(p[0] for p in palm_pts) / 5,
        sum(p[1] for p in palm_pts) / 5,
    )

    features = []

    # --- finger PIP angles (5) ---
    # thumb: average of MCP and IP angles
    thumb_a1 = _angle_at(lm[1], lm[2], lm[3])
    thumb_a2 = _angle_at(lm[2], lm[3], lm[4])
    features.append((thumb_a1 + thumb_a2) / 2.0 / 180.0)  # normalize to 0-1

    finger_joints = [
        (5, 6, 7, 8),      # index
        (9, 10, 11, 12),    # middle
        (13, 14, 15, 16),   # ring
        (17, 18, 19, 20),   # pinky
    ]
    for mcp, pip, dip, tip in finger_joints:
        angle = _angle_at(lm[mcp], lm[pip], lm[dip])
        features.append(angle / 180.0)

    # --- fingertip to wrist ratios (5) ---
    tips = [4, 8, 12, 16, 20]
    for tip_id in tips:
        d = _dist(lm[tip_id], wrist)
        features.append(d / hand_size)

    # --- fingertip to palm ratios (5) ---
    for tip_id in tips:
        d = _dist(lm[tip_id], palm)
        features.append(d / hand_size)

    # --- inter-finger spread angles (4) ---
    tip_pairs = [(8, 12), (12, 16), (16, 20), (4, 8)]
    for t1, t2 in tip_pairs:
        angle = _angle_at(lm[t1], palm, lm[t2])
        features.append(angle / 180.0)

    # --- pinch ratio (1) ---
    pinch_dist = _dist(lm[4], lm[8])
    features.append(pinch_dist / hand_size)

    return features


class GestureTrainer:
    """
    trains a gesture classifier from collected CSV data.

    the CSV format is: x0,y0,z0,x1,y1,z1,...,x20,y20,z20,label
    (63 coordinate columns + 1 label column, same as dataset_collector outputs)
    """

    GESTURES = ["draw", "erase", "change_color", "switch_brush", "grab", "idle"]

    def __init__(self, data_file="gesture_data.csv"):
        self.data_file = data_file
        self.model = None
        self.scaler = None

    def load_data(self):
        """load CSV, extract features, return X and y arrays"""
        if not os.path.exists(self.data_file):
            print(f"no data file found at {self.data_file}")
            print("run the dataset collector first: python -m ml.dataset_collector")
            return None, None

        raw_features = []
        labels = []
        skipped = 0

        with open(self.data_file, "r") as f:
            reader = csv.reader(f)
            next(reader)  # skip header

            for row in reader:
                label = row[-1]
                coords = [float(x) for x in row[:-1]]

                # rebuild into landmark format
                landmarks = []
                for i in range(21):
                    x, y, z = coords[i * 3], coords[i * 3 + 1], coords[i * 3 + 2]
                    landmarks.append((i, x, y, z))

                feat = extract_features(landmarks)
                if feat is None:
                    skipped += 1
                    continue

                raw_features.append(feat)
                labels.append(label)

        if skipped > 0:
            print(f"skipped {skipped} incomplete samples")

        X = np.array(raw_features, dtype=np.float64)
        y = np.array(labels)

        print(f"loaded {len(X)} samples ({len(set(y))} classes)")
        for cls in sorted(set(y)):
            print(f"  {cls}: {sum(y == cls)} samples")

        return X, y

    def train(self, test_split=0.2, method="svm"):
        """
        train a classifier and print evaluation metrics.

        method: "svm" or "knn"
        """
        X, y = self.load_data()
        if X is None or len(X) < 10:
            print("not enough data to train")
            return None

        from sklearn.model_selection import train_test_split, cross_val_score
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import classification_report

        # scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # split
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=test_split, random_state=42, stratify=y
        )

        # pick classifier
        if method == "svm":
            from sklearn.svm import SVC
            clf = SVC(kernel="rbf", C=10.0, gamma="scale", probability=True)
        elif method == "knn":
            from sklearn.neighbors import KNeighborsClassifier
            clf = KNeighborsClassifier(n_neighbors=5, weights="distance")
        else:
            print(f"unknown method: {method}")
            return None

        # train
        clf.fit(X_train, y_train)

        # evaluate
        y_pred = clf.predict(X_test)
        accuracy = np.mean(y_pred == y_test)

        print(f"\n{'=' * 50}")
        print(f"  Method: {method.upper()}")
        print(f"  Accuracy: {accuracy:.1%}")
        print(f"  Train: {len(X_train)}, Test: {len(X_test)}")
        print(f"{'=' * 50}")
        print(classification_report(y_test, y_pred, zero_division=0))

        # cross-validation for a more honest estimate
        cv_scores = cross_val_score(clf, X_scaled, y, cv=5)
        print(f"5-fold CV accuracy: {cv_scores.mean():.1%} (+/- {cv_scores.std() * 2:.1%})")

        # confusion matrix — the real test of whether gestures bleed into each other
        from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

        cm = confusion_matrix(y_test, y_pred, labels=sorted(set(y)))
        precision, recall, f1, support = precision_recall_fscore_support(
            y_test, y_pred, labels=sorted(set(y)), zero_division=0
        )

        class_labels = sorted(set(y))
        print(f"\n  CONFUSION MATRIX")
        print(f"  {'':>15}", end="")
        for lbl in class_labels:
            print(f"  {lbl[:8]:>8}", end="")
        print()
        for i, row_label in enumerate(class_labels):
            print(f"  {row_label:>15}", end="")
            for j in range(len(class_labels)):
                val = cm[i][j]
                marker = f"  {val:>8}" if i != j or val == 0 else f"  {val:>7}*"
                print(marker, end="")
            print(f"    P={precision[i]:.2f}  R={recall[i]:.2f}  F1={f1[i]:.2f}")
        print()

        # per-class metrics dict for saving
        per_class = {}
        for i, lbl in enumerate(class_labels):
            per_class[lbl] = {
                "precision": round(float(precision[i]), 4),
                "recall": round(float(recall[i]), 4),
                "f1": round(float(f1[i]), 4),
                "support": int(support[i]),
            }

        # save everything
        self.model = {
            "classifier": clf,
            "scaler_mean": self.scaler.mean_.tolist(),
            "scaler_scale": self.scaler.scale_.tolist(),
            "classes": list(clf.classes_),
            "method": method,
            "accuracy": float(accuracy),
            "cv_accuracy": float(cv_scores.mean()),
            "cv_std": float(cv_scores.std()),
            "per_class_metrics": per_class,
            "confusion_matrix": cm.tolist(),
            "confusion_labels": class_labels,
            "n_features": X.shape[1],
            "n_samples": len(X),
            "trained_at": datetime.now().isoformat(),
        }

        return accuracy

    def save_model(self, path="gesture_model.pkl"):
        """save trained model + scaler to a pickle file"""
        if self.model is None:
            print("nothing to save, train first")
            return

        with open(path, "wb") as f:
            pickle.dump(self.model, f)
        print(f"model saved to {path}")

        # also save a human-readable metadata file
        meta = {k: v for k, v in self.model.items()
                if k not in ("classifier",)}
        meta_path = path.replace(".pkl", "_meta.json")
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        print(f"metadata saved to {meta_path}")


if __name__ == "__main__":
    import sys

    data_file = sys.argv[1] if len(sys.argv) > 1 else "gesture_data.csv"
    method = sys.argv[2] if len(sys.argv) > 2 else "svm"

    trainer = GestureTrainer(data_file)
    acc = trainer.train(method=method)
    if acc is not None:
        trainer.save_model()

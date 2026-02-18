import csv
import os
import numpy as np
import pickle
from collections import Counter


class GestureTrainer:
    """
    trains a simple classifier on collected gesture data.
    uses a basic approach - no heavy frameworks needed.
    """

    def __init__(self, data_file="gesture_data.csv"):
        self.data_file = data_file
        self.model = None

    def load_data(self):
        """load csv and split into features + labels"""
        if not os.path.exists(self.data_file):
            print(f"no data file found at {self.data_file}")
            return None, None

        features = []
        labels = []

        with open(self.data_file, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)  # skip header
            for row in reader:
                features.append([float(x) for x in row[:-1]])
                labels.append(row[-1])

        X = np.array(features)
        y = np.array(labels)

        print(f"loaded {len(X)} samples")
        print(f"label distribution: {Counter(y)}")

        return X, y

    def normalize(self, X):
        """simple min-max normalization"""
        mins = X.min(axis=0)
        maxs = X.max(axis=0)
        ranges = maxs - mins
        ranges[ranges == 0] = 1  # avoid division by zero
        return (X - mins) / ranges, mins, ranges

    def train(self, test_split=0.2):
        """
        train a nearest-centroid classifier. its simple but works
        surprisingly well for this kind of data.
        """
        X, y = self.load_data()
        if X is None:
            return

        X_norm, mins, ranges = self.normalize(X)

        # shuffle and split
        indices = np.arange(len(X_norm))
        np.random.shuffle(indices)
        split = int(len(indices) * (1 - test_split))

        X_train, X_test = X_norm[indices[:split]], X_norm[indices[split:]]
        y_train, y_test = y[indices[:split]], y[indices[split:]]

        # compute centroids for each class
        classes = np.unique(y_train)
        centroids = {}
        for cls in classes:
            mask = y_train == cls
            centroids[cls] = X_train[mask].mean(axis=0)

        # test accuracy
        correct = 0
        for xi, yi in zip(X_test, y_test):
            # find nearest centroid
            best_cls = None
            best_dist = float('inf')
            for cls, centroid in centroids.items():
                dist = np.linalg.norm(xi - centroid)
                if dist < best_dist:
                    best_dist = dist
                    best_cls = cls
            if best_cls == yi:
                correct += 1

        accuracy = correct / len(y_test) if len(y_test) > 0 else 0
        print(f"accuracy: {accuracy:.2%} on {len(y_test)} test samples")

        # save model
        self.model = {
            "centroids": centroids,
            "mins": mins,
            "ranges": ranges,
        }
        self.save_model()

        return accuracy

    def save_model(self, path="gesture_model.pkl"):
        if self.model:
            with open(path, 'wb') as f:
                pickle.dump(self.model, f)
            print(f"model saved to {path}")

    def load_model(self, path="gesture_model.pkl"):
        if os.path.exists(path):
            with open(path, 'rb') as f:
                self.model = pickle.load(f)
            print("model loaded")
        else:
            print(f"no model found at {path}")


if __name__ == "__main__":
    trainer = GestureTrainer()
    trainer.train()

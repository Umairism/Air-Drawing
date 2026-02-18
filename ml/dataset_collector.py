import csv
import os
import cv2
from core.camera import Camera
from core.hand_tracker import HandTracker


class DatasetCollector:
    """
    collects hand landmark data and saves it to CSV for training.
    run this as a standalone script to build up your gesture dataset.
    """

    def __init__(self, output_file="gesture_data.csv"):
        self.output_file = output_file
        self.cam = Camera()
        self.tracker = HandTracker()

        # create csv with headers if it doesnt exist
        if not os.path.exists(output_file):
            with open(output_file, 'w', newline='') as f:
                writer = csv.writer(f)
                # 21 landmarks * 3 coords (x, y, z) + label
                header = []
                for i in range(21):
                    header.extend([f"x{i}", f"y{i}", f"z{i}"])
                header.append("label")
                writer.writerow(header)

    def collect(self, label, num_samples=100):
        """collect num_samples frames of landmark data for a given gesture label"""
        print(f"collecting '{label}' - do the gesture and press 's' to start")

        count = 0
        recording = False

        while count < num_samples:
            ok, frame = self.cam.read()
            if not ok:
                break

            hand_data = self.tracker.find_hands(frame)
            landmarks = hand_data.get("right") or hand_data.get("left")

            status = f"[{count}/{num_samples}] {'RECORDING' if recording else 'WAITING'}"
            cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Label: {label}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.imshow("Dataset Collector", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                recording = True
            elif key == ord('q'):
                break

            if recording and landmarks is not None:
                self._save_sample(landmarks, label)
                count += 1

        cv2.destroyAllWindows()
        print(f"done! collected {count} samples for '{label}'")

    def _save_sample(self, landmarks, label):
        """flatten landmarks and write one row to csv"""
        row = []
        for idx, x, y, z in landmarks:
            row.extend([x, y, z])
        row.append(label)

        with open(self.output_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)


if __name__ == "__main__":
    collector = DatasetCollector()

    gestures = ["draw", "erase", "change_color", "switch_brush", "idle"]
    print("gesture dataset collector")
    print("available gestures:", gestures)

    while True:
        label = input("\nenter gesture name (or 'quit'): ").strip()
        if label == 'quit':
            break
        if label not in gestures:
            print(f"unknown gesture. pick from: {gestures}")
            continue

        samples = input("how many samples? [100]: ").strip()
        samples = int(samples) if samples else 100

        collector.collect(label, samples)

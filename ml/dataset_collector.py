"""
collects hand landmark data and saves it to CSV for training.

run standalone:
    python -m ml.dataset_collector

it opens your webcam, you do a gesture, and it records the landmark
positions to a CSV file. each row is one frame of data.

supports both the default gestures and custom user-defined ones.
"""
import csv
import os
import cv2
from core.camera import Camera
from core.hand_tracker import HandTracker


DEFAULT_GESTURES = ["draw", "erase", "change_color", "switch_brush", "grab", "idle"]


class DatasetCollector:
    """
    collects hand landmark data and saves it to CSV for training.
    """

    def __init__(self, output_file="gesture_data.csv"):
        self.output_file = output_file
        self.cam = Camera()
        self.tracker = HandTracker()

        # create csv with headers if it doesnt exist
        if not os.path.exists(output_file):
            with open(output_file, "w", newline="") as f:
                writer = csv.writer(f)
                header = []
                for i in range(21):
                    header.extend([f"x{i}", f"y{i}", f"z{i}"])
                header.append("label")
                writer.writerow(header)

    def collect(self, label, num_samples=100):
        """collect num_samples frames of landmark data for a given gesture label"""
        print(f"\ncollecting '{label}' — hold the gesture and press 's' to start recording")
        print(f"press 'q' to stop early\n")

        count = 0
        recording = False

        while count < num_samples:
            ok, frame = self.cam.read()
            if not ok:
                break

            hand_data = self.tracker.find_hands(frame)
            landmarks = hand_data.get("right") or hand_data.get("left")

            # draw hand skeleton
            frame = self.tracker.draw_landmarks(frame, hand_data)

            # status overlay
            status = "RECORDING" if recording else "WAITING (press 's')"
            color = (0, 0, 255) if recording else (0, 200, 255)
            cv2.putText(frame, f"[{count}/{num_samples}] {status}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.putText(frame, f"Gesture: {label}",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            if landmarks is None:
                cv2.putText(frame, "No hand detected",
                            (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 200), 1)

            # progress bar
            if num_samples > 0:
                prog = int(600 * count / num_samples)
                cv2.rectangle(frame, (10, 440), (10 + prog, 450), (0, 255, 0), -1)
                cv2.rectangle(frame, (10, 440), (610, 450), (100, 100, 100), 1)

            cv2.imshow("Dataset Collector", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("s") and not recording:
                recording = True
                print("recording started...")
            elif key == ord("q"):
                break

            if recording and landmarks is not None:
                self._save_sample(landmarks, label)
                count += 1

        cv2.destroyAllWindows()
        print(f"done — collected {count} samples for '{label}'")
        return count

    def _save_sample(self, landmarks, label):
        """flatten landmarks and write one row to csv"""
        row = []
        for idx, x, y, z in landmarks:
            row.extend([x, y, z])
        row.append(label)

        with open(self.output_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(row)

    def get_sample_count(self):
        """count existing samples per label"""
        if not os.path.exists(self.output_file):
            return {}

        counts = {}
        with open(self.output_file, "r") as f:
            reader = csv.reader(f)
            next(reader)  # skip header
            for row in reader:
                label = row[-1]
                counts[label] = counts.get(label, 0) + 1
        return counts


if __name__ == "__main__":
    collector = DatasetCollector()

    print("gesture dataset collector")
    print("=" * 40)

    # show existing data
    counts = collector.get_sample_count()
    if counts:
        print("\nexisting data:")
        for label, n in sorted(counts.items()):
            print(f"  {label}: {n} samples")
    else:
        print("\nno existing data yet")

    print(f"\ndefault gestures: {', '.join(DEFAULT_GESTURES)}")
    print("you can also type any custom gesture name\n")

    while True:
        label = input("enter gesture name (or 'quit'): ").strip().lower()
        if label in ("quit", "q", "exit"):
            break
        if not label:
            continue

        samples = input("how many samples? [100]: ").strip()
        samples = int(samples) if samples.isdigit() else 100

        collector.collect(label, samples)

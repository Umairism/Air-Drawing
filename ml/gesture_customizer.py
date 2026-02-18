"""
gesture customization framework.

lets users define, record, train, and use their own gestures
without touching any code. the workflow:

  1. record: hold a pose, record N frames of landmarks
  2. label: give it a name (e.g. "thumbs_up", "rock_on")
  3. train: retrain the classifier with the new data mixed in
  4. use: the hybrid engine picks up the new gesture automatically

custom gestures are stored in a JSON registry so they persist
across sessions. the actual landmark data goes into the same
CSV that the standard gestures use.
"""
import os
import json
import csv
from datetime import datetime
from ml.trainer import extract_features, GestureTrainer


REGISTRY_PATH = "custom_gestures.json"


class GestureRegistry:
    """
    keeps track of user-defined custom gestures.

    the registry is a JSON file mapping gesture names to metadata:
    {
        "thumbs_up": {
            "description": "thumb up, others curled",
            "samples": 150,
            "created": "2026-02-18T14:30:00",
            "action": "save_drawing"
        }
    }

    'action' is what happens when the gesture is detected. built-in
    actions: "save_drawing", "clear_canvas", "toggle_pause", "undo",
    or a custom callback name that the main loop can handle.
    """

    BUILT_IN_ACTIONS = [
        "save_drawing",
        "clear_canvas",
        "toggle_pause",
        "undo",
        "screenshot",
        "none",
    ]

    def __init__(self, path=REGISTRY_PATH):
        self.path = path
        self.gestures = {}
        self._load()

    def _load(self):
        if os.path.exists(self.path):
            try:
                with open(self.path, "r") as f:
                    self.gestures = json.load(f)
            except (json.JSONDecodeError, IOError):
                self.gestures = {}

    def _save(self):
        with open(self.path, "w") as f:
            json.dump(self.gestures, f, indent=2)

    def register(self, name, description="", action="none", samples=0):
        """add or update a custom gesture in the registry"""
        name = name.strip().lower().replace(" ", "_")
        self.gestures[name] = {
            "description": description,
            "samples": samples,
            "created": datetime.now().isoformat(),
            "action": action,
        }
        self._save()
        return name

    def remove(self, name):
        """remove a gesture from the registry"""
        if name in self.gestures:
            del self.gestures[name]
            self._save()
            return True
        return False

    def get_action(self, gesture_name):
        """look up what action a custom gesture should trigger"""
        if gesture_name in self.gestures:
            return self.gestures[gesture_name].get("action", "none")
        return None

    def list_gestures(self):
        """return list of (name, description, action, samples) tuples"""
        result = []
        for name, meta in sorted(self.gestures.items()):
            result.append((
                name,
                meta.get("description", ""),
                meta.get("action", "none"),
                meta.get("samples", 0),
            ))
        return result

    @property
    def names(self):
        return list(self.gestures.keys())


class GestureRecorder:
    """
    records landmark data for a custom gesture and retrains the model.

    typical flow:
        recorder = GestureRecorder()
        recorder.record_gesture("thumbs_up", description="thumb extended upward")
        recorder.retrain()
    """

    def __init__(self, data_file="gesture_data.csv", model_path="gesture_model.pkl"):
        self.data_file = data_file
        self.model_path = model_path
        self.registry = GestureRegistry()

    def record_gesture(self, name, description="", action="none", num_samples=100):
        """
        open webcam and record landmark data for a named gesture.
        returns the number of samples actually collected.
        """
        from ml.dataset_collector import DatasetCollector

        name = name.strip().lower().replace(" ", "_")
        collector = DatasetCollector(output_file=self.data_file)

        print(f"\nrecording custom gesture: '{name}'")
        if description:
            print(f"description: {description}")
        print(f"target: {num_samples} samples\n")

        count = collector.collect(name, num_samples)

        if count > 0:
            self.registry.register(name, description, action, count)
            print(f"\nregistered '{name}' with {count} samples")
        else:
            print(f"\nno samples collected for '{name}'")

        return count

    def retrain(self, method="svm"):
        """retrain the model with all data including any new custom gestures"""
        trainer = GestureTrainer(self.data_file)
        accuracy = trainer.train(method=method)
        if accuracy is not None:
            trainer.save_model(self.model_path)
            print(f"\nmodel retrained — accuracy: {accuracy:.1%}")
            return accuracy
        return None

    def record_and_train(self, name, description="", action="none",
                         num_samples=100, method="svm"):
        """convenience: record + retrain in one call"""
        count = self.record_gesture(name, description, action, num_samples)
        if count > 0:
            return self.retrain(method)
        return None


if __name__ == "__main__":
    recorder = GestureRecorder()
    registry = recorder.registry

    print("gesture customization tool")
    print("=" * 40)

    # show existing custom gestures
    custom = registry.list_gestures()
    if custom:
        print("\ncustom gestures:")
        for name, desc, action, samples in custom:
            print(f"  {name}: {desc} (action={action}, {samples} samples)")
    else:
        print("\nno custom gestures defined yet")

    print("\ncommands:")
    print("  record <name>  — record a new gesture")
    print("  remove <name>  — delete a gesture")
    print("  train          — retrain the model")
    print("  list           — show all gestures")
    print("  quit           — exit")

    while True:
        cmd = input("\n> ").strip().lower()

        if cmd in ("quit", "q", "exit"):
            break

        elif cmd == "list":
            custom = registry.list_gestures()
            if custom:
                for name, desc, action, samples in custom:
                    print(f"  {name}: {desc} (action={action}, {samples} samples)")
            else:
                print("  (none)")

        elif cmd.startswith("record"):
            parts = cmd.split(maxsplit=1)
            if len(parts) < 2:
                name = input("  gesture name: ").strip()
            else:
                name = parts[1]

            desc = input("  description (optional): ").strip()
            action = input(f"  action {GestureRegistry.BUILT_IN_ACTIONS} [none]: ").strip()
            if not action:
                action = "none"

            samples = input("  samples to record [100]: ").strip()
            samples = int(samples) if samples.isdigit() else 100

            recorder.record_gesture(name, desc, action, samples)

        elif cmd.startswith("remove"):
            parts = cmd.split(maxsplit=1)
            if len(parts) < 2:
                name = input("  gesture name to remove: ").strip()
            else:
                name = parts[1]
            if registry.remove(name):
                print(f"  removed '{name}'")
            else:
                print(f"  '{name}' not found")

        elif cmd == "train":
            method = input("  method [svm/knn]: ").strip() or "svm"
            recorder.retrain(method)

        else:
            print(f"  unknown command: {cmd}")

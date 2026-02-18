# Air Drawing

Draw in the air with your hands. Uses a webcam and MediaPipe to track hand landmarks in real time, then maps finger positions to drawing gestures. Point your index finger to draw, open your hand to erase, make a peace sign to change color.

Built with OpenCV for rendering, MediaPipe Tasks API for hand tracking, and NumPy for canvas math. Optional ML classifier (scikit-learn) for custom gesture training.

---

## Demo

| Gesture | What it does |
|---------|-------------|
| Index finger up (others curled) | **Draw** on canvas |
| Index + middle up | **Change color** (cycles through palette) |
| All fingers open | **Erase** under your hand |
| Fist | **Switch tool** between pen and brush |
| Pinch (thumb + index) | **Grab and move** the drawing |
| Both hands fist | **Clear** entire canvas |

---

## How it works

### Architecture

The project is broken into three layers:

```
main.py                    - Main loop. Reads camera, runs gesture engine, draws to canvas.

core/                      - Detection + logic layer
  camera.py                  Webcam capture, mirror flip, resolution config
  hand_tracker.py            MediaPipe Tasks API wrapper (VIDEO mode, downscaled detection)
  gesture_engine.py          Finger state detection + gesture mapping + debounce
  noise_filter.py            One-euro filter + velocity clamping + confidence scoring
  profiler.py                Per-section timing with rolling window stats
  state_manager.py           Tracks active tool, color, drawing state

app/                       - Drawing + display layer
  canvas.py                  Black surface overlay, blend onto camera feed, save to file
  tools.py                   Pen (thin line), Brush (smoothed thick line), Eraser (black circle)
  ui.py                      FPS counter, tool name, color palette, gesture label, confidence bar
  config.py                  All settings and thresholds in one place

ml/                        - ML pipeline (optional, needs scikit-learn)
  dataset_collector.py       Record landmark vectors + labels to CSV, visual progress bar
  trainer.py                 SVM/kNN classifier, 20 hand-relative features, cross-validation
  model_inference.py         Load model + predict gesture from landmarks, confidence scoring
  gesture_customizer.py      User-defined gesture registry, recording, and retraining
  benchmark.py               Synthetic + live performance benchmarking
```

### Signal Processing Pipeline

Raw MediaPipe landmarks are noisy. Before reaching the gesture engine, they pass through two filter layers:

1. **Velocity clamping** — if a landmark jumps more than 180px between frames, it's pulled back. This catches the worst spikes from MediaPipe momentarily losing the hand.

2. **One-euro filter** — adaptive low-pass filter on each landmark. When the hand is still, it smooths aggressively (removes micro-jitter). When the hand moves fast, it reduces smoothing (keeps up with motion). Much better than a fixed moving average.

3. **Confidence scoring** — the filter tracks how much raw and filtered positions differ over a sliding window. High divergence means noisy input, so a confidence score (0-1) is published. Downstream code uses this to skip drawing during bad frames and show a "tracking unstable" warning.

### Gesture Detection Pipeline

1. **Camera** grabs a 640x480 frame, flips it horizontally (mirror view)
2. **HandTracker** downscales to 480px wide, converts BGR to RGB, feeds to MediaPipe HandLandmarker in VIDEO mode
3. MediaPipe returns 21 landmarks per hand (x, y, z normalized), plus handedness labels
4. Landmarks are scaled back to original frame coordinates
5. **NoiseFilter** cleans up the landmarks (velocity clamping, one-euro smoothing, confidence scoring)
6. **GestureEngine** receives the filtered landmarks and runs through this process:

#### Finger detection (angle + distance, not just Y-coordinate)

Each finger is checked two ways:

- **PIP joint angle**: The angle at the main knuckle bend (MCP to PIP to DIP). A straight finger is around 170 degrees, a curled one is around 60-90 degrees.
- **Tip-to-wrist distance ratio**: Is the fingertip further from the wrist than the PIP joint? Curled fingers have their tip closer to the wrist than the knuckle.

Both must agree before the finger state changes. This prevents jitter from noisy landmarks.

**Hysteresis** is applied on top: once a finger is detected as "open", it stays open until the angle drops well below the threshold (not just slightly). Same in reverse. This creates a dead zone that stops flickering.

#### Thumb detection

Thumb is tricky because it moves laterally, not vertically. We use:
- Average angle at thumb MCP + IP joints
- Spread ratio: distance from thumb tip to index MCP, divided by hand size

The hand size (wrist to middle MCP distance) normalizes everything so it works at any distance from the camera.

#### Landmark smoothing

Raw landmarks jitter frame-to-frame. We keep a rolling average of the last 3 frames of landmark positions. The smoothed positions are used for both finger detection and drawing coordinates.

#### Gesture mapping

| Finger state | Gesture |
|-------------|---------|
| Index up, rest down | draw |
| Index + middle up, rest down | change_color |
| 3+ fingers up | erase |
| All down (fist) | switch_brush |
| Thumb-index pinch, rest down | grab |
| Nothing detected | idle |

#### Debounce

Raw gesture detection can flicker between states. The debounce system requires N consecutive frames of the same gesture before committing:

- **draw / erase**: 2 frames (~66ms) - needs to feel instant
- **grab**: 3 frames (~100ms) - slightly more confirmation to avoid false pinch
- **idle**: 3 frames - prevents brief tracking drops from killing your stroke
- **color / brush switch**: 5+ frames + 0.5s cooldown - these are intentional actions, not rapid switches

#### ML hybrid mode (optional)

When a trained model exists (`gesture_model.pkl`), the ML predictor runs alongside the rule engine every frame. If the ML model's confidence exceeds the threshold (default 0.75) and it disagrees with the rules, the ML prediction wins. This lets you:

- Override edge cases where rules struggle (weird hand angles, partial occlusion)
- Add entirely new gestures that rules can't express
- Fall back gracefully to rules when the model isn't sure

### Drawing pipeline

1. When gesture is "draw", the index fingertip position becomes the cursor
2. A line is drawn from the previous tip position to the current one using OpenCV line drawing
3. The canvas is a separate black NumPy array, overlaid onto the camera feed each frame using bitwise masking
4. Brush tool averages the last 4 points for smoother strokes
5. Eraser draws black circles at all fingertip positions + palm center

### Saving

- Press **s** to save drawing on white background as drawing_YYYYMMDD_HHMMSS.png
- Press **f** to save full frame (camera + drawing overlay) as capture_YYYYMMDD_HHMMSS.png

---

## Setup

**Requirements**: Python 3.10+, a webcam, decent lighting.

```bash
# clone and enter
git clone https://github.com/Umairism/Air-Drawing.git
cd Air-Drawing

# set up a virtual environment
python -m venv .venv
source .venv/bin/activate

# install dependencies
pip install .

# or install with ML support
pip install ".[ml]"

# download the hand landmark model (required)
wget https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
```

You can also install dependencies manually:

```bash
pip install opencv-python mediapipe numpy
pip install scikit-learn   # only if you want ML features
```

## Run

```bash
python main.py
```

### Controls

| Key | Action |
|-----|--------|
| q / ESC | Quit |
| c | Clear canvas |
| s | Save drawing (white background) |
| f | Save full frame (camera + drawing) |
| p | Print profiler report (timing breakdown) |

The window is resizable. You can also close it with the X button.

---

## Configuration

Everything lives in `app/config.py`:

| Setting | Default | What it does |
|---------|---------|-------------|
| CAMERA_WIDTH | 640 | Capture width |
| CAMERA_HEIGHT | 480 | Capture height |
| DETECTION_CONFIDENCE | 0.55 | Min confidence for hand detection |
| TRACKING_CONFIDENCE | 0.5 | Min confidence for landmark tracking |
| DETECTION_WIDTH | 480 | Downscaled width for detection (speed) |
| PEN_THICKNESS | 3 | Pen line width |
| BRUSH_THICKNESS | 12 | Brush line width |
| ERASER_THICKNESS | 40 | Eraser circle radius |
| DEBOUNCE_FRAMES | 3 | Frames before gesture switch |
| GESTURE_COOLDOWN | 0.5 | Seconds between mode switches |
| NOISE_FILTER_ENABLED | True | Enable/disable landmark smoothing |
| LOW_CONFIDENCE_THRESHOLD | 0.3 | Skip drawing below this confidence |
| PROFILER_ENABLED | False | Enable per-frame timing breakdown |
| ML_HYBRID_ENABLED | True | Use ML alongside rules (if model exists) |
| ML_CONFIDENCE_THRESHOLD | 0.75 | Min ML confidence to override rules |
| BENCHMARK_ENABLED | True | Write session metrics on exit |

---

## ML Pipeline

The ML system can replace or augment the rule-based gesture engine. It uses hand-relative features instead of raw coordinates, so the model generalizes across different hand positions and distances.

### Feature extraction (20 features)

All features are normalized by hand size (wrist to middle MCP distance) for scale invariance:

- **5 PIP joint angles** — one per finger, normalized to 0-1. Thumb uses average of MCP + IP angles.
- **5 tip-to-wrist ratios** — how far each fingertip is from the wrist, relative to hand size.
- **5 tip-to-palm ratios** — how far each fingertip is from the palm center.
- **4 inter-finger spread angles** — angles between adjacent fingertips through the palm center.
- **1 pinch ratio** — thumb tip to index tip distance, normalized by hand size.

### Training workflow

```bash
# 1. collect training data (camera feed, press keys to label)
python -m ml.dataset_collector

# 2. train an SVM classifier (default) or kNN
python -m ml.trainer                    # SVM
python -m ml.trainer gesture_data.csv knn  # kNN alternative

# 3. run - the model loads automatically on next launch
python main.py
```

The trainer outputs `gesture_model.pkl` + `gesture_model_meta.json` with accuracy, cross-validation scores, and class distribution.

### Custom gestures

Define your own gestures that aren't in the default set:

```bash
# add and record a custom gesture
python -m ml.gesture_customizer record thumbs_up

# list all registered gestures
python -m ml.gesture_customizer list

# retrain the model with custom + default gestures
python -m ml.gesture_customizer train

# remove a custom gesture
python -m ml.gesture_customizer remove thumbs_up
```

Custom gestures are stored in `custom_gestures.json` with optional actions (save_drawing, clear_canvas, toggle_pause, undo, screenshot).

---

## Benchmarking

Run the synthetic benchmark (no camera needed) to measure pipeline performance:

```bash
python -m ml.benchmark
```

This measures each pipeline stage independently:
- **Gesture engine** — rule-based recognition throughput
- **Noise filter** — one-euro + velocity clamping overhead
- **Feature extraction** — landmark to feature vector conversion
- **Full pipeline** — all stages combined

Results are saved to `benchmark_results.json`. Add `--live 300` to run a live benchmark with actual camera + MediaPipe for 300 frames.

Session-level metrics (gesture counts, ML override rate, total frames) are logged to `benchmark_log.json` automatically on exit when `BENCHMARK_ENABLED = True` in config.

---

## Packaging

### pip install

The project has a `pyproject.toml` for standard Python packaging:

```bash
pip install .              # core only
pip install ".[ml]"        # with ML dependencies
pip install ".[dev]"       # ML + PyInstaller
```

After installing, run with the `air-drawing` command.

### Standalone executable (PyInstaller)

```bash
pip install pyinstaller
pyinstaller AirDrawing.spec
```

Produces a single `dist/AirDrawing` binary (~140MB) with everything bundled. If you've trained a gesture model, it gets included automatically.

---

## Tests

70 tests covering the core gesture engine, noise filter, profiler, and ML feature extraction:

```bash
python -m unittest discover -s tests -v
```

| Test suite | Count | What it covers |
|-----------|-------|---------------|
| test_gesture_engine | 23 | Finger detection, gesture mapping, debounce, multi-hand, edge cases |
| test_noise_filter | 17 | Velocity clamping, one-euro smoothing, confidence scoring, reset |
| test_profiler | 12 | Section timing, FPS calculation, summary output, window rolling |
| test_ml_features | 18 | Feature extraction, position/scale invariance, different gestures, edge cases |

---

## Known Limitations

- **Lighting matters**. MediaPipe struggles in dim rooms or with strong backlighting. Face a window or use a desk lamp.
- **Webcam quality**. Cheap webcams with low resolution or high noise will produce jittery landmarks. 720p+ recommended.
- **Single drawing hand**. The app locks to the first hand it detects for drawing. If you want to switch hands, remove both hands from view first.
- **Thumb detection** is the least reliable finger — it moves laterally and the angle-based check can be inconsistent at certain orientations.
- **Canvas performance** degrades slightly with very complex drawings (lots of blending operations per frame).
- **Not tested on macOS/Windows** extensively. Built and tested on Fedora Linux. Should work on other platforms but may need tweaks for camera permissions.

---

## Tips

- Keep your hand about 1-2 feet from the camera for best landmark detection
- Make deliberate gestures — don't rush between draw and erase
- Good lighting on your hand makes a huge difference
- If gestures feel off, try adjusting DETECTION_CONFIDENCE in config.py
- The brush tool gives smoother lines than the pen if you're going for artistic strokes
- Press **p** while running to see a real-time performance breakdown

---

## Project Structure

```
Air-Drawing/
  main.py
  pyproject.toml                # pip install support
  AirDrawing.spec               # PyInstaller config
  hand_landmarker.task          # MediaPipe model (download separately)
  app/
    __init__.py
    config.py
    canvas.py
    tools.py
    ui.py
  core/
    __init__.py
    camera.py
    hand_tracker.py
    gesture_engine.py
    noise_filter.py
    profiler.py
    state_manager.py
  ml/
    __init__.py
    dataset_collector.py
    trainer.py
    model_inference.py
    gesture_customizer.py
    benchmark.py
  tests/
    test_gesture_engine.py
    test_noise_filter.py
    test_profiler.py
    test_ml_features.py
```

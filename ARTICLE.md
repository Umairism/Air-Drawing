# Designing a Modular Multi-Hand Gesture Recognition Framework with Hybrid Rule-Based and ML Detection

## Abstract

Hand gesture recognition in real-time video is deceptively hard. MediaPipe gives you 21 landmarks per hand, but the gap between raw coordinates and reliable gesture classification is where most projects fall apart. This article documents the architecture of a modular gesture recognition framework that combines deterministic rule-based detection with a machine learning classifier, connected through a confidence-gated hybrid pipeline. The system processes dual hands at 30+ FPS on consumer hardware, implements a three-stage noise filtering pipeline based on the one-euro filter, and uses rotation-invariant feature extraction to maintain accuracy across hand orientations. We present empirical benchmarks quantifying the latency-stability tradeoff of the filtering stack and discuss the design decisions that shaped each module.

---

## 1. Introduction

Most air-drawing demos are single-file scripts: detect a hand, check if the index finger is up, draw a line. They work in controlled conditions and break everywhere else. Jittery landmarks cause wavy lines, gesture detection flickers between states, and the code is too tangled to extend.

This project started from a different premise: build a framework, not a demo. The architecture needed to support multi-hand tracking, tool switching, custom gesture registration, and an ML classification pipeline — all without any single module knowing about the others.

The core challenge is threefold:

1. **Noise**: MediaPipe landmarks jitter by several pixels between frames, even when the hand is perfectly still.
2. **Ambiguity**: The difference between "index finger pointing" and "index finger pointing with a slightly bent middle finger" is a few degrees at a joint. Rule-based thresholds are fragile.
3. **Orientation**: A hand tilted 30° shouldn't produce different gesture classifications, but raw coordinate-based features will.

The system addresses these through layered signal processing, a dual-check gesture engine with hysteresis, and a hybrid detection pipeline that lets rule-based and ML classifiers cross-validate each other.

---

## 2. System Architecture

The codebase is organized into three layers with strict dependency boundaries:

```
core/               # signal processing and detection
  camera.py         # capture with retry logic
  hand_tracker.py   # MediaPipe Tasks API wrapper
  noise_filter.py   # one-euro filter + velocity clamping
  gesture_engine.py # rule-based gesture classification
  state_manager.py  # application state (tool, color, position)
  profiler.py       # per-frame timing with percentile reporting

app/                # presentation and configuration
  canvas.py         # drawing surface management
  tools.py          # pen, brush, eraser implementations
  ui.py             # HUD overlay rendering
  config.py         # centralized configuration

ml/                 # machine learning pipeline
  dataset_collector.py  # landmark recording to CSV
  trainer.py            # feature extraction + model training
  model_inference.py    # real-time prediction
  gesture_customizer.py # user-defined gesture registration
  benchmark.py          # synthetic performance profiling
```

Data flows in one direction: `camera → tracker → noise_filter → gesture_engine → state → canvas/tools → ui`. The ML pipeline sits alongside the gesture engine as an optional enhancer, not a replacement.

Each module communicates through plain data structures — lists of tuples for landmarks `(id, x, y, z)`, strings for gesture names, floats for confidence scores. No module imports from a sibling in the same layer except through `config.py`.

---

## 3. Signal Processing Pipeline

### 3.1 The Noise Problem

MediaPipe's hand landmark model runs in under 10ms per frame, but the landmarks it produces are noisy. Even with a stationary hand, fingertip positions jitter by 3–8 pixels frame-to-frame. For gesture recognition this is annoying; for drawing it's destructive — every drawn line inherits that jitter.

The noise filter sits between the hand tracker and the gesture engine, processing landmarks through three stages:

### 3.2 Stage 1: Velocity Clamping

When MediaPipe briefly loses a hand and re-detects it in the next frame, the landmark positions can teleport across the frame. The velocity clamp catches these discontinuities:

```
max_allowed_movement = MAX_VELOCITY × (dt / 0.033)
```

If any landmark moved further than `max_allowed_movement` pixels since the last frame, its displacement is scaled back to the maximum in the same direction. The velocity threshold (180 px at 30 FPS) corresponds to roughly three hand-widths of movement per frame — fast enough to never interfere with normal motion, strict enough to catch re-detection jumps.

### 3.3 Stage 2: One-Euro Filter

The one-euro filter is an adaptive low-pass filter designed specifically for noisy interactive signals. Unlike a fixed moving average, it adjusts its smoothing based on signal velocity:

- When the hand is still, the cutoff frequency drops and the filter smooths aggressively.
- When the hand moves fast, the cutoff rises and the filter lets the signal through with minimal lag.

The implementation maintains per-landmark state (filtered position and filtered velocity). The smoothing factor α is computed as:

$$\alpha = \frac{1}{1 + \frac{\tau}{\Delta t}}, \quad \text{where } \tau = \frac{1}{2\pi f_c}$$

The adaptive cutoff frequency $f_c$ is:

$$f_c = f_{min} + \beta \cdot |\dot{x}_{filtered}|$$

Where $f_{min}$ (MIN_CUTOFF = 1.5) controls smoothness at rest and $\beta$ (0.007) controls responsiveness during motion. These values were tuned empirically — lower $\beta$ values caused visible lag during fast drawing strokes, higher values let too much jitter through during static gestures.

### 3.4 Stage 3: Confidence Scoring

The filter tracks the average discrepancy between raw and filtered landmark positions over a sliding window. High discrepancy means the input is noisy. This is converted to a 0–1 confidence score through a sigmoid mapping:

$$\text{score} = \frac{1}{1 + \left(\frac{\bar{j}}{j_{threshold}}\right)^2} \times \min\left(\frac{n}{5}, 1\right)$$

Where $\bar{j}$ is the mean jitter over the last 10 frames, $j_{threshold}$ = 4.0 pixels, and the $\min(n/5, 1)$ term is a warmup ramp that prevents false high-confidence on the first few frames.

Downstream modules use this score to:
- Skip drawing when confidence drops below 0.3 (avoids jitter artifacts)
- Display a "tracking unstable" indicator in the UI
- Widen gesture debounce windows during uncertain periods

### 3.5 Latency-Stability Tradeoff: Measured

The filtering stack was benchmarked with 2000 synthetic frames of simulated hand motion (sinusoidal trajectories with Gaussian noise):

| Metric | Value |
|---|---|
| Processing overhead | +0.102 ms per frame (91% over raw) |
| Mean positional lag | 64.0 px |
| p95 positional lag | 101.0 px |
| Jitter reduction ratio | 8.4× (raw avg 4.3 px → smoothed avg 0.5 px) |
| Gesture engine alone | avg 0.115 ms, p99 0.268 ms |
| Noise filter alone | avg 0.118 ms, p99 0.394 ms |

The 0.102 ms overhead is negligible at 30 FPS (each frame has a 33.3 ms budget). The 8.4× jitter reduction transforms hand-drawn lines from visibly shaky to smooth. The 64 px mean positional lag sounds large in absolute terms, but it represents the filter doing its job — the filtered position trails the raw position because the filter is suppressing high-frequency noise. In practice, the adaptive cutoff ensures that during fast strokes the lag drops significantly (the one-euro filter was designed exactly for this scenario).

---

## 4. Gesture Recognition Engine

### 4.1 Dual-Check Detection

Finger state detection uses both joint angle and tip position as independent checks. A finger is classified as "open" only when both conditions agree:

1. **Angle check**: The PIP joint angle exceeds a threshold. A straight finger has a PIP angle near 180°; a curled finger drops below 120°.
2. **Distance check**: The fingertip is further from the wrist than the PIP joint (by at least 85% of the PIP-wrist distance). This catches cases where the finger is technically straight but tucked behind the palm.

Using separate thresholds for opening (>140° and tip further) and closing (<115° or <135° without tip further) creates a **hysteresis band** that prevents rapid toggling when a finger is near the boundary.

### 4.2 Landmark Smoothing

Before evaluating finger states, the engine applies a 3-frame rolling average to the landmark positions. This is lighter than the noise filter's one-euro approach — it's specifically tuned for gesture stability rather than drawing accuracy. The smoothing buffer is independent of the noise filter and operates even when the noise filter is disabled.

### 4.3 Pinch Detection with Hysteresis

The grab gesture (thumb-index pinch) uses hand-size-relative thresholds:

$$\text{pinch\_ratio} = \frac{\text{dist(thumb\_tip, index\_tip)}}{\text{dist(wrist, middle\_MCP)}}$$

The gesture activates when the ratio drops below 0.28 and deactivates when it rises above 0.40. This hysteresis band prevents flickering at the pinch boundary — critical because pinch distance naturally oscillates a few pixels even when the user is holding a steady pinch.

### 4.4 Temporal Debouncing

Every gesture change must persist for `DEBOUNCE_FRAMES` (3) consecutive frames before it's accepted. Additionally, mode-switching gestures (color change, brush toggle) have a `GESTURE_COOLDOWN` (0.5s) lockout to prevent accidental double-triggers.

### 4.5 Gesture Vocabulary

| Gesture | Rule |
|---|---|
| `draw` | Index finger only, extended |
| `erase` | All five fingers open (palm) |
| `change_color` | Index + middle extended, others closed (peace sign) |
| `switch_brush` | Index + middle + ring extended, others closed |
| `grab` | Thumb-index pinch detected |
| `clear_canvas` | Both hands showing simultaneous fists (multi-hand gesture) |
| `idle` | None of the above |

---

## 5. ML Classification Pipeline

### 5.1 Feature Engineering

The ML classifier doesn't see raw landmark coordinates. Instead, we extract a 20-dimensional feature vector engineered for invariance:

**5 PIP joint angles** (one per finger, thumb uses the average of MCP and IP angles). Joint angles are intrinsically rotation and scale-invariant — the angle between two bones doesn't change when you rotate the hand.

**5 fingertip-to-wrist distance ratios** and **5 fingertip-to-palm distance ratios**. Distances are divided by hand_size (wrist-to-middle-MCP distance), making them scale-invariant. But raw distances are NOT rotation-invariant — this is addressed below.

**4 inter-finger spread angles** (index-middle, middle-ring, ring-pinky, thumb-index). These measure how the fingers fan out.

**1 pinch ratio** (thumb-tip to index-tip distance / hand_size).

### 5.2 Rotation Invariance

Distance ratios and spread angles depend on absolute landmark positions. If you compute the distance from a fingertip to the wrist in pixel coordinates, rotating the hand in the camera plane changes the result (different pixels, same physical pose).

The solution is to rotate all landmarks into a **hand-local coordinate frame** before computing features:

1. Set the **origin** at the wrist (landmark 0).
2. Define the **y-axis** as the direction from wrist to middle finger MCP (landmark 9).
3. Compute the perpendicular x-axis.
4. Project all 21 landmarks into this frame using a 2D rotation matrix.

In this local frame, the wrist is always at (0, 0) and the middle finger MCP is always directly above it on the y-axis, regardless of how the hand is rotated in the camera view.

Joint angles (the first 5 features) don't need this treatment — they're already rotation-invariant by construction. But we apply the rotation before computing all distance-based features (the remaining 15) for consistency.

**Verification**: Unit tests confirm that features extracted from a hand rotated by 45° and 90° match the features from the original orientation within a tolerance of 0.05 (normalized feature space).

### 5.3 Model Training and Evaluation

The trainer supports SVM (RBF kernel) and k-NN (k=5, distance-weighted) classifiers. Training uses `StandardScaler` normalization, an 80/20 train-test split, and 5-fold cross-validation.

Evaluation metrics computed during training:
- **Accuracy** (cross-validated mean and standard deviation)
- **Per-class precision, recall, and F1-score**
- **Confusion matrix** (printed as a formatted table with gesture labels)

The trained model, scaler, feature configuration, per-class metrics, and confusion matrix are all serialized into a single `.pkl` file. This means model evaluation results travel with the model — you can inspect a deployed model's training performance without re-running the training pipeline.

### 5.4 Hybrid Detection: Confidence-Gated Override

The ML classifier runs alongside the rule-based engine, not as a replacement. On each frame where hand landmarks are available:

1. The rule-based engine produces a gesture classification.
2. The ML classifier independently predicts a gesture with a confidence score.
3. If both agree → the rule-based result is used (no overhead).
4. If they disagree and the ML confidence exceeds `ML_CONFIDENCE_THRESHOLD` (0.75) → the ML prediction overrides.
5. If they disagree but ML confidence is below threshold → the rule-based result is kept.

This architecture has a key advantage: **the rule-based engine provides a reliable floor**. It doesn't need training data, works immediately, and handles edge cases that the ML model hasn't seen. The ML model provides a ceiling — when it's confident, it's often more accurate than hand-tuned thresholds, especially for gestures with subtle distinctions.

### 5.5 Override Observability

Every override and disagreement event is logged to `ml_overrides.log` with:
- The rule-based gesture name
- The ML gesture name
- The ML confidence score
- The frame number

On session exit, aggregate statistics are written to `benchmark_log.json`:
- Override rate (what percentage of frames did ML actually take over)
- Agreement rate (how often the two systems concurred)
- A confusion matrix mapping rule-based gestures to ML predictions
- Per-gesture confidence statistics (mean, min, max)

This makes the hybrid system auditable. You can answer questions like "Is the ML model consistently overriding the same gesture?" or "Is there a gesture where ML confidence is systematically low?" without adding print statements.

---

## 6. Performance Profiling

### 6.1 Per-Frame Instrumentation

The profiler wraps each pipeline stage with `start(section)` / `stop(section)` calls and maintains a rolling window of the last 120 frames. Reporting uses percentiles, not just averages — p50 tells you the typical case, p95 tells you the occasional spike, and p99 tells you the worst-case-that-actually-happens.

| Section | Avg | p50 | p95 | p99 | Peak |
|---|---|---|---|---|---|
| Gesture engine | 0.115 ms | 0.094 ms | 0.203 ms | 0.268 ms | — |
| Noise filter | 0.118 ms | 0.088 ms | 0.236 ms | 0.394 ms | — |
| Feature extraction | 0.062 ms | 0.043 ms | 0.110 ms | 0.146 ms | — |

These numbers are from the synthetic benchmark (no camera or MediaPipe overhead). In practice, MediaPipe hand detection dominates the frame budget at 15–25 ms per frame. The gesture engine and noise filter together add under 0.5 ms at p99, confirming that the processing pipeline is not the bottleneck.

### 6.2 Why Percentiles Matter

A system averaging 30 FPS might still stutter. If p99 frame time is 80ms, one in every hundred frames takes nearly three frame-budgets to process. The user perceives this as a hitch every few seconds. Mean FPS masks these spikes entirely. The profiler reports p50/p95/p99 for both individual pipeline sections and total frame time, and the benchmark module does the same for the raw-vs-smoothed comparison.

---

## 7. Design Decisions and Tradeoffs

### 7.1 Why Not Just ML?

A pure ML approach would require:
- Training data for every gesture, every lighting condition, every hand size
- Retraining whenever a new gesture is added
- No fallback when the model encounters an out-of-distribution hand pose

The hybrid approach gives you a working system on day zero (rules only) and lets the ML model gradually improve gesture boundaries as training data accumulates. The 0.75 confidence threshold is deliberately conservative — the ML model has to be quite sure before it overrides a rule that's been manually validated.

### 7.2 Why the One-Euro Filter Over a Kalman Filter?

Kalman filters are the textbook answer for noisy time-series data, but they assume a known motion model. Hand gestures are inherently unpredictable — the hand can stop, reverse, or change speed at any moment. The one-euro filter's speed-adaptive cutoff handles this naturally: smooth when still, responsive when moving. It's also simpler to implement and tune (two parameters vs. a full state-space model).

### 7.3 Rotation Invariance: Local Frame vs. Data Augmentation

An alternative to computing rotation-invariant features is to augment the training data with rotated copies. We chose the feature-level approach because:
- It requires zero additional training data
- It works identically for rule-based and ML detection
- It's verifiable through unit tests (rotate landmarks, assert features match)
- Data augmentation increases training time and model size without guaranteeing invariance at unseen angles

### 7.4 Hysteresis Everywhere

Hysteresis appears in three places: finger state detection (separate open/close thresholds), pinch detection (0.28 on / 0.40 off), and gesture debouncing (N frames must agree). This is not accidental — any threshold-based classification on noisy continuous signals will flicker at the boundary. Hysteresis is the simplest and most predictable solution. The alternative (confidence smoothing or probabilistic state machines) adds complexity without meaningful accuracy gains for the gesture vocabulary we're working with.

---

## 8. Testing Strategy

The test suite covers 78 cases across four modules:

- **Gesture engine** (23 tests): Each gesture is tested with synthetic landmark data positioned at exact joint angles. Edge cases include partially curled fingers, hysteresis boundary conditions, and debounce timing.
- **Noise filter** (17 tests): One-euro filter convergence, velocity clamping at exact thresholds, confidence score warmup, and stability transitions.
- **Profiler** (15 tests): Rolling window behavior, percentile computation (including edge cases with empty data), and summary dictionary structure.
- **ML features** (23 tests): Feature vector length, scale invariance (same features at 2× hand size), rotation invariance at 45° and 90°, and the `_rotate_to_local` coordinate frame transformation (verifying wrist maps to origin, middle MCP maps to y-axis).

All tests use deterministic synthetic data — no camera, no MediaPipe, no randomness. A test that passes once passes every time.

---

## 9. Limitations and Future Work

**Depth axis**: The current feature vector uses only x/y coordinates from the 2D projection. MediaPipe does provide z-values (relative depth), but they're significantly noisier than x/y. Incorporating z-features with appropriate filtering could improve classification of gestures that differ primarily in depth (e.g., a flat palm vs. a palm tilted toward the camera).

**Temporal gestures**: The system classifies each frame independently. Gestures defined by motion trajectories (swipe, circle, wave) would require sequence modeling — either rule-based state machines or recurrent neural architectures.

**Personalization**: Joint angles and distance ratios vary between individuals. A calibration step that adjusts thresholds to the user's hand proportions could improve accuracy, particularly for the thumb (which has the highest anatomical variability).

**Lighting robustness**: MediaPipe's detection confidence degrades in low light. The confidence scoring system partially compensates (by signaling low-confidence frames), but the detection model itself is the limiting factor.

---

## 10. Conclusion

Building a reliable gesture recognition system is less about the detection model and more about everything around it: filtering noisy inputs, stabilizing classification decisions, engineering features that survive real-world variation, and making the whole pipeline observable enough to debug. The hybrid rule-based/ML architecture lets each approach cover the other's weaknesses — rules provide deterministic reliability, ML provides adaptive precision. Percentile-based profiling, rotation-invariant features, and per-frame override logging turn what could be a black box into a system you can actually reason about.

The full source code, including the benchmark suite and test coverage, is available in the project repository.

---

*Framework built with Python, OpenCV, MediaPipe, and scikit-learn. Tested on Fedora Linux with a consumer webcam at 640×480 resolution.*

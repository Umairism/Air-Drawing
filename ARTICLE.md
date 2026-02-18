# Building a Gesture Recognition Framework That Actually Works in the Real World

## Abstract

Hand gesture recognition sounds simple until you try to build it. MediaPipe hands you 21 landmarks per hand, and from there you're on your own — the gap between those raw coordinates and gesture classifications that don't flicker every other frame is where most projects quietly die. This article walks through the architecture of a modular gesture recognition framework I built that pairs a deterministic rule-based engine with an ML classifier through a confidence-gated hybrid pipeline. It handles two hands at 30+ FPS on normal hardware, runs a three-stage noise filtering stack rooted in the one-euro filter, and uses rotation-invariant features so the classifier doesn't care which way your hand is tilted. I'll go through the empirical benchmarks that shaped the filter tuning, explain why the gesture engine has hysteresis in three separate places, and get into the design decisions that I think matter most.

---

## Where This Started

If you look at most air-drawing projects on GitHub, they're one file. Detect a hand, check if the index finger is up, draw a line. They demo well. They also break the second you move your hand a little too fast, or the lighting changes, or you tilt your wrist 20 degrees. The landmarks start jittering, the gesture detection starts flickering between "draw" and "idle" ten times a second, and the drawn lines look like they were made during an earthquake.

I wanted to build something that could actually survive real conditions. Multi-hand tracking, tool switching, custom gesture registration, an ML pipeline that can take over when rules fall short — but structured so that no module has to know about any of the others. The kind of thing where you can rip out the noise filter and the gesture engine still works (worse, but it works). Or swap the ML model without touching a single line in the drawing code.

The hard parts turned out to be:

**Noise.** MediaPipe landmarks jitter 3–8 pixels even when your hand is completely still. For gesture recognition you can live with it. For drawing it's a disaster — every line you draw inherits that jitter and looks shaky.

**Ambiguity.** The difference between "index finger pointing" and "index finger pointing but the middle finger is a little bent" is maybe 15 degrees at one joint. Put a hard threshold there and it'll flicker back and forth forever.

**Orientation.** Tilt your hand 30 degrees and suddenly the same gesture produces different landmark coordinates. If your features are based on raw positions, rotation kills you.

The system handles these through layered filtering, a dual-check gesture engine that uses hysteresis to avoid flickering, and a hybrid pipeline where the rules and the ML model cross-validate each other.

---

## How the Code is Organized

Three layers. Strict boundaries between them.

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

Data flows one way: `camera → tracker → noise_filter → gesture_engine → state → canvas/tools → ui`. The ML pipeline sits next to the gesture engine as an optional enhancer. It can improve things, but the system doesn't need it to function.

Modules talk through plain data — lists of tuples for landmarks `(id, x, y, z)`, strings for gesture names, floats for confidence. Nothing fancy. No module reaches into a sibling in the same layer except through `config.py`.

---

## The Noise Problem (And Three Stages of Dealing With It)

MediaPipe's hand model is fast — under 10ms per frame. But the landmarks it spits out are noisy. Hold your hand perfectly still and watch the fingertip coordinates bounce around by 3–8 pixels every frame. For gesture detection this is annoying. For drawing it's actively harmful, because your drawn line inherits all that jitter.

The noise filter lives between the tracker and the gesture engine and processes landmarks through three stages. I'll go through each one.

### Velocity Clamping

Sometimes MediaPipe loses a hand for a frame and then re-detects it. When that happens, the landmarks teleport — suddenly they're 200 pixels away from where they were last frame. The velocity clamp catches these jumps:

```
max_allowed_movement = MAX_VELOCITY × (dt / 0.033)
```

If a landmark moved further than `max_allowed_movement` since last frame, its movement gets scaled back to the maximum in the same direction. The threshold (180px at 30 FPS) is about three hand-widths of movement per frame — way faster than anyone actually moves, but strict enough to catch re-detection teleports.

### The One-Euro Filter

This is the heart of the smoothing. The one-euro filter is an adaptive low-pass filter built specifically for noisy interactive signals, and the key insight is that it changes its behavior based on how fast the signal is moving:

When the hand is still, the cutoff frequency drops low and the filter smooths aggressively — kills the jitter. When the hand moves fast, the cutoff rises and the filter mostly gets out of the way — low lag. This is exactly the behavior you want for drawing: smooth when hovering, responsive when stroking.

The math behind it: the smoothing factor α depends on the cutoff frequency, which itself adapts to velocity:

$$\alpha = \frac{1}{1 + \frac{\tau}{\Delta t}}, \quad \text{where } \tau = \frac{1}{2\pi f_c}$$

The adaptive cutoff $f_c$ is:

$$f_c = f_{min} + \beta \cdot |\dot{x}_{filtered}|$$

$f_{min}$ (set to 1.5) controls how smooth things are at rest. $\beta$ (0.007 for idle, 0.08 for drawing — more on this later) controls how much the filter reacts to speed. Getting these values right took a fair amount of trial and error. Lower β values looked great when the hand was still but created visible lag during fast drawing strokes. Higher values kept up with motion but let too much jitter through during gestures.

### Confidence Scoring

The filter also tracks how much the raw and filtered positions disagree over a sliding window. Lots of disagreement means noisy input. This gets converted into a 0–1 confidence score:

$$\text{score} = \frac{1}{1 + \left(\frac{\bar{j}}{j_{threshold}}\right)^2} \times \min\left(\frac{n}{5}, 1\right)$$

$\bar{j}$ is mean jitter over the last 10 frames, $j_{threshold}$ is 4.0 pixels, and the $\min(n/5, 1)$ piece is a warmup ramp so the score doesn't report high confidence on the first couple frames before it has enough data.

The rest of the system uses this score in a few ways: skip drawing when confidence drops below 0.3 (avoids drawing jitter artifacts), show a "tracking unstable" warning in the UI, and widen the debounce windows when the signal is unreliable.

### What the Numbers Look Like

Benchmarked with 2000 synthetic frames of sinusoidal hand motion plus Gaussian noise:

| Metric | Value |
|---|---|
| Processing overhead | +0.102 ms per frame |
| Mean positional lag | 64.0 px |
| p95 positional lag | 101.0 px |
| Jitter reduction | 8.4× (4.3 px raw → 0.5 px smoothed) |
| Gesture engine alone | avg 0.115 ms, p99 0.268 ms |
| Noise filter alone | avg 0.118 ms, p99 0.394 ms |

The 0.102 ms overhead is nothing at 30 FPS (33.3 ms budget per frame). The 8.4× jitter reduction is the difference between visibly shaky lines and smooth ones. The 64px lag sounds bad, but that's the filter doing its job — it's suppressing high-frequency noise by letting the filtered position trail the raw one. During fast drawing strokes the adaptive cutoff kicks in and lag drops substantially. (The dynamic beta tuning covered in the case study brings this down much further for drawing specifically.)

---

## The Gesture Engine

### How Finger Detection Actually Works

I went through a few iterations on this. Simple angle thresholds? They flicker at the boundary. Tip position only? Doesn't work when the finger is straight but tucked behind the palm. What ended up working is a dual check — both conditions have to agree before a finger counts as "open":

**Angle check**: The PIP joint angle has to exceed a threshold. A straight finger sits near 180°, a curled one drops below 120°.

**Distance check**: The fingertip has to be further from the wrist than the PIP joint is (by at least 85% of the PIP-wrist distance). This catches the annoying case where the finger is technically extended but folded behind the palm.

The thresholds for opening a finger (>140° and tip is further out) are different from the thresholds for closing it (<115°, or <135° without the distance condition met). That gap is the hysteresis band. Without it, a finger sitting right at the boundary would toggle open/closed every frame. With it, the finger has to move decisively past the threshold in one direction before it flips.

### Internal Landmark Smoothing

Before checking finger states, the engine runs a 3-frame rolling average over the landmark positions. This is much lighter than the one-euro filter — it's not trying to produce smooth drawing coordinates, just trying to stabilize the finger state decisions. It runs independently of the noise filter and works even if the noise filter is turned off.

### Pinch Detection

The grab gesture uses a thumb-index pinch, and the distance is measured relative to hand size so it works regardless of how far the hand is from the camera:

$$\text{pinch\_ratio} = \frac{\text{dist(thumb\_tip, index\_tip)}}{\text{dist(wrist, middle\_MCP)}}$$

Activates at ratio < 0.28, deactivates at ratio > 0.40. That 0.12 hysteresis gap is critical — even when you're holding a steady pinch, the measured distance wobbles by a few pixels. Without the gap, the gesture would strobe on and off.

### Debouncing

Every gesture transition has to hold for 3 consecutive frames before the engine accepts it. On top of that, mode-switching gestures (color change, brush toggle) have a 0.5 second cooldown to prevent accidental double-triggers. It sounds like a lot of layers, but each one catches a different class of mistake.

### The Gesture Vocabulary

| Gesture | What triggers it |
|---|---|
| `draw` | Index finger only, extended |
| `idle` | Index + middle extended, others closed |
| `change_color` | Index + middle + ring extended, others closed |
| `grab` | Thumb-index pinch |
| `switch_brush` | Four fingers up (no thumb) |
| `erase` | All five fingers open (full palm) |
| `clear_canvas` | Both hands showing fists simultaneously |

The ordering matters in the code. The engine checks from most-specific to least-specific — five fingers before four, four before three, and so on. If you checked "index up" before "index + middle up," you'd never trigger the two-finger gesture because the first check would always match first.

---

## The ML Pipeline

### Why Not Just Use ML for Everything?

I get asked this a lot. Short answer: the rule-based engine works on day one with zero training data. The ML model needs examples, and it needs examples of every gesture in every condition you care about. Miss a lighting condition or a hand size in training and the model produces garbage for that case — and you won't know until a user hits it.

So the architecture runs both in parallel. The rules provide a reliable baseline. The ML model can improve on it when it's confident.

### Feature Engineering

The classifier never sees raw coordinates. Instead it gets a 20-dimensional feature vector designed for invariance:

**5 PIP joint angles** — one per finger (thumb uses an average of MCP and IP angles). These are naturally rotation and scale invariant because the angle between two bones doesn't change when you rotate or resize the hand.

**10 distance ratios** — 5 fingertip-to-wrist and 5 fingertip-to-palm distances, all divided by hand size (wrist to middle MCP distance). Scale invariant, but NOT rotation invariant in raw form — which is why we need the coordinate frame trick below.

**4 spread angles** — index-middle, middle-ring, ring-pinky, thumb-index. Measures how the fingers fan out.

**1 pinch ratio** — thumb tip to index tip distance, normalized by hand size.

### Making It Rotation Invariant

Here's the problem: if you compute a distance ratio from fingertip to wrist using pixel coordinates, and then the user rotates their hand 30 degrees, those pixel coordinates change even though the physical hand pose is identical. The features drift, the classifier gets confused.

The fix: rotate all landmarks into a hand-local coordinate frame before computing any distance-based features.

1. Wrist (landmark 0) becomes the origin.
2. The direction from wrist to middle finger MCP (landmark 9) becomes the y-axis.
3. Perpendicular to that is the x-axis.
4. Project all 21 landmarks into this frame with a 2D rotation matrix.

Now the wrist is always at (0,0) and the middle finger base is always straight up, regardless of how the hand is actually oriented in the camera view. The joint angle features (first 5) don't even need this — they're rotation invariant by construction. But we apply the rotation before computing all distance-based features (the other 15) anyway for consistency.

I verified this with unit tests: features from a hand rotated 45° and 90° match the original within 0.05 in normalized feature space.

### The Hybrid Override System

Every frame where we have landmarks:

1. The rule engine classifies the gesture.
2. The ML model independently predicts a gesture with a confidence score.
3. If they agree — use the rules result. Done.
4. If they disagree and ML confidence is above 0.75 — let ML override. Log it.
5. If they disagree but ML confidence is below 0.75 — keep the rules result. Log the disagreement.

The 0.75 threshold is deliberately conservative. The ML model has to be quite sure before it gets to override a rule that was hand-tuned and validated through 96 unit tests. This means the rules handle the common cases reliably, and the ML model only steps in when it's confident it knows better — usually at gesture boundaries where the rules' hard thresholds are weakest.

### Logging Everything

Every override and disagreement goes to `ml_overrides.log` with the frame number, both gesture names, and the ML confidence. When the session ends, aggregate stats get dumped to `benchmark_log.json` — override rate, agreement rate, a confusion matrix mapping rule predictions to ML predictions, per-gesture confidence distributions.

This makes the whole hybrid system auditable after the fact. You can answer things like "is the ML model consistently disagreeing on the same gesture?" or "is there a gesture where ML confidence never gets above 0.6?" without adding debug prints and re-running.

### Training

The trainer supports SVM (RBF kernel) and k-NN (k=5, distance-weighted). Standard workflow: `StandardScaler` normalization, 80/20 split, 5-fold cross-validation. It computes accuracy, per-class precision/recall/F1, and a confusion matrix. All of this gets serialized into the model file alongside the weights, so a deployed model carries its own evaluation report.

---

## Performance Profiling

The profiler instruments each pipeline stage with start/stop calls and keeps a rolling window of the last 120 frames. I report percentiles instead of just averages, because averages lie.

A system averaging 30 FPS might still stutter. If your p99 frame time is 80ms, one in every hundred frames takes nearly three frame budgets to process. The user sees a hitch every few seconds. Mean FPS hides this completely.

| Section | Avg | p50 | p95 | p99 |
|---|---|---|---|---|
| Gesture engine | 0.115 ms | 0.094 ms | 0.203 ms | 0.268 ms |
| Noise filter | 0.118 ms | 0.088 ms | 0.236 ms | 0.394 ms |
| Feature extraction | 0.062 ms | 0.043 ms | 0.110 ms | 0.146 ms |

These are from the synthetic benchmark — no camera, no MediaPipe. In production, MediaPipe dominates at 15–25 ms per frame. The gesture engine and noise filter together add under 0.5 ms even at p99. The processing pipeline is definitively not the bottleneck.

---

## Design Decisions I'd Make Again

**Hybrid over pure ML.** A pure ML system needs training data for every gesture, every condition. It needs retraining for new gestures. It has no fallback for out-of-distribution poses. The hybrid approach gives you a working system immediately and lets ML gradually improve the edges. Worth the extra complexity every time.

**One-euro filter over Kalman.** Kalman filters want a motion model, and hand gestures are inherently unpredictable — the hand can stop, reverse, or change speed at any moment. The one-euro filter's speed-adaptive cutoff handles this natively. Two parameters to tune instead of a full state-space model. Simpler to implement, simpler to reason about.

**Feature-level rotation invariance over data augmentation.** You could augment training data with rotated copies instead of computing invariant features. But the feature approach needs zero extra data, works for both rule-based and ML detection identically, and is testable — rotate landmarks, assert features match. Data augmentation grows training time and model size and still doesn't guarantee invariance at angles you didn't include.

**Hysteresis everywhere.** It shows up in finger state detection, pinch detection, and gesture debouncing. Three separate places, same principle: any threshold on a noisy continuous signal will flicker at the boundary. Hysteresis is the simplest fix and the most predictable one. I looked at confidence smoothing and probabilistic state machines as alternatives — they add complexity without meaningful accuracy gains for the gesture set we're working with.

---

## Testing

96 tests across five modules. All deterministic synthetic data, no camera, no MediaPipe, no randomness. A test that passes once passes every time.

The gesture engine tests (25 tests) cover each gesture with synthetic landmarks at exact joint angles, plus edge cases — partially curled fingers, hysteresis boundary conditions, debounce timing, multi-hand locking. The noise filter tests (17) verify one-euro convergence, velocity clamping thresholds, confidence warmup, stability transitions. The profiler tests (15) hit rolling window behavior, percentile math with edge cases, and summary dictionary structure. The ML feature tests (23) check feature vector dimensions, scale/rotation/translation invariance, and the coordinate frame transformation. The distribution shift tests (17) sweep hand sizes from 0.5× to 2.5×, simulate different camera FOVs, vary lighting noise from 2–15px, test rotations from -60° to +90°, combine multiple shifts, and assert that the system fails under documented bad conditions — documenting the limits, not hiding them.

---

## What's Missing

**Depth.** The feature vector only uses x/y from the 2D projection. MediaPipe does give you z-values but they're 3–5× noisier than x/y. Incorporating z with proper filtering could help distinguish gestures that only differ in depth (flat palm vs. tilted palm), but right now it's too noisy to be useful.

**Temporal gestures.** Every frame gets classified independently. Swipes, circles, waves — anything defined by a trajectory — would need sequence modeling. State machines at minimum, or something like an LSTM if you wanted to get serious about it. The debounce logic gives some frame-to-frame continuity but it's not trajectory-aware.

**Per-user calibration.** Joint angles and distance ratios vary between people. A calibration step that adjusts thresholds to the individual user's hand proportions would probably help, especially for the thumb — it has the most anatomical variation across people.

**Lighting.** MediaPipe's detection confidence drops in low light. The confidence scoring system flags it, but there's only so much you can do downstream when the detection model itself is struggling.

---

## Wrapping Up

The interesting part of building a gesture recognition system isn't the detection model. It's everything around it — filtering noisy input so drawing doesn't look terrible, stabilizing decisions so the gesture doesn't flicker, engineering features that survive the real world, and making the pipeline observable enough that when something goes wrong you can figure out why. The hybrid architecture lets rules and ML cover each other's weaknesses. Percentile profiling, rotation-invariant features, and per-frame override logging turn what could easily be a black box into something you can actually debug and reason about.

Full source is in the repository, including the benchmark suite and all tests.

---

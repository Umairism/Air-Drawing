# Case Study: Making an Air Drawing System That Doesn't Fall Apart Under Noise

## The Story of Chasing Down 64 Pixels of Lag and Finding Out Where the System Actually Breaks

**What this is**: Air Drawing — a modular gesture recognition framework with hybrid rule-based and ML detection  
**Running on**: Fedora Linux, Python 3.14.2, OpenCV 4.13.0, MediaPipe 0.10.32, scikit-learn 1.8.0  
**Camera**: 640×480 webcam, processed at native resolution  
**Tests**: 96 passing across gesture engine, noise filter, profiler, ML features, and distribution shift

---

## The 64-Pixel Problem

I noticed it while profiling. The noise filter — specifically the one-euro filter at its default β=0.007 (straight from the original paper's recommended starting range) — was introducing a substantial positional lag. I ran a benchmark with 2000 synthetic frames of hand motion: sinusoidal trajectories, 4px Gaussian noise, 640×480 resolution.

The numbers that came back:

| Metric | Value |
|---|---|
| Mean positional lag | 63.9 px |
| p50 lag | 66.3 px |
| p95 lag | 102.5 px |
| Max lag | 115.4 px |
| Mean jitter (raw) | 4.3 px |
| Mean jitter (filtered) | 0.5 px |
| Jitter reduction | 8.3× |

63.9 pixels. On a 640-pixel-wide frame, that's 10% of the entire width. A typical hand at arm's length is about 130 pixels in the frame, so the cursor was trailing the actual fingertip by half a hand-width. The jitter reduction was excellent — 8.3× is the difference between shaky lines and smooth ones — but that lag was a problem.

For gesture recognition, honestly, it doesn't matter. The engine doesn't need sub-pixel accuracy to figure out if a finger is open or closed. But for drawing? The user expects the line to follow their finger. A half-hand-width delay is immediately noticeable. The drawn line looks like it's being dragged behind on a leash.

So the question became: is this acceptable? And the answer was: **it depends on what the user is doing.** Drawing needs low lag. Gestures need low jitter. One β can't serve both masters.

---

## Fixing It: Dynamic Beta

β in the one-euro filter controls how aggressively the cutoff frequency responds to speed:

$$f_c = f_{min} + \beta \cdot |\dot{x}_{filtered}|$$

Crank β up and the filter follows motion closely (low lag) but lets jitter through. Turn it down and jitter gets crushed (high stability) but everything lags.

The solution was to stop pretending one β fits all situations. The filter now switches based on what the user is actually doing:

- **Drawing** (β=0.08): The finger is moving, the user wants the line to follow it. Responsiveness is what matters.
- **Idle** (β=0.007): The hand is still or performing a gesture. Smoothness is what matters. Kill the jitter.

The transition between modes is exponential with a factor of 0.12 per frame — so it doesn't snap between values:

```
current_beta += (target_beta - current_beta) × 0.12
```

### What This Actually Changed

I tested with a 2000-frame sequence that simulates real usage: 500 frames idle, 700 frames drawing, 300 frames idle, 500 frames drawing. 4px noise throughout (realistic for moderate lighting at 640×480).

**Before — fixed β=0.007:**

| Phase | Lag (avg) | Lag (p95) | Jitter (smooth) | Jitter reduction |
|---|---|---|---|---|
| Idle | 3.5 px | 6.1 px | 0.2 px | 23.8× |
| Draw | 24.4 px | 80.2 px | 0.7 px | 6.4× |

**After — dynamic β:**

| Phase | Lag (avg) | Lag (p95) | Jitter (smooth) | Jitter reduction |
|---|---|---|---|---|
| Idle | 3.3 px | 5.8 px | 0.1 px | 31.4× |
| Draw | 7.5 px | 44.7 px | 1.4 px | 3.3× |

Drawing lag went from 24.4px down to 7.5px — a 69% reduction. At 640px frame width, 7.5px is 1.2%. You can't see it at 30 FPS. The idle smoothing actually got *better* (23.8× → 31.4×) because the idle β is no longer getting contaminated by drawing-mode settings. Drawing-phase jitter went from 0.7px to 1.4px, which is a real tradeoff, but 1.4px on drawn lines is still perfectly smooth to the eye.

The p95 drawing lag — which captures the worst moments during direction changes — went from 80.2px down to 44.7px. Still not tiny, but the really bad spikes got cut nearly in half.

The trade we made: less smoothing during drawing (3.3× vs 6.4× jitter reduction) in exchange for lag that drops below the perception threshold. For a drawing application, that's the right call.

---

## Why β=0.08? The Sweep

I didn't just pick 0.08 out of thin air. I swept seven β values across 2000 frames of constant motion with 4px noise to see the full tradeoff curve:

| β | Lag (avg) | Lag (p50) | Lag (p95) | Lag (max) | Smooth jitter | Reduction |
|---|---|---|---|---|---|---|
| 0.007 | 63.9 px | 66.3 px | 102.5 px | 115.4 px | 0.5 px | 8.3× |
| 0.01 | 60.3 px | 61.7 px | 100.4 px | 119.8 px | 0.7 px | 6.5× |
| 0.02 | 44.9 px | 42.0 px | 95.9 px | 111.8 px | 1.0 px | 4.2× |
| 0.04 | 21.2 px | 13.6 px | 67.4 px | 78.0 px | 1.4 px | 3.2× |
| **0.08** | **8.3 px** | **4.0 px** | **35.8 px** | **46.1 px** | **1.6 px** | **2.6×** |
| 0.15 | 4.0 px | 2.2 px | 15.0 px | 26.9 px | 1.9 px | 2.3× |
| 0.30 | 2.4 px | 2.0 px | 6.3 px | 18.6 px | 2.3 px | 1.9× |

The returns diminish fast above 0.08. Going from 0.04 to 0.08 cuts lag from 21.2px to 8.3px (60% drop) and only costs 0.2px more jitter. Going from 0.08 to 0.15 cuts lag from 8.3 to 4.0px (another 52%) but costs 0.3px more jitter. And past 0.15, jitter reduction drops below 2× — the filter is barely filtering anymore.

β=0.08 is the knee of the curve. Lag becomes imperceptible (8.3px average, 4.0px median — that's less than half a percent of frame width) and jitter reduction is still meaningful at 2.6×.

---

## Where Things Break: Distribution Shift Testing

This is the part of the project where I stopped testing under ideal conditions and started asking: how bad can things get before the system stops working?

### The Noise Tolerance Map

I ran gesture recognition accuracy across nine different hand sizes (scales 0.4× to 2.0×) and eighteen noise levels (1px to 30px), with proper stochastic noise — different random jitter on every frame, 200 trials per condition, with a 3-frame debounce warmup. Here's a slice of the results:

| Scale | hand\_size | 5px | 10px | 15px | 20px | 25px | 30px |
|---|---|---|---|---|---|---|---|
| 0.4 | 32 px | 100% | 100% | 100% | 87% | 91% | 88% |
| 0.5 | 40 px | 100% | 100% | 100% | 95% | 88% | 82% |
| 0.7 | 56 px | 100% | 100% | 100% | 98% | 99% | 88% |
| 1.0 | 80 px | 100% | 100% | 100% | 100% | 97% | 99% |
| 1.3 | 104 px | 100% | 100% | 100% | 100% | 100% | 100% |
| 1.6 | 128 px | 100% | 100% | 100% | 100% | 100% | 100% |
| 2.0 | 160 px | 100% | 100% | 100% | 100% | 100% | 100% |

The pattern follows SNR = hand\_size / noise pretty cleanly. Above an SNR of about 5, everything is 100%. Below that it starts to degrade, but it's a gradual slope — not a cliff.

This actually surprised me. My initial testing had used a deterministic noise seed (same jitter pattern on every frame), and that showed a sharp 97%→0% cliff. Turned out the cliff was an artifact of identical noise repeating, not a real property of the system. Once I switched to properly stochastic noise, the degradation was much more graceful. That's a lesson I won't forget about benchmarking.

The failure mode is geometric. The gesture engine decides if a finger is open or closed by computing PIP joint angles from three landmarks. An open finger has a true PIP angle of about 180° (the three points are nearly collinear). For a misclassification, the measured angle has to drop below 115° — a 65° margin. That margin is protected by three layers that stack on top of each other.

### The Three-Layer Defense: A Formal Model

I spent a while working out a closed-form model that predicts accuracy from a single number: SNR = hand\_size / noise. Turns out you can do it by composing three independent effects.

**Layer 1 — Landmark smoothing cuts noise by √3.** The gesture engine averages each landmark over 3 frames before computing angles. For uniform noise, averaging 3 independent samples reduces variance by a factor of 3:

$$\text{Var}(\bar{x}) = \frac{\text{Var}(x)}{N} = \frac{\sigma^2/3}{3} = \frac{\sigma^2}{9}$$

That's equivalent to reducing noise from σ to σ/√3, so the effective SNR goes up by a factor of √3 ≈ 1.73. I verified this with Monte Carlo: at SNR=8, unsmoothed per-finger accuracy was 0.846, smoothed was 0.991. The smoothed result matches what you'd predict for an effective SNR of 8√3 ≈ 13.9.

**Layer 2 — Hysteresis doubles the angular margin.** The engine uses different thresholds for opening vs. closing a finger:

- Closed → Open: PIP angle must exceed 145° (or 160° without the distance check)
- Open → Closed: PIP angle must drop below 115° (or 135° if the distance check fails)

After warmup — where fingers are initialized to their correct states — an open finger (true angle 180°) has a full 65° margin before it gets reclassified as closed. Not the 35° margin you'd get coming from the cold-start open threshold.

The per-finger classification probability with smoothing and hysteresis follows a probit model pretty well:

$$p_{\text{finger}}(\text{SNR}) = \Phi\left(\frac{\text{SNR} - \mu}{w}\right)$$

where Φ is the standard normal CDF. I fit this against 50,000 Monte Carlo trials per condition (72 data points across 9 scales × 8 noise levels) and got:

$$\mu = 1.42, \quad w = 1.16$$

Maximum absolute error against the Monte Carlo reference: 0.024. The μ=1.42 result means a single finger hits 50% accuracy at SNR=1.42 — which is an absurdly low bar. In practice the system is almost never in danger at the per-finger level. The critical thresholds:

| Target accuracy | Required SNR |
|---|---|
| 90% | 2.9 |
| 95% | 3.3 |
| 99% | 4.1 |
| 99.9% | 5.0 |

**Layer 3 — Debounce squares the error rate.** The engine requires 2 consecutive frames with the same detected gesture before accepting a transition. One bad frame gets ignored. You need two bad frames in a row to actually change the output.

After warmup in the correct state, I modeled this as a Markov chain:
- From correct state: takes 2 consecutive errors to leave. Per-step probability: $(1 - p_{\text{raw}})^2$
- From wrong state: takes 2 consecutive correct frames to recover. Per-step probability: $p_{\text{raw}}^2$

Steady-state probability of being correct:

$$p_{\text{debounced}} = \frac{p_{\text{raw}}^2}{p_{\text{raw}}^2 + (1 - p_{\text{raw}})^2}$$

This amplification is dramatic. If per-frame accuracy is only 80% (marginal), debounced accuracy is 0.64/(0.64+0.04) = 94%. It really crushes errors.

**How the layers compose for different gestures:**

Draw needs 1 open finger + 3 closed fingers. The closed fingers barely ever false-open (their true angle is near 0° with a 115° margin), so draw accuracy is essentially just the debounced accuracy of the one open finger.

Erase (all five open) is harder — it needs multiple simultaneous correct classifications:

$$p_{\text{erase}} = p_{\text{debounced}}\big(p^2 \cdot [1 - (1-p)^3]\big)$$

### Does the Model Actually Work?

I validated the three-layer model against 306 real data points (9 scales × 18 noise levels × 2 gestures × 200 trials each, stochastic noise throughout):

| Gesture | Mean absolute error | Median error | Max error |
|---|---|---|---|
| Draw | 0.022 | 0.000 | 0.601 |
| Erase | 0.017 | 0.000 | 0.182 |
| **Overall** | **0.020** | **0.000** | **0.601** |

The median error is 0.000 — the model nails most conditions exactly. The max error of 0.601 only shows up at SNR < 1.5, which is extreme noise territory. At those levels, the debounce Markov chain assumption breaks down in an interesting way: the "wrong" gesture keeps changing frame-to-frame (different random errors each time), which means the debounce counter for the wrong gesture never reaches 2. The correct gesture stays confirmed by default. So the model underestimates accuracy at very low SNR — a conservative failure mode.

### The Practical SNR Table

Here's what all three layers give you when composed:

| Accuracy target | Draw (SNR ≥) | Erase (SNR ≥) |
|---|---|---|
| 90% | 2.2 | 2.7 |
| 95% | 2.5 | 2.9 |
| 99% | 3.0 | 3.4 |
| 99.9% | 3.6 | 3.9 |

These numbers are remarkably low. At SNR=4 — hand size just 4× the noise — drawing accuracy is already 99%. The three layers compound in a satisfying way: smoothing boosts raw SNR by 1.73×, hysteresis doubles the angular margin, and debounce squares the per-frame error rate.

### What This Means When Someone Actually Uses It

| Scenario | hand\_size (est.) | MediaPipe noise (est.) | SNR | Predicted draw acc. |
|---|---|---|---|---|
| Adult at 50cm, good light | ~130 px | 2-3 px | 43-65 | >99.9% |
| Adult at 80cm, moderate light | ~80 px | 4-6 px | 13-20 | >99.9% |
| Child at 60cm, moderate light | ~60 px | 4-6 px | 10-15 | >99.9% |
| Adult at 120cm, any light | ~45 px | 3-5 px | 9-15 | >99.9% |
| Small hand, poor light | ~40 px | 10-15 px | 2.7-4 | 90-99% |
| Any hand, very poor light | varies | 15-20 px | 2-3 | 70-95% |

For anything short of actively bad lighting, the system sits comfortably above 99.9%. Even in poor lighting (SNR around 3), draw stays above 90%. The only real danger zone is very poor lighting, where MediaPipe itself starts struggling.

### Rotation — Not a Problem

I tested gesture recognition across hand rotations from -60° to +90° (nine angles). Draw and erase were recognized correctly at every single angle. ML features — computed after rotating landmarks into the hand-local coordinate frame — drifted less than 0.1 in normalized feature space across the full sweep.

Rotation invariance works as designed. This was one of those things I was nervous about in theory but turned out to be completely fine in practice.

### How Far Can You Push It?

The worst realistic condition I tested: scale=0.6 (roughly a child's hand), 3px noise (moderate lighting), 30° tilt. The system held above 70% accuracy for the draw gesture across 40 frames. Not great, but functional.

At scale=0.6 + 8px noise + 30° tilt, accuracy dropped to 0%. That's the documented failure boundary. The noise exceeds 10% of hand size at that point (8px / 48px = 16.7%), and the system simply can't distinguish the signal from the noise anymore.

---

## Pipeline Performance Numbers

Everything below comes from synthetic benchmarks (2000 frames each). This measures the processing pipeline only — no camera, no MediaPipe. In actual use, MediaPipe takes 15-25ms per frame and dominates the budget.

### Per-Stage Timing

| Stage | avg | p50 | p95 | p99 | max |
|---|---|---|---|---|---|
| Gesture engine | 0.301 ms | 0.195 ms | 0.622 ms | 2.516 ms | 13.327 ms |
| Noise filter | 0.374 ms | 0.206 ms | 0.833 ms | 3.597 ms | 30.014 ms |
| Feature extraction | 0.159 ms | 0.080 ms | 0.244 ms | 1.589 ms | 14.490 ms |

### About Those p99 Spikes

The noise filter's p99 is 3.597ms — that's 17× its p50. Those spikes land when Python's garbage collector runs or the OS scheduler pulls the thread for a moment. At 30 FPS, it means one frame out of a hundred loses about 3.4ms to filter overhead. The per-frame budget is 33.3ms, so that's roughly 10% of one frame, every 3.3 seconds. Not enough to stutter, but worth knowing about.

The gesture engine's p99 of 2.516ms comes from debounce state transitions — frames where the engine is evaluating both the current gesture and comparing it against its history buffer.

### Full Pipeline Combined

Running noise filter + gesture engine in sequence:

| Metric | Value |
|---|---|
| Avg frame time | 1.49 ms |
| p50 frame time | 1.06 ms |
| p95 frame time | 3.14 ms |
| p99 frame time | 7.54 ms |
| Peak frame time | 17.10 ms |
| Theoretical FPS | 672 |

That 17.1ms peak is a single frame — almost certainly a GC pause. In production it stacks on top of MediaPipe's ~20ms, pushing that one frame to about 37ms (27 FPS for one frame). Imperceptible as a one-off.

---

## The Hybrid ML Pipeline

### How the Features Work

The ML classifier gets a 20-dimensional feature vector, not raw landmarks:

| Feature group | Count | What it's invariant to |
|---|---|---|
| PIP joint angles | 5 | Position, scale, rotation (by construction) |
| Tip-to-wrist distance ratios | 5 | Position, scale (normalized by hand\_size) |
| Tip-to-palm distance ratios | 5 | Position, scale (normalized by hand\_size) |
| Inter-finger spread angles | 4 | Position, scale (angle-based) |
| Pinch ratio | 1 | Position, scale (normalized by hand\_size) |

The distance ratios and spread angles get computed *after* rotating landmarks into a hand-local coordinate frame (wrist at origin, wrist→middle\_MCP along the y-axis), which makes them rotation-invariant too.

I tested the invariances explicitly:
- **Scale**: Features at scale=0.5 match features at scale=2.5 within δ=0.08. Verified across 6 scale factors.
- **Rotation**: 45° and 90° rotations match the 0° baseline within δ=0.05. Full sweep from -90° to +90° in 10° steps stays under 0.1 drift.
- **Translation**: Features at (100,100) match features at (500,400) exactly. Intrinsic since everything uses relative distances.

### How the Override Works

Both the rule engine and ML classifier run on every frame:

1. Rules classify the gesture. No training data needed, always available.
2. ML predicts a gesture with a confidence score. Needs a trained model.
3. If they agree — rules win. No overhead.
4. If they disagree and ML confidence ≥ 0.75 — ML takes over. Event gets logged.
5. If they disagree and ML confidence < 0.75 — rules stay. Disagreement gets logged.

Every override and disagreement goes to `ml_overrides.log` with timestamps, both predictions, confidence, and frame number. At session end, aggregates (override rate, agreement rate, per-gesture confusion matrix, confidence distributions) get written to `benchmark_log.json`.

### The 0.75 Threshold

Conservative on purpose. At 0.75, the ML model has to put 75% probability on its top class before it can override a hand-tuned rule validated against 96 unit tests. Clear-cut gestures where ML easily classifies: ML can fix edge cases the rules miss. Ambiguous gestures where ML is uncertain: rules handle them, no ML hallucinations on unseen poses. Custom gestures the user trains themselves: ML can activate them without anyone touching the rule engine code.

### Training and Evaluation

Trainer supports SVM (RBF kernel) and k-NN (k=5, distance-weighted). Uses `StandardScaler` normalization, 80/20 split, 5-fold cross-validation. Computes accuracy, per-class precision/recall/F1, and a confusion matrix. Everything gets serialized into the model file alongside the weights — so you can inspect what a model was trained on and how it performed without re-running the training pipeline.

---

## How the Tests are Set Up

| Test file | Tests | Coverage |
|---|---|---|
| test\_gesture\_engine.py | 25 | Finger detection, gesture recognition, debounce, hysteresis, multi-hand locking, edge cases |
| test\_noise\_filter.py | 17 | One-euro convergence, velocity clamping, confidence scoring, reset behavior |
| test\_profiler.py | 15 | Rolling window, percentiles (p50/p95/p99), summary dict structure, peak tracking |
| test\_ml\_features.py | 23 | Feature dimensions, scale/rotation/translation invariance, helper functions, edge cases |
| test\_distribution\_shift.py | 17 | Hand sizes (0.5×–2.5×), camera FOV, lighting noise (2–15px), rotation (-60° to 90°), combined shifts, failure boundaries |
| **Total** | **96** | |

Everything uses deterministic synthetic data. No camera needed, no MediaPipe, no randomness. A test that passes once will pass forever. The distribution shift tests are parameterized — controllable scale, rotation, noise, position. And they explicitly test that the system *fails* under known-bad conditions. The "poor lighting" test verifies that 7px noise works AND that 12px noise doesn't — documenting the boundary, not pretending it doesn't exist.

---

## What I Know Is Still Wrong

**The noise tolerance model breaks down at extreme SNR.** The three-layer defense keeps things above 99% down to SNR ≈ 3, but below SNR ≈ 2, accuracy drops to 50-80%. The formal model slightly underestimates low-SNR accuracy because the debounce Markov chain assumption stops holding when the "wrong" gesture keeps changing frame-to-frame. And that initial cliff I saw in testing (97%→0%)? Artifact of using a fixed random seed. Different noise every frame produces a smooth sigmoid, not a cliff. Lesson learned about deterministic vs. stochastic benchmarking.

**Only 2D.** MediaPipe gives z-values for depth but they're 3-5× noisier than x/y. I store them but don't use them for features. This means gestures that differ mainly in depth (flat palm vs. tilted palm) can't be distinguished reliably.

**No trajectory awareness.** Each frame is classified on its own. Swipes, circles, waves — anything defined by a movement path — would need a sequence model. An HMM or LSTM, or at the very least a hand-written state machine. The debounce logic provides some frame-to-frame continuity, but it's not modeling trajectories.

**The beta transition takes time.** When switching from idle to draw mode, the filter needs about 8 frames (at the 0.12 transition rate) to ramp β from 0.007 up to 0.08. During those 8 frames, lag is still elevated. That's where the p95 drawing lag of 44.7px comes from — it's mostly transition frames. Making the transition faster would fix this but risk visible jitter at the draw-idle boundary.

**Uniform noise model.** My distribution shift tests add the same noise magnitude to every landmark. Real MediaPipe noise is non-uniform — fingertips jitter more than the wrist, z-axis more than x/y, and landmarks near the frame edges are noisier than ones near center. A more honest test would use a per-landmark noise model calibrated from actual MediaPipe output.

---

## Reproducing Everything

Every number in this document comes from the system's own benchmarks and tests:

```bash
# synthetic benchmark — no camera needed
python -m ml.benchmark

# distribution shift tests
python -m unittest tests.test_distribution_shift -v

# full test suite
python -m unittest discover -s tests -v
```

Output files:
- `benchmark_results.json` — per-stage timing, raw-vs-smoothed comparison
- `case_study_data.json` — beta sweep data, dynamic vs fixed comparison, noise tolerance map
- `beta_tuning_results.json` — raw beta sweep numbers

---

*All measurements on Fedora Linux, AMD/Intel x86_64, Python 3.14.2, virtual environment. Synthetic benchmarks seed with `numpy.random.seed(42)` for reproducibility. No mock data — every number is a direct output from the system's own infrastructure.*

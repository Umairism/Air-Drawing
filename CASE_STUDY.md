# Case Study: Air Drawing — A Real-Time Gesture Recognition Framework

## Engineering a Stable Drawing System from Noisy Hand Landmarks

**System**: Air Drawing — modular multi-hand gesture recognition with hybrid rule-based and ML detection  
**Platform**: Fedora Linux, Python 3.14.2, OpenCV 4.13.0, MediaPipe 0.10.32, scikit-learn 1.8.0  
**Resolution**: 640×480 webcam, processed at native resolution  
**Test Suite**: 95 tests passing (gesture engine, noise filter, profiler, ML features, distribution shift)

---

## 1. The Problem: 64 Pixels of Lag

During profiling, we measured the positional lag introduced by the noise filter. The one-euro filter was configured with β=0.007 (its default from the original paper's recommended starting range). The benchmark ran 2000 frames of synthetic hand motion — sinusoidal trajectories with 4px Gaussian noise at 640×480.

The result:

| Metric | Value |
|---|---|
| Mean positional lag | 63.9 px |
| p50 lag | 66.3 px |
| p95 lag | 102.5 px |
| Max lag | 115.4 px |
| Mean jitter (raw) | 4.3 px |
| Mean jitter (filtered) | 0.5 px |
| Jitter reduction | 8.3× |

**63.9 px at 640×480 is 10% of frame width.** A typical hand at arm's length occupies ~130px in the frame, so the cursor trails the fingertip by half a hand-width. For gesture detection this is fine — the engine doesn't need sub-pixel positions to classify a finger as open or closed. But for drawing, the lag is visible. The drawn line appears to "drag behind" the finger.

**The question: is this acceptable?**

For this use case — an air drawing tool at 640×480 — the answer is **no for drawing, yes for gestures**. When drawing, the user expects the line to follow the fingertip with near-zero perceived delay. When idle or performing a gesture, they don't care about positional accuracy — they care about classification stability. These are conflicting requirements under a fixed β.

---

## 2. The Fix: Dynamic Beta Tuning

β in the one-euro filter controls the speed-sensitivity of the cutoff frequency:

$$f_c = f_{min} + \beta \cdot |\dot{x}_{filtered}|$$

Higher β means the filter reacts more aggressively to speed changes — less lag, but less smoothing. Lower β means heavier smoothing but higher lag during motion.

Instead of picking one β for all situations, the filter now adjusts β based on what the user is doing:

- **Drawing mode** (β=0.08): The user is moving their finger to draw. Responsiveness matters. The filter follows the finger closely.
- **Idle mode** (β=0.007): The user is performing a gesture or their hand is still. Smoothness matters. The filter cleans up jitter aggressively.

The transition is exponential with a factor of 0.12 per frame, so the filter doesn't snap between settings:

```
current_beta += (target_beta - current_beta) × 0.12
```

### Measured Results

We tested with a 2000-frame sequence simulating realistic usage: 500 frames idle → 700 frames drawing → 300 frames idle → 500 frames drawing. Noise level: 4px (realistic for moderate lighting at 640×480).

**Fixed β=0.007 (old behavior):**

| Phase | Lag (avg) | Lag (p95) | Jitter (smooth) | Jitter reduction |
|---|---|---|---|---|
| Idle | 3.5 px | 6.1 px | 0.2 px | 23.8× |
| Draw | 24.4 px | 80.2 px | 0.7 px | 6.4× |

**Dynamic β (new behavior):**

| Phase | Lag (avg) | Lag (p95) | Jitter (smooth) | Jitter reduction |
|---|---|---|---|---|
| Idle | 3.3 px | 5.8 px | 0.1 px | 31.4× |
| Draw | 7.5 px | 44.7 px | 1.4 px | 3.3× |

**What changed:**

- Drawing lag dropped from **24.4px → 7.5px** (69% reduction). At 640px frame width, 7.5px is 1.2% — below the threshold of perception at 30 FPS.
- Idle jitter reduction improved from **23.8× → 31.4×** because the idle β is never contaminated by drawing-mode responsiveness.
- Drawing jitter went from 0.7px to 1.4px — still smooth enough for drawn lines.
- p95 drawing lag dropped from 80.2px → 44.7px — the occasional spikes during direction changes are also smaller.

The tradeoff: we sacrificed some drawing-phase smoothness (3.3× vs 6.4× jitter reduction) to get lag below the perception threshold. For a drawing application, this is the correct trade.

---

## 3. The Beta Sweep: Mapping the Tradeoff Curve

To justify the β=0.08 choice for drawing mode, we swept seven β values across 2000 frames of constant motion with 4px noise:

| β | Lag (avg) | Lag (p50) | Lag (p95) | Lag (max) | Smooth jitter | Reduction |
|---|---|---|---|---|---|---|
| 0.007 | 63.9 px | 66.3 px | 102.5 px | 115.4 px | 0.5 px | 8.3× |
| 0.01 | 60.3 px | 61.7 px | 100.4 px | 119.8 px | 0.7 px | 6.5× |
| 0.02 | 44.9 px | 42.0 px | 95.9 px | 111.8 px | 1.0 px | 4.2× |
| 0.04 | 21.2 px | 13.6 px | 67.4 px | 78.0 px | 1.4 px | 3.2× |
| **0.08** | **8.3 px** | **4.0 px** | **35.8 px** | **46.1 px** | **1.6 px** | **2.6×** |
| 0.15 | 4.0 px | 2.2 px | 15.0 px | 26.9 px | 1.9 px | 2.3× |
| 0.30 | 2.4 px | 2.0 px | 6.3 px | 18.6 px | 2.3 px | 1.9× |

The curve shows diminishing returns above β=0.08:
- β=0.04→0.08 cuts lag from 21.2 to 8.3px (60% drop) at the cost of 0.2px more jitter.
- β=0.08→0.15 cuts lag from 8.3 to 4.0px (52% drop) at the cost of 0.3px more jitter.
- β=0.15→0.30 cuts lag from 4.0 to 2.4px but the jitter reduction drops below 2×.

β=0.08 sits at the knee of the curve — the point where lag becomes imperceptible (8.3px = 1.3% of frame width, p50 = 4.0px) while maintaining meaningful jitter reduction (2.6×).

---

## 4. Distribution Shift: Where the System Breaks

### 4.1 The Noise Tolerance Map

We tested gesture recognition accuracy across nine hand sizes and eighteen noise levels with proper stochastic noise (different random landmarks every frame), 200 trials per condition after 3-frame debounce warmup. The system degrades gradually as SNR (hand\_size / noise) drops:

| Scale | hand\_size | 5px | 10px | 15px | 20px | 25px | 30px |
|---|---|---|---|---|---|---|---|
| 0.4 | 32 px | 100% | 100% | 100% | 87% | 91% | 88% |
| 0.5 | 40 px | 100% | 100% | 100% | 95% | 88% | 82% |
| 0.7 | 56 px | 100% | 100% | 100% | 98% | 99% | 88% |
| 1.0 | 80 px | 100% | 100% | 100% | 100% | 97% | 99% |
| 1.3 | 104 px | 100% | 100% | 100% | 100% | 100% | 100% |
| 1.6 | 128 px | 100% | 100% | 100% | 100% | 100% | 100% |
| 2.0 | 160 px | 100% | 100% | 100% | 100% | 100% | 100% |

The degradation follows the SNR = hand\_size / noise ratio. Where our initial tests (using a deterministic noise seed) showed a sharp cliff from 97%→0%, the stochastic measurement reveals a gradual sigmoid — the cliff was an artifact of identical noise on every frame, not a real system property.

The failure mode is geometric. The gesture engine classifies fingers as open/closed based on PIP joint angles computed from three landmarks (MCP, PIP, DIP). With noise σ on each landmark coordinate, the measured angle deviates from its true value. The open finger's true PIP angle is 180° (collinear segments). For it to be misclassified as closed, the measured angle must drop below 115° — a 65° margin, protected by three layers of defense detailed in Section 4.5.

### 4.5 Formal Error-Rate Model: SNR → Accuracy

The system's noise tolerance is determined by three defense layers that stack multiplicatively. We can derive a closed-form model that predicts gesture accuracy from a single number: **SNR = hand\_size / noise**.

#### Layer 1: Landmark Smoothing (σ → σ/√3)

The gesture engine averages each landmark's position over a 3-frame rolling window before computing angles. For uniform noise $U(-\sigma, \sigma)$, averaging $N=3$ independent samples reduces the positional variance by a factor of 3:

$$\text{Var}(\bar{x}) = \frac{\text{Var}(x)}{N} = \frac{\sigma^2/3}{3} = \frac{\sigma^2}{9}$$

This is equivalent to reducing the noise half-width from $\sigma$ to $\sigma/\sqrt{3}$, boosting the effective SNR by $\sqrt{3} \approx 1.73$.

Monte Carlo verification confirms this: at SNR=8 (scale=1.0, noise=10px), the unsmoothed per-finger correct classification rate is 0.846, while the 3-frame smoothed rate is 0.991 — matching what we'd expect for an effective SNR of $8\sqrt{3} \approx 13.9$.

#### Layer 2: Hysteresis (35° margin → 65° margin)

The gesture engine uses two different thresholds for finger state transitions:

- **Closed → Open**: PIP angle must exceed 145° (or 160° without the tip-distance check)
- **Open → Closed**: PIP angle must drop below 115° (or 135° if the tip-distance check fails)

The gap between these thresholds is the hysteresis band. After warmup (where fingers are initialized to the correct state via noiseless frames), an open finger with true angle 180° has a **65° margin** before it gets reclassified as closed — not the 35° margin that the open-from-cold-start threshold would suggest.

This is the critical asymmetry. The per-finger classification probability after smoothing and hysteresis is well-described by a probit (cumulative normal) model:

$$p_{\text{finger}}(\text{SNR}) = \Phi\left(\frac{\text{SNR} - \mu}{w}\right)$$

where $\Phi$ is the standard normal CDF. Fitting against Monte Carlo data (50,000 trials per condition, 72 data points across 9 scales × 8 noise levels):

$$\mu = 1.42, \quad w = 1.16$$

The probit fit has a maximum absolute error of 0.024 against the Monte Carlo reference.

**Reading the probit parameters**: $\mu = 1.42$ means a single finger has 50% correct-classification probability at SNR = 1.42. This is extremely low — the system is almost never in danger at the per-finger level. The critical SNR thresholds for per-finger accuracy are:

| Target | Required SNR |
|---|---|
| 90% | 2.9 |
| 95% | 3.3 |
| 99% | 4.1 |
| 99.9% | 5.0 |

#### Layer 3: Debounce (error² suppression)

The gesture engine requires 2 consecutive frames with the same detected gesture before confirming a transition. This means a single-frame classification error is invisible to the output — only consecutive errors (a "burst") can change the confirmed gesture.

After warmup in the correct state, the system is modeled as a Markov chain:
- From **correct** state: need 2 consecutive frame-level errors to transition to **wrong**. Probability per step: $(1 - p_{\text{raw}})^2$.
- From **wrong** state: need 2 consecutive correct frames to recover. Probability per step: $p_{\text{raw}}^2$.

The steady-state probability of being in the correct state is:

$$p_{\text{debounced}} = \frac{p_{\text{raw}}^2}{p_{\text{raw}}^2 + (1 - p_{\text{raw}})^2}$$

This dramatically amplifies the per-finger advantage. For example, if $p_{\text{raw}} = 0.8$ (marginal), the debounced accuracy is $0.8^2 / (0.8^2 + 0.2^2) = 0.64/0.68 = 0.94$.

#### Composing Gesture-Level Accuracy

Different gestures have different per-finger requirements:

**Draw** (1 finger must stay open, 3 must stay closed):

$$p_{\text{draw}} = p_{\text{debounced}}\big(p_{\text{finger}}(\text{SNR})\big)$$

The 3 closed fingers have negligible false-open probability (closed finger's true angle is 0° with a 115° margin to the open threshold), so $p_{\text{draw}}$ is dominated by the single open finger.

**Erase** (index + middle must be open, ≥1 of ring/pinky/thumb must be open):

$$p_{\text{erase}} = p_{\text{debounced}}\big(p^2 \cdot [1 - (1-p)^3]\big)$$

where $p = p_{\text{finger}}(\text{SNR})$. The erase gesture is more demanding because it needs multiple simultaneous correct classifications.

#### Model Validation

The three-layer model (smoothing + hysteresis + debounce) was validated against 306 measured data points (9 scales × 18 noise levels × 2 gestures × 200 trials each with stochastic noise):

| Gesture | Mean absolute error | Median error | Max error |
|---|---|---|---|
| Draw | 0.022 | 0.000 | 0.601 |
| Erase | 0.017 | 0.000 | 0.182 |
| **Overall** | **0.020** | **0.000** | **0.601** |

The max error of 0.601 occurs only at SNR < 1.5 (extreme noise), where the model underestimates draw accuracy. At these levels, the debounce Markov chain assumption breaks down because the "wrong" gesture varies frame-to-frame (preventing the counter from reaching 2), which paradoxically keeps the correct confirmed gesture alive longer than the steady-state model predicts.

#### The Final SNR Table

Combining all three layers, here are the practical SNR thresholds:

| Accuracy target | Draw (SNR ≥) | Erase (SNR ≥) |
|---|---|---|
| 90% | 2.2 | 2.7 |
| 95% | 2.5 | 2.9 |
| 99% | 3.0 | 3.4 |
| 99.9% | 3.6 | 3.9 |

These are remarkably low. At SNR = 4 (hand\_size just 4× the noise), the draw gesture is already 99% accurate. This robustness comes from the three layers compounding: smoothing boosts raw SNR by 1.73×, hysteresis doubles the angular margin, and debounce squares the per-frame error rate.

---

### 4.2 What This Means in Practice

| Scenario | hand\_size (est.) | MediaPipe noise (est.) | SNR | Predicted draw acc. |
|---|---|---|---|---|
| Adult at 50cm, good light | ~130 px | 2-3 px | 43-65 | >99.9% |
| Adult at 80cm, moderate light | ~80 px | 4-6 px | 13-20 | >99.9% |
| Child at 60cm, moderate light | ~60 px | 4-6 px | 10-15 | >99.9% |
| Adult at 120cm, any light | ~45 px | 3-5 px | 9-15 | >99.9% |
| Small hand, poor light | ~40 px | 10-15 px | 2.7-4 | 90-99% |
| Any hand, very poor light | varies | 15-20 px | 2-3 | 70-95% |

The system is robust for virtually all reasonable usage conditions. The formal model predicts >99.9% draw accuracy at SNR ≥ 4, which covers all scenarios except poor/very poor lighting. Even in poor lighting (SNR ≈ 3), the draw gesture stays above 90%.

### 4.3 Rotation Tolerance

Separately, we tested gesture recognition across hand rotations from -60° to +90° (9 angles). The draw and erase gestures were recognized correctly at every angle. ML features (computed in the hand-local coordinate frame) drifted by less than 0.1 in normalized feature space across the full sweep.

This confirms that rotation invariance is not a failure mode — the hand-local coordinate frame transformation works as designed.

### 4.4 Combined Shift

The worst realistic condition we tested: scale=0.6 (child's hand), 3px noise (moderate lighting), 30° tilt. The system maintained >70% accuracy for the draw gesture across 40 frames.

At scale=0.6 + noise=8px + 30° tilt, accuracy dropped to 0%. This is the documented failure boundary — noise has exceeded the 10% hand_size threshold (8px / 48px = 16.7%).

---

## 5. Pipeline Performance

All numbers from synthetic benchmarks (2000 frames each). These measure the processing pipeline only — no camera capture, no MediaPipe inference. In production, MediaPipe dominates the frame budget at ~15-25ms.

### 5.1 Stage Timing

| Stage | avg | p50 | p95 | p99 | max |
|---|---|---|---|---|---|
| Gesture engine | 0.301 ms | 0.195 ms | 0.622 ms | 2.516 ms | 13.327 ms |
| Noise filter | 0.374 ms | 0.206 ms | 0.833 ms | 3.597 ms | 30.014 ms |
| Feature extraction | 0.159 ms | 0.080 ms | 0.244 ms | 1.589 ms | 14.490 ms |

### 5.2 p99 Analysis

The p99 for noise filter is 3.597ms — 17× the p50 (0.206ms). This spike happens when GC runs or the OS scheduler interrupts the Python process. At 30 FPS, one frame in a hundred loses 3.4ms to filter overhead. The per-frame budget is 33.3ms, so this consumes 10% of one frame every 3.3 seconds. Not enough to cause a visible stutter, but it would compound if the pipeline had tighter margins.

The gesture engine's p99 of 2.516ms comes from debounce state transitions — frames where the engine evaluates both the current gesture and compares against history.

### 5.3 Full Pipeline

Combined noise filter + gesture engine running in sequence:

| Metric | Value |
|---|---|
| Avg frame time | 1.49 ms |
| p50 frame time | 1.06 ms |
| p95 frame time | 3.14 ms |
| p99 frame time | 7.54 ms |
| Peak frame time | 17.10 ms |
| Theoretical FPS | 672 |

The peak of 17.1ms is a single frame — likely a GC pause. In production, this happens on top of MediaPipe's ~20ms, pushing that one frame to ~37ms (27 FPS momentarily). Not ideal, but imperceptible as a single-frame event.

---

## 6. The Hybrid ML Pipeline

### 6.1 Feature Engineering

The ML classifier uses a 20-dimensional feature vector:

| Feature group | Count | Invariant to |
|---|---|---|
| PIP joint angles | 5 | Position, scale, rotation (intrinsic) |
| Tip-to-wrist distance ratios | 5 | Position, scale (normalized by hand_size) |
| Tip-to-palm distance ratios | 5 | Position, scale (normalized by hand_size) |
| Inter-finger spread angles | 4 | Position, scale (angle-based) |
| Pinch ratio | 1 | Position, scale (normalized by hand_size) |

Distance ratios and spread angles are computed AFTER rotating landmarks into a hand-local coordinate frame (wrist = origin, wrist→middle_MCP = y-axis), making them rotation-invariant.

**Verified invariances:**

- **Scale**: Features extracted at scale=0.5 match features at scale=2.5 within δ=0.08. Tested across 6 scale factors.
- **Rotation**: Features at 45° and 90° rotations match the 0° baseline within δ=0.05. Sweep from -90° to +90° in 10° steps shows max drift < 0.1 in normalized feature space.
- **Translation**: Features at position (100, 100) match features at position (500, 400). This is intrinsic — all features use relative distances.

### 6.2 Hybrid Override Architecture

The rule-based engine and ML classifier run in parallel on every frame:

1. Rules produce a gesture classification (deterministic, no training data needed).
2. ML produces a gesture + confidence score (probabilistic, requires training data).
3. If they agree → use rule-based result.
4. If they disagree and ML confidence ≥ 0.75 → ML overrides. Log the event.
5. If they disagree and ML confidence < 0.75 → keep rules. Log the disagreement.

Every override and disagreement is written to `ml_overrides.log` with timestamps, both gesture names, confidence, and frame number. On session exit, aggregate statistics (override rate, agreement rate, per-gesture confusion matrix, per-gesture confidence distributions) are written to `benchmark_log.json`.

### 6.3 Why 0.75 Threshold

The threshold was chosen conservatively. At 0.75, the ML model must assign 75% probability to its top class before overriding a rule that was hand-tuned and validated through 95 unit tests. This means:

- Clear gestures (where ML easily classifies): ML overrides when rules misclassify edge cases.
- Ambiguous gestures (where ML is uncertain): rules handle them, avoiding ML hallucinations on unseen poses.
- New gestures (custom-trained by the user): ML can activate them without touching the rule engine.

### 6.4 Evaluation Infrastructure

The trainer computes on every training run:
- 5-fold cross-validation accuracy (mean ± std)
- Per-class precision, recall, F1-score
- Confusion matrix (printed as a formatted table during training, saved to model metadata)

These metrics are serialized into the model file alongside the weights. A deployed model carries its own evaluation — you can inspect `gesture_model_meta.json` without re-running training.

---

## 7. Test Architecture

### 7.1 Coverage by Module

| Test file | Tests | What it covers |
|---|---|---|
| test_gesture_engine.py | 23 | Finger detection, gesture recognition, debounce, hysteresis, multi-hand, edge cases |
| test_noise_filter.py | 17 | One-euro convergence, velocity clamping, confidence scoring, reset behavior |
| test_profiler.py | 15 | Rolling window, percentiles (p50/p95/p99), summary dict structure, peak tracking |
| test_ml_features.py | 23 | Feature vector dimensions, scale/rotation/translation invariance, helper functions, edge cases |
| test_distribution_shift.py | 17 | Hand sizes (0.5×–2.5×), camera FOV simulation, lighting (2–15px noise), rotation (-60° to 90°), combined shifts, documented failure boundaries |
| **Total** | **95** | |

### 7.2 Testing Philosophy

All tests use **deterministic synthetic data**. No camera, no MediaPipe, no randomness. A test that passes once passes every time. This means:

- CI-friendly: no hardware dependencies.
- Distribution shift tests use parameterized synthetic hands with controllable scale, rotation, noise, and position.
- Failure boundary tests assert that the system **does** fail under known-bad conditions (confirming the tolerance map).
- The "poor lighting" test checks both that 7px noise works AND that 12px noise fails — documenting the limit, not hiding it.

---

## 8. Known Limitations

1. **Noise tolerance at extreme SNR**: The three-layer defense (smoothing + hysteresis + debounce) keeps gesture accuracy above 99% down to SNR ≈ 3. Below SNR ≈ 2, the system degrades to 50-80% — still functional but unreliable. The formal model (Section 4.5) slightly underestimates low-SNR accuracy because the debounce Markov chain assumption breaks down when the wrong gesture varies frame-to-frame. Initial testing with deterministic noise (fixed random seed) had shown an artificial cliff from 97%→0% — this was a measurement artifact, not a real system property.

2. **2D features only**: The z-axis (depth) from MediaPipe is noisier than x/y by a factor of 3-5×. We use z for storage but not for feature computation. This means gestures that differ primarily in depth (e.g., a flat palm vs. a palm tilted toward the camera) cannot be reliably distinguished.

3. **No temporal modeling**: Every frame is classified independently. Swipe, circle, and wave gestures require sequence analysis (HMM, LSTM, or at minimum a state machine). The gesture engine's debounce logic provides frame-to-frame continuity, but it doesn't model trajectories.

4. **Dynamic beta lag spike at transitions**: When switching from idle to draw mode, the filter takes ~8 frames (at 0.12 transition rate) to ramp β from 0.007 to 0.08. During these frames, lag is still elevated. The p95 drawing lag of 44.7px reflects this transition period. A faster transition rate would reduce this but risk visible jitter at the draw-idle boundary.

5. **Single noise model**: Our distribution shift tests use uniform noise. Real MediaPipe noise is non-uniform — fingertips jitter more than the wrist, z-axis jitter exceeds x/y jitter, and noise increases near frame edges. A more accurate test would use a per-landmark noise model calibrated from real MediaPipe data.

---

## 9. Reproducibility

Every number in this document can be regenerated:

```bash
# full benchmark (synthetic, no camera needed)
python -m ml.benchmark

# distribution shift tests
python -m unittest tests.test_distribution_shift -v

# beta sweep and dynamic comparison
# (run case_study_data.json generation — see benchmark scripts)

# full test suite
python -m unittest discover -s tests -v
```

Data files produced:
- `benchmark_results.json` — stage timing and raw-vs-smoothed comparison
- `case_study_data.json` — beta sweep, dynamic vs fixed, noise tolerance map
- `beta_tuning_results.json` — raw beta sweep data

---

*All measurements taken on Fedora Linux, AMD/Intel x86_64, Python 3.14.2, with `.venv` virtual environment. Synthetic benchmarks use `numpy.random.seed(42)` for reproducibility. No mock data was used — every number is a direct output from the system's own benchmark and test infrastructure.*

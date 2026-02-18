"""
landmark noise filter.

sits between the hand tracker and the gesture engine.
cleans up jittery landmarks, rejects bad frames, and provides
a confidence score so the rest of the pipeline knows when to
trust the data.
"""
import math
import time
from collections import deque


class NoiseFilter:
    """
    filters raw hand landmarks to reduce jitter and reject garbage frames.

    three layers of filtering:
      1. velocity clamping   - if a landmark jumps further than physically
                               possible between frames, clamp it
      2. one-euro filter     - adaptive low-pass per landmark. smooth when
                               slow, responsive when fast. the gold standard
                               for cursor smoothing.
      3. confidence scoring  - tracks hand stability over time. if the hand
                               is shaking or landmarks are flickering,
                               the score drops and downstream code can
                               back off.
    """

    # one-euro filter defaults (feel free to tune these)
    MIN_CUTOFF = 1.5        # lower = smoother when still
    BETA = 0.007            # higher = less lag when moving fast
    D_CUTOFF = 1.0          # derivative filter cutoff

    # dynamic beta tuning — the key insight from profiling.
    # at beta=0.007, avg lag is 64px (10% of frame width). visible.
    # at beta=0.08, avg lag drops to 8px but jitter reduction drops
    # from 8.5x to 2.6x. the right beta depends on what the user
    # is doing RIGHT NOW.
    #
    # when drawing: responsiveness matters more. use high beta.
    # when idle/gestures: smoothness matters more. use low beta.
    #
    # measured tradeoff (640x480, 4px noise, sinusoidal motion):
    #   beta=0.007 → lag 64px, jitter 0.5px (8.5x reduction)
    #   beta=0.04  → lag 21px, jitter 1.4px (3.2x reduction)
    #   beta=0.08  → lag  8px, jitter 1.6px (2.6x reduction)
    #   beta=0.15  → lag  4px, jitter 1.9px (2.3x reduction)
    BETA_IDLE = 0.007       # heavy smoothing when hand is still
    BETA_DRAW = 0.08        # responsive when drawing (8px avg lag)
    BETA_TRANSITION = 0.12  # how fast beta ramps between modes (0-1)

    # velocity clamp - max pixels a landmark can move per frame
    MAX_VELOCITY = 180      # about 3 hand-widths per frame at 30fps

    # confidence scoring
    STABILITY_WINDOW = 10   # frames to average over
    JITTER_THRESHOLD = 4.0  # avg pixel movement below this = "stable"

    def __init__(self):
        self._prev_lm = None
        self._prev_time = None

        # one-euro state per landmark
        self._oe_x = {}    # filtered x per landmark id
        self._oe_y = {}    # filtered y per landmark id
        self._oe_dx = {}   # filtered dx per landmark id
        self._oe_dy = {}   # filtered dy per landmark id

        # stability tracking
        self._jitter_history = deque(maxlen=self.STABILITY_WINDOW)
        self._confidence = 0.0
        self._frames_seen = 0

        # dynamic beta — ramps between BETA_IDLE and BETA_DRAW
        self._current_beta = self.BETA_IDLE
        self._mode = "idle"  # "idle" or "draw"

    def reset(self):
        """call when the hand disappears so we start fresh next time"""
        self._prev_lm = None
        self._prev_time = None
        self._oe_x.clear()
        self._oe_y.clear()
        self._oe_dx.clear()
        self._oe_dy.clear()
        self._jitter_history.clear()
        self._confidence = 0.0
        self._frames_seen = 0
        self._current_beta = self.BETA_IDLE
        self._mode = "idle"

    @property
    def confidence(self):
        """
        0.0 to 1.0 score of how stable the hand tracking is right now.
        downstream code can use this to:
          - skip drawing on low confidence
          - widen debounce thresholds
          - show a "tracking unstable" hint
        """
        return self._confidence

    @property
    def is_stable(self):
        return self._confidence > 0.5

    @property
    def active_beta(self):
        """current effective beta value after dynamic tuning"""
        return self._current_beta

    def set_mode(self, mode):
        """
        tell the filter what the user is doing so it can pick the
        right beta. call this every frame from the main loop.

        mode="draw" → low lag, moderate smoothing (beta=0.08, ~8px lag)
        mode="idle" → heavy smoothing, higher lag ok (beta=0.007, ~64px lag)

        the transition is gradual (exponential ramp) so the filter
        doesn't snap between settings and cause visible jumps.
        """
        self._mode = mode
        target = self.BETA_DRAW if mode == "draw" else self.BETA_IDLE
        # exponential ramp toward target
        self._current_beta += (target - self._current_beta) * self.BETA_TRANSITION

    # --- public api ---

    def filter(self, landmarks):
        """
        takes raw landmark list [(id, x, y, z), ...], returns
        filtered version in the same format.

        returns None if the frame should be rejected entirely.
        """
        if landmarks is None:
            self.reset()
            return None

        now = time.monotonic()
        dt = 0.033  # assume ~30fps if we cant measure
        if self._prev_time is not None:
            dt = max(now - self._prev_time, 0.001)
        self._prev_time = now

        # build lookup from current frame
        current = {}
        for idx, x, y, z in landmarks:
            current[idx] = (float(x), float(y), float(z))

        # step 1: velocity clamping
        if self._prev_lm is not None:
            current = self._clamp_velocity(current, dt)

        # step 2: one-euro filter
        filtered = self._one_euro_pass(current, dt)

        # step 3: compute confidence from jitter
        self._update_confidence(current, filtered)

        # store for next frame
        self._prev_lm = filtered
        self._frames_seen += 1

        # convert back to list format
        result = []
        for idx in sorted(filtered.keys()):
            x, y, z = filtered[idx]
            result.append((idx, x, y, z))
        return result

    # --- internals ---

    def _clamp_velocity(self, current, dt):
        """
        if any landmark jumped more than MAX_VELOCITY pixels since
        last frame, pull it back. this catches the big spikes from
        mediapipe re-detecting a hand in a totally different position.
        """
        clamped = {}
        max_move = self.MAX_VELOCITY * (dt / 0.033)  # scale with dt

        for idx, (x, y, z) in current.items():
            if idx in self._prev_lm:
                px, py, _ = self._prev_lm[idx]
                dx = x - px
                dy = y - py
                dist = math.sqrt(dx * dx + dy * dy)

                if dist > max_move:
                    # clamp to max distance in same direction
                    scale = max_move / dist
                    x = px + dx * scale
                    y = py + dy * scale

            clamped[idx] = (x, y, z)
        return clamped

    def _one_euro_pass(self, current, dt):
        """
        one-euro filter on each landmark independently.

        this is an adaptive low-pass filter: when the hand moves slowly
        it smooths aggressively (removes jitter), when it moves fast
        it reduces smoothing (keeps up with motion). way better than
        a fixed moving average.

        paper: https://hal.inria.fr/hal-00670496/document
        """
        filtered = {}

        for idx, (x, y, z) in current.items():
            if idx not in self._oe_x:
                # first time seeing this landmark, init everything
                self._oe_x[idx] = x
                self._oe_y[idx] = y
                self._oe_dx[idx] = 0.0
                self._oe_dy[idx] = 0.0
                filtered[idx] = (x, y, z)
                continue

            # compute raw speed
            raw_dx = (x - self._oe_x[idx]) / dt
            raw_dy = (y - self._oe_y[idx]) / dt

            # filter the speed signal
            a_d = self._smoothing_factor(dt, self.D_CUTOFF)
            self._oe_dx[idx] = a_d * raw_dx + (1 - a_d) * self._oe_dx[idx]
            self._oe_dy[idx] = a_d * raw_dy + (1 - a_d) * self._oe_dy[idx]

            # adaptive cutoff based on speed — uses dynamic beta
            speed = math.sqrt(self._oe_dx[idx] ** 2 + self._oe_dy[idx] ** 2)
            cutoff = self.MIN_CUTOFF + self._current_beta * speed

            # filter the position
            a = self._smoothing_factor(dt, cutoff)
            fx = a * x + (1 - a) * self._oe_x[idx]
            fy = a * y + (1 - a) * self._oe_y[idx]

            self._oe_x[idx] = fx
            self._oe_y[idx] = fy
            filtered[idx] = (fx, fy, z)

        return filtered

    def _smoothing_factor(self, dt, cutoff):
        """compute the exponential smoothing factor from dt and cutoff freq"""
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def _update_confidence(self, raw, filtered):
        """
        track how much the raw landmarks differ from the filtered ones.
        high difference = noisy input = low confidence.
        """
        if not raw or not filtered:
            self._jitter_history.append(999.0)
            self._update_score()
            return

        total_jitter = 0.0
        count = 0
        for idx in raw:
            if idx in filtered:
                rx, ry, _ = raw[idx]
                fx, fy, _ = filtered[idx]
                total_jitter += math.sqrt((rx - fx) ** 2 + (ry - fy) ** 2)
                count += 1

        avg_jitter = total_jitter / count if count > 0 else 0.0
        self._jitter_history.append(avg_jitter)
        self._update_score()

    def _update_score(self):
        """
        convert recent jitter into a 0-1 confidence score.
        uses a sigmoid-ish curve so it degrades gracefully.
        """
        if not self._jitter_history:
            self._confidence = 0.0
            return

        avg = sum(self._jitter_history) / len(self._jitter_history)

        # ramp up during the first few frames so we dont start at 1.0
        warmup = min(self._frames_seen / 5.0, 1.0)

        # sigmoid mapping: jitter=0 -> 1.0, jitter=threshold -> 0.5
        # beyond threshold it drops toward 0
        ratio = avg / self.JITTER_THRESHOLD
        score = 1.0 / (1.0 + ratio * ratio)

        self._confidence = score * warmup

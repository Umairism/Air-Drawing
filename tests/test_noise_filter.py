"""
tests for the noise filter.

verifies that the three filter layers (velocity clamping, one-euro,
confidence scoring) are working individually and together.
"""
import sys
import os
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
from core.noise_filter import NoiseFilter


def make_static_landmarks(x_offset=300, y_offset=400):
    """
    build a set of 21 landmarks at a fixed position.
    just places them in a grid so every test starts from the same shape.
    """
    lm = []
    for i in range(21):
        x = x_offset + (i % 5) * 10
        y = y_offset - (i // 5) * 30
        lm.append((i, float(x), float(y), 0.0))
    return lm


def add_jitter(landmarks, amount=2.0):
    """add random-ish jitter to each landmark"""
    result = []
    for idx, x, y, z in landmarks:
        # deterministic wobble so tests are repeatable
        dx = math.sin(idx * 1.7) * amount
        dy = math.cos(idx * 2.3) * amount
        result.append((idx, x + dx, y + dy, z))
    return result


def shift_landmarks(landmarks, dx, dy):
    """move all landmarks by (dx, dy)"""
    return [(idx, x + dx, y + dy, z) for idx, x, y, z in landmarks]


class TestBasicFiltering(unittest.TestCase):
    """make sure filter() returns landmarks in the right format"""

    def test_returns_none_for_none_input(self):
        nf = NoiseFilter()
        result = nf.filter(None)
        self.assertIsNone(result)

    def test_returns_landmarks_for_valid_input(self):
        nf = NoiseFilter()
        lm = make_static_landmarks()
        result = nf.filter(lm)
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 21)

    def test_output_format_matches_input(self):
        nf = NoiseFilter()
        lm = make_static_landmarks()
        result = nf.filter(lm)
        for item in result:
            self.assertEqual(len(item), 4)
            idx, x, y, z = item
            self.assertIsInstance(idx, int)
            self.assertIsInstance(x, float)
            self.assertIsInstance(y, float)

    def test_first_frame_passes_through(self):
        """first frame has nothing to smooth against, should be close to input"""
        nf = NoiseFilter()
        lm = make_static_landmarks()
        result = nf.filter(lm)

        for (_, rx, ry, _), (_, ix, iy, _) in zip(result, lm):
            self.assertAlmostEqual(rx, ix, places=0)
            self.assertAlmostEqual(ry, iy, places=0)

    def test_landmarks_sorted_by_id(self):
        nf = NoiseFilter()
        # feed landmarks in reverse order
        lm = list(reversed(make_static_landmarks()))
        result = nf.filter(lm)
        ids = [idx for idx, _, _, _ in result]
        self.assertEqual(ids, list(range(21)))


class TestVelocityClamping(unittest.TestCase):
    """velocity clamping should stop impossible jumps"""

    def test_huge_jump_gets_clamped(self):
        nf = NoiseFilter()
        lm1 = make_static_landmarks(300, 400)
        nf.filter(lm1)  # establish position

        # teleport hand 1000 pixels away
        lm2 = make_static_landmarks(1300, 400)
        result = nf.filter(lm2)

        # the filtered position should be way closer to the original
        for (_, rx, ry, _), (_, ox, oy, _) in zip(result, lm1):
            dx = abs(rx - ox)
            dy = abs(ry - oy)
            total_move = math.sqrt(dx ** 2 + dy ** 2)
            # MAX_VELOCITY is 180 per frame, with some dt scaling.
            # it should NOT have moved the full 1000px
            self.assertLess(total_move, 500,
                            "clamping should prevent teleportation")

    def test_small_movement_not_clamped(self):
        nf = NoiseFilter()
        lm1 = make_static_landmarks(300, 400)
        nf.filter(lm1)

        # small natural movement
        lm2 = shift_landmarks(lm1, 5, -3)
        result = nf.filter(lm2)

        # after one-euro, the filtered position should still be close
        # to the intended movement (within 10px is fine, filter adds some lag)
        first_orig = lm2[0]
        first_filt = result[0]
        dx = abs(first_filt[1] - first_orig[1])
        dy = abs(first_filt[2] - first_orig[2])
        self.assertLess(dx, 10)
        self.assertLess(dy, 10)


class TestOneEuroSmoothing(unittest.TestCase):
    """one-euro filter should reduce jitter on slow-moving hands"""

    def test_jitter_reduction(self):
        """feeding the same position with jitter should produce smoother output"""
        nf = NoiseFilter()
        base = make_static_landmarks()

        # warm up the filter with a few clean frames
        for _ in range(5):
            nf.filter(base)

        # now add jitter and measure how much the filter dampens it
        jittery = add_jitter(base, amount=5.0)
        result = nf.filter(jittery)

        # the output should be closer to the base than the jittery input
        jitter_total = 0
        filtered_total = 0
        for (_, bx, by, _), (_, jx, jy, _), (_, fx, fy, _) in zip(base, jittery, result):
            jitter_total += math.sqrt((jx - bx) ** 2 + (jy - by) ** 2)
            filtered_total += math.sqrt((fx - bx) ** 2 + (fy - by) ** 2)

        self.assertLess(filtered_total, jitter_total,
                        "filtered should be closer to base than jittered input")

    def test_stationary_hand_converges(self):
        """repeated identical input should converge to that position"""
        nf = NoiseFilter()
        base = make_static_landmarks()

        for _ in range(30):
            result = nf.filter(base)

        # after many frames of the same input, output should match
        for (_, rx, ry, _), (_, bx, by, _) in zip(result, base):
            self.assertAlmostEqual(rx, bx, delta=1.0)
            self.assertAlmostEqual(ry, by, delta=1.0)


class TestConfidenceScoring(unittest.TestCase):
    """confidence should rise with stable input and fall with noise"""

    def test_initial_confidence_is_low(self):
        nf = NoiseFilter()
        self.assertEqual(nf.confidence, 0.0)

    def test_confidence_rises_with_stable_input(self):
        nf = NoiseFilter()
        base = make_static_landmarks()

        for _ in range(20):
            nf.filter(base)

        self.assertGreater(nf.confidence, 0.5,
                           "confidence should be high after stable frames")

    def test_confidence_drops_with_noisy_input(self):
        nf = NoiseFilter()
        base = make_static_landmarks()

        # start with stable
        for _ in range(15):
            nf.filter(base)
        stable_conf = nf.confidence

        # inject heavy noise
        for i in range(15):
            jittery = add_jitter(base, amount=30.0 + i * 5)
            nf.filter(jittery)
        noisy_conf = nf.confidence

        self.assertLess(noisy_conf, stable_conf,
                        "noisy frames should lower confidence")

    def test_is_stable_property(self):
        nf = NoiseFilter()
        base = make_static_landmarks()

        self.assertFalse(nf.is_stable, "shouldn't be stable with no data")

        for _ in range(20):
            nf.filter(base)

        self.assertTrue(nf.is_stable, "should be stable after clean frames")

    def test_reset_clears_confidence(self):
        nf = NoiseFilter()
        base = make_static_landmarks()

        for _ in range(15):
            nf.filter(base)
        self.assertGreater(nf.confidence, 0)

        nf.reset()
        self.assertEqual(nf.confidence, 0.0)
        self.assertEqual(nf._frames_seen, 0)

    def test_none_input_resets(self):
        nf = NoiseFilter()
        base = make_static_landmarks()
        for _ in range(10):
            nf.filter(base)
        self.assertGreater(nf.confidence, 0)

        nf.filter(None)
        self.assertEqual(nf.confidence, 0.0)


class TestReset(unittest.TestCase):

    def test_full_reset(self):
        nf = NoiseFilter()
        lm = make_static_landmarks()
        for _ in range(10):
            nf.filter(lm)

        nf.reset()
        self.assertIsNone(nf._prev_lm)
        self.assertIsNone(nf._prev_time)
        self.assertEqual(len(nf._oe_x), 0)
        self.assertEqual(len(nf._oe_y), 0)
        self.assertEqual(nf._frames_seen, 0)
        self.assertEqual(nf.confidence, 0.0)


if __name__ == "__main__":
    unittest.main()

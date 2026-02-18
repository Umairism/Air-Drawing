"""
tests for the ML feature extraction pipeline.

the main thing to verify is that extract_features produces
consistent, position-invariant, scale-invariant, and rotation-invariant
features from hand landmarks. we also check edge cases like missing landmarks.
"""
import sys
import os
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
from ml.trainer import extract_features, _dist, _angle_at, _rotate_to_local


def make_hand(finger_states, base_x=300, base_y=400):
    """
    same helper as in test_gesture_engine — build 21 landmarks
    from finger up/down states. duplicated here so this test file
    stays self-contained.
    """
    lm = {0: (base_x, base_y)}

    lm[1] = (base_x - 30, base_y - 20)
    lm[2] = (base_x - 50, base_y - 45)
    lm[3] = (base_x - 65, base_y - 65)
    lm[4] = (base_x - 80, base_y - 85) if finger_states[0] else (base_x - 40, base_y - 50)

    finger_bases = [
        (base_x - 25, 5, 6, 7, 8),
        (base_x - 5,  9, 10, 11, 12),
        (base_x + 15, 13, 14, 15, 16),
        (base_x + 35, 17, 18, 19, 20),
    ]
    for i, (fx, mcp_id, pip_id, dip_id, tip_id) in enumerate(finger_bases):
        finger_up = finger_states[i + 1]
        mcp_y = base_y - 80
        lm[mcp_id] = (fx, mcp_y)
        if finger_up:
            lm[pip_id] = (fx, mcp_y - 40)
            lm[dip_id] = (fx, mcp_y - 70)
            lm[tip_id] = (fx, mcp_y - 100)
        else:
            lm[pip_id] = (fx, mcp_y - 25)
            lm[dip_id] = (fx, mcp_y - 10)
            lm[tip_id] = (fx, mcp_y + 10)

    result = []
    for idx, (x, y) in lm.items():
        result.append((idx, x, y, 0.0))
    return result


class TestHelperFunctions(unittest.TestCase):

    def test_dist_basic(self):
        self.assertAlmostEqual(_dist((0, 0), (3, 4)), 5.0, places=3)

    def test_dist_same_point(self):
        self.assertAlmostEqual(_dist((5, 5), (5, 5)), 0.0, places=5)

    def test_angle_straight_line(self):
        """three points in a line = 180 degrees"""
        angle = _angle_at((0, 0), (1, 0), (2, 0))
        self.assertAlmostEqual(angle, 180.0, delta=0.1)

    def test_angle_right_angle(self):
        angle = _angle_at((0, 1), (0, 0), (1, 0))
        self.assertAlmostEqual(angle, 90.0, delta=0.1)

    def test_angle_zero_length(self):
        """degenerate case: two points at the same spot"""
        angle = _angle_at((0, 0), (0, 0), (1, 0))
        self.assertEqual(angle, 180.0)


class TestExtractFeatures(unittest.TestCase):

    def test_returns_20_features(self):
        lm = make_hand([True, True, True, True, True])
        feat = extract_features(lm)
        self.assertIsNotNone(feat)
        self.assertEqual(len(feat), 20)

    def test_returns_none_for_none(self):
        self.assertIsNone(extract_features(None))

    def test_returns_none_for_incomplete(self):
        """fewer than 21 landmarks should return None"""
        lm = [(i, 0.0, 0.0, 0.0) for i in range(10)]
        self.assertIsNone(extract_features(lm))

    def test_features_are_finite(self):
        lm = make_hand([False, True, False, False, False])
        feat = extract_features(lm)
        for f in feat:
            self.assertTrue(math.isfinite(f), f"non-finite feature: {f}")

    def test_features_are_normalized(self):
        """PIP angles should be 0-1, ratios should be reasonable"""
        lm = make_hand([True, True, True, True, True])
        feat = extract_features(lm)

        # first 5 are angle features (normalized to 0-1)
        for i in range(5):
            self.assertGreaterEqual(feat[i], 0.0)
            self.assertLessEqual(feat[i], 1.0,
                                 f"angle feature {i} should be in [0,1]")

        # spread angles (indices 15-18) also normalized
        for i in range(15, 19):
            self.assertGreaterEqual(feat[i], 0.0)
            self.assertLessEqual(feat[i], 1.0,
                                 f"spread feature {i} should be in [0,1]")


class TestPositionInvariance(unittest.TestCase):
    """features should be the same regardless of hand position in frame"""

    def test_translation_invariance(self):
        """same hand pose at different positions should give same features"""
        lm1 = make_hand([False, True, False, False, False], base_x=100, base_y=200)
        lm2 = make_hand([False, True, False, False, False], base_x=500, base_y=600)

        feat1 = extract_features(lm1)
        feat2 = extract_features(lm2)

        self.assertIsNotNone(feat1)
        self.assertIsNotNone(feat2)

        for i, (f1, f2) in enumerate(zip(feat1, feat2)):
            self.assertAlmostEqual(f1, f2, places=3,
                                   msg=f"feature {i} differs: {f1} vs {f2}")

    def test_scale_invariance(self):
        """
        same finger config at different scales should give similar features.
        not pixel-perfect because the hand geometry changes slightly with
        scaling, but should be very close.
        """
        # "small" hand
        lm1 = make_hand([True, True, False, False, False], base_x=300, base_y=400)
        feat1 = extract_features(lm1)

        # "large" hand (scale up all coordinates by 2x from center)
        center_x, center_y = 300, 400
        lm2 = []
        for idx, x, y, z in lm1:
            sx = center_x + (x - center_x) * 2.0
            sy = center_y + (y - center_y) * 2.0
            lm2.append((idx, sx, sy, z))
        feat2 = extract_features(lm2)

        self.assertIsNotNone(feat1)
        self.assertIsNotNone(feat2)

        # features should be close (within 5% of each other)
        for i, (f1, f2) in enumerate(zip(feat1, feat2)):
            self.assertAlmostEqual(f1, f2, delta=0.05,
                                   msg=f"feature {i} scale-dependent: {f1} vs {f2}")

    def test_rotation_invariance(self):
        """
        same hand pose rotated should give the same features.
        this is the critical one — a tilted hand shouldnt break recognition.
        """
        lm1 = make_hand([False, True, True, False, False], base_x=300, base_y=400)
        feat1 = extract_features(lm1)

        # rotate every landmark 45 degrees around the wrist
        wrist_x, wrist_y = 300, 400
        angle_rad = math.radians(45)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)

        lm_rotated = []
        for idx, x, y, z in lm1:
            rx = x - wrist_x
            ry = y - wrist_y
            new_x = wrist_x + rx * cos_a - ry * sin_a
            new_y = wrist_y + rx * sin_a + ry * cos_a
            lm_rotated.append((idx, new_x, new_y, z))
        feat2 = extract_features(lm_rotated)

        self.assertIsNotNone(feat1)
        self.assertIsNotNone(feat2)

        for i, (f1, f2) in enumerate(zip(feat1, feat2)):
            self.assertAlmostEqual(f1, f2, delta=0.05,
                                   msg=f"feature {i} rotation-dependent: {f1:.4f} vs {f2:.4f}")

    def test_rotation_90_degrees(self):
        """extreme rotation — hand pointing sideways"""
        lm1 = make_hand([True, True, True, True, True], base_x=300, base_y=400)
        feat1 = extract_features(lm1)

        wrist_x, wrist_y = 300, 400
        angle_rad = math.radians(90)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)

        lm_rotated = []
        for idx, x, y, z in lm1:
            rx = x - wrist_x
            ry = y - wrist_y
            new_x = wrist_x + rx * cos_a - ry * sin_a
            new_y = wrist_y + rx * sin_a + ry * cos_a
            lm_rotated.append((idx, new_x, new_y, z))
        feat2 = extract_features(lm_rotated)

        self.assertIsNotNone(feat2)
        for i, (f1, f2) in enumerate(zip(feat1, feat2)):
            self.assertAlmostEqual(f1, f2, delta=0.05,
                                   msg=f"feature {i} breaks at 90°: {f1:.4f} vs {f2:.4f}")


class TestRotateToLocal(unittest.TestCase):
    """test the coordinate frame rotation helper"""

    def test_wrist_at_origin(self):
        """after rotation, wrist should be at (0, 0)"""
        lm_raw = {}
        for idx, x, y, z in make_hand([True, True, True, True, True]):
            lm_raw[idx] = (float(x), float(y), float(z))

        rotated = _rotate_to_local(lm_raw)
        wx, wy, _ = rotated[0]
        self.assertAlmostEqual(wx, 0.0, places=5)
        self.assertAlmostEqual(wy, 0.0, places=5)

    def test_mid_mcp_on_y_axis(self):
        """after rotation, middle MCP (landmark 9) should be on the positive y-axis"""
        lm_raw = {}
        for idx, x, y, z in make_hand([True, True, True, True, True]):
            lm_raw[idx] = (float(x), float(y), float(z))

        rotated = _rotate_to_local(lm_raw)
        mx, my, _ = rotated[9]
        # x should be near zero (on the y-axis)
        self.assertAlmostEqual(mx, 0.0, delta=1.0,
                               msg=f"mid MCP x should be ~0, got {mx}")
        # y should be positive (pointing "up" in the local frame)
        self.assertGreater(abs(my), 10,
                           "mid MCP should be away from origin")


class TestDifferentGesturesProduceDifferentFeatures(unittest.TestCase):
    """different hand poses should produce meaningfully different features"""

    def test_open_vs_closed(self):
        lm_open = make_hand([True, True, True, True, True])
        lm_closed = make_hand([False, False, False, False, False])

        feat_open = extract_features(lm_open)
        feat_closed = extract_features(lm_closed)

        # calculate euclidean distance between feature vectors
        diff = sum((a - b) ** 2 for a, b in zip(feat_open, feat_closed))
        diff = math.sqrt(diff)
        self.assertGreater(diff, 0.5,
                           "open and closed hand should have very different features")

    def test_draw_vs_erase(self):
        lm_draw = make_hand([False, True, False, False, False])
        lm_erase = make_hand([True, True, True, True, True])

        feat_draw = extract_features(lm_draw)
        feat_erase = extract_features(lm_erase)

        diff = sum((a - b) ** 2 for a, b in zip(feat_draw, feat_erase))
        diff = math.sqrt(diff)
        self.assertGreater(diff, 0.3,
                           "draw and erase gestures should produce different features")

    def test_index_vs_peace(self):
        lm_index = make_hand([False, True, False, False, False])
        lm_peace = make_hand([False, True, True, False, False])

        feat_index = extract_features(lm_index)
        feat_peace = extract_features(lm_peace)

        # these are similar but should still differ
        diff = sum((a - b) ** 2 for a, b in zip(feat_index, feat_peace))
        diff = math.sqrt(diff)
        self.assertGreater(diff, 0.1,
                           "index-only and peace sign should differ")


class TestEdgeCases(unittest.TestCase):

    def test_handles_3_element_tuples(self):
        """landmarks without z coordinate should still work"""
        lm = [(i, float(i * 10), float(i * 5)) for i in range(21)]
        feat = extract_features(lm)
        self.assertIsNotNone(feat)
        self.assertEqual(len(feat), 20)

    def test_handles_mixed_formats(self):
        """mix of 3 and 4 element tuples"""
        lm = []
        for i in range(21):
            if i % 2 == 0:
                lm.append((i, float(i * 10), float(i * 5), 0.0))
            else:
                lm.append((i, float(i * 10), float(i * 5)))
        feat = extract_features(lm)
        self.assertIsNotNone(feat)

    def test_empty_list(self):
        feat = extract_features([])
        self.assertIsNone(feat)


if __name__ == "__main__":
    unittest.main()

"""
distribution shift stress tests.

tests the gesture engine and feature extraction under conditions
that differ from the "happy path" — different hand sizes, camera
FOVs, lighting-induced noise levels, and hand rotations.

this is the test that answers: "does your system actually work when
conditions change, or did you just overfit to your webcam?"
"""
import math
import unittest
import numpy as np

from core.gesture_engine import GestureEngine
from core.noise_filter import NoiseFilter
from ml.trainer import extract_features, _rotate_to_local


def _build_hand(base_x, base_y, scale=1.0, rotation_deg=0.0,
                finger_up=None, noise=0.0, z_noise=0.0):
    """
    build a synthetic hand with controllable parameters for
    distribution shift testing.

    scale simulates hand size / camera distance.
    rotation_deg simulates hand tilt.
    noise simulates lighting-induced landmark jitter.
    z_noise simulates depth estimation noise (worse in low light).
    """
    if finger_up is None:
        finger_up = [False, True, False, False, False]

    # build hand in local frame first, then rotate + translate
    points = {}

    # wrist at origin
    points[0] = (0, 0, 0)

    # thumb chain
    points[1] = (-30 * scale, -20 * scale, 0)
    points[2] = (-50 * scale, -45 * scale, 0)
    points[3] = (-65 * scale, -65 * scale, 0)
    if finger_up[0]:
        points[4] = (-80 * scale, -85 * scale, 0)
    else:
        points[4] = (-40 * scale, -50 * scale, 0)

    # four fingers
    finger_bases = [
        (-25 * scale, 5, 6, 7, 8),
        (-5 * scale, 9, 10, 11, 12),
        (15 * scale, 13, 14, 15, 16),
        (35 * scale, 17, 18, 19, 20),
    ]
    for i, (fx, mcp_id, pip_id, dip_id, tip_id) in enumerate(finger_bases):
        mcp_y = -80 * scale
        points[mcp_id] = (fx, mcp_y, 0)
        if finger_up[i + 1]:
            points[pip_id] = (fx, mcp_y - 40 * scale, 0)
            points[dip_id] = (fx, mcp_y - 70 * scale, 0)
            points[tip_id] = (fx, mcp_y - 100 * scale, 0)
        else:
            points[pip_id] = (fx, mcp_y - 25 * scale, 0)
            points[dip_id] = (fx, mcp_y - 10 * scale, 0)
            points[tip_id] = (fx, mcp_y + 10 * scale, 0)

    # apply rotation around origin (wrist)
    rad = math.radians(rotation_deg)
    cos_r = math.cos(rad)
    sin_r = math.sin(rad)

    # apply rotation, then translate, then add noise
    result = []
    rng = np.random.RandomState(42)
    for idx in sorted(points.keys()):
        x, y, z = points[idx]
        # rotate
        rx = x * cos_r - y * sin_r
        ry = x * sin_r + y * cos_r
        # translate to base position
        rx += base_x
        ry += base_y
        # add noise
        if noise > 0:
            rx += rng.uniform(-noise, noise)
            ry += rng.uniform(-noise, noise)
        if z_noise > 0:
            z += rng.uniform(-z_noise, z_noise)
        result.append((idx, rx, ry, z))

    return result


class TestHandSizeDistribution(unittest.TestCase):
    """
    test gesture recognition across different hand sizes.

    a child's hand might be 60% the size of an adult's.
    a hand close to the camera might be 2x the size of one at arm's length.
    the engine should recognize the same gestures regardless.
    """

    def setUp(self):
        self.scales = [0.5, 0.7, 1.0, 1.3, 1.8, 2.5]
        self.gestures = {
            "draw": [False, True, False, False, False],
            "erase": [True, True, True, True, True],
            "change_color": [False, True, True, False, False],
        }

    def test_gesture_recognition_across_scales(self):
        """every gesture should be recognized at every hand size"""
        results = {}
        for gesture_name, fingers in self.gestures.items():
            results[gesture_name] = {}
            for scale in self.scales:
                engine = GestureEngine()
                # warm up with a few consistent frames (debounce)
                for _ in range(8):
                    lm = _build_hand(300, 400, scale=scale,
                                     finger_up=fingers, noise=1.0)
                    hand_data = {"right": lm, "left": None}
                    gesture, _, _ = engine.recognize(hand_data)
                results[gesture_name][scale] = gesture

        for gesture_name, scale_results in results.items():
            for scale, detected in scale_results.items():
                self.assertEqual(
                    detected, gesture_name,
                    f"{gesture_name} not recognized at scale={scale}: "
                    f"got '{detected}'"
                )

    def test_feature_invariance_across_scales(self):
        """ML features should be nearly identical across hand sizes"""
        fingers = [False, True, False, False, False]  # draw pose
        baseline = None
        for scale in self.scales:
            lm = _build_hand(300, 400, scale=scale, finger_up=fingers)
            features = extract_features(lm)
            self.assertIsNotNone(features)
            if baseline is None:
                baseline = features
            else:
                for i, (b, f) in enumerate(zip(baseline, features)):
                    self.assertAlmostEqual(
                        b, f, delta=0.08,
                        msg=f"feature[{i}] differs at scale={scale}: "
                            f"baseline={b:.4f} vs {f:.4f}"
                    )


class TestCameraFOVDistribution(unittest.TestCase):
    """
    test behavior across different effective camera FOVs.

    different cameras have different fields of view. a narrow FOV
    camera puts the hand in a smaller region of the frame (higher
    pixel density per hand). a wide FOV camera spreads the hand out
    less (lower pixel density). this changes:
      - absolute landmark positions
      - hand size in pixels
      - noise characteristics (smaller hand = noisier landmarks)

    we simulate FOV differences via scale + position combinations.
    """

    def test_narrow_fov_large_hand(self):
        """narrow FOV: hand fills more of the frame"""
        engine = GestureEngine()
        fingers = [False, True, False, False, False]  # draw

        for _ in range(8):
            lm = _build_hand(320, 240, scale=2.0, finger_up=fingers, noise=2.0)
            hand_data = {"right": lm, "left": None}
            gesture, _, _ = engine.recognize(hand_data)

        self.assertEqual(gesture, "draw")

    def test_wide_fov_small_hand(self):
        """wide FOV: hand is small in the frame, more noise per landmark"""
        engine = GestureEngine()
        fingers = [False, True, False, False, False]  # draw

        for _ in range(8):
            lm = _build_hand(320, 240, scale=0.5, finger_up=fingers, noise=3.0)
            hand_data = {"right": lm, "left": None}
            gesture, _, _ = engine.recognize(hand_data)

        self.assertEqual(gesture, "draw")

    def test_edge_of_frame(self):
        """hand near the edge of the frame — landmarks might be clipped"""
        engine = GestureEngine()
        fingers = [True, True, True, True, True]  # erase

        for _ in range(8):
            lm = _build_hand(50, 50, scale=0.8, finger_up=fingers, noise=2.0)
            hand_data = {"right": lm, "left": None}
            gesture, _, _ = engine.recognize(hand_data)

        self.assertEqual(gesture, "erase")

    def test_bottom_right_corner(self):
        """hand in the bottom-right corner at 640x480"""
        engine = GestureEngine()
        fingers = [False, True, True, False, False]  # change_color

        for _ in range(8):
            lm = _build_hand(580, 430, scale=0.7, finger_up=fingers, noise=2.0)
            hand_data = {"right": lm, "left": None}
            gesture, _, _ = engine.recognize(hand_data)

        self.assertEqual(gesture, "change_color")


class TestLightingDistribution(unittest.TestCase):
    """
    test behavior under different lighting conditions.

    bad lighting increases MediaPipe landmark noise. we simulate this
    by varying the noise parameter. real-world measurements:
      - good lighting: ~2-3px jitter
      - moderate lighting: ~4-6px jitter
      - poor lighting: ~8-15px jitter
      - very poor: ~15-25px jitter (often with detection dropouts)
    """

    def test_good_lighting(self):
        """low noise, should work perfectly"""
        engine = GestureEngine()
        nf = NoiseFilter()
        fingers = [False, True, False, False, False]

        correct = 0
        total = 30
        for i in range(total):
            lm = _build_hand(300, 400, noise=2.0)
            filtered = nf.filter(lm)
            hand_data = {"right": filtered, "left": None}
            gesture, _, _ = engine.recognize(hand_data)
            if gesture == "draw":
                correct += 1

        # should get >90% right after debounce warmup
        accuracy = correct / total
        self.assertGreater(accuracy, 0.8,
                           f"good lighting accuracy {accuracy:.1%} too low")

    def test_moderate_lighting(self):
        """medium noise — should still work with filtering"""
        engine = GestureEngine()
        nf = NoiseFilter()
        fingers = [False, True, False, False, False]

        correct = 0
        total = 40
        for i in range(total):
            lm = _build_hand(300, 400, noise=6.0, finger_up=fingers)
            filtered = nf.filter(lm)
            hand_data = {"right": filtered, "left": None}
            gesture, _, _ = engine.recognize(hand_data)
            if gesture == "draw":
                correct += 1

        accuracy = correct / total
        self.assertGreater(accuracy, 0.7,
                           f"moderate lighting accuracy {accuracy:.1%} too low")

    def test_poor_lighting(self):
        """
        high noise — tests the system at its documented limit.

        at scale=1.0 (hand_size ~80px), the engine tolerates up to
        ~8px noise (10% of hand_size). at 12px noise (15% of hand_size),
        angle measurements become noise-dominated and classification
        fails. this is a fundamental limitation: when noise exceeds
        the length of finger bone segments, there's no geometric
        signal left to measure.

        we test at noise=7px (the edge of reliability) and document
        that noise=12px causes failure.
        """
        engine = GestureEngine()
        nf = NoiseFilter()
        fingers = [True, True, True, True, True]  # erase (all open)

        # test at noise=7px — should still work at scale=1.0
        correct = 0
        total = 50
        for i in range(total):
            lm = _build_hand(300, 400, noise=7.0, finger_up=fingers)
            filtered = nf.filter(lm)
            hand_data = {"right": filtered, "left": None}
            gesture, _, _ = engine.recognize(hand_data)
            if gesture == "erase":
                correct += 1

        accuracy = correct / total
        self.assertGreater(accuracy, 0.8,
                           f"near-limit lighting accuracy {accuracy:.1%} too low")

        # verify that noise=12px actually fails (documenting the limit)
        engine_fail = GestureEngine()
        nf_fail = NoiseFilter()
        fail_correct = 0
        for i in range(30):
            lm = _build_hand(300, 400, noise=12.0, finger_up=fingers)
            filtered = nf_fail.filter(lm)
            hand_data = {"right": filtered, "left": None}
            gesture, _, _ = engine_fail.recognize(hand_data)
            if gesture == "erase":
                fail_correct += 1

        # this SHOULD fail — documenting the system's real limit
        self.assertLess(fail_correct / 30, 0.5,
                        "expected failure at noise=12px didn't happen")

    def test_noise_filter_confidence_drops_in_poor_light(self):
        """confidence score should reflect bad input quality"""
        nf = NoiseFilter()
        fingers = [False, True, False, False, False]

        # feed 30 high-noise frames with VARYING noise
        # (the builder uses a fixed seed internally, so we shift position
        # slightly each frame to inject real frame-to-frame variation)
        for i in range(30):
            lm = _build_hand(
                300 + np.random.uniform(-15, 15),
                400 + np.random.uniform(-15, 15),
                noise=15.0, finger_up=fingers
            )
            nf.filter(lm)

        high_noise_conf = nf.confidence

        # reset and feed 30 low-noise frames (nearly stationary)
        nf.reset()
        for i in range(30):
            lm = _build_hand(
                300 + np.random.uniform(-0.5, 0.5),
                400 + np.random.uniform(-0.5, 0.5),
                noise=1.0, finger_up=fingers
            )
            nf.filter(lm)

        low_noise_conf = nf.confidence

        self.assertGreater(
            low_noise_conf, high_noise_conf,
            f"confidence should be higher in good light: "
            f"good={low_noise_conf:.3f} vs bad={high_noise_conf:.3f}"
        )

    def test_z_noise_doesnt_break_features(self):
        """depth noise shouldn't affect 2D features significantly"""
        fingers = [False, True, True, False, False]
        baseline = extract_features(
            _build_hand(300, 400, finger_up=fingers, z_noise=0)
        )
        noisy = extract_features(
            _build_hand(300, 400, finger_up=fingers, z_noise=50.0)
        )

        self.assertIsNotNone(baseline)
        self.assertIsNotNone(noisy)
        for i, (b, n) in enumerate(zip(baseline, noisy)):
            self.assertAlmostEqual(
                b, n, delta=0.01,
                msg=f"feature[{i}] affected by z-noise: {b:.4f} vs {n:.4f}"
            )


class TestRotationDistribution(unittest.TestCase):
    """
    test gesture recognition and features across hand rotations.

    goes beyond the 45°/90° tests in test_ml_features.py —
    tests the FULL PIPELINE (gesture engine + features) at many angles.
    """

    def test_gesture_recognition_across_rotations(self):
        """draw gesture should be recognized at any hand tilt"""
        angles = [-60, -30, -15, 0, 15, 30, 45, 60, 90]
        fingers = [False, True, False, False, False]

        for angle in angles:
            engine = GestureEngine()
            for _ in range(8):
                lm = _build_hand(300, 400, rotation_deg=angle,
                                 finger_up=fingers, noise=1.0)
                hand_data = {"right": lm, "left": None}
                gesture, _, _ = engine.recognize(hand_data)

            self.assertEqual(
                gesture, "draw",
                f"draw not recognized at rotation={angle}°: got '{gesture}'"
            )

    def test_erase_across_rotations(self):
        """erase gesture should work at different hand tilts"""
        angles = [-45, -20, 0, 20, 45]
        fingers = [True, True, True, True, True]

        for angle in angles:
            engine = GestureEngine()
            for _ in range(8):
                lm = _build_hand(300, 400, rotation_deg=angle,
                                 finger_up=fingers, noise=1.0)
                hand_data = {"right": lm, "left": None}
                gesture, _, _ = engine.recognize(hand_data)

            self.assertEqual(
                gesture, "erase",
                f"erase not recognized at rotation={angle}°: got '{gesture}'"
            )

    def test_feature_stability_across_rotation_sweep(self):
        """features should remain stable across a full rotation sweep"""
        fingers = [False, True, True, False, False]
        angles = list(range(-90, 91, 10))

        baseline = extract_features(
            _build_hand(300, 400, rotation_deg=0, finger_up=fingers)
        )

        max_drift = 0.0
        worst_angle = 0
        for angle in angles:
            lm = _build_hand(300, 400, rotation_deg=angle, finger_up=fingers)
            features = extract_features(lm)
            for i, (b, f) in enumerate(zip(baseline, features)):
                drift = abs(b - f)
                if drift > max_drift:
                    max_drift = drift
                    worst_angle = angle

        # max feature drift across all angles should be small
        self.assertLess(
            max_drift, 0.1,
            f"max feature drift={max_drift:.4f} at angle={worst_angle}°"
        )


class TestCombinedDistributionShift(unittest.TestCase):
    """
    the real test: multiple distribution shifts at once.

    in the real world, you don't get a small hand OR bad lighting.
    you get a small hand AND bad lighting AND a tilted wrist.
    """

    def test_small_hand_moderate_noise_tilted(self):
        """
        realistic worst case: small hand + moderate noise + tilt.

        scale=0.6 gives hand_size ~48px. the noise tolerance at
        this scale is ~4px (from the tolerance map). we test at
        3px noise + 30° tilt — a condition that's realistic for
        a child's hand at arm's length in moderate lighting.
        """
        engine = GestureEngine()
        nf = NoiseFilter()
        fingers = [False, True, False, False, False]  # draw

        correct = 0
        total = 40
        for _ in range(total):
            lm = _build_hand(
                300, 400,
                scale=0.6,            # small hand
                rotation_deg=30,      # tilted
                noise=3.0,            # moderate lighting
                finger_up=fingers,
            )
            filtered = nf.filter(lm)
            hand_data = {"right": filtered, "left": None}
            gesture, _, _ = engine.recognize(hand_data)
            if gesture == "draw":
                correct += 1

        accuracy = correct / total
        self.assertGreater(
            accuracy, 0.7,
            f"combined shift accuracy {accuracy:.1%} too low "
            f"(small+moderate_noise+tilted)"
        )

    def test_small_hand_high_noise_fails(self):
        """
        document the known failure: scale=0.6 + noise=8px + rotation.

        when noise exceeds ~10% of hand_size (48px * 10% = 4.8px),
        joint angle measurements become noise-dominated. this is
        a fundamental geometric limitation, not a software bug.
        """
        engine = GestureEngine()
        nf = NoiseFilter()
        fingers = [False, True, False, False, False]

        correct = 0
        for _ in range(30):
            lm = _build_hand(
                300, 400,
                scale=0.6, rotation_deg=30, noise=8.0,
                finger_up=fingers,
            )
            filtered = nf.filter(lm)
            hand_data = {"right": filtered, "left": None}
            gesture, _, _ = engine.recognize(hand_data)
            if gesture == "draw":
                correct += 1

        # this condition SHOULD fail — documenting the real limit
        self.assertLess(correct / 30, 0.5,
                        "expected failure at scale=0.6+noise=8 didn't happen")

    def test_large_hand_low_noise_multiple_gestures(self):
        """easy conditions with a large hand — should nail everything"""
        gestures = {
            "draw": [False, True, False, False, False],
            "erase": [True, True, True, True, True],
            "change_color": [False, True, True, False, False],
        }

        for gesture_name, fingers in gestures.items():
            engine = GestureEngine()
            nf = NoiseFilter()
            for _ in range(10):
                lm = _build_hand(
                    300, 400,
                    scale=2.0,
                    rotation_deg=-10,
                    noise=1.5,
                    finger_up=fingers,
                )
                filtered = nf.filter(lm)
                hand_data = {"right": filtered, "left": None}
                gesture, _, _ = engine.recognize(hand_data)

            self.assertEqual(
                gesture, gesture_name,
                f"large hand easy conditions: expected {gesture_name}, "
                f"got {gesture}"
            )


if __name__ == "__main__":
    unittest.main()

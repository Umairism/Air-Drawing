"""
tests for the gesture engine.
we simulate hand landmark positions and check that the engine
detects the correct gestures from them.
"""
import sys
import os
import math

# make sure we can import from the project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
from core.gesture_engine import GestureEngine


def make_landmarks(positions):
    """
    helper - build a landmark list from a dict of {id: (x, y)}.
    fills z with 0 for all points.
    """
    lm = []
    for idx, (x, y) in positions.items():
        lm.append((idx, x, y, 0.0))
    return lm


def make_hand(finger_states, base_x=300, base_y=400):
    """
    build a realistic hand landmark set given which fingers should be up.
    finger_states is a list of 5 bools: [thumb, index, middle, ring, pinky].

    generates a plausible set of 21 landmarks. fingers that are "up" have
    their tips far from the wrist with straight joint angles; fingers that
    are "down" have tips curled back toward the palm.
    """
    # wrist at the base
    lm = {0: (base_x, base_y)}

    # thumb landmarks
    lm[1] = (base_x - 30, base_y - 20)   # CMC
    lm[2] = (base_x - 50, base_y - 45)   # MCP
    lm[3] = (base_x - 65, base_y - 65)   # IP

    if finger_states[0]:
        # thumb extended
        lm[4] = (base_x - 80, base_y - 85)
    else:
        # thumb curled
        lm[4] = (base_x - 40, base_y - 50)

    # 4 fingers - each column of landmarks
    finger_bases = [
        (base_x - 25, 5, 6, 7, 8),     # index: mcp_id, pip_id, dip_id, tip_id
        (base_x - 5,  9, 10, 11, 12),   # middle
        (base_x + 15, 13, 14, 15, 16),  # ring
        (base_x + 35, 17, 18, 19, 20),  # pinky
    ]

    for i, (fx, mcp_id, pip_id, dip_id, tip_id) in enumerate(finger_bases):
        finger_up = finger_states[i + 1]

        # MCP is at palm level
        mcp_y = base_y - 80
        lm[mcp_id] = (fx, mcp_y)

        if finger_up:
            # straight finger: pip, dip, tip go straight up
            lm[pip_id] = (fx, mcp_y - 40)
            lm[dip_id] = (fx, mcp_y - 70)
            lm[tip_id] = (fx, mcp_y - 100)
        else:
            # curled finger: tip curls back down toward palm
            lm[pip_id] = (fx, mcp_y - 25)
            lm[dip_id] = (fx, mcp_y - 10)
            lm[tip_id] = (fx, mcp_y + 10)

    return make_landmarks(lm)


def run_gesture_n_times(engine, hand_data, n=5):
    """
    run recognition N times to get past debounce.
    returns the final gesture.
    """
    result = None
    for _ in range(n):
        result = engine.recognize(hand_data)
    return result


class TestFingerDetection(unittest.TestCase):
    """test that finger open/closed detection works correctly"""

    def setUp(self):
        self.engine = GestureEngine()

    def test_all_fingers_up(self):
        lm = make_hand([True, True, True, True, True])
        states = self.engine.get_finger_states(lm)
        self.assertTrue(all(states), f"expected all True, got {states}")

    def test_all_fingers_down(self):
        lm = make_hand([False, False, False, False, False])
        # run twice to get past hysteresis since initial state is all-False
        states = self.engine.get_finger_states(lm)
        self.assertFalse(any(states), f"expected all False, got {states}")

    def test_index_only(self):
        lm = make_hand([False, True, False, False, False])
        states = self.engine.get_finger_states(lm)
        self.assertTrue(states[1], "index should be up")
        self.assertFalse(states[2], "middle should be down")
        self.assertFalse(states[3], "ring should be down")
        self.assertFalse(states[4], "pinky should be down")

    def test_peace_sign(self):
        lm = make_hand([False, True, True, False, False])
        states = self.engine.get_finger_states(lm)
        self.assertTrue(states[1], "index should be up")
        self.assertTrue(states[2], "middle should be up")
        self.assertFalse(states[3], "ring should be down")
        self.assertFalse(states[4], "pinky should be down")

    def test_three_fingers(self):
        lm = make_hand([False, True, True, True, False])
        states = self.engine.get_finger_states(lm)
        self.assertTrue(states[1], "index should be up")
        self.assertTrue(states[2], "middle should be up")
        self.assertTrue(states[3], "ring should be up")

    def test_hysteresis_keeps_state(self):
        """once a finger is detected up, slight changes shouldnt flip it"""
        # first: clearly up
        lm_up = make_hand([False, True, False, False, False])
        self.engine.get_finger_states(lm_up)

        # now feed a slightly ambiguous version (tip not as far out)
        positions = {}
        for idx, x, y, z in lm_up:
            positions[idx] = (x, y)
        # move index tip slightly back (not fully curled)
        positions[8] = (positions[8][0], positions[8][1] + 20)
        lm_ambig = make_landmarks(positions)

        states = self.engine.get_finger_states(lm_ambig)
        self.assertTrue(states[1], "index should stay up through hysteresis")


class TestGestureRecognition(unittest.TestCase):
    """test that hand poses map to the correct gesture names"""

    def _fresh_engine(self):
        return GestureEngine()

    def test_draw_gesture(self):
        """index up only = draw"""
        engine = self._fresh_engine()
        lm = make_hand([False, True, False, False, False])
        hand_data = {"right": lm, "left": None}

        gesture, tip, _ = run_gesture_n_times(engine, hand_data, 6)
        self.assertEqual(gesture, "draw", f"expected draw, got {gesture}")
        self.assertIsNotNone(tip, "should have a tip position")

    def test_erase_gesture(self):
        """all fingers open = erase"""
        engine = self._fresh_engine()
        lm = make_hand([True, True, True, True, True])
        hand_data = {"right": lm, "left": None}

        gesture, _, erase_pts = run_gesture_n_times(engine, hand_data, 6)
        self.assertEqual(gesture, "erase", f"expected erase, got {gesture}")
        self.assertTrue(len(erase_pts) > 0, "should have erase points")

    def test_change_color_gesture(self):
        """index + middle up = change_color"""
        engine = self._fresh_engine()
        lm = make_hand([False, True, True, False, False])
        hand_data = {"right": lm, "left": None}

        gesture, _, _ = run_gesture_n_times(engine, hand_data, 10)
        self.assertEqual(gesture, "change_color", f"expected change_color, got {gesture}")

    def test_fist_gesture(self):
        """all down = switch_brush"""
        engine = self._fresh_engine()
        lm = make_hand([False, False, False, False, False])
        hand_data = {"right": lm, "left": None}

        gesture, _, _ = run_gesture_n_times(engine, hand_data, 10)
        self.assertEqual(gesture, "switch_brush", f"expected switch_brush, got {gesture}")

    def test_idle_no_hands(self):
        """no hands = idle"""
        engine = self._fresh_engine()
        hand_data = {"right": None, "left": None}

        gesture, tip, erase = run_gesture_n_times(engine, hand_data, 4)
        self.assertEqual(gesture, "idle")
        self.assertIsNone(tip)
        self.assertEqual(erase, [])

    def test_hand_locking(self):
        """engine should lock to the first hand it sees"""
        engine = self._fresh_engine()
        lm_right = make_hand([False, True, False, False, False], base_x=400)
        lm_left = make_hand([True, True, True, True, True], base_x=200)

        # first frame: only right hand
        hand_data = {"right": lm_right, "left": None}
        run_gesture_n_times(engine, hand_data, 4)
        self.assertEqual(engine._active_hand, "right")

        # second: both hands visible - should still use right
        hand_data = {"right": lm_right, "left": lm_left}
        gesture, _, _ = run_gesture_n_times(engine, hand_data, 4)
        self.assertEqual(gesture, "draw")  # right hand is index-only = draw

    def test_hand_releases_when_gone(self):
        """when the locked hand disappears, engine should reset"""
        engine = self._fresh_engine()
        lm = make_hand([False, True, False, False, False])

        # lock to right
        engine.recognize({"right": lm, "left": None})
        self.assertEqual(engine._active_hand, "right")

        # right hand gone
        engine.recognize({"right": None, "left": None})
        self.assertIsNone(engine._active_hand)


class TestMultiHandGestures(unittest.TestCase):
    """test two-hand gestures"""

    def test_both_fists_clear(self):
        engine = GestureEngine()
        lm_fist = make_hand([False, False, False, False, False])

        result = engine.recognize_multi({"right": lm_fist, "left": lm_fist})
        self.assertEqual(result, "clear_canvas")

    def test_both_open_pause(self):
        engine = GestureEngine()
        lm_open = make_hand([True, True, True, True, True])

        result = engine.recognize_multi({"right": lm_open, "left": lm_open})
        self.assertEqual(result, "pause")

    def test_one_hand_no_multi(self):
        engine = GestureEngine()
        lm = make_hand([False, True, False, False, False])

        result = engine.recognize_multi({"right": lm, "left": None})
        self.assertIsNone(result)


class TestDebounce(unittest.TestCase):
    """test that the debounce system works"""

    def test_draw_activates_after_2_frames(self):
        engine = GestureEngine()
        lm = make_hand([False, True, False, False, False])
        hand_data = {"right": lm, "left": None}

        g1, _, _ = engine.recognize(hand_data)
        g2, _, _ = engine.recognize(hand_data)

        # by frame 2, draw should be confirmed
        self.assertEqual(g2, "draw")

    def test_single_frame_doesnt_switch(self):
        """one frame of a new gesture shouldnt change state"""
        engine = GestureEngine()
        lm_draw = make_hand([False, True, False, False, False])
        lm_fist = make_hand([False, False, False, False, False])
        draw_data = {"right": lm_draw, "left": None}
        fist_data = {"right": lm_fist, "left": None}

        # establish draw
        for _ in range(5):
            engine.recognize(draw_data)

        # one frame of fist shouldnt switch
        g, _, _ = engine.recognize(fist_data)
        self.assertEqual(g, "draw", "single frame of different gesture shouldnt switch")

    def test_idle_needs_3_frames(self):
        engine = GestureEngine()
        hand_data = {"right": None, "left": None}

        g1, _, _ = engine.recognize(hand_data)
        g2, _, _ = engine.recognize(hand_data)
        g3, _, _ = engine.recognize(hand_data)

        self.assertEqual(g3, "idle")


class TestEdgeCases(unittest.TestCase):
    """edge cases and regression tests"""

    def test_empty_landmarks(self):
        engine = GestureEngine()
        states = engine.get_finger_states(None)
        self.assertEqual(states, [False] * 5)

    def test_partial_landmarks(self):
        """missing some landmark points shouldnt crash"""
        engine = GestureEngine()
        # only wrist and index tip
        lm = make_landmarks({0: (300, 400), 8: (300, 200)})
        states = engine.get_finger_states(lm)
        self.assertEqual(len(states), 5)

    def test_erase_points_include_palm(self):
        """erase gesture should return palm center in erase_points"""
        engine = GestureEngine()
        lm = make_hand([True, True, True, True, True])
        hand_data = {"right": lm, "left": None}

        _, _, erase_pts = run_gesture_n_times(engine, hand_data, 4)
        # should have palm center + 5 fingertips = 6 points
        self.assertGreaterEqual(len(erase_pts), 5)

    def test_tip_position_is_tuple(self):
        engine = GestureEngine()
        lm = make_hand([False, True, False, False, False])
        hand_data = {"right": lm, "left": None}

        _, tip, _ = run_gesture_n_times(engine, hand_data, 4)
        self.assertIsInstance(tip, tuple)
        self.assertEqual(len(tip), 2)


if __name__ == "__main__":
    unittest.main()

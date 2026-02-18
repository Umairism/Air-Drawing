"""
tests for the performance profiler.

we check that timing, FPS calculation, and the summary output
all work correctly.
"""
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
from core.profiler import Profiler


class TestProfilerBasics(unittest.TestCase):

    def test_disabled_profiler_does_nothing(self):
        p = Profiler(enabled=False)
        p.begin_frame()
        p.start("detection")
        p.stop("detection")
        p.end_frame()

        self.assertEqual(p.get_fps(), 0.0)
        self.assertEqual(p.get_section_avg("detection"), 0.0)

    def test_section_timing_positive(self):
        p = Profiler()
        p.start("work")
        time.sleep(0.01)  # 10ms
        p.stop("work")

        avg = p.get_section_avg("work")
        self.assertGreater(avg, 5.0, "10ms sleep should measure >5ms")
        self.assertLess(avg, 50.0, "shouldn't be wildly long")

    def test_stopping_unstarted_section(self):
        """stopping a section that was never started shouldn't crash"""
        p = Profiler()
        p.stop("nonexistent")  # should just return silently

    def test_multiple_sections(self):
        p = Profiler()
        p.start("fast")
        p.stop("fast")

        p.start("slow")
        time.sleep(0.01)
        p.stop("slow")

        fast_avg = p.get_section_avg("fast")
        slow_avg = p.get_section_avg("slow")
        self.assertLess(fast_avg, slow_avg)


class TestFrameTiming(unittest.TestCase):

    def test_fps_calculation(self):
        p = Profiler()
        for _ in range(10):
            p.begin_frame()
            time.sleep(0.01)  # ~100fps target
            p.end_frame()

        fps = p.get_fps()
        self.assertGreater(fps, 20, "FPS should be measurable")
        self.assertLess(fps, 200, "FPS shouldn't be impossibly high")

    def test_frame_count_tracking(self):
        p = Profiler()
        for _ in range(5):
            p.begin_frame()
            p.end_frame()
        self.assertEqual(p._frame_count, 5)

    def test_peak_tracking(self):
        p = Profiler()

        p.begin_frame()
        p.end_frame()

        p.begin_frame()
        time.sleep(0.02)
        p.end_frame()

        # peak should be at least the slow frame
        self.assertGreater(p._peak_total * 1000, 10)


class TestSummaryDict(unittest.TestCase):

    def test_summary_structure(self):
        p = Profiler()
        p.begin_frame()
        p.start("detection")
        time.sleep(0.005)
        p.stop("detection")
        p.end_frame()

        d = p.summary_dict()
        self.assertIn("fps", d)
        self.assertIn("frame_ms", d)
        self.assertIn("frame_p50_ms", d)
        self.assertIn("frame_p95_ms", d)
        self.assertIn("frame_p99_ms", d)
        self.assertIn("sections", d)
        self.assertIn("detection", d["sections"])

    def test_section_summary_fields(self):
        p = Profiler()
        p.begin_frame()
        p.start("gesture")
        p.stop("gesture")
        p.end_frame()

        d = p.summary_dict()
        section = d["sections"]["gesture"]
        self.assertIn("avg_ms", section)
        self.assertIn("p50_ms", section)
        self.assertIn("p95_ms", section)
        self.assertIn("p99_ms", section)
        self.assertIn("peak_ms", section)
        self.assertIn("pct", section)

    def test_empty_summary(self):
        p = Profiler()
        d = p.summary_dict()
        self.assertEqual(d["fps"], 0.0)
        self.assertEqual(d["frame_ms"], 0.0)
        self.assertEqual(len(d["sections"]), 0)


class TestPercentiles(unittest.TestCase):

    def test_section_percentiles_no_data(self):
        p = Profiler()
        p50, p95, p99 = p.get_section_percentiles("nothing")
        self.assertEqual(p50, 0.0)
        self.assertEqual(p95, 0.0)
        self.assertEqual(p99, 0.0)

    def test_frame_percentiles_no_data(self):
        p = Profiler()
        p50, p95, p99 = p.get_frame_percentiles()
        self.assertEqual(p50, 0.0)

    def test_section_percentiles_with_data(self):
        p = Profiler()
        for _ in range(10):
            p.start("work")
            time.sleep(0.001)
            p.stop("work")

        p50, p95, p99 = p.get_section_percentiles("work")
        self.assertGreater(p50, 0)
        self.assertGreaterEqual(p95, p50)
        self.assertGreaterEqual(p99, p95)

    def test_frame_percentiles_ordering(self):
        """p50 <= p95 <= p99 always"""
        p = Profiler()
        for _ in range(20):
            p.begin_frame()
            time.sleep(0.001)
            p.end_frame()

        p50, p95, p99 = p.get_frame_percentiles()
        self.assertGreater(p50, 0)
        self.assertGreaterEqual(p95, p50)
        self.assertGreaterEqual(p99, p95)


class TestWindowRolling(unittest.TestCase):

    def test_window_size_respected(self):
        p = Profiler(window_size=5)

        for _ in range(20):
            p.begin_frame()
            p.start("test")
            p.stop("test")
            p.end_frame()

        # frame totals should be capped at window size
        self.assertLessEqual(len(p._frame_total), 5)
        self.assertLessEqual(len(p._frame_times["test"]), 5)


class TestReportOutput(unittest.TestCase):

    def test_report_no_data(self):
        """report with no data should not crash"""
        p = Profiler()
        p.report()  # should print "no profiling data yet"

    def test_report_with_data(self):
        """report with data should not crash"""
        p = Profiler()
        for _ in range(3):
            p.begin_frame()
            p.start("detection")
            p.stop("detection")
            p.end_frame()
        p.report()  # shouldn't raise


if __name__ == "__main__":
    unittest.main()

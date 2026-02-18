"""
lightweight performance profiler.

tracks per-frame timing for each stage of the pipeline so you can
see where the bottlenecks are. not meant for production - toggle it
on when you need to investigate frame drops.

usage:
    profiler = Profiler()
    profiler.start("detection")
    ...do detection...
    profiler.stop("detection")
    profiler.end_frame()

    # after some frames:
    profiler.report()
"""
import time
import math
from collections import defaultdict


def _percentile(sorted_data, p):
    """compute the pth percentile from a pre-sorted list"""
    if not sorted_data:
        return 0.0
    k = (len(sorted_data) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_data[int(k)]
    return sorted_data[int(f)] * (c - k) + sorted_data[int(c)] * (k - f)


class Profiler:
    """
    measures wall-clock time for named sections of each frame.
    keeps a rolling window of the last N frames and can print
    a summary table whenever you want.

    reports include p50, p95, p99 — mean is cute, percentiles are real.
    """

    def __init__(self, window_size=120, enabled=True):
        self.enabled = enabled
        self._window = window_size

        # per-section timing
        self._active = {}             # section -> start time
        self._frame_times = defaultdict(list)   # section -> [durations]
        self._frame_total = []        # total frame times

        # frame-level tracking
        self._frame_start = None
        self._frame_count = 0

        # peak tracking
        self._peaks = defaultdict(float)  # section -> worst single frame
        self._peak_total = 0.0

    def start(self, section):
        """begin timing a named section"""
        if not self.enabled:
            return
        self._active[section] = time.perf_counter()

    def stop(self, section):
        """end timing a named section"""
        if not self.enabled or section not in self._active:
            return
        elapsed = time.perf_counter() - self._active.pop(section)

        times = self._frame_times[section]
        times.append(elapsed)
        if len(times) > self._window:
            times.pop(0)

        if elapsed > self._peaks[section]:
            self._peaks[section] = elapsed

    def begin_frame(self):
        """call at the very start of each frame"""
        if not self.enabled:
            return
        self._frame_start = time.perf_counter()

    def end_frame(self):
        """call at the very end of each frame"""
        if not self.enabled or self._frame_start is None:
            return
        elapsed = time.perf_counter() - self._frame_start
        self._frame_total.append(elapsed)
        if len(self._frame_total) > self._window:
            self._frame_total.pop(0)

        if elapsed > self._peak_total:
            self._peak_total = elapsed

        self._frame_count += 1
        self._frame_start = None

    def get_section_avg(self, section):
        """average time in ms for a section"""
        times = self._frame_times.get(section, [])
        if not times:
            return 0.0
        return (sum(times) / len(times)) * 1000

    def get_section_percentiles(self, section):
        """return p50, p95, p99 in ms for a section"""
        times = self._frame_times.get(section, [])
        if not times:
            return 0.0, 0.0, 0.0
        sorted_ms = sorted(t * 1000 for t in times)
        return (
            _percentile(sorted_ms, 50),
            _percentile(sorted_ms, 95),
            _percentile(sorted_ms, 99),
        )

    def get_frame_percentiles(self):
        """return p50, p95, p99 in ms for total frame time"""
        if not self._frame_total:
            return 0.0, 0.0, 0.0
        sorted_ms = sorted(t * 1000 for t in self._frame_total)
        return (
            _percentile(sorted_ms, 50),
            _percentile(sorted_ms, 95),
            _percentile(sorted_ms, 99),
        )

    def get_fps(self):
        """average fps over the window"""
        if not self._frame_total:
            return 0.0
        avg = sum(self._frame_total) / len(self._frame_total)
        return 1.0 / avg if avg > 0 else 0.0

    def report(self):
        """print a formatted summary with percentile breakdowns"""
        if not self._frame_total:
            print("no profiling data yet")
            return

        total_avg = (sum(self._frame_total) / len(self._frame_total)) * 1000
        total_peak = self._peak_total * 1000
        fp50, fp95, fp99 = self.get_frame_percentiles()
        fps = self.get_fps()

        sections = sorted(self._frame_times.keys())

        print()
        print(f"  PERFORMANCE PROFILE (last {len(self._frame_total)} frames)")
        print(f"  {'─' * 72}")
        print(f"  {'Section':<20} {'Avg ms':>7} {'p50':>7} {'p95':>7} {'p99':>7} {'Peak':>7} {'%Frame':>7}")
        print(f"  {'─' * 72}")

        for section in sections:
            times = self._frame_times[section]
            avg = (sum(times) / len(times)) * 1000 if times else 0
            p50, p95, p99 = self.get_section_percentiles(section)
            peak = self._peaks[section] * 1000
            pct = (avg / total_avg * 100) if total_avg > 0 else 0
            print(f"  {section:<20} {avg:>6.2f}  {p50:>6.2f}  {p95:>6.2f}  {p99:>6.2f}  {peak:>6.2f}  {pct:>5.1f}%")

        print(f"  {'─' * 72}")
        print(f"  {'TOTAL FRAME':<20} {total_avg:>6.2f}  {fp50:>6.2f}  {fp95:>6.2f}  {fp99:>6.2f}  {total_peak:>6.2f}")
        print(f"  {'FPS':<20} {fps:>6.1f}")
        print(f"  {'─' * 72}")
        print()

    def summary_dict(self):
        """return timing data as a dict for programmatic use"""
        total_avg = 0.0
        if self._frame_total:
            total_avg = (sum(self._frame_total) / len(self._frame_total)) * 1000

        fp50, fp95, fp99 = self.get_frame_percentiles()

        data = {
            "fps": self.get_fps(),
            "frame_ms": total_avg,
            "frame_p50_ms": round(fp50, 2),
            "frame_p95_ms": round(fp95, 2),
            "frame_p99_ms": round(fp99, 2),
            "sections": {},
        }
        for section, times in self._frame_times.items():
            if times:
                avg = (sum(times) / len(times)) * 1000
                p50, p95, p99 = self.get_section_percentiles(section)
                data["sections"][section] = {
                    "avg_ms": round(avg, 2),
                    "p50_ms": round(p50, 2),
                    "p95_ms": round(p95, 2),
                    "p99_ms": round(p99, 2),
                    "peak_ms": round(self._peaks[section] * 1000, 2),
                    "pct": round((avg / total_avg * 100) if total_avg > 0 else 0, 1),
                }
        return data

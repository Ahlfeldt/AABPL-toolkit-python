"""
In-process progress bars for build_disk_region_lookups and search_and_aggregate.

Stage weights are placeholder fractions of total time that sum to 1.0. They will be
calibrated against real timing once benchmark data is collected. Within a session,
total-time estimates self-correct after each completed run using an exponential
moving average (new = 0.75 * old + 0.25 * actual), so runs on a faster or slower
machine converge toward reality within 2-3 calls.

To disable globally::

    import aabpl.utils.progress as aabpl_progress
    aabpl_progress.SHOW_PROGRESS = False
"""
import sys as _sys
import time as _time
import contextvars as _contextvars
try:
    from aabpl import config as _cfg
except Exception:
    _cfg = None

# ---------------------------------------------------------------------------
# Outer-call tracking — used to suppress sub-progress bars when a combined
# top-level progress bar (RadiusSearchProgress / DetectClusterProgress) is
# already active.  ContextVar is thread- and asyncio-task-safe.
# ---------------------------------------------------------------------------
_OUTER_PROGRESS: _contextvars.ContextVar = _contextvars.ContextVar(
    '_aabpl_outer_progress', default=None
)

# Wall time: used for ETA display (what the user actually waits for).
_wall = _time.perf_counter
# CPU time: used for EMA learning (immune to parallel load on a busy machine).
_cpu = _time.process_time

# Set to False to suppress all progress bars regardless of the silent parameter.
SHOW_PROGRESS: bool = True

# Progress bars are suppressed for runs estimated to finish under this many seconds.
_MIN_SECONDS_TO_SHOW: float = 5.0

# Weight given to the new observation vs the old estimate (0.25 → 75% old, 25% new).
_EMA_ALPHA: float = 0.25

_BAR_WIDTH = 18
_LINE_WIDTH = 115  # must exceed the longest line any renderer can produce


_RATIO_EMA_ALPHA: float = 0.25


def _update_ratio(wall: float, cpu: float) -> None:
    """Update WALL_TO_CPU_RATIO EMA from a (wall_elapsed, cpu_elapsed) observation."""
    try:
        if _cfg is not None and cpu > 0.1:
            obs = wall / cpu
            _cfg.WALL_TO_CPU_RATIO = (1 - _RATIO_EMA_ALPHA) * _cfg.WALL_TO_CPU_RATIO + _RATIO_EMA_ALPHA * obs
    except Exception:
        pass


def _ratio() -> float:
    return _cfg.WALL_TO_CPU_RATIO


def _fmt_eta(seconds: float) -> str:
    """Format ETA as '~45s left', '~2m07s left', or '~1h23m left'."""
    s = max(0, int(round(seconds)))
    if s < 60:
        return f"~{s}s left"
    m, s = divmod(s, 60)
    if m < 60:
        return f"~{m}m{s:02d}s left"
    h, m = divmod(m, 60)
    return f"~{h}h{m:02d}m left"


# ---------------------------------------------------------------------------
# build_disk_region_lookups progress
# ---------------------------------------------------------------------------

_STAGE_LABELS = [
    "Classify cells",
    "Init triangle + boundary checks",
    "Split regions",
    "Finalize region cells",
    "Expand to 8 sectors",
    "Build region raster",
    "Expand raster + check structures",
    "Build lookup tables",
]
_STAGE_WEIGHTS = [0.02, 0.04, 0.45, 0.20, 0.05, 0.07, 0.10, 0.07]
assert abs(sum(_STAGE_WEIGHTS) - 1.0) < 1e-9, "Stage weights must sum to 1.0"

# Placeholder total-time estimate for build_disk_region_lookups (seconds).
# Updated within the session by EMA after each completed run.
_BUILD_EST_SECONDS: float = 180.0


class DiskRegionProgress:
    """
    Progress bar for a single build_disk_region_lookups call.

    Suppressed automatically when:
    - ``silent=True``, or
    - module-level ``SHOW_PROGRESS`` is ``False``, or
    - the current session estimate is below 5 s.

    Everything is written with ``\\r`` (no ``\\n``) so the line is fully erased
    on completion or cancellation — no lingering output.
    """

    def __init__(self, silent: bool, r_over_spacing: float, nest_depth: int):
        self._r_over_s = r_over_spacing
        self._nest_depth = nest_depth
        self._stage: int = -1
        self._desc: str = ""
        self._active: bool = False
        self._base_silent = silent

    def start(self) -> None:
        if self._base_silent or not SHOW_PROGRESS or _BUILD_EST_SECONDS < _MIN_SECONDS_TO_SHOW or _OUTER_PROGRESS.get() is not None:
            self._active = False
            return
        self._active = True
        self._run_start_wall = _wall()
        self._run_start_cpu = _cpu()
        self._stage = -1
        self._render()

    def step(self, desc: str) -> None:
        if not self._active:
            return
        self._stage += 1
        self._desc = desc
        self._render()

    def done(self) -> None:
        global _BUILD_EST_SECONDS
        if not self._active:
            return
        self._active = False
        _sys.stdout.write("\r" + " " * _LINE_WIDTH + "\r")
        _sys.stdout.flush()
        wall_actual = _wall() - self._run_start_wall
        cpu_actual  = _cpu()  - self._run_start_cpu
        _BUILD_EST_SECONDS = (1 - _EMA_ALPHA) * _BUILD_EST_SECONDS + _EMA_ALPHA * cpu_actual
        _update_ratio(wall_actual, cpu_actual)

    def cancel(self) -> None:
        """Erase the progress line without updating the EMA (use on error)."""
        if not self._active:
            return
        self._active = False
        _sys.stdout.write("\r" + " " * _LINE_WIDTH + "\r")
        _sys.stdout.flush()

    def _render(self) -> None:
        try:
            elapsed = _wall() - self._run_start_wall
            p = sum(_STAGE_WEIGHTS[: self._stage + 1]) if self._stage >= 0 else 0.0
            eta_elapsed = max(0.0, elapsed / p - elapsed) if p > 0 and elapsed > 0 else _BUILD_EST_SECONDS * _ratio()
            eta_estimate = _BUILD_EST_SECONDS * _ratio() * (1.0 - p)
            blend = min(1.0, p / 0.3)
            eta = (1.0 - blend) * eta_estimate + blend * eta_elapsed
            pct = int(p * 100)
            filled = int(_BAR_WIDTH * p)
            bar = "=" * filled + ">" + " " * (_BAR_WIDTH - filled - 1)
            desc = (self._desc[:16] + "..") if len(self._desc) > 18 else self._desc
            remaining = _fmt_eta(eta)
            info = f"r/s={round(self._r_over_s, 2)} nd={self._nest_depth}"
            line = f"  [{bar}] {pct:3d}%  {info:<14}  {desc:<18}  {remaining:<14}"
            _sys.stdout.write(f"\r{line:<{_LINE_WIDTH}}")
            _sys.stdout.flush()
        except Exception:
            self._active = False


# ---------------------------------------------------------------------------
# search_and_aggregate (disk search loop) progress
# ---------------------------------------------------------------------------

# Placeholder total-time estimate for search_and_aggregate (seconds).
_SEARCH_EST_SECONDS: float = 180.0


class SearchProgress:
    """
    Progress bar for a single search_and_aggregate call.

    Designed for minimal overhead inside a tight per-point loop.  The caller
    drives updates using ``next_threshold``:

        prog = SearchProgress(silent=silent, n_pts=n_pts)
        prog.start()
        _thresh = prog.next_threshold   # copy to local for fast comparison
        for i, ... in enumerate(...):
            ...
            if i >= _thresh:
                _thresh = prog.update(i)
        prog.done()

    ``update`` is called at most 10 times per run, so its stdout cost is
    negligible.  The per-iteration overhead is a single integer comparison.
    When suppressed ``next_threshold`` is ``n_pts + 1`` so the condition is
    never true and there is zero overhead inside the loop.

    Everything is written with ``\\r`` (no ``\\n``) so the line is fully erased
    on completion — no lingering output.
    """

    def __init__(self, silent: bool, n_pts: int):
        self._n_pts = n_pts
        self._base_silent = silent
        self._active: bool = False
        # default: threshold beyond any valid i so the check never fires
        self.next_threshold: int = n_pts + 1

    # Target wall-time interval between redraws (seconds).
    _UPDATE_INTERVAL: float = 2.0

    def start(self) -> None:
        if self._base_silent or not SHOW_PROGRESS or _OUTER_PROGRESS.get() is not None:
            self._active = False
            return
        self._run_start_wall = _wall()
        self._run_start_cpu = _cpu()
        self._last_update_wall = self._run_start_wall
        self._last_update_cpu  = self._run_start_cpu
        self._last_update_i    = 0
        # Initial threshold: whichever fires first — 10 % of points or estimated
        # time for one update interval (so first redraw is never later than that).
        est_wall = _SEARCH_EST_SECONDS * _ratio()
        pts_per_interval = max(1, int(self._n_pts * self._UPDATE_INTERVAL / max(est_wall, 1e-9)))
        self.next_threshold = min(max(1, self._n_pts // 10), pts_per_interval)
        if est_wall >= _MIN_SECONDS_TO_SHOW:
            self._active = True
            remaining = _fmt_eta(est_wall)
            bar = ">" + " " * (_BAR_WIDTH - 1)
            n_str = f"{self._n_pts:,} pts"
            line = f"  [{bar}]   0%  {n_str:<14}                    {remaining:<14}"
            _sys.stdout.write(f"\r{line:<{_LINE_WIDTH}}")
            _sys.stdout.flush()
        else:
            self._active = False  # lazily activated in update() if run is slow

    def update(self, i: int) -> int:
        """Redraw bar and return the next threshold value."""
        if not hasattr(self, '_run_start_wall'):
            return self._n_pts + 1
        now     = _wall()
        now_cpu = _cpu()
        elapsed = now - self._run_start_wall
        if not self._active:
            if elapsed < _MIN_SECONDS_TO_SHOW:
                # Keep checking every interval worth of points
                step = max(1, self.next_threshold - self._last_update_i)
                self._last_update_i = i
                return i + step
            self._active = True  # run is slower than estimated — show bar now
        try:
            p = i / self._n_pts
            est_wall = _SEARCH_EST_SECONDS * _ratio()
            eta_elapsed = max(0.0, elapsed / p - elapsed) if p > 0 and elapsed > 0 else est_wall
            eta_estimate = est_wall * (1.0 - p)
            blend = min(1.0, p / 0.3)
            eta = (1.0 - blend) * eta_estimate + blend * eta_elapsed
            pct = int(p * 100)
            filled = int(_BAR_WIDTH * p)
            bar = "=" * filled + ">" + " " * (_BAR_WIDTH - filled - 1)
            remaining = _fmt_eta(eta)
            n_str = f"{self._n_pts:,} pts"
            line = f"  [{bar}] {pct:3d}%  {n_str:<14}                    {remaining:<14}"
            _sys.stdout.write(f"\r{line:<{_LINE_WIDTH}}")
            _sys.stdout.flush()
        except Exception:
            self._active = False
            return self._n_pts + 1
        # Update wall/cpu ratio from this interval, then recalibrate threshold
        dt     = now - self._last_update_wall
        dt_cpu = now_cpu - self._last_update_cpu
        di     = max(1, i - self._last_update_i)
        _update_ratio(dt, dt_cpu)
        self._last_update_wall = now
        self._last_update_cpu  = now_cpu
        self._last_update_i    = i
        pts_per_interval = max(1, int(di / max(dt, 1e-9) * self._UPDATE_INTERVAL))
        return i + pts_per_interval

    def done(self) -> None:
        global _SEARCH_EST_SECONDS
        if not self._active:
            return
        self._active = False
        _sys.stdout.write("\r" + " " * _LINE_WIDTH + "\r")
        _sys.stdout.flush()
        wall_actual = _wall() - self._run_start_wall
        cpu_actual  = _cpu()  - self._run_start_cpu
        _SEARCH_EST_SECONDS = (1 - _EMA_ALPHA) * _SEARCH_EST_SECONDS + _EMA_ALPHA * cpu_actual
        _update_ratio(wall_actual, cpu_actual)


# ---------------------------------------------------------------------------
# Combined top-level progress bars
# ---------------------------------------------------------------------------

class _CombinedProgress:
    """
    Single-line progress bar for a top-level user-facing call.

    Sub-progress bars (DiskRegionProgress, SearchProgress) suppress themselves
    when ``_OUTER_PROGRESS`` is set, so only this line is visible.

    Subclasses define:
      _STAGE_WEIGHTS  – ordered list of (stage_name, cumulative_fraction) pairs
      _est_seconds    – class-level float: current EMA estimate of total wall time.
                        Updated after each completed run so ETA self-corrects.

    Lifecycle::

        prog = RadiusSearchProgress(silent=silent, n_pts=len(pts))
        token = _OUTER_PROGRESS.set(prog)
        prog.start()
        try:
            prog.step("assigning target")
            ...
            prog.step("searching")
            ...
        finally:
            _OUTER_PROGRESS.reset(token)
            prog.done()
    """

    _label: str = "aabpl"
    _STAGE_WEIGHTS: list = []
    _est_seconds: float = 60.0  # placeholder; updated by EMA after each run

    def __init__(self, silent: bool, n_pts: int, n_tgt: int = None):
        self._n_pts = n_pts
        self._n_tgt = n_tgt
        self._silent = silent
        self._active = False
        self._stage = ""
        self._progress = 0.0
        self._stage_map = {name: frac for name, frac in self._STAGE_WEIGHTS}

    def start(self) -> None:
        if self._silent or not SHOW_PROGRESS:
            return
        self._active = True
        self._t0 = _wall()
        self._cpu0 = _cpu()
        self._progress = 0.0
        self._render()

    def step(self, stage: str) -> None:
        if not self._active:
            return
        self._stage = stage
        self._progress = self._stage_map.get(stage, self._progress)
        self._render()

    def done(self) -> None:
        if not self._active:
            return
        self._active = False
        wall_elapsed = _wall() - self._t0
        cpu_elapsed  = _cpu()  - self._cpu0
        type(self)._est_seconds = (1 - _EMA_ALPHA) * type(self)._est_seconds + _EMA_ALPHA * wall_elapsed
        _update_ratio(wall_elapsed, cpu_elapsed)
        _sys.stdout.write("\r" + " " * _LINE_WIDTH + "\r")
        _sys.stdout.flush()

    def cancel(self) -> None:
        if not self._active:
            return
        self._active = False
        _sys.stdout.write("\r" + " " * _LINE_WIDTH + "\r")
        _sys.stdout.flush()

    def _render(self) -> None:
        try:
            elapsed = _wall() - self._t0
            p = self._progress
            est = type(self)._est_seconds
            eta_elapsed = max(0.0, elapsed / p - elapsed) if p > 0 and elapsed > 0 else est
            eta_estimate = est * (1.0 - p)
            blend = min(1.0, p / 0.3)
            eta = (1.0 - blend) * eta_estimate + blend * eta_elapsed
            filled = int(_BAR_WIDTH * p)
            bar = "=" * filled + (">" if filled < _BAR_WIDTH else "") + " " * max(0, _BAR_WIDTH - filled - 1)
            pct = int(p * 100)
            if self._n_tgt is not None and self._n_tgt != self._n_pts:
                n_str = f"{self._n_pts:,}->{self._n_tgt:,}"
            else:
                n_str = f"{self._n_pts:,} pts"
            stage = (self._stage[:18] + "..") if len(self._stage) > 20 else self._stage
            remaining = _fmt_eta(eta)
            line = f"  {self._label}  [{bar}] {pct:3d}%  {stage:<22}  {n_str:<14}  {elapsed:.1f}s  {remaining:<12}"
            _sys.stdout.write(f"\r{line:<{_LINE_WIDTH}}")
            _sys.stdout.flush()
        except Exception:
            self._active = False


class RadiusSearchProgress(_CombinedProgress):
    """Combined progress bar for a top-level ``radius_search`` call."""
    _label = "radius_search  "
    _est_seconds: float = 30.0  # placeholder: ~30s for a medium dataset
    _STAGE_WEIGHTS = [
        ("initializing",     0.00),
        ("assigning target", 0.05),
        ("assigning source", 0.10),
        ("searching",        0.15),
    ]


class DetectClusterProgress(_CombinedProgress):
    """Combined progress bar for a top-level ``detect_cluster_pts`` call."""
    _label = "detect_clusters"
    _est_seconds: float = 90.0  # placeholder: ~90s (two searches + null distribution)
    _STAGE_WEIGHTS = [
        ("initializing",      0.00),
        ("assigning target",  0.03),
        ("null distribution", 0.08),
        ("assigning source",  0.25),
        ("searching",         0.30),
        ("labeling clusters", 0.90),
    ]


# ---------------------------------------------------------------------------
# Sweep progress bar (benchmarking)
# ---------------------------------------------------------------------------

class SweepProgress:
    """
    Single-line loading bar for run_sweep iterations.

    Usage::

        prog = SweepProgress(n_total=120)
        prog.start()
        for ...:
            prog.update(done, label="r=500 s=100 nd=2 tn=1")
            ...
        prog.done()
    """

    def __init__(self, n_total: int):
        self._n = n_total
        self._active = False

    def start(self) -> None:
        self._active = True
        self._t0 = _wall()
        self._done = 0
        self._label = ""
        self._render()

    def update(self, done: int, label: str = "") -> None:
        if not self._active:
            return
        self._done = done
        self._label = label
        self._render()

    def clear(self) -> None:
        """Erase the bar line so external prints appear on a clean line."""
        if not self._active:
            return
        _sys.stdout.write(f"\r{' ' * _LINE_WIDTH}\r")
        _sys.stdout.flush()

    def redraw(self) -> None:
        """Redraw the bar after external prints have moved the cursor."""
        if not self._active:
            return
        self._render()

    def done(self) -> None:
        if not self._active:
            return
        self._active = False
        _sys.stdout.write(f"\r{' ' * _LINE_WIDTH}\r")
        elapsed = _wall() - self._t0
        _sys.stdout.write(f"  sweep done  {self._n} runs  {elapsed:.1f}s\n")
        _sys.stdout.flush()

    def _render(self) -> None:
        elapsed = _wall() - self._t0
        p = self._done / self._n if self._n > 0 else 0.0
        eta = (elapsed / p - elapsed) if p > 0 and elapsed > 0 else 0.0
        pct = int(p * 100)
        filled = int(_BAR_WIDTH * p)
        bar = "=" * filled + (">" if filled < _BAR_WIDTH else "") + " " * max(0, _BAR_WIDTH - filled - 1)
        label_str = (self._label[:22] + "..") if len(self._label) > 24 else self._label
        eta_str = _fmt_eta(eta) if p > 0 else ""
        line = f"  [{bar}] {pct:3d}%  {self._done}/{self._n}  {label_str:<26}  {elapsed:.1f}s  {eta_str}"
        _sys.stdout.write(f"\r{line:<{_LINE_WIDTH}}")
        _sys.stdout.flush()

"""
Package-wide performance and behaviour configuration for aabpl.

Import and modify these flags at runtime to control algorithm behaviour:

    from aabpl import config
    config.USE_VEC_VERTEX_CHECKS = False   # revert to scalar vertex loop
    config.N_POINTS_TOTAL = 1_000_000      # tell the package your total workload
    config.disk_region_cache.clear()       # invalidate geometry cache

All mutable state here is module-level so changes take effect immediately
on the next call without reloading.
"""
from collections import OrderedDict as _OrderedDict

# ---------------------------------------------------------------------------
# Geometry build
# ---------------------------------------------------------------------------

# Use vectorised numpy batch evaluation of vertex containment/overlap checks
# in classify_subcell_quadrants_vec (disk_region_geometry.py).
# Set to False to fall back to the original scalar per-edge Python loop.
USE_VEC_VERTEX_CHECKS: bool = True

# Shared geometry cache: maps (r/spacing, nest_depth, include_boundary) → lookup result.
# Keyed on geometry only (no point data), so shared across all Grid instances.
# Clear with config.disk_region_cache.clear() to force a rebuild.
# Maximum number of geometry configs held in disk_region_cache before the
# oldest is evicted.  Each entry covers one (r/spacing, nest_depth) combination.
DISK_REGION_CACHE_MAXSIZE: int = 8
disk_region_cache: _OrderedDict = _OrderedDict()

# ---------------------------------------------------------------------------
# Override spacing / nest_depth for testing.  None = choose automatically.
# Set before calling radius_search:
#   import aabpl.config as config
#   config.FIXED_SPACING_RATIO = 2.0   # forces spacing = r / 2.0
#   config.FIXED_NEST_DEPTH = 1
# ---------------------------------------------------------------------------
FIXED_SPACING_RATIO: float | None = None
FIXED_NEST_DEPTH: int | None = None

# ---------------------------------------------------------------------------
# Workload hints — used to inform optimal spacing / nest_depth selection
# ---------------------------------------------------------------------------

# Total number of source points the user expects to process across all calls
# in this session.  Set this before running if you know your workload upfront —
# it allows the package to favour heavier geometry (higher build cost, lower
# search cost per point) when the build amortises over enough points.
# None → the package will estimate from the current call size and N_POINTS_CUM.
N_POINTS_TOTAL: int | None = None

# ---------------------------------------------------------------------------
# Developer mode
# ---------------------------------------------------------------------------

# When True, enables verbose internal diagnostics and skips some guards.
DEV_MODE: bool = False

# Running count of source points processed so far this session.
# Incremented automatically after each search_and_aggregate call.
# Reset to 0 between independent sessions if needed.
N_POINTS_CUM: int = 0

# ---------------------------------------------------------------------------
# Wall-to-CPU time ratio
# ---------------------------------------------------------------------------

# Ratio of wall time to CPU (process) time observed during recent runs.
# Values above 1.0 indicate the machine is under parallel load.
# Updated automatically by progress bars; used to convert CPU-based EMA
# estimates to wall-time ETA for display.
WALL_TO_CPU_RATIO: float = 1.3

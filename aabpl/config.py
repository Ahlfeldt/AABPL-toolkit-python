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

# Per-function CPU-time profiling via the time_func_perf decorator.
# OFF by default: when False the decorator is a zero-overhead passthrough.
# When True it records start/end process_time for every decorated call into
# test_performance.func_timer_dict (consumed by analyze_func_perf). Some of those
# decorated functions run once PER SEARCH POINT, where the wrapper adds ~3 us/call
# and grows an unbounded in-memory call log (~240 MB per 1e6 calls) — so leave
# this False in production. The benchmark harness (run_single_config) turns it on
# for the duration of a measured run; enable it manually if calling
# analyze_func_perf() yourself.
# TEMPORARILY True while collecting benchmark data — set back to False after
# testing (the production default should be False). See roadmap.md.
PROFILE_FUNC_TIMES: bool = True

# Note: the search loop now always uses the integer cell-key codec, the vectorized
# group masks, and the always-on (group x candidates) overlap batch — these were
# previously the experimental flags USE_INT_CELL_KEYS / VECTORIZED_SEARCH_LOOP /
# BATCH_OVERLAP / BATCH_OVERLAP_MIN_GROUP. Benchmarking settled them (always batch
# wins, 1.4-1.8x and growing with scale; all byte-identical to the old scalar loop),
# so they are baked in and the flags removed. See disk_aggregation.search_and_aggregate.

# Shared geometry cache: maps (r/spacing, nest_depth, include_boundary) → lookup result.
# Keyed on geometry only (no point data), so shared across all Grid instances.
# Clear with config.disk_region_cache.clear() to force a rebuild.
# Maximum number of geometry configs held in disk_region_cache before the
# least-recently-used is evicted (LRU: reuse moves an entry to the back; eviction
# pops the front in disk_region_geometry.py). Each entry covers one
# (r/spacing, nest_depth) combination.
#
# Memory footprint is small and bounded — it stores cell geometry only, never the
# point data, and nest_depth/spacing come from small candidate sets. Measured
# pickled entry sizes at r/spacing=1.414 (in-memory ~2-5x larger, but still tens
# of MB total for a full cache):
#     nd=0 → 0.05 MB,  nd=2 → 0.22 MB,  nd=4 → 1.03 MB,
#     nd=5 → 2.22 MB,  nd=6 → 4.64 MB   (~2x per +1 nest_depth; sub-4^nd thanks
#                                        to sparse pruning). Larger r/spacing
#                                        grows entries somewhat but stays in the
#                                        tens-of-MB range, never GB.
# The real cost of deep nesting is BUILD TIME, not memory: constructing the nd=6
# geometry once took ~97s (vs sub-second at low nd) — which is why
# choose_nest_depth caps around 4 in practice. Hence the cache: build once, reuse.
DISK_REGION_CACHE_MAXSIZE: int = 10
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


# ---------------------------------------------------------------------------
# Geometry Amortization Weights
# ---------------------------------------------------------------------------

# Factor used to discount the upfront geometry build cost (geo_s).
# Reflects the expected number of query cycles per geometry lifecycle.
# A value of 0.75 assumes the build cost pays off across multiple runs,
# preventing the optimizer from over-penalizing heavy indexing structures.
GEO_AMORTIZATION_WEIGHT: float = 0.75


# Only for dev purposes and experimental.
USE_OPTIMIZED_METHOD = False
VALIDATE = False
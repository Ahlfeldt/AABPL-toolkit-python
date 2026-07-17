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
from __future__ import annotations
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
PROFILE_FUNC_TIMES: bool = False

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
DISK_REGION_CACHE_MAXSIZE: int = 10*5
disk_region_cache: _OrderedDict = _OrderedDict()

# ---------------------------------------------------------------------------
# Override spacing / nest_depth for testing.  None = choose automatically.
# Set before calling radius_search:
#   import aabpl.config as config
#   config.FIXED_SPACING_RATIO = 2.0   # forces spacing = r / 2.0
#   config.FIXED_NEST_DEPTH = 1
# ---------------------------------------------------------------------------
FIXED_SPACING_RATIO: float | None = 2.0
FIXED_NEST_DEPTH: int | None = None
SINGLE_REGION: bool = False   # undocumented: collapse all regions into one during search

# Narrower than FIXED_NEST_DEPTH: caps the target quadtree depth for adaptive-
# routed runs (FIXED_NEST_DEPTH=None) without affecting routing -- useful when
# benchmarking/forcing a specific nd<0 (supercell) tag via _best_nd_tag, since
# supercells never read the quadtree beyond level 0 and building it to the
# grid's native depth is pure waste. Ignored whenever FIXED_NEST_DEPTH is set.
ADAPTIVE_NEST_DEPTH_CAP: int | None = None

# Safety cap on dense (n_block x n_candidates) pairwise distance-check buffers
# (dx/dy broadcast + inside-radius mask in disk_aggregation*.py). Each pair
# costs ~4 live float64 temporaries during the dx*dx+dy*dy<=r2 computation
# (dx, dy, dx**2+dy**2, and numpy's own intermediate), so this many BYTES
# translates to roughly bytes/(8*4) pairs. When n_block*n_candidates would
# exceed that, the block is processed in row-batches instead of one shot --
# prevents the multi-GB MemoryError crashes seen with extreme forced-nd/ppc
# combos (e.g. nd=-3 on sparse real data, or nd=4 at ppc so high the domain
# collapses to near the search radius) where chunk sizing can't keep pairs
# bounded. 1GB is a reasonable default for a single-process benchmark/
# interactive run; lower it on memory-constrained machines.
MAX_DIST_MATRIX_BYTES: int = 1_000_000_000

# When True, aggregate_point_data_to_cells_adaptive_nd (point_assignment.py) is
# used instead of the uniform-depth aggregate_point_data_to_cells: a coarse
# density scan decides, per spatial region, the max quadtree depth actually
# needed there (via the same best_nd_tag crossover model the hot loop uses),
# instead of building every region down to the grid's global native nest_depth
# unconditionally. Measured at 40-60%+ of total radius_search runtime before
# this, most of it wasted on regions sparse enough to only need nd<0
# (supercells, which never read past level 0) or a shallow nd. Validated
# (exhaustive correctness suite + adversarial cross-boundary dilation test,
# all clean) and measured (81.5%/99.2% node-count reduction and up to 2.35x
# wall-time on real datasets) -- now the default. Set False to force the old
# uniform-depth path (e.g. to isolate whether a regression came from here).
USE_ADAPTIVE_ND_AGGREGATION: bool = True

# When True, the per-chunk nd decision in disk_aggregation_chunk_adaptive_nd.py
# uses nd_choice.best_nd_tag_weighted instead of best_nd_tag: a point-count-
# weighted sum of the cost model over every occupied coarse block (correctly
# reflecting a chunk's real density mixture and occupancy, not just a single
# p75-of-densities percentile that's blind to how many blocks are actually
# occupied), blended by an occupancy-aware alpha with a total-chunk-area ppc.
# Validated as a net positive on both real test datasets earlier this
# session (higher per-decision cost, but better-chosen nd more than pays
# for it) -- promoted to the default.
USE_WEIGHTED_ND_DECISION: bool = True

# Fixed per-chunk overhead (ms) assumed when deciding whether to merge two
# adjacent column-bands with different nd tags: merge only if the estimated
# joint processing time (nd_choice.blended_cost_ms) beats the sum of the two
# separate times plus this overhead. Modeling real per-chunk setup/teardown
# cost. (Same-tag row-strip/column-band merges skip this trade-off entirely
# -- they always merge if they fit the L3 budget, since there's no
# different-nd cost difference to weigh.)
# 2ms is a rough, deliberately modest default -- direct measurement on real
# datasets (15k/521k points) showed no clean, low-noise value: the chunk
# count and wall-clock time did not vary monotonically with tested overhead
# values (0/2/10ms), most likely swamped by ordinary run-to-run system
# noise on the scale of individual chunk timings. Revisit if a cleaner
# measurement setup (isolated machine, more reps) becomes available.
CHUNK_MERGE_OVERHEAD_MS: float = 2.0

# ---------------------------------------------------------------------------
# TEMPORARY bisection toggles (2026-07-14) -- default False = today's fixes
# stay active. Set True to revert that ONE specific change for regression
# testing, without deleting any code. Remove this whole block (and the
# guards using them) once the regression is found and confirmed fixed.
# ---------------------------------------------------------------------------
_DEBUG_REVERT_ROW_ZONE_NOISE_FIX: bool = False   # row pre-planning: re-enable splitting on any nd<0 zone wobble
_DEBUG_REVERT_ROW_MERGE_RELAX: bool = False      # row-strip merge: require exact tag match again
_DEBUG_REVERT_COL_ZONE_NOISE_FIX: bool = False   # column band-building: re-enable per-nd<0-depth banding
_DEBUG_REVERT_COL_SPLIT_TRIGGER_FIX: bool = False  # elif trigger: re-enable on any nd<0 hot/nd mismatch
_DEBUG_REVERT_GREEDY_COL_MERGEBACK: bool = False   # _greedy_col_chunks: skip the merge-back pass
_DEBUG_REVERT_MEM_SCALE_FIX: bool = False          # _nd_mem_scale: flat 0.5 for nd<0 instead of 2**nd

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

# When True, radius_search with any area_weight runs a post-hoc accuracy check:
# picks 5 source points near each decile (0,10,...,100) of the computed
# valid_area_share distribution, computes the exact circle ∩ study_area
# fraction via Shapely, and prints a list of 11 (computed, exact) mean pairs.
# Slow (Shapely intersection per sampled point) — leave False in production.
VALIDATE_AREA: bool = False

# ---------------------------------------------------------------------------
# Chunk-path cache budget (disk_aggregation_chunk.py)
# ---------------------------------------------------------------------------

# L2 cache size in bytes.  Currently used only as the budget for the
# column-split fallback (when a single row exceeds the L3 budget, the col
# splitter falls back to L2 to keep each sub-block hot).  Set to your CPU's
# L2 size for best col-split behaviour; the default (256 KB) is conservative
# and safe on virtually all modern CPUs.
L2_BYTES: int = 256 * 1024

# L3 cache size in bytes.  Controls the primary chunk-row sizing: each chunk's
# block of source points is kept below this budget so it fits in L3 during the
# aggregation pass.  Set to your CPU's L3 size for best throughput.
# Auto-detection is intentionally omitted (subprocess/WMI calls are not
# antivirus-safe in a library).  Detect once with a one-off script and set
# this to the reported value.  Conservative default: 6 MB.
#
# To detect on Windows (run outside the package, once):
#   import subprocess, re
#   out = subprocess.check_output(['wmic', 'cpu', 'get', 'L3CacheSize'], text=True)
#   print(int(re.search(r'\d+', out).group()) * 1024, 'bytes')
L3_BYTES: int = 6 * 1024 * 1024

# ---------------------------------------------------------------------------
# Adaptive nest_depth (disk_aggregation_chunk_adaptive_nd.py)
# ---------------------------------------------------------------------------

# When True, radius_search routes through disk_aggregation_chunk_adaptive_nd
# instead of the standard disk_aggregation_chunk.  The adaptive module picks
# a per-chunk nest_depth based on a cost model (or randomly, during testing).
USE_ADAPTIVE_NEST_DEPTH: bool = False
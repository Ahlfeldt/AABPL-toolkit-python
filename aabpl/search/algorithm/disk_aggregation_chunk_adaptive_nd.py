"""
Chunk-block streaming search_and_aggregate — adaptive nd variant.

Differences from disk_aggregation_chunk.py
-------------------------------------------
* Per-chunk result streaming: sums are written into pts_source[sum_radius_names]
  immediately after each row chunk finishes (not all at once at the end), so peak
  accumulator memory is one chunk rather than all source points.
* Per-chunk adaptive nd selection via best_nd_tag (nd_choice.py).

Architecture
------------
All target points are sorted once by (row, col, subcell_nr) into
pts_vals_xy_full / tgt_rows_full / tgt_cols_full / tgt_scnr_full.
Source points are sorted by (row, col, region_and_trgl_id) as usual.

Source rows are processed in chunks.  For each chunk of source rows
[s_lo, s_hi], the only target points needed are those in rows
[s_lo - r_rows, s_hi + r_rows].  These are a contiguous slice of
pts_vals_xy_full (because it is sorted by row).  Two integer pointers
(blk_lo, blk_hi) advance monotonically through the full array —
no hash table, no eviction bookkeeping, no power-of-2 overallocation.

Within a block the contain path looks up precomputed cell sums directly
from grid.sums_array via codec key (one vectorised searchsorted call).
The overlap path gathers raw target points from the block using
block_cell_slices — a dict keyed by absolute codec int covering level-0
cells and all sub-cell levels up to nest_depth (built by _nest_block_slices
via quadtree descent on tgt_scnr_full).  Contain and overlap sub-cells are
disjoint by construction, matching disk_aggregation.py behaviour.

Chunk size is determined by an L3-budget calculation over sorted row costs.
"""

import itertools
import math
from sys import getsizeof as _sys_getsizeof
import numpy as np
from numpy import (
    zeros as _np_zeros,
    flatnonzero as _np_flatnonzero,
    append as _np_append,
    ones as _np_ones,
    searchsorted as _np_searchsorted,
    exp as _np_exp,
    array as _np_array,
)
from numpy.linalg import norm as np_norm
from aabpl.utils.progress import SearchProgress, progress_print
from aabpl.testing.test_performance import time_func_perf
from aabpl import config as _cfg
from ..point_assignment import cell_count_iter
from .nd_choice import best_nd_tag as _best_nd_tag
from .nd_choice import best_nd_tag_weighted as _best_nd_tag_weighted
from aabpl.utils.dist_matrix import batched_disk_sum as _batched_disk_sum

OVERLAP_BLOCK    = 256
ENABLE_COL_SPLIT  = True   # set False to disable column splitting (for benchmarking)
ENABLE_TALL_STRIPS = False  # allow strips taller than L3 budget when col-split compensates

# ---- adaptive-nd test flags ------------------------------------------------
# When _TEST_RANDOM_ND=True, each col-chunk picks a random nest_depth in
# 0.._TEST_ND_MAX (capped to native nest_depth).  Validates correctness by
# comparing against VALIDATE brute-force; set False to use only the native nd.
_TEST_RANDOM_ND = False
_TEST_ND_MIN    = -1      # min nd to sample (negative = super-cells; -1 groups 2×2 level-0 cells)
_TEST_ND_MAX    = 3       # max nd to sample randomly
# L2/L3 cache budgets are read from config at call time (_cfg.L2_BYTES / _cfg.L3_BYTES)
# so runtime changes to config take effect without reloading this module.
# L2_BYTES: col-split sub-budget (each col block should fit in L2 for the
#   distance-check OVERLAP_BLOCK loop).
# L3_BYTES: primary chunk-row budget — source-point block fits in L3 so
#   pre-aggregation and overlap scatter stay cache-hot.

from typing import NamedTuple

class _AreaWeightOpts(NamedTuple):
    area_weight:       object  # str or None, normalised
    quad_segs:         int
    exact_block_k:     int
    vec_exact:         bool
    keep_raw:          bool
    bnd_frac:          float
    bnd_frac_explicit: bool
    do_weight:         bool
    do_logit:          bool
    do_per_cell:       bool
    do_binary:         bool


def _parse_area_weight(area_weight, cells_rndm_sample) -> '_AreaWeightOpts':
    """Normalise and parse the area_weight option string into structured flags.

    Handles deprecated aliases, comma sub-options (quad_segs, keep_raw, block_k,
    vec), the optional =fraction suffix, and the cells_rndm_sample short-circuit.
    Returns an _AreaWeightOpts namedtuple; all callers should unpack immediately.
    """
    # All cells sampled → every disk is fully valid, no weighting needed.
    if isinstance(cells_rndm_sample, bool) and cells_rndm_sample:
        area_weight = None

    # Deprecated value aliases.
    _ALIASES = {'precise': 'exact', 'estimate': 'logit', 'estimate=0.5': 'logit=0.5'}
    if area_weight in _ALIASES:
        area_weight = _ALIASES[area_weight]

    # Comma sub-options, e.g. 'exact,quad_segs=8'.
    # block_k and vec=0 are undocumented benchmarking backdoors.
    quad_segs     = 8
    exact_block_k = 4      # benchmark confirmed block_k=4 is best
    vec_exact     = True   # vectorised path always used
    keep_raw      = False
    if isinstance(area_weight, str) and ',' in area_weight:
        parts       = [p.strip() for p in area_weight.split(',')]
        area_weight = parts[0]
        for part in parts[1:]:
            if part.startswith('quad_segs='):
                quad_segs = int(part.split('=', 1)[1])
            elif part == 'keep_raw':
                keep_raw = True
            elif part.startswith('block_k='):
                exact_block_k = max(1, int(part.split('=', 1)[1]))
            elif part == 'vec=0':
                vec_exact = False

    # Optional =fraction suffix, e.g. 'logit=0.5', 'flat=0.3'.
    bnd_frac          = 0.5
    bnd_frac_explicit = False
    if isinstance(area_weight, str) and '=' in area_weight:
        area_weight, fstr = area_weight.split('=', 1)
        bnd_frac          = float(fstr)
        bnd_frac_explicit = True

    _VALID = ('exact', 'logit', 'flat', 'binary', None)
    if area_weight not in _VALID:
        progress_print(f"area_weight={area_weight!r} unknown; ignoring. "
                       f"Valid options: {_VALID[:-1]}")
        area_weight = None

    return _AreaWeightOpts(
        area_weight       = area_weight,
        quad_segs         = quad_segs,
        exact_block_k     = exact_block_k,
        vec_exact         = vec_exact,
        keep_raw          = keep_raw,
        bnd_frac          = bnd_frac,
        bnd_frac_explicit = bnd_frac_explicit,
        do_weight         = area_weight is not None,
        do_logit          = area_weight in ('logit', 'flat', 'binary'),
        do_per_cell       = area_weight == 'logit',
        do_binary         = area_weight == 'binary',
    )


@time_func_perf
def search_and_aggregate(
    grid,
    pts_source,
    r,
    c=[],
    y='proj_lat',
    x='proj_lon',
    off_x='offset_x',
    off_y='offset_y',
    pts_target=None,
    row_name='id_y',
    col_name='id_x',
    cell_region_name='cell_region',
    suffix=None,
    exclude_self=True,
    silent=False,
    validate=False,
    area_weight=None,  # None | 'exact' | 'logit' | 'flat' | 'binary'
    plot_pt_disk=None,       # ignored; accepted for API compatibility with orig
    plan_only=False,  # if True: run planning only, return (depth_by_block,
                       # needs_level0) for aggregate_point_data_to_cells_adaptive_nd
                       # to consume, without touching grid.sums_array (may not
                       # exist yet) or executing any chunk.
    run_aggregation=False,  # DIAGNOSTIC/experimental, default off: if True, call
                             # aggregation inline (after planning) then fall
                             # through to execute in this same call, instead of
                             # the normal two-call plan_only flow. Under active
                             # debugging -- do not rely on this for correctness yet.
):
    _single_region = _cfg.SINGLE_REGION
    if pts_target is None:
        pts_target = pts_source

    # ---- grid internals ---------------------------------------------------------
    codec                      = grid._search_internals.cell_codec
    grid_spacing               = grid._search_internals.spacing
    nest_depth                 = grid._search_internals.nest_depth
    contain_region_mult        = grid._search_class.contain_region_mult
    shared_cntd_cells          = grid._search_class.shared_cntd_cells
    cntd_cells_by_region       = grid._search_class.region_and_trgl_id_to_distinct_cntd_cells
    ovlpd_cells_by_region      = grid._search_class.region_and_trgl_id_to_distinct_ovlpd_cells
    ovlpd_cells_by_cell_region = grid._search_class.region_id_to_ovlpd_cells

    n_pts           = len(pts_source)
    n_c             = len(c)
    r2              = r * r
    full_disk_area  = math.pi * r2
    zero_sum        = _np_zeros(n_c, dtype=float)

    # ---- parse / normalise area_weight -----------------------------------------
    cells_rndm_sample = grid._search_internals.cells_rndm_sample
    (area_weight, _quad_segs, _exact_block_k, _vec_exact, _keep_raw,
     _bnd_frac, _bnd_frac_explicit, _do_weight, _do_logit, _do_per_cell, _do_binary
    ) = _parse_area_weight(area_weight, cells_rndm_sample)



    # ---- valid-area setup (only when weighting requested) ----------------------
    if _do_weight:
        pad = -int(-r // grid_spacing)
        _row_lo = int(grid._search_internals.row_ids.min()) - pad
        _row_hi = int(grid._search_internals.row_ids.max()) + pad
        _col_lo = int(grid._search_internals.col_ids.min()) - pad
        _col_hi = int(grid._search_internals.col_ids.max()) + pad
        boundary_frac_raw = getattr(grid._search_internals, 'boundary_cell_valid_fraction', {})

        _sa = getattr(grid, 'study_area', None)
        if _sa is not None:
            _sp_va  = grid_spacing
            _x0_va  = grid._search_internals.bounds.xmin
            _y0_va  = grid._search_internals.bounds.ymin
            # Use cached search-grid classification from intersect_polygon_with_grid
            # (avoids rebuilding Shapely geometry objects every perform_search call).
            _cached_key = getattr(grid._search_internals, 'va_search_grid_key', None)
            _use_cache  = _cached_key == (_x0_va, _y0_va, _sp_va)
            if _use_cache:
                _va_fully  = grid._search_internals.va_fully_rc   # set of (r,c)
                _va_partly = grid._search_internals.va_partly_rc  # set of (r,c) or None
                _search_rc = set(
                    (r, c) for r in range(_row_lo, _row_hi + 1)
                           for c in range(_col_lo, _col_hi + 1)
                )
                if _do_binary:
                    _invalid_cells_rc = _search_rc - _va_fully
                elif _va_partly is not None:
                    _invalid_cells_rc = _search_rc - _va_fully - _va_partly
                else:
                    # partly not cached (binary mode was used in intersect call) — recompute
                    _use_cache = False
            if not _use_cache:
                try:
                    from shapely import box as _shp_box
                    from shapely import contains as _shp_contains
                    from shapely import intersects as _shp_intersects
                    from shapely import prepare as _shp_prepare
                    _shp_prepare(_sa)
                    _rr_va = np.arange(_row_lo, _row_hi + 1)
                    _cc_va = np.arange(_col_lo, _col_hi + 1)
                    _RR_va, _CC_va = np.meshgrid(_rr_va, _cc_va, indexing='ij')
                    _rf_va = _RR_va.ravel(); _cf_va = _CC_va.ravel()
                    _boxes_va = _shp_box(
                        _x0_va + _cf_va * _sp_va, _y0_va + _rf_va * _sp_va,
                        _x0_va + (_cf_va + 1) * _sp_va, _y0_va + (_rf_va + 1) * _sp_va,
                    )
                    _fin = _shp_contains(_sa, _boxes_va)
                    if _do_binary:
                        _invalid_cells_rc = set(
                            (int(_rf_va[i]), int(_cf_va[i])) for i in range(len(_rf_va))
                            if not _fin[i]
                        )
                        _partly_cells_rc = set()
                    else:
                        _pin = ~_fin & _shp_intersects(_sa, _boxes_va)
                        _invalid_cells_rc = set(
                            (int(_rf_va[i]), int(_cf_va[i])) for i in range(len(_rf_va))
                            if not _fin[i] and not _pin[i]
                        )
                        _partly_cells_rc = set(
                            (int(_rf_va[i]), int(_cf_va[i])) for i in range(len(_rf_va))
                            if _pin[i]
                        )
                except ImportError:
                    _invalid_cells_rc = set(
                        (int(rr), int(cc))
                        for rr in range(_row_lo, _row_hi + 1)
                        for cc in range(_col_lo, _col_hi + 1)
                        if (int(rr), int(cc)) not in cells_rndm_sample
                    )
        else:
            # No sample area: data-density fallback (same as orig)
            _invalid_cells_rc = set(
                (int(rr), int(cc))
                for rr in range(_row_lo, _row_hi + 1)
                for cc in range(_col_lo, _col_hi + 1)
                if (int(rr), int(cc)) not in cells_rndm_sample
            )

        invalid_cell_keys = set(int(codec.key(0, rr, cc)) for rr, cc in _invalid_cells_rc)
        boundary_cell_fracs = {int(codec.key(0, int(rr), int(cc))): float(f)
                               for (rr, cc), f in boundary_frac_raw.items()}
        # If compute_fractions=False, boundary_frac_raw is empty; build boundary_cell_keys from:
        # (1) partly_cells_rc computed in-line above (non-cache path), or
        # (2) va_partly_rc stored by intersect_polygon_with_grid (cache path).
        if not boundary_cell_fracs and _sa is not None:
            partial_cells_rc = locals().get('_partly_cells_rc') or getattr(grid._search_internals, 'va_partly_rc', None) or set()
            boundary_cell_keys = set(int(codec.key(0, rr, cc)) for rr, cc in partial_cells_rc)
        else:
            boundary_cell_keys = set(boundary_cell_fracs)
        get_cell_centroid = grid._search_internals.cell_centroid
        _cntd_cells_by_cell_region    = grid._search_class.region_id_to_cntd_cells
        _ovlpd_cells_by_cell_region_va = grid._search_class.region_id_to_ovlpd_cells
        # Level-0 codec-offset arrays keyed by cell_region_id (mirrors orig _l0_offset).
        # nested_cntd_cells / nested_ovlpd_cells store subcell offsets as fractional
        # grid-spacing units (e.g. level-1 at ±0.25/±0.75, level-3 at ±0.125 etc.).
        #
        # Valid-area accounting works at level-0 granularity: a cell is either fully
        # inside, a boundary cell, or outside the study area — sub-cell resolution
        # doesn't change that classification.  _l0_cntd_only / _l0_all_parents
        # intentionally stay at level-0 for this reason.
        def _l0_cntd_only(cells):
            """Level-0 entries only — whole cells guaranteed in disk."""
            if isinstance(cells, np.ndarray) and len(cells):
                mask = cells[:, 0] == 0
                l0 = cells[mask]
                return codec.offset_int(l0) if len(l0) else codec.offset_int(np.empty((0, 3), dtype=np.float32))
            return codec.offset_int(np.empty((0, 3), dtype=np.float32))

        def _l0_all_parents(cells):
            """Map every subcell to its parent level-0 cell (floor(dr+0.5)), deduplicated."""
            if isinstance(cells, np.ndarray) and len(cells):
                l0_rc = np.floor(cells[:, 1:] + 0.5).astype(np.int64)
                unique_rc = np.unique(l0_rc, axis=0)
                l0_arr = np.zeros((len(unique_rc), 3), dtype=np.float32)
                l0_arr[:, 1] = unique_rc[:, 0]
                l0_arr[:, 2] = unique_rc[:, 1]
                return codec.offset_int(l0_arr)
            return codec.offset_int(np.empty((0, 3), dtype=np.float32))

        contain_l0_offsets  = {rid: _l0_cntd_only(c2) for rid, c2 in _cntd_cells_by_cell_region.items()}
        # For overlap: union of parent cells from cntd subcells and ovlpd subcells,
        # minus the level-0 cells already handled by the contain path.
        contain_l0_offset_sets = {rid: set(int(k) for k in arr) for rid, arr in contain_l0_offsets.items()}
        def _ovlpd_parents(cntd_cells, ovlpd_cells, cntd_excl):
            all_cells = np.vstack([cntd_cells, ovlpd_cells]) if len(cntd_cells) and len(ovlpd_cells) else (
                cntd_cells if len(cntd_cells) else ovlpd_cells)
            raw = _l0_all_parents(all_cells)
            if cntd_excl and len(raw):
                raw = raw[~np.isin(raw, list(cntd_excl))]
            return raw
        overlap_l0_offsets = {rid: _ovlpd_parents(
                _cntd_cells_by_cell_region.get(rid, np.empty((0,3), np.float32)),
                _ovlpd_cells_by_cell_region_va.get(rid, np.empty((0,3), np.float32)),
                contain_l0_offset_sets.get(rid, set()))
            for rid in set(list(_cntd_cells_by_cell_region) + list(_ovlpd_cells_by_cell_region_va))}
        # single-region fast path: region_id_to_cntd/ovlpd_cells are empty dicts,
        # so contain/overlap_l0_offsets[0] would KeyError in the main loop.
        # Populate region 0 from shared_cntd_cells + single_region_ovlpd_cells.
        if _single_region and 0 not in contain_l0_offsets:
            contain_l0_offsets[0]  = _l0_cntd_only(grid._search_class.shared_cntd_cells)
            overlap_l0_offsets[0]  = _l0_all_parents(grid._search_class.single_region_ovlpd_cells)
            contain_l0_offset_sets[0] = set(int(k) for k in contain_l0_offsets[0])
        va_contain_cache = {}   # (home_key_int, cell_region_id) -> invalid_area_scalar
        _cell_area    = grid_spacing * grid_spacing
        # Half-diagonal of a grid cell: maximum distance from centroid to any corner.
        # Used as a cheap prefilter: centroid_dist > r + _half_diag guarantees no intersection.
        _half_diag    = grid_spacing * 0.5 * 2 ** 0.5
        if _do_logit:
            _logit_Q = 1 / (0.70628102 + _np_exp(0.57266908 * (grid_spacing / r - 2)))
            _logit_B = 1 / (-0.21443453 + _np_exp(0.76899004 * (grid_spacing / r - 2)))
        # Resolve _bnd_frac (valid fraction for boundary cells, flat/logit/smooth/exact)
        # and _non_fully_frac (valid fraction for all non-fully-inside cells, binary).
        # Priority: (1) explicit =f from user, (2) pre-sampled in intersect_polygon_with_grid,
        # (3) legacy per-call sample from _bkeys_frac, (4) default.
        _non_fully_frac = 0.0   # binary: average valid fraction of non-fully-inside cells
        if _do_weight and not _bnd_frac_explicit:
            if _do_binary:
                _non_fully_frac = getattr(grid._search_internals, 'non_fully_valid_fraction', 0.0)
            elif area_weight != 'logit':
                _pre = getattr(grid._search_internals, 'bnd_frac_sampled', None)
                if _pre is not None:
                    _bnd_frac = _pre
                elif boundary_cell_fracs:
                    _sample = list(boundary_cell_fracs.values())[:100]
                    _bnd_frac = float(np.mean(_sample))
        elif _do_binary and _bnd_frac_explicit:
            _non_fully_frac = _bnd_frac  # user said binary=0.4 → 0.4 valid fraction

    # ---- integer offset templates (absolute codec keys, all levels) -------------
    shared_cntd_offset     = codec.offset_int(shared_cntd_cells)
    cntd_offset_by_region  = {rid: codec.offset_int(cells) for rid, cells in cntd_cells_by_region.items()}
    ovlpd_offset_by_region = {rid: codec.offset_int(cells) for rid, cells in ovlpd_cells_by_region.items()}

    # ---- single-region template (undocumented backdoor: cfg.SINGLE_REGION=True) ----
    # Precomputed in build_region_cell_lookups (cached in disk_region_cache).
    # Contain = shared_cntd_offset; Overlap = all distinct_cntd + nested_ovlpd across
    # all regions, expanded to nest_depth level so granularities match, minus shared.
    # Always built (not gated on cfg.SINGLE_REGION): adaptive per-chunk nd selection
    # can dispatch a single_region=True tag (sT, s1T, smNT) on any chunk regardless
    # of that global flag, so the template needs to exist unconditionally.
    _sr_ovlpd_n_cells = 0  # used below to size candidate_buffer correctly
    single_region_ovlpd_offset = codec.offset_int(
        grid._search_class.single_region_ovlpd_cells)
    _sr_ovlpd_n_cells = len(single_region_ovlpd_offset)

    # ---- multi-nd precomputation (always: needed for adaptive best_nd_tag) -------
    # Build geometry for nd = 0 .. min(_TEST_ND_MAX, native_nd).
    # For each nd: offset arrays for shared contain and per-region contain/overlap.
    # At nd < native_nd: downgrade from native; at native_nd: reuse existing dicts.
    from .disk_geometry import (build_disk_region_lookups as _build_lookups,
                                downgrade_disk_region_cache_entry as _downgrade)
    _adaptive_nd_entries = {}  # nd -> cache entry dict (shared_cntd_cells, ...)
    _nd_max = nest_depth if not _TEST_RANDOM_ND else min(_TEST_ND_MAX, nest_depth)
    # Build/fetch top entry (already cached from grid setup)
    _top_entry = _build_lookups(None, grid_spacing=grid_spacing, r=r,
                                nest_depth=nest_depth, silent=True)
    _adaptive_nd_entries[nest_depth] = _top_entry
    for _nd in range(nest_depth - 1, -1, -1):
        _adaptive_nd_entries[_nd] = _downgrade(_adaptive_nd_entries[_nd + 1], _nd + 1,
                                                grid_spacing=grid_spacing, r=r)
    # Build per-nd offset lookup tables
    _nd_shared_cntd  = {}   # nd -> codec offset array
    _nd_cntd_by_reg  = {}   # nd -> {rid: codec offset array}
    _nd_ovlpd_by_reg = {}   # nd -> {rid: codec offset array}
    _nd_sr_ovlpd     = {}   # nd -> codec offset array (single-region only)
    _nd_sr_ovlpd_n   = {}   # nd -> int (for buffer sizing)
    for _nd in range(_nd_max + 1):
        _e = _adaptive_nd_entries[_nd]
        _nd_shared_cntd[_nd]  = codec.offset_int(_e['shared_cntd_cells'])
        _nd_cntd_by_reg[_nd]  = {rid: codec.offset_int(cells)
                                 for rid, cells in _e['region_and_trgl_id_to_distinct_cntd_cells'].items()}
        _nd_ovlpd_by_reg[_nd] = {rid: codec.offset_int(cells)
                                 for rid, cells in _e['region_and_trgl_id_to_distinct_ovlpd_cells'].items()}
        _sr_arr = _e.get('single_region_ovlpd_cells')
        if _sr_arr is not None and len(_sr_arr):
            _nd_sr_ovlpd[_nd]   = codec.offset_int(_sr_arr)
        else:
            _nd_sr_ovlpd[_nd]   = single_region_ovlpd_offset
        _nd_sr_ovlpd_n[_nd] = len(_nd_sr_ovlpd[_nd])
    _nd_range_test   = list(range(max(_TEST_ND_MIN, -nest_depth), _nd_max + 1))
    _nd_chunk_counts = {}   # (nd, single_region) tag -> int count
    if _nd_sr_ovlpd_n:
        _sr_ovlpd_n_cells = max(_sr_ovlpd_n_cells, max(_nd_sr_ovlpd_n.values()))

    # ---- precomputed sums array (from aggregate_point_data_to_cells) -----------
    # Contain path: instead of summing block_cell_sums in a Python loop, index
    # directly into the contiguous _sums_array using codec keys → one numpy call.
    # This block moved from here (function top) to right after the inline
    # aggregation call below -- see that call site for why.

    # ---- target point arrays sorted by (row, col, subcell_nr) -------------------
    # subcell_nr orders points within a cell by quadtree quadrant at each level,
    # enabling _nest_block_slices to slice sub-cell ranges via searchsorted.
    _xmin = grid._search_internals.bounds.xmin
    _ymin = grid._search_internals.bounds.ymin
    _ox = 0.5 + ((pts_target[x].values - _xmin) % grid_spacing - grid_spacing / 2) / grid_spacing
    _oy = 0.5 + ((pts_target[y].values - _ymin) % grid_spacing - grid_spacing / 2) / grid_spacing
    _tgt_scnr = np.zeros(len(pts_target), dtype=np.int64)
    for _i in range(1, nest_depth + 1):
        _n2 = 2**_i; _m2 = 2**((nest_depth - _i) * 2)
        _tgt_scnr += ((_ox // (1 / _n2)) % _n2 % 2 * _m2 +
                      (_oy // (1 / _n2)) % _n2 % 2 * 2 * _m2).astype(np.int64)
    _sort_idx    = np.lexsort((_tgt_scnr,
                               pts_target[col_name].values,
                               pts_target[row_name].values))
    tgt_rows_full    = pts_target[row_name].values[_sort_idx].astype(np.int64)
    tgt_cols_full    = pts_target[col_name].values[_sort_idx].astype(np.int64)
    tgt_scnr_full    = _tgt_scnr[_sort_idx]
    pts_vals_xy_full = pts_target[c + [x, y]].values[_sort_idx]

    n_tgt             = len(tgt_rows_full)
    tgt_row_lo_global = int(tgt_rows_full[0])  if n_tgt else 0
    tgt_row_hi_global = int(tgt_rows_full[-1]) if n_tgt else 0

    # ---- r_rows: grid-cell reach of a disk of radius r -------------------------
    r_rows = int(math.ceil(r / grid_spacing))
    _sr    = r / grid_spacing  # spacing ratio for nd_choice lookup

    # ---- fine-cell gather margin needed for a given (possibly super-cell) nd ---
    # r_rows above only accounts for the native (fine) grid_spacing. A super-cell
    # (nd<0) chunk's overlap template reaches farther in fine-cell terms
    # (effective coarse_spacing = grid_spacing * 2**k), so the row-band AND
    # column candidate-gather windows (need_row_lo/hi, tgt_col_lo/hi below) must
    # widen to that chunk's own reach or its overlap candidates outside the
    # window are silently never gathered -- not because the underlying target
    # points don't exist, but because they were never loaded into the local
    # block. Computed per (row-)strip from that strip's own resolved nd tags
    # (_sp_nd_tag/_sp_hot_tag, already known from planning -- no extra work),
    # not from the deepest tag anywhere in the whole benchmark table: using the
    # global worst case caused a 2**9-scale margin blowup applied to every
    # chunk regardless of what it actually needed. Memoized per nd since the
    # same handful of tags recur across many strips.
    # ---- exact nd re-decision for margin sizing ---------------------------------
    # Must mirror the REAL dispatch decision (made later in the col-chunk loop)
    # bit-for-bit, including the USE_WEIGHTED_ND_DECISION branch -- an earlier
    # version of this helper only replicated the non-weighted _best_nd_tag(sr, ppc)
    # formula and fell back to the (possibly-disagreeing) planning-time hint
    # whenever USE_WEIGHTED_ND_DECISION was on. Since that flag defaults to True,
    # the "fix" was a silent no-op in the default config: margin sizing kept using
    # the stale hint while the real dispatch used the weighted formula, so a
    # chunk could be margin-sized for nd=0 (small reach) but actually dispatched
    # to nd=-1 (needs a wider reach), causing severe undercounts. This replicates
    # the exact weighted computation (_sparse_box_ppc_and_counts + total-area ppc
    # + share_occupied -> _best_nd_tag_weighted) so margin sizing and real
    # dispatch can never disagree.
    def _exact_chunk_nd(_ecr_lo, _ecr_hi, _ecc_lo, _ecc_hi):
        if _cfg.USE_WEIGHTED_ND_DECISION:
            if _ecc_lo is not None:
                _e_reg_nz, _e_reg_cnt = _sparse_box_ppc_and_counts(_ecr_lo, _ecr_hi, _ecc_lo // _K, _ecc_hi // _K)
                _e_col_span_coarse = (_ecc_hi // _K) - (_ecc_lo // _K) + 1
            else:
                _e_reg_nz, _e_reg_cnt = _sparse_box_ppc_and_counts(_ecr_lo, _ecr_hi)
                _e_col_span_coarse = _max_cc_coarse
            _e_row_span_coarse = _ecr_hi - _ecr_lo + 1
            _e_total_area_fine = max(1, _e_row_span_coarse * _e_col_span_coarse * _K * _K)
            _e_total_pts_reg = float(_e_reg_cnt.sum()) if len(_e_reg_cnt) else 0.0
            _e_total_area_ppc = (_e_total_pts_reg / _e_total_area_fine) * math.pi * _sr * _sr
            _e_n_blocks_possible = max(1, _e_row_span_coarse * _e_col_span_coarse)
            _e_share_occupied = len(_e_reg_nz) / _e_n_blocks_possible
            return _best_nd_tag_weighted(_sr, _e_reg_nz, _e_reg_cnt, _e_total_area_ppc,
                                          share_occupied=_e_share_occupied)[0]
        else:
            if _ecc_lo is not None:
                _e_reg_nz = _sparse_box_ppc(_ecr_lo, _ecr_hi, _ecc_lo // _K, _ecc_hi // _K)
            else:
                _e_reg_nz = _sparse_box_ppc(_ecr_lo, _ecr_hi)
            _e_ppc_est = float(np.percentile(_e_reg_nz, _PPC_CHUNK_PERCENTILE)) if len(_e_reg_nz) > 0 else 1.0
            return _best_nd_tag(_sr, _e_ppc_est)[0]

    _margin_for_nd_cache = {}
    def _margin_for_nd(_nd):
        if _nd >= 0:
            return r_rows
        if _nd in _margin_for_nd_cache:
            return _margin_for_nd_cache[_nd]
        from aabpl.utils.cell_geometry import classify_disk_cells_by_level
        _k = -_nd
        _coarse_spacing = grid_spacing * (2 ** _k)
        _, _, _coarse_ovlpd, _ = classify_disk_cells_by_level(
            grid_spacing=_coarse_spacing, r=r, include_boundary=False, nest_depth=0,
        )
        _max_coarse_offset = 0
        if len(_coarse_ovlpd):
            _max_coarse_offset = int(max(
                max(abs(int(_c[0])), abs(int(_c[1]))) for _c in _coarse_ovlpd.tolist()
            ))
        _margin = max(r_rows, (_max_coarse_offset + 1) * (2 ** _k))
        _margin_for_nd_cache[_nd] = _margin
        return _margin

    # ---- super-cell (nd < 0) pre-computation -----------------------------------
    # Runs unconditionally so the production adaptive-nd path can use super-cells.
    # Lazily built (not eagerly for all depths): each level's contain+overlap parity
    # expansion is O(4^k) — building all 9 depths upfront regardless of whether
    # adaptive dispatch ever selects them is a severe, needless cost. Built on first
    # actual per-chunk use via _ensure_super_cell_nd, memoized per nd.
    _nd_super_shared_cntd  = {}  # nd -> list[4^k] of codec offset arrays (by parity)
    _nd_super_sr_ovlpd     = {}  # nd -> list[4^k] of codec offset arrays (by parity)
    _nd_super_sr_ovlpd_max = {}  # nd -> max len across parities
    from .nd_choice import (_BENCH as _nd_bench, _SR_VALS as _nd_sr_vals,
                             blended_cost_ms as _nd_blended_cost_ms)
    import numpy as _np_sr_key
    _sr_key_idx   = int(_np_sr_key.searchsorted(_nd_sr_vals, _sr, side='right')) - 1
    _sr_key_idx   = max(0, min(_sr_key_idx, len(_nd_sr_vals) - 1))
    _sr_bench_key = float(_nd_sr_vals[_sr_key_idx])  # closest benchmarked sr -- used for merge-decision time estimates too
    _bench_nd_min = min((tag[0] for tag in _nd_bench.get(_sr_bench_key, {})
                         if tag[0] < 0), default=0)

    def _ensure_super_cell_nd(_sc_nd):
        """Build (and cache) the contain+overlap parity templates for one super-cell
        depth, on first use. Cost is O(4^k) where k=-_sc_nd; only ever paid for
        depths adaptive dispatch actually selects for some chunk."""
        if _sc_nd in _nd_super_shared_cntd:
            return
        import numpy as _np_sc
        _k = -_sc_nd
        _coarse_spacing = grid_spacing * (2 ** _k)
        _ce = _build_lookups(None, grid_spacing=_coarse_spacing, r=r,
                             nest_depth=0, silent=True)

        def _expand_coarse(coarse_cells, pr, pc, _step=2**_k):
            """Expand coarse-grid offsets to fine-grid level-0 offsets."""
            if isinstance(coarse_cells, frozenset):
                _tuples = list(coarse_cells)
            else:
                _arr = _np_sc.array(coarse_cells, dtype=float)
                _tuples = [tuple(_arr[_i]) for _i in range(len(_arr))] if len(_arr) else []
            fine = set()
            for (_lv, _dr_c, _dc_c) in _tuples:
                for _i in range(_step):
                    for _j in range(_step):
                        fine.add((0, (int(_step * _dr_c + _i - pr),
                                     int(_step * _dc_c + _j - pc))))
            return frozenset(fine)

        _sc_cntd_parity  = []
        _sc_ovlpd_parity = []
        for _pr in range(2 ** _k):
            for _pc in range(2 ** _k):
                _fc = _expand_coarse(_ce['shared_cntd_cells'], _pr, _pc)
                _sc_cntd_parity.append(codec.offset_int(_fc))
                # Super-cells are single-region by construction (a coarse cell has
                # no per-fine-region breakdown) — needed independent of the
                # cfg.SINGLE_REGION global, since adaptive per-chunk nd selection
                # can dispatch a super-cell tag on any chunk regardless of that flag.
                _fo = _expand_coarse(
                    _ce.get('single_region_ovlpd_cells', frozenset()), _pr, _pc)
                _sc_ovlpd_parity.append(codec.offset_int(_fo))
        _nd_super_shared_cntd[_sc_nd]  = _sc_cntd_parity
        _nd_super_sr_ovlpd[_sc_nd]     = _sc_ovlpd_parity
        _nd_super_sr_ovlpd_max[_sc_nd] = max(len(a) for a in _sc_ovlpd_parity)

    if _TEST_RANDOM_ND:
        # Super-cell templates build lazily now (first per-chunk use, not here),
        # so there's nothing to report about them pre-loop — the old "N parities
        # per nd" printout no longer applies (the coarse-batch fast path has no
        # per-position parity concept at all; the old parity-template path only
        # still runs when _do_weight is set, and even then only per nd actually
        # dispatched, not eagerly for every nd in _nd_range_test).
        progress_print(f"[adaptive-nd] test range: nd={_nd_range_test}; "
                       f"sr_ovlpd cells per nd: "
                       f"{ {nd: (_nd_sr_ovlpd_n.get(nd, _nd_super_sr_ovlpd_max.get(nd, 'n/a'))) for nd in _nd_range_test} }")

    # ---- super-cell fast path: position-independent coarse-cell batching --------
    # For nd<0, coarse-cell size (grid_spacing * 2^k) grows exponentially with k
    # while r stays fixed, so r/coarse_spacing -> 0 for deep nd: essentially no
    # neighbour cell is ever "fully contained" (a contained cell must be entirely
    # within r, but the cell itself is bigger than r), and fine sub-position within
    # a coarse cell barely shifts which coarse neighbours could matter. So instead
    # of the old per-position (`_par`, up to 4^k variants) contain/overlap templates,
    # every point sharing a coarse home cell is batched together and checked against
    # a single fixed-size, position-independent neighbour block, sized generously
    # enough to cover the disk from ANY position within the home cell. Correctness
    # is unaffected by how loose that block is — the final distance check
    # (dx^2+dy^2<=r^2) is always exact; a looser block only means more (cheap,
    # since nd<0 is only ever chosen for sparse regions) candidates get checked.
    _super_tgt_index = {}  # nd -> dict(width, unique_keys, starts, ends, xy, vals) or None

    def _ensure_super_cell_target_index(_sc_nd):
        """Build (once, cached) the coarse-block candidate index for supercell nd.

        Restricted to the union of chunk regions that the pre-scan (see
        _nd_chunk_regions, built right after _strip_plans) found actually use
        this nd, dilated by _super_block_radius(nd) coarse blocks -- same margin
        _super_gather_candidates itself uses, so no true candidate is ever
        excluded. Previously this always indexed the FULL target grid the first
        time any chunk anywhere used a given nd, even if that nd's chunks only
        covered a small corner -- e.g. one nd=-6 chunk in an otherwise nd=0
        521k-point grid still paid for a full 521k-row index. _nd_chunk_regions
        being empty (random-nd testing mode, or an nd never seen in the
        pre-scan) falls back to the old full-grid behaviour, which is always
        safe, just not minimal.
        """
        if _sc_nd in _super_tgt_index:
            return
        if n_tgt == 0:
            _super_tgt_index[_sc_nd] = None
            return
        _k = -_sc_nd
        _regions = _nd_chunk_regions.get(_sc_nd)
        if _regions:
            _bR = _super_block_radius(_sc_nd)
            _dilate_fine = _bR * (2 ** _k)
            _mask = np.zeros(n_tgt, dtype=bool)
            for (_rlo, _rhi, _clo, _chi) in _regions:
                _mask |= ((tgt_rows_full >= _rlo - _dilate_fine) & (tgt_rows_full <= _rhi + _dilate_fine) &
                          (tgt_cols_full >= _clo - _dilate_fine) & (tgt_cols_full <= _chi + _dilate_fine))
            _sel_idx = np.flatnonzero(_mask)
            if len(_sel_idx) == 0:
                _super_tgt_index[_sc_nd] = None
                return
            _tcr = (tgt_rows_full[_sel_idx] >> _k).astype(np.int64)
            _tcc = (tgt_cols_full[_sel_idx] >> _k).astype(np.int64)
            _src_pts_full = pts_vals_xy_full[_sel_idx]
        else:
            _tcr = (tgt_rows_full >> _k).astype(np.int64)
            _tcc = (tgt_cols_full >> _k).astype(np.int64)
            _src_pts_full = pts_vals_xy_full
        _width = int(_tcc.max()) + 1
        _coarse_key = _tcr * _width + _tcc
        _order = np.argsort(_coarse_key, kind='stable')
        _cks = _coarse_key[_order]
        _bdry = np.empty(len(_cks), dtype=bool)
        _bdry[0] = True
        _bdry[1:] = _cks[1:] != _cks[:-1]
        _starts = np.flatnonzero(_bdry)
        _ends = np.empty_like(_starts)
        _ends[:-1] = _starts[1:]
        _ends[-1] = len(_cks)
        _super_tgt_index[_sc_nd] = dict(
            width=_width,
            unique_keys=_cks[_starts],
            starts=_starts, ends=_ends,
            xy=_src_pts_full[_order][:, -2:],
            vals=_src_pts_full[_order][:, :-2].astype(float),
        )

    def _super_block_radius(_sc_nd):
        """Supercells beyond home needed in each direction to guarantee no
        false negative: worst case a point sits at the very edge of its home
        supercell, so any point within radius r of it can be at most
        ceil(r/coarse_spacing) supercells further out. This is already tight
        -- the exact per-point distance check downstream (dx^2+dy^2<=r^2)
        is what actually filters candidates, so this only needs to guarantee
        no under-inclusion, not tightness beyond that.

        Previously had a flat "+1" on top of this, which meant every
        supercell depth searched at least a 5x5 block even once
        coarse_spacing >> r made a 3x3 (or smaller, for shallower sr) block
        already sufficient -- e.g. at sr=2.0, ceil(r/coarse_spacing) is
        already 1 (3x3) starting at nd=-1 and stays there for every deeper
        nd, so the "+1" was pure excess at every depth, worst at deep nd
        where it roughly triples the candidates gathered per group (see
        bench_overlap_micro.py profiling: avg candidates/group grew toward
        ~29% of n_tgt at nd=-10 before this fix).
        """
        _coarse_spacing = grid_spacing * (2 ** (-_sc_nd))
        return max(1, int(math.ceil(r / _coarse_spacing)))

    def _super_gather_candidates(_sc_nd, cr, cc, block_radius):
        """All target points within [cr-block_radius,cr+block_radius] x
        [cc-block_radius,cc+block_radius] coarse cells, concatenated."""
        idx = _super_tgt_index.get(_sc_nd)
        if idx is None:
            return np.empty((0, 2)), np.empty((0, n_c))
        _width = idx['width']
        _uk = idx['unique_keys']
        key_lo = (cr - block_radius) * _width
        key_hi = (cr + block_radius + 1) * _width
        i0 = int(np.searchsorted(_uk, key_lo, side='left'))
        i1 = int(np.searchsorted(_uk, key_hi, side='left'))
        if i0 >= i1:
            return np.empty((0, 2)), np.empty((0, n_c))
        _cc_vals = _uk[i0:i1] % _width
        _m = (_cc_vals >= cc - block_radius) & (_cc_vals <= cc + block_radius)
        if not _m.any():
            return np.empty((0, 2)), np.empty((0, n_c))
        _sel = np.flatnonzero(_m) + i0
        # Vectorised "concatenate multiple ranges" gather, mirroring the same
        # fix applied to gather_overlap_pts: one combined index array covering
        # all matched (start, end) ranges at once, then a single fancy-index
        # gather -- replaces what used to be a Python list comprehension
        # building one slice per matched supercell key + np.concatenate.
        _sel_starts = idx['starts'][_sel]
        _lens = idx['ends'][_sel] - _sel_starts
        _total = int(_lens.sum())
        if _total == 0:
            return np.empty((0, 2)), np.empty((0, n_c))
        _seg_offsets = np.cumsum(_lens) - _lens
        _combined_idx = (np.repeat(_sel_starts, _lens)
                          + np.arange(_total) - np.repeat(_seg_offsets, _lens))
        return idx['xy'][_combined_idx], idx['vals'][_combined_idx]

    def _process_super_cell_chunk(_sc_nd, sp_idx):
        """Batched, position-independent contain+overlap for a super-cell chunk.
        sp_idx: global indices (into rows/cols/point_xy/sums_within_disks) of the
        source points to process, already row/col-chunk-filtered."""
        nonlocal _n_super_grps, _n_super_candidates, _t_super_index_build, _t_super_loop
        nonlocal _t_super_gather, _t_super_distcheck
        if len(sp_idx) == 0:
            return
        _tib0 = _pt()
        _ensure_super_cell_target_index(_sc_nd)
        _t_super_index_build += _pt() - _tib0
        _tlp0 = _pt()
        _k  = -_sc_nd
        _bR = _super_block_radius(_sc_nd)
        _csr = rows[sp_idx] >> _k
        _csc = cols[sp_idx] >> _k
        _coarse_key = _csr.astype(np.int64) * 1_000_000_007 + _csc.astype(np.int64)
        _order = np.argsort(_coarse_key, kind='stable')
        _sp_sorted = sp_idx[_order]
        _cks = _coarse_key[_order]
        _bdry = np.empty(len(_cks), dtype=bool)
        _bdry[0] = True
        _bdry[1:] = _cks[1:] != _cks[:-1]
        _starts = np.flatnonzero(_bdry)
        _ends = np.empty_like(_starts)
        _ends[:-1] = _starts[1:]
        _ends[-1] = len(_cks)
        for _gi in range(len(_starts)):
            _g_idx = _sp_sorted[_starts[_gi]:_ends[_gi]]
            _cr = int(rows[_g_idx[0]] >> _k)
            _cc = int(cols[_g_idx[0]] >> _k)
            _tgg0 = _pt()
            _cand_xy, _cand_vals = _super_gather_candidates(_sc_nd, _cr, _cc, _bR)
            _t_super_gather += _pt() - _tgg0
            _n_super_grps += 1
            _n_super_candidates += len(_cand_xy)
            if len(_cand_xy) == 0:
                continue
            _tgd0 = _pt()
            _pxy = point_xy[_g_idx]
            sums_within_disks[_g_idx] += _batched_disk_sum(_pxy, _cand_xy, _cand_vals, r2)
            _t_super_distcheck += _pt() - _tgd0
        _t_super_loop += _pt() - _tlp0

    # ---- quadtree descent: build sub-cell entries in block_cell_slices ----------
    # Mirrors nest_next_lvl from aggregate_point_data_to_cells. Adds level-1..nd
    # entries keyed by absolute codec int so gather_overlap_pts can retrieve only
    # the sub-cell slices belonging to the overlap sub-cells of a boundary cell.
    def _nest_block_slices(blk_scnr, blk_rows, blk_cols, blk_n, block_cell_slices,
                           depth=None):
        """Add sub-cell (level 1..depth) entries to block_cell_slices via quadtree descent.

        depth defaults to nest_depth (native).  Pass a smaller value for adaptive-nd
        test to stop subdivision early; the scnr values are still encoded at the native
        depth so the bit-width is always correct for the stride formula.
        """
        if depth is None:
            depth = nest_depth
        if blk_n == 0 or depth == 0:
            return
        cell_bdry   = np.concatenate([[True],
                                      (blk_rows[1:] != blk_rows[:-1]) |
                                      (blk_cols[1:] != blk_cols[:-1])])
        cell_starts = _np_flatnonzero(cell_bdry)
        cell_ends   = _np_append(cell_starts[1:], blk_n)
        for cell_start, cell_end in zip(cell_starts.tolist(), cell_ends.tolist()):
            row0 = int(blk_rows[cell_start]); col0 = int(blk_cols[cell_start])
            stack = [(cell_start, cell_end, 0, 1, row0 - 0.5, col0 - 0.5)]
            while stack:
                pos_min, pos_max, subcell_base, lvl, sub_row, sub_col = stack.pop()
                # stride always uses nest_depth (scnr encoding) not chunk depth
                subcell_stride = 2**((nest_depth - lvl) * 2)
                quad_nrs = (blk_scnr[pos_min:pos_max] - subcell_base) // subcell_stride
                cur_pos = pos_min
                for qnr in range(4):
                    if cur_pos >= pos_max: break
                    rel_off = cur_pos - pos_min
                    if quad_nrs[rel_off] > qnr: continue
                    seg_len = int(np.searchsorted(quad_nrs[rel_off:], qnr + 1, side='left'))
                    next_pos        = cur_pos + seg_len
                    next_subcell_base = subcell_base + qnr * subcell_stride
                    child_row       = sub_row + qnr // 2 / (2**lvl)
                    child_col       = sub_col + qnr %  2 / (2**lvl)
                    cell_key        = int(codec.key(lvl,
                                                    child_row + 2**-(lvl + 1),
                                                    child_col + 2**-(lvl + 1)))
                    block_cell_slices[cell_key] = (cur_pos, next_pos)
                    if lvl + 1 <= depth:
                        stack.append((cur_pos, next_pos, next_subcell_base,
                                      lvl + 1, child_row, child_col))
                    cur_pos = next_pos

    # ---- chunk sizing via L2 budget ---------------------------------------------
    # Cost per target pt in the block: (n_c+2)*8 bytes (pts_vals_xy slice)
    # Cost per cell aggregate: CELL_DICT_BYTES_PER_ENTRY (see below)
    # We want  block_pts * pt_bytes + block_cells * cell_bytes  <= 0.9 * L2
    #
    # Estimate block_cells ~ block_pts * (avg cells per pt)
    # but we bound it simply: sort pts_per_row descending, cumsum until budget.
    pt_bytes   = (n_c + 2 + int(_do_weight)) * 8
    # block_cell_slices is a Python dict of int -> (start, end) tuple, not a
    # flat array -- real per-entry cost (measured via sys.getsizeof across
    # dict + boxed int keys + tuple values, temp/track_l3_memory.py) is
    # ~131 bytes, not the max(n_c,1)*8=8 bytes a flat-array model would
    # assume. Independent of n_c: the dict tracks (start,end) index pairs
    # only, never the n_c value columns themselves. Confirmed via regression
    # against measured real chunk memory across 15k/521k runs (R^2=0.994).
    # Was previously uncounted at its true size, causing chunks (especially
    # merged ones) to silently exceed the real L3 budget while this formula
    # still reported them as fitting.
    CELL_DICT_BYTES_PER_ENTRY = 131
    cell_bytes = CELL_DICT_BYTES_PER_ENTRY
    # Source-point bytes touched per point during the chunk's contain/overlap
    # pass (point_xy[start:end] -- a view, not a new allocation, but the
    # bytes still occupy cache lines while being read, so they count toward
    # L3 pressure the same as an allocation would). x,y as float64 = 16
    # bytes/point. Previously not counted in the row-strip sizing formula at
    # all (only used, via the wrong pt_bytes constant, in the column-split
    # memory check below).
    SRC_BYTES_PER_PT = 16

    # candidate buffer: hot in overlap pass, subtract from budget
    max_cells_per_region  = max(
        max((len(cells) for cells in ovlpd_cells_by_cell_region.values()), default=1),
        _sr_ovlpd_n_cells)
    # Per-level-0-cell target point counts, computed directly from raw target
    # points instead of cell_count_iter(grid) (which reads Layer 1's aggregated
    # id_to_vals_xy_by_lvl). This only ever needed point COUNTS per cell, never
    # the aggregated sums themselves, so it doesn't actually need aggregation
    # to have run -- confirmed by the AttributeError this raised when planning
    # was moved to run before aggregation. A direct bincount of tgt_rows_full/
    # tgt_cols_full (already available, already sorted by row) gives the same
    # per-cell counts without that dependency.
    if n_tgt:
        _pop_width = int(tgt_cols_full.max()) + 1
        _pop_key = tgt_rows_full.astype(np.int64) * _pop_width + tgt_cols_full.astype(np.int64)
        _pop_key_sorted = np.sort(_pop_key)
        _pop_bdry = _np_flatnonzero(np.concatenate([[True], _pop_key_sorted[1:] != _pop_key_sorted[:-1]]))
        _pop_counts = _np_append(_pop_bdry[1:], n_tgt) - _pop_bdry
        sorted_cell_pops = sorted(_pop_counts.tolist())
    else:
        sorted_cell_pops = []
    max_cands        = sum(sorted_cell_pops[-max_cells_per_region:]) if sorted_cell_pops else 1
    # candidate buffer is reused across overlap groups, not simultaneously hot with the
    # target block — don't subtract it from the block-sizing budget.
    l3_budget    = int(0.9 * _cfg.L3_BYTES)

    # pts_per_row from the sorted tgt array — O(n_unique_rows), one diff
    tgt_row_bdry  = _np_flatnonzero(_np_ones(n_tgt, dtype=bool)
                                    if n_tgt == 0 else
                                    np.concatenate([[True], tgt_rows_full[1:] != tgt_rows_full[:-1]]))
    pts_per_row   = _np_append(tgt_row_bdry[1:], n_tgt) - tgt_row_bdry  # length = n_unique_rows

    # unique (row,col) cells per unique row — needed for chunk cost estimate
    tgt_cell_bdry = _np_flatnonzero(
        np.concatenate([[True],
                        (tgt_rows_full[1:] != tgt_rows_full[:-1]) |
                        (tgt_cols_full[1:] != tgt_cols_full[:-1])]))
    # each row boundary is also a cell boundary; use searchsorted to count cells per row
    row_cell_pos   = _np_searchsorted(tgt_cell_bdry, _np_append(tgt_row_bdry, n_tgt), side='left')
    cells_per_row  = np.diff(row_cell_pos)   # shape (n_unique_rows,)

    # Source-point count per target row, estimated by scaling each row's
    # target-point share by the global n_src/n_tgt ratio -- cheap proxy for
    # a real per-row source binning, assumes source density is roughly
    # proportional to target density row-by-row (exact for uniform data,
    # approximate otherwise; still far better than the previous 0 estimate).
    src_per_row_est = pts_per_row * (n_pts / max(1, n_tgt))

    # joint cost per row, sort descending, cumsum
    row_cost       = (pts_per_row * pt_bytes + cells_per_row * cell_bytes
                       + src_per_row_est * SRC_BYTES_PER_PT)
    sorted_cost    = np.sort(row_cost)[::-1]
    # fixed: 2*r_rows overlap rows always in block regardless of chunk size
    fixed_cost     = int(np.sum(sorted_cost[:2 * r_rows]))
    remaining      = max(1, l3_budget - fixed_cost)
    cumcost        = np.cumsum(sorted_cost)
    fits           = int(np.searchsorted(cumcost, remaining, side='right'))
    _strip_fits    = max(1, fits)  # baseline strip height (rows that fit in L3)

    # col-split helper: greedy left-to-right scan, cuts on memory OR nd-zone change.
    # strip_rows: source-point row values (parallel to strip_cols) — used to query
    # the sparse coarse density map at each point's own (row//K, col//K) location rather than
    # averaging over the full strip row range.  This avoids blending ppc estimates
    # when the strip spans a vertical density gradient (e.g. dense top, sparse bottom),
    # which could cause the averaged ppc to straddle an nd-zone boundary and pick
    # the wrong nd for one end of the strip.
    #
    # Improvements vs naïve version:
    #   #1 Hysteresis: zone-change cuts require (MIN_COLS consecutive differing columns)
    #      OR (MIN_PTS source points in the pending zone) before committing.
    #      MIN_PTS catches sparse areas (few pts/col, real change spans many cols).
    #      MIN_COLS catches dense areas (many pts/col, MIN_PTS met on one noisy col,
    #      but multiple consecutive confirming columns is a stronger signal).
    #      Memory-overflow cuts are always immediate regardless of hysteresis.
    #      When hysteresis commits, the cut is placed at the first differing column (not
    #      the current one), minimising wrong-nd columns in the preceding chunk.
        #   #2 nd-aware memory accounting: effective bytes per source point scales with 2^nd
    #      (higher nd = finer grid = more target cells held in L3 simultaneously).
    #      Super-cells (nd<0) scale the SAME way (2^nd, nd negative), not a flat
    #      factor -- deeper supercells batch exponentially larger groups (nd=-15
    #      groups span 2^14x more area per side than nd=-1's), so per-source-point
    #      L3 pressure keeps shrinking with depth, it doesn't floor at some fixed
    #      "coarser than nd=0" discount. A flat 0.5 for all nd<0 was previously
    #      splitting very deep supercell chunks (e.g. nd=-15 on a 100k-point null-
    #      distribution search) into hundreds of tiny same-tag column bands that
    #      never got merged back, each paying full per-chunk setup overhead for
    #      what should have been a handful of large batched supercell ops.
    #   #3 Better zero-ppc fallback: empty coarse cells use the strip's mean ppc rather
    #      than a hardcoded 1.0, which avoids artificially anchoring to super-cell tags
    #      at sparse periphery cells.
    _HYS_MIN_COLS = 3
    # Percentile for chunk nd dispatch: biases nd choice toward the denser
    # sub-region within a chunk footprint rather than the area-weighted mean.
    # Row zone-change detection uses row-max ppc (separate signal, not a percentile).
    _PPC_CHUNK_PERCENTILE = 75
    _HYS_MIN_PTS  = 5

    def _nd_mem_scale(tag):
        nd = tag[0]
        if getattr(_cfg, '_DEBUG_REVERT_MEM_SCALE_FIX', False):
            return float(2 ** max(0, nd)) if nd >= 0 else 0.5
        return float(2.0 ** nd)

    def _greedy_col_chunks(strip_cols, strip_rows, reverse):
        if len(strip_cols) == 0:
            return [(None, None)]

        # Per-coarse-column p75 ppc from the sparse coarse cell map (cell-based,
        # not weighted by how many source points happen to sit in a column —
        # matches the strip-level _PPC_CHUNK_PERCENTILE convention and the
        # metric used by the nd-heterogeneity col-split below, so both
        # col-split paths now agree on what "this column's density" means).
        _cr_lo_g = int(strip_rows.min()) // _K
        _cr_hi_g = int(strip_rows.max()) // _K
        _cc_lo_g = int(strip_cols.min()) // _K
        _cc_hi_g = int(strip_cols.max()) // _K
        _cc_vals_g, _cc_p75_g = _sparse_col_percentile(
            _cr_lo_g, _cr_hi_g, _cc_lo_g, _cc_hi_g, _PPC_CHUNK_PERCENTILE)
        _cc_to_ppc = dict(zip(_cc_vals_g.tolist(), _cc_p75_g.tolist()))

        # Strip-level fallback ppc (#3): mean of nonzero coarse-column values,
        # avoids anchoring to ppc=1 which biases toward super-cell tags in
        # sparse-periphery coarse cells.
        _strip_ppc_default = float(np.mean(_cc_p75_g)) if len(_cc_p75_g) > 0 else 1.0

        # Aggregate to per-unique-column: sort by col, dedupe, map each fine
        # column to its coarse column's p75 value.
        _ord     = np.argsort(strip_cols, kind='stable')
        _scols   = strip_cols[_ord]
        _bdry    = np.empty(len(_scols), dtype=bool)
        _bdry[0] = True
        _bdry[1:] = _scols[1:] != _scols[:-1]
        _cs      = np.flatnonzero(_bdry)
        _ce      = np.empty_like(_cs); _ce[:-1] = _cs[1:]; _ce[-1] = len(_scols)
        unique_cols = _scols[_cs]
        counts      = (_ce - _cs).astype(np.intp)
        _ucc        = (unique_cols // _K).astype(np.int64)
        col_to_ppc  = {int(c): _cc_to_ppc.get(int(cc), _strip_ppc_default)
                       for c, cc in zip(unique_cols.tolist(), _ucc.tolist())}

        chunks = []; chunk_lo = int(unique_cols[0]); accum = 0; prev_col = chunk_lo
        ppc0 = col_to_ppc.get(chunk_lo, _strip_ppc_default) or _strip_ppc_default
        ppc_chunk_start = max(1.0, float(ppc0))
        zone_start = _best_nd_tag(_sr, ppc_chunk_start)

        # Hysteresis state (#1)
        zone_candidate       = None   # nd-tag of pending zone change
        zone_change_col      = None   # first column where zone_candidate was seen
        zone_change_prev_col = None   # column just before zone_change_col
        accum_before_change  = None   # accum value before zone_change_col was added
        zone_pending_cols = 0
        zone_pending_pts  = 0

        for col, cnt in zip(unique_cols.tolist(), counts.tolist()):
            ppc_here = col_to_ppc.get(col, _strip_ppc_default) or _strip_ppc_default
            ppc_here = max(1.0, float(ppc_here))
            zone_here = _best_nd_tag(_sr, ppc_here)
            # nd-aware memory (#2): scale by 2^nd relative to nd=0. pt_bytes
            # here estimates TARGET-side data pulled into cache per source
            # point processed (contain/overlap gathering), not the source
            # point's own footprint -- that's a separate, nd-independent
            # cost (raw x,y storage, SRC_BYTES_PER_PT) that was previously
            # missing entirely from this estimate.
            effective_bytes = int(cnt * (pt_bytes * _nd_mem_scale(zone_start) + SRC_BYTES_PER_PT))
            accum += effective_bytes

            # Both supercell tags (nd<0) share the same cheap, batched
            # _process_super_cell_chunk mechanism regardless of exact depth
            # -- unlike the general path, where nd genuinely changes what
            # gets built (quadtree depth, overlap templates). At very low
            # ppc (deep supercell territory), per-column density is noisy
            # by nature (a coarse column with 0 vs 1 vs 2 points swings its
            # local ppc hugely), which the hysteresis guard alone doesn't
            # fully damp -- a real 15km-radius/sparse-null-distribution run
            # fragmented into ~300 single-column nd=-15 chunks from exactly
            # this. A transition between two negative-nd tags isn't a real
            # mechanism change worth splitting over; only supercell<->general
            # (crossing nd=0) is.
            _both_supercell_col = zone_start[0] < 0 and zone_here[0] < 0
            if getattr(_cfg, '_DEBUG_REVERT_COL_ZONE_NOISE_FIX', False):
                _both_supercell_col = False
            _real_zone_change = (zone_here != zone_start) and not _both_supercell_col
            if _real_zone_change:
                if zone_candidate != zone_here:
                    # new candidate (or reversal) — reset hysteresis from this column
                    zone_candidate       = zone_here
                    zone_change_col      = col
                    zone_change_prev_col = prev_col
                    accum_before_change  = accum - effective_bytes
                    zone_pending_cols = 1
                    zone_pending_pts  = cnt
                else:
                    zone_pending_cols += 1
                    zone_pending_pts  += cnt
            else:
                # zone reverted — discard pending state
                zone_candidate = None; zone_change_col = None
                zone_change_prev_col = None; accum_before_change = None
                zone_pending_cols = 0; zone_pending_pts = 0

            mem_overflow  = col > chunk_lo and accum > remaining
            hysteresis_ok = (col > chunk_lo and zone_candidate is not None and
                             (zone_pending_cols >= _HYS_MIN_COLS or
                              zone_pending_pts  >= _HYS_MIN_PTS))

            if mem_overflow or hysteresis_ok:
                if hysteresis_ok and not mem_overflow:
                    # Cut at the column where zone change began — minimises wrong-nd
                    # columns in the preceding chunk.  Carry forward accumulated cost
                    # from that column onward into the new chunk.
                    chunks.append((chunk_lo, zone_change_prev_col))
                    chunk_lo = zone_change_col
                    accum    = accum - accum_before_change
                else:
                    # Memory pressure: cut immediately at current column boundary.
                    chunks.append((chunk_lo, prev_col))
                    chunk_lo = col
                    accum    = effective_bytes
                ppc_chunk_start = ppc_here
                zone_start = _best_nd_tag(_sr, ppc_chunk_start)
                zone_candidate = None; zone_change_col = None
                zone_change_prev_col = None; accum_before_change = None
                zone_pending_cols = 0; zone_pending_pts = 0

            prev_col = col

        chunks.append((chunk_lo, int(unique_cols[-1])))

        # Merge-back pass: adjacent chunks sharing the same nd tag have no
        # different-nd trade-off to weigh (nothing about processing would
        # differ by keeping them split -- same reasoning as the row-strip/
        # column-band merge passes elsewhere), so collapse them whenever the
        # combined range still fits L3. Catches boundaries that were purely
        # memory-driven (mem_overflow fired without an actual zone change) --
        # exactly what an over-conservative _nd_mem_scale used to produce for
        # deep supercell nd (see its docstring).
        if len(chunks) > 1 and not getattr(_cfg, '_DEBUG_REVERT_GREEDY_COL_MERGEBACK', False):
            def _chunk_stats(c_lo, c_hi):
                _i0 = int(np.searchsorted(unique_cols, c_lo, side='left'))
                _i1 = int(np.searchsorted(unique_cols, c_hi, side='right'))
                _n = int(counts[_i0:_i1].sum())
                _ppc = max(1.0, float(col_to_ppc.get(c_lo, _strip_ppc_default) or _strip_ppc_default))
                return _n, _best_nd_tag(_sr, _ppc)
            _merged_cols = []
            _cur_lo, _cur_hi = chunks[0]
            _cur_n, _cur_tag = _chunk_stats(_cur_lo, _cur_hi)
            for (c_lo, c_hi) in chunks[1:]:
                _n, _tag = _chunk_stats(c_lo, c_hi)
                _comb_n = _cur_n + _n
                _comb_bytes = _comb_n * (pt_bytes * _nd_mem_scale(_cur_tag) + SRC_BYTES_PER_PT)
                if _tag == _cur_tag and _comb_bytes <= remaining:
                    _cur_hi = c_hi
                    _cur_n  = _comb_n
                else:
                    _merged_cols.append((_cur_lo, _cur_hi))
                    _cur_lo, _cur_hi, _cur_n, _cur_tag = c_lo, c_hi, _n, _tag
            _merged_cols.append((_cur_lo, _cur_hi))
            chunks = _merged_cols

        return chunks[::-1] if reverse else chunks

    if getattr(_cfg, '_DEBUG_PER_CHUNK_CAPTURE', False):
        progress_print(
            f"Chunk-block: r_rows={r_rows}, chunk_rows={_strip_fits}, "
            f"col_split={'enabled' if ENABLE_COL_SPLIT else 'disabled'}, "
            f"tall_strips={'enabled' if ENABLE_TALL_STRIPS else 'disabled'}, "
            f"l3_budget={l3_budget//1024} KB, "
            f"~{(fixed_cost + int(np.sum(sorted_cost[:_strip_fits])))//1024} KB est. hot set")

    # ---- sort source points -----------------------------------------------------
    if suffix is None:
        suffix = '_' + str(r)
    sum_radius_names = [cn + suffix for cn in c]
    # Allocate writable numpy arrays for streaming output (pandas .values can be
    # read-only under COW; writing directly to these avoids that).
    _stream_out = {_cn: _np_zeros(n_pts) for _cn in sum_radius_names}
    pts_source[sum_radius_names] = 0

    # Snake sort: even rows L→R, odd rows R→L — keeps adjacent rows spatially
    # contiguous so the next chunk's leading edge is cache-warm.
    # Sort pts_source in-place (keeping the same object) so write-backs via
    # pts_source[col] = ... remain visible to the caller after we restore order.
    _snake_col = pts_source[col_name] * np.where(pts_source[row_name] % 2 == 0, 1, -1)
    pts_source['_snake_col']   = _snake_col
    pts_source['_orig_order']  = np.arange(n_pts)
    pts_source.sort_values([row_name, '_snake_col', 'region_and_trgl_id'], inplace=True)
    pts_source.drop(columns=['_snake_col'], inplace=True)
    point_xy        = pts_source[[x, y]].values
    point_offset    = pts_source[[off_x, off_y]].values if _do_weight else None
    rows            = pts_source[row_name].values.astype(np.int64)
    cols            = pts_source[col_name].values.astype(np.int64)
    cell_region     = pts_source[cell_region_name].values
    region_and_trgl = pts_source['region_and_trgl_id'].values
    # ---- exact super-cell classification (per unique source home cell / block) ----
    # For each unique home cell (or K×K block when block_k>1), build a super-cell
    # polygon covering all grid cells the disk can possibly reach, then intersect
    # with the study area once.  Per-point invalid_area is computed in the pre-loop
    # below using a fine-grained cell-ID trigger: only points whose disk touches a
    # non-fully-interior cell pay the Shapely disk.intersection(clip_poly) cost.
    # block_k>1 groups K×K home cells into one shared super-cell, reducing the
    # number of Shapely pre-computations by K² with no accuracy loss on the trigger.
    _do_exact_supercell = False
    if _do_weight and area_weight == 'exact':
        _sa_pc = getattr(grid, 'study_area', None)
        if _sa_pc is not None:
            try:
                try:
                    from shapely import unary_union as _shp_union
                except ImportError:
                    from shapely.ops import unary_union as _shp_union
                from shapely import (
                    box        as _shp_box_pc,
                    prepare    as _shp_prepare_pc,
                    contains   as _shp_cp_pc,
                    intersects as _shp_ix_pc,
                )
                from shapely.affinity import translate as _shp_translate_pc
                from shapely.geometry import Point as _ShpPoint
                _shp_prepare_pc(_sa_pc)
                _sp_pc = grid_spacing
                _x0_pc = grid._search_internals.bounds.xmin
                _y0_pc = grid._search_internals.bounds.ymin

                # Super-cell template relative to block origin (bottom-left of cell
                # (0,0) within the block).  For block_k=1 this is the standard
                # single-home-cell template; for block_k=K it expands to cover the
                # union of all K×K home cells' disk footprints.
                # A cell (dr, dc) is included when the rectangle-to-rectangle gap
                # between the K×K block [0, K*sp]² and the cell is ≤ r, i.e. ANY
                # point in the block can reach the cell.  This guarantees that for
                # every point p in a home cell within the block, disk(p) ⊆ super_cell,
                # so disk(p) ∩ study_area = disk(p) ∩ clip_poly exactly.
                # Gap formula: max(0, dc−K, −dc−1)·sp along each axis (rectangle gap).
                _K = _exact_block_k
                _sc_rel_boxes = []
                for _dr in range(-r_rows - 2, r_rows + _K + 2):
                    _dy = max(0.0, (_dr - _K) * _sp_pc, -(_dr + 1) * _sp_pc)
                    for _dc in range(-r_rows - 2, r_rows + _K + 2):
                        _dx = max(0.0, (_dc - _K) * _sp_pc, -(_dc + 1) * _sp_pc)
                        if math.sqrt(_dx * _dx + _dy * _dy) <= r:
                            _sc_rel_boxes.append(_shp_box_pc(
                                _dc * _sp_pc,       _dr * _sp_pc,
                                (_dc + 1) * _sp_pc, (_dr + 1) * _sp_pc,
                            ))
                _sc_tmpl = _shp_union(_sc_rel_boxes)   # computed once per rs() call

                # Classify each unique block (block_k=1 → each home cell is its own block).
                # Result: 'interior' | 'exterior' | clip_poly (= study_area ∩ super_cell).
                _unique_home_cells = set(zip(rows.tolist(), cols.tolist()))
                _unique_blocks = set((_hr // _K, _hc // _K) for _hr, _hc in _unique_home_cells)
                _block_clips = {}   # (block_r, block_c) -> 'interior'|'exterior'|clip_poly
                for _br, _bc in _unique_blocks:
                    _bx0 = _x0_pc + _bc * _K * _sp_pc
                    _by0 = _y0_pc + _br * _K * _sp_pc
                    _sc  = _shp_translate_pc(_sc_tmpl, xoff=_bx0, yoff=_by0)
                    if _shp_cp_pc(_sa_pc, _sc):
                        _block_clips[(_br, _bc)] = 'interior'
                    elif not _shp_ix_pc(_sa_pc, _sc):
                        _block_clips[(_br, _bc)] = 'exterior'
                    else:
                        _block_clips[(_br, _bc)] = _sa_pc.intersection(_sc)
                # Map each home cell to its block's clip result.
                _precise_home_cells = {
                    (_hr, _hc): _block_clips.get((_hr // _K, _hc // _K), 'interior')
                    for _hr, _hc in _unique_home_cells
                }
                _do_exact_supercell = True
            except ImportError:
                pass   # shapely unavailable → fall through to _invalid_keys path

    # group boundaries — compare (row, col) directly to avoid int64 overflow
    # that would occur with codec keys when row_stride * row_range overflows.
    cell_changed    = _np_ones(n_pts, dtype=bool)
    contain_changed = _np_ones(n_pts, dtype=bool)
    overlap_changed = _np_ones(n_pts, dtype=bool)
    if n_pts > 1:
        cell_changed[1:]    = (rows[1:] != rows[:-1]) | (cols[1:] != cols[:-1])
        if _single_region:
            # All points in the same cell share one template — only split at cell boundaries.
            contain_changed[1:] = cell_changed[1:]
            overlap_changed[1:] = cell_changed[1:]
        else:
            region_trgl_changed = region_and_trgl[1:] != region_and_trgl[:-1]
            contain_changed[1:] = (cell_changed[1:]
                                   | (cell_region[1:] // contain_region_mult != cell_region[:-1] // contain_region_mult)
                                   | region_trgl_changed)
            overlap_changed[1:] = (cell_changed[1:]
                                   | (cell_region[1:] % contain_region_mult != cell_region[:-1] % contain_region_mult)
                                   | region_trgl_changed)

    sums_within_disks = _np_zeros((n_pts, n_c))
    if _do_weight:
        invalid_area = _np_zeros(n_pts)

    # Sub-timers and counters (injected into func_timer_dict at end).
    from time import process_time as _pt, perf_counter as _wall_pt
    _t_contain      = 0.0
    _t_overlap      = 0.0
    _t_ov_gather    = 0.0   # gather_overlap_pts sub-step
    _t_ov_distcheck = 0.0   # distance-check matrix sub-step
    _n_contain_grps = 0     # total contain groups processed
    _n_overlap_grps = 0     # total overlap groups processed
    _n_candidates   = 0     # total candidate points gathered (overlap)
    _n_ov_keys      = 0     # total occupied overlap cell-keys touched (cache hit or miss)
    _n_super_grps       = 0  # supercell fast-path: total groups processed
    _n_super_candidates = 0  # supercell fast-path: total candidates gathered across all groups
    _t_super_index_build = 0.0  # supercell fast-path: one-time target-index build (per nd, memoized)
    _t_super_loop        = 0.0  # supercell fast-path: sort + per-group loop (gather + distance-check)
    _t_super_gather      = 0.0  # supercell fast-path: candidate-gathering sub-step within the loop
    _t_super_distcheck   = 0.0  # supercell fast-path: distance-check + sum sub-step within the loop
    _t_block_cell_slices_build = 0.0  # nd>=0 path: block_cell_slices dict build, rebuilt every chunk
    _n_ov_template  = 0     # total template size (len(codec_offsets)) before occupancy filter
    _t_ov_filter    = 0.0   # gather sub-step: raw_keys build + occupancy filter + frozenset
    _t_ov_cachelkup = 0.0   # gather sub-step: overlap_fp_cache dict .get()
    _t_ov_copy      = 0.0   # gather sub-step: slice lookups + candidate_buffer copy loop
    _n_contain_pts  = 0     # total target-point × source-point pairs via contain path
    _n_o_cache_hit  = 0     # overlap_fp_cache hits
    _n_o_cache_miss = 0
    _n_buffer_resizes = 0   # candidate_buffer grew past its upfront max_cands estimate

    # ---- pre-loop invalid_area pass for exact super-cell mode -------------------
    # Cell-ID trigger: only pay the Shapely cost for points whose level-0 disk
    # cells (contain ∪ overlap) include at least one non-fully-interior cell.
    # Points are grouped by block (row // K, col // K): all points in the same
    # block share one clip polygon, so buffer + intersection + area are vectorized
    # across the whole group in a single C-level call.
    if _do_exact_supercell:
        # ---- Opt 1: vectorised trigger mask ---------------------------------
        # Build the dilation footprint as a symmetric disk of radius r_rows+1
        # cells, directly from the search radius/grid spacing -- NOT derived
        # from the per-region contain/overlap templates.
        #
        # BUG (found via VALIDATE_AREA, this session): the per-region
        # disk-reach templates (_cntd_cells_by_cell_region /
        # _ovlpd_cells_by_cell_region_va) are keyed by sub-cell (triangle)
        # position and are inherently one-directional in row for every
        # region (fp_row_min stayed 0 -- never negative -- even after
        # unioning ALL regions, not just active ones; this isn't a "which
        # regions are active" issue, the templates themselves never carry
        # negative row offsets). Using them to build the trigger-dilation
        # kernel silently shrinks its reach in one direction: a source point
        # could have real invalid/boundary cells within its true disk radius
        # that the kernel could never propagate into its trigger status --
        # points near a study_area edge got silently treated as fully
        # interior. A kernel derived straight from the radius has no such
        # asymmetry by construction. (+1 cell margin for float/round-off
        # safety at the boundary.)
        _trig_reach = r_rows + 1
        footprint_offsets: set = {
            (_dr, _dc)
            for _dr in range(-_trig_reach, _trig_reach + 1)
            for _dc in range(-_trig_reach, _trig_reach + 1)
            if _dr * _dr + _dc * _dc <= _trig_reach * _trig_reach
        }
        # Build 2D boolean kernel centred at (0,0).
        if footprint_offsets:
            footprint_arr  = np.array(sorted(footprint_offsets), dtype=np.int64)
            fp_row_min, fp_col_min = int(footprint_arr[:, 0].min()), int(footprint_arr[:, 1].min())
            fp_n_rows = int(footprint_arr[:, 0].max()) - fp_row_min + 1
            fp_n_cols = int(footprint_arr[:, 1].max()) - fp_col_min + 1
            dilation_kernel = np.zeros((fp_n_rows, fp_n_cols), dtype=bool)
            dilation_kernel[footprint_arr[:, 0] - fp_row_min, footprint_arr[:, 1] - fp_col_min] = True
        else:
            dilation_kernel = np.ones((1, 1), dtype=bool)
            fp_row_min = fp_col_min = 0
        # Build the trigger grid: mark every invalid and boundary (rr, cc) cell.
        # _invalid_cells_rc and _partly_src are both sets of (rr, cc) int tuples
        # already in scope for the exact path (compute_fractions=False).
        trig_row_min = int(rows.min()) + fp_row_min   # source rows shifted by kernel reach
        trig_col_min = int(cols.min()) + fp_col_min
        trig_row_max = int(rows.max()) - fp_row_min   # fp_row_min is negative → subtract negates
        trig_col_max = int(cols.max()) - fp_col_min
        trig_n_rows = trig_row_max - trig_row_min + 1
        trig_n_cols = trig_col_max - trig_col_min + 1
        trigger_grid = np.zeros((trig_n_rows, trig_n_cols), dtype=np.int8)
        partial_cells_rc = locals().get('partial_cells_rc') or set()
        for _rr, _cc in itertools.chain(_invalid_cells_rc, partial_cells_rc):
            _ri, _ci = _rr - trig_row_min, _cc - trig_col_min
            if 0 <= _ri < trig_n_rows and 0 <= _ci < trig_n_cols:
                trigger_grid[_ri, _ci] = 1
        # Dilate with the disk-shaped footprint; source points index directly into
        # the result — no boundary issues since source rows ∈ [rows.min, rows.max]
        # which maps to [−fp_row_min, trig_n_rows+fp_row_min−1] after the offset, always valid.
        try:
            from scipy.ndimage import maximum_filter as _max_flt
            trigger_dilated = _max_flt(
                trigger_grid.view(bool), footprint=dilation_kernel,
                mode='constant', cval=False,
            )
        except ImportError:
            if not getattr(_parse_area_weight, '_scipy_hint_shown', False):
                progress_print(
                    "Tip: install scipy for faster trigger dilation in area_weight='exact' "
                    "(falling back to numpy shift loop).")
                _parse_area_weight._scipy_hint_shown = True
            tg_bool = trigger_grid.view(bool)
            trigger_dilated = np.zeros((trig_n_rows, trig_n_cols), dtype=bool)
            for dr, dc in footprint_offsets:
                r_src_lo = max(0, -dr); r_src_hi = trig_n_rows - max(0, dr)
                r_dst_lo = r_src_lo + dr; r_dst_hi = r_src_hi + dr
                c_src_lo = max(0, -dc); c_src_hi = trig_n_cols - max(0, dc)
                c_dst_lo = c_src_lo + dc; c_dst_hi = c_src_hi + dc
                trigger_dilated[r_dst_lo:r_dst_hi, c_dst_lo:c_dst_hi] |= \
                    tg_bool[r_src_lo:r_src_hi, c_src_lo:c_src_hi]
        src_row_idx = rows - trig_row_min
        src_col_idx = cols - trig_col_min
        # Clip to grid bounds — source points outside the padded grid are untriggered.
        in_trigger_bounds = (
            (src_row_idx >= 0) & (src_row_idx < trig_n_rows) &
            (src_col_idx >= 0) & (src_col_idx < trig_n_cols)
        )
        triggered_mask = np.zeros(n_pts, dtype=bool)
        triggered_mask[in_trigger_bounds] = trigger_dilated[src_row_idx[in_trigger_bounds], src_col_idx[in_trigger_bounds]]
        triggered_indices = np.flatnonzero(triggered_mask)

        if not _vec_exact:
            # old per-point loop (vec=0) — used as benchmark baseline; trigger
            # check now vectorised, body unchanged for fair comparison.
            for _i in triggered_indices.tolist():
                home_class = _precise_home_cells.get((int(rows[_i]), int(cols[_i])), 'interior')
                if home_class == 'exterior':
                    invalid_area[_i] = full_disk_area
                elif home_class != 'interior':
                    _disk = _ShpPoint(
                        float(point_xy[_i, 0]), float(point_xy[_i, 1])
                    ).buffer(r, quad_segs=_quad_segs)
                    invalid_area[_i] = full_disk_area - _disk.intersection(home_class).area
        else:
            import shapely as _shp_vec
            block_buckets = {}   # (br, bc) -> list of point indices needing clip
            for _i in triggered_indices.tolist():
                home_class = _precise_home_cells.get((int(rows[_i]), int(cols[_i])), 'interior')
                if home_class == 'exterior':
                    invalid_area[_i] = full_disk_area
                elif home_class != 'interior':
                    block_key = (int(rows[_i]) // _K, int(cols[_i]) // _K)
                    if block_key not in block_buckets:
                        block_buckets[block_key] = []
                    block_buckets[block_key].append(_i)
            for (_br, _bc), pt_indices in block_buckets.items():
                clip_poly = _block_clips[(_br, _bc)]
                if len(pt_indices) < 4:
                    for _i in pt_indices:
                        _disk = _ShpPoint(
                            float(point_xy[_i, 0]), float(point_xy[_i, 1])
                        ).buffer(r, quad_segs=_quad_segs)
                        invalid_area[_i] = full_disk_area - _disk.intersection(clip_poly).area
                else:
                    _shp_vec.prepare(clip_poly)
                    _xy   = point_xy[pt_indices]
                    _disks = _shp_vec.buffer(
                        _shp_vec.points(_xy[:, 0], _xy[:, 1]), r, quad_segs=_quad_segs)
                    invalid_area[pt_indices] = (
                        full_disk_area - _shp_vec.area(_shp_vec.intersection(clip_poly, _disks)))

    progress = SearchProgress(silent=silent, n_pts=n_pts)
    progress.start()
    next_threshold = progress.next_threshold

    # ---- source row boundaries --------------------------------------------------
    src_row_bdry  = _np_flatnonzero(
        np.concatenate([[True], rows[1:] != rows[:-1]]))
    src_unique_rows = rows[src_row_bdry]
    n_src_unique    = len(src_unique_rows)

    contain_starts_all = _np_flatnonzero(contain_changed)
    contain_ends_all   = _np_append(contain_starts_all[1:], n_pts)
    overlap_starts_all = _np_flatnonzero(overlap_changed)
    overlap_ends_all   = _np_append(overlap_starts_all[1:], n_pts)

    # candidate buffer — reused across overlap groups
    candidate_buffer = _np_zeros((max(max_cands, 1), n_c + 2), dtype=float)

    # ---- two pointers into tgt sorted array -------------------------------------
    blk_lo = 0  # index of first tgt pt in current block
    blk_hi = 0  # index past last tgt pt in current block
    # tracks which target row range is currently loaded
    loaded_row_lo = None
    loaded_row_hi = None

    # ---- coarse density map for per-chunk ppc estimation -----------------------
    # Sparse map: only OCCUPIED coarse cells are stored (as parallel arrays,
    # sorted in row-major order by coarse key = cr*width+cc), never a dense
    # (n_rows x n_cols) array. This matters because a single target point far
    # from the rest of the cloud can make the row/col span — and therefore a
    # dense array sized to it — enormous, even though the actual occupied-cell
    # count stays proportional to the data. `_max_cc_coarse` ("width") below is
    # only ever used as a cheap int64 key multiplier, never as an allocation size.
    # Built from TARGET pts (the data being summed), not source pts — ppc
    # reflects how many target pts each source cell will encounter.
    # K aligns with r_rows so each coarse cell is roughly one disk-radius wide.
    _K = max(4, r_rows)
    if n_tgt > 0:
        _cr_arr = (tgt_rows_full // _K).astype(np.int64)
        _cc_arr = (tgt_cols_full // _K).astype(np.int64)
        _max_cc_coarse = int(_cc_arr.max()) + 1
        _max_cr_coarse = int(_cr_arr.max()) + 1
        # flat keys: sort by (coarse_key, fine_key) so fine keys are ordered per group
        _coarse_key = _cr_arr * _max_cc_coarse + _cc_arr
        _fine_key   = tgt_rows_full * (_max_cc_coarse * _K + 1) + tgt_cols_full
        _order      = np.lexsort((_fine_key, _coarse_key))
        _ck_s = _coarse_key[_order]
        _fk_s = _fine_key[_order]
        # group boundaries in sorted array
        _new_coarse        = np.empty(n_tgt, dtype=bool)
        _new_coarse[0]     = True
        _new_coarse[1:]    = _ck_s[1:] != _ck_s[:-1]
        _new_fine          = np.empty(n_tgt, dtype=bool)
        _new_fine[0]       = True
        _new_fine[1:]      = _fk_s[1:] != _fk_s[:-1]  # True at coarse AND fine boundaries
        _cs_starts         = np.flatnonzero(_new_coarse)
        _cs_ends           = np.empty_like(_cs_starts)
        _cs_ends[:-1]      = _cs_starts[1:]
        _cs_ends[-1]       = n_tgt
        _n_pts_cc          = (_cs_ends - _cs_starts).astype(np.float32)
        # n unique fine cells per coarse group via cumsum of new_fine
        _fcum              = np.cumsum(_new_fine)
        _n_fine_cc         = (_fcum[_cs_ends - 1]
                              - np.concatenate([[0], _fcum[_cs_starts[1:] - 1]])).astype(np.float32)
        _ppc_cc            = _n_pts_cc / (_K * _K) * math.pi * _sr ** 2
        # sparse arrays, length = n_occupied_coarse_cells (NOT max_cr*max_cc)
        _ck_unique         = _ck_s[_cs_starts]          # sorted row-major coarse keys
        # 1-D row profiles for density-guided row cuts, built by grouping the
        # already-sparse per-cell arrays by cr (row-major sort keeps same-cr
        # entries contiguous), never touching a dense (row x col) structure.
        # Mean ppc: used as the zone anchor for the current strip's start row —
        #   stable, not distorted by single hot cells.
        # Max ppc: used as the zone-change trigger for subsequent rows —
        #   catches a single dense hot-spot column that mean would bury.
        _cr_of_cc      = _ck_unique // _max_cc_coarse
        _row_bdry      = np.empty(len(_ck_unique), dtype=bool)
        _row_bdry[0]   = True
        _row_bdry[1:]  = _cr_of_cc[1:] != _cr_of_cc[:-1]
        _row_starts    = np.flatnonzero(_row_bdry)
        _row_cr_unique = _cr_of_cc[_row_starts]
        _row_pts_s     = np.add.reduceat(_n_pts_cc, _row_starts)
        _row_fine_s    = np.add.reduceat(_n_fine_cc, _row_starts)
        _row_ppc_max_s = np.maximum.reduceat(_ppc_cc, _row_starts)
        _row_ppc_mean_s = np.where(_row_fine_s > 0, _row_pts_s / _row_fine_s, 0.0).astype(np.float32)
    else:  # n_tgt == 0
        _max_cr_coarse = 1; _max_cc_coarse = 1
        _ck_unique = np.empty(0, dtype=np.int64)
        _ppc_cc    = np.empty(0, dtype=np.float32)
        _n_pts_cc  = np.empty(0, dtype=np.float32)
        _row_cr_unique  = np.empty(0, dtype=np.int64)
        _row_ppc_mean_s = np.empty(0, dtype=np.float32)
        _row_ppc_max_s  = np.empty(0, dtype=np.float32)

    def _sparse_row_lookup(cr, unique_arr, val_arr, default=0.0):
        """Single-row value lookup (mean/max ppc) via binary search — O(log n)."""
        i = int(np.searchsorted(unique_arr, cr))
        if i < len(unique_arr) and unique_arr[i] == cr:
            return float(val_arr[i])
        return default

    def _sparse_box_ppc(cr_lo, cr_hi, cc_lo=None, cc_hi=None):
        """All occupied-cell ppc values with cr in [cr_lo, cr_hi] (optionally cc too)."""
        if len(_ck_unique) == 0:
            return _ppc_cc[:0]
        key_lo = cr_lo * _max_cc_coarse
        key_hi = (cr_hi + 1) * _max_cc_coarse
        i0 = int(np.searchsorted(_ck_unique, key_lo, side='left'))
        i1 = int(np.searchsorted(_ck_unique, key_hi, side='left'))
        if cc_lo is None:
            return _ppc_cc[i0:i1]
        cc_vals = _ck_unique[i0:i1] % _max_cc_coarse
        m = (cc_vals >= cc_lo) & (cc_vals <= cc_hi)
        return _ppc_cc[i0:i1][m]

    def _sparse_box_ppc_and_counts(cr_lo, cr_hi, cc_lo=None, cc_hi=None):
        """Same slicing as _sparse_box_ppc, but also returns each occupied
        block's raw point count (_n_pts_cc) alongside its ppc -- needed for
        best_nd_tag_weighted's point-count-weighted sum (see cfg.
        USE_WEIGHTED_ND_DECISION). Point counts are what _ppc_cc was derived
        from in the first place (_ppc_cc = _n_pts_cc/(K*K)*pi*sr^2), sliced
        identically so the two arrays stay in matching order."""
        if len(_ck_unique) == 0:
            return _ppc_cc[:0], _n_pts_cc[:0]
        key_lo = cr_lo * _max_cc_coarse
        key_hi = (cr_hi + 1) * _max_cc_coarse
        i0 = int(np.searchsorted(_ck_unique, key_lo, side='left'))
        i1 = int(np.searchsorted(_ck_unique, key_hi, side='left'))
        if cc_lo is None:
            return _ppc_cc[i0:i1], _n_pts_cc[i0:i1]
        cc_vals = _ck_unique[i0:i1] % _max_cc_coarse
        m = (cc_vals >= cc_lo) & (cc_vals <= cc_hi)
        return _ppc_cc[i0:i1][m], _n_pts_cc[i0:i1][m]

    def _sparse_col_percentile(cr_lo, cr_hi, cc_lo, cc_hi, pct):
        """Per-occupied-column percentile of cell ppc within [cr_lo,cr_hi] x [cc_lo,cc_hi].

        Returns (cc_values, pct_ppc) — sparse, only occupied columns — rather
        than a dense array indexed 0..n_coarse_cols-1 (n_coarse_cols can be the
        full grid width; bounding to the strip's cc range keeps this cheap).

        Cell-based (unweighted by how many source points happen to sit in a
        column), unlike a per-point average — a column with many points at
        moderate density shouldn't outrank a column with fewer points at
        genuinely higher spatial density. pct=100 recovers the old max
        behaviour (used as the strip-level hot-spot trigger); pct=75 is used
        for actual per-column zone classification, matching the strip-level
        _PPC_CHUNK_PERCENTILE convention.
        """
        if len(_ck_unique) == 0:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)
        key_lo = cr_lo * _max_cc_coarse
        key_hi = (cr_hi + 1) * _max_cc_coarse
        i0 = int(np.searchsorted(_ck_unique, key_lo, side='left'))
        i1 = int(np.searchsorted(_ck_unique, key_hi, side='left'))
        cc_vals = _ck_unique[i0:i1] % _max_cc_coarse
        vals    = _ppc_cc[i0:i1]
        m = (cc_vals >= cc_lo) & (cc_vals <= cc_hi)
        cc_vals = cc_vals[m]; vals = vals[m]
        if len(cc_vals) == 0:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)
        order = np.argsort(cc_vals, kind='stable')
        cc_s = cc_vals[order]; v_s = vals[order]
        bdry = np.empty(len(cc_s), dtype=bool)
        bdry[0] = True
        bdry[1:] = cc_s[1:] != cc_s[:-1]
        starts = np.flatnonzero(bdry)
        ends = np.empty_like(starts); ends[:-1] = starts[1:]; ends[-1] = len(cc_s)
        out = np.empty(len(starts), dtype=np.float32)
        for _gi in range(len(starts)):
            out[_gi] = np.percentile(v_s[starts[_gi]:ends[_gi]], pct)
        return cc_s[starts], out

    n_chunks_total = math.ceil(n_src_unique / _strip_fits)  # updated per-strip
    chunk_num      = 0

    # ---- strip pre-planning pass (v6) ------------------------------------------
    # Phase 1: greedy row scan with look-ahead dampening.
    #   Cut on zone change only after ≥2 consecutive coarse rows in the new zone
    #   (prevents spurious cuts from isolated hot cells).
    # Phase 2: merge consecutive strips with the same p75-resolved nd that fit in
    #   L3, guarded by a p90 check — skip merge if the combined strip would have
    #   p90 nd ≠ p75 nd (prevents over-tall strips that get col-fragmented).
    # Phase 3: within each merged strip, if p90 nd ≠ p75 nd, split into hot/cold
    #   column bands; adjacent same-nd bands are then collapsed (within-strip
    #   col-merge) and bands smaller than _MIN_COLSPLIT_PTS are not split.
    _MIN_COLSPLIT_PTS  = 1000   # minimum source points per col-split band
    _LOOKAHEAD_ROWS    = 2      # coarse rows of new zone required before cutting

    _planned_strips = []   # list of (row_chunk_idx, chunk_rows)
    _tmp_idx = 0
    while _tmp_idx < n_src_unique:
        _tmp_pt_s   = int(src_row_bdry[_tmp_idx])
        _tmp_cr_s   = int(src_unique_rows[_tmp_idx]) // _K
        _tmp_ppc_rs = _sparse_row_lookup(_tmp_cr_s, _row_cr_unique, _row_ppc_mean_s)
        if _tmp_ppc_rs <= 0.0:
            _tmp_ppc_rs = 1.0
        _tmp_zone    = _best_nd_tag(_sr, _tmp_ppc_rs)
        _tmp_mem_scale = _nd_mem_scale(_tmp_zone)
        _tmp_h       = 1
        _new_zone_run = 0  # consecutive coarse rows already seen in the candidate new zone
        while _tmp_idx + _tmp_h < n_src_unique:
            _tmp_pt_e = int(src_row_bdry[_tmp_idx + _tmp_h])
            # scaled by the resolved zone's actual per-point memory footprint
            # (2^nd, both directions -- see _nd_mem_scale) — not flat
            # pt_bytes, which silently let a strip already committed to a
            # high nd blow past the L3 budget it was supposed to respect.
            if (_tmp_pt_e - _tmp_pt_s) * pt_bytes * _tmp_mem_scale > remaining:
                break
            _tmp_cr_n   = int(src_unique_rows[_tmp_idx + _tmp_h]) // _K
            _tmp_ppc_nm = _sparse_row_lookup(_tmp_cr_n, _row_cr_unique, _row_ppc_max_s)
            _tmp_zone_nm = _best_nd_tag(_sr, _tmp_ppc_nm) if _tmp_ppc_nm > 0.0 else _tmp_zone
            # Both supercell tags (nd<0) share the same cheap batched
            # mechanism regardless of exact depth -- see the matching note
            # in _greedy_col_chunks. Noisy per-row density at very low ppc
            # shouldn't cut a strip over a same-mechanism nd wobble.
            _both_supercell_row = _tmp_zone[0] < 0 and _tmp_zone_nm[0] < 0
            if getattr(_cfg, '_DEBUG_REVERT_ROW_ZONE_NOISE_FIX', False):
                _both_supercell_row = False
            if _tmp_ppc_nm > 0.0 and _tmp_zone_nm != _tmp_zone and not _both_supercell_row:
                _new_zone_run += 1
                if _new_zone_run >= _LOOKAHEAD_ROWS:
                    break  # confirmed zone change: cut here
                # not yet confirmed — include this row tentatively
            else:
                _new_zone_run = 0  # reset: back to same zone
            _tmp_h += 1
        _planned_strips.append((_tmp_idx, _tmp_h))
        _tmp_idx += _tmp_h

    # helper: resolve nd tag and ppc from coarse-map percentile for a strip
    def _strip_nd_tag(idx, h):
        _cr_lo_ = int(src_unique_rows[idx]) // _K
        _cr_hi_ = int(src_unique_rows[idx + h - 1]) // _K
        _nz_ = _sparse_box_ppc(_cr_lo_, _cr_hi_)
        _ppc_ = float(np.percentile(_nz_, _PPC_CHUNK_PERCENTILE)) if len(_nz_) > 0 else 1.0
        return _best_nd_tag(_sr, _ppc_), _ppc_

    def _strip_p90_tag(idx, h):
        _cr_lo_ = int(src_unique_rows[idx]) // _K
        _cr_hi_ = int(src_unique_rows[idx + h - 1]) // _K
        _nz_ = _sparse_box_ppc(_cr_lo_, _cr_hi_)
        _ppc90_ = float(np.percentile(_nz_, 90)) if len(_nz_) > 0 else 1.0
        return _best_nd_tag(_sr, _ppc90_), _ppc90_

    # merge pass (v6): collapse consecutive strips with same p75 nd, L3 check,
    # and p90 guard (don't merge if combined strip becomes nd-heterogeneous).
    _merged_strips = []   # (row_chunk_idx, chunk_rows, nd_tag, ppc_est, hot_nd_tag)
    _si = 0
    while _si < len(_planned_strips):
        _m_idx, _m_h = _planned_strips[_si]
        _m_tag, _m_ppc = _strip_nd_tag(_m_idx, _m_h)
        _sj = _si + 1
        while _sj < len(_planned_strips):
            _n_idx, _n_h = _planned_strips[_sj]
            _n_tag, _n_ppc = _strip_nd_tag(_n_idx, _n_h)
            # Both supercell tags (nd<0) share the same cheap batched
            # mechanism regardless of exact depth -- see _greedy_col_chunks'
            # matching note. Don't refuse a merge just because per-strip
            # density noise flipped the "best" supercell depth by one level.
            _both_supercell_strip = _m_tag[0] < 0 and _n_tag[0] < 0
            if getattr(_cfg, '_DEBUG_REVERT_ROW_MERGE_RELAX', False):
                _both_supercell_strip = False
            if _n_tag != _m_tag and not _both_supercell_strip:
                break
            # combined L3 check, scaled by _m_tag's actual memory footprint
            # (same reasoning as the pre-planning check above)
            _comb_pt_s = int(src_row_bdry[_m_idx])
            _end_row   = _n_idx + _n_h
            _comb_pt_e = (int(src_row_bdry[_end_row]) if _end_row < n_src_unique else n_pts)
            if (_comb_pt_e - _comb_pt_s) * pt_bytes * _nd_mem_scale(_m_tag) > remaining:
                break
            # _n_tag == _m_tag is already guaranteed by the while-condition
            # above, so there's no per-chunk overhead trade-off to weigh here
            # (no different-nd combination is ever considered at this point,
            # unlike the column-band merge below) -- two strips that already
            # agree on nd and fit the L3 budget together are strictly better
            # merged than kept separate, so just merge.
            _m_h   += _n_h
            _m_ppc  = max(_m_ppc, _n_ppc)
            _sj    += 1
        # check p90 tag for nd-heterogeneity col-split hint
        _hot_tag, _ = _strip_p90_tag(_m_idx, _m_h)
        _merged_strips.append((_m_idx, _m_h, _m_tag, _m_ppc, _hot_tag))
        _si = _sj

    # Debug/experiment override: force an explicit row-strip merge pattern
    # instead of the greedy nd-homogeneity merge above. cfg._DEBUG_FORCE_ROW_MERGE_PATTERN
    # is a list of bools, one per boundary between consecutive ATOMIC strips
    # in _planned_strips (length len(_planned_strips)-1): True merges that
    # boundary, False keeps it split. Lets chunk-overhead experiments force
    # an EXACT merge/no-merge pattern across real atomic row-strips, rather
    # than indirectly coaxing the L3 budget / nd-homogeneity heuristic into
    # producing a desired pattern. Off by default. Still uses the real
    # _strip_nd_tag/_strip_p90_tag on whatever range results, so the nd
    # choice for a forced-merged range reflects its actual data, only the
    # merge/no-merge decision itself is forced.
    _dbg_row_pattern = getattr(_cfg, '_DEBUG_FORCE_ROW_MERGE_PATTERN', None)
    if _dbg_row_pattern is not None:
        _merged_strips = []
        _si = 0
        while _si < len(_planned_strips):
            _m_idx, _m_h = _planned_strips[_si]
            _sj = _si + 1
            while _sj < len(_planned_strips) and _sj - 1 < len(_dbg_row_pattern) and _dbg_row_pattern[_sj - 1]:
                _n_idx, _n_h = _planned_strips[_sj]
                _m_h += _n_h
                _sj += 1
            _m_tag, _m_ppc = _strip_nd_tag(_m_idx, _m_h)
            _hot_tag, _ = _strip_p90_tag(_m_idx, _m_h)
            _merged_strips.append((_m_idx, _m_h, _m_tag, _m_ppc, _hot_tag))
            _si = _sj
        if getattr(_cfg, '_DEBUG_CAPTURE_MERGED_STRIP_COUNT', False):
            _cfg._DEBUG_MERGED_STRIP_COUNT = (
                getattr(_cfg, '_DEBUG_MERGED_STRIP_COUNT', 0) + len(_merged_strips))

    # ---- col-chunk planning pre-pass (v7) ---------------------------------------
    # Resolves the same col-split decision the main loop used to compute inline,
    # for every merged strip, before any per-chunk work (target-band loading,
    # contain/overlap slicing, actual point processing) begins. Nothing in this
    # decision depends on anything computed by earlier per-chunk work -- it only
    # reads the coarse density map and each strip's own pre-resolved metadata
    # (_sp_nd_tag/_sp_hot_tag from the merge pass above) -- so hoisting it here
    # is behaviour-preserving, not a algorithmic change. Doing this upfront means
    # the full (region, nd_tag) chunk plan is known before the hot loop starts,
    # which is what a future selective-quadtree-depth build (only constructing
    # the target aggregation levels a chunk's resolved nd actually needs) would
    # key off of.
    def _plan_col_chunks(_pci_row_chunk_idx, _pci_chunk_rows, _pci_sp_nd_tag,
                          _pci_sp_hot_tag, _pci_reverse):
        _pt_s      = int(src_row_bdry[_pci_row_chunk_idx])
        _pt_e      = (int(src_row_bdry[_pci_row_chunk_idx + _pci_chunk_rows])
                      if _pci_row_chunk_idx + _pci_chunk_rows < n_src_unique else n_pts)
        _strip_cost = (_pt_e - _pt_s) * pt_bytes

        # col-split (v6): L3-overflow split OR nd-heterogeneity split.
        # After splitting, apply within-strip col-merge (collapse adjacent same-nd
        # bands) and enforce a minimum src_pts guard per band.
        _cr_lo_strip = int(src_unique_rows[_pci_row_chunk_idx]) // _K
        _cr_hi_strip = int(src_unique_rows[_pci_row_chunk_idx + _pci_chunk_rows - 1]) // _K
        if _strip_cost > remaining and ENABLE_COL_SPLIT:
            _col_chunks = _greedy_col_chunks(cols[_pt_s:_pt_e], rows[_pt_s:_pt_e], _pci_reverse)
        elif (_pci_sp_hot_tag != _pci_sp_nd_tag
              and not (_pci_sp_hot_tag[0] < 0 and _pci_sp_nd_tag[0] < 0
                       and not getattr(_cfg, '_DEBUG_REVERT_COL_SPLIT_TRIGGER_FIX', False))
              and ENABLE_COL_SPLIT):
            # nd-heterogeneity split: assign each coarse column its best nd tag
            # based on that column's max ppc across the strip's coarse rows.
            # Bounded to the strip's own source-column footprint (not the full
            # target extent) so this stays cheap even when the grid spans a
            # huge, mostly-empty area (e.g. one far-away outlier point).
            _src_cols_strip = cols[_pt_s:_pt_e]
            _src_rows_strip = rows[_pt_s:_pt_e]
            _cc_lo_bound = int(_src_cols_strip.min()) // _K
            _cc_hi_bound = int(_src_cols_strip.max()) // _K
            _n_coarse_cols = _cc_hi_bound - _cc_lo_bound + 1
            _cc_vals_sp, _cc_p75_sp = _sparse_col_percentile(
                _cr_lo_strip, _cr_hi_strip, _cc_lo_bound, _cc_hi_bound, _PPC_CHUNK_PERCENTILE)
            _col_ppc = np.zeros(_n_coarse_cols, dtype=np.float32)
            if len(_cc_vals_sp):
                _col_ppc[_cc_vals_sp - _cc_lo_bound] = _cc_p75_sp
            # assign nd tag per coarse column
            _cc_tags = [_best_nd_tag(_sr, float(_col_ppc[_cc]) if _col_ppc[_cc] > 0 else 1.0)
                        for _cc in range(_n_coarse_cols)]
            # build raw bands as runs of same nd tag -- all supercell tags
            # (nd<0) are grouped as one run regardless of exact depth: they
            # share the same cheap, batched _process_super_cell_chunk
            # mechanism, so noisy per-column density at very low ppc (a
            # coarse column with 0 vs 1 vs 2 points swings its local
            # estimate hugely) flipping between e.g. nd=-14/-15/-16 isn't a
            # real mechanism change worth banding on -- only a crossing
            # between supercell and the general (nd>=0) path is.
            def _band_key(_tag):
                if getattr(_cfg, '_DEBUG_REVERT_COL_ZONE_NOISE_FIX', False):
                    return _tag
                return ('sc', _tag[1]) if _tag[0] < 0 else _tag
            _raw_bands = []  # list of (cc_lo, cc_hi, nd_tag) — cc offsets local to _cc_lo_bound
            _cc_run_start = 0
            for _cc in range(1, _n_coarse_cols):
                if _band_key(_cc_tags[_cc]) != _band_key(_cc_tags[_cc_run_start]):
                    _raw_bands.append((_cc_run_start, _cc - 1, _cc_tags[_cc_run_start]))
                    _cc_run_start = _cc
            _raw_bands.append((_cc_run_start, _n_coarse_cols - 1, _cc_tags[_cc_run_start]))
            # convert to fine-cell column ranges, tracking each band's own
            # absolute coarse-column bounds (alongside its tag/pt-count) so
            # the merge decision below can re-query real per-block ppc/count
            # data on demand for any candidate (or merged) band.
            _col_chunks_raw = []
            for (_bcc_lo, _bcc_hi, _btag) in _raw_bands:
                _fc_lo = (_bcc_lo + _cc_lo_bound) * _K
                _fc_hi = (_bcc_hi + _cc_lo_bound) * _K + _K - 1
                # count source pts in this band
                _band_mask  = (_src_cols_strip >= _fc_lo) & (_src_cols_strip <= _fc_hi)
                _band_n_src = int(_band_mask.sum())
                _bcc_lo_abs = _bcc_lo + _cc_lo_bound
                _bcc_hi_abs = _bcc_hi + _cc_lo_bound
                _col_chunks_raw.append((_fc_lo, _fc_hi, _btag, _band_n_src, _bcc_lo_abs, _bcc_hi_abs))

            # Estimated total ms for `tag` processing the coarse-column range
            # [bcc_lo_abs, bcc_hi_abs] (n_pts source points) -- the SAME
            # point-count-weighted-sum / occupancy-aware-average blend used
            # to pick nd in the first place (nd_choice.blended_cost_ms),
            # rather than a single max-ppc estimate: keeps the merge
            # decision consistent with the cost model that actually chose
            # the tags being compared.
            _row_span_coarse_strip = _cr_hi_strip - _cr_lo_strip + 1
            def _band_cost_ms(_tag, _bcc_lo_abs, _bcc_hi_abs, _n_pts):
                _bp, _bc = _sparse_box_ppc_and_counts(_cr_lo_strip, _cr_hi_strip,
                                                       _bcc_lo_abs, _bcc_hi_abs)
                _col_span = _bcc_hi_abs - _bcc_lo_abs + 1
                _n_blocks_possible = max(1, _row_span_coarse_strip * _col_span)
                _total_area_fine = max(1, _n_blocks_possible * _K * _K)
                _total_area_ppc = (_n_pts / _total_area_fine) * math.pi * _sr * _sr
                _share = len(_bp) / _n_blocks_possible
                _alpha = 0.9 * max(0.0, min(1.0, _share))
                return _nd_blended_cost_ms(_sr_bench_key, _tag, _bp, _bc, _total_area_ppc,
                                            _alpha, total_pts=float(_n_pts))

            # Merge adjacent column bands: always merge if either side is
            # below the minimum pts threshold (undersized bands are never
            # worth processing standalone); otherwise merge only if the
            # estimated joint processing time beats the sum of the two
            # separate times plus the per-chunk overhead
            # (cfg.CHUNK_MERGE_OVERHEAD_MS). Either way, a merge is only
            # taken if it still fits the L3 budget.
            if len(_col_chunks_raw) > 1:
                _overhead_ms = getattr(_cfg, 'CHUNK_MERGE_OVERHEAD_MS', 2.0)
                _merged_bands = [list(_col_chunks_raw[0])]
                for _bnd in _col_chunks_raw[1:]:
                    _prev = _merged_bands[-1]
                    _prev_n, _bnd_n = _prev[3], _bnd[3]
                    _comb_n   = _prev_n + _bnd_n
                    _comb_tag = _prev[2] if _prev_n >= _bnd_n else _bnd[2]
                    _l3_ok = _comb_n * pt_bytes * _nd_mem_scale(_comb_tag) <= remaining
                    if _prev[2] == _bnd[2]:
                        # Same nd tag already -- no different-nd trade-off to
                        # weigh (nothing about processing would differ by
                        # keeping them split), so merge whenever it fits L3.
                        _do_merge = _l3_ok
                    elif _prev_n < _MIN_COLSPLIT_PTS or _bnd_n < _MIN_COLSPLIT_PTS:
                        _do_merge = _l3_ok
                    else:
                        _time_sep = (_band_cost_ms(_prev[2], _prev[4], _prev[5], _prev_n)
                                     + _band_cost_ms(_bnd[2], _bnd[4], _bnd[5], _bnd_n)
                                     + _overhead_ms)
                        _time_joint = _band_cost_ms(_comb_tag, _prev[4], _bnd[5], _comb_n)
                        _do_merge = _l3_ok and (_time_joint < _time_sep)
                    if _do_merge:
                        _merged_bands[-1] = [_prev[0], _bnd[1], _comb_tag, _comb_n, _prev[4], _bnd[5]]
                    else:
                        _merged_bands.append(list(_bnd))
                _col_chunks_raw = _merged_bands
            # final: merge consecutive same-nd bands (within-strip col-merge)
            _final_bands = [_col_chunks_raw[0]]
            for _bnd in _col_chunks_raw[1:]:
                if _bnd[2] == _final_bands[-1][2]:
                    _final_bands[-1][1] = _bnd[1]  # extend
                    _final_bands[-1][3] += _bnd[3]
                else:
                    _final_bands.append(_bnd)
            # only actually split if there are genuinely different nd tags
            if len(_final_bands) > 1 and len({_b[2] for _b in _final_bands}) > 1:
                _max_col_src = int(_src_cols_strip.max()) if _pt_e > _pt_s else _final_bands[-1][1]
                _col_chunks = [(max(0, _b[0]), min(_b[1], _max_col_src))
                               for _b in _final_bands
                               if _b[0] <= _max_col_src]
            else:
                _col_chunks = [(None, None)]
        else:
            _col_chunks = [(None, None)]

        # Debug/experiment override: force explicit column-chunk boundaries
        # instead of the greedy/nd-heterogeneity split above. Used by
        # chunk-overhead experiments that need EXACT control over which
        # adjacent chunks are merged vs. kept separate (rather than
        # indirectly coaxing the L3 budget into producing a desired
        # pattern). Off by default; single-strip datasets only (no attempt
        # to map the override across multiple row-strips).
        _dbg_force = getattr(_cfg, '_DEBUG_FORCE_COL_CHUNKS', None)
        if _dbg_force is not None:
            _col_chunks = _dbg_force

        # Debug/experiment override: force an explicit column-BAND merge
        # pattern, mirroring _DEBUG_FORCE_ROW_MERGE_PATTERN but at the
        # column level -- discovered (this session) to be the axis that
        # actually matters for elongated test datasets, where real row-
        # strip count is tiny (~5) and nearly all chunking happens via
        # column-splitting within those few strips. Takes whichever REAL
        # atomic column partition the natural logic above already produced
        # (memory-overflow split OR nd-heterogeneity split, whichever
        # fired) as the atomic base -- nd-heterogeneity splits can produce
        # small bands too in practice, so this doesn't force only the
        # memory path -- then merges adjacent atomic bands per
        # cfg._DEBUG_FORCE_COL_MERGE_PATTERN (list of bools, one per
        # boundary between atomic bands, True=merge). No-op if the natural
        # logic didn't split at all (_col_chunks == [(None,None)]). Off by
        # default.
        # Debug/experiment override: replace the natural atomic column bands
        # with a SYNTHETIC partition of varied sizes relative to the current
        # (pseudo-shrunk) L3 budget -- ~60% of bands sized 90-100% of the
        # budget, ~40% sized 2-80% -- rather than the natural planner's
        # bands, which tend to cluster tightly near the budget cap (greedy
        # fill-until-full). Gives the merge-pattern experiment more size
        # diversity to work with. Reproducible: a fresh RNG seeded from
        # n_pts+n_tgt (deterministic dataset-derived seed, not a fixed
        # constant) is used for every strip, so the same dataset always
        # produces the same synthetic band sizes. Off by default; only
        # takes effect when _DEBUG_FORCE_COL_MERGE_PATTERN is also set,
        # since the synthetic bands only matter as the merge-pattern's
        # atomic base.
        _dbg_synthetic = getattr(_cfg, '_DEBUG_SYNTHETIC_COL_BANDS', False)
        if _dbg_synthetic and _pt_e > _pt_s:
            _syn_cols_all = cols[_pt_s:_pt_e]
            _syn_ord = np.argsort(_syn_cols_all, kind='stable')
            _syn_scols = _syn_cols_all[_syn_ord]
            _syn_bdry = np.empty(len(_syn_scols), dtype=bool)
            _syn_bdry[0] = True
            _syn_bdry[1:] = _syn_scols[1:] != _syn_scols[:-1]
            _syn_cs = np.flatnonzero(_syn_bdry)
            _syn_ce = np.empty_like(_syn_cs); _syn_ce[:-1] = _syn_cs[1:]; _syn_ce[-1] = len(_syn_scols)
            _syn_uniq_cols = _syn_scols[_syn_cs]
            _syn_counts = (_syn_ce - _syn_cs).astype(np.intp)
            _syn_seed = int(n_pts + n_tgt)
            _syn_rng = np.random.default_rng(_syn_seed)
            _syn_cap = remaining
            _syn_bands = []
            _syn_b_lo = int(_syn_uniq_cols[0]); _syn_accum = 0.0
            _syn_prev_col = _syn_b_lo
            def _syn_next_frac():
                return (_syn_rng.uniform(0.9, 1.0) if _syn_rng.random() < 0.6
                        else _syn_rng.uniform(0.02, 0.8))
            _syn_target_frac = _syn_next_frac()
            for _syn_col, _syn_cnt in zip(_syn_uniq_cols.tolist(), _syn_counts.tolist()):
                _syn_cost = _syn_cnt * (pt_bytes * _nd_mem_scale(_pci_sp_nd_tag) + SRC_BYTES_PER_PT)
                if _syn_accum > 0 and _syn_accum + _syn_cost > _syn_cap * _syn_target_frac:
                    _syn_bands.append((_syn_b_lo, _syn_prev_col))
                    _syn_b_lo = _syn_col
                    _syn_accum = 0.0
                    _syn_target_frac = _syn_next_frac()
                _syn_accum += _syn_cost
                _syn_prev_col = _syn_col
            _syn_bands.append((_syn_b_lo, _syn_prev_col))
            _col_chunks = _syn_bands

        _dbg_col_pattern = getattr(_cfg, '_DEBUG_FORCE_COL_MERGE_PATTERN', None)
        if _dbg_col_pattern is not None and _col_chunks != [(None, None)]:
            _atomic_col_bands = _col_chunks
            _col_chunks = []
            _ci = 0
            while _ci < len(_atomic_col_bands):
                _b_lo, _b_hi = _atomic_col_bands[_ci]
                _cj = _ci + 1
                while (_cj < len(_atomic_col_bands) and _cj - 1 < len(_dbg_col_pattern)
                       and _dbg_col_pattern[_cj - 1]):
                    _n_lo, _n_hi = _atomic_col_bands[_cj]
                    _b_hi = _n_hi
                    _cj += 1
                _col_chunks.append((_b_lo, _b_hi))
                _ci = _cj
            if getattr(_cfg, '_DEBUG_CAPTURE_MERGED_STRIP_COUNT', False):
                _cfg._DEBUG_MERGED_STRIP_COUNT = (
                    getattr(_cfg, '_DEBUG_MERGED_STRIP_COUNT', 0) + len(_atomic_col_bands))

        # Per-band nd hint, aligned with _col_chunks by index, used by the main
        # loop to size each column-chunk's own candidate-gather margin instead
        # of a single strip-wide value -- a strip that's mostly nd=0 with one
        # sparse sub-band needing nd=-1 (exactly what column-heterogeneity
        # splitting exists to handle) would otherwise never get that sub-band's
        # margin widened, since the strip-level tags reflect the dominant
        # density, not a minority sub-region's. Recomputed generically here
        # (cheap density lookup per final band) rather than threaded through
        # each construction path above, since only the nd-heterogeneity branch
        # has precise per-band tags on hand; this way every path (greedy
        # memory-split, no-split, debug overrides) gets a consistent estimate.
        _col_chunk_nd_hints = []
        for (_hb_lo, _hb_hi) in _col_chunks:
            if _hb_lo is None:
                _col_chunk_nd_hints.append(min(_pci_sp_nd_tag[0], _pci_sp_hot_tag[0]))
                continue
            _hb_reg_nz = _sparse_box_ppc(_cr_lo_strip, _cr_hi_strip, _hb_lo // _K, _hb_hi // _K)
            _hb_ppc = float(np.percentile(_hb_reg_nz, _PPC_CHUNK_PERCENTILE)) if len(_hb_reg_nz) > 0 else 1.0
            _col_chunk_nd_hints.append(_best_nd_tag(_sr, _hb_ppc)[0])

        return _col_chunks, _col_chunk_nd_hints, _pt_s, _pt_e, _strip_cost, _cr_lo_strip, _cr_hi_strip

    _strip_plans = []  # (row_chunk_idx, chunk_rows, sp_nd_tag, sp_ppc, sp_hot_tag,
                        #  col_chunks, col_chunk_nd_hints, pt_s, pt_e, strip_cost, cr_lo_strip, cr_hi_strip)
    _rci = 0
    for _pp_i, (_sp_idx, _sp_h, _sp_nd_tag, _sp_ppc, _sp_hot_tag) in enumerate(_merged_strips):
        _reverse_plan = (_pp_i % 2 == 1)  # matches the main loop's row_iter parity below
        _col_chunks, _col_chunk_nd_hints, _pt_s, _pt_e, _strip_cost, _cr_lo_strip, _cr_hi_strip = _plan_col_chunks(
            _rci, _sp_h, _sp_nd_tag, _sp_hot_tag, _reverse_plan)
        _strip_plans.append((_rci, _sp_h, _sp_nd_tag, _sp_ppc, _sp_hot_tag,
                              _col_chunks, _col_chunk_nd_hints, _pt_s, _pt_e, _strip_cost, _cr_lo_strip, _cr_hi_strip))
        _rci += _sp_h
    # Full plan known upfront now -- exact chunk count, no more running estimate.
    n_chunks_total = sum(len(_sp[5]) for _sp in _strip_plans)
    # Chunk-progress print cadence: update after every chunk if there are few,
    # otherwise cap it at ~20 updates total regardless of chunk count.
    _chunk_progress_step = max(1, n_chunks_total // 20)

    # ---- rasterize each chunk's (dilated) rectangle + nd onto a coarse K×K -----
    # block grid, then run Layer 1 aggregation with it (if run_aggregation), all
    # inline before executing this same call. Lets aggregation be driven by the
    # REAL, final (post-merge) chunk decision instead of an independently-
    # estimated one -- see this session's finding that re-estimating nd on a
    # single coarse block in isolation disagrees with the real (possibly much
    # larger, merged) chunk box's decision, since total_area_ppc/share_occupied
    # depend on the box actually evaluated. Previously this ran as a separate
    # plan_only=True call from disk_search.py, with aggregation sandwiched
    # between two search_and_aggregate calls -- folded into one call here so
    # nothing before this point (e.g. _parse_area_weight above) runs twice.
    # Margin/nd computation here intentionally mirrors the main loop's strip- and
    # chunk-level logic below (same _exact_chunk_nd / _margin_for_nd calls) --
    # duplicated rather than shared inline because threading this through the
    # ~700-line main loop (many branches: _TEST_RANDOM_ND, _do_weight, area_weight
    # variants, single vs multi-region) is higher-risk than a small, self-
    # contained pre-pass. Both copies call the same canonical functions, so they
    # cannot disagree with each other the way independently-implemented
    # estimates did.
    depth_by_block = {}
    needs_level0 = {}
    for (_pci_rci, _pci_h, _pci_sp_nd_tag, _pci_sp_ppc, _pci_sp_hot_tag,
         _pci_col_chunks, _pci_col_chunk_nd_hints, _pci_pt_s, _pci_pt_e,
         _pci_strip_cost, _pci_cr_lo_strip, _pci_cr_hi_strip) in _strip_plans:
        if _pci_pt_e <= _pci_pt_s:
            continue
        _pci_chunk_src_rows = src_unique_rows[_pci_rci : _pci_rci + _pci_h]
        _pci_src_row_min = int(_pci_chunk_src_rows[0])
        _pci_src_row_max = int(_pci_chunk_src_rows[-1])
        _pci_strip_margin = r_rows
        _pci_nd_candidates = [
            _exact_chunk_nd(_pci_cr_lo_strip, _pci_cr_hi_strip, _sccl, _scch)
            for (_sccl, _scch) in _pci_col_chunks
        ]
        for _nd_cand in _pci_nd_candidates:
            if _nd_cand < 0:
                _pci_strip_margin = max(_pci_strip_margin, _margin_for_nd(_nd_cand))
        _pci_need_row_lo = max(tgt_row_lo_global, _pci_src_row_min - _pci_strip_margin)
        _pci_need_row_hi = min(tgt_row_hi_global, _pci_src_row_max + _pci_strip_margin)
        for (_ccl, _cch), _pci_nd in zip(_pci_col_chunks, _pci_nd_candidates):
            if _ccl is not None:
                _pci_margin = r_rows if _pci_nd >= 0 else _margin_for_nd(_pci_nd)
                _pci_col_lo, _pci_col_hi = _ccl - _pci_margin, _cch + _pci_margin
            else:
                _pci_col_lo, _pci_col_hi = 0, _max_cc_coarse * _K - 1
            _blo, _bhi = _pci_need_row_lo // _K, _pci_need_row_hi // _K
            _clo, _chi = _pci_col_lo // _K, _pci_col_hi // _K
            for _br in range(int(_blo), int(_bhi) + 1):
                for _bc in range(int(_clo), int(_chi) + 1):
                    _key = (_br, _bc)
                    depth_by_block[_key] = max(depth_by_block.get(_key, 0), max(0, _pci_nd))
                    needs_level0[_key] = needs_level0.get(_key, False) or (_pci_nd >= 0)

    import os as _diag_os3
    _DIAG_SINGLE_CALL = _diag_os3.environ.get('AABPL_DIAG_SINGLE_CALL')
    if _DIAG_SINGLE_CALL:
        print(f'DIAGTMP pre-agg tgt_rows_full[:5]={tgt_rows_full[:5].tolist()} '
              f'tgt_cols_full[:5]={tgt_cols_full[:5].tolist()} '
              f'sum_tgt_rows={int(tgt_rows_full.sum())} sum_tgt_cols={int(tgt_cols_full.sum())} '
              f'pts_vals_sum={float(pts_vals_xy_full[:, :n_c].sum())}', flush=True)

    if plan_only:
        return depth_by_block, needs_level0

    if run_aggregation and getattr(_cfg, 'USE_ADAPTIVE_ND_AGGREGATION', False):
        from aabpl.search.point_assignment import aggregate_point_data_to_cells_adaptive_nd
        aggregate_point_data_to_cells_adaptive_nd(
            grid=grid, pts=pts_target, y=y, x=x, c=c,
            row_name=row_name, col_name=col_name,
            nest_depth=nest_depth, sr=_sr, silent=silent,
            depth_by_block=depth_by_block, needs_level0=needs_level0,
        )
        if _DIAG_SINGLE_CALL:
            print(f'DIAGTMP post-agg tgt_rows_full[:5]={tgt_rows_full[:5].tolist()} '
                  f'tgt_cols_full[:5]={tgt_cols_full[:5].tolist()} '
                  f'sum_tgt_rows={int(tgt_rows_full.sum())} sum_tgt_cols={int(tgt_cols_full.sum())} '
                  f'pts_vals_sum={float(pts_vals_xy_full[:, :n_c].sum())} '
                  f'sums_array_shape={grid.sums_array.shape} '
                  f'sums_array_sum={float(grid.sums_array.sum())} '
                  f'id_to_sums_by_lvl_len={len(grid._search_internals.id_to_sums_by_lvl or {})}',
                  flush=True)

    # ---- precomputed sums array (from aggregate_point_data_to_cells) -----------
    # Contain path: instead of summing block_cell_sums in a Python loop, index
    # directly into the contiguous _sums_array using codec keys → one numpy call.
    _sums_array  = grid.sums_array                           # (n_nodes, n_c) float64

    # Build sorted key/index arrays once for vectorised lookup in _contain_sum,
    # then cache them on the grid (not just the source dict) -- callers that
    # invoke search_and_aggregate more than once on the same grid (e.g.
    # detect_cluster_pts's null-distribution passes) need these arrays on
    # every call, but id_to_sums_by_lvl itself is freed after first use (see
    # below), so a naive "build from the dict every call" breaks on call two.
    _ck_keys = getattr(grid._search_internals, '_ck_keys_cache', None)
    _ck_idxs = getattr(grid._search_internals, '_ck_idxs_cache', None)
    if _ck_keys is None:
        _id_to_lvl = grid._search_internals.id_to_sums_by_lvl  # codec_int -> row index
        _ck_keys = np.array(sorted(_id_to_lvl.keys()), dtype=np.int64)
        _ck_idxs = np.array([_id_to_lvl[k] for k in _ck_keys], dtype=np.int64)
        grid._search_internals._ck_keys_cache = _ck_keys
        grid._search_internals._ck_idxs_cache = _ck_idxs
        # id_to_sums_by_lvl (the dict _id_to_lvl points at) is never read again
        # past this point -- _ck_keys/_ck_idxs (now cached above) are what
        # _contain_sum actually uses from here on, on this and future calls.
        # A dict entry costs ~6-9x a sorted-int64-array slot (boxed key +
        # hash table overhead vs 16 bytes), and at deep nest_depth this dict
        # can hold well over a million entries -- drop it rather than let it
        # sit alive for the rest of the Grid's lifetime doing nothing.
        grid._search_internals.id_to_sums_by_lvl = None
        del _id_to_lvl

    def _contain_sum(codec_offsets, home_codec_key):
        """Sum precomputed cell sums for a set of contain cells."""
        if n_c == 0 or len(codec_offsets) == 0:
            return zero_sum
        keys = codec_offsets + home_codec_key
        pos  = np.searchsorted(_ck_keys, keys)
        safe_pos = np.minimum(pos, len(_ck_keys) - 1)
        mask = (pos < len(_ck_keys)) & (_ck_keys[safe_pos] == keys)
        if not mask.any():
            return zero_sum
        return _sums_array[_ck_idxs[pos[mask]]].sum(axis=0)

    # ---- supercell region pre-scan: which nd<0 tags are used where ---------------
    # Re-derives each planned chunk's nd via the same _best_nd_tag(_sr, ppc_est)
    # call the main loop makes (cheap -- already proven so, it's what the main
    # loop itself does per chunk), purely to let _ensure_super_cell_target_index
    # restrict its index to the chunk regions that actually use each nd (dilated),
    # instead of indexing the whole target grid. Skipped for _TEST_RANDOM_ND
    # (that mode picks nd randomly at chunk-execution time, so a pre-scan via
    # _best_nd_tag wouldn't match -- _nd_chunk_regions stays empty, and
    # _ensure_super_cell_target_index falls back to its old, safe, full-grid
    # behaviour for every nd in that mode).
    _nd_chunk_regions = {}  # nd (nd<0 only) -> list of (row_lo, row_hi, col_lo, col_hi), fine-cell units
    if not _TEST_RANDOM_ND:
        for (_rci2, _sp_h2, _sp_nd_tag2, _sp_ppc2, _sp_hot_tag2,
             _col_chunks2, _col_chunk_nd_hints2, _pt_s2, _pt_e2, _strip_cost2, _cr_lo2, _cr_hi2) in _strip_plans:
            if _pt_e2 <= _pt_s2:
                continue
            _strip_cols2 = cols[_pt_s2:_pt_e2]
            _strip_rows2 = rows[_pt_s2:_pt_e2]
            for (_cc_lo2, _cc_hi2) in _col_chunks2:
                if _cc_lo2 is not None:
                    _bmask2 = (_strip_cols2 >= _cc_lo2) & (_strip_cols2 <= _cc_hi2)
                    if not _bmask2.any():
                        continue
                    _b_rows2 = _strip_rows2[_bmask2]; _b_cols2 = _strip_cols2[_bmask2]
                    _reg_nz2 = _sparse_box_ppc(_cr_lo2, _cr_hi2, _cc_lo2 // _K, _cc_hi2 // _K)
                else:
                    _b_rows2 = _strip_rows2; _b_cols2 = _strip_cols2
                    _reg_nz2 = _sparse_box_ppc(_cr_lo2, _cr_hi2)
                _nd2 = _exact_chunk_nd(_cr_lo2, _cr_hi2, _cc_lo2, _cc_hi2)
                if _nd2 < 0:
                    _rlo2, _rhi2 = int(_b_rows2.min()), int(_b_rows2.max())
                    _clo2, _chi2 = int(_b_cols2.min()), int(_b_cols2.max())
                    _nd_chunk_regions.setdefault(_nd2, []).append((_rlo2, _rhi2, _clo2, _chi2))

    # ---- main chunk loop --------------------------------------------------------
    # Consumes the precomputed _strip_plans instead of recomputing col-split
    # decisions inline -- everything below is otherwise unchanged from before
    # the planning pre-pass was hoisted out.
    row_iter = 0  # tracks even/odd for snake column order (kept for readability;
                  # _strip_plans was already built with matching parity above)
    for (row_chunk_idx, chunk_rows, _sp_nd_tag, _sp_ppc, _sp_hot_tag,
         _col_chunks, _col_chunk_nd_hints, _pt_s, _pt_e, _strip_cost, _cr_lo_strip, _cr_hi_strip) in _strip_plans:
        _reverse = (row_iter % 2 == 1)

        chunk_src_rows = src_unique_rows[row_chunk_idx : row_chunk_idx + chunk_rows]
        src_row_min    = int(chunk_src_rows[0])
        src_row_max    = int(chunk_src_rows[-1])

        # Row-band margin: widened beyond r_rows to cover the deepest ACTUAL nd
        # among this strip's own column-chunks -- re-decided here with the same
        # exact density estimate the non-weighted dispatch branch uses (not the
        # planning-time hint, which can disagree with it near a crossover
        # threshold -- see the col-chunk loop below for the same fix applied
        # there). The row-band window covers the whole strip regardless of
        # which column-chunk needs the extra reach, so it must use the worst
        # case among all of them.
        _strip_margin = r_rows
        _strip_nd_candidates = [_exact_chunk_nd(_cr_lo_strip, _cr_hi_strip, _scc_lo, _scc_hi)
                                 for (_scc_lo, _scc_hi) in _col_chunks]
        for _nd_cand in _strip_nd_candidates:
            if _nd_cand < 0:
                _strip_margin = max(_strip_margin, _margin_for_nd(_nd_cand))

        # target row band needed for this chunk
        need_row_lo = max(tgt_row_lo_global, src_row_min - _strip_margin)
        need_row_hi = min(tgt_row_hi_global, src_row_max + _strip_margin)

        # advance blk_lo / blk_hi to cover [need_row_lo, need_row_hi]
        if loaded_row_lo is None or need_row_lo != loaded_row_lo:
            blk_lo = int(_np_searchsorted(tgt_rows_full, need_row_lo,     side='left'))
        if loaded_row_hi is None or need_row_hi != loaded_row_hi:
            blk_hi = int(_np_searchsorted(tgt_rows_full, need_row_hi + 1, side='left'))
        loaded_row_lo = need_row_lo
        loaded_row_hi = need_row_hi

        # ---- source point range and group slices for this row chunk -------------
        batch_pt_start = int(src_row_bdry[row_chunk_idx])
        next_chunk     = row_chunk_idx + chunk_rows
        batch_pt_end   = int(src_row_bdry[next_chunk]) if next_chunk < n_src_unique else n_pts

        cg_lo    = int(_np_searchsorted(contain_starts_all, batch_pt_start, side='left'))
        cg_hi    = int(_np_searchsorted(contain_starts_all, batch_pt_end,   side='left'))
        c_starts = contain_starts_all[cg_lo:cg_hi]
        c_ends   = contain_ends_all[cg_lo:cg_hi]

        og_lo    = int(_np_searchsorted(overlap_starts_all, batch_pt_start, side='left'))
        og_hi    = int(_np_searchsorted(overlap_starts_all, batch_pt_end,   side='left'))
        o_starts = overlap_starts_all[og_lo:og_hi]
        o_ends   = overlap_ends_all[og_lo:og_hi]

        # coarse row range for this strip (used inside col-chunk loop)
        _cr_lo = _cr_lo_strip
        _cr_hi = _cr_hi_strip

        # ---- col chunk inner loop -----------------------------------------------
        for (_cc_lo, _cc_hi), _cc_nd_hint in zip(_col_chunks, _col_chunk_nd_hints):
            if _cc_lo is not None:
                # Re-decide this chunk's nd NOW, before slicing, using the exact
                # same density estimate + best_nd_tag call the non-weighted
                # branch below makes -- the planning-time hint (_cc_nd_hint)
                # was computed independently and can disagree at density values
                # near a crossover threshold (confirmed via brute-force
                # validation: e.g. ppc~413 classified nd=0 by one estimate and
                # nd=-1 by the other for the same chunk), silently leaving the
                # margin unwidened for chunks that actually need it. This
                # duplicates a cheap lookup (not the full weighted-decision
                # path) purely to size the gather window correctly; the
                # downstream ppc-estimate block still runs its own decision
                # unchanged for the actual dispatch.
                _cc_nd_actual = _exact_chunk_nd(_cr_lo, _cr_hi, _cc_lo, _cc_hi)
                # target: include this chunk's own margin either side, widened
                # beyond r_rows only if its actual (re-decided) nd is a
                # super-cell -- other, denser column-chunks in the same strip
                # keep the cheap native margin instead of all paying for this
                # one chunk's wider reach.
                _cc_margin = r_rows if _cc_nd_actual >= 0 else _margin_for_nd(_cc_nd_actual)
                tgt_col_lo       = _cc_lo - _cc_margin
                tgt_col_hi       = _cc_hi + _cc_margin
                tgt_cols_in_band = tgt_cols_full[blk_lo:blk_hi]
                col_filter_idx   = _np_flatnonzero((tgt_cols_in_band >= tgt_col_lo) &
                                                   (tgt_cols_in_band <= tgt_col_hi))
                blk_rows = tgt_rows_full[blk_lo:blk_hi][col_filter_idx]
                blk_cols = tgt_cols_in_band[col_filter_idx]
                blk_pts  = pts_vals_xy_full[blk_lo:blk_hi][col_filter_idx]
                blk_scnr = tgt_scnr_full[blk_lo:blk_hi][col_filter_idx]
                blk_n    = len(col_filter_idx)
                # restrict contain/overlap groups to this source col range.
                # Groups are sorted by (row, col) not col alone, so boolean masking
                # is required — searchsorted on cols would fail for chunk_rows > 1
                # because the same col can recur in a later row.
                if len(c_starts):
                    _c_mask       = (cols[c_starts] >= _cc_lo) & (cols[c_starts] <= _cc_hi)
                    _c_starts_col = c_starts[_c_mask]
                    _c_ends_col   = c_ends[_c_mask]
                else:
                    _c_starts_col = c_starts; _c_ends_col = c_ends
                if len(o_starts):
                    _o_mask       = (cols[o_starts] >= _cc_lo) & (cols[o_starts] <= _cc_hi)
                    _o_starts_col = o_starts[_o_mask]
                    _o_ends_col   = o_ends[_o_mask]
                else:
                    _o_starts_col = o_starts; _o_ends_col = o_ends
            else:
                blk_rows = tgt_rows_full[blk_lo:blk_hi]
                blk_cols = tgt_cols_full[blk_lo:blk_hi]
                blk_pts  = pts_vals_xy_full[blk_lo:blk_hi]
                blk_scnr = tgt_scnr_full[blk_lo:blk_hi]
                blk_n    = blk_hi - blk_lo
                _c_starts_col = c_starts; _c_ends_col = c_ends
                _o_starts_col = o_starts; _o_ends_col = o_ends

            _blk_row_lo = need_row_lo  # kept for blk_pts row-range reference

            chunk_num += 1
            if not silent and (chunk_num % _chunk_progress_step == 0 or chunk_num == n_chunks_total):
                print(f'  chunk {chunk_num}/{n_chunks_total}', flush=True)

            # Debug/experiment per-chunk timing: snapshot before this chunk's
            # work starts, compared against the same snapshot taken at the
            # end of the loop body below. Off by default (cfg._DEBUG_PER_CHUNK_CAPTURE).
            _dbg_per_chunk = getattr(_cfg, '_DEBUG_PER_CHUNK_CAPTURE', False)
            if _dbg_per_chunk:
                _dbg_t0 = _pt()
                _dbg_wall0 = _wall_pt()
                _dbg_contain0 = _t_contain
                _dbg_overlap0 = _t_overlap

            # ---- ppc estimate from sparse coarse density map --------------------
            # Query the coarse cell region covering this chunk (strip rows × col range).
            # Handles tall strips and col splits correctly — both dimensions matter.
            if _cfg.USE_WEIGHTED_ND_DECISION:
                if _cc_lo is not None:
                    _reg_nz, _reg_cnt = _sparse_box_ppc_and_counts(_cr_lo, _cr_hi, _cc_lo // _K, _cc_hi // _K)
                    _col_span_coarse = (_cc_hi // _K) - (_cc_lo // _K) + 1
                else:
                    _reg_nz, _reg_cnt = _sparse_box_ppc_and_counts(_cr_lo, _cr_hi)
                    _col_span_coarse = _max_cc_coarse
                _row_span_coarse = _cr_hi - _cr_lo + 1
                _total_area_fine = max(1, _row_span_coarse * _col_span_coarse * _K * _K)
                _total_pts_reg = float(_reg_cnt.sum()) if len(_reg_cnt) else 0.0
                _total_area_ppc = (_total_pts_reg / _total_area_fine) * math.pi * _sr * _sr
                _ppc_est = float(np.percentile(_reg_nz, _PPC_CHUNK_PERCENTILE)) if len(_reg_nz) > 0 else 1.0
                # Share of coarse blocks in the chunk's bounding box that are
                # actually occupied -- how much total_area_ppc's full-footprint
                # averaging dilutes true local density. High share: little
                # empty space, the average is close to unbiased. Low share:
                # points cluster in a few blocks out of many, and the average
                # badly understates density where points actually are.
                _n_blocks_possible = max(1, _row_span_coarse * _col_span_coarse)
                _share_occupied = len(_reg_nz) / _n_blocks_possible
            else:
                if _cc_lo is not None:
                    _reg_nz = _sparse_box_ppc(_cr_lo, _cr_hi, _cc_lo // _K, _cc_hi // _K)
                else:
                    _reg_nz = _sparse_box_ppc(_cr_lo, _cr_hi)
                _ppc_est   = float(np.percentile(_reg_nz, _PPC_CHUNK_PERCENTILE)) if len(_reg_nz) > 0 else 1.0

            # ---- per-chunk adaptive nd selection --------------------------------
            # _chunk_single_region is resolved from the CHOSEN TAG, not the static
            # cfg.SINGLE_REGION flag: adaptive selection can dispatch a
            # single_region=True tag (sT, s1T, smNT — smNT always is one, by
            # construction) on any chunk regardless of that global flag, and the
            # contain/overlap templates must match whichever tag was actually chosen.
            _is_super_cell_nd = False
            if _TEST_RANDOM_ND:
                import random as _rng
                _chunk_nd = _rng.choice(_nd_range_test)
                _chunk_tag = (_chunk_nd, False)
                _chunk_single_region = _chunk_tag[1]
                _nd_chunk_counts[_chunk_tag] = _nd_chunk_counts.get(_chunk_tag, 0) + 1
                _is_super_cell_nd = (_chunk_nd < 0)
                if _is_super_cell_nd:
                    if _do_weight:
                        _ensure_super_cell_nd(_chunk_nd)
                else:
                    shared_cntd_offset     = _nd_shared_cntd[_chunk_nd]
                    cntd_offset_by_region  = _nd_cntd_by_reg[_chunk_nd]
                    ovlpd_offset_by_region = _nd_ovlpd_by_reg[_chunk_nd]
                    if _chunk_single_region:
                        single_region_ovlpd_offset = _nd_sr_ovlpd[_chunk_nd]
            else:
                if _cfg.USE_WEIGHTED_ND_DECISION:
                    _chunk_tag = _best_nd_tag_weighted(_sr, _reg_nz, _reg_cnt, _total_area_ppc,
                                                        share_occupied=_share_occupied)
                else:
                    _chunk_tag = _best_nd_tag(_sr, _ppc_est)
                _chunk_nd  = _chunk_tag[0]
                # nd_choice's cost model is built independent of the actual
                # runtime-selected native nest_depth for this grid/radius — clamp
                # so a recommendation deeper than what geometry was built for
                # (_nd_max) never causes an out-of-bounds dict lookup below.
                if _chunk_nd > _nd_max:
                    _chunk_nd = _nd_max
                    _chunk_tag = (_chunk_nd, _chunk_tag[1])
                _chunk_single_region = _chunk_tag[1]
                _nd_chunk_counts[_chunk_tag] = _nd_chunk_counts.get(_chunk_tag, 0) + 1
                _is_super_cell_nd = (_chunk_nd < 0)
                if _is_super_cell_nd:
                    if _do_weight:
                        # area-weighting still uses the old per-position (parity)
                        # templates — not yet ported to the coarse-batch fast path.
                        # Only the sparse/supercell chunks pay this; nd>=0 chunks in
                        # the same run are unaffected either way.
                        #
                        # Port plan, if this combination turns out to matter in
                        # practice: split by area_weight mode.
                        #   'logit'/'flat'/'binary' (cheap, not 'exact'): classify
                        #     each COARSE candidate cell once as interior/exterior/
                        #     boundary via Shapely against study_area (few dozen
                        #     coarse cells per nd, cheap — same idea as the existing
                        #     _do_exact_supercell block classification, just at
                        #     coarse-cell size instead of exact_block_k). Then reuse
                        #     the existing centroid-distance logit/flat estimate per
                        #     point, keyed off that classification, inside
                        #     _process_super_cell_chunk. No _expand_coarse needed.
                        #   'exact': needs true per-point disk∩study_area area, which
                        #     is the expensive/precise path already — porting it
                        #     doesn't remove that cost, only the (now-fixed)
                        #     geometry-build overhead around it. Lower priority;
                        #     scope separately if ever needed.
                        # Not started because it's unconfirmed whether area_weight
                        # is actually used together with sparse (supercell-triggering)
                        # data in practice — check before investing in this.
                        _ensure_super_cell_nd(_chunk_nd)
                else:
                    shared_cntd_offset     = _nd_shared_cntd[_chunk_nd]
                    cntd_offset_by_region  = _nd_cntd_by_reg[_chunk_nd]
                    ovlpd_offset_by_region = _nd_ovlpd_by_reg[_chunk_nd]
                    if _chunk_single_region:
                        single_region_ovlpd_offset = _nd_sr_ovlpd[_chunk_nd]

            if _cc_lo is not None:
                _col_slice = cols[batch_pt_start:batch_pt_end]
                _chunk_mask = (_col_slice >= _cc_lo) & (_col_slice <= _cc_hi)
                _n_src_chunk = int(np.sum(_chunk_mask))
            else:
                _chunk_mask = None
                _n_src_chunk = batch_pt_end - batch_pt_start

            # ---- super-cell fast path: coarse-cell batching, skip fine machinery -
            if _is_super_cell_nd and not _do_weight:
                if _cc_lo is not None:
                    _sp_mask = ((cols[batch_pt_start:batch_pt_end] >= _cc_lo) &
                                (cols[batch_pt_start:batch_pt_end] <= _cc_hi))
                    _sp_idx = np.flatnonzero(_sp_mask) + batch_pt_start
                else:
                    _sp_idx = np.arange(batch_pt_start, batch_pt_end)
                _process_super_cell_chunk(_chunk_nd, _sp_idx)
                if len(_sp_idx) and int(_sp_idx.max()) >= next_threshold:
                    next_threshold = progress.update(int(_sp_idx.max()))
                if _dbg_per_chunk:
                    if not hasattr(_cfg, '_DEBUG_PER_CHUNK_STATS'):
                        _cfg._DEBUG_PER_CHUNK_STATS = []
                    _cfg._DEBUG_PER_CHUNK_STATS.append({
                        'chunk_num': chunk_num,
                        'cc_lo': _cc_lo, 'cc_hi': _cc_hi,
                        'n_src': len(_sp_idx),
                        'nd': _chunk_nd,
                        'ppc_est': _ppc_est,
                        'is_super_cell': True,
                    })
                continue

            # ---- pre-aggregate pts -> cell slices within block ------------------
            # block_cell_slices: absolute-codec-key -> (start, end) into blk_pts.
            # Used by the overlap path (gather raw pts into candidate_buffer).
            # Contains level-0 entries (one per occupied cell) plus level-1..nd
            # sub-cell entries built by _nest_block_slices quadtree descent.
            _tbcs0 = _pt()
            block_cell_slices = {}
            if _do_weight and not _do_exact_supercell:
                block_cell_valid_area = {}
            if blk_n > 0:
                cell_bdry = np.concatenate([[True],
                                            (blk_rows[1:] != blk_rows[:-1]) |
                                            (blk_cols[1:] != blk_cols[:-1])])
                cell_starts = _np_flatnonzero(cell_bdry)
                cell_ends   = _np_append(cell_starts[1:], blk_n)
                # Vectorised key computation: codec.key() already accepts array
                # inputs (np.asarray/np.round throughout), but was previously
                # called once per occupied cell -- a full scaled-int computation
                # plus numpy-call overhead per call for what's really one
                # elementwise formula. One batched call instead of N.
                _cell_keys = codec.key(0, blk_rows[cell_starts], blk_cols[cell_starts]).astype(np.int64)
                for _ck, s, e in zip(_cell_keys.tolist(), cell_starts.tolist(), cell_ends.tolist()):
                    block_cell_slices[_ck] = (s, e)
                    if _do_weight and not _do_exact_supercell:
                        if _ck in invalid_cell_keys:
                            va = 0.0
                        elif _ck in boundary_cell_keys:
                            # .get(..., 1.0) not [_ck]: boundary_cell_fracs can be
                            # empty (compute_fractions=False) while boundary_cell_keys
                            # is still populated from a different source (partial_cells_rc)
                            # -- confirmed pre-existing KeyError in that gap, reproduces
                            # identically in pip 0.4.1. Defaulting to 1.0 (fully valid,
                            # no extra invalid-area penalty) is the conservative choice:
                            # it never crashes, and is a no-op whenever the dict IS
                            # populated (the common case, unaffected by this change).
                            va = (boundary_cell_fracs.get(_ck, 1.0) if _do_per_cell else _bnd_frac) * _cell_area
                        else:
                            va = _cell_area
                        block_cell_valid_area[_ck] = va
                _nest_block_slices(blk_scnr, blk_rows, blk_cols, blk_n, block_cell_slices,
                                   depth=max(0, _chunk_nd))
            _t_block_cell_slices_build += _pt() - _tbcs0

            if _cfg.PROFILE_FUNC_TIMES:
                block_cell_counts = {k: e - s for k, (s, e) in block_cell_slices.items()}
            nonempty_block = set(block_cell_slices)

            # ---- overlap gather cache keyed on occupied-cell fingerprint --------
            # fs = frozenset of absolute codec keys that are both in the disk template
            # and in block_cell_slices (i.e. actually occupied by target points).
            # Cache stores (slice_list, n) so hits re-fill candidate_buffer without
            # re-scanning block_cell_slices.
            _FP_CAP = 512
            overlap_fp_cache = {}

            def gather_overlap_pts(codec_offsets, home_codec_key):
                nonlocal _n_o_cache_hit, _n_o_cache_miss, _n_ov_keys, _n_ov_template
                nonlocal _t_ov_filter, _t_ov_cachelkup, _t_ov_copy
                nonlocal candidate_buffer, _n_buffer_resizes
                _tf0 = _pt()
                raw_keys = codec_offsets + home_codec_key
                _n_ov_template += len(codec_offsets)
                # Check cache with a cheap frozenset key first; only sort on miss.
                occupied_overlap_keys = frozenset(int(k) for k in raw_keys if k in nonempty_block)
                _n_ov_keys += len(occupied_overlap_keys)
                _t_ov_filter += _pt() - _tf0
                _tc0 = _pt()
                cached = overlap_fp_cache.get(occupied_overlap_keys)
                _t_ov_cachelkup += _pt() - _tc0
                _tcp0 = _pt()
                if cached is not None:
                    _n_o_cache_hit += 1
                    s_arr, lens, total_pts = cached
                else:
                    _n_o_cache_miss += 1
                    if occupied_overlap_keys:
                        # dict lookup itself still needs a per-key Python loop (keys
                        # come from a dict, not a sorted array), but the actual point
                        # COPY below is now one batched fancy-index gather instead of
                        # one small numpy slice-assign per key -- that copy loop was
                        # measured as ~60-67% of total gather time (see bench_overlap_micro.py).
                        s_list = []
                        e_list = []
                        for k in occupied_overlap_keys:
                            s, e = block_cell_slices[k]
                            s_list.append(s); e_list.append(e)
                        s_arr = _np_array(s_list, dtype=np.int64)
                        lens  = _np_array(e_list, dtype=np.int64) - s_arr
                        total_pts = int(lens.sum())
                    else:
                        s_arr = _np_array([], dtype=np.int64)
                        lens  = _np_array([], dtype=np.int64)
                        total_pts = 0
                    if len(overlap_fp_cache) < _FP_CAP:
                        # cache compact (start, length) arrays, not the materialised
                        # candidate points -- rebuilding the combined index below is
                        # cheap (vectorised) relative to the copy it replaces, so this
                        # keeps the cache small while still letting hits skip the
                        # per-key dict-lookup loop entirely.
                        overlap_fp_cache[occupied_overlap_keys] = (s_arr, lens, total_pts)
                if total_pts:
                    # Vectorised "concatenate multiple ranges" gather: build one
                    # combined index array covering all (s, s+len) ranges at once,
                    # then a single fancy-index copy -- replaces what used to be
                    # one candidate_buffer[...] = blk_pts[s:e] numpy call per key.
                    seg_offsets  = np.cumsum(lens) - lens
                    combined_idx = (np.repeat(s_arr, lens)
                                    + np.arange(total_pts) - np.repeat(seg_offsets, lens))
                    if total_pts > len(candidate_buffer):
                        # Upfront max_cands estimate (cell_count_iter, based on
                        # id_to_sums) undercounts whenever part of the grid is
                        # supercell-only (nd<0 regions skip id_to_sums entirely
                        # -- see aggregate_point_data_to_cells_adaptive_nd's
                        # docstring) while a nearby nd>=0 chunk's overlap
                        # template still reaches into a genuinely dense region.
                        # Rather than trying to compute a tighter upfront bound
                        # (same class of bug could resurface for a different
                        # density shape), grow on demand -- rare in practice,
                        # correctness matters more than avoiding a realloc.
                        _n_buffer_resizes += 1
                        candidate_buffer = _np_zeros((total_pts, n_c + 2), dtype=float)
                    candidate_buffer[:total_pts] = blk_pts[combined_idx]
                _t_ov_copy += _pt() - _tcp0
                return candidate_buffer[:total_pts]

            # ---- merged contain + overlap pass ----------------------------------
            cur_cell_sum   = zero_sum.copy()
            cur_home_key   = None   # absolute codec key of the last-seen home cell
            contain_idx = 0; overlap_idx = 0
            n_contain_groups = len(_c_starts_col); n_overlap_groups = len(_o_starts_col)

            while contain_idx < n_contain_groups or overlap_idx < n_overlap_groups:
                do_contain = (contain_idx < n_contain_groups and
                              (overlap_idx >= n_overlap_groups or
                               _c_starts_col[contain_idx] <= _o_starts_col[overlap_idx]))
                do_overlap = (overlap_idx < n_overlap_groups and
                              (contain_idx >= n_contain_groups or
                               _o_starts_col[overlap_idx] <= _c_starts_col[contain_idx]))

                if do_contain:
                    _n_contain_grps += 1
                    _t0c = _pt()
                    start, end = int(_c_starts_col[contain_idx]), int(_c_ends_col[contain_idx])
                    contain_home_key = int(codec.key(0, int(rows[start]), int(cols[start])))
                    if cur_home_key is None or cell_changed[start] or cur_home_key != contain_home_key:
                        cur_home_key = contain_home_key
                        if _is_super_cell_nd:
                            _step = 2 ** (-_chunk_nd)
                            _par  = (int(rows[start]) % _step) * _step + (int(cols[start]) % _step)
                            cur_cell_sum = _contain_sum(_nd_super_shared_cntd[_chunk_nd][_par], cur_home_key)
                        else:
                            cur_cell_sum = _contain_sum(shared_cntd_offset, cur_home_key)
                    if _chunk_single_region:
                        sums_within_disks[start:end] += cur_cell_sum
                    else:
                        region_trgl_id = region_and_trgl[start]
                        region_sum = _contain_sum(cntd_offset_by_region[region_trgl_id], cur_home_key)
                        sums_within_disks[start:end] += cur_cell_sum + region_sum
                    if _cfg.PROFILE_FUNC_TIMES:
                        _cntd_keys_for_prof = (list(shared_cntd_offset + cur_home_key) if _chunk_single_region else
                            list(shared_cntd_offset + cur_home_key) + list(cntd_offset_by_region[region_trgl_id] + cur_home_key))
                        _n_contain_pts += (end - start) * sum(
                            block_cell_counts.get(k, 0)
                            for k in set(_cntd_keys_for_prof) if k in nonempty_block
                        )
                    if _do_weight and not _do_exact_supercell:
                        cr_id = cell_region[start]
                        cache_key = (cur_home_key, cr_id)
                        cached_invalid_area = va_contain_cache.get(cache_key)
                        if cached_invalid_area is None:
                            contain_cell_keys = set(int(k) for k in (contain_l0_offsets[cr_id] + cur_home_key))
                            n_invalid_cells = len(contain_cell_keys & invalid_cell_keys)
                            if _do_binary:
                                # invalid_cell_keys contains all non-fully-inside cells; apply
                                # _non_fully_frac as average valid fraction across that set
                                invalid_area_contrib = (1.0 - _non_fully_frac) * n_invalid_cells * _cell_area
                            else:
                                invalid_area_contrib = n_invalid_cells * _cell_area
                                if boundary_cell_keys:
                                    boundary_keys = contain_cell_keys & boundary_cell_keys
                                    if _do_per_cell:
                                        for k in boundary_keys:
                                            # see .get(..., 1.0) rationale at the
                                            # other boundary_cell_fracs site above
                                            invalid_area_contrib += (1.0 - boundary_cell_fracs.get(k, 1.0)) * _cell_area
                                    else:  # flat
                                        invalid_area_contrib += (1.0 - _bnd_frac) * len(boundary_keys) * _cell_area
                            va_contain_cache[cache_key] = invalid_area_contrib
                            cached_invalid_area = invalid_area_contrib
                        invalid_area[start:end] += cached_invalid_area
                    if end - 1 >= next_threshold:
                        next_threshold = progress.update(end - 1)
                    contain_idx += 1
                    _t_contain += _pt() - _t0c

                if do_overlap:
                    _n_overlap_grps += 1
                    _t0o = _pt()
                    start, end = int(_o_starts_col[overlap_idx]), int(_o_ends_col[overlap_idx])
                    overlap_home_key = int(codec.key(0, int(rows[start]), int(cols[start])))
                    _t0g = _pt()
                    if _is_super_cell_nd:
                        _step = 2 ** (-_chunk_nd)
                        _par  = (int(rows[start]) % _step) * _step + (int(cols[start]) % _step)
                        _ovlpd_tmpl = (_nd_super_sr_ovlpd[_chunk_nd][_par] if _chunk_single_region
                                       else ovlpd_offset_by_region.get(region_and_trgl[start],
                                                                        _nd_super_shared_cntd[_chunk_nd][_par]))
                    else:
                        _ovlpd_tmpl = (single_region_ovlpd_offset if _chunk_single_region
                                       else ovlpd_offset_by_region[region_and_trgl[start]])
                    candidates = gather_overlap_pts(_ovlpd_tmpl, overlap_home_key)
                    _t_ov_gather += _pt() - _t0g
                    _n_candidates += len(candidates)
                    if _do_weight and not _do_exact_supercell:
                        cr_id = cell_region[start]
                        overlap_cell_keys = set(int(k) for k in (overlap_l0_offsets[cr_id] + overlap_home_key))
                        invalid_overlap_keys = overlap_cell_keys & invalid_cell_keys
                        boundary_overlap_keys = overlap_cell_keys & boundary_cell_keys
                        if invalid_overlap_keys or boundary_overlap_keys:
                            # logit / flat / binary: vectorised centroid path.
                            # Build combined cell list with per-cell invalid weights:
                            #   inv cells  → full invalid weight (1.0 or 1-_non_fully_frac)
                            #   bnd cells  → partial invalid weight (per-cell or fixed frac)
                            invalid_cell_rcs  = []
                            invalid_cell_weights = []
                            invalid_weight = (1.0 - _non_fully_frac) if _do_binary else 1.0
                            for k in invalid_overlap_keys:
                                invalid_cell_rcs.append(codec.decode_tuple(k)[1])
                                invalid_cell_weights.append(invalid_weight)
                            if _do_per_cell:
                                for k in boundary_overlap_keys:
                                    invalid_cell_rcs.append(codec.decode_tuple(k)[1])
                                    # see .get(..., 1.0) rationale at the other
                                    # boundary_cell_fracs sites above
                                    invalid_cell_weights.append(1.0 - boundary_cell_fracs.get(k, 1.0))
                            elif _do_binary:
                                for k in boundary_overlap_keys:
                                    invalid_cell_rcs.append(codec.decode_tuple(k)[1])
                                    invalid_cell_weights.append(1.0 - _non_fully_frac)
                            else:  # flat
                                for k in boundary_overlap_keys:
                                    invalid_cell_rcs.append(codec.decode_tuple(k)[1])
                                    invalid_cell_weights.append(1.0 - _bnd_frac)
                            if invalid_cell_rcs:
                                src_xy_block = point_xy[start:end]
                                centroids = _np_array(
                                    [get_cell_centroid(rr, cc) for rr, cc in invalid_cell_rcs])
                                weights = _np_array(invalid_cell_weights)
                                centroid_dists = np_norm(
                                    src_xy_block[:, None, :] - centroids[None, :, :], axis=2)
                                # Prefilter: centroid farther than r + half_diag means
                                # no corner of the cell can touch the disk → zero share.
                                if _do_logit:
                                    share = 1 - 1 / (1 + _logit_Q * _np_exp(
                                        -_logit_B * (centroid_dists / r - 1)))
                                    share[centroid_dists > r + _half_diag] = 0.0
                                else:
                                    # flat / binary: uniform weight per cell, just filter
                                    share = (centroid_dists <= r + _half_diag).astype(float)
                                invalid_area[start:end] += (
                                    share * weights[None, :]).sum(axis=1) * _cell_area
                    if len(candidates):
                        cand_xy   = candidates[:, -2:]
                        cand_vals = candidates[:, :n_c].astype(float)
                        _t0d = _pt()
                        # OVERLAP_BLOCK caps rows per batch but not candidate count --
                        # when cand_xy itself is huge (e.g. extreme ppc where most of
                        # the dataset is within range), even one OVERLAP_BLOCK-sized
                        # row batch can blow past several GB. _batched_disk_sum
                        # enforces the MAX_DIST_MATRIX_BYTES cap regardless of
                        # OVERLAP_BLOCK, further sub-batching by rows when needed.
                        for blk_s in range(start, end, OVERLAP_BLOCK):
                            blk_e        = min(blk_s + OVERLAP_BLOCK, end)
                            src_xy_block = point_xy[blk_s:blk_e]
                            sums_within_disks[blk_s:blk_e] += _batched_disk_sum(
                                src_xy_block, cand_xy, cand_vals, r2)
                        _t_ov_distcheck += _pt() - _t0d
                    if end - 1 >= next_threshold:
                        next_threshold = progress.update(end - 1)
                    overlap_idx += 1
                    _t_overlap += _pt() - _t0o

            # Debug/experiment per-chunk timing: record this chunk's elapsed
            # time and its own contribution to the contain/overlap
            # ("cell aggregation") sub-timers. Not reached by the supercell
            # fast path above (which `continue`s past this point) -- only
            # meaningful for nd>=0 fixed-nd experiments.
            if _dbg_per_chunk:
                if not hasattr(_cfg, '_DEBUG_PER_CHUNK_STATS'):
                    _cfg._DEBUG_PER_CHUNK_STATS = []
                _dbg_row = {
                    'chunk_num': chunk_num,
                    'cc_lo': _cc_lo, 'cc_hi': _cc_hi,
                    'n_src': _n_src_chunk,
                    'n_tgt': blk_n,
                    'nd': _chunk_nd,
                    'ppc_est': _ppc_est,
                }
                # Occupancy/count-weighted covariates (only in scope when
                # USE_WEIGHTED_ND_DECISION computed them above) -- n_occupied
                # (distinct occupied coarse blocks), total point count across
                # them, and the occupancy-aware total-chunk-area ppc. Worth
                # tracking as regression covariates for the chunk-overhead
                # cost model regardless of whether they drove this chunk's
                # actual nd choice (fixed-nd experiments force nd separately).
                if _cfg.USE_WEIGHTED_ND_DECISION:
                    _dbg_row['n_occupied'] = int(len(_reg_nz))
                    _dbg_row['total_pts_reg'] = _total_pts_reg
                    _dbg_row['total_area_ppc'] = _total_area_ppc
                if getattr(_cfg, '_DEBUG_TRACK_L3_MEMORY', False):
                    # Real object sizes vs. what the row-strip planner's L3
                    # budget formula (pt_bytes/cell_bytes, see l3_budget above)
                    # actually assumes -- the formula only ever accounts for
                    # blk_pts (target-value-xy block, matches exactly by
                    # construction) and a flat 8 bytes/occupied-cell for
                    # block_cell_slices. It never counts: (a) the REAL Python
                    # dict overhead of block_cell_slices (int key + tuple
                    # value, far more than 8 bytes/entry), or (b) any
                    # source-point memory at all (point_xy slices touched
                    # during contain/overlap). This directly measures both
                    # gaps so we can compare real vs assumed L3 pressure.
                    _real_tgt_bytes = int(blk_pts.nbytes)
                    _real_cell_dict_bytes = int(
                        _sys_getsizeof(block_cell_slices)
                        + sum(_sys_getsizeof(k) for k in block_cell_slices)
                        + sum(_sys_getsizeof(v) for v in block_cell_slices.values())
                    )
                    _assumed_cell_bytes = len(block_cell_slices) * cell_bytes
                    _real_src_xy_bytes = int(_n_src_chunk * point_xy.itemsize * point_xy.shape[1])
                    _assumed_bytes_formula = blk_n * pt_bytes + len(block_cell_slices) * cell_bytes
                    _real_bytes_total = _real_tgt_bytes + _real_cell_dict_bytes + _real_src_xy_bytes
                    _dbg_row['real_tgt_bytes'] = _real_tgt_bytes
                    _dbg_row['real_cell_dict_bytes'] = _real_cell_dict_bytes
                    _dbg_row['assumed_cell_bytes'] = _assumed_cell_bytes
                    _dbg_row['real_src_xy_bytes'] = _real_src_xy_bytes
                    _dbg_row['assumed_bytes_formula'] = _assumed_bytes_formula
                    _dbg_row['real_bytes_total'] = _real_bytes_total
                    _dbg_row['l3_budget_bytes'] = l3_budget
                _cfg._DEBUG_PER_CHUNK_STATS.append(_dbg_row)
                _dbg_row_ref = _cfg._DEBUG_PER_CHUNK_STATS[-1]
                _dbg_row_ref.update({
                    'process_time_ms': (_pt() - _dbg_t0) * 1000,
                    'wall_time_ms': (_wall_pt() - _dbg_wall0) * 1000,
                    'aggregation_time_ms': ((_t_contain - _dbg_contain0) + (_t_overlap - _dbg_overlap0)) * 1000,
                })

        # end col chunk loop

        # Stream this chunk's sums into _stream_out (avoids pandas read-only COW issue).
        for _ci, _cn in enumerate(sum_radius_names):
            _stream_out[_cn][batch_pt_start:batch_pt_end] += \
                sums_within_disks[batch_pt_start:batch_pt_end, _ci]
        sums_within_disks[batch_pt_start:batch_pt_end] = 0.0

        row_iter += 1  # row_chunk_idx/chunk_rows now come from the _strip_plans
                       # tuple each iteration -- no manual advance needed.

    progress.done()

    # ---- write streamed results back to pts_source ------------------------------
    for _cn in sum_radius_names:
        pts_source[_cn] = _stream_out[_cn]

    if exclude_self:
        if grid._search_class.tgt_df_contains_src_df:
            for sum_name, value_col in zip(sum_radius_names, c):
                pts_source[sum_name] = pts_source[sum_name].values - pts_source[value_col]
        else:
            progress_print(
                "Option `exclude_self=True` but source and target DataFrames differ — "
                "point own values are not subtracted."
            )

    if _do_weight:
        valid_area_shares = (full_disk_area - invalid_area) / full_disk_area
        share_name = f'valid_area_share_{r}'
        pts_source[share_name] = valid_area_shares
        _safe_shares = np.where(valid_area_shares > 0, valid_area_shares, np.nan)
        for sum_name in sum_radius_names:
            if _keep_raw:
                pts_source[sum_name + '_raw'] = pts_source[sum_name].values
            pts_source[sum_name] = pts_source[sum_name].values / _safe_shares

        # Area-path diagnostics: attached to grid (not just printed) so the
        # weighting pass can be inspected/asserted-on directly, same pattern
        # as area_weight_validation/sum_validation below. n_buffer_resizes
        # tracks how often candidate_buffer's upfront size estimate (based on
        # cell_count_iter/id_to_sums, which undercounts whenever part of the
        # grid is supercell-only) had to grow on demand -- see the dynamic
        # resize fix in gather_overlap_pts. Non-zero here on a mode/dataset
        # that previously crashed is the direct signal the fix is engaging.
        _boundary_mask = valid_area_shares < 0.999
        if not hasattr(grid, 'area_weight_diagnostics'):
            grid.area_weight_diagnostics = {}
        grid.area_weight_diagnostics[area_weight] = {
            'n_points':            int(len(valid_area_shares)),
            'n_boundary_points':   int(np.sum(_boundary_mask)),
            'valid_area_share_min': round(float(np.nanmin(valid_area_shares)), 4) if len(valid_area_shares) else None,
            'valid_area_share_mean': round(float(np.nanmean(valid_area_shares)), 4) if len(valid_area_shares) else None,
            'n_buffer_resizes':    int(_n_buffer_resizes),
        }

        if _cfg.VALIDATE_AREA and getattr(grid, 'study_area', None) is not None:
            from aabpl.testing.validate_area_weight import validate_area_shares as _vaw
            _val = _vaw(pts_source, share_name, grid.study_area, r, x, y)
            _diffs = [abs(c - e) for c, e in _val]
            if not hasattr(grid, 'area_weight_validation'):
                grid.area_weight_validation = {}
            grid.area_weight_validation[area_weight] = {
                'mad':      round(float(sum(_diffs) / len(_diffs)), 6),
                'max_diff': round(float(max(_diffs)), 6),
            }

    # Restore original row order and drop the helper column.
    pts_source.sort_values('_orig_order', inplace=True)
    pts_source.drop(columns=['_orig_order'], inplace=True)

    # Inject sub-timers and counters as synthetic func_timer_dict entries.
    if _cfg.PROFILE_FUNC_TIMES:
        from aabpl.testing.test_performance import func_timer_dict as _ftd
        _now = _pt()
        def _inj(name, val):
            _ftd['times'].append({'func_name': name, 'start_time': _now - val, 'end_time': _now})
        _inj('_saa_contain',      _t_contain)
        _inj('_saa_overlap',      _t_overlap)
        _inj('_saa_ov_gather',    _t_ov_gather)
        _inj('_saa_ov_distcheck', _t_ov_distcheck)
        _inj('_saa_ov_filter',    _t_ov_filter)
        _inj('_saa_ov_cachelkup', _t_ov_cachelkup)
        _inj('_saa_ov_copy',      _t_ov_copy)
        _inj('_saa_super_index_build', _t_super_index_build)
        _inj('_saa_super_loop',        _t_super_loop)
        _inj('_saa_super_gather',      _t_super_gather)
        _inj('_saa_super_distcheck',   _t_super_distcheck)
        _inj('_saa_block_cell_slices_build', _t_block_cell_slices_build)
        # Counters stored as fake 0-duration entries with value in start_time field.
        for name, val in [
            ('_cnt_contain_grps', _n_contain_grps),
            ('_cnt_contain_pts',  _n_contain_pts),
            ('_cnt_overlap_grps', _n_overlap_grps),
            ('_cnt_candidates',   _n_candidates),
            ('_cnt_ov_keys',      _n_ov_keys),
            ('_cnt_super_grps',       _n_super_grps),
            ('_cnt_super_candidates', _n_super_candidates),
            ('_cnt_ov_template',  _n_ov_template),
            ('_cnt_o_cache_hit',  _n_o_cache_hit),
            ('_cnt_o_cache_miss', _n_o_cache_miss),
        ]:
            _ftd['times'].append({'func_name': name, 'start_time': 0, 'end_time': 0, '_count': val})

    if not validate and not _cfg.VALIDATE:
        return pts_source[sum_radius_names]

    # ---- brute-force validation -------------------------------------------------
    all_xy   = pts_target[[x, y]].values
    all_vals = pts_target[c].values if n_c > 1 else pts_target[c[0]].values.reshape(-1, 1)
    # Build per-region sorted point lists (descending cell population so denser
    # cells are tested first), then round-robin fill the budget.
    # Budget = min(max(100, n_regions), n_pts_src): cover every region at least
    # once, test at least 100 pts when possible, never exceed source size.
    cell_pop = {(row, col): cnt for row, col, cnt in cell_count_iter(grid) if cnt > 0}
    pts_source['_cell_pop'] = pts_source.apply(
        lambda row: cell_pop.get((int(row[row_name]), int(row[col_name])), 0), axis=1)
    region_pt_lists = (pts_source.sort_values('_cell_pop', ascending=False)
                       .groupby(cell_region_name, sort=False)
                       .apply(lambda g: list(g.index)))
    pts_source.drop(columns=['_cell_pop'], inplace=True)
    n_regions = len(region_pt_lists)
    _val_budget = min(max(100, n_regions), len(pts_source))
    val_indices = []  # list of (cell_region, pt_idx)
    slot = 0
    while len(val_indices) < _val_budget:
        added = False
        for cr, idxs in region_pt_lists.items():
            if slot < len(idxs) and len(val_indices) < _val_budget:
                val_indices.append((cr, idxs[slot]))
                added = True
        if not added:
            break
        slot += 1
    errors = []
    for cr, rep_idx in val_indices:
        rep_xy     = pts_source.loc[rep_idx, [x, y]].values.astype(float)
        dists      = np_norm(all_xy - rep_xy, axis=1)
        brute_sums = all_vals[dists <= r].sum(axis=0).flatten()
        own_vals   = 0.0
        if exclude_self and grid._search_class.tgt_df_contains_src_df:
            own_vals   = all_vals[pts_target.index == rep_idx].sum(axis=0).flatten()
            brute_sums = brute_sums - own_vals
        algo_sums = pts_source.loc[rep_idx, sum_radius_names].values.astype(float).flatten()
        diff = abs(brute_sums - algo_sums).max()
        if diff > 1e-6:
            errors.append((rep_idx, cr, brute_sums, algo_sums, diff))
    if errors:
        progress_print(f"VALIDATION FAILED: {len(errors)}/{len(val_indices)} point(s) have wrong sums (tested {len(val_indices)} pts across {n_regions} region(s)):")
        for rep_idx, cr, bf, algo, diff in errors:
            progress_print(f"  pt_id={rep_idx} cell_region={cr} brute={bf} algo={algo} diff={diff}")
    else:
        progress_print(f"VALIDATION OK: all {len(val_indices)} point(s) correct ({n_regions} region(s), budget={_val_budget}).")

    # Attach results as a grid attribute (mirrors grid.area_weight_validation)
    # so callers/tests can assert on outcomes directly instead of parsing
    # progress_print output.
    if not hasattr(grid, 'sum_validation'):
        grid.sum_validation = {}
    grid.sum_validation[tuple(sum_radius_names)] = {
        'n_tested':  len(val_indices),
        'n_regions': n_regions,
        'n_errors':  len(errors),
        'max_diff':  round(float(max((e[4] for e in errors), default=0.0)), 6),
    }

    return pts_source[sum_radius_names]

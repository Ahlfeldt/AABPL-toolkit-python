"""
Chunk-block streaming search_and_aggregate.

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
from ..study_area import compute_disk_cell_overlap

OVERLAP_BLOCK  = 256
ENABLE_COL_SPLIT = True   # set False to disable column splitting (for benchmarking)
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
    _sr_ovlpd_n_cells = 0  # used below to size candidate_buffer correctly
    if _single_region:
        single_region_ovlpd_offset = codec.offset_int(
            grid._search_class.single_region_ovlpd_cells)
        _sr_ovlpd_n_cells = len(single_region_ovlpd_offset)
    # ---- precomputed sums array (from aggregate_point_data_to_cells) -----------
    # Contain path: instead of summing block_cell_sums in a Python loop, index
    # directly into the contiguous _sums_array using codec keys → one numpy call.
    _sums_array  = grid.sums_array                           # (n_nodes, n_c) float64
    _id_to_lvl   = grid._search_internals.id_to_sums_by_lvl  # codec_int -> row index

    # Build sorted key/index arrays once for vectorised lookup in _contain_sum.
    _ck_keys = np.array(sorted(_id_to_lvl.keys()), dtype=np.int64)
    _ck_idxs = np.array([_id_to_lvl[k] for k in _ck_keys], dtype=np.int64)

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

    # ---- quadtree descent: build sub-cell entries in block_cell_slices ----------
    # Mirrors nest_next_lvl from aggregate_point_data_to_cells. Adds level-1..nd
    # entries keyed by absolute codec int so gather_overlap_pts can retrieve only
    # the sub-cell slices belonging to the overlap sub-cells of a boundary cell.
    def _nest_block_slices(blk_scnr, blk_rows, blk_cols, blk_n, block_cell_slices):
        """Add sub-cell (level 1..nest_depth) entries to block_cell_slices via quadtree descent."""
        if blk_n == 0 or nest_depth == 0:
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
                    if lvl + 1 <= nest_depth:
                        stack.append((cur_pos, next_pos, next_subcell_base,
                                      lvl + 1, child_row, child_col))
                    cur_pos = next_pos

    # ---- chunk sizing via L2 budget ---------------------------------------------
    # Cost per target pt in the block: (n_c+2)*8 bytes (pts_vals_xy slice)
    # Cost per cell aggregate: max(n_c,1)*8 bytes (cell_sums dict value)
    # We want  block_pts * pt_bytes + block_cells * cell_bytes  <= 0.9 * L2
    #
    # Estimate block_cells ~ block_pts * (avg cells per pt)
    # but we bound it simply: sort pts_per_row descending, cumsum until budget.
    pt_bytes   = (n_c + 2 + int(_do_weight)) * 8
    cell_bytes = max(n_c, 1) * 8

    # candidate buffer: hot in overlap pass, subtract from budget
    max_cells_per_region  = max(
        max((len(cells) for cells in ovlpd_cells_by_cell_region.values()), default=1),
        _sr_ovlpd_n_cells)
    sorted_cell_pops = sorted(cnt for _, _, cnt in cell_count_iter(grid))
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

    # joint cost per row, sort descending, cumsum
    row_cost       = pts_per_row * pt_bytes + cells_per_row * cell_bytes
    sorted_cost    = np.sort(row_cost)[::-1]
    # fixed: 2*r_rows overlap rows always in block regardless of chunk size
    fixed_cost     = int(np.sum(sorted_cost[:2 * r_rows]))
    remaining      = max(1, l3_budget - fixed_cost)
    cumcost        = np.cumsum(sorted_cost)
    fits           = int(np.searchsorted(cumcost, remaining, side='right'))
    chunk_rows     = max(1, fits)
    # column splitting: only needed when even a single row exceeds the budget
    _do_col_split  = ENABLE_COL_SPLIT and len(sorted_cost) > 0 and sorted_cost[0] > remaining

    if not silent:
        progress_print(
            f"Chunk-block: r_rows={r_rows}, chunk_rows={chunk_rows}, "
            f"col_split={'yes' if _do_col_split else 'no'}, "
            f"l3_budget={l3_budget//1024} KB, "
            f"~{(fixed_cost + int(np.sum(sorted_cost[:chunk_rows])))//1024} KB est. hot set")

    # ---- sort source points -----------------------------------------------------
    if suffix is None:
        suffix = '_' + str(r)
    sum_radius_names = [cn + suffix for cn in c]
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
    from time import process_time as _pt
    _t_contain      = 0.0
    _t_overlap      = 0.0
    _t_ov_gather    = 0.0   # gather_overlap_pts sub-step
    _t_ov_distcheck = 0.0   # distance-check matrix sub-step
    _n_contain_grps = 0     # total contain groups processed
    _n_overlap_grps = 0     # total overlap groups processed
    _n_candidates   = 0     # total candidate points gathered (overlap)
    _n_contain_pts  = 0     # total target-point × source-point pairs via contain path
    _n_o_cache_hit  = 0     # overlap_fp_cache hits
    _n_o_cache_miss = 0

    # ---- pre-loop invalid_area pass for exact super-cell mode -------------------
    # Cell-ID trigger: only pay the Shapely cost for points whose level-0 disk
    # cells (contain ∪ overlap) include at least one non-fully-interior cell.
    # Points are grouped by block (row // K, col // K): all points in the same
    # block share one clip polygon, so buffer + intersection + area are vectorized
    # across the whole group in a single C-level call.
    if _do_exact_supercell:
        # ---- Opt 1: vectorised trigger mask ---------------------------------
        # Build the dilation footprint from the union of (dr, dc) offsets across
        # all cell regions that actually have source points.  This matches the
        # disk geometry used by the trigger check and avoids the square-kernel
        # overestimate while being reliable (per-region level-0 lists are the
        # authoritative source).  Microscopic corner regions with no points are
        # excluded automatically via np.unique(cell_region).
        _active_cr_ids = np.unique(cell_region)
        footprint_offsets: set = set()
        for _cr in _active_cr_ids:
            for _drc in _cntd_cells_by_cell_region.get(int(_cr), []):
                footprint_offsets.add((_drc[0], _drc[1]))
            for _drc in _ovlpd_cells_by_cell_region_va.get(int(_cr), []):
                footprint_offsets.add((_drc[0], _drc[1]))
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

    n_chunks_total = math.ceil(n_src_unique / chunk_rows)  # updated below if col split
    chunk_num      = 0

    # ---- main chunk loop --------------------------------------------------------
    row_chunk_idx = 0
    row_iter      = 0  # tracks even/odd for snake column order
    while row_chunk_idx < n_src_unique:
        chunk_src_rows = src_unique_rows[row_chunk_idx : row_chunk_idx + chunk_rows]
        src_row_min    = int(chunk_src_rows[0])
        src_row_max    = int(chunk_src_rows[-1])

        # target row band needed for this chunk
        need_row_lo = max(tgt_row_lo_global, src_row_min - r_rows)
        need_row_hi = min(tgt_row_hi_global, src_row_max + r_rows)

        # advance blk_lo / blk_hi to cover [need_row_lo, need_row_hi]
        if loaded_row_lo is None or need_row_lo > loaded_row_lo:
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

        # ---- column chunk list for this row -------------------------------------
        # col splitting only fires when chunk_rows==1 and a single row > budget.
        # In that case source pts in this row are sorted by col → searchsorted works.
        if _do_col_split and len(c_starts) > 0:
            src_cols_in_row  = cols[batch_pt_start:batch_pt_end]
            src_unique_cols  = np.unique(src_cols_in_row)
            # cost per unique source col (pts in that col × pt_bytes)
            col_boundaries   = np.concatenate([[True],
                                               src_cols_in_row[1:] != src_cols_in_row[:-1]])
            col_starts       = _np_flatnonzero(col_boundaries)
            pts_per_col      = _np_append(col_starts[1:], len(src_cols_in_row)) - col_starts
            col_costs        = pts_per_col * pt_bytes
            sorted_col_costs = np.sort(col_costs)[::-1]
            cols_that_fit    = int(_np_searchsorted(np.cumsum(sorted_col_costs), remaining, side='right'))
            cols_per_chunk   = max(1, cols_that_fit)
            _col_chunks      = [(int(src_unique_cols[i]),
                                 int(src_unique_cols[min(i + cols_per_chunk - 1,
                                                         len(src_unique_cols) - 1)]))
                                for i in range(0, len(src_unique_cols), cols_per_chunk)]
            if row_iter % 2 == 1:
                _col_chunks = _col_chunks[::-1]  # snake: reverse direction on odd rows
            # update total chunk count estimate on first row that uses col split
            if row_iter == 0:
                n_chunks_total = math.ceil(n_src_unique / chunk_rows) * len(_col_chunks)
        else:
            _col_chunks = [(None, None)]

        # ---- col chunk inner loop -----------------------------------------------
        for (_cc_lo, _cc_hi) in _col_chunks:
            if _cc_lo is not None:
                # target: include r_rows padding either side
                tgt_col_lo       = _cc_lo - r_rows
                tgt_col_hi       = _cc_hi + r_rows
                tgt_cols_in_band = tgt_cols_full[blk_lo:blk_hi]
                col_filter_idx   = _np_flatnonzero((tgt_cols_in_band >= tgt_col_lo) &
                                                   (tgt_cols_in_band <= tgt_col_hi))
                blk_rows = tgt_rows_full[blk_lo:blk_hi][col_filter_idx]
                blk_cols = tgt_cols_in_band[col_filter_idx]
                blk_pts  = pts_vals_xy_full[blk_lo:blk_hi][col_filter_idx]
                blk_scnr = tgt_scnr_full[blk_lo:blk_hi][col_filter_idx]
                blk_n    = len(col_filter_idx)
                # restrict contain/overlap groups to this source col range
                # c_starts is sorted by source pt index = sorted by col (chunk_rows==1)
                contain_start_cols = cols[c_starts] if len(c_starts) else c_starts
                contain_grp_lo     = int(_np_searchsorted(contain_start_cols, _cc_lo,     side='left'))
                contain_grp_hi     = int(_np_searchsorted(contain_start_cols, _cc_hi + 1, side='left'))
                _c_starts_col      = c_starts[contain_grp_lo:contain_grp_hi]
                _c_ends_col        = c_ends  [contain_grp_lo:contain_grp_hi]
                overlap_start_cols = cols[o_starts] if len(o_starts) else o_starts
                overlap_grp_lo     = int(_np_searchsorted(overlap_start_cols, _cc_lo,     side='left'))
                overlap_grp_hi     = int(_np_searchsorted(overlap_start_cols, _cc_hi + 1, side='left'))
                _o_starts_col      = o_starts[overlap_grp_lo:overlap_grp_hi]
                _o_ends_col        = o_ends  [overlap_grp_lo:overlap_grp_hi]
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
            if n_chunks_total > 1:
                _col_tag = f'  cols=[{_cc_lo},{_cc_hi}]' if _cc_lo is not None else ''
                progress_print(
                    f"  chunk {chunk_num}/{n_chunks_total}  "
                    f"block_pts={blk_n}  rows=[{need_row_lo},{need_row_hi}]{_col_tag}  "
                    f"~{blk_n * pt_bytes // 1024} KB")

            # ---- pre-aggregate pts -> cell slices within block ------------------
            # block_cell_slices: absolute-codec-key -> (start, end) into blk_pts.
            # Used by the overlap path (gather raw pts into candidate_buffer).
            # Contains level-0 entries (one per occupied cell) plus level-1..nd
            # sub-cell entries built by _nest_block_slices quadtree descent.
            block_cell_slices = {}
            if _do_weight and not _do_exact_supercell:
                block_cell_valid_area = {}
            if blk_n > 0:
                cell_bdry = np.concatenate([[True],
                                            (blk_rows[1:] != blk_rows[:-1]) |
                                            (blk_cols[1:] != blk_cols[:-1])])
                cell_starts = _np_flatnonzero(cell_bdry)
                cell_ends   = _np_append(cell_starts[1:], blk_n)
                for s, e in zip(cell_starts.tolist(), cell_ends.tolist()):
                    row_val = int(blk_rows[s])
                    col_val = int(blk_cols[s])
                    _ck = int(codec.key(0, row_val, col_val))
                    block_cell_slices[_ck] = (s, e)
                    if _do_weight and not _do_exact_supercell:
                        if _ck in invalid_cell_keys:
                            va = 0.0
                        elif _ck in boundary_cell_keys:
                            va = (boundary_cell_fracs[_ck] if _do_per_cell else _bnd_frac) * _cell_area
                        else:
                            va = _cell_area
                        block_cell_valid_area[_ck] = va
                _nest_block_slices(blk_scnr, blk_rows, blk_cols, blk_n, block_cell_slices)

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
                nonlocal _n_o_cache_hit, _n_o_cache_miss
                raw_keys = codec_offsets + home_codec_key
                # Check cache with a cheap frozenset key first; only sort on miss.
                occupied_overlap_keys = frozenset(int(k) for k in raw_keys if k in nonempty_block)
                cached = overlap_fp_cache.get(occupied_overlap_keys)
                if cached is not None:
                    _n_o_cache_hit += 1
                    slices, total_pts = cached
                    pos = 0
                    for s, e, seg_len in slices:
                        candidate_buffer[pos:pos + seg_len] = blk_pts[s:e]
                        pos += seg_len
                    return candidate_buffer[:total_pts]
                _n_o_cache_miss += 1
                total_pts = 0
                slices = []
                for k in occupied_overlap_keys:
                    s, e = block_cell_slices[k]
                    seg_len = e - s
                    candidate_buffer[total_pts:total_pts + seg_len] = blk_pts[s:e]
                    slices.append((s, e, seg_len))
                    total_pts += seg_len
                if len(overlap_fp_cache) < _FP_CAP:
                    overlap_fp_cache[occupied_overlap_keys] = (slices, total_pts)
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
                        cur_cell_sum = _contain_sum(shared_cntd_offset, cur_home_key)
                    if _single_region:
                        sums_within_disks[start:end] += cur_cell_sum
                    else:
                        region_trgl_id = region_and_trgl[start]
                        region_sum = _contain_sum(cntd_offset_by_region[region_trgl_id], cur_home_key)
                        sums_within_disks[start:end] += cur_cell_sum + region_sum
                    if _cfg.PROFILE_FUNC_TIMES:
                        _cntd_keys_for_prof = (list(shared_cntd_offset + cur_home_key) if _single_region else
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
                                            invalid_area_contrib += (1.0 - boundary_cell_fracs[k]) * _cell_area
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
                    _ovlpd_tmpl = (single_region_ovlpd_offset if _single_region
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
                                    invalid_cell_weights.append(1.0 - boundary_cell_fracs[k])
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
                        for blk_s in range(start, end, OVERLAP_BLOCK):
                            blk_e        = min(blk_s + OVERLAP_BLOCK, end)
                            src_xy_block = point_xy[blk_s:blk_e]
                            sq_dists     = ((src_xy_block[:, 0:1] - cand_xy[None, :, 0]) ** 2 +
                                            (src_xy_block[:, 1:2] - cand_xy[None, :, 1]) ** 2)
                            within_radius = sq_dists <= r2
                            sums_within_disks[blk_s:blk_e] += within_radius.astype(float) @ cand_vals
                        _t_ov_distcheck += _pt() - _t0d
                    if end - 1 >= next_threshold:
                        next_threshold = progress.update(end - 1)
                    overlap_idx += 1
                    _t_overlap += _pt() - _t0o

        # end col chunk loop
        row_iter      += 1
        row_chunk_idx += chunk_rows

    progress.done()

    # ---- write results and exclude self -----------------------------------------
    pts_source[sum_radius_names] = pts_source[sum_radius_names].values + sums_within_disks

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
        # Counters stored as fake 0-duration entries with value in start_time field.
        for name, val in [
            ('_cnt_contain_grps', _n_contain_grps),
            ('_cnt_contain_pts',  _n_contain_pts),
            ('_cnt_overlap_grps', _n_overlap_grps),
            ('_cnt_candidates',   _n_candidates),
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

    return pts_source[sum_radius_names]

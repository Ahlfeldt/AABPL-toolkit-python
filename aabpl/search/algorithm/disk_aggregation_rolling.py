"""
Rolling-window search_and_aggregate — draft / work in progress.

Stripped from disk_aggregation.py:
  - numba / USE_OPTIMIZED_METHOD path removed
  - weight_valid_area path removed (to be re-added later)
  - plot_pt_disk path removed

Key structural differences from disk_aggregation.py:
  - Window covers only target rows needed by the current source batch:
      [src_row_min - window_depth, src_row_max + window_depth]
    where window_depth = ceil(r / spacing).
  - Circular buffer keyed by codec_key % window_size; win_tags guards stale slots.
  - Both contained and overlap sums are computed per batch (not two full passes).
  - batch_rows targets L2/4 working set to stay in cache.
  - pts_vals_xy_full is the full sorted target array; win_slices stores absolute indices.
  - TODO: nest_depth > 0 (subcell levels) not yet handled in window load/evict.
"""

import math
import numpy as np
from numpy import (
    zeros as _np_zeros,
    flatnonzero as _np_flatnonzero,
    append as _np_append,
    ones as _np_ones,
    diff as _np_diff,
    searchsorted as _np_searchsorted,
)
from numpy.linalg import norm as np_norm
from aabpl.utils.progress import SearchProgress, progress_print
from aabpl.testing.test_performance import time_func_perf
from aabpl import config as _cfg
from ..point_assignment import cell_count_iter

OVERLAP_BLOCK = 256
L2_BYTES = 256 * 1024  # 256 KB — typical L2; target L2/4


def _next_pow2(n):
    return max(1, 1 << (n - 1).bit_length()) if n > 1 else 1


class _WindowState:
    """
    Circular-buffer window over target-point cell data.

    Attributes
    ----------
    sums      : (window_size, n_c) float64  — cell column sums
    slices    : (window_size, 2)   int64    — absolute [start, end] into pts_vals_xy_full
    tags      : (window_size,)     int64    — codec key occupying each slot; -1 = empty
    nonempty  : set[int]                    — codec keys currently in window
    row_lo/hi : int                         — target row range currently loaded
    """

    def __init__(self, window_size, n_c):
        self.sums     = _np_zeros((window_size, max(n_c, 1)), dtype=float)
        self.slices   = _np_zeros((window_size, 2), dtype=np.int64)
        self.tags     = _np_zeros(window_size, dtype=np.int64) - 1
        self.nonempty = set()
        self.row_lo   = None
        self.row_hi   = None
        self.size     = window_size

    def load_row(self, row, tgt_rows, tgt_cols, pts_vals_xy_full, codec, n_c, lo_ptr, hi_ptr):
        """Load all cells for a single target row into the circular buffer."""
        # find slice of target pts for this row (pts_vals_xy_full already sorted by row,col)
        r_lo = int(_np_searchsorted(tgt_rows, row,     side='left'))
        r_hi = int(_np_searchsorted(tgt_rows, row + 1, side='left'))
        if r_lo >= r_hi:
            return  # empty row

        row_cols = tgt_cols[r_lo:r_hi]
        # find column-group boundaries within this row
        col_change = _np_ones(r_hi - r_lo, dtype=bool)
        col_change[1:] = row_cols[1:] != row_cols[:-1]
        cell_starts = r_lo + _np_flatnonzero(col_change)
        cell_ends   = _np_append(cell_starts[1:], r_hi)

        size = self.size
        for s, e in zip(cell_starts, cell_ends):
            col = int(tgt_cols[s])
            key = int(codec.key(0, row, col))
            slot = key % size
            self.sums[slot]   = pts_vals_xy_full[s:e, :-2].sum(axis=0)
            self.slices[slot] = [s, e]
            self.tags[slot]   = key
            self.nonempty.add(key)

    def evict_row(self, row, tgt_rows, tgt_cols, codec):
        """Remove all cells for a single target row from the circular buffer."""
        r_lo = int(_np_searchsorted(tgt_rows, row,     side='left'))
        r_hi = int(_np_searchsorted(tgt_rows, row + 1, side='left'))
        if r_lo >= r_hi:
            return

        row_cols = tgt_cols[r_lo:r_hi]
        col_change = _np_ones(r_hi - r_lo, dtype=bool)
        col_change[1:] = row_cols[1:] != row_cols[:-1]
        cell_starts = r_lo + _np_flatnonzero(col_change)

        size = self.size
        for s in cell_starts:
            col = int(tgt_cols[s])
            key = int(codec.key(0, row, col))
            slot = key % size
            if self.tags[slot] == key:
                self.tags[slot] = -1
            self.nonempty.discard(key)

    def update(self, new_row_lo, new_row_hi, tgt_rows, tgt_cols, pts_vals_xy_full, codec, n_c):
        """Slide window to [new_row_lo, new_row_hi], loading/evicting as needed."""
        if self.row_lo is None:
            # first load
            for row in range(new_row_lo, new_row_hi + 1):
                self.load_row(row, tgt_rows, tgt_cols, pts_vals_xy_full, codec, n_c, None, None)
            self.row_lo = new_row_lo
            self.row_hi = new_row_hi
            return

        # evict rows falling out the bottom
        for row in range(self.row_lo, min(new_row_lo, self.row_hi + 1)):
            self.evict_row(row, tgt_rows, tgt_cols, codec)

        # load rows entering the top
        load_from = max(self.row_hi + 1, new_row_lo)
        for row in range(load_from, new_row_hi + 1):
            self.load_row(row, tgt_rows, tgt_cols, pts_vals_xy_full, codec, n_c, None, None)

        self.row_lo = new_row_lo
        self.row_hi = new_row_hi


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
):
    if pts_target is None:
        pts_target = pts_source

    # ---- grid internals ---------------------------------------------------------
    codec                      = grid._search_internals.cell_codec
    grid_spacing               = grid._search_internals.spacing
    contain_region_mult        = grid._search_class.contain_region_mult
    shared_cntd_cells          = grid._search_class.shared_cntd_cells
    cntd_cells_by_region       = grid._search_class.region_and_trgl_id_to_distinct_cntd_cells
    ovlpd_cells_by_region      = grid._search_class.region_and_trgl_id_to_distinct_ovlpd_cells
    ovlpd_cells_by_cell_region = grid._search_class.region_id_to_ovlpd_cells

    n_pts  = len(pts_source)
    n_c    = len(c)
    r2     = r * r
    zero_sum = _np_zeros(n_c, dtype=float)

    # ---- integer offset templates -----------------------------------------------
    shared_cntd_offset     = codec.offset_int(shared_cntd_cells)
    cntd_offset_by_region  = {rid: codec.offset_int(cells) for rid, cells in cntd_cells_by_region.items()}
    ovlpd_offset_by_region = {rid: codec.offset_int(cells) for rid, cells in ovlpd_cells_by_region.items()}

    # ---- target point arrays (sorted by row, col) -------------------------------
    pts_target_sorted = pts_target.sort_values([row_name, col_name])
    tgt_rows         = pts_target_sorted[row_name].values.astype(int)
    tgt_cols         = pts_target_sorted[col_name].values.astype(int)
    pts_vals_xy_full = pts_target_sorted[c + [x, y]].values  # absolute index base

    tgt_row_lo_global = int(tgt_rows.min()) if len(tgt_rows) else 0
    tgt_row_hi_global = int(tgt_rows.max()) if len(tgt_rows) else 0

    # ---- window sizing ----------------------------------------------------------
    # r_rows: how many neighbouring rows a disk of radius r can reach
    r_rows = int(math.ceil(r / grid_spacing))

    # cells_per_row: actual non-empty cell count for each distinct target row
    tgt_row_change = _np_ones(len(tgt_rows), dtype=bool)
    if len(tgt_rows) > 1:
        tgt_row_change[1:] = (tgt_rows[1:] != tgt_rows[:-1]) | (tgt_cols[1:] != tgt_cols[:-1])
    cell_start_idx = _np_flatnonzero(tgt_row_change)
    distinct_rows  = tgt_rows[cell_start_idx]
    # cells per distinct row (number of (row,col) pairs)
    cells_per_row  = _np_append(cell_start_idx[1:], len(tgt_rows)) - cell_start_idx

    # worst-case batch sizing: sort rows by combined cost descending, cumsum until budget.
    # Budget = 0.9 * L2, minus candidate_buffer (also hot in overlap pass).
    # Per-cell cost: win_sums (n_c*8) + win_tags (8) + win_slices (16) bytes.
    bytes_per_cell     = max(n_c, 1) * 8 + 8 + 16
    # pts_per_row not yet tracked here — added once pts_vals_xy is windowed
    max_cells_per_region = max((len(cells) for cells in ovlpd_cells_by_cell_region.values()), default=1)
    _counts = sorted(cnt for _, _, cnt in cell_count_iter(grid))
    max_candidates = sum(_counts[-max_cells_per_region:]) if _counts else 1
    candidate_budget   = max_candidates * (n_c + 2) * 8
    l2_budget          = int(0.9 * L2_BYTES) - candidate_budget

    sorted_cells  = np.sort(cells_per_row)[::-1]  # densest rows first
    # fixed cost: 2*r_rows rows always in window regardless of batch size
    fixed_budget  = int(np.sum(sorted_cells[:2 * r_rows])) * bytes_per_cell
    remaining     = max(1, l2_budget - fixed_budget)
    cumcost       = np.cumsum(sorted_cells) * bytes_per_cell
    fits          = int(np.searchsorted(cumcost, remaining, side='right'))
    batch_rows    = max(1, fits)

    # circular buffer size: worst-case cells in window at ~50% load factor
    max_cells_in_window = int(np.sum(sorted_cells[:2 * r_rows + batch_rows]))
    window_size = _next_pow2(max(max_cells_in_window * 2, 8))

    if not silent:
        ws_kb = max_cells_in_window * bytes_per_cell / 1024
        progress_print(f"Rolling window: r_rows={r_rows}, batch_rows={batch_rows}, "
                       f"window_size={window_size} slots, "
                       f"~{ws_kb:.1f} KB cell working set + {candidate_budget/1024:.1f} KB candidates")

    win = _WindowState(window_size, n_c)

    # ---- helpers that close over win -------------------------------------------
    def covered_cell_keys(offset_template, home_key):
        _ne = win.nonempty
        return [k for k in (offset_template + home_key) if k in _ne]

    def sum_over_cells(cell_keys):
        if not cell_keys:
            return zero_sum.copy()
        total = zero_sum.copy()
        _tags = win.tags; _sums = win.sums; _size = win.size
        for k in cell_keys:
            slot = k % _size
            if _tags[slot] == k:
                total += _sums[slot]
        return total

    candidate_buffer = _np_zeros((max(max_candidates, 1), n_c + 2), dtype=float)

    def gather_overlap_candidates(home_key, region_id):
        _buffer = candidate_buffer
        _tags = win.tags; _slices = win.slices; _size = win.size
        _pvxy = pts_vals_xy_full
        n = 0
        for k in covered_cell_keys(ovlpd_offset_by_region[region_id], home_key):
            slot = k % _size
            if _tags[slot] != k:
                continue
            s, e = int(_slices[slot, 0]), int(_slices[slot, 1])
            cell_array = _pvxy[s:e]
            length = len(cell_array)
            _buffer[n : n + length] = cell_array
            n += length
        return _buffer[:n]

    # ---- sort source points: (row, col, region_and_trgl_id) --------------------
    if suffix is None:
        suffix = '_' + str(r)
    sum_radius_names = [cname + suffix for cname in c]
    pts_source[sum_radius_names] = 0

    pts_source.sort_values([row_name, col_name, 'region_and_trgl_id'], inplace=True)
    point_xy        = pts_source[[x, y]].values
    rows            = pts_source[row_name].values.astype(int)
    cols            = pts_source[col_name].values.astype(int)
    cell_region     = pts_source[cell_region_name].values
    region_and_trgl = pts_source['region_and_trgl_id'].values

    home_key = (rows * codec.scale - codec._rlo) * codec.row_stride + (cols * codec.scale - codec._clo)

    # group boundaries
    cell_changed    = _np_ones(n_pts, dtype=bool)
    contain_changed = _np_ones(n_pts, dtype=bool)
    overlap_changed = _np_ones(n_pts, dtype=bool)
    if n_pts > 1:
        cell_changed[1:]    = home_key[1:] != home_key[:-1]
        rt_changed          = region_and_trgl[1:] != region_and_trgl[:-1]
        contain_changed[1:] = (cell_changed[1:]
                               | (cell_region[1:] // contain_region_mult != cell_region[:-1] // contain_region_mult)
                               | rt_changed)
        overlap_changed[1:] = (cell_changed[1:]
                               | (cell_region[1:] % contain_region_mult != cell_region[:-1] % contain_region_mult)
                               | rt_changed)

    sums_within_disks = _np_zeros((n_pts, n_c))
    progress = SearchProgress(silent=silent, n_pts=n_pts)
    progress.start()
    next_threshold = progress.next_threshold

    # ---- find source row batch boundaries ---------------------------------------
    src_row_change_idx = _np_flatnonzero(_np_ones(n_pts, dtype=bool)
                                          if n_pts == 0 else
                                          np.concatenate([[True], rows[1:] != rows[:-1]]))
    src_unique_rows = rows[src_row_change_idx]
    n_src_unique    = len(src_unique_rows)

    # precompute contain/overlap group start arrays for slicing per batch
    contain_starts_all = _np_flatnonzero(contain_changed)
    contain_ends_all   = _np_append(contain_starts_all[1:], n_pts)
    overlap_starts_all = _np_flatnonzero(overlap_changed)
    overlap_ends_all   = _np_append(overlap_starts_all[1:], n_pts)

    # ---- main batch loop --------------------------------------------------------
    row_batch_idx = 0
    n_batches_total = math.ceil(n_src_unique / batch_rows)
    batch_num = 0

    while row_batch_idx < n_src_unique:
        batch_src_rows = src_unique_rows[row_batch_idx : row_batch_idx + batch_rows]
        src_row_min    = int(batch_src_rows[0])
        src_row_max    = int(batch_src_rows[-1])

        # target rows needed for this batch
        tgt_win_lo = max(tgt_row_lo_global, src_row_min - r_rows)
        tgt_win_hi = min(tgt_row_hi_global, src_row_max + r_rows)

        # slide the window
        win.update(tgt_win_lo, tgt_win_hi, tgt_rows, tgt_cols, pts_vals_xy_full, codec, n_c)
        batch_num += 1
        if n_batches_total > 1:
            win_kb = (win.sums.nbytes + win.tags.nbytes + win.slices.nbytes) // 1024
            progress_print(f"  batch {batch_num}/{n_batches_total}  window={win_kb} KB  nonempty={len(win.nonempty)}")

        # source point index range for this batch
        batch_pt_start = int(src_row_change_idx[row_batch_idx])
        next_row_batch = row_batch_idx + batch_rows
        batch_pt_end   = int(src_row_change_idx[next_row_batch]) if next_row_batch < n_src_unique else n_pts

        # find contain groups within this batch
        cg_lo = int(_np_searchsorted(contain_starts_all, batch_pt_start, side='left'))
        cg_hi = int(_np_searchsorted(contain_starts_all, batch_pt_end,   side='left'))
        c_starts = contain_starts_all[cg_lo:cg_hi]
        c_ends   = contain_ends_all[cg_lo:cg_hi]

        # find overlap groups within this batch
        og_lo = int(_np_searchsorted(overlap_starts_all, batch_pt_start, side='left'))
        og_hi = int(_np_searchsorted(overlap_starts_all, batch_pt_end,   side='left'))
        o_starts = overlap_starts_all[og_lo:og_hi]
        o_ends   = overlap_ends_all[og_lo:og_hi]

        # process contain and overlap groups together (merged single pass)
        # reset cur_hk each batch: window may have evicted cells since last batch
        cur_cell_sum = zero_sum.copy()
        cur_hk = None
        ci = 0; oi = 0
        n_cg = len(c_starts); n_og = len(o_starts)

        while ci < n_cg or oi < n_og:
            do_contain = ci < n_cg and (oi >= n_og or c_starts[ci] <= o_starts[oi])
            do_overlap = oi < n_og and (ci >= n_cg or o_starts[oi] <= c_starts[ci])

            if do_contain:
                start, end = int(c_starts[ci]), int(c_ends[ci])
                hk = int(home_key[start])
                if cur_hk is None or cell_changed[start] or cur_hk != hk:
                    cur_cell_sum = sum_over_cells(covered_cell_keys(shared_cntd_offset, hk))
                    cur_hk = hk
                region_sum = sum_over_cells(covered_cell_keys(cntd_offset_by_region[region_and_trgl[start]], hk))
                sums_within_disks[start:end] += cur_cell_sum + region_sum
                if end - 1 >= next_threshold:
                    next_threshold = progress.update(end - 1)
                ci += 1

            if do_overlap:
                start, end = int(o_starts[oi]), int(o_ends[oi])
                hk = int(home_key[start])
                candidates = gather_overlap_candidates(hk, region_and_trgl[start])
                if len(candidates):
                    candidate_xy   = candidates[:, -2:]
                    candidate_vals = candidates[:, :-2].astype(float)
                    for block_start in range(start, end, OVERLAP_BLOCK):
                        block_end = min(block_start + OVERLAP_BLOCK, end)
                        block_xy  = point_xy[block_start:block_end]
                        dx = block_xy[:, 0][:, None] - candidate_xy[None, :, 0]
                        dy = block_xy[:, 1][:, None] - candidate_xy[None, :, 1]
                        inside = (dx * dx + dy * dy) <= r2
                        sums_within_disks[block_start:block_end] += inside.astype(float) @ candidate_vals
                if end - 1 >= next_threshold:
                    next_threshold = progress.update(end - 1)
                oi += 1

        row_batch_idx += batch_rows

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

    if not validate and not _cfg.VALIDATE:
        return pts_source[sum_radius_names]

    # ---- brute-force validation -------------------------------------------------
    all_xy   = pts_target[[x, y]].values
    all_vals = pts_target[c].values if n_c > 1 else pts_target[c[0]].values.reshape(-1, 1)
    cell_pop = {(row, col): cnt for row, col, cnt in cell_count_iter(grid) if cnt > 0}
    pts_source['_cell_pop'] = pts_source.apply(
        lambda row: cell_pop.get((int(row[row_name]), int(row[col_name])), 0), axis=1)
    rep_indices = (pts_source.sort_values('_cell_pop', ascending=False)
                   .groupby(cell_region_name, sort=False).apply(lambda g: g.index[0]))
    pts_source.drop(columns=['_cell_pop'], inplace=True)
    errors = []
    for cr, rep_idx in rep_indices.items():
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
        progress_print(f"VALIDATION FAILED: {len(errors)}/{len(rep_indices)} cell_region(s) have wrong sums:")
        for rep_idx, cr, bf, algo, diff in errors:
            progress_print(f"  pt_id={rep_idx} cell_region={cr} brute={bf} algo={algo} diff={diff}")
    else:
        progress_print(f"VALIDATION OK: all {len(rep_indices)} cell_region(s) correct.")

    return pts_source[sum_radius_names]

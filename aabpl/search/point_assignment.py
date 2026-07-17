import math as _math
import numpy as np
from numpy import (
    array as _np_array,
    zeros as _np_zeros,
    linspace as _np_linspace,
    diff as _np_diff,
    where as _np_where,
    searchsorted as _np_searchsorted,
    bincount as _np_bincount,
    maximum as _np_maximum,
)
from pandas import (DataFrame as _pd_DataFrame, cut as _pd_cut, concat as _pd_concat) 
from aabpl.utils.misc import find_column_name,arr_to_tpls
from aabpl.testing.test_performance import time_func_perf
# from aabpl.doc.docstrings import fixdocstring

################ assign_points_to_cells ######################################################################################
# @fixdocstring
@time_func_perf
def assign_points_to_cells(
    grid:dict,
    pts:_pd_DataFrame,
    y:str='lat',
    x:str='lon',
    row_name:str='id_y',
    col_name:str='id_x',
    silent:bool=False,
) -> _pd_DataFrame:
    """Compute integer grid-cell indices for every point and write them into `pts`.

    Each point is assigned to the search cell whose south-west corner is at
    ``(grid.total_bounds.ymin + row * spacing, grid.total_bounds.xmin + col * spacing)``.
    The assignment is a simple floor-division of the point's coordinate offset by
    ``grid._search_spacing``, so it runs in O(N) with no loops.

    Args:
        grid: Grid object with attributes ``total_bounds.ymin``, ``total_bounds.xmin``,
            and ``_search_spacing`` (cell side length in the projected CRS units).
        pts: Point DataFrame.  Two integer columns are added **in-place**:
            ``row_name`` (row index, north-south) and ``col_name`` (column index,
            east-west).  Points outside the grid bounds receive negative or
            out-of-range indices and are handled by the caller.
        y: Column in ``pts`` holding the northing / latitude coordinate.
        x: Column in ``pts`` holding the easting / longitude coordinate.
        row_name: Name for the new row-index column written into ``pts``.
        col_name: Name for the new column-index column written into ``pts``.
        silent: Unused; reserved for future progress logging.

    Returns:
        DataFrame slice ``pts[[row_name, col_name]]`` — the two index columns only.
        The full ``pts`` DataFrame is also modified in-place.
    """
    # to do change to cut
    # for each row select relevant points, then refine selection with columns to obtain cells
    pts[row_name] = ((pts[y]-grid._search_internals.bounds.ymin) // grid._search_internals.spacing).astype(int)
    pts[col_name] = ((pts[x]-grid._search_internals.bounds.xmin) // grid._search_internals.spacing).astype(int)
        
    return pts[[row_name, col_name]]

def translate_subcell_row_col_to_value(row_nr, col_nr, lvl, nest_depth):
    """TODO create function that that translates subcell row col to value"""
    subcell_value = 0
    for i in range(1,nest_depth+1):
            n = 2**(i)
            subcell_mult = 2**((nest_depth-i)*2)
            print("nest_depth", nest_depth, "i", i, "subcell_mult",subcell_mult, "n", n)
            print([int(x//(1/n)%n%2) for x in _np_linspace(0,1,10)])
            # TODO this is not yet ready.
            subcell_value += (col_nr//(1/n)%n%2 * subcell_mult  + row_nr//(1/n)%n%2 * 2 * subcell_mult).astype(int)
        #
    return subcell_value

@time_func_perf
def aggregate_point_data_to_cells(
    grid:dict,
    pts:_pd_DataFrame,
    c:list=['employment'],
    y:str='lat',
    x:str='lon',
    row_name:str='id_y',
    col_name:str='id_x',
    nest_depth:int=5,
    silent:bool=False,
) -> _pd_DataFrame:
    """Sort points into grid cells and build per-cell (and per-quadtree-subcell) lookup dicts.

    The function sorts ``pts`` by ``(row_name, col_name, subcell_nr)`` and then
    scans the sorted array once to aggregate the value columns ``c`` for every
    non-empty cell and every non-empty quadtree subcell up to ``nest_depth`` levels
    deep.  All per-cell and per-subcell data are written directly onto ``grid``.

    **Quadtree encoding** (``nest_depth > 0``): each cell is recursively split into
    four quadrants (NW/NE/SW/SE) up to ``nest_depth`` times.  A scalar ``subcell_nr``
    encodes the full path from the root: at each level *i* the two bits
    ``(row_bit, col_bit)`` are packed as ``row_bit*2 + col_bit`` and shifted left by
    ``2*(nest_depth - i)`` bits, then summed across levels.  Points are sorted by
    this value so subcell ranges can be found with ``searchsorted`` rather than
    linear scans.

    Args:
        grid: Grid object.  Must already have ``total_bounds``, ``_search_spacing``,
            and optionally ``cell_codec`` (integer-key codec for the ``_by_lvl`` dicts).
            Six attributes are written on return — see *Writes to grid* below.
        pts: Point DataFrame sorted by ``(row_name, col_name)`` on entry (or any
            order — the function re-sorts internally).  A temporary ``sc_nr`` column
            is appended when ``nest_depth > 0``; it is **not** removed afterward.
        c: Value columns to aggregate.  Every column must exist in ``pts``.
        y: Northing / latitude column in ``pts``.
        x: Easting / longitude column in ``pts``.
        row_name: Integer row-index column (written by ``assign_points_to_cells``).
        col_name: Integer column-index column (written by ``assign_points_to_cells``).
        nest_depth: Number of quadtree levels below the base cell.  0 disables
            subcell nesting entirely (no ``sc_nr`` column, no ``_by_lvl`` subcell
            entries beyond level 0).  Higher values give finer spatial resolution
            for the radius search at the cost of more dict entries
            (up to ``4^1 + … + 4^nest_depth`` extra entries per non-empty cell).
        silent: Unused; reserved for future progress logging.

    Writes to grid:
        id_to_sums        : ``{(row, col): ndarray}`` — sum of ``c`` columns over all
                            points in the cell.  Used by ``cell_count_iter`` (below)
                            and directly by the search hot path.
        id_to_sums_by_lvl : ``{key: ndarray}`` — same as ``id_to_sums`` at level 0,
                            plus one entry per non-empty subcell at each deeper level.
                            Keys are codec-packed int64 when ``grid.cell_codec`` is set,
                            else ``(lvl, (row_centroid, col_centroid))`` tuples.
                            The main radius-search loop reads this dict exclusively.
        id_to_vals_xy_by_lvl : same structure, stores packed (start<<32)|end index
                            ranges into ``grid.pts_vals_xy`` (not per-cell arrays).

    Note: id_to_pt_ids / id_to_pt_ids_by_lvl were removed (see
    aabpl/testing/run_all_tests.py's regression check that they stay gone) --
    per-point ids for the whole sorted block are available via grid.pts_ids
    instead.
    """
    # what is points data initally sorted by
    # aggregate cells to super cells to save lookup time 
    aggregate_level = 0

    # sort by row, then by col = resulting in cell wise sorting
    # then sort for quadrants
    cols_for_sort = [row_name, col_name]
    subcell_nr = find_column_name('sc_nr', existing_columns=pts.columns)

    # -- optional: auto-reduce nest_depth when cells are too sparse to benefit --
    if False and nest_depth > 0:
        from aabpl.search.spacing_topology import count_cells_per_level, recommend_max_nest_depth
        _x_arr = pts[x].values
        _y_arr = pts[y].values
        _cell_counts = count_cells_per_level(
            _x_arr, _y_arr,
            xmin=grid._search_internals.bounds.xmin,
            ymin=grid._search_internals.bounds.ymin,
            spacing=grid._search_internals.spacing,
            max_nd=nest_depth,
        )
        _recommended_nd = recommend_max_nest_depth(
            n_pts_src=getattr(grid, '_n_pts_src', len(pts)),
            n_pts_tgt=len(pts),
            cell_counts=_cell_counts,
            spacing_ratio=getattr(grid, '_spacing_ratio', 2.0),
        )
        if _recommended_nd < nest_depth:
            nest_depth = _recommended_nd
            if nest_depth == 0:
                grid._search_internals.cell_codec = None
            else:
                from aabpl.utils.cell_keys import CellKeyCodec
                grid._search_internals.cell_codec = CellKeyCodec(
                    nest_depth=nest_depth,
                    row_lo=int(grid.row_ids.min()), row_hi=int(grid.row_ids.max()),
                    col_lo=int(grid.col_ids.min()), col_hi=int(grid.col_ids.max()),
                    offset_margin=16,
                )

    if nest_depth > 0:
        # offsets normalized to 0-1
        offset_x = 0.5 + ((pts[x]-grid._search_internals.bounds.xmin)%grid._search_internals.spacing - grid._search_internals.spacing/2) / grid._search_internals.spacing
        offset_y = 0.5 + ((pts[y]-grid._search_internals.bounds.ymin)%grid._search_internals.spacing - grid._search_internals.spacing/2) / grid._search_internals.spacing
        pts[subcell_nr] = 0

        # loop through nest levels starting from the broadest/most aggregate end with smallest/most narrow
        for i in range(1,nest_depth+1):
            n = 2**(i)
            subcell_mult = 2**((nest_depth-i)*2)
            # print("nest_depth", nest_depth, "i", i, "subcell_mult",subcell_mult, "n", n)
            # print([int(x//(1/n)%n%2) for x in _np_linspace(0,1,10)])
            pts[subcell_nr] += (offset_x//(1/n)%n%2 * subcell_mult  + offset_y//(1/n)%n%2 * 2 * subcell_mult).astype(int)
        #
        cols_for_sort.append(subcell_nr)
    #
    
    # Sort via a local index array instead of pts.sort_values(inplace=True) --
    # see the identical fix (and rationale) in aggregate_point_data_to_cells_adaptive_nd
    # below: pts can alias pts_source in a self-search, and physically
    # reordering it here can silently desynchronize a caller's own arrays built
    # from pts_source's original row order.
    _sort_idx = np.lexsort(tuple(pts[col].values for col in reversed(cols_for_sort)))

    # extract variables from dataframe for faster access speed
    pts_rows = pts[row_name].values[_sort_idx]
    pts_cols = pts[col_name].values[_sort_idx]
    pts_vals = pts[c].values[_sort_idx]
    pts_vals_xy = pts[c+[x,y]].values[_sort_idx]
    pts_ids = pts.index.values[_sort_idx]
    pts_subcell_nrs = pts[subcell_nr].values[_sort_idx] if nest_depth > 0 else _np_zeros(len(pts_cols))
    n_pts = len(pts)

    # row group boundaries: positions where row_name changes value (pts is sorted)
    row_id_indexes = list(_np_where(_np_diff(pts_rows, prepend=pts_rows[0] - 1) != 0)[0])
    row_ids = [int(pts_rows[i]) for i in row_id_indexes]

    # output dicts
    # NOTE (future min/max/range support): id_to_sums stores a per-cell SUM of the
    # value columns. min/max/range are not additive — supporting them will require
    # an analogous per-cell MIN/MAX dict here (e.g. id_to_min / id_to_max via
    # .min(axis=0)/.max(axis=0) over each cell's points) that disk_aggregation can
    # reduce as min-of-mins / max-of-maxes for contained cells. Not implemented yet.
    id_to_sums = {}
    id_to_sums_by_lvl = {}
    id_to_vals_xy_by_lvl = {}

    # pre-allocate contiguous array for all subcell sums (avoids one small ndarray
    # object per node; values are indexed by int stored in id_to_sums_by_lvl)
    _max_nodes = n_pts * (nest_depth + 1)
    _n_cols_arr = max(len(c), 1)
    _sums_array = _np_zeros((_max_nodes, _n_cols_arr), dtype=float)
    _node_count = 0

    # Key builder for the *_by_lvl dicts: packed int64 when the integer-key codec
    # is active, else the original (lvl,(row,col)) tuple.
    # The level-0 (row,col) dicts above keep tuple keys (clusters/plots/exports).
    _codec = getattr(grid._search_internals, 'cell_codec', None)
    if _codec is not None:
        def _k(lvl, rc):
            return int(_codec.key(lvl, rc[0], rc[1]))
    else:
        def _k(lvl, rc):
            return (lvl, rc)

    def nest_next_lvl(pos_min_init, pos_max_init, row_init, col_init):
        """Iterative quadtree descent: replaces recursion with an explicit stack.
        Also replaces the inner generator scan with searchsorted — same fix as
        the cell-boundary loop above. No behavioural change for nest_depth 0 or 1.
        """
        stack = [(pos_min_init, pos_max_init, 0, 1, row_init, col_init)]
        while stack:
            pos_min, pos_max, subcell_val, lvl, row, col = stack.pop()
            subcell_mult = 2**((nest_depth - lvl) * 2)
            # precompute quadrant index for every point in this slice once
            quad_nrs = (pts_subcell_nrs[pos_min:pos_max] - subcell_val) // subcell_mult
            cur_pos = pos_min
            for cur_quad_nr in range(4):
                if cur_pos >= pos_max:
                    break
                offset = cur_pos - pos_min
                if quad_nrs[offset] > cur_quad_nr:
                    continue
                # searchsorted to find end of this quadrant — O(log n) not O(n)
                rel = int(_np_searchsorted(quad_nrs[offset:], cur_quad_nr + 1, side='left'))
                pos_next = cur_pos + rel
                next_subcell_val = subcell_val + cur_quad_nr * subcell_mult
                row_c = row + cur_quad_nr // 2 / (2**lvl)
                col_c = col + cur_quad_nr %  2 / (2**lvl)
                rc = (row_c + 2**-(lvl+1), col_c + 2**-(lvl+1))
                _kc = _k(lvl, rc)
                nonlocal _node_count
                _sums_array[_node_count] = pts_vals[cur_pos:pos_next].sum(axis=0)
                id_to_sums_by_lvl[_kc]    = _node_count
                # id_to_vals_xy_by_lvl is only ever read at level 0 (_lvl0_packed/
                # cell_count_iter both hardcode codec.key(0,...)) -- populating it
                # for lvl>0 here is pure dead weight (same entry count as
                # id_to_sums_by_lvl, never read back). Level-0 entries are still
                # written below in the row/col loop.
                _node_count += 1
                if lvl + 1 <= nest_depth:
                    stack.append((cur_pos, pos_next, next_subcell_val, lvl+1, row_c, col_c))
                cur_pos = pos_next

    # loop over rows; use searchsorted to find column-group boundaries in O(log n)
    row_ends = row_id_indexes[1:] + [n_pts]
    for cur_row, cur_col_i, next_row_i in zip(row_ids, row_id_indexes, row_ends):
        cur_col_i = int(cur_col_i)
        next_row_i = int(next_row_i)
        row_cols = pts_cols[cur_col_i:next_row_i]   # sorted col ids for this row
        cur_col = int(row_cols[0])

        while True:
            # first position in this row where col > cur_col
            rel = int(_np_searchsorted(row_cols, cur_col, side='right'))
            next_col_i = cur_col_i + rel
            next_col   = int(row_cols[rel]) if rel < len(row_cols) else -1

            cell_vals  = pts_vals[cur_col_i:next_col_i]
            # cell_ids = pts_ids[cur_col_i:next_col_i]  # dead: fed id_to_pt_ids,
            # which was removed (see docstring note above) -- this slice was
            # computed every cell iteration and discarded, never stored.
            cell_sum   = cell_vals.sum(axis=0)

            # level-0 cell dicts (used by cell_count_iter and the search hot path)
            rc = (cur_row, cur_col)
            id_to_sums[rc] = cell_sum

            # level-0 entry in the by-lvl dicts
            _k0 = _k(0, rc)
            _sums_array[_node_count] = cell_sum
            id_to_sums_by_lvl[_k0]    = _node_count
            id_to_vals_xy_by_lvl[_k0] = (cur_col_i << 32) | next_col_i
            _node_count += 1

            if nest_depth > 0:
                nest_next_lvl(cur_col_i, next_col_i, cur_row - 0.5, cur_col - 0.5)

            if next_col == -1:
                break
            cur_col_i = next_col_i
            cur_col   = next_col
            row_cols  = row_cols[rel:]   # advance the view
        #
    #

    grid._search_internals.id_to_sums           = id_to_sums
    grid._search_internals.id_to_sums_by_lvl    = id_to_sums_by_lvl
    grid._search_internals.id_to_vals_xy_by_lvl = id_to_vals_xy_by_lvl
    grid.sums_array           = _sums_array[:_node_count]
    grid.pts_vals_xy          = pts_vals_xy
    grid.pts_ids              = pts_ids
    return
#

def _lvl0_packed(grid, row, col):
    """Return the packed int64 position for a level-0 cell, or None if empty."""
    codec = grid._search_internals.cell_codec
    k = codec.key(0, row, col) if codec is not None else (0, (row, col))
    return grid._search_internals.id_to_vals_xy_by_lvl.get(k)

def cell_count(grid, row, col):
    """Number of points in level-0 cell (row, col); 0 if empty."""
    pos = _lvl0_packed(grid, row, col)
    return (pos & 0xFFFFFFFF) - (pos >> 32) if pos is not None else 0

def cell_count_iter(grid):
    """Yield (row, col, count) for every non-empty level-0 cell."""
    codec = grid._search_internals.cell_codec
    vxy = grid._search_internals.id_to_vals_xy_by_lvl
    for rc in grid._search_internals.id_to_sums:
        k = codec.key(0, rc[0], rc[1]) if codec is not None else (0, rc)
        pos = vxy[k]
        yield rc[0], rc[1], (pos & 0xFFFFFFFF) - (pos >> 32)


################ aggregate_point_data_to_cells_adaptive_nd ###################################################################
def _estimate_local_max_depths(pts_rows, pts_cols, sr, K, r_rows):
    """Coarse KxK-block density scan -> per-block (max depth needed, level-0 needed).

    Mirrors disk_aggregation_chunk_adaptive_nd.py's ppc convention (ppc = fine-cell
    density * pi*sr^2, see nd_choice.get_exact_L's docstring) and reuses best_nd_tag
    for the nd decision, so this agrees with the same crossover thresholds the
    hot-loop planning uses -- it is an independent coarse estimate (not literally
    the same per-chunk boundaries search_and_aggregate's planning pass computes),
    but built from the same density signal.

    nd<0 (supercell) tags never read id_to_sums_by_lvl/sums_array at all -- see
    disk_aggregation_chunk_adaptive_nd.py's _process_super_cell_chunk, which builds
    its own index straight from raw pts_vals_xy via _ensure_super_cell_target_index,
    completely bypassing the quadtree dicts this function builds. So a coarse block
    whose OWN density says nd<0 doesn't need any id_to_sums_by_lvl entry there --
    not even level 0.

    BUT: an nd>=0 query chunk's contain/overlap template can reach across chunk
    boundaries (that's why chunking carries a row-margin at all) -- a dense nd=3
    region sitting next to a sparse block would have its contain lookups silently
    miss keys if the sparse neighbor was never built past a shallower depth (or
    skipped level 0 entirely). _contain_sum has no way to distinguish "genuinely
    empty cell" from "cell exists but wasn't built at this depth" -- both look
    like a missing key -- so an under-build here is a silent undercount, not a
    crash. To stay safe this dilates both the depth requirement AND the
    needs-level-0 flag outward by the same reach margin _super_block_radius uses
    (ceil(r/coarse_spacing), floored at 1 coarse block): any block within reach of
    an nd>=0 block inherits that block's requirements, so cross-boundary contain
    reads always land on something that was actually built.

    Returns:
        (depth_by_block: dict[(coarse_row,coarse_col), int] -- dilated max depth,
         needs_level0: dict[(coarse_row,coarse_col), bool] -- dilated; blocks
         absent from this dict (never occupied, never within reach of an
         occupied nd>=0 block) may safely skip level 0 too)
    """
    from .algorithm.nd_choice import best_nd_tag, best_nd_tag_weighted
    import aabpl.config as _cfg
    _use_weighted = getattr(_cfg, 'USE_WEIGHTED_ND_DECISION', False)
    if len(pts_rows) == 0:
        return {}, {}
    coarse_rows = (pts_rows // K).astype(np.int64) if hasattr(pts_rows, 'astype') else (pts_rows // K)
    coarse_cols = (pts_cols // K).astype(np.int64) if hasattr(pts_cols, 'astype') else (pts_cols // K)
    coarse_rows = _np_array(coarse_rows); coarse_cols = _np_array(coarse_cols)
    cr_min, cr_max = int(coarse_rows.min()), int(coarse_rows.max())
    cc_min, cc_max = int(coarse_cols.min()), int(coarse_cols.max())
    n_r, n_c = cr_max - cr_min + 1, cc_max - cc_min + 1
    block_key = (coarse_rows - cr_min).astype(np.int64) * n_c + (coarse_cols - cc_min)
    counts = _np_bincount(block_key, minlength=n_r * n_c).reshape(n_r, n_c)

    eff = _np_zeros((n_r, n_c), dtype=np.int64)      # max(0, nd) per block, 0 where unoccupied
    needs_lvl0 = _np_zeros((n_r, n_c), dtype=bool)   # nd >= 0 per block
    occ_ri, occ_ci = _np_where(counts > 0)
    for ri, ci in zip(occ_ri.tolist(), occ_ci.tolist()):
        cnt = int(counts[ri, ci])
        ppc = (cnt / (K * K)) * _math.pi * sr * sr
        if _use_weighted:
            # Mirrors the real (weighted) dispatch decision exactly, evaluated
            # on this single block: block_ppcs/block_counts is a length-1
            # array for this block, total_area_ppc reduces to the same ppc
            # (one fully-occupied block, no empty-block dilution), and
            # share_occupied=1.0 since the block itself is, by construction,
            # occupied (only occupied blocks reach this loop). Previously
            # this always used the plain (non-weighted) best_nd_tag even when
            # cfg.USE_WEIGHTED_ND_DECISION was on, silently disagreeing with
            # the real per-chunk dispatch near crossover thresholds and
            # causing aggregation to build cells too shallow -- confirmed via
            # brute-force validation (contain-path lookups then silently miss
            # keys that were never aggregated, a partial undercount).
            nd, _single_region = best_nd_tag_weighted(
                sr, [ppc], [cnt], ppc, share_occupied=1.0)
        else:
            nd, _single_region = best_nd_tag(sr, max(ppc, 1e-6))
        eff[ri, ci] = max(0, nd)
        needs_lvl0[ri, ci] = (nd >= 0)

    # dilate outward by reach margin, same convention as _super_block_radius:
    # max(1, ceil(r_rows / K)) coarse blocks in every direction.
    pad = max(1, int(_math.ceil(r_rows / K))) if K > 0 else 1
    eff_p = _np_zeros((n_r + 2*pad, n_c + 2*pad), dtype=np.int64)
    eff_p[pad:pad+n_r, pad:pad+n_c] = eff
    lvl0_p = _np_zeros((n_r + 2*pad, n_c + 2*pad), dtype=bool)
    lvl0_p[pad:pad+n_r, pad:pad+n_c] = needs_lvl0

    eff_dil = _np_zeros((n_r, n_c), dtype=np.int64)
    lvl0_dil = _np_zeros((n_r, n_c), dtype=bool)
    for dr in range(-pad, pad + 1):
        for dc in range(-pad, pad + 1):
            eff_dil = _np_maximum(eff_dil, eff_p[pad+dr:pad+dr+n_r, pad+dc:pad+dc+n_c])
            lvl0_dil |= lvl0_p[pad+dr:pad+dr+n_r, pad+dc:pad+dc+n_c]

    depth_by_block = {}
    needs_level0 = {}
    ri_idx, ci_idx = _np_where(lvl0_dil | (eff_dil > 0) | (counts > 0))
    for ri, ci in zip(ri_idx.tolist(), ci_idx.tolist()):
        key = (cr_min + ri, cc_min + ci)
        depth_by_block[key] = int(eff_dil[ri, ci])
        needs_level0[key] = bool(lvl0_dil[ri, ci])
    return depth_by_block, needs_level0


@time_func_perf
def aggregate_point_data_to_cells_adaptive_nd(
    grid:dict,
    pts:_pd_DataFrame,
    c:list=['employment'],
    y:str='lat',
    x:str='lon',
    row_name:str='id_y',
    col_name:str='id_x',
    nest_depth:int=5,
    sr:float=2.0,
    silent:bool=False,
    depth_by_block:dict=None,
    needs_level0:dict=None,
) -> _pd_DataFrame:
    """Same as aggregate_point_data_to_cells, but caps quadtree depth PER SPATIAL
    REGION instead of building every level-0 cell down to the grid's global native
    nest_depth unconditionally.

    Rationale: aggregate_point_data_to_cells previously measured at 40-60%+ of
    total radius_search runtime, and most of that is wasted whenever a region is
    sparse enough that best_nd_tag would pick a shallow nd (or a supercell, nd<0,
    which never reads past level 0 at all) for the chunks covering it -- see the
    session's finding that e.g. nd=-8 chunks paid for a full level-0 quadtree
    build that was never read.

    A coarse KxK-block density scan (_estimate_local_max_depths) decides, per
    block, the max depth actually needed there via the same best_nd_tag crossover
    model the hot loop itself uses. Each level-0 cell then only descends the
    quadtree to its block's local cap instead of the global nest_depth.

    K matches disk_aggregation_chunk_adaptive_nd.py's coarse-block convention
    (max(4, ceil(r/spacing))) so the density estimate is computed at roughly the
    same spatial granularity as the search's own coarse density map, though this
    is an independently-computed scan (see _estimate_local_max_depths docstring),
    not literally the same chunk boundaries as the hot-loop planning pass.

    Args: identical to aggregate_point_data_to_cells, plus:
        sr: spacing ratio (r / grid spacing) -- needed for the ppc estimate that
            drives the local depth decision.

    Caveat: id_to_sums (the plain, non-by_lvl per-cell dict) is skipped entirely
    for cells in a region with no nd>=0 chunk (dilated) within reach -- confirmed
    via grep that the search hot path (disk_aggregation_chunk_adaptive_nd.py)
    never reads it directly, only id_to_sums_by_lvl/sums_array. Other consumers
    of id_to_sums (cell_count_iter -- used by cluster/plot/export code, not
    search) will see those cells as absent rather than zero-count. Acceptable
    for this opt-in, search-focused mode; not a concern for the default (False)
    uniform path, which is unaffected.
    """
    aggregate_level = 0
    cols_for_sort = [row_name, col_name]
    subcell_nr = find_column_name('sc_nr', existing_columns=pts.columns)

    pts_rows_raw = pts[row_name].values
    pts_cols_raw = pts[col_name].values
    r_rows = int(_math.ceil(sr)) if sr > 0 else 1
    K = max(4, r_rows)
    if depth_by_block is None:
        # Fallback for callers that never ran chunk planning first (e.g. direct
        # use of this function outside search_and_aggregate's pipeline). When
        # planning HAS run (the normal search_and_aggregate(plan_only=True)
        # path), its real, final (post-merge) per-chunk nd decision is passed
        # in instead -- an independent re-estimate here was confirmed (this
        # session) to disagree with the real dispatch even using the identical
        # weighted formula, because it evaluates a single isolated K-block in
        # place of the real (possibly much larger, merged) chunk box the
        # dispatch actually decides over -- total_area_ppc/share_occupied
        # depend on that box, so the two are not interchangeable.
        depth_by_block, needs_level0 = _estimate_local_max_depths(pts_rows_raw, pts_cols_raw, sr, K, r_rows)
    _default_depth = nest_depth  # blocks with no points never get scanned; N/A in practice

    if nest_depth > 0:
        offset_x = 0.5 + ((pts[x]-grid._search_internals.bounds.xmin)%grid._search_internals.spacing - grid._search_internals.spacing/2) / grid._search_internals.spacing
        offset_y = 0.5 + ((pts[y]-grid._search_internals.bounds.ymin)%grid._search_internals.spacing - grid._search_internals.spacing/2) / grid._search_internals.spacing
        pts[subcell_nr] = 0
        for i in range(1,nest_depth+1):
            n = 2**(i)
            subcell_mult = 2**((nest_depth-i)*2)
            pts[subcell_nr] += (offset_x//(1/n)%n%2 * subcell_mult  + offset_y//(1/n)%n%2 * 2 * subcell_mult).astype(int)
        cols_for_sort.append(subcell_nr)

    # Sort via a local index array instead of pts.sort_values(inplace=True): pts
    # can be the SAME object as pts_source in a self-search (pts_target is
    # pts_source), and physically reordering it here was found (this session,
    # debugging an attempted search_and_aggregate single-call merge) to silently
    # desynchronize search_and_aggregate's own source-side arrays -- built
    # earlier in that function, from pts_source's original row order -- from
    # pts_source's actual row order by the time results get written back.
    # Confirmed via evidence: total sum unchanged (a pure permutation preserves
    # it) but individual points got another point's computed value. Sorting a
    # local index instead of the DataFrame itself leaves pts's row order
    # exactly as the caller provided it, no matter what object it aliases.
    _sort_idx = np.lexsort(tuple(pts[col].values for col in reversed(cols_for_sort)))

    pts_rows = pts[row_name].values[_sort_idx]
    pts_cols = pts[col_name].values[_sort_idx]
    pts_vals = pts[c].values[_sort_idx]
    pts_vals_xy = pts[c+[x,y]].values[_sort_idx]
    pts_ids = pts.index.values[_sort_idx]
    pts_subcell_nrs = pts[subcell_nr].values[_sort_idx] if nest_depth > 0 else _np_zeros(len(pts_cols))
    n_pts = len(pts)

    row_id_indexes = list(_np_where(_np_diff(pts_rows, prepend=pts_rows[0] - 1) != 0)[0])
    row_ids = [int(pts_rows[i]) for i in row_id_indexes]

    id_to_sums = {}
    id_to_sums_by_lvl = {}
    id_to_vals_xy_by_lvl = {}

    # Same worst-case upper bound as the uniform version -- per-region capping
    # only ever REDUCES node count relative to this, never exceeds it.
    _max_nodes = n_pts * (nest_depth + 1)
    _n_cols_arr = max(len(c), 1)
    _sums_array = _np_zeros((_max_nodes, _n_cols_arr), dtype=float)
    _node_count = 0

    _codec = getattr(grid._search_internals, 'cell_codec', None)
    if _codec is not None:
        def _k(lvl, rc):
            return int(_codec.key(lvl, rc[0], rc[1]))
    else:
        def _k(lvl, rc):
            return (lvl, rc)

    def nest_next_lvl(pos_min_init, pos_max_init, row_init, col_init, local_max_depth):
        if local_max_depth <= 0:
            return
        stack = [(pos_min_init, pos_max_init, 0, 1, row_init, col_init)]
        while stack:
            pos_min, pos_max, subcell_val, lvl, row, col = stack.pop()
            subcell_mult = 2**((nest_depth - lvl) * 2)
            quad_nrs = (pts_subcell_nrs[pos_min:pos_max] - subcell_val) // subcell_mult
            cur_pos = pos_min
            for cur_quad_nr in range(4):
                if cur_pos >= pos_max:
                    break
                offset = cur_pos - pos_min
                if quad_nrs[offset] > cur_quad_nr:
                    continue
                rel = int(_np_searchsorted(quad_nrs[offset:], cur_quad_nr + 1, side='left'))
                pos_next = cur_pos + rel
                next_subcell_val = subcell_val + cur_quad_nr * subcell_mult
                row_c = row + cur_quad_nr // 2 / (2**lvl)
                col_c = col + cur_quad_nr %  2 / (2**lvl)
                rc = (row_c + 2**-(lvl+1), col_c + 2**-(lvl+1))
                _kc = _k(lvl, rc)
                nonlocal _node_count
                _sums_array[_node_count] = pts_vals[cur_pos:pos_next].sum(axis=0)
                id_to_sums_by_lvl[_kc]    = _node_count
                # id_to_vals_xy_by_lvl is only ever read at level 0 -- see the
                # matching comment in aggregate_point_data_to_cells above.
                _node_count += 1
                # local_max_depth (not the global nest_depth) gates further descent
                if lvl + 1 <= local_max_depth:
                    stack.append((cur_pos, pos_next, next_subcell_val, lvl+1, row_c, col_c))
                cur_pos = pos_next

    row_ends = row_id_indexes[1:] + [n_pts]
    for cur_row, cur_col_i, next_row_i in zip(row_ids, row_id_indexes, row_ends):
        cur_col_i = int(cur_col_i)
        next_row_i = int(next_row_i)
        row_cols = pts_cols[cur_col_i:next_row_i]
        cur_col = int(row_cols[0])
        _coarse_row = cur_row // K

        while True:
            rel = int(_np_searchsorted(row_cols, cur_col, side='right'))
            next_col_i = cur_col_i + rel
            next_col   = int(row_cols[rel]) if rel < len(row_cols) else -1

            _block_key = (_coarse_row, cur_col // K)
            # Default True (safe) if a block is somehow missing from the scan --
            # should not happen (every occupied fine cell's coarse block is
            # scanned), but an unbuilt level 0 is a silent undercount downstream,
            # so the safe default on any uncertainty is "build it".
            if needs_level0.get(_block_key, True):
                cell_vals  = pts_vals[cur_col_i:next_col_i]
                cell_sum   = cell_vals.sum(axis=0)

                rc = (cur_row, cur_col)
                id_to_sums[rc] = cell_sum

                _k0 = _k(0, rc)
                _sums_array[_node_count] = cell_sum
                id_to_sums_by_lvl[_k0]    = _node_count
                id_to_vals_xy_by_lvl[_k0] = (cur_col_i << 32) | next_col_i
                _node_count += 1

                if nest_depth > 0:
                    _local_depth = min(nest_depth, depth_by_block.get(_block_key, _default_depth))
                    nest_next_lvl(cur_col_i, next_col_i, cur_row - 0.5, cur_col - 0.5, _local_depth)

            if next_col == -1:
                break
            cur_col_i = next_col_i
            cur_col   = next_col
            row_cols  = row_cols[rel:]
        #
    #

    grid._search_internals.id_to_sums           = id_to_sums
    grid._search_internals.id_to_sums_by_lvl    = id_to_sums_by_lvl
    grid._search_internals.id_to_vals_xy_by_lvl = id_to_vals_xy_by_lvl
    grid.sums_array           = _sums_array[:_node_count]
    grid.pts_vals_xy          = pts_vals_xy
    grid.pts_ids              = pts_ids
    return

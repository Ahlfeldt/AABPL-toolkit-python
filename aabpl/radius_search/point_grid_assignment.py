from numpy import (
    array as _np_array,
    zeros as _np_zeros,
    linspace as _np_linspace,
    diff as _np_diff,
    where as _np_where,
    searchsorted as _np_searchsorted,
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
                            points in the cell.  Used by ``clusters.py`` and
                            ``grid_class.py``.
        id_to_pt_ids      : ``{(row, col): ndarray}`` — point index array for the cell.
                            Used by ``null_distribution.py`` and ``plot_grid.py``.
        id_to_vals_xy     : ``{(row, col): ndarray}`` — stacked ``[c + [x, y]]`` array
                            for the cell.  Used by ``disk_aggregation.py`` to compute
                            candidate-count ceilings before the search loop.
        id_to_sums_by_lvl : ``{key: ndarray}`` — same as ``id_to_sums`` at level 0,
                            plus one entry per non-empty subcell at each deeper level.
                            Keys are codec-packed int64 when ``grid.cell_codec`` is set,
                            else ``(lvl, (row_centroid, col_centroid))`` tuples.
                            The main radius-search loop in ``disk_aggregation.py`` reads
                            this dict exclusively.
        id_to_pt_ids_by_lvl  : same structure, stores point-index arrays.
        id_to_vals_xy_by_lvl : same structure, stores stacked ``[c + [x, y]]`` arrays.
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
        from aabpl.radius_search.spacing_topology import count_cells_per_level, recommend_max_nest_depth
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
    
    pts.sort_values(cols_for_sort,inplace=True)
    
    # extract variables from dataframe for faster access speed
    pts_rows = pts[row_name].values
    pts_cols = pts[col_name].values
    pts_vals = pts[c].values
    pts_vals_xy = pts[c+[x,y]].values
    pts_ids = pts.index.values
    pts_subcell_nrs = pts[subcell_nr].values if nest_depth > 0 else _np_zeros(len(pts_cols))
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
                id_to_vals_xy_by_lvl[_kc] = (cur_pos << 32) | pos_next
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
            cell_ids   = pts_ids[cur_col_i:next_col_i]
            cell_sum   = cell_vals.sum(axis=0)

            # level-0 cell dicts (tuple keys, used by clusters/plots/exports)
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

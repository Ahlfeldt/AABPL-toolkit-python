from pandas import DataFrame as _pd_DataFrame
from numpy import (
    array as _np_array, column_stack as _np_column_stack, ndarray as _np_ndarray, vstack as _np_vstack,
    ones as _np_ones, empty as _np_empty, percentile as _np_percentile, bool_ as _np_bool,
    zeros as _np_zeros,
)
from numpy.random import ( random as _np_random,  randint as _np_randint, seed as _np_seed, )
from shapely.geometry import Polygon as _shapely_Polygon, Point as _shapely_Point
try:
    from shapely import contains_xy as _shapely_contains_xy
except ImportError:
    from shapely.vectorized import contains as _shapely_contains_xy
from aabpl.utils.misc import flatten_list
from aabpl.testing.test_performance import time_func_perf

@time_func_perf
def draw_random_points_in_sample_area(
    grid:dict,
    cell_width:float,
    n_random_points:int=int(1e5),
    sample_area:_shapely_Polygon=None,
    cells_rndm_sample:dict=None,
    random_seed:float=None,
    cell_height:float=None,
    extra_share_of_pts_to_create:float = 0.01,
    fix_extra_pts_to_create:int = 1000,
)->_np_array:
    """
    Draw n random points within non-excluded region
    if grid is provided it will first draw a grid cell that is not excluded 
    then it will choose a random point within that grid cell
    if the grid cell is partly excluded and the randomly generated point falls 
    into the excluded area the point is discarded and a new cell is drawn 

    Args:
    -------
    partly_or_fully_included_cells (??):
        list cells with attributes (centroid coords, excluded_property)
    cell_width (float):
        width of cells
    n_random_points (int):
        number of random points to be drawn (default=1e5)
    random_seed (int):
        seed to make random draws replicable. TODO not yet implemented.
    cell_height (float):
        height of cells. (default=None, cell_height will be set equal to cell_width)
    Returns:
    random_points_coordinates (array):
        vector of coordinates (x,y) of randomly drawn points within included area. shape=(n_random_points, 2)
    random_points_cell_ids (array):
        vector cell ids where random points fall into. TODO not yet implemented.  
    """
    if sample_area is None:
        sample_area = grid.sample_area

    # SET RANDOM SEED IF ANY SUPPLIED AND ASSERT TYPE
    if type(random_seed) == int:
        _np_seed(random_seed)
    elif random_seed is not None:
        raise TypeError(
            "random_seed should be int if supplied, otherwise None (of type NoneType)."+
            "\nSeed suplied is of type "+str(type(random_seed))+
            ". Seed suplied:\n", random_seed
        )
    #
    
    # IF NOT SPECIFIED OTHERWISE CELL HEIGHT EQUAL CELL WIDTH
    if cell_height is None:
        cell_height = cell_width
    #

    # ── Fast path: output-cell-level sampling ────────────────────────────────
    # When infer_sample_area_from_pts has pre-classified output cells (buff_cells
    # methods), sample uniformly within output cells.  Fully-valid cells need no
    # geometry check; partly-valid boundary cells do.
    _out_fully = getattr(grid, 'output_cells_fully_valid', None)
    _out_partly = getattr(grid, 'output_cells_partly_valid', None)
    if _out_fully is not None and _out_partly is not None:
        _sx = grid._output_sample_spacing_x
        _sy = grid._output_sample_spacing_y
        _ox = grid.x_steps_bounds[0]
        _oy = grid.y_steps_bounds[0]
        _fully_arr = _np_array(sorted(_out_fully), dtype=float)   # (row, col)
        _partly_arr = _np_array(sorted(_out_partly), dtype=float) if _out_partly else _np_array([]).reshape(0, 2)
        _n_fully  = len(_fully_arr)
        _n_partly = len(_partly_arr)
        _n_total  = _n_fully + _n_partly
        # Expected keep-rate for partly cells (estimated lazily from first batch).
        _partly_keep_rate = 0.5
        _result = _np_ndarray(shape=(0, 2))
        _pts_attempted = 0
        while _result.shape[0] < n_random_points:
            _need = n_random_points - _result.shape[0]
            # Allocate attempts proportionally; oversample partly cells.
            _w_fully  = _n_fully
            _w_partly = _n_partly / max(_partly_keep_rate, 0.01)
            _w_total  = _w_fully + _w_partly
            _n_full_draw  = int(_need * _w_fully  / _w_total) + fix_extra_pts_to_create + 1
            _n_part_draw  = int(_need * _w_partly / _w_total) + fix_extra_pts_to_create + 1
            coords_out = _np_ndarray(shape=(0, 2))
            if _n_fully:
                _idx = _np_randint(0, _n_fully, _n_full_draw)
                _rc  = _fully_arr[_idx]
                _pts = _np_array([_ox, _oy]) + _np_array([_sx, _sy]) * (
                    _np_random((_n_full_draw, 2)) + _rc[:, ::-1]
                )
                coords_out = _pts
            if _n_partly:
                _idx = _np_randint(0, _n_partly, _n_part_draw)
                _rc  = _partly_arr[_idx]
                _pts = _np_array([_ox, _oy]) + _np_array([_sx, _sy]) * (
                    _np_random((_n_part_draw, 2)) + _rc[:, ::-1]
                )
                _inside = _shapely_contains_xy(sample_area, _pts[:, 0], _pts[:, 1])
                _kept = _pts[_inside]
                if _n_part_draw > 0:
                    _partly_keep_rate = max(len(_kept) / _n_part_draw, 1e-3)
                coords_out = _np_vstack([coords_out, _kept]) if len(coords_out) and len(_kept) else (coords_out if len(coords_out) else _kept)
            if len(coords_out):
                _result = _np_vstack([_result, coords_out]) if _result.shape[0] else coords_out
            _pts_attempted += _n_full_draw + _n_part_draw
        return _result[:n_random_points]
    # ── End fast path ─────────────────────────────────────────────────────────

    # cells_fully_valid_ref = grid.cells_fully_valid_max_lvl
    cells_fully_valid_ref = grid._search_internals.cells_fully_valid
    cells_partly_valid_ref = grid._search_internals.cells_partly_valid_max_lvl
    # col_min = int((sample_area.bounds[0] - grid.total_bounds.xmin) // cell_width)
    # row_min = int((sample_area.bounds[1] - grid.total_bounds.ymin) // cell_height)
    # col_max = int((sample_area.bounds[2] - grid.total_bounds.xmin) // cell_width)
    # row_max = int((sample_area.bounds[3] - grid.total_bounds.ymin) // cell_height)
    # col_min = grid.sample_col_min
    # row_min = grid.sample_row_min
    # col_max = grid.sample_col_max
    # row_max = grid.sample_row_max
    # centroid_left_x = grid.total_bounds.xmin + grid._search_spacing / 2 
    # centroid_bottom_y = grid.total_bounds.ymin + grid._search_spacing / 2
    # centroid_left_x = grid.total_bounds.xmin
    # centroid_bottom_y = grid.total_bounds.ymin
    # grid.sample_grid_bounds = [
    #     grid.total_bounds.xmin + col_min * cell_width,
    #     grid.total_bounds.ymin + row_min * cell_height,
    #     grid.total_bounds.xmin + (col_max+1) * cell_width,
    #     grid.total_bounds.ymin + (row_max+1) * cell_height,
    # ]



    max_cells_fully_covered = max([
            sum([2**-(2*lvl) for lvl,(row, col) in cells_fully_valid_ref if lvl==lvl_i])
            for lvl_i in set([lvl for lvl, (row, col) in cells_fully_valid_ref])
        ])
    all_cells_eligible = sample_area is None or max_cells_fully_covered >= grid._search_internals.n_cells 
    

    # update cells_rndm_sample with grid cells outside the grid

    max_lvl_partly = max([lvl for lvl,(row,col) in (cells_partly_valid_ref if len(cells_partly_valid_ref)>0 else cells_fully_valid_ref)])
    sample_cells_arr = _np_array(sorted([
        (lvl,row,col) for lvl,(row,col) in cells_fully_valid_ref.union(
            [(lvl,(row,col)) for lvl,(row,col) in cells_partly_valid_ref if lvl==max_lvl_partly]
        )
        ]))
    
    min_lvl = int(min(sample_cells_arr[:,0]))
    max_lvl = int(max(sample_cells_arr[:,0]))
    cum_count_start_by_lvl, cum_int_start_by_lvl, cum_int_stop_by_lvl = {min_lvl:0}, {min_lvl:0}, {}
    
    if all_cells_eligible:
        # no need to potenitally keep multiple levels of sample cells as all drawn pts are valid.
        # Thus keep only one (arbirary) level.
        sample_cells_arr = sample_cells_arr
        def rand_int_transformer(rand_ints:_np_ndarray)->_np_ndarray:
            return rand_ints
        rand_int_stop = int(sum(2**(max_lvl-sample_cells_arr[:,0])))
    else:
        n_sample_cells_by_lvl = {}
        for lvl in range(min_lvl, max_lvl+1):
            n_sample_cells_by_lvl[lvl] = sum(sample_cells_arr[:,0]==lvl)
            if lvl > min_lvl:
                # cum_int_start_by_lvl[lvl] = cum_int_start_by_lvl[lvl-1] + n_sample_cells_by_lvl[lvl-1]*(2**(2*(max_lvl-lvl)))
                cum_int_start_by_lvl[lvl] = cum_int_stop_by_lvl[lvl-1]
                cum_count_start_by_lvl[lvl] = cum_count_start_by_lvl[lvl-1] + n_sample_cells_by_lvl[lvl-1]
            cum_int_stop_by_lvl[lvl] = cum_int_start_by_lvl[lvl] + n_sample_cells_by_lvl[lvl]*(2**(2*(max_lvl-lvl)))
        rand_int_stop = cum_int_stop_by_lvl[max_lvl]
  
    def rand_int_transformer(rand_ints:_np_ndarray)->_np_ndarray:
        """Transform random integers in [0, rand_int_stop) to cell indices in sample_cells  
        """
        if min_lvl == max_lvl:
            return rand_ints
        transformed_rand_ints = _np_empty(len(rand_ints), int)
        for lvl in range(min_lvl, max_lvl+1):
            mask = (rand_ints >= cum_int_start_by_lvl[lvl]) & (rand_ints < cum_int_stop_by_lvl[lvl])
            # transformed_rand_ints[mask] = cum_int_start_by_lvl[lvl] + ((rand_ints[mask]-cum_int_start_by_lvl[lvl])//(2**(2*(max_lvl-lvl))))
            transformed_rand_ints[mask] = cum_count_start_by_lvl[lvl] + ((rand_ints[mask]-cum_int_start_by_lvl[lvl])//(2**(2*(max_lvl-lvl))))
            # rand_ints[mask] = cum_count_start_by_lvl[lvl] + ((rand_ints[mask]-cum_int_start_by_lvl[lvl])//(2**(2*(max_lvl-lvl))))
        return transformed_rand_ints
    
    grid.rand_int_transformer = rand_int_transformer
    cell_to_poly = grid._search_internals.cell_to_poly if hasattr(grid._search_internals, 'cell_to_poly') else {}

    grid_bbox = _shapely_Polygon([
            (grid._search_internals.bounds.xmin,grid._search_internals.bounds.ymin),
            (grid._search_internals.bounds.xmax,grid._search_internals.bounds.ymin),
            (grid._search_internals.bounds.xmax,grid._search_internals.bounds.ymax),
            (grid._search_internals.bounds.xmin,grid._search_internals.bounds.ymax),
    ])
    
    sample_area_contains_grid = sample_area.contains(grid_bbox)
    # estimate the share of invalid area to draw additionally to create points (as some get discarded when they fall in invalid area)
    share_of_invalid_cells = .0 if all_cells_eligible else  sum(
        [2**(2*-lvl) for lvl, (row, col) in cells_partly_valid_ref]
        )/(
            sum([2**(2*-lvl) for lvl, (row, col) in cells_fully_valid_ref]) + 
            sum([2**(2*-lvl) for lvl, (row, col) in cells_partly_valid_ref])
        )
    share_of_invalid_geometry = sum(
        [cell_height * 2**(-lvl) * cell_width * 2**(-lvl) - cell_to_poly[lvl,(row,col)].area for lvl, (row, col) in cells_partly_valid_ref]
        ) / sample_area.area
    # make a guess upward biased guess how large the share of invalid random points may be. 
    share_of_invalid_area = 1 - (
        0.125 *  (1 - share_of_invalid_cells) * (1 - share_of_invalid_geometry) + 
        0.125 * (1 - share_of_invalid_cells) +
        0.75 * (1 - share_of_invalid_geometry)
    )
    # CREATE POINTS AND DISCARD POINTS UNTIL ENOUGH POINTS ARE DRWAN IN VALID AREA
    random_points_coordinates = _np_ndarray(shape=(0,2))
    pts_attempted_to_create = 0
    it = 0
    while random_points_coordinates.shape[0] < n_random_points:
        # update estimation of share of invalid area for iterations after first
        # TODO THIS MIGHT NOT BE NECESSARY ONCE PERCENTAGE OF INVALID AREA IS KNOWN
        if pts_attempted_to_create > 0:
            # otherwise update guess for iterations after first
            share_of_invalid_area = len(random_points_coordinates)/pts_attempted_to_create
        
        # set number of additional points to create
        n_rndm_points_to_create = int(
            (1+share_of_invalid_area+extra_share_of_pts_to_create*int(share_of_invalid_area>0)) * 
            (n_random_points-len(random_points_coordinates)) + 
            fix_extra_pts_to_create*(1+it)*int(share_of_invalid_area>0)
        )

        # larger cells (higher levels) shall have a higher chance to be drawn. 
        # Also it must be ensured that no subcell is part of sample_cells if parent cell is part of sample_cells
        # rndm_cells[:,:0:-1] gives col,row and leaves out level
        rand_ints = rand_int_transformer(_np_randint(0, rand_int_stop, n_rndm_points_to_create))
        rndm_cells = sample_cells_arr[rand_ints]
        # _np_array([sample_bounds_xmin, sample_bounds_ymin]) 
        new_random_point_coordinates = _np_array([grid._search_internals.bounds.xmin, grid._search_internals.bounds.ymin]) + grid._search_internals.spacing * (
            _np_random((n_rndm_points_to_create,2)) *
            (2**-rndm_cells[:,0].reshape(-1,1)) +
            rndm_cells[:,1:][:,::-1] # TODO this part might not put the points into the right position if lvl>0
        )
        
        # if anywhere is valid area        
        if sample_area_contains_grid:
            new_random_point_coordinates_in_sample_area = new_random_point_coordinates
            # 
        else: # filter out points in invalid area
            rndm_cells_fully_valid = _np_array([
                (int(lvl),(
                (int if lvl >= 0 else float)(row),
                (int if lvl >= 0 else float)(col)
                )) in cells_fully_valid_ref  for lvl,row,col in rndm_cells])

            # Points from fully-valid cells are guaranteed inside sample_area — no geometry check needed.
            pts_from_full_cells = new_random_point_coordinates[rndm_cells_fully_valid]

            # Points from partial cells need a geometry check; use vectorised contains for speed.
            pts_from_partial = new_random_point_coordinates[~rndm_cells_fully_valid]
            if len(pts_from_partial):
                in_area = _shapely_contains_xy(sample_area, pts_from_partial[:, 0], pts_from_partial[:, 1])
                pts_from_partial = pts_from_partial[in_area]

            new_random_point_coordinates_in_sample_area = (
                _np_vstack([pts_from_full_cells, pts_from_partial])
                if len(pts_from_full_cells) and len(pts_from_partial)
                else pts_from_full_cells if len(pts_from_full_cells)
                else pts_from_partial
            )
            grid.new_random_point_coordinates_partly_valid = new_random_point_coordinates[~rndm_cells_fully_valid]
            grid.rndm_cells_partly_valid = rndm_cells[~rndm_cells_fully_valid]
            
            #
        # save valid random points
        if len(new_random_point_coordinates_in_sample_area) > 0:
            random_points_coordinates = _np_vstack([random_points_coordinates, new_random_point_coordinates_in_sample_area])
        # update loop vars
        it += 1
        pts_attempted_to_create += n_rndm_points_to_create

    # return n_random_points coordinates
    return random_points_coordinates[:n_random_points]

@time_func_perf
def compute_null_distribution(
    grid:dict,
    pts:_pd_DataFrame,
    sample_area:_shapely_Polygon=None,
    min_pts_to_sample_cell:int=1,
    null_distribution=int(1e5),
    k_th_percentile:float=[99.5],
    c:list=[],
    x:str='lon',
    y:str='lat',
    row_name:str='id_y',
    col_name:str='id_x',
    suffix:str='_750m',
    random_seed:int=None,
    silent:bool=False,
    r=None,
    stat:str='sum',
):
    """Draws random points within sample_area and aggregates data within the search radius to
    build a null distribution. From those values it calculates the k_th_percentile threshold.

    k_th_percentile: in [0,100] k-th percentile

    min_pts_to_sample_cell (int):
        minimum number of points in dataset that need to be in cell s.t. random points are allowed to be drawn within it. (default=1)
    null_distribution (int | numpy.ndarray | pandas.DataFrame):
        Controls the null distribution used for cluster detection:

        - **int** (default ``100_000``): draw this many points uniformly at random within
          ``sample_area`` and compute their radius sums as the reference distribution.
        - **numpy.ndarray of shape (N, 2)**: use these coordinates directly — first column is
          **x**, second column is **y**, both in the projected CRS (metres). The internal
          random draw is skipped.
        - **pandas.DataFrame**: converted to a numpy array via ``.values``; must have at least
          two columns in **x, y order** (projected CRS). Column names are ignored.

        In all coordinate-array cases the radius sums are still computed by the package.
        Useful for non-uniform reference distributions (e.g. stratified, clustered, or
        empirically derived reference points).
    """
    if type(k_th_percentile) != list:
        k_th_percentiles = [k_th_percentile for i in range(len(c))]
    else: 
        k_th_percentiles = k_th_percentile
    if any([k_th_percentile >= 100 or k_th_percentile <= 0 for k_th_percentile in k_th_percentiles]):
        raise ValueError(
            'Values for k_th_percentile must be >0 and <100. Provided values do not fullfill that condition',
            set([k_th_percentile for k_th_percentile in k_th_percentiles if k_th_percentile >= 100 or k_th_percentile <= 0])
        )
    from aabpl.radius_search.point_grid_assignment import cell_count_iter as _cell_count_iter
    grid._search_internals.cells_rndm_sample = True if min_pts_to_sample_cell == 0 else set((row, col) for row, col, cnt in _cell_count_iter(grid) if cnt >= min_pts_to_sample_cell)
    grid.sample_area = sample_area

    if isinstance(null_distribution, int):
        # Draw N random points uniformly within sample_area.
        random_point_coords = draw_random_points_in_sample_area(
            grid=grid,
            cell_width=grid._search_internals.spacing,
            n_random_points=null_distribution,
            random_seed=random_seed,
            cell_height=grid._search_internals.spacing,
        )
        rndm_pts = _pd_DataFrame(data=random_point_coords, columns=[x, y])
    else:
        # User-supplied coordinates — array (N,2) or DataFrame (columns ignored, x-first).
        arr = _np_array(null_distribution.values if isinstance(null_distribution, _pd_DataFrame) else null_distribution)
        if arr.ndim != 2 or arr.shape[1] < 2:
            raise ValueError(
                "null_distribution array/DataFrame must have shape (N, 2) with x in the first "
                "column and y in the second column, both in the projected CRS."
            )
        rndm_pts = _pd_DataFrame(data=arr[:, :2], columns=[x, y])

    grid.null_distribution = rndm_pts

    cols_c = [c] if isinstance(c, str) else list(c)

    # ── multi-radius path ─────────────────────────────────────────────────────
    _mr_grids = getattr(grid, '_mr_grids', None)
    if r is not None and _mr_grids is not None:
        from aabpl.radius_search.multi_radius import _parse_r_spec, _fmt_r, _AGG_ABBR
        spec_type, data = _parse_r_spec(r)

        if spec_type != 'single':
            stat_str = _AGG_ABBR.get(stat, stat)
            _INT = '__mrnull__'

            if spec_type == 'list':
                unique_radii = sorted(set(data))
                bands = None
            else:
                bands = data
                unique_radii = sorted(set(rv for a, b, *_ in bands for rv in (a, b)))

            # aggregate rndm_pts at each unique radius using the pre-built grids
            for ri in unique_radii:
                int_sfx = f'{_INT}{_fmt_r(ri)}'
                if ri == 0.0:
                    # No real pts share random point coords → r=0 contribution is zero
                    for col in cols_c:
                        rndm_pts[f'{col}{int_sfx}'] = 0.0
                else:
                    grid_i = _mr_grids.get(ri)
                    if grid_i is None:
                        raise ValueError(
                            f"No grid for radius {ri} in grid._mr_grids. "
                            "Ensure multi_radius_search was called before compute_null_distribution."
                        )
                    _gi_sc = grid_i._search_class
                    _gi_sc.set_source(
                        pts=rndm_pts, c=cols_c, x=x, y=y,
                        row_name=row_name, col_name=col_name,
                        suffix=int_sfx, silent=True,
                    )
                    _gi_sc.set_target(
                        pts=pts, c=cols_c, x=x, y=y,
                        row_name=row_name, col_name=col_name, silent=silent,
                    )
                    _excl_orig = _gi_sc.exclude_self
                    _gi_sc.exclude_self = False
                    _gi_sc.perform_search(silent=True)
                    _gi_sc.exclude_self = _excl_orig

            # build output columns and thresholds
            thresholds: dict = {}
            band_cols_tmp: list = []

            if spec_type == 'list':
                for i, ri in enumerate(data):
                    for col in cols_c:
                        src = f'{col}{_INT}{_fmt_r(ri)}'
                        final = f'{col}_{stat_str}_r{i}'
                        rndm_pts[final] = rndm_pts[src].values
                        ki = k_th_percentiles[cols_c.index(col)]
                        thresholds[final] = _np_percentile(rndm_pts[final].values, ki, axis=0)
            else:
                for i, band in enumerate(bands):
                    r_in, r_out = band[0], band[1]
                    for col in cols_c:
                        outer = f'{col}{_INT}{_fmt_r(r_out)}'
                        inner = f'{col}{_INT}{_fmt_r(r_in)}'
                        bname = f'{col}_{stat_str}_b{i}'
                        rndm_pts[bname] = rndm_pts[outer].values - rndm_pts[inner].values
                        band_cols_tmp.append(bname)
                        if spec_type == 'bands':
                            ki = k_th_percentiles[cols_c.index(col)]
                            thresholds[bname] = _np_percentile(rndm_pts[bname].values, ki, axis=0)

                if spec_type == 'wbands':
                    total_w = sum(band[2] for band in bands)
                    for col in cols_c:
                        arr = _np_empty(len(rndm_pts))
                        arr[:] = 0.0
                        for i, band in enumerate(bands):
                            r_in, r_out, w = band
                            bname = f'{col}_{stat_str}_b{i}'
                            arr += (w / total_w) * rndm_pts[bname].values
                        wgt_col = f'{col}_{stat_str}_wgt'
                        rndm_pts[wgt_col] = arr
                        ki = k_th_percentiles[cols_c.index(col)]
                        thresholds[wgt_col] = _np_percentile(rndm_pts[wgt_col].values, ki, axis=0)

            # clean up temp __mrnull__ cols
            for tmp_col in list(rndm_pts.columns):
                if _INT in tmp_col:
                    rndm_pts.drop(columns=[tmp_col], inplace=True)

            return (thresholds, rndm_pts)

    # ── single-radius path (original) ────────────────────────────────────────
    _sc = grid._search_class
    _sc.set_source(
        pts=rndm_pts,
        c=c,
        x=x,
        y=y,
        row_name=row_name,
        col_name=col_name,
        suffix=suffix,
        silent=True,
    )

    _sc.set_target(
        pts=pts,
        c=c,
        x=x,
        y=y,
        row_name=row_name,
        col_name=col_name,
        silent=silent,
    )

    # Random points have no "self" to exclude — suppress exclude_self for this search.
    _exclude_self_orig = _sc.exclude_self
    _sc.exclude_self = False
    _sc.perform_search(silent=True,)
    _sc.exclude_self = _exclude_self_orig

    sum_radius_names = [(cname+suffix) for cname in c]
    disk_sums_for_random_points = rndm_pts[sum_radius_names].values

    thresholds = {name: _np_percentile(disk_sums_for_random_points[:,i], k_th_percentiles[i], axis=0)
                  for i, name in enumerate(sum_radius_names)}

    return (thresholds, rndm_pts)


def draw_random_coords(
    n_pts: int,
    sample_area=None,
    crs: str = None,
    proj_crs: str = 'auto',
    coord_generator=None,
    random_seed=None,
    x: str = 'x',
    y: str = 'y',
) -> _pd_DataFrame:
    """Draw ``n_pts`` random coordinate pairs, optionally constrained to a valid area.

    Use it to inspect, visualise, or pre-generate null-distribution coordinates
    before passing them to ``detect_cluster_pts`` / ``detect_cluster_cells``
    via the ``null_distribution`` argument.

    Parameters
    ----------
    n_pts : int
        Number of coordinate pairs to return.
    sample_area : Polygon, MultiPolygon, or list of (x, y) tuples, optional
        Valid area. Points that fall outside are rejected.
        A coordinate list is interpreted as a ``shapely.Polygon`` automatically.
        When ``None`` every coordinate produced by ``coord_generator`` is
        accepted — ``coord_generator`` is then required.
    crs : str, optional
        CRS of ``sample_area`` (e.g. ``'EPSG:4326'``). When provided the
        geometry is reprojected to ``proj_crs`` before sampling, using the
        same ``convert_MultiPolygon_crs`` utility used internally by
        ``detect_cluster_pts``. Leave ``None`` when ``sample_area`` is already
        in the target projected CRS (metres).
    proj_crs : str, default ``'auto'``
        Target projected CRS. ``'auto'`` picks the best UTM zone from the
        centroid of ``sample_area`` (requires ``sample_area`` to be supplied
        and ``crs`` to be set). Pass an explicit EPSG string (e.g.
        ``'EPSG:32618'``) to override. Ignored when ``crs`` is ``None``.
    coord_generator : callable, optional
        ``coord_generator(n, rng) -> array-like of shape (n, 2)``
        where ``n`` is the number of candidates to produce in one batch and
        ``rng`` is a ``numpy.random.Generator``. The function may return more
        or fewer than ``n`` rows; the loop keeps calling it until ``n_pts``
        valid points have been collected.

        When ``None`` (default) a uniform draw over the bounding box of
        ``sample_area`` is used — ``sample_area`` must be supplied in that
        case.
    random_seed : int, optional
        Seed for the internal ``numpy.random.Generator``.
    x, y : str
        Column names for the returned DataFrame (default ``'x'``, ``'y'``).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ``[x, y]`` and exactly ``n_pts`` rows,
        in the projected CRS (metres) when ``crs`` is supplied.

    Examples
    --------
    Uniform draw inside a polygon (already projected)::

        from shapely.geometry import Polygon
        import aabpl
        poly = Polygon([(0,0),(10000,0),(10000,10000),(0,10000)])
        pts = aabpl.draw_random_coords(5000, sample_area=poly, random_seed=42)

    From a coordinate list, auto-reprojected from WGS-84::

        coords = [(-74.05, 40.65), (-73.85, 40.65), (-73.85, 40.85), (-74.05, 40.85)]
        pts = aabpl.draw_random_coords(5000, sample_area=coords,
                                        crs='EPSG:4326', random_seed=42)

    Custom generator (e.g. clustered reference points)::

        import numpy as np
        def clustered(n, rng):
            centres = rng.uniform(0, 10000, (10, 2))
            idx = rng.integers(0, 10, n)
            return centres[idx] + rng.normal(0, 500, (n, 2))

        pts = aabpl.draw_random_coords(5000, sample_area=poly,
                                        coord_generator=clustered)
    """
    import numpy as _np
    from numpy.random import default_rng as _default_rng
    from shapely.geometry import Polygon as _Polygon, MultiPolygon as _MultiPolygon

    if sample_area is None and coord_generator is None:
        raise ValueError(
            "Provide at least one of 'sample_area' or 'coord_generator'. "
            "When 'sample_area' is None every produced coordinate is accepted, "
            "so 'coord_generator' must define the distribution."
        )

    # --- coerce coordinate list to Shapely Polygon ----------------------------
    if sample_area is not None and not isinstance(sample_area, (_Polygon, _MultiPolygon)):
        try:
            sample_area = _Polygon(sample_area)
        except Exception as e:
            raise ValueError(
                "sample_area must be a Shapely Polygon, MultiPolygon, or a list of "
                f"(x, y) coordinate tuples. Could not interpret as Polygon: {e}"
            )

    # --- CRS reprojection (reuses shared utility from crs_transformation) -----
    if crs is not None and sample_area is not None:
        from aabpl.utils.crs_transformation import convert_MultiPolygon_crs, convert_wgs_to_utm
        from pyproj import Transformer as _Transformer
        if proj_crs == 'auto':
            if crs != 'EPSG:4326':
                # convert centroid to WGS-84 first to pick the right UTM zone
                cx, cy = sample_area.centroid.coords[0]
                t = _Transformer.from_crs(crs, 'EPSG:4326', always_xy=True)
                cx_wgs, cy_wgs = t.transform(cx, cy)
            else:
                cx_wgs, cy_wgs = sample_area.centroid.coords[0]
            proj_crs = 'EPSG:' + str(convert_wgs_to_utm(cx_wgs, cy_wgs))
        if crs != proj_crs:
            sample_area = convert_MultiPolygon_crs(sample_area, initial_crs=crs, target_crs=proj_crs)
    elif crs is not None and sample_area is None:
        raise ValueError(
            "crs is set but sample_area is None — cannot determine target projection "
            "without a geometry anchor. Either supply sample_area or set proj_crs explicitly "
            "and reproject your coord_generator output yourself."
        )

    rng = _default_rng(random_seed)

    # --- default generator: uniform over bounding box of sample_area ----------
    if coord_generator is None:
        xmin, ymin, xmax, ymax = sample_area.bounds
        def coord_generator(n, rng):
            return _np.column_stack([
                rng.uniform(xmin, xmax, n),
                rng.uniform(ymin, ymax, n),
            ])

    # --- draw loop ------------------------------------------------------------
    collected = _np_ndarray(shape=(0, 2))
    batch = max(1000, int(n_pts * 1.2))

    while len(collected) < n_pts:
        candidates = _np_array(coord_generator(batch, rng))
        if candidates.ndim != 2 or candidates.shape[1] < 2:
            raise ValueError(
                "coord_generator must return an array-like of shape (n, 2). "
                f"Got shape {candidates.shape}."
            )
        coords = candidates[:, :2]

        if sample_area is not None:
            coords = _np_array([
                c for c in coords if sample_area.covers(_shapely_Point(c))
            ])

        if len(coords) > 0:
            collected = _np_vstack([collected, coords])

        # adapt batch size based on observed acceptance rate
        accepted = len(coords)
        if accepted > 0:
            rate = accepted / len(candidates)
            needed = n_pts - len(collected)
            batch = max(1000, int(needed / rate * 1.2))

    result = collected[:n_pts]
    return _pd_DataFrame(data=result, columns=[x, y])

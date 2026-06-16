from warnings import simplefilter
from pandas.errors import PerformanceWarning as _pd_PerformanceWarning
from pandas import DataFrame as _pd_DataFrame
from numpy import array as _np_array, nan as _np_nan
from shapely.geometry import Polygon as _shapely_Polygon, MultiPolygon as _shapely_MultiPolygon

simplefilter(action='ignore', category=_pd_PerformanceWarning)
simplefilter(action='ignore', category=FutureWarning)

from .radius_search.null_distribution import compute_null_distribution
from .radius_search.sample_area import infer_sample_area_from_pts, subtract_invalid_area, intersect_polygon_with_grid
from .testing.test_performance import time_func_perf
from .radius_search.disk_search_state import DiskSearch
from .radius_search.grid_class import Grid
from .illustrations.plot_pt_vars import create_plots_for_vars
from .illustrations.distribution_plot import create_distribution_plot
from .utils.misc import count_polygon_edges, find_column_name
from .utils.crs_transformation import convert_MultiPolygon_crs, convert_coords_to_local_crs, convert_pts_to_crs, convert_wgs_to_utm
from .utils.progress import _OUTER_PROGRESS, RadiusSearchProgress, DetectClusterProgress

def _validate_kwargs(
        pts:_pd_DataFrame,
        crs:str,
        sample_area_crs:str,
        r:float,
        c:list=[],
        agg:str='sum',
        x:str='lon',
        y:str='lat',
        row_name:str='id_y',
        col_name:str='id_x',
        sum_suffix:str=None,
        pts_target:_pd_DataFrame=None,
        x_tgt:str=None,
        y_tgt:str=None,
        row_name_tgt:str=None,
        col_name_tgt:str=None,
        trynew:int=0,
        proj_crs:str='auto',
        silent:bool=None,
):
    """
    check shared keyword arguments and apply defaults
    """
    # locals() TODO use locals to make this take in only locals
    if type(row_name) != str:
        raise TypeError('`row_name` must be of type str. Instead provided of type',type(row_name),row_name)
    if type(col_name) != str:
        raise TypeError('`col_name` must be of type str. Instead provided of type',type(col_name),col_name)
    if row_name_tgt is None:
        row_name_tgt = row_name
    elif type(row_name_tgt) != str:
        raise TypeError('`row_name_tgt` must be of type str. Instead provided of type',type(row_name_tgt),row_name_tgt)
    if col_name_tgt is None:
        col_name_tgt = col_name
    elif type(col_name_tgt) != str:
        raise TypeError('`col_name_tgt` must be of type str. Instead provided of type',type(col_name_tgt),col_name_tgt)
    if type(pts) != _pd_DataFrame:
        raise TypeError('`pts` must be a pandas.DataFrame or None. Instead provided of type',type(pts))
    if type(x) != str:
        raise TypeError('`x` must be of type str. Instead provided of type',type(x),x)
    if type(y) != str:
        raise TypeError('`x` must be of type str. Instead provided of type',type(y),y)
    if not x in pts.columns:
        raise ValueError('`x` (x-coord column name) must be in columns of pts')
    if not y in pts.columns:
        raise ValueError('`y` (y-coord column name) must be in columns of pts')
    if not type(sum_suffix) is str:
        if not sum_suffix is None:
            sum_suffix = str(sum_suffix)
        else:
            r_suffix = int(r) if r%1==0 or len(str(int(r))) > 5 else round(r,6-len(str(int(r))))
            sum_suffix = '_' + str(r_suffix)+'m'
    if x_tgt is None:
        x_tgt = x
    if y_tgt is None:
        y_tgt = y
    same_target = pts_target is None or pts is pts_target
    if pts_target is None:
        pts_target = pts
    else:
        if type(pts_target) != _pd_DataFrame:
            raise TypeError('`pts_target` must be a pandas.DataFrame or None. Instead provided of type',type(pts_target))
    if type(c) == str:
        c = [c]
    else:
        if c is None or len(c)==0:
            print("Warning: No columns specified for aggregation - will simply count number of points within radius.")
            agg = 'count'
        try:
            if any([type(column)!=str for column in c]):
                raise TypeError
        except:
            raise TypeError('`c` must be either a string of single column name or a list of column name strings')
    if any([not column in pts_target.columns for column in c]):
        raise ValueError('not all columns(',c,') are in columns of search target pts_target(',pts.columns,')')
    if not x_tgt in pts_target.columns:
        raise ValueError('`x_tgt` (x-coord column name) must be in columns of pts_target')
    if not y_tgt in pts_target.columns:
        raise ValueError('`y_tgt` (y-coord column name) must be in columns of pts_target')
    if sample_area_crs is None:
        sample_area_crs = crs
    if proj_crs == 'auto': 
        x_center = (min([pts[x].min(), pts_target[x_tgt].min()])+max([pts[x].max(), pts_target[x_tgt].max()]))/2
        y_center = (min([pts[y].min(), pts_target[y_tgt].min()])+max([pts[y].max(), pts_target[y_tgt].max()]))/2
        local_crs = 'EPSG:'+str(convert_wgs_to_utm(x_center, y_center))
    else:
        local_crs = proj_crs
    if crs != local_crs:
        x,y,local_crs = convert_pts_to_crs(pts=pts, x=x, y=y, initial_crs=crs, target_crs=proj_crs, silent=bool(silent))
        if not same_target:
            x_tgt,y_tgt,local_crs = convert_pts_to_crs(pts=pts_target, x=x_tgt, y=y_tgt, initial_crs=crs, target_crs=proj_crs, silent=bool(silent))
        else:
            x_tgt,y_tgt = x,y
    
    grid = build_grid(
        pts_source=pts,
        initial_crs=local_crs,
        local_crs=local_crs,
        data_crs=crs,
        r=r,
        x=x,
        y=y,
        pts_target=pts_target,
        x_tgt=x_tgt,
        y_tgt=y_tgt,
        silent=silent,
    )
    grid.trynew = trynew

    return (pts, local_crs,  sample_area_crs, c, x, y, sum_suffix, pts_target, x_tgt, y_tgt, row_name_tgt, col_name_tgt, grid, agg)
#


@time_func_perf
def resolve_sample_area(
    pts:_pd_DataFrame,
    r:float,
    sample_area='buffered_cells',
    sample_area_crs=None, 
    local_crs:str=None,
    x:str='lon',
    y:str='lat',
    grid:Grid=None,
    min_pts_to_sample_cell:int=0,
    no_plot:bool=True
):
    if type(sample_area)==bool and sample_area==False:
        return _shapely_Polygon([
            (grid.total_bounds.xmin, grid.total_bounds.ymin),
            (grid.total_bounds.xmax, grid.total_bounds.ymin),
            (grid.total_bounds.xmax, grid.total_bounds.ymax),
            (grid.total_bounds.xmin, grid.total_bounds.ymax)
        ])

    if sample_area is None:
        sample_area = 'grid'
    if type(sample_area) == str:
        if no_plot:
            print("Creating sample area with method '"+sample_area+"' and buffer=tolerance="+str(r)+". Use 'grid.sample_area' to inspect.")
        sample_area = infer_sample_area_from_pts(
            pts=pts,
            grid=grid,
            x=x,
            y=y,
            hull_type=sample_area,
            buffer=r,
            min_pts_to_sample_cell=min_pts_to_sample_cell,
            plot_sample_area=None,
        )
        # sample_area = subtract_invalid_area(sample_area, invalid_areas=_shapely_Polygon([]))

    elif type(sample_area) in [_shapely_Polygon, _shapely_MultiPolygon]:
        sample_area = convert_MultiPolygon_crs(multipoly=sample_area, initial_crs=sample_area_crs,target_crs=local_crs)
    else:
        raise ValueError('sample_area must parameter most be one of ["str","Poylgon","MultiPolygon"] instead of type', type(sample_area))
    
    
    
    return sample_area

# TODO remove cell_region from kwargs
@time_func_perf
def build_grid(
    pts_source:_pd_DataFrame,
    initial_crs:str,
    local_crs:str,
    data_crs:str,
    r:float,
    x:str='lon',
    y:str='lat',
    pts_target:_pd_DataFrame=None,
    x_tgt:str=None,
    y_tgt:str=None,
    silent:bool=None,
):
    """
    Returns a Grid that covers all points and will
    - can be used to represent clusters
    - and is leverage for performance gains of radius search

    Spacing and nest_depth are chosen automatically. To override, set
    config.FIXED_SPACING_RATIO and/or config.FIXED_NEST_DEPTH before calling.
    """
    if pts_target is None:
        xmin = pts_source[x].min()
        xmax = pts_source[x].max()
        ymin = pts_source[y].min()
        ymax = pts_source[y].max()
    else:
        if y_tgt is None:
            y_tgt = y
        if x_tgt is None:
            x_tgt = x
        xmin = min([pts_source[x].min(), pts_target[x_tgt].min()])
        xmax = max([pts_source[x].max(), pts_target[x_tgt].max()])
        ymin = min([pts_source[y].min(), pts_target[y_tgt].min()])
        ymax = max([pts_source[y].max(), pts_target[y_tgt].max()])

    _pts_tgt = pts_target if pts_target is not None else pts_source
    _x_tgt   = x_tgt if x_tgt is not None else x
    _y_tgt   = y_tgt if y_tgt is not None else y

    return Grid(
        xmin=xmin,
        xmax=xmax,
        ymin=ymin,
        ymax=ymax,
        initial_crs=initial_crs,
        local_crs=local_crs,
        r=r,
        n_pts_src=len(pts_source),
        n_pts_tgt=len(_pts_tgt),
        pts_tgt_xy=_pts_tgt[[_x_tgt, _y_tgt]].values,
        data_crs=data_crs,
        silent=silent,
    )
#

def radius_sum(pts, crs:str, r:float, c:list=[], **kwargs):
    """Aggregate neighbouring points within radius r by summing each column in c.
    Convenience wrapper for ``radius_search(..., agg='sum')``.
    All keyword arguments are forwarded to ``radius_search``.
    """
    return radius_search(pts=pts, crs=crs, r=r, c=c, agg='sum', **kwargs)

def radius_count(pts, crs:str, r:float, c:list=[], **kwargs):
    """Count neighbouring points within radius r for each column in c.
    Convenience wrapper for ``radius_search(..., agg='count')``.
    All keyword arguments are forwarded to ``radius_search``.
    """
    return radius_search(pts=pts, crs=crs, r=r, c=c, agg='count', **kwargs)

def radius_mean(pts, crs:str, r:float, c:list=[], **kwargs):
    """Compute the mean of neighbouring points within radius r for each column in c.
    Convenience wrapper for ``radius_search(..., agg='mean')``.
    All keyword arguments are forwarded to ``radius_search``.
    """
    return radius_search(pts=pts, crs=crs, r=r, c=c, agg='mean', **kwargs)


@time_func_perf
def radius_search(
    pts:_pd_DataFrame,
    crs:str,
    r:float,
    c:list=[],
    exclude_pt_itself:bool=True,
    weight_valid_area:str=None,
    sample_area=False,
    sample_area_crs:str=None,
    include_boundary:bool=False,
    agg:str=['sum','count','mean'][0],
    x:str='lon',
    y:str='lat',
    row_name:str='id_y',
    col_name:str='id_x',
    sum_suffix:str='_r_sum', 
    pts_target:_pd_DataFrame=None,
    x_tgt:str=None,
    y_tgt:str=None,
    row_name_tgt:str=None,
    col_name_tgt:str=None,
    trynew:int=0,
    proj_crs:str='auto',
    keep_cols:bool=False,
    _dev:dict=None,
    silent:bool=None,
):
    """
    Aggregates data from neighboring points within a search radius for every point in ``pts``.
    Results are appended in-place to ``pts`` as new column(s) named ``{c}{sum_suffix}``.
    The Grid object returned provides access to plots and internal search state.

    Args:
    -------
    pts (pandas.DataFrame):
        Points for which neighborhood aggregates are computed. Results are appended to this DataFrame in-place.
        Note: row order of ``pts`` may change after the call.
    crs (str):
        CRS of the coordinates in ``pts``, e.g. ``'EPSG:4326'``.
    r (float):
        Search radius in metres (after reprojection to a metric CRS).
    c (str or list):
        Column name or list of column names to aggregate within the search radius.
        If empty or None, points within the radius are counted.
        Columns must exist in ``pts`` (or in ``pts_target`` if provided).
    exclude_pt_itself (bool):
        If True, each point's own value is subtracted from its radius aggregate (default=True).
    weight_valid_area (str):
        Inverse-area weighting for edge effects. ``'estimate'`` uses a fast approximation
        (MSE ≈ 5 % of cell area); ``'precise'`` is exact but slow. ``None`` disables weighting (default=None).
    sample_area (shapely.Polygon | shapely.MultiPolygon | str):
        Area used for valid-area weighting. Accepted string values:
            - ``'buffered_cells'``: non-empty grid cells plus a radius-sized buffer (default)
            - ``'concave'``: concave hull around points
            - ``'convex'``: convex hull around points
            - ``'buffer'``: buffer around individual points (slow for large datasets)
            - ``'bounding_box'``: axis-aligned bounding box
            - ``'grid'`` or ``None``: full grid extent
        Alternatively pass a Shapely geometry directly (must already be in the metric projection).
        See ``infer_sample_area_from_pts`` for finer control (default=False).
    sample_area_crs (str):
        CRS of the ``sample_area`` polygon. Ignored when ``sample_area`` is a string.
        Defaults to ``crs`` when None (default=None).
    include_boundary (bool):
        NOT YET IMPLEMENTED. When implemented, points at distance exactly equal to ``r`` will be
        included. Currently the boundary is always excluded (strict ``distance < r``) (default=False).
    x (str):
        Column name of the x-coordinate (longitude) in ``pts`` (default=``'lon'``).
    y (str):
        Column name of the y-coordinate (latitude) in ``pts`` (default=``'lat'``).
    row_name (str):
        Name for the grid row-index column appended to ``pts`` (default=``'id_y'``).
    col_name (str):
        Name for the grid column-index column appended to ``pts`` (default=``'id_x'``).
    sum_suffix (str):
        Suffix appended to each column name in ``c`` to form the result column names.
        When None, defaults to ``'_{r}m'`` (e.g. ``'_750m'`` for ``r=750``) (default=None).
    pts_target (pandas.DataFrame):
        Points to aggregate over. If None, ``pts`` is used as both source and target (default=None).
    x_tgt (str):
        X-coordinate column in ``pts_target``. Defaults to ``x`` when None (default=None).
    y_tgt (str):
        Y-coordinate column in ``pts_target``. Defaults to ``y`` when None (default=None).
    row_name_tgt (str):
        Grid row-index column name for ``pts_target``. Defaults to ``row_name`` when None (default=None).
    col_name_tgt (str):
        Grid column-index column name for ``pts_target``. Defaults to ``col_name`` when None (default=None).
    trynew (int):
        Selects which overlapped-cell lookup table to use (0 or 1). Ignored when ``grid``
        is provided. 0 uses the full nested overlapped cell list; 1 uses the deduplicated
        distinct list, which is faster for some geometries (default=0).
    proj_crs (str):
        Metric CRS used internally. ``'auto'`` selects the appropriate UTM zone from the data extent.
        Pass an explicit EPSG string (e.g. ``'EPSG:32632'``) to override, or ``None`` to skip
        reprojection (default=``'auto'``).
    keep_cols (bool):
        If False, intermediate columns added during processing (grid indices, offsets, etc.)
        are removed from ``pts`` before returning. If True they are retained (default=False).
    _dev (dict):
        Development only. Dict of kwargs for internal debug plots. Supported keys:
        ``plot_pt_disk``, ``plot_cell_reg_assign``, ``plot_offset_checks``,
        ``plot_offset_regions``, ``plot_offset_raster``. None disables all (default=None).
    silent (bool):
        If True, suppresses all progress output. None is treated as False (default=None).

    Returns:
    -------
    grid (aabpl.Grid):
        The Grid object used for the search. Aggregated values are written directly into ``pts``
        as columns ``{c}{sum_suffix}``. Use ``grid.plot.vars()`` to visualise results.

    Examples:
    -------
    from aabpl.main import radius_search
    from pandas import read_csv
    pts = read_csv('C:/path/to/file.txt', sep=',', header=None)
    pts.columns = ["eid", "employment", "industry", "lat", "lon", "moved"]
    grid = radius_search(pts, crs="EPSG:4326", r=750, c=['employment'])
    grid.plot.vars(filename='employment_750m')
    """
    _cols_before = set(pts.columns)
    _orig_pts_target = pts_target  # capture before _validate_kwargs may set it to pts
    _cols_before_tgt = set(pts_target.columns) if pts_target is not None else None
    init_sort = find_column_name('initial_sort', existing_columns=pts.columns)
    pts[init_sort] = range(len(pts))

    (pts, local_crs, sample_area_crs, c, x, y, sum_suffix, pts_target, x_tgt, y_tgt, row_name_tgt, col_name_tgt, grid, agg
     ) = _validate_kwargs(
            pts=pts, crs=crs, sample_area_crs=sample_area_crs, r=r, c=c, x=x, y=y, row_name=row_name,
            col_name=col_name, sum_suffix=sum_suffix, pts_target=pts_target, x_tgt=x_tgt, y_tgt=y_tgt,
            row_name_tgt=row_name_tgt, col_name_tgt=col_name_tgt,
            trynew=trynew, proj_crs=proj_crs, silent=silent,
    )

    if agg in ['count','mean']:
        count_helper_col = find_column_name('count','_helper_col', existing_columns=pts_target.columns)
        c.append(count_helper_col)
        pts_target[count_helper_col] = 1

    _is_internal = _OUTER_PROGRESS.get() is not None
    if not _is_internal:
        _prog = RadiusSearchProgress(silent=bool(silent), n_pts=len(pts), n_tgt=len(pts_target))
        _token = _OUTER_PROGRESS.set(_prog)
        _prog.start()
        _prog.step("initializing")

    # initialize disk_search
    grid.search = DiskSearch(
        grid=grid,
        r=r,
        exclude_pt_itself=exclude_pt_itself,
        weight_valid_area=weight_valid_area,
        include_boundary=include_boundary
    )

    if not _is_internal:
        _prog.step("assigning target")
    # prepare target points data
    grid.search.set_target(
        pts=pts_target,
        c=c,
        x=x_tgt,
        y=y_tgt,
        row_name=row_name_tgt,
        col_name=col_name_tgt,
        silent=silent,
    )

    if not _is_internal:
        _prog.step("assigning source")
    # prepare source points data
    _d = _dev or {}
    grid.search.set_source(
        pts=pts,
        c=c,
        x=x,
        y=y,
        row_name=row_name,
        col_name=col_name,
        sum_suffix=sum_suffix,
        plot_cell_reg_assign=_d.get('plot_cell_reg_assign'),
        plot_offset_checks=_d.get('plot_offset_checks'),
        plot_offset_regions=_d.get('plot_offset_regions'),
        plot_offset_raster=_d.get('plot_offset_raster'),
        silent=silent,
    )

    # in case sums shall be weighted by sample area
    grid.sample_area = resolve_sample_area(
        pts=pts,r=r,sample_area=sample_area,
        sample_area_crs=sample_area_crs,local_crs=local_crs,x=x,y=y,
        grid=grid, min_pts_to_sample_cell=0)
    intersect_polygon_with_grid(grid)

    if not _is_internal:
        _prog.step("searching")
    disk_sums_for_pts = grid.search.perform_search(silent=False if silent is None else silent,plot_pt_disk=_d.get('plot_pt_disk'))

    if agg in ['mean']:
        n_rs = 1 # TODO for distance bands... later
        radius_count_cols = disk_sums_for_pts.columns[-n_rs:]
        radius_count_col = radius_count_cols[0] # TODO for distance bands... later
        for s_name in disk_sums_for_pts.columns[:-n_rs]:
            if s_name not in radius_count_cols:
                pts[s_name][pts[radius_count_col]>0] = pts[s_name][pts[radius_count_col]>0] / pts[radius_count_col][pts[radius_count_col]>0]
                pts[s_name][pts[radius_count_col]==0] = _np_nan
        pts.drop(columns=[count_helper_col], inplace=True)

    # Assign output grid cell indices — needed by clustering when output_spacing
    # differs from internal spacing. No-op (alias only) when spacings match.
    grid.assign_output_cell_ids(pts, x=x, y=y, row_name=row_name, col_name=col_name)

    _suffixes = [sum_suffix] if isinstance(sum_suffix, str) else list(sum_suffix)
    _result_cols = [c for c in pts.columns if c not in _cols_before and any(c.endswith(s) for s in _suffixes)]
    if _result_cols:
        grid.aggregate_pts_to_output_cells(pts, val_cols=_result_cols, x=x, y=y, agg='sum')

    pts.sort_values(init_sort, inplace=True)
    pts.drop(columns=[init_sort], inplace=True)

    if not keep_cols:
        _keep_extra = {grid.output_row_name, grid.output_col_name}
        _to_drop = [
            c for c in pts.columns
            if c not in _cols_before and c not in _keep_extra and not any(c.endswith(s) for s in _suffixes)
        ]
        if _to_drop:
            pts.drop(columns=_to_drop, inplace=True)

    # Clean up temporary projection columns from pts_target (e.g. proj_x/proj_y added
    # by _validate_kwargs). pts_target is often the caller's original DataFrame and is
    # NOT copied, so without this cleanup columns accumulate across repeated calls.
    if _orig_pts_target is not None and _orig_pts_target is not pts and _cols_before_tgt is not None:
        _tgt_to_drop = [c for c in _orig_pts_target.columns if c not in _cols_before_tgt]
        if _tgt_to_drop:
            _orig_pts_target.drop(columns=_tgt_to_drop, inplace=True)

    if not _is_internal:
        _OUTER_PROGRESS.reset(_token)
        _prog.done()
    return grid
#

@time_func_perf
def detect_cluster_pts(
    pts:_pd_DataFrame,
    crs:str,
    r:float,
    c:list=[],
    agg:str=['sum','count','mean'][0],
    exclude_pt_itself:bool=True,
    sample_area='buffered_cells',
    sample_area_crs:str=None,
    min_pts_to_sample_cell:int=0,
    weight_valid_area:str=None,
    k_th_percentile:float=99.5,
    n_random_points:int=int(1e5),
    random_seed:int=None,
    include_boundary:bool=False,
    x:str='lon',
    y:str='lat',
    row_name:str='id_y',
    col_name:str='id_x',
    sum_suffix:str='_750m',
    cluster_suffix:str='_cluster',
    proj_crs:str='auto',
    pts_target:_pd_DataFrame=None,
    x_tgt:str=None,
    y_tgt:str=None,
    row_name_tgt:str=None,
    col_name_tgt:str=None,
    plot_distribution:dict=None,
    plot_cluster_points:dict=None,
    _dev:dict=None,
    silent:bool=None,
):
    """
    For all points in a DataFrame it searches for all other points (potentially of another DataFrame) within the specified radius and aggregate the values for specified column(s).
    It draws random the bounding box containing all points from DataFrame(s) and aggregate the values within the radius to obtain a random distribution.
    Then all points from DataFrame which exceed the k_th_percentile of the random distribution are labeld as clustered.
    The results will be appended to DataFrame.
    """
    init_sort = find_column_name('initial_sort', existing_columns=pts.columns)
    pts[init_sort] = range(len(pts))

    (pts, local_crs, sample_area_crs, c, x, y, sum_suffix, pts_target, x_tgt, y_tgt, row_name_tgt, col_name_tgt, grid, agg
     ) = _validate_kwargs(
            pts=pts, crs=crs, sample_area_crs=sample_area_crs, r=r, c=c, agg=agg,
            x=x, y=y, row_name=row_name, col_name=col_name, sum_suffix=sum_suffix,
            pts_target=pts_target, x_tgt=x_tgt, y_tgt=y_tgt, row_name_tgt=row_name_tgt, col_name_tgt=col_name_tgt,
            proj_crs=proj_crs, silent=silent,
    )
    if type(k_th_percentile) not in [list,_np_array, tuple]:
        k_th_percentile = [k_th_percentile for column in c]
    elif len(k_th_percentile) < len(c):
        k_th_percentile = [k_th_percentile[i%len(k_th_percentile)] for i in range(len(c))]

    _prog = DetectClusterProgress(silent=bool(silent), n_pts=len(pts))
    _token = _OUTER_PROGRESS.set(_prog)
    _prog.start()
    _prog.step("initializing")

    # initialize disk_search
    grid.search = DiskSearch(
        grid,
        r=r,
        exclude_pt_itself=exclude_pt_itself,
        weight_valid_area=weight_valid_area,
        include_boundary=include_boundary
    )

    _prog.step("assigning target")
    grid.search.set_target(
        pts=pts_target,
        c=c,
        x=x_tgt,
        y=y_tgt,
        row_name=row_name_tgt,
        col_name=col_name_tgt,
        silent=silent,
    )

    grid.sample_area = resolve_sample_area(
        pts=pts, r=r,
        sample_area=sample_area, sample_area_crs=sample_area_crs,local_crs=local_crs,x=x, y=y, grid=grid,
        min_pts_to_sample_cell=min_pts_to_sample_cell,
        no_plot=plot_distribution is None and plot_cluster_points is None)
    intersect_polygon_with_grid(grid=grid)

    _prog.step("null distribution")
    (cluster_threshold_values, rndm_pts) = compute_null_distribution(
        grid=grid,
        pts=pts,
        sample_area=grid.sample_area,
        min_pts_to_sample_cell=min_pts_to_sample_cell,
        c=c,
        x=x,
        y=y,
        row_name=row_name,
        col_name=col_name,
        sum_suffix=sum_suffix,
        n_random_points=n_random_points,
        k_th_percentile=k_th_percentile,
        random_seed=random_seed,
        silent=silent,
    )

    if not silent:
        for (colname, threshold_value, k_th_p) in zip(c, cluster_threshold_values,k_th_percentile):
            print("Threshold value for "+str(k_th_p)+"th-percentile is "+str(threshold_value)+" for "+str(colname)+" within "+str(r)+" meters.")

    _prog.step("assigning source")
    _d = _dev or {}
    grid.search.set_source(
        pts=pts,
        c=c,
        x=x,
        y=y,
        row_name=row_name,
        col_name=col_name,
        sum_suffix=sum_suffix,
        plot_cell_reg_assign=_d.get('plot_cell_reg_assign'),
        plot_offset_checks=_d.get('plot_offset_checks'),
        plot_offset_regions=_d.get('plot_offset_regions'),
        plot_offset_raster=_d.get('plot_offset_raster'),
        silent=silent,
    )

    _prog.step("searching")
    disk_sums_for_pts = grid.search.perform_search(silent=silent,plot_pt_disk=_d.get('plot_pt_disk'))

    _prog.step("labeling clusters")
    for j, cname in enumerate(c):
        pts[str(cname)+str(cluster_suffix)] = disk_sums_for_pts.values[:,j]>cluster_threshold_values[j]

    if plot_distribution is not None:
        create_distribution_plot(
            pts=pts,
            x=x,
            y=y,
            radius_sum_columns=[n+sum_suffix for n in c],
            grid=grid,
            rndm_pts=rndm_pts,
            cluster_threshold_values=cluster_threshold_values,
            k_th_percentile=k_th_percentile,
            r=r,
            plot_kwargs=plot_distribution
            )

    def plot_rand_dist(
            filename:str="",
            pts=pts,
            x=x,
            y=y,
            radius_sum_columns=[n+sum_suffix for n in c],
            rndm_pts=rndm_pts,
            cluster_threshold_values=cluster_threshold_values,
            k_th_percentile=k_th_percentile,
            r=r,
            grid=grid,
            **plot_kwargs
    ):
        create_distribution_plot(
            filename=filename,
            plot_kwargs=plot_kwargs,
            pts=pts,
            x=x,
            y=y,
            radius_sum_columns=radius_sum_columns,
            grid=grid,
            rndm_pts=rndm_pts,
            cluster_threshold_values=cluster_threshold_values,
            k_th_percentile=k_th_percentile,
            r=r,
            )
    grid.plot.rand_dist = plot_rand_dist

    plot_colnames = list(c) + [n+sum_suffix for n in c] + [str(cname)+str(cluster_suffix) for cname in c]
    def plot_cluster_pts(
            self=grid,
            colnames=_np_array(plot_colnames),
            filename:str="",
            **plot_kwargs,
    ):
        return create_plots_for_vars(
            grid=self,
            colnames=colnames,
            filename=filename,
            plot_kwargs=plot_kwargs,
        )
    grid.plot.cluster_pts = plot_cluster_pts

    if plot_cluster_points is not None:
        grid.plot.cluster_pts(**plot_cluster_points)
    pts.sort_values(init_sort, inplace=True)
    pts.drop(columns=[init_sort], inplace=True)

    _OUTER_PROGRESS.reset(_token)
    _prog.done()
    return grid
# done

def detect_cluster_cells(
    pts:_pd_DataFrame,
    crs:str,
    r:float,
    c:list=[],
    agg:str=['sum','count','mean'][0],
    exclude_pt_itself:bool=True,
    sample_area='buffered_cells',
    sample_area_crs:str=None,
    min_pts_to_sample_cell:int=0,
    weight_valid_area:str=None,
    k_th_percentile:float=99.5,
    n_random_points:int=int(1e5),
    random_seed:int=None,
    queen_contingency:int=1,
    rook_contingency:int=1,
    centroid_dist_threshold:float=None,
    border_dist_threshold:float=None,
    min_cluster_share_after_contingency:float=0.05,
    min_cluster_share_after_centroid_dist:float=0.00,
    min_cluster_share_after_convex:float=0.00,
    make_convex:bool=True,
    include_boundary:bool=False,
    x:str='lon',
    y:str='lat',
    row_name:str='id_y',
    col_name:str='id_x',
    sum_suffix:str='_750m',
    cluster_suffix:str='_cluster',
    proj_crs:str='auto',
    pts_target:_pd_DataFrame=None,
    x_tgt:str=None,
    y_tgt:str=None,
    row_name_tgt:str=None,
    col_name_tgt:str=None,
    plot_distribution:dict=None,
    plot_cluster_points:dict=None,
    _dev:dict=None,
    silent:bool=None,
):
    """
    For all points in a DataFrame it searches for all other points (potentially of another DataFrame) within the specified radius and aggregate the values for specified column(s).
    It draws random the bounding box containing all points from DataFrame(s) and aggregate the values within the radius to obtain a random distribution.
    Then all points from DataFrame which exceed the k_th_percentile of the random distribution are labeld as clustered.
    The results will be appended to DataFrame.

    Args:
    -------
    pts (pandas.DataFrame):
        DataFrame of points for which a search for other points within the specified radius shall be performed
    crs (str):
        crs of coordinates, e.g. 'EPSG:4326'
    r (float):
        radius within which other points shall be found in meters
    c (str or list):
        column name or list of column name(s) in DataFrame for which data within search radius shall be aggregated. If None provided it will simply count the points within the radius. Column name must be in pts(DataFrame) unless a different search target is specified - then columns must exist in pts_target.
    exclude_pt_itself (bool):
        whether the sums within search radius point shall exlclude the point data itself (default=True)
    weight_valid_area (str):
        if set to 'estimate' or 'precise' the radius aggregate will be weighted inversely by the share of area of valid cells within search radius. 'precise' is very slow, 'estimate' has MSE of 5% of cell area. (default=None)
    sample_area (shapely.Polygon | shapely.MultiPolygon | str):
        Area used for drawing random comparison points. Accepted string values:
            - ``'buffered_cells'``: non-empty grid cells plus a radius-sized buffer (default)
            - ``'concave'``: concave hull around points
            - ``'convex'``: convex hull around points
            - ``'buffer'``: buffer around individual points (slow for large datasets)
            - ``'bounding_box'``: axis-aligned bounding box
            - ``'grid'`` or ``None``: full grid extent
        Alternatively pass a Shapely geometry directly (must already be in the metric projection).
        See ``infer_sample_area_from_pts`` for finer control (default='buffered_cells').
    sample_area_crs (str):
        CRS of the ``sample_area`` polygon. Ignored when ``sample_area`` is a string.
        Defaults to ``crs`` when None (default=None).
    min_pts_to_sample_cell (int):
        Minimum number of data points a grid cell must contain for random points to be drawn in it (default=0).
    k_th_percentile (float):
        Percentile of the random distribution a point must exceed to be labelled as clustered (default=99.5).
    n_random_points (int):
        Number of random points drawn to build the comparison distribution (default=100000).
    random_seed (int):
        Random seed for reproducibility. None means no seed is set (default=None).
    queen_contingency (int):
        Merge neighbouring clustered cells (including diagonals) into the same cluster.
        Values ≥ 2 also pull in cells that many steps away (default=1).
    rook_contingency (int):
        Merge horizontally/vertically neighbouring clustered cells into the same cluster.
        Ignored when ``queen_contingency`` is set higher. Values ≥ 2 extend the reach (default=1).
    centroid_dist_threshold (float):
        Maximum centroid-to-centroid distance for merging two clusters. None disables centroid merging (default=r*10/3).
    border_dist_threshold (float):
        Maximum border-to-border distance for merging two clusters. None disables border merging (default=r*4/3).
    min_cluster_share_after_contingency (float):
        Minimum share of total clustered points a cluster must represent after contingency merging to be retained (default=0.05).
    min_cluster_share_after_centroid_dist (float):
        Minimum share after centroid-distance merging (default=0.00).
    min_cluster_share_after_convex (float):
        Minimum share after convex-hull infill (default=0.00).
    make_convex (bool):
        If True, all grid cells within the convex hull of each cluster are added to it (default=True).
    include_boundary (bool):
        NOT YET IMPLEMENTED. When implemented, points at distance exactly equal to ``r`` will be
        included. Currently the boundary is always excluded (strict ``distance < r``) (default=False).
    x (str):
        Column name of the x-coordinate (longitude) in ``pts`` (default=``'lon'``).
    y (str):
        Column name of the y-coordinate (latitude) in ``pts`` (default=``'lat'``).
    row_name (str):
        Name for the grid row-index column appended to ``pts`` (default=``'id_y'``).
    col_name (str):
        Name for the grid column-index column appended to ``pts`` (default=``'id_x'``).
    sum_suffix (str):
        Suffix appended to each column name in ``c`` to form the radius-aggregate column names.
        When None, defaults to ``'_{r}m'`` (e.g. ``'_750m'`` for ``r=750``) (default='_750m').
    pts_target (pandas.DataFrame):
        Points to aggregate over. If None, ``pts`` is used as both source and target (default=None).
    x_tgt (str):
        X-coordinate column in ``pts_target``. Defaults to ``x`` when None (default=None).
    y_tgt (str):
        Y-coordinate column in ``pts_target``. Defaults to ``y`` when None (default=None).
    row_name_tgt (str):
        Grid row-index column name for ``pts_target``. Defaults to ``row_name`` when None (default=None).
    col_name_tgt (str):
        Grid column-index column name for ``pts_target``. Defaults to ``col_name`` when None (default=None).
    plot_distribution (dict):
        Kwargs for the random-distribution plot. None disables it (default=None).
    _dev (dict):
        Development only. Dict of kwargs for internal debug plots. Supported keys:
        ``plot_pt_disk``, ``plot_cell_reg_assign``, ``plot_offset_checks``,
        ``plot_offset_regions``, ``plot_offset_raster``. None disables all (default=None).
    silent (bool):
        If True, suppresses all progress output. None is treated as False (default=None).

    Returns:
    -------
    grid (aabpl.Grid):
        The Grid object used for the search, with spatial cluster polygons stored at
        ``grid.clustering``. Boolean cluster columns ``{c}{cluster_suffix}`` and radius
        aggregates ``{c}{sum_suffix}`` are appended to ``pts``.
    """
    (pts, local_crs, sample_area_crs, c, x, y, sum_suffix, pts_target, x_tgt, y_tgt, row_name_tgt, col_name_tgt, grid, agg
     ) = _validate_kwargs(
            pts=pts, crs=crs, sample_area_crs=sample_area_crs, r=r, c=c, agg=agg,
            x=x, y=y, row_name=row_name, col_name=col_name, sum_suffix=sum_suffix, pts_target=pts_target, x_tgt=x_tgt, y_tgt=y_tgt,
            row_name_tgt=row_name_tgt, col_name_tgt=col_name_tgt, grid=grid,
            proj_crs=proj_crs, silent=silent,
    )
    if centroid_dist_threshold is None:
        centroid_dist_threshold = r * 10/3
    if border_dist_threshold is None:
        border_dist_threshold = r * 4/3
    
    grid = detect_cluster_pts(
        pts=pts,
        crs=local_crs,
        r=r,
        c=c,
        exclude_pt_itself=exclude_pt_itself,
        weight_valid_area=weight_valid_area,
        sample_area=sample_area,
        sample_area_crs=sample_area_crs,
        min_pts_to_sample_cell=min_pts_to_sample_cell,
        k_th_percentile=k_th_percentile,
        n_random_points=n_random_points,
        random_seed=random_seed,
        grid=grid,
        x=x,
        y=y,
        row_name=row_name,
        col_name=col_name,
        sum_suffix=sum_suffix,
        cluster_suffix=cluster_suffix,
        proj_crs=local_crs,
        pts_target=pts_target,
        x_tgt=x_tgt,
        y_tgt=y_tgt,
        row_name_tgt=row_name_tgt,
        col_name_tgt=col_name_tgt,
        include_boundary=include_boundary,
        plot_distribution=plot_distribution,
        plot_cluster_points=plot_cluster_points,
        _dev=_dev,
        silent=silent,
    )
    
    grid.clustering.create_clusters(
        pts=pts,
        c=c,
        queen_contingency=queen_contingency,
        rook_contingency=rook_contingency,
        centroid_dist_threshold=centroid_dist_threshold,
        border_dist_threshold=border_dist_threshold,
        min_cluster_share_after_contingency=min_cluster_share_after_contingency,
        min_cluster_share_after_centroid_dist=min_cluster_share_after_centroid_dist,
        min_cluster_share_after_convex=min_cluster_share_after_convex,
        make_convex=make_convex,
        row_name=grid.output_row_name,
        col_name=grid.output_col_name,
        cluster_suffix=cluster_suffix,
        )
    
    return grid
#
def detect_cluster_cells_from_labeled_pts():
    pass

from functools import wraps as _wraps
from warnings import simplefilter
from pandas.errors import PerformanceWarning as _pd_PerformanceWarning
from pandas import DataFrame as _pd_DataFrame
from numpy import array as _np_array, nan as _np_nan
from shapely.geometry import Polygon as _shapely_Polygon, MultiPolygon as _shapely_MultiPolygon, GeometryCollection as _shapely_GeometryCollection

simplefilter(action='ignore', category=_pd_PerformanceWarning)
simplefilter(action='ignore', category=FutureWarning)

from .radius_search.sample_area import infer_sample_area_from_pts, subtract_invalid_area, intersect_polygon_with_grid
from .testing.test_performance import time_func_perf
from .radius_search.disk_search_state import DiskSearch
from .radius_search.grid_class import Grid
from .utils.misc import count_polygon_edges, find_column_name
from .utils.crs_transformation import convert_MultiPolygon_crs, convert_coords_to_local_crs, convert_pts_to_crs, convert_wgs_to_utm
from .utils.progress import _OUTER_PROGRESS, RadiusSearchProgress, DetectClusterProgress, progress_print
from .utils.param_docs import attach_params
from typing import NamedTuple as _NamedTuple

class _SearchParams(_NamedTuple):
    pts:           _pd_DataFrame
    local_crs:     str
    c:             list
    x:             str
    y:             str
    suffix:        object
    pts_target:    _pd_DataFrame
    x_tgt:         str
    y_tgt:         str
    row_name_tgt:  str
    col_name_tgt:  str
    grid:          object
    stat:          object

def _validate_kwargs(
        pts:_pd_DataFrame,
        crs:str,
        r:float,
        c:list=[],
        stat:str='sum',
        x:str='lon',
        y:str='lat',
        row_name:str='id_y',
        col_name:str='id_x',
        suffix:str=None,
        pts_target:_pd_DataFrame=None,
        x_tgt:str=None,
        y_tgt:str=None,
        row_name_tgt:str=None,
        col_name_tgt:str=None,
        output_spacing:float=None,
        output_spacing_y:float=None,
        build_grid_obj:bool=True,
        n_pts_src_extra:int=0,
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
    if suffix is not None and not isinstance(suffix, (str, dict, list, tuple)):
        suffix = str(suffix)
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
            progress_print("Warning: No columns specified for aggregation - will simply count number of points within radius.")
            stat = 'count'
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
    if crs is None:
        raise ValueError(
            "crs is required. Pass your coordinate reference system string (e.g. crs='EPSG:4326'), "
            "or pass crs='' to skip reprojection entirely when your coordinates are already in a "
            "Cartesian/projected plane and r is expressed in the same units as your coordinates."
        )
    if not crs:
        # Cartesian mode: coordinates are already in the target unit system; skip all reprojection.
        # r is interpreted directly in the same units as the x/y columns.
        local_crs = proj_crs if proj_crs != 'auto' else None
    elif proj_crs == 'auto':
        x_center = (min([pts[x].min(), pts_target[x_tgt].min()])+max([pts[x].max(), pts_target[x_tgt].max()]))/2
        y_center = (min([pts[y].min(), pts_target[y_tgt].min()])+max([pts[y].max(), pts_target[y_tgt].max()]))/2
        local_crs = 'EPSG:'+str(convert_wgs_to_utm(x_center, y_center))
    else:
        local_crs = proj_crs
    if crs and crs != local_crs:
        x,y,local_crs = convert_pts_to_crs(pts=pts, x=x, y=y, initial_crs=crs, target_crs=proj_crs)
        if not same_target:
            x_tgt,y_tgt,local_crs = convert_pts_to_crs(pts=pts_target, x=x_tgt, y=y_tgt, initial_crs=crs, target_crs=proj_crs)
        else:
            x_tgt,y_tgt = x,y
    
    if build_grid_obj:
        grid = build_grid(
            pts_source=pts,
            initial_crs=local_crs,
            local_crs=local_crs,
            data_crs=crs or None,
            r=r,
            x=x,
            y=y,
            pts_target=pts_target,
            x_tgt=x_tgt,
            y_tgt=y_tgt,
            output_spacing=output_spacing,
            output_spacing_y=output_spacing_y,
            n_pts_src_extra=n_pts_src_extra,
            silent=silent,
        )
    else:
        # Caller (detect_cluster_cells) only needs the validated/reprojected kwargs;
        # the grid is built by the detect_cluster_pts it delegates to. Skip the
        # throwaway build here.
        grid = None

    return _SearchParams(pts, local_crs, c, x, y, suffix, pts_target, x_tgt, y_tgt, row_name_tgt, col_name_tgt, grid, stat)
#


def _unpack_contingency(v):
    if isinstance(v, (tuple, list)):
        return int(v[0]), int(v[1])
    v = int(v)
    return v, v

def _unpack_merge_dist(v):
    if v is None:
        return None, None
    if isinstance(v, (tuple, list)):
        return v[0], v[1]
    f = float(v)
    return f, f

def _unpack_min_cluster_share(v):
    if isinstance(v, (tuple, list)):
        return float(v[0]), float(v[1]), float(v[2])
    f = float(v)
    return f, f, f

def _parse_sample_area_spec(spec):
    "Parse 'method,key=val,key=val' into (method, {key: val})."
    if ',' not in spec or '=' not in spec:
        return _SAMPLE_AREA_ALIASES.get(spec, spec), {}
    parts = spec.split(',')
    method = _SAMPLE_AREA_ALIASES.get(parts[0].strip(), parts[0].strip())
    params = {}
    for part in parts[1:]:
        if '=' in part:
            k, v = part.split('=', 1)
            params[k.strip()] = v.strip()
    return method, params

_SAMPLE_AREA_ALIASES = {
    'buff_cells':     'buff_cells_min_pts',
    'buff_non_empty': 'buff_non_empty_cells',
    'concave_hull':   'concave',
    'convex_hull':    'convex',
    'bbox':           'bounding_box',
}

_SAMPLE_AREA_SPEC_DOCS = (
    "sample_area spec string:  'method'  or  'method,key=val,key=val'\n"
    "\n"
    "Methods and parameters\n"
    "----------------------------------------------------------------------\n"
    "buff_non_empty_cells / buff_non_empty  [DEPRECATED - use buff_cells]\n"
    "    Buffer around all non-empty grid cells.\n"
    "    buf=<float>       Buffer radius in CRS units (default: search radius r).\n"
    "\n"
    "buff_cells_min_pts / buff_cells\n"
    "    Buffer around cells that contain >= min_pts data points.\n"
    "    min_pts=<int>     Minimum points per cell (default: 1 = non-empty).\n"
    "    buf=<float>       Buffer radius (default: r).\n"
    "\n"
    "buff_pts\n"
    "    Buffer around every individual point. (!) Slow for large datasets.\n"
    "    buf=<float>       Buffer radius (default: r).\n"
    "\n"
    "concave / concave_hull\n"
    "    Concave hull of all points + buffer.\n"
    "    concavity=<float> Shape tightness: 0=very concave, large=convex (default: 1).\n"
    "    buf=<float>       Buffer radius (default: r).\n"
    "    tol=<float>       Douglas-Peucker simplification tolerance (default: None).\n"
    "\n"
    "convex / convex_hull\n"
    "    Convex hull of all points + buffer.\n"
    "    buf=<float>       Buffer radius (default: r).\n"
    "    tol=<float>       Simplification tolerance (default: None).\n"
    "\n"
    "bounding_box / bbox\n"
    "    Axis-aligned bounding box of all points + buffer.\n"
    "    buf=<float>       Buffer radius (default: r).\n"
    "\n"
    "grid\n"
    "    Full grid extent (no parameters).\n"
    "\n"
    "----------------------------------------------------------------------\n"
    "Examples\n"
    "    sample_area='buff_cells'\n"
    "    sample_area='buff_cells,min_pts=2,buf=500'\n"
    "    sample_area='buff_cells_min_pts,min_pts=1,buf=53.3'\n"
    "    sample_area='concave,concavity=0.5,buf=1000,tol=50'\n"
    "    sample_area='convex'\n"
    "    sample_area='bounding_box'\n"
    "\n"
    "You can also pass a Shapely Polygon or MultiPolygon directly (must be\n"
    "in the same projected CRS as the pts coordinates).\n"
)


@time_func_perf
def resolve_sample_area(
    pts:_pd_DataFrame,
    r:float,
    sample_area='buf_non_empty_cells',
    crs:str=None,
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
        hull_type, parsed = _parse_sample_area_spec(sample_area)

        if hull_type == 'buff_non_empty_cells':
            progress_print("sample_area='buff_non_empty_cells' is deprecated. Use sample_area='buff_cells' instead.")
            hull_type = 'buff_cells_min_pts'
            parsed.setdefault('min_pts', '1')

        _buf       = float(parsed['buf'])       if 'buf'       in parsed else r
        _tol       = float(parsed['tol'])       if 'tol'       in parsed else None
        _concavity = float(parsed['concavity']) if 'concavity' in parsed else 1
        if 'min_pts' in parsed:
            _min_pts = int(parsed['min_pts'])
        elif min_pts_to_sample_cell != 0:
            _min_pts = min_pts_to_sample_cell
        else:
            _min_pts = 1

        _tol_resolved = _tol if _tol is not None else _buf
        def _fmt(v): return str(int(v)) if v == int(v) else str(v)

        _param_parts = []
        if hull_type in ('buff_cells_min_pts', 'buff_pts', 'concave', 'convex', 'bounding_box', 'grid'):
            if hull_type != 'grid':
                _param_parts.append(f'buf={_fmt(_buf)}')
        if hull_type == 'buff_cells_min_pts':
            _param_parts.append(f'min_pts={_min_pts}')
        if hull_type in ('concave', 'convex', 'buff_pts'):
            _param_parts.append(f'tol={_fmt(_tol_resolved)}')
        if hull_type == 'concave':
            _param_parts.append(f'concavity={_fmt(_concavity)}')
        _params_str = (', '.join(_param_parts) + '. ') if _param_parts else ''
        progress_print(f"Creating sample area: method='{hull_type}', {_params_str}Use 'grid.sample_area' to inspect.")

        sample_area = infer_sample_area_from_pts(
            pts=pts,
            grid=grid,
            x=x,
            y=y,
            hull_type=hull_type,
            buffer=_buf,
            min_pts_to_sample_cell=_min_pts,
            plot_sample_area=None,
        )

    elif isinstance(sample_area, (_shapely_Polygon, _shapely_MultiPolygon, _shapely_GeometryCollection)):
        if crs and local_crs and crs != local_crs:
            sample_area = convert_MultiPolygon_crs(multipoly=sample_area, initial_crs=crs, target_crs=local_crs)
    else:
        raise ValueError(
            f"sample_area must be a str spec, shapely Polygon, or shapely MultiPolygon; got {type(sample_area).__name__}. "
            "Call resolve_sample_area.params() for valid string spec options."
        )
    
    
    
    return sample_area

def _resolve_sample_area_params():
    print(_SAMPLE_AREA_SPEC_DOCS)
resolve_sample_area.params = _resolve_sample_area_params

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
    output_spacing:float=None,
    output_spacing_y:float=None,
    n_pts_src_extra:int=0,
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
        output_spacing=output_spacing,
        output_spacing_y=output_spacing_y,
        n_pts_src_extra=n_pts_src_extra,
        silent=silent,
    )
#

# Moment-based aggregations and the raw-moment powers each needs (besides the
# mean): variance needs E[x^2]; skewness adds E[x^3]; kurtosis adds E[x^4].
_MOMENT_AGGS = {
    'variance': [2],
    'std': [2],       # standard deviation = sqrt(variance)
    'cv': [2],        # coefficient of variation = std / mean (unitless spread)
    'skewness': [2, 3],
    'kurtosis': [2, 3, 4],
}

# Maps full stat names to their short identifiers used in column names.
# Short forms are also accepted directly as stat= values (e.g. stat='cnt').
_AGG_ABBR = {
    'sum':      'sum',
    'count':    'cnt',
    'mean':     'avg',
    'variance': 'var',
    'std':      'std',
    'cv':       'cv',
    'skewness': 'skw',
    'kurtosis': 'krt',
}

def _default_suffix(stat, r):
    """Return the auto-generated suffix for a given stat and radius.

    Format: ``_{stat}_{r}`` e.g. ``_sum_2000``, ``_avg_750``, ``_krt_1000``.
    stat is normalised via _AGG_ABBR so both 'mean' and 'avg' produce '_avg_750'.
    """
    stat_str = _AGG_ABBR.get(stat, stat)
    if r == int(r):
        r_str = str(int(r))
    else:
        r_str = repr(r).replace('.', 'p').replace('-', 'n').replace('+', '')
    return f'_{stat_str}_{r_str}'


# TODO: min/max/range are NOT yet implemented. Unlike sum/mean/variance/skewness/
# kurtosis they are NOT additive — they cannot be derived from radius-sums of cell
# totals. Implementing them needs a separate aggregation path (see the NOTE blocks
# in radius_search below and in disk_aggregation / point_grid_assignment): the grid
# must hold a per-cell MIN/MAX (not a sum) for fully-contained cells, and overlapped
# cells must still be scanned point-by-point. Wrappers prepared but commented out.
# def radius_max(pts, crs:str, r:float, c:list=[], suffix:str='_r_max', **kwargs):
#     """Compute the max of neighbouring points within radius r for each column in c.
#     Convenience wrapper for ``radius_search(..., statt='max')``.
#     """
#     return radius_search(pts=pts, crs=crs, r=r, c=c, statt='max', suffix=suffix, **kwargs)

# def radius_min(pts, crs:str, r:float, c:list=[], suffix:str='_r_min', **kwargs):
#     """Compute the min of neighbouring points within radius r for each column in c.
#     Convenience wrapper for ``radius_search(..., statt='min')``.
#     """
#     return radius_search(pts=pts, crs=crs, r=r, c=c, statt='min', suffix=suffix, **kwargs)

# def radius_range(pts, crs:str, r:float, c:list=[], suffix:str='_r_range', **kwargs):
#     """Compute the range (max - min) of neighbouring points within radius r for each column in c.
#     Convenience wrapper for ``radius_search(..., statt='range')``.
#     """
#     # implement by computing both max and min in one search, then subtract.
#     return radius_search(pts=pts, crs=crs, r=r, c=c, statt='range', suffix=suffix, **kwargs)



@attach_params
@time_func_perf
def radius_search(
    pts:_pd_DataFrame,
    crs:str,    # e.g. 'EPSG:4326'; pass '' to skip reprojection (Cartesian coords)
    r,          # float | list of floats | list of (r_in,r_out) bands | list of (r_in,r_out,w) weighted bands
    c:list=[],
    x:str='lon',
    y:str='lat',
    stat:str=['sum','count','mean','variance','std','cv','skewness','kurtosis'][0],
    exclude_self:bool=True,
    proj_crs:str='auto',
    suffix=None,
    overwrite:bool=False,
    weight_valid_area:str=None,
    spacing:float=None,
    sample_area=False,
    row_name:str='id_y',
    col_name:str='id_x',
    pts_target:_pd_DataFrame=None,
    x_tgt:str=None,
    y_tgt:str=None,
    row_name_tgt:str=None,
    col_name_tgt:str=None,
    keep_cols:bool=False,
    exclude_pt_itself:bool=None,   # deprecated alias for exclude_self
    _dev:dict=None,
    silent:bool=None,
):
    """
    Aggregates data from neighboring points within a search radius for every point in ``pts``.
    Results are appended in-place to ``pts`` as new column(s) named ``{c}{suffix}``.
    The Grid object returned provides access to plots and internal search state.

    Args:
    -------
    pts (pandas.DataFrame):
        Points for which neighborhood aggregates are computed. Results are appended to this DataFrame in-place.
        Note: row order of ``pts`` may change after the call.
    crs (str):
        CRS of the coordinates in ``pts``, e.g. ``'EPSG:4326'``.
        Pass ``crs=''`` to skip reprojection entirely — use this when your
        coordinates are already in a Cartesian/projected plane and ``r`` is in
        the same units as ``x``/``y``.
    r (float or list or list of tuples):
        Search radius or multiple radii, or a (weighted) distance bands:

        - ``float`` — single radius in metres (or in ``x``/``y`` units when ``crs=''``).
        - ``[500, 750, 1000]`` — list of radii; produces one result column per radius
          named ``{col}_{stat}_{r}`` (e.g. ``employment_sum_750``).
        - ``[(0,500), (500,750)]`` — distance bands ``(r_inner, r_outer)``; produces one
          column per band named ``{col}_{stat}_{r_in}_{r_out}`` (e.g. ``employment_sum_0_500``).
        - ``[(0,500,1), (500,750,2)]`` — weighted bands; weights are normalised and the
          single weighted aggregate is named ``{col}_{stat}_wgt``.
          Intermediate band columns are dropped unless ``keep_cols=True``.

        For distance bands only additive statistics (``sum``, ``count``) are strictly
        correct; other stats are subtracted numerically.
    c (str or list):
        Column name or list of column names to aggregate within the search radius.
        If empty or None, points within the radius are counted.
        Columns must exist in ``pts`` (or in ``pts_target`` if provided).
    x (str):
        Column name of the x-coordinate (longitude) in ``pts`` (default=``'lon'``).
    y (str):
        Column name of the y-coordinate (latitude) in ``pts`` (default=``'lat'``).
    stat (str or list):
        Statistic to compute within the search radius. One of ``'sum'``, ``'count'``,
        ``'mean'``, ``'variance'``, ``'std'``, ``'cv'``, ``'skewness'``, ``'kurtosis'``
        (default=``'sum'``). Pass a **list** to compute multiple statistics in a single
        search pass, e.g. ``stat=['sum', 'mean', 'variance']``. When used with
        ``detect_cluster_pts`` / ``detect_cluster_cells`` the **first** stat in the list
        drives cluster detection; any additional stats are appended to the output grid as
        extra cell-level aggregates.
    exclude_self (bool):
        If True, each point's own value is subtracted from its radius aggregate (default=True).
        Formerly ``exclude_pt_itself`` (deprecated).
    proj_crs (str):
        Metric CRS used internally. ``'auto'`` selects the appropriate UTM zone from the data extent.
        Pass an explicit EPSG string (e.g. ``'EPSG:32632'``) to override, or ``None`` to skip
        reprojection (default=``'auto'``).
    keep_cols (bool):
        If False, intermediate columns added during processing (grid indices, offsets, proj x+y, etc.)
        are removed from ``pts`` before returning. If None proj x+y are retained. If True they are retained (default=False).
    overwrite (bool):
        If True, existing output columns with the same names are overwritten. Raises ValueError
        when False and a collision is detected (default=False).
    weight_valid_area (str):
        Inverse-area weighting for edge effects. ``'estimate'`` uses a fast approximation
        (MSE ≈ 5 % of cell area); ``'precise'`` is exact but slow. ``None`` disables weighting (default=None).

        **Limitation:** contained cells (fully inside the search disk) are treated as binary —
        fully valid or fully invalid — based on whether they fall inside ``sample_area``.
        Cells that straddle the ``sample_area`` boundary but were classified as fully valid
        contribute zero invalid area, causing a slight upward bias in ``valid_area_share``
        when ``sample_area`` is a custom Polygon or MultiPolygon with sharp edges. For
        method-string sample areas (``'buff_non_empty_cells'``, ``'concave'``, etc.) the
        boundary follows cell edges exactly and this bias does not occur.
    spacing (float):
        Output cell size, in the same unit as ``r`` (metres after reprojection). Controls the
        resolution of the output grid used for exports and plots — NOT the internal search grid,
        whose cell size is chosen automatically for speed. Per-point aggregates are exact
        regardless of this value, so a finer output grid is well-defined. When None, defaults to
        ``r/3`` (default=None).
    sample_area (shapely.Polygon | shapely.MultiPolygon | str):
        Area used for valid-area weighting. Accepted string values:
            - ``'buff_non_empty_cells'``: non-empty grid cells plus a radius-sized buffer (default)
            - ``'buf_cells_min_pts'``: grid cells with at least 'min_pts_to_sample_cell' plus a radius-sized buffer (default)
            - ``'concave'``: concave hull around points
            - ``'convex'``: convex hull around points
            - ``'buffer'``: buffer around individual points (slow for large datasets)
            - ``'bounding_box'``: axis-aligned bounding box
            - ``'grid'`` or ``None``: full grid extent
        Alternatively pass any Shapely ``Polygon`` or ``MultiPolygon`` directly. The geometry
        is assumed to be in ``crs`` and is reprojected to ``local_crs`` automatically. If your
        geometry is already in the projected CRS (``local_crs``), pass ``crs=local_crs`` to
        prevent a double reprojection.
        See ``infer_sample_area_from_pts`` for finer control (default=False).
    suffix (str):
        Suffix appended to each column name in ``c`` to form the result column names.
        When None (default), derived from ``stat`` and ``r``: ``_{stat}_{r}``
        e.g. ``employment_sum_2000``, ``employment_avg_750``, ``employment_krt_2000``.
        When ``stat`` is a list, pass a dict ``{stat: suffix}`` for per-stat control.
    row_name (str):
        Name for the grid row-index column appended to ``pts`` (default=``'id_y'``).
    col_name (str):
        Name for the grid column-index column appended to ``pts`` (default=``'id_x'``).
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
        as columns ``{c}{suffix}`` (e.g. ``employment_sum_750``). Use ``grid.plot.vars()`` to visualise results.

    Examples:
    -------
    from aabpl.main import radius_search
    from pandas import read_csv
    pts = read_csv('C:/path/to/file.txt', sep=',', header=None)
    pts.columns = ["eid", "employment", "industry", "lat", "lon", "moved"]
    grid = radius_search(pts, crs="EPSG:4326", r=750, c=['employment'])
    # Result column: employment_sum_750
    grid.plot.vars(filename='employment_750')
    """
    # ── multi-radius delegation ───────────────────────────────────────────────
    # When r is not a scalar (list of radii or distance bands), hand off to the
    # dedicated implementation which calls radius_search once per unique radius.
    from .radius_search.multi_radius import _parse_r_spec, _multi_radius_search
    spec_type, spec_data = _parse_r_spec(r)
    if spec_type != 'single':
        return _multi_radius_search(
            pts=pts, r=r, c=c, x=x, y=y, stat=stat,
            suffix=suffix, keep_cols=keep_cols,
            exclude_self=exclude_self, silent=silent,
            _radius_search_fn=radius_search,
            _parsed_spec=(spec_type, spec_data),
            crs=crs, proj_crs=proj_crs,
            sample_area=sample_area,
            spacing=spacing, row_name=row_name, col_name=col_name,
            pts_target=pts_target, x_tgt=x_tgt, y_tgt=y_tgt,
            row_name_tgt=row_name_tgt, col_name_tgt=col_name_tgt,
            weight_valid_area=weight_valid_area, overwrite=overwrite,
            _dev=_dev,
        )
    # ── end multi-radius ──────────────────────────────────────────────────────

    # ---- multi-stat normalisation -------------------------------------------
    # When stat is a list/tuple we run the search ONCE using the union of all
    # required helper columns (the search always sums), then derive each stat's
    # output columns from those sums in post-processing.
    _stat_list = None       # None → single-stat path (unchanged behaviour)
    _stat_suffixes = {}     # {stat_name: output_suffix}
    _multi_pow_hcols = {}   # {power: [helper_col_name per orig_col]} (multi-stat only)
    if isinstance(stat, (list, tuple)):
        _stat_list = [str(a) for a in stat]
        if isinstance(suffix, dict):
            _stat_suffixes = {a: suffix.get(a, _default_suffix(a, r)) for a in _stat_list}
        elif isinstance(suffix, (list, tuple)):
            _stat_suffixes = dict(zip(_stat_list, suffix))
        else:
            _stat_suffixes = {a: _default_suffix(a, r) for a in _stat_list}
        stat = 'sum'         # grid building and search are stat-agnostic
        suffix = '__rs_int__'  # internal suffix dropped after post-processing
    # -------------------------------------------------------------------------

    _cols_before = set(pts.columns)
    _orig_pts_target = pts_target  # capture before _validate_kwargs may set it to pts
    _cols_before_tgt = set(pts_target.columns) if pts_target is not None else None
    init_sort = find_column_name('initial_sort', existing_columns=pts.columns)
    pts[init_sort] = range(len(pts))

    vk = _validate_kwargs(
        pts=pts, crs=crs, r=r, c=c, stat=stat, x=x, y=y, row_name=row_name,
        col_name=col_name, suffix=suffix, pts_target=pts_target, x_tgt=x_tgt, y_tgt=y_tgt,
        row_name_tgt=row_name_tgt, col_name_tgt=col_name_tgt,
        output_spacing=spacing,
        proj_crs=proj_crs, silent=silent,
    )
    pts, local_crs = vk.pts, vk.local_crs
    c, x, y, suffix = vk.c, vk.x, vk.y, vk.suffix
    pts_target, x_tgt, y_tgt = vk.pts_target, vk.x_tgt, vk.y_tgt
    row_name_tgt, col_name_tgt, grid, stat = vk.row_name_tgt, vk.col_name_tgt, vk.grid, vk.stat
    # Always work on a local copy so the caller's c list is never mutated by the
    # helper-column appends below (count_helper_col, moment_helper_cols, etc.).
    c = list(c)
    # TODO: replace with the actual output column names once spacing/nesting
    # changes how row/col indices are stored (may differ from input row_name/col_name)
    grid.output_row_name = row_name
    grid.output_col_name = col_name

    # Resolve dynamic default suffix for the single-stat path (multi-stat already set
    # suffix='__rs_int__' above; _validate_kwargs may have changed stat, e.g. to 'count').
    if suffix is None:
        suffix = _default_suffix(stat, r)

    # Snapshot the user's requested columns before any helper columns are appended.
    # Used to define _output_cols and to drive the moment-combine block.
    orig_cols = list(c)

    count_helper_col = None
    moment_helper_cols = []

    if _stat_list is not None:
        # Multi-stat: add the union of all helpers required across every requested stat.
        # Powers are grouped (all x^2 first, then all x^3, …) so the result-column
        # layout mirrors the single-stat moment path and the same combine logic applies.
        _needs_count = any(a in ['count', 'mean'] + list(_MOMENT_AGGS) for a in _stat_list)
        _max_power = max((max(_MOMENT_AGGS[a]) for a in _stat_list if a in _MOMENT_AGGS), default=0)
        for pw in range(2, _max_power + 1):
            _multi_pow_hcols[pw] = []
            for colname in orig_cols:
                hcol = find_column_name(f'{colname}pow{pw}', '_helper_col', existing_columns=pts_target.columns)
                pts_target[hcol] = pts_target[colname].astype(float) ** pw
                c.append(hcol)
                moment_helper_cols.append(hcol)
                _multi_pow_hcols[pw].append(hcol)
        if _needs_count:
            count_helper_col = find_column_name('count', '_helper_col', existing_columns=pts_target.columns)
            pts_target[count_helper_col] = 1
            c.append(count_helper_col)
        # Actual output columns across all stats (used for collision check + cleanup)
        _output_cols = set(col + _stat_suffixes[a] for a in _stat_list for col in orig_cols)
        if not orig_cols and 'count' in _stat_list and count_helper_col:
            _output_cols.add(count_helper_col + _stat_suffixes['count'])
    else:
        if stat in _MOMENT_AGGS:
            # variance/skewness/kurtosis are computed from raw moments: radius-sum x^p
            # for the required powers (one helper column per original column per power)
            # plus a count column. All helpers are dropped again after combining below.
            # Append powers grouped (all x^2, then all x^3, ...) so the result columns
            # have a predictable [Sum(x) | Sum(x^2) | ... | count] layout.
            moment_powers = _MOMENT_AGGS[stat]
            for pw in moment_powers:
                for colname in orig_cols:
                    hcol = find_column_name(f'{colname}pow{pw}', '_helper_col', existing_columns=pts_target.columns)
                    pts_target[hcol] = pts_target[colname].astype(float) ** pw
                    c.append(hcol)
                    moment_helper_cols.append(hcol)
            count_helper_col = find_column_name('count', '_helper_col', existing_columns=pts_target.columns)
            pts_target[count_helper_col] = 1
            c.append(count_helper_col)
        if stat in ['count', 'mean']:
            count_helper_col = find_column_name('count', '_helper_col', existing_columns=pts_target.columns)
            pts_target[count_helper_col] = 1
            if stat == 'count':
                # Only the count is needed — drop all value columns from the search.
                # Reassign to a new local list so the caller's list is not mutated.
                c = []
            c.append(count_helper_col)
        # Track the exact set of columns radius_search is supposed to produce so the
        # cleanup filter below can simply check membership instead of guessing via
        # prefix/suffix heuristics.
        _output_cols = set(col + suffix for col in orig_cols)

    if not overwrite:
        _collision = _output_cols & _cols_before
        if _collision:
            raise ValueError(
                f"Output columns {sorted(_collision)} already exist in pts. "
                "Pass overwrite=True to overwrite them."
            )
    # if agg in ['min', 'max', 'range']:
    #     # NOT YET IMPLEMENTED. No summed helper columns exist for min/max — they are
    #     # not additive. The grid/search engine itself must aggregate per-cell min/max
    #     # (see NOTE in disk_aggregation.search_and_aggregate and in
    #     # point_grid_assignment.aggregate_point_data_to_cells). 'range' needs both.
    #     pass


    _is_internal = _OUTER_PROGRESS.get() is not None
    if not _is_internal:
        _prog = RadiusSearchProgress(silent=bool(silent), n_pts=len(pts), n_tgt=len(pts_target))
        _token = _OUTER_PROGRESS.set(_prog)
        _prog.start()
        _prog.step("initializing")

    if exclude_pt_itself is not None:
        progress_print("DeprecationWarning: `exclude_pt_itself` is deprecated, use `exclude_self` instead.")
        exclude_self = exclude_pt_itself

    # initialize disk_search
    grid.search = DiskSearch(
        grid=grid,
        r=r,
        exclude_self=exclude_self,
        weight_valid_area=weight_valid_area,
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
        suffix=suffix,
        plot_cell_reg_assign=_d.get('plot_cell_reg_assign'),
        plot_offset_checks=_d.get('plot_offset_checks'),
        plot_offset_regions=_d.get('plot_offset_regions'),
        plot_offset_raster=_d.get('plot_offset_raster'),
        silent=silent,
    )

    # in case sums shall be weighted by sample area
    grid.sample_area = resolve_sample_area(
        pts=pts,r=r,sample_area=sample_area,
        crs=crs,local_crs=local_crs,x=x,y=y,
        grid=grid, min_pts_to_sample_cell=0)
    intersect_polygon_with_grid(grid)

    if not _is_internal:
        _prog.step("searching")
    disk_sums_for_pts = grid.search.perform_search(silent=False if silent is None else silent,plot_pt_disk=_d.get('plot_pt_disk'))

    if _stat_list is not None:
        # Multi-stat post-processing: derive each requested stat from the internal
        # radius sums and write results under the per-stat output suffix.
        _int_suf = suffix  # the internal suffix used during the search
        count_rs = count_helper_col + _int_suf if count_helper_col else None
        _pow_block_rs = {pw: [hc + _int_suf for hc in hcs] for pw, hcs in _multi_pow_hcols.items()}
        for _st in _stat_list:
            _suf = _stat_suffixes[_st]
            if _st == 'sum':
                for col in orig_cols:
                    pts[col + _suf] = pts[col + _int_suf]
            elif _st == 'count':
                for col in orig_cols:
                    pts[col + _suf] = pts[count_rs]
                if not orig_cols and count_helper_col:
                    pts[count_helper_col + _suf] = pts[count_rs]
            elif _st == 'mean':
                n = pts[count_rs].where(pts[count_rs] > 0)
                for col in orig_cols:
                    pts[col + _suf] = pts[col + _int_suf] / n
            elif _st in _MOMENT_AGGS:
                _mpowers = _MOMENT_AGGS[_st]
                n = pts[count_rs].where(pts[count_rs] > 0)
                for i, col in enumerate(orig_cols):
                    m1 = pts[col + _int_suf] / n
                    E = {pw: pts[_pow_block_rs[pw][i]] / n for pw in _mpowers}
                    m2 = E[2] - m1 ** 2
                    if _st == 'variance':
                        _result = m2
                    elif _st == 'std':
                        _result = m2.clip(lower=0) ** 0.5
                    elif _st == 'cv':
                        _result = m2.clip(lower=0) ** 0.5 / m1
                    elif _st == 'skewness':
                        m3 = E[3] - 3 * m1 * E[2] + 2 * m1 ** 3
                        _result = m3 / m2 ** 1.5
                    else:  # kurtosis
                        m4 = E[4] - 4 * m1 * E[3] + 6 * m1 ** 2 * E[2] - 3 * m1 ** 4
                        _result = m4 / m2 ** 2 - 3
                    pts[col + _suf] = _result
        # Drop all internal result columns (they carry the internal suffix)
        pts.drop(columns=[col for col in pts.columns if col.endswith(_int_suf)], inplace=True, errors='ignore')
        # Drop source helper columns from pts_target
        _hcols_to_drop = moment_helper_cols + ([count_helper_col] if count_helper_col else [])
        if _hcols_to_drop:
            pts_target.drop(columns=_hcols_to_drop, inplace=True, errors='ignore')

    if stat in ['mean']:
        n_rs = 1 # TODO for distance bands... later
        radius_count_cols = disk_sums_for_pts.columns[-n_rs:]
        radius_count_col = radius_count_cols[0] # TODO for distance bands... later
        for s_name in disk_sums_for_pts.columns[:-n_rs]:
            if s_name not in radius_count_cols:
                n = pts[radius_count_col]
                pts.loc[n > 0, s_name] = pts.loc[n > 0, s_name] / n[n > 0]
                pts.loc[n == 0, s_name] = _np_nan
        pts.drop(columns=[count_helper_col], inplace=True)
    if stat in ['count']:
        # Rename the single count-result column to each user-requested col+suffix.
        # When orig_cols is non-empty, copy the count into each expected output col
        # and drop the internal name. When orig_cols is empty keep the internal col.
        count_result_col = count_helper_col + suffix
        for col in orig_cols:
            pts[col + suffix] = pts[count_result_col]
        if orig_cols:
            pts.drop(columns=[count_result_col], inplace=True, errors='ignore')
        pts_target.drop(columns=[count_helper_col], inplace=True, errors='ignore')
    if stat in _MOMENT_AGGS:
        # result-column layout: [Sum(x) per col | Sum(x^p) per col for each power |
        # count]. Combine raw moments into the central statistic, written back into
        # each original column's slot, then drop every helper column.
        moment_powers = _MOMENT_AGGS[stat]
        cols = list(disk_sums_for_pts.columns)
        k = len(orig_cols)
        count_col = cols[-1]
        n = pts[count_col].where(pts[count_col] > 0)  # 0 neighbours -> NaN
        pow_block = {pw: cols[(gi + 1) * k:(gi + 2) * k] for gi, pw in enumerate(moment_powers)}
        for i in range(k):
            m1 = pts[cols[i]] / n                                   # E[x]
            E = {pw: pts[pow_block[pw][i]] / n for pw in moment_powers}  # E[x^p]
            m2 = E[2] - m1 ** 2                                     # variance (central 2nd moment)
            if stat == 'variance':
                _result = m2
            elif stat == 'std':
                _result = m2.clip(lower=0) ** 0.5
            elif stat == 'cv':
                _result = m2.clip(lower=0) ** 0.5 / m1
            elif stat == 'skewness':
                m3 = E[3] - 3 * m1 * E[2] + 2 * m1 ** 3
                _result = m3 / m2 ** 1.5
            else:  # 'kurtosis' (excess: normal -> 0)
                m4 = E[4] - 4 * m1 * E[3] + 6 * m1 ** 2 * E[2] - 3 * m1 ** 4
                _result = m4 / m2 ** 2 - 3
            pts[cols[i]] = _result
        # drop helper result columns (all Sum(x^p) blocks + count) ...
        pts.drop(columns=cols[k:], inplace=True)
        # ... and the source helper columns we added to pts_target
        pts_target.drop(columns=moment_helper_cols + [count_helper_col], inplace=True, errors='ignore')
    # ---- min / max / range (NOT YET IMPLEMENTED) ----------------------------
    # These are NOT additive and cannot be derived from radius sums. The search
    # engine must produce a per-point radius MIN/MAX directly (see the NOTE in the
    # agg-setup block above and in disk_aggregation / point_grid_assignment). Once
    # the engine supports an statt='min'/'max', the result column will already hold
    # the min/max and need no moment combine; 'range' computes both and subtracts:
    # if agg in ['min', 'max']:
    #     # result already holds the per-point radius min/max — nothing to combine.
    #     pass
    # if agg in ['range']:
    #     # needs both min and max from the engine, written e.g. as <col>_min/<col>_max
    #     # result columns; range = max - min, then drop the two helper result cols.
    #     pass

    # NOTE: the output grid (cell ids + aggregates) is built lazily via
    # grid.update_spacing(), called by the plots/exports and by
    # detect_cluster_pts/detect_cluster_cells. radius_search deliberately skips it
    # so a search-only call does not pay the per-point output-aggregation cost.
    pts.sort_values(init_sort, inplace=True)
    pts.drop(columns=[init_sort], inplace=True)

    if not keep_cols:
        if keep_cols is None:
            _keep_extra = {grid.output_row_name, grid.output_col_name, x, y}
        else:
            _keep_extra = {x, y}
        _to_drop = [
            clmn for clmn in pts.columns
            if clmn not in _cols_before and clmn not in _output_cols and clmn not in _keep_extra
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

@_wraps(radius_search)
def radius_sum(pts, crs:str, r:float, c:list=[], suffix=None, **kwargs):
    """Sum values of neighbouring points within radius r. Wraps ``radius_search(stat='sum')``.\n\n"""
    return radius_search(pts=pts, crs=crs, r=r, c=c, stat='sum', suffix=suffix, **kwargs)
radius_sum.__doc__ = "Sum values of neighbouring points within radius r. Wraps ``radius_search(stat='sum')``.\n\n" + (radius_search.__doc__ or "")

@_wraps(radius_search)
def radius_count(pts, crs:str, r:float, c:list=[], suffix=None, **kwargs):
    """Count neighbouring points within radius r. Wraps ``radius_search(stat='count')``.\n\n"""
    return radius_search(pts=pts, crs=crs, r=r, c=c, stat='count', suffix=suffix, **kwargs)
radius_count.__doc__ = "Count neighbouring points within radius r. Wraps ``radius_search(stat='count')``.\n\n" + (radius_search.__doc__ or "")

@_wraps(radius_search)
def radius_mean(pts, crs:str, r:float, c:list=[], suffix=None, **kwargs):
    """Mean of neighbouring point values within radius r. Wraps ``radius_search(stat='mean')``.\n\n"""
    return radius_search(pts=pts, crs=crs, r=r, c=c, stat='mean', suffix=suffix, **kwargs)
radius_mean.__doc__ = "Mean of neighbouring point values within radius r. Wraps ``radius_search(stat='mean')``.\n\n" + (radius_search.__doc__ or "")

@_wraps(radius_search)
def radius_variance(pts, crs:str, r:float, c:list=[], suffix=None, **kwargs):
    """Variance of neighbouring point values within radius r. Wraps ``radius_search(stat='variance')``.\n\n"""
    return radius_search(pts=pts, crs=crs, r=r, c=c, stat='variance', suffix=suffix, **kwargs)
radius_variance.__doc__ = "Variance of neighbouring point values within radius r. Wraps ``radius_search(stat='variance')``.\n\n" + (radius_search.__doc__ or "")

@_wraps(radius_search)
def radius_std(pts, crs:str, r:float, c:list=[], suffix=None, **kwargs):
    """Standard deviation of neighbouring point values within radius r. Wraps ``radius_search(stat='std')``.\n\n"""
    return radius_search(pts=pts, crs=crs, r=r, c=c, stat='std', suffix=suffix, **kwargs)
radius_std.__doc__ = "Standard deviation of neighbouring point values within radius r. Wraps ``radius_search(stat='std')``.\n\n" + (radius_search.__doc__ or "")

@_wraps(radius_search)
def radius_cv(pts, crs:str, r:float, c:list=[], suffix=None, **kwargs):
    """Coefficient of variation (std/mean) of neighbouring point values within radius r. Wraps ``radius_search(stat='cv')``.\n\n"""
    return radius_search(pts=pts, crs=crs, r=r, c=c, stat='cv', suffix=suffix, **kwargs)
radius_cv.__doc__ = "Coefficient of variation of neighbouring point values within radius r. Wraps ``radius_search(stat='cv')``.\n\n" + (radius_search.__doc__ or "")

@_wraps(radius_search)
def radius_skewness(pts, crs:str, r:float, c:list=[], suffix=None, **kwargs):
    """Skewness of neighbouring point values within radius r. Wraps ``radius_search(stat='skewness')``.\n\n"""
    return radius_search(pts=pts, crs=crs, r=r, c=c, stat='skewness', suffix=suffix, **kwargs)
radius_skewness.__doc__ = "Skewness of neighbouring point values within radius r. Wraps ``radius_search(stat='skewness')``.\n\n" + (radius_search.__doc__ or "")

@_wraps(radius_search)
def radius_kurtosis(pts, crs:str, r:float, c:list=[], suffix=None, **kwargs):
    """Excess kurtosis of neighbouring point values within radius r. Wraps ``radius_search(stat='kurtosis')``.\n\n"""
    return radius_search(pts=pts, crs=crs, r=r, c=c, stat='kurtosis', suffix=suffix, **kwargs)
radius_kurtosis.__doc__ = "Excess kurtosis of neighbouring point values within radius r. Wraps ``radius_search(stat='kurtosis')``.\n\n" + (radius_search.__doc__ or "")


@attach_params
@time_func_perf
def detect_cluster_pts(
    pts:_pd_DataFrame,
    crs:str,
    r,          # float | list | bands | weighted bands — see radius_search.params.r
    c:list=[],
    stat:str=['sum','count','mean','variance','std','cv','skewness','kurtosis'][0],
    exclude_self:bool=True,
    sample_area='buff_cells_min_pts',
    weight_valid_area:str=None,
    k_th_percentile:float=99.5,
    n_random_points:int=int(1e5),
    random_seed:int=None,
    null_distribution=None,
    x:str='lon',
    y:str='lat',
    row_name:str='id_y',
    col_name:str='id_x',
    suffix:str=None,
    cluster_suffix:str=None,
    proj_crs:str='auto',
    pts_target:_pd_DataFrame=None,
    spacing:float=None,
    plot_distribution:dict=None,
    plot_cluster_points:dict=None,
    keep_cols:bool=False,
    overwrite:bool=False,
    exclude_pt_itself:bool=None,   # deprecated alias for exclude_self
    _dev:dict=None,
    silent:bool=None,
    **kwargs,
):
    """
    For all points in a DataFrame it searches for all other points (potentially of another DataFrame) within the specified radius and aggregate the values for specified column(s).
    It draws random the bounding box containing all points from DataFrame(s) and aggregate the values within the radius to obtain a random distribution.
    Then all points from DataFrame which exceed the k_th_percentile of the random distribution are labeld as clustered.
    The results will be appended to DataFrame.

    null_distribution (array-like or pd.DataFrame, optional):
        User-supplied null-distribution coordinates. When provided the internal uniform-random draw
        is skipped and these coordinates are used as reference points for computing the null radius
        sums. Accepts a DataFrame with the same x/y column names used in the search, or a
        2-column array ``[[x, y], ...]``.

        **Important:** coordinates must be in the same projected CRS that ``pts`` is reprojected
        into (metres), not in the original input CRS. When a real ``crs`` is supplied, use
        ``aabpl.draw_random_coords()`` with the already-projected ``grid.sample_area`` to generate
        compatible coordinates, or project your own points beforehand.
    """
    if 'min_pts_to_sample_cell' in kwargs:
        min_pts_to_sample_cell = kwargs.pop('min_pts_to_sample_cell')
        progress_print(
            "DeprecationWarning: min_pts_to_sample_cell= is deprecated. Pass it inline via sample_area= instead, "
            "e.g. sample_area='buff_cells,min_pts=1'. "
            "Call resolve_sample_area.params() for all options."
        )
    else:
        min_pts_to_sample_cell = 0
    x_tgt        = kwargs.pop('x_tgt', None)
    y_tgt        = kwargs.pop('y_tgt', None)
    row_name_tgt = kwargs.pop('row_name_tgt', None)
    col_name_tgt = kwargs.pop('col_name_tgt', None)
    if kwargs:
        raise TypeError(f"detect_cluster_pts() got unexpected keyword argument(s): {sorted(kwargs)}")
    # ── multi-radius delegation ───────────────────────────────────────────────
    # When r is not a scalar, run a search per unique radius, then build the
    # null distribution across all bands and label cluster points.
    from .radius_search.multi_radius import _parse_r_spec, _multi_radius_search
    spec_type, spec_data = _parse_r_spec(r)
    if spec_type != 'single':
        max_r = max(spec_data) if spec_type == 'list' else max(r_out for _, r_out, *_ in spec_data)

        columns_before = set(pts.columns)
        sort_order_col = find_column_name('initial_sort', existing_columns=pts.columns)
        pts[sort_order_col] = range(len(pts))

        # Reproject pts and resolve column names using the largest radius
        # (build_grid_obj=False avoids building a grid we will not use).
        vk = _validate_kwargs(
            pts=pts, crs=crs, r=max_r, c=c,
            stat=stat, x=x, y=y, row_name=row_name, col_name=col_name,
            suffix=suffix, pts_target=pts_target, x_tgt=x_tgt, y_tgt=y_tgt,
            row_name_tgt=row_name_tgt, col_name_tgt=col_name_tgt,
            build_grid_obj=False, n_pts_src_extra=n_random_points,
            proj_crs=proj_crs, silent=silent,
        )
        pts, local_crs = vk.pts, vk.local_crs
        value_cols = list(vk.c)
        x, y, stat = vk.x, vk.y, vk.stat
        pts_target, x_tgt, y_tgt = vk.pts_target, vk.x_tgt, vk.y_tgt
        row_name_tgt, col_name_tgt = vk.row_name_tgt, vk.col_name_tgt
        # For multi-radius, output cols already embed _{stat}_{r}; append _cluster only.
        if cluster_suffix is None:
            cluster_suffix = '_cluster'

        # Run the multi-radius search (one radius_search call per unique radius).
        # keep_cols=True so the output columns are available for null-distribution labelling.
        last_grid = _multi_radius_search(
            pts=pts, r=r, c=value_cols, x=x, y=y, stat=stat,
            keep_cols=True, exclude_self=exclude_self, silent=silent,
            _radius_search_fn=radius_search,
            _parsed_spec=(spec_type, spec_data),
            crs=local_crs or '', proj_crs=local_crs,
            row_name=row_name, col_name=col_name,
            pts_target=pts_target, x_tgt=x_tgt, y_tgt=y_tgt,
            row_name_tgt=row_name_tgt, col_name_tgt=col_name_tgt,
            weight_valid_area=weight_valid_area,
        )

        # Build the sampling area and compute null-distribution thresholds.
        last_grid.sample_area = resolve_sample_area(
            pts=pts, r=max_r, sample_area=sample_area,
            crs=crs, local_crs=local_crs,
            x=x, y=y, grid=last_grid,
            min_pts_to_sample_cell=min_pts_to_sample_cell, no_plot=True,
        )
        intersect_polygon_with_grid(grid=last_grid)
        from .radius_search.null_distribution import compute_null_distribution
        (thresholds_by_col, random_pts) = compute_null_distribution(
            grid=last_grid, pts=pts, sample_area=last_grid.sample_area,
            min_pts_to_sample_cell=min_pts_to_sample_cell,
            c=value_cols, x=x, y=y, row_name=row_name, col_name=col_name,
            suffix='__mr_null__', n_random_points=n_random_points,
            k_th_percentile=k_th_percentile, random_seed=random_seed,
            silent=silent, null_distribution=null_distribution,
            r=r, stat=stat,
        )

        # Label each output column: True where the aggregate exceeds its threshold.
        for output_col, threshold in thresholds_by_col.items():
            if output_col in pts.columns:
                pts[output_col + cluster_suffix] = pts[output_col] > threshold

        # Drop intermediate search columns unless the caller wants them.
        if not keep_cols:
            intermediate_cols = [col for col in pts.columns
                                  if col not in columns_before
                                  and not col.endswith(cluster_suffix)
                                  and col != sort_order_col]
            if intermediate_cols:
                pts.drop(columns=intermediate_cols, inplace=True)

        pts.sort_values(sort_order_col, inplace=True)
        pts.drop(columns=[sort_order_col], inplace=True)
        last_grid.rndm_pts = random_pts
        return last_grid
    # ── end multi-radius ──────────────────────────────────────────────────────

    _cols_before = set(pts.columns)
    init_sort = find_column_name('initial_sort', existing_columns=pts.columns)
    pts[init_sort] = range(len(pts))

    vk = _validate_kwargs(
        pts=pts, crs=crs, r=r, c=c, stat=stat,
        x=x, y=y, row_name=row_name, col_name=col_name, suffix=suffix,
        pts_target=pts_target, x_tgt=x_tgt, y_tgt=y_tgt, row_name_tgt=row_name_tgt, col_name_tgt=col_name_tgt,
        output_spacing=spacing,
        # the null distribution searches n_random_points extra sources over the
        # same grid, so include them in the spacing/timing estimate.
        n_pts_src_extra=n_random_points,
        proj_crs=proj_crs, silent=silent,
    )
    pts, local_crs = vk.pts, vk.local_crs
    c, x, y, suffix = vk.c, vk.x, vk.y, vk.suffix
    pts_target, x_tgt, y_tgt = vk.pts_target, vk.x_tgt, vk.y_tgt
    row_name_tgt, col_name_tgt, grid, stat = vk.row_name_tgt, vk.col_name_tgt, vk.grid, vk.stat
    if suffix is None:
        suffix = _default_suffix(stat, r)
    if cluster_suffix is None:
        cluster_suffix = f'_cluster{_default_suffix(stat, r)}'
    c = list(c)
    orig_cols = list(c)
    _output_cols = set(str(col)+suffix for col in orig_cols) | set(str(col)+cluster_suffix for col in orig_cols)
    if not overwrite:
        _collision = _output_cols & _cols_before
        if _collision:
            raise ValueError(
                f"Output columns {sorted(_collision)} already exist in pts. "
                "Pass overwrite=True to overwrite them."
            )

    if type(k_th_percentile) not in [list,_np_array, tuple]:
        k_th_percentile = [k_th_percentile for column in c]
    elif len(k_th_percentile) < len(c):
        k_th_percentile = [k_th_percentile[i%len(k_th_percentile)] for i in range(len(c))]

    _is_internal = _OUTER_PROGRESS.get() is not None
    if not _is_internal:
        _prog = DetectClusterProgress(silent=bool(silent), n_pts=len(pts))
        _token = _OUTER_PROGRESS.set(_prog)
        _prog.start()
        _prog.step("initializing")

    if exclude_pt_itself is not None:
        progress_print("DeprecationWarning: `exclude_pt_itself` is deprecated, use `exclude_self` instead.")
        exclude_self = exclude_pt_itself

    # initialize disk_search
    grid.search = DiskSearch(
        grid,
        r=r,
        exclude_self=exclude_self,
        weight_valid_area=weight_valid_area,
    )

    if not _is_internal: _prog.step("assigning target")
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
        sample_area=sample_area, crs=crs, local_crs=local_crs, x=x, y=y, grid=grid,
        min_pts_to_sample_cell=min_pts_to_sample_cell,
        no_plot=plot_distribution is None and plot_cluster_points is None)
    intersect_polygon_with_grid(grid=grid)

    if not _is_internal: _prog.step("null distribution")
    if null_distribution is not None and local_crs and local_crs != crs:
        progress_print(
            "WARNING: null_distribution coordinates must already be in the projected CRS "
            f"'{local_crs}', not the input CRS '{crs}'. "
            "pts were reprojected automatically but null_distribution is used as-is. "
            "Use draw_random_coords() with the projected sample_area to generate valid coordinates."
        )
    from .radius_search.null_distribution import compute_null_distribution
    (_cluster_thresholds_dict, rndm_pts) = compute_null_distribution(
        grid=grid,
        pts=pts,
        sample_area=grid.sample_area,
        min_pts_to_sample_cell=min_pts_to_sample_cell,
        c=c,
        x=x,
        y=y,
        row_name=row_name,
        col_name=col_name,
        suffix=suffix,
        n_random_points=n_random_points,
        k_th_percentile=k_th_percentile,
        random_seed=random_seed,
        silent=silent,
        null_distribution=null_distribution,
    )
    # unpack dict → list keyed by c[j]+suffix for the single-radius path
    cluster_threshold_values = [_cluster_thresholds_dict[col+suffix] for col in c]

    if not silent:
        for (colname, threshold_value, k_th_p) in zip(c, cluster_threshold_values,k_th_percentile):
            progress_print("Threshold value for "+str(k_th_p)+"th-percentile is "+str(threshold_value)+" for "+str(colname)+" within "+str(r)+" meters.")

    if not _is_internal: _prog.step("assigning source")
    _d = _dev or {}
    grid.search.set_source(
        pts=pts,
        c=c,
        x=x,
        y=y,
        row_name=row_name,
        col_name=col_name,
        suffix=suffix,
        plot_cell_reg_assign=_d.get('plot_cell_reg_assign'),
        plot_offset_checks=_d.get('plot_offset_checks'),
        plot_offset_regions=_d.get('plot_offset_regions'),
        plot_offset_raster=_d.get('plot_offset_raster'),
        silent=silent,
    )

    if not _is_internal: _prog.step("searching")
    disk_sums_for_pts = grid.search.perform_search(silent=silent,plot_pt_disk=_d.get('plot_pt_disk'))

    if not _is_internal: _prog.step("labeling clusters")
    for j, cname in enumerate(c):
        pts[str(cname)+str(cluster_suffix)] = disk_sums_for_pts.values[:,j]>cluster_threshold_values[j]

    if plot_distribution is not None:
        from .illustrations.distribution_plot import create_distribution_plot
        create_distribution_plot(
            pts=pts,
            x=x,
            y=y,
            radius_sum_columns=[n+suffix for n in c],
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
            radius_sum_columns=[n+suffix for n in c],
            rndm_pts=rndm_pts,
            cluster_threshold_values=cluster_threshold_values,
            k_th_percentile=k_th_percentile,
            r=r,
            grid=grid,
            **plot_kwargs
    ):
        from .illustrations.distribution_plot import create_distribution_plot
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

    plot_colnames = list(c) + [n+suffix for n in c] + [str(cname)+str(cluster_suffix) for cname in c]
    def plot_cluster_pts(
            self=grid,
            colnames=_np_array(plot_colnames),
            filename:str="",
            **plot_kwargs,
    ):
        from .illustrations.plot_pt_vars import create_plots_for_vars
        return create_plots_for_vars(
            grid=self,
            colnames=colnames,
            filename=filename,
            plot_kwargs=plot_kwargs,
        )
    grid.plot.cluster_pts = plot_cluster_pts

    # Cluster detection always materialises the output grid (cell aggregates etc.).
    grid.update_spacing()

    if plot_cluster_points is not None:
        grid.plot.cluster_pts(**plot_cluster_points)

    if not keep_cols:
        _keep_extra = {grid.output_row_name, grid.output_col_name, init_sort, x, y}
        _to_drop = [
            col for col in pts.columns
            if col not in _cols_before and col not in _output_cols and col not in _keep_extra
        ]
        if _to_drop:
            pts.drop(columns=_to_drop, inplace=True)

    pts.sort_values(init_sort, inplace=True)
    pts.drop(columns=[init_sort], inplace=True)

    if not _is_internal:
        _OUTER_PROGRESS.reset(_token)
        _prog.done()
    return grid
# done

@attach_params
def detect_cluster_cells(
    pts:_pd_DataFrame,
    crs:str,
    r,          # float | list | bands | weighted bands — see radius_search.params.r
    c:list=[],
    stat:str=['sum','count','mean','variance','std','cv','skewness','kurtosis'][0],
    exclude_self:bool=True,
    spacing:float=None,
    sample_area='buff_cells_min_pts',
    weight_valid_area:str=None,
    k_th_percentile:float=99.5,
    n_random_points:int=int(1e5),
    random_seed:int=None,
    null_distribution=None,
    contingency=1,
    merge_dist=None,
    min_cluster_share=(0.05, 0.0, 0.0),
    make_convex:bool=True,
    x:str='lon',
    y:str='lat',
    row_name:str='id_y',
    col_name:str='id_x',
    suffix:str=None,
    cluster_suffix:str=None,
    proj_crs:str='auto',
    pts_target:_pd_DataFrame=None,
    plot_distribution:dict=None,
    plot_cluster_points:dict=None,
    keep_cols:bool=False,
    overwrite:bool=False,
    _dev:dict=None,
    silent:bool=None,
    **kwargs,
):
    """
    For all points in a DataFrame it searches for all other points (potentially of another DataFrame) within the specified radius and aggregate the values for specified column(s).
    It draws random the bounding box containing all points from DataFrame(s) and aggregate the values within the radius to obtain a random distribution.
    Then all points from DataFrame which exceed the k_th_percentile of the random distribution are labeld as clustered.
    The results will be appended to DataFrame.

    Args:
    -------
    pts (pandas.DataFrame):
        Points for which neighborhood aggregates are computed. Results are appended to this DataFrame in-place.
        Note: row order of ``pts`` may change after the call.
    crs (str):
        CRS of the coordinates in ``pts``, e.g. ``'EPSG:4326'``.
        Pass ``crs=''`` to skip reprojection entirely — use this when your
        coordinates are already in a Cartesian/projected plane and ``r`` is in
        the same units as ``x``/``y``.
    r (float or list or list of tuples):
        Search radius or multiple radii, or a (weighted) distance bands:

        - ``float`` — single radius in metres (or in ``x``/``y`` units when ``crs=''``).
        - ``[500, 750, 1000]`` — list of radii; produces one result column per radius
          named ``{col}_{stat}_{r}`` (e.g. ``employment_sum_750``).
        - ``[(0,500), (500,750)]`` — distance bands ``(r_inner, r_outer)``; produces one
          column per band named ``{col}_{stat}_{r_in}_{r_out}`` (e.g. ``employment_sum_0_500``).
        - ``[(0,500,1), (500,750,2)]`` — weighted bands; weights are normalised and the
          single weighted aggregate is named ``{col}_{stat}_wgt``.
          Intermediate band columns are dropped unless ``keep_cols=True``.

        For distance bands only additive statistics (``sum``, ``count``) are strictly
        correct; other stats are subtracted numerically.
    c (str or list):
        Column name or list of column names to aggregate within the search radius.
        If empty or None, points within the radius are counted.
        Columns must exist in ``pts`` (or in ``pts_target`` if provided).
    stat (str or list):
        Statistic to compute within the search radius. One of ``'sum'``, ``'count'``,
        ``'mean'``, ``'variance'``, ``'std'``, ``'cv'``, ``'skewness'``, ``'kurtosis'``
        (default=``'sum'``). Pass a **list** to compute multiple statistics in a single
        search pass, e.g. ``stat=['sum', 'mean', 'variance']``. When used with
        ``detect_cluster_pts`` / ``detect_cluster_cells`` the **first** stat in the list
        drives cluster detection; any additional stats are appended to the output grid as
        extra cell-level aggregates.
    exclude_self (bool):
        If True, each point's own value is subtracted from its radius aggregate (default=True).
        Formerly ``exclude_pt_itself`` (deprecated).
    spacing (float):
        Output cell size, in the same unit as ``r`` (metres after reprojection). Controls the
        resolution of the output grid used for exports and plots — NOT the internal search grid,
        whose cell size is chosen automatically for speed. Per-point aggregates are exact
        regardless of this value, so a finer output grid is well-defined. When None, defaults to
        ``r/3`` (default=None).
    sample_area (shapely.Polygon | shapely.MultiPolygon | str):
        Area used for drawing random comparison points. Accepted string values:
            - ``'buff_non_empty_cells'``: non-empty grid cells plus a radius-sized buffer (default)
            - ``'buf_cells_min_pts'``: grid cells with at least 'min_pts_to_sample_cell' plus a radius-sized buffer (default)
            - ``'concave'``: concave hull around points
            - ``'convex'``: convex hull around points
            - ``'buffer'``: buffer around individual points (slow for large datasets)
            - ``'bounding_box'``: axis-aligned bounding box
            - ``'grid'`` or ``None``: full grid extent
        Alternatively pass any Shapely ``Polygon`` or ``MultiPolygon`` directly. The geometry
        is assumed to be in ``crs`` and is reprojected to ``local_crs`` automatically. If your
        geometry is already in the projected CRS (``local_crs``), pass ``crs=local_crs`` to
        prevent a double reprojection.
        See ``infer_sample_area_from_pts`` for finer control (default='buff_non_empty_cells').
    min_pts_to_sample_cell (int):
        Minimum number of data points a grid cell must contain for random points to be drawn in it (default=0).
    weight_valid_area (str):
        If set to ``'estimate'`` or ``'precise'`` the radius aggregate will be weighted
        inversely by the share of valid area within the search radius. ``'precise'`` is
        exact but slow; ``'estimate'`` uses a fast logit approximation (MSE ≈ 5 % of cell
        area). ``None`` disables weighting (default=None).

        **Limitation:** contained cells (fully inside the search disk) are treated as binary —
        fully valid or fully invalid — based on whether they fall inside ``sample_area``.
        Cells that straddle the ``sample_area`` boundary but were classified as fully valid
        contribute zero invalid area, causing a slight upward bias in ``valid_area_share``
        when ``sample_area`` is a custom Polygon or MultiPolygon with sharp edges. For
        method-string sample areas (``'buff_non_empty_cells'``, ``'concave'``, etc.) the
        boundary follows cell edges exactly and this bias does not occur.
    k_th_percentile (float):
        Percentile of the random distribution a point must exceed to be labelled as clustered (default=99.5).
    n_random_points (int):
        Number of random points drawn to build the comparison distribution (default=100000).
    random_seed (int):
        Random seed for reproducibility. None means no seed is set (default=None).
    null_distribution (array-like or pd.DataFrame, optional):
        User-supplied null-distribution coordinates. When provided the internal uniform-random draw
        is skipped and these coordinates are used as reference points for computing the null radius
        sums. Accepts a DataFrame with the same x/y column names used in the search, or a
        2-column array ``[[x, y], ...]``.

        **Important:** coordinates must be in the same projected CRS that ``pts`` is reprojected
        into (metres), not in the original input CRS. Use ``aabpl.draw_random_coords()`` with
        ``crs`` set to generate correctly projected coordinates.
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
    x (str):
        Column name of the x-coordinate (longitude) in ``pts`` (default=``'lon'``).
    y (str):
        Column name of the y-coordinate (latitude) in ``pts`` (default=``'lat'``).
    row_name (str):
        Name for the grid row-index column appended to ``pts`` (default=``'id_y'``).
    col_name (str):
        Name for the grid column-index column appended to ``pts`` (default=``'id_x'``).
    suffix (str):
        Suffix appended to each column name in ``c`` to form the radius-aggregate column names.
        When None (default), auto-generates ``'_{stat}_{r}'`` (e.g. ``'_sum_750'`` for ``stat='sum', r=750``).
    cluster_suffix (str):
        Suffix appended to each column name in ``c`` to form the boolean cluster indicator column names.
        When None (default), auto-generates ``'_cluster_{stat}_{r}'`` (e.g. ``'_cluster_sum_750'``).
    proj_crs (str):
        Metric CRS used internally. ``'auto'`` selects the appropriate UTM zone from the data extent.
        Pass an explicit EPSG string (e.g. ``'EPSG:32632'``) to override, or ``None`` to skip
        reprojection (default=``'auto'``).
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
    plot_cluster_points (dict):
        Kwargs for a map of the identified cluster points. None disables it (default=None).
    keep_cols (bool):
        If False, intermediate columns added during processing (grid indices, offsets, proj x+y, etc.)
        are removed from ``pts`` before returning. If None proj x+y are retained. If True they are retained (default=False).
    overwrite (bool):
        If True, existing output columns with the same names are overwritten. Raises ValueError
        when False and a collision is detected (default=False).
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
        aggregates ``{c}{suffix}`` are appended to ``pts``.
    """
    if 'min_pts_to_sample_cell' in kwargs:
        min_pts_to_sample_cell = kwargs.pop('min_pts_to_sample_cell')
        progress_print(
            "DeprecationWarning: min_pts_to_sample_cell= is deprecated. Pass it inline via sample_area= instead, "
            "e.g. sample_area='buff_cells,min_pts=1'. "
            "Call resolve_sample_area.params() for all options."
        )
    else:
        min_pts_to_sample_cell = 0
    x_tgt             = kwargs.pop('x_tgt', None)
    y_tgt             = kwargs.pop('y_tgt', None)
    row_name_tgt      = kwargs.pop('row_name_tgt', None)
    col_name_tgt      = kwargs.pop('col_name_tgt', None)
    exclude_pt_itself = kwargs.pop('exclude_pt_itself', None)
    queen_contingency, rook_contingency = _unpack_contingency(contingency)
    centroid_dist_threshold, border_dist_threshold = _unpack_merge_dist(merge_dist)
    (min_cluster_share_after_contingency,
     min_cluster_share_after_centroid_dist,
     min_cluster_share_after_convex) = _unpack_min_cluster_share(min_cluster_share)
    _dep_cluster = {k: kwargs.pop(k) for k in (
        'queen_contingency', 'rook_contingency',
        'centroid_dist_threshold', 'border_dist_threshold',
        'min_cluster_share_after_contingency',
        'min_cluster_share_after_centroid_dist',
        'min_cluster_share_after_convex',
    ) if k in kwargs}
    if _dep_cluster:
        progress_print(
            f"DeprecationWarning: kwarg(s) {sorted(_dep_cluster)} are deprecated. "
            "Use contingency=, merge_dist=, min_cluster_share= instead. See docstring."
        )
        if 'queen_contingency'                     in _dep_cluster: queen_contingency                     = int(_dep_cluster['queen_contingency'])
        if 'rook_contingency'                      in _dep_cluster: rook_contingency                      = int(_dep_cluster['rook_contingency'])
        if 'centroid_dist_threshold'               in _dep_cluster: centroid_dist_threshold               = _dep_cluster['centroid_dist_threshold']
        if 'border_dist_threshold'                 in _dep_cluster: border_dist_threshold                 = _dep_cluster['border_dist_threshold']
        if 'min_cluster_share_after_contingency'   in _dep_cluster: min_cluster_share_after_contingency   = float(_dep_cluster['min_cluster_share_after_contingency'])
        if 'min_cluster_share_after_centroid_dist' in _dep_cluster: min_cluster_share_after_centroid_dist = float(_dep_cluster['min_cluster_share_after_centroid_dist'])
        if 'min_cluster_share_after_convex'        in _dep_cluster: min_cluster_share_after_convex        = float(_dep_cluster['min_cluster_share_after_convex'])
    if kwargs:
        raise TypeError(f"detect_cluster_cells() got unexpected keyword argument(s): {sorted(kwargs)}")
    _cols_before = set(pts.columns)
    vk = _validate_kwargs(
        pts=pts, crs=crs, r=r, c=c, stat=stat,
        x=x, y=y, row_name=row_name, col_name=col_name, suffix=suffix, pts_target=pts_target, x_tgt=x_tgt, y_tgt=y_tgt,
        row_name_tgt=row_name_tgt, col_name_tgt=col_name_tgt,
        build_grid_obj=False,  # grid is built by the detect_cluster_pts call below
        proj_crs=proj_crs, silent=silent,
    )
    pts, local_crs = vk.pts, vk.local_crs
    c, x, y, suffix = vk.c, vk.x, vk.y, vk.suffix
    pts_target, x_tgt, y_tgt = vk.pts_target, vk.x_tgt, vk.y_tgt
    row_name_tgt, col_name_tgt, stat = vk.row_name_tgt, vk.col_name_tgt, vk.stat
    if suffix is None:
        suffix = _default_suffix(stat, r)
    if cluster_suffix is None:
        cluster_suffix = f'_cluster{_default_suffix(stat, r)}'
    if centroid_dist_threshold is None:
        centroid_dist_threshold = r * 10/3
    if border_dist_threshold is None:
        border_dist_threshold = r * 4/3

    grid = detect_cluster_pts(
        pts=pts,
        crs=local_crs if local_crs is not None else '',
        r=r,
        c=c,
        stat=stat,
        exclude_self=exclude_self,
        exclude_pt_itself=exclude_pt_itself,
        weight_valid_area=weight_valid_area,
        sample_area=sample_area,
        k_th_percentile=k_th_percentile,
        n_random_points=n_random_points,
        random_seed=random_seed,
        null_distribution=null_distribution,
        x=x,
        y=y,
        row_name=row_name,
        col_name=col_name,
        suffix=suffix,
        cluster_suffix=cluster_suffix,
        proj_crs=local_crs,
        pts_target=pts_target,
        x_tgt=x_tgt,
        y_tgt=y_tgt,
        row_name_tgt=row_name_tgt,
        col_name_tgt=col_name_tgt,
        spacing=spacing,
        plot_distribution=plot_distribution,
        plot_cluster_points=plot_cluster_points,
        keep_cols=True,  # cell ids (row_name/col_name) are needed for clustering below
        overwrite=overwrite,
        _dev=_dev,
        silent=silent,
    )
    # TODO: replace with actual output names once spacing/nesting changes column naming
    if not hasattr(grid, 'output_row_name'):
        grid.output_row_name = row_name
        grid.output_col_name = col_name

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
        # Cluster on the SEARCH grid: clusters.py looks cell totals up in
        # grid.id_to_sums, which is keyed by the search-cell ids (id_y/id_x). The
        # output grid (out_id_*) is for display/export only.
        row_name=row_name,
        col_name=col_name,
        cluster_suffix=cluster_suffix,
        )

    if not grid._silent:
        grid.update_spacing()
        nr, nc = len(grid.row_ids), len(grid.col_ids)
        n_nonempty = len(grid.id_to_sums)
        s = grid._search_spacing
        out_s  = grid.output_spacing
        out_sy = getattr(grid, 'output_spacing_y', None) or out_s
        ratio_x = max(1, round(out_s  / s))
        ratio_y = max(1, round(out_sy / s))
        n_clusters = sum(len(col_cl.clusters) for col_cl in grid.clustering.by_column.values())
        n_cluster_cells = sum(
            len(set((row // ratio_y, col // ratio_x) for row, col in cl.cells))
            for col_cl in grid.clustering.by_column.values()
            for cl in col_cl.clusters
        )
        progress_print(
            f'Built output grid: {nr}*{nc}={nr*nc} cells with spacing {grid.output_spacing} '
            f'({n_nonempty} non-empty) — {n_clusters} cluster(s) across {n_cluster_cells} output cells'
        )

    if not keep_cols:
        _cluster_id_suffix = cluster_suffix.replace('_cluster_', '_cluster_id_', 1)
        _outer_output_cols = (
            set(str(col)+suffix for col in c) |
            set(str(col)+cluster_suffix for col in c) |
            set(str(col)+_cluster_id_suffix for col in c)
        )
        _keep_extra = {grid.output_row_name, grid.output_col_name, x, y}
        _to_drop = [
            col for col in pts.columns
            if col not in _cols_before and col not in _outer_output_cols and col not in _keep_extra
        ]
        if _to_drop:
            pts.drop(columns=_to_drop, inplace=True)

    return grid
#
# detect_cluster_pts has the same parameter set — share its docs
detect_cluster_pts.params = detect_cluster_cells.params

@time_func_perf
def detect_cluster_cells_from_labeled_pts(
    pts:_pd_DataFrame,
    crs:str,
    r:float,
    c:list=[],
    is_cluster_column:str='cluster',
    cluster_suffix:str='_cluster',
    exclude_self:bool=True,
    exclude_pt_itself:bool=None,
    x:str='lon',
    y:str='lat',
    row_name:str='id_y',
    col_name:str='id_x',
    contingency=1,
    merge_dist=None,
    min_cluster_share=(0.0, 0.0, 0.0),
    make_convex:bool=True,
    spacing:float=None,
    suffix:str=None,
    proj_crs:str='auto',
    keep_cols:bool=False,
    overwrite:bool=False,
    silent:bool=None,
    **kwargs,
):
    """
    Build spatial cell-clusters from points that are ALREADY labelled as clustered.

    Unlike ``detect_cluster_cells`` this skips the radius search AND the random null
    distribution entirely. The caller supplies ``pts`` with a boolean column
    ``is_cluster_column`` (default ``'cluster'``) marking which points belong to a
    cluster. The points are assigned to grid cells, per-cell mass is aggregated for
    the column(s) in ``c`` (defaults to counting the clustered points), and contiguous
    clustered cells are merged into clusters and output exactly as in
    ``detect_cluster_cells``.

    Parameters mirror ``detect_cluster_cells`` minus all null-distribution arguments
    (``k_th_percentile``, ``n_random_points``, ``sample_area`` ...).

    Returns the Grid with cluster polygons at ``grid.clustering``.
    """
    queen_contingency, rook_contingency = _unpack_contingency(contingency)
    centroid_dist_threshold, border_dist_threshold = _unpack_merge_dist(merge_dist)
    (min_cluster_share_after_contingency,
     min_cluster_share_after_centroid_dist,
     min_cluster_share_after_convex) = _unpack_min_cluster_share(min_cluster_share)
    _dep_cluster = {k: kwargs.pop(k) for k in (
        'queen_contingency', 'rook_contingency',
        'centroid_dist_threshold', 'border_dist_threshold',
        'min_cluster_share_after_contingency',
        'min_cluster_share_after_centroid_dist',
        'min_cluster_share_after_convex',
    ) if k in kwargs}
    if _dep_cluster:
        progress_print(
            f"DeprecationWarning: kwarg(s) {sorted(_dep_cluster)} are deprecated. "
            "Use contingency=, merge_dist=, min_cluster_share= instead. See docstring."
        )
        if 'queen_contingency'                     in _dep_cluster: queen_contingency                     = int(_dep_cluster['queen_contingency'])
        if 'rook_contingency'                      in _dep_cluster: rook_contingency                      = int(_dep_cluster['rook_contingency'])
        if 'centroid_dist_threshold'               in _dep_cluster: centroid_dist_threshold               = _dep_cluster['centroid_dist_threshold']
        if 'border_dist_threshold'                 in _dep_cluster: border_dist_threshold                 = _dep_cluster['border_dist_threshold']
        if 'min_cluster_share_after_contingency'   in _dep_cluster: min_cluster_share_after_contingency   = float(_dep_cluster['min_cluster_share_after_contingency'])
        if 'min_cluster_share_after_centroid_dist' in _dep_cluster: min_cluster_share_after_centroid_dist = float(_dep_cluster['min_cluster_share_after_centroid_dist'])
        if 'min_cluster_share_after_convex'        in _dep_cluster: min_cluster_share_after_convex        = float(_dep_cluster['min_cluster_share_after_convex'])
    if kwargs:
        raise TypeError(f"detect_cluster_cells_from_labeled_pts() got unexpected keyword argument(s): {sorted(kwargs)}")
    if is_cluster_column not in pts.columns:
        raise ValueError(f"`is_cluster_column` '{is_cluster_column}' is not a column of pts.")
    # Without an explicit value column the cluster mass is the count of clustered
    # points per cell — aggregate the boolean label itself.
    if c is None or (not isinstance(c, str) and len(c) == 0):
        c = [is_cluster_column]

    _cols_before = set(pts.columns)
    init_sort = find_column_name('initial_sort', existing_columns=pts.columns)
    pts[init_sort] = range(len(pts))

    vk = _validate_kwargs(
        pts=pts, crs=crs, r=r, c=c, stat='sum',
        x=x, y=y, row_name=row_name, col_name=col_name, suffix=suffix,
        output_spacing=spacing, proj_crs=proj_crs, silent=silent,
    )
    pts, local_crs = vk.pts, vk.local_crs
    c, x, y, suffix = vk.c, vk.x, vk.y, vk.suffix
    pts_target, x_tgt, y_tgt = vk.pts_target, vk.x_tgt, vk.y_tgt
    row_name_tgt, col_name_tgt, grid = vk.row_name_tgt, vk.col_name_tgt, vk.grid
    orig_cols = list(c)
    _output_cols = set(str(col)+cluster_suffix for col in orig_cols)
    if not overwrite:
        _collision = _output_cols & _cols_before
        if _collision:
            raise ValueError(
                f"Output columns {sorted(_collision)} already exist in pts. "
                "Pass overwrite=True to overwrite them."
            )
    if centroid_dist_threshold is None:
        centroid_dist_threshold = r * 10/3
    if border_dist_threshold is None:
        border_dist_threshold = r * 4/3

    # Assign points to grid cells and pre-aggregate per-cell mass (used for cluster
    # totals). No radius search and no null distribution is performed.
    if exclude_pt_itself is not None:
        progress_print("DeprecationWarning: `exclude_pt_itself` is deprecated, use `exclude_self` instead.")
        exclude_self = exclude_pt_itself
    grid.search = DiskSearch(grid, r=r, exclude_self=exclude_self,
                             weight_valid_area=False)
    grid.search.set_target(pts=pts_target, c=c, x=x_tgt, y=y_tgt,
                           row_name=row_name_tgt, col_name=col_name_tgt, silent=silent)
    grid.search.set_source(pts=pts, c=c, x=x, y=y, row_name=row_name, col_name=col_name,
                           suffix=suffix, silent=silent)

    # Use the caller-provided boolean as the cluster label for every column in c.
    is_cluster = pts[is_cluster_column].astype(bool)
    for column in c:
        pts[str(column)+str(cluster_suffix)] = is_cluster

    if not hasattr(grid, 'output_row_name'):
        grid.output_row_name = row_name
        grid.output_col_name = col_name

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
        row_name=row_name,
        col_name=col_name,
        cluster_suffix=cluster_suffix,
    )

    if not keep_cols:
        _keep_extra = {grid.output_row_name, grid.output_col_name, init_sort, x, y}
        _to_drop = [
            col for col in pts.columns
            if col not in _cols_before and col not in _output_cols and col not in _keep_extra
        ]
        if _to_drop:
            pts.drop(columns=_to_drop, inplace=True)

    pts.sort_values(init_sort, inplace=True)
    pts.drop(columns=[init_sort], inplace=True)
    return grid

from functools import wraps as _wraps
from warnings import simplefilter
from pandas.errors import PerformanceWarning as _pd_PerformanceWarning
from pandas import DataFrame as _pd_DataFrame
from numpy import array as _np_array, nan as _np_nan, concatenate as _np_concatenate, zeros as _np_zeros, array_equal as _np_array_equal
from shapely.geometry import Polygon as _shapely_Polygon, MultiPolygon as _shapely_MultiPolygon, GeometryCollection as _shapely_GeometryCollection

simplefilter(action='ignore', category=_pd_PerformanceWarning)
simplefilter(action='ignore', category=FutureWarning)

from .search.study_area import (
    infer_study_area_from_pts, intersect_polygon_with_grid, intersect_polygon_with_grid_aw,
    resolve_study_area, _STUDY_AREA_SPEC_DOCS,
)
from .testing.test_performance import time_func_perf
from .search.algorithm.disk_search import DiskSearch
from .search.grid_class import Grid
from .utils.misc import find_column_name
from .utils.crs_transformation import convert_MultiPolygon_crs, convert_pts_to_crs, convert_wgs_to_utm
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

def _resolve_local_crs(pts, x, y, crs, proj_crs, pts_target=None, x_tgt=None, y_tgt=None):
    """Resolve the local (projected) CRS for a point set.

    ``proj_crs='auto'`` (default) picks the local UTM zone from the combined
    bounds of ``pts`` and (if given) ``pts_target``.
    """
    if crs is None:
        raise ValueError(
            "crs is required. Pass your coordinate reference system string (e.g. crs='EPSG:4326'), "
            "or pass crs='' to skip reprojection entirely when your coordinates are already in a "
            "Cartesian/projected plane and r is expressed in the same units as your coordinates."
        )
    if not crs:
        # Cartesian mode: coordinates are already in the target unit system; skip all reprojection.
        # r is interpreted directly in the same units as the x/y columns.
        return proj_crs if proj_crs != 'auto' else None
    if proj_crs != 'auto':
        return proj_crs
    _pts_tgt = pts_target if pts_target is not None else pts
    _x_tgt = x_tgt if x_tgt is not None else x
    _y_tgt = y_tgt if y_tgt is not None else y
    x_center = (min([pts[x].min(), _pts_tgt[_x_tgt].min()]) + max([pts[x].max(), _pts_tgt[_x_tgt].max()])) / 2
    y_center = (min([pts[y].min(), _pts_tgt[_y_tgt].min()]) + max([pts[y].max(), _pts_tgt[_y_tgt].max()])) / 2
    return 'EPSG:' + str(convert_wgs_to_utm(x_center, y_center))


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
        grid_bounds=(None, None, None, None),
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
    _nan_cols = [col for col in c if pts_target[col].isna().any()]
    if _nan_cols:
        progress_print(
            f"Warning: column(s) {_nan_cols} contain NaN values. "
            "The current implementation does not distinguish NaN from valid values: "
            "NaN points still contribute 1 to count and may contaminate sums. "
            "Consider filling or dropping NaN rows before calling radius_search."
        )
    if not x_tgt in pts_target.columns:
        raise ValueError('`x_tgt` (x-coord column name) must be in columns of pts_target')
    if not y_tgt in pts_target.columns:
        raise ValueError('`y_tgt` (y-coord column name) must be in columns of pts_target')
    local_crs = _resolve_local_crs(pts, x, y, crs, proj_crs, pts_target, x_tgt, y_tgt)
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
            grid_bounds=grid_bounds,
            silent=silent,
        )
    else:
        # Caller (detect_cluster_cells) only needs the validated/reprojected kwargs;
        # the grid is built by the detect_cluster_pts it delegates to. Skip the
        # throwaway build here.
        grid = None

    return _SearchParams(pts, local_crs, c, x, y, suffix, pts_target, x_tgt, y_tgt, row_name_tgt, col_name_tgt, grid, stat)
#


from .cluster.params import (
    unpack_contingency as _unpack_contingency,
    unpack_merge_dist as _unpack_merge_dist,
    unpack_min_cluster_share as _unpack_min_cluster_share,
)

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
    grid_bounds=(None, None, None, None),
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
        grid_bounds=grid_bounds,
        silent=silent,
    )
#


def build_study_area(
    pts:_pd_DataFrame,
    r:float,
    crs:str,
    study_area='cells,m=1,b=0',
    x:str='lon',
    y:str='lat',
    proj_crs:str='auto',
    area_weight:str=None,
    silent:bool=None,
):
    """
    Resolve a study-area spec into a Shapely polygon, without running a full search.

    Builds the grid a cell/hull-based ``study_area`` method needs internally, so
    callers only ever pass ``pts``/``r``/``crs`` — never a ``grid``. Useful when you
    want to inspect or edit the study area (e.g. cut out a sub-region) before
    feeding it back into ``detect_cluster_cells``/``detect_cluster_pts``/
    ``radius_search`` via their own ``study_area=`` argument.

    Args:
    -------
    pts (pandas.DataFrame):
        Points used to build the grid and (for cell/hull-based methods) the
        study-area footprint.
    r (float):
        Search radius in metres (or in ``x``/``y`` units when ``crs=''``). Only
        used to size the grid and, for the ``'cells'`` method, to warn if a
        requested buffer would make a downstream ``area_weight`` a no-op.
    crs (str):
        CRS of the coordinates in ``pts``, e.g. ``'EPSG:4326'``. Pass ``crs=''``
        to skip reprojection when coordinates are already Cartesian/projected.
    study_area (str or shapely.Polygon or shapely.MultiPolygon):
        Study-area spec string (e.g. ``'cells,min_pts=1'``, ``'concave,buffer=500'``),
        an existing Shapely Polygon/MultiPolygon in the same projected CRS, or
        ``False`` for the full grid bounding box. Call
        ``aabpl.build_study_area.params()`` to print all spec options.
    x, y (str):
        x/y-coordinate column names in ``pts`` (default ``'lon'``/``'lat'``).
    proj_crs (str):
        Target projected CRS for internal computation. ``'auto'`` (default)
        picks the local UTM zone.
    area_weight (str):
        Pass the same ``area_weight`` you intend to use downstream (e.g.
        ``'exact'``) to get a warning if the study-area buffer would make it a
        no-op. ``None`` (default) skips that check.
    silent (bool):
        Suppress progress output.

    Returns:
    -------
    study_area (shapely.Polygon or shapely.MultiPolygon):
        The resolved study-area polygon, in the projected CRS.

    Example:
    -------
    >>> poly = aabpl.build_study_area(pts, r=750, crs='EPSG:4326', study_area='cells,min_pts=1')
    >>> poly = poly.difference(area_to_exclude)  # user edits the polygon
    >>> aabpl.detect_cluster_cells(pts=pts, r=750, crs='EPSG:4326', study_area=poly)
    """
    local_crs = _resolve_local_crs(pts, x, y, crs, proj_crs)
    if crs and crs != local_crs:
        x, y, local_crs = convert_pts_to_crs(pts=pts, x=x, y=y, initial_crs=crs, target_crs=proj_crs)
    grid = build_grid(
        pts_source=pts, initial_crs=local_crs, local_crs=local_crs,
        data_crs=crs or None, r=r, x=x, y=y, silent=silent,
    )
    return resolve_study_area(
        pts=pts, r=r, study_area=study_area,
        crs=crs, local_crs=local_crs, x=x, y=y,
        grid=grid, min_pts_to_sample_cell=0,
        area_weight=area_weight,
    )
build_study_area.params = resolve_study_area.params


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

def _r_scalar(r):
    """Return a single representative radius from any r spec (scalar, list, bands, wbands)."""
    if not isinstance(r, (list, tuple)):
        return float(r)
    first = r[0]
    return max(b[1] for b in r) if isinstance(first, (list, tuple)) else max(float(v) for v in r)


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
    exclude_self:bool=False,
    proj_crs:str='auto',
    suffix=None,
    overwrite:bool=False,
    area_weight:str=None,
    cell_size:float=None,
    study_area=False,
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
    **kwargs,
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
        search pass, e.g. ``stat=['sum', 'mean', 'variance']``.
    exclude_self (bool):
        If True, each point's own value is subtracted from its radius aggregate (default=False).
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
    area_weight (str):
        Inverse-area weighting for edge effects. Scales the radius aggregate by the inverse
        of the share of valid (inside ``study_area``) area within each point's search disk.

        Variants:

        - ``'exact'``: per-point Shapely disk∩study_area intersection. Scales **linearly
          with the number of points** (O(n)), making it the recommended choice for most
          datasets.
        - ``'logit'``: per-cell logit approximation using precomputed cell-boundary fractions.
          Fast setup, but overhead per point is similar to ``'exact'`` at typical densities.
        - ``'flat'``: like ``'logit'`` but uses a single average fraction for all boundary
          cells instead of per-cell values. Slightly less accurate.
        - ``'binary'``: treats each cell as fully inside or fully outside the study area —
          no fractional contribution. Fastest, but least accurate near boundaries.
        - ``None``: disables weighting (default).

        Append ``',keep_raw'`` to any variant (e.g. ``'exact,keep_raw'``) to retain the
        unweighted aggregate in an additional ``{col}_raw`` column alongside the weighted
        result.

        Append ``',quad_segs=N'`` to ``'exact'`` to control the Shapely disk approximation
        quality (default 8).

        Aliases ``'precise'`` → ``'exact'`` and ``'estimate'`` → ``'logit'`` are accepted
        for backwards compatibility.

        **Limitation:** contained cells (fully inside the search disk) are treated as binary —
        fully valid or fully invalid — based on whether they fall inside ``study_area``.
        Cells that straddle the ``study_area`` boundary but were classified as fully valid
        contribute zero invalid area, causing a slight upward bias in ``valid_area_share``
        when ``study_area`` is a custom Polygon or MultiPolygon with sharp edges. For
        method-string study areas (``'buff_non_empty_cells'``, ``'concave'``, etc.) the
        boundary follows cell edges exactly and this bias does not occur.
    cell_size (float):
        Output cell size, in the same unit as ``r`` (metres after reprojection). Controls the
        resolution of the output grid used for exports and plots — NOT the internal search grid,
        whose cell size is chosen automatically for speed. Per-point aggregates are exact
        regardless of this value, so a finer output grid is well-defined. When None, defaults to
        ``r/3`` (default=None).
    study_area (shapely.Polygon | shapely.MultiPolygon | str | False):
        Study area polygon used for valid-area weighting (only relevant when
        ``area_weight`` is set). ``False`` disables study-area processing entirely
        and is the correct default when no edge-effect weighting is needed (default=False).
        Accepted string values:
            - ``'buff_cells,min_pts=1'``: grid cells with at least 1 point plus a radius-sized buffer
            - ``'buff_non_empty_cells'``: non-empty grid cells plus a radius-sized buffer
            - ``'concave'``: concave hull around points
            - ``'convex'``: convex hull around points
            - ``'buffer'``: buffer around individual points (slow for large datasets)
            - ``'bounding_box'``: axis-aligned bounding box
            - ``'grid'`` or ``None``: full grid extent
        Alternatively pass any Shapely ``Polygon`` or ``MultiPolygon`` directly. The geometry
        is assumed to be in ``crs`` and is reprojected to the projected CRS automatically.
        Append ``,min_pts=N`` to any string option to require at least N data points per cell.
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
    # backward-compat: spacing= was renamed to cell_size=
    if 'spacing' in kwargs:
        import warnings
        warnings.warn("'spacing' is deprecated, use 'cell_size' instead.", DeprecationWarning, stacklevel=2)
        if cell_size is None:
            cell_size = kwargs.pop('spacing')
        else:
            kwargs.pop('spacing')
    # Backwards-compat aliases
    _dep_rs = {k: kwargs.pop(k) for k in ('weight_valid_area', 'sample_area') if k in kwargs}
    if _dep_rs:
        progress_print(
            f"DeprecationWarning: kwarg(s) {sorted(_dep_rs)} are deprecated. "
            "Use area_weight= and study_area= instead. See docstring."
        )
        if 'weight_valid_area' in _dep_rs and area_weight is None:
            area_weight = _dep_rs['weight_valid_area']
        if 'sample_area' in _dep_rs and study_area is None:
            study_area = _dep_rs['sample_area']
    if kwargs:
        raise TypeError(f"radius_search() got unexpected keyword argument(s): {sorted(kwargs)}")
    # ── multi-radius delegation ───────────────────────────────────────────────
    # When r is not a scalar (list of radii or distance bands), hand off to the
    # dedicated implementation which calls radius_search once per unique radius.
    from .search.multi_radius import _parse_r_spec, _multi_radius_search
    spec_type, spec_data = _parse_r_spec(r)
    if spec_type == 'single':
        # Unwrap single-element lists/tuples (e.g. r=[750]) to the plain
        # scalar _parse_r_spec already resolved them to -- previously this
        # branch fell through without reassigning r, so a single-element list
        # still failed the scalar check below with a confusing error even
        # though _parse_r_spec had already correctly identified it as 'single'.
        r = spec_data
    if spec_type != 'single':
        return _multi_radius_search(
            pts=pts, r=r, c=c, x=x, y=y, stat=stat,
            suffix=suffix, keep_cols=keep_cols,
            exclude_self=exclude_self, silent=silent,
            _radius_search_fn=radius_search,
            _parsed_spec=(spec_type, spec_data),
            crs=crs, proj_crs=proj_crs,
            study_area=study_area,
            spacing=cell_size, row_name=row_name, col_name=col_name,
            pts_target=pts_target, x_tgt=x_tgt, y_tgt=y_tgt,
            row_name_tgt=row_name_tgt, col_name_tgt=col_name_tgt,
            area_weight=area_weight, overwrite=overwrite,
            _dev=_dev,
        )
    # ── end multi-radius ──────────────────────────────────────────────────────

    # ── r validation ─────────────────────────────────────────────────────────
    if not (isinstance(r, (int, float)) and r > 0):
        raise ValueError(f"r must be a positive number, got r={r!r}.")
    # ─────────────────────────────────────────────────────────────────────────

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
        output_spacing=cell_size,
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
    grid.cell_row_name = row_name
    grid.cell_col_name = col_name

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
        if area_weight:
            _output_cols.add(f'valid_area_share_{r}')
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
        if area_weight:
            _output_cols.add(f'valid_area_share_{r}')

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
        import warnings; warnings.warn("'exclude_pt_itself' is deprecated, use 'exclude_self' instead.", DeprecationWarning, stacklevel=2)
        exclude_self = exclude_pt_itself

    # initialize disk_search
    DiskSearch(
        grid=grid,
        r=r,
        exclude_self=exclude_self,
        area_weight=area_weight,
    )

    if not _is_internal:
        _prog.step("assigning target")
    # prepare target points data
    grid._search_class.set_target(
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
    grid._search_class.set_source(
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
    grid.study_area = resolve_study_area(
        pts=pts,r=r,study_area=study_area,
        crs=crs,local_crs=local_crs,x=x,y=y,
        grid=grid, min_pts_to_sample_cell=0,
        area_weight=area_weight)
    if study_area is not False:
        intersect_polygon_with_grid_aw(grid, area_weight)

    if not _is_internal:
        _prog.step("searching")
    disk_sums_for_pts = grid._search_class.perform_search(silent=False if silent is None else silent,plot_pt_disk=_d.get('plot_pt_disk'))

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
        # count_helper_col (raw 'count') is a helper added to pts_target, not
        # pts (see 'if stat in ['count', 'mean']:' above) -- dropping it from
        # pts only ever worked by accident when pts_target is pts (self-search,
        # same object). errors='ignore' since the result column here is
        # radius_count_col (count_helper_col + suffix), already handled above;
        # this only cleans up the raw source-side helper, matching the
        # 'count'/_MOMENT_AGGS branches below.
        pts_target.drop(columns=[count_helper_col], inplace=True, errors='ignore')
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
            _keep_extra = {grid.cell_row_name, grid.cell_col_name, x, y}
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

    # Store per-column provenance so plot functions can decide whether to show
    # the radius indicator and what r value(s) to use.
    # Skip temporary intermediate columns (those written by _multi_radius_search
    # with the __mr__ marker) — they are cleaned up before the user sees them.
    from .search.multi_radius import _TEMP_COL_MARKER
    if not hasattr(grid, '_aabpl_col_meta'):
        grid._aabpl_col_meta = {}
    for col in orig_cols:
        out_col = col + suffix
        if _TEMP_COL_MARKER not in out_col:
            grid._aabpl_col_meta[out_col] = {'c': col, 'stat': stat, 'r': r}

    return grid
#

@_wraps(radius_search)
def radius_sum(pts, crs:str, r:float, c:list=[], suffix=None, **kwargs):
    return radius_search(pts=pts, crs=crs, r=r, c=c, stat='sum', suffix=suffix, **kwargs)
radius_sum.__doc__ = "Sum values of neighbouring points within radius r. Wraps ``radius_search(stat='sum')``.\n\n" + (radius_search.__doc__ or "")

@_wraps(radius_search)
def radius_count(pts, crs:str, r:float, c:list=[], suffix=None, **kwargs):
    return radius_search(pts=pts, crs=crs, r=r, c=c, stat='count', suffix=suffix, **kwargs)
radius_count.__doc__ = "Count neighbouring points within radius r. Wraps ``radius_search(stat='count')``.\n\n" + (radius_search.__doc__ or "")

@_wraps(radius_search)
def radius_mean(pts, crs:str, r:float, c:list=[], suffix=None, **kwargs):
    return radius_search(pts=pts, crs=crs, r=r, c=c, stat='mean', suffix=suffix, **kwargs)
radius_mean.__doc__ = "Mean of neighbouring point values within radius r. Wraps ``radius_search(stat='mean')``.\n\n" + (radius_search.__doc__ or "")

@_wraps(radius_search)
def radius_variance(pts, crs:str, r:float, c:list=[], suffix=None, **kwargs):
    return radius_search(pts=pts, crs=crs, r=r, c=c, stat='variance', suffix=suffix, **kwargs)
radius_variance.__doc__ = "Variance of neighbouring point values within radius r. Wraps ``radius_search(stat='variance')``.\n\n" + (radius_search.__doc__ or "")

@_wraps(radius_search)
def radius_std(pts, crs:str, r:float, c:list=[], suffix=None, **kwargs):
    return radius_search(pts=pts, crs=crs, r=r, c=c, stat='std', suffix=suffix, **kwargs)
radius_std.__doc__ = "Standard deviation of neighbouring point values within radius r. Wraps ``radius_search(stat='std')``.\n\n" + (radius_search.__doc__ or "")

@_wraps(radius_search)
def radius_cv(pts, crs:str, r:float, c:list=[], suffix=None, **kwargs):
    return radius_search(pts=pts, crs=crs, r=r, c=c, stat='cv', suffix=suffix, **kwargs)
radius_cv.__doc__ = "Coefficient of variation of neighbouring point values within radius r. Wraps ``radius_search(stat='cv')``.\n\n" + (radius_search.__doc__ or "")

@_wraps(radius_search)
def radius_skewness(pts, crs:str, r:float, c:list=[], suffix=None, **kwargs):
    return radius_search(pts=pts, crs=crs, r=r, c=c, stat='skewness', suffix=suffix, **kwargs)
radius_skewness.__doc__ = "Skewness of neighbouring point values within radius r. Wraps ``radius_search(stat='skewness')``.\n\n" + (radius_search.__doc__ or "")

@_wraps(radius_search)
def radius_kurtosis(pts, crs:str, r:float, c:list=[], suffix=None, **kwargs):
    return radius_search(pts=pts, crs=crs, r=r, c=c, stat='kurtosis', suffix=suffix, **kwargs)
radius_kurtosis.__doc__ = "Excess kurtosis of neighbouring point values within radius r. Wraps ``radius_search(stat='kurtosis')``.\n\n" + (radius_search.__doc__ or "")


from .cluster.detection import _detect_cluster_pts_multi


@attach_params
@time_func_perf
def detect_cluster_pts(
    pts:_pd_DataFrame,
    crs:str,
    r,          # float | list | bands | weighted bands — see radius_search.params.r
    c:list=[],
    x:str='lon',
    y:str='lat',
    stat:str=['sum','count','mean','variance','std','cv','skewness','kurtosis'][0],
    exclude_self:bool=True,
    cell_size:float=None,
    study_area='buff_cells,min_pts=1',
    area_weight:str=None,
    k_th_percentile:float=99.5,
    null_distribution=int(1e5),
    random_seed:int=None,
    proj_crs:str='auto',
    row_name:str='id_y',
    col_name:str='id_x',
    suffix:str=None,
    pts_target:_pd_DataFrame=None,
    keep_cols:bool=False,
    overwrite:bool=False,
    exclude_pt_itself:bool=None,   # deprecated alias for exclude_self
    silent:bool=None,
    **kwargs,
):
    """
    Compute radius aggregates for all points, compare them against a null distribution of
    randomly placed points within the study area, and label each **point** above the k-th
    percentile as clustered. A boolean column ``{c}{cluster_suffix}`` is appended to ``pts``
    in-place. Use ``detect_cluster_cells`` instead to get spatial cluster polygons.

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
        Search radius or multiple radii, or (weighted) distance bands.
        Same format as ``radius_search`` — see its docstring for full details.
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
        (default=``'sum'``). The **first** stat drives cluster detection; additional stats
        are appended as extra columns.
    exclude_self (bool):
        If True, each point's own value is subtracted from its radius aggregate (default=True).
    cell_size (float):
        Output cell size in metres (or coordinate units when ``crs=''``). Controls the output
        grid resolution — NOT the internal search grid. Defaults to ``r/3`` (default=None).
    study_area (shapely.Polygon | shapely.MultiPolygon | str):
        Study area from which random null-distribution points are drawn. Accepted string values:
            - ``'buff_cells,min_pts=1'``: grid cells with at least 1 point plus a radius-sized buffer (default)
            - ``'buff_non_empty_cells'``: non-empty grid cells plus a radius-sized buffer
            - ``'concave'``: concave hull around points
            - ``'convex'``: convex hull around points
            - ``'buffer'``: buffer around individual points (slow for large datasets)
            - ``'bounding_box'``: axis-aligned bounding box
            - ``'grid'`` or ``None``: full grid extent
        Alternatively pass any Shapely ``Polygon`` or ``MultiPolygon`` directly.
        Append ``,min_pts=N`` to any string option to require at least N data points per cell.
    area_weight (str):
        Inverse-area weighting for edge effects. See ``radius_search`` for full variant
        documentation. ``'exact'`` scales linearly with the number of points and is the
        recommended choice. ``None`` disables weighting (default=None).
    k_th_percentile (float):
        Percentile of the null distribution a point must exceed to be labelled as clustered (default=99.5).
    null_distribution (int | numpy.ndarray | pandas.DataFrame):
        Controls the null distribution used for cluster detection (default ``100_000``):

        - **int**: draw this many points uniformly at random within ``study_area``.
        - **numpy.ndarray of shape (N, 2)**: use these coordinates directly — **first column x,
          second column y**, both in the projected CRS (metres).
        - **pandas.DataFrame**: treated as an (N, 2) array via ``.values`` — **first column x,
          second column y** (column names are ignored).

        **Important for coordinate inputs:** coordinates must be in the projected CRS (metres),
        not the original input CRS. Use ``aabpl.draw_random_coords()`` with ``grid.study_area``
        to generate compatible coordinates.
    random_seed (int):
        Random seed for reproducibility. None means no seed is set (default=None).
    proj_crs (str):
        Metric CRS used internally. ``'auto'`` selects the appropriate UTM zone from the data extent.
        Pass an explicit EPSG string (e.g. ``'EPSG:32632'``) to override (default=``'auto'``).
    row_name (str):
        Name for the grid row-index column appended to ``pts`` (default=``'id_y'``).
    col_name (str):
        Name for the grid column-index column appended to ``pts`` (default=``'id_x'``).
    suffix (str):
        Suffix for the radius-aggregate column names. Defaults to ``'_{stat}_{r}'`` (default=None).
    pts_target (pandas.DataFrame):
        Points to aggregate over. If None, ``pts`` is used as both source and target (default=None).
    keep_cols (bool):
        If False, intermediate columns are removed from ``pts`` before returning (default=False).
    overwrite (bool):
        If True, existing output columns are overwritten. Raises ValueError on collision when False (default=False).
    silent (bool):
        If True, suppresses all progress output (default=None).

    Returns:
    -------
    grid (aabpl.Grid):
        The Grid object used for the search. Boolean cluster columns ``{c}_cluster_{stat}_{r}``
        and radius aggregates ``{c}{suffix}`` are appended to ``pts``.
    """
    # backward-compat: spacing= was renamed to cell_size=
    if 'spacing' in kwargs:
        import warnings
        warnings.warn("'spacing' is deprecated, use 'cell_size' instead.", DeprecationWarning, stacklevel=2)
        if cell_size is None:
            cell_size = kwargs.pop('spacing')
        else:
            kwargs.pop('spacing')
    # backward-compat: n_random_points= merged into null_distribution=
    if 'n_random_points' in kwargs:
        import warnings
        warnings.warn("'n_random_points' is deprecated, pass an integer to 'null_distribution' instead.", DeprecationWarning, stacklevel=2)
        if isinstance(null_distribution, int):
            null_distribution = kwargs.pop('n_random_points')
        else:
            kwargs.pop('n_random_points')
    if 'min_pts_to_sample_cell' in kwargs:
        min_pts_to_sample_cell = kwargs.pop('min_pts_to_sample_cell')
        progress_print(
            "DeprecationWarning: min_pts_to_sample_cell= is deprecated. Pass it inline via study_area= instead, "
            "e.g. study_area='buff_cells,min_pts=1'. "
            "Call resolve_study_area.params() for all options."
        )
    else:
        min_pts_to_sample_cell = 0
    _dev = kwargs.pop('_dev', None)
    grid_bounds = kwargs.pop('grid_bounds', (None, None, None, None))
    _max_output_cells = kwargs.pop('max_output_cells', None)
    _cluster_suffix_internal = kwargs.pop('_cluster_suffix', None)  # passed by detect_cluster_cells
    if _cluster_suffix_internal is not None:
        cluster_suffix = _cluster_suffix_internal
    elif 'cluster_suffix' in kwargs:
        import warnings
        warnings.warn("'cluster_suffix' is deprecated as a function argument. It will be removed in a future version.", DeprecationWarning, stacklevel=2)
        cluster_suffix = kwargs.pop('cluster_suffix', None)
    else:
        cluster_suffix = None
    if 'plot_distribution' in kwargs:
        import warnings
        warnings.warn("'plot_distribution' is deprecated as a function argument. Call grid.plot.rand_dist() on the returned grid instead.", DeprecationWarning, stacklevel=2)
        kwargs.pop('plot_distribution')
    if 'plot_cluster_points' in kwargs:
        import warnings
        warnings.warn("'plot_cluster_points' is deprecated as a function argument. Call grid.plot.cluster_pts() on the returned grid instead.", DeprecationWarning, stacklevel=2)
        kwargs.pop('plot_cluster_points')
    x_tgt         = kwargs.pop('x_tgt', None)
    y_tgt         = kwargs.pop('y_tgt', None)
    row_name_tgt  = kwargs.pop('row_name_tgt', None)
    col_name_tgt  = kwargs.pop('col_name_tgt', None)
    # _parsed_spec is an internal shortcut passed by detect_cluster_cells so that
    # _detect_cluster_pts_multi can skip re-parsing r.
    _parsed_spec  = kwargs.pop('_parsed_spec', None)
    if kwargs:
        raise TypeError(f"detect_cluster_pts() got unexpected keyword argument(s): {sorted(kwargs)}")
    # ── multi-radius delegation ───────────────────────────────────────────────
    from .search.multi_radius import _parse_r_spec
    spec_type, spec_data = _parse_r_spec(r)
    if spec_type == 'single':
        # Unwrap single-element lists/tuples (e.g. r=[750]) to the plain
        # scalar _parse_r_spec already resolved them to.
        r = spec_data
    if spec_type != 'single':
        # Use the already-parsed spec if detect_cluster_cells passed it through.
        effective_parsed_spec = _parsed_spec if _parsed_spec is not None else (spec_type, spec_data)
        return _detect_cluster_pts_multi(
            pts=pts, crs=crs, r=r, c=c, x=x, y=y, stat=stat,
            exclude_self=exclude_self, cell_size=cell_size,
            study_area=study_area, area_weight=area_weight,
            k_th_percentile=k_th_percentile, null_distribution=null_distribution,
            random_seed=random_seed, proj_crs=proj_crs,
            row_name=row_name, col_name=col_name,
            cluster_suffix=cluster_suffix, pts_target=pts_target,
            keep_cols=keep_cols, overwrite=overwrite, silent=silent,
            parsed_spec=effective_parsed_spec,
            min_pts_to_sample_cell=min_pts_to_sample_cell,
            _dev=_dev, grid_bounds=grid_bounds,
            x_tgt=x_tgt, y_tgt=y_tgt,
            row_name_tgt=row_name_tgt, col_name_tgt=col_name_tgt,
        )
    # ── single-radius path ────────────────────────────────────────────────────

    _cols_before = set(pts.columns)
    init_sort = find_column_name('initial_sort', existing_columns=pts.columns)
    pts[init_sort] = range(len(pts))

    vk = _validate_kwargs(
        pts=pts, crs=crs, r=r, c=c, stat=stat,
        x=x, y=y, row_name=row_name, col_name=col_name, suffix=suffix,
        pts_target=pts_target, x_tgt=x_tgt, y_tgt=y_tgt, row_name_tgt=row_name_tgt, col_name_tgt=col_name_tgt,
        output_spacing=cell_size,
        # the null distribution searches extra source pts over the same grid
        n_pts_src_extra=null_distribution if isinstance(null_distribution, int) else 0,
        grid_bounds=grid_bounds,
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
        import warnings; warnings.warn("'exclude_pt_itself' is deprecated, use 'exclude_self' instead.", DeprecationWarning, stacklevel=2)
        exclude_self = exclude_pt_itself

    # initialize disk_search
    DiskSearch(
        grid,
        r=r,
        exclude_self=exclude_self,
        area_weight=area_weight,
    )

    # pts_target defaults to pts itself (self-search) -- resolved here (not left
    # to the nested radius_search call below) so we always pass an EXPLICIT
    # target through: the nested call's own pts_target=None default would
    # default to the *combined* (real + null-sample) source we build below,
    # which is wrong -- null-distribution sample points must never appear as
    # a target, only as extra query locations.
    if pts_target is None:
        pts_target = pts

    grid.study_area = resolve_study_area(
        pts=pts, r=r,
        study_area=study_area, crs=crs, local_crs=local_crs, x=x, y=y, grid=grid,
        min_pts_to_sample_cell=min_pts_to_sample_cell,
        area_weight=area_weight)
    intersect_polygon_with_grid_aw(grid, area_weight)

    if not _is_internal: _prog.step("null distribution")
    if not isinstance(null_distribution, int) and local_crs and local_crs != crs:
        progress_print(
            "WARNING: null_distribution coordinates must already be in the projected CRS "
            f"'{local_crs}', not the input CRS '{crs}'. "
            "pts were reprojected automatically but null_distribution is used as-is. "
            "Use draw_random_coords() with the projected study_area to generate valid coordinates."
        )
    # Draw the null-distribution sample coordinates only (no search yet), then
    # fold them into ONE combined search alongside the real source points --
    # replaces the old flow of two separate searches (real vs pts_target, then
    # random-sample vs pts hardcoded), which (a) paid chunk-planning/grid setup
    # overhead twice and (b) silently ignored a caller-supplied pts_target in
    # the null-distribution half (it always searched against `pts`, not
    # `pts_target`). One combined search, target set once, fixes both.
    from .search.null_distribution import draw_null_distribution_points, compute_null_thresholds
    rndm_pts = draw_null_distribution_points(
        grid=grid,
        study_area=grid.study_area,
        min_pts_to_sample_cell=min_pts_to_sample_cell,
        null_distribution=null_distribution,
        x=x,
        y=y,
        random_seed=random_seed,
    )
    n_real = len(pts)

    # Combined source = real points followed by null-distribution sample
    # points. Sample rows get 0 in every c-column: they're synthetic query
    # locations, not real data, so they never contribute their own value to
    # anything -- and it makes the exclude_self subtraction below a correct
    # no-op for them without needing separate per-row logic.
    _combined_src = _pd_DataFrame({
        x: _np_concatenate([pts[x].values, rndm_pts[x].values]),
        y: _np_concatenate([pts[y].values, rndm_pts[y].values]),
    })
    for col in c:
        _combined_src[col] = _np_concatenate([pts[col].values, _np_zeros(len(rndm_pts))])

    if not _is_internal: _prog.step("assigning source")
    if not _is_internal: _prog.step("searching")
    # Delegate the actual aggregation to the real radius_search() -- the same
    # public entry point users call directly -- instead of driving
    # grid._search_class.set_source/perform_search by hand. This is what
    # gives single-radius, multi-radius, and (weighted) distance-band r specs
    # all for free: whatever radius_search already supports here, works here
    # too, with no separate handling needed.
    # x/y here are already the projected coordinates resolved by _validate_kwargs
    # above, so crs='' skips a second, redundant reprojection.
    _exclude_self_orig = exclude_self
    # resolved above on the preliminary grid (by intersect_polygon_with_grid_aw
    # and draw_null_distribution_points respectively); carry them over below --
    # plotting reads all three off the grid it's handed, and the nested
    # radius_search() call below builds its own separate grid that never had
    # any of this set on it.
    _study_area = grid.study_area
    _study_grid_bounds = grid.study_grid_bounds
    _cells_rndm_sample = grid._search_internals.cells_rndm_sample
    # sample_x_steps/y_steps (set by intersect_polygon_with_grid_aw on the
    # preliminary grid) extend the output-grid extent to cover the null-
    # distribution sample area, same as a true self-search would. Without
    # carrying these over, update_spacing() below anchors the output grid to
    # the real-points-only extent, shifting cell boundaries relative to a
    # true self-search.
    _sample_x_steps = getattr(grid, 'sample_x_steps', None)
    _sample_y_steps = getattr(grid, 'sample_y_steps', None)
    _sample_col_ids = getattr(grid, 'sample_col_ids', None)
    _sample_row_ids = getattr(grid, 'sample_row_ids', None)
    # raw_bounds (real pts/pts_target extent only, no null-distribution samples)
    # drives update_spacing()'s output-grid origin (x_steps_bounds/row_ids/col_ids),
    # which in turn is what assign_output_cell_ids() anchors on. The nested
    # radius_search() call below builds its own grid from _combined_src (real +
    # samples), whose wider extent would otherwise shift the output-grid origin
    # by a few metres relative to a true self-search over pts alone.
    _raw_bounds = grid._search_internals.raw_bounds
    grid = radius_search(
        pts=_combined_src,
        crs='',
        r=r,
        c=c,
        x=x,
        y=y,
        stat=stat,
        exclude_self=False,  # applied manually below, uniformly (see comment above _combined_src)
        proj_crs=None,
        # Defensive copy: radius_search()/set_target() may reorder its pts_target
        # argument in place (row sorting for cell-region processing). When
        # pts_target is pts itself (the common self-search default resolved
        # above), that reordering would silently corrupt pts's row order out
        # from under the positional write-back below (disk_sums_for_pts.values
        # assumes pts is still in the same order _combined_src was built from).
        pts_target=(pts_target.copy() if pts_target is pts else pts_target),
        x_tgt=x_tgt,
        y_tgt=y_tgt,
        row_name_tgt=row_name_tgt,
        col_name_tgt=col_name_tgt,
        row_name=row_name,
        col_name=col_name,
        suffix=suffix,
        overwrite=True,
        keep_cols=True,
        silent=silent,
    )
    # Use the grid radius_search() just built as our working grid from here on
    # (instead of the earlier preliminary one): it actually processed the full
    # combined point set (real + null-distribution samples), matching what the
    # old two-search flow gave for free by running both through the same
    # grid._search_class. Also picks up cell_row_name/cell_col_name etc. that
    # radius_search() sets on its own grid.
    grid.study_area = _study_area
    grid.study_grid_bounds = _study_grid_bounds
    grid._search_internals.cells_rndm_sample = _cells_rndm_sample
    _bounds_changed = grid._search_internals.raw_bounds != _raw_bounds
    _sample_steps_changed = (
        _sample_x_steps is not None and not _np_array_equal(getattr(grid, 'sample_x_steps', None), _sample_x_steps)
    ) or (
        _sample_y_steps is not None and not _np_array_equal(getattr(grid, 'sample_y_steps', None), _sample_y_steps)
    )
    if _bounds_changed or _sample_steps_changed:
        grid._search_internals.raw_bounds = _raw_bounds
        if _sample_x_steps is not None:
            grid.sample_x_steps = _sample_x_steps
            grid.sample_col_ids = _sample_col_ids
        if _sample_y_steps is not None:
            grid.sample_y_steps = _sample_y_steps
            grid.sample_row_ids = _sample_row_ids
        grid.update_spacing(recompute=True)

    sum_radius_names = [(cname + suffix) for cname in c]
    if _exclude_self_orig:
        for name, col in zip(sum_radius_names, c):
            _combined_src[name] = _combined_src[name].values - _combined_src[col].values

    for name in sum_radius_names:
        rndm_pts[name] = _combined_src[name].values[n_real:]
    disk_sums_for_pts = _combined_src.iloc[:n_real][sum_radius_names]
    # radius_search() mutated _combined_src (a separate object from pts) in
    # place, not pts itself -- write the real-point results back onto pts
    # explicitly (the old two-search flow got this for free by passing pts
    # directly as its own search source).
    for name in sum_radius_names:
        pts[name] = disk_sums_for_pts[name].values

    # grid._search_class.source.pts currently IS _combined_src (real + sample
    # rows) -- downstream code (clustering's add_cluster_id_to_pts, plotting)
    # reads/writes that object directly, not the caller's pts, so it must be
    # repointed at the real pts here (carrying over any cell-id columns the
    # search assigned) to restore the pre-refactor invariant that
    # grid._search_class.source.pts is the caller's own pts object.
    _sc = grid._search_class
    _src_row_name, _src_col_name = _sc.source.row_name, _sc.source.col_name
    for _cell_id_col in (_src_row_name, _src_col_name):
        if _cell_id_col in _combined_src.columns:
            pts[_cell_id_col] = _combined_src[_cell_id_col].values[:n_real]
    _sc.source.pts = pts

    _cluster_thresholds_dict = compute_null_thresholds(rndm_pts, c, suffix, k_th_percentile)
    # unpack dict → list keyed by c[j]+suffix for the single-radius path
    cluster_threshold_values = [_cluster_thresholds_dict[col+suffix] for col in c]
    grid.null_distribution = rndm_pts

    if not silent:
        _r_str = (f'{r/1000:g} km' if r >= 1000 else f'{r:g} m') if grid._proj_is_metric else f'{r:g}'
        for (colname, threshold_value, k_th_p) in zip(c, cluster_threshold_values, k_th_percentile):
            progress_print(f"Threshold for {colname} within {_r_str}: {k_th_p}th-percentile = {threshold_value:g}.")

    if not _is_internal: _prog.step("labeling clusters")
    for j, cname in enumerate(c):
        pts[str(cname)+str(cluster_suffix)] = disk_sums_for_pts.values[:,j]>cluster_threshold_values[j]

    aggregate_cols = [col + suffix for col in orig_cols]
    cluster_cols   = [col + cluster_suffix for col in orig_cols]
    _k_th_list = k_th_percentile if isinstance(k_th_percentile, list) else [k_th_percentile]
    n_agg = len(aggregate_cols)
    k_ths_expanded = (_k_th_list * ((n_agg + len(_k_th_list) - 1) // len(_k_th_list)))[:n_agg]
    col_threshold_info = {
        col: {k_ths_expanded[i]: _cluster_thresholds_dict[col]}
        for i, col in enumerate(aggregate_cols)
    }
    grid._cluster_result = {
        'aggregate_cols':     aggregate_cols,
        'thresholds':         _cluster_thresholds_dict,
        'k_th_percentiles':   k_ths_expanded,
        'col_threshold_info': col_threshold_info,
        'display_radius':     r,
        'plot_colnames':      _np_array(list(orig_cols) + aggregate_cols + cluster_cols),
    }

    # Cluster detection always materialises the output grid (cell aggregates etc.).
    if _max_output_cells is not None:
        grid._max_output_cells = int(_max_output_cells)
    grid.update_spacing()

    if not keep_cols:
        _keep_extra = {grid.cell_row_name, grid.cell_col_name, init_sort, x, y}
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

    if not hasattr(grid, '_aabpl_col_meta'):
        grid._aabpl_col_meta = {}
    for col in orig_cols:
        grid._aabpl_col_meta[col + suffix] = {'c': col, 'stat': stat, 'r': r}
        grid._aabpl_col_meta[col + cluster_suffix] = {'c': col, 'stat': stat, 'r': r}

    return grid
# done

@attach_params
def detect_cluster_cells(
    pts:_pd_DataFrame,
    crs:str,
    r,          # float | list | bands | weighted bands — see radius_search.params.r
    c:list=[],
    x:str='lon',
    y:str='lat',
    stat:str=['sum','count','mean','variance','std','cv','skewness','kurtosis'][0],
    exclude_self:bool=True,
    cell_size:float=None,
    study_area='buff_cells,min_pts=1',
    area_weight:str=None,
    k_th_percentile:float=99.5,
    null_distribution=int(1e5),
    random_seed:int=None,
    proj_crs:str='auto',
    row_name:str='id_y',
    col_name:str='id_x',
    suffix:str=None,
    contingency=1,
    merge_dist=None,
    min_cluster_share=(0.05, 0.0, 0.0),
    make_convex:bool=True,
    pts_target:_pd_DataFrame=None,
    keep_cols:bool=False,
    overwrite:bool=False,
    silent:bool=None,
    **kwargs,
):
    """
    Compute radius aggregates for all points, compare them against a null distribution of
    randomly placed points within the study area, and label points above the k-th percentile
    as clustered. Results are appended to ``pts`` in-place and spatial cluster polygons are
    stored on the returned Grid object.

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
        search pass, e.g. ``stat=['sum', 'mean', 'variance']``. The **first** stat in the
        list drives cluster detection; any additional stats are appended to the output grid as
        extra cell-level aggregates.
    exclude_self (bool):
        If True, each point's own value is subtracted from its radius aggregate (default=True).
    cell_size (float):
        Output cell size, in the same unit as ``r`` (metres after reprojection). Controls the
        resolution of the output grid used for exports and plots — NOT the internal search grid,
        whose cell size is chosen automatically for speed. Per-point aggregates are exact
        regardless of this value, so a finer output grid is well-defined. When None, defaults to
        ``r/3`` (default=None).
    study_area (shapely.Polygon | shapely.MultiPolygon | str):
        Area used for drawing random comparison points. Accepted string values:
            - ``'buff_cells,min_pts=1'``: grid cells with at least 1 point plus a radius-sized buffer (default)
            - ``'buff_non_empty_cells'``: non-empty grid cells plus a radius-sized buffer
            - ``'concave'``: concave hull around points
            - ``'convex'``: convex hull around points
            - ``'buffer'``: buffer around individual points (slow for large datasets)
            - ``'bounding_box'``: axis-aligned bounding box
            - ``'grid'`` or ``None``: full grid extent
        Alternatively pass any Shapely ``Polygon`` or ``MultiPolygon`` directly. The geometry
        is assumed to be in ``crs`` and is reprojected to the projected CRS automatically.
        Append ``,min_pts=N`` to any string option to require at least N data points per cell
        before random points are drawn there (e.g. ``'buff_cells,min_pts=2'``).
    area_weight (str):
        Inverse-area weighting for edge effects. See ``radius_search`` for full variant
        documentation. ``'exact'`` scales linearly with the number of points and is the
        recommended choice. ``None`` disables weighting (default=None).

        **Limitation:** contained cells (fully inside the search disk) are treated as binary —
        fully valid or fully invalid — based on whether they fall inside ``study_area``.
        Cells that straddle the ``study_area`` boundary but were classified as fully valid
        contribute zero invalid area, causing a slight upward bias in ``valid_area_share``
        when ``study_area`` is a custom Polygon or MultiPolygon with sharp edges. For
        method-string study areas (``'buff_non_empty_cells'``, ``'concave'``, etc.) the
        boundary follows cell edges exactly and this bias does not occur.
    k_th_percentile (float):
        Percentile of the null distribution a point must exceed to be labelled as clustered (default=99.5).
    null_distribution (int | numpy.ndarray | pandas.DataFrame):
        Controls the null distribution used for cluster detection (default ``100_000``):

        - **int**: draw this many points uniformly at random within ``study_area``.
        - **numpy.ndarray of shape (N, 2)**: use these coordinates directly — **first column x,
          second column y**, both in the projected CRS (metres).
        - **pandas.DataFrame**: treated as an (N, 2) array via ``.values`` — **first column x,
          second column y** (column names are ignored).

        **Important for coordinate inputs:** coordinates must be in the projected CRS (metres),
        not the original input CRS. Use ``aabpl.draw_random_coords()`` with ``grid.study_area``
        to generate compatible coordinates.
    random_seed (int):
        Random seed for reproducibility. None means no seed is set (default=None).
    contingency (int or tuple):
        Controls how adjacent clustered cells are merged into clusters. Accepts:

        - **int** — queen contingency (includes diagonals); values >= 2 extend the reach by
          that many steps (default=1).
        - **``(queen, rook)``** — explicit queen and rook values. Rook contingency merges only
          horizontally/vertically adjacent cells and is ignored when queen is higher.
    merge_dist (None | float | tuple | list of tuples):
        Distance threshold(s) for merging clusters after contingency merging.
        Each condition is a ``(centroid_dist, border_dist)`` pair — both must hold (AND).
        Pass a list of such pairs to merge when ANY pair is satisfied (OR between pairs, AND within each).
        A ``None`` element in a pair disables that measure for that condition.
        Accepts:

        - ``None``                         — no distance merging (default).
        - ``float``                        — same threshold for both centroid and border.
        - ``(centroid, border)``           — single AND-condition.
        - ``[(c1,b1), (c2,b2), ...]``      — merge if condition 1 OR condition 2 OR ... is met.
    min_cluster_share (float or tuple):
        Minimum share of total clustered points a cluster must represent to be retained.
        Pass a single float to apply the same threshold at all three stages, or a
        ``(after_contingency, after_merge_dist, after_convex)`` tuple for stage-specific
        thresholds (default=``(0.05, 0.0, 0.0)``).
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
    grid_bounds (tuple of 4 floats|None):
        ``(xmin, ymin, xmax, ymax)`` in the **input CRS** (same units as ``x``/``y``).
        Any component may be ``None`` to fall back to the data extent for that edge.

        - ``(lon_west, lat_south, None, None)`` — anchor the south-west corner only.
        - ``(None, None, lon_east, lat_north)`` — extend grid at least this far east/north.
        - ``(None, lat_south, None, lat_north)`` — fix latitudinal edges only.

        **xmin/ymin are binding** (west edge of col 0, south edge of row 0; points outside
        get negative indices). **xmax/ymax are soft minimums** — grid extends to
        ``max(data_extent, specified_value)``. x = easting/longitude, y = northing/latitude.
        Default: ``(None, None, None, None)``.
    keep_cols (bool):
        If False, intermediate columns added during processing (grid indices, offsets, proj x+y, etc.)
        are removed from ``pts`` before returning. If None proj x+y are retained. If True they are retained (default=False).
    overwrite (bool):
        If True, existing output columns with the same names are overwritten. Raises ValueError
        when False and a collision is detected (default=False).
    silent (bool):
        If True, suppresses all progress output. None is treated as False (default=None).

    Returns:
    -------
    grid (aabpl.Grid):
        The Grid object used for the search, with spatial cluster polygons stored at
        ``grid.clustering``. Boolean cluster columns ``{c}_cluster_{stat}_{r}`` and radius
        aggregates ``{c}{suffix}`` are appended to ``pts``.
    """
    # backward-compat: spacing= was renamed to cell_size=
    if 'spacing' in kwargs:
        import warnings
        warnings.warn("'spacing' is deprecated, use 'cell_size' instead.", DeprecationWarning, stacklevel=2)
        if cell_size is None:
            cell_size = kwargs.pop('spacing')
        else:
            kwargs.pop('spacing')
    # backward-compat: n_random_points= merged into null_distribution=
    if 'n_random_points' in kwargs:
        import warnings
        warnings.warn("'n_random_points' is deprecated, pass an integer to 'null_distribution' instead.", DeprecationWarning, stacklevel=2)
        if isinstance(null_distribution, int):
            null_distribution = kwargs.pop('n_random_points')
        else:
            kwargs.pop('n_random_points')
    if 'min_pts_to_sample_cell' in kwargs:
        min_pts_to_sample_cell = kwargs.pop('min_pts_to_sample_cell')
        progress_print(
            "DeprecationWarning: min_pts_to_sample_cell= is deprecated. Pass it inline via study_area= instead, "
            "e.g. study_area='buff_cells,min_pts=1'. "
            "Call resolve_study_area.params() for all options."
        )
    else:
        min_pts_to_sample_cell = 0
    if 'cluster_suffix' in kwargs:
        import warnings
        warnings.warn("'cluster_suffix' is deprecated as a function argument. It will be removed in a future version.", DeprecationWarning, stacklevel=2)
    cluster_suffix = kwargs.pop('cluster_suffix', None)
    _dev = kwargs.pop('_dev', None)
    grid_bounds = kwargs.pop('grid_bounds', (None, None, None, None))
    if 'plot_distribution' in kwargs:
        import warnings
        warnings.warn("'plot_distribution' is deprecated as a function argument. Call grid.plot.rand_dist() on the returned grid instead.", DeprecationWarning, stacklevel=2)
        kwargs.pop('plot_distribution')
    if 'plot_cluster_points' in kwargs:
        import warnings
        warnings.warn("'plot_cluster_points' is deprecated as a function argument. Call grid.plot.cluster_pts() on the returned grid instead.", DeprecationWarning, stacklevel=2)
        kwargs.pop('plot_cluster_points')
    x_tgt             = kwargs.pop('x_tgt', None)
    y_tgt             = kwargs.pop('y_tgt', None)
    row_name_tgt      = kwargs.pop('row_name_tgt', None)
    col_name_tgt      = kwargs.pop('col_name_tgt', None)
    exclude_pt_itself = kwargs.pop('exclude_pt_itself', None)
    queen_contingency, rook_contingency = _unpack_contingency(contingency)
    _merge_dist_conditions = _unpack_merge_dist(merge_dist)
    # Legacy scalar unpacking kept for _dep_cluster backward compat.
    centroid_dist_threshold = _merge_dist_conditions[0][0] if _merge_dist_conditions else None
    border_dist_threshold   = _merge_dist_conditions[0][1] if _merge_dist_conditions else None
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
    _max_output_cells = kwargs.pop('max_output_cells', None)
    # Backwards-compat aliases
    _dep_dcc = {k: kwargs.pop(k) for k in ('weight_valid_area', 'sample_area') if k in kwargs}
    if _dep_dcc:
        progress_print(
            f"DeprecationWarning: kwarg(s) {sorted(_dep_dcc)} are deprecated. "
            "Use area_weight= and study_area= instead. See docstring."
        )
        if 'weight_valid_area' in _dep_dcc and area_weight is None:
            area_weight = _dep_dcc['weight_valid_area']
        if 'sample_area' in _dep_dcc and study_area is None:
            study_area = _dep_dcc['sample_area']
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
    from .search.multi_radius import _parse_r_spec
    _spec_type, _spec_data = _parse_r_spec(r)
    if _spec_type != 'single':
        if suffix is not None or cluster_suffix is not None:
            progress_print("WARNING: custom suffix/cluster_suffix is not supported for multi-radius or distance-band r. Ignoring.")
        suffix = None
        if cluster_suffix is None:
            cluster_suffix = '_cluster'
    else:
        # Unwrap single-element lists/tuples (e.g. r=[750]) to the plain
        # scalar _parse_r_spec already resolved them to.
        r = _spec_data
        if suffix is None:
            suffix = _default_suffix(stat, r)
        if cluster_suffix is None:
            cluster_suffix = f'_cluster{_default_suffix(stat, r)}'
    if centroid_dist_threshold is None:
        centroid_dist_threshold = _r_scalar(r) * 10/3
    if border_dist_threshold is None:
        border_dist_threshold = _r_scalar(r) * 4/3

    if isinstance(study_area, (_shapely_Polygon, _shapely_MultiPolygon, _shapely_GeometryCollection)) and crs and local_crs and crs != local_crs:
        study_area = convert_MultiPolygon_crs(study_area, initial_crs=crs, target_crs=local_crs)

    grid = detect_cluster_pts(
        pts=pts,
        crs=local_crs if local_crs is not None else '',
        r=r,
        c=c,
        stat=stat,
        exclude_self=exclude_self,
        exclude_pt_itself=exclude_pt_itself,
        area_weight=area_weight,
        study_area=study_area,
        k_th_percentile=k_th_percentile,
        null_distribution=null_distribution,
        random_seed=random_seed,
        x=x,
        y=y,
        row_name=row_name,
        col_name=col_name,
        suffix=suffix,
        _cluster_suffix=cluster_suffix,
        proj_crs=local_crs,
        pts_target=pts_target,
        x_tgt=x_tgt,
        y_tgt=y_tgt,
        row_name_tgt=row_name_tgt,
        col_name_tgt=col_name_tgt,
        cell_size=cell_size,
        grid_bounds=grid_bounds,
        keep_cols=True,  # cell ids (row_name/col_name) are needed for clustering below
        overwrite=overwrite,
        _dev=_dev,
        silent=silent,
        _parsed_spec=(_spec_type, _spec_data),
    )
    # TODO: replace with actual output names once spacing/nesting changes column naming
    if not hasattr(grid, 'cell_row_name'):
        grid.cell_row_name = row_name
        grid.cell_col_name = col_name

    # When pts_target was separate, assign_output_cell_ids and aggregate_pts_to_output_cells
    # ran on pts_target only. Rebuild both from source pts before clustering.
    if grid.cell_row_name not in pts.columns:
        _src = getattr(getattr(grid, '_search_class', None), 'source', None)
        _x_src = getattr(_src, 'x', x)
        _y_src = getattr(_src, 'y', y)
        grid.assign_output_cell_ids(pts, x=_x_src, y=_y_src, row_name=row_name, col_name=col_name)
        _agg_cols = [col for col in c if col in pts.columns]
        if _agg_cols:
            grid.aggregate_pts_to_output_cells(pts, val_cols=_agg_cols, x=_x_src, y=_y_src, agg='sum', overwrite=True)

    # Shared keyword arguments for every create_clusters call.
    _clustering_kwargs = dict(
        pts=pts,
        queen_contingency=queen_contingency,
        rook_contingency=rook_contingency,
        centroid_dist_threshold=centroid_dist_threshold,
        border_dist_threshold=border_dist_threshold,
        min_cluster_share_after_contingency=min_cluster_share_after_contingency,
        min_cluster_share_after_centroid_dist=min_cluster_share_after_centroid_dist,
        min_cluster_share_after_convex=min_cluster_share_after_convex,
        make_convex=make_convex,
        row_name=grid.cell_row_name,
        col_name=grid.cell_col_name,
    )

    if _spec_type == 'single':
        # Single radius: one create_clusters call covering all original columns.
        grid.clustering._merge_dist_conditions_for_col = {col: _merge_dist_conditions for col in c}
        grid.clustering.create_clusters(
            c=list(c),
            cluster_suffix=cluster_suffix,
            **_clustering_kwargs,
        )
    else:
        # Multi-radius / bands: one create_clusters call per cluster column.
        # _detect_cluster_pts_multi stored a map {cluster_col: originating_value_col}
        # so we can pass c=[orig_col] (column_id=0 is always valid) with the exact
        # suffix that reconstructs the cluster column name.
        cluster_col_map = getattr(grid, '_cluster_col_map', {})
        for cluster_col, orig_col in cluster_col_map.items():
            if cluster_col not in pts.columns:
                if not silent:
                    progress_print(
                        f'Warning: cluster column "{cluster_col}" missing from pts '
                        f'(pts has {len(pts.columns)} cols); skipping create_clusters. '
                        f'This is a bug — please report it.'
                    )
                continue
            adapted_suffix = cluster_col[len(orig_col):]  # e.g. '_sum_wgt_cluster'
            grid.clustering._merge_dist_conditions_for_col = {orig_col: _merge_dist_conditions}
            grid.clustering.create_clusters(
                c=[orig_col],
                cluster_suffix=adapted_suffix,
                **_clustering_kwargs,
            )

    # Attach threshold + k to each ClustersForColumn so the user can inspect
    # e.g. grid.clustering.by_column['employment_sum_15000_cluster'].threshold
    _cti = getattr(grid, '_cluster_result', {}).get('col_threshold_info', {})
    for _cluster_col, _cfc in grid.clustering.by_column.items():
        for _agg_col, _kt in _cti.items():
            if _cluster_col.startswith(_agg_col):
                _cfc.k, _cfc.threshold = next(iter(_kt.items()))
                break

    # Always update spacing so grid.cell_size and grid.row_ids are current.
    if _max_output_cells is not None:
        grid._max_output_cells = int(_max_output_cells)
    grid.update_spacing()

    if not grid._silent:
        nr, nc = len(grid.row_ids), len(grid.col_ids)
        n_nonempty = len(grid._search_internals.id_to_sums)
        _cs = grid.cell_size
        _cs_str = (f'{_cs/1000:g} km' if _cs >= 1000 else f'{_cs:g} m') if grid._proj_is_metric else f'{_cs:g}'

        per_col_clusters = [
            len(col_cl.clusters) for col_cl in grid.clustering.by_column.values()
        ]
        per_col_cells = [
            len(set(cell for cl in col_cl.clusters for cell in cl.cells))
            for col_cl in grid.clustering.by_column.values()
        ]
        n_cols = len(per_col_clusters)
        if n_cols <= 1:
            n_cl = per_col_clusters[0] if per_col_clusters else 0
            n_ce = per_col_cells[0] if per_col_cells else 0
            _cl_str = (
                f'Detected {n_cl} cluster{"s" if n_cl != 1 else ""}'
                f' spanning {n_ce} cells.'
            )
        else:
            cl_lo, cl_hi = min(per_col_clusters), max(per_col_clusters)
            ce_lo, ce_hi = min(per_col_cells), max(per_col_cells)
            cl_range = str(cl_lo) if cl_lo == cl_hi else f'{cl_lo}-{cl_hi}'
            ce_range = str(ce_lo) if ce_lo == ce_hi else f'{ce_lo}-{ce_hi}'
            _cl_str = (
                f'Detected {cl_range} cluster{"s" if cl_hi != 1 else ""}'
                f' spanning {ce_range} cells'
                f' (across {n_cols} cluster columns).'
            )

        _thr_parts = []
        for _col, _cfc in grid.clustering.by_column.items():
            _k   = getattr(_cfc, 'k', None)
            _thr = getattr(_cfc, 'threshold', None)
            if _k is not None and _thr is not None:
                _thr_parts.append(f'{_col}: {{{_k}: {_thr:g}}}')
        _thr_str = ('  |  thresholds: ' + ', '.join(_thr_parts)) if _thr_parts else ''
        progress_print(
            f'Output grid: {nr}x{nc} = {nr*nc:,} cells'
            f'  |  {n_nonempty:,} non-empty  |  cell size {_cs_str}  |  {_cl_str}{_thr_str}'
        )

    if not keep_cols:
        _cluster_id_suffix = cluster_suffix.replace('_cluster', '_cluster_id', 1)
        if _spec_type == 'single':
            _outer_output_cols = (
                set(str(col)+suffix for col in c) |
                set(str(col)+cluster_suffix for col in c) |
                set(str(col)+_cluster_id_suffix for col in c)
            )
        else:
            # Multi-radius: output cols are what _detect_cluster_pts_multi stored
            # plus any cluster_id cols that create_clusters added.
            _mr_output = getattr(grid, '_multi_radius_output_cols', set())
            _cluster_id_cols = {
                col.replace(cluster_suffix, _cluster_id_suffix)
                for col in _mr_output if col.endswith(cluster_suffix)
            }
            _outer_output_cols = _mr_output | _cluster_id_cols
        _keep_extra = {grid.cell_row_name, grid.cell_col_name, x, y}
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
def build_cluster_cells_from_labels(
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
    cell_size:float=None,
    suffix:str=None,
    proj_crs:str='auto',
    keep_cols:bool=False,
    overwrite:bool=False,
    silent:bool=None,
    **kwargs,
):
    """
    Build spatial cell-clusters from points that are already labelled as clustered,
    skipping the radius search and null distribution entirely.

    The caller supplies ``pts`` with a boolean column (``is_cluster_column``) marking
    which points belong to a cluster. Those points are assigned to output grid cells,
    per-cell mass is aggregated for the column(s) in ``c`` (defaults to counting
    clustered points), and contiguous clustered cells are merged into spatial cluster
    polygons exactly as in ``detect_cluster_cells``.

    Use this when you have already run ``detect_cluster_pts`` (or any other method to
    label points) and only need the cell-aggregation and polygon-building step.

    Args:
    -------
    pts (pandas.DataFrame):
        Points with a boolean cluster-membership column. Results are appended in-place.
    crs (str):
        CRS of the coordinates in ``pts``, e.g. ``'EPSG:4326'``.
        Pass ``crs=''`` for Cartesian coordinates already in the same units as ``r``.
    r (float):
        Search radius in metres (or coordinate units when ``crs=''``). Determines the
        internal search grid resolution; no actual radius search is performed.
    c (str or list):
        Column(s) to aggregate per output cell. If empty, the ``is_cluster_column``
        boolean is summed (i.e. clustered-point count per cell).
    is_cluster_column (str):
        Name of the boolean column in ``pts`` that marks clustered points (default=``'cluster'``).
    cluster_suffix (str):
        Suffix appended to each column in ``c`` to form the boolean cluster-indicator
        column names (default=``'_cluster'``).
    exclude_self (bool):
        If True, each point's own value is excluded from its cell aggregate (default=True).
    x (str):
        Column name of the x-coordinate (longitude) in ``pts`` (default=``'lon'``).
    y (str):
        Column name of the y-coordinate (latitude) in ``pts`` (default=``'lat'``).
    row_name (str):
        Name for the grid row-index column appended to ``pts`` (default=``'id_y'``).
    col_name (str):
        Name for the grid column-index column appended to ``pts`` (default=``'id_x'``).
    contingency (int or tuple):
        Controls how adjacent clustered cells are merged. An int sets queen contingency
        (includes diagonals); pass ``(queen, rook)`` for explicit control (default=1).
    merge_dist (None | float | tuple | list of tuples):
        Distance threshold(s) for merging clusters after contingency merging.
        Same format as ``detect_cluster_cells`` (default=None).
    min_cluster_share (float or tuple):
        Minimum share of clustered points a cluster must represent to be retained.
        Pass a float or ``(after_contingency, after_merge_dist, after_convex)`` tuple
        (default=``(0.0, 0.0, 0.0)``).
    make_convex (bool):
        If True, all cells within the convex hull of each cluster are added to it (default=True).
    cell_size (float):
        Output cell size in metres. Defaults to ``r/3`` (default=None).
    suffix (str):
        Suffix for the per-cell aggregate column names. Defaults to ``'_sum_{r}'`` (default=None).
    proj_crs (str):
        Metric CRS used internally. ``'auto'`` selects the appropriate UTM zone (default=``'auto'``).
    keep_cols (bool):
        If False, intermediate columns are removed from ``pts`` before returning (default=False).
    overwrite (bool):
        If True, existing output columns are overwritten (default=False).
    silent (bool):
        If True, suppresses all progress output (default=None).
    grid_bounds (tuple of 4 floats|None):
        Advanced, passed via ``**kwargs``. ``(xmin, ymin, xmax, ymax)`` in the
        **input CRS** (same units as ``x``/``y``); any component may be ``None`` to
        fall back to the data extent for that edge. Same semantics as
        ``detect_cluster_cells``'s ``grid_bounds`` (default=``(None, None, None, None)``).

    Returns:
    -------
    grid (aabpl.Grid):
        Grid object with spatial cluster polygons at ``grid.clustering``.
    """
    # backward-compat: spacing= was renamed to cell_size=
    if 'spacing' in kwargs:
        import warnings
        warnings.warn("'spacing' is deprecated, use 'cell_size' instead.", DeprecationWarning, stacklevel=2)
        if cell_size is None:
            cell_size = kwargs.pop('spacing')
        else:
            kwargs.pop('spacing')
    queen_contingency, rook_contingency = _unpack_contingency(contingency)
    _merge_dist_conditions = _unpack_merge_dist(merge_dist)
    # Legacy scalar unpacking kept for _dep_cluster backward compat.
    centroid_dist_threshold = _merge_dist_conditions[0][0] if _merge_dist_conditions else None
    border_dist_threshold   = _merge_dist_conditions[0][1] if _merge_dist_conditions else None
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
    # grid_bounds: same undocumented advanced kwarg detect_cluster_cells accepts,
    # forwarded to grid construction so callers can pin the output grid's extent
    # even though this function never runs a radius search itself.
    grid_bounds = kwargs.pop('grid_bounds', (None, None, None, None))
    _max_output_cells = kwargs.pop('max_output_cells', None)
    if kwargs:
        raise TypeError(f"build_cluster_cells_from_labels() got unexpected keyword argument(s): {sorted(kwargs)}")
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
        output_spacing=cell_size, proj_crs=proj_crs, silent=silent,
        grid_bounds=grid_bounds,
    )
    pts, local_crs = vk.pts, vk.local_crs
    c, x, y, suffix = vk.c, vk.x, vk.y, vk.suffix
    pts_target, x_tgt, y_tgt = vk.pts_target, vk.x_tgt, vk.y_tgt
    row_name_tgt, col_name_tgt, grid = vk.row_name_tgt, vk.col_name_tgt, vk.grid
    # _validate_kwargs (a shared low-level helper) does not auto-derive a suffix
    # default the way detect_cluster_cells does for itself -- this function never
    # applied one either, which is what caused the pre-existing "can only
    # concatenate str (not NoneType)" crash in set_source below whenever a caller
    # didn't pass suffix= explicitly (reproduces identically in pip 0.4.1). suffix
    # isn't semantically used by this function (no search results are produced),
    # but DiskSearchSource still requires a non-None string.
    if suffix is None:
        suffix = _default_suffix('sum', r)
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
        centroid_dist_threshold = _r_scalar(r) * 10/3
    if border_dist_threshold is None:
        border_dist_threshold = _r_scalar(r) * 4/3

    # Assign points to grid cells and pre-aggregate per-cell mass (used for cluster
    # totals). No radius search and no null distribution is performed -- DiskSearch
    # + set_source is still needed because create_clusters' add_cluster_id_to_pts
    # reads grid._search_class.source downstream. suffix is now guaranteed non-None
    # (see above), which was the actual root cause of the pre-existing crash here
    # (reproduces identically in pip 0.4.1) -- set_target/set_source themselves are
    # correct and necessary, they just weren't given a valid suffix.
    if exclude_pt_itself is not None:
        import warnings; warnings.warn("'exclude_pt_itself' is deprecated, use 'exclude_self' instead.", DeprecationWarning, stacklevel=2)
        exclude_self = exclude_pt_itself
    DiskSearch(grid, r=r, exclude_self=exclude_self,
               area_weight=False)
    grid._search_class.set_target(pts=pts_target, c=c, x=x_tgt, y=y_tgt,
                                  row_name=row_name_tgt, col_name=col_name_tgt, silent=silent)
    grid._search_class.set_source(pts=pts, c=c, x=x, y=y, row_name=row_name, col_name=col_name,
                                  suffix=suffix, silent=silent)

    # Use the caller-provided boolean as the cluster label for every column in c.
    is_cluster = pts[is_cluster_column].astype(bool)
    for column in c:
        pts[str(column)+str(cluster_suffix)] = is_cluster

    if _max_output_cells is not None:
        grid._max_output_cells = int(_max_output_cells)
    grid.update_spacing()

    grid.clustering._merge_dist_conditions_for_col = {col: _merge_dist_conditions for col in c}
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
        row_name=grid.cell_row_name,
        col_name=grid.cell_col_name,
        cluster_suffix=cluster_suffix,
    )

    if not keep_cols:
        _keep_extra = {grid.cell_row_name, grid.cell_col_name, init_sort, x, y}
        _to_drop = [
            col for col in pts.columns
            if col not in _cols_before and col not in _output_cols and col not in _keep_extra
        ]
        if _to_drop:
            pts.drop(columns=_to_drop, inplace=True)

    pts.sort_values(init_sort, inplace=True)
    pts.drop(columns=[init_sort], inplace=True)
    return grid


def detect_cluster_cells_from_labeled_pts(*args, **kwargs):
    """Deprecated alias for ``build_cluster_cells_from_labels``. Same signature."""
    import warnings
    warnings.warn(
        "'detect_cluster_cells_from_labeled_pts' is deprecated, use "
        "'build_cluster_cells_from_labels' instead.",
        DeprecationWarning, stacklevel=2,
    )
    return build_cluster_cells_from_labels(*args, **kwargs)

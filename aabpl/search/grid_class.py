from numpy import (
    array as _np_array,
    linspace as _np_linspace,
    stack as _np_stack,
    arange as _np_arange,
    unique as _np_unique,
    zeros as _np_zeros,
    min as _np_min,
    max as _np_max,
    mean as _np_mean,
    median as _np_median,
    std as _np_std,
    ceil as _np_ceil,
    pi as _np_pi,
    sum as _np_sum,
    log2 as _np_log2,
)
from pyproj import Transformer
from pandas import DataFrame as _pd_DataFrame
from math import log10 as _math_log10, inf as _math_inf
from aabpl.utils.misc import flatten_list, find_column_name
from aabpl.utils.crs_transformation import convert_bounds_to_local_crs
from aabpl.illustrations.plot_utils import map_2D_to_rgb, get_2D_rgb_colobar_kwargs
from .algorithm.disk_search import (
    aggregate_point_data_to_cells,
    search_and_aggregate
)
from .point_region_assignment import assign_points_to_cell_regions
from .sample_area import compute_disk_cell_overlap, intersect_polygon_with_grid
from aabpl.testing.test_performance import time_func_perf
from aabpl.cluster.clusters import Clustering
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union
from geopandas import GeoDataFrame as _gpd_GeoDataFrame
# from decimal import Decimal as _decimal_Decimal, getcontext as _decimal_getcontext
import math
from .spacing_topology import (
    compute_spatial_stats, compute_spacing_breakpoints,
    choose_nest_depth,
    predict_timing, choose_spacing_and_depth,
)
from aabpl import config as _cfg

# def get_spacing_decimal(xmin, xmax, n_steps, ndec = 20):
#     _decimal_getcontext(ndec)
#     return _decimal_Decimal(xmax-xmin)/_decimal_Decimal(n_steps)


class _DFList(list):
    """list of GeoDataFrames that also supports dict-style lookup by cluster column name.

    Backwards-compatible: existing code using ``dfs[0]``, ``for df in dfs``, ``len(dfs)``
    continues to work unchanged. New code can use ``dfs['employment_cluster_sum_750']``
    or iterate via ``dfs.keys()`` / ``dfs.items()``.
    """
    def __init__(self):
        super().__init__()
        self._by_column = {}

    def _append(self, key, df):
        self.append(df)
        self._by_column[key] = df

    def __getitem__(self, item):
        if isinstance(item, str):
            return self._by_column[item]
        return super().__getitem__(item)

    def keys(self):
        return self._by_column.keys()

    def values(self):
        return self._by_column.values()

    def items(self):
        return self._by_column.items()


class Bounds(object):
    __slots__ = ('xmin', 'xmax', 'ymin', 'ymax', 'np_array_of_bounds') # use this syntax to save some memory. also only create vars that are really neccessary
    def __init__(self, xmin:float, xmax:float, ymin:float, ymax:float):
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
    #
#


class SearchInternals:
    """Namespace for internal search-grid state. Not part of the public API."""
    pass



        

class Grid(object):
    """
    Result object returned by ``detect_cluster_pts`` and ``detect_cluster_cells``.

    Call ``grid.info()`` for a concise interactive summary.

    Key attributes
    --------------
    proj_crs : str
        Projected CRS used for all coordinates and geometries (e.g. ``'EPSG:32632'``).
    cell_size / cell_size_y : float
        Output cell width / height in ``proj_crs`` units (default ``r / 3``).
    x_steps / y_steps : numpy.ndarray
        Left-edge coordinates of output columns / rows.
    cell_aggregates : dict
        ``{(row, col): values_array}`` — aggregated indicator values per non-empty cell.
    sample_area : shapely.Geometry or None
        Valid sampling polygon within which null-distribution points are drawn.
    null_distribution : pandas.DataFrame or None
        Random points used for the null distribution with their radius-sum columns.
    clustering : Clustering
        ``grid.clustering.by_column[col].clusters`` — list of ``Cluster`` objects, each
        with ``.id``, ``.geometry``, ``.centroid``, ``.cells``.

    Plots (all return matplotlib.Figure)
    -------------------------------------
    grid.plot.clusters()     — cluster polygons over the aggregated grid
    grid.plot.vars()         — choropleth of radius-sum values per cell
    grid.plot.cluster_pts()  — scatter coloured by radius-sum, clusters highlighted
    grid.plot.rand_dist()    — observed vs. null-distribution cumulative plot

    Export
    ------
    grid.save_sparse_grid(filename)              → shapefile / CSV of non-empty cells + cluster ids
    grid.save_cell_clusters(filename)            → shapefile / CSV of cluster polygons
    grid.create_sparse_grid_df()                 → GeoDataFrame of non-empty cells
    grid.create_clusters_df_for_column(col)      → GeoDataFrame of cluster polygons
    """
    @time_func_perf
    def __init__(
        self,
        xmin:float,
        xmax:float,
        ymin:float,
        ymax:float,
        initial_crs:str,
        r:float,
        local_crs:str='auto',
        n_pts_src:int=None,
        n_pts_tgt:int=None,
        n_pts_src_extra:int=0,
        pts_tgt_xy=None,
        data_crs:str=None,
        output_spacing:float=None,
        output_spacing_y:float=None,
        grid_bounds=(None, None, None, None),
        silent=False,
    ):
        """
        Internal constructor — call via ``detect_cluster_pts`` / ``detect_cluster_cells``.

        The public-facing cell size is ``spacing`` (default ``r/3``).  An internal
        search grid with a different cell size is chosen automatically for performance;
        all internal search-grid state lives in ``_search_internals`` and is not part
        of the public API.
        """
        self._silent = silent
        # if local crs should be found automatically or
        if local_crs == 'auto' or initial_crs != local_crs:
            local_crs, (xmin,xmax,ymin,ymax) = convert_bounds_to_local_crs(
                xmin=xmin,
                xmax=xmax,
                ymin=ymin,
                ymax=ymax,
                initial_crs=initial_crs,
                target_crs=local_crs,
                silent=silent
            )

        _extent_r_ratio = max(xmax - xmin, ymax - ymin) / r if r > 0 else 0
        if _extent_r_ratio > 10_000_000:
            from aabpl.utils.progress import progress_print as _pp
            _approx_cells = int((xmax - xmin) / r * 1.414) * int((ymax - ymin) / r * 1.414)
            _pp(
                f'Warning: max(W,H)/r = {_extent_r_ratio:,.0f} (extent {max(xmax-xmin,ymax-ymin):,.0f}, r={r}). '
                f'Search grid ~{_approx_cells:,.0f} cells — will require very large arrays and may exhaust memory. '
                f'Check that r is in the same units as your coordinates.'
            )

        # auto choose spacing ratio and depth unless explictly set by the user via config.
        spacing, nest_depth = choose_spacing_and_depth(
            r=r,
            spacing_ratio=_cfg.FIXED_SPACING_RATIO,
            nest_depth=_cfg.FIXED_NEST_DEPTH,
            n_pts_src=n_pts_src,
            n_pts_tgt=n_pts_tgt,
            n_pts_src_extra=n_pts_src_extra,
            pts_tgt_xy=pts_tgt_xy,
            silent=silent,
        )

        self.clustering = Clustering(self)
        from aabpl.illustrations.plot_grid import GridPlots
        self.plot = GridPlots(self)
        # TODO _search_internals.bounds should also contain excluded area if not cntd
        # min(points._search_internals.bounds+r, max(points._search_internals.bounds, excluded_area_total_bound))
        self.data_crs = data_crs if data_crs is not None else local_crs
        self.proj_crs = local_crs
        try:
            from pyproj import CRS as _CRS
            self._proj_is_metric = _CRS(local_crs).axis_info[0].unit_name == 'metre'
        except Exception:
            self._proj_is_metric = False
        # Project grid_bounds from input CRS to projected CRS.
        # Each component is independent — None means "use data extent".
        _gb = list(grid_bounds) if grid_bounds is not None else [None, None, None, None]
        if any(v is not None for v in _gb) and initial_crs and initial_crs != local_crs:
            from aabpl.utils.crs_transformation import _pyproj_Transformer
            _t = _pyproj_Transformer.from_crs(crs_from=initial_crs, crs_to=local_crs, always_xy=True)
            for i, (bx, by) in enumerate([(0, 1), (2, 3)]):  # (xmin,ymin), (xmax,ymax)
                if _gb[bx] is not None or _gb[by] is not None:
                    px = _gb[bx] if _gb[bx] is not None else (xmin if i == 0 else xmax)
                    py = _gb[by] if _gb[by] is not None else (ymin if i == 0 else ymax)
                    px_proj, py_proj = _t.transform(px, py)
                    if _gb[bx] is not None:
                        _gb[bx] = px_proj
                    if _gb[by] is not None:
                        _gb[by] = py_proj
        self._grid_bounds_proj = tuple(_gb)  # (xmin|None, ymin|None, xmax|None, ymax|None) in proj CRS
        x_padding = ((xmin-xmax) % spacing)/2
        y_padding = ((ymin-ymax) % spacing)/2
        _sb = Bounds(xmin=xmin-x_padding,xmax=xmax+x_padding,ymin=ymin-y_padding,ymax=ymax+y_padding)
        n_xsteps = -int((xmin-xmax)/spacing)+2 # round up
        n_ysteps = -int((ymin-ymax)/spacing)+2 # round up
        x_steps = _np_linspace(_sb.xmin, _sb.xmax, n_xsteps)
        y_steps = _np_linspace(_sb.ymin, _sb.ymax, n_ysteps)
        _search_row_ids = _np_arange(n_ysteps-1)
        _search_col_ids = _np_arange(n_xsteps-1)
        _search_n_cells = len(_search_row_ids)*len(_search_col_ids)
        # Output grid step arrays and ids — initialised to None here; set by update_spacing().
        # row_ids / col_ids / n_cells / x_steps / y_steps point to the OUTPUT grid.
        self.x_steps = None
        self.y_steps = None
        self.x_steps_bounds = None
        self.y_steps_bounds = None
        self._x_anchor_offset = 0
        self._y_anchor_offset = 0
        self.row_ids = _search_row_ids
        self.col_ids = _search_col_ids
        self.n_cells = _search_n_cells

        # Integer cell-key codec: packs (lvl, row, col) into one int64 so the
        # cell-sum aggregation and the per-search offset templates share the same
        # packing and a template translates to a point with one vector add. Sized
        # from the search-grid extent. Always built — the search loop relies on it.
        from aabpl.utils.cell_keys import CellKeyCodec
        _cell_codec = CellKeyCodec(
            nest_depth=nest_depth,
            row_lo=int(_search_row_ids.min()), row_hi=int(_search_row_ids.max()),
            col_lo=int(_search_col_ids.min()), col_hi=int(_search_col_ids.max()),
            offset_margin=16,
        )

        # Output grid — user-facing cell size for exports and plots.
        # When unset, defaults to r/3: the search grid is chosen for speed (a coarser
        # cell, typically ~0.35r-0.71r), but the output raster should be fine enough to
        # render results crisply. Per-point aggregates are exact regardless of the
        # search cell size, so a finer output grid is well-defined.
        # Output spacing (the user-facing cell size) is fixed at construction, default r/3.
        # The output grid itself (arrays and cell aggregates) is built LAZILY by
        # update_spacing() — radius_search alone skips it to avoid the per-point
        # aggregation overhead when no output grid is needed.
        self.cell_size = output_spacing if output_spacing is not None else r / 3
        self.cell_size_y = output_spacing_y if output_spacing_y is not None else self.cell_size
        self.cell_aggregates = {}
        self._output_val_cols = []
        self._spacing_computed = False
        # Bundle all search-grid internals into a namespace for convenience.
        si = SearchInternals()
        si.spacing    = spacing
        si.x_steps    = x_steps
        si.y_steps    = y_steps
        si.row_ids    = _search_row_ids
        si.col_ids    = _search_col_ids
        si.n_cells    = _search_n_cells
        si.nest_depth = nest_depth
        si.cell_codec = _cell_codec
        si.raw_bounds = (xmin, xmax, ymin, ymax)
        si.bounds     = _sb
        si.cells_rndm_sample = set()
        self._search_internals = si

        self.bounds = {'data': (xmin, ymin, xmax, ymax)}
        self.sample_area = None
        self.sample_grid_bounds = None

        grid_xmin = _sb.xmin
        grid_ymin = _sb.ymin
        grid_xmax = _sb.xmax
        grid_ymax = _sb.ymax
        # self.get_cell_centroid = lambda row, col: (float(grid_xmin+col*(spacing+.5)),float(grid_ymin+row*(spacing+.5)))
        min_row, max_row, min_col, max_col = int(_search_row_ids.min()), int(_search_row_ids.max()), int(_search_col_ids.min()), int(_search_col_ids.max())
        si.cell_centroid = lambda row, col, x_steps=x_steps, y_steps=y_steps: (
            float(
                x_steps[int(col):int(col)+2].sum()/2 if min_col<=col<=max_col else 
                grid_xmin+(col+.5)*spacing if min_col<col else 
                grid_xmax+(col-max_col-1+.5)*spacing
            ),
            float(
                y_steps[int(row):int(row)+2].sum()/2 if min_row<=row<=max_row else
                grid_ymin+(row+.5)*spacing if min_row<row else 
                grid_ymax+(row-max_row-1+.5)*spacing
            )
            ) if row%1==0 and col%1==0 else (
            float(
                x_steps[int(round(col)):int(round(col))+2].sum()/2 + (col-round(col))*spacing if min_col<=col<=max_col else
                grid_xmin+(col+.5)*spacing if min_col<col else
                grid_xmax+(col-max_col-1+.5)*spacing
            ),
            float(
                y_steps[int(round(row)):int(round(row))+2].sum()/2 + (row-round(row))*spacing if min_row<=row<=max_row else
                grid_ymin+(row+.5)*spacing if min_row<row else
                grid_ymax+(row-max_row-1+.5)*spacing
            )
            )
        si.cell_poly = lambda row, col, x_steps=x_steps, y_steps=y_steps: [
            ((x_mean-spacing/(2**(1+lvl)),y_mean-spacing/(2**(1+lvl))), 
             (x_mean+spacing/(2**(1+lvl)),y_mean-spacing/(2**(1+lvl))), 
             (x_mean+spacing/(2**(1+lvl)),y_mean+spacing/(2**(1+lvl))), 
             (x_mean-spacing/(2**(1+lvl)),y_mean+spacing/(2**(1+lvl))))
            for x_mean,y_mean,lvl in [(
            float(
                x_steps[int(col):int(col)+2].sum()/2 if min_col<=col<=max_col else 
                grid_xmin+(col+.5)*spacing if min_col<col else 
                grid_xmax+(col-max_col-1+.5)*spacing
            ),
            float(
                y_steps[int(row):int(row)+2].sum()/2 if min_row<=row<=max_row else
                grid_ymin+(row+.5)*spacing if min_row<row else 
                grid_ymax+(row-max_row-1+.5)*spacing
            ),
            0
            ) if row%1==0 and col%1==0 else (
            float(
                x_steps[int(round(col)):int(round(col))+2].sum()/2 + (col-round(col))*spacing if min_col<=col<=max_col else 
                grid_xmin+(col+.5)*spacing if min_col<col else 
                grid_xmax+(col-max_col-1+.5)*spacing
            ),
            float(
                y_steps[int(round(row)):int(round(row))+2].sum()/2 + (row-round(row))*spacing if min_row<=row<=max_row else
                grid_ymin+(row+.5)*spacing if min_row<row else 
                grid_ymax+(row-max_row-1+.5)*spacing
            ),
            next((i for i,n in enumerate(range(max(20,nest_depth+1))) if row%.5%(2**-(n+1))==0),max(21,nest_depth+2)) # UPDATE if max_nest_level > 20
            )]][0]
        
        si.cell_bounds = lambda row, col, x_steps=x_steps, y_steps=y_steps: [
            ((x_mean-spacing/(2**(1+lvl)),
              y_mean-spacing/(2**(1+lvl))), 
             (x_mean+spacing/(2**(1+lvl)),
              y_mean+spacing/(2**(1+lvl))))
            for x_mean,y_mean,lvl in [(
            float(
                x_steps[int(col):int(col)+2].sum()/2 if min_col<=col<=max_col else 
                grid_xmin+(col+.5)*spacing if min_col<col else 
                grid_xmax+(col-max_col-1+.5)*spacing
            ),
            float(
                y_steps[int(row):int(row)+2].sum()/2 if min_row<=row<=max_row else
                grid_ymin+(row+.5)*spacing if min_row<row else 
                grid_ymax+(row-max_row-1+.5)*spacing
            ),
            0
            ) if row%1==0 and col%1==0 else (
            float(
                x_steps[int(round(col)):int(round(col))+2].sum()/2 + (col-round(col))*spacing if min_col<=col<=max_col else 
                grid_xmin+(col+.5)*spacing if min_col<col else 
                grid_xmax+(col-max_col-1+.5)*spacing
            ),
            float(
                y_steps[int(round(row)):int(round(row))+2].sum()/2 + (row-round(row))*spacing if min_row<=row<=max_row else
                grid_ymin+(row+.5)*spacing if min_row<row else 
                grid_ymax+(row-max_row-1+.5)*spacing
            ),
            next((i for i,n in enumerate(range(max(20,nest_depth+1))) if row%.5%(2**-(n+1))==0),max(21,nest_depth+2)) # UPDATE if max_nest_level > 20
            )]][0]
        # NOTE: no grid-creation print here. The internal search grid is not
        # user-facing; the informative print happens in update_spacing() when the
        # OUTPUT grid is actually (re)built, so it fires once per spacing value.
        #
     
    #
    # ── Info ──────────────────────────────────────────────────────────────────
    def info(self):
        """Print a concise summary of the grid's contents and available operations."""
        _metric = getattr(self, '_proj_is_metric', False)
        cs = getattr(self, 'cell_size', None)
        if cs is not None:
            cs_str = (f'{cs/1000:.3g} km' if cs >= 1000 else f'{cs:.3g} m') if _metric else f'{cs:.3g}'
        else:
            cs_str = '?'

        row_ids = getattr(self, 'row_ids', None)
        col_ids = getattr(self, 'col_ids', None)
        nr = len(row_ids) if row_ids is not None else '?'
        nc = len(col_ids) if col_ids is not None else '?'
        cell_agg = getattr(self, 'cell_aggregates', None)
        n_nonempty = len(cell_agg) if cell_agg else 0

        def _f(v): return f'{v/1000:,.1f} km' if _metric and abs(v) >= 1000 else f'{v:,.1f}'

        def _steps_preview(arr):
            if arr is None or len(arr) == 0:
                return 'not set'
            vals = [_f(arr[0])]
            if len(arr) > 1: vals.append(_f(arr[1]))
            if len(arr) > 3: vals.append('...')
            if len(arr) > 2: vals.append(_f(arr[-2]))
            if len(arr) > 1: vals.append(_f(arr[-1]))
            return '[' + '  '.join(vals) + ']'

        lines = ['-' * 60]

        # Grid dimensions
        lines.append(f'  grid.row_ids / col_ids    {nr} x {nc} cells  |  cell_size {cs_str}')
        lines.append(f'  grid.cell_aggregates      {n_nonempty:,} non-empty cells')
        xb = getattr(self, 'x_steps_bounds', None)
        yb = getattr(self, 'y_steps_bounds', None)
        lines.append(f'  grid.x_steps_bounds       {_steps_preview(xb)}')
        lines.append(f'  grid.y_steps_bounds       {_steps_preview(yb)}')

        # CRS
        data_crs = getattr(self, 'data_crs', None)
        proj_crs = getattr(self, 'proj_crs', '?')
        if data_crs and data_crs != proj_crs:
            lines.append(f'  grid.data_crs             {data_crs}')
        lines.append(f'  grid.proj_crs             {proj_crs}')

        # Tracked columns
        val_cols = getattr(self, '_output_val_cols', [])
        tgt = getattr(getattr(self, '_search_class', None), 'target', None)
        if not val_cols and tgt is not None:
            val_cols = list(getattr(tgt, 'c', []))
        lines.append(f'  grid._output_val_cols     {val_cols if val_cols else "(none)"}')

        # Cell ID columns written to pts
        row_name = getattr(self, 'cell_row_name', None) or getattr(getattr(self, '_search_internals', None), 'row_name', None)
        col_name_attr = getattr(self, 'cell_col_name', None) or getattr(getattr(self, '_search_internals', None), 'col_name', None)
        if row_name and col_name_attr:
            lines.append(f'  pts["{row_name}"] / ["{col_name_attr}"]   cell row/col ids written to pts')

        # Sample area
        sa = getattr(self, 'sample_area', None)
        if sa is not None:
            x0, y0, x1, y1 = sa.bounds
            geom_type = type(sa).__name__
            lines.append(f'  grid.sample_area          ({geom_type})  x [{_f(x0)} to {_f(x1)}]  y [{_f(y0)} to {_f(y1)}]')
        else:
            lines.append(f'  grid.sample_area          not set')

        # Null distribution
        nd = getattr(self, 'null_distribution', None)
        if nd is not None:
            nd_type = type(nd).__name__
            lines.append(f'  grid.null_distribution    ({nd_type})  {len(nd):,} random points')
        else:
            lines.append(f'  grid.null_distribution    not computed')

        # Clusters
        cl = getattr(self, 'clustering', None)
        if cl and getattr(cl, 'by_column', None):
            for cl_col, col_cl in cl.by_column.items():
                n_cl = len(col_cl.clusters)
                lines.append(f'  grid.clustering.by_column["{cl_col}"]   {n_cl} cluster{"s" if n_cl != 1 else ""}')
                mr = getattr(col_cl, 'merge_rules', {})
                if mr:
                    _metric = getattr(self, '_proj_is_metric', False)
                    def _d(v):
                        if v is None: return 'None'
                        if isinstance(v, list): return str([(f'{a/1000:g}km',f'{b/1000:g}km') if _metric else (a,b) for a,b in v])
                        return f'{v/1000:g} km' if _metric and v >= 1000 else f'{v:g}'
                    parts = [f'contingency={mr["contingency"]}']
                    if 'merge_dist' in mr: parts.append(f'merge_dist={_d(mr["merge_dist"])}')
                    if 'centroid_dist' in mr: parts.append(f'centroid_dist={_d(mr["centroid_dist"])}')
                    if 'border_dist' in mr: parts.append(f'border_dist={_d(mr["border_dist"])}')
                    lines.append(f'    merge rules: {", ".join(parts)}')
                for cl_obj in col_cl.clusters:
                    n_c = cl_obj.n_cells
                    tot = cl_obj.total
                    cid = cl_obj.id
                    lines.append(f'    cluster {cid}: {n_c} cells  total={tot:g}  .cells  .centroid  .geometry  .area')
        else:
            lines.append(f'  grid.clustering           no clusters')

        # bounds dict
        bd = getattr(self, 'bounds', None) or {}
        if 'data' in bd:
            x0, y0, x1, y1 = bd['data']
            lines.append(f'  grid.bounds["data"]       x [{_f(x0)} to {_f(x1)}]  y [{_f(y0)} to {_f(y1)}]')
        if 'grid' in bd:
            x0, y0, x1, y1 = bd['grid']
            lines.append(f'  grid.bounds["grid"]       x [{_f(x0)} to {_f(x1)}]  y [{_f(y0)} to {_f(y1)}]')

        # Other user-facing attributes not listed above
        _internal = {
            'sums_array', 'pts_ids', 'pts_vals_xy', 'sample_grid_bounds',
            '_output_sample_spacing_x', '_output_sample_spacing_y', '_silent',
            'sample_col_ids', 'sample_col_max', 'sample_col_min',
            'sample_row_ids', 'sample_row_max', 'sample_row_min',
            'x_steps', 'y_steps', 'sample_x_steps', 'sample_y_steps',
        }
        _mentioned = {
            'row_ids', 'col_ids', 'cell_aggregates', 'x_steps_bounds', 'y_steps_bounds',
            'proj_crs', 'data_crs', '_output_val_cols', 'cell_row_name', 'cell_col_name',
            'sample_area', 'null_distribution', 'clustering', '_search_class',
            '_search_internals', '_spacing_computed', '_proj_is_metric',
            '_grid_bounds_proj', '_x_anchor_offset', '_y_anchor_offset',
            'cell_size', 'cell_size_y', 'n_cells', 'plot', 'bounds',
        } | _internal
        _all = [k for k in vars(self) if not k.startswith('__') and k not in _mentioned]
        _public = sorted(k for k in _all if not k.startswith('_'))
        _private = sorted(k for k in _all if k.startswith('_'))
        _other = _public + _private
        if _other:
            lines.append(f'  other attrs               grid.' + '  grid.'.join(_other))

        lines += [
            '-' * 60,
            '  Plots   grid.plot.clusters() / .vars() / .cluster_pts() / .rand_dist()',
            '          grid.plot.sample_area()',
            '  Export  grid.save_sparse_grid(filename)      non-empty cells + cluster ids',
            '          grid.save_full_grid(filename)         all cells (dense)',
            '          grid.save_cell_clusters(filename)     cluster polygons',
            '          grid.create_sparse_grid_df()          as GeoDataFrame',
            '          grid.create_full_grid_df()            dense GeoDataFrame',
            '          grid.create_clusters_df_for_column(col)',
            '  Cell    grid.get_cell_centroid(row, col)  .get_cell_bounds()  .get_cell_poly()',
            '  Rebin   grid.update_spacing(new_cell_size)   rebuild output grid at new resolution',
            '  Add     grid.aggregate_pts_to_output_cells(pts, val_cols, x, y)',
            '-' * 60,
        ]
        print('\n'.join(lines))

    # ── Public output-cell helpers ────────────────────────────────────────────
    def get_cell_centroid(self, row, col):
        """Return the (x, y) centroid of output cell (row, col) in proj_crs.

        col/row may be negative (data left of/below the declared grid_bounds anchor).
        """
        xo, yo = self._x_anchor_offset, self._y_anchor_offset
        xs, ys = self.x_steps, self.y_steps
        return (
            (xs[xo + col] + xs[xo + col + 1]) / 2,
            (ys[yo + row] + ys[yo + row + 1]) / 2,
        )

    def get_cell_bounds(self, row, col):
        """Return (x0, y0, x1, y1) bounds of output cell (row, col) in proj_crs.

        col/row may be negative (data left of/below the declared grid_bounds anchor).
        """
        xo, yo = self._x_anchor_offset, self._y_anchor_offset
        xs, ys = self.x_steps, self.y_steps
        return (xs[xo + col], ys[yo + row], xs[xo + col + 1], ys[yo + row + 1])

    def get_cell_poly(self, row, col):
        """Return corner list [(x0,y0),(x1,y0),(x1,y1),(x0,y1)] for output cell."""
        x0, y0, x1, y1 = self.get_cell_bounds(row, col)
        return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]

    # Keep old names as aliases so existing call-sites that were not yet updated still work.
    cell_centroid = get_cell_centroid
    cell_bounds   = get_cell_bounds
    cell_poly     = get_cell_poly

    # add functions
    aggregate_point_data_to_cells = aggregate_point_data_to_cells
    assign_points_to_cell_regions = assign_points_to_cell_regions # OUTDATED? REPLACE WITH assign_points_to_mirco_regions?
    intersect_polygon_with_grid = intersect_polygon_with_grid
    search_and_aggregate = search_and_aggregate
    compute_disk_cell_overlap = compute_disk_cell_overlap
    
    def get_all_ids(
            self,
            store:bool=False,
    ):
        if hasattr(self,'ids') and not self.ids is None:
            return self.ids 
        col_ids = self.col_ids
        row_ids = self.row_ids
        ids = tuple(flatten_list([[(int(row_id), int(col_id)) for col_id in col_ids] for row_id in row_ids]))
        if store:
            self.ids = ids
        return ids
    #



    def assign_output_cell_ids(
        self,
        pts: _pd_DataFrame,
        x: str,
        y: str,
        row_name: str,
        col_name: str,
    ) -> None:
        """
        Assign each point to its output grid cell and store the column names
        on the grid as ``self.cell_row_name`` / ``self.cell_col_name``
        

        When output spacing equals internal spacing the existing row/col
        columns are reused (no new columns written to pts).  Otherwise new
        columns ``'out_{row_name}'`` / ``'out_{col_name}'`` are added to pts.
        """
        from numpy import floor as _np_floor
        # row/col columns are assigned on the search grid, so compare against it.
        if self.cell_size == self._search_internals.spacing and self.cell_size_y == self._search_internals.spacing:
            self.cell_row_name = row_name
            self.cell_col_name = col_name
            return
        out_col_name = 'out_' + col_name
        out_row_name = 'out_' + row_name
        pts[out_col_name] = _np_floor(
            (pts[x].values - self.x_steps_bounds[0]) / self.cell_size
        ).astype(int)
        pts[out_row_name] = _np_floor(
            (pts[y].values - self.y_steps_bounds[0]) / self.cell_size_y
        ).astype(int)
        self.cell_row_name = out_row_name
        self.cell_col_name = out_col_name

    def aggregate_pts_to_output_cells(
        self,
        pts: _pd_DataFrame,
        val_cols: list,
        x: str = 'lon',
        y: str = 'lat',
        agg: str = 'sum',
        overwrite: bool = False,
        add_to_exports: bool = False,
    ) -> dict:
        """
        Aggregate per-point values into output grid cells.

        Useful for adding extra indicators to an existing grid without rerunning
        the full radius search.  The ``pts`` coordinates must already be in
        ``grid.proj_crs`` (projected, same units as the grid).

        Parameters
        ----------
        pts : DataFrame
            Must contain columns ``x``, ``y``, and all ``val_cols``.
        val_cols : list[str]
            Columns to aggregate.
        x, y : str
            Coordinate column names in ``pts``, in ``grid.proj_crs`` units.
        agg : str
            Aggregation method: ``'sum'``, ``'mean'``, or ``'count'``.
        overwrite : bool
            If ``False`` (default), results are merged into ``grid.cell_aggregates``
            — existing cells keep their values for columns not in ``val_cols``.
            If ``True``, ``grid.cell_aggregates`` is replaced entirely.
        add_to_exports : bool
            If ``True``, append ``val_cols`` to ``grid._output_val_cols`` so that
            ``create_sparse_grid_df`` / ``save_sparse_grid`` include them.
            Leave ``False`` (default) when the new columns follow a different naming
            convention from the radius-sum columns or are only needed for custom analysis.

        Returns
        -------
        dict
            ``{(row, col): numpy_array}`` mapping output-grid cells to aggregated
            values (one entry per val_col, in the same order as ``val_cols``).
        """
        from numpy import floor as _np_floor
        from pandas import DataFrame as _pd_DataFrame_local

        if self.x_steps_bounds is None or self.y_steps_bounds is None:
            raise RuntimeError(
                "Output grid not yet built — call grid.update_spacing() first, "
                "or run detect_cluster_pts / detect_cluster_cells to completion."
            )
        for col in [x, y] + list(val_cols):
            if col not in pts.columns:
                raise KeyError(
                    f"Column '{col}' not found in pts. "
                    f"Note: x/y must be in grid.proj_crs ({self.proj_crs}), not the original CRS."
                )

        xmin_out = self.x_steps_bounds[0]
        ymin_out = self.y_steps_bounds[0]
        sx = self.cell_size
        sy = self.cell_size_y

        val_cols = list(val_cols)
        col_idx = _np_floor((pts[x].values - xmin_out) / sx).astype(int).clip(0, len(self.col_ids) - 1)
        row_idx = _np_floor((pts[y].values - ymin_out) / sy).astype(int).clip(0, len(self.row_ids) - 1)

        tmp = _pd_DataFrame_local({'_row': row_idx, '_col': col_idx})
        for col in val_cols:
            tmp[col] = pts[col].values

        grp = tmp.groupby(['_row', '_col'])[val_cols]
        if agg == 'sum':
            agg_df = grp.sum()
        elif agg == 'mean':
            agg_df = grp.mean()
        elif agg == 'count':
            agg_df = grp.count()
        else:
            raise ValueError(f"agg must be 'sum', 'mean', or 'count'; got {agg!r}")

        new_result = {rc: row.values.copy() for rc, row in agg_df.iterrows()}

        if overwrite:
            self.cell_aggregates = new_result
        else:
            for rc, vals in new_result.items():
                if rc in self.cell_aggregates:
                    from numpy import concatenate as _np_concatenate
                    self.cell_aggregates[rc] = _np_concatenate([self.cell_aggregates[rc], vals])
                else:
                    self.cell_aggregates[rc] = vals

        if add_to_exports:
            existing = list(getattr(self, '_output_val_cols', []))
            self._output_val_cols = existing + [c for c in val_cols if c not in existing]

        return new_result

    def update_spacing(self, spacing: float = None, spacing_y: float = None, recompute: bool = False):
        """
        (Re)build the output grid and its cached cell aggregates.

        Intentionally lazy: radius_search skips it to avoid per-point aggregation
        overhead when no output grid is needed. Called by detect_cluster_pts /
        detect_cluster_cells (always) and by plot/export methods.

        Parameters
        ----------
        spacing : float, optional
            New output cell size; rebuilds grid if different from current cell_size.
        spacing_y : float, optional
            Output cell height; defaults to spacing.
        recompute : bool
            Force recomputation even if already current.
        """
        import math

        # Capture anchor before any rebuild so we can compute sample offsets afterwards.
        _old_xsb0 = self.x_steps_bounds[0] if self.x_steps_bounds is not None else None
        _old_ysb0 = self.y_steps_bounds[0] if self.y_steps_bounds is not None else None

        changed = False
        if spacing is not None:
            new_y = spacing_y if spacing_y is not None else spacing
            if spacing != self.cell_size or new_y != self.cell_size_y:
                self.cell_size = spacing
                self.cell_size_y = new_y
                changed = True
        if self._spacing_computed and not changed and not recompute:
            return self

        # 1. Data extent: raw points + sample area extension (if set).
        xmin, xmax, ymin, ymax = self._search_internals.raw_bounds
        sx = getattr(self, 'sample_x_steps', None)
        sy = getattr(self, 'sample_y_steps', None)
        if sx is not None and len(sx):
            xmin = min(xmin, float(sx[0]))
            xmax = max(xmax, float(sx[-1]))
        if sy is not None and len(sy):
            ymin = min(ymin, float(sy[0]))
            ymax = max(ymax, float(sy[-1]))

        gb_xmin, gb_ymin, gb_xmax, gb_ymax = self._grid_bounds_proj

        # 2. X-axis
        if gb_xmin is None and gb_xmax is None:
            # Fall A: no user bounds — symmetrical padding
            x_padding = ((xmin - xmax) % self.cell_size) / 2
            full_xmin = xmin - x_padding
            full_xmax = xmax + x_padding
            n_x = max(2, -int((xmin - xmax) / self.cell_size) + 2)
            self.x_steps = _np_linspace(full_xmin, full_xmax, n_x)
            self.x_steps_bounds = self.x_steps
            xoff = 0
        else:
            # Fall B: align to user-specified anchor
            b_xmin = gb_xmin if gb_xmin is not None else xmin
            b_xmax = gb_xmax if gb_xmax is not None else xmax
            full_xmin = b_xmin - math.ceil(max(0, b_xmin - xmin) / self.cell_size) * self.cell_size
            full_xmax = b_xmax + math.ceil(max(0, xmax - b_xmax) / self.cell_size) * self.cell_size
            n_x = round((full_xmax - full_xmin) / self.cell_size) + 1
            self.x_steps = _np_linspace(full_xmin, full_xmin + (n_x - 1) * self.cell_size, n_x)
            xoff = round((b_xmin - full_xmin) / self.cell_size)
            n_xb = round((b_xmax - b_xmin) / self.cell_size) + 1
            self.x_steps_bounds = self.x_steps[xoff : xoff + n_xb]

        # 3. Y-axis
        if gb_ymin is None and gb_ymax is None:
            y_padding = ((ymin - ymax) % self.cell_size_y) / 2
            full_ymin = ymin - y_padding
            full_ymax = ymax + y_padding
            n_y = max(2, -int((ymin - ymax) / self.cell_size_y) + 2)
            self.y_steps = _np_linspace(full_ymin, full_ymax, n_y)
            self.y_steps_bounds = self.y_steps
            yoff = 0
        else:
            b_ymin = gb_ymin if gb_ymin is not None else ymin
            b_ymax = gb_ymax if gb_ymax is not None else ymax
            full_ymin = b_ymin - math.ceil(max(0, b_ymin - ymin) / self.cell_size_y) * self.cell_size_y
            full_ymax = b_ymax + math.ceil(max(0, ymax - b_ymax) / self.cell_size_y) * self.cell_size_y
            n_y = round((full_ymax - full_ymin) / self.cell_size_y) + 1
            self.y_steps = _np_linspace(full_ymin, full_ymin + (n_y - 1) * self.cell_size_y, n_y)
            yoff = round((b_ymin - full_ymin) / self.cell_size_y)
            n_yb = round((b_ymax - b_ymin) / self.cell_size_y) + 1
            self.y_steps_bounds = self.y_steps[yoff : yoff + n_yb]

        self._x_anchor_offset = xoff
        self._y_anchor_offset = yoff

        # 4. Output grid IDs and cell count.
        self.row_ids = _np_arange(len(self.y_steps_bounds) - 1)
        self.col_ids = _np_arange(len(self.x_steps_bounds) - 1)
        self.n_cells = len(self.row_ids) * len(self.col_ids)

        # 5. Warn if user-specified bounds add substantially more cells than data alone.
        if not self._silent and any(v is not None for v in (gb_xmin, gb_ymin, gb_xmax, gb_ymax)):
            _raw = self._search_internals.raw_bounds
            data_cells = max(1, round((_raw[1]-_raw[0]) / self.cell_size)) * max(1, round((_raw[3]-_raw[2]) / self.cell_size_y))
            bnd_cells  = len(self.col_ids) * len(self.row_ids)
            added = bnd_cells - data_cells
            if data_cells > 0 and added > 20_000 and added / data_cells > 0.5:
                from aabpl.utils.misc import progress_print
                progress_print(
                    f"grid_bounds adds {added:,} cells ({added/data_cells:.0%} of data extent). "
                    f"Grid expanded from ~{data_cells:,} to ~{bnd_cells:,} cells."
                )

        # 6. Record how many cells the sample area added left/bottom of the initial grid.
        _scol = max(0, round((_old_xsb0 - self.x_steps_bounds[0]) / self.cell_size)) if _old_xsb0 is not None else 0
        _srow = max(0, round((_old_ysb0 - self.y_steps_bounds[0]) / self.cell_size_y)) if _old_ysb0 is not None else 0
        if not isinstance(getattr(self, 'bounds', None), dict):
            self.bounds = {}
        self.bounds['sample_col_offset'] = _scol
        self.bounds['sample_row_offset'] = _srow
        self.bounds['grid'] = (
            float(self.x_steps_bounds[0]), float(self.y_steps_bounds[0]),
            float(self.x_steps_bounds[-1]), float(self.y_steps_bounds[-1]),
        )

        # 7. Aggregate pts to output cells and assign cell IDs (only when pts available).
        search = getattr(self, '_search_class', None)
        tgt = getattr(search, 'target', None) if search is not None else None
        if tgt is not None and len(getattr(tgt, 'c', [])):
            self._search_internals.row_name = tgt.row_name
            self._search_internals.col_name = tgt.col_name
            _agg_cols = [col for col in tgt.c if col in tgt.pts.columns]
            if _agg_cols:
                self.aggregate_pts_to_output_cells(tgt.pts, val_cols=_agg_cols, x=tgt.x, y=tgt.y, agg='sum', overwrite=True)
            self.assign_output_cell_ids(tgt.pts, x=tgt.x, y=tgt.y, row_name=tgt.row_name, col_name=tgt.col_name)

        self._spacing_computed = True
        return self


    @time_func_perf
    def calc_micro_region_stats(self):
        """
        Calculate statistics on the micro regions created by the radius search, 
        such as the average number of distinct cntd and overlapped cells per region, 
        and their area. 
        Useful for understanding the characteristics of the micro regions and 
        for comparing different radius search configurations in optimization.
        """
        # Helper function to avoid redundant list comprehension syntax
        def calc_cells_area(cells):
            import numpy as _np_local
            if isinstance(cells, _np_local.ndarray):
                return float(_np_local.sum(2.0 ** (-2.0 * cells[:, 0]))) if len(cells) else 0.0
            return sum(2**(-2 * lvl) for lvl, _ in cells)

        _sc = self._search_class
        # 1. Base shared metrics
        n_shared = {
            "cntd": len(_sc.shared_cntd_cells),
            "ovlpd": 0
        }
        area_shared = {
            "cntd": calc_cells_area(_sc.shared_cntd_cells),
            "ovlpd": 0
        }

        # 2. Accumulators for regions
        totals = {
            "count": {"cntd": 0.0, "ovlpd": 0.0},
            "area":  {"cntd": 0.0, "ovlpd": 0.0}
        }
        total_reg_area = 0.0
        total_reg_count = 0

        # 3. Process regions
        mult = _sc.region_and_trgl_mult
        for reg_id, reg_area in _sc.region_id_to_area.items():
            total_reg_area += reg_area
            total_reg_count += 1

            # Use triangle 1 as representative (all triangles share the same list unless
            # by_trgl overrides exist; for aggregate stats any triangle is equivalent)
            cells_data = {
                "cntd": _sc.region_and_trgl_id_to_distinct_cntd_cells[reg_id * mult + 1],
                "ovlpd": _sc.region_and_trgl_id_to_distinct_ovlpd_cells[reg_id * mult + 1]
            }

            for state in ["cntd", "ovlpd"]:
                cells = cells_data[state]
                c_count = len(cells)
                c_area = calc_cells_area(cells)

                # Accumulate values
                totals["count"][state] += c_count
                totals["area"][state]  += c_area
                
                # Weighted by region area
                totals["count"][f"{state}_weighted"] += c_count * reg_area
                totals["area"][f"{state}_weighted"]  += c_area * reg_area

        # 4. Normalization and final aggregation
        for state in ["cntd", "ovlpd"]:
            if total_reg_count > 0:
                # Normalize by count
                totals["count"][state] /= total_reg_count
                totals["area"][state]  /= total_reg_count
            
            if total_reg_area > 0:
                # Normalize by area
                totals["count"][f"{state}_weighted"] /= total_reg_area
                totals["area"][f"{state}_weighted"]  /= total_reg_area

            # Add shared components
            totals["count"][state] += n_shared[state]
            totals["area"][state]  += n_shared[state]

            totals["count"][f"{state}_weighted"] += area_shared[state]
            totals["area"][f"{state}_weighted"]  += area_shared[state]

        # 5. Normalise areas by circle area (π·r²), expressed in spacing² units.
        #    This ratio depends only on r/spacing and nest_depth, so it is
        #    reusable across different point clouds with the same geometry.
        from math import pi as _pi
        r = _sc.r
        spacing = self._search_internals.spacing
        circle_area_in_spacing_units = _pi * (r / spacing) ** 2
        totals["area_share"] = {}
        for state in ["cntd", "ovlpd"]:
            for sfx in ["", "_weighted"]:
                key = state + sfx
                totals["area_share"][key] = (
                    totals["area"][key] / circle_area_in_spacing_units
                    if circle_area_in_spacing_units > 0 else 0.0
                )

        self.micro_region_stats = totals
        return totals
    
    # append plots
    # # append cluster functions
    # create_clusters = create_clusters
    # add_geom_to_cluster = add_geom_to_cluster
    # connect_cells_to_clusters = connect_cells_to_clusters
    # make_cluster_orthogonally_convex = make_cluster_orthogonally_convex
    # make_cluster_convex = make_cluster_convex
    # merge_clusters = merge_clusters
    # add_cluster_tags_to_cells = add_cluster_tags_to_cells
    # # # save options
    # save_full_grid = Clustering.save_full_grid
    # save_sparse_grid = Clustering.save_sparse_grid
    # save_cell_clusters = Clustering.save_cell_clusters

    def plot_sample_area(self, *args, **kwargs):
        import warnings
        warnings.warn("grid.plot_sample_area() is deprecated. Use grid.plot.sample_area() instead.", DeprecationWarning, stacklevel=2)
        return self.plot.sample_area(*args, **kwargs)

    def create_full_grid_df(self, target_crs:str=['initial','local','EPSG:4326'][0], max_column_name_length:int=10):
        """returns geopandas.GeoDataFrame with entry for each grid cell with attributes on its Polygon, centroid, sum of indicator(s), and cluster id
        
        Args:
        -------
        target_crs (str):
            crs in which data shall be projected. If 'initial' then it will be projected in same crs as input data. If 'local' a local projection will be used. Otherwise specify the target crs directly like 'EPSG:4326' (default='initial') 
        max_column_name_length (int):
            maximum length of automatically chosen target name (shapefiles allow a maximum column name length of 10)
        
        Returns:
        -------
        df (geopandas.GeoDataFrame):
            with entry for each grid cell
        """
        self.update_spacing()  # ensure the output grid is materialised
        id_to_sums = self.cell_aggregates if hasattr(self, 'cell_aggregates') and self.cell_aggregates else {}
        all_row_col = [(row, col) for row in self.row_ids for col in self.col_ids]
        n_cells = len(all_row_col)
        n_cols_cluster = len(self.clustering.by_column)
        c_ids = _np_zeros((n_cells, n_cols_cluster), int)
        sums = _np_zeros((n_cells, max(n_cols_cluster, 1)), int)
        polys = []
        centroids_x_local = _np_zeros(n_cells, float)
        centroids_y_local = _np_zeros(n_cells, float)
        target_crs = self.proj_crs if target_crs in ('initial', 'local') else target_crs
        if target_crs != self.proj_crs:
            transformer = Transformer.from_crs(crs_from=self.proj_crs, crs_to=target_crs, always_xy=True)
        clusters_for_columns = list(self.clustering.by_column.values())
        _sc = self._search_class
        for i, (row, col) in enumerate(all_row_col):
            x0, y0, x1, y1 = self.get_cell_bounds(row, col)
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            if target_crs != self.proj_crs:
                cx, cy = transformer.transform(cx, cy)
            centroids_x_local[i] = cx
            centroids_y_local[i] = cy
            for j, clusters_for_column in enumerate(clusters_for_columns):
                if (row, col) in clusters_for_column.cell_to_cluster_id:
                    c_ids[i, j] = clusters_for_column.cell_to_cluster_id[(row, col)]
            if (row, col) in id_to_sums:
                vals = id_to_sums[(row, col)]
                sums[i, :len(vals)] = vals
            polys.append(Polygon(((x0, y0), (x1, y0), (x1, y1), (x0, y1))))
        _row_name = _sc.source.row_name if _sc else 'id_y'
        _col_name = _sc.source.col_name if _sc else 'id_x'
        df = _gpd_GeoDataFrame({
            _row_name: [row for row, col in all_row_col],
            _col_name: [col for row, col in all_row_col],
            'centroid_x': centroids_x_local,
            'centroid_y': centroids_y_local,
            }, geometry=polys,
            crs=self.proj_crs
            )
        if n_cols_cluster <= 1:
            df['cluster_id'] = c_ids[:, 0] if n_cols_cluster == 1 else 0
            df['sum'] = sums[:, 0]
        else:
            for j, column in enumerate(self.clustering.by_column):
                c_id_colname = find_column_name("cluster_id", column, df.columns, max_column_name_length)
                agg_colname = find_column_name("sum_radius", column, df.columns, max_column_name_length)
                df[c_id_colname] = c_ids[:,j]
                df[agg_colname] = sums[:,j]
        if target_crs != self.proj_crs:
            df.to_crs(target_crs, inplace=True)

        return df
    #

    def create_sparse_grid_df(
            self, target_crs:str=['initial','local','EPSG:4326'][0], max_column_name_length:int=10
        ):
        """
        returns geopandas.GeoDataFrame with entry for grid cells that either has points inside or is part of a cluster with attributes on their Polygon, centroid, sum of indicator(s), and cluster id
        
        Args:
        -------
        target_crs (str):
            crs in which data shall be projected. If 'initial' then it will be projected in same crs as input data. If 'local' a local projection will be used. Otherwise specify the target crs directly like 'EPSG:4326' (default='initial') 
        max_column_name_length (int):
            maximum length of automatically chosen target name (shapefiles allow a maximum column name length of 10)
        
        Returns:
        -------
        df (geopandas.GeoDataFrame):
            with entry for each grid cell
        """
        self.update_spacing()  # ensure the output grid + aggregates are materialised
        id_to_sums = self.cell_aggregates if hasattr(self, 'cell_aggregates') and self.cell_aggregates else {}
        x_steps = self.x_steps_bounds
        y_steps = self.y_steps_bounds
        out_row_ids = self.row_ids
        out_col_ids = self.col_ids
        n_out_cells = len(out_row_ids) * len(out_col_ids)
        polys = []
        target_crs = self.proj_crs if target_crs in ('initial', 'local') else target_crs
        if target_crs != self.proj_crs:
            transformer = Transformer.from_crs(crs_from=self.proj_crs, crs_to=target_crs, always_xy=True)
        centroids_x = _np_zeros(n_out_cells, float)
        centroids_y = _np_zeros(n_out_cells, float)
        sparse_row_ids = _np_zeros(n_out_cells, int)
        sparse_col_ids = _np_zeros(n_out_cells, int)
        n_val_cols = max(
            max((len(v) if hasattr(v, '__len__') else 1 for v in id_to_sums.values()), default=0),
            len(self.clustering.by_column),
            1,
        )
        sums = _np_zeros((n_out_cells, n_val_cols), float)
        c_ids = _np_zeros((n_out_cells, len(self.clustering.by_column)), int)
        i = 0
        js_clusters_for_columns = list(enumerate(self.clustering.by_column.values()))
        for row in out_row_ids:
            ymin_c, ymax_c = y_steps[row], y_steps[row + 1]
            sparse_row_ids[i:] = row
            for col in out_col_ids:
                cell_in_a_cluster = False
                for j, clusters_for_column in js_clusters_for_columns:
                    if (row, col) in clusters_for_column.cell_to_cluster_id:
                        c_ids[i, j] = clusters_for_column.cell_to_cluster_id[(row, col)]
                        cell_in_a_cluster = True
                if (row, col) in id_to_sums:
                    vals = id_to_sums[(row, col)]
                    sums[i, :len(vals)] = vals
                elif not cell_in_a_cluster:
                    continue
                sparse_col_ids[i] = col
                xmin_c, xmax_c = x_steps[col], x_steps[col + 1]
                cx, cy = (xmin_c + xmax_c) / 2, (ymin_c + ymax_c) / 2
                if target_crs != self.proj_crs:
                    cx, cy = transformer.transform(cx, cy)
                centroids_x[i] = cx
                centroids_y[i] = cy
                polys.append(Polygon(((xmin_c, ymin_c), (xmax_c, ymin_c), (xmax_c, ymax_c), (xmin_c, ymax_c))))
                i += 1
        _sc2 = self._search_class
        out_row_name = getattr(self, 'cell_row_name', getattr(self._search_internals, 'row_name', 'id_y'))
        out_col_name = getattr(self, 'cell_col_name', getattr(self._search_internals, 'col_name', 'id_x'))
        df = _gpd_GeoDataFrame({
            out_row_name: sparse_row_ids[:i],
            out_col_name: sparse_col_ids[:i],
            'centroid_x': centroids_x[:i],
            'centroid_y': centroids_y[:i],
            }, geometry=polys,
            crs=self.proj_crs
            )
        val_cols = getattr(self, '_output_val_cols', [])
        if len(self.clustering.by_column) <= 1:
            df['cluster_id'] = c_ids[:i, 0] if c_ids.shape[1] > 0 else 0
            for k, vcol in enumerate(val_cols):
                df[vcol] = sums[:i, k]
        else:
            for j, column in enumerate(self.clustering.by_column):
                c_id_colname = find_column_name("cluster_id", column, df.columns, max_column_name_length)
                df[c_id_colname] = c_ids[:i, j]
            for k, vcol in enumerate(val_cols):
                df[vcol] = sums[:i, k]

        if target_crs != self.proj_crs:
            df.to_crs(target_crs, inplace=True)

        return df
    #

    def create_clusters_df_for_column(self, cluster_column:str, target_crs:str=['initial','local','EPSG:4326'][0]):
        """
        returns geopandas.GeoDataFrame with entry for grid cells that either has points inside or is part of a cluster with attributes on their Polygon, centroid, sum of indicator(s), and cluster id
        
        Args:
        cluster_column (str):
            column containing the variable by which the clustering shall be performed
        target_crs (str):
            crs in which data shall be projected. If 'initial' then it will be projected in same crs as input data. If 'local' a local projection will be used. Otherwise specify the target crs directly like 'EPSG:4326' (default='initial') 
        
        Returns:
        df (geopandas.GeoDataFrame):
            with entry for each grid cell attributes: centroid_x, centroid_y, cluster_id, sum, n_cells, area
        """
        self.update_spacing()  # TODO not sure if this is needed - ensure the output grid is materialised
        clusters_for_column = self.clustering.by_column[cluster_column]
        df = _gpd_GeoDataFrame({
            'centroid_x': [cluster.centroid[0] for cluster in clusters_for_column.clusters],
            'centroid_y': [cluster.centroid[1] for cluster in clusters_for_column.clusters],
            'cluster_id': [cluster.id for cluster in clusters_for_column.clusters],
            'sum': [cluster.total for cluster in clusters_for_column.clusters],
            "n_cells": [cluster.n_cells for cluster in clusters_for_column.clusters],
            'area': [cluster.area for cluster in clusters_for_column.clusters],
            },
            geometry = [cluster.geometry for cluster in clusters_for_column.clusters],
            crs=self.proj_crs)
        
        target_crs = self.proj_crs if target_crs in ('initial', 'local') else target_crs
        if target_crs != self.proj_crs:
            transformer = Transformer.from_crs(crs_from=self.proj_crs, crs_to=target_crs, always_xy=True)
            df['centroid_x'], df['centroid_y'] = transformer.transform(df['centroid_x'], df['centroid_y'])
            df.to_crs(target_crs, inplace=True)
        
        return df
    #

    def save_full_grid(
            self,
            filename:str="full_grid",
            file_format:str=['shp','csv'][0],
            target_crs:str=['initial','local','EPSG:4326'][0],
    ):
        """save geopandas.DataFrame with entry for each grid cell with attributes on their Polygon, centroid, sum of indicator(s), and cluster id
        
        filename (str):
            name of the output file excluding file format extension. It can contain full path like 'output_folder/fname' (default='full_grid')
        file_format (str):
            format in which the file shall be saved. Currently available options are 'shp' and 'csv'. Extension will be appended to filename. (default='shp')
        target_crs (str):
            crs in which data shall be projected. If 'initial' then it will be projected in same crs as input data. If 'local' a local projection will be used. Otherwise specify the target crs directly like 'EPSG:4326' (default='initial') 
        
        Returns:
        df (geopandas.GeoDataFrame):
            with entry for each grid cell
        """
        
        df = self.create_full_grid_df(target_crs=target_crs, max_column_name_length=10 if file_format=='shp' else 20)
        # save
        filename = filename +'.'+file_format
        if file_format == 'shp':
            df.to_file(filename, driver="ESRI Shapefile", index=False)
        elif file_format == 'csv':
            df.to_csv(filename, index=False)
        else:
            raise ValueError('Unknown file_format:',file_format,'Choose on out of shp, csv')
        return df
    #

    def save_sparse_grid(
            self,
            filename:str="sparse_grid",
            file_format:str=['shp','csv'][0],
            target_crs:str=['initial','local','EPSG:4326'][0],
        ):
        """save geopandas.GeoDataFrame with entry for grid cells that either has points inside or is part of a cluster with attributes on their Polygon, centroid, sum of indicator(s), and cluster id
        
        Args:
        filename (str):
            name of the output file excluding file format extension. It can contain full path like 'output_folder/fname' (default='sparse_grid')
        file_format (str):
            format in which the file shall be saved. Currently available options are 'shp' and 'csv'. Extension will be appended to filename. (default='shp')
        target_crs (str):
            crs in which data shall be projected. If 'initial' then it will be projected in same crs as input data. If 'local' a local projection will be used. Otherwise specify the target crs directly like 'EPSG:4326' (default='initial') 
        
        Returns:
        df (geopandas.GeoDataFrame):
            with entry for each grid cell that is non empty or part of cluster
        """
        
        df = self.create_sparse_grid_df(
            target_crs=target_crs, max_column_name_length=10 if file_format=='shp' else 20
        )
        # save
        filename = filename +'.'+file_format
        if file_format == 'shp':
            df.to_file(filename, driver="ESRI Shapefile", index=False)
        elif file_format == 'csv':
            df.to_csv(filename, index=False)
        else:
            raise ValueError('Unknown file_format:',file_format,'Choose on out of shp, csv')
        return df
    #

    def save_cell_clusters_for_column(
            self,
            cluster_column:str,
            filename:str="grid_clusters",
            file_format:str=['shp','csv'][0],
            target_crs:str=['initial','local','EPSG:4326'][0],
    ):
        """Save geopandas.GeoDataFrame that has one entry for each clusters including attributes on its polygon geometry, column totals, n_cells, id  

        filename (str):
            name of the output file excluding file format extension. It can contain full path like 'output_folder/fname' (default='grid_clusters')
        file_format (str):
            format in which the file shall be saved. Currently available options are 'shp' and 'csv'. Extension will be appended to filename. (default='shp')
        target_crs (str):
            crs in which data shall be projected. If 'initial' then it will be projected in same crs as input data. If 'local' a local projection will be used. Otherwise specify the target crs directly like 'EPSG:4326' (default='initial') 
        
        Returns:
        df (geopandas.GeoDataFrame)
            with one entry for each cluster
        """
        df = self.create_clusters_df_for_column(cluster_column=cluster_column, target_crs=target_crs)
        filename = filename+'.'+file_format
        if file_format == 'shp':
            df.to_file(filename, driver="ESRI Shapefile", index=False)
        elif file_format == 'csv':
            df.to_csv(filename, index=False)
        else:
            raise ValueError('Unknown file_format:',file_format,'Choose on out of shp, csv')
        return df
        #
    #
    def save_cell_clusters(
            self,
            filename:str="grid_clusters",
            file_format:str=['shp','csv'][0],
            target_crs:str=['initial','local','EPSG:4326'][0],
        ):
        """For each cluster column saves a geopandas.GeoDataFrame that has one entry for each clusters including attributes on its polygon geometry, column totals, n_cells, id  

        filename (str or list):
            name of the output file excluding file format extension. If there are more than 1 cluster column it will append the column name to the file. You can also provide a list of filenames to contain the filename indvidually. It can contain full path like 'output_folder/fname' (default='grid_clusters')
        file_format (str):
            format in which the file shall be saved. Currently available options are 'shp' and 'csv'. Extension will be appended to filename. (default='shp')
        target_crs (str):
            crs in which data shall be projected. If 'initial' then it will be projected in same crs as input data. If 'local' a local projection will be used. Otherwise specify the target crs directly like 'EPSG:4326' (default='initial') 
        
        Returns:
        dfs (_DFList)
            List of GeoDataFrames, one per cluster column. Also supports dict-style access by
            cluster column name: ``dfs['employment_cluster_sum_750']``, ``dfs.keys()``, ``dfs.items()``.
            Existing code using ``dfs[0]`` or ``for df in dfs`` is unchanged.
        """
        dfs = _DFList()
        filenames = filename if type(filename) == list else [filename + (("_"+cluster_column) if len(self.clustering.by_column) > 1 else '') for cluster_column in self.clustering.by_column]
        for (cluster_column, clusters), filename in zip(self.clustering.by_column.items(), filenames):
            df = self.save_cell_clusters_for_column(cluster_column=cluster_column, filename=filename, file_format=file_format, target_crs=target_crs)
            dfs._append(cluster_column, df)
        return dfs
    #
    #


#

class ExcludedArea:
    def __init__(self,excluded_area_geometry_or_list, grid:Grid):
        # recursively split exluded area geometry along grid 
        # then sort it into grid cell
        
        pass
#



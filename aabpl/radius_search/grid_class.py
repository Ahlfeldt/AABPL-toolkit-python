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
from .disk_search_state import (
    aggregate_point_data_to_cells,
    search_and_aggregate
)
from .point_region_assignment import assign_points_to_cell_regions
from .sample_area import compute_disk_cell_overlap, intersect_polygon_with_grid
from aabpl.testing.test_performance import time_func_perf
# from .clusters import (
#     create_clusters, add_geom_to_cluster, connect_cells_to_clusters,
#     make_cluster_orthogonally_convex, make_cluster_convex, merge_clusters,
#     add_cluster_tags_to_cells, save_full_grid, save_sparse_grid, save_cell_clusters)
from .clusters import Clustering
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


class Bounds(object):
    __slots__ = ('xmin', 'xmax', 'ymin', 'ymax', 'np_array_of_bounds') # use this syntax to save some memory. also only create vars that are really neccessary
    def __init__(self, xmin:float, xmax:float, ymin:float, ymax:float):
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
    #
#



        

class Grid(object):
    """
    A grid used to facilitate radius search and to delineate point clusters
    It store attributes from radius_search / clustering methods, like aggregates per cell 

    ...

    Attributes:
    ----------
    clustering (str): 
        custom class exhibiting methods to map clustered points to cells, merge cluster cells and making clusters convex and adding attributes. For more info help(Clustering)
    plot (aabpl.GridPlots):
        custom class exhibiting methods to create plots. For more info help(aabpl.GridPlots)
    initial_crs (str):
        initial crs of points DataFrame supplied to radius_search, detect_cluster_pts or detect_cluster_cells
    local_crs (str):
        crs automatically choosen by algorithm based on center coordinate of bounding box covering input point data coordinates
    total_bounds (aabpl.Bounds):
        object contaning xmin, xmax, ymin, ymax of full grid  
    spacing (float): 
        the length and width of each grid cell (in meters if no custom projection is used)
    x_steps (numpy.ndarray):
        all x values of grid from xmin to xmax with step size of spacing. Its length is one more than the number of columns of grid.
    y_steps (numpy.ndarray):
        all y values of grid from ymin to ymax with step size of spacing. Its length is one more than the number of rows of grid.
    row_ids (numpy.ndarray):
        ids for grid starting at 0
    col_ids (numpy.ndarray):
        ids for grid starting at 0
    ids (tuple):
        tuple containing all tuple of each cell (row_id, col_id). Sorted row-wise going starting row 0, column 0->n_cols, row 1, column 0->n_cols, ..., row n_rows, column 0->n_cols
    n_cells (int):
        number of cells in grid (=n_rows*n_cols) 
    centroids (numpy.ndarray):
        2D array containing cell centroids. (sorted row-wise)
    row_col_to_centroid (dict):
        dictionary to look up the cells centroid by their row/col index tuple(row_id,col_id)
    row_col_to_bounds (dict):
        dictionary to look up the cells bounds (tuple(tuple(xmin,ymin),tuple(xmax,ymax))) by their row/col index tuple(row_id,col_id)
    
    Methods:
    -------
    create_full_grid_df(target_crs:str=['initial','local','EPSG:4326'][0], max_column_name_length:int=10)
        returns geopandas.GeoDataFrame with entry for each grid cell. Attributes: row, col, geometry, centroid_xy, aggregate of indicator(s), and cluster_id
    create_sparse_grid_df(target_crs:str=['initial','local','EPSG:4326'][0], max_column_name_length:int=10)
        returns geopandas.GeoDataFrame with entry for grid cells that contain a point or is part of a cluster. Attributes: row, col, geometry, centroid_xy, aggregate of indicator(s), and cluster_id
    create_clusters_df_for_column(cluster_column:str, target_crs:str=['initial','local','EPSG:4326'][0], max_column_name_length:int=10)
        returns geopandas.GeoDataFrame with entry for grid cells that either has points inside or is part of a cluster with attributes on their Polygon, centroid, sum of indicator(s), and cluster id
    save_full_grid(filename:str="full_grid", file_format:str=['shp','csv'][0], target_crs:str=['initial','local','EPSG:4326'][0])
        returns and saves geopandas.GeoDataFrame with entry for each grid cell. Attributes: row, col, geometry, centroid_xy, aggregate of indicator(s), and cluster_id
    save_sparse_grid(filename:str="sparse_grid",file_format:str=['shp','csv'][0], target_crs:str=['initial','local','EPSG:4326'][0])
        returns and saves with entry for grid cells that contain a point or is part of a cluster. Attributes: row, col, geometry, centroid_xy, aggregate of indicator(s), and cluster_id
    save_cell_clusters(filename:str="grid_clusters", file_format:str=['shp','csv'][0], target_crs:str=['initial','local','EPSG:4326'][0])
        save each cluster with the Polygon, centroid, sum of indicator(s), area, and cluster id
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
        silent=False,
    ):
        """
        Returns an object of Grid class used to enhance point search and bundle results/methods.

        Two grids are tracked:

        - **Search grid** (``_search_spacing`` and the ``_search_*`` step/id arrays): the cell
          size used internally for the radius search. Chosen automatically via timing models for
          speed (typically ~0.35r-0.71r), or overridden via ``config.FIXED_SPACING_RATIO`` /
          ``config.FIXED_NEST_DEPTH`` before calling. Not meant for user consumption.
        - **Output grid** (``output_spacing`` / ``output_spacing_y`` and ``output_x_steps`` /
          ``output_y_steps``): the user-facing cell size for exports and plots. Defaults to
          ``r/3`` when ``output_spacing`` is None. Per-point aggregates are exact regardless of
          the search cell size, so the output grid may be finer than the search grid.

        ``nest_depth`` belongs to the search grid only.
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
        if data_crs is None:
            data_crs = initial_crs
        self.data_crs = data_crs

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

        self.spacing = spacing
        # Internal search-grid spacing. Alias introduced for the output_spacing
        # refactor (see roadmap): consumers that mean the *search* grid should read
        # `_search_spacing`; user-facing output cell size lives in `output_spacing`.
        # Kept as an alias (not a rename) so existing `grid.spacing` readers are
        # unaffected during the gradual migration.
        self._search_spacing = spacing
        self.nest_depth = nest_depth

        self.nest_height = 0
        self.levels = [0]

        self.clustering = Clustering(self)
        from aabpl.illustrations.plot_grid import GridPlots
        self.plot = GridPlots(self)
        # TODO total_bounds should also contain excluded area if not cntd 
        # min(points.total_bounds+r, max(points.total_bounds, excluded_area_total_bound))  
        self.initial_crs = initial_crs
        self.local_crs = local_crs
        x_padding = ((xmin-xmax) % spacing)/2
        y_padding = ((ymin-ymax) % spacing)/2
        self.total_bounds = total_bounds = Bounds(xmin=xmin-x_padding,xmax=xmax+x_padding,ymin=ymin-y_padding,ymax=ymax+y_padding)
        # n_xsteps = -int((total_bounds.xmin-total_bounds.xmax)/spacing)+1 # round up
        # n_ysteps = -int((total_bounds.ymin-total_bounds.ymax)/spacing)+1 # round up 
        n_xsteps = -int((xmin-xmax)/spacing)+2 # round up
        n_ysteps = -int((ymin-ymax)/spacing)+2 # round up 
        self.x_steps = x_steps = _np_linspace(total_bounds.xmin, total_bounds.xmax, n_xsteps)
        self.y_steps = y_steps = _np_linspace(total_bounds.ymin, total_bounds.ymax, n_ysteps)
        self.row_ids = _np_arange(n_ysteps-1)#[::-1]
        self.col_ids = _np_arange(n_xsteps-1)
        self.n_cells = len(self.row_ids)*len(self.col_ids)
        # Search-grid aliases (see _search_spacing note above). Additive only.
        self._search_x_steps = x_steps
        self._search_y_steps = y_steps
        self._search_row_ids = self.row_ids
        self._search_col_ids = self.col_ids
        self._search_n_cells = self.n_cells

        # Integer cell-key codec: packs (lvl, row, col) into one int64 so the
        # cell-sum aggregation and the per-search offset templates share the same
        # packing and a template translates to a point with one vector add. Sized
        # from the search-grid extent. Always built — the search loop relies on it.
        from aabpl.utils.cell_keys import CellKeyCodec
        self.cell_codec = CellKeyCodec(
            nest_depth=nest_depth,
            row_lo=int(self.row_ids.min()), row_hi=int(self.row_ids.max()),
            col_lo=int(self.col_ids.min()), col_hi=int(self.col_ids.max()),
            offset_margin=16,
        ) 
       
        # Output grid — user-facing cell size for exports and plots.
        # When unset, defaults to r/3: the search grid is chosen for speed (a coarser
        # cell, typically ~0.35r-0.71r), but the output raster should be fine enough to
        # render results crisply. Per-point aggregates are exact regardless of the
        # search cell size, so a finer output grid is well-defined.
        # Output spacing (the user-facing cell size) is fixed at construction, default r/3.
        # The output grid itself (arrays, public-attribute flip, cell aggregates) is built
        # LAZILY by update_spacing() — radius_search alone skips it to avoid the per-point
        # aggregation overhead when no output grid is needed. self.spacing is set to the
        # output spacing immediately so it is truthful right away; the heavier arrays stay
        # on the search grid until update_spacing() runs.
        self.output_spacing = output_spacing if output_spacing is not None else r / 3
        self.output_spacing_y = output_spacing_y if output_spacing_y is not None else self.output_spacing
        self.spacing = self.output_spacing
        self.spacing_y = self.output_spacing_y
        self.output_x_steps = None
        self.output_y_steps = None
        self.output_id_to_sums = {}
        self._output_val_cols = []
        self._spacing_computed = False
        self._raw_bounds = (xmin, xmax, ymin, ymax)

        self.sample_area = None
        self.sample_grid_bounds = None
        self.cells_rndm_sample = set()

        
        # TODO replace row_col_to_centroid with function fetching it from 1d arrays to no clog up memory for fine grids
        
        self.centroids = None
        self.row_col_to_centroid = None

        grid_xmin = total_bounds.xmin
        grid_ymin = total_bounds.ymin
        grid_xmax = total_bounds.xmax
        grid_ymax = total_bounds.ymax
        # self.get_cell_centroid = lambda row, col: (float(grid_xmin+col*(spacing+.5)),float(grid_ymin+row*(spacing+.5)))
        min_row, max_row, min_col, max_col = min(self.row_ids), max(self.row_ids), min(self.row_ids), max(self.col_ids) 
        self.get_cell_centroid = lambda row, col, x_steps=self.x_steps, y_steps=self.y_steps: (
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
        
        self.get_cell_poly = lambda row, col, x_steps=self.x_steps, y_steps=self.y_steps: [
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
        
        self.get_cell_bounds = lambda row, col, x_steps=self.x_steps, y_steps=self.y_steps: [
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

    def get_all_centroids(
                self,
                store:bool=False,
        ):
            if not self.centroids is None:
                return self.centroids
            row_ids = self.row_ids
            col_ids = self.col_ids
            x_steps = self.x_steps
            y_steps = self.y_steps
            centroids = _np_array([centroid for centroid in flatten_list([
                [(x_steps[col_id:col_id+2].mean(), y_steps[row_id:row_id+2].mean()) for col_id in col_ids] 
                for row_id in row_ids]
            )])
            if store:
                self.centroids = centroids
            return centroids
        # self.bounds = flatten_list([
        #         [((x_steps[col_id], y_steps[row_id]), (x_steps[col_id+1], y_steps[row_id+1])) for col_id in col_ids] 
        #         for row_id in row_ids])
        # TODO replace row_col_to_centroid with function fetching it from 1d arrays to no clog up memory for fine grids
        
    def get_all_row_col_to_centroids(
            self,
            store:bool=False,
        ):
        
        if not self.row_col_to_centroid is None:
            return self.row_col_to_centroid
        row_ids = self.row_ids
        col_ids = self.col_ids
        x_steps = self.x_steps
        y_steps = self.y_steps
        row_col_to_centroid = {g_row_col:centroid for (g_row_col,centroid) in flatten_list([
            [((row_id,col_id),(x_steps[col_id:col_id+2].mean(), y_steps[row_id:row_id+2].mean())) for col_id in col_ids] 
            for row_id in row_ids]
            )}
        if store:
            self.row_col_to_centroid = row_col_to_centroid
        return row_col_to_centroid


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
        on the grid as ``self.output_row_name`` / ``self.output_col_name``.

        When output spacing equals internal spacing the existing row/col
        columns are reused (no new columns written to pts).  Otherwise new
        columns ``'out_{row_name}'`` / ``'out_{col_name}'`` are added to pts.
        """
        from numpy import floor as _np_floor
        # row/col columns are assigned on the search grid, so compare against it.
        if self.output_spacing == self._search_spacing and self.output_spacing_y == self._search_spacing:
            self.output_row_name = row_name
            self.output_col_name = col_name
            return
        out_col_name = 'out_' + col_name
        out_row_name = 'out_' + row_name
        pts[out_col_name] = _np_floor(
            (pts[x].values - self.output_x_steps[0]) / self.output_spacing
        ).astype(int)
        pts[out_row_name] = _np_floor(
            (pts[y].values - self.output_y_steps[0]) / self.output_spacing_y
        ).astype(int)
        self.output_row_name = out_row_name
        self.output_col_name = out_col_name

    def aggregate_pts_to_output_cells(
        self,
        pts: _pd_DataFrame,
        val_cols: list,
        x: str = 'lon',
        y: str = 'lat',
        agg: str = 'sum',
    ) -> dict:
        """
        Aggregate per-point values into output grid cells and store in
        ``self.output_id_to_sums``.

        Uses ``self.output_spacing`` / ``self.output_spacing_y`` and
        ``self.output_x_steps`` / ``self.output_y_steps`` — all set in
        ``__init__``.

        Args:
            pts: DataFrame containing ``x``, ``y``, and all ``val_cols``.
            val_cols: Column names to aggregate (must already exist in ``pts``).
            x: X-coordinate column (must be in the same projection as the grid).
            y: Y-coordinate column.
            agg: Aggregation method — ``'sum'``, ``'mean'``, or ``'count'``.

        Returns:
            ``self.output_id_to_sums`` — dict mapping ``(row, col)`` to a
            numpy array of aggregated values (one entry per val_col).
        """
        from numpy import floor as _np_floor
        from pandas import DataFrame as _pd_DataFrame_local

        xmin_out = self.output_x_steps[0]
        ymin_out = self.output_y_steps[0]
        sx = self.output_spacing
        sy = self.output_spacing_y

        col_idx = _np_floor((pts[x].values - xmin_out) / sx).astype(int)
        row_idx = _np_floor((pts[y].values - ymin_out) / sy).astype(int)

        val_cols = list(val_cols)
        tmp = _pd_DataFrame_local({'_row': row_idx, '_col': col_idx})
        for col in val_cols:
            tmp[col] = pts[col].values

        grp = tmp.groupby(['_row', '_col'])[val_cols]
        if stat == 'sum':
            agg_df = grp.sum()
        elif stat == 'mean':
            agg_df = grp.mean()
        elif stat == 'count':
            agg_df = grp.count()
        else:
            raise ValueError(f"agg must be 'sum', 'mean', or 'count'; got {agg!r}")

        self.output_id_to_sums = {rc: row.values for rc, row in agg_df.iterrows()}
        self._output_val_cols = val_cols
        return self.output_id_to_sums

    def update_spacing(self, spacing: float = None, spacing_y: float = None, recompute: bool = False):
        """
        (Re)build the output grid and its cached cell aggregates.

        This is intentionally LAZY: it is NOT run when the grid is created, and
        ``radius_search`` does not call it — computing the output grid means a
        per-point aggregation that is wasted overhead if the caller only wants the
        radius search. It IS called by every consumer that needs the output grid:
        ``detect_cluster_pts`` / ``detect_cluster_cells`` (always), and the plot /
        export methods (``cell_aggregates``, ``clusters``, ``save_*``).

        What it (re)computes:
          - output grid arrays ``output_x_steps`` / ``output_y_steps``;
          - the public grid attributes ``spacing`` / ``x_steps`` / ``y_steps`` /
            ``row_ids`` / ``col_ids`` / ``n_cells`` (re-pointed to the output grid);
          - the cached per-output-cell aggregate of the raw indicator columns in
            ``output_id_to_sums`` (read by ``cell_aggregates`` / ``clusters``);
          - the ``out_*`` cell-id columns on the target points.

        Parameters
        ----------
        spacing : float, optional
            New output cell size. If given and different, the output grid is rebuilt
            for it. Defaults to keeping the current ``output_spacing`` (r/3 unless set
            at construction).
        spacing_y : float, optional
            Output cell height; defaults to ``spacing``.
        recompute : bool
            Force recomputation even if already current.
        """
        changed = False
        if spacing is not None:
            new_y = spacing_y if spacing_y is not None else spacing
            if spacing != self.output_spacing or new_y != self.output_spacing_y:
                self.output_spacing = spacing
                self.output_spacing_y = new_y
                changed = True
        if self._spacing_computed and not changed and not recompute:
            return self

        xmin, xmax, ymin, ymax = self._raw_bounds
        tb = self.total_bounds
        n_out_xsteps = -int((xmin - xmax) / self.output_spacing) + 2
        n_out_ysteps = -int((ymin - ymax) / self.output_spacing_y) + 2
        self.output_x_steps = _np_linspace(tb.xmin, tb.xmax, n_out_xsteps)
        self.output_y_steps = _np_linspace(tb.ymin, tb.ymax, n_out_ysteps)
        # Re-point public grid attributes to the output grid. Search internals read
        # grid._search_*; the get_cell_* lambdas captured the search arrays at init.
        self.spacing = self.output_spacing
        self.spacing_y = self.output_spacing_y
        self.x_steps = self.output_x_steps
        self.y_steps = self.output_y_steps
        self.row_ids = _np_arange(len(self.output_y_steps) - 1)
        self.col_ids = _np_arange(len(self.output_x_steps) - 1)
        self.n_cells = len(self.row_ids) * len(self.col_ids)
        self.row_col_to_centroid = None  # centroid cache depends on the grid
        self.centroids = None

        # Cache per-output-cell aggregate of the RAW indicator columns (what
        # cell_aggregates / clusters display) and assign out_* cell ids.
        search = getattr(self, 'search', None)
        tgt = getattr(search, 'target', None) if search is not None else None
        if tgt is not None and len(getattr(tgt, 'c', [])):
            # Use the projected-coords snapshot taken at set_target so this works
            # even after radius_search dropped proj_x/proj_y from the user's pts.
            snap = getattr(self, '_output_snapshot', None)
            out_pts = snap if snap is not None else tgt.pts
            self.aggregate_pts_to_output_cells(out_pts, val_cols=list(tgt.c), x=tgt.x, y=tgt.y, stat='sum')
            self.assign_output_cell_ids(out_pts, x=tgt.x, y=tgt.y, row_name=tgt.row_name, col_name=tgt.col_name)
            # propagate the out_* cell ids back onto the user's target pts (by index)
            if snap is not None and str(self.output_row_name).startswith('out_'):
                for _oc in (self.output_row_name, self.output_col_name):
                    if _oc in snap.columns:
                        tgt.pts[_oc] = snap[_oc].reindex(tgt.pts.index)
        self._spacing_computed = True
        if not self._silent:
            from aabpl.utils.progress import progress_print
            nr, nc = len(self.row_ids), len(self.col_ids)
            progress_print('Built output grid: ' + str(nr) + '*' + str(nc) + '=' + str(nr * nc) +
                           ' cells with spacing ' + str(self.output_spacing))
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

        # 1. Base shared metrics
        n_shared = {
            "cntd": len(self.search.shared_cntd_cells),
            "ovlpd": 0
        }
        area_shared = {
            "cntd": calc_cells_area(self.search.shared_cntd_cells),
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
        mult = self.search.region_and_trgl_mult
        for reg_id, reg_area in self.search.region_id_to_area.items():
            total_reg_area += reg_area
            total_reg_count += 1

            # Use triangle 1 as representative (all triangles share the same list unless
            # by_trgl overrides exist; for aggregate stats any triangle is equivalent)
            cells_data = {
                "cntd": self.search.region_and_trgl_id_to_distinct_cntd_cells[reg_id * mult + 1],
                "ovlpd": self.search.region_and_trgl_id_to_distinct_ovlpd_cells[reg_id * mult + 1]
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
        r = self.search.r
        spacing = self._search_spacing
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

    def plot_sample_area(self,
        pts:_pd_DataFrame=None,
        x:str=None,
        y:str=None,
        filename:str='',
        plot_kwargs:dict={},
        show:bool=True,):
        from aabpl.illustrations.plot_sample_area import plot_sample_area
        plot_sample_area(
            grid=self,
            pts=pts or self.pts if hasattr(self,"pts") else None,
            x=x or self.x if hasattr(self,"x") else None,
            y=y or self.y if hasattr(self,"y") else None,
            filename=filename,
            plot_kwargs=plot_kwargs,
            show=show,
        )
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
        c_ids = _np_zeros((self.n_cells, len(self.clustering.by_column)),int)#-1
        sums = _np_zeros((self.n_cells, len(self.clustering.by_column)),int)
        polys = []
        id_to_sums = self.id_to_sums
        centroids = _np_array(list(self.get_all_row_col_to_centroids().values()))
        target_crs = self.initial_crs if target_crs=='initial' else self.local_crs if target_crs=='local' else target_crs
        if target_crs != self.local_crs:
            transformer = Transformer.from_crs(crs_from=self.local_crs, crs_to=self.initial_crs, always_xy=True)
            centroids_x, centroids_y = transformer.transform(centroids[:,0], centroids[:,1])
        else:
            centroids_x, centroids_y = centroids[:,0], centroids[:,1]
        clusters_for_columns = list(self.clustering.by_column.values())
        for (i, row_col), ((xmin,ymin),(xmax,ymax)) in zip(enumerate(self.ids), list(self.row_col_to_bounds.values())):
            for clusters_for_column in clusters_for_columns:
                if row_col in clusters_for_column.cell_to_cluster_id: 
                    c_ids[i] = clusters_for_column.cell_to_cluster_id[row_col]
            if row_col in id_to_sums: 
                sums[i] = id_to_sums[row_col]
            polys.append(Polygon(((xmin,ymin),(xmax,ymin),(xmax,ymax),(xmin,ymax))))
        df = _gpd_GeoDataFrame({
            self.search.source.row_name: [row for row,col in self.ids],
            self.search.source.col_name: [col for row,col in self.ids],
            'centroid_x': centroids_x,
            'centroid_y': centroids_y,
            }, geometry=polys,
            crs=self.local_crs
            )
        if len(self.clustering.by_column)<=1:
            df['cluster_id'] = c_ids
            df['sum'] = sums
        else:
            for j, column in enumerate(self.clustering.by_column):
                c_id_colname = find_column_name("cluster_id", column, df.columns, max_column_name_length)
                agg_colname = find_column_name("sum_radius", column, df.columns, max_column_name_length)
                df[c_id_colname] = c_ids[:,j]
                df[agg_colname] = sums[:,j]
        if target_crs != self.local_crs:
            df.to_crs(self.initial_crs, inplace=True)

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
        id_to_sums = self.output_id_to_sums
        x_steps = self.output_x_steps
        y_steps = self.output_y_steps
        out_row_ids = _np_arange(len(y_steps) - 1)
        out_col_ids = _np_arange(len(x_steps) - 1)
        n_out_cells = len(out_row_ids) * len(out_col_ids)
        polys = []
        target_crs = self.initial_crs if target_crs=='initial' else self.local_crs if target_crs=='local' else target_crs
        if target_crs != self.local_crs:
            transformer = Transformer.from_crs(crs_from=self.local_crs, crs_to=target_crs, always_xy=True)
        centroids_x = _np_zeros(n_out_cells, float)
        centroids_y = _np_zeros(n_out_cells, float)
        sparse_row_ids = _np_zeros(n_out_cells, int)
        sparse_col_ids = _np_zeros(n_out_cells, int)
        n_val_cols = max(len(self.clustering.by_column), 1)
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
                if target_crs != self.local_crs:
                    cx, cy = transformer.transform(cx, cy)
                centroids_x[i] = cx
                centroids_y[i] = cy
                polys.append(Polygon(((xmin_c, ymin_c), (xmax_c, ymin_c), (xmax_c, ymax_c), (xmin_c, ymax_c))))
                i += 1
        out_row_name = getattr(self, 'output_row_name', self.search.source.row_name)
        out_col_name = getattr(self, 'output_col_name', self.search.source.col_name)
        df = _gpd_GeoDataFrame({
            out_row_name: sparse_row_ids[:i],
            out_col_name: sparse_col_ids[:i],
            'centroid_x': centroids_x[:i],
            'centroid_y': centroids_y[:i],
            }, geometry=polys,
            crs=self.local_crs
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

        if target_crs != self.local_crs:
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
            crs=self.local_crs)
        
        target_crs = self.initial_crs if target_crs=='initial' else self.local_crs if target_crs=='local' else target_crs
        if target_crs != self.local_crs:
            transformer = Transformer.from_crs(crs_from=self.local_crs, crs_to=target_crs, always_xy=True)
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
        dfs (list)
            list that for each cluster column contains a geopandas.GeoDataFrame with one entry for each cluster 
        """
        dfs = []
        filenames = filename if type(filename) == list else [filename + (("_"+cluster_column) if len(self.clustering.by_column) > 1 else '') for cluster_column in self.clustering.by_column]
        for (cluster_column, clusters), filename in zip(self.clustering.by_column.items(), filenames):
            df = self.save_cell_clusters_for_column(cluster_column=cluster_column, filename=filename, file_format=file_format, target_crs=target_crs)
            dfs.append(df)
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



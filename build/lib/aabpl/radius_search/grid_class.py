from numpy import (
    array as _np_array, 
    linspace as _np_linspace,
    stack as _np_stack,
    arange as _np_arange, 
    unique as _np_unique,
    zeros as _np_zeros,
)
from pyproj import Transformer
from pandas import DataFrame as _pd_DataFrame
from math import log10 as _math_log10, inf as _math_inf
from aabpl.utils.general import flatten_list, find_column_name
from aabpl.utils.crs_transformation import convert_bounds_to_local_crs
from aabpl.illustrations.plot_utils import map_2D_to_rgb, get_2D_rgb_colobar_kwargs
from aabpl.illustrations.grid import GridPlots#plot_cell_sums, plot_grid_ids, plot_clusters, plot_cluster_vars
from aabpl.illustrations.illustrate_sample_area import plot_sample_area
from aabpl.illustrations.plot_pt_vars import create_plots_for_vars
from .radius_search_class import (
    aggregate_point_data_to_cells,
    aggregate_point_data_to_disks_vectorized
)
from .pts_to_offset_regions import assign_points_to_cell_regions
from aabpl.valid_area import disk_cell_intersection_area
from aabpl.testing.test_performance import time_func_perf
# from .clusters import (
#     create_clusters, add_geom_to_cluster, connect_cells_to_clusters,
#     make_cluster_orthogonally_convex, make_cluster_convex, merge_clusters,
#     add_cluster_tags_to_cells, save_full_grid, save_sparse_grid, save_cell_clusters)
from .clusters import Clustering
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union
from geopandas import GeoDataFrame as _gpd_GeoDataFrame
from aabpl.valid_area import process_sample_poly_to_grid
# from decimal import Decimal as _decimal_Decimal, getcontext as _decimal_getcontext

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

def find_optimal_spacing(
    r:float,
    n_pts_src:int, 
    n_pts_tgt:int, 
    x_range:float, 
    y_range:float,
    spacings:list,
    ):
    return r/3
    min_score = _math_inf
    best_s = None
    for s in spacings:
        
        grid_creation_time = 0.1*x_range/s*y_range/s
        create_micro_region_time = 0.1*(r/s)**2+0.2*(r/s)+0.3
        aggregate_cell_based_time = 0.1*x_range/s*y_range/s+0.1*n_pts_tgt+0.1*x_range/s*y_range/s*n_pts_tgt
        search_pts_time = n_pts_src*n_pts_tgt
        score = grid_creation_time + create_micro_region_time + aggregate_cell_based_time + search_pts_time
        if score < min_score:
            min_score = score
            best_s = s
    return best_s
        

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
        local_crs:str='auto',
        set_fixed_spacing:float=None,
        r:float=750,
        n_pts_src:int=None,
        n_pts_tgt:int=None,
        nest_depth:int=None,
        data_crs:str=None,
        silent = False,
        ):

        """
        Returns an object of Grid class, that is used to enhance point search and bundle results and methods
        (xmax-xmin)/spacing is not an integer it will add space evenly on both sides s.t. 'real xmin' is is xmin-((xmax-xmin)%spacing)/2
        Args:
        -------

        xmin:float,
        xmax:float,
        ymin:float,
        ymax:float,
        initial_crs:str,
        local_crs:str,
        set_fixed_spacing:float=None,
        r:float=750,
        n_points:int=10000,
        silent = False,
        """

        if nest_depth is None:
            nest_depth = 0
        

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
        
        
        if not set_fixed_spacing is None:
            spacing = set_fixed_spacing
            if False: # add this after testing
                spacings = [set_fixed_spacing*2**i for i in range(6)[::-1]]+[set_fixed_spacing*(1/2**i) for i in range(1,6)]
                spacings = [set_fixed_spacing]
                spacing = find_optimal_spacing(
                    r=r,
                    n_pts_src=n_pts_src,
                    n_pts_tgt=n_pts_tgt,
                    x_range=xmax-xmin, 
                    y_range=ymax-ymin,
                    spacings=spacings
                )
        else:
            # f(x_range, y_range, r, n_pts_src,n_pts_tgt) -> (spacing, nest_depth)
            # distribution concentration of pts might matter. 
            # but this metric can be slow to compute - careful here.
            if n_pts_src is None:
                n_pts_src = 10000
                print("Specifying 'n_pts_src' can improve performance when searching along grid.")
            if n_pts_tgt is None:
                n_pts_tgt = n_pts_src
            
            spacings = []
            for i in range(1,32):
                spacings.append(r/i)
                for j in range(1,i+1):
                    k = (i**2+j**2)**.5
                    if k <=32:
                        spacings.append(r/k)
            #spacings add break points like r/2**.5
            spacing = find_optimal_spacing(
                n_pts_src=n_pts_src,
                n_pts_tgt=n_pts_tgt,
                x_range=xmax-xmin, 
                y_range=ymax-ymin,
                spacings=spacings
            )

        self.spacing = spacing
        self.nest_depth = nest_depth
        
        self.nest_height = 0
        self.levels = [0]
        self.ref_lvl = 0
        
        self.clustering = Clustering(self)
        self.plot = GridPlots(self)
        # TODO total_bounds should also contain excluded area if not contained 
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

        if not silent:
            print('Create grid with '+str(n_ysteps-1)+'*'+str(n_xsteps-1)+'='+str((n_ysteps-1)*(n_xsteps-1))+" with spacing "+str(spacing))
        #
    #
    # add functions
    aggregate_point_data_to_cells = aggregate_point_data_to_cells
    assign_points_to_cell_regions = assign_points_to_cell_regions # OUTDATED? REPLACE WITH assign_points_to_mirco_regions?
    process_sample_poly_to_grid = process_sample_poly_to_grid
    aggregate_point_data_to_disks_vectorized = aggregate_point_data_to_disks_vectorized
    disk_cell_intersection_area = disk_cell_intersection_area
    
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
        close_plot:bool=False,):
        plot_sample_area(
            grid=self,
            pts=pts or self.pts if hasattr(self,"pts") else None,
            x=x or self.x if hasattr(self,"x") else None,
            y=y or self.y if hasattr(self,"y") else None,
            filename=filename,
            plot_kwargs=plot_kwargs,
            close_plot=close_plot,
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
        
        id_to_sums = self.id_to_sums
        x_steps = self.x_steps
        y_steps = self.y_steps
        col_ids = self.col_ids
        centroids = self.get_all_centroids()
        polys = []
        target_crs = self.initial_crs if target_crs=='initial' else self.local_crs if target_crs=='local' else target_crs
        if target_crs != self.local_crs:
            transformer = Transformer.from_crs(crs_from=self.local_crs, crs_to=target_crs, always_xy=True)
            centroids_x_full, centroids_y_full = transformer.transform(centroids[:,0], centroids[:,1])
        else:
            centroids_x_full, centroids_y_full = centroids[:,0], centroids[:,1]
        centroids_x = _np_zeros(self.n_cells,float)
        centroids_y = _np_zeros(self.n_cells,float)
        sparse_row_ids = _np_zeros(self.n_cells,int)
        sparse_col_ids = _np_zeros(self.n_cells,int)
        sums = _np_zeros((self.n_cells, len(self.clustering.by_column)),float)
        c_ids = _np_zeros((self.n_cells, len(self.clustering.by_column)),int)
        i = 0
        js_clusters_for_columns = [x for x in enumerate(self.clustering.by_column.values())]
        for row in self.row_ids:
            (ymin,ymax) = (y_steps[row], y_steps[row+1])
            sparse_row_ids[i:] = row
            for col in col_ids:
                cell_in_a_cluster = False
                for j, clusters_for_column in js_clusters_for_columns:
                    if (row,col) in clusters_for_column.cell_to_cluster_id: 
                        c_ids[i,j] = clusters_for_column.cell_to_cluster_id[(row,col)]
                        cell_in_a_cluster = True
                
                if (row,col) in id_to_sums: 
                    sums[i] = id_to_sums[(row,col)]
                elif not cell_in_a_cluster:
                    continue
                sparse_col_ids[i] = col
                (xmin, xmax) = (x_steps[col], x_steps[col+1])
                centroids_x[i] = centroids_x_full[row*col+col] 
                centroids_y[i] = centroids_y_full[row*col+col]
                polys.append(Polygon(((xmin,ymin),(xmax,ymin),(xmax,ymax),(xmin,ymax))))
                i += 1
            #
        #
        df = _gpd_GeoDataFrame({
            self.search.source.row_name: sparse_row_ids[i],
            self.search.source.col_name: sparse_col_ids[i],
            'centroid_x': centroids_x[:i],
            'centroid_y': centroids_y[:i],
            }, geometry=polys,
            crs=self.local_crs
            )
        if len(self.clustering.by_column)<=1:
            df['cluster_id'] = c_ids[:i]
            df['sum'] = sums[:i]
        else:
            for j, column in enumerate(self.clustering.by_column):
                c_id_colname = find_column_name("cluster_id", column, df.columns, max_column_name_length)
                agg_colname = find_column_name("sum_radius", column, df.columns, max_column_name_length)
                df[c_id_colname] = c_ids[:i,j]
                df[agg_colname] = sums[:i,j]
        
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

class ExludedArea:
    def __init__(self,excluded_area_geometry_or_list, grid:Grid):
        # recursively split exluded area geometry along grid 
        # then sort it into grid cell
        
        pass
#



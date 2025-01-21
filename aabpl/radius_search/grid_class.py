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
from math import log10 as _math_log10
from aabpl.utils.general import flatten_list, find_column_name
from aabpl.illustrations.plot_utils import map_2D_to_rgb, get_2D_rgb_colobar_kwargs
from aabpl.illustrations.grid import GridPlots#plot_cell_sums, plot_grid_ids, plot_clusters, plot_cluster_vars
from aabpl.illustrations.plot_pt_vars import create_plots_for_vars
from .radius_search_class import (
    aggregate_point_data_to_cells,
    aggreagate_point_data_to_disks_vectorized
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

class Bounds(object):
    __slots__ = ('xmin', 'xmax', 'ymin', 'ymax', 'np_array_of_bounds') # use this syntax to save some memory. also only create vars that are really neccessary
    def __init__(self, xmin:float, xmax:float, ymin:float, ymax:float):
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
    #
#

# TODO potentially add nestes GridCell classtype the 
class GridCell(object):
    # __slots__ = ('id', 'row_col_nr', ...) # use this syntax to save some memory. also only create vars that are really neccessary
    def __init__(self, row_nr, col_nr, y_steps, x_steps):
        self.id = (row_nr,col_nr)
        self.centroid = (y_steps[row_nr:row_nr+2].sum()/2,x_steps[col_nr:col_nr+2].sum()/2)
        self.bounds = Bounds(xmin=x_steps[col_nr], ymin=y_steps[row_nr], xmax=x_steps[col_nr+1], ymax=y_steps[row_nr+1])
        self.pt_ids = []
        self.excluded = None

    def add_pt_id(self, pt_id):
        self.pt_ids = [*self.pt_ids, pt_id]
        
    def add_pt_ids(self, pt_ids):
        self.pt_ids = [*self.pt_ids, *pt_ids]
    
    def set_excluded_area(self,excluded_area):
        pass

    def make_sparse(self):
        # self.pt_ids # make _np_array or tuple
        # tighten bounds
        pass            
    def tighten_bounds(self, pt_ids):
        self.pt_ids = [*self.pt_ids, *pt_ids]
    
    def plot(self, facecolor, edgecolor, ):
        pass
#

    

class Grid(object):
    """
    A grid used to facilitate radius search and to delineate point clusters
    It store attributes from radius_search / clustering methods, like aggregates per cell 

    ...

    Attributes
    ----------
    spacing : float
        the length and width of each grid cell (in meters if no custom projection is used)
    name : str
        the name of the animal
    plot (aabpl.GridPlots):
        custom class exhibiting methods to create plots
    
    Methods
    -------
    save_full_grid(filename:str="full_grid", file_format:str=['shp','csv'][0])
        save each grid cell with attributes on their Polygon, centroid, sum of indicator(s), and cluster id
    save_sparse_grid(filename:str="sparse_grid",file_format:str=['shp','csv'][0])
        save each grid cell, that is non-empty or part of a cluster, with attributes on their Polygon, centroid, sum of indicator(s), and cluster id
    save_cell_clusters(filename:str="grid_clusters", file_format:str=['shp','csv'][0])
        save each cluster with the Polygon, centroid, sum of indicator(s), area, and cluster id
    plot_clusters(fig=None, axs=None, filename:str='')
        xxx
    """
    @time_func_perf
    def __init__(
        self,
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
        ):

        """

        """
        if set_fixed_spacing:
            spacing = set_fixed_spacing
        else:
            # find optimal spacing TODO
            print("TODO find optimal spacing for",r, n_points)
            spacing = 1.
        self.spacing = spacing

        # TODO total_bounds should also contain excluded area if not contained 
        # min(points.total_bounds+r, max(points.total_bounds, excluded_area_total_bound))  
        self.initial_crs = initial_crs
        self.local_crs = local_crs
        x_padding = ((xmin-xmax) % spacing)/2
        y_padding = ((ymin-ymax) % spacing)/2
        self.total_bounds = total_bounds = Bounds(xmin=xmin-x_padding,xmax=xmax+x_padding,ymin=ymin-y_padding,ymax=ymax+y_padding)
        self.n_x_steps = n_x_steps = -int((self.total_bounds.xmin-self.total_bounds.xmax)/spacing)+1 # round up
        self.n_y_steps = n_y_steps = -int((self.total_bounds.ymin-self.total_bounds.ymax)/spacing)+1 # round up 
        self.x_steps = x_steps = _np_linspace(total_bounds.xmin, total_bounds.xmax, n_x_steps)
        self.y_steps = y_steps = _np_linspace(total_bounds.ymin, total_bounds.ymax, n_y_steps)
        self.id_y_mult = id_y_mult = 10**(int(_math_log10(n_x_steps))+1)
        self.row_ids = row_ids = _np_arange(n_y_steps-1)
        self.col_ids = col_ids =  _np_arange(n_x_steps-1)
        self.ids = tuple(flatten_list([[(row_id, col_id) for col_id in col_ids] for row_id in row_ids]))
        self.n_cells = len(self.ids)
        class CellDict(dict):
            def __init__(self, x_steps, y_steps, id_y_mult):
                self.x_steps = x_steps
                self.y_steps = y_steps
                self.id_y_mult = id_y_mult

            def id_to_row_col(self,id:int): 
                return (id // self.id_y_mult, id % self.id_y_mult)
            
            def row_col_to_id(self,row_nr:int,col_nr:int): 
                return row_nr * self.id_y_mult + col_nr
            
            def add_new(self,row,col):
                setattr(self, str(self.row_col_to_id(row,col)), GridCell(
                    row, col, self.x_steps, self.y_steps
                ))
            
            def get_by_row_col(self,row,col):
                return getattr(self, str(self.row_col_to_id(row,col)))
            
            def get_by_id(self,id):
                return getattr(self, str(id))
            
            def get_or_create(self,row,col):
                id = str(self.row_col_to_id(row,col))
                if not hasattr(self, id):
                    self.add_new(row,col)
                return self.get_by_id(id)
            
            def add_pts(self, pts, row, col):
                cell = self.get_or_create(row,col)
                cell.add_pts(pts)
            
            def add_pt(self, pt, row, col):
                cell = self.get_or_create(row,col)
                cell.add_pt(pt)
            
        self.cells = CellDict(x_steps=x_steps, y_steps=y_steps, id_y_mult=id_y_mult,)

        self.cells = _np_array([[
            GridCell(row_id, col_id, y_steps=y_steps, x_steps=x_steps) for col_id in col_ids
            ] for row_id in row_ids]).flatten()

        self.row_col_to_centroid = {g_row_col:centroid for (g_row_col,centroid) in flatten_list([
                [((row_id,col_id),(x_steps[col_id:col_id+2].mean(), y_steps[row_id:row_id+2].mean())) for col_id in col_ids] 
                for row_id in row_ids]
                )}
        self.centroids = _np_array([centroid for centroid in flatten_list([
                [(x_steps[col_id:col_id+2].mean(), y_steps[row_id:row_id+2].mean()) for col_id in col_ids] 
                for row_id in row_ids]
                )])
        self.row_col_to_bounds = {(row_id,col_id): bounds for ((row_id,col_id),bounds) in flatten_list([
                [((row_id,col_id),((x_steps[col_id], y_steps[row_id]), (x_steps[col_id+1], y_steps[row_id+1]))) for col_id in col_ids] 
                for row_id in row_ids]
                )}
        self.bounds = flatten_list([
                [((x_steps[col_id], y_steps[row_id]), (x_steps[col_id+1], y_steps[row_id+1])) for col_id in col_ids] 
                for row_id in row_ids])
        self.cell_dict = dict()
        self.clustering = Clustering(self)
        self.plot = GridPlots(self)
        if not silent:
            print('Create grid with '+str(n_y_steps-1)+'x'+str(n_x_steps-1)+'='+str((n_y_steps-1)*(n_x_steps-1)))
        #
    #

        
    # add functions
    aggregate_point_data_to_cells = aggregate_point_data_to_cells
    assign_points_to_cell_regions = assign_points_to_cell_regions
    aggreagate_point_data_to_disks_vectorized = aggreagate_point_data_to_disks_vectorized
    disk_cell_intersection_area = disk_cell_intersection_area
    
    # append plots
    # plt = {
    #     cell_sums = plot_cell_sums
    #     grid_ids = plot_grid_ids
    #     clusters = plot_clusters
    #     cluster_vars = plot_cluster_vars
    #     vars = create_plots_for_vars
    # }
    
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

    def create_full_grid_df(self, keep_initial_crs:bool=True, max_column_name_length:int=20):
        """create df with row for each grid cell with attributes on their Polygon, centroid, sum of indicator(s), and cluster id
        """
        c_ids = _np_zeros((self.n_cells, len(self.clustering.by_column)),int)#-1
        sums = _np_zeros((self.n_cells, len(self.clustering.by_column)),int)
        polys = []
        id_to_sums = self.id_to_sums
        centroids = _np_array(list(self.row_col_to_centroid.values()))
        if keep_initial_crs:
            transformer = Transformer.from_crs(crs_from=self.local_crs, crs_to=self.initial_crs, always_xy=True)
            centroids_x, centroids_y = transformer.transform(centroids[:,0], centroids[:,1])
        else:
            centroids_x, centroids_y = centroids[:,0], centroids[:,1]
        clusters_for_columns = list(self.clustering.by_column.values())
        for (i, row_col), ((xmin,ymin),(xmax,ymax)) in zip(enumerate(self.ids), self.bounds):
            for clusters_for_column in clusters_for_columns:
                if row_col in clusters_for_column.cell_to_cluster_id: 
                    c_ids[i] = clusters_for_column.cell_to_cluster_id[row_col]
            if row_col in id_to_sums: 
                sums[i] = id_to_sums[row_col]
            polys.append(Polygon(((xmin,ymin),(xmax,ymin),(xmax,ymax),(xmin,ymax))))
        df = _gpd_GeoDataFrame({
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
        if keep_initial_crs:
            df.to_crs(self.initial_crs, inplace=True)

        return df
    #

    def create_sparse_grid_df(
            self, keep_initial_crs:bool=True, max_column_name_length:int=20
        ):
        """
        save each grid cell with attributes on their Polygon, centroid, sum of indicator(s), and cluster id
        
        """
        id_to_sums = self.id_to_sums
        x_steps = self.x_steps
        y_steps = self.y_steps
        col_ids = self.col_ids
        polys = []
        if keep_initial_crs:
            transformer = Transformer.from_crs(crs_from=self.local_crs, crs_to=self.initial_crs, always_xy=True)
            centroids_x_full, centroids_y_full = transformer.transform(self.centroids[:,0], self.centroids[:,1])
        else:
            centroids_x_full, centroids_y_full = self.centroids[:,0], self.centroids[:,1]
        centroids_x = _np_zeros(self.n_cells,float)
        centroids_y = _np_zeros(self.n_cells,float)
        sums = _np_zeros((self.n_cells, len(self.clustering.by_column)),float)
        c_ids = _np_zeros((self.n_cells, len(self.clustering.by_column)),int)
        i = 0
        js_clusters_for_columns = [x for x in enumerate(self.clustering.by_column.values())]
        for row in self.row_ids:
            (ymin,ymax) = (y_steps[row], y_steps[row+1])
            for col in col_ids:
                cell_in_a_cluster = False
                for j, clusters_for_column in js_clusters_for_columns:
                    if (row,col) in clusters_for_column.cell_to_cluster_id: 
                        c_ids[i] = clusters_for_column.cell_to_cluster_id[(row,col)]
                        cell_in_a_cluster = True
                
                if (row,col) in id_to_sums: 
                    sums[i] = id_to_sums[(row,col)]
                elif not cell_in_a_cluster:
                    continue
                (xmin, xmax) = (x_steps[col], x_steps[col+1])
                centroids_x[i] = centroids_x_full[row*col+col] 
                centroids_y[i] = centroids_y_full[row*col+col]
                polys.append(Polygon(((xmin,ymin),(xmax,ymin),(xmax,ymax),(xmin,ymax))))
                i += 1
            #
        #
        df = _gpd_GeoDataFrame({
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
        
        if keep_initial_crs:
            df.to_crs(self.initial_crs, inplace=True)
        
        return df
    #

    def create_clusters_df_for_column(self, cluster_column, keep_initial_crs:bool=True):
        clusters_for_column = self.clustering.by_column[cluster_column]
        df = _gpd_GeoDataFrame({
            'centroid_x': [cluster.centroid[0] for cluster in clusters_for_column.clusters],
            'centroid_y': [cluster.centroid[1] for cluster in clusters_for_column.clusters],
            'cluster_id': [cluster.id for cluster in clusters_for_column.clusters],
            'sum': [cluster.aggregate for cluster in clusters_for_column.clusters],
            "n_cells": [cluster.n_cells for cluster in clusters_for_column.clusters],
            'area': [cluster.area for cluster in clusters_for_column.clusters],
            },
            geometry = [cluster.geometry for cluster in clusters_for_column.clusters],
            crs=self.local_crs)
        
        if keep_initial_crs:
            transformer = Transformer.from_crs(crs_from=self.local_crs, crs_to=self.initial_crs, always_xy=True)
            df['centroid_x'], df['centroid_y'] = transformer.transform(df['centroid_x'], df['centroid_y'])
            df.to_crs(self.initial_crs, inplace=True)
        
        return df
    #

    def save_full_grid(
            self,
            filename:str="full_grid",
            file_format:str=['shp','csv'][0],
            keep_initial_crs:bool=True,
    ):
        """save each grid cell with attributes on their Polygon, centroid, sum of indicator(s), and cluster id
        
        Args:
        filename (str):
            name of the output file excluding file format extension. It can contain full path like 'output_folder/fname'
        file_format (str):
            format in which the file shall be saved. Currently available options are 'shp' and 'csv'. Extension will be appended to filename.
        """
        
        df = self.create_full_grid_df(keep_initial_crs=keep_initial_crs, max_column_name_length=10 if file_format=='shp' else 20)
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
            keep_initial_crs:bool=True,
        ):
        """
        save each grid cell with attributes on their Polygon, centroid, sum of indicator(s), and cluster id
        
        Args:
        filename (str):
            name of the output file excluding file format extension. It can contain full path like 'output_folder/fname'
        file_format (str):
            format in which the file shall be saved. Currently available options are 'shp' and 'csv'. Extension will be appended to filename.
        
        """
        
        df = self.create_sparse_grid_df(
            keep_initial_crs=keep_initial_crs, max_column_name_length=10 if file_format=='shp' else 20
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
            keep_initial_crs:bool=True,
    ):
        df = self.create_clusters_df_for_column(cluster_column=cluster_column, keep_initial_crs=keep_initial_crs)
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
        ):
        """
        Saves all cell clusters
        """
        dfs = []
        for (cluster_column, clusters) in self.clustering.by_column.items():
            filename = filename + (("_"+cluster_column) if len(self.clustering.by_column) > 1 else '')
            df = self.save_cell_clusters_for_column(cluster_column=cluster_column, filename=filename, file_format=file_format)
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



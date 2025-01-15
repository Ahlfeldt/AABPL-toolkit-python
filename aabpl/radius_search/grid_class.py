from numpy import (
    array as _np_array, 
    linspace as _np_linspace,
    stack as _np_stack,
    arange as _np_arange, 
    unique, invert, flip, transpose, concatenate, sign, zeros, 
    min, max, equal, where, logical_or, logical_and, all, newaxis
)
# from numpy.linalg import norm
# from numpy.random import randint, random
from pandas import DataFrame as _pd_DataFrame
from math import log10 as _math_log10#,ceil,asin,acos
from aabpl.utils.general import ( flatten_list, visualize, depth )
from aabpl.illustrations.illustrate_optimal_grid_spacing import ( create_optimal_grid_spacing_gif, )
from aabpl.illustrations.plot_utils import map_2D_to_rgb, get_2D_rgb_colobar_kwargs
from aabpl.utils.distances_to_cell import ( get_always_contained_potentially_overlapped_cells, )
# from .nested_search import (aggregate_point_data_to_nested_cells, aggreagate_point_data_to_disks_vectorized_nested)
from .radius_search_class import (
    aggregate_point_data_to_cells,
    assign_points_to_cell_regions,
    aggreagate_point_data_to_disks_vectorized
)
from aabpl.valid_area import disk_cell_intersection_area
# from .radius_search.optimal_grid_spacing import (select_optimal_grid_spacing,)
from aabpl.testing.test_performance import time_func_perf, func_timer_dict
import matplotlib.pyplot as plt


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

    """
    @time_func_perf
    def __init__(
        self,
        xmin:float,
        xmax:float,
        ymin:float,
        ymax:float,
        set_fixed_spacing:float=None,
        radius:float=750,
        n_points:int=10000,
        silent = False,
        ):

        """

        """
        if set_fixed_spacing:
            spacing = set_fixed_spacing
        else:
            # find optimal spacing TODO
            print("TODO find optimal spacing for",radius, n_points)
            spacing = 1.
        self.spacing = spacing

        # TODO total_bounds should also contain excluded area if not contained 
        # min(points.total_bounds+radius, max(points.total_bounds, excluded_area_total_bound))  
      
        x_padding = ((xmin-xmax) % spacing)/2
        y_padding = ((ymin-ymax) % spacing)/2
        
        self.n_x_steps = n_x_steps = -int((xmin-xmax)/spacing) # round up
        self.n_y_steps = n_y_steps = -int((ymin-ymax)/spacing) # round up 
        self.total_bounds = total_bounds = Bounds(xmin=xmin-x_padding,xmax=xmax+x_padding,ymin=ymin-y_padding,ymax=ymax+y_padding)
        
        self.x_steps = x_steps = _np_linspace(total_bounds.xmin, total_bounds.xmax, n_x_steps)
        # self.y_steps = y_steps = _np_linspace(total_bounds.ymax, total_bounds.ymin, n_y_steps)
        self.y_steps = y_steps = _np_linspace(total_bounds.ymin, total_bounds.ymax, n_y_steps)
        
        self.id_y_mult = id_y_mult = 10**(int(_math_log10(n_x_steps))+1)
        
        self.row_ids = _np_arange(n_y_steps-1)
        self.col_ids = _np_arange(n_x_steps-1)
        
        self.ids = tuple(flatten_list([[(row_id, col_id) for col_id in range(n_x_steps-1)] for row_id in range(n_y_steps-1)]))
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
            
        self.row_col_stack = _np_stack([
                    _np_array([[row_id for col_id in range(n_x_steps-1)] for row_id in range(n_y_steps-1)]).flatten(),
                    _np_array([[col_id for col_id in range(n_x_steps-1)] for row_id in range(n_y_steps-1)]).flatten(),
                ])
        self.cells = CellDict(x_steps=x_steps, y_steps=y_steps, id_y_mult=id_y_mult,)

        self.cells = _np_array([[
            GridCell(row_id, col_id, y_steps=y_steps, x_steps=x_steps) for col_id in range(n_x_steps-1)
            ] for row_id in range(n_y_steps-1)]).flatten()

        self.row_col_to_centroid = {g_row_col:centroid for (g_row_col,centroid) in flatten_list([
                [((row_id,col_id),(x_steps[col_id:col_id+2].mean(), y_steps[row_id:row_id+2].mean())) for col_id in range(n_x_steps-1)] 
                for row_id in range(n_y_steps-1)]
                )}
        self.centroids = _np_array([centroid for centroid in flatten_list([
                [(x_steps[col_id:col_id+2].mean(), y_steps[row_id:row_id+2].mean()) for col_id in range(n_x_steps-1)] 
                for row_id in range(n_y_steps-1)]
                )])
        self.id_to_bounds = {row_id*id_y_mult+col_id: bounds for ((row_id,col_id),bounds) in flatten_list([
                [((row_id,col_id),((x_steps[col_id], y_steps[row_id]), (x_steps[col_id+1], y_steps[row_id+1]))) for col_id in range(n_x_steps-1)] 
                for row_id in range(n_y_steps-1)]
                )}
        self.bounds = flatten_list([
                [((x_steps[col_id], y_steps[row_id]), (x_steps[col_id+1], y_steps[row_id+1])) for col_id in range(n_x_steps-1)] 
                for row_id in range(n_y_steps-1)])
        self.cell_dict = dict()
        if not silent:
            print('Create grid with '+str(n_y_steps-1)+'x'+str(n_x_steps-1)+'='+str((n_y_steps-1)*(n_x_steps-1)))
        #
    #
    # add functions
    aggregate_point_data_to_cells = aggregate_point_data_to_cells
    # aggregate_point_data_to_nested_cells = aggregate_point_data_to_nested_cells
    assign_points_to_cell_regions = assign_points_to_cell_regions
    aggreagate_point_data_to_disks_vectorized = aggreagate_point_data_to_disks_vectorized
    # aggreagate_point_data_to_disks_vectorized_nested = aggreagate_point_data_to_disks_vectorized_nested
    # append a variable to pts_df that indicates the share of valid area float[0,1] 
    disk_cell_intersection_area = disk_cell_intersection_area
    def create_sparse_grid(self):
        
        
        return
        # TODO shall this be another class? A new instane of the grid? Or create a sparse copy within itself? Or overwrite itself with an own c?
        print("Make this grid sparse")
    def create_nested_grid(self):
        print("Create nested grid")
    #

    def plot_grid(self, fig=None, ax=None,):
        if ax is None:
            fig, ax = plt.subplots(nrows=3, figsize=(15,25))
            imshow_kwargs = {
                'xmin':self.x_steps.min(),
                'ymin':self.y_steps.min(),
                'xmax':self.x_steps.max(),
                'ymax':self.y_steps.max(),
            }
            extent=[imshow_kwargs['xmin'],imshow_kwargs['xmax'],imshow_kwargs['ymax'],imshow_kwargs['ymin']]
            X = _np_array([[map_2D_to_rgb(x,y, **imshow_kwargs) for x in  self.x_steps[:-1]] for y in self.y_steps[:-1]])
            # ax.flat[0].imshow(X=X, interpolation='none', extent=extent)
            # ax.flat[0].pcolormesh([self.x_steps, self.y_steps], X)
            ax.flat[0].pcolormesh(X, edgecolor="black", linewidth=1/max([self.n_x_steps, self.n_y_steps])/1.35)
            # ax.flat[0].set_aspect(2)
            colorbar_kwargs = get_2D_rgb_colobar_kwargs(**imshow_kwargs)
            cb = plt.colorbar(**colorbar_kwargs[2], ax=ax.flat[0])
            cb.ax.set_xlabel("diagonal")
            cb = plt.colorbar(**colorbar_kwargs[0], ax=ax.flat[0])
            cb.ax.set_xlabel("x/lon")
            cb = plt.colorbar(**colorbar_kwargs[1], ax=ax.flat[0])
            cb.ax.set_xlabel("y/lat") 
            ax.flat[0].set_xlabel('x/lon') 
            ax.flat[0].set_ylabel('y/lat') 
            ax.flat[0].title.set_text("Grid lat / lon coordinates")
            # ax.flat[0].set_xticks(self.x_steps, minor=True)
            # ax.flat[0].set_yticks(self.y_steps, minor=True)
            # ax.flat[0].grid(which='minor', color='w', linestyle='-', linewidth=0.002)

            imshow_kwargs = {
                'xmin':self.col_ids.min(),
                'ymin':self.row_ids.min(),
                'xmax':self.col_ids.max(),
                'ymax':self.row_ids.max(),
            }
            extent=[imshow_kwargs['xmin'],imshow_kwargs['xmax'],imshow_kwargs['ymax'],imshow_kwargs['ymin']]

            X = _np_array([[map_2D_to_rgb(x,y, **imshow_kwargs) for x in  self.col_ids] for y in self.row_ids])
            ax.flat[1].imshow(X=X, interpolation='none', extent=extent)
            # ax.flat[1].set_aspect(2)
            colorbar_kwargs = get_2D_rgb_colobar_kwargs(**imshow_kwargs)
            cb = plt.colorbar(**colorbar_kwargs[2], ax=ax.flat[1])
            cb.ax.set_xlabel("diagonal")
            cb = plt.colorbar(**colorbar_kwargs[0], ax=ax.flat[1])
            cb.ax.set_xlabel("col nr")
            cb = plt.colorbar(**colorbar_kwargs[1], ax=ax.flat[1])
            cb.ax.set_xlabel("row nr") 
            ax.flat[1].set_xlabel('row nr') 
            ax.flat[1].set_ylabel('col nr') 
            ax.flat[1].title.set_text("Grid row / col indices")
            # ax.flat[1].set_xticks(self.col_ids, minor=True)
            # ax.flat[1].set_yticks(self.row_ids, minor=True)
            # ax.flat[1].grid(which='minor', color='black', linestyle='-', linewidth=0.003)
            
            X = _np_array([[len(self.id_to_pt_ids[(row_id, col_id)]) if (row_id, col_id) in self.id_to_pt_ids else 0 for col_id in self.col_ids] for row_id in self.row_ids])
            p = ax.flat[2].pcolormesh(X, cmap='Reds')
            ax.flat[2].set_xlabel('row nr') 
            ax.flat[2].set_ylabel('col nr') 
            plt.colorbar(p)
#

class ExludedArea:
    def __init__(self,excluded_area_geometry_or_list, grid:Grid):
        # recursively split exluded area geometry along grid 
        # then sort it into grid cell
        
        pass
#



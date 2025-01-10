from numpy import (
    array as _np_array,
    ndarray as _np_ndarray,
    arange as _np_arange,
    hstack as _np_hstack,
    zeros as _np_zeros,
    unique as _np_unique, 
    concatenate as _np_concatenate,
    equal as _np_equal, 
    logical_or as _np_logical_or, 
    all as _np_all, 
)
from pandas import (DataFrame as _pd_DataFrame, cut as _pd_cut, concat as _pd_concat) 
from primelocations.utils.general import ( DataFrameRelation, arr_to_tpls)
# from primelocations.illustrations.illustrate_point_to_disk import (illustrate_point_disk)
# from primelocations.illustrations.plot_pt_vars import create_plots_for_vars
from primelocations.utils.distances_to_cell import (get_cells_relevant_for_disk_by_type,)
from .two_dimensional_weak_ordering_class import (gen_weak_order_rel_to_convex_set,)
from .pts_to_cells import (assign_points_to_cells, aggregate_point_data_to_cells,)
from .pts_to_offset_regions import (assign_points_to_cell_regions,assign_points_to_mirco_regions)
from .pts_radius_search import (aggreagate_point_data_to_disks_vectorized)
from primelocations.testing.test_performance import time_func_perf, func_timer_dict


################ DiskSearchSource ######################################################################################
class DiskSearchObject(object):

    def assign_pts_to_cells(
        self,
        silent:bool = False
    ):
        return assign_points_to_cells(
            grid=self.grid,
            pts_df=self.pts_df,
            y_coord_name=self.y_coord_name,
            x_coord_name=self.x_coord_name,
            row_name=self.row_name,
            col_name=self.col_name,
            silent=silent,
        )
    #
#

################ DiskSearchSource ######################################################################################
class DiskSearchSource(DiskSearchObject):
    def __init__(
        self,
        grid,
        pts_df:_pd_DataFrame,
        y_coord_name:str='lat',
        x_coord_name:str='lon',
        row_name:str='id_y',
        col_name:str='id_x',
        cell_region_name:str='cell_region',
        sum_suffix:str='_750m',
    ):
        self.grid = grid 
        self.pts_df = pts_df
        self.y_coord_name = y_coord_name
        self.x_coord_name = x_coord_name
        self.row_name = row_name
        self.col_name = col_name
        self.cell_region_name = cell_region_name 
        self.sum_suffix = sum_suffix
    #

    def assign_pts_to_cell_regions(
            self,
            plot_cell_reg_assign:dict=None,
            plot_offset_checks:dict=None,
            plot_offset_regions:dict=None,
            plot_offset_raster:dict=None,
            silent:bool=False,
    ):
        # TODO 
        return assign_points_to_mirco_regions( 
        # return assign_points_to_cell_regions(
            grid=self.grid,
            pts_df=self.pts_df,
            radius=self.grid.search.radius,
            include_boundary=self.grid.search.include_boundary,
            y_coord_name=self.y_coord_name,
            x_coord_name=self.x_coord_name,
            row_name=self.row_name,
            col_name=self.col_name,
            cell_region_name=self.cell_region_name,
            plot_cell_reg_assign=plot_cell_reg_assign,
            plot_offset_checks=plot_offset_checks,
            plot_offset_regions=plot_offset_regions,
            plot_offset_raster=plot_offset_raster,
            silent=silent,
        )
    #
#

################ DiskSearchTarget ######################################################################################
class DiskSearchTarget(DiskSearchObject):
    def __init__(
        self,
        grid,
        pts_df:_pd_DataFrame,
        sum_names:list=['employment'],
        y_coord_name:str='lat',
        x_coord_name:str='lon',
        row_name:str='id_y',
        col_name:str='id_x',
    ):
        self.grid = grid 
        self.pts_df = pts_df
        self.sum_names = sum_names
        self.x_coord_name = x_coord_name
        self.y_coord_name = y_coord_name
        self.row_name = row_name
        self.col_name = col_name
        
        # prepare dicts for fast lookup of values for point ids
        self.pt_id_to_xy_coords = {
            pt_id:xy for (pt_id,xy) in zip(pts_df.index, pts_df[[x_coord_name,y_coord_name]].values)
        }
        self.pt_id_to_vals = {
            pt_id:pt_vals for (pt_id,pt_vals) in zip(pts_df.index, pts_df[sum_names].values)
        }
    #
    
    def aggregate_pt_data_to_cells(
            self,
            silent
    ):
        return aggregate_point_data_to_cells(
            grid=self.grid,
            pts_df=self.pts_df,
            sum_names=self.sum_names,
            row_name=self.row_name,
            col_name=self.col_name,
            silent=silent,
        )
    #
#
   

################ DiskSearch ######################################################################################
class DiskSearch(object):
    def __init__(
        self,
        grid,
        exclude_pt_itself:bool=True,
        radius:float=0.0075,
        include_boundary:bool=False,
    ):
            
        """
        
        """
        # link to grid
        grid.search = self
        self.grid = grid

        self.update_search_params(
            grid=grid,
            exclude_pt_itself=exclude_pt_itself,
            radius=radius,
            include_boundary=include_boundary,
        )
        #
    #

    @time_func_perf
    def update_search_params(
        self,
        grid,
        exclude_pt_itself:bool=None,
        radius:float=0.0075,
        include_boundary:bool=False,
        relation_tgt_to_src:str=None,
    ):
        if exclude_pt_itself is not None:
            self.exclude_pt_itself = exclude_pt_itself
        if relation_tgt_to_src is not None:
            self.relation_tgt_to_src = relation_tgt_to_src
        
        # early return if radius and include_boundary have not changed
        if hasattr(self, 'radius') and hasattr(self, 'include_boundary'):
            if self.radius == radius and self.include_boundary == include_boundary:
                return
        
        # store params
        self.radius = radius
        self.include_boundary = include_boundary
        self.overlap_checks = []
        self.contain_checks = []
        
        # get relative position of cells that are always included within radius for current gridsize    
        (
        self.cells_contained_in_all_disks, 
        self.cells_contained_in_all_trgl_disks, 
        self.cells_maybe_overlapping_a_disk, 
        self.cells_maybe_overlapping_a_trgl_disk
        ) = get_cells_relevant_for_disk_by_type(
                grid_spacing=grid.spacing,
                radius=radius,
                include_boundary=include_boundary,
        )
        # hierarchically order all cells with respect to any point in triangle. 
        # Some cells are at least as far away as others 
        # e.g. (2,2) is weakly closer than (-2,-2) as any pt in triangle is P(x>=0,y>=0)
        # e.g. (2,3) is weakly closer than (?,?) as any pt in triangle is P(x,y<=0.5*grid.spacing)
        triangle_1_vertices = _np_array([[0,0],[0.5,0],[0.5,0.5]])
        vertices_is_inside_triangle_1 = _np_array([True,True,False],dtype=bool)
        # TODO radius,grid_spacing, include_boundary could be removed from weak_order_tree generation
        self.weak_order_tree = gen_weak_order_rel_to_convex_set(
                cells=self.cells_maybe_overlapping_a_trgl_disk,
                convex_set_vertices = triangle_1_vertices,
                vertex_is_inside_convex_set = vertices_is_inside_triangle_1,
                radius=radius,
                grid_spacing=grid.spacing,
                include_boundary=include_boundary,
        )
        self.cells_contained_in_all_disks = arr_to_tpls(self.cells_contained_in_all_disks,int)
    #
    
    # @time_func_perf
    def check_if_tgt_df_contains_src_df(
        self,
    )->bool:
        if not hasattr(self, 'target'): return False
        if not hasattr(self, 'source'): return False
        if not hasattr(self.target, 'pts_df'): return False
        if not hasattr(self.source, 'pts_df'): return False
        return DataFrameRelation.check_if_df_is_contained(self.source.pts_df, self.target.pts_df,)
    
    # @time_func_perf
    def check_if_search_obj_already_exist(
        self,
        pts_df:_pd_DataFrame,
        obj:str=['source','target'],
        **kwargs
    ):
        """
        check if search sortarget already created
        kwarg at pos 0 is pandas.DataFrame and will thus be checked for
        equlity by .equals insted of ==  
        Args:
        self : DiskSearch
          Checks on disk searchs object
        Returns:
        
        """
        
        # check if this attribute is already set as source
        alr_added_pts_to_grid = (
            hasattr(self, 'target') and 
            all([hasattr(self.target, k) and 
                v == getattr(self.target, k)
                for k,v in kwargs.items()]) and
                hasattr(self.target, 'pts_df') and 
                DataFrameRelation.check_if_df_is_contained(pts_df, self.target.pts_df)
        )
        alr_assg_to_cell_regions = obj=='source' and (
            hasattr(self, 'source') and all([
                hasattr(self.source, k) and 
                v == getattr(self.source, k)
                for k,v in kwargs.items()]) and
                hasattr(self.source, 'pts_df') and 
                DataFrameRelation.check_if_df_is_contained(pts_df, self.source.pts_df)
        )
        alr_assg_to_cells = (alr_added_pts_to_grid or alr_assg_to_cell_regions)
        
        return (alr_assg_to_cells, alr_assg_to_cell_regions, alr_added_pts_to_grid)
    #

    @time_func_perf
    def set_source(
        self,

        pts_df:_pd_DataFrame,
        y_coord_name:str='lat',
        x_coord_name:str='lon',
        row_name:str='id_y',
        col_name:str='id_x',
        cell_region_name:str='cell_region',

        sum_suffix:str='_750m',

        plot_cell_reg_assign:dict=None,
        plot_offset_checks:dict=None,
        plot_offset_regions:dict=None,
        plot_offset_raster:dict=None,
        silent:bool=False,
    ):
        """
        TODO also make shortcut if  grid.search.tgt_df_contains_src_df
        """
        (alr_assg_to_cells,
        alr_assg_to_cell_regions,
        alr_added_pts_to_grid) = self.check_if_search_obj_already_exist(
            **dict(list(locals().items())[1:8])
        )

        self.source = DiskSearchSource(
            grid=self.grid,
            pts_df=pts_df,
            y_coord_name=y_coord_name,
            x_coord_name=x_coord_name,
            row_name=row_name,
            col_name=col_name,
            cell_region_name=cell_region_name,
            sum_suffix=sum_suffix,
        )
        
        self.tgt_df_contains_src_df = self.check_if_tgt_df_contains_src_df()

        if not alr_assg_to_cells:
            self.source.assign_pts_to_cells(silent=silent,)
            self.source.pts_df.sort_values(
                [self.source.row_name, self.source.col_name],
                inplace=True
            )
        #
        if not alr_assg_to_cell_regions:
            self.source.assign_pts_to_cell_regions(
                plot_cell_reg_assign=plot_cell_reg_assign,
                plot_offset_checks=plot_offset_checks,
                plot_offset_regions=plot_offset_regions,
                plot_offset_raster=plot_offset_raster,
                silent=silent,
            )
            self.source.pts_df.sort_values(
                [self.source.row_name, self.source.col_name, self.source.cell_region_name],
                inplace=True
            )
        #
    #

    @time_func_perf
    def set_target(
        self,

        pts_df:_pd_DataFrame,
        sum_names:list=['employment'],
        y_coord_name:str='lat',
        x_coord_name:str='lon',
        row_name:str='id_y',
        col_name:str='id_x',

        silent:bool=False,
    ):
        
        (alr_assg_to_cells,
        alr_assg_to_cell_regions,
        alr_added_pts_to_grid) = self.check_if_search_obj_already_exist(
            **dict(list(locals().items())[1:8])
        )

        self.target = DiskSearchTarget(
            grid=self.grid,
            pts_df=pts_df,
            sum_names=sum_names,
            y_coord_name=y_coord_name,
            x_coord_name=x_coord_name,
            row_name=row_name,
            col_name=col_name,
        )

        self.tgt_df_contains_src_df = self.check_if_tgt_df_contains_src_df()

        if not alr_assg_to_cells:
            self.target.assign_pts_to_cells(silent=silent,)

            # also sort according to grid cell region!
            self.target.pts_df.sort_values(
                [self.target.row_name, self.target.col_name],
                inplace=True
            )
        #
        if not alr_added_pts_to_grid:
            self.target.aggregate_pt_data_to_cells(silent=silent,)
        #
    #

    def perform_search(
            self,
            plot_radius_sums:dict=None,
            plot_pt_disk:dict=None,
            silent:bool=False,
    ):
        
        return aggreagate_point_data_to_disks_vectorized(
            grid=self.grid,
            pts_df_search_source=self.source.pts_df,
            pts_df_search_target=self.target.pts_df,
            radius=self.radius,
            sum_names=self.target.sum_names,
            y_coord_name=self.source.y_coord_name,
            x_coord_name=self.source.x_coord_name,
            row_name=self.source.row_name,
            col_name=self.source.col_name,
            cell_region_name=self.source.cell_region_name,
            sum_suffix=self.source.sum_suffix,
            exclude_pt_itself=self.exclude_pt_itself,
            plot_radius_sums=plot_radius_sums,
            plot_pt_disk=plot_pt_disk,
            silent=silent,
        )
    #
#
    
    

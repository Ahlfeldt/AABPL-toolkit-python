from numpy import array as _np_array
from pandas import (DataFrame as _pd_DataFrame, cut as _pd_cut, concat as _pd_concat) 
from aabpl.utils.general import ( DataFrameRelation, arr_to_tpls)
from aabpl.utils.distances_to_cell import get_cells_relevant_for_disk_by_type
from .two_dimensional_weak_ordering_class import gen_weak_order_rel_to_convex_set
from .pts_to_cells import assign_points_to_cells, aggregate_point_data_to_cells
from .pts_to_offset_regions import assign_points_to_mirco_regions
from .pts_radius_search import aggregate_point_data_to_disks_vectorized
from aabpl.testing.test_performance import time_func_perf


################ DiskSearchSource ######################################################################################
class DiskSearchObject(object):

    def assign_pts_to_cells(
        self,
        silent:bool = False
    ):
        return assign_points_to_cells(
            grid=self.grid,
            pts=self.pts,
            y=self.y,
            x=self.x,
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
        pts:_pd_DataFrame,
        c:list=[],
        y:str='lat',
        x:str='lon',
        row_name:str='id_y',
        col_name:str='id_x',
        sum_suffix:str='_750m',
    ):
        self.grid = grid 
        self.pts = pts
        self.y = y
        self.x = x
        self.row_name = row_name
        self.col_name = col_name
        self.cell_region_name = next(('cell_reg'+str(i) for i in (['']+list(range(len(pts.columns)))) if not 'cell_reg'+str(i) in pts.columns)) 
        self.sum_suffix = sum_suffix
        self.aggregate_columns = [str(column)+sum_suffix for column in c]

    #next(('helper_col'+i for i in (['']+list(range(len(pts_target.columns))))))

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
            pts=self.pts,
            r=self.grid.search.r,
            include_boundary=self.grid.search.include_boundary,
            y=self.y,
            x=self.x,
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
        pts:_pd_DataFrame,
        c:list=[],
        y:str='lat',
        x:str='lon',
        row_name:str='id_y',
        col_name:str='id_x',
    ):
        self.grid = grid 
        self.pts = pts
        self.c = c
        self.x = x
        self.y = y
        self.row_name = row_name
        self.col_name = col_name
        
        # prepare dicts for fast lookup of values for point ids
        self.pt_id_to_xy_coords = {
            pt_id:xy for (pt_id,xy) in zip(pts.index, pts[[x,y]].values)
        }
        self.pt_id_to_vals = {
            pt_id:pt_vals for (pt_id,pt_vals) in zip(pts.index, pts[c].values)
        }
    #
    
    def aggregate_pt_data_to_cells(
            self,
            silent
    ):
        return aggregate_point_data_to_cells(
            grid=self.grid,
            pts=self.pts,
            c=self.c,
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
        r:float=0.0075,
        exclude_pt_itself:bool=True,
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
            r=r,
            include_boundary=include_boundary,
        )
        #
    #

    @time_func_perf
    def update_search_params(
        self,
        grid,
        exclude_pt_itself:bool=None,
        r:float=0.0075,
        include_boundary:bool=False,
        relation_tgt_to_src:str=None,
    ):
        if exclude_pt_itself is not None:
            self.exclude_pt_itself = exclude_pt_itself
        if relation_tgt_to_src is not None:
            self.relation_tgt_to_src = relation_tgt_to_src
        
        # early return if r and include_boundary have not changed
        if hasattr(self, 'r') and hasattr(self, 'include_boundary'):
            if self.r == r and self.include_boundary == include_boundary:
                return
        
        # store params
        self.r = r
        self.include_boundary = include_boundary
        self.overlap_checks = []
        self.contain_checks = []
        
        # get relative position of cells that are always included within r for current gridsize    
        (
        self.cells_contained_in_all_disks, 
        self.cells_contained_in_all_trgl_disks, 
        self.cells_maybe_overlapping_a_disk, 
        self.cells_maybe_overlapping_a_trgl_disk
        ) = get_cells_relevant_for_disk_by_type(
                grid_spacing=grid.spacing,
                r=r,
                include_boundary=include_boundary,
        )
        # hierarchically order all cells with respect to any point in triangle. 
        # Some cells are at least as far away as others 
        # e.g. (2,2) is weakly closer than (-2,-2) as any pt in triangle is P(x>=0,y>=0)
        # e.g. (2,3) is weakly closer than (?,?) as any pt in triangle is P(x,y<=0.5*grid.spacing)
        triangle_1_vertices = _np_array([[0,0],[0.5,0],[0.5,0.5]])
        vertices_is_inside_triangle_1 = _np_array([True,True,False],dtype=bool)
        # TODO r,grid_spacing, include_boundary could be removed from weak_order_tree generation
        self.weak_order_tree = gen_weak_order_rel_to_convex_set(
                cells=self.cells_maybe_overlapping_a_trgl_disk,
                convex_set_vertices = triangle_1_vertices,
                vertex_is_inside_convex_set = vertices_is_inside_triangle_1,
                r=r,
                grid_spacing=grid.spacing,
                include_boundary=include_boundary,
        )
        self.cells_contained_in_all_disks = arr_to_tpls(self.cells_contained_in_all_disks,int)
    #
    
    # @time_func_perf
    def check_if_tgt_df_contains_src_df(
        self,
        silent:bool=False,
    )->bool:
        if not hasattr(self, 'target'): return False
        if not hasattr(self, 'source'): return False
        if not hasattr(self.target, 'pts'): return False
        if not hasattr(self.source, 'pts'): return False
        return DataFrameRelation.check_if_df_is_contained(self.source.pts, self.target.pts,silent=silent)
    
    # @time_func_perf
    def check_if_search_obj_already_exist(
        self,
        pts:_pd_DataFrame,
        obj:str=['source','target'],
        silent:bool=False,
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
                hasattr(self.target, 'pts') and 
                DataFrameRelation.check_if_df_is_contained(pts, self.target.pts,silent=silent)
        )
        alr_assg_to_cell_regions = obj=='source' and (
            hasattr(self, 'source') and all([
                hasattr(self.source, k) and 
                v == getattr(self.source, k)
                for k,v in kwargs.items()]) and
                hasattr(self.source, 'pts') and 
                DataFrameRelation.check_if_df_is_contained(pts, self.source.pts,silent=silent)
        )
        alr_assg_to_cells = (alr_added_pts_to_grid or alr_assg_to_cell_regions)
        
        return (alr_assg_to_cells, alr_assg_to_cell_regions, alr_added_pts_to_grid)
    #

    @time_func_perf
    def set_source(
        self,
        pts:_pd_DataFrame,
        c:list=[],
        y:str='lat',
        x:str='lon',
        row_name:str='id_y',
        col_name:str='id_x',
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
            pts=pts,
            c=c,
            y=y,
            x=x,
            row_name=row_name,
            col_name=col_name,
            sum_suffix=sum_suffix,
        )
        
        self.tgt_df_contains_src_df = self.check_if_tgt_df_contains_src_df(silent=silent)

        if not alr_assg_to_cells:
            self.source.assign_pts_to_cells(silent=silent,)
            # self.source.pts.sort_values(
            #     [self.source.row_name, self.source.col_name],
            #     inplace=True
            # )
        #
        if not alr_assg_to_cell_regions:
            self.source.assign_pts_to_cell_regions(
                plot_cell_reg_assign=plot_cell_reg_assign,
                plot_offset_checks=plot_offset_checks,
                plot_offset_regions=plot_offset_regions,
                plot_offset_raster=plot_offset_raster,
                silent=silent,
            )
            # self.source.pts.sort_values(
            #     [self.source.row_name, self.source.col_name, self.source.cell_region_name],
            #     inplace=True
            # )
        #
    #

    @time_func_perf
    def set_target(
        self,

        pts:_pd_DataFrame,
        c:list=['employment'],
        y:str='lat',
        x:str='lon',
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
            pts=pts,
            c=c,
            y=y,
            x=x,
            row_name=row_name,
            col_name=col_name,
        )

        self.tgt_df_contains_src_df = self.check_if_tgt_df_contains_src_df(silent=silent)

        if not alr_assg_to_cells:
            self.target.assign_pts_to_cells(silent=silent,)

            # also sort according to grid cell region!
            # self.target.pts.sort_values(
            #     [self.target.row_name, self.target.col_name],
            #     inplace=True
            # )
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
        
        return aggregate_point_data_to_disks_vectorized(
            grid=self.grid,
            pts_source=self.source.pts,
            pts_target=self.target.pts,
            r=self.r,
            c=self.target.c,
            y=self.source.y,
            x=self.source.x,
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
    
    

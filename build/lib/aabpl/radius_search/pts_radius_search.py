from numpy import array as _np_array, zeros as _np_zeros
from numpy.linalg import norm as _np_linalg_norm
from pandas import (DataFrame as _pd_DataFrame, cut as _pd_cut, concat as _pd_concat) 
from aabpl.utils.general import flatten_list
from aabpl.illustrations.illustrate_point_to_disk import illustrate_point_disk
from aabpl.illustrations.plot_pt_vars import create_plots_for_vars
from aabpl.testing.test_performance import time_func_perf


################ aggreagate_point_data_to_disks_vectorized ######################################################################################
@time_func_perf
def aggreagate_point_data_to_disks_vectorized(
    grid:dict,
    pts_source:_pd_DataFrame,
    pts_target:_pd_DataFrame=None,
    radius:float=0.0075,
    columns:list=['employment'],
    y:str='lat',
    x:str='lon',
    row_name:str='id_y',
    col_name:str='id_x',
    cell_region_name:str='cell_region',
    sum_suffix:str='_750m',
    exclude_pt_itself:bool=True,
    plot_radius_sums:dict=None,
    plot_pt_disk:dict=None,
    silent = False,
):
    """
    
    """
    if pts_target is None:
        pts_target = pts_source 
    
    # unpack grid_data 
    grid_id_to_pt_ids = grid.id_to_pt_ids
    grid_id_to_sums = grid.id_to_sums
    pt_id_to_row_col = grid.pt_id_to_row_col
    sparse_grid_ids = set(grid_id_to_pt_ids)
    
    region_id_to_contained_cells = grid.search.region_id_to_contained_cells
    region_id_to_overlapped_cells = grid.search.region_id_to_overlapped_cells
    cells_contained_in_all_disks = grid.search.cells_contained_in_all_disks

    id_y_mult=grid.id_y_mult
    pt_id_to_xy_coords = grid.search.target.pt_id_to_xy_coords
    pt_id_to_vals = grid.search.target.pt_id_to_vals
    n_pts = len(pts_source)

    # initialize columns and/or reset to zero 
    sum_radius_names = [(cname+sum_suffix) for cname in columns]
    pts_source[sum_radius_names] = 0
    
  
    sums_within_disks = _np_zeros((n_pts, len(columns)))
    
    if plot_pt_disk is not None:
        if not 'pt_id' in plot_pt_disk:
            plot_pt_disk['pt_id'] = pts_source.index[int(n_pts//2)]
        
    zero_sums = _np_zeros(len(columns),dtype=int) if len(columns) > 1 else 0
    pts_source['initial_sort'] = range(len(pts_source))
    pts_source.sort_values([row_name, col_name, cell_region_name], inplace=True)
    last_pt_row_col = (-1, -1)
    last_cell_region_id = -1
    counter_new_cell = 0
    counter_new_contain_region = 0
    counter_new_overlap_region = 0

    if len(columns) > 1:
        @time_func_perf
        def sum_contained_all_offset_regions(
                pt_row,
                pt_col,
        ):
            cells_cntd_by_pt_cell = sparse_grid_ids.intersection([(row+pt_row,col+pt_col) for row,col in cells_contained_in_all_disks])
            # cells_cntd_by_pt_cell = [cell_id for cell_id in (id_y_mult*(cells_contained_in_all_disks[:,0]+(pt_row))+(
            #     cells_contained_in_all_disks[:,1]+pt_col)) if cell_id in sparse_grid_ids] 
            if len(cells_cntd_by_pt_cell)>0:
                return _np_array([grid_id_to_sums[g_id] for g_id in cells_cntd_by_pt_cell]).sum(axis=0) 
            return zero_sums
        #
    else:
        @time_func_perf
        def sum_contained_all_offset_regions(
                pt_row,
                pt_col,
        ):
            cells_cntd_by_pt_cell = sparse_grid_ids.intersection([(row+pt_row,col+pt_col) for row,col in cells_contained_in_all_disks])
            return sum([grid_id_to_sums[g_id] for g_id in cells_cntd_by_pt_cell]) 
        #
    
    if len(columns) > 1:
        @time_func_perf
        def sum_contained_by_offset_region(
                pt_row,
                pt_col,
                cell_region_id,
        ):
            cells_contained_by_pt_region = sparse_grid_ids.intersection([(row+pt_row,col+pt_col) for row,col in region_id_to_contained_cells[cell_region_id]])
            if len(cells_contained_by_pt_region)>0:
                return _np_array([grid_id_to_sums[g_id] for g_id in cells_contained_by_pt_region]).sum(axis=0) 
            return zero_sums
        #
    else:
        @time_func_perf
        def sum_contained_by_offset_region(
                pt_row,
                pt_col,
                cell_region_id,
        ):
            cells_contained_by_pt_region = sparse_grid_ids.intersection([(row+pt_row,col+pt_col) for row,col in region_id_to_contained_cells[cell_region_id]])
            return sum([grid_id_to_sums[g_id] for g_id in cells_contained_by_pt_region]) 
        #
    
    
    @time_func_perf
    def get_pts_overlapped_by_region(
            pt_row,
            pt_col,
            cell_region_id,
    ):  
        
        cells_overlapped_by_pt_region = sparse_grid_ids.intersection([(row+pt_row,col+pt_col) for row,col in region_id_to_overlapped_cells[cell_region_id]])
        return _np_array(flatten_list([
            grid_id_to_pt_ids[cell_id] for cell_id in cells_overlapped_by_pt_region
        ]))
    #
    
    

    if len(columns) > 1:
        @time_func_perf
        def sum_overlapped_pts_in_radius(
            pts_in_cells_overlapped_by_pt_region,
            a_pt_xycoord
        ):
            if len(pts_in_cells_overlapped_by_pt_region) > 0:
                pts_in_radius = pts_in_cells_overlapped_by_pt_region[(_np_linalg_norm( # create a mask of boolean values indicating if point are within radius 
                    _np_array([pt_id_to_xy_coords[pt_id] for pt_id in pts_in_cells_overlapped_by_pt_region]) -
                    a_pt_xycoord, 
                axis=1) <= radius)]
                
                return _np_array([pt_id_to_vals[pt_id] for pt_id in pts_in_radius]).sum(axis=0) if len(pts_in_radius) > 0 else zero_sums
                # else no points in radius thus return vector of _np_zeros
            return zero_sums
    else:
        @time_func_perf
        def sum_overlapped_pts_in_radius(
            pts_in_cells_overlapped_by_pt_region,
            a_pt_xycoord
        ):
            if len(pts_in_cells_overlapped_by_pt_region) > 0:
                pts_in_radius = pts_in_cells_overlapped_by_pt_region[(_np_linalg_norm( # create a mask of boolean values indicating if point are within radius 
                    _np_array([pt_id_to_xy_coords[pt_id] for pt_id in pts_in_cells_overlapped_by_pt_region]) -
                    a_pt_xycoord, 
                axis=1) <= radius)]
                
                return sum([pt_id_to_vals[pt_id] for pt_id in pts_in_radius]) if len(pts_in_radius) > 0 else 0
                # else no points in radius thus return vector of _np_zeros
            return 0
    

    @time_func_perf
    def do_nothing():
        pass
    
    for (i, pt_id, a_pt_xycoord, (pt_row,pt_col), contain_region_id, overlap_region_id, cell_region_id) in zip(
        range(n_pts),
        pts_source.index,
        pts_source[[x, y,]].values, 
        pts_source[[row_name, col_name]].values,
        pts_source[cell_region_name].values // grid.search.contain_region_mult,
        pts_source[cell_region_name].values % grid.search.contain_region_mult,
        pts_source[cell_region_name].values,
        
        
        ):
        # (pt_row, pt_col) = pt_id_to_row_col[pt_id]
        
        # as pts are sorted by grid cell update only if grid cell changed
        if not (pt_row, pt_col) == last_pt_row_col:
            counter_new_cell += 1
            # cells_cntd_by_pt_cell = sparse_grid_ids.intersection([(row+pt_row,col+pt_col) for row,col in cells_contained_in_all_disks])
            # # cells_cntd_by_pt_cell = [cell_id for cell_id in (id_y_mult*(cells_contained_in_all_disks[:,0]+(pt_row))+(
            # #     cells_contained_in_all_disks[:,1]+pt_col)) if cell_id in sparse_grid_ids] 
            # sums_cells_cntd_by_pt_cell = (_np_array([grid_id_to_sums[g_id] for g_id in cells_cntd_by_pt_cell]).sum(axis=0) 
            #                           if len(cells_cntd_by_pt_cell)>0 else zero_sums)
            sums_cells_cntd_by_pt_cell = sum_contained_all_offset_regions(pt_row, pt_col)
        #
            
        if not (pt_row, pt_col) == last_pt_row_col or last_contain_region_id != contain_region_id:
            counter_new_contain_region += 1
            # if cell changed or cell region changed
            # this can be improve to capture only changes of cell region conatain ids 
            # cells_contained_by_pt_region = sparse_grid_ids.intersection([(row+pt_row,col+pt_col) for row,col in region_id_to_contained_cells[cell_region_id]])
            # # cells_contained_by_pt_region = [cell_id for cell_id in (id_y_mult*(region_id_to_contained_cells[cell_region_id][:,0]+(pt_row))+(
            # #     region_id_to_contained_cells[cell_region_id][:,1]+pt_col)) if cell_id in sparse_grid_ids]
                           
            # sums_contained_by_pt_region = (_np_array([grid_id_to_sums[g_id] for g_id in cells_contained_by_pt_region]).sum(axis=0) 
            #                                  if len(cells_contained_by_pt_region)>0 else zero_sums)
            sums_contained_by_pt_region = sum_contained_by_offset_region(pt_row, pt_col, cell_region_id)
        do_nothing()

        if not (pt_row, pt_col) == last_pt_row_col or last_overlap_region_id != overlap_region_id:
            counter_new_overlap_region += 1
            # cells_overlapped_by_pt_region = sparse_grid_ids.intersection([(row+pt_row,col+pt_col) for row,col in region_id_to_overlapped_cells[cell_region_id]])
            # # cells_overlapped_by_pt_region = [cell_id for cell_id in (id_y_mult*(region_id_to_overlapped_cells[cell_region_id][:,0]+(pt_row))+(
            # #     region_id_to_overlapped_cells[cell_region_id][:,1]+pt_col)) if cell_id in sparse_grid_ids]
            # pts_in_cells_overlapped_by_pt_region = _np_array(flatten_list([
            #     grid_id_to_pt_ids[cell_id] for cell_id in cells_overlapped_by_pt_region
            # ]))
            pts_in_cells_overlapped_by_pt_region = get_pts_overlapped_by_region(pt_row, pt_col, cell_region_id)
        #
        
        # if len(pts_in_cells_overlapped_by_pt_region) > 0:
        #     pts_in_radius = pts_in_cells_overlapped_by_pt_region[(_np_linalg_norm( # create a mask of boolean values indicating if point are within radius 
        #         _np_array([pt_id_to_xy_coords[pt_id] for pt_id in pts_in_cells_overlapped_by_pt_region]) -
        #         a_pt_xycoord, 
        #     axis=1) <= radius)]
            
        #     overlapping_cells_sums = _np_array([pt_id_to_vals[pt_id] for pt_id in pts_in_radius]).sum(axis=0) if len(pts_in_radius) > 0 else zero_sums
        # else:
        #     # else no points in radius thus return vector of _np_zeros
        #     overlapping_cells_sums = zero_sums
        do_nothing()
        
        overlapping_cells_sums = sum_overlapped_pts_in_radius(pts_in_cells_overlapped_by_pt_region, a_pt_xycoord)

        do_nothing()

        # combine sums from three steps.
        # append result 
        sums_within_disks[i,:] = sums_cells_cntd_by_pt_cell + sums_contained_by_pt_region + overlapping_cells_sums
        
        if plot_pt_disk is not None and pt_id == plot_pt_disk['pt_id']:
            cells_cntd_by_pt_cell = sparse_grid_ids.intersection([(row+pt_row,col+pt_col) for row,col in cells_contained_in_all_disks])
            cells_contained_by_pt_region = sparse_grid_ids.intersection([(row+pt_row,col+pt_col) for row,col in region_id_to_contained_cells[cell_region_id]])
            pts_in_radius = pts_in_cells_overlapped_by_pt_region[(_np_linalg_norm( # create a mask of boolean values indicating if point are within radius 
                _np_array([pt_id_to_xy_coords[pt_id] for pt_id in pts_in_cells_overlapped_by_pt_region]) -
                a_pt_xycoord, 
            axis=1) <= radius)]
            illustrate_point_disk(
                grid=grid,
                pts_source=pts_source,
                pts_target=pts_target,
                radius=radius,
                columns=columns,
                x=x,
                y=y,
                cells_cntd_by_pt_cell=[(row+pt_row,col+pt_col) for row,col in cells_contained_in_all_disks],
                cells_contained_by_pt_region=[(row+pt_row,col+pt_col) for row,col in region_id_to_contained_cells[cell_region_id]],
                cells_overlapped_by_pt_region=[(row+pt_row,col+pt_col) for row,col in region_id_to_overlapped_cells[cell_region_id]],
                pts_in_cell_contained_by_pt_region=_np_array(flatten_list([
                        grid_id_to_pt_ids[cell_id] for cell_id in cells_cntd_by_pt_cell
                    ]+[
                        grid_id_to_pt_ids[cell_id] for cell_id in cells_contained_by_pt_region
                    ])),
                pts_in_cells_overlapped_by_pt_region=pts_in_cells_overlapped_by_pt_region,
                pts_in_radius=pts_in_radius,
                **plot_pt_disk,
            )
        # #

        # set id as last id for next iteration
        last_pt_row_col = (pt_row, pt_col)
        last_contain_region_id = contain_region_id
        last_overlap_region_id = overlap_region_id
    #
    pts_source[sum_radius_names] = pts_source[sum_radius_names].values + sums_within_disks
            
    if exclude_pt_itself and grid.search.tgt_df_contains_src_df:
        # substract data from point itself unless specified otherwise
        pts_source[sum_radius_names] = pts_source[sum_radius_names].values - pts_source[columns]
    
    # print(
    #     "Share of pts in",
    #     "\n- same cell as previous:", 100-int(counter_new_cell/len(pts_source)*100),"%",
    #     "\n- same cell and containing same surrounding cells:",100 - int(counter_new_contain_region/len(pts_source)*100),"%",
    #     "\n- same cell and overlapping same surrounding cells",100 - int(counter_new_overlap_region/len(pts_source)*100),"%")
    def plot_vars(
        self = grid,
        colnames = _np_array([columns, sum_radius_names]), 
        filename:str='',
        **plot_kwargs:dict,
    ):
        return create_plots_for_vars(
            grid=self,
            colnames=colnames,
            filename=filename,
            plot_kwargs=plot_kwargs,
        )

    grid.plot_vars = plot_vars
    
    if plot_radius_sums is not None:
        print('create plot for radius sums')
        create_plots_for_vars(
            grid=grid,
            colnames=_np_array([columns, sum_radius_names]),
            plot_kwargs=plot_radius_sums,
        )
    #
    pts_source.sort_values(['initial_sort'], inplace=True)
    pts_source.drop('initial_sort', axis=1, inplace=True)

    return pts_source[sum_radius_names]
#



from numpy import array as _np_array, zeros as _np_zeros,exp as _np_exp 
from numpy.linalg import norm as _np_linalg_norm
from pandas import (DataFrame as _pd_DataFrame, cut as _pd_cut, concat as _pd_concat) 
from aabpl.utils.general import flatten_list
from aabpl.utils.distances_to_cell import get_cells_by_lvl_relevant_for_disk_by_type
from aabpl.illustrations.illustrate_point_to_disk import illustrate_point_disk
from aabpl.illustrations.plot_pt_vars import create_plots_for_vars
from aabpl.testing.test_performance import time_func_perf
from math import pi as _math_pi
from aabpl.valid_area import disk_cell_intersection_area, disk_cell_intersection_estimate

################ aggregate_point_data_to_disks_vectorized ######################################################################################
@time_func_perf
def aggregate_point_data_to_disks_vectorized(
    grid:dict,
    pts_source:_pd_DataFrame,
    r:float,
    c:list=[],
    y:str='proj_lat',
    x:str='proj_lon',
    off_x='offset_x',
    off_y='offset_y',
    pts_target:_pd_DataFrame=None,
    row_name:str='id_y',
    col_name:str='id_x',
    cell_region_name:str='cell_region',
    sum_suffix:str=None,
    exclude_pt_itself:bool=True,
    weight_valid_area:str=None,
    plot_pt_disk:dict=None,
    silent:bool=False,
):
    """
    Aggregates Data around each point
    """
    if pts_target is None:
        pts_target = pts_source 
    # unpack grid_data 
    grid_id_to_pt_ids = grid.id_to_pt_ids
    grid_id_to_vals_xy = grid.id_to_vals_xy
    grid_id_to_sums = grid.id_to_sums
    
    grid_id_to_pt_ids_by_lvl = grid.id_to_pt_ids_by_lvl
    grid_id_to_sums_by_lvl = grid.id_to_sums_by_lvl
    grid_id_to_vals_xy_by_lvl = grid.id_to_vals_xy_by_lvl
    trynew = grid.trynew if hasattr(grid, 'trynew') else 1 
    
    sparse_grid_ids = set(grid_id_to_sums_by_lvl)
    # print("sparse_grid_ids",sparse_grid_ids)
    cells_rndm_sample = grid.cells_rndm_sample
    grid_spacing = grid.spacing
    if type(cells_rndm_sample)==bool and cells_rndm_sample:
        weight_valid_area=False # as for each point 100% of area would be valid
    else:
        grid_padding = -int(-grid_spacing//r)
        # take all cells that are part of the sampling grid
        invalid_cells = set([id for id in 
                             tuple(flatten_list([
            [(0, (int(row_id), int(col_id))) for col_id in range(min(grid.col_ids)-grid_padding, max(grid.col_ids)+grid_padding)] 
            for row_id in range(min(grid.row_ids)-grid_padding, max(grid.row_ids)+grid_padding)]))
             if not id in cells_rndm_sample])
    
    region_id_to_contained_cells = grid.search.region_id_to_contained_cells
    region_id_to_overlapped_cells = grid.search.region_id_to_overlapped_cells
    region_id_to_nested_contained_cells = grid.search.region_id_to_nested_contained_cells
    region_id_to_nested_overlapped_cells = grid.search.region_id_to_nested_overlapped_cells
    # print("region_id_to_nested_overlapped_cells[0]",region_id_to_nested_overlapped_cells[0])
    region_id_to_distinct_contained_cells = grid.search.region_id_to_distinct_contained_cells
    region_id_to_distinct_overlapped_cells = grid.search.region_id_to_distinct_overlapped_cells
    # print("region_id_to_distinct_overlapped_cells[0]",region_id_to_distinct_overlapped_cells[0])
    # cells_contained_in_all_disks = grid.search.cells_contained_in_all_disks
    # print("cells_contained_in_all_disks",cells_contained_in_all_disks)
    shared_contained_cells  = grid.search.shared_contained_cells
    shared_overlapped_cells = grid.search.shared_overlapped_cells
    
    
    # print("region_id_to_nested_overlapped_cells",region_id_to_nested_overlapped_cells)
    row_col_to_centroid = grid.row_col_to_centroid
    get_cell_centroid = grid.get_cell_centroid
    pt_id_to_xy_coords = grid.search.target.pt_id_to_xy_coords
    n_pts = len(pts_source)
    if trynew in [1]:
        shared_contained_cells_lookup = shared_contained_cells
        region_id_to_contained_cells_lookup = region_id_to_distinct_contained_cells
        shared_overlapped_cells_lookup = shared_overlapped_cells
        region_id_to_overlapped_cells_lookup = region_id_to_distinct_overlapped_cells
    elif trynew in [2]:
        shared_contained_cells_lookup = shared_contained_cells
        region_id_to_contained_cells_lookup = region_id_to_distinct_contained_cells
        shared_overlapped_cells_lookup = []
        region_id_to_overlapped_cells_lookup = region_id_to_nested_overlapped_cells
    elif trynew in [3]:
        shared_contained_cells_lookup = []
        region_id_to_contained_cells_lookup = region_id_to_nested_contained_cells
        shared_overlapped_cells_lookup = shared_overlapped_cells
        region_id_to_overlapped_cells_lookup = region_id_to_distinct_overlapped_cells
    else: 
        shared_contained_cells_lookup = []
        region_id_to_contained_cells_lookup = region_id_to_nested_contained_cells
        shared_overlapped_cells_lookup = []
        region_id_to_overlapped_cells_lookup = region_id_to_nested_overlapped_cells
    # print("///shared_contained_cells_lookup",len(shared_contained_cells_lookup), 
    #       "///shared_overlapped_cells_lookup",len(shared_overlapped_cells_lookup))
    # print("region_id_to_overlapped_cells_lookup[0]",region_id_to_overlapped_cells_lookup[0])
    # print("cells_contained_in_all_disks_incl_nested",cells_contained_in_all_disks_incl_nested)
    # initialize columns and/or reset to zero 
    
    
  
    
    # prepare plot #
    if plot_pt_disk is not None:
        if not 'pt_id' in plot_pt_disk:
            plot_pt_disk['pt_id'] = pts_source.index[int(n_pts//2)]
            plot_pt_disk['pt_id'] = sorted([(len(pt_ids), pt_ids[0] if len(pt_ids)>0 else []) for pt_ids in grid_id_to_pt_ids_by_lvl.values()])[-1][1]

    ##################### set up loop ############################
    if sum_suffix is None:
        sum_suffix = '_'+str(r)
    sum_radius_names = [(cname+sum_suffix) for cname in c]
    pts_source[sum_radius_names] = 0
     
    all_sums_cells_cntd_by_pt_cell = _np_zeros((n_pts, len(c)))
    all_sums_contained_by_pt_region = _np_zeros((n_pts, len(c)))
    all_distinct_overlap_sums = _np_zeros((n_pts, len(c)))
    all_shared_overlap_sums = _np_zeros((n_pts, len(c)))
    sums_within_disks = _np_zeros((n_pts, len(c)))
    valid_area_shares = _np_zeros(n_pts)
    valid_search_area = 2 * _math_pi * r**2
    column_dtypes = pts_target[c].dtypes
    zero_sums = _np_zeros(len(c),dtype=int) if len(c) > 1 else 0

    # create empty arrays for storing intermediate results to avoid reallocation in loop. choose size based on max possible number of cells relevant for disk search
    # TODO the nest depth can be chosen according to density of the region (super cell count)
    # TODO ensure this works as intended if multiple lvls area used
    n_cells_pot_relevant = max([len(cells) for cells in region_id_to_overlapped_cells.values()])
    max_len_empty = sum(sorted([len(v) for v in grid_id_to_vals_xy.values()])[-n_cells_pot_relevant:])
    empty_sums = _np_zeros((max_len_empty,len(c)),dtype=int)
    empty_xy_vals = _np_zeros((max_len_empty,len(c)+2),dtype=int)
    
    shared_overlap_sums = zero_sums
    pts_source.sort_values([row_name, col_name, cell_region_name], inplace=True)
    last_pt_row_col = (None,None)
    # last_cell_region_id = -1
    counter_new_cell = 0
    counter_new_contain_region = 0
    counter_new_overlap_region = 0
    
    # print("weight_valid_area",weight_valid_area, "trynew", trynew, 'nest_depth', grid.nest_depth)
    ############################ sum_contained_all_offset_regions #######################################################################
    if len(c) > 1:
        if weight_valid_area:
            @time_func_perf
            def sum_contained_all_offset_regions(
                    pt_row,
                    pt_col,
            ):
                """
                returns sum for cells contained in search radius for all points within cell. Additionally returns invalid area as float
                """
                cells_cntd_by_pt_cell = sparse_grid_ids.intersection(
                    [(lvl,(row+pt_row,col+pt_col)) for lvl,(row,col) in shared_contained_cells_lookup])
                # invalid_area = len(invalid_cells.intersection(
                #     [(lvl, (row+pt_row,col+pt_col)) for lvl,(row,col) in shared_contained_cells_lookup])) * grid_spacing**2
                # invalid_area = grid_spacing*sum(
                #     [2**(-2*lvl) for lvl, cell in invalid_cells.intersection([(lvl, (row+pt_row,col+pt_col)) for lvl,(row,col) in cells_contained_in_all_disks])]
                # )
                if len(cells_cntd_by_pt_cell)>0:
                    i = 0
                    for lvl_cell in cells_cntd_by_pt_cell:
                        sums_in_cell = grid_id_to_sums_by_lvl[lvl_cell]
                        empty_sums[i:i+len(sums_in_cell)] = sums_in_cell
                        i +=len(sums_in_cell)
                    return empty_sums[:i].sum(axis=0), 0#invalid_area 
                return zero_sums, 0#invalid_area 
            #
        else:# not weight_valid_area
            @time_func_perf
            def sum_contained_all_offset_regions(
                    pt_row,
                    pt_col,
            ):
                """
                returns sum for cells contained in search radius for all points within cell. Additionally returns invalid area as float
                """
                cells_cntd_by_pt_cell = sparse_grid_ids.intersection(
                    [(lvl, (row+pt_row,col+pt_col)) for lvl,(row,col) in shared_contained_cells_lookup])
                if len(cells_cntd_by_pt_cell)>0:
                    i = 0
                    for lvl_cell in cells_cntd_by_pt_cell:
                        sums_in_cell = grid_id_to_sums_by_lvl[lvl_cell]
                        empty_sums[i:i+len(sums_in_cell)] = sums_in_cell
                        i +=len(sums_in_cell)
                    return empty_sums[:i].sum(axis=0), 0 
                    # return _np_array([grid_id_to_sums_by_lvl[lvl_cell] for lvl_cell in cells_cntd_by_pt_cell]).sum(axis=0) 
                return zero_sums, 0
                #
                #
        #
    else: # len(c)==1
        if weight_valid_area:
            @time_func_perf
            def sum_contained_all_offset_regions(
                    pt_row,
                    pt_col,
            ):
                cells_cntd_by_pt_cell = sparse_grid_ids.intersection(
                    [(lvl, (row+pt_row,col+pt_col)) for lvl,(row,col) in shared_contained_cells_lookup])
                # invalid_area = len(invalid_cells.intersection(
                #     [(lvl, (row+pt_row,col+pt_col)) for lvl,(row,col) in shared_contained_cells_lookup])
                #     ) * grid_spacing**2
                return sum([grid_id_to_sums_by_lvl[lvl_cell] for lvl_cell in cells_cntd_by_pt_cell]), 0#invalid_area 
            #
        else:# not weight_valid_area
            @time_func_perf
            def sum_contained_all_offset_regions(
                    pt_row,
                    pt_col,
            ):
                cells_cntd_by_pt_cell = sparse_grid_ids.intersection(
                    [(lvl, (row+pt_row,col+pt_col)) for lvl,(row,col) in shared_contained_cells_lookup])
                return sum([grid_id_to_sums_by_lvl[lvl_cell] for lvl_cell in cells_cntd_by_pt_cell]), 0
            #
            
        #
    ############################ sum_contained_by_offset_region #######################################################################
    if len(c) > 1:
        if weight_valid_area:
            @time_func_perf
            def sum_contained_by_offset_region(
                    pt_row,
                    pt_col,
                    cell_region_id,
            ):
                cells_contained_by_pt_region = sparse_grid_ids.intersection(
                    [(lvl,(row+pt_row,col+pt_col)) for lvl,(row,col) in region_id_to_contained_cells_lookup[cell_region_id]])
                invalid_area = len(invalid_cells.intersection(
                    [(lvl,(row+pt_row,col+pt_col)) for lvl,(row,col) in region_id_to_contained_cells[cell_region_id]])
                    ) * grid_spacing**2
                if len(cells_contained_by_pt_region)>0:
                    return _np_array([grid_id_to_sums_by_lvl[lvl_cell] for lvl_cell in cells_contained_by_pt_region]).sum(axis=0), invalid_area
                return zero_sums, invalid_area
            #
        else:# not weight_valid_area
            @time_func_perf
            def sum_contained_by_offset_region(
                    pt_row,
                    pt_col,
                    cell_region_id,
            ):
                cells_contained_by_pt_region = sparse_grid_ids.intersection([(lvl,(row+pt_row,col+pt_col)) for lvl,(row,col) in region_id_to_contained_cells_lookup[cell_region_id]])
                if len(cells_contained_by_pt_region)>0:
                    return _np_array([grid_id_to_sums_by_lvl[lvl_cell] for lvl_cell in cells_contained_by_pt_region]).sum(axis=0), 0
                return zero_sums, 0
            #
        #
    else:# len(c)==1
        if weight_valid_area:
            @time_func_perf
            def sum_contained_by_offset_region(
                    pt_row,
                    pt_col,
                    cell_region_id,
            ):
                cells_contained_by_pt_region = sparse_grid_ids.intersection(
                    [(lvl,(row+pt_row,col+pt_col)) for lvl,(row,col) in region_id_to_contained_cells_lookup[cell_region_id]])
                invalid_area = len(invalid_cells.intersection(
                    [(row+pt_row,col+pt_col) for lvl,(row,col) in region_id_to_contained_cells[cell_region_id]])
                    ) * grid_spacing**2
                return sum([grid_id_to_sums_by_lvl[lvl_cell] for lvl_cell in cells_contained_by_pt_region]), invalid_area
            #
        else:# not weight_valid_area
            @time_func_perf
            def sum_contained_by_offset_region(
                    pt_row,
                    pt_col,
                    cell_region_id,
            ):
                cells_contained_by_pt_region = sparse_grid_ids.intersection(
                    [(lvl,(row+pt_row,col+pt_col)) for lvl,(row,col) in region_id_to_contained_cells_lookup[cell_region_id]])
                return sum([grid_id_to_sums_by_lvl[lvl_cell] for lvl_cell in cells_contained_by_pt_region]), 0
            #
        #
    #

    ############################ get_pts_overlapped_by_all #######################################################################
    @time_func_perf
    def get_pts_overlapped_by_all(
                pt_row,
                pt_col
        ):
        i = 0
        for lvl_cell in sparse_grid_ids.intersection(
                [(lvl, (row+pt_row,col+pt_col)) for lvl,(row,col) in shared_overlapped_cells_lookup]):
            xy_vals_in_cell = grid_id_to_vals_xy_by_lvl[lvl_cell]
            empty_xy_vals[i:i+len(xy_vals_in_cell)] = xy_vals_in_cell
            i +=len(xy_vals_in_cell)
        return empty_xy_vals[:i]

    ############################ get_pts_overlapped_by_region #######################################################################
    if weight_valid_area:
        @time_func_perf
        def get_pts_overlapped_by_region(
                pt_row,
                pt_col,
                cell_region_id,
                n:int,
        ):  
            
            overlapped_invalid_cells = invalid_cells.intersection(
                [(row+pt_row,col+pt_col) for lvl,(row,col) in region_id_to_overlapped_cells[cell_region_id]])
            i = n
            for lvl_cell in sparse_grid_ids.intersection(
                    [(lvl, (row+pt_row,col+pt_col)) for lvl,(row,col) in region_id_to_overlapped_cells_lookup[cell_region_id]]):
                xy_vals_in_cell = grid_id_to_vals_xy_by_lvl[lvl_cell]
                empty_xy_vals[i:i+len(xy_vals_in_cell)] = xy_vals_in_cell
                i +=len(xy_vals_in_cell)
            return empty_xy_vals[n:i], overlapped_invalid_cells
        #
    else:# not weight_valid_area
        @time_func_perf
        def get_pts_overlapped_by_region(
                pt_row,
                pt_col,
                cell_region_id,
                n:int,
        ):  
            i = n
            for lvl_cell in sparse_grid_ids.intersection(
                    [(lvl, (row+pt_row,col+pt_col)) for lvl,(row,col) in region_id_to_overlapped_cells_lookup[cell_region_id]]):
                xy_vals_in_cell = grid_id_to_vals_xy_by_lvl[lvl_cell]
                empty_xy_vals[i:i+len(xy_vals_in_cell)] = xy_vals_in_cell
                i +=len(xy_vals_in_cell)
            return empty_xy_vals[n:i], []
            
    

    ############################ sum_overlapped_pts_in_radius #######################################################################
    if len(c) > 1:
        @time_func_perf
        def sum_overlapped_pts_in_radius(
            vals_xy_distinct_overlapped,
            pt_xycoord
        ):
            if len(vals_xy_distinct_overlapped) > 0:
                vals_in_radius = vals_xy_distinct_overlapped[:,:-2][(_np_linalg_norm( # create a mask of boolean values indicating if point are within radius 
                    vals_xy_distinct_overlapped[:,-2:] -
                    pt_xycoord, 
                axis=1) <= r)]
                
                return vals_in_radius.sum(axis=0) if len(vals_in_radius) > 0 else zero_sums
                # else no points in radius thus return vector of _np_zeros
            return zero_sums
    else:# len(c)==1
        @time_func_perf
        def sum_overlapped_pts_in_radius(
            vals_xy_distinct_overlapped,
            pt_xycoord
        ):
            if len(vals_xy_distinct_overlapped) > 0:
                return vals_xy_distinct_overlapped[:,:-2][(_np_linalg_norm( # create a mask of boolean values indicating if point are within radius 
                    vals_xy_distinct_overlapped[:,-2:] -
                    pt_xycoord, 
                axis=1) <= r)].sum(axis=0)
                # else no points in radius thus return vector of _np_zeros
            return 0

    ############################ weight_valid_area #######################################################################
    if weight_valid_area == 'precise':
        if r**2<2*grid_spacing**2:
            print("WARNING: Precise intersection method of search circle and grid cells is only implemented for search radius >= (2*grid_spacing**2)**0.5. Calculation of valid area thus might be false.")
        
        
        @time_func_perf
        def calculate_overlapped_invalid_area(
            pt_xyoffset:tuple,
            pt_row:int, 
            pt_col:int,
            invalid_overlapped_cells,
            **kwargs
        ) -> float:
            # This is slow. Either increase the speed or make a simple function that maps centroid distance to area estimate.
            
            return sum([disk_cell_intersection_area(
                    pt_xyoffset,
                    row_col=(int(row-pt_row),int(col-pt_col)), # TO-DO this
                    grid_spacing=grid_spacing,
                    r=r,
                    silent=True,
                    ) for row,col in invalid_overlapped_cells])
            
    elif weight_valid_area == 'estimate':
        
        # define here as it depends on grid_spacing / r
        @time_func_perf
        def estimate_overlapped_area_share(
            disk_center_pt_s:_np_array,      
            centroid_s:tuple=_np_array,
            logit_Q:float=1 / (0.70628102 + _np_exp(0.57266908 * (grid_spacing / r - 2))),
            logit_B:float=1 / (-0.21443453 + _np_exp(0.76899004 * (grid_spacing / r - 2))),
            r:float=r,
        ) -> _np_array:
            """
            either disk_center_pt_s or centroid_s can be more than one element not both
            returns numpy.array with share of grid cells that is overlapped by radius each element is in [0,1] or in (0,1) if cell is truly only overlapped
            """
            return 1 - 1 / (
                1.0 + logit_Q * _np_exp(
                    -logit_B * 
                        (1/r * _np_linalg_norm(disk_center_pt_s-centroid_s, axis=1) - 1)
                    )
                ) 
        
        @time_func_perf
        def calculate_overlapped_invalid_area(
            pt_xycoord,
            invalid_overlapped_cells:set,
            **kwargs
            ) -> float:
            """
            Call intersection area estimation function based on distance, radius and grid_spacing.
            Mean estimation error of 5% of cell area. Largest error for cells where only one vertex of cell lies within radius (~20%)
            """
            # This is slow. Either increase the speed or make a simple function that maps centroid distance to area estimate.
            return 0.0 if len(invalid_overlapped_cells)==0 else estimate_overlapped_area_share(
                    disk_center_pt_s=pt_xycoord,
                    centroid_s=_np_array([row_col_to_centroid.get((int(row),int(col)),get_cell_centroid(row,col)) for row,col in invalid_overlapped_cells]),
                    # centroid_s=_np_array([row_col_to_centroid[(int(row),int(col))] for row,col in invalid_overlapped_cells]),
                    ).sum() * grid_spacing ** 2
    
    else:
        
        if weight_valid_area != False and not weight_valid_area is None:
            # move to handle inputs
            print("Value for 'weight_valid_area' must be in ['precise', 'estimate', 'guess', False]. Instead",weight_valid_area,"was provided.")
        weight_valid_area = False
    #

    @time_func_perf
    def do_nothing():
        pass
    
    for (i, pt_id, pt_xycoord, pt_xyoffset, (pt_row,pt_col), contain_region_id, overlap_region_id, cell_region_id) in zip(
        range(n_pts),
        pts_source.index,
        pts_source[[x, y,]].values, 
        pts_source[[off_x, off_y,]].values, 
        pts_source[[row_name, col_name]].values,
        pts_source[cell_region_name].values // grid.search.contain_region_mult,
        pts_source[cell_region_name].values % grid.search.contain_region_mult,
        pts_source[cell_region_name].values,
        
        
        ):
        (pt_row, pt_col) = (int(pt_row), int(pt_col))
        # as pts are sorted by grid cell update only if grid cell changed
        if not (pt_row, pt_col) == last_pt_row_col:
            counter_new_cell += 1
            sums_cells_cntd_by_pt_cell, invalid_search_area_cntd_by_pt_cell = sum_contained_all_offset_regions(pt_row, pt_col)
            vals_xy_shared_overlapped = get_pts_overlapped_by_all(pt_row, pt_col)
            do_nothing()
        #
            
        if (pt_row, pt_col) != last_pt_row_col or last_contain_region_id != contain_region_id:
            counter_new_contain_region += 1
            # if cell changed or cell region changed
            (sums_contained_by_pt_region, 
             invalid_search_area_cntd_by_pt_region) = sum_contained_by_offset_region(pt_row, pt_col, cell_region_id)
            do_nothing()

        if (pt_row, pt_col) != last_pt_row_col or last_overlap_region_id != overlap_region_id:
            counter_new_overlap_region += 1
            (vals_xy_distinct_overlapped, 
             invalid_overlapped_cells) = get_pts_overlapped_by_region(pt_row, pt_col, cell_region_id, len(vals_xy_shared_overlapped))

        #
        shared_overlap_sums = sum_overlapped_pts_in_radius(vals_xy_shared_overlapped, pt_xycoord)
        distinct_overlap_sums = sum_overlapped_pts_in_radius(vals_xy_distinct_overlapped, pt_xycoord)


        # combine sums from the steps.
        # append result 
        sums_within_disks[i,:] = (
            sums_cells_cntd_by_pt_cell + 
            sums_contained_by_pt_region + 
            shared_overlap_sums + 
            distinct_overlap_sums)
        # for inspecting
        all_sums_cells_cntd_by_pt_cell[i,:] = sums_cells_cntd_by_pt_cell
        all_sums_contained_by_pt_region[i,:] = sums_contained_by_pt_region
        all_shared_overlap_sums[i,:] = shared_overlap_sums
        all_distinct_overlap_sums[i,:] = distinct_overlap_sums
        
        # calculate share of valid area
        if weight_valid_area:
            invalid_search_area_overlaps = calculate_overlapped_invalid_area(
                    pt_xyoffset=pt_xyoffset,
                    pt_xycoord=pt_xycoord,
                    pt_row=pt_row, 
                    pt_col=pt_col,
                    invalid_overlapped_cells=invalid_overlapped_cells,
                )
            valid_area_shares[i] = (
                valid_search_area - 
                invalid_search_area_cntd_by_pt_cell - 
                invalid_search_area_cntd_by_pt_region - 
                invalid_search_area_overlaps
                ) / valid_search_area
    
        # plot example pint
        if plot_pt_disk is not None and pt_id == plot_pt_disk['pt_id']:
            # cells_cntd_by_pt_cell = sparse_grid_ids.intersection([(row+pt_row,col+pt_col) for lvl,(row,col) in cells_contained_in_all_disks])
            # cells_cntd_by_pt_cell = sparse_grid_ids.intersection([(lvl, (row+pt_row,col+pt_col)) for lvl,(row,col) in shared_contained_cells_lookup])
            # print("cells_cntd_by_pt_cell",cells_cntd_by_pt_cell)
            # cells_contained_by_pt_region = sparse_grid_ids.intersection(
            #     [(lvl,(row+pt_row,col+pt_col)) for lvl,(row,col) in region_id_to_contained_cells[cell_region_id]+cells_contained_in_all_disks])
            # print("cells_contained_by_pt_region",cells_contained_by_pt_region)
            # # print("(pt_row,pt_col)",(pt_row,pt_col))
            # # print("pt_xycoord",pt_xycoord)
            # a =  _np_array(flatten_list([
            #     grid_id_to_vals_xy_by_lvl[cell_id] for cell_id 
            #     in cells_contained_by_pt_region
            # ]))
            # print("region_id_to_overlapped_cells_incl_nested[cell_region_id]\n",region_id_to_overlapped_cells_lookup[cell_region_id])
           
            # # print("a",a)
            # # print("a.shape",a.shape)
            # # print("a[:,-2:]",a[:,-2:])
            # # print("vals_xy_distinct_overlapped",vals_xy_distinct_overlapped)
            # # for cell_id in cells_contained_by_pt_region:
            # #     print("XX",cell_id, grid_id_to_vals_xy_by_lvl[cell_id])
            
            # print("vals_xy_distinct_overlapped",vals_xy_distinct_overlapped)
            # print("linalg",_np_linalg_norm( 
            #     vals_xy_distinct_overlapped[:,-2:] -
            #     pt_xycoord, 
            # axis=1) <= r)
            vals_xy_overlapped = _np_array(list(vals_xy_distinct_overlapped)+list(vals_xy_shared_overlapped))
            
            pts_xy_in_radius = vals_xy_overlapped[:,-2:][(_np_linalg_norm( 
                vals_xy_overlapped[:,-2:] -
                pt_xycoord, 
            axis=1) <= r)]
            
            pts_xy_in_cells_overlapped_by_pt_region = vals_xy_overlapped[:,-2:][(_np_linalg_norm( 
                vals_xy_overlapped[:,-2:] -
                pt_xycoord, 
            axis=1) > r)]

            pts_xy_in_cell_contained_by_pt_region = _np_array(flatten_list([
                grid_id_to_vals_xy_by_lvl[cell_id] for cell_id 
                in sparse_grid_ids.intersection(
                    [(lvl,(row+pt_row,col+pt_col)) for lvl,(row,col) in 
                     set(list(shared_contained_cells_lookup)+list(region_id_to_contained_cells_lookup[cell_region_id]))])
            ]))[:,-2:]
            
            # shared_contained_cells_lookup = []
            # region_id_to_contained_cells_lookup = region_id_to_nested_contained_cells
            # shared_overlapped_cells_lookup = []
            # region_id_to_overlapped_cells_lookup = region_id_to_nested_overlapped_cells
            illustrate_point_disk(
                grid=grid,
                pts_source=pts_source,
                pts_target=pts_target,
                r=r,
                c=c,
                x=x,
                y=y,
                shared_contained_cells=[(lvl,(row+pt_row,col+pt_col)) for lvl,(row,col) in shared_contained_cells_lookup],
                shared_overlapped_cells=[(lvl,(row+pt_row,col+pt_col)) for lvl,(row,col) in shared_overlapped_cells_lookup],
                distinct_contained_cells=[(lvl,(row+pt_row,col+pt_col)) for lvl,(row,col) in region_id_to_contained_cells_lookup[cell_region_id]],
                distinct_overlapped_cells=[(lvl,(row+pt_row,col+pt_col)) for lvl,(row,col) in region_id_to_overlapped_cells_lookup[cell_region_id]],
                pts_xy_in_cell_contained_by_pt_region = pts_xy_in_cell_contained_by_pt_region,
                pts_xy_in_cells_overlapped_by_pt_region=pts_xy_in_cells_overlapped_by_pt_region,
                pts_xy_in_radius=pts_xy_in_radius,
                home_cell=(pt_row,pt_col),
                region_id=contain_region_id*grid.search.contain_region_mult+overlap_region_id,
                **plot_pt_disk,
            )
        # #

        # set id as last id for next iteration
        last_pt_row_col = (pt_row, pt_col)
        last_contain_region_id = contain_region_id
        last_overlap_region_id = overlap_region_id
    #
    pts_source[sum_radius_names] = pts_source[sum_radius_names].values + sums_within_disks
    # ensure correct dtypes
    pts_source = pts_source.astype({n:dt for n,dt, in zip(sum_radius_names, column_dtypes)})
    
    if exclude_pt_itself and grid.search.tgt_df_contains_src_df:
        # substract data from point itself unless specified otherwise
        for sum_radius_name, col_name in zip(sum_radius_names, c):
            pts_source[sum_radius_name] = pts_source[sum_radius_name].values - pts_source[col_name]
    
    if weight_valid_area:
        pts_source['valid_area_share'+sum_suffix] = valid_area_shares
        for sum_radius_name in sum_radius_names:
            pts_source[sum_radius_name] = pts_source[sum_radius_name].values / pts_source['valid_area_share'+sum_suffix].values
        if silent != True:
            print("Appended radius sum"+("" if len(c)<=1 else "s")+" (r="+str(r)+") for " +', '.join(["'"+cname+"' as '"+sname+"'" for (cname,sname) in zip(c, sum_radius_names)])+" to pts DataFrame. (Sum names can be controlled by setting sum_suffix='...')")    
            if weight_valid_area:
                print("Appended valid area share as "+"'valid_area_share"+sum_suffix+"' to pts DataFrame.")    
    # print("all_sums_cells_cntd_by_pt_cell", sum(all_sums_cells_cntd_by_pt_cell))
    # print("all_sums_contained_by_pt_region",sum(all_sums_contained_by_pt_region))#
    # print("all_shared_overlap_sums",sum(all_shared_overlap_sums))
    # print("all_distinct_overlap_sums",sum(all_distinct_overlap_sums))
    # print("FINAL RESULT---------------",sum(sums_within_disks)) 
    # print(
    #     "Share of pts in",
    #     "\n- same cell as previous:", 100-int(counter_new_cell/len(pts_source)*100),"%",
    #     "\n- same cell and containing same surrounding cells:",100 - int(counter_new_contain_region/len(pts_source)*100),"%",
    #     "\n- same cell and overlapping same surrounding cells",100 - int(counter_new_overlap_region/len(pts_source)*100),"%")
    def plot_vars(
        self = grid,
        colnames = _np_array([c, sum_radius_names]), 
        filename:str='',
        **plot_kwargs:dict,
    ):
        return create_plots_for_vars(
            grid=self,
            colnames=colnames,
            filename=filename,
            plot_kwargs=plot_kwargs,
        )

    grid.plot.vars = plot_vars


    return pts_source[sum_radius_names]
#



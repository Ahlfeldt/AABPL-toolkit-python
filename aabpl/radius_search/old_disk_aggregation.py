from numpy import array as _np_array, zeros as _np_zeros,exp as _np_exp 
from numpy.linalg import norm as _np_linalg_norm
from pandas import (DataFrame as _pd_DataFrame, cut as _pd_cut, concat as _pd_concat) 
from aabpl.utils.misc import flatten_list
from aabpl.utils.progress import SearchProgress as _SearchProgress
from aabpl.utils.cell_geometry import classify_disk_cells_by_level
from aabpl.illustrations.plot_disk import illustrate_point_disk
from aabpl.illustrations.plot_pt_vars import create_plots_for_vars
from aabpl.testing.test_performance import time_func_perf
from math import pi as _math_pi
from .sample_area import compute_disk_cell_overlap, estimate_disk_cell_overlap

################ search_and_aggregate ######################################################################################
@time_func_perf
def search_and_aggregate(
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
    validate:bool=False,
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
    
    region_id_to_cntd_cells = grid.search.region_id_to_cntd_cells
    region_id_to_ovlpd_cells = grid.search.region_id_to_ovlpd_cells
    shared_cntd_cells  = grid.search.shared_cntd_cells

    # All nested/distinct lookups are keyed by region_and_trgl_id = region_id * REGION_AND_TRGL_MULT + trgl_nr.
    # This encodes the triangle so that merged symmetric regions use the correct per-triangle
    # nested cells without any triangle logic here in the search loop.
    region_and_trgl_to_nested_cntd_cells   = grid.search.region_and_trgl_id_to_nested_cntd_cells
    region_and_trgl_to_distinct_cntd_cells = grid.search.region_and_trgl_id_to_distinct_cntd_cells
    region_and_trgl_to_distinct_ovlpd_cells= grid.search.region_and_trgl_id_to_distinct_ovlpd_cells

    row_col_to_centroid = grid.row_col_to_centroid
    get_cell_centroid = grid.get_cell_centroid
    pt_id_to_xy_coords = grid.search.target.pt_id_to_xy_coords
    n_pts = len(pts_source)
    _search_prog = _SearchProgress(silent=silent, n_pts=n_pts)
    shared_cntd_cells_lookup = shared_cntd_cells
    region_id_to_cntd_cells_lookup = region_and_trgl_to_distinct_cntd_cells
    region_id_to_ovlpd_cells_lookup = region_and_trgl_to_distinct_ovlpd_cells
    
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
    all_sums_cntd_by_pt_region = _np_zeros((n_pts, len(c)))
    all_distinct_overlap_sums = _np_zeros((n_pts, len(c)))
    sums_within_disks = _np_zeros((n_pts, len(c)))
    valid_area_shares = _np_zeros(n_pts)
    valid_search_area = _math_pi * r**2
    column_dtypes = pts_target[c].dtypes
    zero_sums = _np_zeros(len(c),dtype=int) if len(c) > 1 else 0
    r2 = r * r

    # create empty arrays for storing intermediate results to avoid reallocation in loop. choose size based on max possible number of cells relevant for disk search
    # TODO the nest depth can be chosen according to density of the region (super cell count)
    # TODO ensure this works as intended if multiple lvls area used
    n_cells_pot_relevant = max([len(cells) for cells in region_id_to_ovlpd_cells.values()])
    max_len_empty = sum(sorted([len(v) for v in grid_id_to_vals_xy.values()])[-n_cells_pot_relevant:])
    empty_sums = _np_zeros((max_len_empty,len(c)),dtype=int)
    empty_xy_vals = _np_zeros((max_len_empty,len(c)+2),dtype=float)
    
    pts_source.sort_values([row_name, col_name, 'region_and_trgl_id'], inplace=True)
    last_pt_row_col = (None,None)
    last_contain_region_id = -1
    last_overlap_region_id = -1
    last_region_and_trgl_id = -1
    counter_new_cell = 0
    counter_new_contain_region = 0
    counter_new_overlap_region = 0
    
    # print("weight_valid_area",weight_valid_area, "trynew", trynew, 'nest_depth', grid.nest_depth)
    ############################ sum_cntd_all_offset_regions #######################################################################
    if len(c) > 1:
        if weight_valid_area:
            @time_func_perf
            def sum_cntd_all_offset_regions(
                    pt_row,
                    pt_col,
            ):
                """
                returns sum for cells cntd in search radius for all points within cell. Additionally returns invalid area as float
                """
                cells_cntd_by_pt_cell = sparse_grid_ids.intersection(
                    [(lvl,(row+pt_row,col+pt_col)) for lvl,(row,col) in shared_cntd_cells_lookup])
                # invalid_area = len(invalid_cells.intersection(
                #     [(lvl, (row+pt_row,col+pt_col)) for lvl,(row,col) in shared_cntd_cells_lookup])) * grid_spacing**2
                # invalid_area = grid_spacing*sum(
                #     [2**(-2*lvl) for lvl, cell in invalid_cells.intersection([(lvl, (row+pt_row,col+pt_col)) for lvl,(row,col) in cells_cntd_in_all_disks])]
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
            def sum_cntd_all_offset_regions(
                    pt_row,
                    pt_col,
            ):
                """
                returns sum for cells cntd in search radius for all points within cell. Additionally returns invalid area as float
                """
                cells_cntd_by_pt_cell = sparse_grid_ids.intersection(
                    [(lvl, (row+pt_row,col+pt_col)) for lvl,(row,col) in shared_cntd_cells_lookup])
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
            def sum_cntd_all_offset_regions(
                    pt_row,
                    pt_col,
            ):
                cells_cntd_by_pt_cell = sparse_grid_ids.intersection(
                    [(lvl, (row+pt_row,col+pt_col)) for lvl,(row,col) in shared_cntd_cells_lookup])
                # invalid_area = len(invalid_cells.intersection(
                #     [(lvl, (row+pt_row,col+pt_col)) for lvl,(row,col) in shared_cntd_cells_lookup])
                #     ) * grid_spacing**2
                return sum([grid_id_to_sums_by_lvl[lvl_cell] for lvl_cell in cells_cntd_by_pt_cell]), 0#invalid_area 
            #
        else:# not weight_valid_area
            @time_func_perf
            def sum_cntd_all_offset_regions(
                    pt_row,
                    pt_col,
            ):
                cells_cntd_by_pt_cell = sparse_grid_ids.intersection(
                    [(lvl, (row+pt_row,col+pt_col)) for lvl,(row,col) in shared_cntd_cells_lookup])
                return sum([grid_id_to_sums_by_lvl[lvl_cell] for lvl_cell in cells_cntd_by_pt_cell]), 0
            #
            
        #
    ############################ sum_cntd_by_offset_region #######################################################################
    if len(c) > 1:
        if weight_valid_area:
            @time_func_perf
            def sum_cntd_by_offset_region(
                    pt_row,
                    pt_col,
                    cell_region_id,
                    region_and_trgl_id,
            ):
                nc = region_id_to_cntd_cells_lookup[region_and_trgl_id]
                cells_cntd_by_pt_region = sparse_grid_ids.intersection(
                    [(lvl,(row+pt_row,col+pt_col)) for lvl,(row,col) in nc])
                invalid_area = len(invalid_cells.intersection(
                    [(lvl,(row+pt_row,col+pt_col)) for lvl,(row,col) in region_id_to_cntd_cells[cell_region_id]])
                    ) * grid_spacing**2
                if len(cells_cntd_by_pt_region)>0:
                    return _np_array([grid_id_to_sums_by_lvl[lvl_cell] for lvl_cell in cells_cntd_by_pt_region]).sum(axis=0), invalid_area
                return zero_sums, invalid_area
            #
        else:# not weight_valid_area
            @time_func_perf
            def sum_cntd_by_offset_region(
                    pt_row,
                    pt_col,
                    cell_region_id,
                    region_and_trgl_id,
            ):
                nc = region_id_to_cntd_cells_lookup[region_and_trgl_id]
                cells_cntd_by_pt_region = sparse_grid_ids.intersection([(lvl,(row+pt_row,col+pt_col)) for lvl,(row,col) in nc])
                if len(cells_cntd_by_pt_region)>0:
                    return _np_array([grid_id_to_sums_by_lvl[lvl_cell] for lvl_cell in cells_cntd_by_pt_region]).sum(axis=0), 0
                return zero_sums, 0
            #
        #
    else:# len(c)==1
        if weight_valid_area:
            @time_func_perf
            def sum_cntd_by_offset_region(
                    pt_row,
                    pt_col,
                    cell_region_id,
                    region_and_trgl_id,
            ):
                nc = region_id_to_cntd_cells_lookup[region_and_trgl_id]
                cells_cntd_by_pt_region = sparse_grid_ids.intersection(
                    [(lvl,(row+pt_row,col+pt_col)) for lvl,(row,col) in nc])
                invalid_area = len(invalid_cells.intersection(
                    [(row+pt_row,col+pt_col) for lvl,(row,col) in region_id_to_cntd_cells[cell_region_id]])
                    ) * grid_spacing**2
                return sum([grid_id_to_sums_by_lvl[lvl_cell] for lvl_cell in cells_cntd_by_pt_region]), invalid_area
            #
        else:# not weight_valid_area
            @time_func_perf
            def sum_cntd_by_offset_region(
                    pt_row,
                    pt_col,
                    cell_region_id,
                    region_and_trgl_id,
            ):
                nc = region_id_to_cntd_cells_lookup[region_and_trgl_id]
                cells_cntd_by_pt_region = sparse_grid_ids.intersection(
                    [(lvl,(row+pt_row,col+pt_col)) for lvl,(row,col) in nc])
                return sum([grid_id_to_sums_by_lvl[lvl_cell] for lvl_cell in cells_cntd_by_pt_region]), 0
            #
        #
    #

    ############################ get_pts_ovlpd_by_region #######################################################################
    if weight_valid_area:
        @time_func_perf
        def get_pts_ovlpd_by_region(
                pt_row,
                pt_col,
                cell_region_id,
                region_and_trgl_id,
        ):
            no = region_id_to_ovlpd_cells_lookup[region_and_trgl_id]
            ovlpd_invalid_cells = invalid_cells.intersection(
                [(row+pt_row,col+pt_col) for lvl,(row,col) in region_id_to_ovlpd_cells[cell_region_id]])
            i = 0
            for lvl_cell in sparse_grid_ids.intersection(
                    [(lvl, (row+pt_row,col+pt_col)) for lvl,(row,col) in no]):
                xy_vals_in_cell = grid_id_to_vals_xy_by_lvl[lvl_cell]
                empty_xy_vals[i:i+len(xy_vals_in_cell)] = xy_vals_in_cell
                i +=len(xy_vals_in_cell)
            return empty_xy_vals[:i], ovlpd_invalid_cells
        #
    else:# not weight_valid_area
        @time_func_perf
        def get_pts_ovlpd_by_region(
                pt_row,
                pt_col,
                cell_region_id,
                region_and_trgl_id,
        ):
            no = region_id_to_ovlpd_cells_lookup[region_and_trgl_id]
            i = 0
            for lvl_cell in sparse_grid_ids.intersection(
                    [(lvl, (row+pt_row,col+pt_col)) for lvl,(row,col) in no]):
                xy_vals_in_cell = grid_id_to_vals_xy_by_lvl[lvl_cell]
                empty_xy_vals[i:i+len(xy_vals_in_cell)] = xy_vals_in_cell
                i +=len(xy_vals_in_cell)
            return empty_xy_vals[:i], []
            
    

    ############################ sum_ovlpd_pts_in_radius #######################################################################
    if len(c) > 1:
        @time_func_perf
        def sum_ovlpd_pts_in_radius(
            vals_xy_distinct_ovlpd,
            pt_xycoord
        ):
            if len(vals_xy_distinct_ovlpd) == 0:
                return zero_sums
            xy = vals_xy_distinct_ovlpd[:, -2:]
            dx = xy[:, 0] - pt_xycoord[0]
            dy = xy[:, 1] - pt_xycoord[1]
            bbox = (dx <= r) & (dx >= -r) & (dy <= r) & (dy >= -r)
            if not bbox.any():
                return zero_sums
            dx2 = dx[bbox]; dy2 = dy[bbox]
            mask = dx2 * dx2 + dy2 * dy2 <= r2
            if not mask.any():
                return zero_sums
            return vals_xy_distinct_ovlpd[bbox, :-2][mask].sum(axis=0)
    else:# len(c)==1
        @time_func_perf
        def sum_ovlpd_pts_in_radius(
            vals_xy_distinct_ovlpd,
            pt_xycoord
        ):
            if len(vals_xy_distinct_ovlpd) == 0:
                return 0
            xy = vals_xy_distinct_ovlpd[:, -2:]
            dx = xy[:, 0] - pt_xycoord[0]
            dy = xy[:, 1] - pt_xycoord[1]
            bbox = (dx <= r) & (dx >= -r) & (dy <= r) & (dy >= -r)
            if not bbox.any():
                return 0
            dx2 = dx[bbox]; dy2 = dy[bbox]
            mask = dx2 * dx2 + dy2 * dy2 <= r2
            return vals_xy_distinct_ovlpd[bbox, :-2][mask].sum(axis=0)

    ############################ weight_valid_area #######################################################################
    if weight_valid_area == 'precise':
        if r**2<2*grid_spacing**2:
            print("WARNING: Precise intersection method of search circle and grid cells is only implemented for search radius >= (2*grid_spacing**2)**0.5. Calculation of valid area thus might be false.")
        
        
        @time_func_perf
        def calculate_ovlpd_invalid_area(
            pt_xyoffset:tuple,
            pt_row:int, 
            pt_col:int,
            invalid_ovlpd_cells,
            **kwargs
        ) -> float:
            # This is slow. Either increase the speed or make a simple function that maps centroid distance to area estimate.
            
            return sum([compute_disk_cell_overlap(
                    pt_xyoffset,
                    row_col=(int(row-pt_row),int(col-pt_col)), # TO-DO this
                    grid_spacing=grid_spacing,
                    r=r,
                    silent=True,
                    ) for row,col in invalid_ovlpd_cells])
            
    elif weight_valid_area == 'estimate':
        
        # define here as it depends on grid_spacing / r
        @time_func_perf
        def estimate_ovlpd_area_share(
            disk_center_pt_s:_np_array,      
            centroid_s:tuple=_np_array,
            logit_Q:float=1 / (0.70628102 + _np_exp(0.57266908 * (grid_spacing / r - 2))),
            logit_B:float=1 / (-0.21443453 + _np_exp(0.76899004 * (grid_spacing / r - 2))),
            r:float=r,
        ) -> _np_array:
            """
            either disk_center_pt_s or centroid_s can be more than one element not both
            returns numpy.array with share of grid cells that is ovlpd by radius each element is in [0,1] or in (0,1) if cell is truly only ovlpd
            """
            return 1 - 1 / (
                1.0 + logit_Q * _np_exp(
                    -logit_B * 
                        (1/r * _np_linalg_norm(disk_center_pt_s-centroid_s, axis=1) - 1)
                    )
                ) 
        
        @time_func_perf
        def calculate_ovlpd_invalid_area(
            pt_xycoord,
            invalid_ovlpd_cells:set,
            **kwargs
            ) -> float:
            """
            Call intersection area estimation function based on distance, radius and grid_spacing.
            Mean estimation error of 5% of cell area. Largest error for cells where only one vertex of cell lies within radius (~20%)
            """
            # This is slow. Either increase the speed or make a simple function that maps centroid distance to area estimate.
            return 0.0 if len(invalid_ovlpd_cells)==0 else estimate_ovlpd_area_share(
                    disk_center_pt_s=pt_xycoord,
                    centroid_s=_np_array([row_col_to_centroid.get((int(row),int(col)),get_cell_centroid(row,col)) for row,col in invalid_ovlpd_cells]),
                    # centroid_s=_np_array([row_col_to_centroid[(int(row),int(col))] for row,col in invalid_ovlpd_cells]),
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

    _search_prog.start()
    _search_thresh = _search_prog.next_threshold  # local copy: single int comparison per iteration

    for (i, pt_id, pt_xycoord, pt_xyoffset, (pt_row,pt_col), contain_region_id, overlap_region_id, cell_region_id, region_and_trgl_id) in zip(
        range(n_pts),
        pts_source.index,
        pts_source[[x, y,]].values,
        pts_source[[off_x, off_y,]].values,
        pts_source[[row_name, col_name]].values,
        pts_source[cell_region_name].values // grid.search.contain_region_mult,
        pts_source[cell_region_name].values % grid.search.contain_region_mult,
        pts_source[cell_region_name].values,
        pts_source['region_and_trgl_id'].values,
        ):
        (pt_row, pt_col) = (int(pt_row), int(pt_col))
        # as pts are sorted by grid cell update only if grid cell changed
        if not (pt_row, pt_col) == last_pt_row_col:
            counter_new_cell += 1
            sums_cells_cntd_by_pt_cell, invalid_search_area_cntd_by_pt_cell = sum_cntd_all_offset_regions(pt_row, pt_col)
            # do_nothing()

        # if cell changed or cell region changed update sums for cntd and overlapped cells in region.
        if (pt_row, pt_col) != last_pt_row_col or last_contain_region_id != contain_region_id or last_region_and_trgl_id != region_and_trgl_id:
            counter_new_contain_region += 1
            (sums_cntd_by_pt_region,
             invalid_search_area_cntd_by_pt_region) = sum_cntd_by_offset_region(pt_row, pt_col, cell_region_id, region_and_trgl_id)
            # do_nothing()

        # if cell changed or overlap region changed update pts ovlpd by region. as these are costly to retrieve and filter for radius, we want to avoid retrieving them if not necessary.
        if (pt_row, pt_col) != last_pt_row_col or last_overlap_region_id != overlap_region_id or last_region_and_trgl_id != region_and_trgl_id:
            counter_new_overlap_region += 1
            (vals_xy_distinct_ovlpd,
             invalid_ovlpd_cells) = get_pts_ovlpd_by_region(pt_row, pt_col, cell_region_id, region_and_trgl_id)

        #
        distinct_overlap_sums = sum_ovlpd_pts_in_radius(vals_xy_distinct_ovlpd, pt_xycoord)

        # combine sums from the steps.
        # append result
        sums_within_disks[i,:] = (
            sums_cells_cntd_by_pt_cell +
            sums_cntd_by_pt_region +
            distinct_overlap_sums)
        # for inspecting
        # all_sums_cells_cntd_by_pt_cell[i,:] = sums_cells_cntd_by_pt_cell
        # all_sums_cntd_by_pt_region[i,:] = sums_cntd_by_pt_region
        # all_shared_overlap_sums[i,:] = shared_overlap_sums
        # all_distinct_overlap_sums[i,:] = distinct_overlap_sums
        
        # calculate share of valid area
        if weight_valid_area:
            invalid_search_area_overlaps = calculate_ovlpd_invalid_area(
                    pt_xyoffset=pt_xyoffset,
                    pt_xycoord=pt_xycoord,
                    pt_row=pt_row, 
                    pt_col=pt_col,
                    invalid_ovlpd_cells=invalid_ovlpd_cells,
                )
            valid_area_shares[i] = (
                valid_search_area - 
                invalid_search_area_cntd_by_pt_cell - 
                invalid_search_area_cntd_by_pt_region - 
                invalid_search_area_overlaps
                ) / valid_search_area
    
        # plot example point
        if plot_pt_disk is not None and pt_id == plot_pt_disk['pt_id']:
            vals_xy_ovlpd = vals_xy_distinct_ovlpd
            pts_xy_in_radius = vals_xy_ovlpd[:,-2:][(_np_linalg_norm(
                vals_xy_ovlpd[:,-2:] - pt_xycoord, axis=1) <= r)]
            pts_xy_in_cells_ovlpd_by_pt_region = vals_xy_ovlpd[:,-2:][(_np_linalg_norm(
                vals_xy_ovlpd[:,-2:] - pt_xycoord, axis=1) > r)]
            _illus_nc = region_id_to_cntd_cells_lookup[region_and_trgl_id]
            _illus_no = region_id_to_ovlpd_cells_lookup[region_and_trgl_id]
            pts_xy_in_cell_cntd_by_pt_region = _np_array(flatten_list([
                grid_id_to_vals_xy_by_lvl[cell_id] for cell_id
                in sparse_grid_ids.intersection(
                    [(lvl,(row+pt_row,col+pt_col)) for lvl,(row,col) in
                     set(list(shared_cntd_cells_lookup)+list(_illus_nc))])
            ]))[:,-2:]
            illustrate_point_disk(
                grid=grid,
                pts_source=pts_source,
                pts_target=pts_target,
                r=r,
                c=c,
                x=x,
                y=y,
                shared_cntd_cells=[(lvl,(row+pt_row,col+pt_col)) for lvl,(row,col) in shared_cntd_cells_lookup],
                shared_ovlpd_cells=[],
                distinct_cntd_cells=[(lvl,(row+pt_row,col+pt_col)) for lvl,(row,col) in _illus_nc],
                distinct_ovlpd_cells=[(lvl,(row+pt_row,col+pt_col)) for lvl,(row,col) in _illus_no],
                pts_xy_in_cell_cntd_by_pt_region=pts_xy_in_cell_cntd_by_pt_region,
                pts_xy_in_cells_ovlpd_by_pt_region=pts_xy_in_cells_ovlpd_by_pt_region,
                pts_xy_in_radius=pts_xy_in_radius,
                sums_within_disk=sums_within_disks[i,:],
                sum_names=sum_radius_names,
                cell_region_id=cell_region_id,
                home_cell=(pt_row,pt_col),
                region_id=contain_region_id*grid.search.contain_region_mult+overlap_region_id,
                **plot_pt_disk,
            )
        # #

        # set id as last id for next iteration
        last_pt_row_col = (pt_row, pt_col)
        last_contain_region_id = contain_region_id
        last_overlap_region_id = overlap_region_id
        last_region_and_trgl_id = region_and_trgl_id
        if i >= _search_thresh:
            _search_thresh = _search_prog.update(i)
    #
    _search_prog.done()
    pts_source[sum_radius_names] = pts_source[sum_radius_names].values + sums_within_disks
    # ensure correct dtypes
    pts_source = pts_source.astype({n:dt for n,dt, in zip(sum_radius_names, column_dtypes)})

    if exclude_pt_itself and grid.search.tgt_df_contains_src_df:
        # substract data from point itself unless specified otherwise
        for sum_radius_name, _excl_cname in zip(sum_radius_names, c):
            pts_source[sum_radius_name] = pts_source[sum_radius_name].values - pts_source[_excl_cname]
    
    if weight_valid_area:
        pts_source['valid_area_share'+sum_suffix] = valid_area_shares
        for sum_radius_name in sum_radius_names:
            pts_source[sum_radius_name] = pts_source[sum_radius_name].values / pts_source['valid_area_share'+sum_suffix].values
        if silent != True:
            print("Appended radius sum"+("" if len(c)<=1 else "s")+" (r="+str(r)+") for " +', '.join(["'"+cname+"' as '"+sname+"'" for (cname,sname) in zip(c, sum_radius_names)])+" to pts DataFrame. (Sum names can be controlled by setting sum_suffix='...')")    
            if weight_valid_area:
                print("Appended valid area share as "+"'valid_area_share"+sum_suffix+"' to pts DataFrame.")    
    # ---- brute-force validation ----
    if not validate:
        return pts_source[sum_radius_names]
    # For each cell_region, pick one representative point from the most populated cell
    # that contains a point in that cell_region, then verify against O(n^2) distances.
    all_xy = pts_target[[x, y]].values
    all_vals = pts_target[c].values if len(c) > 1 else pts_target[c[0]].values.reshape(-1, 1)
    lvl0_cells = {k: v for k, v in grid_id_to_pt_ids.items() if k[0] == 0}
    cell_pop = {k: len(v) for k, v in lvl0_cells.items() if len(v) > 0}
    pts_source['_cell_key'] = list(zip(
        pts_source[row_name].astype(int).map(lambda r: (0, r)),
        pts_source[col_name].astype(int),
    ))
    pts_source['_cell_pop'] = pts_source.apply(
        lambda row: cell_pop.get((0, (int(row[row_name]), int(row[col_name]))), 0), axis=1
    )
    # for each cell_region pick the representative from the most populated cell
    rep_indices = (
        pts_source
        .sort_values('_cell_pop', ascending=False)
        .groupby(cell_region_name, sort=False)
        .apply(lambda g: g.index[0])
    )
    pts_source.drop(columns=['_cell_key', '_cell_pop'], inplace=True)
    errors = []
    for cr, rep_idx in rep_indices.items():
        rep_xy = pts_source.loc[rep_idx, [x, y]].values.astype(float)
        dists = _np_linalg_norm(all_xy - rep_xy, axis=1)
        brute_sums = all_vals[dists <= r].sum(axis=0)
        if exclude_pt_itself and grid.search.tgt_df_contains_src_df:
            own_vals = (pts_target.loc[rep_idx, c].values.astype(float)
                        if len(c) > 1
                        else _np_array([float(pts_target.loc[rep_idx, c[0]])]))
            brute_sums = brute_sums - own_vals
        algo_sums = pts_source.loc[rep_idx, sum_radius_names].values.astype(float)
        diff = abs(brute_sums - algo_sums).max()
        if diff > 0:
            errors.append((rep_idx, cr, brute_sums, algo_sums, diff))
    if errors:
        print(f"VALIDATION FAILED: {len(errors)}/{len(rep_indices)} cell_region(s) have wrong sums:")
        for rep_idx, cr, bf, algo, diff in errors:
            print(f"  pt_id={rep_idx} cell_region={cr} brute={bf} algo={algo} diff={diff}")
    else:
        print(f"VALIDATION OK: all {len(rep_indices)} cell_region(s) correct.")

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



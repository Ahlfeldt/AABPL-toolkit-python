from numpy import (
    array as _np_array,
    append as _np_append,
    ndarray as _np_ndarray,
    arange as _np_arange,
    column_stack as _np_column_stack,
    hstack as _np_hstack,
    ones as _np_ones,
    zeros as _np_zeros,
    unique as _np_unique, 
    concatenate as _np_concatenate,
    equal as _np_equal, 
    logical_or as _np_logical_or, 
    all as _np_all, 
    sort as _np_sort
)
from pandas import (DataFrame as _pd_DataFrame, cut as _pd_cut, concat as _pd_concat) 
from math import log10 as _math_log10#,asin,acos
from aabpl.utils.general import (flatten_list,DataFrameRelation,arr_to_tpls, make_bins_from_vals, get_vals_from_bins)
from aabpl.illustrations.illustrate_point_to_cell_region_assignment import (illustrate_point_to_cell_region_assignment)
from .two_dimensional_weak_ordering_class import (gen_weak_order_rel_to_convex_set,recursive_cell_region_inference)
from aabpl.testing.test_performance import time_func_perf, func_timer_dict
from .offset_regions import prepare_offset_regions
# from primelocations.doc.docstrings import fixdocstring

################ classify_point_triangle ######################################################################################
@time_func_perf
def classify_point_triangle(
        x:_np_array,
        y:_np_array,
)->_np_array:    
    """
    Classifies 2D coordinate plane into eight triangle regions (quadrants splitted along 45deg lines)

    Args:
      x (numpy.array):
        Array of x/longtitude coordinates
      y (numpy.array):
        Array of y/lattitude coordinates 
    Returns:
      triangle_ids (numpy.array):
        triangle number (int) for each input coordinate. 2D plane is defieded as follows:
        - Triangle 1: x >  0, y >= 0 and |x| >  |y|. OR x=y=0. 
        - Triangle 2: x >  0, y >  0 and |x| <= |y|. 
        - Triangle 3: x <= 0, y >  0 and |x| <  |y|.
        - Triangle 4: x <  0, y >  0 and |x| >= |y|.
        - Triangle 5: x <  0, y <= 0 and |x| >  |y|.
        - Triangle 6: x <  0, y <  0 and |x| <= |y|.
        - Triangle 7: x >= 0, y <  0 and |x| <  |y|.
        - Triangle 8: x >  0, y <  0 and |x| >= |y|.
    """

    triangle_ids = _np_ones(len(x),dtype=int)
    triangle_ids[(x >  0) * (y >= 0) * (abs(x) >  abs(y))] = 1
    triangle_ids[(x >  0) * (y >  0) * (abs(x) <= abs(y))] = 2
    triangle_ids[(x <= 0) * (y >  0) * (abs(x) <  abs(y))] = 3
    triangle_ids[(x <  0) * (y >  0) * (abs(x) >= abs(y))] = 4
    triangle_ids[(x <  0) * (y <= 0) * (abs(x) >  abs(y))] = 5
    triangle_ids[(x <  0) * (y <  0) * (abs(x) <  abs(y))] = 6
    triangle_ids[(x >= 0) * (y <  0) * (abs(x) <  abs(y))] = 7
    triangle_ids[(x >  0) * (y <  0) * (abs(x) >= abs(y))] = 8

    return triangle_ids
#

################ transform_cell_pattern ######################################################################################
def transform_cell_pattern(
    cells:_np_array,
    i:int,
)->_np_array:
    """
    Transform a pattern of cells (row_nr, col_nr) around origin cell (0,0) to match a 
    transformation of point from Triangle 1 into any other Triangle 1-8.

    Transformation from Traingle 1 to Triangle X: cell(row, col): explanation
    - 1: ( r, c): keep unchanged
    - 2: ( c, r): mirror along +45deg line (y=x)
    - 3: ( c,-r): rotate by 90° counter clockwise
    - 4: ( r,-c): mirror along +45deg line (y=x). rotate by 90° counter clockwise
    - 5: (-r,-c): rotate by 180° counter clockwise
    - 6: (-c,-r): mirror along +45deg line (y=x). rotate by 180° counter clockwise
    - 7: (-c, r): rotate by 270° counter clockwise
    - 8: (-r, c): mirror along +45deg line (y=x). rotate by 270° counter clockwise

    Args:
        cells (numpy.array):
        Array of cells that will be rotated. each cell has (row_nr, col_nr). shape=(n,2)
        i (int):
        Tranform a cell pattern relative to Triangle 1 such that is has the same properties relative
        to Triangle i   
    Returns:
        rotated_cells (numpy.array):
        cell rotated from Triangle to Triangle [i].
    """
    if i not in [1,2,3,4,5,6,7,8]:
        raise ValueError("Triangle number must be an integer between 1 and 8")
    if type(cells) != _np_array:
        cells = _np_array(cells)
    else:
        cells = cells.copy()

    if i == 1: return cells[:,:] # no change *array([1,1]) # Triangle 1
    if i == 2: return cells[:,::-1] * _np_array([ 1, 1])   # Triangle 2
    if i == 3: return cells[:,::-1] * _np_array([ 1,-1])   # Triangle 3
    if i == 4: return cells[:,:]    * _np_array([ 1,-1])   # Triangle 4
    if i == 5: return cells[:,:]    * _np_array([-1,-1])   # Triangle 5
    if i == 6: return cells[:,::-1] * _np_array([-1,-1])   # Triangle 6
    if i == 7: return cells[:,::-1] * _np_array([-1, 1])   # Triangle 7
    if i == 8: return cells[:,:]    * _np_array([-1, 1])   # Triangle 8
#


################ get_pt_to_cell_centroid_triangle_offset ######################################################################################
@time_func_perf
def get_pt_to_cell_centroid_triangle_offset(
    grid:dict,
    pts_df:_pd_DataFrame,
    x_coord_name:str='lon',
    y_coord_name:str='lat',
    row_name:str='id_y',
    col_name:str='id_x',
)->tuple:
    """
    Calculate the offset of each point relative to the centroid of its grid cell
    Infer and store into which of the eight triangles (quadrants divided along 45deg lines) the points falls
    Transform the offset coordiante such that the point lies within Triangle1 (x>=0,y>=0,x>=y) TODO
    Args:
      TODO (TODO):
        TODO
    Returns:
      triangle_offset_info (tuple): 
      - cntrd_offset_xy (numpy.array): Point offset to centroid of its cell. Shape=len(pts_df),2
      - cntrd_trngl_offset_xy (numpy.array): Point offset to centroid coord transformed into Triangle1. Shape=len(pts_df),2
      - triangle_ids (numpy.array): int 1-8 indicating Triangle in which point offset to centroid coord falls. Shape=len(pts_df),
        
    """    
    # unpack values from Grid dictionary
    # get vectors of row columns boundary values
    y_steps=grid.y_steps
    x_steps=grid.x_steps
    
    # each point is contained in grid cell. Calculate the offset of each point to the centroid of its grid cell.
    # TODO cut may contain repeated expensive operation - thus already assigned cell vectors could be used to just query cell centroid by cell_id from dict 
    # lon/x offset
    offset_x = pts_df[x_coord_name] - _pd_cut(
        x=pts_df[x_coord_name],
        bins=x_steps, labels=[(x_low+x_up)/2 for x_low,x_up in zip(x_steps[:-1], x_steps[1:])],
        include_lowest=True).astype(float)
    
    # lat/y offset
    offset_y = pts_df[y_coord_name] - _pd_cut(
        x=pts_df[y_coord_name],
        bins=y_steps, labels=[(y_low+y_up)/2 for y_low,y_up in zip(y_steps[:-1], y_steps[1:])],
        # bins=y_steps, labels=[(y_low+y_up)/2 for y_low,y_up in zip(y_steps[1:], y_steps[:-1])],
        include_lowest=True).astype(float)
    
    # id_y_mult = grid.id_y_mult
    # cell_centroid = _np_array([grid.row_col_to_centroid[row*id_y_mult+col] for row,col in pts_df[[row_name, col_name]]])

    # offset_x = cell_centroid[:,0] - pts_df[x_coord_name]
    # offset_y = cell_centroid[:,1] - pts_df[y_coord_name]
    pts_df['offset_x'] = offset_x.copy()
    pts_df['offset_y'] = offset_y.copy()
    # combine into single 2D _np_array of shape=(n_pts, 2)
    cntrd_offset_xy = _np_column_stack([offset_x, offset_y])
    # infer triangle in which the points falls
    triangle_ids = classify_point_triangle(x=offset_x,y=offset_y)
    pts_df['triangle_id'] = triangle_ids
    # Transform offset pt s.t. each offset pt is contained in triangle 1.
    # as we have a store the triangle id, we can reverse the transformation later on.
    cntrd_trngl_offset_xy = _np_sort(abs(cntrd_offset_xy),axis=1)[:,::-1]
    
    return (cntrd_offset_xy, cntrd_trngl_offset_xy, triangle_ids)
#

################ get_pt_to_cell_centroid_cell_offset ######################################################################################
@time_func_perf
def get_pt_to_cell_centroid_cell_offset(
    grid:dict,
    pts_df:_pd_DataFrame,
    x_coord_name:str='lon',
    y_coord_name:str='lat',
    row_name:str='id_y',
    col_name:str='id_x',
)->_np_array:
    """
    Calculate the offset of each point relative to the centroid of its grid cell
    Infer and store into which of the eight triangles (quadrants divided along 45deg lines) the points falls
    Transform the offset coordiante such that the point lies within Triangle1 (x>=0,y>=0,x>=y) TODO
    Args:
      TODO (TODO):
        TODO
    Returns:
      triangle_offset_info (tuple): 
      - cntrd_offset_xy (numpy.array): Point offset to centroid of its cell. Shape=len(pts_df),2
      - cntrd_trngl_offset_xy (numpy.array): Point offset to centroid coord transformed into Triangle1. Shape=len(pts_df),2
      - triangle_ids (numpy.array): int 1-8 indicating Triangle in which point offset to centroid coord falls. Shape=len(pts_df),
        
    """    
    # unpack values from Grid dictionary
    # get vectors of row columns boundary values
    y_steps=grid.y_steps
    x_steps=grid.x_steps
    
    # each point is contained in grid cell. Calculate the offset of each point to the centroid of its grid cell.
    # TODO cut may contain repeated expensive operation - thus already assigned cell vectors could be used to just query cell centroid by cell_id from dict 
    # lon/x offset
    offset_x = pts_df[x_coord_name] - _pd_cut(
        x=pts_df[x_coord_name],
        bins=x_steps, labels=[(x_low+x_up)/2 for x_low,x_up in zip(x_steps[:-1], x_steps[1:])],
        include_lowest=True).astype(float)
    
    # lat/y offset
    offset_y = pts_df[y_coord_name] - _pd_cut(
        x=pts_df[y_coord_name],
        bins=y_steps, labels=[(y_low+y_up)/2 for y_low,y_up in zip(y_steps[1:], y_steps[:-1])],
        # bins=y_steps[::-1], labels=[(y_low+y_up)/2 for y_low,y_up in zip(y_steps[1:], y_steps[:-1])][::-1],
        include_lowest=True).astype(float)
    
    # id_y_mult = grid.id_y_mult
    # cell_centroid = _np_array([grid.row_col_to_centroid[row*id_y_mult+col] for row,col in pts_df[[row_name, col_name]]])
    offset_xy_unscaled = _np_column_stack([offset_x, offset_y]) / grid.spacing
    # offset_x = cell_centroid[:,0] - pts_df[x_coord_name]
    # offset_y = cell_centroid[:,1] - pts_df[y_coord_name]
    pts_df['offset_x'] = offset_xy_unscaled[:,0].copy()
    pts_df['offset_y'] = offset_xy_unscaled[:,1].copy()
    # combine into single 2D _np_array of shape=(n_pts, 2)
    
    return offset_xy_unscaled
#


def combine_rotated_regions_if_same_relevant_cells(
    relevant_cells:_np_array,
)->dict:
    """
    shares region ?
    """
    # check whether this regions shares its contained regions 
    # with no other triangle,
    # a neighbouring triangle (along diagonal OR along vertical=horizontal axis),
    # or all triangles (implied by symmetrie sharing when sharing with both neighbours )
    a=set(arr_to_tpls(_np_unique(relevant_cells, axis=0),int))
    b=set(arr_to_tpls(_np_unique(transform_cell_pattern(relevant_cells,2), axis=0),int))
    c=set(arr_to_tpls(_np_unique(transform_cell_pattern(relevant_cells,8), axis=0),int))
    
    shared_along_diagonal = a == b 
    shared_along_vertical = a == c

    if shared_along_diagonal and shared_along_vertical:
        return {1:1,2:1,3:1,4:1,5:1,6:1,7:1,8:1}
    if shared_along_diagonal:
        return {1:1,2:1,3:3,4:3,5:5,6:5,7:7,8:7}  
    elif shared_along_vertical:
        return {1:1,2:3,3:3,4:5,5:5,6:7,7:7,8:1}
    return {1:1,2:2,3:3,4:4,5:5,6:6,7:7,8:8}
#

################ create_cell_region ######################################################################################
@time_func_perf
def create_cell_region(
    grid,
    region_id:int,
    combined_disk_check_result:_np_array,
):
    """
    
    TODO explain
    """

    disk_contains_cells = combined_disk_check_result[:grid.search.weak_order_tree.n_checks_if_contained]
    disk_overlaps_cells = combined_disk_check_result[grid.search.weak_order_tree.n_checks_if_contained:]

    # get row, col for cells contained
    region_contained_cells = grid.search.weak_order_tree.check_if_contained_order[disk_contains_cells]
    # add cells that are always inside
    if len(region_contained_cells) > 0:
        region_contained_cells = _np_concatenate([region_contained_cells, grid.search.cells_contained_in_all_trgl_disks])
    else: # handle case for no fully covered cells
        region_contained_cells = grid.search.cells_contained_in_all_trgl_disks
    # get row, col for cells overlapped
    region_overlapped_cells = grid.search.weak_order_tree.check_if_overlapped_order[disk_overlaps_cells] if disk_overlaps_cells.any() else _np_ndarray(shape=(0,2))
    
    # add cells that were not checked for overlap as they are always overlapped
    additional_always_overlapped_cells = [
        cell for cell in grid.search.weak_order_tree.cells_overlapped_by_all_trgl1_disks 
        if not any((region_contained_cells[:]==cell).all(1))
    ]
    if len(additional_always_overlapped_cells) > 0:
        region_overlapped_cells = _np_concatenate([
            region_overlapped_cells, _np_array(additional_always_overlapped_cells)
        ])
    
    # # sort cells to perform equality checks
    region_contained_cells = _np_unique(region_contained_cells, axis=0)
    region_overlapped_cells = _np_unique(region_overlapped_cells, axis=0)
    # region_contained_cells = arr_to_tpls(region_contained_cells, int)
    # region_overlapped_cells = arr_to_tpls(region_overlapped_cells, int)
     
    trans_dict_contained = combine_rotated_regions_if_same_relevant_cells(region_contained_cells)
    trans_dict_overlapped = combine_rotated_regions_if_same_relevant_cells(region_overlapped_cells)

    # NEW CELL REGIONS
    trgl_region_to_cell_region =  {i:region_id + trans_dict_contained[i]*10+trans_dict_overlapped[i] for i in [1,2,3,4,5,6,7,8]}
    
    return trgl_region_to_cell_region, region_contained_cells, region_overlapped_cells
#

################ assign_points_to_cell_regions ######################################################################################
@time_func_perf
def assign_points_to_cell_regions(
    grid:dict,
    pts_df,
    radius:float=0.0075,
    include_boundary:bool=False,
    x_coord_name:str='lon',
    y_coord_name:str='lat',
    row_name:str='id_y',
    col_name:str='id_x',
    cell_region_name:str='cell_region',
    plot_cell_reg_assign:dict=None,
    silent:bool=False,
) -> _pd_DataFrame:
    """
    region to determine in which relative region within cell the point lies
    TODO explain
    """

    pts_df.sort_values([y_coord_name, x_coord_name], inplace=True)
    n_pts = len(pts_df)

    # TODO can this function be split up here?
    (cntrd_offset_xy, cntrd_trngl_offset_xy, triangle_ids) = get_pt_to_cell_centroid_triangle_offset(
        grid=grid,
        pts_df=pts_df,
        x_coord_name=x_coord_name,
        y_coord_name=y_coord_name,
        row_name=row_name,
        col_name=col_name,
    )

    pts_df['triangle_id'] = triangle_ids # TODO remove this
    # intialize 2D matrices to store results  
    disks_by_cells_contains = _np_zeros((n_pts, grid.search.weak_order_tree.n_checks_if_contained),dtype=bool)
    disks_by_cells_overlaps = _np_zeros((n_pts, grid.search.weak_order_tree.n_checks_if_overlapped),dtype=bool)
    
    # apply recursive checks on cells (defined in family_tree_flat) on whether disks around pts fully or partially contain them 
    scaled_to_grid_cntrd_trngl_offset_xy = cntrd_trngl_offset_xy / grid.spacing
    pts_df['scaled_cntrd_trngl_offset_x']=scaled_to_grid_cntrd_trngl_offset_xy[:,0]
    pts_df['scaled_cntrd_trngl_offset_y']=scaled_to_grid_cntrd_trngl_offset_xy[:,1]
    recursive_cell_region_inference(
        grid=grid,
        transformed_offset_xy=scaled_to_grid_cntrd_trngl_offset_xy,
        reference_ids=_np_arange(n_pts),
        disks_by_cells_contains=disks_by_cells_contains,
        disks_by_cells_overlaps=disks_by_cells_overlaps,
        family_tree_pos=grid.search.weak_order_tree.root,
        grid_spacing=grid.spacing,
        radius=radius,
        include_boundary=include_boundary,
    )
    #

    # stack together to get determine unique combination of fully and partly covered cells by disks around all pts offsets 
    combined_disk_check_results = _np_hstack([disks_by_cells_contains,disks_by_cells_overlaps])
    
    unique_combined_disk_check_results = _np_unique(combined_disk_check_results,axis=0)

    # initialize, update in loop
    region_ids = _np_zeros(n_pts,dtype=int)-1
    grid.search.region_id_to_contained_cells = {}
    grid.search.region_id_to_overlapped_cells = {}
    region_id = 0 
    # region id shall consist of contain_region_id * mult + overlap_region_id
    # how to get number of overlap patterns? its bounded above by _np_unique(disks_by_cells_overlaps, axis=1)*8
    print("_np_unique(disks_by_cells_overlaps, axis=1)",len(_np_unique(disks_by_cells_overlaps, axis=0)), (_np_unique(disks_by_cells_overlaps, axis=0).shape))
    grid.search.contain_region_mult =  10**(int(_math_log10(len(_np_unique(disks_by_cells_overlaps, axis=0))*8))+1)

    contain_region_ids = dict()
    overlap_region_ids = dict()

    # loop over all unique combinations of fully and partly covered cells by disks around all pts offsets
    for combined_disk_check_result in unique_combined_disk_check_results:
        
        # create mask to reference pts that fullfill this condition
        filter_pts_in_region = _np_all(_np_equal(combined_disk_check_results,combined_disk_check_result),axis=1)
        (triangle_id_to_new_cell_region, region_contained_cells, region_overlapped_cells )= create_cell_region(
            grid,
            region_id,
            combined_disk_check_result
        )
        trgl_i_to_region_id = dict()
        for i in [1,2,3,4,5,6,7,8]:
            contained_regions_rotated_i = sorted(arr_to_tpls(transform_cell_pattern(region_contained_cells, i),int))
            overlapped_regions_rotated_i = sorted(arr_to_tpls(transform_cell_pattern(region_overlapped_cells, i),int))

            if tuple(contained_regions_rotated_i) not in contain_region_ids:
                contain_region_ids[tuple(contained_regions_rotated_i)] = len(contain_region_ids)
            #
            if tuple(overlapped_regions_rotated_i) not in overlap_region_ids:
                overlap_region_ids[tuple(overlapped_regions_rotated_i)] = len(overlap_region_ids)
            #

            contain_region_id = contain_region_ids[tuple(contained_regions_rotated_i)]
            overlap_region_id = overlap_region_ids[tuple(overlapped_regions_rotated_i)]
            region_id_i = contain_region_id * grid.search.contain_region_mult + overlap_region_id

            if not region_id_i in grid.search.region_id_to_contained_cells:
                grid.search.region_id_to_contained_cells[region_id_i] = contained_regions_rotated_i
            if not region_id_i in grid.search.region_id_to_overlapped_cells:
                grid.search.region_id_to_overlapped_cells[region_id_i] = overlapped_regions_rotated_i
            trgl_i_to_region_id[i] = region_id_i

        region_ids[filter_pts_in_region] = _np_array([
            trgl_i_to_region_id[int(t_id)] for t_id in triangle_ids[filter_pts_in_region]
        ])

        # print("triangle_id_to_new_cell_region",triangle_id_to_new_cell_region)
        # # save pattern and its rotated variations in dict
        # for (i, region_id_i) in triangle_id_to_new_cell_region.items():
        #     if not region_id_i in grid.search.region_id_to_contained_cells:
        #         grid.search.region_id_to_contained_cells[region_id_i] = arr_to_tpls(transform_cell_pattern(region_contained_cells, i),int)
        #     if not region_id_i in grid.search.region_id_to_overlapped_cells:
        #         grid.search.region_id_to_overlapped_cells[region_id_i] = arr_to_tpls(transform_cell_pattern(region_overlapped_cells, i),int)
        #     #
        # #
        # # save cell regions ids to vector entries for pts in current cell region
        # region_ids[filter_pts_in_region] = _np_array([
        #     triangle_id_to_new_cell_region[int(t_id)] for t_id in triangle_ids[filter_pts_in_region]
        # ])
        # # increase region id to host next gen without conflicts  
        # region_id += 100

    # SAVE cell_region_id TO POINT DATAFRAME
    pts_df[cell_region_name] = region_ids
    
    # add triangle id (to recover full information of pts after forcing them into triangle 1) and substract 1 (as triangle ids start with 1)
    # except for the a potential region around centroid that is shared among all triangles
    if plot_cell_reg_assign is not None:
        illustrate_point_to_cell_region_assignment(
            grid=grid,
            triangle_ids=triangle_ids,
            region_ids=region_ids,
            offset_xy = cntrd_offset_xy,
            transformed_offset_xy=cntrd_trngl_offset_xy,
            radius=radius,
            include_boundary=include_boundary,
            **plot_cell_reg_assign,
        )
    #
    
    # TODO MAKE PRINTS SMALLER
    if not silent:
        print(
            len(grid.search.cells_maybe_overlapping_a_disk),'cells are potentially within radius of a point within cell. For a point in cell region:',
            round(_np_array([len(v) for v in grid.search.region_id_to_overlapped_cells.values()]).mean(),1),
            ' are potentially within radius',
            round(_np_array([len(v) for v in grid.search.region_id_to_contained_cells.values()]).mean(),1),
            ' are fully contained additional to',len(grid.search.cells_contained_in_all_disks),'that are contained for any cell.'
        )
        print(
            str(len(pts_df.index) - _np_logical_or(pts_df[col_name]==-1, pts_df[row_name]==-1).sum())+
            '/'+str(len(pts_df.index)) +
            ' Points assigned to '+str(len(_np_unique(pts_df[[row_name, col_name]].values,axis=0)))+
            '/'
            
            
            +str(len(grid.ids))+' cells ' +
            'with '+ str(len(pts_df[cell_region_name].unique())) +' regions resulting in '+
            str(len(_np_unique(pts_df[[row_name, col_name,cell_region_name]].values,axis=0))) +' unique cell region combinations.'
            
        )
    # now convert results of recursive function into an integer that call a dicitionary to retrieve relevant cells
    # region_id = _np_zeros(len(offset_xy),dtype=int)-1
    return
#

################ assign_points_to_mirco_regions ######################################################################################
@time_func_perf
def assign_points_to_mirco_regions(
    grid:dict,
    pts_df,
    radius:float=0.0075,
    include_boundary:bool=False,
    x_coord_name:str='lon',
    y_coord_name:str='lat',
    row_name:str='id_y',
    col_name:str='id_x',
    cell_region_name:str='cell_region',
    plot_cell_reg_assign:dict=None,
    plot_offset_checks:dict=None,
    plot_offset_regions:dict=None,
    plot_offset_raster:dict=None,
    silent:bool=False,
) -> _pd_DataFrame:
    """
    region to determine in which relative region within cell the point lies
    TODO explain
    """

    pts_df.sort_values([y_coord_name, x_coord_name], inplace=True)
    n_pts = len(pts_df)

    # TODO can this function be split up here?
    offset_xy_unscaled = get_pt_to_cell_centroid_cell_offset(
        grid=grid,
        pts_df=pts_df,
        x_coord_name=x_coord_name,
        y_coord_name=y_coord_name,
        row_name=row_name,
        col_name=col_name,
    )
    # scale offset to cell side length of 1
    
    # print("Type offset_xy_unscaled", offset_xy_unscaled)
    raster_cell_to_region_comb_nr, offset_region_comb_nr_to_check, offset_all_x_vals, offset_all_y_vals, id_to_offset_regions, contain_region_mult = prepare_offset_regions(
        grid_spacing=1, radius=radius/grid.spacing, include_boundary=include_boundary, 
        plot_offset_checks=plot_offset_checks, plot_offset_regions=plot_offset_regions, plot_offset_raster=plot_offset_raster,
        silent=silent,
    )
    grid.search.contain_region_mult = contain_region_mult
    grid.id_to_offset_regions = id_to_offset_regions
    

    lbls_x = [-ix for ix in reversed(range(1, (len(offset_all_x_vals)-1)//2+1))] + list(range(1,(len(offset_all_x_vals)-1)//2+1))
    lbls_y = [-iy for iy in reversed(range(1, (len(offset_all_y_vals)-1)//2+1))] + list(range(1,(len(offset_all_y_vals)-1)//2+1))
 
    # cells are inprecise - some points slightly overlap
    offset_all_x_vals[ 0] -= 0.01
    offset_all_x_vals[-1] += 0.01
    offset_all_y_vals[ 0] -= 0.01
    offset_all_y_vals[-1] += 0.01

    pts_df['initial_sort'] = range(len(pts_df))
    pts_df.sort_values('offset_x', inplace=True)

    micro_raster_x = _pd_cut(
        x = pts_df['offset_x'],
        bins = offset_all_x_vals,
        labels = lbls_x,
        include_lowest = True
    ).astype(int)
    # pts_df['offset_ny'] = -pts_df['offset_y']
    micro_raster_y = _pd_cut(
        x = pts_df['offset_y'],
        bins = offset_all_y_vals,
        labels = lbls_y,
        include_lowest = True
    ).astype(int)
    # TODO remove this next line
    pts_df['micro_raster_x'] = micro_raster_x
    pts_df['micro_raster_y'] = micro_raster_y
    
    pts_df['region_comb_nr'] = [raster_cell_to_region_comb_nr[(ix, iy)] for ix,iy in zip(micro_raster_x, micro_raster_y)]
    pts_df.sort_values('region_comb_nr', inplace=True)
    region_comb_nr = pts_df['region_comb_nr'].values
    pts_offset_xy = pts_df[['offset_x', 'offset_y']].values
    n_pts = len(pts_offset_xy)
    i=0
    pts_offset_region = -_np_ones(n_pts, dtype=int)
    # remaining_pts_region_comb_nr
    count=0
    grid.search.offset_region_comb_nr_to_check = offset_region_comb_nr_to_check
    while i < n_pts:
        count+=1
        current_offset_region_comb_nr = region_comb_nr[i]
        j = next((n+i for n, x in enumerate(region_comb_nr[i:]) if x != current_offset_region_comb_nr), n_pts)
        check = offset_region_comb_nr_to_check[current_offset_region_comb_nr]
        res = check(pts_offset_xy[i:j])
        pts_offset_region[i:j] = res
        i = j
    pts_df[cell_region_name] = pts_offset_region
    
    pts_df.sort_values('initial_sort', inplace=True)

    grid.search.id_to_offset_regions = id_to_offset_regions
    grid.search.region_id_to_contained_cells = {id: reg.contained_cells for id,reg in id_to_offset_regions.items()}
    grid.search.region_id_to_overlapped_cells = {id: reg.overlapped_cells for id,reg in id_to_offset_regions.items()}

    if 'testing' != 'testing':
        pts_df.drop('initial_sort', axis=1, inplace=True)
        print('Mean contained:', sum([len(v) for v in grid.search.region_id_to_contained_cells.values()])/len(grid.search.region_id_to_contained_cells))
        print('Mean overlapped:', sum([len(v) for v in grid.search.region_id_to_overlapped_cells.values()])/len(grid.search.region_id_to_overlapped_cells))
    else:
        # pts_df['triangle_id'] = classify_point_triangle(x=pts_df['offset_x'], y=pts_df['offset_y']) # TODO remove this
        pts_df.drop(['initial_sort', 'micro_raster_x', 'micro_raster_y','offset_x', 'offset_y'], axis=1, inplace=True)
   
#

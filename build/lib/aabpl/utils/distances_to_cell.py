from numpy import (
    array as _np_array,
    append as _np_append,
    invert as _np_invert,
    sign as _np_sign,
    all as _np_all,
    any as _np_any,
    zeros as _np_zeros,
    ones as _np_ones,
    dstack as _np_dstack,
    logical_or,
    where as _np_where, 
    arange, transpose as _np_transpose, unique, linspace, flip,  concatenate,
    min,
    max as _np_max, 
    equal, logical_and, 
    newaxis as _np_newaxis, 
    sort
)
from numpy.linalg import norm as _np_linalg_norm
from math import ceil as _math_ceil
from ..utils.general import ( flatten_list, visualize, depth, list_dict_keys, )


# TODO vectorize them such that they can take any input from shape(1,2) to shape(n,2)
# TODO add descriptions
def min_possible_dist_cells_to_cell(
        cell_orig:_np_array,
        cell_dest:_np_array=_np_array([0,0]),
        ) -> float:
    """
    TODO
    Returns:

    """
    if len(cell_orig.shape)==1:
        cell_orig = cell_orig[_np_newaxis]

    cell_diff = cell_orig - cell_dest
    
    return _np_linalg_norm(cell_diff - _np_sign(cell_diff), axis=1)
    # return grid_spacing*((row_diff-_np_sign(row_diff))**2 + (col_diff-_np_sign(col_diff))**2)**.5

#
def max_possible_dist_cells_to_cell(
        cell_dest:_np_array,
        cell_orig:_np_array=_np_array([0,0]),
    ) -> float:
    """
    TODO
    Returns:

    """
    if len(cell_orig.shape)==1:
        cell_orig = cell_orig[_np_newaxis]
        
    cell_diff = cell_orig - cell_dest
    return _np_linalg_norm(cell_diff + _np_where(cell_diff>=0,1,-1), axis=1)
    # return grid_spacing*((row_diff+(1 if row_diff>=0 else -1))**2 + (col_diff+(1 if col_diff>=0 else -1))**2)**.5
#
def min_possible_dist_trgl1_to_cell(
        cell_dest:_np_array,
        ) -> float:
    """
    TODO
    TODO link triangle 1 definition
    Returns:

    """
    (row, col) = cell_dest

    # return grid_spacing*((row-(.5 if row>0 else 0))**2 + (col-(.5 if col>0 else 0))**2)**.5
    return (
        (row-(1 if row>0 else 0 if row==0 else -0.5))**2 + 
        (col-(1 if col>0 else 0 if col==0 else -0.5))**2
        )**.5
#
def max_possible_dist_trgl1_to_cell(
        cell_dest:_np_array,
        ) -> float:
    """
    TODO
    unscaled
    Returns:

    """
    (row, col) = cell_dest
    return (
        (row+(0.5 if row>=0 else -1))**2 + 
        (col+(0.5 if col>=0 else -1))**2
        )**.5
#

def get_cell_closest_point_to_points(
        pts_xy:_np_array,
        cell:_np_array,
        ):
    """
    TODO THIS MATTER FOR CELL REGIONS, 
    THIS COULD ALSO RETURN LINES FOR POINTS  
    Returns:
    """
    # TODO also check if handle correctly case for same row / column
    pts_x_unscaled = pts_xy[:, 0]
    pts_y_unscaled = pts_xy[:,1]
    n_pts = pts_xy.shape[0]

    row,col = cell
    ys = (
        # TODO this could also account for cases where point is not within cell (0,0)
        pts_y_unscaled
    ) if row == 0 else (
        # pts_y_unscaled - (row - .5 * _np_sign(row))
       _np_zeros(n_pts) + row - .5 * _np_sign(row)
    )
    
    xs = (
        # TODO this could also account for cases where point is not within cell (0,0)
        pts_x_unscaled
    ) if col == 0 else (
        # pts_x_unscaled - (col - .5 * _np_sign(col))
        _np_zeros(n_pts) + col - .5 * _np_sign(col)
    )

    return _np_transpose([xs, ys], (1,0))
#

def get_cell_closest_point_to_point(
        pt_xy:_np_array,
        cell:_np_array,
        ):
    """
    TODO THIS MATTER FOR CELL REGIONS, 
    THIS COULD ALSO RETURN LINES FOR POINTS  
    Returns:
    """
    # TODO also check if handle correctly case for same row / column
    x,y = pt_xy
    row,col = cell
    return _np_array([
        x if col == 0 else col - .5 * _np_sign(col),
        y if row == 0 else row - .5 * _np_sign(row)
    ])
#

def min_dist_points_to_cell(
        pts_xy:_np_array,
        cell:_np_array,
        ) -> _np_array:
    """
    TODO
    TODO link triangle 1 definition
    Returns:
    ONLYACCEPTS POINTS WITHIN CELL (0,0)!
    POINTS MUST BE RELATIVE TO GRID [-0.5,0.5]
    """

    # TODO also check if handle correctly case for same row / column
    pts_x_unscaled = pts_xy[:, 0]
    pts_y_unscaled = pts_xy[:,1]
    n = pts_xy.shape[0]

    row,col = cell
    
    ydist = (
        # TODO this could also account for cases where point is not within cell (0,0)
        _np_zeros(n)
    ) if row == 0 else (
        pts_y_unscaled - (row - .5 * _np_sign(row))
    )
    
    xdist = (
        # TODO this could also account for cases where point is not within cell (0,0)
        _np_zeros(n)
    ) if col == 0 else (
        pts_x_unscaled - (col - .5 * _np_sign(col))
    )
    
    return _np_linalg_norm([xdist, ydist], axis=0)
#


def get_cell_farthest_vertex_to_point(
        pts_xy:_np_array,
        cell:_np_array,
        ):
    """
    TODO THIS MATTER FOR CELL REGIONS, 
    THIS COULD ALSO RETURN LINES FOR POINTS  
    Returns:

    """
    return (cell[::-1] + (cell[::-1] - pts_xy >= 0)-.5)
#

def max_dist_points_to_cell(
        pts_xy:_np_array,
        cell:_np_array,
        ) -> _np_array:
    """
    TODO
    Returns:
        1D array of min distance from points to cell
    """
    if len(pts_xy)==0:
        print("NOOOOPOINT1",pts_xy)
        return _np_array([])
    if len(cell)==0:
        print("NOOOOPOINTcell_cell",cell)
    return _np_linalg_norm(
        _np_array([
            get_cell_farthest_vertex_to_point(
                pts_xy,
                cell
            ) for pts_xy in pts_xy
            ]) - pts_xy,
        axis=1
    )
#

def is_within_radius(distances,radius:float,include_boundary):
    if include_boundary:
        return distances <= radius
    return distances < radius

def get_cells_relevant_for_disk_by_type(
        grid_spacing:float=250,
        radius:float=750,
        include_boundary:bool=False,
        return_type:str=[None, 'contained_by_all', 'contained_by_trgl', 'overlapped_by_all', 'overlapped_by_trgl'][0]
    ) -> tuple:
    """
    TODO turn it into smaller function to return a specific type only 
    For an undefined point in cell(0,0): Determines which cells are always fully within radius and which cell might be overlaped 
    If count == True returns only count of cells. If not True returns list of tuples of cell row col locations
    Note due to symmetry only 1/8 of radius needs to be checked 
    Returns
    - cells_contained_in_all_disks, 
    - cells_contained_in_all_trgl_disks, 
    - cells_maybe_overlapping_a_disk, 
    - cells_maybe_overlapping_a_trgl_disk
    """

    ratio = radius/grid_spacing
    cell_steps_max = _math_ceil(ratio+1)
    
    # create all cells within square
    unassigned_cells_in_max_steps_square = _np_array(flatten_list([[(row,col) for col in range(-cell_steps_max, cell_steps_max+1)] for row in range(-cell_steps_max, cell_steps_max+1)]))
    # compare farthest vertex of cells: bool if cell is in contained in disk of radius centered around any pt within grid cell 
    cell_is_contained_in_all_disks = grid_spacing * max_possible_dist_cells_to_cell(unassigned_cells_in_max_steps_square) <= radius
    cells_contained_in_all_disks = unassigned_cells_in_max_steps_square[cell_is_contained_in_all_disks,:]
    
    if 'contained_by_all' == return_type:
        return cells_contained_in_all_disks
    
    # those cells should not be part of other sets
    unassigned_cells_in_max_steps_square = unassigned_cells_in_max_steps_square[_np_invert(cell_is_contained_in_all_disks),:]
    # compare closest vertex of cells: bool if cell is potentially overlaped by a disk of radius around pt within grid cell 
    cell_may_overlap_a_disk = (
        grid_spacing * min_possible_dist_cells_to_cell(unassigned_cells_in_max_steps_square) <= radius
        ) if include_boundary else (
        grid_spacing * min_possible_dist_cells_to_cell(unassigned_cells_in_max_steps_square) < radius
        )
    # [min_possible_dist_cell_to_cell(cell, grid_spacing) <= radius for cell in unassigned_cells_in_max_steps_square]
    cells_maybe_overlapping_a_disk = unassigned_cells_in_max_steps_square[cell_may_overlap_a_disk,:]
    if 'overlapped_by_all' == return_type:
        return cells_maybe_overlapping_a_disk
    
    # compare farthest vertex of cells: bool if cell is in contained in disk of radius centered around any pt within triangle 1 
    cell_is_contained_in_all_trgl_disks = [grid_spacing * max_possible_dist_trgl1_to_cell(cell) <= radius for cell in unassigned_cells_in_max_steps_square]
    cells_contained_in_all_trgl_disks = unassigned_cells_in_max_steps_square[cell_is_contained_in_all_trgl_disks]    
    
    if 'contained_by_trgl' == return_type:
        return cells_contained_in_all_trgl_disks
    
    # those cells should not be in cells maybe overlapping disk around pts in triangle as they are surely contain
    unassigned_cells_in_max_steps_square = unassigned_cells_in_max_steps_square[_np_invert(cell_is_contained_in_all_trgl_disks),:]
    # compare closest vertex of cells: bool if cell is potentially overlaped by a disk of radius around pt within triangle 1
    cell_may_overlap_trgl_disk = (
        [grid_spacing * min_possible_dist_trgl1_to_cell(cell) <= radius for cell in unassigned_cells_in_max_steps_square]
        ) if include_boundary else (
        [grid_spacing * min_possible_dist_trgl1_to_cell(cell) < radius for cell in unassigned_cells_in_max_steps_square]
        )
    cells_maybe_overlapping_a_trgl_disk = unassigned_cells_in_max_steps_square[cell_may_overlap_trgl_disk,:]

    if 'overlapped_by_trgl' == return_type:
        return cells_maybe_overlapping_a_trgl_disk
    
    return (
        cells_contained_in_all_disks, 
        cells_contained_in_all_trgl_disks, 
        cells_maybe_overlapping_a_disk, 
        cells_maybe_overlapping_a_trgl_disk
    )
#

# relevant cell types:
# cells always contained

def filter_cells_alw_contained(
        grid_spacing:float=250,
        radius:float=750,
        include_boundary:bool=False
    ) -> tuple:
    return get_cells_relevant_for_disk_by_type(
       grid_spacing=grid_spacing, radius=radius, include_boundary=include_boundary, return_type='contained_by_all'
    )
#

def filter_cells_alw_contained(
        grid_spacing:float=250,
        radius:float=750,
        include_boundary:bool=False
    ) -> tuple:
    return get_cells_relevant_for_disk_by_type(
       grid_spacing=grid_spacing, radius=radius, include_boundary=include_boundary, return_type='contained_by_all'
    )
#

def filter_cells_alw_overlapped(
        grid_spacing:float=250,
        radius:float=750,
        include_boundary:bool=False
    ) -> tuple:
    return get_cells_relevant_for_disk_by_type(
       grid_spacing=grid_spacing, radius=radius, include_boundary=include_boundary, return_type='overlapped_by_trgl'
    )
#

def filter_cells_alw_overlapped(
        grid_spacing:float=250,
        radius:float=750,
        include_boundary:bool=False
    ) -> tuple:
    return get_cells_relevant_for_disk_by_type(
       grid_spacing=grid_spacing, radius=radius, include_boundary=include_boundary, return_type='overlapped_by_trgl'
    )
#

def check_if_always_overlaps_convex_set(
        cells:_np_array,
        convex_set_vertices:_np_array = _np_array([[0,0],[.5,0],[.5,.5]]),
        vertex_is_inside_convex_set=True,
        radius:float=0.00750,
        grid_spacing:float=0.00250, 
        include_boundary:bool=False,
):
    if type(vertex_is_inside_convex_set)==bool:
        if vertex_is_inside_convex_set:
            vertex_is_inside_convex_set = _np_ones(len(convex_set_vertices), dtype=bool)
        else:
            vertex_is_inside_convex_set = _np_zeros(len(convex_set_vertices), dtype=bool)
    
    # smallest possible distance of cell to convex set
    dist_lower_bounds = grid_spacing * _np_array([
        min_dist_points_to_cell(convex_set_vertices,cell) 
        for cell in cells]
    )

    set_always_overlaps_cells = _np_all([
            (dist_lower_bounds[:,i] <= radius) if include_boundary and point_in_set or 0. in cell else (dist_lower_bounds[:,i] < radius) 
            for i,point_in_set, cell in zip(range(dist_lower_bounds.shape[1]), vertex_is_inside_convex_set, cells)
        ], axis=0)
    
    return set_always_overlaps_cells
#
  
def check_if_always_overlaps_full_convex_set(
        cells:_np_array,
        convex_set_vertices:_np_array = _np_array([[0,0],[.5,0],[.5,.5]]),
        vertex_is_inside_convex_set=True,
        radius:float=0.00750,
        grid_spacing:float=0.00250,
        include_boundary:bool=False,
):
    if type(vertex_is_inside_convex_set)==bool:
        if vertex_is_inside_convex_set:
            vertex_is_inside_convex_set = _np_ones(len(convex_set_vertices), dtype=bool)
        else:
            vertex_is_inside_convex_set = _np_zeros(len(convex_set_vertices), dtype=bool)
    
    always_overlaps_full_convex_set = [max([
        (pt_x-vtx_x)**2+(pt_y-vtx_y)**2 for (pt_x, pt_y), vtx_x, vtx_y in [
            (get_cell_closest_point_to_point((vtx_x, vtx_y), cell), vtx_x, vtx_y) for (vtx_x, vtx_y) in convex_set_vertices
        ]
        ]) <= (radius/grid_spacing)**2 
        for cell in cells]

    return always_overlaps_full_convex_set
#
  
def check_if_never_contains_convex_set(
        cells:_np_array,
        convex_set_vertices:_np_array = _np_array([[0,0],[.5,0],[.5,.5]]),
        vertex_is_inside_convex_set=True,
        radius:float=0.00750,
        grid_spacing:float=0.00250,
        include_boundary:bool=False,
):
    if type(vertex_is_inside_convex_set)==bool:
        if vertex_is_inside_convex_set:
            vertex_is_inside_convex_set = _np_ones(len(convex_set_vertices), dtype=bool)
        else:
            vertex_is_inside_convex_set = _np_zeros(len(convex_set_vertices), dtype=bool)
    
    # largest possible distance of cell to convex set
    dist_upper_bounds = grid_spacing * _np_array([
        max_dist_points_to_cell(convex_set_vertices,cell)
        for cell in cells]
    )

    set_maybe_contains_cells = _np_any(dist_upper_bounds <= radius, axis=1)
    set_maybe_contains_cells = _np_any([
            (dist_upper_bounds[:,i] <= radius) if include_boundary and point_in_set else (dist_upper_bounds[:,i] < radius) 
            for i,point_in_set in zip(range(dist_upper_bounds.shape[1]), vertex_is_inside_convex_set)
        ], axis=0)
    set_never_contains_cells = _np_invert(set_maybe_contains_cells)

    return set_never_contains_cells
#









































def get_always_contained_potentially_overlapped_cells(
        radius:float=750,
        grid_spacing:float=750,
        count_only:bool=True
    ) -> tuple:
    """
    TODO this method and get_cells_relevant_for_disk_by_type contain repeated logic. Only one shall be kept.
    For any point within cell in grid (at row=0,col=0) of spacing=grid_spacing=cell width=cell height:
    Determines which cells (relative to the cell where the point falls into) are:
    - contained within radius for all possible points in the cell
    - which cells might be overlaped, i.e. that are not fully contained or partly 
      overlappend for at least one BUT NOT all possible points in the cell
    If count == True returns only count of cells. If not True returns list of tuples of cell row col locations
    Note due to symmetry only 1/8 of radius needs to be checked 
    TODO wording needed to say home grid cell

    Args:
      radius (float):
        the radius of disk drawn around any point within the cell, in which other points are searched,
        this is used to check whether other grid cells are fully contained in that disk or overlaped
        by any possble point in the grid cell
      grid_spacing (float):
        the spacing of the grid defining the width and heigth of each grid cell
    
    Returns:
      cell_info (tuple):
        tuple containg:

          - overlap_count (int):
            Number of cells that are contained within disk around all point in cell
          - overlap_count (int):
            Number of cells that are overlap with disk around a point in cell but 
            not contained in disk around all points of cell
        
        and if not count_only additionally containing:
          - overlap_ids (list):
            relative position of cells that are contained within disk around all point in cell
          - overlap_ids (list):
            relative position of cells that are overlap with disk around a point in cell but 
            not contained in disk around all points of cell
          - cell_steps_max (int):
            maximum row (=column) distance of a cell to be potentially overlaped by a disk of
            radius around any point in cell 
     
     """

    ratio = radius/grid_spacing
    cell_steps_max = _math_ceil(ratio+1)
    overlap_count, overlap_count = 0,0

    if not count_only:
        overlap_ids, overlap_ids = [], []
    
    for i in range(cell_steps_max):
        for j in range(i+1):
            if ratio > (max(i-1,0)**2+max(j-1,0)**2)**.5:
                # account for symmetry. 
                # Center cell exist once. 
                # 90Deg Diagonals exist 4times. 
                # 45Deg Diagonals exist 4times.
                # Gaps exist 8times.  
                cellCountMultiplier = 1 if (i==0 and j==0) else 4 if (j==0 or i==j) else 8
                i_ids = list(set([-i,i]))
                j_ids = list(set([-j,j]))
                cell_ids = flatten_list([
                    flatten_list([
                        [(i_id,j_id)]+([(j_id,i_id)] if i!=j else []) for j_id in j_ids
                        ]) for i_id in i_ids
                    ])
                n_cells = len(cell_ids)
           
                if ratio >= ((1+i)**2+(1+j)**2)**.5:
                    if not count_only:
                        overlap_ids += cell_ids
                    overlap_count += cellCountMultiplier
                else:
                    if not count_only:
                        overlap_ids += cell_ids
                    overlap_count += cellCountMultiplier
    
    # multiply to get n for full circle not only quadrant
    # overlap_count = max((overlap_count-1)*8+1,0)
    # overlap_count = overlap_count*8

    if count_only:
        return overlap_count, overlap_count

    return overlap_count, overlap_count, sorted(overlap_ids), sorted(overlap_ids), cell_steps_max
#
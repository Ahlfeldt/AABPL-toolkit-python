from numpy import (
    array as _np_array, 
    unique, linspace, invert, flip, transpose, concatenate, sign, zeros, 
    min as _np_min, max as _np_max, equal, where, logical_or, logical_and, all, newaxis)

def apply_invalid_area_on_grid(
        
):
    return


def disk_cell_intersection_area(
    disk_center_pt:_np_array,      
    cell:float=0.0075,
    grid_spacing:float=0.0075,
    radius:float=0.0075,
    silent = False,
):
    """
    Returns intersection area. area if intesection (0,grid_spacing**2)
    Case for no intersection will be handled before (fully included or fully excluded).
    Case 1: two intersection points (more than half of square are within radius) - 3 vertices are within circle 
    Case 2: two intersection points (more than half of square are within radius) - 1 vertex is within circle
    Case 3: two intersection points (less than half of square within radius) - 0 vertices within circle (same row or col)
    Case 4: two intersection points (unclear wheter more or less than half) - 2 vertices within circle (same row or col)
    Case 5: four intersection points (more than half of circle is included) - 2 vertices within circle (same row or col)
    
    TODO: if grid_spacing/2 is greater than radius there will be weird instances 

    This can also be done already as a function of the point offset
    and is also symmetrical towards the triangle
    the intersection area only needs to be computed for those cases where excluded cells are intersected 

    """
    # case no intersection will be handled
    area = grid_spacing**2
    return area


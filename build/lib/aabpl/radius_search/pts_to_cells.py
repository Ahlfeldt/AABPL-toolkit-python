from numpy import (
    array as _np_array,
    append as _np_append,
    zeros as _np_zeros,
    unique as _np_unique, 
    logical_or as _np_logical_or, 
)
from pandas import (DataFrame as _pd_DataFrame, cut as _pd_cut, concat as _pd_concat) 
from aabpl.utils.general import arr_to_tpls
from aabpl.testing.test_performance import time_func_perf
# from aabpl.doc.docstrings import fixdocstring

################ assign_points_to_cells ######################################################################################
# @fixdocstring
@time_func_perf
def assign_points_to_cells(
    grid:dict,
    pts:_pd_DataFrame,
    y:str='lat',
    x:str='lon',
    row_name:str='id_y',
    col_name:str='id_x',
    silent:bool = False,
) -> _pd_DataFrame:
    """
    # TODO Move to class and Properly describe.
    # TODO it modifies pts AND grid?
    Modifies input pandas.DataFrame grid and pts: 
    sorts by 1) y coordinate and 2) by x coordinate

    Args:
    <y>
    
    Returns:
    gridcell_id_name: name to be appended in pts to indicate gridcell. If False then information will not be stored in pts 
    """
    # TO Do this might be significantly faster when looping through pts instead of through cells
    pts.sort_values([y, x], inplace=True)

    # . 
    row_ids = grid.row_ids
    col_ids = grid.col_ids
    # get vectors of row columns boundary values
    x_steps = grid.x_steps
    y_steps = grid.y_steps
    # store len and digits for index
    n_pts = len(pts)
    
    if not silent:
        print(
            'Aggregate Data from '+str(n_pts)+' points'+
            ' into '+str(len(y_steps))+'x'+str(len(x_steps))+
            '='+str(len(y_steps)*len(x_steps))+' cells.' 
        )

    # to do change to cut
    # for each row select relevant points, then refine selection with columns to obtain cells
    pts[row_name] = _pd_cut(
        x = pts[y],
        # bins = y_steps[::-1],
        # labels = row_ids[::-1],
        bins = y_steps,
        labels = row_ids,
        include_lowest = True
    ).astype(int)
    
    pts[col_name] = _pd_cut(
        x = pts[x],
        bins = x_steps,
        labels = col_ids,
        include_lowest = True
    ).astype(int)
    
    
    return pts[[row_name, col_name]]

@time_func_perf
def aggregate_point_data_to_cells(
    grid:dict,
    pts:_pd_DataFrame,
    c:list=['employment'],
    row_name:str='id_y',
    col_name:str='id_x',
    silent = False,
) -> _pd_DataFrame:
    """
    TODO
    """
    # initialize dicts for later lookups )
    sums_zero = _np_zeros(len(c),dtype=int)
    cells_containing_pts = arr_to_tpls(_np_unique(pts[[row_name, col_name]],axis=0),int)
    grid.id_to_pt_ids = {pt_row_col:_np_array([],dtype=int) for pt_row_col in cells_containing_pts}
    grid.id_to_sums = {pt_row_col:sums_zero for pt_row_col in cells_containing_pts}
    # grid.id_to_sums = {g_id:sums_zero for g_id in grid.ids} 
    grid.pt_id_to_row_col = {}
    
    # TODO this could be also done in batches of points belonging to a single cell
    for pt_id, pt_row_col,pt_vals in zip(
        pts.index, 
        arr_to_tpls(pts[[row_name, col_name]].values,int), 
        pts[c].values
        ):
        grid.id_to_pt_ids[pt_row_col] = _np_append(grid.id_to_pt_ids[pt_row_col], pt_id)
        grid.id_to_sums[pt_row_col] = grid.id_to_sums[pt_row_col]+pt_vals
        grid.pt_id_to_row_col[pt_id] = pt_row_col

        #
    #
    if not silent:
        print(
            'Points assigned to grid cell:'+
            str(len(pts.index) - _np_logical_or(pts[col_name]==-1, pts[row_name]==-1).sum())+
            '/'+str(len(pts.index))
        )
        print('sum in grid:', _np_array([s for s in grid.id_to_sums.values()]).sum(axis=0), 'sum in pts', pts[c].values.sum(axis=0))
    #
    return 
#
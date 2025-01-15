
from numpy import (
    array as _np_array,
    append as _np_append,
    arange,
    invert as _np_invert,
    where,
    sign,
    zeros as _np_zeros,
    transpose,
    unique, linspace, flip,  concatenate,
    min as _np_min, 
    max as _np_max, 
    equal, 
    logical_or as _np_logical_or, 
    logical_and, 
    all as _np_all, 
    newaxis,
    sort
)
from numpy.linalg import norm as _np_linalg_norm
from pandas import (
    DataFrame as _pd_DataFrame, 
    cut as _pd_cut,
) 
from math import ceil#,asin,acos
from aabpl.utils.general import ( flatten_list, visualize, depth, list_dict_keys )
from aabpl.illustrations.illustrate_point_to_cell_region_assignment import (illustrate_point_to_cell_region_assignment)
from aabpl.utils.distances_to_cell import (  get_cells_relevant_for_disk_by_type )

def create_nested_cell_structure(
    cell_dict:dict,
    nest_levels_remaining:int=5,
    max_pts_per_cell:int=10,
):
    """
    input cell contains at least one pt
    """
    

    pts_lat_lon = cell_dict['pts_lat_lon']
    # shrink cell bounds to minimal bounding box containing all its points
    if len(pts_lat_lon)==0:
        print(nest_levels_remaining, "NOLEN",cell_dict)
    lat_min, lon_min, lat_max, lon_max = pts_lat_lon[:,0].min(), pts_lat_lon[:,1].min(), pts_lat_lon[:,0].max(), pts_lat_lon[:,1].max()
    cell_dict['bounds'] = ((lat_min, lon_min), (lat_max, lon_max))

    # early return if
    if ( 
        # less pts then max_pts_per_cell specifies 
        len(cell_dict['pt_ids']) <= max_pts_per_cell or
        # maximum nest levels reached
        nest_levels_remaining == 0 or
        # if all pts are exactly in the same location  
        (lat_min==lat_max and lon_min==lon_max)
        ):
        # return unchanged (except for tightend bounds)
        return cell_dict 
    
    
    pt_ids = cell_dict['pt_ids']
    pts_vals = cell_dict['pts_vals']
    
    # find largest extension in dimensions and split in half s.t.
    # split along north-south border
    # split along west - east border 
    threshold_dimension = int(lat_max-lat_min < lon_max-lon_min)
    # find value that splits the pts in half
    threshold_upper_value = sorted(pts_lat_lon[:,threshold_dimension])[int(len(pts_lat_lon)/2)]
    if not (pts_lat_lon[:,threshold_dimension]<threshold_upper_value).any():
        # handle 
        threshold_lower_value = threshold_upper_value
        threshold_upper_value = _np_min(pts_lat_lon[:,threshold_dimension][pts_lat_lon[:,threshold_dimension]>threshold_upper_value])
    else: 
        threshold_lower_value = _np_max(pts_lat_lon[:,threshold_dimension][pts_lat_lon[:,threshold_dimension]<threshold_upper_value])
    # use this itermediate value to ensure to split pts even if threshold value is repeated accross pts
    threshold_value = (threshold_upper_value+threshold_lower_value)/2
  
    
    pt_is_below_threshold = pts_lat_lon[:,threshold_dimension] <= threshold_value

    
    bounds_below_threshold = (
        (lat_min, lon_min), 
        ((lat_max,threshold_lower_value)[threshold_dimension], (threshold_lower_value,lon_max)[threshold_dimension])
        )
    bounds_above_threshold = (
        ((lat_min,threshold_upper_value)[threshold_dimension], (threshold_upper_value, lon_min)[threshold_dimension]), 
        (lat_max, lon_max)
        )


    # append list of quadrants
    cell_dict['quadrants'] = [create_nested_cell_structure(subcell,nest_levels_remaining-1,max_pts_per_cell) for subcell in [
        {
            'sums': pts_vals[pt_is_below_threshold].sum(axis=0),
            'pt_ids': pt_ids[pt_is_below_threshold],
            'pts_vals': pts_vals[pt_is_below_threshold],
            'pts_lat_lon': pts_lat_lon[pt_is_below_threshold],
            'bounds':bounds_below_threshold,
        },
        {
            'sums': pts_vals[_np_invert(pt_is_below_threshold)].sum(axis=0),
            'pt_ids': pt_ids[_np_invert(pt_is_below_threshold)],
            'pts_vals': pts_vals[_np_invert(pt_is_below_threshold)],
            'pts_lat_lon': pts_lat_lon[_np_invert(pt_is_below_threshold)],
            'bounds':bounds_above_threshold,
        }
    ]]
    
    # remove keys that are not necesssary any longer
    del cell_dict['pt_ids']
    del cell_dict['pts_vals']
    del cell_dict['pts_lat_lon']

    return cell_dict
#

def create_nested_cell_structure_quadrants(
    cell_dict:dict,
    nest_levels_remaining:int=5,
    max_pts_per_cell:int=10,
):
    """
    TODO this is not currently used. 
    TODO it does not recursively call itself but another function. is that right? 
    input cell contains at least one pt
    """
    

    pts_lat_lon = cell_dict['pts_lat_lon']
    # shrink cell bounds to minimal bounding box containing all its points

    lat_min, lon_min, lat_max, lon_max = pts_lat_lon[:,0].min(), pts_lat_lon[:,1].min(), pts_lat_lon[:,0].max(), pts_lat_lon[:,1].max()
    cell_dict['bounds'] = ((lat_min, lon_min), (lat_max, lon_max))

    # early return if
    if ( 
        # less pts then max_pts_per_cell specifies 
        len(cell_dict['pt_ids']) <= max_pts_per_cell or
        # maximum nest levels reached
        nest_levels_remaining == 0 or
        # if all pts are exactly in the same location  
        (lat_min==lat_max and lon_min==lon_max)
        ):
        # return unchanged (except for tightend bounds)
        return cell_dict 
    
    
    pt_ids = cell_dict['pt_ids']
    pts_vals = cell_dict['pts_vals']
    
    quadrants_bounds = [
        # quadrant 1: north west
        [( (lat_min+lat_max)/2,             lon_min ), (             lat_max, (lon_min+lon_max)/2 )],
        # quadrant 2: north east
        [( (lat_min+lat_max)/2, (lon_min+lon_max)/2 ), (             lat_max,             lon_max )],
        # quadrant 3: south west
        [(             lat_min,             lon_min ), ( (lat_min+lat_max)/2, (lon_min+lon_max)/2 )],
        # quadrant 4: south east
        [(             lat_min, (lon_min+lon_max)/2 ), ( (lat_min+lat_max)/2,             lon_max )],
        ]
    
    quadrants_pt_is_in = [
            # ensure that that top row includes pts on its northern boundary
            (q_lat_min <= pts_lat_lon[:,0]) * (pts_lat_lon[:,0] < q_lat_max+int(q_nr+1 in (1,2))) * 
            # ensure that that right col includes pts on its eastern boundary
            (q_lon_min <= pts_lat_lon[:,1]) * (pts_lat_lon[:,1] < q_lon_max+int(q_nr+1 in (2,4)))   
          for q_nr, ((q_lat_min, q_lon_min), (q_lat_max, q_lon_max)) in enumerate(quadrants_bounds)]

    non_empty_quadrant_dict_list = [{
            'sums': pts_vals[pt_is_in_quad].sum(axis=0),
            'pt_ids': pt_ids[pt_is_in_quad],
            'pts_vals': pts_vals[pt_is_in_quad],
            'pts_lat_lon': pts_lat_lon[pt_is_in_quad],
            'bounds':quad_bounds,
        } for pt_is_in_quad, quad_bounds in  zip(quadrants_pt_is_in, quadrants_bounds) if any(pt_is_in_quad)]
    if len(non_empty_quadrant_dict_list)==0:
        print("ERROR!")
    # append list of quadrants
    cell_dict['quadrants'] = [create_nested_cell_structure(quadrant,nest_levels_remaining-1,max_pts_per_cell) for quadrant in non_empty_quadrant_dict_list]
    
    # remove keys that are not necesssary any longer
    del cell_dict['pt_ids']
    del cell_dict['pts_vals']
    del cell_dict['pts_lat_lon']

    return cell_dict
#

def aggregate_point_data_to_nested_cells(
    grid:dict,
    pts_df:_pd_DataFrame,
    sum_names:list=['employment'],
    y_coord_name:str='lat',
    x_coord_name:str='lon',
    cell_name:str='cell_id',
    row_name:str='id_y',
    col_name:str='id_x',
    max_nest_levels:int=5,
    max_pts_per_cell:int=10,
    silent = False,
) -> _pd_DataFrame:
    """
    Inputs:
    Modifies input pandas.DataFrame grid and pts_df: 
    - sorts by 1) y coordinate and 2) by x coordinate
    - 
    Returns 
    gridcell_id_name: name to be appended in pts_df to indicate gridcell. If False then information will not be stored in pts_df 
    """
    # TO Do this might be significantly faster when looping through pts_df instead of through cells
    pts_df.sort_values([y_coord_name, x_coord_name], inplace=True)

    # . 
    row_ids = grid.row_ids
    col_ids = grid.col_ids
    # get vectors of row columns boundary values
    y_steps=grid.y_steps
    x_steps=grid.x_steps
    # store len and digits for index
    id_y_mult = grid.id_y_mult
    len_pts_df = len(pts_df)

    # initialize dicts for later lookups

    grid.pt_id_to_row_col = {}
    nested_grid = {}
    grid_id_to_bounds = grid.id_to_bounds
    if not silent:
        print(
            'Aggregate Data from '+str(len_pts_df)+' points'+
            ' into '+str(len(y_steps))+'x'+str(len(x_steps))+
            '='+str(len(y_steps)*len(x_steps))+' cells for '+str(len(sum_names))+' indicator(s).' 
            )

   
    # to do change to cut
    # for each row select relevant points, then refine selection with columns to obtain cells
    pts_df[row_name]=_pd_cut(
        x=pts_df[y_coord_name],
        bins=y_steps,labels=row_ids,include_lowest=True).astype(int)
    
    pts_df[col_name]=_pd_cut(
        x=pts_df[x_coord_name],
        bins=x_steps,labels=col_ids,include_lowest=True).astype(int)
    
    pts_df[cell_name] = id_y_mult * pts_df[row_name] + pts_df[col_name]

    for pt_id, pt_cell_id, pt_row_id, pt_col_id, pt_vals, pt_lat_lon in zip(
        pts_df.index, 
        pts_df[cell_name], 
        pts_df[row_name],
        pts_df[col_name], 
        pts_df[sum_names].values,
        pts_df[[y_coord_name,x_coord_name]].values
        ):
        # grid.id_to_pt_ids[pt_cell_id] = _np_append(grid.id_to_pt_ids[pt_cell_id], pt_id)
        # grid.id_to_sums[pt_cell_id]=grid.id_to_sums[pt_cell_id]+pt_vals
        grid.pt_id_to_row_col[pt_id] = (pt_row_id, pt_col_id)
        
        # grid.pt_id_to_lat_lon[pt_id] = (pt_row_id, pt_col_id)

        # nested TODO use classes here.
        if not pt_cell_id in nested_grid:
            # initialize cell dict
            nested_grid[pt_cell_id] = {
            'sums':pt_vals,
            'pt_ids':[pt_id], 
            'pts_vals':[pt_vals],
            'pts_lat_lon':[pt_lat_lon],
            'bounds': grid_id_to_bounds[pt_cell_id],
        }
        else:
            cell_dict = nested_grid[pt_cell_id]
            cell_dict['sums'] = cell_dict['sums'] + pt_vals 
            cell_dict['pt_ids'].append(pt_id) 
            cell_dict['pts_vals'].append(pt_vals) 
            cell_dict['pts_lat_lon'].append(pt_lat_lon) 
        #
    #
    
   
    # create neste structure of grid cell 
    # set parameters
    for grid_id,cell_dict in nested_grid.items():

        # convert to arrays
        cell_dict['pt_ids'] = _np_array(cell_dict['pt_ids'])
        cell_dict['pts_vals'] = _np_array(cell_dict['pts_vals'])
        cell_dict['pts_lat_lon'] = _np_array(cell_dict['pts_lat_lon'])

        nested_grid[grid_id] = create_nested_cell_structure(
            cell_dict=cell_dict,
            nest_levels_remaining=max_nest_levels,
            max_pts_per_cell=max_pts_per_cell,
            )
        
    grid.nested = nested_grid

    if not silent:
        print(
            'Points assigned to grid cell:'+
            str(len(pts_df.index) - _np_logical_or(pts_df[col_name]==-1, pts_df[row_name]==-1).sum())+
            '/'+str(len(pts_df.index))
        )
    return 
#

######################## PERFORM_SERACH #####################


def nested_distance_checks_directional(
    cell_dict:dict,
        pt_lat:float,
        pt_lon:float,
        pt_is_north_of_cell:bool,
        pt_is_east_of_cell:bool,
        radius:float=0.0074,
        zeros:_np_array=_np_zeros(1,dtype=int),
):
    """
    
    """
    # cell_dict['bounds]
    (cell_lat_min, cell_lon_min), (cell_lat_max, cell_lon_max) = cell_dict['bounds']
    (cell_lat_min, cell_lat_max)[int(pt_is_north_of_cell)]
    

    # else point is in between cell bounds at leas in one dimension thus two points have to be checked
    farthest_vertex_in_disk = (
            (pt_lat-(cell_lat_min if pt_is_north_of_cell else cell_lat_max))**2 + 
            (pt_lon-(cell_lon_min if pt_is_east_of_cell  else cell_lon_max))**2
            )**.5 <= radius

    # cell contained in disk if
    if farthest_vertex_in_disk:
        return cell_dict['sums']
    
    closest_vertex_in_disk = (
            (pt_lat-(cell_lat_max if pt_is_north_of_cell else cell_lat_min))**2 + 
            (pt_lon-(cell_lon_max if pt_is_east_of_cell  else cell_lon_min))**2
            )**.5 <= radius

    # overlapped if
    if closest_vertex_in_disk:
        # if there are quadrants within cell search downwwards
        if 'quadrants' in cell_dict:
            return _np_array([nested_distance_checks_directional(
                    cell_dict=quadrant,
                    pt_lat=pt_lat,
                    pt_lon=pt_lon,
                    pt_is_north_of_cell=pt_is_north_of_cell,
                    pt_is_east_of_cell=pt_is_east_of_cell,
                    radius=radius,
                    zeros=zeros,
            ) for quadrant in cell_dict['quadrants']]).sum(axis=0)
        
        # else: There are not quadrants - thus deepest level has been reached   
        # check which pts are within disk and retrieve their vals and create sum
        return cell_dict['pts_vals'][_np_linalg_norm(cell_dict['pts_lat_lon']-_np_array([pt_lat,pt_lon]),axis=1)<=radius].sum(axis=0) 
    
    # cell does not intersects disk
    return zeros
#

def nested_distance_checks(
        cell_dict:dict,
            pt_lat:float,
            pt_lon:float,
            radius:float=0.0074,
            zeros:_np_array=_np_zeros(1,dtype=int),
):
    """
    
    """
    # directly evalute pairwise distance if at lowest nest level to avoid unnessary overhead
    if 'pts_vals' in cell_dict:
        return cell_dict['pts_vals'][_np_linalg_norm(cell_dict['pts_lat_lon']-_np_array([pt_lat,pt_lon]),axis=1)<=radius].sum(axis=0)


    # unpack for better readability
    (cell_lat_min, cell_lon_min), (cell_lat_max, cell_lon_max) = cell_dict['bounds']
        
    # check if pt not within cell lat or lon 'range'
    if (
        # pt is not (btwn_lat_min_max)
        not ((pt_lat <= cell_lat_min) != (pt_lat < cell_lat_max)) and 
        # pt is not (btwn_lon_min_max)
        not ((pt_lon <= cell_lon_min) != (pt_lon < cell_lon_max))
    ):
        # pt is not between max and min lat / lon values of cell
        # thus it is clear which vertex (north-west, north-east,south-east,south-west) is closest and which farthest from pt
        # the same vertex will also be closest for any possibly nested subcells
        
        return nested_distance_checks_directional(
                cell_dict=cell_dict,
                    pt_lat=pt_lat,
                    pt_lon=pt_lon,
                    # pt is north of cell lower if true else its south (as beeing in between is excluded)
                    pt_is_north_of_cell=pt_lat>cell_lat_min,
                    # pt is east of cell lower if true else its west (as beeing in between is excluded)
                    pt_is_east_of_cell=pt_lon>cell_lon_min,
                    radius=radius,
                    zeros=zeros,
            )
    
    return _np_array([nested_distance_checks(
                cell_dict=quadrant,
                    pt_lat=pt_lat,
                    pt_lon=pt_lon,
                    radius=radius,
                    zeros=zeros,
            ) for quadrant in cell_dict['quadrants']]).sum(axis=0) 


    # AVOID TROUGH EARLY RETURN 


    # else point is in between cell bounds at leas in one dimension thus two points have to be checked
    vertex_1_in_disk = ((pt_lat-cell_lat_min)**2+(pt_lon-cell_lon_min)**2)**.5 <= radius
    vertex_2_in_disk = ((pt_lat-cell_lat_max)**2+(pt_lon-cell_lon_min)**2)**.5 <= radius
    vertex_3_in_disk = ((pt_lat-cell_lat_min)**2+(pt_lon-cell_lon_max)**2)**.5 <= radius
    vertex_4_in_disk = ((pt_lat-cell_lat_max)**2+(pt_lon-cell_lon_max)**2)**.5 <= radius

    # if cell contained in disk return cell sum
    if _np_all([vertex_1_in_disk, vertex_2_in_disk, vertex_3_in_disk, vertex_4_in_disk]):
        return cell_dict['sums']
    
    # if overlapped
    if any([vertex_1_in_disk, vertex_2_in_disk, vertex_3_in_disk, vertex_4_in_disk]):
        
        # if no further sub quadrants check distance to pts
        # handle case where disk overlaps but no pts within it are within disk
        return _np_array([nested_distance_checks(
                cell_dict=quadrant,
                    pt_lat=pt_lat,
                    pt_lon=pt_lon,
                    radius=radius,
                    zeros=zeros,
            ) for quadrant in cell_dict['quadrants']]).sum(axis=0) 
    # cell does not intersects disk
    return zeros
#


def aggreagate_point_data_to_disks_vectorized_nested(
    grid:dict,
    pts_df:_pd_DataFrame,
    radius:float=0.0075,
    sum_names:list=['employment'],
    y_coord_name:str='lat',
    x_coord_name:str='lon',
    row_name:str='id_y',
    col_name:str='id_x',
    cell_name:str='cell_id',
    cell_region_name:str='cell_region',
    sum_suffix:str='_750m',
    exclude_pt_itself:bool=True,
    reset_sum_cols_to_zero:bool=True,
    silent = False,
):
    """
    
    """
    # unpack grid_data 
    grid_spacing = grid.spacing
    # grid_id_to_sums = grid.id_to_sums
    pt_id_to_grid_xy = grid.pt_id_to_row_col
    region_id_to_contained_cells = grid.search.region_id_to_contained_cells
    region_id_to_overlapped_cells = grid.search.region_id_to_overlapped_cells
    nested_grid = grid.nested
    id_y_mult=grid.id_y_mult
    cells_contained_in_all_disks =grid.search.cells_contained_in_all_disks

    # prepare dicts for fast lookup of values for point ids
 
    # initialize columns and/or reset to zero unless specified differently
    sum_radius_names = [(cname+sum_suffix) for cname in sum_names]
    pts_df[[cname for cname in sum_radius_names if reset_sum_cols_to_zero or not cname in pts_df.columns]] = 0
    
    # sort according to grid cell
    # also sort according to grid cell region!
    pts_df.sort_values([row_name, col_name, cell_region_name], inplace=True)
    store_index = pts_df.index 
    pts_df.reset_index(inplace=True,drop=True)
    
  
    sums_within_disks = []
    last_grid_id = -1
    last_cell_region_id = -1
    zero_sums = _np_zeros(len(sum_names),dtype=int)
    sparse_grid_ids = set(pts_df[cell_name])

    for pt_id, a_pt_ycoord, a_pt_xcoord, cell_id, cell_region_id in zip(
        store_index,
        pts_df[y_coord_name], 
        pts_df[x_coord_name],
        pts_df[cell_name],
        pts_df[cell_region_name],
        ):
        (grid_id_y, grid_id_x) = pt_id_to_grid_xy[pt_id]
        
        # as pts are sorted by grid cell update only if grid cell changed
        if not (grid_id_y, grid_id_x) == last_grid_id:
            
            contained_cell_ids = [cell_id for cell_id in (id_y_mult*(cells_contained_in_all_disks[:,0]+(grid_id_y))+(
                cells_contained_in_all_disks[:,1]+grid_id_x)) if cell_id in nested_grid] 
       
            contained_cells_sums = (_np_array([nested_grid[g_id]['sums'] for g_id in contained_cell_ids]).sum(axis=0) 
                                      if len(contained_cell_ids)>0 else zero_sums)
        #
            
        if not (grid_id_y, grid_id_x) == last_grid_id or last_cell_region_id != cell_region_id:
            # if cell changed or cell region changed
            cell_ids_full_in_cell_region = [cell_id for cell_id in (id_y_mult*(region_id_to_contained_cells[cell_region_id][:,0]+(grid_id_y))+(
                region_id_to_contained_cells[cell_region_id][:,1]+grid_id_x)) if cell_id in nested_grid]
                           
            contained_cells_sums_region = (_np_array([nested_grid[g_id]['sums'] for g_id in cell_ids_full_in_cell_region if g_id in nested_grid]).sum(axis=0) 
                                             if len(cell_ids_full_in_cell_region)>0 else zero_sums)

            cell_ids_maybe_in = [cell_id for cell_id in (id_y_mult*(region_id_to_overlapped_cells[cell_region_id][:,0]+(grid_id_y))+(
                region_id_to_overlapped_cells[cell_region_id][:,1]+grid_id_x)) if cell_id in nested_grid]
        #

        sums_from_overlapping_cells = [nested_distance_checks(
                cell_dict = nested_grid[cell_id],
                pt_lat = a_pt_ycoord,
                pt_lon = a_pt_xcoord,
                zeros = zero_sums
        ) for cell_id in cell_ids_maybe_in]

        overlapping_cells_sums = _np_array(sums_from_overlapping_cells).sum(axis=0) if len(sums_from_overlapping_cells)>0 else zero_sums

        sums_full_disk = contained_cells_sums + contained_cells_sums_region + overlapping_cells_sums
        
        # append result 
        sums_within_disks.append(sums_full_disk)
        
        # set id as last id for next iteration
        last_grid_id=(grid_id_y, grid_id_x)
        last_cell_region_id = cell_region_id
    #

    pts_df[sum_radius_names] = pts_df[sum_radius_names].values + sums_within_disks
            
    if exclude_pt_itself:
        # substract data from point itself unless specified otherwise
        pts_df[sum_radius_names] = pts_df[sum_radius_names].values-pts_df[sum_names]

    # restore index 
    pts_df.index = store_index

    return 
#
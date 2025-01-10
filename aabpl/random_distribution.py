from pandas import DataFrame as _pd_DataFrame
from numpy import (
    array as _np_array, arange as _np_arange, ndarray as _np_ndarray, vstack as _np_vstack, ones as _np_ones, percentile as _np_percentile, bool_ as _np_bool
)
from numpy.random import ( random as _np_random,  randint as _np_randint, seed as _np_seed, )
from aabpl.illustrations.distribution_plot import create_distribution_plot

def draw_random_points_within_valid_area(
  cell_centroid_xy_coords:_np_array,
  cell_inclusion_indicator:_np_array,
  cell_width:float,
  n_random_points:int=int(1e6),
  random_seed:float=None,
  cell_height:float=None,
  extra_share_of_pts_to_create:float = 0.02,
  fix_extra_pts_to_create:int = 1000,
)->_np_array:
    """
    Draw n random points within non-excluded region
    if grid is provided it will first draw a grid cell that is not excluded 
    then it will choose a random point within that grid cell
    if the grid cell is partly excluded and the randomly generated point falls 
    into the excluded area the point is discarded and a new cell is drawn 

    Args:
    partly_or_fully_included_cells (??):
        list cells with attributes (centroid coords, excluded_property)
    cell_width (float):
        width of cells
    n_random_points (int):
        number of random points to be drawn (default=1e6)
    random_seed (int):
        seed to make random draws replicable. TODO not yet implemented.
    cell_height (float):
        height of cells. (default=None, cell_height will be set equal to cell_width)
    Returns:
    random_points_coordinates (array):
        vector of coordinates (x,y) of randomly drawn points within included area. shape=(n_random_points, 2)
    random_points_cell_ids (array):
        vector cell ids where random points fall into. TODO not yet implemented.  
    """
    
    # SET RANDOM SEED IF ANY SUPPLIED AND ASSERT TYPE
    if type(random_seed)==int:
        _np_seed(random_seed)
    elif random_seed is not None:
        raise TypeError(
            "random_seed should be int if supplied, otherwise None (of type NoneType)."+
            "\nSeed suplied is of type "+str(type(random_seed))+
            ". Seed suplied:\n", random_seed
        )
    #
    
    # IF NOT SPECIFIED OTHERWISE CELL HEIGHT EQUAL CELL WIDTH
    if cell_height is None:
        cell_height = cell_width
    #
    
    # FILTER OUT FULLY INVALID CELLS TODO CHECK IF NECESSARY IN FINAL IMPLEMENTATION
    cell_centroid_xy_coords = _np_array([
        coords for inclusion, coords in zip(cell_inclusion_indicator, cell_centroid_xy_coords) if not (type(inclusion)==bool and inclusion==False)
    ])
    cell_inclusion_indicator = _np_array([
        inclusion for inclusion in cell_inclusion_indicator if not (type(inclusion)==bool and inclusion==False)
    ])

    # estimate the share of invalid area to draw additionally to create points (as some get discarded when they fall in invalid area)
    share_of_invalid_area_estimate = sum([0 if type(inclusion)==bool else 0.5 for inclusion in cell_inclusion_indicator])/len(cell_inclusion_indicator) 
    
    # if all points will be valid 
    # draw and immeditatley n random points in valid area
    if share_of_invalid_area_estimate == 0.0:
        return cell_centroid_xy_coords[
            _np_randint(
                0, len(cell_centroid_xy_coords),
                n_random_points
            )
        ] + (
            _np_random((n_random_points,2))-0.5
            ) * _np_array([cell_width, cell_height])
    

    # OTHERWISE CREATE POINTS AND DISCARD INVALID UNTIL ENOUGH POINTS ARE DRWAN IN VALID AREA
    random_points_coordinates = _np_ndarray(shape=(0,2))
    pts_attempted_to_create = 0
    while random_points_coordinates.shape[0] < n_random_points:
        # update estimation of share of invalid area for iterations after first
        # TODO THIS MIGHT NOT BE NECESSARY ONCE PERCENTAGE OF INVALID AREA IS KNOWN
        if pts_attempted_to_create > 0:
            # otherwise update guess for iterations after first
            share_of_invalid_area_estimate = len(random_points_coordinates)/pts_attempted_to_create
        
        # set number of additional points to create
        attempt_to_create_n_points = int(
            (1+share_of_invalid_area_estimate+extra_share_of_pts_to_create) * 
            (n_random_points-len(random_points_coordinates)) + 
            fix_extra_pts_to_create
        )

        # cells in which random points are drawn
        cell_nrs = _np_randint(
            0, len(cell_centroid_xy_coords),
            attempt_to_create_n_points
        )
        # randomly drawn points within not fully invalid cells
        new_random_point_coordinates = cell_centroid_xy_coords[cell_nrs]+(
            _np_random((attempt_to_create_n_points,2))-0.5)*_np_array([cell_width, cell_height])
        
        # filter out points in invalid area
        new_random_point_coordinates_in_valid_area =_np_array([
            coords for (inclusion, coords) 
            in zip(
                cell_inclusion_indicator[cell_nrs],
                new_random_point_coordinates
            ) if (
                (type(inclusion) in [bool, _np_bool] and inclusion == True) or 
                (not type(inclusion) in [bool, _np_bool] and not inclusion.contains(coords) )
            )
        ])

        # save valid random points
        random_points_coordinates = _np_vstack([random_points_coordinates, new_random_point_coordinates_in_valid_area])
        
        # update loop vars
        pts_attempted_to_create += attempt_to_create_n_points
    
    # return n_random_points coordinates
    return random_points_coordinates[:n_random_points]


def get_distribution_for_random_points(
    grid:dict,
    pts_df:_pd_DataFrame,
    radius:float=0.0075,
    n_random_points:int=int(1e6),
    k_th_percentiles:float=[99.5],
    sum_names:list=['employment'],
    x_coord_name:str='lon',
    y_coord_name:str='lat',
    row_name:str='id_y',
    col_name:str='id_x',
    cell_region_name:str='cell_region',
    sum_suffix:str='_750m',
    plot_distribution:dict={},
    random_seed:int=None,
    silent:bool=False,
    maybe_allow_for_some_random_vals_for_pts=None,
):
    """
    execute methods
    
    k_th_percentile: in [0,100] k-th percentile 

    1. draw n_random_points with draw_random_points_within_valid_area
    2. aggreagate_point_data_to_disks_vectorized
    TODO Check if how cluster value 
    """
    if type(k_th_percentiles) != list:
        k_th_percentiles = [k_th_percentiles for i in range(len(sum_names))]
    if any([k_th_percentile >= 100 or k_th_percentile <= 0 for k_th_percentile in k_th_percentiles]):
        raise ValueError(
            'Values for k_th_percentiles must be >0 and <100. Provided values do not fullfill that condition',
            set([k_th_percentile for k_th_percentile in k_th_percentiles if k_th_percentile >= 100 or k_th_percentile <= 0])
        )

    random_point_coords = draw_random_points_within_valid_area(
        cell_centroid_xy_coords=grid.centroids,
        cell_inclusion_indicator=_np_ones(len(grid.centroids),bool),
        cell_width=grid.spacing,
        n_random_points=n_random_points,
        random_seed=random_seed,
        cell_height=grid.spacing,
    )

    rndm_pts_df = _pd_DataFrame(
        data = random_point_coords,
        columns=[x_coord_name,y_coord_name]
    )

    grid.search.set_source(
        pts_df=rndm_pts_df,
        x_coord_name=x_coord_name,
        y_coord_name=y_coord_name,
        row_name=row_name,
        col_name=col_name,
        cell_region_name=cell_region_name,
        sum_suffix=sum_suffix,
        silent=silent,
    )

    grid.search.set_target(
        pts_df=pts_df,
        sum_names=sum_names,
        x_coord_name=x_coord_name,
        y_coord_name=y_coord_name,
        row_name=row_name,
        col_name=col_name,
        silent=silent,
    )
    
    grid.search.perform_search(silent=silent,)

    sum_radius_names = [(cname+sum_suffix) for cname in sum_names]
    disk_sums_for_random_points = rndm_pts_df[sum_radius_names].values

    cluster_threshold_values  = [_np_percentile(disk_sums_for_random_points[:,i], k_th_percentile,axis=0) for i, k_th_percentile in enumerate(k_th_percentiles)]
    
    if plot_distribution is not None:
        create_distribution_plot(
            sums_df=rndm_pts_df[sum_radius_names],
            cluster_threshold_values=cluster_threshold_values,
            k_th_percentiles=k_th_percentiles,
            radius=radius,
            plot_kwargs=plot_distribution
            )


    return (cluster_threshold_values, disk_sums_for_random_points)

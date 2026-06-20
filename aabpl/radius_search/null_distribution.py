from pandas import DataFrame as _pd_DataFrame
from numpy import (
    array as _np_array, column_stack as _np_column_stack, ndarray as _np_ndarray, vstack as _np_vstack, 
    ones as _np_ones, empty as _np_empty, percentile as _np_percentile, bool_ as _np_bool
)
from numpy.random import ( random as _np_random,  randint as _np_randint, seed as _np_seed, )
from shapely.geometry import Polygon as _shapely_Polygon, Point as _shapely_Point
from aabpl.utils.misc import flatten_list
from aabpl.testing.test_performance import time_func_perf

@time_func_perf
def draw_random_points_in_sample_area(
    grid:dict,
    cell_width:float,
    n_random_points:int=int(1e5),
    sample_area:_shapely_Polygon=None,
    cells_rndm_sample:dict=None,
    random_seed:float=None,
    cell_height:float=None,
    extra_share_of_pts_to_create:float = 0.01,
    fix_extra_pts_to_create:int = 1000,
)->_np_array:
    """
    Draw n random points within non-excluded region
    if grid is provided it will first draw a grid cell that is not excluded 
    then it will choose a random point within that grid cell
    if the grid cell is partly excluded and the randomly generated point falls 
    into the excluded area the point is discarded and a new cell is drawn 

    Args:
    -------
    partly_or_fully_included_cells (??):
        list cells with attributes (centroid coords, excluded_property)
    cell_width (float):
        width of cells
    n_random_points (int):
        number of random points to be drawn (default=1e5)
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
    if sample_area is None:
        sample_area = grid.sample_area

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
    
    
    # cells_fully_valid_ref = grid.cells_fully_valid_max_lvl
    cells_fully_valid_ref = grid.cells_fully_valid
    cells_partly_valid_ref = grid.cells_partly_valid_max_lvl
    # col_min = int((sample_area.bounds[0] - grid.total_bounds.xmin) // cell_width)
    # row_min = int((sample_area.bounds[1] - grid.total_bounds.ymin) // cell_height)
    # col_max = int((sample_area.bounds[2] - grid.total_bounds.xmin) // cell_width)
    # row_max = int((sample_area.bounds[3] - grid.total_bounds.ymin) // cell_height)
    # col_min = grid.sample_col_min
    # row_min = grid.sample_row_min
    # col_max = grid.sample_col_max
    # row_max = grid.sample_row_max
    # centroid_left_x = grid.total_bounds.xmin + grid._search_spacing / 2 
    # centroid_bottom_y = grid.total_bounds.ymin + grid._search_spacing / 2
    # centroid_left_x = grid.total_bounds.xmin
    # centroid_bottom_y = grid.total_bounds.ymin
    # grid.sample_grid_bounds = [
    #     grid.total_bounds.xmin + col_min * cell_width,
    #     grid.total_bounds.ymin + row_min * cell_height,
    #     grid.total_bounds.xmin + (col_max+1) * cell_width,
    #     grid.total_bounds.ymin + (row_max+1) * cell_height,
    # ]



    max_cells_fully_covered = max([
            sum([2**-(2*lvl) for lvl,(row, col) in cells_fully_valid_ref if lvl==lvl_i])
            for lvl_i in set([lvl for lvl, (row, col) in cells_fully_valid_ref])
        ])
    all_cells_eligible = sample_area is None or max_cells_fully_covered >= grid._search_n_cells 
    

    # update cells_rndm_sample with grid cells outside the grid

    max_lvl_partly = max([lvl for lvl,(row,col) in (cells_partly_valid_ref if len(cells_partly_valid_ref)>0 else cells_fully_valid_ref)])
    sample_cells_arr = _np_array(sorted([
        (lvl,row,col) for lvl,(row,col) in cells_fully_valid_ref.union(
            [(lvl,(row,col)) for lvl,(row,col) in cells_partly_valid_ref if lvl==max_lvl_partly]
        )
        ]))
    
    min_lvl = int(min(sample_cells_arr[:,0]))
    max_lvl = int(max(sample_cells_arr[:,0]))
    cum_count_start_by_lvl, cum_int_start_by_lvl, cum_int_stop_by_lvl = {min_lvl:0}, {min_lvl:0}, {}
    
    if all_cells_eligible:
        # no need to potenitally keep multiple levels of sample cells as all drawn pts are valid.
        # Thus keep only one (arbirary) level.
        sample_cells_arr = sample_cells_arr
        def rand_int_transformer(rand_ints:_np_ndarray)->_np_ndarray:
            return rand_ints
        rand_int_stop = int(sum(2**(max_lvl-sample_cells_arr[:,0])))
    else:
        n_sample_cells_by_lvl = {}
        for lvl in range(min_lvl, max_lvl+1):
            n_sample_cells_by_lvl[lvl] = sum(sample_cells_arr[:,0]==lvl)
            if lvl > min_lvl:
                # cum_int_start_by_lvl[lvl] = cum_int_start_by_lvl[lvl-1] + n_sample_cells_by_lvl[lvl-1]*(2**(2*(max_lvl-lvl)))
                cum_int_start_by_lvl[lvl] = cum_int_stop_by_lvl[lvl-1]
                cum_count_start_by_lvl[lvl] = cum_count_start_by_lvl[lvl-1] + n_sample_cells_by_lvl[lvl-1]
            cum_int_stop_by_lvl[lvl] = cum_int_start_by_lvl[lvl] + n_sample_cells_by_lvl[lvl]*(2**(2*(max_lvl-lvl)))
        rand_int_stop = cum_int_stop_by_lvl[max_lvl]
  
    def rand_int_transformer(rand_ints:_np_ndarray)->_np_ndarray:
        """Transform random integers in [0, rand_int_stop) to cell indices in sample_cells  
        """
        if min_lvl == max_lvl:
            return rand_ints
        transformed_rand_ints = _np_empty(len(rand_ints), int)
        for lvl in range(min_lvl, max_lvl+1):
            mask = (rand_ints >= cum_int_start_by_lvl[lvl]) & (rand_ints < cum_int_stop_by_lvl[lvl])
            # transformed_rand_ints[mask] = cum_int_start_by_lvl[lvl] + ((rand_ints[mask]-cum_int_start_by_lvl[lvl])//(2**(2*(max_lvl-lvl))))
            transformed_rand_ints[mask] = cum_count_start_by_lvl[lvl] + ((rand_ints[mask]-cum_int_start_by_lvl[lvl])//(2**(2*(max_lvl-lvl))))
            # rand_ints[mask] = cum_count_start_by_lvl[lvl] + ((rand_ints[mask]-cum_int_start_by_lvl[lvl])//(2**(2*(max_lvl-lvl))))
        return transformed_rand_ints
    
    grid.rand_int_transformer = rand_int_transformer
    cell_to_poly = grid.cell_to_poly if hasattr(grid, 'cell_to_poly') else {}
    
    grid_bbox = _shapely_Polygon([
            (grid.total_bounds.xmin,grid.total_bounds.ymin),
            (grid.total_bounds.xmax,grid.total_bounds.ymin),
            (grid.total_bounds.xmax,grid.total_bounds.ymax),
            (grid.total_bounds.xmin,grid.total_bounds.ymax),
    ])
    
    sample_area_contains_grid = grid_bbox.area == sample_area.intersection(grid_bbox).area
    # estimate the share of invalid area to draw additionally to create points (as some get discarded when they fall in invalid area)
    share_of_invalid_cells = .0 if all_cells_eligible else  sum(
        [2**(2*-lvl) for lvl, (row, col) in cells_partly_valid_ref]
        )/(
            sum([2**(2*-lvl) for lvl, (row, col) in cells_fully_valid_ref]) + 
            sum([2**(2*-lvl) for lvl, (row, col) in cells_partly_valid_ref])
        )
    share_of_invalid_geometry = sum(
        [cell_height * 2**(-lvl) * cell_width * 2**(-lvl) - cell_to_poly[lvl,(row,col)].area for lvl, (row, col) in cells_partly_valid_ref]
        ) / sample_area.area
    # make a guess upward biased guess how large the share of invalid random points may be. 
    share_of_invalid_area = 1 - (
        0.125 *  (1 - share_of_invalid_cells) * (1 - share_of_invalid_geometry) + 
        0.125 * (1 - share_of_invalid_cells) +
        0.75 * (1 - share_of_invalid_geometry)
    )
    # CREATE POINTS AND DISCARD POINTS UNTIL ENOUGH POINTS ARE DRWAN IN VALID AREA
    random_points_coordinates = _np_ndarray(shape=(0,2))
    pts_attempted_to_create = 0
    it = 0
    while random_points_coordinates.shape[0] < n_random_points:
        # update estimation of share of invalid area for iterations after first
        # TODO THIS MIGHT NOT BE NECESSARY ONCE PERCENTAGE OF INVALID AREA IS KNOWN
        if pts_attempted_to_create > 0:
            # otherwise update guess for iterations after first
            share_of_invalid_area = len(random_points_coordinates)/pts_attempted_to_create
        
        # set number of additional points to create
        n_rndm_points_to_create = int(
            (1+share_of_invalid_area+extra_share_of_pts_to_create*int(share_of_invalid_area>0)) * 
            (n_random_points-len(random_points_coordinates)) + 
            fix_extra_pts_to_create*(1+it)*int(share_of_invalid_area>0)
        )

        # larger cells (higher levels) shall have a higher chance to be drawn. 
        # Also it must be ensured that no subcell is part of sample_cells if parent cell is part of sample_cells
        # rndm_cells[:,:0:-1] gives col,row and leaves out level
        rand_ints = rand_int_transformer(_np_randint(0, rand_int_stop, n_rndm_points_to_create))
        rndm_cells = sample_cells_arr[rand_ints]
        # _np_array([sample_bounds_xmin, sample_bounds_ymin]) 
        new_random_point_coordinates = _np_array([grid.total_bounds.xmin, grid.total_bounds.ymin]) + grid._search_spacing * (
            _np_random((n_rndm_points_to_create,2)) * 
            (2**-rndm_cells[:,0].reshape(-1,1)) +
            rndm_cells[:,1:][:,::-1] # TOOD this part might not put the points into the right postion if lvl>0
        )#  rndm_cells[:,:0:-1]
        
        # if anywhere is valid area        
        if sample_area_contains_grid:
            new_random_point_coordinates_in_sample_area = new_random_point_coordinates
            # 
        else: # filter out points in invalid area
            # lookup which rndm cells are fully valid as checking whether sample_area.covers for all points is slow
            rndm_cells_fully_valid = _np_array([
                (int(lvl),(
                (int if lvl >= 0 else float)(row),
                (int if lvl >= 0 else float)(col)
                )) in cells_fully_valid_ref  for lvl,row,col in rndm_cells])
            
            new_random_point_coordinates_in_sample_area =_np_array(
                [
                coords for coords in new_random_point_coordinates[rndm_cells_fully_valid]
             ] +
            [
                coords for coords,(lvl,row,col) in zip(
                    new_random_point_coordinates[~rndm_cells_fully_valid],
                    rndm_cells[~rndm_cells_fully_valid]
                )
                if cell_to_poly.get(
                    (int(lvl),(
                        (int if lvl >= 0 else float)(row),
                        (int if lvl >= 0 else float)(col)
                    )), sample_area
                ).covers(_shapely_Point(coords)) 
            ] #+
            # [
            #     coords for coords in new_random_point_coordinates[~rndm_cells_fully_valid]
            #     if sample_area.covers(_shapely_Point(coords)) 
            # ]
            )
            grid.new_random_point_coordinates_partly_valid = new_random_point_coordinates[~rndm_cells_fully_valid]
            grid.rndm_cells_partly_valid = rndm_cells[~rndm_cells_fully_valid]
            
            #
        # save valid random points
        if len(new_random_point_coordinates_in_sample_area) > 0:
            random_points_coordinates = _np_vstack([random_points_coordinates, new_random_point_coordinates_in_sample_area])
        # update loop vars
        it += 1
        pts_attempted_to_create += n_rndm_points_to_create

    # return n_random_points coordinates
    return random_points_coordinates[:n_random_points]

@time_func_perf
def compute_null_distribution(
    grid:dict,
    pts:_pd_DataFrame,
    sample_area:_shapely_Polygon=None,
    min_pts_to_sample_cell:int=1,
    n_random_points:int=int(1e5),
    k_th_percentile:float=[99.5],
    c:list=[],
    x:str='lon',
    y:str='lat',
    row_name:str='id_y',
    col_name:str='id_x',
    suffix:str='_750m',
    random_seed:int=None,
    silent:bool=False,
):
    """Draws n_random_points within sample_area and aggregates data from points within search radius. 
    From those values it calculates the k_th_percentile threshold value for the variable(s). This 
    execute methods
    
    k_th_percentile: in [0,100] k-th percentile 

    1. draw n_random_points with draw_random_points_within_valid_area
    2. aggregate_point_data_to_disks_vectorized
    TODO Check if how cluster value 


    
    min_pts_to_sample_cell (int):
        minimum number of points in dataset that need to be in cell s.t. random points are allowed to be drawn within it. (default=1)
    """
    if type(k_th_percentile) != list:
        k_th_percentiles = [k_th_percentile for i in range(len(c))]
    else: 
        k_th_percentiles = k_th_percentile
    if any([k_th_percentile >= 100 or k_th_percentile <= 0 for k_th_percentile in k_th_percentiles]):
        raise ValueError(
            'Values for k_th_percentile must be >0 and <100. Provided values do not fullfill that condition',
            set([k_th_percentile for k_th_percentile in k_th_percentiles if k_th_percentile >= 100 or k_th_percentile <= 0])
        )
    grid.cells_rndm_sample = True if min_pts_to_sample_cell == 0 else set([(row,col) for (row,col),pts in grid.id_to_pt_ids.items() if len(pts)>=min_pts_to_sample_cell])
    grid.sample_area = sample_area

    random_point_coords = draw_random_points_in_sample_area(
        grid=grid,
        cell_width=grid._search_spacing,
        n_random_points=n_random_points,
        random_seed=random_seed,
        cell_height=grid._search_spacing,
    )

    rndm_pts = _pd_DataFrame(
        data = random_point_coords,
        columns=[x,y]
    )

    grid.search.set_source(
        pts=rndm_pts,
        c=c,
        x=x,
        y=y,
        row_name=row_name,
        col_name=col_name,
        suffix=suffix,
        silent=True,
    )

    grid.search.set_target(
        pts=pts,
        c=c,
        x=x,
        y=y,
        row_name=row_name,
        col_name=col_name,
        silent=silent,
    )
    
    grid.rndm_pts = rndm_pts
    
    grid.search.perform_search(silent=True,)

    sum_radius_names = [(cname+suffix) for cname in c]
    disk_sums_for_random_points = rndm_pts[sum_radius_names].values

    cluster_threshold_values  = [_np_percentile(disk_sums_for_random_points[:,i], k_th_percentile,axis=0) for i, k_th_percentile in enumerate(k_th_percentiles)]
    
  
    return (cluster_threshold_values, rndm_pts)

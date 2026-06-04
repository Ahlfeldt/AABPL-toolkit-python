from warnings import simplefilter 
from pandas.errors import PerformanceWarning as _pd_PerfromanceWarning
simplefilter(action="ignore", category=_pd_PerfromanceWarning)
simplefilter(action='ignore', category=FutureWarning)
from pandas import DataFrame as _pd_DataFrame
from numpy import array as _np_array, nan as _np_nan
from .random_distribution import get_distribution_for_random_points
from .valid_area import infer_sample_area_from_pts, remove_invalid_area_from_sample_poly, process_sample_poly_to_grid
from aabpl.testing.test_performance import time_func_perf
from aabpl.radius_search.radius_search_class import DiskSearch
from aabpl.radius_search.grid_class import Grid
from aabpl.illustrations.plot_pt_vars import create_plots_for_vars
from aabpl.illustrations.distribution_plot import create_distribution_plot
from aabpl.utils.general import count_polygon_edges,find_column_name
from aabpl.utils.crs_transformation import convert_MultiPolygon_crs, convert_coords_to_local_crs, convert_pts_to_crs, convert_wgs_to_utm
from shapely.geometry import (Polygon as _shapely_Polygon, MultiPolygon as _shapely_MultiPolygon)

def check_kwargs(
        pts:_pd_DataFrame,
        crs:str,
        sample_area_crs:str,
        r:float,
        c:list=[],
        aggregation_method:str='sum',
        x:str='lon',
        y:str='lat',
        row_name:str='id_y',
        col_name:str='id_x',
        sum_suffix:str=None,
        pts_target:_pd_DataFrame=None,
        x_tgt:str=None,
        y_tgt:str=None,
        row_name_tgt:str=None,
        col_name_tgt:str=None,
        grid:Grid=None,
        nest_depth:int=None,
        proj_crs:str='auto',
        silent:bool=None,
):
    """
    check shared keyword arguments and apply defaults
    """
    # locals() TODO use locals to make this take in only locals
    if type(row_name) != str:
        raise TypeError('`row_name` must be of type str. Instead provided of type',type(row_name),row_name)
    if type(col_name) != str:
        raise TypeError('`col_name` must be of type str. Instead provided of type',type(col_name),col_name)
    if row_name_tgt is None:
        row_name_tgt = row_name
    elif type(row_name_tgt) != str:
        raise TypeError('`row_name_tgt` must be of type str. Instead provided of type',type(row_name_tgt),row_name_tgt)
    if col_name_tgt is None:
        col_name_tgt = col_name
    elif type(col_name_tgt) != str:
        raise TypeError('`col_name_tgt` must be of type str. Instead provided of type',type(col_name_tgt),col_name_tgt)
    if type(pts) != _pd_DataFrame:
        raise TypeError('`pts` must be a pandas.DataFrame or None. Instead provided of type',type(pts))
    if type(x) != str:
        raise TypeError('`x` must be of type str. Instead provided of type',type(x),x)
    if type(y) != str:
        raise TypeError('`x` must be of type str. Instead provided of type',type(y),y)
    if not x in pts.columns:
        raise ValueError('`x` (x-coord column name) must be in columns of pts')
    if not y in pts.columns:
        raise ValueError('`y` (y-coord column name) must be in columns of pts')
    if not type(sum_suffix) is str:
        if not sum_suffix is None:
            sum_suffix = str(sum_suffix)
        else:
            r_suffix = int(r) if r%1==0 or len(str(int(r))) > 5 else round(r,6-len(str(int(r))))
            sum_suffix = '_' + str(r_suffix)+'m'
    if x_tgt is None:
        x_tgt = x
    if y_tgt is None:
        y_tgt = y
    same_target = pts_target is None or pts is pts_target
    if pts_target is None:
        pts_target = pts
    else:
        if type(pts_target) != _pd_DataFrame:
            raise TypeError('`pts_target` must be a pandas.DataFrame or None. Instead provided of type',type(pts_target))
    if type(c) == str:
        c = [c]
    else:
        if c is None or len(c)==0:
            print("Warning: No columns specified for aggregation - will simply count number of points within radius.")
            aggregation_method = 'count'
        try:
            if any([type(column)!=str for column in c]):
                raise TypeError
        except:
            raise TypeError('`c` must be either a string of single column name or a list of column name strings')
    if any([not column in pts_target.columns for column in c]):
        raise ValueError('not all columns(',c,') are in columns of search target pts_target(',pts.columns,')')
    if not x_tgt in pts_target.columns:
        raise ValueError('`x_tgt` (x-coord column name) must be in columns of pts_target')
    if not y_tgt in pts_target.columns:
        raise ValueError('`y_tgt` (y-coord column name) must be in columns of pts_target')
    if sample_area_crs is None:
        sample_area_crs = crs
    if proj_crs == 'auto': 
        x_center = (min([pts[x].min(), pts_target[x_tgt].min()])+max([pts[x].max(), pts_target[x_tgt].max()]))/2
        y_center = (min([pts[x].min(), pts_target[x_tgt].min()])+max([pts[x].max(), pts_target[x_tgt].max()]))/2
        local_crs = 'EPSG:'+str(convert_wgs_to_utm(x_center, y_center))
    else:
        local_crs = proj_crs
    if crs != local_crs:
        x,y,local_crs = convert_pts_to_crs(pts=pts, x=x, y=y, initial_crs=crs, target_crs=proj_crs)
        if not same_target:
            x_tgt,y_tgt,local_crs = convert_pts_to_crs(pts=pts_target, x=x_tgt, y=y_tgt, initial_crs=crs, target_crs=proj_crs)
        else:
            x_tgt,y_tgt = x,y
    
    # OVERWRITE DEFAULTS
    if grid is None:
        if not nest_depth is None and int(nest_depth) != nest_depth:
            raise TypeError('`nest_depth` must be either of type int or None. Instead:', nest_depth, "of type "+str(type(nest_depth))+' was provided.')
        elif not nest_depth is None:
            nest_depth = int(nest_depth)
        grid = create_auto_grid_for_radius_search(
            pts_source=pts,
            initial_crs=local_crs,
            local_crs=local_crs,
            data_crs=crs,
            r=r,
            nest_depth=nest_depth,
            x=x,
            y=y,
            pts_target=pts_target,
            x_tgt=x_tgt,
            y_tgt=y_tgt,
            silent=silent,
        )
    elif type(grid) != Grid:
        raise TypeError('`grid` must be either of type Grid or None. Instead:', grid, "of type "+str(type(grid))+' was provided.')


    return (pts, local_crs,  sample_area_crs, c, x, y, sum_suffix, pts_target, x_tgt, y_tgt, row_name_tgt, col_name_tgt, grid, aggregation_method)
#


@time_func_perf
def handle_sample_area_input(
    pts:_pd_DataFrame,
    r:float,
    sample_area='buffered_cells',
    sample_area_crs=None, 
    local_crs:str=None,
    x:str='lon',
    y:str='lat',
    grid:Grid=None,
    min_pts_to_sample_cell:int=0,
    no_plot:bool=True
):
    if type(sample_area)==bool and sample_area==False:
        return _shapely_Polygon([
            (grid.total_bounds.xmin, grid.total_bounds.ymin),
            (grid.total_bounds.xmax, grid.total_bounds.ymin),
            (grid.total_bounds.xmax, grid.total_bounds.ymax),
            (grid.total_bounds.xmin, grid.total_bounds.ymax)
        ])

    if sample_area is None:
        sample_area = 'grid'
    if type(sample_area) == str:
        if no_plot:
            print("Creating sample area with method '"+sample_area+"' and buffer=tolerance="+str(r)+". Use 'grid.sample_area' to inspect.")
        sample_area = infer_sample_area_from_pts(
            pts=pts,
            grid=grid,
            x=x,
            y=y,
            hull_type=sample_area,
            buffer=r,
            min_pts_to_sample_cell=min_pts_to_sample_cell,
            plot_sample_area=None,
        )
        # sample_area = remove_invalid_area_from_sample_poly(sample_area, invalid_areas=_shapely_Polygon([]))

    elif type(sample_area) in [_shapely_Polygon, _shapely_MultiPolygon]:
        sample_area = convert_MultiPolygon_crs(multipoly=sample_area, initial_crs=sample_area_crs,target_crs=local_crs)
    else:
        raise ValueError('sample_area must parameter most be one of ["str","Poylgon","MultiPolygon"] instead of type', type(sample_area))
    
    
    
    return sample_area

# TODO remove cell_region from kwargs
@time_func_perf
def create_auto_grid_for_radius_search(
    pts_source:_pd_DataFrame,
    initial_crs:str,
    local_crs:str,
    data_crs:str,
    r:float,
    nest_depth:int=None,
    x:str='lon',
    y:str='lat',
    pts_target:_pd_DataFrame=None,
    x_tgt:str=None,
    y_tgt:str=None,
    silent:bool=None,
):
    """
    Returns a Grid that covers all points and will 
    - can be used to represent clusters
    - and is leverage for performance gains of radius search 

    Args:
    -------
    pts_source (pandas.DataFrame):
        DataFrame of points for which a search for other points within the specified radius shall be performed
    crs (str):
        crs of coordinates, e.g. 'EPSG:4326'
    r (float):
        radius within which other points shall be found in meters 
    x (str):
        column name of x-coordinate (=longtitude) in pts_source (default='lon')
    y (str):
        column name of y-coordinate (=lattitude) in pts_source (default='lat')
    pts_target (pandas.DataFrame):
        DataFrame of points that shall be found/aggregated when searching within radius of points from pts_source. If None specified its assumed to be the same as pts_source. (default=None)
    x_tgt (str):
        column name of x-coordinate (=longtitude) in pts_target. If None its assumed to be same as x (default=None)
    y_tgt (str):
        column name of y-coordinate (=lattitude) in pts_target. If None its assumed to be same as y (default=None)
    silent (bool):
        Whether information on progress shall be printed to console (default=False)
    
    Returns:
    -------
    grid (aabl.Grid):
        a grid covering all points (custom class containing 
    """

    if pts_target is None:
        xmin = pts_source[x].min()
        xmax = pts_source[x].max()
        ymin = pts_source[y].min()
        ymax = pts_source[y].max()
    else:
        if y_tgt is None:
            y_tgt = y
        if x_tgt is None:
            x_tgt = x
        xmin = min([pts_source[x].min(), pts_target[x_tgt].min()])
        xmax = max([pts_source[x].max(), pts_target[x_tgt].max()])
        ymin = min([pts_source[y].min(), pts_target[y_tgt].min()])
        ymax = max([pts_source[y].max(), pts_target[y_tgt].max()])
    
    return Grid(
            xmin=xmin,
            xmax=xmax,
            ymin=ymin,
            ymax=ymax,
            initial_crs=initial_crs,
            local_crs=local_crs,
            set_fixed_spacing=r/3, # TODO don t set fixed spacing but
            r=r,
            n_pts_src=len(pts_source),
            n_pts_tgt=len(pts_target),
            nest_depth=nest_depth,
            data_crs=data_crs,
            silent=silent,
        )
#

@time_func_perf
def radius_search(
    pts:_pd_DataFrame,
    crs:str,
    r:float,
    c:list=[],
    exclude_pt_itself:bool=True,
    weight_valid_area:str=None,
    sample_area=False,
    sample_area_crs:str=None,
    include_boundary:bool=False,
    aggregation_method:str=['sum','count','mean'][0],
    x:str='lon',
    y:str='lat',
    row_name:str='id_y',
    col_name:str='id_x',
    sum_suffix:str='_r_sum', 
    pts_target:_pd_DataFrame=None,
    x_tgt:str=None,
    y_tgt:str=None,
    row_name_tgt:str=None,
    col_name_tgt:str=None,
    grid:Grid=None,
    nest_depth:int=None,
    proj_crs:str='auto',
    plot_pt_disk:dict=None,
    plot_cell_reg_assign:dict=None,
    plot_offset_checks:dict=None,
    plot_offset_regions:dict=None,
    plot_offset_raster:dict=None,
    silent:bool=None,
):
    """
    For all points in DataFrame it searches for all other points (potentially of another DataFrame) within the specified radius and aggregate the values for specified column(s)
    The result will be appended to DataFrame.

    Args:
    -------
    pts (pandas.DataFrame):
        DataFrame of points for which a search for other points within the specified radius shall be performed
    crs (str):
        crs of coordinates, e.g. 'EPSG:4326'
    r (float):
        radius within which other points shall be found in meters 
    c (str or list):
        column name or list of column name(s) in DataFrame for which data within search radius shall be aggregated. If None provided it will simply count the points within the radius. Column name must be in pts(DataFrame) unless a different search target is specified - then columns must exist in pts_target.
    exclude_pt_itself (bool):
        whether the sums within search radius point shall exlclude the point data itself (default=True)
    sample_area (shapely.geometry.Polygon | shapely.geometry.MultiPolygon | str):
        Specifies the area in which random points can be drawn.
        Either geometry is supplied directly (ensure that it uses the same projection in meters that the rest of the algorithm does).
        Or a string can be supplied that will be passed on into infer_sample_area_from_pts to create the sample area:
            - 'buffered_cells': each non-empty cell plus buffer around them
            - 'concave': a concave hull will be drawn around points. 
            - 'convex': a convex hull will be drawn around points.
            - 'buffer': points will only be drawn within buffer around points. WARNING: Very slow if many points in df.
            - 'bounding_box': a bounding box around points will be drawn
            - 'grid' or None: a box covering full grid will be drawn
        However using the function infer_sample_area_from_pts directly is recommended as it give you more control over its parameters.
        (default='buffered_cells')
    sample_area_crs (str):
        crs of the sample_area polygon. Ignored if sample_area param is not a (Multi-)Polygon. When None it will assume same crs as pts. (Default=None) 
    weight_valid_area (str):
        if set to 'estimate' or 'precise' the radius aggregate will be weighted inversely by the share of area of valid cells within search radius. 'precise' is very slow, 'estimate' has MSE of 5% of cell area. (default=None)
    x (str):
        column name of x-coordinate (=longtitude) in pts (default='lon')
    y (str):
        column name of y-coordinate (=lattitude) in pts (default='lat')
    row_name (str):
        name for column that will be appended to pts indicating grid cell row (default='id_y')
    col_name (str):
        name for column that will be appended to pts indicating grid cell column (default='id_x')
    sum_suffix (str):
        suffix used for new column(s) creating by aggregating data of columns , 
    pts_target (pandas.DataFrame):
        DataFrame of points that shall be found/aggregated when searching within radius of points from pts_source. If None specified its assumed to be the same as pts_source. (default=None)
    x_tgt (str):
        column name of x-coordinate (=longtitude) in pts_target. If None its assumed to be same as x (default=None)
    y_tgt (str):
        column name of y-coordinate (=lattitude) in pts_target. If None its assumed to be same as y (default=None)
    row_name_tgt (str):
        name for column that will be appended to pts indicating grid cell row. If None its assumed to be same as x (default=None)
    col_name_tgt (str):
        name for column that will be appended to pts indicating grid cell column (default='id_x')
    proj_crs (crs):
        crs projection into which pts for search (source and target) shall be mapped. If 'auto' local crs will be determined automatically. If None no reprojection will be performed (default='auto')
    grid (aabpl.Grid):
        grid of custom class containing points. If None it will automatically one (default=None)
    include_boundary (bool):
        FEATURE NOT YET IMPLEMENTED. Define whether points that are at the distance of exactly the radius shall be considered within (Default=False)
    plot_pt_disk (dict):
        Only needed for development. Dictionary with kwargs to create plot for example radius search. If None no plot will be created (default=None)
    plot_cell_reg_assign (dict):
        Only needed for development. Dictionary with kwargs to create plot for assginments of points to cell offset regions. If None no plot will be created (default=None)
    plot_offset_checks (dict):
        Only needed for development. Dictionary with kwargs to create plot for checks creating offset regions. If None no plot will be created (default=None)
    plot_offset_regions (dict):
        Only needed for development. Dictionary with kwargs to create plot for offset regions. If None no plot will be created (default=None)
    plot_offset_raster (dict):
        Only needed for development. Dictionary with kwargs to create plot for raster for offset regions. If None no plot will be created (default=None)
    silent (bool):
        Whether information on progress shall be printed to console (default=False)
    
    Returns:
    -------
    grid (aabl.Grid):
        a grid covering all points (custom class containing  
    
          
    Examples:
    -------
    from aabpl.main import radius_search
    from pandas import read_csv
    pts = read_csv('C:/path/to/file.txt',sep=',',header=None)
    pts.columns = ["eid", "employment", "industry", "lat","lon","moved"]
    grid = radius_search(pts,crs="EPSG:4326",r=750,columns=['employment'])
    grid.plot_vars(filename='employoment_750m')
    """
    init_sort = find_column_name('initial_sort', existing_columns=pts.columns)
    pts[init_sort] = range(len(pts))

    (pts, local_crs, sample_area_crs, c, x, y, sum_suffix, pts_target, x_tgt, y_tgt, row_name_tgt, col_name_tgt, grid, aggregation_method
     ) = check_kwargs(
            pts=pts, crs=crs, sample_area_crs=sample_area_crs, r=r, c=c, x=x, y=y, row_name=row_name,
            col_name=col_name, sum_suffix=sum_suffix, pts_target=pts_target, x_tgt=x_tgt, y_tgt=y_tgt,
            row_name_tgt=row_name_tgt, col_name_tgt=col_name_tgt, grid=grid, nest_depth=nest_depth, 
            proj_crs=proj_crs, silent=silent,
    )

    if aggregation_method in ['count','mean']:
        count_helper_col = find_column_name('count','_helper_col', existing_columns=pts_target.columns)
        c.append(count_helper_col)
        pts_target[count_helper_col] = 1
   
    # initialize disk_search
    grid.search = DiskSearch(
        grid=grid,
        r=r,
        nest_depth=nest_depth,
        exclude_pt_itself=exclude_pt_itself,
        weight_valid_area=weight_valid_area,
        include_boundary=include_boundary
    )
    # for i in range(10000):
    #     for j in range(40000):
    #         2243.8**.1324549

    # prepare target points data
    grid.search.set_target(
        pts=pts_target,
        c=c,
        x=x_tgt,
        y=y_tgt,
        row_name=row_name_tgt,
        col_name=col_name_tgt,
        silent=silent,
    )
    
    # prepare source points data
    grid.search.set_source(
        pts=pts,
        c=c,
        x=x,
        y=y,
        row_name=row_name,
        col_name=col_name,
        sum_suffix=sum_suffix,
        plot_cell_reg_assign=plot_cell_reg_assign,
        plot_offset_checks=plot_offset_checks,
        plot_offset_regions=plot_offset_regions,
        plot_offset_raster=plot_offset_raster,
        silent=silent,
    )
    
    # in case sums shall be weighted by sample area
    grid.sample_area = handle_sample_area_input(
        pts=pts,r=r,sample_area=sample_area,
        sample_area_crs=sample_area_crs,local_crs=local_crs,x=x,y=y,
        grid=grid, min_pts_to_sample_cell=0)
    process_sample_poly_to_grid(grid)
    disk_sums_for_pts = grid.search.perform_search(silent=False if silent is None else silent,plot_pt_disk=plot_pt_disk)
    
    if aggregation_method in ['mean']:
        n_rs = 1 # TODO for distance bands... later
        radius_count_cols = disk_sums_for_pts.columns[-n_rs:]
        radius_count_col = radius_count_cols[0] # TODO for distance bands... later
        for s_name in disk_sums_for_pts.columns[:-n_rs]:
            if s_name not in radius_count_cols:
                pts[s_name][pts[radius_count_col]>0] = pts[s_name][pts[radius_count_col]>0] / pts[radius_count_col][pts[radius_count_col]>0]  
                pts[s_name][pts[radius_count_col]==0] = _np_nan
        pts.drop(columns=[count_helper_col], inplace=True)
        # pts.drop(columns=[disk_sums_for_pts.columns[-1]], inplace=True)


    pts.sort_values(init_sort, inplace=True)
    pts.drop(columns=[init_sort], inplace=True)

    return grid
#

@time_func_perf
def detect_cluster_pts(
    pts:_pd_DataFrame,
    crs:str,
    r:float=0.0075,
    c:list=[],
    aggregation_method:str=['sum','count','mean'][0],
    exclude_pt_itself:bool=True,
    sample_area='buffered_cells',
    sample_area_crs:str=None,
    min_pts_to_sample_cell:int=0,
    weight_valid_area:str=None,
    k_th_percentile:float=99.5,
    n_random_points:int=int(1e5),
    random_seed:int=None,
    include_boundary:bool=False,
    x:str='lon',
    y:str='lat',
    row_name:str='id_y',
    col_name:str='id_x',
    sum_suffix:str='_750m',
    cluster_suffix:str='_cluster',
    proj_crs:str='auto',
    pts_target:_pd_DataFrame=None,
    x_tgt:str=None,
    y_tgt:str=None,
    row_name_tgt:str=None,
    col_name_tgt:str=None,
    grid:Grid=None,
    nest_depth:int=None,
    plot_distribution:dict=None,
    plot_cluster_points:dict=None,
    plot_pt_disk:dict=None,
    plot_cell_reg_assign:dict=None,
    plot_offset_checks:dict=None,
    plot_offset_regions:dict=None,
    plot_offset_raster:dict=None,
    silent:bool=None,
):
    """
    For all points in a DataFrame it searches for all other points (potentially of another DataFrame) within the specified radius and aggregate the values for specified column(s).
    It draws random the bounding box containing all points from DataFrame(s) and aggregate the values within the radius to obtain a random distribution.   
    Then all points from DataFrame which exceed the k_th_percentile of the random distribution are labeld as clustered.
    The results will be appended to DataFrame.

    Args:
    -------
    pts (pandas.DataFrame):
        DataFrame of points for which a search for other points within the specified radius shall be performed
    crs (str):
        crs of coordinates, e.g. 'EPSG:4326'
    r (float):
        radius within which other points shall be found in meters 
    c (str or list):
        column name or list of column name(s) in DataFrame for which data within search radius shall be aggregated. If None provided it will simply count the points within the radius. Column name must be in pts(DataFrame) unless a different search target is specified - then columns must exist in pts_target.
    exclude_pt_itself (bool):
        whether the sums within search radius point shall exlclude the point data itself (default=True)
    weight_valid_area (str):
        if set to 'estimate' or 'precise' the radius aggregate will be weighted inversely by the share of area of valid cells within search radius. 'precise' is very slow, 'estimate' has MSE of 5% of cell area. (default=None)
    sample_area (shapely.geometry.Polygon | shapely.geometry.MultiPolygon | str):
        Specifies the area in which random points can be drawn.
        Either geometry is supplied directly (ensure that it uses the same projection in meters that the rest of the algorithm does).
        Or a string can be supplied that will be passed on into infer_sample_area_from_pts to create the sample area:
            - 'buffered_cells': each non-empty cell plus buffer around them
            - 'concave': a concave hull will be drawn around points. 
            - 'convex': a convex hull will be drawn around points.
            - 'buffer': points will only be drawn within buffer around points. WARNING: Very slow if many points in df.
            - 'bounding_box': a bounding box around points will be drawn
            - 'grid' or None: a box covering full grid will be drawn
        However using the function infer_sample_area_from_pts directly is recommended as it give you more control over its parameters.
        (default='buffered_cells')
    sample_area_crs (str):
        crs of the sample_area polygon. Ignored if sample_area param is not a (Multi-)Polygon. When None it will assume same crs as pts. (Default=None) 
    min_pts_to_sample_cell (int):
        minimum number of points in dataset that need to be in cell s.t. random points are allowed to be drawn within it. (default=0)
    k_th_percentile (float):
        percentile of random distribution that a point needs to exceed to be classified as clustered.
    n_random_points (int):
        number of random points to be drawn to create random distribution (default=100000)
    random_seed (int):
        random seed to be applied when drawing random points to create random distribution. If None no seed will be set (default=None)
    x (str):
        column name of x-coordinate (=longtitude) in pts (default='lon')
    y (str):
        column name of y-coordinate (=lattitude) in pts (default='lat')
    row_name (str):
        name for column that will be appended to pts indicating grid cell row (default='id_y')
    col_name (str):
        name for column that will be appended to pts indicating grid cell column (default='id_x')
    sum_suffix (str):
        suffix used for new column(s) creating by aggregating data of columns , 
    pts_target (pandas.DataFrame):
        DataFrame of points that shall be found/aggregated when searching within radius of points from pts_source. If None specified its assumed to be the same as pts_source. (default=None)
    x_tgt (str):
        column name of x-coordinate (=longtitude) in pts_target. If None its assumed to be same as x (default=None)
    y_tgt (str):
        column name of y-coordinate (=lattitude) in pts_target. If None its assumed to be same as y (default=None)
    row_name_tgt (str):
        name for column that will be appended to pts indicating grid cell row. If None its assumed to be same as x (default=None)
    col_name_tgt (str):
        name for column that will be appended to pts indicating grid cell column (default='id_x')
    grid (aabpl.Grid):
        grid of custom class containing points. If None it will automatically one (default=None)
    include_boundary (bool):
        FEATURE NOT YET IMPLEMENTED. Define whether points that are at the distance of exactly the radius shall be considered within (Default=False)
    plot_distribution (dict):
        dictionary with kwargs to create plot for random distribution. If None no plot will be created (default=None)
    plot_pt_disk (dict):
        Only needed for development. Dictionary with kwargs to create plot for example radius search. If None no plot will be created (default=None)
    plot_cell_reg_assign (dict):
        Only needed for development. Dictionary with kwargs to create plot for assginments of points to cell offset regions. If None no plot will be created (default=None)
    plot_offset_checks (dict):
        Only needed for development. Dictionary with kwargs to create plot for checks creating offset regions. If None no plot will be created (default=None)
    plot_offset_regions (dict):
        Only needed for development. Dictionary with kwargs to create plot for offset regions. If None no plot will be created (default=None)
    plot_offset_raster (dict):
        Only needed for development. Dictionary with kwargs to create plot for raster for offset regions. If None no plot will be created (default=None)
    silent (bool):
        Whether information on progress shall be printed to console (default=False)
    
    Returns:
    -------
    grid (aabl.Grid):
        a grid covering all points (custom class) with cluster attributes stored to it
    """
    init_sort = find_column_name('initial_sort', existing_columns=pts.columns)
    pts[init_sort] = range(len(pts))

    (pts, local_crs, sample_area_crs, c, x, y, sum_suffix, pts_target, x_tgt, y_tgt, row_name_tgt, col_name_tgt, grid, aggregation_method
     ) = check_kwargs(
            pts=pts, crs=crs, sample_area_crs=sample_area_crs, r=r, c=c, aggregation_method=aggregation_method,
            x=x, y=y, row_name=row_name, col_name=col_name, sum_suffix=sum_suffix, 
            pts_target=pts_target, x_tgt=x_tgt, y_tgt=y_tgt, row_name_tgt=row_name_tgt, col_name_tgt=col_name_tgt, 
            grid=grid, nest_depth=nest_depth, proj_crs=proj_crs, silent=silent,
    )
    if type(k_th_percentile) not in [list,_np_array, tuple]:
        k_th_percentile = [k_th_percentile for column in c]
    elif len(k_th_percentile) < len(c):
        k_th_percentile = [k_th_percentile[i%len(k_th_percentile)] for i in range(len(c))]
    # initialize disk_search
    grid.search = DiskSearch(
        grid,
        r=r,
        nest_depth=nest_depth,
        exclude_pt_itself=exclude_pt_itself,
        weight_valid_area=weight_valid_area,
        include_boundary=include_boundary
    )

    grid.search.set_target(
        pts=pts_target,
        c=c,
        x=x_tgt,
        y=y_tgt,
        row_name=row_name_tgt,
        col_name=col_name_tgt,
        silent=silent,
    )

    #
    grid.sample_area = handle_sample_area_input(
        pts=pts, r=r, 
        sample_area=sample_area, sample_area_crs=sample_area_crs,local_crs=local_crs,x=x, y=y, grid=grid,
        min_pts_to_sample_cell=min_pts_to_sample_cell,
        no_plot=plot_distribution is None and plot_cluster_points is None)
    process_sample_poly_to_grid(grid=grid)
    
    (cluster_threshold_values, rndm_pts) = get_distribution_for_random_points(
        grid=grid,
        pts=pts,
        sample_area=grid.sample_area,
        min_pts_to_sample_cell=min_pts_to_sample_cell,
        c=c,
        x=x,
        y=y,
        row_name=row_name,
        col_name=col_name,
        sum_suffix=sum_suffix,
        n_random_points=n_random_points,
        k_th_percentile=k_th_percentile,
        random_seed=random_seed,
        silent=silent,
    )

    if not silent:
        for (colname, threshold_value, k_th_p) in zip(c, cluster_threshold_values,k_th_percentile):
            print("Threshold value for "+str(k_th_p)+"th-percentile is "+str(threshold_value)+" for "+str(colname)+" within "+str(r)+" meters.")
    

    grid.search.set_source(
        pts=pts,
        c=c,
        x=x,
        y=y,
        row_name=row_name,
        col_name=col_name,
        sum_suffix=sum_suffix,
        plot_cell_reg_assign=plot_cell_reg_assign,
        plot_offset_checks=plot_offset_checks,
        plot_offset_regions=plot_offset_regions,
        plot_offset_raster=plot_offset_raster,
        silent=silent,
    )


    disk_sums_for_pts = grid.search.perform_search(silent=silent,plot_pt_disk=plot_pt_disk)
    
    # save bool of whether pt is part of a cluster 
    for j, cname in enumerate(c):
        pts[str(cname)+str(cluster_suffix)] = disk_sums_for_pts.values[:,j]>cluster_threshold_values[j]


    if plot_distribution is not None:
        # print("disk_sums_for_random_points", disk_sums_for_random_points)
        create_distribution_plot(
            pts=pts,
            x=x,
            y=y,
            radius_sum_columns=[n+sum_suffix for n in c],
            grid=grid,
            rndm_pts=rndm_pts,
            cluster_threshold_values=cluster_threshold_values,
            k_th_percentile=k_th_percentile,
            r=r,
            plot_kwargs=plot_distribution
            )
    #

    def plot_rand_dist(
            filename:str="",
            pts=pts,
            x=x,
            y=y,
            radius_sum_columns=[n+sum_suffix for n in c],
            rndm_pts=rndm_pts,
            cluster_threshold_values=cluster_threshold_values,
            k_th_percentile=k_th_percentile,
            r=r,
            grid=grid,
            **plot_kwargs
            
    ):
        create_distribution_plot(
            filename=filename,
            plot_kwargs=plot_kwargs,
            pts=pts,
            x=x,
            y=y,
            radius_sum_columns=radius_sum_columns,
            grid=grid,
            rndm_pts=rndm_pts,
            cluster_threshold_values=cluster_threshold_values,
            k_th_percentile=k_th_percentile,
            r=r,
            )
    grid.plot.rand_dist = plot_rand_dist
    
    plot_colnames = list(c) + [n+sum_suffix for n in c] + [str(cname)+str(cluster_suffix) for cname in c]
    def plot_cluster_pts(
            self=grid,
            colnames=_np_array(plot_colnames),
            filename:str="",
            **plot_kwargs,
    ):
        return create_plots_for_vars(
            grid=self,
            colnames=colnames,
            filename=filename,
            plot_kwargs=plot_kwargs,
        )
    grid.plot.cluster_pts = plot_cluster_pts

    if plot_cluster_points is not None:
        grid.plot.cluster_pts(**plot_cluster_points)
        pass
    pts.sort_values(init_sort, inplace=True)
    pts.drop(columns=[init_sort], inplace=True)
    
    return grid
# done

def detect_cluster_cells(
    pts:_pd_DataFrame,
    crs:str,
    r:float=750,
    c:list=[],
    aggregation_method:str=['sum','count','mean'][0],
    exclude_pt_itself:bool=True,
    sample_area='buffered_cells',
    sample_area_crs:str=None,
    min_pts_to_sample_cell:int=0,
    weight_valid_area:str=None,
    k_th_percentile:float=99.5,
    n_random_points:int=int(1e5),
    random_seed:int=None,
    queen_contingency:int=1,
    rook_contingency:int=1,
    centroid_dist_threshold:float=None,
    border_dist_threshold:float=None,
    min_cluster_share_after_contingency:float=0.05,
    min_cluster_share_after_centroid_dist:float=0.00,
    min_cluster_share_after_convex:float=0.00,
    make_convex:bool=True,
    include_boundary:bool=False,
    x:str='lon',
    y:str='lat',
    row_name:str='id_y',
    col_name:str='id_x',
    sum_suffix:str='_750m',
    cluster_suffix:str='_cluster',
    proj_crs:str='auto',
    pts_target:_pd_DataFrame=None,
    x_tgt:str=None,
    y_tgt:str=None,
    row_name_tgt:str=None,
    col_name_tgt:str=None,
    grid:Grid=None,
    nest_depth:int=None,
    plot_distribution:dict=None,
    plot_cluster_points:dict=None,
    plot_pt_disk:dict=None,
    plot_cell_reg_assign:dict=None,
    plot_offset_checks:dict=None,
    plot_offset_regions:dict=None,
    plot_offset_raster:dict=None,
    silent:bool=None,
):
    """
    For all points in a DataFrame it searches for all other points (potentially of another DataFrame) within the specified radius and aggregate the values for specified column(s).
    It draws random the bounding box containing all points from DataFrame(s) and aggregate the values within the radius to obtain a random distribution.   
    Then all points from DataFrame which exceed the k_th_percentile of the random distribution are labeld as clustered.
    The results will be appended to DataFrame.

    Args:
    -------
    pts (pandas.DataFrame):
        DataFrame of points for which a search for other points within the specified radius shall be performed
    crs (str):
        crs of coordinates, e.g. 'EPSG:4326'
    r (float):
        radius within which other points shall be found in meters 
    c (str or list):
        column name or list of column name(s) in DataFrame for which data within search radius shall be aggregated. If None provided it will simply count the points within the radius. Column name must be in pts(DataFrame) unless a different search target is specified - then columns must exist in pts_target. 
    exclude_pt_itself (bool):
        whether the sums within search radius point shall exlclude the point data itself (default=True)
    weight_valid_area (str):
        if set to 'estimate' or 'precise' the radius aggregate will be weighted inversely by the share of area of valid cells within search radius. 'precise' is very slow, 'estimate' has MSE of 5% of cell area. (default=None)
    sample_area (shapely.geometry.Polygon | shapely.geometry.MultiPolygon | str):
        Specifies the area in which random points can be drawn.
        Either geometry is supplied directly (ensure that it uses the same projection in meters that the rest of the algorithm does).
        Or a string can be supplied that will be passed on into infer_sample_area_from_pts to create the sample area:
            - 'buffered_cells': each non-empty cell plus buffer around them
            - 'concave': a concave hull will be drawn around points. 
            - 'convex': a convex hull will be drawn around points.
            - 'bounding_box': a bounding box around points will be drawn
            - 'grid' or None: a box covering full grid will be drawn
        However using the function infer_sample_area_from_pts directly is recommended as it give you more control over its parameters.
        (default='buffered_cells')
    sample_area_crs (str):
        crs of the sample_area polygon. Ignored if sample_area param is not a (Multi-)Polygon. When None it will assume same crs as pts. (Default=None) 
    min_pts_to_sample_cell (int):
        minimum number of points in dataset that need to be in cell s.t. random points are allowed to be drawn within it. (default=0)
    k_th_percentile (float):
        percentile of random distribution that a point needs to exceed to be classified as clustered.
    n_random_points (int):
        number of random points to be drawn to create random distribution (default=100000)
    random_seed (int):
        random seed to be applied when drawing random points to create random distribution. If None no seed will be set (default=None)
    queen_contingency (int):
        if contigent (vertical, horizontal, diagonal) cells that are also classified as clustered shall be part of the same cluster. If set to a value>=2 then it also adds non-contingent cells that are that many steps away to the same cluster. (default=1) 
    rook_contingency (int):
        if contigent (vertical, horizontal) cells that are also classified as clustered shall be part of the same cluster. Ignored if queen_contingency is set to a higher value. If set to a value>=2 then it also adds non-contingent cells that are that many steps away to the same cluster. (default=1) 
    centroid_dist_threshold (float):
        maximum distance between centroids of clusters to be merged into a single cluster. If None clusters won't be merged based on centroid distance. (default=r*10/3)
    border_dist_threshold (float):
        maximum distance between borders of clusters to be merged into a single cluster. If None clusters won't be merged based on boundary distance (default=r*4/3)
    min_cluster_share_after_contingency (float):
        minimum share of cluster of total to not be dropped after cells are merged to clusters based on contingency
    min_cluster_share_after_centroid_dist (float):
        minimum share of cluster of total to not be dropped after clusters are merged based on centroid
    min_cluster_share_after_convex (float):
        minimum share of cluster of total to not be dropped after clusters are made convex by adding cells within its convex hull
    make_convex (bool):
        Whether all cells within the convex hull of a cluster shall be added to it (default=True)
    include_boundary (bool):
        FEATURE NOT YET IMPLEMENTED. Define whether points that are at the distance of exactly the radius shall be considered within (Default=False)
    x (str):
        column name of x-coordinate (=longtitude) in pts (default='lon')
    y (str):
        column name of y-coordinate (=lattitude) in pts (default='lat')
    row_name (str):
        name for column that will be appended to pts indicating grid cell row (default='id_y')
    col_name (str):
        name for column that will be appended to pts indicating grid cell column (default='id_x')
    sum_suffix (str):
        suffix used for new column(s) creating by aggregating data of columns , 
    pts_target (pandas.DataFrame):
        DataFrame of points that shall be found/aggregated when searching within radius of points from pts_source. If None specified its assumed to be the same as pts_source. (default=None)
    x_tgt (str):
        column name of x-coordinate (=longtitude) in pts_target. If None its assumed to be same as x (default=None)
    y_tgt (str):
        column name of y-coordinate (=lattitude) in pts_target. If None its assumed to be same as y (default=None)
    row_name_tgt (str):
        name for column that will be appended to pts indicating grid cell row. If None its assumed to be same as x (default=None)
    col_name_tgt (str):
        name for column that will be appended to pts indicating grid cell column (default='id_x')
    grid (aabpl.Grid):
        grid of custom class containing points. If None it will automatically one (default=None)
    plot_distribution (dict):
        dictionary with kwargs to create plot for random distribution. If None no plot will be created (default=None)
    plot_pt_disk (dict):
        Only needed for development. Dictionary with kwargs to create plot for example radius search. If None no plot will be created (default=None)
    plot_cell_reg_assign (dict):
        Only needed for development. Dictionary with kwargs to create plot for assginments of points to cell offset regions. If None no plot will be created (default=None)
    plot_offset_checks (dict):
        Only needed for development. Dictionary with kwargs to create plot for checks creating offset regions. If None no plot will be created (default=None)
    plot_offset_regions (dict):
        Only needed for development. Dictionary with kwargs to create plot for offset regions. If None no plot will be created (default=None)
    plot_offset_raster (dict):
        Only needed for development. Dictionary with kwargs to create plot for raster for offset regions. If None no plot will be created (default=None)
    silent (bool):
        Whether information on progress shall be printed to console (default=False)
    
    Returns:
    -------
    grid (aabl.Grid):
        a grid covering all points (custom class) with cluster attributes stored to it  
    """
    (pts, local_crs, sample_area_crs, c, x, y, sum_suffix, pts_target, x_tgt, y_tgt, row_name_tgt, col_name_tgt, grid, aggregation_method
     ) = check_kwargs(
            pts=pts, crs=crs, sample_area_crs=sample_area_crs, r=r, c=c, aggregation_method=aggregation_method,
            x=x, y=y, row_name=row_name, col_name=col_name, sum_suffix=sum_suffix, pts_target=pts_target, x_tgt=x_tgt, y_tgt=y_tgt,
            row_name_tgt=row_name_tgt, col_name_tgt=col_name_tgt, grid=grid, nest_depth=nest_depth, 
            proj_crs=proj_crs, silent=silent,
    )
    if centroid_dist_threshold is None:
        centroid_dist_threshold = r * 10/3
    if border_dist_threshold is None:
        border_dist_threshold = r * 4/3
    
    grid = detect_cluster_pts(
        pts=pts,
        crs=local_crs,
        r=r,
        c=c,
        exclude_pt_itself=exclude_pt_itself,
        weight_valid_area=weight_valid_area,
        sample_area=sample_area,
        sample_area_crs=sample_area_crs,
        min_pts_to_sample_cell=min_pts_to_sample_cell,
        k_th_percentile=k_th_percentile,
        n_random_points=n_random_points,
        random_seed=random_seed,
        grid=grid,
        nest_depth=nest_depth,
        x=x,
        y=y,
        row_name=row_name,
        col_name=col_name,
        sum_suffix=sum_suffix,
        cluster_suffix=cluster_suffix,
        proj_crs=local_crs,
        pts_target=pts_target,
        x_tgt=x_tgt,
        y_tgt=y_tgt,
        row_name_tgt=row_name_tgt,
        col_name_tgt=col_name_tgt,
        include_boundary=include_boundary,
        plot_distribution=plot_distribution,
        plot_cluster_points=plot_cluster_points,
        plot_pt_disk=plot_pt_disk,
        plot_cell_reg_assign=plot_cell_reg_assign,
        plot_offset_checks=plot_offset_checks,
        plot_offset_regions=plot_offset_regions,
        plot_offset_raster=plot_offset_raster,
        silent=silent,
    )
    
    grid.clustering.create_clusters(
        pts=pts,
        c=c,
        queen_contingency=queen_contingency,
        rook_contingency=rook_contingency,
        centroid_dist_threshold=centroid_dist_threshold,
        border_dist_threshold=border_dist_threshold,
        min_cluster_share_after_contingency=min_cluster_share_after_contingency,
        min_cluster_share_after_centroid_dist=min_cluster_share_after_centroid_dist,
        min_cluster_share_after_convex=min_cluster_share_after_convex,
        make_convex=make_convex,
        row_name=row_name,
        col_name=col_name,
        cluster_suffix=cluster_suffix,
        )
    
    return grid
#
def detect_cluster_cells_from_labeled_pts():
    pass    
def radius_sum():
    pass
def radius_count():
    pass
def radius_mean():
    pass

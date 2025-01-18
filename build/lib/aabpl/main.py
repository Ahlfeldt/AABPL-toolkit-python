from pandas import DataFrame as _pd_DataFrame
from numpy import array as _np_array
import math
from pyproj import Transformer
from .random_distribution import get_distribution_for_random_points
from aabpl.testing.test_performance import time_func_perf
from aabpl.radius_search.radius_search_class import DiskSearch
from aabpl.radius_search.grid_class import Grid
from aabpl.illustrations.plot_pt_vars import create_plots_for_vars
from aabpl.illustrations.distribution_plot import create_distribution_plot

def convert_coords_to_local_crs(
        pts,
        x:str='lon',
        y:str='lat',
        proj_x:str='proj_lon',
        proj_y:str='proj_lat',
        initial_crs:str="EPSG:4326",
        target_crs:str='auto',
        silent:bool=False,
) -> str:
    
    # https://gis.stackexchange.com/a/269552
    
    if target_crs == 'auto': 
        
        # convert_wgs_to_utm function, see https://stackoverflow.com/a/40140326/4556479
        def convert_wgs_to_utm(lon: float, lat: float):
            """Based on lat and lng, return best utm epsg-code"""
            utm_band = str((math.floor((lon + 180) / 6 ) % 60) + 1)
            if len(utm_band) == 1:
                utm_band = '0'+utm_band
            if lat >= 0:
                epsg_code = '326' + utm_band
                return epsg_code
            epsg_code = '327' + utm_band
            return epsg_code
            
        # Get best UTM Zone SRID/EPSG Code for center coordinate pair
        utm_code = convert_wgs_to_utm(*pts[[x,y]].mean(axis=0))
        # define project_from_to with iteration
        # see https://gis.stackexchange.com/a/127432/33092
        local_crs = 'EPSG:'+str(utm_code)
    else:
        local_crs = target_crs
    transformer = Transformer.from_crs(crs_from=initial_crs, crs_to=local_crs, always_xy=True)
    pts[proj_x],pts[proj_y] = transformer.transform(pts[x], pts[y])
    if not silent and initial_crs != local_crs:
        print("Reproject from " +str(initial_crs)+' to '+local_crs)
    return local_crs

def ensure_local_crs(
    pts:_pd_DataFrame=None,
    x:str='lon',
    y:str='lat',
    initial_crs:str='EPSG:4326', 
    target_crs:str='auto',
    silent:bool=True,
):
        
    proj_x = next(('proj_x'+str(i) for i in ['']+list(range(len(pts.columns))) if 'proj_x'+str(i) not in pts.columns))
    proj_y = next(('proj_y'+str(i) for i in ['']+list(range(len(pts.columns))) if 'proj_y'+str(i) not in pts.columns))
    if not target_crs is None:
        local_crs = convert_coords_to_local_crs(pts=pts, initial_crs=initial_crs, target_crs=target_crs, x=x, y=y, proj_x=proj_x, proj_y=proj_y,silent=silent)
        if local_crs == initial_crs:
            pts.drop(columns=[proj_x, proj_y], inplace=True)
        else:
            x = proj_x
            y = proj_y
        return x,y,local_crs
    return x,y,initial_crs


# TODO remove cell_region from kwargs
@time_func_perf
def create_auto_grid_for_radius_search(
    pts_source:_pd_DataFrame,
    initial_crs:str,
    local_crs:str,
    r:float,
    x:str='lon',
    y:str='lat',
    pts_target:_pd_DataFrame=None,
    x_tgt:str=None,
    y_tgt:str=None,
    silent:bool=True,
):
    """
    Returns a Grid that covers all points and will 
    - can be used to represent clusters
    - and is leverage for performance gains of radius search 

    Args:
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
            silent=silent,
        )
#

@time_func_perf
def radius_search(
    pts:_pd_DataFrame,
    crs:str,
    r:float,
    columns:list=[],
    exclude_pt_itself:bool=True,
    x:str='lon',
    y:str='lat',
    row_name:str='id_y',
    col_name:str='id_x',
    sum_suffix:str='_r_sum', 
    pts_target:_pd_DataFrame=None,
    x_tgt:str=None,
    y_tgt:str=None,
    tgt_row_name:str=None,
    tgt_col_name:str=None,
    grid:Grid=None,
    proj_crs:str='auto',
    include_boundary:bool=False,
    plot_radius_sums:dict=None,
    plot_pt_disk:dict=None,
    plot_cell_reg_assign:dict=None,
    plot_offset_checks:dict=None,
    plot_offset_regions:dict=None,
    plot_offset_raster:dict=None,
    silent:bool=True,
):
    """
    For all points in DataFrame it searches for all other points (potentially of another DataFrame) within the specified radius and aggregate the values for specified column(s)
    The result will be appended to DataFrame.

    Args:
    pts (pandas.DataFrame):
        DataFrame of points for which a search for other points within the specified radius shall be performed
    crs (str):
        crs of coordinates, e.g. 'EPSG:4326'
    r (float):
        radius within which other points shall be found in meters 
    columns (list or str):
        column(s) in DataFrame for which data within search radius shall be aggregated. If None provided it will simply count the points within the radius. 
    exclude_pt_itself (bool):
        whether the sums within search radius point shall exlclude the point data itself (default=True)
    x (str):
        column name of x-coordinate (=longtitude) in pts (default='lon')
    y (str):
        column name of y-coordinate (=lattitude) in pts (default='lat')
    row_name (str):
        name for column that will be appended to pts indicating grid cell row (default='id_y')
    col_name (str):
        name for column that will be appended to pts indicating grid cell column (default='id_y')
    sum_suffix (str):
        suffix used for new column(s) creating by aggregating data of columns , 
    pts_target (pandas.DataFrame):
        DataFrame of points that shall be found/aggregated when searching within radius of points from pts_source. If None specified its assumed to be the same as pts_source. (default=None)
    x_tgt (str):
        column name of x-coordinate (=longtitude) in pts_target. If None its assumed to be same as x (default=None)
    y_tgt (str):
        column name of y-coordinate (=lattitude) in pts_target. If None its assumed to be same as y (default=None)
    tgt_row_name (str):
        name for column that will be appended to pts indicating grid cell row. If None its assumed to be same as x (default=None)
    col_name (str):
        name for column that will be appended to pts indicating grid cell column (default='id_y')
    proj_crs (crs):
        crs projection into which pts for search (source and target) shall be mapped. If 'auto' local crs will be determined automatically. If None no reprojection will be performed (default='auto')
    grid (aabpl.Grid):
        grid of custom class containing points. If None it will automatically one (default=None)
    include_boundary (bool):
        FEATURE NOT YET IMPLEMENTED. Define whether points that are at the distance of exactly the radius shall be considered within (Default=False)
    plot_radius_sums (dict):
        dictionary with kwargs to create plot for radius sums. If None no plot will be created (default=None)
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
    grid (aabl.Grid):
        a grid covering all points (custom class containing  
    """


    x,y,local_crs = ensure_local_crs(pts=pts, x=x, y=y, initial_crs=crs, target_crs=proj_crs)
    if pts_target is None:
        pts_target = pts
    else:
        x,y,local_crs = ensure_local_crs(pts=pts_target, x=x, y=y, initial_crs=crs, target_crs=proj_crs)
    if x_tgt is None:
        x_tgt = x
    if y_tgt is None:
        y_tgt = y
    
    # OVERWRITE DEFAULTS
    if grid is None:
        grid = create_auto_grid_for_radius_search(
            pts_source=pts,
            initial_crs=crs,
            local_crs=local_crs,
            r=r,
            x=x,
            y=y,
            pts_target=pts_target,
            x_tgt=x_tgt,
            y_tgt=y_tgt,
            silent=silent,
        )
    
    if tgt_row_name is None:
        tgt_row_name = row_name
    if tgt_col_name is None:
        tgt_col_name = col_name
    help_col = None
    if columns is None or len(columns)==0:
        help_col = [next(('helper_col'+str(i) for i in (['']+list(range(len(pts_target.columns)))) if not 'helper_col'+str(i) in pts_target.columns))]
        pts_target[help_col] = 1
        columns = [help_col]

    # initialize disk_search
    grid.search = DiskSearch(
        grid=grid,
        r=r,
        exclude_pt_itself=exclude_pt_itself,
        include_boundary=include_boundary
    )
    

    # prepare target points data
    grid.search.set_target(
        pts=pts_target,
        columns=columns,
        x=x_tgt,
        y=y_tgt,
        row_name=tgt_row_name,
        col_name=tgt_col_name,
        silent=silent,
    )

    # prepare source points data
    grid.search.set_source(
        pts=pts,
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
    
    disk_sums_for_pts = grid.search.perform_search(silent=silent,plot_radius_sums=plot_radius_sums,plot_pt_disk=plot_pt_disk)
    if help_col is not None:
        pts_target.drop(columns=[help_col], inplace=True)
    return grid
#

@time_func_perf
def detect_cluster_pts(
    pts:_pd_DataFrame,
    crs:str,
    r:float=0.0075,
    columns:list=[],
    exclude_pt_itself:bool=True,
    k_th_percentiles:float=[99.5],
    n_random_points:int=int(1e5),
    random_seed:int=None,
    x:str='lon',
    y:str='lat',
    row_name:str='id_y',
    col_name:str='id_x',
    sum_suffix:str='_750m',
    cluster_suffix:str='_cluster',
    proj_crs:str='auto',
    grid:Grid=None,
    include_boundary:bool=False,
    plot_distribution:dict=None,
    plot_radius_sums:dict=None,
    plot_cluster_points:dict=None,
    plot_pt_disk:dict=None,
    plot_cell_reg_assign:dict=None,
    plot_offset_checks:dict=None,
    plot_offset_regions:dict=None,
    plot_offset_raster:dict=None,
    silent:bool=True,
):
    """
    For all points in a DataFrame it searches for all other points (potentially of another DataFrame) within the specified radius and aggregate the values for specified column(s).
    It draws random the bounding box containing all points from DataFrame(s) and aggregate the values within the radius to obtain a random distribution.   
    Then all points from DataFrame which exceed the k_th_percentile of the random distribution are labeld as clustered.
    The results will be appended to DataFrame.

    Args:
    pts (pandas.DataFrame):
        DataFrame of points for which a search for other points within the specified radius shall be performed
    crs (str):
        crs of coordinates, e.g. 'EPSG:4326'
    r (float):
        radius within which other points shall be found in meters 
    columns (list or str):
        column(s) in DataFrame for which data within search radius shall be aggregated. If None provided it will simply count the points within the radius. 
    exclude_pt_itself (bool):
        whether the sums within search radius point shall exlclude the point data itself (default=True)
    k_th_percentiles (float):
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
        name for column that will be appended to pts indicating grid cell column (default='id_y')
    sum_suffix (str):
        suffix used for new column(s) creating by aggregating data of columns , 
    pts_target (pandas.DataFrame):
        DataFrame of points that shall be found/aggregated when searching within radius of points from pts_source. If None specified its assumed to be the same as pts_source. (default=None)
    x_tgt (str):
        column name of x-coordinate (=longtitude) in pts_target. If None its assumed to be same as x (default=None)
    y_tgt (str):
        column name of y-coordinate (=lattitude) in pts_target. If None its assumed to be same as y (default=None)
    tgt_row_name (str):
        name for column that will be appended to pts indicating grid cell row. If None its assumed to be same as x (default=None)
    col_name (str):
        name for column that will be appended to pts indicating grid cell column (default='id_y')
    grid (aabpl.Grid):
        grid of custom class containing points. If None it will automatically one (default=None)
    include_boundary (bool):
        FEATURE NOT YET IMPLEMENTED. Define whether points that are at the distance of exactly the radius shall be considered within (Default=False)
    plot_distribution (dict):
        dictionary with kwargs to create plot for random distribution. If None no plot will be created (default=None)
    plot_radius_sums (dict):
        dictionary with kwargs to create plot for radius sums. If None no plot will be created (default=None)
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
    grid (aabl.Grid):
        a grid covering all points (custom class) with cluster attributes stored to it  
    """
    # OVERWRITE DEFAULTS
    
    x,y,local_crs = ensure_local_crs(pts=pts, x=x, y=y, initial_crs=crs, target_crs=proj_crs)
    if grid is None:
        grid = create_auto_grid_for_radius_search(
            pts_source=pts,
            initial_crs=crs,
            local_crs=local_crs,
            r=r,
            y=y,
            x=x,
        )
    # initialize disk_search
    grid.search = DiskSearch(
        grid,
        r=r,
        exclude_pt_itself=exclude_pt_itself,
        include_boundary=include_boundary
    )

    grid.search.set_target(
        pts=pts,
        columns=columns,
        x=x,
        y=y,
        row_name=row_name,
        col_name=col_name,
        silent=silent,
    )

    (cluster_threshold_values, rndm_pts) = get_distribution_for_random_points(
        grid=grid,
        pts=pts,
        columns=columns,
        x=x,
        y=y,
        row_name=row_name,
        col_name=col_name,
        sum_suffix=sum_suffix,
        n_random_points=n_random_points,
        k_th_percentiles=k_th_percentiles,
        plot_distribution=plot_distribution,
        random_seed=random_seed,
        silent=silent,
    )

    if not silent:
        for (colname, threshold_value, k_th_percentile) in zip(columns, cluster_threshold_values,k_th_percentiles):
            print("Threshold value for "+str(k_th_percentile)+"th-percentile is "+str(threshold_value)+" for "+str(colname)+".")
    
    grid.search.set_source(
        pts=pts,
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


    disk_sums_for_pts = grid.search.perform_search(silent=silent,plot_radius_sums=plot_radius_sums,plot_pt_disk=plot_pt_disk)
    
    # save bool of whether pt is part of a cluster 
    pts[
        [str(cname)+str(cluster_suffix) for cname in columns]
    ] = disk_sums_for_pts>cluster_threshold_values


    if plot_distribution is not None:
        # print("disk_sums_for_random_points", disk_sums_for_random_points)
        create_distribution_plot(
            pts=pts,
            x=x,
            y=y,
            radius_sum_columns=[n+sum_suffix for n in columns],
            rndm_pts=rndm_pts,
            cluster_threshold_values=cluster_threshold_values,
            k_th_percentiles=k_th_percentiles,
            r=r,
            plot_kwargs=plot_distribution
            )
    #

    def plot_rand_dist(
            pts=pts,
            x=x,
            y=y,
            radius_sum_columns=[n+sum_suffix for n in columns],
            rndm_pts=rndm_pts,
            cluster_threshold_values=cluster_threshold_values,
            k_th_percentiles=k_th_percentiles,
            r=r,
            filename:str="",
            **plot_kwargs
            
    ):
        create_distribution_plot(
            pts=pts,
            x=x,
            y=y,
            radius_sum_columns=radius_sum_columns,
            rndm_pts=rndm_pts,
            cluster_threshold_values=cluster_threshold_values,
            k_th_percentiles=k_th_percentiles,
            r=r,
            filename=filename,
            plot_kwargs=plot_kwargs
            )
    grid.plot_rand_dist = plot_rand_dist
    
    plot_colnames = list(columns) + [n+sum_suffix for n in columns] + [str(cname)+str(cluster_suffix) for cname in columns]
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
    grid.plot_cluster_pts = plot_cluster_pts

    if plot_cluster_points is not None:
        
        create_plots_for_vars(
            grid=grid,
            colnames=_np_array(plot_colnames),
            plot_kwargs=plot_cluster_points,
        )
        print('create plot for cluster points')
        pass
    
    return grid
# done

def detect_cluster_cells(
    pts:_pd_DataFrame,
    crs:str,
    r:float=750,
    columns:list=[],
    exclude_pt_itself:bool=True,
    k_th_percentiles:float=[99.5],
    n_random_points:int=int(1e5),
    random_seed:int=None,
    distance_thresholds:float=2500,
    make_convex:bool=True,
    x:str='lon',
    y:str='lat',
    row_name:str='id_y',
    col_name:str='id_x',
    sum_suffix:str='_750m',
    cluster_suffix:str='_cluster',
    proj_crs:str='auto',
    grid:Grid=None,
    include_boundary:bool=False,
    plot_distribution:dict=None,
    plot_radius_sums:dict=None,
    plot_cluster_points:dict=None,
    plot_pt_disk:dict=None,
    plot_cell_reg_assign:dict=None,
    plot_offset_checks:dict=None,
    plot_offset_regions:dict=None,
    plot_offset_raster:dict=None,
    plot_cluster_cells:dict=None,
    silent:bool=True,
):
    """
    For all points in a DataFrame it searches for all other points (potentially of another DataFrame) within the specified radius and aggregate the values for specified column(s).
    It draws random the bounding box containing all points from DataFrame(s) and aggregate the values within the radius to obtain a random distribution.   
    Then all points from DataFrame which exceed the k_th_percentile of the random distribution are labeld as clustered.
    The results will be appended to DataFrame.

    Args:
    pts (pandas.DataFrame):
        DataFrame of points for which a search for other points within the specified radius shall be performed
    crs (str):
        crs of coordinates, e.g. 'EPSG:4326'
    r (float):
        radius within which other points shall be found in meters 
    columns (list or str):
        column(s) in DataFrame for which data within search radius shall be aggregated. If None provided it will simply count the points within the radius. 
    exclude_pt_itself (bool):
        whether the sums within search radius point shall exlclude the point data itself (default=True)
    k_th_percentiles (float):
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
        name for column that will be appended to pts indicating grid cell column (default='id_y')
    sum_suffix (str):
        suffix used for new column(s) creating by aggregating data of columns , 
    pts_target (pandas.DataFrame):
        DataFrame of points that shall be found/aggregated when searching within radius of points from pts_source. If None specified its assumed to be the same as pts_source. (default=None)
    x_tgt (str):
        column name of x-coordinate (=longtitude) in pts_target. If None its assumed to be same as x (default=None)
    y_tgt (str):
        column name of y-coordinate (=lattitude) in pts_target. If None its assumed to be same as y (default=None)
    tgt_row_name (str):
        name for column that will be appended to pts indicating grid cell row. If None its assumed to be same as x (default=None)
    col_name (str):
        name for column that will be appended to pts indicating grid cell column (default='id_y')
    grid (aabpl.Grid):
        grid of custom class containing points. If None it will automatically one (default=None)
    include_boundary (bool):
        FEATURE NOT YET IMPLEMENTED. Define whether points that are at the distance of exactly the radius shall be considered within (Default=False)
    plot_distribution (dict):
        dictionary with kwargs to create plot for random distribution. If None no plot will be created (default=None)
    plot_radius_sums (dict):
        dictionary with kwargs to create plot for radius sums. If None no plot will be created (default=None)
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
    grid (aabl.Grid):
        a grid covering all points (custom class) with cluster attributes stored to it  
    """
    grid = detect_cluster_pts(
        pts=pts,
        crs=crs,
        r=r,
        columns=columns,
        exclude_pt_itself=exclude_pt_itself,
        k_th_percentiles=k_th_percentiles,
        n_random_points=n_random_points,
        random_seed=random_seed,
        grid=grid,
        x=x,
        y=y,
        row_name=row_name,
        col_name=col_name,
        sum_suffix=sum_suffix,
        cluster_suffix=cluster_suffix,
        include_boundary=include_boundary,
        plot_distribution=plot_distribution,
        plot_radius_sums=plot_radius_sums,
        plot_cluster_points=plot_cluster_points,
        plot_pt_disk=plot_pt_disk,
        plot_cell_reg_assign=plot_cell_reg_assign,
        plot_offset_checks=plot_offset_checks,
        plot_offset_regions=plot_offset_regions,
        plot_offset_raster=plot_offset_raster,
        silent=silent,
    )

    grid.create_clusters(
        pts=pts,
        columns=columns,
        distance_thresholds=distance_thresholds,
        make_convex=make_convex,
        row_name=row_name,
        col_name=col_name,
        cluster_suffix=cluster_suffix,
        plot_cluster_cells=plot_cluster_cells,
        )
    
    return grid

# next thing would be to label cells as clustered or not
# then to create orthogonal convex hull around clusters
# then to maybe wrap everything in one final function  
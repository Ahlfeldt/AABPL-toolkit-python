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

@time_func_perf
def create_auto_grid_for_radius_search(
    pts_df_source:_pd_DataFrame,
    crs:str,
    radius:float,
    pts_df_target:_pd_DataFrame=None,
    x_coord_name:str='lon',
    y_coord_name:str='lat',
    tgt_x_coord_name:str=None,
    tgt_y_coord_name:str=None,
    silent:bool=True,
):
    """
    automatially choose 
    """

    if pts_df_target is None:
        xmin = pts_df_source[x_coord_name].min()
        xmax = pts_df_source[x_coord_name].max()
        ymin = pts_df_source[y_coord_name].min()
        ymax = pts_df_source[y_coord_name].max()
    else:
        if tgt_y_coord_name is None:
            tgt_y_coord_name = y_coord_name
        if tgt_x_coord_name is None:
            tgt_x_coord_name = x_coord_name
        xmin = min([pts_df_source[x_coord_name].min(), pts_df_target[tgt_x_coord_name].min()])
        xmax = max([pts_df_source[x_coord_name].max(), pts_df_target[tgt_x_coord_name].max()])
        ymin = min([pts_df_source[y_coord_name].min(), pts_df_target[tgt_y_coord_name].min()])
        ymax = max([pts_df_source[y_coord_name].max(), pts_df_target[tgt_y_coord_name].max()])

    return Grid(
            xmin=xmin,
            xmax=xmax,
            ymin=ymin,
            ymax=ymax,
            crs=crs,
            set_fixed_spacing=radius/3, # TODO don t set fixed spacing but
            silent=silent,
        )
#

@time_func_perf
def radius_search(
    pts_df:_pd_DataFrame,
    crs:str,
    radius:float,

    include_boundary:bool=False,
    exclude_pt_itself:bool=True,

    pts_df_target:_pd_DataFrame=None,
    relation_src_tgt:str=['equal','subset','superset','none'][0],

    grid=None,
    
    x_coord_name:str='lon',
    y_coord_name:str='lat',
    row_name:str='id_y',
    col_name:str='id_x',
    cell_region_name:str='cell_region',
    sum_suffix:str='_750m', 

    sum_names:list=['employment'],

    tgt_x_coord_name:str=None,
    tgt_y_coord_name:str=None,
    tgt_row_name:str=None,
    tgt_col_name:str=None,

    plot_radius_sums:dict=None,
    plot_pt_disk:dict=None,
    plot_cell_reg_assign:dict=None,
    plot_offset_checks:dict=None,
    plot_offset_regions:dict=None,
    plot_offset_raster:dict=None,
    silent:bool=True,
):
    """
    execute methods
    1. pts_df data -> aggreagate_point_data_to_disks_vectorized
    2. create columns to check whether points are within cluster depending on the various parameters
    """
    # OVERWRITE DEFAULTS
    if grid is None:
        grid = create_auto_grid_for_radius_search(
            pts_df_source=pts_df,
            crs=crs,
            radius=radius,
            pts_df_target=pts_df_target,
            x_coord_name=x_coord_name,
            y_coord_name=y_coord_name,
            tgt_x_coord_name=tgt_x_coord_name,
            tgt_y_coord_name=tgt_y_coord_name,
            silent=silent,
        )
    if pts_df_target is None:
        pts_df_target = pts_df
    if tgt_x_coord_name is None:
        tgt_x_coord_name = x_coord_name
    if tgt_y_coord_name is None:
        tgt_y_coord_name = y_coord_name
    if tgt_row_name is None:
        tgt_row_name = row_name
    if tgt_col_name is None:
        tgt_col_name = col_name


    # initialize disk_search
    grid.search = DiskSearch(
        grid=grid,
        radius=radius,
        exclude_pt_itself=exclude_pt_itself,
        include_boundary=include_boundary
    )
    

    # prepare target points data
    grid.search.set_target(
        pts_df=pts_df_target,
        sum_names=sum_names,
        x_coord_name=tgt_x_coord_name,
        y_coord_name=tgt_y_coord_name,
        row_name=tgt_row_name,
        col_name=tgt_col_name,
        silent=silent,
    )

    # prepare source points data
    grid.search.set_source(
        pts_df=pts_df,
        x_coord_name=x_coord_name,
        y_coord_name=y_coord_name,
        row_name=row_name,
        col_name=col_name,
        cell_region_name=cell_region_name,
        sum_suffix=sum_suffix,
        plot_cell_reg_assign=plot_cell_reg_assign,
        plot_offset_checks=plot_offset_checks,
        plot_offset_regions=plot_offset_regions,
        plot_offset_raster=plot_offset_raster,
        silent=silent,
    )
    
    disk_sums_for_pts_df = grid.search.perform_search(silent=silent,plot_radius_sums=plot_radius_sums,plot_pt_disk=plot_pt_disk)

    return grid
#

@time_func_perf
def detect_cluster_pts(
    pts_df:_pd_DataFrame,
    crs:str,
    
    radius:float=0.0075,
    include_boundary:bool=False,
    exclude_pt_itself:bool=True,

    k_th_percentiles:float=[99.5],
    n_random_points:int=int(1e5),
    random_seed:int=None,

    grid=None,

    sum_names:list=['employment'],
    x_coord_name:str='lon',
    y_coord_name:str='lat',
    row_name:str='id_y',
    col_name:str='id_x',
    cell_region_name:str='cell_region',
    sum_suffix:str='_750m',
    cluster_suffix:str='_cluster',
    
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
    execute methods
    1. pts_df data -> aggreagate_point_data_to_disks_vectorized
    2. create columns to check whether points are within cluster depending on the various parameters
    """
    # OVERWRITE DEFAULTS
    if grid is None:
        grid = create_auto_grid_for_radius_search(
            pts_df_source=pts_df,
            crs=crs,
            radius=radius,
            y_coord_name=y_coord_name,
            x_coord_name=x_coord_name,
        )
    # initialize disk_search
    grid.search = DiskSearch(
        grid,
        radius=radius,
        exclude_pt_itself=exclude_pt_itself,
        include_boundary=include_boundary
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

    (cluster_threshold_values, rndm_pts_df) = get_distribution_for_random_points(
        grid=grid,
        pts_df=pts_df,
        radius=radius,
        sum_names=sum_names,
        x_coord_name=x_coord_name,
        y_coord_name=y_coord_name,
        row_name=row_name,
        col_name=col_name,
        cell_region_name=cell_region_name,
        sum_suffix=sum_suffix,
        n_random_points=n_random_points,
        k_th_percentiles=k_th_percentiles,
        plot_distribution=plot_distribution,
        random_seed=random_seed,
        silent=silent,
    )

    if not silent:
        for (colname, threshold_value, k_th_percentile) in zip(sum_names, cluster_threshold_values,k_th_percentiles):
            print("Threshold value for "+str(k_th_percentile)+"th-percentile is "+str(threshold_value)+" for "+str(colname)+".")
    
    grid.search.set_source(
        pts_df=pts_df,
        x_coord_name=x_coord_name,
        y_coord_name=y_coord_name,
        row_name=row_name,
        col_name=col_name,
        cell_region_name=cell_region_name,
        sum_suffix=sum_suffix,
        plot_cell_reg_assign=plot_cell_reg_assign,
        plot_offset_checks=plot_offset_checks,
        plot_offset_regions=plot_offset_regions,
        plot_offset_raster=plot_offset_raster,
        silent=silent,
    )


    disk_sums_for_pts_df = grid.search.perform_search(silent=silent,plot_radius_sums=plot_radius_sums,plot_pt_disk=plot_pt_disk)
    
    # save bool of whether pt is part of a cluster 
    pts_df[
        [str(cname)+str(cluster_suffix) for cname in sum_names]
    ] = disk_sums_for_pts_df>cluster_threshold_values


    if plot_distribution is not None:
        # print("disk_sums_for_random_points", disk_sums_for_random_points)
        create_distribution_plot(
            pts_df=pts_df,
            x_coord_name=x_coord_name,
            y_coord_name=y_coord_name,
            sum_radius_names=[n+sum_suffix for n in sum_names],
            rndm_pts_df=rndm_pts_df,
            cluster_threshold_values=cluster_threshold_values,
            k_th_percentiles=k_th_percentiles,
            radius=radius,
            plot_kwargs=plot_distribution
            )
    #

    def plot_rand_dist(
            pts_df=pts_df,
            x_coord_name=x_coord_name,
            y_coord_name=y_coord_name,
            sum_radius_names=[n+sum_suffix for n in sum_names],
            rndm_pts_df=rndm_pts_df,
            cluster_threshold_values=cluster_threshold_values,
            k_th_percentiles=k_th_percentiles,
            radius=radius,
            filename:str="",
            **plot_kwargs
            
    ):
        create_distribution_plot(
            pts_df=pts_df,
            x_coord_name=x_coord_name,
            y_coord_name=y_coord_name,
            sum_radius_names=sum_radius_names,
            rndm_pts_df=rndm_pts_df,
            cluster_threshold_values=cluster_threshold_values,
            k_th_percentiles=k_th_percentiles,
            radius=radius,
            filename=filename,
            plot_kwargs=plot_kwargs
            )
    grid.plot_rand_dist = plot_rand_dist
    
    plot_colnames = list(sum_names) + [n+sum_suffix for n in sum_names] + [str(cname)+str(cluster_suffix) for cname in sum_names]
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

def convert_coords_to_local_crs(
        pts_df,
        x_coord_name:str='lon',
        y_coord_name:str='lat',
        initial_crs:str="EPSG:4326",
) -> str:
    
    # https://gis.stackexchange.com/a/269552
    
    
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
    utm_code = convert_wgs_to_utm(*pts_df[[x_coord_name,y_coord_name]].mean(axis=0))
    # define project_from_to with iteration
    # see https://gis.stackexchange.com/a/127432/33092
    local_crs = 'EPSG:'+str(utm_code)
    transformer = Transformer.from_crs(crs_from=initial_crs, crs_to=local_crs, always_xy=True)
    pts_df[x_coord_name],pts_df[y_coord_name] = transformer.transform(pts_df[x_coord_name], pts_df[y_coord_name])
    return local_crs

def detect_cells_with_cluster_pts(
    pts_df:_pd_DataFrame,
    crs:str,
    
    radius:float=750,
    distance_thresholds:float=2500,
    make_convex:bool=True,
    include_boundary:bool=False,
    exclude_pt_itself:bool=True,

    k_th_percentiles:float=[99.5],
    n_random_points:int=int(1e5),
    random_seed:int=None,

    grid=None,

    sum_names:list=['employment'],
    x_coord_name:str='lon',
    y_coord_name:str='lat',
    row_name:str='id_y',
    col_name:str='id_x',
    cell_region_name:str='cell_region',
    sum_suffix:str='_750m',
    cluster_suffix:str='_cluster',
    
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
    grid = detect_cluster_pts(
        pts_df=pts_df,
        crs=crs,
        radius=radius,
        include_boundary=include_boundary,
        exclude_pt_itself=exclude_pt_itself,
        k_th_percentiles=k_th_percentiles,
        n_random_points=n_random_points,
        random_seed=random_seed,
        grid=grid,
        sum_names=sum_names,
        x_coord_name=x_coord_name,
        y_coord_name=y_coord_name,
        row_name=row_name,
        col_name=col_name,
        cell_region_name=cell_region_name,
        sum_suffix=sum_suffix,
        cluster_suffix=cluster_suffix,
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
        pts_df=pts_df,
        sum_names=sum_names,
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
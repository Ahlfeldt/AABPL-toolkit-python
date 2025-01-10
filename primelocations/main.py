from pandas import DataFrame as _pd_DataFrame
from primelocations.random_distribution import (get_distribution_for_random_points)
from primelocations.radius_search.radius_search_class import (aggreagate_point_data_to_disks_vectorized, DiskSearch)
from primelocations.radius_search.grid_class import Grid
from primelocations.testing.test_performance import time_func_perf, func_timer_dict

@time_func_perf
def create_auto_grid_for_radius_search(
    pts_df_source:_pd_DataFrame,
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
            set_fixed_spacing=radius/3, # TODO don t set fixed spacing but
            silent=silent,
        )
#

@time_func_perf
def radius_search(
    pts_df:_pd_DataFrame,
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

    time_dict:dict={'i':0,'clean':-1},

    plot_radius_sums:dict=None,
    plot_pt_disk:dict=None,
    plot_cell_reg_assign:dict=None,
    plot_offset_checks:dict=None,
    plot_offset_regions:dict=None,
    plot_offset_raster:dict=None,
    silent:bool = False,
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
def detect_clusters(
    pts_df:_pd_DataFrame,
    
    radius:float=0.0075,
    include_boundary:bool=False,
    exclude_pt_itself:bool=True,

    k_th_percentiles:float=[99.5],
    n_random_points:int=int(1e6),
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
    
    time_dict:dict={'i':0,'clean':-1},

    plot_distribution:dict=None,
    plot_radius_sums:dict=None,
    plot_cluster_points:dict=None,
    plot_pt_disk:dict=None,
    plot_cell_reg_assign:dict=None,
    plot_offset_checks:dict=None,
    plot_offset_regions:dict=None,
    plot_offset_raster:dict=None,
    silent:bool = False,
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

    (cluster_threshold_value, disk_sums_for_random_points) = get_distribution_for_random_points(
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
        for (colname, threshold_value, k_th_percentile) in zip(sum_names, cluster_threshold_value,k_th_percentiles):
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
    ] = disk_sums_for_pts_df>cluster_threshold_value

    if plot_cluster_points is not None:
        print('create plot for cluster points')
        pass
    
    return grid
# done
# next thing would be to label cells as clustered or not
# then to create orthogonal convex hull around clusters
# then to maybe wrap everything in one final function  
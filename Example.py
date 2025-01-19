# install package into your environment through your console via
# pip install ABRSQOL
# or install it from this script:
import subprocess, sys
try:
    __import__('aabpl')
    from aabpl import __version__
    if __version__ in ['0.1.'+str(i) for i in range(6)]:
        subprocess.check_call([sys.executable, "-m", "pip", "uninstall", 'aabpl'])
        raise ImportError("Old version, updated needed")
except ImportError:
    print("Installing / updating package aabpl...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", 'aabpl', "--upgrade"])

### set up working directory and folders
import os
# Create output folders if they don't exist
working_directory = "./" # specify your folder path here C:/User/YourName/YourFolder
output_data_folder = os.path.join(working_directory, "output_data/")
output_gis_folder = os.path.join(working_directory, "output_gis/")
output_maps_folder = os.path.join(working_directory, "output_maps/")
temp_folder = os.path.join(working_directory, "temp")
os.makedirs(output_data_folder, exist_ok=True)
os.makedirs(output_gis_folder, exist_ok=True)
os.makedirs(output_maps_folder, exist_ok=True)
os.makedirs(temp_folder, exist_ok=True)

### Import packages
from pandas import read_csv
from aabpl.main import detect_cluster_pts, radius_search, convert_coords_to_local_crs, detect_cells_with_cluster_pts
from aabpl.testing.test_performance import analyze_func_perf, func_timer_dict

path_to_your_csv = '../../cbsa_sample_data/plants_10180.txt'
crs_of_your_csv =  "EPSG:4326"
pts_df = read_csv(path_to_your_csv, sep=",", header=None)
pts_df.columns = ["eid", "employment", "industry", "lat","lon","moved"]

# automatically detect most local projection
local_crs = convert_coords_to_local_crs(pts_df)

# ## Detecting clustered points and cells: 
# 1. Calculate the sum for each point of pts_df for specified variable(s) within specified radius
# 2. Draw random points within area and calculate the sum for each point of the original dataset for specified variable(s) within specified radius pts_df
# 3. Label all pts in which radius the sum exceeds the threshold value for specified percentile of given region. label those points as cluster points
# 4. Label all cells that at least contain one point exceeding that threshold as cluster cells 
# 5. Merge all cells that lie in the vecinity of each other (distance between region centroids below distance_thresholds) into single clusters
# 6. Additionaly include cells within a cluster that lie in its convex hull.
# 7. Returning pts_df with columns on radius sums and cluster ids aswell as grid including cluster information 

# ### Functions and the steps they execute
# - radius_search: 1
# - get_distribution_for_random_points: 2.
# - detect_cluster_pts: 3.
# - detect_cells_with_cluster_pts: 4.-7. 

grid = detect_cells_with_cluster_pts(
    pts_df=pts_df,
    crs=local_crs,
    radius=750,
    include_boundary=False,
    exclude_pt_itself=True,
    distance_thresholds=2500,
    k_th_percentiles=[99.97],
    n_random_points=int(1e5),
    make_convex=True,
    random_seed=0,
    sum_names=['employment'],
    silent = True,
)

## Save DataFrames with radius sums and clusters
# Using all the save options below is most likely excessive. 
# saving the shapefile for save_cell_clusters and save_sparse_grid is most
# likely sufficient

# save files as needed
# save only only clusters including their geometry, aggregate values, area and id
grid.save_cell_clusters(filename=output_gis_folder+'grid_clusters', file_format='shp', target_crs=crs_of_your_csv)
# grid.save_cell_clusters(filename=output_data_folder+'grid_clusters', file_format='csv', target_crs=crs_of_your_csv)
# save sparse grid including cells only those cells that at least contain one point
grid.save_sparse_grid(filename=output_gis_folder+'grid_clusters', file_format='shp', target_crs=crs_of_your_csv)
# grid.save_sparse_grid(filename=output_data_folder+'grid_clusters', file_format='csv', target_crs=crs_of_your_csv)
# save full grid including cells that have no points in them (through many empty cells this will occuppy unecessary disk space)
# grid.save_full_grid(filename=output_gis_folder+'grid_clusters', file_format='shp', target_crs=crs_of_your_csv)
# grid.save_full_grid(filename=output_data_folder+'grid_clusters', file_format='csv', target_crs=crs_of_your_csv)

pts_df.to_csv(output_data_folder+'pts_df_w_clusters.csv')

# CREATE PLOTS
grid.plot_clusters(output_maps_folder+'clusters_employment_750m_9975th')
grid.plot_vars(filename=output_maps_folder+'employment_vars')
grid.plot_cluster_pts(filename=output_maps_folder+'employment_cluster_pts')
grid.plot_rand_dist(filename=output_maps_folder+'rand_dist_employment')



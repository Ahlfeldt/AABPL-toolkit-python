# install package into your environment through your console via
# pip install ABRSQOL
# or install it from this script:
import subprocess, sys
subprocess.check_call([sys.executable, "-m", "pip", "install", 'aabpl', "--upgrade"])

### set up working directory and folders
import os
# Create output folders if they don't exist
working_directory = "P:/Algorithm/PL_python/AABPL-toolkit-python" # specify your folder path here C:/User/YourName/YourFolder
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
from aabpl import (
    radius_search, radius_sum, radius_count, radius_mean,
    detect_cluster_pts, detect_cluster_cells
)

path_to_your_csv = 'input_data/hist_New_York.txt'
crs_of_your_csv =  "EPSG:4326"
pts = read_csv(path_to_your_csv, sep=",", header=None)
pts.columns = ["eid", "employment", "industry", "lat","lon","moved"]

grid = detect_cluster_cells(
    pts=pts,
    crs=crs_of_your_csv,
    r=750,
    c='employment', # Name of the column or list of columns for which values shall be aggregated within search radius 
    exclude_pt_itself=True, # False
    sample_area='buff_cells_min_pts', # 'concave', 'convex', 'buffer', 'bounding_box', 'grid' or None
    min_pts_to_sample_cell=1, # 0, 10
    weight_valid_area=None, # 'estimate', 'precise'
    k_th_percentile=99.5,
    n_random_points=100000,
    random_seed=0,
    queen_contingency=1, # 0, 2
    centroid_dist_threshold=2500,
    border_dist_threshold=1000,
    min_cluster_share_after_contingency=0.05,
    spacing=250, # sets the cell width and height, if None defaults to r/3 
    make_convex=True, # False
)

## Save DataFrames with radius sums and clusters
# Using all the save options below is most likely excessive. 
# saving the shapefile for save_cell_clusters and save_sparse_grid is most
# likely sufficient

# save files as needed
# save only only clusters including their geometry, aggregate values, area and id
df_clusters = grid.save_cell_clusters(filename=output_gis_folder+'clusters', file_format='shp')
df_clusters = grid.save_cell_clusters(filename=output_data_folder+'clusters', file_format='csv')

# save sparse grid including cells only those cells that at least contain one point
df_sparse_grid = grid.save_sparse_grid(filename=output_gis_folder+'sparse_grid', file_format='shp')
df_sparse_grid = grid.save_sparse_grid(filename=output_data_folder+'sparse_grid', file_format='csv')

# save full grid including cells that have no points in them (through many empty cells this will occuppy unecessary disk space)
# df_full_grid = grid.save_full_grid(filename=output_gis_folder+'full_grid', file_format='shp')
# df_full_grid = grid.save_full_grid(filename=output_data_folder+'full_grid', file_format='csv')

pts.to_csv(output_data_folder+'pts_df_w_clusters.csv')

# CREATE PLOTS
fig = grid.plot.clusters(output_maps_folder+'clusters_employment_750m_995th')
fig = grid.plot.vars(filename=output_maps_folder+'employment_vars')
fig = grid.plot.cluster_pts(filename=output_maps_folder+'employment_cluster_pts')
fig = grid.plot.rand_dist(filename=output_maps_folder+'rand_dist_employment')

print("Successfully executed Example.py")


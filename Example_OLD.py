# install package into your environment through your console via
# pip install ABRSQOL
# or install it from this script:
import subprocess, sys
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
from aabpl.main import radius_search, detect_cluster_pts, detect_cluster_cells

path_to_your_csv = '../../cbsa_sample_data/plants_10180.txt'
crs_of_your_csv =  "EPSG:4326"
pts = read_csv(path_to_your_csv, sep=",", header=None)
pts.columns = ["eid", "employment", "industry", "lat","lon","moved"]

grid = detect_cluster_cells(
    pts=pts,
    crs=crs_of_your_csv,
    r=750,
    columns=['employment'],
    exclude_pt_itself=True,
    distance_thresholds=2500,
    k_th_percentiles=[99.97],
    n_random_points=int(1e5),
    make_convex=True,
    random_seed=0,
    silent = True,
)

## Save DataFrames with radius sums and clusters
# Using all the save options below is most likely excessive. 
# saving the shapefile for save_cell_clusters and save_sparse_grid is most
# likely sufficient

# save files as needed
# save only only clusters including their geometry, aggregate values, area and id
grid.save_cell_clusters(filename=output_gis_folder+'grid_clusters', file_format='shp')
# grid.save_cell_clusters(filename=output_data_folder+'grid_clusters', file_format='csv')
# save sparse grid including cells only those cells that at least contain one point
grid.save_sparse_grid(filename=output_gis_folder+'grid_clusters', file_format='shp')
# grid.save_sparse_grid(filename=output_data_folder+'grid_clusters', file_format='csv')
# save full grid including cells that have no points in them (through many empty cells this will occuppy unecessary disk space)
# grid.save_full_grid(filename=output_gis_folder+'grid_clusters', file_format='shp')
# grid.save_full_grid(filename=output_data_folder+'grid_clusters', file_format='csv')

pts.to_csv(output_data_folder+'pts_df_w_clusters.csv')

# CREATE PLOTS
grid.plot.clusters(output_maps_folder+'clusters_employment_750m_9975th')
grid.plot.vars(filename=output_maps_folder+'employment_vars')
grid.plot.cluster_pts(filename=output_maps_folder+'employment_cluster_pts')
grid.plot.rand_dist(filename=output_maps_folder+'rand_dist_employment')

print("Successfully executed Example.py")

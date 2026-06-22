# install package into your environment through your console via
# pip install aabpl
# or install it from this script:
import subprocess, sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "aabpl", "--upgrade", "--no-cache-dir"])

### Set up working directory and output folders
import os
working_directory = "C:/Users/YourName/YourFolder"   # <-- change this
output_data_folder = os.path.join(working_directory, "output_data/")
output_gis_folder  = os.path.join(working_directory, "output_gis/")
output_maps_folder = os.path.join(working_directory, "output_maps/")
os.makedirs(output_data_folder, exist_ok=True)
os.makedirs(output_gis_folder,  exist_ok=True)
os.makedirs(output_maps_folder, exist_ok=True)

### Import packages
from pandas import read_csv
from aabpl import (
    radius_search, radius_sum, radius_count, radius_mean,
    detect_cluster_pts, detect_cluster_cells,
)

### Load data
path_to_your_csv = 'input_data/hist_New_York.txt'
crs_of_your_csv  = "EPSG:4326"                       # coordinate system of lat/lon columns
pts = read_csv(path_to_your_csv, sep=",", header=None)
pts.columns = ["eid", "employment", "industry", "lat", "lon", "moved"]

### Detect employment clusters
grid = detect_cluster_cells(
    pts=pts,
    crs=crs_of_your_csv,
    r=750,                                  # search radius in CRS units (metres after reprojection)
    c='employment',                         # column(s) to aggregate within radius; list for multiple
    stat='sum',                             # aggregation: sum|count|mean|variance|std|cv|skewness|kurtosis
    exclude_self=True,                      # exclude the point itself from its own neighbourhood sum
    sample_area='buff_cells_min_pts',       # sample-area: 'concave'|'convex'|'buffer'|'bounding_box'|'grid'|None or a Shapely Polygon/MultiPolygon
    min_pts_to_sample_cell=1,               # min points a cell must contain to be part of the sample area
    weight_valid_area=None,                 # correct edge effects: None|'estimate'|'precise'
    k_th_percentile=99.5,                   # null-distribution percentile used as cluster threshold
    n_random_points=100000,                 # random points drawn to build the null distribution
    random_seed=0,                          # set for reproducibility; None for random
    queen_contingency=1,                    # merge adjacent clusters within this many cells (0 = no merge)
    centroid_dist_threshold=2500,           # merge clusters whose centroids are within this distance (CRS units)
    border_dist_threshold=1000,             # merge clusters whose borders are within this distance (CRS units)
    min_cluster_share_after_contingency=0.05, # drop clusters smaller than this share of the largest cluster
    make_convex=True,                       # replace each cluster polygon with its convex hull
    spacing=250,                            # output grid cell size in metres (always projected); defaults to r/3
)

### Save outputs
# Cluster polygons with aggregate values and area
df_clusters    = grid.save_cell_clusters(filename=output_gis_folder+'clusters',   file_format='shp')
df_clusters    = grid.save_cell_clusters(filename=output_data_folder+'clusters',  file_format='csv')

# Sparse grid: only cells that contain at least one point
df_sparse_grid = grid.save_sparse_grid(filename=output_gis_folder+'sparse_grid',  file_format='shp')
df_sparse_grid = grid.save_sparse_grid(filename=output_data_folder+'sparse_grid', file_format='csv')

# Full grid (many empty cells — large file, usually not needed):
# df_full_grid = grid.save_full_grid(filename=output_gis_folder+'full_grid', file_format='shp')

pts.to_csv(output_data_folder+'pts_df_w_clusters.csv')

### Plots
grid.plot.clusters(  output_maps_folder+'clusters_employment_750m_995th')
grid.plot.vars(      filename=output_maps_folder+'employment_vars')
grid.plot.cluster_pts(filename=output_maps_folder+'employment_cluster_pts')
grid.plot.rand_dist( filename=output_maps_folder+'rand_dist_employment')

### Radius search only (no clustering)
grid = radius_sum(
    pts=pts,
    crs=crs_of_your_csv,
    r=750,
    c='employment',   # column(s) to sum
    exclude_self=True,
)

print("Successfully executed Example.py")

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
# Result columns appended to pts:
#   employment_sum_15000        — radius-sum aggregate (auto-named: {col}_{stat}_{r})
#   employment_cluster_sum_15000 — True/False cluster label (auto-named: {col}_cluster_{stat}_{r})
grid = detect_cluster_cells(
    pts=pts,
    crs=crs_of_your_csv,
    r=15000,                        # search radius in metres (after reprojection); also accepts r=[500,750] or r=[(0,500),(500,750)]
    c='employment',                 # column(s) to aggregate within radius; list for multiple
    stat='sum',                     # aggregation: sum|count|mean|variance|std|cv|skewness|kurtosis
    exclude_self=True,              # exclude the point itself from its own neighbourhood sum
    sample_area='buff_cells,min_pts=1',  # sampling region for null distribution; call resolve_sample_area.params() for all options
                                    # alternatives: 'concave,concavity=0.5' | 'convex' | 'bbox' | 'grid' | Shapely Polygon/MultiPolygon
    weight_valid_area=None,         # edge-effect correction: None|'estimate'|'precise'
    k_th_percentile=99.5,           # null-distribution percentile used as cluster threshold
    n_random_points=100000,         # random points drawn to build the null distribution
    random_seed=0,                  # set for reproducibility; None for random
    contingency=1,                  # merge adjacent clusters within this many cells (0 = no merge)
    merge_dist=(25000, 10000),      # (centroid_dist, border_dist): merge clusters closer than these distances
    min_cluster_share=(0.05, 0.0, 0.0),  # drop clusters smaller than this share of the largest (after contingency, centroid, convex steps)
    make_convex=True,               # replace each cluster polygon with its convex hull
    spacing=15000,                  # output grid cell size in metres; defaults to r/3
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
grid.plot.clusters(  output_maps_folder+'clusters_employment_995th')
grid.plot.vars(      filename=output_maps_folder+'employment_vars')
grid.plot.cluster_pts(filename=output_maps_folder+'employment_cluster_pts')
grid.plot.rand_dist( filename=output_maps_folder+'rand_dist_employment')

### Radius search only (no clustering)
grid = radius_sum(
    pts=pts,
    crs=crs_of_your_csv,
    r=15000,
    c='employment',   # column(s) to sum; result appended as employment_sum_15000
    exclude_self=True,
)

print("Successfully executed Example.py")

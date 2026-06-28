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
    r=15000,                            # search radius in metres (after reprojection); also accepts multiple radii r=[500,750] or (weighted) distance bands r=[(0,500,.004),(500,750,.0016)]
    c='employment',                     # column(s) to aggregate within radius; c=['col1','col2'] for multiple
    stat='sum',                         # aggregation statistic: sum|count|mean|std|variance|cv|skewness|kurtosis
    exclude_self=True,                  # exclude the point's own value from its radius sum
    sample_area='buff_cells,min_pts=1,buf=30000',  # region used to draw null-distribution random points; call aabpl.main.resolve_sample_area.params() for all options
                                        # alternatives: 'concave,concavity=0.5' | 'convex' | 'bbox' | 'grid' | Shapely Polygon/MultiPolygon
    weight_valid_area=None,             # edge-effect correction near sample boundary: None | 'estimate' | 'precise'
    k_th_percentile=99.5,               # cluster threshold = this percentile of the null distribution (0–100); lower → more clusters
    null_distribution=100_000,          # int → draw N random points uniformly within sample area for null distribution; or pass an (N,2) array/DataFrame of pre-drawn coords (x first)
    random_seed=0,                      # for reproducibility; None = different result each run
    contingency=(1,1),                  # gap tolerance (queen_cells, rook_cells) when merging near-adjacent clusters; (0,0) = no gap merge
    merge_dist=(25000, 15000),          # (centroid_dist, border_dist) in metres: merge cluster pairs where both distances fall below these thresholds
    min_cluster_share=(0.05, 0.0, 0.0),# drop clusters below this share of the largest: applied after (contingency merge, centroid merge, convex-hull step)
    make_convex=True,                   # replace each cluster polygon with its convex hull (fills concavities)
    cell_size=5000,                     # output grid resolution in metres; rule of thumb: r/3; smaller = finer detail but slower
    overwrite=True,                     # allow overwriting existing radius-sum columns in pts DataFrame
)
# grid.info()  # inspect grid layout, cell size, cluster summary, bounds, and available plot methods

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

### Radius search only (no clustering) — exclude_self defaults to False for radius_sum/radius_search
grid = radius_sum(
    pts=pts,
    crs=crs_of_your_csv,
    r=15000,                            # search radius in metres; also accepts r=[500,750] or distance bands r=[(0,500,.004),(500,750,.0016)]
    c='employment',                     # column(s) to aggregate; c=['col1','col2'] for multiple; result appended as employment_sum_15000
    exclude_self=False,                 # False by default: each point's own value is included in its radius sum; set True to exclude self-contribution
    overwrite=True,
)

print("Successfully executed Example.py")

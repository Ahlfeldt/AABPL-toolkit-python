# Check and install packages

import os
import subprocess
import sys

# Function to ensure all required packages are installed
def ensure_packages(packages):
    for package in packages:
        try:
            __import__(package)
        except ImportError:
            print(f"Package '{package}' not found. Installing...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        else:
            print(f"Package '{package}' is already installed.")

# List of required packages
required_packages = [
    "numpy", "pandas", "geopandas", "matplotlib", "shapely",
    "scipy", "pyproj", "aabpl", "matplotlib-scalebar" , "geopy"
]

# Ensure all packages are installed
ensure_packages(required_packages)

# Import libraries after ensuring installation
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import box, Point
from aabpl.main import detect_clusters, convert_coords_to_local_crs
from scipy.ndimage import label
from matplotlib.colors import BoundaryNorm
from pyproj import Transformer
try:
    from matplotlib_scalebar.scalebar import ScaleBar
except ImportError:
    pass

print("All required packages are installed and imported successfully.")

# Packages installed

# Define user choices
working_directory = "P:/Algorithm/PL_python/DebuggingArena/GA"
input_file = "plants_10180.txt"

# Define output folder paths
output_data_folder = os.path.join(working_directory, "output_data")
output_gis_folder = os.path.join(working_directory, "output_gis")
output_maps_folder = os.path.join(working_directory, "output_maps")
temp_folder = os.path.join(working_directory, "temp")

# Create output folders if they don't exist
os.makedirs(output_data_folder, exist_ok=True)
os.makedirs(output_gis_folder, exist_ok=True)
os.makedirs(output_maps_folder, exist_ok=True)
os.makedirs(temp_folder, exist_ok=True)

# Update file paths to reflect new folders
output_csv = os.path.join(output_data_folder, "pts_df_with_radius_sums.csv")
grid_shapefile = os.path.join(temp_folder, "grid_250m.shp")
clustered_grid_shapefile = os.path.join(temp_folder, "grid_with_clusters.shp")
clustered_grid_with_ids_shapefile = os.path.join(output_gis_folder, "grid_with_cluster_ids.shp")
output_grid_csv = os.path.join(output_data_folder, "grid_with_cluster_ids.csv")
output_map = os.path.join(output_maps_folder, "grid_with_cluster_ids.png")

os.chdir(working_directory)  # Set working directory
print("Working directory set to:", os.getcwd())

# Define CRS and clustering parameters
crs_of_your_csv = "EPSG:4326"
RADIUS = 750
K_TH_PERCENTILES = [99.975]
N_RANDOM_POINTS = int(1e5)
CELL_SIZE = 250  # Define cell size (edge length of squares) of grid that will be generated
distance_threshold = 2500  # Distance threshold in meters, smaller clusters within this distance will be merged to larger clusters

# Import required libraries
import subprocess, sys
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import box, Point
from aabpl.main import detect_clusters, convert_coords_to_local_crs
from scipy.ndimage import label
from matplotlib.colors import BoundaryNorm

# Install necessary packages
subprocess.check_call([sys.executable, "-m", "pip", "install", "aabpl"])
print("Imports complete")

# Remaining steps follow unchanged

# STEP 1: Perform clustering ##################################################

# Load the input CSV
pts_df = pd.read_csv(input_file, sep=",", header=None)
pts_df.columns = ["eid", "employment", "industry", "lat", "lon", "moved"]
print("Initial DataFrame:")
print(pts_df.info())

# Convert coordinates to projected CRS
convert_coords_to_local_crs(pts_df)
print("DataFrame after CRS conversion:")
print(pts_df.head())

# Detect clusters
detect_clusters(
    pts_df=pts_df,
    radius=RADIUS,
    include_boundary=False,
    exclude_pt_itself=True,
    k_th_percentiles=K_TH_PERCENTILES,
    n_random_points=N_RANDOM_POINTS,
    random_seed=0,
    sum_names=["employment"],
    silent=True,
)
print("Clustering done.")

# Rename columns
pts_df.rename(columns={"lat": "y_proj", "lon": "x_proj"}, inplace=True)
pts_df["lat"] = pd.read_csv(input_file, sep=",", header=None)[3]
pts_df["lon"] = pd.read_csv(input_file, sep=",", header=None)[4]

# Replace projected coordinates
from pyproj import Transformer

print("Reprojecting coordinates to ensure accuracy...")
transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
pts_df["x_proj"], pts_df["y_proj"] = transformer.transform(pts_df["lon"], pts_df["lat"])

# Save output CSV
pts_df.to_csv(output_csv, index=False, encoding="utf-8")
print(f"Clustering results saved to {output_csv}")


# STEP 1: Perform clustering ##################################################

# Load the input CSV
pts_df = pd.read_csv(input_file, sep=",", header=None)
pts_df.columns = ["eid", "employment", "industry", "lat", "lon", "moved"]
print("Initial DataFrame:")
print(pts_df.info())

# Convert coordinates to projected CRS
convert_coords_to_local_crs(pts_df)
print("DataFrame after CRS conversion:")
print(pts_df.head())

# Detect clusters
detect_clusters(
    pts_df=pts_df,
    radius=RADIUS,
    include_boundary=False,
    exclude_pt_itself=True,
    k_th_percentiles=K_TH_PERCENTILES,
    n_random_points=N_RANDOM_POINTS,
    random_seed=0,
    sum_names=["employment"],
    silent=True,
)
print("Clustering done.")

# Rename columns
pts_df.rename(columns={"lat": "y_proj", "lon": "x_proj"}, inplace=True)
pts_df["lat"] = pd.read_csv(input_file, sep=",", header=None)[3]
pts_df["lon"] = pd.read_csv(input_file, sep=",", header=None)[4]

# Replace projected coordinates
from pyproj import Transformer

print("Reprojecting coordinates to ensure accuracy...")
transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
pts_df["x_proj"], pts_df["y_proj"] = transformer.transform(pts_df["lon"], pts_df["lat"])

# Save output CSV
pts_df.to_csv(output_csv, index=False, encoding="utf-8")
print(f"Clustering results saved to {output_csv}")


# STEP 2: Generate grid #######################################################

# Load establishments and compute bounding box
pts_df = pd.read_csv(output_csv)
minx, maxx = pts_df["x_proj"].min(), pts_df["x_proj"].max()
miny, maxy = pts_df["y_proj"].min(), pts_df["y_proj"].max()
cell_size = CELL_SIZE

# Create grid cells
x_coords = np.arange(minx, maxx, cell_size)
y_coords = np.arange(miny, maxy, cell_size)
grid_cells = [box(x, y, x + cell_size, y + cell_size) for x in x_coords for y in y_coords]

# Create grid GeoDataFrame
grid = gpd.GeoDataFrame({"geometry": grid_cells}, crs="EPSG:3857")
grid["grid_id"] = range(1, len(grid) + 1)
grid.to_file(grid_shapefile, driver="ESRI Shapefile")
print(f"Grid saved to {grid_shapefile}")

# STEP 3: Assign establishments to grid #######################################

# Load grid and establishments
grid = gpd.read_file(grid_shapefile)
pts_df = pd.read_csv(output_csv)
clustered_pts = pts_df[pts_df["employment_cluster"]]

# Convert establishments to GeoDataFrame
clustered_gdf = gpd.GeoDataFrame(
    clustered_pts,
    geometry=gpd.points_from_xy(clustered_pts["x_proj"], clustered_pts["y_proj"]),
    crs="EPSG:3857",
)
grid["has_cluster"] = 0
grid["employment_sum"] = 0

# Assign clusters and sum employment
for idx, cell in grid.iterrows():
    points_in_cell = clustered_gdf[clustered_gdf.geometry.within(cell.geometry)]
    grid.at[idx, "has_cluster"] = int(not points_in_cell.empty)
    all_points_in_cell = pts_df[
        (pts_df["x_proj"] >= cell.geometry.bounds[0])
        & (pts_df["x_proj"] <= cell.geometry.bounds[2])
        & (pts_df["y_proj"] >= cell.geometry.bounds[1])
        & (pts_df["y_proj"] <= cell.geometry.bounds[3])
    ]
    grid.at[idx, "employment_sum"] = all_points_in_cell["employment"].sum()

grid.to_file(clustered_grid_shapefile, driver="ESRI Shapefile")
print(f"Updated grid saved to {clustered_grid_shapefile}")

# STEP 4: Assign cluster IDs ##################################################

# STEP 4: Assign cluster IDs ##################################################

grid = gpd.read_file(clustered_grid_shapefile)

if "has_cluster" not in grid.columns:
    print("Column 'has_cluster' not found, renaming truncated column...")
    grid.rename(columns={"has_cluste": "has_cluster"}, inplace=True)

# Create raster-like matrix for cluster identification
grid_bounds = grid.total_bounds
ncols, nrows = int((grid_bounds[2] - grid_bounds[0]) / cell_size), int((grid_bounds[3] - grid_bounds[1]) / cell_size)
cluster_matrix = np.zeros((nrows, ncols), dtype=int)

# Map grid cells to matrix positions
matrix_to_cell = {}
for idx, cell in grid.iterrows():
    col = int((cell.geometry.bounds[0] - grid_bounds[0]) / cell_size)
    row = int((grid_bounds[3] - cell.geometry.bounds[1]) / cell_size)
    if 0 <= row < nrows and 0 <= col < ncols:
        cluster_matrix[row, col] = cell["has_cluster"]
        matrix_to_cell[(row, col)] = idx

# Identify contiguous clusters
labeled_array, num_features = label(cluster_matrix)
grid["cluster_id"] = 0

# Assign cluster IDs to the grid
for (row, col), grid_idx in matrix_to_cell.items():
    cluster_label = labeled_array[row, col]
    grid.at[grid_idx, "cluster_id"] = cluster_label if cluster_matrix[row, col] == 1 else 0

# Calculate centroids for grid cells in projected coordinates
grid["centroid_x_proj"] = grid.geometry.centroid.x  # Projected X
grid["centroid_y_proj"] = grid.geometry.centroid.y  # Projected Y

# Reproject centroids to latitude and longitude (EPSG:4326)
print("Reprojecting grid cell centroids to lat/lon...")
grid_centroids = grid.copy().to_crs("EPSG:4326")
grid["grid_lon"] = grid_centroids.geometry.centroid.x  # Longitude
grid["grid_lat"] = grid_centroids.geometry.centroid.y  # Latitude

# Save updated grid with cluster IDs and centroids as shapefile
grid.drop(columns=["centroid_x_proj", "centroid_y_proj"], inplace=True)  # Optional to avoid redundancy
grid.to_file(clustered_grid_with_ids_shapefile, driver="ESRI Shapefile")
print(f"Clustered grid saved to {clustered_grid_with_ids_shapefile}")

# Save grid table as CSV (including centroids)
grid.drop(columns=["geometry"]).to_csv(output_grid_csv, index=False)
print(f"Grid data table saved to {output_grid_csv}")


# STEP 5: Generate map ########################################################

grid = gpd.read_file(clustered_grid_with_ids_shapefile)

if "employment_sum" not in grid.columns:
    print("Handling truncated column for 'employment_sum'...")
    grid.rename(columns={"employment": "employment_sum"}, inplace=True)

employment_norm = BoundaryNorm([0.5, 1.5, 50, 100, 250, 1000, 1e6], ncolors=256)

fig, ax = plt.subplots(1, 1, figsize=(12, 10))
background_cells = grid.copy()
background_cells["employment_sum"] = background_cells["employment_sum"].clip(upper=1e6)
background_cells.plot(column="employment_sum", cmap="Greys", norm=employment_norm, ax=ax, alpha=0.8)

sm = plt.cm.ScalarMappable(cmap="Greys", norm=employment_norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, orientation="vertical", fraction=0.03, pad=0.02)
cbar.set_label("Employment Range", fontsize=12)

cluster_cells = grid[grid["cluster_id"] > 0]
cluster_cells.plot(ax=ax, color="red", alpha=0.5)

cluster_centroids = cluster_cells.dissolve(by="cluster_id").centroid
for cluster_id, centroid in zip(cluster_centroids.index, cluster_centroids.geometry):
    ax.text(centroid.x + 200, centroid.y + 200, str(cluster_id), color="red", fontsize=12, ha="left", va="bottom")

ax.set_title("Cluster Map with Employment Background", fontsize=16)
ax.set_axis_off()
plt.savefig(output_map, dpi=300, bbox_inches="tight")
print(f"Map saved to {output_map}")
plt.show()

# STEP &: AGGREGATE TO PRIME LOCATIONS #######################################

# Import required libraries
import os
import pandas as pd
import geopandas as gpd
from geopy.distance import geodesic as geopy_distance

# Define parameters
working_directory = "P:/Algorithm/PL_python/DebuggingArena/GA"
input_shapefile = os.path.join(working_directory, "output_gis", "grid_with_cluster_ids.shp")
output_shapefile = os.path.join(working_directory, "temp", "grid_with_pl_ids.shp")

# Load input shapefile
print("Loading input shapefile...")
grid = gpd.read_file(input_shapefile)

# Tabulate cluster IDs in the input shapefile
input_cluster_tabulation = grid["cluster_id"].value_counts().reset_index()
input_cluster_tabulation.columns = ["cluster_id", "count"]
print("\nTabulation of cluster IDs in the input shapefile:")
print(input_cluster_tabulation)

# Save tabulation of input clusters
input_tabulation_csv = os.path.join(working_directory, "temp", "input_cluster_tabulation.csv")
input_cluster_tabulation.to_csv(input_tabulation_csv, index=False)

# Compute employment-weighted centroids for each cluster
print("\nComputing employment-weighted centroids...")
cluster_summary = grid.groupby("cluster_id", group_keys=False).apply(
    lambda group: pd.Series({
        "emp_sum": group["employment"].sum(),
        "cent_lat": (group["grid_lat"] * group["employment"]).sum() / group["employment"].sum(),
        "cent_lon": (group["grid_lon"] * group["employment"]).sum() / group["employment"].sum()
    })
).reset_index()

# Merge centroids and employment sums back into the grid
grid = grid.merge(cluster_summary, on="cluster_id", how="left")

# Merging clusters
print("Merging clusters based on distance threshold...")
for current_cluster_id in cluster_summary["cluster_id"]:
    if current_cluster_id == 0:
        continue  # Skip unclustered cells

    # Check if current cluster has any remaining cells
    current_cluster_cells = grid[grid["cluster_id"] == current_cluster_id]
    if current_cluster_cells.empty:
        continue

    current_centroid = (
        current_cluster_cells["cent_lat"].iloc[0],
        current_cluster_cells["cent_lon"].iloc[0],
    )

    for neighbor_cluster_id in cluster_summary["cluster_id"]:
        if neighbor_cluster_id == 0 or neighbor_cluster_id == current_cluster_id:
            continue  # Skip unclustered cells and self-comparison

        # Check if neighbor cluster has any remaining cells
        neighbor_cluster_cells = grid[grid["cluster_id"] == neighbor_cluster_id]
        if neighbor_cluster_cells.empty:
            continue

        neighbor_centroid = (
            neighbor_cluster_cells["cent_lat"].iloc[0],
            neighbor_cluster_cells["cent_lon"].iloc[0],
        )

        # Compute distance between centroids
        distance = geopy_distance(current_centroid, neighbor_centroid).meters
        print(f"  Comparing cluster {current_cluster_id} to {neighbor_cluster_id}: Distance = {distance:.2f} meters")

        if distance < distance_threshold:
            print(f"    Merging cluster {neighbor_cluster_id} into {current_cluster_id}")
            grid.loc[grid["cluster_id"] == neighbor_cluster_id, "cluster_id"] = current_cluster_id

            # Recompute centroids and employment sums after merging
            cluster_summary = grid.groupby("cluster_id", group_keys=False).apply(
                lambda group: pd.Series({
                    "emp_sum": group["employment"].sum(),
                    "cent_lat": (group["grid_lat"] * group["employment"]).sum() / group["employment"].sum(),
                    "cent_lon": (group["grid_lon"] * group["employment"]).sum() / group["employment"].sum()
                })
            ).reset_index()

# Save output shapefile
grid.rename(columns={
    "cent_lat": "cl_lat",  # Shortened for shapefile compatibility
    "cent_lon": "cl_lon",
    "emp_sum": "emp_sum"
}, inplace=True)
grid.to_file(output_shapefile, driver="ESRI Shapefile")
print(f"\nOutput shapefile saved to {output_shapefile}")

# Tabulate cluster IDs in the output shapefile
output_cluster_tabulation = grid["cluster_id"].value_counts().reset_index()
output_cluster_tabulation.columns = ["cluster_id", "count"]
print("\nTabulation of cluster IDs in the output shapefile:")
print(output_cluster_tabulation)

# Save tabulation of output clusters
output_tabulation_csv = os.path.join(working_directory, "temp", "output_cluster_tabulation.csv")
output_cluster_tabulation.to_csv(output_tabulation_csv, index=False)

print("\nTabulations saved as CSV files:")
print(f"- {input_tabulation_csv}")
print(f"- {output_tabulation_csv}")
 
 
# STEP 7: CONVEX HULL ####################################################################### 

import os
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon

# Define paths and folders
input_shapefile = os.path.join(working_directory, "temp", "grid_with_pl_ids.shp")
output_shapefile = os.path.join(working_directory, "output_gis", "grid_with_final_pl_ids.shp")

# Ensure working directory is set
os.chdir(working_directory)
print("Working directory set to:", os.getcwd())

# Load the input shapefile
print("Loading input shapefile...")
grid = gpd.read_file(input_shapefile)
print("Input shapefile loaded successfully!")

# Initialize a new column for pl_id
grid["pl_id"] = 0

# Process clusters by cluster ID
print("Processing clusters and building convex hulls...")
cluster_ids = grid["cluster_id"].unique()
for cluster_id in cluster_ids:
    if cluster_id == 0:  # Skip unclustered cells
        continue

    # Get all cells belonging to the current cluster
    cluster_cells = grid[grid["cluster_id"] == cluster_id]

    # Build the convex hull for the cluster
    convex_hull = cluster_cells.unary_union.convex_hull

    # Assign pl_id to all cells within the convex hull
    grid.loc[grid.geometry.within(convex_hull), "pl_id"] = cluster_id

# Compute total employment by pl_id, excluding pl_id = 0
print("Computing total employment by pl_id...")
pl_summary = grid[grid["pl_id"] > 0].groupby("pl_id", group_keys=False)["employment"].sum().reset_index()
pl_summary.columns = ["pl_id", "emp_sum"]

# Sort pl_ids by employment and reassign ranks
print("Reassigning pl_ids based on employment ranks...")
pl_summary = pl_summary.sort_values("emp_sum", ascending=False).reset_index(drop=True)
pl_summary["new_pl_id"] = range(1, len(pl_summary) + 1)

# Merge updated pl_ids back into the grid
grid = grid.merge(pl_summary[["pl_id", "new_pl_id"]], on="pl_id", how="left")
grid["pl_id"] = grid["new_pl_id"].fillna(0).astype(int)  # Ensure unclustered cells remain with pl_id = 0
grid.drop(columns=["new_pl_id"], inplace=True)

# Save the updated shapefile
print(f"Saving output shapefile to {output_shapefile}...")
grid.to_file(output_shapefile, driver="ESRI Shapefile")
print(f"Output shapefile saved successfully to {output_shapefile}!")


# STEP 8: PL SHAPE ####################################################################### 

import os
import pandas as pd
import geopandas as gpd

# Define paths
input_shapefile = os.path.join(working_directory, "output_gis", "grid_with_final_pl_ids.shp")
output_shapefile = os.path.join(working_directory, "output_gis", "pl_shape.shp")
output_csv = os.path.join(working_directory, "output_data", "pl_data.csv")

# Ensure working directory is set
os.chdir(working_directory)
print("Working directory set to:", os.getcwd())

# Load the input shapefile
print("Loading input shapefile...")
grid = gpd.read_file(input_shapefile)
print("Input shapefile loaded successfully!")

# Debugging: Print available columns
print("Available columns in the input shapefile:", grid.columns)

# Check if 'pl_id' exists
if "pl_id" not in grid.columns:
    raise ValueError("The input shapefile does not contain a 'pl_id' column. Verify the input file.")

# Filter grid cells with positive pl_id
print("Filtering grid cells with positive pl_id...")
positive_pl_grid = grid[grid["pl_id"] > 0]

# Dissolve grid cells by pl_id
print("Dissolving grid cells by pl_id...")
dissolved = positive_pl_grid.dissolve(by="pl_id", aggfunc={"employment": "sum"})

# Calculate total area and centroids
print("Calculating total area and centroids...")
dissolved["total_area"] = dissolved.geometry.area  # Area in CRS units (e.g., square meters)
projected_centroids = dissolved.geometry.centroid  # Centroids in the current CRS

# Reproject to WGS84 for geographic centroids
dissolved = dissolved.to_crs("EPSG:4326")
dissolved["centroid_lat"] = dissolved.geometry.centroid.y
dissolved["centroid_lon"] = dissolved.geometry.centroid.x

# Prepare final output attributes
print("Preparing final attributes...")
dissolved = dissolved.reset_index()
dissolved.rename(columns={"employment": "total_employment", "pl_id": "pl_id"}, inplace=True)

# Save the shapefile
print(f"Saving shapefile to {output_shapefile}...")
dissolved.to_file(output_shapefile, driver="ESRI Shapefile")
print(f"Shapefile saved successfully to {output_shapefile}!")

# Save the CSV
print(f"Saving data to CSV at {output_csv}...")
dissolved[["pl_id", "total_employment", "total_area", "centroid_lat", "centroid_lon"]].to_csv(
    output_csv, index=False
)
print(f"CSV file saved successfully to {output_csv}!")

# STEP 9: FINAL MAP WITH OUTCOME #############################################################

# Import required libraries 
import subprocess
import sys

# Ensure matplotlib-scalebar is installed
try:
    from matplotlib_scalebar.scalebar import ScaleBar
except ModuleNotFoundError:
    print("matplotlib-scalebar not found. Installing...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "matplotlib-scalebar"])
    from matplotlib_scalebar.scalebar import ScaleBar

import os
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import FuncFormatter

# Define paths
working_directory = "P:/Algorithm/PL_python/DebuggingArena/GA"
input_shapefile_grid = os.path.join(working_directory, "output_gis", "grid_with_final_pl_ids.shp")
input_shapefile_pl = os.path.join(working_directory, "output_gis", "pl_shape.shp")
output_map = os.path.join(working_directory, "output_maps", "prime_location_map.png")

os.makedirs(os.path.dirname(output_map), exist_ok=True)

# Set working directory
os.chdir(working_directory)
print("Working directory set to:", os.getcwd())

# Load shapefiles
print("Loading shapefiles...")
grid = gpd.read_file(input_shapefile_grid)
pl_shape = gpd.read_file(input_shapefile_pl)
print("Shapefiles loaded successfully!")

# Ensure both are in the same CRS
if grid.crs != pl_shape.crs:
    print("Reprojecting shapefiles to a common CRS...")
    pl_shape = pl_shape.to_crs(grid.crs)

# Define cutoff values for employment and create a normalization
cutoff_values = [0, 0.5, 1.5, 50, 100, 250, 1000, 1e6]
norm = BoundaryNorm(cutoff_values, ncolors=256, clip=True)

# Function to format the color bar labels
def format_cbar_ticks(x, _):
    if x >= 1000:
        return f"{int(x/1000)}e3"
    return str(int(x))

# Create the map
print("Creating the map...")
fig, ax = plt.subplots(1, 1, figsize=(12, 10))

# Plot employment background
grid.plot(
    column="employment",
    cmap="Greys",
    norm=norm,
    ax=ax,
    legend=True,
    legend_kwds={
        "label": "Employment Levels",
        "orientation": "vertical",
        "shrink": 0.7
    },
    alpha=0.8,
)

# Adjust the color bar labels
cbar = ax.get_figure().axes[-1]  # Retrieve color bar axis
cbar.yaxis.set_major_formatter(FuncFormatter(format_cbar_ticks))

# Add very light grey grid lines
grid.boundary.plot(ax=ax, color="lightgrey", linewidth=0.2, alpha=0.5)

# Overlay prime locations with transparency and red fill (no outlines)
pl_shape.plot(ax=ax, color="red", alpha=0.5, edgecolor=None)

# Annotate prime locations in red
for idx, row in pl_shape.iterrows():
    centroid = row.geometry.centroid
    ax.text(
        centroid.x + 200, centroid.y + 200,
        str(int(row["pl_id"])),
        color="red",
        fontsize=12,
        ha="left",
        va="bottom",
        weight="bold"
    )

# Add scale bar
scalebar = ScaleBar(1, location="lower right", units="m", scale_loc="bottom", color="black")
ax.add_artist(scalebar)

# Finalize map
ax.set_title("Prime Location Map with Employment Levels", fontsize=16)
ax.set_axis_off()

# Save and show map
plt.savefig(output_map, dpi=300, bbox_inches="tight")
print(f"Map saved to {output_map}")
plt.show()

# ALL DONE #############################################################
# AABPL-toolkit-python (beta version)

(c) Gabriel M. Ahlfeldt, Thilo N. H. Albers, Kristian Behrens, [Max von Mylius](https://github.com/maximylius), Version 0.3.6.0, 2024-10





## About

This repository is part of the **[Toolkit of Prime Locations (AABPL)](https://github.com/Ahlfeldt/AABPL-toolkit/blob/main/README.md)**. It contains a Python version of the prime locations delineation algorithm developed by Ahlfeldt, Albers, and Behrens (2024). The algorithm uses arbitrary spatial point patterns as input and returns a gridded version of the data along with polygons of the delineated spatial clusters as outputs.

When using the algorithm in your work, **please cite Ahlfeldt, Albers, Behrens (2024): Prime locations. American Economic Review: Insights, forthcoming.**

<details>
<summary>Differences from the published version and replication notes</summary>

Note that while this implementation of the algorithm follows the same basic steps as the one used by Ahlfeldt, Albers, and Behrens (2024), it will not necessarily generate exactly the same results. The Python package is designed to enhance usability. There are subtle differences in the way counterfactual distributions are generated, establishments are assigned to grid cells, clusters are aggregated, and convex hulls are generated. Importantly, the current version of the algorithm samples from a bounding box built around the establishments input into the algorithm, whereas Ahlfeldt, Albers, and Behrens (2024) condition on the presence of employment. Therefore, the parameter values that need to be defined in the program syntax cannot be directly transferred from Ahlfeldt, Albers, and Behrens (2024).

We recommend that users find their own preferred values depending on the context and purpose of the clustering. We aim to allow for a user-specified sampling area so that users can, akin to Ahlfeldt, Albers, and Behrens (2024), exclude arbitrary areas when generating counterfactual establishment distributions. For replication of the results reported in Ahlfeldt, Albers, and Behrens (2024), we refer to the official replication directory.

</details>



## Installation

To install the Python package of the AABPL-toolkit, run the following command in your python environment in your terminal. 



`pip install aabpl`



If you are **new to Python**, you can download the Anaconda distrbution from [this website](https://www.anaconda.com/download). Then enter the command into the Anaconda Promt.



Alternatively you can also install it from within your python script:

```python

import subprocess, sys

subprocess.check_call([sys.executable, "-m", "pip", "install", 'aabpl', "--upgrade"])

```

If you use the ready-to-use file described below, the package will install automatically.



<details>

<summary>In case an error occurs at the installation...</summary>



with an erorr message like 'metadata-generation-failed', it is likely caused by incompatabile versions of `setuptools` and `packaging`. 

This can be fixed by upgrading `setuptools` and `packaging` to compatible versions:

```console

pip install --upgrade setuptools>=74.1.1

pip install --upgrade packaging>=22.0

```

Or by downgrading `setuptools`:

```console

pip install --upgrade setuptools==70.0.0

```



</details>







## Usage

```python
from pandas import read_csv
from aabpl import detect_cluster_cells, radius_search
```

All parameters are documented inline — no need to leave your script:

```python
import aabpl
aabpl.radius_search.params.r          # description of the r parameter
aabpl.detect_cluster_cells.params     # list all parameters
```

### Examples

```python
from pandas import read_csv
from aabpl import detect_cluster_cells, radius_sum

path_to_your_csv = 'input_data/hist_New_York.txt'
crs_of_your_csv  = "EPSG:4326"                       # coordinate system of lat/lon columns
pts = read_csv(path_to_your_csv, sep=",", header=None)
pts.columns = ["eid", "employment", "industry", "lat", "lon", "moved"]

# Result columns appended to pts:
#   employment_sum_15000         — radius-sum aggregate  (auto-named: {col}_{stat}_{r})
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

# Save outputs
df_clusters    = grid.save_cell_clusters(filename='output_gis/clusters',   file_format='shp')
df_sparse_grid = grid.save_sparse_grid( filename='output_gis/sparse_grid', file_format='shp')
pts.to_csv('output_data/pts_df_w_clusters.csv')

# Plots
grid.plot.clusters(  'output_maps/clusters_employment_995th')
grid.plot.vars(      filename='output_maps/employment_vars')
grid.plot.cluster_pts(filename='output_maps/employment_cluster_pts')
grid.plot.rand_dist( filename='output_maps/rand_dist_employment')

# Radius search only (no clustering) — exclude_self defaults to False for radius_sum/radius_search
grid = radius_sum(
    pts=pts,
    crs=crs_of_your_csv,
    r=15000,                            # search radius in metres; also accepts r=[500,750] or distance bands r=[(0,500,.004),(500,750,.0016)]
    c='employment',                     # column(s) to aggregate; c=['col1','col2'] for multiple; result appended as employment_sum_15000
    exclude_self=False,                 # False by default: each point's own value is included in its radius sum; set True to exclude self-contribution
    overwrite=True,
)
```





<details>
<summary><strong>Ready-to-use script</strong> — If you are new to Python, you may find it useful to execute the Example.py (or Example.ipynb) script saved in this folder...</summary>

### Ready-to-use script



If you are new to Python, you may find it useful to execute the [`Example.py`](https://github.com/Ahlfeldt/AABPL-toolkit-python/blob/main/Example.py) (or [`Example.ipynb`](https://github.com/Ahlfeldt/AABPL-toolkit-python/blob/main/Example.ipynb)) script saved in this folder. It will install the package, load the testing data set (we provide on example file in the [`input_data`](https://github.com/Ahlfeldt/AABPL-toolkit-python/tree/main/input_data) subfolder), generate clusters, and save various outputs to your working directory.  It should be straightforward to adapt the script to your data and preferred parameter values. 



You have many options for executing the `Example.py` script. One convenient option is to open the script in Sypder, a development environment that can be launched from the Anaconda Navigator. Spyder will automatically set the working directory to the folder to which you have copied the 'Example.py' file. If you name you name your input file `plants.txt` and save it in an  `input_data` subfolder, you will not have to make any adjustments to the script. For a first trial, we recommend that you just copy the `input_data` (with its content) to the same directory where you save `Example.py` file and then run the script from Spyder.



</details>

<details>
<summary><strong>Inputs</strong> — The compulsory input is a file containing spatial point pattern data (establishments, buildings, individuals, etc.) with geographic coordinates and an importance weight...</summary>

### Inputs



The **compulsory input** into the algorithm is a file containing spatial point pattern data. In the application by Ahlfeldt, Albers, and Behrens (2024), spatial points are establishments. However, these could also be individuals, buildings, or any other subjects or objects whose location can be referenced by geographic coordinates. The data file should contain geographic coordinates in standard decimal degrees and a variable that defines the importance of a subject or object. In the application by Ahlfeldt, Albers, and Behrens (2024), the importance is represented by the employment of an establishment. However, it could also be the productivity of a worker, the height of a building, or any weight that summarizes the importance of a data point. Of course, equal importance will be reflected by a uniform value.



In case you wish to use the above `Example.py` script without having to make any adjustments (except for setting your root directory), you should create a comma-separated file with exactly the same name and structure as the `plants.txt` file provided in this repository (this is just the renamed `prime_points_weighted_79.txt` file from the [AABPL-toolkit](https://github.com/Ahlfeldt/AABPL-toolkit/blob/main/DATA/GlobalCities/prime_points_weighted.zip)). Note that this exemplary input file **does not include variable names**. It includes variables in the following order (separated by commas):



- **identifier variable**: In our case, this is an establishment identifier. If you do not need this, you can set all values to 1.

- **importance weight**: In our case, this is predicted employment. If you want to use equal weights, you can set all values to 1.

- **category identifier**: In our case, this is the type of establishment (e.g., accounting, consulting, etc.). If you do not care, you can set all values to 1.

- **latitude**: Given in decimal degrees in the standard WGS1984 geographic coordinate system.

- **longitude**: Given in decimal degrees in the standard WGS1984 geographic coordinate system.

- **placebolder for another variable**: You can ignore it.





Variable names will then be assigned by the script. Of course, with some adjustments to the 'Example.py' script, you can also import data sets that already contain variable names. Just make sure that latitudes and longitudes are defined by variables named `lat` and `lon`. You can define the name of the variable representing your importance weights in the program syntax.



An **optional input** is a shapefile (or Shapely Polygon/MultiPolygon) that defines the sampling area of the counterfactual distribution, passed via the `sample_area` parameter. Ahlfeldt, Albers, and Behrens (2024) exclude residential and undevelopable areas. Such a shapefile could also restrict the sampling area for counterfactual spatial distributions to inhabitable areas or to areas zoned for the development of tall buildings. The parameter also accepts a method name string with optional inline parameters, e.g. `'buff_cells,min_pts=1'` or `'concave,concavity=0.5,buf=1000'`. Call `aabpl.resolve_sample_area.params()` at any time for a full list of methods and their parameters.



</details>

<details>
<summary><strong>Outputs</strong> — The package creates output_data, output_gis, and output_maps folders with CSVs, shapefiles, and maps...</summary>

### Outputs



The package will create the a number of folders in your working directory into which the outputs will be saved. File names are those specified in the `Example.py` file (you may choose different names). 



| Folder | File | Description |
|:---|:---|:---|
| output_data | `clusters.csv` | CSV file containing information on the final delineated clusters, including geographic coordinates in decimal degrees, a cluster id that corresponds to the rank in the distribution of total mass within the cluster (in our case employment), the number of cells within the cluster, the total area of the cluster (in square meters). You may choose another file name in the 'Example.py' script. |
| output_data | `grid_clusters.csv` | CSV file containing a gridded version of the data set, including groups of clustered grid cells identified by the cluster id, geographic coordinates in decimal degrees, and the total mass in the grid cell (in our case employment). You may choose another file name in the 'Example.py' script.   |
| output_data | `pts_df_w_clusters.csv` | CSV file containing the plants with the input data and, in addition, an identifier for the cluster to which a plant belongs. You may choose another file name in the 'Example.py' script. |
| output_gis | `grid_clusters.*` | Shapefile of the gridded data set including the same information as in  `grid_clusters.csv`. You may choose another file name in the 'Example.py' script. |
| output_gis | `clusters.*` | Shapefile of final output, i.e. aggregated clusters (in our case prime locations) along with the same information as in 'clusters.csv'. You may choose another file name in the 'Example.py' script.  |
| output_maps | `clusters_employment_995th.png` | Map showing the boundaries of the final output, i.e. clusters after aggregation (in our case to prime locations), with the density of the selected importance weight (in our case employment) in the background. You may choose another file name in the 'Example.py' script.  |
| output_maps | `employment_cluster_pts.png` | Map showing the plants and how clustered they are. You may choose another file name in the 'Example.py' script.  |
| output_maps | `rand_dist_employment.png` | Technical output to inform the choice of the p-value. You may choose another file name in the 'Example.py' script.  |



Other outputs can be generated by activating the respective lines (by removing the '#') in the 'Exmaple.py' script.



</details>

<details>
<summary><strong>Recommendations</strong> — The default parameter values are calibrated for a dataset covering roughly a large city...</summary>

### Recommendations



The results of the clustering algorithm naturally depend on the chosen parameter values. The recommended baseline parameter values have been tested for areas that in terms of geography coverage conform roughly to a large city. For example, if you obtain establishments as point-pattern data for an area that covers roughly New York City (the New York grid in the Global Cities sample in the Prime Locations research paper), you will likely obtain two prime locations (in Midtown and Wallstreet). If your point-pattern data covers a much larger area (e.g. the state of New York), there will many emty areas that affect the counterfactual distributions. Dense places will be in relative terms denser, and, hence, a greater p-value might be required to obtain the same to two prime locations (else, the algorithm may return many more prime locations). You would also have to use more than 100,000 points to have decent coverage of such a large area.

</details>

## User-facing functions



All functions are available directly on the `aabpl` module after `import aabpl`. Full parameter documentation is available via `help(aabpl.<function>)` or your IDE.



| Function | Description |
|:---|:---|
| **`radius_search(pts, crs, r, c, stat, ...)`** | **Core function.** For every point in `pts`, aggregates values of neighbouring points within radius `r` (or distance bands). Adds the result as a new column. Supports `stat` in `{sum, count, mean, variance, std, cv, skewness, kurtosis}`. |
| **`detect_cluster_cells(pts, crs, r, c, ...)`** | **Core function.** Full pipeline: runs `radius_search`, builds a null distribution from random points, delineates contiguous clustered cells into cluster polygons. Returns a `Grid` object; polygons at `grid.clustering`. |
| `detect_cluster_pts(pts, crs, r, c, ...)` | Labels each point as clustered or not. Same pipeline as `detect_cluster_cells` but skips the output grid and polygon steps. |
| `detect_cluster_cells_from_labeled_pts(pts, crs, r, ...)` | Delineates cluster polygons from points with a pre-existing cluster label column, skipping the radius search and null distribution. |
| `infer_sample_area_from_pts(pts, grid, ...)` | Derives the valid sample area polygon from the point pattern. Used internally; available for inspection. |
| `draw_random_coords(n_pts, sample_area, crs, ...)` | Draws `n_pts` random coordinate pairs. `sample_area` accepts a Shapely Polygon/MultiPolygon or a plain coordinate list; coordinates outside it are rejected. Set `crs` to reproject the geometry from a geographic CRS (e.g. `'EPSG:4326'`) into the best UTM zone automatically — the same reprojection used internally by `detect_cluster_pts`. Pass `sample_area=None` with a custom `coord_generator(n, rng)` to accept all produced coordinates. Returns a two-column DataFrame ready to pass as `null_distribution` to `detect_cluster_pts` / `detect_cluster_cells`. |
| `radius_sum(pts, crs, r, c, ...)` | Shorthand for `radius_search(..., stat='sum')`. |
| `radius_count(pts, crs, r, c, ...)` | Shorthand for `radius_search(..., stat='count')`. |
| `radius_mean(pts, crs, r, c, ...)` | Shorthand for `radius_search(..., stat='mean')`. |
| `radius_variance(pts, crs, r, c, ...)` | Shorthand for `radius_search(..., stat='variance')`. |
| `radius_std(pts, crs, r, c, ...)` | Shorthand for `radius_search(..., stat='std')`. |
| `radius_cv(pts, crs, r, c, ...)` | Shorthand for `radius_search(..., stat='cv')` (coefficient of variation). |
| `radius_skewness(pts, crs, r, c, ...)` | Shorthand for `radius_search(..., stat='skewness')`. |
| `radius_kurtosis(pts, crs, r, c, ...)` | Shorthand for `radius_search(..., stat='kurtosis')`. |

<details>
<summary><strong>Grid object methods</strong> — methods available on the <code>grid</code> object returned by <code>detect_cluster_cells</code></summary>

### Grid object methods

The `grid` object returned by `detect_cluster_cells` exposes the following methods:

| Method | Description |
|:---|:---|
| `grid.info()` | Print grid layout, cell size, bounds, cluster summary, and available plot/save methods. |
| `grid.aggregate_pts_to_output_cells(pts, val_cols, agg)` | Aggregate additional point columns into existing grid cells without rerunning the full pipeline. Coordinates must be in `grid.proj_crs`. Use `add_to_exports=True` to include the new columns in `save_sparse_grid` output. |
| `grid.plot.clusters(filename)` | Map of cluster polygons overlaid on cell aggregates. |
| `grid.plot.cell_aggregates(filename)` | Heatmap of raw aggregated values per output cell. |
| `grid.plot.vars(filename)` | Scatter of source points coloured by any column value. |
| `grid.plot.rand_dist(filename)` | Observed vs null distribution — use to calibrate `k_th_percentile`. |
| `grid.plot_sample_area(filename)` | Map of the sampling region used for the null distribution. |
| `grid.save_cell_clusters(filename, file_format)` | Export cluster polygons as shapefile / GeoJSON / GeoParquet / CSV. |
| `grid.save_sparse_grid(filename, file_format)` | Export non-empty grid cells with cluster IDs and aggregates. |
| `grid.create_sparse_grid_df()` | Return non-empty grid cells as a GeoDataFrame for custom analysis without saving to disk. |
| `grid.clustering.by_column['col']` | Access individual cluster objects directly: `.cells`, `.centroid`, `.geometry`, `.area`. |

All plot methods accept `show=False` to suppress the pop-up, `display_dpi` to control screen resolution, and `save_kwargs={'dpi': 300}` to control saved-file resolution.

</details>



### Selected files



| | File | Description |
|:---|:---|:---|
| [-](https://github.com/Ahlfeldt/ABRSQOL-toolkit) | `AABPL-Codebook.pdf` | **Codebook** laying out the **structure of the delineation algorithm in pseudo code**. |



# References 



Ahlfeldt, Albers, Behrens (2024): Prime locations. American Economic Review: Insights, forthcoming.



---



## Algorithm details



*This section documents the internal mechanics of the algorithm; it is not needed for normal usage.*



### Cluster detection pipeline



`detect_cluster_cells` (and the point-level `detect_cluster_pts`) proceed in four stages, each building on the previous.



#### Stage 1 — Radius aggregation



For every point *i*, `radius_search` computes the sum (or count/mean/variance etc.) of a variable across all other points within a circle of radius *r*:



```

agg_i = Σ_{j: d(i,j) ≤ r, j ≠ i}  value_j

```



This produces one number per point that reflects local concentration — a point with a high aggregate is surrounded by many (or high-valued) neighbours. The grid and offset-region machinery described below is what makes this step fast. Edge effects near the study-area boundary are corrected by weighting each aggregate by the inverse of the fraction of the circle that falls within the valid sampling area.



#### Stage 2 — Null distribution



To decide whether an aggregate is *significantly* elevated, a **null distribution** is built by drawing random points from the sample area and running `radius_search` on them with the same radius and source points. The **k-th percentile** of this distribution becomes the cluster threshold `τ`; a point is labelled clustered if `agg_i > τ`. Pass an integer to `null_distribution=` to control how many random points are drawn (default 100 000), or pass an (N, 2) array/DataFrame with x in the first column and y in the second (projected CRS) to supply your own reference coordinates. Because the null distribution reflects the study-area geometry, it automatically accounts for irregular boundaries and gaps.



#### Stage 3 — Cell-level delineation



`detect_cluster_cells` aggregates radius sums onto a regular output grid (default cell size `r/3`) and applies the threshold cell-by-cell. Contiguous groups of cells that all exceed the threshold form **raw cluster patches**. Adjacent patches are merged when close enough to represent the same concentration:



- **Queen / rook contiguity** — cells sharing a corner (queen) or an edge (rook) are joined.

- **Centroid-distance merging** — two patches merge if their centroids are within `centroid_dist_threshold` (default `r × 10/3`) and their borders within `border_dist_threshold` (default `r × 4/3`).



Clusters whose total aggregate falls below `min_cluster_share_after_contingency` of the dataset total are dropped. If `make_convex=True` (default), all cells inside each cluster's convex hull are added, filling internal gaps.



#### Stage 4 — Cluster polygons



Each final cluster is dissolved from its constituent cells into a single polygon, available at `grid.clustering` and exportable via `grid.save_cell_clusters`.



```
pts ──► radius_search ──► agg_i per point
                                │
         n random pts ──► agg_j per random point ──► k-th percentile = τ
                                │
                          agg_i > τ ? ──► cluster_i (point label)
                                │
                     aggregate to output grid cells
                                │
                     contiguous cell patches ──► merge ──► convexify
                                │
                         cluster polygons
```



<details>
<summary><strong>Grid and offset regions / Adaptive spacing / Nest depth</strong> — Internal implementation details; not needed for normal usage.</summary>

### Grid and offset regions



The algorithm avoids O(n²) point-by-point distance checks by overlaying a regular grid on the target points and pre-aggregating each variable into cell sums. A radius search then reduces to summing over the grid cells that fall within the search circle — O(cells) rather than O(points).



The central insight is that which neighbouring cells a point's search circle **contains or overlaps depends only on where the point sits within its own cell** — not on its absolute position in space. Two source points in different parts of the map but at the same relative position within their respective cells will always have the same circle“cell neighbourhood topology.



The algorithm exploits this by expressing each source point as a **sub-cell offset** — its displacement `(dx, dy)` from its cell centre. The set of all possible offsets is partitioned into **offset regions**: areas within the cell bounded by the grid lines and the arcs where the search circle crosses cell boundaries, such that every point inside a given region shares exactly the same set of fully-contained and potentially-overlapping neighbouring cells. This partition is precomputed once from the geometry of circle“grid intersections.



At search time, assigning a source point to its offset region requires only a modulo to obtain the sub-cell offset, followed by a region classification against the precomputed arc boundaries. From there, the neighbourhood lookup is a direct table read: the precomputed entry lists which cells are **fully contained** (contributing their full aggregated sum) and which are **boundary cells** (partially overlapping, contributing a fractional weight). No per-point distance checks are needed.



![Offset regions within a grid cell](docs/img/illustrate_offset_regions.png)

*Each panel shows one offset region (shaded, left) and the corresponding set of fully-contained cells (green) and boundary cells (pink) that apply to all points within that region (right).*



Once a source point's neighbourhood is resolved, the search circle is applied:



![Radius search around an example point](docs/img/illustrate_radius_search.png)

*Green cells are fully contained — their pre-aggregated sums are added directly. Orange cells overlap the boundary and require individual distance checks (red crosses = outside radius, black dots = inside). Grey crosses fall in cells entirely outside the circle and are never visited.*



### Adaptive grid spacing



The grid spacing is not fixed — it is chosen automatically relative to the search radius `r`. A coarser grid (large spacing) means fewer cells to traverse but more points per boundary cell; a finer grid means more cells but sparser boundary zones. The algorithm selects the spacing as a dimensionless ratio `r / spacing` from a set of candidates at topology breakpoints — values where the circle“cell intersection pattern changes structurally — and jointly optimises over nest depth using a fitted timing model. The result is a spacing that minimises predicted runtime given the dataset size, point density, and spatial distribution.



### Boundary cells and nest depth

Cells that lie entirely inside the search radius are aggregated in bulk using nested cell sums — no individual point lookups needed. Cells that straddle the boundary cannot be bulk-aggregated; their points are checked individually against the radius.

The `nest_depth` parameter controls how aggressively boundary cells are pre-aggregated before that individual check. At `nest_depth=0` every point in a boundary cell is checked one by one. At `nest_depth=d` each boundary cell is recursively subdivided into a 2^d × 2^d sub-grid: sub-cells fully inside the radius are bulk-summed, and only the remaining sub-cells (a thin ring near the circle edge) fall through to point-level checks. Higher nest depth means fewer individual point lookups at the cost of more sub-cell traversals. The optimal value depends on point density and cell size, which is why `nest_depth` is chosen jointly with grid spacing by the adaptive timing model.

</details>

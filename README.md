# AABPL-toolkit-python

(c) Gabriel M. Ahlfeldt, Thilo N. H. Albers, Kristian Behrens, [Max von Mylius](https://github.com/maximylius), Version 0.1.0, 2024-10

## About

This repository is part of the **[Toolkit of Prime Locations (AABPL)](https://github.com/Ahlfeldt/AABPL-toolkit/blob/main/README.md)**. It contains a Python implementation of the prime locations delineation algorithm developed by Ahlfeldt, Albers, and Behrens (2024).

The package takes an arbitrary spatial point pattern as input and produces:
- **Radius aggregates** — for each point, the sum/count/mean of a variable across all neighbouring points within a given search radius
- **Spatial clusters** — contiguous groups of grid cells whose radius aggregate significantly exceeds a null distribution of randomly distributed points

The algorithm is designed for large datasets (100k+ points) and uses a vectorised grid-based approach that avoids O(n²) brute-force distance checks.

> **Note:** While this implementation follows the same basic steps as Ahlfeldt, Albers, and Behrens (2024), it does not necessarily reproduce exactly the same results. There are subtle differences in how counterfactual distributions are generated, establishments are assigned to grid cells, and clusters are aggregated. Parameter values cannot be transferred directly from the original paper.

When using this package in your work, **please cite**:

> Ahlfeldt, Albers, Behrens (2024): Prime locations. *American Economic Review: Insights*, forthcoming.

---

## Installation

```console
pip install aabpl
```

<details>
<summary>Troubleshooting installation errors</summary>

If you see `metadata-generation-failed`, upgrade `setuptools` and `packaging`:

```console
pip install --upgrade "setuptools>=74.1.1" "packaging>=22.0"
```

Or from within Python:

```python
import subprocess, sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "aabpl", "--upgrade"])
```

</details>

---

## Quick start

```python
from pandas import read_csv
from aabpl.main import radius_search, detect_cluster_cells

pts = read_csv('my_data.csv')
# pts must have columns for longitude ('lon') and latitude ('lat') in decimal degrees

# --- Radius aggregation only ---
grid = radius_search(pts, crs='EPSG:4326', r=750, c='employment')
# pts now has a new column 'employment_750m' with the employment sum within 750 m of each point
grid.plot.vars(filename='output_maps/employment_750m')

# --- Full cluster delineation ---
grid = detect_cluster_cells(pts, crs='EPSG:4326', r=750, c='employment')
grid.plot.clusters(filename='output_maps/clusters')
```

---

## Functions

All user-facing functions live in `aabpl.main`.

### `radius_search(pts, crs, r, c, ...)`

Aggregates a variable from neighbouring points within a search radius for every point in `pts`. Results are appended in-place to `pts` as new column(s) `{c}{sum_suffix}`.

| Parameter | Type | Description |
|---|---|---|
| `pts` | `DataFrame` | Input points. Must contain longitude and latitude columns. Modified in-place. |
| `crs` | `str` | CRS of the coordinates, e.g. `'EPSG:4326'`. |
| `r` | `float` | Search radius in metres. |
| `c` | `str` or `list` | Column name(s) to aggregate. If empty, points are counted. |
| `agg` | `str` | Aggregation method: `'sum'` (default), `'count'`, or `'mean'`. |
| `exclude_pt_itself` | `bool` | Subtract each point's own value from its aggregate (default `True`). |
| `pts_target` | `DataFrame` | Points to aggregate over. If `None`, same as `pts` (default `None`). |
| `sum_suffix` | `str` | Suffix for result columns. Defaults to `'_{r}m'`, e.g. `'_750m'`. |
| `x` / `y` | `str` | Column names for longitude / latitude (default `'lon'` / `'lat'`). |
| `weight_valid_area` | `str` | Edge-effect correction: `'estimate'` (fast) or `'precise'` (exact). `None` disables it. |
| `sample_area` | `str` or geometry | Area for valid-area weighting. See [Sample area](#sample-area) below. |
| `proj_crs` | `str` | Internal metric CRS. `'auto'` picks the UTM zone automatically. |
| `keep_cols` | `bool` | Keep intermediate processing columns in `pts` (default `False`). |
| `silent` | `bool` | Suppress progress output (default `False`). |

**Returns:** `grid` — a `Grid` object. Aggregated values are written directly into `pts`.

**Convenience wrappers** with the same signature (minus `agg`):

```python
radius_sum(pts, crs, r, c, ...)    # agg='sum'
radius_count(pts, crs, r, c, ...)  # agg='count'
radius_mean(pts, crs, r, c, ...)   # agg='mean'
```

---

### `detect_cluster_pts(pts, crs, r, c, ...)`

Identifies **point-level** clusters: runs `radius_search` then labels each point as clustered or not based on a percentile threshold of a random null distribution. Appends boolean column(s) `{c}{cluster_suffix}` to `pts`.

| Parameter | Type | Description |
|---|---|---|
| `pts` | `DataFrame` | Input points. |
| `crs` | `str` | CRS of coordinates. |
| `r` | `float` | Search radius in metres. |
| `c` | `str` or `list` | Column(s) to aggregate. |
| `k_th_percentile` | `float` | Percentile of random distribution a point must exceed to be labelled clustered (default `99.5`). |
| `n_random_points` | `int` | Random points drawn to build the null distribution (default `100000`). |
| `sample_area` | `str` or geometry | Area for random point sampling (default `'buffered_cells'`). |
| `random_seed` | `int` | Seed for reproducibility (default `None`). |

**Returns:** `grid` — a `Grid` object. Boolean cluster columns are appended to `pts`.

---

### `detect_cluster_cells(pts, crs, r, c, ...)`

Identifies **cell-level** clusters: delineates contiguous groups of grid cells whose radius aggregate exceeds the null distribution threshold, then merges and optionally convexifies them into spatial cluster polygons.

All parameters from `detect_cluster_pts` apply, plus:

| Parameter | Type | Description |
|---|---|---|
| `queen_contingency` | `int` | Merge neighbouring clustered cells (including diagonals) into the same cluster (default `1`). |
| `rook_contingency` | `int` | Merge horizontally/vertically neighbouring cells (default `1`). |
| `centroid_dist_threshold` | `float` | Maximum centroid distance for merging clusters (default `r*10/3`). |
| `border_dist_threshold` | `float` | Maximum border distance for merging clusters (default `r*4/3`). |
| `make_convex` | `bool` | Add all cells within each cluster's convex hull to it (default `True`). |
| `min_cluster_share_after_contingency` | `float` | Minimum share of total to retain a cluster after contingency merging (default `0.05`). |

**Returns:** `grid` — a `Grid` object with cluster polygons at `grid.clustering`.

---

## The Grid object

All three functions return a `Grid` object that exposes plots and saved outputs.

### Plots

```python
grid.plot.vars(filename='output_maps/employment_750m')       # radius aggregate map
grid.plot.clusters(filename='output_maps/clusters')          # cluster polygon map
grid.plot.cluster_pts(filename='output_maps/cluster_pts')    # clustered points map
grid.plot.rand_dist(filename='output_maps/rand_dist')        # null distribution plot
```

### Saving outputs

```python
# Cluster polygons (shapefile + CSV)
grid.save_cell_clusters(filename='output_gis/clusters',    file_format='shp')
grid.save_cell_clusters(filename='output_data/clusters',   file_format='csv')

# Non-empty grid cells with aggregate values
grid.save_sparse_grid(filename='output_gis/sparse_grid',   file_format='shp')
grid.save_sparse_grid(filename='output_data/sparse_grid',  file_format='csv')

# Full grid including empty cells (can be large)
grid.save_full_grid(filename='output_data/full_grid',      file_format='csv')
```

---

## Sample area

The `sample_area` parameter controls where random null-distribution points are drawn. It accepts a string shorthand or a Shapely geometry:

| Value | Description |
|---|---|
| `'buffered_cells'` | Non-empty grid cells plus a radius-sized buffer (default) |
| `'concave'` | Concave hull around points |
| `'convex'` | Convex hull around points |
| `'buffer'` | Buffer around individual points (slow for large datasets) |
| `'bounding_box'` | Axis-aligned bounding box |
| `'grid'` / `None` | Full grid extent |
| Shapely geometry | Custom polygon (must be in the metric CRS) |

For finer control, build the area explicitly with `infer_sample_area_from_pts`:

```python
from aabpl.sample_area import infer_sample_area_from_pts

sample_area = infer_sample_area_from_pts(
    pts=pts, x='lon', y='lat',
    hull_type='concave',
    concavity=0.2,
    buffer=750,
    tolerance=750,
    plot_sample_area=True,
)

grid = detect_cluster_cells(pts, crs='EPSG:4326', r=750, c='employment',
                             sample_area=sample_area)
```

---

## Input data format

The minimum required input is a `pandas.DataFrame` with longitude and latitude columns (by default named `'lon'` and `'lat'`) in decimal degrees (WGS84 / EPSG:4326). Any additional numeric column can be passed as `c` for aggregation.

Example using the bundled test data:

```python
from pandas import read_csv
pts = read_csv('input_data/hist_New_York.txt', sep=',', header=None)
pts.columns = ['eid', 'employment', 'industry', 'lat', 'lon', 'moved']
```

If your coordinate columns have different names, pass `x='my_lon', y='my_lat'` to any function.

---

## Full example

```python
import os
from pandas import read_csv
from aabpl.main import detect_cluster_cells
from aabpl.sample_area import infer_sample_area_from_pts

# Folders
os.makedirs('output_data', exist_ok=True)
os.makedirs('output_gis',  exist_ok=True)
os.makedirs('output_maps', exist_ok=True)

# Load data
pts = read_csv('input_data/hist_New_York.txt', sep=',', header=None)
pts.columns = ['eid', 'employment', 'industry', 'lat', 'lon', 'moved']

# Build sample area (optional — controls where random points are drawn)
sample_area = infer_sample_area_from_pts(
    pts=pts, x='lon', y='lat',
    hull_type='concave', concavity=0.2,
    buffer=750, tolerance=750,
)

# Detect clusters
grid = detect_cluster_cells(
    pts=pts,
    crs='EPSG:4326',
    r=750,
    c='employment',
    exclude_pt_itself=True,
    sample_area=sample_area,
    k_th_percentile=99.5,
    n_random_points=100000,
    random_seed=0,
    queen_contingency=1,
    centroid_dist_threshold=2500,
    border_dist_threshold=1000,
    min_cluster_share_after_contingency=0.05,
    make_convex=True,
)

# Save outputs
grid.save_cell_clusters(filename='output_gis/clusters',  file_format='shp')
grid.save_cell_clusters(filename='output_data/clusters', file_format='csv')
grid.save_sparse_grid(filename='output_gis/sparse_grid', file_format='shp')
pts.to_csv('output_data/pts_with_clusters.csv', index=False)

# Plots
grid.plot.clusters(filename='output_maps/clusters')
grid.plot.vars(filename='output_maps/employment_750m')
grid.plot.rand_dist(filename='output_maps/rand_dist')
```

See [`Example.py`](Example.py) and [`Example.ipynb`](Example.ipynb) for a ready-to-run version.

---

## References

Ahlfeldt, Albers, Behrens (2024): Prime locations. *American Economic Review: Insights*, forthcoming.

---

## Performance scaling

The following is based on benchmark testing across varied dataset sizes, search radii, and spatial configurations. Timings reflect wall time on a single CPU core.

**Search phase** scales approximately as:

```
time ∝ n_src^0.61 × n_tgt^0.69
```

**Aggregation phase** scales approximately as:

```
time ∝ n_src^0.5–0.9 × n_tgt^0.83
```

The aggregation n_src exponent is regime-dependent: it rises from ~0.5 at small datasets to ~0.9 at n_src > 500k, reflecting a log-linear relationship between source count and per-lookup overhead. Both phases scale sub-linearly with n_src and n_tgt — the grid structure means neither phase requires visiting all target points for each source point individually.

**Comparison to KD-tree / Ball-tree:**

At low density (few expected points per search circle), KD-trees are competitive due to effective tree pruning. At high density — the typical use case for urban point patterns — the grid-based approach has a structural advantage: KD-trees degrade toward O(n_src × ppc) as they must iterate every result point individually, while AABPL's aggregation step operates on pre-summed grid cells and is largely insensitive to how many points fall within each cell. The crossover is roughly at ppc > 5–10 expected points per search circle.

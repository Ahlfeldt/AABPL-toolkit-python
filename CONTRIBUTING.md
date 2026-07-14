# Contributing

## Repository layout

## Folder structure

| Folder | File | Description |
|:---|:---|:---|
| `aabpl/` | `main.py` | All user-facing functions (see table above). |
| `aabpl/` | `config.py` | Runtime configuration flags (`FIXED_SPACING_RATIO`, `FIXED_NEST_DEPTH`, `DISK_REGION_CACHE_MAXSIZE`, etc.). |
| `aabpl/radius_search/` | `grid_class.py` | `Grid` object: holds reprojected points, the search grid, aggregated cell sums, and cluster output. Auto-selects spacing and nest depth. |
| `aabpl/radius_search/` | `disk_aggregation.py` | Inner search loop: contain kernel (hash lookup of cell aggregates) and overlap kernel (per-point distance checks for boundary cells). |
| `aabpl/radius_search/` | `disk_region_geometry.py` | Precomputes and caches the disk region geometry (contained/overlapping cell offsets) for each `(spacing_ratio, nest_depth)` pair. |
| `aabpl/radius_search/` | `spacing_topology.py` | Timing model and auto-selector: `choose_spacing_and_depth`, `predict_timing`, `predict_geo_build_time`. |
| `aabpl/radius_search/` | `point_grid_assignment.py` | Assigns target points to grid cells and pre-aggregates cell sums and sub-cell trees used by the search. |
| `aabpl/radius_search/` | `point_region_assignment.py` | Assigns each source point to its offset region (sub-cell position determining which neighbours are contained vs overlapping). |
| `aabpl/radius_search/` | `region_classes.py` | Geometry of individual disk regions: intersection logic, split-with-edge, contained/overlap classification. |
| `aabpl/radius_search/` | `null_distribution.py` | Draws uniform random points within the sample area and runs `radius_search` on them to build the null distribution. |
| `aabpl/radius_search/` | `clusters.py` | Cell-level cluster delineation: contiguity merging, centroid-distance merging, convex-hull fill, polygon output. |
| `aabpl/radius_search/` | `study_area.py` | Infers the valid study area polygon from the point pattern; computes valid-area weights for boundary correction. |
| `aabpl/radius_search/` | `optimal_grid_spacing.py` | Legacy grid-spacing analysis utilities (analytical breakpoints). |
| `aabpl/illustrations/` | `*.py` | Visualisation helpers for the algorithm (offset regions, disk, nested grid, sample area, variable plots). Not required for normal use. |
| `aabpl/utils/` | `crs_transformation.py` | CRS reprojection helpers (WGS-84 ↔ local metric CRS). |
| `aabpl/utils/` | `grid_aggregate.py` | `aggregate_to_grid`: aggregates point values onto output grid cells. |
| `aabpl/utils/` | `misc.py`, `precision.py`, `progress.py`, others | Small shared utilities (progress bars, floating-point helpers, polygon edge counting, etc.). |
| `aabpl/testing/` | `run_all_tests.py` | End-to-end test suite covering correctness across stat types, nest depths, and edge cases. |


## Running the tests

```
python aabpl/testing/run_all_tests.py
```
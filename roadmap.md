# Roadmap

## Documentation
- Emphasise that a local CRS should be chosen to minimise projection error; accuracy degrades over large areas (city level: good, country level: less accurate)

## User-facing grid spacing refactor

**Goal:** All user-facing outputs — plots, `Grid` attributes, column names — should reflect the **user-specified output cell size** (`spacing`, default `r/3`), not the internal search-grid spacing chosen automatically for computation efficiency.

**Terminology used here:**
- `spacing` — the public, user-facing output cell size (what the user sets; default `r/3`). After the refactor this is `grid.spacing`.
- `_search_spacing` — the internal cell size selected by `choose_spacing_and_depth` from `CANDIDATE_SPACINGS_BREAKPOINTS`. Always `<= spacing`. Never exposed to users.

**Current state:** `Grid.__init__` already stores both values (lines 174, 202 of `grid_class.py`) and has partial infrastructure (`output_x_steps`, `output_y_steps`, `assign_output_cell_ids`, `aggregate_pts_to_output_cells`). The problem is that almost every downstream consumer still reads `grid.spacing` (the internal value) instead of `grid.output_spacing`.

---

### 1  Public API — `radius_search` / `detect_cluster_pts`

- Add a `spacing` parameter (default `r / 3`); document it as "output cell size in the same unit as `r`".
- Pass it into `Grid.__init__` as `output_spacing=spacing`.
- Validate `spacing >= _search_spacing` and raise a clear error if violated (finer output than the search grid is undefined).
- `Grid.__init__` already accepts `output_spacing` and `output_spacing_y` — wire these up from the public param.

### 2  `Grid` object attribute rename

- Rename `grid.spacing` (currently internal) → `grid._search_spacing`.
- Rename `grid.output_spacing` → `grid.spacing` so `grid.spacing` is always what the user set.
- Keep `grid.output_spacing_y` / `grid.output_x_steps` / `grid.output_y_steps` working (just adjust the source they read from after the rename).
- `grid.nest_depth`, `grid.nest_height`, `grid.ref_lvl` remain unchanged — they belong to `_search_spacing`.
- Update every internal use of `grid.spacing` (for the search computation) to `grid._search_spacing`. Files affected:
  - `grid_class.py` — `total_bounds` padding, `x_steps`, `y_steps`, `get_cell_centroid`, `get_cell_poly`, `get_cell_bounds` lambdas
  - `point_grid_assignment.py` — floor-division row/col assignment, offset columns
  - `point_region_assignment.py` — `offset_x2`/`offset_y2`, scaled triangle offsets, `r/grid.spacing`
  - `sample_area.py` — `sample_col_min/max/row_min/max`, `sample_grid_bounds`
  - `null_distribution.py` — random point coordinate generation, `cell_width`/`cell_height`
  - `disk_aggregation.py` — cached `grid_spacing = grid.spacing`
  - `disk_region_geometry.py` — cache key `r / grid_spacing`, subcell geometry
  - `disk_search_state.py` — `nest_depth` retrieval
  - `plot_disk.py` — cell centre and subcell rectangle coordinates

### 2b  Grid dimension attributes — rename for clarity

`Grid.__init__` currently stores search-grid dimensions under names that give no hint they belong to the internal grid:

| Current name | Belongs to | Rename to |
|---|---|---|
| `grid.spacing` | internal search | `grid._search_spacing` |
| `grid.x_steps` | internal search | `grid._search_x_steps` |
| `grid.y_steps` | internal search | `grid._search_y_steps` |
| `grid.row_ids` | internal search | `grid._search_row_ids` |
| `grid.col_ids` | internal search | `grid._search_col_ids` |
| `grid.n_cells` | internal search | `grid._search_n_cells` |
| `grid.output_spacing` | user-facing | `grid.spacing` |
| `grid.output_spacing_y` | user-facing | `grid.spacing_y` |
| `grid.output_x_steps` | user-facing | `grid.x_steps` |
| `grid.output_y_steps` | user-facing | `grid.y_steps` |

After the rename, `grid.spacing`, `grid.x_steps`, `grid.y_steps` are always the user-facing output grid. Internal consumers (radius search, disk geometry, point assignment, null distribution) all switch to `grid._search_*`.

Also audit every place that reads `n_xsteps` / `n_ysteps` (local variables in `__init__`) and verify it uses the correct grid's step count — the internal and output grids have different numbers of rows and columns whenever `spacing != _search_spacing`.

### 3  Column naming

- `grid.output_row_name` / `grid.output_col_name` are set in `main.py` with TODO comments noting they must match the output spacing once that is tracked.
- After step 2: derive column names from `spacing` (e.g. `row_500m`, `col_500m` when `spacing=500`), or keep the current name and document what spacing it corresponds to.
- The existing `assign_output_cell_ids()` in `grid_class.py` already handles the case where output spacing differs from search spacing — it creates `'out_' + row_name` columns. Remove the TODO once wired up.
- Ensure round-trip: a user can reconstruct which output cell a point belongs to from the column values + `grid.spacing`.

### 4  `sample_area.py` — sample grid bounds

- `sample_col_min/max`, `sample_row_min/max`, and `sample_grid_bounds` are computed using `grid.spacing` (internal).
- After step 2 these should use `grid._search_spacing` (they describe the *search* grid extent, not the output grid).
- Add a parallel `grid.output_sample_grid_bounds` computed from `grid.spacing` (the output cell size) for use in plots.

### 5  `null_distribution.py` — random point sampling

- Random points are placed on a jittered grid using `grid.spacing` as cell size (lines 195, 295, 298).
- This should remain `_search_spacing` — the null distribution is computed at the search resolution, not the output resolution. No change needed here, but verify the downstream aggregation that *uses* the null distribution respects output spacing.

### 6  `detect_cluster_cells` — aggregation binning

- Currently groups points into `grid.spacing`-sized cells. Change to `grid.spacing` (output, after rename) so the output raster matches what the user requested.
- `grid_aggregate.py` already accepts an `output_spacing` parameter — wire this through.
- This is the main computational change.

### 7  Plots — use output spacing throughout

All illustration files that convert between grid indices and coordinates currently use `grid.spacing` (internal). After the rename they will automatically read the correct value, but audit each:
- `distribution_plot.py` lines 247–250: index computation for the sample-area overlay raster — switch to `grid._search_spacing` (these are search-grid indices) or to `output_sample_grid_bounds` (step 4).
- `plot_pt_vars.py`, `plot_sample_area.py`: same pattern, same fix.
- `plot_disk.py`: cell geometry for diagnostic disk plots — stays `_search_spacing` (these show the internal search structure, not output cells).
- `plot_grid.py` / `plot_grid_spacing.py`: any title or axis label that mentions cell size should read from `grid.spacing` (the public output value).

### 8  Documentation

- Docstring on `Grid.__init__`: explain the two-level grid (search vs. output), what `spacing` controls, and why `_search_spacing` is chosen automatically.
- Docstring on `radius_search` / `detect_cluster_pts`: document `spacing` parameter with the `r/3` default and note that finer values reduce output resolution but never affect accuracy.
- User-facing docstring note: "`grid.spacing` is always the output cell size you requested. The internal search grid may use a different (finer or equal) cell size for efficiency; this is not exposed."
- `CANDIDATE_SPACINGS_BREAKPOINTS` in `spacing_topology.py`: add a comment explaining these are dimensionless `r / _search_spacing` ratios, not user-visible.

### Suggested implementation order

1. Rename `grid.spacing` → `grid._search_spacing` and `grid.output_spacing` → `grid.spacing` throughout. Fix all internal consumers in one pass (step 2 + step 7 partial).
2. Add `spacing` parameter to `radius_search` / `detect_cluster_pts`; thread into `Grid.__init__` (step 1).
3. Wire `assign_output_cell_ids()` and resolve `output_row_name`/`output_col_name` TODOs (step 3).
4. Add `output_sample_grid_bounds` to `sample_area.py`; fix plot index computations (steps 4 + 7).
5. Fix `detect_cluster_cells` aggregation to bin at output spacing (step 6).
6. Add input validation (step 1).
7. Write docstrings (step 8).

---

## Open items

### Correctness / precision
- Validate user inputs at public API boundary

### Performance
- Investigate line-based comparisons as alternative to bilateral point distance checks
- Benchmark super-cell approach (e.g. 3×3 macro-cells) to reduce dict lookups at fine grid sizes


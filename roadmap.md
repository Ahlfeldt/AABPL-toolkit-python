# Roadmap

## Documentation
- Emphasise that a local CRS should be chosen to minimise projection error; accuracy degrades over large areas (city level: good, country level: less accurate).

## Output-spacing refactor — DONE

User-facing outputs now reflect the user-specified output cell size (`grid.spacing`, default `r/3`), decoupled from the internal search grid (`grid._search_spacing`, chosen automatically from `SPACINGS_BREAKPOINTS`).

Summary of what shipped (all verified with `temp/spacing_harness.py --cmp base`):
- **Public API:** `spacing` param on `radius_search` / `detect_cluster_pts` / `detect_cluster_cells` → `Grid(output_spacing=...)`; `None` ⇒ `r/3`. A user value set at creation is remembered and reused by no-arg `update_spacing()`.
- **Two-grid separation:** every search/cluster/sampling computation reads `grid._search_*`; public `grid.spacing`/`x_steps`/`y_steps`/`row_ids`/`col_ids`/`n_cells` reflect the output grid.
- **Lazy `update_spacing()`:** the output grid + cached per-output-cell aggregate (`output_id_to_sums`) are built on demand, not at construction. `radius_search` alone skips it (no overhead); `detect_cluster_*`, plots, and `save_*` call it. Re-pass `spacing=` to rebuild at a new size.
- **Plots/exports on the output grid:** `cell_aggregates` / `clusters` render the raw indicator per output cell via a `PatchCollection` of non-empty cells (no dense full-grid raster — robust to wide extents). Valid-area buffer cells (`infer_sample_area_from_pts`) bin to output cells.
- **Fixes made along the way:** `proj_x` plot crash after reprojection; `show=`/`display_dpi=` kwarg leak into `ax.scatter`; clustering keyed on the search grid (latent `KeyError`); removed the throwaway second `Grid` build in `detect_cluster_cells`; grid-creation print moved into `update_spacing` (fires once, only on build/spacing-change).
- **Docs:** two-grid model documented in `Grid.__init__`, the `radius_search` docstring, the README param table, and the `SPACINGS_BREAKPOINTS` comment.

### Remaining caveats (not blocking)
- **`weight_valid_area='estimate'/'precise'`** not exercised with the output-grid flip (default is off). It reads `grid.row_col_to_centroid` / invalid-cell geometry — search-grid concepts — via the export centroid builder, which now builds from output-flipped public arrays. Verify before relying on edge-weighting. (The search↔valid-area-polygon interaction itself is correct: `intersect_polygon_with_grid` clips against `grid._search_*`.)
- **Plots validated numerically** (conservation, no errors), **not visually** — worth an eyeball on orientation and that the 5 NY clusters look sensible.
- **`grid_ids`** still renders the search grid by design (search-structure diagnostic).
- **`create_full_grid_df`** allocates a dense `n_cells` array — `MemoryError` on continent-scale extents is expected; use `save_sparse_grid`.
- Optional polish: a separate `grid.output_sample_grid_bounds` for plot overlays (vs. search `sample_grid_bounds`) — only if overlay alignment looks off.

## Performance

### Experiment with Implementing numba
- Create an alternative version of disk_aggregation and its performance.

### Done
- **`disk_region_cache` LRU:** reuse now `move_to_end`s the entry, so eviction (`popitem(last=False)`) drops the genuinely least-recently-used config, not the oldest-inserted. Cache is small/bounded (geometry only; ~0.05 MB at nd=0 → ~4.6 MB at nd=6 per entry — see comment in `config.py`).
- **`time_func_perf` gated by `config.PROFILE_FUNC_TIMES`** (see TODO above) — removes per-point instrumentation (~3 µs/call + ~240 MB/1e6-calls call-log) from production runs; benchmark `run_single_config` enables it for measured runs only.
- **Generator instead of list** in the `.intersection(...)` calls of the hot search loop (`disk_aggregation.py`) — avoids materialising the intermediate cell list (marginal).

### Candidate improvements (bigger, not done)
- **Integer-encode cell ids + vectorised membership** — the real structural speedup. Full implementation plan in *Integer cell-key indexing* below.
- **Flatten cell-key tuples** `(lvl, (row, col))` → `(lvl, row, col)` — same blast radius as int-keys (touches every site below) but only ~1.36× on the intersection (measured) and no vectorised-translation headroom. Not recommended on its own; see plan below.
- Investigate line-based comparisons as an alternative to bilateral point distance checks.
- Benchmark a super-cell approach (e.g. 3×3 macro-cells) to reduce dict lookups at fine grid sizes.

### Timing-model benchmark (coverage gaps)
- The collected `perf_test` data is heavily skewed (~88% of runs at `n_source=2000`); `predict_timing` is unreliable at low `r/spacing` (1.41–1.58), low `nest_depth` (0–1), and high/uneven `n_src`×`n_tgt`. Gap-filling sweeps (`temp/gap_sweep*.py`) target `n_src ≫ n_tgt` (the `detect_cluster_pts` regime), low-`r/s`/low-`nd`, and multiple point distributions (clustered / uniform / 2nd dataset). Refit the model once enough gap data is collected.

### Allow for rectangualar cells
- Simply apply scaling on axis and remove it in the end. Needs to be kept in mind for plotting.

## Open items
- Validate user inputs at the public API boundary.

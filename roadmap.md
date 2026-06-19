# Roadmap

## Documentation
- Emphasise that a local CRS should be chosen to minimise projection error; accuracy degrades over large areas (city level: good, country level: less accurate).

## TODO / reminders
- **Set `config.PROFILE_FUNC_TIMES` back to `False`** once benchmark data collection is finished. It is **temporarily `True`** to collect per-function timings; the production default is `False` (the `time_func_perf` decorator is a zero-overhead passthrough when off — it otherwise adds ~3 µs per decorated call in the hot search loop and grows an unbounded in-memory call log).

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

### Done
- **`disk_region_cache` LRU:** reuse now `move_to_end`s the entry, so eviction (`popitem(last=False)`) drops the genuinely least-recently-used config, not the oldest-inserted. Cache is small/bounded (geometry only; ~0.05 MB at nd=0 → ~4.6 MB at nd=6 per entry — see comment in `config.py`).
- **`time_func_perf` gated by `config.PROFILE_FUNC_TIMES`** (see TODO above) — removes per-point instrumentation (~3 µs/call + ~240 MB/1e6-calls call-log) from production runs; benchmark `run_single_config` enables it for measured runs only.
- **Generator instead of list** in the `.intersection(...)` calls of the hot search loop (`disk_aggregation.py`) — avoids materialising the intermediate cell list (marginal).

### Candidate improvements (bigger, not done)
- **Integer-encode cell ids + vectorised membership** — the real structural speedup. Full implementation plan in *Integer cell-key indexing* below.
- **Flatten cell-key tuples** `(lvl, (row, col))` → `(lvl, row, col)` — same blast radius as int-keys (touches every site below) but only ~1.36× on the intersection (measured) and no vectorised-translation headroom. Not recommended on its own; see plan below.
- Investigate line-based comparisons as an alternative to bilateral point distance checks.
- Benchmark a super-cell approach (e.g. 3×3 macro-cells) to reduce dict lookups at fine grid sizes.

### Integer cell-key indexing — IMPLEMENTED (behind flag, verified)

**Status:** done and equivalence-verified behind `config.USE_INT_CELL_KEYS` (default `False`). With the flag on, `radius_search`, `detect_cluster_cells`, and multi-column runs produce **byte-identical** results to the tuple path (per-point aggregate hashes match; same 5 NY clusters; harness `--cmp base` PASS with flag off). Measured **~1.23× on the NY set** (2,596 pts, low-`ppc`); larger low-`ppc` data gains more. Files: `aabpl/utils/cell_keys.py` (codec), `config.USE_INT_CELL_KEYS`, `Grid.__init__` (`grid.cell_codec`), `point_grid_assignment.py` (int-key the `*_by_lvl` dicts via `_k`), `disk_aggregation.py` (`_isect_shared/_cntd/_ovlpd` helpers + int hot loop).
- **Key gotcha (cost a debugging round):** nested **centroid** coords are `row_c + 2**-(lvl+1)`, one level finer than `2**-lvl`, so the codec uses **`SCALE = 2**(nest_depth+1)`** (not `2**nest_depth`) — otherwise distinct cells collide.
- **`weight_valid_area` now int-routed (verified).** The invalid-cell membership (`invalid_cells` ∩ translated region cells) is preconverted to int in `disk_aggregation.py`: `invalid_int = {key(0,r,c)}`, region templates encoded via `offset_int([(0,dc) …])` with **lvl forced to 0** (membership is a level-0 `(row,col)` comparison — the tuple path drops lvl), set-intersection for dedup. Equivalent to the tuple logic across 5000 randomized trials (fractional lvl>0 cells never spuriously match; integer lvl-0 cells match exactly). Helpers `_invalid_count` / `_invalid_ovlpd_cells` provided in both int and tuple branches.
- **Pre-existing bugs found in the ovlpd-invalid-area path (independent of the flag — baseline crashes identically):** `calculate_ovlpd_invalid_area` → `compute_disk_cell_overlap` is broken regardless of `USE_INT_CELL_KEYS`. Fixed the `sample_area.py:667` unpack typo (`(xmin,ymin),(xmax,ymax) = a,b,c,d` → proper pairs). Still broken downstream: `circle_line_segment_intersection(..., precision=…)` (unexpected kwarg) and `row_col_to_centroid` is `None` in `calculate_ovlpd_invalid_area`. This ovlpd-invalid-area code (both `'precise'` and `'estimate'` when invalid overlap cells exist) has apparently never been exercised — fix separately before relying on edge-weighting. The cntd-weight (`len·spacing²`) path is unaffected.
- **Still guarded → `NotImplementedError` with the flag on:** the `plot_pt_disk` diagnostic (reads tuple cell keys). Level-0 dicts, clusters, exports, `cells_rndm_sample` remain tuple-keyed (unchanged) — extend later if wanted.
- **Before flipping the default:** measure the real payoff on clean (serial) large low-`ppc` data; the NY 1.23× is modest. Keep `False` until justified.

Original design notes below (kept for reference).

**Goal:** replace the per-region/per-point nested-tuple build + hash + `set.intersection` in the search hot loop with integer keys + vectorised membership, so translating an offset template to a point's home cell is one scalar add.

**Expected payoff:** ~5–10× on the three intersection functions only; linear in `n_source` in the low-`ppc` regime (where the cell-change cache doesn't fire), near-zero otherwise. Estimated ~0.5–1 s at `n_source=200k`, low-`ppc`. **Measure first:** sum `func_sum_cntd_*` + `func_get_pts_ovlpd_by_region` ÷ `func_search_and_aggregate` from clean (serial) `perf_test` rows to confirm the share before investing.

**The codec** — pack `(lvl, row, col)` → one `int64`, **linear** in row/col (proven in chat, incl. fractional `(2,(-1.25,2.25))`):
- `SCALE = 2**nest_depth` → sub-cell coords (multiples of `2**-lvl`) become exact integers.
- `key(lvl,row,col) = lvl*LVL_STRIDE + (round(row*SCALE)+BIAS)*ROW_STRIDE + (round(col*SCALE)+BIAS)`.
- **Grid-fit strides:** `LVL_STRIDE`/`ROW_STRIDE`/`BIAS` sized from *this grid's* padded extent + disk-reach margin (NOT fixed/global — see below), with `assert packed_max < 2**63`.
- **Offset/translation identity:** `key(lvl, pt_row+dr, pt_col+dc) = key(0, pt_row, pt_col) + offset_int(lvl,dr,dc)` where `offset_int(lvl,dr,dc) = lvl*LVL_STRIDE + round(dr*SCALE)*ROW_STRIDE + round(dc*SCALE)`. `offset_int` depends only on `(lvl,dr,dc)`, so one template array + one `home_key` per point.
- `decode(key) → (lvl, row, col)` and `decode_tuple(key) → (lvl,(row,col))` (old nested format, floats at lvl>0) + vectorised `decode_tuples` — for plots that want the original representation.
- Live in `aabpl/utils/cell_keys.py` (`CellKeyCodec`); was prototyped + unit-tested in chat (round-trip, translation-equivalence, collision-freedom, int64 safety all passed) — re-create from there.

**Codec lives on the grid, built once.** `choose_spacing_and_depth` sets `nest_depth`+spacing in `Grid.__init__` before point assignment and geometry build, and the extent is known there → build `grid.cell_codec` in `__init__`. Both the cell-sum aggregation and the per-search template encoding use this same codec, so keys are consistent by construction. Grid-fit strides are fine because nothing int-keyed is cached across grids (templates stay tuples in the cache; see below).

**Scope — only the `*_by_lvl` dicts (search-only), level-0 tuples untouched:**

| dict | key today | who reads it | action |
|---|---|---|---|
| `id_to_sums_by_lvl`, `id_to_vals_xy_by_lvl`, `id_to_pt_ids_by_lvl` | `(lvl,(row,col))` | **search loop only** (`disk_aggregation`) + `plot_pt_disk` diagnostic | **int-key at aggregation** (`point_grid_assignment.py`) using `grid.cell_codec` |
| `id_to_sums`, `id_to_pt_ids`, `id_to_vals_xy` (level-0) | `(row,col)` | `clusters.py`, `plot_grid`, df-export builders, `cells_rndm_sample` | **leave as tuples** (unchanged) |

**Change list (minimal — geometry build + cache untouched):**
1. `Grid.__init__`: build + store `grid.cell_codec` (SCALE from `nest_depth`, strides from extent).
2. `point_grid_assignment.py`: key the three `*_by_lvl` dicts with `codec.key(lvl,row,col)` (lines ~164–166, 198–202, 225–235). Level-0 `(row,col)` dicts (~192–194, 242–244) unchanged.
3. `disk_aggregation.py` setup (once per search): `sparse_int = set(grid.id_to_sums_by_lvl)` (already int); encode each region template (`region_and_trgl_id_to_{nested_cntd,distinct_cntd,distinct_ovlpd}_cells`, `shared_cntd_cells`) → `offset_int` arrays via `grid.cell_codec`. The cached tuple templates from `disk_region_geometry.py` are encoded here per-search — **`disk_region_geometry.py` and `disk_region_cache` are not touched.**
4. `disk_aggregation.py` hot loop: `home_key = codec.key(0, pt_row, pt_col)`; `abs = template_int + home_key`; membership via `np.isin(abs, sparse_int_sorted)` / int-set; value lookups `grid.id_to_sums_by_lvl[int]` (lines 150/170/201/222/236/255/268/288/305). Buffer slicing (`empty_sums`/`empty_xy_vals`) unchanged.
5. `plot_pt_disk` diagnostic (~line 500) + any per-level plot: use `codec.decode_tuple(key)` to recover `(lvl,(row,col))` for display.

**Gotchas / invariants:**
- Fractional nested coords → `SCALE=2**nest_depth` makes them exact; assert no rounding drift.
- `offset_int` can be a negative additive delta — fine; but `BIAS`/strides must keep the *absolute* row/col field in `[0, stride)` over padded extent + disk-reach margin (no field carry). Assert.
- `region_and_trgl_id` / `contain_region_mult` (`disk_aggregation.py` ~434–435) are separate region/triangle int encodings, **not** cell keys — leave untouched.
- Cluster cells can include empty cells — the level-0 tuple path is unchanged, so its `.get`/`in` guards stay.

**Verification (automatic):** gate behind `config.USE_INT_CELL_KEYS` (default `False`); build both paths; `temp/spacing_harness.py --cmp base` with flag off vs on must give byte-identical `rs_agg_hash` + cluster labels across several `(r, sr, nd, ppc)`. Do NOT run this refactor concurrently with a benchmark sweep (memory). Flip the default only after equivalence holds and the measured speedup justifies it.

**Staging:** (1) re-add + unit-test `cell_keys.py`; (2) `grid.cell_codec` in `__init__`; (3) int-key the `*_by_lvl` dicts in `point_grid_assignment`; (4) per-search template encoding + int hot loop in `disk_aggregation`, behind the flag; (5) `decode_tuple` in the diagnostic/plots; (6) harness equivalence off-vs-on; (7) measure on clean low-`ppc` large-`n_src` runs; (8) decide on flipping default.

### Timing-model benchmark (coverage gaps)
- The collected `perf_test` data is heavily skewed (~88% of runs at `n_source=2000`); `predict_timing` is unreliable at low `r/spacing` (1.41–1.58), low `nest_depth` (0–1), and high/uneven `n_src`×`n_tgt`. Gap-filling sweeps (`temp/gap_sweep*.py`) target `n_src ≫ n_tgt` (the `detect_cluster_pts` regime), low-`r/s`/low-`nd`, and multiple point distributions (clustered / uniform / 2nd dataset). Refit the model once enough gap data is collected.

## Open items
- Validate user inputs at the public API boundary.

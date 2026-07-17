# AABPL Toolkit — Roadmap

---

## Performance self-tuning guide for users

**Status:** idea / not started

**Summary:**
Expose a helper (e.g. `aabpl.print_performance_tips()` or a config method) that
tells the user how to maximise throughput for their hardware and workload:

- How to detect their L2/L3 cache sizes and set `config.L2_BYTES` / `config.L3_BYTES`.
- When to use the chunk path vs the orig path (crossover radius, dataset size).
- How `config.N_POINTS_TOTAL` affects geometry selection and amortisation.
- What `config.PROFILE_FUNC_TIMES` + `analyze_func_perf()` reveal about bottlenecks.
- Guidance on `weight_valid_area` variant trade-offs (precise / estimate / guess / fast).

The goal is that a user can run one command and get a personalised checklist rather
than having to read scattered config comments.

---

## Adaptive multi-resolution disk aggregation (chunk)

**Status:** idea / not started

**Summary:**
Precompute region templates (contained cells + overlap cells) at multiple nest depths,
where depth `k` uses `spacing = r / 2^k`. During chunk processing, choose depth
dynamically based on local point density in the chunk.

**Motivation:**
At the current fixed `spacing = r`, nearly the entire disk falls in the expensive
boundary (overlap) region — contained cells are few or zero and require only a dict
lookup, while overlap cells require per-point `compute_disk_cell_overlap` or the logit
estimate. At finer grids the boundary ring shrinks:

| spacing | approx. contained cells | approx. overlap cells | boundary fraction |
|---|---|---|---|
| `r` (current) | 0–4 | 8–16 | ~100% |
| `r/2` | ~12 | ~16 | ~75% |
| `r/4` | ~50 | ~24 | ~38% |
| `r/8` | ~200 | ~32 | ~19% |

In a dense chunk, many points share precomputed `cell_sums`; the contained-cell
contribution then costs O(1) per point regardless of chunk size. This amplifies the
benefit of finer resolution for dense chunks.

**Decision rule (per chunk):**
If the chunk has ≥ `4^k` points in its core rows, using depth `k` is net positive.
Estimated cheaply from the `pts_per_row` array already computed for chunk sizing.

**Implementation notes:**
- Grid construction: optionally build `_depth_k` variants of `cntd_cells_by_region`,
  `ovlpd_cells_by_region`, and a per-depth codec at `spacing = r / 2^k`.
- Memory scales as `1 + 4 + 16 + 64 = 85×` current template size for `k_max=3` — acceptable.
- `cell_sums` is rebuilt per chunk, so incompatible codecs across chunks are fine.
- Chunk loop change is straightforward once multi-level templates exist; most work is
  in grid construction.
- `weight_valid_area` benefit: coarser depths make the polygon validity grid fast,
  naturally solving the "tiny-r → huge validity grid" problem (see below).

**Related issue — tiny-r validity grid:**
For small `r` (e.g., `r=1`), `grid_spacing=1` and the padded cell range can reach
10^10 cells, making the upfront shapely box classification infeasible. Fix: use a
minimum validity-check spacing (e.g., `max(grid_spacing, 50.0)` or the study_area's
own construction spacing) for the polygon classification, independent of `r`.
Multi-resolution templates would solve this naturally for the chunk path.

**Highest ROI:** large datasets with spatially variable density (e.g., `n=521k`,
`r=750`), where some chunks are very dense and benefit from depth-2/3 while sparse
peripheral chunks stay at depth 0.

---

## make multi radius search native to our core algorithm

**Status:** not started: idea: we use our largest r to start out process as usally. But then we define the contained and overlapped cells for those same regions additionally for the smaller r (not sure if thats easy). Also imagine if our rs are like 1000,3,2,1. Then our cell itself is way to big s.t. we might be better of handling r=1000 individually. but groupin 3,2,1 might still have its merits.

---

## NaN handling in aggregation columns

**Status:** partially done (user warning added); core behaviour unchanged

## Re-add proper timing benchmark for progress prints

AFter testing is done.  

### Sub-issues

**1. Count columns with NaN inputs**
Currently a point with `NaN` in variable `x` still contributes 1 to `count_x`.
It should contribute 0 so that `count_x` equals the number of non-missing
observations in the disk — making it a valid denominator for computing a mean.

**2. Sum/mean contamination by NaN**
A single `NaN` in `x` propagates through the aggregation and silently zeroes
(or NaN-fills) the sum for every disk that contains that point.
`np.nansum` / `np.nanmean` semantics (skip NaN, count remaining) are almost
certainly what users expect; SQL-style NaN propagation is rarely desired here.

**3. User warning (done)**
If any column passed via `c=` contains NaN values, a one-line warning is
printed before aggregation so the user is not silently surprised.
The warning notes that the current implementation does not distinguish NaN from
valid values (i.e. NaN contributes to count and contaminates sums).

### Proposed behaviour (not yet implemented)
- `count_x`: count non-NaN values only.
- `sum_x`: use `nansum` — NaN points are skipped, not propagated.
- Add an option (e.g. `nan_policy='propagate'`) to restore the old behaviour
  for users who want strict NaN propagation.

---

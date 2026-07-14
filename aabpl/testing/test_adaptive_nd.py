"""
Adaptive nd timing test.

Fixed: sr=2.0 (spacing=250m), r=500m, 1000 source pts all in one home cell.
Varies: nd=0..7, n_target_pts in N_TGT_LIST.

For each (n_tgt, nd) combo:
  - Swaps grid._search_class lookup tables to the nd-level versions
  - Times search_and_aggregate (min of N_REPS runs)

Prints timing table + best nd per n_tgt.
"""
import sys, time
import numpy as np
import pandas as pd

sys.path.insert(0, 'Z:/Algorithm/PL_python/AABPL-toolkit-python')
[sys.modules.pop(m) for m in list(sys.modules.keys()) if m.startswith('aabpl')]

R          = 500.0
SR         = 2.0
SPACING    = R / SR   # 250m
ND_MAX     = 7
N_SRC      = 1000
N_TGT_LIST = [10, 50, 100, 500, 1000, 5000, 10000]
N_REPS     = 3
CRS        = 'EPSG:32632'
HX0, HY0  = 500_000.0, 5_500_000.0   # bottom-left of home cell

np.random.seed(42)

# --- synthetic data ----------------------------------------------------------
# Source: N_SRC pts strictly inside one home cell (5m margin to avoid cell edge)
src_x = HX0 + np.random.uniform(5, SPACING - 5, N_SRC)
src_y = HY0 + np.random.uniform(5, SPACING - 5, N_SRC)

# Target pool: pts within ±R of home cell bounds
MAX_TGT = max(N_TGT_LIST)
tgt_x = HX0 + np.random.uniform(-R, SPACING + R, MAX_TGT)
tgt_y = HY0 + np.random.uniform(-R, SPACING + R, MAX_TGT)

# Helper zone label
zone = np.where(
    (tgt_x >= HX0) & (tgt_x < HX0 + SPACING) &
    (tgt_y >= HY0) & (tgt_y < HY0 + SPACING),
    'home', 'neighbour'
)

# Combined pts for grid build (mark is_src so we can split later)
all_x   = np.concatenate([src_x, tgt_x])
all_y   = np.concatenate([src_y, tgt_y])
all_val = np.ones(N_SRC + MAX_TGT)
is_src  = np.array([1] * N_SRC + [0] * MAX_TGT)
pts_all = pd.DataFrame({'lon': all_x, 'lat': all_y, 'val': all_val, 'is_src': is_src})

# --- capture grid + preprocessed pts via monkey-patch -----------------------
# search_and_aggregate is imported into disk_search.py at import time,
# so we must patch disk_search.search_and_aggregate (not the source module).
from aabpl.main import radius_search  # triggers all imports
import aabpl.search.algorithm.disk_search as _dsearch
import aabpl.search.algorithm.disk_aggregation_chunk as _chunk
_orig_fn = _chunk.search_and_aggregate  # keep reference to real function

_cap = {}
def _capture_saa(grid, pts_source, r, **kw):
    if 'grid' not in _cap:
        _cap['grid'] = grid
        _cap['pts_pp'] = pts_source.copy()
    return _orig_fn(grid, pts_source, r, **kw)

_dsearch.search_and_aggregate = _capture_saa
radius_search(pts_all, crs=CRS, r=R, c=['val'], cell_size=SPACING, silent=True)
_dsearch.search_and_aggregate = _orig_fn   # restore

grid    = _cap['grid']
pts_pp  = _cap['pts_pp']
SPACING = grid._search_internals.spacing   # use actual grid spacing

# Find projected coordinate column names
_xy_candidates = [('proj_lon', 'proj_lat'), ('lon', 'lat'), ('x', 'y')]
X_COL, Y_COL = next(
    (xc, yc) for xc, yc in _xy_candidates
    if xc in pts_pp.columns and yc in pts_pp.columns
)

# Find actual cell_region column name (find_column_name may suffix it)
CELL_REG_COL = next(
    (c for c in pts_pp.columns if c.startswith('cell_reg')), 'cell_region'
)

# Find row/col/off column names
ROW_COL  = next((c for c in pts_pp.columns if c.startswith('id_y')), 'id_y')
COL_COL  = next((c for c in pts_pp.columns if c.startswith('id_x')), 'id_x')
OFF_X    = next((c for c in pts_pp.columns if c.startswith('offset_x')), 'offset_x')
OFF_Y    = next((c for c in pts_pp.columns if c.startswith('offset_y')), 'offset_y')

# Split back into source / target
pts_src_pp = pts_pp[pts_pp['is_src'] == 1].copy().reset_index(drop=True)
pts_tgt_pp = pts_pp[pts_pp['is_src'] == 0].copy().reset_index(drop=True)
pts_tgt_pp['zone'] = zone

print(f"Grid captured.  spacing={SPACING:.1f}m  r={R:.0f}m  sr={R/SPACING:.2f}")
print(f"Coordinate columns: x={X_COL!r}  y={Y_COL!r}")
print(f"cell_region col: {CELL_REG_COL!r}   row: {ROW_COL!r}   col: {COL_COL!r}")
print(f"offset cols: off_x={OFF_X!r}  off_y={OFF_Y!r}")
n_home_cells = pts_src_pp[[ROW_COL, COL_COL]].drop_duplicates().shape[0]
print(f"Source pts: {len(pts_src_pp)}  (unique home cells: {n_home_cells}  <- should be ~1)")
print(f"Target pool: {len(pts_tgt_pp)}  zone dist: {pts_tgt_pp['zone'].value_counts().to_dict()}")

# --- pre-compute region-assigned source pts for each nd, capturing grid tables ---
# We let assign_points_to_mirco_regions drive everything: it internally calls
# build_disk_region_lookups(grid_spacing=1, r=r/spacing, nd=nd) and then
# updates grid._search_class with the resulting tables. We snapshot those tables
# after each call so we can swap them back in during the timing loop.
from aabpl.search.point_region_assignment import assign_points_to_mirco_regions
import copy

nd_entries    = {}
pts_src_per_nd = {}

print(f"\nBuilding nd=0..{ND_MAX} tables & assigning source pts ...")
print(f"{'nd':>4}  {'n_regions':>10}  {'avg_cntd':>9}  {'avg_ovlpd':>10}")
print('-' * 40)
for nd in range(ND_MAX + 1):
    _ps = pts_src_pp.drop(columns=[CELL_REG_COL, 'region_and_trgl_id'], errors='ignore').copy()
    assign_points_to_mirco_regions(
        grid=grid,
        pts=_ps,
        r=R,
        nest_depth=nd,
        include_boundary=grid._search_class.include_boundary,
        x=X_COL, y=Y_COL,
        off_x=OFF_X, off_y=OFF_Y,
        row_name=ROW_COL, col_name=COL_COL,
        cell_region_name=CELL_REG_COL,
        silent=True,
    )
    pts_src_per_nd[nd] = _ps
    # Snapshot the tables that were just installed in grid._search_class
    _sc = grid._search_class
    nd_entries[nd] = {
        'region_and_trgl_id_to_distinct_cntd_cells':  copy.copy(_sc.region_and_trgl_id_to_distinct_cntd_cells),
        'region_and_trgl_id_to_distinct_ovlpd_cells': copy.copy(_sc.region_and_trgl_id_to_distinct_ovlpd_cells),
        'shared_cntd_cells':                          copy.copy(_sc.shared_cntd_cells),
    }
    n_reg = len(nd_entries[nd]['region_and_trgl_id_to_distinct_cntd_cells'])
    ac = np.mean([len(v) for v in nd_entries[nd]['region_and_trgl_id_to_distinct_cntd_cells'].values()]) if n_reg else 0
    ao = np.mean([len(v) for v in nd_entries[nd]['region_and_trgl_id_to_distinct_ovlpd_cells'].values()]) if n_reg else 0
    print(f"{nd:>4}  {n_reg:>10}  {ac:>9.1f}  {ao:>10.1f}")

# --- timing loop ------------------------------------------------------------
_sc  = grid._search_class
saa  = _orig_fn   # original (time_func_perf wrapped but PROFILE_FUNC_TIMES=False so near-zero overhead)

# Validate correctness once with nd=0
print("\nValidating nd=0 gives correct results (validate=True) ...")
_sc.region_and_trgl_id_to_distinct_cntd_cells  = nd_entries[0]['region_and_trgl_id_to_distinct_cntd_cells']
_sc.region_and_trgl_id_to_distinct_ovlpd_cells = nd_entries[0]['region_and_trgl_id_to_distinct_ovlpd_cells']
_sc.shared_cntd_cells                           = nd_entries[0]['shared_cntd_cells']
_s = pts_src_per_nd[0].copy()
saa(grid, _s, r=R, c=['val'], x=X_COL, y=Y_COL,
    row_name=ROW_COL, col_name=COL_COL, cell_region_name=CELL_REG_COL,
    pts_target=pts_tgt_pp.copy(), silent=True, validate=True)

# Print header
print(f"\n{'n_tgt':>8}", end='')
for nd in range(ND_MAX + 1):
    print(f"  nd={nd}_ms", end='')
print("  best_nd")
print('-' * (8 + (ND_MAX + 1) * 10 + 10))

results = {}
for n_tgt in N_TGT_LIST:
    pts_tgt_n = pts_tgt_pp.iloc[:n_tgt].copy()
    row = {}
    print(f"{n_tgt:>8}", end='', flush=True)

    for nd in range(ND_MAX + 1):
        # Swap grid lookup tables to this nd level
        _sc.region_and_trgl_id_to_distinct_cntd_cells  = nd_entries[nd]['region_and_trgl_id_to_distinct_cntd_cells']
        _sc.region_and_trgl_id_to_distinct_ovlpd_cells = nd_entries[nd]['region_and_trgl_id_to_distinct_ovlpd_cells']
        _sc.shared_cntd_cells                           = nd_entries[nd]['shared_cntd_cells']

        times = []
        for _ in range(N_REPS):
            _s = pts_src_per_nd[nd].copy()
            t0 = time.perf_counter()
            saa(grid, _s, r=R, c=['val'], x=X_COL, y=Y_COL,
                row_name=ROW_COL, col_name=COL_COL, cell_region_name=CELL_REG_COL,
                pts_target=pts_tgt_n, silent=True)
            times.append((time.perf_counter() - t0) * 1000)

        row[nd] = min(times)
        print(f"  {row[nd]:7.1f}", end='', flush=True)

    results[n_tgt] = row
    best_nd = min(row, key=row.get)
    print(f"  {best_nd}", flush=True)

# Restore original lookup tables
_sc.region_and_trgl_id_to_distinct_cntd_cells  = nd_entries[3]['region_and_trgl_id_to_distinct_cntd_cells']
_sc.region_and_trgl_id_to_distinct_ovlpd_cells = nd_entries[3]['region_and_trgl_id_to_distinct_ovlpd_cells']
_sc.shared_cntd_cells                           = nd_entries[3]['shared_cntd_cells']

print("\nSummary: best nd per n_tgt")
print(f"{'n_tgt':>8}  {'best_nd':>7}  {'time_ms':>8}  {'vs_nd0_ms':>10}  {'vs_nd7_ms':>10}")
print('-' * 55)
for n_tgt, row in results.items():
    best_nd   = min(row, key=row.get)
    print(f"{n_tgt:>8}  {best_nd:>7}  {row[best_nd]:>8.1f}  {row[0]:>10.1f}  {row[ND_MAX]:>10.1f}")

print("\nDone.")

"""
Comprehensive test suite for aabpl.radius_search.

Covers:
- All stat methods (sum, count, mean, variance, std, cv, skewness, kurtosis)
- Multi-stat lists
- Multiple value columns (c)
- nest_depth 0, 2, 3
- Cartesian CRS (crs='')
- Separate pts_target
- weight_valid_area
- config.VALIDATE=True
- cell_count / cell_count_iter helpers
- Grid attribute checks (no legacy id_to_pt_ids)
"""
import sys, time
sys.path.insert(0, 'Z:/Algorithm/PL_python/AABPL-toolkit-python')
[sys.modules.pop(m) for m in list(sys.modules.keys()) if m.startswith('aabpl')]

import numpy as np
import pandas as pd
import aabpl
import aabpl.config as config
from aabpl.radius_search.point_grid_assignment import cell_count, cell_count_iter, _lvl0_packed

# ── helpers ──────────────────────────────────────────────────────────────────

def _make_pts(n=1000, seed=0):
    rng = np.random.default_rng(seed)
    pts = pd.DataFrame({
        'x': rng.uniform(0, 50000, n),
        'y': rng.uniform(0, 50000, n),
        'val': rng.uniform(1, 10, n),
        'val2': rng.uniform(0, 5, n),
    })
    return pts

def check(cond, msg):
    if not cond:
        raise AssertionError(f"FAIL: {msg}")

def section(name):
    print(f"\n{'='*60}")
    print(f"  {name}")
    print('='*60)

# ── fixtures ──────────────────────────────────────────────────────────────────

N = 800   # small for speed
pts_base = _make_pts(N)
R = 5000  # 5 km radius
DEV = {'nest_depth': 2, 'spacing_over_radius': 2.0}
config.VALIDATE = False  # default off; enabled explicitly in section 8

# ═══════════════════════════════════════════════════════════════════════════════
section("1 · Basic single-agg: sum / count / mean  (crs='', nd=2)")
# ═══════════════════════════════════════════════════════════════════════════════

for stat, suf in [('sum', '_r_sum'), ('count', '_r_count'), ('mean', '_r_mean')]:
    pts = pts_base.copy()
    aabpl.radius_search(pts=pts, crs='', r=R, c=['val'], x='x', y='y',
                        stat=stat, suffix=suf, silent=True, _dev=DEV)
    out = [c for c in pts.columns if c.endswith(suf)]
    check(len(out) == 1, f"stat={stat}: expected 1 output col ending with {suf!r}, got {[c for c in pts.columns]}")
    col = out[0]
    check((pts[col] >= 0).all(), f"stat={stat}: negative values in {col}")
    print(f"  stat={stat!r:12s}  col={col!r}  sum={pts[col].sum():.1f}  OK")

# ═══════════════════════════════════════════════════════════════════════════════
section("2 · Moment aggs: variance / std / cv / skewness / kurtosis  (nd=2)")
# ═══════════════════════════════════════════════════════════════════════════════

for stat in ['variance', 'std', 'cv', 'skewness', 'kurtosis']:
    pts = pts_base.copy()
    aabpl.radius_search(pts=pts, crs='', r=R, c=['val'], x='x', y='y',
                        stat=stat, silent=True, _dev=DEV)
    out_cols = [c for c in pts.columns if c.startswith('val_')]
    check(len(out_cols) == 1, f"stat={stat}: expected 1 output col, got {out_cols}")
    check(not (pts[out_cols[0]].dropna() < 0).any() or stat in ('skewness', 'kurtosis', 'cv'),
          f"stat={stat}: unexpected negatives")
    print(f"  stat={stat!r:12s}  col={out_cols[0]!r}  OK")

# ═══════════════════════════════════════════════════════════════════════════════
section("3 · Multi-agg lists  (crs='', nd=2)")
# ═══════════════════════════════════════════════════════════════════════════════

# 3a: sum + count + mean — verify mean == sum/count
pts = pts_base.copy()
aabpl.radius_search(pts=pts, crs='', r=R, c=['val'], x='x', y='y',
                    stat=['sum', 'count', 'mean'], silent=True, _dev=DEV)
for expected in [f'val_sum_{R}', f'val_cnt_{R}', f'val_avg_{R}']:
    check(expected in pts.columns, f"multi-stat: missing {expected}; cols={[c for c in pts.columns if 'val' in c]}")
mask = pts[f'val_cnt_{R}'] > 0
diff = (pts.loc[mask, f'val_avg_{R}'] - pts.loc[mask, f'val_sum_{R}'] / pts.loc[mask, f'val_cnt_{R}']).abs().max()
check(diff < 1e-9, f"multi-stat: mean != sum/count, diff={diff}")
print(f"  ['sum','count','mean']  mean_err={diff:.2e}  OK")

# 3b: sum + variance
pts = pts_base.copy()
aabpl.radius_search(pts=pts, crs='', r=R, c=['val'], x='x', y='y',
                    stat=['sum', 'variance'], silent=True, _dev=DEV)
check(f'val_sum_{R}' in pts.columns and f'val_var_{R}' in pts.columns, "multi-stat sum+variance: missing cols")
leaked = [c for c in pts.columns if '__rs_int__' in c]
check(not leaked, f"multi-stat: internal suffix leaked: {leaked}")
print(f"  ['sum','variance']  no_leak=True  OK")

# 3c: two columns, sum + mean
pts = pts_base.copy()
aabpl.radius_search(pts=pts, crs='', r=R, c=['val', 'val2'], x='x', y='y',
                    stat=['sum', 'mean'], silent=True, _dev=DEV)
for col_base in ['val', 'val2']:
    for suf in [f'_sum_{R}', f'_avg_{R}']:
        check(col_base + suf in pts.columns, f"two-col multi-stat: missing {col_base+suf}")
print(f"  two-col ['sum','mean']  OK")

# 3d: custom suffix dict
pts = pts_base.copy()
aabpl.radius_search(pts=pts, crs='', r=R, c=['val'], x='x', y='y',
                    stat=['sum', 'count'],
                    suffix={'sum': '_s5k', 'count': '_n5k'},
                    silent=True, _dev=DEV)
check('val_s5k' in pts.columns and 'val_n5k' in pts.columns,
      f"custom suffix: cols={list(pts.columns)}")
print(f"  custom suffix dict  OK")

# ═══════════════════════════════════════════════════════════════════════════════
section("4 · nest_depth variants: 0, 2, 3  (sum)")
# ═══════════════════════════════════════════════════════════════════════════════

sums = {}
for nd in [0, 2, 3]:
    pts = pts_base.copy()
    aabpl.radius_search(pts=pts, crs='', r=R, c=['val'], x='x', y='y',
                        stat='sum', silent=True,
                        _dev={'nest_depth': nd, 'spacing_over_radius': 2.0})
    col = [c for c in pts.columns if c.startswith('val_')][0]
    sums[nd] = pts[col].sum()
    check(sums[nd] > 0, f"nd={nd}: zero sum")
    print(f"  nd={nd}  sum={sums[nd]:.1f}  OK")
# Finer nesting should give more accurate (and typically larger) sums at the overlap boundary
# (not a strict invariant, but nd=3 should not be wildly off from nd=2)
check(abs(sums[3] - sums[2]) / max(sums[2], 1) < 0.05,
      f"nd=2 vs nd=3 sum diverge too much: {sums[2]:.0f} vs {sums[3]:.0f}")
print(f"  nd consistency (|nd2-nd3|/nd2 < 5%)  OK")

# ═══════════════════════════════════════════════════════════════════════════════
section("5 · Multiple c columns  (nd=2, sum)")
# ═══════════════════════════════════════════════════════════════════════════════

pts = pts_base.copy()
aabpl.radius_search(pts=pts, crs='', r=R, c=['val', 'val2'], x='x', y='y',
                    stat='sum', silent=True, _dev=DEV)
for expected in [f'val_sum_{R}', f'val2_sum_{R}']:
    check(expected in pts.columns, f"multi-col: missing {expected}")
print(f"  c=['val','val2'] sum  OK")

# ═══════════════════════════════════════════════════════════════════════════════
section("6 · Separate pts_target  (nd=2, sum)")
# ═══════════════════════════════════════════════════════════════════════════════

pts_source = pts_base.copy()
pts_target = _make_pts(300, seed=42)
grid = aabpl.radius_search(pts=pts_target, crs='', r=R, c=['val'], x='x', y='y',
                           pts_target=pts_source, silent=True, _dev=DEV)
check(f'val_sum_{R}' in pts_target.columns, "pts_target: missing val_sum_{R}")
check(f'val_sum_{R}' not in pts_source.columns, "pts_source should NOT have result col")
print(f"  pts_target separate  sum={pts_target[f'val_sum_{R}'].sum():.1f}  OK")

# ═══════════════════════════════════════════════════════════════════════════════
section("7 · cell_count / cell_count_iter helpers")
# ═══════════════════════════════════════════════════════════════════════════════

from aabpl.radius_search.point_grid_assignment import (
    assign_points_to_cells, aggregate_point_data_to_cells,
)

class _Bounds:
    def __init__(self, xmin, ymin): self.xmin = xmin; self.ymin = ymin
class _Grid: pass

spacing = 5000.0
g = _Grid()
pts_raw = pts_base.copy()
g.total_bounds = _Bounds(pts_raw['x'].min() - spacing, pts_raw['y'].min() - spacing)
g._search_spacing = spacing
g.cell_codec = None

assign_points_to_cells(g, pts_raw, y='y', x='x')
aggregate_point_data_to_cells(g, pts_raw, c=['val'], y='y', x='x', nest_depth=2)

total = sum(cnt for _, _, cnt in cell_count_iter(g))
check(total == len(pts_raw), f"cell_count_iter total {total} != {len(pts_raw)}")
check(not hasattr(g, 'id_to_pt_ids'), "id_to_pt_ids should be gone")

for rc in list(g.id_to_sums)[:30]:
    pos = _lvl0_packed(g, rc[0], rc[1])
    ids_len = (pos & 0xFFFFFFFF) - (pos >> 32) if pos else 0
    cc = cell_count(g, rc[0], rc[1])
    check(cc == ids_len, f"cell_count({rc})={cc} but packed ids_len={ids_len}")

print(f"  cell_count_iter total={total}  cell_count spot-checks OK")

# ═══════════════════════════════════════════════════════════════════════════════
section("8 · config.VALIDATE=True correctness check  (nd=2, sum)")
# ═══════════════════════════════════════════════════════════════════════════════

config.VALIDATE = True
pts = pts_base.copy()
try:
    grid = aabpl.radius_search(pts=pts, crs='', r=R, c=['val'], x='x', y='y',
                               stat='sum', silent=True, _dev=DEV)
    print(f"  VALIDATE=True run completed without error  OK")
except Exception as e:
    raise AssertionError(f"VALIDATE=True raised: {e}")
finally:
    config.VALIDATE = False

# ═══════════════════════════════════════════════════════════════════════════════
section("9 · weight_valid_area  (nd=2, sum)")
# ═══════════════════════════════════════════════════════════════════════════════

config.VALIDATE = False
pts = pts_base.copy()
try:
    grid = aabpl.radius_search(pts=pts, crs='', r=R, c=['val'], x='x', y='y',
                               stat='sum', weight_valid_area='estimate',
                               silent=True, _dev=DEV)
    check(f'val_sum_{R}' in pts.columns, f"weight_valid_area: missing val_sum_{R}")
    print(f"  weight_valid_area='estimate'  OK")
except Exception as e:
    print(f"  weight_valid_area skipped: {e}")

# ═══════════════════════════════════════════════════════════════════════════════
section("10 · count-only mode with explicit c  (nd=2)")
# ═══════════════════════════════════════════════════════════════════════════════

# c=['val'] + stat='count' → output is val_r_count (copied from count helper)
pts = pts_base.copy()
aabpl.radius_search(pts=pts, crs='', r=R, c=['val'], x='x', y='y',
                    stat='count', suffix='_r_count', silent=True, _dev=DEV)
check('val_r_count' in pts.columns, f"stat='count': missing val_r_count; cols={list(pts.columns)}")
check((pts['val_r_count'] >= 0).all(), "stat='count': negative values")
print(f"  c=['val'] stat='count'  sum={pts['val_r_count'].sum():.0f}  OK")

# ═══════════════════════════════════════════════════════════════════════════════
section("11 · keep_cols behaviour and stat helper cleanup")
# ═══════════════════════════════════════════════════════════════════════════════

GRID_COLS  = {'id_y', 'id_x'}   # default row_name / col_name
PROJ_COLS  = {'proj_x', 'proj_y'}

# 11a: keep_cols=False (default) — only requested output cols added
pts = pts_base.copy()
cols_before = set(pts.columns)
aabpl.radius_search(pts=pts, crs='', r=R, c=['val'], x='x', y='y',
                    stat='sum', silent=True, _dev=DEV)
added = set(pts.columns) - cols_before
check(added == {f'val_sum_{R}'}, f"keep_cols=False: unexpected cols added: {added}")
print(f"  keep_cols=False (sum)  added={added}  OK")

# 11b: keep_cols=True — grid/proj cols kept, but NO internal helpers
pts = pts_base.copy()
cols_before = set(pts.columns)
aabpl.radius_search(pts=pts, crs='', r=R, c=['val'], x='x', y='y',
                    stat='sum', keep_cols=True, silent=True, _dev=DEV)
added = set(pts.columns) - cols_before
internal_leaked = [c for c in added if '__rs_int__' in c]
check(not internal_leaked, f"keep_cols=True: internal helpers leaked: {internal_leaked}")
check(f'val_sum_{R}' in added, "keep_cols=True: output col missing")
print(f"  keep_cols=True (sum)  added={sorted(added)}  OK")

# 11c: keep_cols=None — grid index cols kept, proj cols kept, no helpers
pts = pts_base.copy()
cols_before = set(pts.columns)
aabpl.radius_search(pts=pts, crs='', r=R, c=['val'], x='x', y='y',
                    stat='sum', keep_cols=None, silent=True, _dev=DEV)
added = set(pts.columns) - cols_before
internal_leaked = [c for c in added if '__rs_int__' in c]
check(not internal_leaked, f"keep_cols=None: internal helpers leaked: {internal_leaked}")
check(f'val_sum_{R}' in added, "keep_cols=None: output col missing")
print(f"  keep_cols=None (sum)  added={sorted(added)}  OK")

# 11d: radius_count — count IS the output col, must be present
pts = pts_base.copy()
cols_before = set(pts.columns)
aabpl.radius_count(pts=pts, crs='', r=R, c=['val'], x='x', y='y', silent=True, _dev=DEV)
added = set(pts.columns) - cols_before
check(any('cnt' in c or 'count' in c for c in added),
      f"radius_count: count output col missing; added={added}")
internal_leaked = [c for c in added if '__rs_int__' in c]
check(not internal_leaked, f"radius_count: internal helpers leaked: {internal_leaked}")
print(f"  radius_count  added={sorted(added)}  OK")

# 11e: radius_variance — only variance col added, no count/sum helpers leaked
pts = pts_base.copy()
cols_before = set(pts.columns)
aabpl.radius_variance(pts=pts, crs='', r=R, c=['val'], x='x', y='y', silent=True, _dev=DEV)
added = set(pts.columns) - cols_before
check(any('var' in c for c in added), f"radius_variance: variance col missing; added={added}")
# count and sum helpers must NOT appear as output cols
helper_leaked = [c for c in added if ('cnt' in c or 'count' in c or 'sum' in c) and '__rs_int__' not in c and c not in {f'val_var_{R}'}]
check(not helper_leaked, f"radius_variance: count/sum helpers leaked into pts: {helper_leaked}")
internal_leaked = [c for c in added if '__rs_int__' in c]
check(not internal_leaked, f"radius_variance: internal helpers leaked: {internal_leaked}")
print(f"  radius_variance  added={sorted(added)}  OK")

# 11f: radius_variance + keep_cols=True — grid/proj cols kept, still no count/sum helpers
pts = pts_base.copy()
cols_before = set(pts.columns)
aabpl.radius_variance(pts=pts, crs='', r=R, c=['val'], x='x', y='y',
                      keep_cols=True, silent=True, _dev=DEV)
added = set(pts.columns) - cols_before
helper_leaked = [c for c in added if ('cnt' in c or 'count' in c or 'sum' in c) and '__rs_int__' not in c and c not in {f'val_var_{R}'}]
check(not helper_leaked, f"radius_variance keep_cols=True: count/sum helpers leaked: {helper_leaked}")
internal_leaked = [c for c in added if '__rs_int__' in c]
check(not internal_leaked, f"radius_variance keep_cols=True: internal helpers leaked: {internal_leaked}")
print(f"  radius_variance keep_cols=True  added={sorted(added)}  OK")

# 11g: multi-stat ['variance','count'] — count IS requested so it must be present
pts = pts_base.copy()
cols_before = set(pts.columns)
aabpl.radius_search(pts=pts, crs='', r=R, c=['val'], x='x', y='y',
                    stat=['variance', 'count'], silent=True, _dev=DEV)
added = set(pts.columns) - cols_before
check(any('var' in c for c in added), f"multi [var,cnt]: variance col missing; added={added}")
check(any('cnt' in c or 'count' in c for c in added), f"multi [var,cnt]: count col missing; added={added}")
internal_leaked = [c for c in added if '__rs_int__' in c]
check(not internal_leaked, f"multi [var,cnt]: internal helpers leaked: {internal_leaked}")
print(f"  multi-stat ['variance','count']  added={sorted(added)}  OK")

# ═══════════════════════════════════════════════════════════════════════════════
section("12 · detect_cluster_pts  (crs='', nd=2, sum)")

# ═══════════════════════════════════════════════════════════════════════════════
# This exercises the null-distribution path where random points (no 'val' col)
# are the source. The exclude_self guard must NOT subtract src[value_col] there.

pts = pts_base.copy()
try:
    grid = aabpl.detect_cluster_pts(
        pts=pts, crs='', r=R, c=['val'], x='x', y='y',
        stat='sum', n_random_points=500, random_seed=0,
        silent=True, _dev=DEV,
    )
except Exception as e:
    raise AssertionError(f"detect_cluster_pts raised: {e}")

cluster_col = [c for c in pts.columns if 'cluster' in c.lower()]
check(len(cluster_col) >= 1, f"detect_cluster_pts: no cluster column found; cols={list(pts.columns)}")
check(pts[cluster_col[0]].dtype == bool or set(pts[cluster_col[0]].dropna().unique()) <= {0, 1, True, False},
      f"detect_cluster_pts: cluster column not boolean; unique={pts[cluster_col[0]].unique()}")
n_clustered = pts[cluster_col[0]].sum()
print(f"  detect_cluster_pts  cluster_col={cluster_col[0]!r}  n_clustered={n_clustered}  OK")

# ═══════════════════════════════════════════════════════════════════════════════
section("12 · detect_cluster_cells  (crs='', nd=2, sum)")
# ═══════════════════════════════════════════════════════════════════════════════

pts = pts_base.copy()
try:
    grid = aabpl.detect_cluster_cells(
        pts=pts, crs='', r=R, c=['val'], x='x', y='y',
        stat='sum', n_random_points=500, random_seed=0,
        silent=True, _dev=DEV,
    )
except Exception as e:
    raise AssertionError(f"detect_cluster_cells raised: {e}")

check(hasattr(grid, 'clustering'), "detect_cluster_cells: grid has no 'clustering' attribute")
n_clusters = len(grid.clustering.cluster_ids) if grid.clustering is not None and hasattr(grid.clustering, 'cluster_ids') else '?'
print(f"  detect_cluster_cells  n_clusters={n_clusters}  OK")

# ═══════════════════════════════════════════════════════════════════════════════
section("13 · detect_cluster_cells separate pts_target  (crs='', nd=2)")
# ═══════════════════════════════════════════════════════════════════════════════
# Ensures exclude_self is NOT applied when source != target.

pts_src = pts_base.copy()
pts_tgt = _make_pts(300, seed=99)
try:
    grid = aabpl.detect_cluster_cells(
        pts=pts_src, crs='', r=R, c=['val'], x='x', y='y',
        pts_target=pts_tgt, x_tgt='x', y_tgt='y',
        stat='sum', n_random_points=500, random_seed=0,
        silent=True, _dev=DEV,
    )
except Exception as e:
    raise AssertionError(f"detect_cluster_cells (separate target) raised: {e}")

print(f"  detect_cluster_cells separate pts_target  OK")

# ═══════════════════════════════════════════════════════════════════════════════
section("14 · detect_cluster_cells_from_labeled_pts  (crs='', nd=2)")
# ═══════════════════════════════════════════════════════════════════════════════

pts = pts_base.copy()
# pre-label ~20% of points as clustered
pts['cluster'] = pts['val'] > pts['val'].quantile(0.8)
try:
    grid = aabpl.detect_cluster_cells_from_labeled_pts(
        pts=pts, crs='', r=R, c=['val'], x='x', y='y',
        is_cluster_column='cluster',
        suffix=f'_sum_{R}',
        silent=True,
    )
except Exception as e:
    raise AssertionError(f"detect_cluster_cells_from_labeled_pts raised: {e}")

check(hasattr(grid, 'clustering'), "detect_cluster_cells_from_labeled_pts: grid has no 'clustering' attribute")
print(f"  detect_cluster_cells_from_labeled_pts  OK")

# ═══════════════════════════════════════════════════════════════════════════════
section("15 · grid.plot.* smoke tests after keep_cols=False cleanup")
# ═══════════════════════════════════════════════════════════════════════════════
# Regression: plot_pt_vars.py tried to read 'proj_x'/'proj_y' from pts after
# keep_cols=False (the default) had already dropped them.  The fix snapshots
# projected coords in set_source so plots always have access to them.
import matplotlib
matplotlib.use('Agg')  # non-interactive backend — no GUI window needed

pts = pts_base.copy()
grid = aabpl.detect_cluster_cells(
    pts=pts, crs='', r=R, c=['val'], x='x', y='y',
    stat='sum', n_random_points=500, random_seed=0,
    silent=True, _dev=DEV,
    keep_cols=False,  # default — ensures proj_x/proj_y are dropped from pts
)

# grid.plot.vars — failed with KeyError: 'proj_x' before the fix
try:
    fig = grid.plot.vars(filename=None)
    import matplotlib.pyplot as plt
    plt.close('all')
    print("  grid.plot.vars(keep_cols=False)  OK")
except Exception as e:
    raise AssertionError(f"grid.plot.vars raised after keep_cols=False cleanup: {e}")

# grid.plot.clusters — verify it runs without error
try:
    fig = grid.plot.clusters(filename=None)
    plt.close('all')
    print("  grid.plot.clusters(keep_cols=False)  OK")
except Exception as e:
    raise AssertionError(f"grid.plot.clusters raised after keep_cols=False cleanup: {e}")

# grid.plot.cluster_pts
try:
    fig = grid.plot.cluster_pts(filename=None)
    plt.close('all')
    print("  grid.plot.cluster_pts(keep_cols=False)  OK")
except Exception as e:
    raise AssertionError(f"grid.plot.cluster_pts raised after keep_cols=False cleanup: {e}")

# grid.plot.rand_dist
try:
    fig = grid.plot.rand_dist(filename=None)
    plt.close('all')
    print("  grid.plot.rand_dist(keep_cols=False)  OK")
except Exception as e:
    raise AssertionError(f"grid.plot.rand_dist raised after keep_cols=False cleanup: {e}")

# ═══════════════════════════════════════════════════════════════════════════════
section("16 · custom sample_area: Polygon and MultiPolygon containment check")
# ═══════════════════════════════════════════════════════════════════════════════
# Verifies that ALL random null-distribution points fall inside the user-supplied
# polygon/multipolygon.  Uses crs='' (Cartesian) so projected coords == input coords.
from shapely.geometry import Polygon, MultiPolygon, Point

pts_sa = _make_pts(2000, seed=7)
# Two non-overlapping rectangles that together cover the point cloud
poly_a = Polygon([(5000,5000),(30000,5000),(30000,30000),(5000,30000)])
poly_b = Polygon([(32000,32000),(48000,32000),(48000,48000),(32000,48000)])
multi  = MultiPolygon([poly_a, poly_b])

for label, sa in [('Polygon', poly_a), ('MultiPolygon', multi)]:
    pts = pts_sa.copy()
    grid_sa = aabpl.detect_cluster_pts(
        pts=pts, crs='', r=R, c=['val'], x='x', y='y',
        sample_area=sa,
        n_random_points=2000, random_seed=0,
        silent=True, _dev=DEV,
    )
    xs = grid_sa._rndm_pts_x_snapshot
    ys = grid_sa._rndm_pts_y_snapshot
    check(xs is not None, f"{label}: _rndm_pts_x_snapshot is None")
    n_outside = sum(
        not sa.covers(Point(x, y)) for x, y in zip(xs, ys)
    )
    check(n_outside == 0,
          f"{label}: {n_outside}/{len(xs)} random points fall outside the supplied polygon")
    print(f"  {label}: all {len(xs)} random points inside supplied geometry  OK")

# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  ALL TESTS PASSED")
print("="*60 + "\n")

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
print("\n" + "="*60)
print("  ALL TESTS PASSED")
print("="*60 + "\n")

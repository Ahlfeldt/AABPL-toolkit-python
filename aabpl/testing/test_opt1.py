"""
Opt 1 correctness + timing test.

Compares vec=0 (baseline, per-point set ops but now using trigger grid)
and vec=1 (vectorised block grouping) for the exact area_weight path.

Run from repo root:
    python aabpl/testing/test_opt1.py
"""
import sys, time, statistics
sys.path.insert(0, 'Z:/Algorithm/PL_python/AABPL-toolkit-python')
[sys.modules.pop(m) for m in list(sys.modules.keys()) if m.startswith('aabpl')]

import numpy as np
import pandas as pd
import aabpl
import aabpl.config as config

rs = aabpl.radius_search

_COLS = ['eid', 'employment', 'industry', 'lat', 'lon', 'moved']
_raw_15k  = pd.read_csv('Z:/Algorithm/cbsa_sample_data/plants_10180.txt',
                         sep=',', header=None)
_raw_15k.columns = _COLS
_raw_521k = pd.read_csv('Z:/Algorithm/cbsa_sample_data/plants_16980.txt',
                         sep=',', header=None)
_raw_521k.columns = _COLS

CRS, X, Y, C, R = 'EPSG:4326', 'lon', 'lat', 'employment', 4000

# Study area combos that exercise the boundary trigger path.
# m=1,b=400 is the worst-case (tightest boundary, most boundary cells).
COMBOS = [
    ('cells,m=1,b=400',  'tight'),
    ('cells,m=1,b=4000', 'wide'),
    ('cells,m=5,b=400',  'tight+dense'),
]

N_REPS   = 3
N_WARMUP = 1

def run_one(pts_template, wva, sa, n_reps, n_warmup):
    """Return list of saa times (seconds) for n_reps runs."""
    times = []
    for rep in range(n_warmup + n_reps):
        pts = pts_template.copy()
        t0 = time.perf_counter()
        rs(pts, crs=CRS, r=R, c=[C], x=X, y=Y, study_area=sa,
           area_weight=wva, silent=True)
        elapsed = time.perf_counter() - t0
        if rep >= n_warmup:
            times.append(elapsed)
    return times

def get_sums(pts_template, wva, sa):
    """Return the result sum column as a numpy array for correctness check."""
    pts = pts_template.copy()
    rs(pts, crs=CRS, r=R, c=[C], x=X, y=Y, study_area=sa,
       area_weight=wva, silent=True)
    # radius_search adds result columns in-place on pts
    added = [c for c in pts.columns if c not in pts_template.columns]
    sum_cols = [c for c in added if C in c]
    return pts[sum_cols[0]].values if sum_cols else None

# ---- Correctness: vec=0 vs vec=1 should match exactly --------------------
print("=== Correctness: exact,vec=0 vs exact (vec=1) ===")
_pts_small = _raw_15k[['lat', 'lon', 'employment']].head(500).copy()
config.VALIDATE_AREA = True
for sa, label in COMBOS:
    s0 = get_sums(_pts_small, 'exact,vec=0', sa)
    s1 = get_sums(_pts_small, 'exact',       sa)
    if s0 is None or s1 is None:
        print(f"  {label:20s}  could not extract sum column")
        continue
    # NaNs from points outside study area are expected to match too
    nan_match = np.all(np.isnan(s0) == np.isnan(s1))
    val_mask  = ~np.isnan(s0) & ~np.isnan(s1)
    if val_mask.sum() == 0:
        max_diff = 0.0
    else:
        max_diff = float(np.abs(s0[val_mask] - s1[val_mask]).max())
    status = 'PASS' if nan_match and max_diff < 1e-6 else 'FAIL'
    print(f"  {label:20s}  {status}  nan_match={nan_match}  max_diff={max_diff:.2e}")
    sys.stdout.flush()
config.VALIDATE_AREA = False

# Also run VALIDATE_AREA for the grid-level validation
print("\n=== VALIDATE_AREA check ===")
config.VALIDATE_AREA = True
for sa, label in COMBOS:
    pts = _pts_small.copy()
    g = rs(pts, crs=CRS, r=R, c=[C], x=X, y=Y, study_area=sa,
           area_weight='exact', silent=True)
    val = getattr(g, 'area_weight_validation', {}).get('exact') or getattr(g, 'area_weight_validation', {}).get('valid_area_share_4000')
    if val:
        status = 'PASS' if val['max_diff'] < 1e-6 else 'FAIL'
        print(f"  {label:20s}  {status}  max_diff={val['max_diff']:.2e}  mad={val['mad']:.2e}")
    else:
        print(f"  {label:20s}  no validation data")
    sys.stdout.flush()
config.VALIDATE_AREA = False

# ---- Timing: vec=0 vs vec=1, 15k and 521k -----------------------------------
print("\n=== Timing (15k n=500) ===")
print(f"  {'combo':<22}  {'vec=0 ms':>10}  {'vec=1 ms':>10}  {'speedup':>8}")
for sa, label in COMBOS:
    t0 = statistics.median(run_one(_pts_small, 'exact,vec=0', sa, N_REPS, N_WARMUP)) * 1000
    t1 = statistics.median(run_one(_pts_small, 'exact',       sa, N_REPS, N_WARMUP)) * 1000
    print(f"  {label:<22}  {t0:>10.0f}  {t1:>10.0f}  {t0/t1:>8.2f}x")
    sys.stdout.flush()

print("\n=== Timing (15k full) ===")
_pts_15k = _raw_15k[['lat', 'lon', 'employment']].copy()
print(f"  {'combo':<22}  {'vec=0 ms':>10}  {'vec=1 ms':>10}  {'speedup':>8}")
for sa, label in COMBOS:
    t0 = statistics.median(run_one(_pts_15k, 'exact,vec=0', sa, N_REPS, N_WARMUP)) * 1000
    t1 = statistics.median(run_one(_pts_15k, 'exact',       sa, N_REPS, N_WARMUP)) * 1000
    print(f"  {label:<22}  {t0:>10.0f}  {t1:>10.0f}  {t0/t1:>8.2f}x")
    sys.stdout.flush()

print("\n=== Timing (521k, 2 reps) ===")
_pts_521k = _raw_521k[['lat', 'lon', 'employment']].copy()
print(f"  {'combo':<22}  {'vec=0 ms':>10}  {'vec=1 ms':>10}  {'speedup':>8}")
for sa, label in COMBOS[:2]:   # skip dense for time
    t0 = statistics.median(run_one(_pts_521k, 'exact,vec=0', sa, 2, 1)) * 1000
    t1 = statistics.median(run_one(_pts_521k, 'exact',       sa, 2, 1)) * 1000
    print(f"  {label:<22}  {t0:>10.0f}  {t1:>10.0f}  {t0/t1:>8.2f}x")
    sys.stdout.flush()

print("\nDone.")

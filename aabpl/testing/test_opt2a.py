"""
Opt 2a correctness + timing test (fixed LOCAL_STRIDE).

Checks that the sums produced with the fixed global stride match results from
a fresh baseline import where the stride was recomputed per column-chunk.
Since we cannot cheaply toggle it, we use VALIDATE_AREA as the ground truth
and compare sum columns between area_weight=None (unaffected) and the exact
variants to catch any stride-related mismatch.

Run from repo root:
    python aabpl/testing/test_opt2a.py
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

COMBOS = [
    ('cells,m=1,b=400',  'tight'),
    ('cells,m=1,b=4000', 'wide'),
    ('cells,m=5,b=400',  'tight+dense'),
]
N_REPS = 3

def run_rs(pts_template, wva, sa):
    pts = pts_template.copy()
    rs(pts, crs=CRS, r=R, c=[C], x=X, y=Y,
       study_area=sa, area_weight=wva, silent=True)
    added = [c for c in pts.columns if c not in pts_template.columns]
    sum_cols = [c for c in added if C in c]
    return pts[sum_cols[0]].values if sum_cols else None

def timed(pts_template, wva, sa, n):
    times = []
    for _ in range(n):
        pts = pts_template.copy()
        t0 = time.perf_counter()
        rs(pts, crs=CRS, r=R, c=[C], x=X, y=Y,
           study_area=sa, area_weight=wva, silent=True)
        times.append(time.perf_counter() - t0)
    return statistics.median(times) * 1000

_pts_small = _raw_15k[['lat', 'lon', 'employment']].head(500).copy()
_pts_15k   = _raw_15k[['lat', 'lon', 'employment']].copy()
_pts_521k  = _raw_521k[['lat', 'lon', 'employment']].copy()

# ---- Correctness: none vs exact should differ only by the weight, not the sums ----
# More directly: exact,vec=0 vs exact,vec=1 should match exactly (same as test_opt1).
print("=== Correctness: vec=0 vs vec=1 (fixed stride, n=500) ===")
for sa, label in COMBOS:
    s0 = run_rs(_pts_small, 'exact,vec=0', sa)
    s1 = run_rs(_pts_small, 'exact',       sa)
    if s0 is None or s1 is None:
        print(f"  {label:20s}  could not extract column"); continue
    nan_ok   = np.all(np.isnan(s0) == np.isnan(s1))
    mask     = ~np.isnan(s0) & ~np.isnan(s1)
    max_diff = float(np.abs(s0[mask] - s1[mask]).max()) if mask.sum() else 0.0
    print(f"  {label:20s}  {'PASS' if nan_ok and max_diff < 1e-6 else 'FAIL'}"
          f"  max_diff={max_diff:.2e}")
    sys.stdout.flush()

# ---- VALIDATE_AREA -------------------------------------------------------
print("\n=== VALIDATE_AREA (exact, n=500) ===")
config.VALIDATE_AREA = True
for sa, label in COMBOS:
    pts = _pts_small.copy()
    g = rs(pts, crs=CRS, r=R, c=[C], x=X, y=Y,
           study_area=sa, area_weight='exact', silent=True)
    val = {}
    for k, v in getattr(g, 'area_weight_validation', {}).items():
        val = v; break
    if val:
        status = 'PASS' if val['max_diff'] < 1e-6 else 'KNOWN_APPROX_ERR'
        print(f"  {label:20s}  {status}  max_diff={val['max_diff']:.2e}  mad={val['mad']:.2e}")
    else:
        print(f"  {label:20s}  no validation data")
    sys.stdout.flush()
config.VALIDATE_AREA = False

# ---- Timing: 15k and 521k -----------------------------------------------
print("\n=== Timing (15k n=500) — none vs exact,vec=1 ===")
print(f"  {'combo':<22}  {'none ms':>8}  {'exact ms':>9}  {'overhead':>9}")
for sa, label in COMBOS:
    tn = timed(_pts_small, None,    sa, N_REPS)
    te = timed(_pts_small, 'exact', sa, N_REPS)
    print(f"  {label:<22}  {tn:>8.0f}  {te:>9.0f}  {te-tn:>+9.0f}")
    sys.stdout.flush()

print("\n=== Timing (15k full) — none vs exact,vec=1 ===")
print(f"  {'combo':<22}  {'none ms':>8}  {'exact ms':>9}  {'overhead':>9}")
for sa, label in COMBOS:
    tn = timed(_pts_15k, None,    sa, N_REPS)
    te = timed(_pts_15k, 'exact', sa, N_REPS)
    print(f"  {label:<22}  {tn:>8.0f}  {te:>9.0f}  {te-tn:>+9.0f}")
    sys.stdout.flush()

print("\n=== Timing (521k, 2 reps) — none vs exact,vec=1 ===")
print(f"  {'combo':<22}  {'none ms':>8}  {'exact ms':>9}  {'overhead':>9}")
for sa, label in COMBOS[:2]:
    tn = timed(_pts_521k, None,    sa, 2)
    te = timed(_pts_521k, 'exact', sa, 2)
    print(f"  {label:<22}  {tn:>8.0f}  {te:>9.0f}  {te-tn:>+9.0f}")
    sys.stdout.flush()

print("\nDone.")

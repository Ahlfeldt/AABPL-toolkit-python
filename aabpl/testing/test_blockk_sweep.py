"""
block_k sweep: block_k=2..12, r=250/750/1500, buffer=0/r//2.
Datasets: 15k then 521k. Flush after every combination.

SAA overhead is approximated as:  t_exact(block_k) - t_none
  t_none  captures grid setup + basic aggregation (block_k-independent).
  The difference isolates the exact area-weight computation cost.

Informational columns stored per row:
  dataset, n_pts, r, spacing, buf, block_k, triggered, t_none_ms, t_exact_ms, saa_ms

Run from repo root:
    python aabpl/testing/test_blockk_sweep.py
"""
import sys, time, statistics
sys.path.insert(0, 'Z:/Algorithm/PL_python/AABPL-toolkit-python')
[sys.modules.pop(m) for m in list(sys.modules.keys()) if m.startswith('aabpl')]

import numpy as np
import pandas as pd
import aabpl

rs = aabpl.radius_search

_COLS = ['eid', 'employment', 'industry', 'lat', 'lon', 'moved']
_raw_15k = pd.read_csv('Z:/Algorithm/cbsa_sample_data/plants_10180.txt',
                        sep=',', header=None)
_raw_15k.columns = _COLS
_raw_521k = pd.read_csv('Z:/Algorithm/cbsa_sample_data/plants_16980.txt',
                         sep=',', header=None)
_raw_521k.columns = _COLS

CRS, X, Y, C = 'EPSG:4326', 'lon', 'lat', 'employment'

BLOCK_KS = list(range(2, 13))
RS       = [500, 1000, 2000]
N_REPS   = 2

# ---------------------------------------------------------------------------

def _call(pts_template, r, sa, wva):
    pts = pts_template.copy()
    g   = rs(pts, crs=CRS, r=r, c=[C], x=X, y=Y,
              study_area=sa, area_weight=wva, silent=True)
    return pts, g

def timed_median(pts_template, r, sa, wva, n):
    times = []
    last_pts = None
    for _ in range(n):
        t0 = time.perf_counter()
        pts, _ = _call(pts_template, r, sa, wva)
        times.append(time.perf_counter() - t0)
        last_pts = pts
    return statistics.median(times) * 1000, last_pts

def get_spacing(g):
    """Pull grid cell size from the returned Grid object if available."""
    for attr in ('spacing', 'cell_size', '_spacing', '_cell_size', 'h'):
        if hasattr(g, attr):
            v = getattr(g, attr)
            if v is not None:
                return float(v)
    return float('nan')

# ---------------------------------------------------------------------------
# Header
HDR = (f"{'dataset':>8} {'n_pts':>7} {'r':>5} {'spacing':>9} "
       f"{'buf':>5} {'block_k':>8} {'triggered':>10} "
       f"{'t_none':>9} {'t_exact':>9} {'saa_oh':>9}")
SEP = '-' * len(HDR)
print(HDR)
print(SEP)
sys.stdout.flush()

results = []

for dataset_name, raw in [('15k', _raw_15k), ('521k', _raw_521k)]:
    pts_template = raw[['lat', 'lon', 'employment']].copy()
    n_pts = len(pts_template)

    for r in RS:
        for buf in [0, r // 2]:
            sa = f'cells,m=1,b={buf}'

            # --- baseline: area_weight=None (grid setup + plain aggregation) ---
            t_none_ms, _ = timed_median(pts_template, r, sa, None, N_REPS)

            # grab spacing once from a none run
            _, g_ref = _call(pts_template, r, sa, None)
            spacing  = get_spacing(g_ref)

            for bk in BLOCK_KS:
                wva = f'exact,block_k={bk}'
                t_exact_ms, last_pts = timed_median(pts_template, r, sa, wva, N_REPS)
                saa_oh = t_exact_ms - t_none_ms

                # triggered = pts where valid_area_share < 1
                share_col = f'valid_area_share_{r}'
                if share_col in last_pts.columns:
                    triggered = int((last_pts[share_col] < 1.0 - 1e-9).sum())
                else:
                    triggered = -1

                row = dict(
                    dataset=dataset_name, n_pts=n_pts, r=r,
                    spacing=spacing, buf=buf, block_k=bk,
                    triggered=triggered,
                    t_none_ms=t_none_ms, t_exact_ms=t_exact_ms, saa_oh_ms=saa_oh,
                )
                results.append(row)

                sp_str = f'{spacing:.1f}' if not np.isnan(spacing) else '?'
                print(
                    f"{dataset_name:>8} {n_pts:>7} {r:>5} {sp_str:>9} "
                    f"{buf:>5} {bk:>8} {triggered:>10} "
                    f"{t_none_ms:>9.0f} {t_exact_ms:>9.0f} {saa_oh:>+9.0f}"
                )
                sys.stdout.flush()

print(SEP)
print("Done.")

# Optional: save results to CSV for further analysis
try:
    out = pd.DataFrame(results)
    out_path = 'Z:/Algorithm/PL_python/AABPL-toolkit-python/aabpl/testing/blockk_sweep_results.csv'
    out.to_csv(out_path, index=False)
    print(f"Results saved to {out_path}")
except Exception as e:
    print(f"Could not save CSV: {e}")

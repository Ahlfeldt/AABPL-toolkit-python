"""
Timing benchmark: area_weight variants (chunk path) + old vs chunk comparison.

Run from repo root:
    python aabpl/testing/run_timing_benchmark.py

Results saved to tests/timing_benchmark.json
"""
import sys, time, json, os, statistics
sys.path.insert(0, 'Z:/Algorithm/PL_python/AABPL-toolkit-python')
[sys.modules.pop(m) for m in list(sys.modules.keys()) if m.startswith('aabpl')]

import numpy as np
import pandas as pd

import aabpl
import aabpl.config as config
from aabpl.search.algorithm.disk_aggregation_chunk import search_and_aggregate as _chunk_saa
from aabpl.search.algorithm.disk_aggregation   import search_and_aggregate as _orig_saa
import aabpl.search.algorithm.disk_search as _ds

rs = aabpl.radius_search

# -- Datasets ------------------------------------------------------------------
_COLS = ['eid', 'employment', 'industry', 'lat', 'lon', 'moved']
_CSV_15K  = 'Z:/Algorithm/cbsa_sample_data/plants_10180.txt'
_CSV_521K = 'Z:/Algorithm/cbsa_sample_data/plants_16980.txt'
_raw_15k  = pd.read_csv(_CSV_15K,  sep=',', header=None); _raw_15k.columns  = _COLS
_raw_521k = pd.read_csv(_CSV_521K, sep=',', header=None); _raw_521k.columns = _COLS

AW_CRS = 'EPSG:4326'
AW_X, AW_Y, AW_C = 'lon', 'lat', 'employment'

AW_R        = 4000
AW_VARIANTS = [
    'exact,vec=0', 'exact,block_k=2,vec=0', 'exact,block_k=4,vec=0',  # old per-point loop
    'exact',       'exact,block_k=2',        'exact,block_k=4',        # new vectorised loop
    'logit', 'flat', 'binary',
]
AW_MIN_PTS  = [0, 1, 5]
AW_BUFFS    = [AW_R, 0.1 * AW_R, 2 * AW_R]   # r, 0.1r, 2r
N_REPS_A    = 3   # reps per combo for 15k dataset
N_REPS_A_LARGE = 2  # reps per combo for 521k dataset (fewer to save time)

# Timing hooks: wrap chunk saa and intersect_polygon_with_grid to capture
# sub-timings and isolate them from reprojection / grid setup overhead.
# intersect_polygon_with_grid is called from main.py under its locally-imported
# name, so we must patch aabpl.main.intersect_polygon_with_grid (not just the
# source module) to intercept calls made from radius_search.
import aabpl.search.study_area as _sa_mod
import aabpl.main as _main_mod
_orig_ipwg = _sa_mod.intersect_polygon_with_grid
_t_saa  = []   # seconds spent in search_and_aggregate
_t_ipwg = []   # seconds spent in intersect_polygon_with_grid

def _timed_chunk_saa(*args, **kwargs):
    t0 = time.perf_counter()
    result = _chunk_saa(*args, **kwargs)
    _t_saa.append(time.perf_counter() - t0)
    return result

def _timed_ipwg(*args, **kwargs):
    t0 = time.perf_counter()
    result = _orig_ipwg(*args, **kwargs)
    _t_ipwg.append(time.perf_counter() - t0)
    return result

def _time_rs(pts_template, r, wva, n_reps, poly=None, use_chunk=False,
             crs=None, x='x', y='y', c=None, return_grid=False):
    """Run radius_search n_reps times; return (total_times, saa_times, ipwg_times[, grid])."""
    if c is None:
        c = ['val']
    if use_chunk:
        _ds.search_and_aggregate = _timed_chunk_saa
        _main_mod.intersect_polygon_with_grid = _timed_ipwg
    else:
        _ds.search_and_aggregate = _orig_saa
        _main_mod.intersect_polygon_with_grid = _orig_ipwg
    total_times, saa_times, ipwg_times = [], [], []
    last_grid = None
    for _ in range(n_reps):
        pts = pts_template.copy()
        _t_saa.clear(); _t_ipwg.clear()
        t0 = time.perf_counter()
        last_grid = rs(pts, crs=crs or '', r=r, c=c, x=x, y=y,
                       study_area=poly, area_weight=wva, silent=True)
        total_times.append(time.perf_counter() - t0)
        saa_times.append(sum(_t_saa))
        ipwg_times.append(sum(_t_ipwg))
    _main_mod.intersect_polygon_with_grid = _orig_ipwg
    if return_grid:
        return total_times, saa_times, ipwg_times, last_grid
    return total_times, saa_times, ipwg_times

def _run_section_a(pts_template, label, n_reps=None):
    if n_reps is None:
        n_reps = N_REPS_A
    print(f"\n--A ({label}): area_weight variants -- {len(pts_template)} pts, r={AW_R}m, reps={n_reps} --")
    print("     9 combos: min_pts in {0,1,5} x buffer in {r, 0.1r, 2r}")
    print("     Warming up...")
    _time_rs(pts_template, AW_R, None, 1, poly=f'cells,m=0,b={AW_R}',
             use_chunk=True, crs=AW_CRS, x=AW_X, y=AW_Y, c=[AW_C])

    ratio_table      = {v: [] for v in AW_VARIANTS}
    combo_rows       = []
    for min_pts in AW_MIN_PTS:
        for buff in AW_BUFFS:
            sa = f'cells,m={min_pts},b={int(buff)}'
            _, saas, _ = _time_rs(pts_template, AW_R, None, n_reps, poly=sa,
                                  use_chunk=True, crs=AW_CRS, x=AW_X, y=AW_Y, c=[AW_C])
            saa_none = statistics.median(saas)
            t_var_saa = {}
            for wva in AW_VARIANTS:
                _, saas2, _ = _time_rs(pts_template, AW_R, wva, n_reps, poly=sa,
                                       use_chunk=True, crs=AW_CRS, x=AW_X, y=AW_Y, c=[AW_C])
                t_var_saa[wva] = statistics.median(saas2)
            oh_exact = t_var_saa['exact'] - saa_none
            row = {'spec': sa, 'saa_none_ms': round(saa_none*1000, 1)}
            for wva in AW_VARIANTS:
                oh = t_var_saa[wva] - saa_none
                ratio_table[wva].append(oh / oh_exact if oh_exact > 0 else float('nan'))
                row[f'{wva}_saa_ms'] = round(t_var_saa[wva]*1000, 1)
            combo_rows.append(row)
            spec_short = f"m={min_pts} b={int(buff)}"
            print(f"  {spec_short:15s}  none={saa_none*1000:.0f}ms  " +
                  "  ".join(f"{v}:{t_var_saa[v]*1000:.0f}ms" for v in AW_VARIANTS))
            sys.stdout.flush()

    print(f"\n  Avg SAA time (ms) and ratio vs new-exact across 9 combos:")
    print(f"  {'variant':<28}  {'avg_ms':>7}  {'vs_new_exact':>12}")
    for wva in AW_VARIANTS:
        avg_ms  = statistics.mean(row[f'{wva}_saa_ms'] for row in combo_rows)
        valid   = [x for x in ratio_table[wva] if x == x]
        avg_rat = sum(valid) / len(valid) if valid else float('nan')
        print(f"  {wva:<28}  {avg_ms:>7.0f}  {avg_rat:>12.3f}x")
    return combo_rows, ratio_table

# -- Section A: 15k dataset (small + full) ------------------------------------
_pts_15k_small = _raw_15k[['lat', 'lon', 'employment']].head(500).copy()
_pts_15k_full  = _raw_15k[['lat', 'lon', 'employment']].copy()

combo_rows_15k_small, ratio_15k_small = _run_section_a(_pts_15k_small, f"15k-dataset n=500",     n_reps=N_REPS_A)
combo_rows_15k_full,  ratio_15k_full  = _run_section_a(_pts_15k_full,  f"15k-dataset n={len(_raw_15k)}", n_reps=N_REPS_A)

# -- Section A: 521k dataset (full only, fewer reps) -------------------------
_pts_521k = _raw_521k[['lat', 'lon', 'employment']].copy()
combo_rows_521k, ratio_521k = _run_section_a(_pts_521k, f"521k-dataset n={len(_raw_521k)}", n_reps=N_REPS_A_LARGE)

aw_results = {
    '15k_n500':  {'combos': combo_rows_15k_small, 'saa_ratio_per_combo': {v: [round(x,3) for x in ratio_15k_small[v]] for v in AW_VARIANTS}},
    '15k_full':  {'combos': combo_rows_15k_full,  'saa_ratio_per_combo': {v: [round(x,3) for x in ratio_15k_full[v]]  for v in AW_VARIANTS}},
    '521k_full': {'combos': combo_rows_521k,       'saa_ratio_per_combo': {v: [round(x,3) for x in ratio_521k[v]]      for v in AW_VARIANTS}},
}

# --Section A2: precision (MAD + max_diff vs exact Shapely) --------------------
print("\n--A2: precision — MAD and max_diff vs exact Shapely (one combo each) --")
config.VALIDATE_AREA = True
_prec_sa = f'cells,m=1,b={int(0.1 * AW_R)}'
prec_results = {}
_prec_seen = set()  # skip vec=0 duplicates — same precision as vec=1
for wva in AW_VARIANTS:
    _val_key = wva.split(',', 1)[0] if ',' in wva else wva
    _dedup_key = wva.replace(',vec=0', '')
    if _dedup_key in _prec_seen:
        prec_results[wva] = prec_results.get(_dedup_key)
        print(f"  {wva:<28}  (same as {_dedup_key})")
        sys.stdout.flush()
        continue
    _, _, _, _g = _time_rs(_pts_15k_small, AW_R, wva, 1, poly=_prec_sa,
                           use_chunk=True, crs=AW_CRS, x=AW_X, y=AW_Y,
                           c=[AW_C], return_grid=True)
    val = getattr(_g, 'area_weight_validation', {}).get(_val_key)
    if val:
        prec_results[wva] = val
        print(f"  {wva:<28}  MAD={val['mad']:.6f}  max_diff={val['max_diff']:.6f}")
    else:
        prec_results[wva] = None
        print(f"  {wva:<28}  (no validation data)")
    _prec_seen.add(_dedup_key)
    sys.stdout.flush()
config.VALIDATE_AREA = False

# --Section B: old vs chunk, multiple n and r ----------------------------------
N_REPS_B = 5
print("\n--B: old vs chunk path (area_weight=None, real data, varying n) --")

PATH_CONFIGS = [
    {'n':   100},
    {'n':   500},
    {'n':  5000},
    {'n': 15000},
]
path_results = {}
for cfg in PATH_CONFIGS:
    n = cfg['n']
    key = f"n={n}_r={AW_R}"
    pts_t = _raw_15k[['lat', 'lon', 'employment']].head(n).copy()
    orig_times,  _, _ = _time_rs(pts_t, AW_R, None, N_REPS_B, use_chunk=False,
                                  crs=AW_CRS, x=AW_X, y=AW_Y, c=[AW_C])
    chunk_times, _, _ = _time_rs(pts_t, AW_R, None, N_REPS_B, use_chunk=True,
                                  crs=AW_CRS, x=AW_X, y=AW_Y, c=[AW_C])
    om = statistics.median(orig_times)
    cm = statistics.median(chunk_times)
    path_results[key] = {
        'orig_times_s':  [round(t, 4) for t in orig_times],
        'chunk_times_s': [round(t, 4) for t in chunk_times],
        'orig_median_s':  round(om, 4),
        'chunk_median_s': round(cm, 4),
        'speedup': round(om / cm, 2),
    }
    print(f"  n={n:6d} r={AW_R}  orig={om:.3f}s  chunk={cm:.3f}s  speedup={om/cm:.2f}x")
    sys.stdout.flush()
    sys.stdout.flush()

# --Save -----------------------------------------------------------------------
out = {
    'area_weight_variants': {
        'dataset': 'plants_10180_real_EPSG4326',
        'r': AW_R,
        'path': 'chunk',
        'n_reps': N_REPS_A,
        'specs': '9 buff_cells combos: min_pts in {0,1,5} x buff in {r,0.1r,2r}',
        **aw_results,
    },
    'path_comparison': {
        'area_weight': None,
        'n_reps': N_REPS_B,
        'configs': path_results,
    },
}
out_path = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', 'tests', 'timing_benchmark.json'))
with open(out_path, 'w') as f:
    json.dump(out, f, indent=2)
print(f"\nSaved to {out_path}")

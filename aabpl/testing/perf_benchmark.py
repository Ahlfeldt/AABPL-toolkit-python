"""
Performance benchmarking for aabpl radius_search.

Two-phase workflow
------------------
Phase 1 – sweep
    run_sweep() iterates over all combinations of (radius, spacing, nest_depth,
    trynew=1), calls radius_search, captures timing + micro_region_stats, and
    saves one JSON file per run to perf_test/.

Phase 2 – predict
    load_results() aggregates all saved JSONs into a DataFrame.
    build_predictor() fits a model so that, for a new (pts, radius) scenario,
    we can estimate the best (spacing, nest_depth) without running exhaustive
    tests again.

Key insight
-----------
micro_region_stats depend only on (radius / spacing) geometrically, not on the
point data.  So we can reuse micro_region_stats from prior runs that share the
same radius/spacing ratio when predicting on new datasets.
"""

import json
import os
import hashlib
import platform
import socket
import time as _time
from typing import Optional

import numpy as np
import pandas as pd

from aabpl.main import radius_search, Grid
from aabpl.radius_search.spacing_topology import compute_spatial_stats, compute_spacing_breakpoints
from aabpl import config as _aabpl_config
from aabpl.testing.test_performance import (
    reset_perf_times,
    analyze_func_perf,
    func_timer_dict,
)


# ---------------------------------------------------------------------------
# Machine identification
# ---------------------------------------------------------------------------

def get_machine_info() -> dict:
    """
    Collect stable hardware/OS identifiers for controlling across machines.
    Uses only stdlib — no extra dependencies.
    """
    cpu_count = os.cpu_count()
    try:
        import psutil
        ram_gb = round(psutil.virtual_memory().total / 1024**3, 1)
    except ImportError:
        ram_gb = None

    info = {
        "hostname":    socket.gethostname(),
        "os":          platform.system(),
        "os_version":  platform.version(),
        "cpu":         platform.processor() or platform.machine(),
        "cpu_count":   cpu_count,
        "ram_gb":      ram_gb,
        "python":      platform.python_version(),
    }
    # Short stable hash of (hostname, cpu, cpu_count, ram) as a compact ID
    key = f"{info['hostname']}|{info['cpu']}|{cpu_count}|{ram_gb}"
    info["machine_id"] = hashlib.md5(key.encode()).hexdigest()[:8]
    return info


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _scenario_hash(dist_stats: dict, radius: float) -> str:
    """Stable identifier for a (point-cloud, radius) scenario."""
    key = (
        round(radius, 6),
        round(dist_stats.get("spatial_width", 0), 2),
        round(dist_stats.get("spatial_height", 0), 2),
        round(dist_stats.get("total_target_points", 0), 0),
        round(dist_stats.get("density_skewness_max_to_mean", 0), 4),
        round(dist_stats.get("density_skewness_max_to_mean", 0), 4),
    )
    return hashlib.md5(str(key).encode()).hexdigest()[:10]


def _total_process_time(perf_dict: dict) -> float:
    if "grp_df" in perf_dict and perf_dict["grp_df"] is not None:
        return float(perf_dict["grp_df"]["process_time"].sum())
    times = perf_dict.get("times", [])
    if not times:
        return 0.0
    t0 = min(t["start_time"] for t in times)
    t1 = max(t["end_time"] for t in times)
    return float(t1 - t0)


_SEARCH_FUNCS = frozenset({
    "sum_cntd_all_offset_regions",
    "sum_cntd_by_offset_region",
    "get_pts_ovlpd_by_region",
    "assign_points_to_mirco_regions",  # includes build_disk_region_lookups when uncached
})
_AGGREGATE_FUNCS = frozenset({
    "sum_ovlpd_pts_in_radius",
    "aggregate_point_data_to_cells",
})


def _geom_cache_key(radius: float, spacing: float, nest_depth: int) -> tuple:
    """Canonical cache key matching what build_disk_region_lookups stores."""
    return (round(radius / spacing, 8), nest_depth, False)


def _func_timing_summary(perf_dict: dict) -> dict:
    """Collapse per-function timing into a flat dict keyed by func_name.

    Also computes derived categories:
      search_cpu_s    – time in cell-lookup / candidate-retrieval functions
      aggregate_cpu_s – time in value-summation functions
    """
    summary = {}
    for t in perf_dict.get("times", []):
        name = t["func_name"]
        elapsed = t.get("time_elapsed", t["end_time"] - t["start_time"])
        summary[name] = summary.get(name, 0.0) + elapsed

    summary["search_cpu_s"] = sum(summary.get(f, 0.0) for f in _SEARCH_FUNCS)
    summary["aggregate_cpu_s"] = sum(summary.get(f, 0.0) for f in _AGGREGATE_FUNCS)
    return summary


def candidate_spacings_from_radius(radius: float, max_offset: int = 4) -> np.ndarray:
    """Return the canonical candidate spacings for a given radius."""
    ratios = compute_spacing_breakpoints(max_offset=max_offset)
    return radius / ratios


# ---------------------------------------------------------------------------
# Single-run executor
# ---------------------------------------------------------------------------

def run_single_config(
    pts_source: pd.DataFrame,
    crs: str,
    radius: float,
    spacing_ratio: float,
    nest_depth: int,
    *,
    pts_target: Optional[pd.DataFrame] = None,
    col: str = "employment",
    local_crs: str = "auto",
    dist_stats: Optional[dict] = None,
    silent: bool = True,
) -> dict:
    """
    Run radius_search for one (radius, spacing, nest_depth) combination
    and return a result dict containing all metrics needed for Phase-2 prediction.

    Parameters
    ----------
    pts_source  : DataFrame of search origins (can be a sample for screening)
    crs         : CRS string for pts
    radius      : search radius (metres if CRS is metric)
    spacing_ratio : r/spacing ratio
    nest_depth  : Grid nest_depth (0–9)
    pts_target  : DataFrame of points to aggregate over (defaults to pts_source).
                  Pass the full dataset here when pts_source is a sample.
    col         : column name to aggregate
    local_crs   : projected CRS for internal computation
    dist_stats  : pre-computed compute_spatial_stats result (saves time across calls)
    silent      : suppress radius_search console output
    """
    pts_source = pts_source.copy()
    pts_grid = pts_target if pts_target is not None else pts_source

    # -- scenario distribution stats (can be shared across configs) ----------
    if dist_stats is None:
        pts_xy = pts_grid[["lon", "lat"]].values if "lat" in pts_grid.columns else pts_grid.iloc[:, :2].values
        dist_stats = compute_spatial_stats(
            target_points=pts_xy,
            search_radii=[radius],
        )

    scenario_id = _scenario_hash(dist_stats, radius)

    # -- run -----------------------------------------------------------------
    import aabpl.config as _cfg
    _cfg.FIXED_SPACING_RATIO = spacing_ratio
    _cfg.FIXED_NEST_DEPTH = nest_depth
    reset_perf_times()
    _cache_keys_before = set(_aabpl_config.disk_region_cache.keys())
    t_wall_start = _time.perf_counter()
    grid_result = radius_search(
        pts=pts_source,
        crs=crs,
        r=radius,
        c=col,
        exclude_pt_itself=True,
        silent=silent,
        trynew=1,
        proj_crs=local_crs,
        pts_target=pts_target,
    )
    _cfg.FIXED_SPACING_RATIO = None
    _cfg.FIXED_NEST_DEPTH = None
    t_wall = _time.perf_counter() - t_wall_start
    geometry_cached = set(_aabpl_config.disk_region_cache.keys()) == _cache_keys_before

    # -- collect timing ------------------------------------------------------
    perf = analyze_func_perf(plot=False)
    total_cpu = _total_process_time(perf)
    func_times = _func_timing_summary(perf)


    # -- micro-region stats --------------------------------------------------
    try:
        micro = grid_result.calc_micro_region_stats()
        # flatten nested dict
        micro_flat = {}
        for outer_key, inner in micro.items():
            for inner_key, val in inner.items():
                micro_flat[f"micro_{outer_key}_{inner_key}"] = val
    except Exception:
        micro_flat = {}

    # -- assemble result record ----------------------------------------------
    machine = get_machine_info()
    result = {
        "meta": {
            "scenario_id": scenario_id,
            "timestamp": _time.strftime("%Y-%m-%dT%H:%M:%S"),
            "machine_id": machine["machine_id"],
        },
        "machine": machine,
        "config": {
            "radius": radius,
            "spacing_ratio": spacing_ratio,
            "radius_over_spacing": spacing_ratio,
            "nest_depth": nest_depth,
            "col": col,
            "n_source": len(pts_source),
            "n_target": len(pts_grid),
            "geometry_cached": geometry_cached,
        },
        "scenario_stats": dist_stats,
        "micro_region_stats": micro_flat,
        "timing": {
            "total_cpu_s": total_cpu,
            "total_wall_s": t_wall,   # wall time — unreliable under parallel load; use total_cpu_s
            "search_cpu_s": func_times.pop("search_cpu_s", 0.0),
            "aggregate_cpu_s": func_times.pop("aggregate_cpu_s", 0.0),
            **{f"func_{k}": v for k, v in func_times.items()},
        },
    }
    return result


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_run_result(result: dict, output_folder: str) -> str:
    """Save a run-result dict as a JSON file. Returns the file path.

    Any existing file for the same (scenario, radius, spacing, nest_depth,
    n_source, machine) is deleted first so load_results never sees
    stale duplicates from earlier runs.
    """
    os.makedirs(output_folder, exist_ok=True)
    cfg = result["config"]
    meta = result["meta"]
    s_str = str(round(cfg['spacing_ratio'], 2))
    stem = (
        f"run_{meta['scenario_id']}"
        f"_r{cfg['radius']}"
        f"_s{s_str}"
        f"_nd{cfg['nest_depth']}"
        f"_ns{cfg['n_source']}"
        f"_nt{cfg['n_target']}"
        f"_m{meta['machine_id']}_"
    )
    # remove any previous file for this exact config; return deleted names
    deleted = []
    for existing in os.listdir(output_folder):
        if existing.startswith(stem) and existing.endswith(".json"):
            try:
                os.remove(os.path.join(output_folder, existing))
                deleted.append(existing)
            except OSError:
                pass
    fname = stem + meta['timestamp'].replace(':', '').replace('-', '') + ".json"
    fpath = os.path.join(output_folder, fname)
    with open(fpath, "w") as f:
        json.dump(result, f, indent=2, default=float)
    return fpath, deleted


def load_results(output_folder: str) -> pd.DataFrame:
    """
    Load all JSON run files from output_folder into a flat DataFrame.
    Each row is one (radius, spacing, nest_depth) run.

    Results are cached in _cache.parquet inside output_folder. On subsequent
    calls only new JSON files (not yet in the cache) are read, making repeated
    loads fast regardless of how many files accumulate.
    """
    cache_path = os.path.join(output_folder, "_cache.parquet")

    # load existing cache and find which filenames are already in it
    cached_df = pd.DataFrame()
    cached_files: set = set()
    if os.path.isfile(cache_path):
        try:
            cached_df = pd.read_parquet(cache_path)
            if "_source_file" in cached_df.columns:
                cached_files = set(cached_df["_source_file"].dropna())
        except Exception:
            cached_df = pd.DataFrame()

    # read only JSON files not yet in the cache
    new_rows = []
    for fname in os.listdir(output_folder):
        if not fname.endswith(".json") or fname in cached_files:
            continue
        with open(os.path.join(output_folder, fname)) as f:
            try:
                d = json.load(f)
            except Exception:
                continue
        row = {"_source_file": fname}
        row.update(d.get("meta", {}))
        for k, v in d.get("machine", {}).items():
            row[f"machine_{k}"] = v
        row.update(d.get("config", {}))
        for k, v in d.get("scenario_stats", {}).items():
            if isinstance(v, list):
                row[f"scen_{k}"] = v[0] if v else None
            else:
                row[f"scen_{k}"] = v
        row.update(d.get("micro_region_stats", {}))
        row.update(d.get("timing", {}))
        new_rows.append(row)

    if new_rows:
        new_df = pd.DataFrame(new_rows)
        df = pd.concat([cached_df, new_df], ignore_index=True) if not cached_df.empty else new_df
        try:
            df.to_parquet(cache_path, index=False)
        except Exception:
            pass
    else:
        df = cached_df

    if df.empty:
        return pd.DataFrame()

    # Enrich with geometry cache stats (n_regions, mean_cntd_count, mean_ovlpd_count).
    # These depend only on (radius_over_spacing, nest_depth) so one cache lookup fills
    # all rows with the same combination, even if most JSON files predate this feature.
    _enrich_geo_stats(df)
    return df


def _enrich_geo_stats(df: pd.DataFrame) -> None:
    """Add n_regions / mean_cntd_count / mean_ovlpd_count from the geometry cache in-place."""
    import aabpl.config as _cfg
    import math

    def _stats(r_over_s, nest_depth):
        key = (round(float(r_over_s), 8), int(nest_depth), False)
        cached = _cfg.disk_region_cache.get(key)
        if cached is None:
            return None, None, None
        region_id_to_cntd  = cached["region_id_to_cntd_cells"]
        region_id_to_ovlpd = cached["region_id_to_ovlpd_cells"]
        region_id_to_area  = cached["region_id_to_area"]
        total_area = sum(region_id_to_area.values()) or 1.0
        n_regions  = len(region_id_to_cntd)
        mean_cntd  = sum(len(v) * region_id_to_area[k] for k, v in region_id_to_cntd.items())  / total_area
        mean_ovlpd = sum(len(v) * region_id_to_area[k] for k, v in region_id_to_ovlpd.items()) / total_area
        return n_regions, mean_cntd, mean_ovlpd

    combos = df[["radius_over_spacing", "nest_depth"]].drop_duplicates()
    stats_map = {
        (row.radius_over_spacing, row.nest_depth): _stats(row.radius_over_spacing, row.nest_depth)
        for row in combos.itertuples(index=False)
    }
    df["n_regions"]       = df.apply(lambda r: stats_map.get((r["radius_over_spacing"], r["nest_depth"]), (None, None, None))[0], axis=1)
    df["mean_cntd_count"] = df.apply(lambda r: stats_map.get((r["radius_over_spacing"], r["nest_depth"]), (None, None, None))[1], axis=1)
    df["mean_ovlpd_count"]= df.apply(lambda r: stats_map.get((r["radius_over_spacing"], r["nest_depth"]), (None, None, None))[2], axis=1)


# ---------------------------------------------------------------------------
# Static geometry stats
# ---------------------------------------------------------------------------

def compute_static_stats(
    r_over_s: float,
    nest_depth: int,
    include_boundary: bool = False,
    saved_df: Optional[pd.DataFrame] = None,
) -> dict:
    """
    Return static geometric statistics for a (r/s, nest_depth) combination.

    Geometric counts (cntd/ovlpd cells per region) come from the geometry cache —
    building it if not already cached.  Build time is taken from ``saved_df`` when
    a matching row exists (keyed on radius_over_spacing and nest_depth), so repeated
    calls avoid re-timing already-measured configs.  Pass ``load_results(folder)``
    as ``saved_df`` to use saved JSON results.
    """
    import time as _t
    from aabpl.radius_search import disk_region_geometry

    _key = (r_over_s, nest_depth, include_boundary)
    already_cached = _key in _aabpl_config.disk_region_cache

    # -- build time: prefer saved results, fall back to measuring --------------
    build_s = None
    if saved_df is not None and not saved_df.empty:
        mask = (
            (saved_df['radius_over_spacing'].round(8) == round(r_over_s, 8))
            & (saved_df['nest_depth'] == nest_depth)
            & (~saved_df.get('geometry_cached', pd.Series([False]*len(saved_df))).astype(bool))
        )
        matched = saved_df[mask]
        if not matched.empty and 'func_build_disk_region_lookups' in matched.columns:
            build_s = float(matched['func_build_disk_region_lookups'].median())

    if build_s is None:
        if already_cached:
            build_s = 0.0
        else:
            t0 = _t.perf_counter()
            disk_region_geometry.build_disk_region_lookups(
                grid={}, grid_spacing=1, r=r_over_s,
                nest_depth=nest_depth, include_boundary=include_boundary,
                silent=True,
            )
            build_s = _t.perf_counter() - t0
    elif not already_cached:
        disk_region_geometry.build_disk_region_lookups(
            grid={}, grid_spacing=1, r=r_over_s,
            nest_depth=nest_depth, include_boundary=include_boundary,
            silent=True,
        )

    cached = _aabpl_config.disk_region_cache[_key]

    region_id_to_cntd  = cached['region_id_to_cntd_cells']
    region_id_to_ovlpd = cached['region_id_to_ovlpd_cells']
    region_id_to_area  = cached['region_id_to_area']
    shared_cntd        = cached['shared_cntd_cells']

    total_area = sum(region_id_to_area.values()) or 1.0
    n_shared   = len(shared_cntd)

    mean_cntd  = (
        sum(len(cells) * region_id_to_area[rid] for rid, cells in region_id_to_cntd.items())
        / total_area
    ) + n_shared
    mean_ovlpd = (
        sum(len(cells) * region_id_to_area[rid] for rid, cells in region_id_to_ovlpd.items())
        / total_area
    )

    return {
        'r_over_s':          r_over_s,
        'nest_depth':        nest_depth,
        'build_s':           build_s,
        'n_regions':         len(region_id_to_cntd),
        'n_shared_cntd':     n_shared,
        'mean_cntd_count':   mean_cntd,
        'mean_ovlpd_count':  mean_ovlpd,
        'mean_total_count':  mean_cntd + mean_ovlpd,
        'area_ratio':        total_area / (3.141592653589793 * r_over_s ** 2),
    }


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------

def run_sweep(
    pts: pd.DataFrame,
    crs: str,
    radii: list,
    spacing_ratios: list,
    nest_depths: list,
    sample_sizes: list,
    *,
    col: str = "employment",
    local_crs: str = "auto",
    output_folder: str = "./perf_test",
    silent: bool = True,
    skip_existing: bool = True,
    verbose: bool = True,
    n_screen: int = 2000,
    top_k: int = 5,
) -> pd.DataFrame:
    """
    Two-phase sweep over (radius, spacing, nest_depth) combinations.

    Phase 1 — screening
        All combinations are run with ``n_screen`` source points (default 2000)
        against the full ``pts`` as target. Loops are ordered sr → nd → r
        so geometry is built once per (sr, nd) group and reused across all radii
        while the cache is warm.

    Phase 2 — validation
        For each radius, the ``top_k`` fastest configs from Phase 1 are re-run
        at each full sample size in ``sample_sizes``.

    Parameters
    ----------
    n_screen  : number of source points used in the screening phase (default 2000)
    top_k     : number of configs per radius to promote to full-scale validation
    """
    from aabpl.utils.crs_transformation import convert_pts_to_crs as _convert_pts_to_crs
    _pts_proj = pts.copy()
    _x_proj, _y_proj, _local_crs = _convert_pts_to_crs(
        pts=_pts_proj,
        x="lon" if "lon" in pts.columns else pts.columns[0],
        y="lat" if "lat" in pts.columns else pts.columns[1],
        initial_crs=crs,
        target_crs=local_crs,
        silent=True,
    )
    pts_xy = _pts_proj[[_x_proj, _y_proj]].values
    dist_stats_cache = {r: compute_spatial_stats(target_points=pts_xy, search_radii=[r]) for r in radii}

    completed = set()
    if skip_existing and os.path.isdir(output_folder):
        # Migrate old-format files (no _n{n_source} segment) by reading the JSON
        # and renaming in place so the new skip pattern matches them.
        for fname in os.listdir(output_folder):
            if not fname.endswith(".json") or "_n" in fname:
                continue
            fpath = os.path.join(output_folder, fname)
            try:
                with open(fpath) as _f:
                    _d = json.load(_f)
                _n = _d.get("config", {}).get("n_source")
                if _n is not None:
                    stem = fname[:-5]  # strip .json
                    # insert _n<value> before _m<machine_id>
                    new_fname = stem.replace("_m", f"_n{_n}_m", 1) + ".json"
                    os.rename(fpath, os.path.join(output_folder, new_fname))
            except Exception:
                pass
        completed = {f for f in os.listdir(output_folder) if f.endswith(".json")}

    from aabpl.utils.progress import SweepProgress

    results = []

    def _run_one(pts_source, r, s, nd, dist_stats, prog=None, done=0, n_total=0, tag=""):
        scenario_id = _scenario_hash(dist_stats, r)
        n_tgt = len(pts) if pts is not None else len(pts_source)
        skip_pattern = f"run_{scenario_id}_r{r}_s{round(s, 2)}_nd{nd}_ns{len(pts_source)}_nt{n_tgt}_"
        label = f"r={r} sr={round(s, 2)} nd={nd}"
        if skip_existing and any(f.startswith(skip_pattern) for f in completed):
            match = next(f for f in completed if f.startswith(skip_pattern))
            if prog:
                prog.clear()
            if verbose:
                print(f"  load  {label}{tag}")
                print(f"         pattern: {skip_pattern}")
                print(f"         matched: {match}")
            if prog:
                prog.redraw()
            try:
                with open(os.path.join(output_folder, match)) as _f:
                    result = json.load(_f)
                results.append(result)
                return result
            except Exception:
                # file was deleted (e.g. by a previous save_run_result) — fall through and rerun
                completed.discard(match)
        if prog:
            prog.update(done, label=f"run   {label}")
        elif verbose:
            print(f"  run   {label} n={len(pts_source)}{tag} ...", end=" ", flush=True)
        try:
            result = run_single_config(
                pts_source=pts_source, crs=crs, radius=r, spacing_ratio=s,
                nest_depth=nd, col=col, pts_target=pts,
                local_crs=local_crs, dist_stats=dist_stats, silent=silent,
            )
            fpath, deleted = save_run_result(result, output_folder)
            for d in deleted:
                completed.discard(d)
            completed.add(os.path.basename(fpath))
            results.append(result)
            if not prog and verbose:
                cached_str = "cached" if result["config"]["geometry_cached"] else "UNCACHED"
                print(f"done  cpu={result['timing']['total_cpu_s']:.2f}s  geom={cached_str}")
            return result
        except Exception as e:
            import traceback as _tb
            import aabpl.config as _cfg
            _cfg.FIXED_SPACING_RATIO = None
            _cfg.FIXED_NEST_DEPTH = None
            if not prog and verbose:
                print(f"ERROR: {e}")
                _tb.print_exc()
            elif prog and verbose:
                print(f"\n  ERROR {label}:")
                _tb.print_exc()
            return None

    # ------------------------------------------------------------------ #
    # Phase 1: screen all configs at n_screen source points               #
    # Loop order: r → s → nd  (geometry cached within r/s/nd group) #
    # ------------------------------------------------------------------ #
    if n_screen > 0:
        screen_sample = pts.sample(min(n_screen, len(pts)), random_state=0)
        n_total_screen = len(radii) * len(spacing_ratios) * len(nest_depths)

        # raise cache limit so no entry is evicted mid-sweep
        import aabpl.config as _cfg
        _cfg.DISK_REGION_CACHE_MAXSIZE = max(
            _cfg.DISK_REGION_CACHE_MAXSIZE,
            len(spacing_ratios) * len(nest_depths) + 4,
        )

        if verbose:
            print(f"\n=== Phase 1: screening {n_total_screen} configs with {len(screen_sample):,} source points ===")
        prog1 = SweepProgress(n_total=n_total_screen)
        prog1.start()

        screen_results = {r: [] for r in radii}
        done = 0
        # outer loop: (sr, nd) — geometry cache key is (sr, nd, False), independent of
        # absolute radius, so one clear per (sr, nd) group lets all radii share the cache.
        for sr in spacing_ratios:
            for nd in nest_depths:
                _aabpl_config.disk_region_cache.clear()
                for r in radii:
                    dist_stats = dist_stats_cache[r]
                    done += 1
                    result = _run_one(screen_sample, r, sr, nd, dist_stats,
                                        prog=prog1, done=done, n_total=n_total_screen,
                                        tag=" [screen]")
                    if result is not None:
                        screen_results[r].append((result["timing"]["total_cpu_s"], sr, nd))

        prog1.done()

    # ------------------------------------------------------------------ #
    # Phase 2: validate top_k configs per radius at full sample sizes     #
    # ------------------------------------------------------------------ #
    n_total_validate = len(radii) * top_k * len(sample_sizes)
    if verbose:
        print(f"\n=== Phase 2: validating top {top_k} configs per radius across {len(sample_sizes)} sample size(s) ===")

    # group Phase 2 configs by (sr, nd) so each geometry is built once and
    # reused across all radii and sample sizes that share that cache key.
    from collections import defaultdict
    config_to_radii: dict = defaultdict(list)
    top_configs_per_r: dict = {}
    for r in radii:
        top = sorted(screen_results.get(r, []))[:top_k]
        top_configs_per_r[r] = top
        if verbose and top:
            print(f"\n  Top {top_k} for r={r}: " + ", ".join(f"sr={sr} nd={nd} ({t:.2f}s)" for t, sr, nd in top))
        for _, sr, nd in top:
            config_to_radii[(sr, nd)].append(r)

    n_total_validate = sum(len(rs) * len(sample_sizes) for rs in config_to_radii.values())
    prog2 = SweepProgress(n_total=n_total_validate)
    prog2.start()
    done = 0
    for (sr, nd), rs in sorted(config_to_radii.items()):
        _aabpl_config.disk_region_cache.clear()
        for r in rs:
            dist_stats = dist_stats_cache[r]
            for sample_size in sample_sizes:
                done += 1
                full_sample = pts.sample(min(sample_size, len(pts)), random_state=0)
                _run_one(full_sample, r, sr, nd, dist_stats,
                         prog=prog2, done=done, n_total=n_total_validate,
                         tag=" [validate]")
    prog2.done()

    if not results:
        return load_results(output_folder)

    rows = []
    for d in results:
        row = {}
        row.update(d.get("meta", {}))
        row.update(d.get("config", {}))
        for k, v in d.get("scenario_stats", {}).items():
            row[f"scen_{k}"] = v[0] if isinstance(v, list) and v else v
        row.update(d.get("micro_region_stats", {}))
        row.update(d.get("timing", {}))
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Phase-2 predictor
# ---------------------------------------------------------------------------

FEATURE_COLS = [
    # scenario features
    "scen_spatial_width", "scen_spatial_height", "scen_spatial_area",
    "scen_spatial_aspect_ratio", "scen_density_skewness_max_to_mean",
    "scen_total_target_points",
    # geometric ratio
    "radius_over_spacing",
    # micro-region stats (geometry-only, reusable across point clouds)
    "micro_count_cntd", "micro_count_ovlpd",
    "micro_count_cntd_weighted", "micro_count_ovlpd_weighted",
    "micro_area_cntd", "micro_area_ovlpd",
    # area as share of circle area π·r² (depends only on r/spacing + nest_depth)
    "micro_area_share_cntd", "micro_area_share_ovlpd",
    "micro_area_share_cntd_weighted", "micro_area_share_ovlpd_weighted",
    # config
    "nest_depth",
]

TARGET_COL = "total_cpu_s"


def build_predictor(df: pd.DataFrame):
    """
    Fit a simple predictor: given scenario + config features, predict CPU time.

    Returns a callable predict(features_df) -> predicted_times array,
    plus the fitted model object.

    Requires scikit-learn.
    """
    try:
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.impute import SimpleImputer
    except ImportError:
        raise ImportError("scikit-learn is required for build_predictor()")

    available = [c for c in FEATURE_COLS if c in df.columns]
    X = df[available].copy()
    y = df[TARGET_COL].copy()

    pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("scl", StandardScaler()),
        ("mdl", GradientBoostingRegressor(n_estimators=200, max_depth=4, random_state=42)),
    ])
    pipe.fit(X, y)

    def predict(features_df: pd.DataFrame) -> np.ndarray:
        cols = [c for c in available if c in features_df.columns]
        return pipe.predict(features_df[cols])

    return predict, pipe, available


def recommend_config(
    pts: pd.DataFrame,
    crs: str,
    radius: float,
    spacings: list,
    nest_depths: list,
    predictor,
) -> pd.DataFrame:
    """
    Given a predictor from build_predictor(), estimate the best (spacing, nest_depth)
    for a new (pts, radius) scenario WITHOUT running the full algorithm.

    Note: micro_region_stats are geometric and depend only on radius/spacing —
    they are interpolated from the training data stored in the predictor pipeline.
    For a quick estimate we reuse exact micro_region_stats from the training set
    by matching on the closest radius_over_spacing ratio.

    Returns a DataFrame ranked by predicted CPU time (best first).
    """
    pts_xy = pts[["lon", "lat"]].values if "lat" in pts.columns else pts.iloc[:, :2].values
    dist_stats = compute_spatial_stats(target_points=pts_xy, search_radii=[radius])

    rows = []
    for s in spacings:
        for nd in nest_depths:
            row = {
                "spacing": s,
                "nest_depth": nd,
                "radius_over_spacing": radius / s,
                "scen_spatial_width": dist_stats.get("spatial_width"),
                "scen_spatial_height": dist_stats.get("spatial_height"),
                "scen_spatial_area": dist_stats.get("spatial_area"),
                "scen_spatial_aspect_ratio": dist_stats.get("spatial_aspect_ratio"),
                "scen_density_skewness_max_to_mean": dist_stats.get("density_skewness_max_to_mean"),
                "scen_total_target_points": dist_stats.get("total_target_points"),
            }
            rows.append(row)

    feat_df = pd.DataFrame(rows)
    feat_df["predicted_cpu_s"] = predictor(feat_df)
    return feat_df.sort_values("predicted_cpu_s").reset_index(drop=True)

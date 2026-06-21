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

import os
import json
import pandas as pd
import aabpl.config as _cfg
from typing import Optional, Dict, Any
import numpy as np

from aabpl.main import radius_search, Grid
from aabpl.radius_search.spacing_topology import compute_spatial_stats, compute_spacing_breakpoints, choose_spacing_and_depth
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
    exclude_self=True,
    use_int_cell_keys: Optional[bool] = None,
    vectorized_search_loop: Optional[bool] = None,
    batch_overlap: Optional[bool] = None,
    batch_overlap_min_group: Optional[int] = None,
    generation: Optional[str] = None,
    geo_stats_folder: Optional[str] = None,
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
    geo_stats_folder : if given, persist geometric stats (cell counts/areas relative to
                  circle area) for this (spacing_ratio, nest_depth) pair to a JSON file
                  named geo_nd{nd}_sr{sr:.8f}.json.  Written only on first occurrence.
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
    # Enable per-function profiling only for this measured run (it is off by
    # default in production — see config.PROFILE_FUNC_TIMES). Restored in finally.
    _profile_was = _cfg.PROFILE_FUNC_TIMES
    _cfg.PROFILE_FUNC_TIMES = True
    # The search path is now fixed (int keys + vectorized + always-on overlap batch),
    # so the old per-flag toggles are no-ops; the params are kept for backward
    # compatibility and recorded as constants for result metadata.
    eff_flags = {
        "use_int_cell_keys": True,
        "vectorized_search_loop": True,
        "batch_overlap": True,
        "batch_overlap_min_group": 1,
    }
    if generation is None:
        generation = getattr(_cfg, "PERF_GENERATION", None)
    reset_perf_times()
    _cache_keys_before = set(_aabpl_config.disk_region_cache.keys())
    t_wall_start = _time.perf_counter()
    try:
        grid_result = radius_search(
            pts=pts_source,
            crs=crs,
            r=radius,
            c=col,
            exclude_self=exclude_self,
            silent=silent,
            proj_crs=local_crs,
            pts_target=pts_target,
            stat='sum',
            suffix="_sum",
        )
    finally:
        _cfg.FIXED_SPACING_RATIO = None
        _cfg.FIXED_NEST_DEPTH = None
    t_wall = _time.perf_counter() - t_wall_start
    geometry_cached = set(_aabpl_config.disk_region_cache.keys()) == _cache_keys_before
    if geo_stats_folder is not None and not os.path.isfile(
        _geo_stats_path(spacing_ratio, nest_depth, geo_stats_folder)
    ):
        compute_and_save_geo_stats(spacing_ratio, nest_depth, geo_stats_folder)
    av_pts_per_circle = next((pts_source[col].mean() for col in pts_source.columns[::-1] if col.endswith("_sum")),-1)
    # print("av_pts_per_circle",av_pts_per_circle)
    # -- collect timing ------------------------------------------------------
    perf = analyze_func_perf(plot=False)
    _cfg.PROFILE_FUNC_TIMES = _profile_was
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
            "generation": generation,
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
            "av_pts_per_circle":av_pts_per_circle,
            "geometry_cached": geometry_cached,
            **eff_flags,
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
# Single-run executor
# ---------------------------------------------------------------------------

def run_optimal_config(
    pts_source: pd.DataFrame,
    crs: str,
    radius: float,
    clear:bool=True,
    *,
    pts_target: Optional[pd.DataFrame] = None,
    col: str = "employment",
    local_crs: str = "auto",
    dist_stats: Optional[dict] = None,
    silent: bool = True,
    generation: Optional[str] = None,
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
    import aabpl.config as _cfg
    if clear:
        _cfg.disk_region_cache.clear()
    spacing_ratio, nest_depth = choose_spacing_and_depth(
        r=radius, n_pts_src=len(pts_source), n_pts_tgt=len(pts_target), pts_tgt_xy=pts_target
    )
    scenario_id = _scenario_hash(dist_stats, radius)

    # -- run -----------------------------------------------------------------
    _cfg.FIXED_SPACING_RATIO = spacing_ratio
    _cfg.FIXED_NEST_DEPTH = nest_depth
    # Enable per-function profiling only for this measured run (it is off by
    # default in production — see config.PROFILE_FUNC_TIMES). Restored in finally.
    _profile_was = _cfg.PROFILE_FUNC_TIMES
    _cfg.PROFILE_FUNC_TIMES = True
    # The search path is now fixed (int keys + vectorized + always-on overlap batch),
    # so the old per-flag toggles are no-ops; the params are kept for backward
    # compatibility and recorded as constants for result metadata.
    eff_flags = {
        "use_int_cell_keys": True,
        "vectorized_search_loop": True,
        "batch_overlap": True,
        "batch_overlap_min_group": 1,
    }
    if generation is None:
        generation = getattr(_cfg, "PERF_GENERATION", None)
    reset_perf_times()
    _cache_keys_before = set(_aabpl_config.disk_region_cache.keys())
    t_wall_start = _time.perf_counter()
    try:
        grid_result = radius_search(
            pts=pts_source,
            crs=crs,
            r=radius,
            c=col,
            exclude_self=True,
            silent=silent,
            proj_crs=local_crs,
            pts_target=pts_target,
        )
    finally:
        _cfg.FIXED_SPACING_RATIO = None
        _cfg.FIXED_NEST_DEPTH = None
    t_wall = _time.perf_counter() - t_wall_start
    geometry_cached = set(_aabpl_config.disk_region_cache.keys()) == _cache_keys_before

    # -- collect timing ------------------------------------------------------
    perf = analyze_func_perf(plot=False)
    _cfg.PROFILE_FUNC_TIMES = _profile_was
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
            "generation": generation,
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
            **eff_flags,
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
    # flag tag: index I(nt)/T(uple), loop P(lain)/V(ectorized)/B(atch). Keeps
    # different search-path variants in distinct files so they don't overwrite or
    # falsely skip each other. Absent flags default to the historical scalar path.
    _idx = "I" if cfg.get("use_int_cell_keys") else "T"
    if cfg.get("batch_overlap"):
        _loop = f"B{cfg.get('batch_overlap_min_group', 0)}"
    elif cfg.get("vectorized_search_loop"):
        _loop = "V"
    else:
        _loop = "P"
    flag_tag = f"_x{_idx}{_loop}"
    gen = meta.get("generation")
    gen_tag = f"_g{gen}" if gen else ""
    stem = (
        f"run_{meta['scenario_id']}"
        f"_r{cfg['radius']}"
        f"_s{s_str}"
        f"_nd{cfg['nest_depth']}"
        f"_ns{cfg['n_source']}"
        f"_nt{cfg['n_target']}"
        f"{flag_tag}{gen_tag}"
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


def load_results(output_folder: str, only_new: bool = True,
                 generation: Optional[str] = None) -> pd.DataFrame:
    """
    Load all JSON run files from output_folder into a flat DataFrame.
    Each row is one (radius, spacing, nest_depth) run.

    Results are cached in _cache.parquet inside output_folder. On subsequent
    calls only new JSON files (not yet in the cache) are read, making repeated
    loads fast regardless of how many files accumulate.

    Generation filtering
    --------------------
    Each result carries a ``generation`` tag (meta.generation). To avoid mixing
    results produced before a code change with the current ones:
      - ``generation=<label>``  → return only that generation's rows.
      - ``only_new=True`` (default) and no explicit generation → return only the
        most recent generation present (max label). Rows without a generation tag
        (legacy files) are treated as one group, so old folders behave as before.
      - ``only_new=False`` → return every row across all generations.
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

    # read only JSON files not yet in the cache. Reading is network-latency bound
    # (files live on a remote share), so do it with a thread pool — the GIL is
    # released during file I/O, letting many round-trips overlap (≈10-30x faster
    # than the serial loop for hundreds of small files).
    def _parse_one(fname):
        try:
            with open(os.path.join(output_folder, fname)) as f:
                d = json.load(f)
        except Exception:
            return None
        row = {"_source_file": fname}
        row.update(d.get("meta", {}))
        for k, v in d.get("machine", {}).items():
            row[f"machine_{k}"] = v
        row.update(d.get("config", {}))
        for k, v in d.get("scenario_stats", {}).items():
            row[f"scen_{k}"] = (v[0] if v else None) if isinstance(v, list) else v
        row.update(d.get("micro_region_stats", {}))
        row.update(d.get("timing", {}))
        return row

    new_files = [f for f in os.listdir(output_folder)
                 if f.endswith(".json") and f not in cached_files]
    new_rows = []
    if new_files:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=32) as _ex:
            new_rows = [r for r in _ex.map(_parse_one, new_files) if r is not None]

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

    # -- generation filtering -------------------------------------------------
    if "generation" in df.columns:
        if generation is not None:
            df = df[df["generation"] == generation]
        elif only_new:
            _gens = df["generation"].dropna().unique()
            if len(_gens) > 0:
                _latest = max(_gens)
                df = df[df["generation"] == _latest]
        df = df.reset_index(drop=True)
        if df.empty:
            return df

    # Enrich with geometry cache stats (n_regions, mean_cntd_count, mean_ovlpd_count).
    # These depend only on (radius_over_spacing, nest_depth) so one cache lookup fills
    # all rows with the same combination, even if most JSON files predate this feature.
    _enrich_geo_stats(df)
    return df


def _cell_area_sum(cells) -> float:
    """Total normalised area covered by a list of (lvl, rc) quadtree cells.
    A cell at level *lvl* has side 2^-lvl, so area = 4^-lvl (in units where the
    base cell is 1×1).
    """
    import numpy as _np_local
    if isinstance(cells, _np_local.ndarray):
        return float(_np_local.sum(4.0 ** (-cells[:, 0]))) if len(cells) else 0.0
    return sum(4.0 ** (-lvl) for lvl, _ in cells)


def _geo_stats_path(r_over_s: float, nest_depth: int, geo_folder: str) -> str:
    return os.path.join(geo_folder, f"geo_nd{int(nest_depth)}_sr{round(float(r_over_s), 8):.8f}.json")


def compute_and_save_geo_stats(
    r_over_s: float,
    nest_depth: int,
    geo_folder: str,
) -> Optional[dict]:
    """Compute full geometric stats for a (r/s, nest_depth) pair and persist them.

    Reads from the in-memory geometry cache (must already be populated by
    build_disk_region_lookups).  Skips silently if the cache entry is absent.
    Returns the stats dict (or None on cache miss).

    Saved stats (all area-weighted by region size unless prefixed ``mean_``):
        n_regions, n_shared_cntd, total_area, circle_area, area_ratio
        mean_count_cntd / mean_count_ovlpd   — simple mean over regions (count)
        mean_area_cntd  / mean_area_ovlpd    — simple mean over regions (cell area)
        w_count_cntd    / w_count_ovlpd      — area-weighted mean count
        w_area_cntd     / w_area_ovlpd       — area-weighted mean cell area
    """
    import math
    cache_key = (round(float(r_over_s), 8), int(nest_depth), False)
    cached = _aabpl_config.disk_region_cache.get(cache_key)
    if cached is None:
        return None

    region_id_to_cntd  = cached["region_id_to_cntd_cells"]
    region_id_to_ovlpd = cached["region_id_to_ovlpd_cells"]
    region_id_to_area  = cached["region_id_to_area"]
    shared_cntd        = cached.get("shared_cntd_cells", set())

    total_area = sum(region_id_to_area.values()) or 1.0
    n_regions  = len(region_id_to_cntd)

    sum_count_cntd = sum_count_ovlpd = 0.0
    sum_area_cntd  = sum_area_ovlpd  = 0.0
    w_count_cntd   = w_count_ovlpd   = 0.0
    w_area_cntd    = w_area_ovlpd    = 0.0

    for rid, cntd_cells in region_id_to_cntd.items():
        reg_area = region_id_to_area.get(rid, 0.0)
        ovlpd_cells = region_id_to_ovlpd.get(rid, [])
        cnt_c = len(cntd_cells);            cnt_o = len(ovlpd_cells)
        a_c   = _cell_area_sum(cntd_cells); a_o   = _cell_area_sum(ovlpd_cells)

        sum_count_cntd += cnt_c;   sum_count_ovlpd += cnt_o
        sum_area_cntd  += a_c;     sum_area_ovlpd  += a_o
        w_count_cntd   += cnt_c * reg_area;  w_count_ovlpd += cnt_o * reg_area
        w_area_cntd    += a_c   * reg_area;  w_area_ovlpd  += a_o   * reg_area

    n = n_regions or 1
    circle_area = math.pi * r_over_s ** 2
    mean_ac = sum_area_cntd  / n
    mean_ao = sum_area_ovlpd / n
    w_ac    = w_area_cntd    / total_area
    w_ao    = w_area_ovlpd   / total_area
    stats = {
        "r_over_s":                  float(r_over_s),
        "nest_depth":                int(nest_depth),
        "n_regions":                 n_regions,
        "n_shared_cntd":             len(shared_cntd),
        "total_area":                total_area,
        "circle_area":               circle_area,
        "area_ratio":                total_area / circle_area,
        # simple means (one value per region, equally weighted)
        "mean_count_cntd":           sum_count_cntd  / n,
        "mean_count_ovlpd":          sum_count_ovlpd / n,
        "mean_area_cntd":            mean_ac,
        "mean_area_ovlpd":           mean_ao,
        # area-weighted means (each region weighted by its spatial area)
        "w_count_cntd":              w_count_cntd  / total_area,
        "w_count_ovlpd":             w_count_ovlpd / total_area,
        "w_area_cntd":               w_ac,
        "w_area_ovlpd":              w_ao,
        # cell area as share of circle area (= micro_area_share_* from calc_micro_region_stats)
        "area_share_cntd":           mean_ac / circle_area if circle_area > 0 else 0.0,
        "area_share_ovlpd":          mean_ao / circle_area if circle_area > 0 else 0.0,
        "area_share_cntd_weighted":  w_ac    / circle_area if circle_area > 0 else 0.0,
        "area_share_ovlpd_weighted": w_ao    / circle_area if circle_area > 0 else 0.0,
    }

    os.makedirs(geo_folder, exist_ok=True)
    with open(_geo_stats_path(r_over_s, nest_depth, geo_folder), "w") as f:
        json.dump(stats, f, indent=2, default=float)
    return stats


def load_geo_stats(r_over_s: float, nest_depth: int, geo_folder: str) -> Optional[dict]:
    """Load persisted geo stats for one (r/s, nest_depth) pair, or None if missing."""
    path = _geo_stats_path(r_over_s, nest_depth, geo_folder)
    if not os.path.isfile(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def _enrich_geo_stats(df: pd.DataFrame, geo_folder: Optional[str] = None) -> None:
    """Enrich df in-place with geometric stats for each (radius_over_spacing, nest_depth).

    Stats are sourced in order of preference:
      1. Saved JSON files in *geo_folder* (survives across sessions, no cache needed).
      2. In-memory geometry cache (populated if geometry was built this session).

    Columns added:
        n_regions, n_shared_cntd, area_ratio
        mean_count_cntd, mean_count_ovlpd   (simple mean over regions)
        mean_area_cntd,  mean_area_ovlpd    (simple mean over regions)
        w_count_cntd,    w_count_ovlpd      (area-weighted mean count)
        w_area_cntd,     w_area_ovlpd       (area-weighted mean cell area)
    """
    import math

    GEO_COLS = [
        "n_regions", "n_shared_cntd", "area_ratio",
        "mean_count_cntd", "mean_count_ovlpd",
        "mean_area_cntd",  "mean_area_ovlpd",
        "w_count_cntd",    "w_count_ovlpd",
        "w_area_cntd",     "w_area_ovlpd",
        "area_share_cntd", "area_share_ovlpd",
        "area_share_cntd_weighted", "area_share_ovlpd_weighted",
    ]

    def _from_cache(r_over_s, nest_depth):
        key = (round(float(r_over_s), 8), int(nest_depth), False)
        cached = _aabpl_config.disk_region_cache.get(key)
        if cached is None:
            return None
        # compute on the fly (same logic as compute_and_save_geo_stats)
        cntd  = cached["region_id_to_cntd_cells"]
        ovlpd = cached["region_id_to_ovlpd_cells"]
        areas = cached["region_id_to_area"]
        shared = cached.get("shared_cntd_cells", set())
        total_area = sum(areas.values()) or 1.0
        n = len(cntd) or 1
        sc, so, ac, ao, wcc, wco, wac, wao = 0., 0., 0., 0., 0., 0., 0., 0.
        for rid, cc in cntd.items():
            oc = ovlpd.get(rid, [])
            ra = areas.get(rid, 0.)
            c, o = len(cc), len(oc)
            a, b = _cell_area_sum(cc), _cell_area_sum(oc)
            sc += c; so += o; ac += a; ao += b
            wcc += c*ra; wco += o*ra; wac += a*ra; wao += b*ra
        ca = math.pi * r_over_s ** 2
        mac = ac/n; mao = ao/n; wac_ = wac/total_area; wao_ = wao/total_area
        return {
            "n_regions": n, "n_shared_cntd": len(shared),
            "area_ratio": total_area / ca,
            "mean_count_cntd": sc/n, "mean_count_ovlpd": so/n,
            "mean_area_cntd":  mac,  "mean_area_ovlpd":  mao,
            "w_count_cntd": wcc/total_area, "w_count_ovlpd": wco/total_area,
            "w_area_cntd":  wac_,           "w_area_ovlpd":  wao_,
            "area_share_cntd":           mac  / ca if ca > 0 else 0.0,
            "area_share_ovlpd":          mao  / ca if ca > 0 else 0.0,
            "area_share_cntd_weighted":  wac_ / ca if ca > 0 else 0.0,
            "area_share_ovlpd_weighted": wao_ / ca if ca > 0 else 0.0,
        }

    combos = df[["radius_over_spacing", "nest_depth"]].drop_duplicates()
    stats_map = {}
    for row in combos.itertuples(index=False):
        ros, nd = row.radius_over_spacing, row.nest_depth
        s = None
        if geo_folder:
            s = load_geo_stats(ros, nd, geo_folder)
        if s is None:
            s = _from_cache(ros, nd)
        stats_map[(ros, nd)] = s

    for col in GEO_COLS:
        df[col] = df.apply(
            lambda r, c=col: (stats_map.get((r["radius_over_spacing"], r["nest_depth"])) or {}).get(c),
            axis=1,
        )

    # back-compat aliases kept for existing code that reads these names
    if "mean_cntd_count" not in df.columns:
        df["mean_cntd_count"]  = df["w_count_cntd"]
    if "mean_ovlpd_count" not in df.columns:
        df["mean_ovlpd_count"] = df["w_count_ovlpd"]


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
            if verbose:
                from aabpl.utils.progress import progress_print as _pp
                _pp(f"  load  {label}{tag}")
                _pp(f"         pattern: {skip_pattern}")
                _pp(f"         matched: {match}")
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
            from aabpl.utils.progress import progress_print as _pp
            _pp(f"  run   {label} n={len(pts_source)}{tag} ...", end=" ", flush=True)
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
                from aabpl.utils.progress import progress_print as _pp
                cached_str = "cached" if result["config"]["geometry_cached"] else "UNCACHED"
                _pp(f"done  cpu={result['timing']['total_cpu_s']:.2f}s  geom={cached_str}")
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


def run_direct_test(
    pts_source: pd.DataFrame,
    pts_target: pd.DataFrame,
    crs: str,
    radii: list,
    spacing_ratios: list,
    nest_depths: list,
    cfg_updates: Dict[str, Any],
    *,
    col: str = "employment",
    local_crs: str = "auto",
    output_folder: str = "./perf_test",
    silent: bool = True,
    skip_existing: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Runs a direct grid search over (radius, spacing, nest_depth) using fixed source 
    and target datasets, applying custom configuration overwrites.
    """
    # 1. Dynamically update the configuration parameters
    for key, value in cfg_updates.items():
        if hasattr(_cfg, key):
            setattr(_cfg, key, value)
        else:
            if verbose:
                print(f"Warning: aabpl.config has no attribute '{key}'. Setting anyway.")
            setattr(_cfg, key, value)

    # 2. Project target points to local CRS for spatial statistics calculation
    from aabpl.utils.crs_transformation import convert_pts_to_crs as _convert_pts_to_crs
    _pts_proj = pts_target.copy()
    _x_proj, _y_proj, _local_crs = _convert_pts_to_crs(
        pts=_pts_proj,
        x="lon" if "lon" in pts_target.columns else pts_target.columns[0],
        y="lat" if "lat" in pts_target.columns else pts_target.columns[1],
        initial_crs=crs,
        target_crs=local_crs,
        silent=True,
    )
    pts_xy = _pts_proj[[_x_proj, _y_proj]].values
    dist_stats_cache = {r: compute_spatial_stats(target_points=pts_xy, search_radii=[r]) for r in radii}

    # 3. Handle old-format output migration and build completion index
    completed = set()
    if skip_existing and os.path.isdir(output_folder):
        for fname in os.listdir(output_folder):
            if not fname.endswith(".json") or "_n" in fname:
                continue
            fpath = os.path.join(output_folder, fname)
            try:
                with open(fpath) as _f:
                    _d = json.load(_f)
                _n = _d.get("config", {}).get("n_source")
                if _n is not None:
                    stem = fname[:-5]
                    new_fname = stem.replace("_m", f"_n{_n}_m", 1) + ".json"
                    os.rename(fpath, os.path.join(output_folder, new_fname))
            except Exception:
                pass
        completed = {f for f in os.listdir(output_folder) if f.endswith(".json")}

    results = []
    n_tgt = len(pts_target)

    # 4. Iterate directly through the configurations
    for sr in spacing_ratios:
        for nd in nest_depths:
            # Clear cache between groups to safely reuse geometry footprints
            if hasattr(_cfg, 'disk_region_cache') and _cfg.disk_region_cache is not None:
                _cfg.disk_region_cache.clear()
            
            for r in radii:
                dist_stats = dist_stats_cache[r]
                scenario_id = _scenario_hash(dist_stats, r)
                
                # Format string to match run_sweep's internal schema exactly
                skip_pattern = f"run_{scenario_id}_r{r}_s{round(sr, 2)}_nd{nd}_ns{len(pts_source)}_nt{n_tgt}_"
                label = f"r={r} sr={round(sr, 2)} nd={nd}"
                
                # Check for cached runs
                if skip_existing and any(f.startswith(skip_pattern) for f in completed):
                    match = next(f for f in completed if f.startswith(skip_pattern))
                    if verbose:
                        print(f"  load  {label} [direct-test]")
                    try:
                        with open(os.path.join(output_folder, match)) as _f:
                            result = json.load(_f)
                        results.append(result)
                        continue
                    except Exception:
                        completed.discard(match)

                if verbose:
                    print(f"  run   {label} n={len(pts_source)} [direct-test] ... ", end="", flush=True)
                
                # Execute single run configuration
                try:
                    result = run_single_config(
                        pts_source=pts_source, crs=crs, radius=r, spacing_ratio=sr,
                        nest_depth=nd, col=col, pts_target=pts_target,
                        local_crs=local_crs, dist_stats=dist_stats, silent=silent,
                    )
                    fpath, deleted = save_run_result(result, output_folder)
                    for d in deleted:
                        completed.discard(d)
                    completed.add(os.path.basename(fpath))
                    results.append(result)
                    
                    if verbose:
                        cached_str = "cached" if result["config"]["geometry_cached"] else "UNCACHED"
                        print(f"done  cpu={result['timing']['total_cpu_s']:.2f}s  geom={cached_str}")
                        
                except Exception as e:
                    import traceback as _tb
                    _cfg.FIXED_SPACING_RATIO = None
                    _cfg.FIXED_NEST_DEPTH = None
                    if verbose:
                        print(f"ERROR: {e}")
                        _tb.print_exc()

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Testing generation: blank-start adaptive sweep
# ---------------------------------------------------------------------------

# Search-path variants to compare. Each is one (index, loop) combination passed
# straight to run_single_config. Edit/extend as needed in the notebook.
DEFAULT_FLAG_COMBOS = [
    {"name": "int_batch",  "use_int_cell_keys": True,  "vectorized_search_loop": True,  "batch_overlap": True},
    {"name": "int_plain",  "use_int_cell_keys": True,  "vectorized_search_loop": False, "batch_overlap": False},
    {"name": "tuple_plain","use_int_cell_keys": False, "vectorized_search_loop": False, "batch_overlap": False},
]


def _flag_key(combo: dict) -> tuple:
    return (bool(combo.get("use_int_cell_keys")),
            bool(combo.get("vectorized_search_loop")),
            bool(combo.get("batch_overlap")),
            int(combo.get("batch_overlap_min_group", 0)) if combo.get("batch_overlap") else 0)


def _row_flag_key(row) -> tuple:
    return (bool(row.get("use_int_cell_keys")),
            bool(row.get("vectorized_search_loop")),
            bool(row.get("batch_overlap")),
            int(row.get("batch_overlap_min_group", 0)) if row.get("batch_overlap") else 0)


def run_generation(
    pts: pd.DataFrame,
    crs: str,
    radii: list,
    spacing_ratios: list,
    nest_depths: list,
    *,
    generation: Optional[str] = None,
    flag_combos: Optional[list] = None,
    screen_size: int = 3000,
    sample_sizes: Optional[list] = None,
    col: str = "employment",
    local_crs: str = "auto",
    output_folder: str = "./perf_test_gen",
    prior_folder: Optional[str] = None,
    seed_top_k: int = 3,
    max_rounds: int = 6,
    time_budget_s: Optional[float] = None,
    validate_top_k: int = 2,
    silent: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Blank-start, self-extending benchmark "generation".

    Explores the (radius x spacing_ratio x nest_depth x flag_combo) space without
    running it exhaustively: it seeds from prior results (``prior_folder``) when
    available — otherwise a coarse grid — then *recursively* expands around the
    best-so-far cell for each (radius, flag_combo), measuring grid neighbours until
    no unmeasured neighbour of the optimum remains (a local-search / coordinate
    expansion over the spacing x nest_depth grid). Finally the best ``validate_top_k``
    configs per (radius, flag_combo) are re-run at each full ``sample_sizes``.

    All runs are stamped with ``generation`` and saved immediately, so the run is
    restartable (rerunning skips finished combos) and ``load_results(folder)`` —
    which defaults to ``only_new=True`` — returns just this generation.

    Parameters
    ----------
    generation     : label for this batch (default: timestamp ``gen<YYYYMMDDHHMMSS>``).
    flag_combos    : list of search-path variants (see DEFAULT_FLAG_COMBOS).
    screen_size    : source sample size used during exploration.
    sample_sizes   : full sizes for the validation pass (default [len(pts)]).
    prior_folder   : folder of older results used only to pick good seed cells.
    time_budget_s  : stop starting new runs after this many seconds (None = no cap).
    """
    import time as _t
    from aabpl.utils.crs_transformation import convert_pts_to_crs as _convert_pts_to_crs

    if generation is None:
        generation = "gen" + _time.strftime("%Y%m%d%H%M%S")
    flag_combos = flag_combos or DEFAULT_FLAG_COMBOS
    sample_sizes = sample_sizes or [len(pts)]
    sr_grid = sorted(spacing_ratios)
    nd_grid = sorted(nest_depths)
    os.makedirs(output_folder, exist_ok=True)
    _cfg.PERF_GENERATION = generation
    t_start = _t.perf_counter()

    if verbose:
        print(f"=== generation {generation} ===")
        print(f"  grid: {len(radii)} radii x {len(sr_grid)} spacing_ratios x "
              f"{len(nd_grid)} nest_depths x {len(flag_combos)} flag_combos")
        print(f"  output: {os.path.abspath(output_folder)}")

    # -- project once for spatial stats --------------------------------------
    _pp = pts.copy()
    _xp, _yp, _lcrs = _convert_pts_to_crs(
        pts=_pp, x="lon" if "lon" in pts.columns else pts.columns[0],
        y="lat" if "lat" in pts.columns else pts.columns[1],
        initial_crs=crs, target_crs=local_crs, silent=True)
    pts_xy = _pp[[_xp, _yp]].values
    dist_stats_cache = {r: compute_spatial_stats(target_points=pts_xy, search_radii=[r]) for r in radii}

    # -- resume: which (r, sr, nd, n, flag) are already measured this gen -----
    measured: dict = {}   # (r, fkey) -> {(si, ni): cpu}  at screen_size
    done: set = set()     # (round(r), round(sr,4), nd, n, fkey)
    existing = load_results(output_folder, generation=generation)
    if not existing.empty:
        for _, row in existing.iterrows():
            fk = _row_flag_key(row)
            key = (round(float(row["radius"]), 4), round(float(row["spacing_ratio"]), 4),
                   int(row["nest_depth"]), int(row["n_source"]), fk)
            done.add(key)
            if int(row["n_source"]) == screen_size:
                try:
                    si = sr_grid.index(min(sr_grid, key=lambda s: abs(s - row["spacing_ratio"])))
                    ni = nd_grid.index(int(row["nest_depth"]))
                    measured.setdefault((round(float(row["radius"]), 4), fk), {})[(si, ni)] = float(row["total_cpu_s"])
                except ValueError:
                    pass

    # -- seed cells (indices into sr_grid x nd_grid) per (r, flag) -----------
    def _coarse_seed():
        sis = sorted({0, len(sr_grid)//2, len(sr_grid)-1})
        nis = sorted({0, len(nd_grid)//2, len(nd_grid)-1})
        return [(si, ni) for si in sis for ni in nis]

    prior_df = None
    if prior_folder and os.path.isdir(prior_folder):
        prior_df = load_results(prior_folder, only_new=False)

    def _prior_seed(r, fk):
        if prior_df is None or prior_df.empty or "total_cpu_s" not in prior_df.columns:
            return []
        sub = prior_df[(prior_df["radius"].round(4) == round(r, 4))]
        if sub.empty:
            return []
        cells = []
        for _, row in sub.sort_values("total_cpu_s").head(seed_top_k).iterrows():
            try:
                si = sr_grid.index(min(sr_grid, key=lambda s: abs(s - row["spacing_ratio"])))
                ni = nd_grid.index(min(nd_grid, key=lambda n: abs(n - int(row["nest_depth"]))))
                cells.append((si, ni))
            except ValueError:
                pass
        return cells

    sample_screen = pts.sample(min(screen_size, len(pts)), random_state=0)
    n_tgt = len(pts)
    results: list = []

    def _do(si, ni, r, fk_combo, n_source, pts_source, tag):
        sr = sr_grid[si]; nd = nd_grid[ni]; fk = _flag_key(fk_combo)
        key = (round(float(r), 4), round(float(sr), 4), int(nd), int(n_source), fk)
        if key in done:
            return measured.get((round(float(r), 4), fk), {}).get((si, ni))
        if time_budget_s is not None and (_t.perf_counter() - t_start) > time_budget_s:
            return None
        if verbose:
            print(f"  [{tag}] r={r} sr={round(sr,3)} nd={nd} n={n_source} {fk_combo['name']} ...",
                  end=" ", flush=True)
        try:
            res = run_single_config(
                pts_source=pts_source, crs=crs, radius=r, spacing_ratio=sr, nest_depth=nd,
                col=col, pts_target=pts, local_crs=local_crs,
                dist_stats=dist_stats_cache[r], silent=silent, generation=generation,
                batch_overlap_min_group=fk_combo.get("batch_overlap_min_group"),
                **{k: fk_combo[k] for k in
                   ("use_int_cell_keys", "vectorized_search_loop", "batch_overlap") if k in fk_combo},
            )
            save_run_result(res, output_folder)
            results.append(res)
            done.add(key)
            cpu = res["timing"]["total_cpu_s"]
            if n_source == screen_size:
                measured.setdefault((round(float(r), 4), fk), {})[(si, ni)] = cpu
            if verbose:
                print(f"{cpu:.2f}s")
            return cpu
        except Exception as e:
            import traceback as _tb
            _cfg.FIXED_SPACING_RATIO = None; _cfg.FIXED_NEST_DEPTH = None
            if verbose:
                print(f"ERROR: {e}")
                _tb.print_exc()
            return None

    # -- exploration: seed then expand around best per (r, flag) -------------
    for r in radii:
        for combo in flag_combos:
            fk = _flag_key(combo)
            queue = list(dict.fromkeys(_prior_seed(r, fk) + _coarse_seed()))
            rounds = 0
            while queue and rounds < max_rounds:
                rounds += 1
                for (si, ni) in queue:
                    if 0 <= si < len(sr_grid) and 0 <= ni < len(nd_grid):
                        _do(si, ni, r, combo, screen_size, sample_screen, "screen")
                if time_budget_s is not None and (_t.perf_counter() - t_start) > time_budget_s:
                    break
                cells = measured.get((round(float(r), 4), fk), {})
                if not cells:
                    break
                (bsi, bni) = min(cells, key=cells.get)
                neighbours = [(bsi-1, bni), (bsi+1, bni), (bsi, bni-1), (bsi, bni+1)]
                queue = [(s, n) for (s, n) in neighbours
                         if 0 <= s < len(sr_grid) and 0 <= n < len(nd_grid) and (s, n) not in cells]

    # -- validation: best configs per (r, flag) at full sample sizes ---------
    for r in radii:
        for combo in flag_combos:
            fk = _flag_key(combo)
            cells = measured.get((round(float(r), 4), fk), {})
            for (si, ni) in sorted(cells, key=cells.get)[:validate_top_k]:
                for n_source in sample_sizes:
                    full = pts.sample(min(n_source, len(pts)), random_state=0)
                    _do(si, ni, r, combo, len(full), full, "validate")

    if verbose:
        print(f"\ngeneration {generation} complete: {len(results)} new runs "
              f"in {(_t.perf_counter()-t_start)/60:.1f} min")
    return load_results(output_folder, generation=generation)


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

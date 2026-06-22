import math
from numpy import (
    array       as _np_array,
    zeros       as _np_zeros,
    min         as _np_min,
    max         as _np_max,
    add         as _np_add,
    clip        as _np_clip,
    atleast_1d  as _np_atleast_1d,
    percentile  as _np_percentile,
    random      as _np_random,
)


def compute_spatial_stats(
    target_points,
    search_radii,
    source_points=None,
    grid_resolution=8,
    random_seed=42,
    max_pts=50_000,
):
    """
    Compute descriptive spatial statistics for a point cloud.

    Radius-dependent metrics are returned as lists aligned with search_radii.

    Parameters
    ----------
    target_points   : (N, 2) NumPy array
    search_radii    : list/array of search radii
    source_points   : unused, kept for backward compatibility
    grid_resolution : cells per side for legacy 8×8 skewness (default 8)
    max_pts         : subsample target_points to this size; keeps runtime
                      under ~0.5 s for any N
    """
    if random_seed is not None:
        _np_random.seed(random_seed)

    N_full = len(target_points)
    if N_full > max_pts:
        idx = _np_random.choice(N_full, size=max_pts, replace=False)
        target_points = target_points[idx]
    N = len(target_points)
    radii_list = [float(r) for r in _np_atleast_1d(search_radii)]
    stats = {}

    # --- 1. Bounding box & aspect ratio ---
    p_min = _np_min(target_points, axis=0)
    p_max = _np_max(target_points, axis=0)
    W = float(p_max[0] - p_min[0])
    H = float(p_max[1] - p_min[1])
    max_world_dim = max(W, H)
    area = W * H

    stats['spatial_width']        = W
    stats['spatial_height']       = H
    stats['spatial_area']         = area
    stats['spatial_aspect_ratio'] = W / H if H > 0 else 1.0

    denom_x = W if W > 0 else 1.0
    denom_y = H if H > 0 else 1.0
    global_mean_density = N / area if area > 0 else 0.0

    # --- 2. Legacy fixed-grid skewness (8×8, scalar) ---
    xi8 = _np_clip(((target_points[:, 0] - p_min[0]) / denom_x * grid_resolution).astype(int), 0, grid_resolution - 1)
    yi8 = _np_clip(((target_points[:, 1] - p_min[1]) / denom_y * grid_resolution).astype(int), 0, grid_resolution - 1)
    gc8 = _np_zeros((grid_resolution, grid_resolution))
    _np_add.at(gc8, (xi8, yi8), 1)
    cell_area_8 = area / (grid_resolution ** 2) if grid_resolution > 0 else 1.0
    max_dens_8  = float(_np_max(gc8)) / cell_area_8 if cell_area_8 > 0 else 0.0
    stats['density_skewness_max_to_mean']         = max_dens_8 / global_mean_density if global_mean_density > 0 else 1.0
    stats['density_max_target_points_in_test_cell']= float(_np_max(gc8))

    # --- 3. Per-radius metrics ---
    r_to_world_list      = []
    boundary_overhead_list = []
    skewness_2r_list     = []

    cell_w_fixed = W / grid_resolution if grid_resolution > 0 else 1.0

    for r in radii_list:
        r_to_world_list.append((2.0 * r) / max_world_dim if max_world_dim > 0 else 0.0)
        boundary_overhead_list.append((2.0 * r) / cell_w_fixed if cell_w_fixed > 0 else 0.0)

        # Radius-aware skewness: cell size = 2r (search diameter).
        # Uses 90th percentile of non-empty cells / global mean — focuses on
        # the dense tail while ignoring empty cells.
        nx = max(int(W / (2.0 * r)), 1)
        ny = max(int(H / (2.0 * r)), 1)
        xi = _np_clip(((target_points[:, 0] - p_min[0]) / denom_x * nx).astype(int), 0, nx - 1)
        yi = _np_clip(((target_points[:, 1] - p_min[1]) / denom_y * ny).astype(int), 0, ny - 1)
        gc = _np_zeros((nx, ny))
        _np_add.at(gc, (xi, yi), 1)
        cell_area_2r = (W / nx) * (H / ny) if nx > 0 and ny > 0 else 1.0
        nonempty = gc.ravel()[gc.ravel() > 0]
        if len(nonempty) > 0 and global_mean_density > 0 and cell_area_2r > 0:
            p90_density = float(_np_percentile(nonempty, 90)) / cell_area_2r
            skewness_2r_list.append(p90_density / global_mean_density)
        else:
            skewness_2r_list.append(1.0)

    stats['ratio_radius_to_world']        = r_to_world_list
    stats['boundary_overhead_factor_radius'] = boundary_overhead_list
    stats['density_skewness_2r']          = skewness_2r_list
    stats['total_target_points']          = N_full

    return stats


def count_cells_per_level(x, y, xmin, ymin, spacing, max_nd):
    """
    Count non-empty grid cells at each nesting level 0..max_nd.

    Uses a smart upward-propagation strategy: compute cell indices at the
    deepest level once from all pts, then derive shallower levels via integer
    ``//2`` on the unique pairs — much faster than re-scanning all pts per level.

    Parameters
    ----------
    x, y    : 1-D array-like of projected coordinates (same units as spacing)
    xmin, ymin : grid origin (lower-left corner)
    spacing : base cell size (level-0 cell side length)
    max_nd  : maximum nesting depth (inclusive)

    Returns
    -------
    list of int, length max_nd+1 — n_unique cells at levels 0, 1, …, max_nd
    """
    import numpy as _np
    import pandas as _pd

    x = _np.asarray(x, dtype=_np.float64)
    y = _np.asarray(y, dtype=_np.float64)
    if len(x) == 0:
        return [0] * (max_nd + 1)

    # compute cell indices at deepest level
    sub_deep = spacing / (2 ** max_nd)
    rows = ((y - ymin) / sub_deep).astype(_np.int64)
    cols = ((x - xmin) / sub_deep).astype(_np.int64)

    # unique pairs at deepest level via combined key
    grid_w = int(_np.ceil((x.max() - xmin) / sub_deep)) + 2
    combined = rows * grid_w + cols
    uniq = _pd.unique(combined)
    u_rows = (uniq // grid_w).astype(_np.int64)
    u_cols = (uniq  % grid_w).astype(_np.int64)
    del rows, cols, combined, uniq

    counts = [0] * (max_nd + 1)
    counts[max_nd] = len(u_rows)

    # propagate upward: each level has cells 2× bigger → indices halve
    for k in range(max_nd - 1, -1, -1):
        u_rows >>= 1
        u_cols >>= 1
        grid_w_k = int(_np.ceil((x.max() - xmin) / (spacing / (2 ** k)))) + 2
        combined = u_rows * grid_w_k + u_cols
        counts[k] = len(_pd.unique(combined))

    return counts


def estimate_template_cell_counts(nest_depth, spacing_ratio):
    """
    Estimate cell counts and area shares for the disk search template at a
    given (nest_depth, spacing_ratio).

    All coefficients from empirical log-linear fits to disk_region_geometry
    output across a grid of (sr, nd) values:

      stat               R²      formula
      n_ovlpd          0.990   exp(2.0641 + 0.9530·log(sr) + 0.7776·nd)  [∝ sr·2^nd]
      n_cntd           0.915   exp(0.1319 + 2.2716·log(sr) + 0.8451·nd)  [∝ sr²·4^nd approx]
      area_share_ovlpd 0.979   exp(0.9632 - 1.1021·log(sr) - 0.3187·nd)  [∝ 1/(sr·2^nd)]
      area_share_cntd  0.903   exp(-1.3045 + 0.7389·log(sr) + 0.1862·nd)
      area_ratio       1.000   exp(-1.1447 - 2.0000·log(sr))              [∝ 1/sr², nd-independent]

    area_share_ovlpd and area_share_cntd are fractions of the total disk area
    covered by overlapped / contained cells respectively.
    area_ratio is total cell area / circle area (captures how much the cell
    approximation over-covers the circle).

    Parameters
    ----------
    nest_depth    : int   — nesting depth (0 = no sub-cells)
    spacing_ratio : float — r / spacing (sr)

    Returns
    -------
    dict with keys: n_cntd, n_ovlpd, area_share_cntd, area_share_ovlpd, area_ratio
    """
    import math as _math
    log_sr = _math.log(float(spacing_ratio))
    nd     = int(nest_depth)
    return {
        'n_cntd':           _math.exp(0.1319 + 2.2716 * log_sr + 0.8451 * nd),
        'n_ovlpd':          _math.exp(2.0641 + 0.9530 * log_sr + 0.7776 * nd),
        'area_share_cntd':  _math.exp(-1.3045 + 0.7389 * log_sr + 0.1862 * nd),
        'area_share_ovlpd': _math.exp( 0.9632 - 1.1021 * log_sr - 0.3187 * nd),
        'area_ratio':       _math.exp(-1.1447 - 2.0000 * log_sr),
    }


def predict_time_excl_build(nest_depth, spacing_ratio, n_pts_src, n_pts_tgt,
                            skew=1.0, ppc=None, ngc=None):
    """
    Predict search + aggregation time (excl. geometry build) in seconds.

    Empirical OLS model (R²=0.857, 21 features, no ros dummies).

    Coefficients  (R²=0.857, 21 features, no ros dummies)
    ------------
    Intercept                              -2.52740
    log(r/s)³                              -0.08831
    nd³                                    +0.01070
    log(r/s)×nd                            +0.17594
    log(n_tgt)²                            +0.02267
    log(n_src)²                            +0.02075
    log(ppc)                               -0.19208
    log(ppc)²×log(r/s)²                   -0.01038
    log(ppc)²×nd²                          +0.00027
    log(ppc)³×nd                           -0.00059
    log(ppc)²×log(n_src)                   -0.00010
    log(skew)                              -0.32633
    log(skew)×log(r/s)                     +0.09258
    log(skew)×nd                           +0.01695
    log(skew)×nd²                          -0.03483
    log(skew)×log(ppc)²                   +0.01421
    log(n_tgt/ngc)×log(r/s)               -0.01743  [skipped if ngc not given]
    log(n_src/ngc)×nd                      -0.01555  [skipped if ngc not given]
    log(r/s)²×nd²                          -0.02791
    log(ppc)×log(r/s)²×nd                  +0.01092
    log(area_share_ovlpd)×log(ppc)        -0.05970

    Parameters
    ----------
    nest_depth    : int
    spacing_ratio : float — r / spacing (sr)
    n_pts_src     : int   — number of source points
    n_pts_tgt     : int   — number of target points
    skew          : float — spatial skewness (default 1.0)
    ppc           : float — points per circle area = n_tgt·π·r²/(W·H);
                            if None falls back to n_tgt^0.5
    ngc           : float — total grid cells = (W/s)·(H/s)·4^nd;
                            if None the two ngc interaction terms are skipped

    Returns
    -------
    float — predicted time in seconds
    """
    import math as _math

    nd    = int(nest_depth)
    sr    = float(spacing_ratio)
    n_src = float(n_pts_src)
    n_tgt = float(n_pts_tgt)
    sk    = max(float(skew), 1e-6)
    if ppc is None:
        ppc = max(n_tgt ** 0.5, 1.0)
    ppc = max(float(ppc), 1e-6)

    log_sr  = _math.log(sr)
    log_src = _math.log(max(n_src, 1))
    log_tgt = _math.log(max(n_tgt, 1))
    log_ppc = _math.log(ppc)
    log_sk  = _math.log(sk)
    log_sr2 = log_sr ** 2
    log_ppc2 = log_ppc ** 2
    nd2     = nd ** 2

    _est           = estimate_template_cell_counts(nd, sr)
    log_area_ovlpd = _math.log(max(_est['area_share_ovlpd'], 1e-12))

    log_time = (
        -2.52740
        - 0.08831 * log_sr ** 3
        + 0.01070 * nd ** 3
        + 0.17594 * log_sr   * nd
        + 0.02267 * log_tgt ** 2
        + 0.02075 * log_src ** 2
        - 0.19208 * log_ppc
        - 0.01038 * log_ppc2 * log_sr2
        + 0.00027 * log_ppc2 * nd2
        - 0.00059 * log_ppc ** 3 * nd
        - 0.00010 * log_ppc2 * log_src
        - 0.32633 * log_sk
        + 0.09258 * log_sk   * log_sr
        + 0.01695 * log_sk   * nd
        - 0.03483 * log_sk   * nd2
        + 0.01421 * log_sk   * log_ppc2
        - 0.02791 * log_sr2  * nd2
        + 0.01092 * log_ppc  * log_sr2 * nd
        - 0.05970 * log_area_ovlpd * log_ppc
    )
    if ngc is not None:
        log_ngc   = _math.log(max(float(ngc), 1.0))
        log_time += (
            - 0.01743 * (log_tgt - log_ngc) * log_sr
            - 0.01555 * (log_src - log_ngc) * nd
        )
    return _math.exp(log_time)


def predict_geo_build_time(spacing_ratio: float, nest_depth: int) -> float:
    """Predict geometry build time (seconds, CPU) for a given spacing_ratio and nest_depth.

    Fitted on process_time() sweeps over sr ∈ [1.0, 6.0], nd ∈ [0, 9].
    R² ≈ 0.80 (single-run measurements; true R² on repeated mins would be higher).
    Captures:
      - cubic polynomial in log(sr) and nd with cross-terms
      - sr_is_int dummy (sr is an exact integer) and its nd interactions
    Valid range: sr > 0, nd ≥ 0.  Returns seconds.
    """
    import math as _math
    sr  = float(spacing_ratio)
    nd  = float(nest_depth)
    lsr = _math.log(sr)
    si  = 1.0 if sr % 1.0 == 0.0 else 0.0  # sr_is_int
    log_t = (
        14.0419
        - 16.4806 * sr
        + 25.3188 * lsr
        +  0.0526 * nd
        -  3.6885 * lsr**2
        -  0.0477 * nd**2
        +  0.0461 * lsr * nd
        +  9.1441 * lsr**3
        +  0.0112 * nd**3
        -  0.0355 * lsr**2 * nd
        -  0.0262 * lsr * nd**2
        +  0.0112 * lsr**2 * nd**2
        +  0.0782 * si
        +  0.0472 * si * nd
        -  0.0140 * si * nd**2
    )
    return _math.exp(log_t)


def recommend_max_nest_depth(
    n_pts_src,
    n_pts_tgt,
    cell_counts,
    spacing_ratio,
    skew=1.0,
    build_cost_per_cell=1e-7,
):
    """
    Return the nest_depth that minimises estimated total runtime.

    Cost model
    ----------
    total_cost(k) = predict_time_excl_build(k, ...) + build_cost(k)

    where build_cost(k) = build_cost_per_cell × Σ cell_counts[0..k]
    (aggregate_point_data_to_cells writes one node per non-empty cell per level).

    The search + aggregation time is predicted from the empirical regression
    model (R²=0.859); see predict_time_excl_build() for coefficients.
    Build time is kept separate as it scales linearly with total non-empty cells
    across all levels and is cached in disk_region_geometry after the first call.

    Parameters
    ----------
    n_pts_src          : number of source (search-origin) points
    n_pts_tgt          : number of target (aggregated) points
    cell_counts        : list of int, length nd+1 — non-empty cells per level
                         (level 0 = coarsest) from count_cells_per_level()
    spacing_ratio      : r / spacing (sr)
    skew               : spatial skewness of target pts (default 1.0 = uniform)
    build_cost_per_cell: seconds per non-empty cell for aggregation node build
                         (default 1e-7 s / cell)

    Returns
    -------
    int — recommended max nest_depth in 0 .. len(cell_counts)-1
    """
    nd_max = len(cell_counts) - 1
    if nd_max == 0 or n_pts_tgt <= 0:
        return 0

    ppc = n_pts_tgt / max(cell_counts[0], 1)

    best_k, best_cost = 0, float('inf')
    build_cumulative  = 0.0

    for k in range(nd_max + 1):
        build_cumulative += cell_counts[k] * build_cost_per_cell
        search_agg_cost  = predict_time_excl_build(
            k, spacing_ratio, n_pts_src, n_pts_tgt, skew=skew, ppc=ppc,
        )
        total_cost = search_agg_cost + build_cumulative

        if total_cost < best_cost:
            best_cost = total_cost
            best_k    = k

    return best_k


def compute_spacing_breakpoints(max_offset: int = 4, silent: bool = True):
    """
    Generate the unique 2D analytical breakpoints (R/spacing ratios) where the
    required grid cell search pattern changes — i.e. where a new ring of cells
    enters or leaves the search disk.
    """
    breakpoints = {}

    for i in range(max_offset + 1):
        for j in range(max_offset + 1):
            r2_center_corner = (i + 0.5) ** 2 + (j + 0.5) ** 2
            r2_center_line   = (i + 0.5) ** 2
            r2_corner_line   = float(i ** 2) if i > 0 else None
            r2_corner_corner = float(i ** 2 + j ** 2) if (i > 0 or j > 0) else None

            candidates = [r2_center_corner, r2_center_line]
            if r2_corner_line   is not None: candidates.append(r2_corner_line)
            if r2_corner_corner is not None: candidates.append(r2_corner_corner)

            for r2 in candidates:
                key = round(r2, 4)
                if key > 0:
                    if key not in breakpoints:
                        breakpoints[key] = {'ratio': math.sqrt(r2), 'squared': r2, 'offsets': []}
                    if (i, j) not in breakpoints[key]['offsets']:
                        breakpoints[key]['offsets'].append((i, j))

    sorted_keys = sorted(breakpoints.keys())

    if not silent:
        print("Generating Complete 2D Topological Break Points:\n")
        print(f"{'Source Offsets':<20} | {'Exact (R/w)²':<15} | {'Break Point (R/w)':<20}")
        print("-" * 62)

    unique_ratios = []
    for key in sorted_keys:
        bp = breakpoints[key]
        if not silent:
            offset_str = ", ".join(str(o) for o in bp['offsets'][:2])
            if len(bp['offsets']) > 2:
                offset_str += "..."
            print(f"{offset_str:<20} | {bp['squared']:<15.2f} | {bp['ratio']:<20.4f}")
        unique_ratios.append(bp['ratio'])

    return _np_array(unique_ratios)


# ---------------------------------------------------------------------------
# Candidate spacing ratios and nest-depth heuristic
# ---------------------------------------------------------------------------

# Candidate values for the dimensionless ratio r / search_spacing (NOT the
# user-facing output spacing). The chosen search spacing is r divided by one of
# these; larger ratios mean finer (slower) search grids. These pick the internal
# *search* grid only — the output grid resolution is set separately by the
# `spacing` parameter (default r/3) and is not user-visible here.
SPACINGS_BREAKPOINTS = _np_array([
    2**.5, 1.5, 2.5**.5, 2, 4.5**0.5, 5**0.5, 2.5, (13/2)**0.5, (17/2)**0.5, 3
])
CANDIDATE_DEPTHS = range(4)  # nd 0-3; expand once full sweep (incl. nd=4+) is complete

def choose_nest_depth(r_over_s: float) -> int:
    """Heuristic nest_depth for a given r/spacing ratio."""
    sqrt2 = math.sqrt(2)
    nd_min = 0 if r_over_s >= sqrt2 else max(0, math.ceil(math.log2(sqrt2 / r_over_s)))
    if r_over_s >= 3.0:
        return nd_min
    elif r_over_s >= 1.5:
        return nd_min + 1
    else:
        return min(nd_min + 2, 4)


# ---------------------------------------------------------------------------
# Timing model — LASSO→OLS benchmark regression (2026-06, v3)
# geo: 55 rows (R²=0.732)  total_excl_build: (R²=0.859, search+agg merged)
# ---------------------------------------------------------------------------

def predict_timing(
    r_over_s: float,
    nd: int,
    n_src: float,
    n_tgt: float,
    r: float,
    spatial_width: float,
    spatial_height: float,
    skewness: float,
    geometry_cached: bool,
) -> tuple:
    """Return (geometry_s, search_s, aggregate_s) predicted CPU seconds.

    r          : actual search radius in the same CRS units as spatial_width/height
    spatial_*  : bounding-box dimensions from compute_spatial_stats (same CRS units as r)
    """
    s   = r / max(r_over_s, 1e-9)
    W, H = spatial_width, spatial_height
    # All dimensionless — require consistent units between r and W/H
    ngc  = (W / s) * (H / s) * (4 ** nd)
    ppc  = n_tgt * math.pi * r ** 2 / max(W * H, 1e-30)

    log_ros  = math.log(max(r_over_s,  1e-9))
    log_ngc  = math.log(max(ngc,       1.0))
    log_nts  = math.log(max(n_src,     1.0))
    log_ntt  = math.log(max(n_tgt,     1.0))
    log_skw  = math.log(max(skewness,  1e-9))
    log_ppc  = math.log(max(ppc,       1e-9))
    log_ros2 = log_ros ** 2
    log_ros3 = log_ros ** 3
    log_nts2 = log_nts ** 2
    log_ntt2 = log_ntt ** 2
    nd2      = nd ** 2
    nd3      = nd ** 3
    tol = 1e-4
    ros_2_5000 = int(abs(r_over_s - 2.5) < tol)
    ros_3_0000 = int(abs(r_over_s - 3.0) < tol)

    # ── Geometry (absolute seconds, uncached only, topology features only) ──
    # geo: 55 rows (R²=0.732)
    if geometry_cached:
        geo_s = 0.0
    else:
        lv = (2.6429
              + 0.077  * log_ros2
              + 0.724  * log_ros3
              - 0.064  * nd
              + 0.012  * nd3
              - 0.018  * log_ngc
              + 0.018  * log_ros2 * nd2
              - 0.006  * log_ngc  * nd)
        geo_s = math.exp(min(lv, 50.0))
    # ── Search + Aggregation combined (absolute seconds) — total excl. geo build
    srch_s = predict_time_excl_build(nd, r_over_s, n_src, n_tgt,
                                     skew=skewness, ppc=ppc, ngc=ngc)
    agg_s  = 0.0

    return geo_s, srch_s, agg_s


def choose_spacing_and_depth(
    r: float,
    spacing_ratio: int = None,
    nest_depth: int = None,
    n_pts_src: int = None,
    n_pts_tgt: int = None,
    n_pts_src_extra: int = 0,
    pts_tgt_xy=None,
    silent: bool = True,
) -> tuple:
    """
    Choose the best (spacing, nest_depth) pair from *candidate_spacings* using
    the hard-coded timing model (geometry + search + aggregation).

    ``n_pts_src_extra`` adds to the effective source count used for the timing
    estimate. ``detect_cluster_pts`` passes the number of random null-distribution
    points here, since those are searched over the same grid in addition to the
    real source points. Defaults to 0, so a plain ``radius_search`` is unaffected.
    """
    from aabpl import config as _cfg
    from aabpl.utils import progress as _prog
    candidate_spacings = SPACINGS_BREAKPOINTS if spacing_ratio is None else [spacing_ratio]
    candidate_depths = CANDIDATE_DEPTHS if nest_depth is None else [nest_depth]
    n_src = float((n_pts_src or 10_000) + (n_pts_src_extra or 0))
    n_tgt = float(n_pts_tgt or (n_pts_src or 10_000))
    skewness       = 10.0
    spatial_width  = r * 100.0   # fallback: pretend world is 100× radius
    spatial_height = r * 100.0
    if pts_tgt_xy is not None and n_tgt >= 1000:
        _stats = compute_spatial_stats(target_points=pts_tgt_xy, search_radii=[r])
        _skew_2r = _stats.get('density_skewness_2r')
        skewness       = max((_skew_2r[0] if isinstance(_skew_2r, list) and _skew_2r else None)
                             or _stats.get('density_skewness_max_to_mean', 1.0), 1e-9)
        spatial_width  = max(_stats.get('spatial_width',  1e-9), 1e-9)
        spatial_height = max(_stats.get('spatial_height', 1e-9), 1e-9)

    import math as _math
    _ppc_debug = n_tgt * _math.pi * r**2 / max(spatial_width * spatial_height, 1e-30)
    if _cfg.DEV_MODE:
        print(f"  [spacing dbg] r={r}  n_tgt={n_tgt:.0f}  W={spatial_width:.1f}  H={spatial_height:.1f}"
            f"  ppc={_ppc_debug:.3f}  skew={skewness:.2f}"
            f"  {'(fallback)' if pts_tgt_xy is None or n_tgt <= 1000 else '(from pts)'}")

    cache = _cfg.disk_region_cache

    # s iterates over dimensionless r/spacing ratios; actual spacing = r/s
    def _is_cached(s, nd):
        return (round(s, 8), nd, False) in cache or (round(s, 8), nd, True) in cache


    best_total        = math.inf
    best_pair         = (candidate_spacings[0], candidate_depths[0])
    best_cached_total = math.inf
    best_cached_pair  = None
    for s in candidate_spacings:
        for nd in candidate_depths:
            cached = _is_cached(s, nd)
            geo_s, srch_s, agg_s = predict_timing(
                s, nd, n_src, n_tgt,
                r, spatial_width, spatial_height,
                skewness, geometry_cached=cached,
            )
            total = geo_s*(_cfg.GEO_AMORTIZATION_WEIGHT-max(0,min(_cfg.GEO_AMORTIZATION_WEIGHT-0.01,nd*0.05))) + srch_s + agg_s # slightly decrease the weight of geo_s as it might pay off when the user calls multiple times.
            if total < best_total:
                best_total = total
                best_pair  = (s, nd)
            if cached and total < best_cached_total:
                best_cached_total = total
                best_cached_pair  = (s, nd)

    chosen_s, chosen_nd = best_pair
    if best_cached_pair is not None and best_cached_total <= best_total:
        chosen_s, chosen_nd = best_cached_pair

    geo_s, srch_s, agg_s = predict_timing(
        chosen_s, chosen_nd, n_src, n_tgt,
        r, spatial_width, spatial_height,
        skewness, geometry_cached=_is_cached(chosen_s, chosen_nd),
    )
    _prog._BUILD_EST_SECONDS  = geo_s
    _prog._SEARCH_EST_SECONDS = srch_s + agg_s

    if _cfg.DEV_MODE:
        print(f"  chosen spacing_ratio={chosen_s:.3f}  nest_depth={chosen_nd}"
              f"  pred={geo_s+srch_s+agg_s:.2f}s"
              f"  (geo={geo_s:.2f}s  srch={srch_s:.2f}s  agg={agg_s:.2f}s)")

    return r / chosen_s, chosen_nd

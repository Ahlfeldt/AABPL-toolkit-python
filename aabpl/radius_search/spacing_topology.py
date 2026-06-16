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

CANDIDATE_SPACINGS_BREAKPOINTS = _np_array([
    2**.5, 1.5, 2.5**.5, 2, 4.5**0.5, 5**0.5, 2.5, (13/2)**0.5, (17/2)**0.5,
])
CANDIDATE_DEPTHS = range(7)

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
# Timing model — LASSO→OLS benchmark regression (2026-06)
# geo: 55 rows (R²=0.734)  srch: 621 rows (R²=0.941)  agg: 927 rows (R²=0.965)
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
    ngc       = (W / s) * (H / s) * (4 ** nd)
    ppc       = n_tgt * math.pi * r ** 2 / max(W * H, 1e-30)
    wrld_crcl = W * H / (math.pi * r ** 2)

    log_ros  = math.log(max(r_over_s,  1e-9))
    log_ngc  = math.log(max(ngc,       1.0))
    log_nts  = math.log(max(n_src,     1.0))
    log_ntt  = math.log(max(n_tgt,     1.0))
    log_skw  = math.log(max(skewness,  1e-9))
    log_ppc  = math.log(max(ppc,       1e-9))
    log_wc   = math.log(max(wrld_crcl, 1e-9))
    log_4nd  = math.log1p(4.0 ** nd)
    log_ros2 = log_ros ** 2
    log_nts2 = log_nts ** 2
    log_ntt2 = log_ntt ** 2
    nd2      = nd ** 2

    # ── Geometry (absolute seconds, uncached only, topology features only) ──
    if geometry_cached:
        geo_s = 0.0
    else:
        lv = (3.8803
              - 1.802  * log_ros
              + 2.193  * log_ros2
              + 1.048  * nd
              + 0.114  * nd2
              - 1.214  * log_4nd
              - 0.019  * log_ngc
              - 0.080  * log_ros2 * nd
              + 0.061  * log_ros  * nd2
              - 0.004  * log_ngc  * nd)
        geo_s = math.exp(lv)

    # ── Search (absolute seconds) ────────────────────────────────────────────
    lv = (2.7203
          + 1.663  * log_ros2
          + 0.390  * nd
          + 0.027  * nd2
          + 0.309  * log_4nd
          + 0.342  * log_ngc
          - 0.026  * log_ros  * nd2
          - 2.485  * log_ntt
          + 0.124  * log_ntt2
          + 0.613  * log_nts
          - 0.006  * log_ntt2 * nd
          + 0.117  * log_ppc  * log_ros
          - 0.116  * log_skw  * log_ros
          + 0.229  * log_wc   * log_ros)
    srch_s = math.exp(lv)

    # ── Aggregation (absolute seconds) ──────────────────────────────────────
    lv = (-8.3060
          + 0.555  * log_ros2
          + 0.681  * log_ros2 * nd
          + 0.005  * log_ntt2
          + 0.033  * log_nts2
          + 0.005  * log_ntt2 * nd
          + 0.001  * log_ntt2 * nd2
          + 0.938  * log_ppc
          - 0.344  * log_ppc  * log_ros
          - 0.158  * log_ppc  * nd
          + 0.872  * log_skw
          - 1.038  * log_skw  * log_ros
          - 0.091  * log_skw  * nd
          - 1.047  * log_wc   * log_ros
          + 0.724  * log_wc   * log_ros2
          - 0.019  * log_wc   * nd
          + 0.020  * log_wc   * nd2)
    agg_s = math.exp(lv)

    return (0.0 if geometry_cached else max(geo_s, 1.0)), max(srch_s, 1.0), max(agg_s, 1.0)


def choose_spacing_and_depth(
    r: float,
    candidate_spacings: list,
    spacing_ratio: int = None,
    nest_depth: int = None,
    n_pts_src: int = None,
    n_pts_tgt: int = None,
    pts_tgt_xy=None,
    silent: bool = True,
) -> tuple:
    """
    Choose the best (spacing, nest_depth) pair from *candidate_spacings* using
    the hard-coded timing model (geometry + search + aggregation).
    """
    from aabpl import config as _cfg
    from aabpl.utils import progress as _prog
    candidate_spacings = CANDIDATE_SPACINGS_BREAKPOINTS if spacing_ratio is None else [spacing_ratio]
    candidate_depths = CANDIDATE_DEPTHS if nest_depth is None else [nest_depth]
    n_src = float(n_pts_src or 10_000)
    n_tgt = float(n_pts_tgt or n_src)
    skewness       = 10.0
    spatial_width  = r * 100.0   # fallback: pretend world is 100× radius
    spatial_height = r * 100.0
    if pts_tgt_xy is not None and n_tgt > 1000:
        _stats = compute_spatial_stats(target_points=pts_tgt_xy, search_radii=[r])
        _skew_2r = _stats.get('density_skewness_2r')
        skewness       = max((_skew_2r[0] if isinstance(_skew_2r, list) and _skew_2r else None)
                             or _stats.get('density_skewness_max_to_mean', 1.0), 1e-9)
        spatial_width  = max(_stats.get('spatial_width',  1e-9), 1e-9)
        spatial_height = max(_stats.get('spatial_height', 1e-9), 1e-9)

    cache = _cfg.disk_region_cache

    def _is_cached(s, nd):
        rs = round(r / s, 8)
        return (rs, nd, False) in cache or (rs, nd, True) in cache

    pairs = [(s, nd) for s in candidate_spacings for nd in candidate_depths]

    best_total        = math.inf
    best_pair         = pairs[0]
    best_cached_total = math.inf
    best_cached_pair  = None

    for s, nd in pairs:
        cached = _is_cached(s, nd)
        geo_s, srch_s, agg_s = predict_timing(
            r / s, nd, n_src, n_tgt,
            r, spatial_width, spatial_height,
            skewness, geometry_cached=cached,
        )
        total = geo_s + srch_s + agg_s
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
        r / chosen_s, chosen_nd, n_src, n_tgt,
        r, spatial_width, spatial_height,
        skewness, geometry_cached=_is_cached(chosen_s, chosen_nd),
    )
    _prog._BUILD_EST_SECONDS  = geo_s
    _prog._SEARCH_EST_SECONDS = srch_s + agg_s

    if not silent:
        print(f"  chosen spacing_ratio={r/chosen_s:.3f}  nest_depth={chosen_nd}"
              f"  pred={geo_s+srch_s+agg_s:.2f}s"
              f"  (geo={geo_s:.2f}s  srch={srch_s:.2f}s  agg={agg_s:.2f}s)")

    if _cfg.DEV_MODE:
        print("spacing, nest_depth",chosen_s/r, chosen_nd)
    return chosen_s, chosen_nd

"""
Consolidated test suite for aabpl.

Covers:
  radius_search  — all stats, multi-stat, multi-c, nest_depth, pts_target,
                   keep_cols, weight_valid_area, config.VALIDATE, helpers
  detect_cluster_pts — all r spec types (single / list / bands / wbands),
                       multi-c, stat variants, cluster boolean integrity
  guard checks   — rand_dist / cluster_pts raise clear errors on radius_search grids
  plot smoke     — grid.plot.* run without error (Agg backend)

Run with:  python -m pytest tests/run_all_tests.py -v
"""
import pytest
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _make_pts(n=800, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        'x':    rng.uniform(0, 50_000, n),
        'y':    rng.uniform(0, 50_000, n),
        'val':  rng.uniform(1, 10, n),
        'val2': rng.uniform(0, 5, n),
    })


# Tiny dataset for cluster tests — 30 pts, ~2 km extent, r <= 500 m
_RNG = np.random.default_rng(42)
_CX = _RNG.uniform(580_000, 580_400, 10)
_CY = _RNG.uniform(4_510_000, 4_510_400, 10)
_SX = _RNG.uniform(580_000, 582_000, 20)
_SY = _RNG.uniform(4_510_000, 4_512_000, 20)
_X  = np.concatenate([_CX, _SX])
_Y  = np.concatenate([_CY, _SY])
_EMP = np.concatenate([_RNG.integers(50, 200, 10).astype(float),
                       _RNG.integers(1,  10,  20).astype(float)])
_POP = _RNG.integers(100, 500, 30).astype(float)
_CLUSTER_CRS = 'EPSG:32618'


def _make_cluster_pts():
    return pd.DataFrame({'x': _X, 'y': _Y, 'emp': _EMP, 'pop': _POP})


_SEARCH_DEV = {'nest_depth': 2, 'spacing_over_radius': 2.0}
_R = 5_000  # radius for radius_search tests (Cartesian, no CRS)
_CR = 500   # radius for cluster tests (metres, EPSG:32618)


def _rs(pts, stat='sum', c=None, **kw):
    """Thin wrapper around radius_search with test defaults."""
    import aabpl
    if c is None:
        c = ['val']
    return aabpl.radius_search(
        pts=pts, crs='', r=_R, c=c, x='x', y='y',
        stat=stat, silent=True, _dev=_SEARCH_DEV, **kw,
    )


def _detect(r, c='emp', stat='sum', keep_cols=False, **kw):
    """Thin wrapper around detect_cluster_pts with test defaults."""
    import aabpl
    pts = _make_cluster_pts()
    grid = aabpl.detect_cluster_pts(
        pts=pts, crs=_CLUSTER_CRS, r=r, c=c, stat=stat,
        x='x', y='y',
        null_distribution=500,
        k_th_percentile=95.0,
        random_seed=0,
        keep_cols=keep_cols,
        silent=True,
        **kw,
    )
    return grid, pts


# ===========================================================================
# A. radius_search — stat coverage
# ===========================================================================

@pytest.mark.parametrize('stat,suf', [
    ('sum',   f'_sum_{_R}'),
    ('count', f'_cnt_{_R}'),
    ('mean',  f'_avg_{_R}'),
])
def test_rs_basic_stat(stat, suf):
    pts = _make_pts()
    _rs(pts, stat=stat, suffix=suf)
    out = [c for c in pts.columns if c.endswith(suf)]
    assert len(out) == 1, f'stat={stat}: expected 1 col ending {suf!r}, got {list(pts.columns)}'
    assert (pts[out[0]] >= 0).all()


@pytest.mark.parametrize('stat', ['variance', 'std', 'cv', 'skewness', 'kurtosis'])
def test_rs_moment_stat(stat):
    pts = _make_pts()
    _rs(pts, stat=stat)
    out = [c for c in pts.columns if c.startswith('val_')]
    assert len(out) == 1, f'stat={stat}: expected 1 output col, got {out}'


# ===========================================================================
# B. radius_search — multi-stat lists
# ===========================================================================

def test_rs_multi_stat_sum_count_mean():
    pts = _make_pts()
    _rs(pts, stat=['sum', 'count', 'mean'])
    for col in [f'val_sum_{_R}', f'val_cnt_{_R}', f'val_avg_{_R}']:
        assert col in pts.columns, f'missing {col}'
    # mean == sum / count where count > 0
    mask = pts[f'val_cnt_{_R}'] > 0
    diff = (pts.loc[mask, f'val_avg_{_R}']
            - pts.loc[mask, f'val_sum_{_R}'] / pts.loc[mask, f'val_cnt_{_R}']).abs().max()
    assert diff < 1e-9, f'mean != sum/count, diff={diff}'


def test_rs_multi_stat_sum_variance():
    pts = _make_pts()
    _rs(pts, stat=['sum', 'variance'])
    assert f'val_sum_{_R}' in pts.columns
    assert f'val_var_{_R}' in pts.columns
    leaked = [c for c in pts.columns if '__rs_int__' in c]
    assert not leaked, f'internal suffix leaked: {leaked}'


def test_rs_multi_stat_two_cols():
    pts = _make_pts()
    _rs(pts, stat=['sum', 'mean'], c=['val', 'val2'])
    for base in ['val', 'val2']:
        for suf in [f'_sum_{_R}', f'_avg_{_R}']:
            assert base + suf in pts.columns


def test_rs_multi_stat_custom_suffix():
    pts = _make_pts()
    _rs(pts, stat=['sum', 'count'], suffix={'sum': '_s5k', 'count': '_n5k'})
    assert 'val_s5k' in pts.columns
    assert 'val_n5k' in pts.columns


# ===========================================================================
# C. radius_search — nest_depth variants
# ===========================================================================

@pytest.mark.parametrize('nd', [0, 2, 3])
def test_rs_nest_depth(nd):
    import aabpl
    pts = _make_pts()
    aabpl.radius_search(pts=pts, crs='', r=_R, c=['val'], x='x', y='y',
                        stat='sum', silent=True,
                        _dev={'nest_depth': nd, 'spacing_over_radius': 2.0})
    col = [c for c in pts.columns if c.startswith('val_')][0]
    assert pts[col].sum() > 0


def test_rs_nest_depth_consistency():
    import aabpl
    sums = {}
    for nd in [2, 3]:
        pts = _make_pts()
        aabpl.radius_search(pts=pts, crs='', r=_R, c=['val'], x='x', y='y',
                            stat='sum', silent=True,
                            _dev={'nest_depth': nd, 'spacing_over_radius': 2.0})
        col = [c for c in pts.columns if c.startswith('val_')][0]
        sums[nd] = pts[col].sum()
    rel = abs(sums[3] - sums[2]) / max(sums[2], 1)
    assert rel < 0.05, f'nd=2 vs nd=3 diverge {rel:.1%}'


# ===========================================================================
# D. radius_search — pts_target
# ===========================================================================

def test_rs_pts_target():
    import aabpl
    pts_source = _make_pts()
    pts_target = _make_pts(300, seed=42)
    aabpl.radius_search(pts=pts_target, crs='', r=_R, c=['val'], x='x', y='y',
                        pts_target=pts_source, silent=True, _dev=_SEARCH_DEV)
    assert f'val_sum_{_R}' in pts_target.columns
    assert f'val_sum_{_R}' not in pts_source.columns


# ===========================================================================
# E. radius_search — keep_cols / helper cleanup
# ===========================================================================

def test_rs_keep_cols_false():
    pts = _make_pts()
    before = set(pts.columns)
    _rs(pts, stat='sum')
    added = set(pts.columns) - before
    assert added == {f'val_sum_{_R}'}, f'unexpected cols: {added}'


def test_rs_keep_cols_true_no_internals():
    pts = _make_pts()
    before = set(pts.columns)
    _rs(pts, stat='sum', keep_cols=True)
    added = set(pts.columns) - before
    leaked = [c for c in added if '__rs_int__' in c]
    assert not leaked
    assert f'val_sum_{_R}' in added


def test_rs_variance_no_count_sum_leak():
    import aabpl
    pts = _make_pts()
    before = set(pts.columns)
    aabpl.radius_variance(pts=pts, crs='', r=_R, c=['val'], x='x', y='y',
                          silent=True, _dev=_SEARCH_DEV)
    added = set(pts.columns) - before
    leaked = [c for c in added if ('cnt' in c or 'count' in c or 'sum' in c)
              and c != f'val_var_{_R}']
    assert not leaked, f'count/sum leaked: {leaked}'


def test_rs_multi_variance_count_both_present():
    pts = _make_pts()
    _rs(pts, stat=['variance', 'count'])
    assert any('var' in c for c in pts.columns)
    assert any('cnt' in c or 'count' in c for c in pts.columns)


# ===========================================================================
# F. radius_search — weight_valid_area and config.VALIDATE
# ===========================================================================

def test_rs_weight_valid_area():
    pts = _make_pts()
    _rs(pts, stat='sum', weight_valid_area='estimate')
    assert f'val_sum_{_R}' in pts.columns


def test_rs_config_validate():
    import aabpl.config as config
    config.VALIDATE = True
    try:
        pts = _make_pts()
        _rs(pts, stat='sum')
        assert f'val_sum_{_R}' in pts.columns
    finally:
        config.VALIDATE = False



# ===========================================================================
# H. detect_cluster_pts — r spec types
# ===========================================================================

def test_cluster_r_single():
    # Single-r naming: {c}_cluster_{stat}_{r}
    grid, pts = _detect(r=_CR, keep_cols=True)
    assert f'emp_sum_{_CR}' in pts.columns
    assert f'emp_cluster_sum_{_CR}' in pts.columns
    assert pts[f'emp_cluster_sum_{_CR}'].dtype == bool


def test_cluster_r_list():
    grid, pts = _detect(r=[300, _CR], keep_cols=True)
    assert f'emp_sum_300' in pts.columns
    assert f'emp_sum_{_CR}' in pts.columns
    assert f'emp_sum_300_cluster' in pts.columns
    assert f'emp_sum_{_CR}_cluster' in pts.columns


def test_cluster_r_list_cluster_cols_no_keep():
    grid, pts = _detect(r=[300, _CR])
    assert 'emp_sum_300_cluster' in pts.columns
    assert f'emp_sum_{_CR}_cluster' in pts.columns
    assert pts['emp_sum_300_cluster'].dtype == bool


def test_cluster_r_bands():
    grid, pts = _detect(r=[(0, 300), (300, _CR)])
    assert 'emp_sum_0_300_cluster' in pts.columns
    assert f'emp_sum_300_{_CR}_cluster' in pts.columns
    assert pts['emp_sum_0_300_cluster'].dtype == bool
    assert pts[f'emp_sum_300_{_CR}_cluster'].dtype == bool


def test_cluster_r_bands_keep_cols():
    grid, pts = _detect(r=[(0, 300), (300, _CR)], keep_cols=True)
    assert 'emp_sum_0_300' in pts.columns
    assert f'emp_sum_300_{_CR}' in pts.columns


def test_cluster_r_wbands():
    grid, pts = _detect(r=[(0, 300, 1.0), (300, _CR, 0.5)])
    assert 'emp_sum_wgt_cluster' in pts.columns
    assert pts['emp_sum_wgt_cluster'].dtype == bool


def test_cluster_r_wbands_keep_cols():
    grid, pts = _detect(r=[(0, 300, 1.0), (300, _CR, 0.5)], keep_cols=True)
    assert 'emp_sum_wgt' in pts.columns


# ===========================================================================
# I. detect_cluster_pts — multi-c
# ===========================================================================

def test_cluster_multi_c_single_r():
    grid, pts = _detect(r=_CR, c=['emp', 'pop'], keep_cols=True)
    assert f'emp_sum_{_CR}' in pts.columns
    assert f'pop_sum_{_CR}' in pts.columns
    assert f'emp_cluster_sum_{_CR}' in pts.columns
    assert f'pop_cluster_sum_{_CR}' in pts.columns


def test_cluster_multi_c_multi_r():
    grid, pts = _detect(r=[300, _CR], c=['emp', 'pop'], keep_cols=True)
    assert 'emp_sum_300' in pts.columns
    assert f'pop_sum_{_CR}' in pts.columns
    assert 'emp_sum_300_cluster' in pts.columns
    assert f'pop_sum_{_CR}_cluster' in pts.columns


# ===========================================================================
# J. detect_cluster_pts — stat variants
# ===========================================================================

@pytest.mark.parametrize('stat,abbr', [
    ('sum',   'sum'),
    ('count', 'cnt'),
    ('mean',  'avg'),
])
def test_cluster_stat_single_r(stat, abbr):
    grid, pts = _detect(r=_CR, c='emp', stat=stat, keep_cols=True)
    assert f'emp_{abbr}_{_CR}' in pts.columns, \
        f'Expected emp_{abbr}_{_CR}, got: {list(pts.columns)}'


def test_cluster_stat_sum_multi_r():
    grid, pts = _detect(r=[300, _CR], c='emp', stat='sum', keep_cols=True)
    assert 'emp_sum_300' in pts.columns
    assert 'emp_sum_300_cluster' in pts.columns


def test_cluster_stat_mean_multi_r():
    grid, pts = _detect(r=[300, _CR], c='emp', stat='mean', keep_cols=True)
    assert 'emp_avg_300' in pts.columns


def test_cluster_stat_count_multi_r():
    grid, pts = _detect(r=[300, _CR], c='emp', stat='count', keep_cols=True)
    assert 'emp_cnt_300' in pts.columns


# ===========================================================================
# K. detect_cluster_pts — regression: band boolean not overwritten
# ===========================================================================

def test_rand_dist_shows_all_bands():
    """k_th_percentiles must have one entry per aggregate col, not one per value col.
    Previously the zip in create_distribution_plot stopped at len(value_cols), silently
    dropping any extra band panels."""
    grid, _ = _detect(r=[(0, 200), (200, 300), (300, _CR)])
    result = grid._cluster_result
    n_agg = len(result['aggregate_cols'])
    n_k   = len(result['k_th_percentiles'])
    assert n_agg == 3, f'expected 3 aggregate cols for 3 bands, got {result["aggregate_cols"]}'
    assert n_k == n_agg, \
        f'k_th_percentiles has {n_k} entries but aggregate_cols has {n_agg} — ' \
        f'rand_dist would silently drop the last {n_agg - n_k} band panel(s)'


def test_band_cluster_bool_not_overwritten():
    grid, pts = _detect(r=[(0, 300), (300, _CR)])
    for col in ('emp_sum_0_300_cluster', f'emp_sum_300_{_CR}_cluster'):
        assert col in pts.columns
        assert pts[col].dtype == bool, f'{col} dtype={pts[col].dtype}, expected bool'


# ===========================================================================
# L. detect_cluster_pts — custom sample area (Polygon / MultiPolygon)
# ===========================================================================

def test_custom_sample_area_polygon():
    import aabpl
    from shapely.geometry import Polygon, Point
    pts = _make_pts(2000, seed=7)
    poly = Polygon([(5000, 5000), (30000, 5000), (30000, 30000), (5000, 30000)])
    grid = aabpl.detect_cluster_pts(
        pts=pts, crs='', r=_R, c=['val'], x='x', y='y',
        sample_area=poly, null_distribution=1000, random_seed=0, silent=True,
        _dev=_SEARCH_DEV,
    )
    nd = grid.null_distribution
    outside = sum(not poly.covers(Point(x, y))
                  for x, y in zip(nd['x'].values, nd['y'].values))
    assert outside == 0, f'{outside} random points outside polygon'


def test_custom_sample_area_multipolygon():
    import aabpl
    from shapely.geometry import Polygon, MultiPolygon, Point
    pts = _make_pts(2000, seed=7)
    poly_a = Polygon([(5000, 5000), (30000, 5000), (30000, 30000), (5000, 30000)])
    poly_b = Polygon([(32000, 32000), (48000, 32000), (48000, 48000), (32000, 48000)])
    multi = MultiPolygon([poly_a, poly_b])
    grid = aabpl.detect_cluster_pts(
        pts=pts, crs='', r=_R, c=['val'], x='x', y='y',
        sample_area=multi, null_distribution=1000, random_seed=0, silent=True,
        _dev=_SEARCH_DEV,
    )
    nd = grid.null_distribution
    outside = sum(not multi.covers(Point(x, y))
                  for x, y in zip(nd['x'].values, nd['y'].values))
    assert outside == 0, f'{outside} random points outside multipolygon'


# ===========================================================================
# M. guard: rand_dist / cluster_pts raise clear error on radius_search grid
# ===========================================================================

def test_rand_dist_guard_on_radius_search_grid():
    import aabpl
    pts = _make_cluster_pts()
    grid = aabpl.radius_search(pts=pts, crs=_CLUSTER_CRS, r=_CR,
                               c='emp', x='x', y='y', silent=True)
    with pytest.raises(RuntimeError, match='detect_cluster_pts.*detect_cluster_cells'):
        grid.plot.rand_dist()


def test_cluster_pts_guard_on_radius_search_grid():
    import aabpl
    pts = _make_cluster_pts()
    grid = aabpl.radius_search(pts=pts, crs=_CLUSTER_CRS, r=_CR,
                               c='emp', x='x', y='y', silent=True)
    with pytest.raises(RuntimeError, match='detect_cluster_pts.*detect_cluster_cells'):
        grid.plot.cluster_pts()


# ===========================================================================
# N. plot smoke tests (Agg backend, no file I/O)
# ===========================================================================

def test_plot_smoke():
    import aabpl
    import matplotlib.pyplot as plt
    pts = _make_pts()
    grid = aabpl.detect_cluster_cells(
        pts=pts, crs='', r=_R, c=['val'], x='x', y='y',
        stat='sum', null_distribution=500, random_seed=0,
        keep_cols=False, silent=True, _dev=_SEARCH_DEV,
    )
    for method in ('vars', 'clusters', 'cluster_pts', 'rand_dist'):
        try:
            getattr(grid.plot, method)(filename=None)
            plt.close('all')
        except Exception as e:
            raise AssertionError(f'grid.plot.{method}() raised: {e}')

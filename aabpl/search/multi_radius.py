"""
Multi-radius and distance-band aggregation.

This module is an internal implementation detail of ``radius_search``.
Users interact with it only through ``radius_search(r=...)``.

Radius specification conventions
---------------------------------
Single radius:
    r = 750

List of radii (one search per entry):
    r = [250, 500, 750]

Distance bands as (r_inner, r_outer) — small to large:
    r = [(0, 250), (250, 500), (500, 750)]
    band_result = search(r_outer) - search(r_inner)

Weighted aggregation across bands, with (r_inner, r_outer, weight):
    r = [(0, 250, 1), (250, 500, 1), (500, 750, 2)]
    Weights are normalised internally so they sum to 1.
    Output column: ``{col}_{stat_abbr}_wgt`` (or custom suffix).
    Intermediate band columns are dropped from pts unless keep_cols=True.

r=0 handling
------------
radius_search cannot handle r=0 (no lookup table is built).  A dedicated
group-by-coordinate implementation is used: each point's neighbourhood is
itself plus any other points that share the exact same coordinates.

Warning
-------
When more than 5 distance bands are requested in a single call, a one-time
per-session warning is printed because each band triggers a full radius search.
"""
from __future__ import annotations
import numpy as _np
import pandas as _pd

# One-time session warning flag for large band counts.
_WARNED_BAND_COUNT: bool = False

# Maps full stat names to their short identifiers used in column names.
# Short forms are also accepted directly as stat= values (e.g. stat='cnt').
_AGG_ABBR = {
    'sum':      'sum',
    'count':    'cnt',
    'mean':     'avg',
    'variance': 'var',
    'std':      'std',
    'cv':       'cv',
    'skewness': 'skw',
    'kurtosis': 'krt',
}

# Marker embedded in temporary intermediate column names so they can be
# found and removed reliably without accidentally matching user columns.
_TEMP_COL_MARKER = '__mr__'


def _fmt_r(r: float) -> str:
    """Format a radius value for embedding in column names.
    Always uses fixed-point notation (never scientific) and is exact for integer-valued
    floats.  Non-integers use fixed decimal with trailing zeros stripped."""
    if r == int(r):
        return str(int(r))
    return f'{r:f}'.rstrip('0').rstrip('.')


def _parse_r_spec(r):
    """
    Parse the r parameter into a canonical (spec_type, data) pair.

    spec_type   data
    ----------  -------------------------------------------------------
    'single'    float
    'list'      list[float]
    'bands'     list[(r_inner, r_outer)]
    'wbands'    list[(r_inner, r_outer, weight)]
    """
    if not isinstance(r, (list, tuple)):
        return ('single', float(r))
    if len(r) == 0:
        raise ValueError("r must not be empty.")
    first = r[0]
    if isinstance(first, (int, float)):
        radii = [float(v) for v in r]
        return ('single', radii[0]) if len(radii) == 1 else ('list', radii)
    if not isinstance(first, (list, tuple)):
        raise ValueError(f"Cannot parse r specification: {r!r}")
    n_elements = len(first)
    if n_elements == 2:
        for i, band in enumerate(r):
            if float(band[0]) > float(band[1]):
                raise ValueError(
                    f"Band {i}: r_inner={band[0]} > r_outer={band[1]}. "
                    "Use (r_inner, r_outer) ordering."
                )
        return ('bands', [(float(r_in), float(r_out)) for r_in, r_out in r])
    if n_elements == 3:
        for i, band in enumerate(r):
            if float(band[0]) > float(band[1]):
                raise ValueError(
                    f"Band {i}: r_inner={band[0]} > r_outer={band[1]}. "
                    "Use (r_inner, r_outer, weight) ordering."
                )
        return ('wbands', [(float(r_in), float(r_out), float(w)) for r_in, r_out, w in r])
    raise ValueError(f"Band tuples must have 2 or 3 elements, got {n_elements}.")


def _agg_at_r0(pts: _pd.DataFrame, cols: list, x: str, y: str,
               stat: str, exclude_self: bool) -> dict:
    """
    Aggregate within r=0: the point itself plus any exact coordinate duplicates.

    Returns dict {col: pd.Series} with the same index as pts.
    For stats that are not additive (variance, std, cv, skewness, kurtosis),
    returns zeros — the caller can subtract zero safely.
    """
    result = {}
    for col in cols:
        point_values = pts[col]
        if stat == 'count':
            group_count = pts.groupby([x, y])[col].transform('count')
            result[col] = (group_count - 1).astype(float) if exclude_self else group_count.astype(float)
        elif stat == 'sum':
            group_sum = pts.groupby([x, y])[col].transform('sum')
            result[col] = (group_sum - point_values) if exclude_self else group_sum
        elif stat == 'mean':
            group_sum = pts.groupby([x, y])[col].transform('sum')
            group_count = pts.groupby([x, y])[col].transform('count')
            if exclude_self:
                numerator = group_sum - point_values
                denominator = group_count - 1
                result[col] = _pd.Series(
                    _np.where(denominator > 0, numerator / denominator, 0.0),
                    index=pts.index,
                )
            else:
                result[col] = group_sum / group_count
        else:
            result[col] = _pd.Series(_np.zeros(len(pts), dtype=float), index=pts.index)
    return result


def _combine_bands_on_df(
    dataframe,
    spec_type,
    spec_data,
    value_cols,
    stat_str,
    radius_col_fn,
    wgt_col_suffix=None,
    keep_band_cols=False,
    k_th_percentiles=None,
):
    """
    Apply band or weighted-band combination to a DataFrame in-place.

    Called for both the real data DataFrame (pts) and the null-distribution
    DataFrame so that band algebra lives in exactly one place.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Modified in-place.  Must already contain per-radius columns whose names
        are returned by radius_col_fn.
    spec_type : str
        'bands' or 'wbands'.
    spec_data : list
        Band specs — (r_inner, r_outer) for 'bands' or
        (r_inner, r_outer, weight) for 'wbands'.
    value_cols : list[str]
        Original value column names (e.g. ['employment']).
    stat_str : str
        Short stat identifier embedded in output column names (e.g. 'sum').
    radius_col_fn : callable(col: str, radius: float) -> str
        Returns the per-radius aggregate column name for a given value column
        and radius.  The caller maps to the correct naming convention:
        temporary columns (with _TEMP_COL_MARKER) when building pts, or
        permanent '{col}_{stat_str}_{radius}' columns when building null_df.
    wgt_col_suffix : str | None
        For wbands: the suffix appended to the value column name for the
        final weighted aggregate.  None uses the default '{col}_{stat_str}_wgt'.
    keep_band_cols : bool
        Keep intermediate per-band columns when spec_type='wbands' (default False).
    k_th_percentiles : list[float] | None
        If provided, compute a threshold for each output column as the given
        percentile of its values.  Must have one entry per entry in value_cols.
        When None no thresholds are computed and an empty dict is returned.

    Returns
    -------
    thresholds : dict {output_col: threshold_value}
        Empty when k_th_percentiles is None.
    output_cols : set[str]
        Names of the final output columns written to dataframe.
    """
    thresholds = {}
    output_cols = set()
    band_columns = []

    # Subtract inner-radius aggregate from outer-radius aggregate for each band.
    for r_inner, r_outer, *_ in spec_data:
        for col in value_cols:
            outer_col = radius_col_fn(col, r_outer)
            inner_col = radius_col_fn(col, r_inner)
            band_col = f'{col}_{stat_str}_{_fmt_r(r_inner)}_{_fmt_r(r_outer)}'
            dataframe[band_col] = dataframe[outer_col].values - dataframe[inner_col].values
            band_columns.append(band_col)
            if spec_type == 'bands':
                output_cols.add(band_col)

    if spec_type == 'bands':
        if k_th_percentiles is not None:
            for r_inner, r_outer, *_ in spec_data:
                for col in value_cols:
                    band_col = f'{col}_{stat_str}_{_fmt_r(r_inner)}_{_fmt_r(r_outer)}'
                    percentile_for_col = k_th_percentiles[value_cols.index(col)]
                    thresholds[band_col] = _np.percentile(dataframe[band_col].values, percentile_for_col)

    elif spec_type == 'wbands':
        total_weight = sum(weight for _, _, weight in spec_data)
        for col in value_cols:
            weighted_values = _np.zeros(len(dataframe), dtype=float)
            for r_inner, r_outer, weight in spec_data:
                band_col = f'{col}_{stat_str}_{_fmt_r(r_inner)}_{_fmt_r(r_outer)}'
                weighted_values += (weight / total_weight) * dataframe[band_col].values
            if wgt_col_suffix is not None:
                weighted_col = f'{col}{wgt_col_suffix}'
            else:
                weighted_col = f'{col}_{stat_str}_wgt'
            dataframe[weighted_col] = weighted_values
            output_cols.add(weighted_col)
            if k_th_percentiles is not None:
                percentile_for_col = k_th_percentiles[value_cols.index(col)]
                thresholds[weighted_col] = _np.percentile(weighted_values, percentile_for_col)

        if not keep_band_cols:
            for band_col in band_columns:
                if band_col in dataframe.columns:
                    dataframe.drop(columns=[band_col], inplace=True)

    return thresholds, output_cols


def _multi_radius_search(
    pts: _pd.DataFrame,
    r,
    c=None,
    x: str = 'x',
    y: str = 'y',
    stat: str = 'sum',
    suffix=None,
    keep_cols=False,
    exclude_self: bool = False,
    silent: bool = False,
    _radius_search_fn=None,
    _parsed_spec=None,   # (spec_type, spec_data) — pass if already parsed to skip re-parsing
    **kwargs,
):
    """
    Multi-radius and distance-band wrapper around radius_search.

    Called internally by ``radius_search`` when ``r`` is not a scalar.
    Do not call this function directly — use ``radius_search(r=...)`` instead.

    Parameters
    ----------
    r
        Single radius, list of radii, list of (r_inner, r_outer) band tuples,
        or list of (r_inner, r_outer, weight) weighted band tuples.
    c : str | list[str]
        Value column(s) to aggregate.
    stat : str
        Aggregation statistic.  For distance bands, only additive statistics
        (``'sum'``, ``'count'``) give a strictly correct result.  Other stats
        are subtracted numerically but the interpretation is the user's
        responsibility.  Multi-stat lists are not supported for bands.
    suffix : str | None
        For list mode: appended after the stat identifier (default: ``_{stat}_{r}``).
        For weighted-band mode: the final column suffix (default: ``_wgt``).
    keep_cols : bool
        If True, intermediate band columns are retained in pts (default False).
    exclude_self : bool
        Forwarded to radius_search and used in r=0 grouping.

    Returns
    -------
    The grid object returned by the last radius_search call (or None for r=0 only).
    """
    global _WARNED_BAND_COUNT

    if _radius_search_fn is None:
        from aabpl.main import radius_search as _radius_search_fn

    spec_type, spec_data = _parsed_spec if _parsed_spec is not None else _parse_r_spec(r)

    # ── single radius: pass through unchanged ────────────────────────────────
    if spec_type == 'single':
        call_kwargs = dict(c=c, x=x, y=y, stat=stat, keep_cols=keep_cols,
                           exclude_self=exclude_self, silent=silent, **kwargs)
        if suffix is not None:
            call_kwargs['suffix'] = suffix
        return _radius_search_fn(pts=pts, r=spec_data, **call_kwargs)

    if isinstance(stat, (list, tuple)) and spec_type in ('bands', 'wbands'):
        raise ValueError(
            "Multi-stat lists are not supported for distance bands. "
            "Pass a single stat string."
        )

    value_cols = [c] if isinstance(c, str) else list(c)
    stat_str = _AGG_ABBR.get(stat, stat) if isinstance(stat, str) else 'agg'
    original_columns = set(pts.columns)

    # ── collect unique radii and warn if many ────────────────────────────────
    if spec_type == 'list':
        unique_radii = sorted(set(spec_data))
    else:
        unique_radii = sorted(set(rv for r_in, r_out, *_ in spec_data for rv in (r_in, r_out)))

    if len(spec_data) > 5 and not _WARNED_BAND_COUNT:
        from aabpl.utils.progress import progress_print
        progress_print(
            "Note that calculating each distance band results in a new radius "
            "search, which might be slow."
        )
        _WARNED_BAND_COUNT = True

    # ── run radius_search once per unique radius ──────────────────────────────
    # Intermediate columns are named {col}__mr__{r} and cleaned up below.
    # radius_to_temp_cols maps each radius to its {value_col: temp_col_name} dict.
    radius_to_temp_cols: dict = {}
    grids_by_radius: dict = {}   # kept on the returned grid for compute_null_distribution
    last_grid = None

    for radius in unique_radii:
        temp_suffix = f'{_TEMP_COL_MARKER}{_fmt_r(radius)}'
        radius_to_temp_cols[radius] = {col: f'{col}{temp_suffix}' for col in value_cols}
        if radius == 0.0:
            r0_aggregates = _agg_at_r0(pts, cols=value_cols, x=x, y=y,
                                       stat=stat, exclude_self=exclude_self)
            for col in value_cols:
                pts[f'{col}{temp_suffix}'] = r0_aggregates[col].values
            grids_by_radius[radius] = None
        else:
            last_grid = _radius_search_fn(
                pts=pts, r=radius, c=value_cols, x=x, y=y, stat=stat,
                suffix=temp_suffix, keep_cols=True,
                exclude_self=exclude_self, silent=silent, **kwargs,
            )
            grids_by_radius[radius] = last_grid

    # ── build final output columns ────────────────────────────────────────────
    output_columns: set = set()

    if spec_type == 'list':
        r_index = {i: radius for i, radius in enumerate(spec_data)}
        for i, radius in enumerate(spec_data):
            for col in value_cols:
                temp_col = radius_to_temp_cols[radius][col]
                output_col = (f'{col}{suffix}{_fmt_r(radius)}' if suffix is not None
                              else f'{col}_{stat_str}_{_fmt_r(radius)}')
                pts[output_col] = pts[temp_col].values
                output_columns.add(output_col)

    else:  # bands or wbands
        r_index = {i: (r_inner, r_outer)
                   for i, (r_inner, r_outer, *_) in enumerate(spec_data)}
        temp_col_fn = lambda col, radius: radius_to_temp_cols[radius][col]
        # The band combination logic is shared with the null-distribution path
        # via _combine_bands_on_df so both DataFrames always use the same algebra.
        _, output_columns = _combine_bands_on_df(
            dataframe=pts,
            spec_type=spec_type,
            spec_data=spec_data,
            value_cols=value_cols,
            stat_str=stat_str,
            radius_col_fn=temp_col_fn,
            wgt_col_suffix=suffix,   # None → '{col}_{stat_str}_wgt', else '{col}{suffix}'
            keep_band_cols=keep_cols,
        )

    # ── remove temporary intermediate columns ────────────────────────────────
    for col_name in list(pts.columns):
        if _TEMP_COL_MARKER in col_name:
            pts.drop(columns=[col_name], inplace=True)

    # ── remove grid-assignment columns added internally by radius_search ─────
    if not keep_cols:
        cols_to_drop = set(pts.columns) - original_columns - output_columns
        if cols_to_drop:
            pts.drop(columns=list(cols_to_drop), inplace=True)

    # Attach grids_by_radius to the returned grid so detect_cluster_pts can
    # reuse the already-built search structures for the null distribution.
    if last_grid is not None:
        last_grid._mr_grids = grids_by_radius
        last_grid._r_index = r_index

    return last_grid


# Keep the old public name as an alias so any existing external call sites
# and tests still work without changes.
multi_radius_search = _multi_radius_search

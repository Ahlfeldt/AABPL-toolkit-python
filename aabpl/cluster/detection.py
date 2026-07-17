"""
Multi-radius cluster detection.

``_detect_cluster_pts_multi`` is the implementation path for detect_cluster_pts
when r is not a single scalar.  It is called by detect_cluster_pts in main.py
and is not intended for direct use.

After running, it stores ``grid._cluster_result`` with the data that
``grid.plot.rand_dist()`` and ``grid.plot.cluster_pts()`` read from.
"""
from __future__ import annotations
from numpy import array as _np_array
from pandas import DataFrame as _pd_DataFrame


def _fmt_r_human(v):
    """Format radius value as human-readable string (e.g. 15000 -> '15km', 500 -> '500m')."""
    if v >= 1000:
        scaled = v / 1000.0
        suffix = 'km'
    else:
        scaled = float(v)
        suffix = 'm'
    return (str(int(scaled)) if scaled == int(scaled) else f'{scaled:.4g}') + suffix


def _detect_cluster_pts_multi(
    pts,
    crs,
    r,
    c,
    x,
    y,
    stat,
    exclude_self,
    cell_size,
    study_area,
    area_weight,
    k_th_percentile,
    null_distribution,
    random_seed,
    proj_crs,
    row_name,
    col_name,
    cluster_suffix,
    pts_target,
    keep_cols,
    overwrite,
    silent,
    parsed_spec,
    min_pts_to_sample_cell,
    _dev,
    grid_bounds,
    x_tgt,
    y_tgt,
    row_name_tgt,
    col_name_tgt,
):
    """
    Multi-radius implementation of detect_cluster_pts.

    Runs one radius_search per unique radius, combines bands or weighted bands,
    draws a shared null distribution, and labels each point.

    Called by detect_cluster_pts when r is not a single scalar.
    Not intended for direct use — call detect_cluster_pts instead.
    """
    from aabpl.main import radius_search, _validate_kwargs, resolve_study_area
    from aabpl.utils.misc import find_column_name
    from aabpl.search.multi_radius import (
        _parse_r_spec, _multi_radius_search, _fmt_r, _AGG_ABBR, _combine_bands_on_df,
    )
    from aabpl.search.null_distribution import compute_null_distribution
    from aabpl.search.study_area import intersect_polygon_with_grid

    spec_type, spec_data = parsed_spec if parsed_spec is not None else _parse_r_spec(r)
    max_radius = (
        max(spec_data)
        if spec_type == 'list'
        else max(r_outer for _, r_outer, *_ in spec_data)
    )

    initial_columns = set(pts.columns)
    sort_order_col = find_column_name('initial_sort', existing_columns=pts.columns)
    pts[sort_order_col] = range(len(pts))

    validated = _validate_kwargs(
        pts=pts, crs=crs, r=max_radius, c=c,
        stat=stat, x=x, y=y, row_name=row_name, col_name=col_name,
        suffix=None,
        pts_target=pts_target, x_tgt=x_tgt, y_tgt=y_tgt,
        row_name_tgt=row_name_tgt, col_name_tgt=col_name_tgt,
        build_grid_obj=False,
        n_pts_src_extra=null_distribution if isinstance(null_distribution, int) else 0,
        proj_crs=proj_crs, silent=silent,
    )
    pts          = validated.pts
    local_crs    = validated.local_crs
    value_cols   = list(validated.c)
    x, y, stat   = validated.x, validated.y, validated.stat
    pts_target   = validated.pts_target
    x_tgt, y_tgt = validated.x_tgt, validated.y_tgt
    row_name_tgt  = validated.row_name_tgt
    col_name_tgt  = validated.col_name_tgt

    if cluster_suffix is None:
        cluster_suffix = '_cluster'

    stat_str = _AGG_ABBR.get(stat, stat)

    last_grid = _multi_radius_search(
        pts=pts, r=r, c=value_cols, x=x, y=y, stat=stat,
        keep_cols=True, exclude_self=exclude_self, silent=silent,
        _radius_search_fn=radius_search,
        _parsed_spec=(spec_type, spec_data),
        crs=local_crs or '', proj_crs=local_crs,
        row_name=row_name, col_name=col_name,
        pts_target=pts_target, x_tgt=x_tgt, y_tgt=y_tgt,
        row_name_tgt=row_name_tgt, col_name_tgt=col_name_tgt,
        area_weight=area_weight,
    )

    last_grid.study_area = resolve_study_area(
        pts=pts, r=max_radius, study_area=study_area,
        crs=crs, local_crs=local_crs,
        x=x, y=y, grid=last_grid,
        min_pts_to_sample_cell=min_pts_to_sample_cell,
    )
    intersect_polygon_with_grid(grid=last_grid, area_weight=area_weight)

    grids_by_radius = last_grid._mr_grids
    k_th_percentiles = (
        [k_th_percentile] * len(value_cols)
        if not isinstance(k_th_percentile, list)
        else k_th_percentile
    )

    # Iterate radii largest-first so the first call draws the random points;
    # subsequent calls receive the DataFrame and append their columns to it.
    thresholds_by_col = {}
    null_df = null_distribution
    for radius in sorted(grids_by_radius.keys(), reverse=True):
        grid_for_radius = grids_by_radius[radius]
        if grid_for_radius is None or radius == 0.0:
            if isinstance(null_df, _pd_DataFrame):
                for col in value_cols:
                    null_df[f'{col}_{stat_str}_{_fmt_r(radius)}'] = 0.0
            continue
        radius_suffix = f'_{stat_str}_{_fmt_r(radius)}'
        thresholds_for_radius, null_df = compute_null_distribution(
            grid=grid_for_radius, pts=pts, study_area=last_grid.study_area,
            min_pts_to_sample_cell=min_pts_to_sample_cell,
            c=value_cols, x=x, y=y, row_name=row_name, col_name=col_name,
            suffix=radius_suffix, null_distribution=null_df,
            k_th_percentile=k_th_percentile, random_seed=random_seed,
            silent=silent,
        )
        if spec_type == 'list':
            thresholds_by_col.update(thresholds_for_radius)

    if 0.0 in grids_by_radius and isinstance(null_df, _pd_DataFrame):
        for col in value_cols:
            null_df[f'{col}_{stat_str}_{_fmt_r(0.0)}'] = 0.0

    if spec_type in ('bands', 'wbands'):
        def null_radius_col_fn(col, radius):
            return f'{col}_{stat_str}_{_fmt_r(radius)}'

        thresholds_from_bands, _ = _combine_bands_on_df(
            dataframe=null_df,
            spec_type=spec_type,
            spec_data=spec_data,
            value_cols=value_cols,
            stat_str=stat_str,
            radius_col_fn=null_radius_col_fn,
            keep_band_cols=False,
            k_th_percentiles=k_th_percentiles,
        )
        thresholds_by_col.update(thresholds_from_bands)
        for radius in grids_by_radius:
            for col in value_cols:
                per_radius_col = f'{col}_{stat_str}_{_fmt_r(radius)}'
                if per_radius_col in null_df.columns:
                    null_df.drop(columns=[per_radius_col], inplace=True)

    cluster_col_map = {}
    for output_col, threshold in thresholds_by_col.items():
        if output_col not in pts.columns:
            if not silent:
                from aabpl.utils.progress import progress_print
                progress_print(
                    f'Warning: aggregate column "{output_col}" not found in pts '
                    f'(columns: {list(pts.columns)}); cluster boolean skipped.'
                )
            continue
        cluster_col = output_col + cluster_suffix
        n_above = int((pts[output_col] > threshold).sum())
        pts[cluster_col] = pts[output_col] > threshold
        if not silent:
            from aabpl.utils.progress import progress_print
            progress_print(
                f'Cluster column "{cluster_col}": {n_above}/{len(pts)} pts above '
                f'threshold {threshold:g} (max value: {pts[output_col].max():g}).'
            )
        for col in value_cols:
            if output_col == col or output_col.startswith(col + '_'):
                cluster_col_map[cluster_col] = col
                break
    last_grid._cluster_col_map = cluster_col_map

    last_grid._multi_radius_output_cols = (
        set(thresholds_by_col.keys()) | set(cluster_col_map.keys())
    )

    last_grid._search_class.set_source(
        pts=pts, c=value_cols, x=x, y=y,
        row_name=row_name, col_name=col_name,
        suffix=f'_{stat_str}_{_fmt_r(max_radius)}',
        silent=True,
    )

    last_grid.null_distribution = null_df

    # Build a per-column r-spec so the radius indicator can show the correct
    # visual for each output column (circle, donut, or weighted donuts).
    if spec_type == 'list':
        # Each output col corresponds to one specific radius — extract it from
        # the column name suffix so the indicator shows just that ring.
        col_to_r_display = {}
        for output_col in thresholds_by_col:
            for radius in spec_data:
                if output_col.endswith(f'_{stat_str}_{_fmt_r(radius)}'):
                    col_to_r_display[output_col] = radius
                    break
            else:
                col_to_r_display[output_col] = max_radius
    elif spec_type == 'bands':
        # Each band column b{i} maps to one (r_inner, r_outer) tuple.
        col_to_r_display = {}
        for output_col in thresholds_by_col:
            for r_inner, r_outer in spec_data:
                if output_col.endswith(f'_{stat_str}_{_fmt_r(r_inner)}_{_fmt_r(r_outer)}'):
                    col_to_r_display[output_col] = (r_inner, r_outer)
                    break
            else:
                col_to_r_display[output_col] = max_radius
    else:  # wbands — the weighted column represents the full set of bands
        col_to_r_display = {output_col: spec_data for output_col in thresholds_by_col}

    if not hasattr(last_grid, '_aabpl_col_meta'):
        last_grid._aabpl_col_meta = {}
    for output_col in thresholds_by_col:
        originating_col = next(
            col for col in value_cols
            if output_col == col or output_col.startswith(col + '_')
        )
        r_for_col = col_to_r_display.get(output_col, r)
        meta = {'c': originating_col, 'stat': stat, 'r': r_for_col}
        last_grid._aabpl_col_meta[output_col] = meta
        last_grid._aabpl_col_meta[output_col + cluster_suffix] = meta

    # Print one threshold line per output column with radius and k context.
    if not silent:
        from aabpl.utils.progress import progress_print
        for i, (output_col, threshold) in enumerate(thresholds_by_col.items()):
            k_val = k_th_percentiles[i % len(k_th_percentiles)]
            meta = last_grid._aabpl_col_meta.get(output_col, {})
            r_disp = meta.get('r', max_radius)
            orig_c = meta.get('c', output_col)
            # wbands: r_disp is a list of (r_in, r_out, w) tuples
            _is_wbands = (
                isinstance(r_disp, list) and len(r_disp) > 0
                and isinstance(r_disp[0], (list, tuple)) and len(r_disp[0]) == 3
            )
            # bands: r_disp is a 2-element tuple of scalars
            _is_band = (
                isinstance(r_disp, tuple) and len(r_disp) == 2
                and isinstance(r_disp[0], (int, float))
            )
            if _is_wbands:
                r_outer = max(t[1] for t in r_disp)
                stat_label = 'weighted sum'
                r_str = f'within {_fmt_r_human(r_outer)}'
            elif _is_band:
                stat_label = meta.get('stat', stat)
                r_str = f'{_fmt_r_human(r_disp[0])}-{_fmt_r_human(r_disp[1])} band'
            elif isinstance(r_disp, (int, float)):
                stat_label = meta.get('stat', stat)
                r_str = _fmt_r_human(r_disp)
            else:
                stat_label = meta.get('stat', stat)
                r_str = str(r_disp)
            progress_print(
                f'Threshold for {orig_c} ({stat_label}, {r_str}): '
                f'{k_val}th-percentile = {threshold:g}.'
            )

    aggregate_cols = list(thresholds_by_col.keys())
    cluster_cols   = list(cluster_col_map.keys())
    # Expand k_th_percentiles so it has one entry per aggregate_col, not one per
    # value_col.  For bands/list r, aggregate_cols = len(bands)*len(value_cols);
    # k_th_percentiles was sized to len(value_cols).  The zip in create_distribution_plot
    # stops at the shortest iterable, silently dropping the extra panels without this fix.
    _k_per_agg = []
    for _col in aggregate_cols:
        _orig_idx = next(
            (j for j, vc in enumerate(value_cols)
             if _col == vc or _col.startswith(vc + '_')),
            0,
        )
        _k_per_agg.append(k_th_percentiles[_orig_idx % len(k_th_percentiles)])
    # Per-column lookup: output_col -> {k_percentile: threshold_value}
    col_threshold_info = {
        col: {_k_per_agg[i]: thresholds_by_col[col]}
        for i, col in enumerate(aggregate_cols)
    }
    last_grid._cluster_result = {
        'aggregate_cols':     aggregate_cols,
        'thresholds':         thresholds_by_col,
        'k_th_percentiles':   _k_per_agg,
        'col_threshold_info': col_threshold_info,
        'display_radius':     max_radius,
        'plot_colnames':      _np_array(list(value_cols) + aggregate_cols + cluster_cols),
    }

    if not keep_cols:
        intermediate_cols = [
            col for col in pts.columns
            if col not in initial_columns
            and not col.endswith(cluster_suffix)
            and col != sort_order_col
        ]
        if intermediate_cols:
            pts.drop(columns=intermediate_cols, inplace=True)

    pts.sort_values(sort_order_col, inplace=True)
    pts.drop(columns=[sort_order_col], inplace=True)

    return last_grid

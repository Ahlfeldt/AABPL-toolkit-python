from pandas import DataFrame as _pd_DataFrame
from numpy import (
    array as _np_array,
    zeros as _np_zeros,
    linspace as _np_linspace,
    searchsorted as _np_searchsorted,
    sort as _np_sort,
)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable as _make_axes_locatable
from matplotlib.pyplot import close as _plt_close
from matplotlib.colors import LogNorm as _plt_LogNorm, Normalize as _plt_Normalize, LinearSegmentedColormap as _plt_LinearSegmentedColormap, ListedColormap as _plt_ListedColormap
from matplotlib.pyplot import (subplots as _plt_subplots, colorbar as _plt_colorbar, get_cmap as _plt_get_cmap)
from matplotlib.ticker import FuncFormatter as _FuncFormatter
from matplotlib.patches import Patch as _plt_Patch
from aabpl.illustrations.plot_utils import add_color_bar_ax, set_map_frame, draw_radius_indicator, truncate_colormap, plot_polygon, format_col_title
from shapely.geometry import Polygon as _shapely_Polygon

# Colour used for zero-value points and the colorbar's 'under' extension.
# Change here to update it globally across all distribution plots.
_ZERO_GREY = '#e9e9e9'

def create_distribution_plot(
        pts:_pd_DataFrame,
        rndm_pts:_pd_DataFrame,
        cluster_threshold_values:list,
        k_th_percentile:list,
        radius_sum_columns:list,
        grid:dict=None,
        r:float=None,
        x:str='lon',
        y:str='lat',
        filename:str='',
        plot_kwargs:dict={},
        show:bool=True,
        display_dpi:int=100,
        save_kwargs:dict={},
):
    """
    Four-panel plot comparing the random-point null distribution to the
    observed radius-sum distribution. Called via ``grid.plot.rand_dist()``.

    Parameters
    ----------
    pts : DataFrame
        Source points with radius-sum columns already appended.
    rndm_pts : DataFrame
        Random points with radius-sum columns appended by ``compute_null_distribution``.
    cluster_threshold_values : list[float]
        One threshold per column — the k-th percentile cutoff value.
    k_th_percentile : list[float]
        Percentile values used to derive ``cluster_threshold_values``.
    radius_sum_columns : list[str]
        Column names of the radius sums to plot.
    grid
        Grid object (used to draw sample-area overlay).
    r : float
        Search radius in metres (used for the plot title).
    x, y : str
        Coordinate column names in ``pts`` and ``rndm_pts``.
    filename : str
        Save path. Empty string skips saving (default ``''``).
    plot_kwargs : dict
        Supported keys:

        Layout / figure
        ~~~~~~~~~~~~~~~
        figsize : tuple
            Figure size (default ``(10, 10)``).
        maps_side_by_side : bool or None
            ``True`` = maps side by side (horizontal), ``False`` = stacked
            vertically.  ``None`` (default) = auto-detect from data aspect
            ratio, with a small bias towards horizontal.
        hspace : float
            Vertical spacing between distribution axis and map panels
            (default ``0.12``).
        wspace : float
            Horizontal spacing between side-by-side maps (default ``0.08``).
        colorbar_width : float
            Fractional width of the colorbar column when maps are side by
            side (default ``0.04``).
        dist_map_ratio : float
            Height of the distribution panel relative to the map panel
            (default ``0.3``). Decrease to give more vertical space to the maps.

        Scatter appearance
        ~~~~~~~~~~~~~~~~~~
        s : float
            Marker size for scatter points (default auto-scaled to figsize).
        color : str
            Colour for random-point distribution scatter (default ``'#eaa'``).
        pts_color : str
            Colour for dataset-point distribution scatter (default
            ``'#66aabb'``).
        zero_color : str
            Colour for zero-value points and the colorbar under-extension
            (default ``'#aaaaaa'``).
        cmap : str or Colormap
            Colormap for the radius-sum maps (default ``'Reds'``).
            Accepts any matplotlib colormap name or object.
        cmap_minval : float
            Lower truncation of the colormap, 0–1 (default ``0.2``).
        color_scale : str
            ``'log'``, ``'linear'``, or ``'auto'`` — log when vmin > 0
            (default ``'auto'``).
        color_cluster : str
            Colour for points above the cluster threshold (default
            ``'#2a07ee'``).

        Sample-area overlay
        ~~~~~~~~~~~~~~~~~~~
        non_valid_area_color : str
            Fill colour for the non-valid (excluded) area (default
            ``'#bedbe6'``).
        sample_area_color : str
            Fill colour for the valid sample area (default ``'#ffffff'``).
        sample_area_linewidth : float
            Border linewidth of the valid-area polygon (default ``0.2``).

        Threshold lines
        ~~~~~~~~~~~~~~~
        hlines : dict
            kwargs for ``ax.hlines`` threshold line (default
            ``{'color': 'red', 'linewidth': 1}``).
        vlines : dict
            kwargs for ``ax.vlines`` percentile line (default
            ``{'color': 'red', 'linewidth': 1}``).

        Titles
        ~~~~~~
        title : str or None
            Override the auto suptitle. Use ``{auto}`` to embed the generated
            title: ``"My prefix: {auto}"``.
        suptitle : dict
            Extra kwargs forwarded to ``fig.suptitle()``.
        ax_titles : dict
            Override individual panel titles, keyed by
            ``'rand_dist'``, ``'pts_dist'``, ``'rand_map'``, ``'pts_map'``.

        Existing figure
        ~~~~~~~~~~~~~~~
        fig, axs
            Existing Figure / Axes to draw into.

    show : bool
        Display the figure (default ``True``).
    display_dpi : int
        Resolution for inline/screen display (default ``100``).

    Returns
    -------
    matplotlib.figure.Figure
    """
    x_coord_name, y_coord_name = x, y
    for _df, _col, _label in [
        (pts, x_coord_name, 'pts'), (pts, y_coord_name, 'pts'),
        (rndm_pts, x_coord_name, 'rndm_pts'), (rndm_pts, y_coord_name, 'rndm_pts'),
    ]:
        if _col not in _df.columns:
            raise KeyError(
                f"Coordinate column '{_col}' not found in {_label}. "
                f"The column must be the local-projection coordinate (in metres). "
                f"Pass the correct column name via x= / y= to the plot call, "
                f"or check that keep_cols did not drop it. "
                f"Available columns: {list(_df.columns)}"
            )
    _pts_x = pts[x_coord_name]
    _pts_y = pts[y_coord_name]
    _rnd_x = rndm_pts[x_coord_name]
    _rnd_y = rndm_pts[y_coord_name]
    disk_sums_for_random_points = rndm_pts[radius_sum_columns]
    (n_random_points, ncols) = disk_sums_for_random_points.shape

    # ── Defaults ──────────────────────────────────────────────────────────────
    default_kwargs = {
        's': 0.8,
        'color': '#eaa',
        'pts_color': '#66aabb',
        'zero_color': _ZERO_GREY,
        'cmap': 'Reds',
        'cmap_minval': 0.2,
        'color_scale': 'auto',
        'color_cluster': '#2a07ee',

        'figsize': (10, 13),
        'maps_side_by_side': None,   # None = auto
        'hspace': 0.5,
        'wspace': 0.08,
        'colorbar_width': 0.06,
        'dist_map_ratio': 0.3,

        'non_valid_area_color': '#bedbe6',
        'sample_area_color': '#ffffff',
        'sample_area_linewidth': 0.2,

        'fig': None,
        'axs': None,
        'title': None,
        'suptitle': {},
        'ax_titles': {},
        'hlines': {'color': 'red', 'linewidth': 1},
        'vlines': {'color': 'red', 'linewidth': 1},
    }

    show = plot_kwargs.pop('show', show)
    display_dpi = plot_kwargs.pop('display_dpi', display_dpi)
    plot_kwargs.pop('filename', None)

    # Merge dict-typed defaults with caller overrides before flattening.
    kwargs = {}
    dict_keys = {k for k, v in default_kwargs.items() if isinstance(v, dict)}
    for k in list(plot_kwargs.keys()):
        if k in dict_keys:
            kwargs[k] = {**default_kwargs.pop(k), **plot_kwargs.pop(k)}
    kwargs.update(default_kwargs)
    kwargs.update(plot_kwargs)

    figsize              = kwargs.pop('figsize')
    maps_side_by_side_kw = kwargs.pop('maps_side_by_side')
    hspace               = kwargs.pop('hspace')
    wspace               = kwargs.pop('wspace')
    colorbar_width       = kwargs.pop('colorbar_width')
    dist_map_ratio       = kwargs.pop('dist_map_ratio')
    fig                  = kwargs.pop('fig')
    axs                  = kwargs.pop('axs')
    title_override       = kwargs.pop('title')
    suptitle_kwargs      = kwargs.pop('suptitle')
    ax_titles            = kwargs.pop('ax_titles')
    sample_area_lw       = kwargs.pop('sample_area_linewidth')
    color_scale          = kwargs.pop('color_scale')
    zero_color           = kwargs.pop('zero_color')
    non_valid_area_color = kwargs.pop('non_valid_area_color')
    sample_area_color    = kwargs.pop('sample_area_color')
    color_cluster        = kwargs.pop('color_cluster')
    cmap_name            = kwargs.pop('cmap')
    cmap_minval          = kwargs.pop('cmap_minval')
    pts_color            = kwargs.pop('pts_color')

    # Remove any remaining control keys that must not leak into ax.scatter().
    for k in ['fig', 'axs', 'figsize', 'title', 'suptitle', 'ax_titles',
              'sample_area_linewidth', 'color_scale', 'maps_side_by_side',
              'hspace', 'wspace', 'colorbar_width']:
        plot_kwargs.pop(k, None)

    if fig is None or axs is None:
        fig = plt.figure(figsize=figsize, dpi=display_dpi)
        outer = gridspec.GridSpec(ncols, 1, figure=fig, hspace=0.08,
                                  top=0.93, bottom=0.04, left=0.08, right=0.97)

    _col_meta_dist = getattr(grid, '_aabpl_col_meta', {})

    def _fmt_r_for_title(rv):
        if isinstance(rv, (int, float)):
            km = rv / 1000.0
            if rv >= 1000:
                return 'within ' + (str(int(km)) if km == int(km) else f'{km:.4g}') + 'km'
            return 'within ' + str(int(rv)) + 'm'
        if isinstance(rv, (list, tuple)) and len(rv) > 0:
            outer = max((t[1] if isinstance(t, (list, tuple)) else t) for t in rv)
            km = outer / 1000.0
            if outer >= 1000:
                return 'within ' + (str(int(km)) if km == int(km) else f'{km:.4g}') + 'km'
            return 'within ' + str(int(outer)) + 'm'
        return 'within ' + str(rv)

    _title_parts = []
    for _cn in radius_sum_columns:
        _m  = _col_meta_dist.get(_cn, {})
        _c  = _m.get('c', _cn)
        _st = _m.get('stat', '')
        _rv = _m.get('r', r)
        _title_parts.append(
            (_c + ' ' + _st + ' ' if _st else _c + ' ') + _fmt_r_for_title(_rv)
        )
    _auto_title = ', '.join(_title_parts) + ' - Null distribution vs Data'
    _suptitle = (
        title_override.replace('{auto}', _auto_title) if title_override
        else _auto_title
    )
    fig.suptitle(_suptitle, **suptitle_kwargs)

    _sa_xmin, _sa_ymin, _sa_xmax, _sa_ymax = grid.sample_area.bounds
    _sa_w = _sa_xmax - _sa_xmin
    _sa_h = _sa_ymax - _sa_ymin
    _sa_span = max(_sa_w, _sa_h)
    # Equal absolute padding in both dimensions (same distance regardless of aspect).
    _map_pad2 = 0.02 * _sa_span
    _ax_xmin = _sa_xmin - _map_pad2
    _ax_xmax = _sa_xmax + _map_pad2
    _ax_ymin = _sa_ymin - _map_pad2
    _ax_ymax = _sa_ymax + _map_pad2
    # Outer rectangle slightly larger so excluded fringe is visible even for bbox sample areas.
    _map_pad3 = 0.03 * _sa_span
    _out_xmin = _sa_xmin - _map_pad3
    _out_xmax = _sa_xmax + _map_pad3
    _out_ymin = _sa_ymin - _map_pad3
    _out_ymax = _sa_ymax + _map_pad3
    non_valid_area = _shapely_Polygon([
        (_out_xmin, _out_ymin), (_out_xmax, _out_ymin),
        (_out_xmax, _out_ymax), (_out_xmin, _out_ymax),
    ]).difference(grid.sample_area)

    pct_xmin, pct_xmax = 0, 100
    xs_random_pts = _np_linspace(pct_xmin, pct_xmax, n_random_points)
    xs_pts        = _np_linspace(pct_xmin, pct_xmax, len(pts))
    random_vals   = disk_sums_for_random_points.values
    pts_vals      = pts[radius_sum_columns].values

    for (i, colname, cluster_threshold_value, k) in zip(
            range(ncols), radius_sum_columns, cluster_threshold_values, k_th_percentile):
        # Human-readable column label used in default panel titles
        col_label = format_col_title(colname, _col_meta_dist.get(colname, {}))

        # ── Determine map layout ──────────────────────────────────────────────
        map_w = max(
            (_pts_x.max() if hasattr(_pts_x, 'max') else max(_pts_x)) -
            (_pts_x.min() if hasattr(_pts_x, 'min') else min(_pts_x)), 1e-9
        )
        map_h = max(
            (_pts_y.max() if hasattr(_pts_y, 'max') else max(_pts_y)) -
            (_pts_y.min() if hasattr(_pts_y, 'min') else min(_pts_y)), 1e-9
        )
        if maps_side_by_side_kw is None:
            # Prefer horizontal (side by side) unless data is notably portrait.
            maps_side_by_side = map_h < map_w * 1.2
        else:
            maps_side_by_side = maps_side_by_side_kw

        # ── GridSpec layout ───────────────────────────────────────────────────
        panel = gridspec.GridSpecFromSubplotSpec(
            2, 1, subplot_spec=outer[i], hspace=hspace,
            height_ratios=[dist_map_ratio, 1.0],
        )
        if maps_side_by_side:
            bottom_gs = gridspec.GridSpecFromSubplotSpec(
                1, 2, subplot_spec=panel[1], wspace=wspace,
            )
        else:
            # Two rows: rand_map (top) | pts_map (bottom); colorbar spans both.
            bottom_gs = gridspec.GridSpecFromSubplotSpec(
                2, 1, subplot_spec=panel[1], hspace=hspace,
            )

        # ── Cumulative distribution panel ─────────────────────────────────────
        ys_rnd = _np_sort(random_vals[:, i])
        ys_pts = _np_sort(pts_vals[:, i])
        ymin   = min(ys_rnd.min(), ys_pts.min())
        ymax   = max(ys_rnd.max(), ys_pts.max())

        idx = _np_searchsorted(ys_rnd, cluster_threshold_value)
        next_smaller_val = ys_rnd[max(0, idx - 1)]
        next_larger_val  = ys_rnd[idx] if idx < len(ys_rnd) else ys_rnd[-1]
        sufficient_digits = next((
            d for d in range(100) if (
                (
                    (next_smaller_val == next_larger_val or
                     cluster_threshold_value == next_smaller_val) and
                    round(next_smaller_val, d) == next_smaller_val
                ) or (
                    round(next_larger_val, d) != round(cluster_threshold_value, d) and
                    round(next_smaller_val, d) != round(cluster_threshold_value, d)
                )
            )
        ), 100)

        ax_dist = fig.add_subplot(panel[0])
        ax_title = ax_titles.get('rand_dist', "{col}: {k}th-percentile threshold = {thresh}")
        ax_title = ax_title.format(
            col=col_label, k=k,
            thresh=round(cluster_threshold_value, sufficient_digits),
            r=r, n=len(rndm_pts),
        )
        ax_dist.set_title(ax_title, fontdict={'fontsize': 6})
        ax_dist.set_ylabel("Radius sum", fontsize=7)
        ax_dist.set_xlabel("Percentile rank", fontsize=7)

        xtick_steps, ytick_steps = 5, 5
        xticks = _np_array(sorted(
            [x for x in _np_linspace(pct_xmin, pct_xmax, xtick_steps)
             if abs(x - k) > (pct_xmax - pct_xmin) / (xtick_steps * 2)] + [k]
        ))
        ax_dist.set_xticks(xticks, labels=xticks)
        yticks = _np_array(sorted(
            [y for y in _np_linspace(ymin, ymax, ytick_steps)
             if abs(cluster_threshold_value - y) > (ymax - ymin) / (ytick_steps * 10)] +
            [cluster_threshold_value]
        ))
        ax_dist.set_yticks(yticks, labels=[round(t, sufficient_digits) for t in yticks])

        ax_dist.hlines(y=cluster_threshold_value, xmin=pct_xmin, xmax=pct_xmax, **kwargs['hlines'])
        ax_dist.vlines(x=k, ymin=ymin, ymax=ymax, **kwargs['vlines'])

        _s = plot_kwargs.get('s', 0.8)
        ax_dist.scatter(
            x=xs_random_pts, y=ys_rnd, s=_s, marker='.',
            label=f"Random pts (n={len(rndm_pts):,})",
            **{k_: v for k_, v in plot_kwargs.items() if k_ != 's'},
        )
        ax_dist.scatter(
            x=xs_pts, y=ys_pts, color=pts_color, s=_s, marker='.',
            label=f"Dataset pts (n={len(pts):,})",
        )
        ax_dist.set_xlim([pct_xmin, pct_xmax])
        if ymin != ymax:
            ax_dist.set_ylim([ymin, ymax])
        ax_dist.legend(fontsize=6, markerscale=5)

        # ── Colormap / norm setup ─────────────────────────────────────────────
        map_xmin = min(_pts_x.min(), _rnd_x.min())
        map_xmax = max(_pts_x.max(), _rnd_x.max())
        map_ymin = min(_pts_y.min(), _rnd_y.min())
        map_ymax = max(_pts_y.max(), _rnd_y.max())

        _pts_nonzero = pts_vals[:, i][pts_vals[:, i] != 0]
        _rnd_nonzero = random_vals[:, i][random_vals[:, i] != 0]
        _has_range = len(_pts_nonzero) > 0 and len(_rnd_nonzero) > 0
        if _has_range:
            vmin = min(_pts_nonzero.min(), _rnd_nonzero.min())
            vmax = max(pts_vals[:, i].max(), random_vals[:, i].max())
            _has_range = vmin < vmax and vmin > 0
        if not _has_range:
            vmin = 1.0
            vmax = max(float(cluster_threshold_value), 2.0)

        _base_cmap = cmap_name if hasattr(cmap_name, 'N') else _plt_get_cmap(cmap_name)
        cmap_scatter = truncate_colormap(
            cmap=_base_cmap, minval=cmap_minval, maxval=1.0, n=100
        )
        # Zero colour: sample the base cmap at half cmap_minval so it sits
        # visually just below the data range — same hue family, clearly lighter.
        # Falls back to the explicit zero_color parameter if the user set one.
        _zero_col = zero_color if zero_color != _ZERO_GREY else _base_cmap(cmap_minval / 2)
        cmap_scatter.set_under(_zero_col)
        cmap_scatter.set_bad(_zero_col)
        # Values above the cluster threshold → cluster colour.
        cmap_scatter.set_over(color_cluster)

        _norm_vmax = float(cluster_threshold_value) if cluster_threshold_value > vmin else vmax
        _use_log   = (color_scale == 'log') or (color_scale == 'auto' and vmin > 0)
        norm = (
            _plt_LogNorm(vmin=vmin, vmax=_norm_vmax, clip=False) if _use_log
            else _plt_Normalize(vmin=vmin, vmax=_norm_vmax, clip=False)
        )
        s_map = 0.2 * figsize[0] / 10

        # ── Sample-area grid overlay (shared by both map panels) ──────────────
        X = None
        extent = None
        cmap_binary = None
        if grid is not None:
            cells_rndm_sample = grid._search_internals.cells_rndm_sample
            _si = grid._search_internals
            col_min = int(round((grid.sample_grid_bounds[0] - _si.bounds.xmin) / _si.spacing, 0))
            row_min = int(round((grid.sample_grid_bounds[1] - _si.bounds.ymin) / _si.spacing, 0))
            col_max = int(round((grid.sample_grid_bounds[2] - _si.bounds.xmin) / _si.spacing - 1, 0))
            row_max = int(round((grid.sample_grid_bounds[3] - _si.bounds.ymin) / _si.spacing - 1, 0))
            n_rows_x = row_max - row_min + 1
            n_cols_x = col_max - col_min + 1
            if type(cells_rndm_sample) == bool:
                X = _np_zeros((n_rows_x, n_cols_x), dtype=bool)
                if not cells_rndm_sample:
                    X[:] = True
            else:
                X = _np_zeros((n_rows_x, n_cols_x), dtype=bool)
                X[:] = True
                for (row, col) in cells_rndm_sample:
                    ri = row_max - row
                    ci = col - col_min
                    if 0 <= ri < n_rows_x and 0 <= ci < n_cols_x:
                        X[ri, ci] = False
            cmap_binary = _plt_ListedColormap([sample_area_color, non_valid_area_color])
            extent = [
                grid.sample_grid_bounds[0], grid.sample_grid_bounds[2],
                grid.sample_grid_bounds[1], grid.sample_grid_bounds[3],
            ]

        # ── Map: random points ────────────────────────────────────────────────
        ax_rnd = fig.add_subplot(bottom_gs[0])
        ax_rnd.set_facecolor(sample_area_color)
        ax_rnd.set_title(
            ax_titles.get('rand_map', "Random points (n={n})").format(
                col=col_label, k=k,
                thresh=round(cluster_threshold_value, sufficient_digits),
                r=r, n=len(rndm_pts),
            ),
            fontdict={'fontsize': 6},
        )
        if X is not None:
            ax_rnd.imshow(X=X, interpolation='none', cmap=cmap_binary, extent=extent)
            non_valid_patch = _plt_Patch(facecolor=non_valid_area_color, label='Non-valid area', edgecolor='black')
            sample_patch    = _plt_Patch(facecolor=sample_area_color,    label='Sample area',    edgecolor='black')
            ax_rnd.legend(handles=[non_valid_patch, sample_patch], loc='upper left', fontsize=5)
        plot_polygon(ax=ax_rnd, poly=non_valid_area, facecolor=non_valid_area_color,
                     edgecolor='black', linewidth=sample_area_lw)
        sc = ax_rnd.scatter(
            x=_rnd_x, y=_rnd_y, c=random_vals[:, i],
            s=s_map, marker='.', norm=norm, cmap=cmap_scatter, linewidths=0.3,
        )
        plot_polygon(ax=ax_rnd, poly=grid.sample_area, facecolor="none",
                     edgecolor='black', linewidth=sample_area_lw)
        set_map_frame(ax=ax_rnd, xmin=_ax_xmin, xmax=_ax_xmax, ymin=_ax_ymin, ymax=_ax_ymax, padding_frac=0)

        # ── Map: dataset points ───────────────────────────────────────────────
        ax_pts = fig.add_subplot(bottom_gs[1])
        ax_pts.set_facecolor(sample_area_color)
        ax_pts.set_title(
            ax_titles.get('pts_map', "Dataset points (n={n})").format(
                col=col_label, k=k,
                thresh=round(cluster_threshold_value, sufficient_digits),
                r=r, n=len(pts),
            ),
            fontdict={'fontsize': 6},
        )
        if X is not None:
            ax_pts.imshow(X=X, interpolation='none', cmap=cmap_binary, extent=extent)
        plot_polygon(ax=ax_pts, poly=non_valid_area, facecolor=non_valid_area_color,
                     edgecolor='black', linewidth=sample_area_lw)
        sc_pts = ax_pts.scatter(
            x=_pts_x, y=_pts_y, c=pts_vals[:, i],
            s=s_map, marker='.', norm=norm, cmap=cmap_scatter, linewidths=0.3,
        )
        set_map_frame(ax=ax_pts, xmin=_ax_xmin, xmax=_ax_xmax, ymin=_ax_ymin, ymax=_ax_ymax, padding_frac=0)

        # ── Coordinate tick formatter ─────────────────────────────────────────
        # Pick scale per axis from actual data magnitude, not a fixed threshold.
        def _make_coord_fmt(vmin, vmax):
            span = abs(vmax - vmin) or 1.0
            _metric = getattr(grid, '_proj_is_metric', False) if grid is not None else False
            if _metric and span >= 1000:
                div, suffix = 1e3, ' km'
            elif _metric:
                div, suffix = 1.0, ' m'
            else:
                div, suffix = 1.0, ''
            return _FuncFormatter(lambda v, p, d=div, s=suffix: f'{round(v/d):,}{s}')
        _xfmt = _make_coord_fmt(map_xmin, map_xmax)
        _yfmt = _make_coord_fmt(map_ymin, map_ymax)
        for _ax in [ax_rnd, ax_pts]:
            _ax.xaxis.set_major_formatter(_xfmt)
            _ax.yaxis.set_major_formatter(_yfmt)

        # ── Axis label cleanup ────────────────────────────────────────────────
        if maps_side_by_side:
            # Both maps at same height — suppress redundant y-axis on right map.
            ax_pts.tick_params(labelleft=False)
            ax_pts.set_ylabel('')
        else:
            # Top map (rand) shares x-extent with bottom map (pts) — suppress x labels.
            ax_rnd.tick_params(labelbottom=False)
            ax_rnd.set_xlabel('')

        # ── Shared colorbar ───────────────────────────────────────────────────
        # 'both': grey triangle at bottom (zero), cluster colour at top (over-threshold).
        # Force a full draw so aspect='equal' axes have their final rendered positions.
        fig.canvas.draw()

        # aspect='equal' centers maps inside their GridSpec cell, leaving whitespace
        # above and below.  Shift both maps up so their tops sit just below the chart.
        if maps_side_by_side:
            _pos_dist = ax_dist.get_position()
            _pos_rnd  = ax_rnd.get_position()
            _pos_pts  = ax_pts.get_position()
            _gap = _pos_dist.y0 - max(_pos_rnd.y1, _pos_pts.y1)
            if _gap > 0.005:
                _shift = _gap * 0.5
                for _max in [ax_rnd, ax_pts]:
                    _p = _max.get_position()
                    _max.set_position([_p.x0, _p.y0 + _shift, _p.width, _p.height])
            fig.canvas.draw()

        if r is not None:
            draw_radius_indicator(
                fig, ax_pts, r,
                xmin=_ax_xmin, xmax=_ax_xmax, ymin=_ax_ymin, ymax=_ax_ymax,
                placement='y',
            )

        pos_rnd = ax_rnd.get_position()
        pos_pts = ax_pts.get_position()
        cbar_bottom = min(pos_rnd.y0, pos_pts.y0)
        cbar_top    = max(pos_rnd.y1, pos_pts.y1)
        cbar_left   = pos_pts.x1 + 0.01
        cbar_width_fig = colorbar_width * (pos_pts.x1 - pos_pts.x0)
        ax_cbar = fig.add_axes([cbar_left, cbar_bottom, cbar_width_fig, cbar_top - cbar_bottom])
        fig.colorbar(sc_pts, cax=ax_cbar, extend='both')

    if filename:
        fig.savefig(filename, **{'dpi': 300, 'bbox_inches': 'tight', **save_kwargs})
    if not show:
        _plt_close(fig)
    return fig

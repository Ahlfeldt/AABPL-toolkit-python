from numpy import zeros as _np_zeros
from matplotlib.pyplot import subplots as _plt_subplots
from matplotlib.pyplot import close as _plt_close
from matplotlib.colors import ListedColormap as _plt_ListedColormap
from matplotlib.patches import Patch as _plt_Patch
from matplotlib.lines import Line2D as _plt_Line2D
from aabpl.illustrations.plot_utils import set_map_frame, plot_polygon
from shapely.geometry import Polygon as _shapely_Polygon


def plot_sample_area(
        grid,
        filename:str='',
        show_grid_bounds:bool=False,
        show:bool=True,
        display_dpi:int=100,
        save_kwargs:dict={},
        **plot_kwargs,
):
    """
    Plot the sample area used for drawing null-distribution random points.

    Shows which grid cells are included in the sample area (white) versus
    excluded (blue), with the sample-area boundary drawn on top.

    Parameters
    ----------
    grid
        Grid object returned by ``radius_search`` / ``detect_cluster_pts``.
    filename : str
        Save path. Empty string skips saving (default ``''``).
    show_grid_bounds : bool
        Draw the full grid extent as a dashed rectangle (default ``False``).
    show : bool
        Display the figure (default ``True``).
    display_dpi : int
        Resolution for inline/screen display (default ``100``).
    save_kwargs : dict
        Forwarded to ``fig.savefig()``.
    **plot_kwargs
        figsize : tuple, default ``(10, 8)``
        fig, ax — existing Figure/Axes to draw into

    Returns
    -------
    matplotlib.figure.Figure
    """
    save_kwargs = {'dpi': 300, 'bbox_inches': 'tight', **save_kwargs}
    fig_in  = plot_kwargs.pop('fig', None)
    ax_in   = plot_kwargs.pop('ax', None)

    sa_bounds = grid.sample_area.bounds   # (xmin, ymin, xmax, ymax)
    xmin, ymin, xmax, ymax = sa_bounds
    w = xmax - xmin
    h = ymax - ymin
    # Use the longer dimension so padding is equal in both x and y (same absolute
    # distance regardless of aspect ratio, consistent with equal-aspect axes).
    span = max(w, h)
    pad2 = 0.02 * span
    ax_xmin, ax_xmax = xmin - pad2, xmax + pad2
    ax_ymin, ax_ymax = ymin - pad2, ymax + pad2
    # Outer rectangle for the non-valid-area polygon: slightly larger so it always
    # extends beyond the plot edge regardless of sample_area shape.
    pad3 = 0.03 * span
    outer_xmin, outer_xmax = xmin - pad3, xmax + pad3
    outer_ymin, outer_ymax = ymin - pad3, ymax + pad3

    if ax_in is None:
        fig_w = plot_kwargs.pop('figsize', (10,))[0] if 'figsize' in plot_kwargs else 10
        plot_kwargs.pop('figsize', None)
        fig_h = fig_w * h / w
        fig, ax = _plt_subplots(1, 1, figsize=(fig_w, fig_h), dpi=display_dpi)
    else:
        plot_kwargs.pop('figsize', None)
        fig, ax = fig_in, ax_in

    non_valid_area_color = '#bedbe6'
    sample_area_color    = '#ffffff'

    # ── cell raster ──────────────────────────────────────────────────────────
    _si  = grid._search_internals
    sgb  = grid.sample_grid_bounds
    col_min = int(round((sgb[0] - _si.bounds.xmin) / _si.spacing, 0))
    row_min = int(round((sgb[1] - _si.bounds.ymin) / _si.spacing, 0))
    col_max = int(round((sgb[2] - _si.bounds.xmin) / _si.spacing - 1, 0))
    row_max = int(round((sgb[3] - _si.bounds.ymin) / _si.spacing - 1, 0))
    n_rows  = row_max - row_min + 1
    n_cols  = col_max - col_min + 1

    cells = grid._search_internals.cells_rndm_sample
    X = _np_zeros((n_rows, n_cols), dtype=bool)
    if isinstance(cells, bool):
        X[:] = not cells
    else:
        X[:] = True
        for (row, col) in cells:
            ri = row_max - row
            ci = col - col_min
            if 0 <= ri < n_rows and 0 <= ci < n_cols:
                X[ri, ci] = False

    cmap_binary = _plt_ListedColormap([sample_area_color, non_valid_area_color])
    extent = [sgb[0], sgb[2], sgb[1], sgb[3]]
    ax.set_facecolor(sample_area_color)
    ax.imshow(X=X, interpolation='none', cmap=cmap_binary, extent=extent)

    # ── sample-area boundary ──────────────────────────────────────────────────
    # Outer rectangle is 3% larger than the sample area bounds so it always
    # extends past the plot edge; this makes the excluded fringe visible even
    # when the sample_area itself is a bounding box.
    non_valid_area = _shapely_Polygon([
        (outer_xmin, outer_ymin), (outer_xmax, outer_ymin),
        (outer_xmax, outer_ymax), (outer_xmin, outer_ymax),
    ]).difference(grid.sample_area)
    plot_polygon(ax=ax, poly=non_valid_area,
                 facecolor=non_valid_area_color, edgecolor='#4a8fa8', linewidth=0.6)

    # ── optional grid-extent rectangle ───────────────────────────────────────
    if show_grid_bounds:
        b = _si.bounds
        rect = _shapely_Polygon([
            (b.xmin, b.ymin), (b.xmax, b.ymin),
            (b.xmax, b.ymax), (b.xmin, b.ymax),
        ])
        plot_polygon(ax=ax, poly=rect, facecolor='none',
                     edgecolor='#555555', linewidth=0.8, linestyle='--')

    set_map_frame(ax=ax, xmin=ax_xmin, xmax=ax_xmax, ymin=ax_ymin, ymax=ax_ymax, padding_frac=0)

    # ── legend ───────────────────────────────────────────────────────────────
    legend_handles = [
        _plt_Patch(facecolor=sample_area_color,    edgecolor='#888', label='Sample area'),
        _plt_Patch(facecolor=non_valid_area_color, edgecolor='#4a8fa8', label='Excluded area'),
    ]
    if show_grid_bounds:
        legend_handles.append(
            _plt_Line2D([0], [0], color='#555555', linewidth=0.8, linestyle='--',
                        label='Grid bounds')
        )
    ax.legend(handles=legend_handles, loc='best', fontsize=7,
              handlelength=1.0, handleheight=0.9, borderpad=0.5)

    ax.set_title('Sample area', fontsize=9)

    fig.tight_layout()
    if filename:
        fig.savefig(filename, **save_kwargs)
    if not show:
        _plt_close(fig)
    return fig

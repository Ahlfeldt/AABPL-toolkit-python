from matplotlib.pyplot import (subplots as _plt_subplots)
from matplotlib.pyplot import get_cmap as _plt_get_cmap
from matplotlib.pyplot import close as _plt_close
from matplotlib.colors import LogNorm as _plt_LogNorm, Normalize as _plt_Normalize, LinearSegmentedColormap as _plt_LinearSegmentedColormap, ListedColormap as _plt_ListedColormap, BoundaryNorm as _plt_BoundaryNorm
from matplotlib.patches import Patch as _plt_Patch
from numpy import array as _np_array, zeros as _np_zeros
from aabpl.illustrations.plot_utils import truncate_colormap, set_map_frame, draw_radius_indicator, plot_polygon, format_col_title
from shapely.geometry import Polygon as _shapely_Polygon, MultiPoint as _shapely_MultiPoint

def handle_plot_kwargs(default_kwargs:dict={}, **kwargs):
    plot_kwargs = {
        **default_kwargs
    }
    sub_group_kwargs = ['fig','ax','hlines','vlines','figsize','set_aspect','set_xticks','set_yticks','set_xlim','set_ylim','suptitle']
    # groups of kwargs from which only the last one shall be applied
    kwargs_groups = (('cmap','color','c'),)
    kwarg_to_group = {}
    for grp in kwargs_groups:
        for k in grp:
            if k not in kwarg_to_group:
                kwarg_to_group[k] = [x for x in grp if x!=k]
            else:
                kwarg_to_group[k].extend([x for x in grp if x!=k])
            #
        #
    #
    # 'hlines':{'color':'red', 'linewidth':1},
    # 'vlines':{'color':'red', 'linewidth':1},
    for kwarg, v in kwargs.items():
        if kwarg in kwargs_groups:
            for k in kwargs_groups[kwarg]:
                plot_kwargs.pop(k,None)
        plot_kwargs[kwarg] = v
    used_sub_groups = [k for k in sub_group_kwargs if k in plot_kwargs]
    def pop_all_subgroups(used_sub_groups=used_sub_groups, plot_kwargs=plot_kwargs):
        for k in used_sub_groups:
            plot_kwargs.pop(k,None)
    # plot_kwargs['pop_all'] = pop_all_subgroups
    return plot_kwargs


def create_plots_for_vars(
        grid,
        colnames:_np_array,
        filename:str="",
        save_kwargs:dict={},
        plot_kwargs:dict={},
        show:bool=True,
        display_dpi:int=100,
):
    """
    Scatter plot of source points coloured by the value of one or more columns.

    Parameters
    ----------
    grid
        Grid object returned by ``radius_search`` / ``detect_cluster_pts``.
    colnames : np.ndarray
        Array of column names to plot. One subplot per entry.
    filename : str
        Save path. Empty string skips saving (default ``''``).
    save_kwargs : dict
        Forwarded to ``fig.savefig()``.
        Defaults: ``{'dpi': 300, 'bbox_inches': 'tight'}``.
    plot_kwargs : dict
        Supported keys:
        - ``figsize`` : tuple — figure size (default ``(10*n_cols, 8*n_rows)``)
        - ``cmap`` : str — colormap name (default ``'Reds'``)
        - ``s`` : float — scatter marker size (default auto)
        - ``vmin``, ``vmax`` : float — colormap range
        - ``sample_area_colors`` : list[str, str] — colours for [valid, excluded] area
        - ``sample_area_linewidth`` : float — border linewidth of the valid-area polygon (default ``0.2``)
        - ``fig``, ``axs`` — existing Figure/Axes to draw into
    show : bool
        Display the figure (default ``True``).
    display_dpi : int
        Resolution for inline/screen display (default ``100``).
        Use ``save_kwargs={'dpi': 300}`` to control saved-file resolution.

    Returns
    -------
    matplotlib.figure.Figure
    """
    nrows = colnames.shape[0]
    ncols = 1 if len(colnames.shape)==1 else colnames.shape[1]
    # Callers (e.g. grid.plot.vars / grid.plot.cluster_pts) forward **plot_kwargs;
    # extract control args that must not leak into ax.scatter().
    show = plot_kwargs.pop('show', show)
    display_dpi = plot_kwargs.pop('display_dpi', display_dpi)
    if 'save_kwargs' in plot_kwargs:
        save_kwargs = plot_kwargs.pop('save_kwargs')
    plot_kwargs.pop('filename', None)
    plot_kwargs.pop('colnames', None)
    # specify default plot kwargs and add defaults
    s = 1 if not 'figsize' in plot_kwargs else 0.1*plot_kwargs['figsize'][0]
    plot_kwargs = handle_plot_kwargs(default_kwargs={
        'fig': None,
        'axs': None,
        's': s,
        'cmap': 'Reds',
        'figsize': (10*ncols, 5*nrows),
        'additional_varnames':[],
        'sample_area_colors': ["#ffffff",  "#bedbe6"],
        'sample_area_linewidth': 0.2,
    }, **plot_kwargs)
    save_kwargs = {'dpi':300, 'bbox_inches':"tight", **save_kwargs}
    
    plot_kwargs.pop('color', None)
    figsize = plot_kwargs.pop('figsize')
    fig = plot_kwargs.pop('fig')
    axs = plot_kwargs.pop('axs')
    
    sample_area_color, non_valid_area_color = plot_kwargs.pop('sample_area_colors')
    sample_area_lw = plot_kwargs.pop('sample_area_linewidth')
    
    additional_varnames = plot_kwargs.pop('additional_varnames')
    if len(additional_varnames)>nrows:
        # TODO this is not compelted
        nrows = len(additional_varnames)
    cmap_name = plot_kwargs.pop('cmap', 'Reds')
    cmap = truncate_colormap(cmap_name if hasattr(cmap_name, 'N') else _plt_get_cmap(cmap_name), 0.1, 1)
    if fig is None or axs is None:
        fig, axs = _plt_subplots(nrows, ncols, figsize=figsize, dpi=display_dpi,
                                  constrained_layout=True,
                                  sharex=(ncols == 1 and nrows > 1),
                                  sharey=(nrows == 1 and ncols > 1))
    if not grid.study_grid_bounds is None and not grid.study_area is None:
        _sa_xmin, _sa_ymin, _sa_xmax, _sa_ymax = grid.study_area.bounds
        non_valid_area = _shapely_Polygon([
            (_sa_xmin, _sa_ymin), (_sa_xmax, _sa_ymin),
            (_sa_xmax, _sa_ymax), (_sa_xmin, _sa_ymax),
        ]).difference(grid.study_area)
        xmin, xmax, ymin, ymax = grid._search_internals.bounds.xmin, grid._search_internals.bounds.xmax, grid._search_internals.bounds.ymin, grid._search_internals.bounds.ymax,
    else:    
        xmin, xmax, ymin, ymax = grid._search_internals.bounds.xmin, grid._search_internals.bounds.xmax, grid._search_internals.bounds.ymin, grid._search_internals.bounds.ymax,
    _sc = grid._search_class
    x_col, y_col = _sc.source.x, _sc.source.y
    src = _sc.source
    xs = src.pts[x_col]
    ys = src.pts[y_col]
    _col_meta = getattr(grid, '_aabpl_col_meta', {})
    _axes_with_r = []  # list of (ax, r) pairs — radius indicator drawn on every axis
    for i, colname in enumerate(colnames.flat):
        # SELECT AX (IF MULTIPLE)
        ax = axs.flat[i] if nrows > 1 else axs
        row_idx = i // ncols
        col_idx = i % ncols

        # SET TITLE
        ax.set_title(format_col_title(colname, _col_meta.get(colname, {})))
        # suppress redundant axis labels on shared axes
        if ncols == 1 and nrows > 1 and row_idx < nrows - 1:
            ax.set_xlabel('')
            ax.tick_params(labelbottom=False)
        if nrows == 1 and ncols > 1 and col_idx > 0:
            ax.set_ylabel('')
            ax.tick_params(labelleft=False)
        # CPOLOR NON SAMPLE AREA
        if not grid.study_grid_bounds is None and not grid.study_area is None:
            ax.set_facecolor(sample_area_color)
            cells_rndm_sample = grid._search_internals.cells_rndm_sample
            _si = grid._search_internals
            col_min = int(round((grid.study_grid_bounds[0]-_si.bounds.xmin)/_si.spacing,0))
            row_min = int(round((grid.study_grid_bounds[1]-_si.bounds.ymin)/_si.spacing,0))
            col_max = int(round((grid.study_grid_bounds[2]-_si.bounds.xmin)/_si.spacing-1,0))
            row_max = int(round((grid.study_grid_bounds[3]-_si.bounds.ymin)/_si.spacing-1,0))
            # row_min, row_max = min([row for row,col in grid.cells_rndm_sample]), max([row for row,col in grid.cells_rndm_sample])
            # col_min, col_max = min([col for row,col in grid.cells_rndm_sample]), max([col for row,col in grid.cells_rndm_sample])
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
            extent = [grid.study_grid_bounds[0],grid.study_grid_bounds[2],grid.study_grid_bounds[1],grid.study_grid_bounds[3]]
            p = ax.imshow(X=X, interpolation='none', cmap=cmap_binary, extent=extent)#, To-Do the extent is imprecise as it does not cover the full grid only its points
            non_valid_patch = _plt_Patch(facecolor=non_valid_area_color, label='Non-valid area', edgecolor=non_valid_area_color)
            sample_patch = _plt_Patch(facecolor=sample_area_color, label='Sample area', edgecolor=sample_area_color)
            ax.legend(handles=[non_valid_patch, sample_patch], loc='best',
                      handlelength=0.8, handleheight=0.8, fontsize=8, borderpad=0.5)
            plot_polygon(poly=non_valid_area, ax=ax, facecolor=non_valid_area_color, edgecolor='black', linewidth=sample_area_lw)
        # ADD DISTRIBUTION PLOT
        c = src.pts[colname]
        scatter_kwargs = {k: v for k, v in plot_kwargs.items() if k not in ('vmin', 'vmax', 'norm')}

        # Detect discrete integer columns (e.g. cluster IDs): ≤20 unique integer values
        _unique_vals = c.dropna().unique()
        _is_discrete = (
            len(_unique_vals) <= 20 and
            all(float(v) == int(v) for v in _unique_vals if v == v)
        )

        if _is_discrete:
            _sorted = sorted(int(v) for v in _unique_vals)
            _n = len(_sorted)
            _qual_cmap = _plt_get_cmap('tab10' if _n <= 10 else 'tab20')
            _colors = ["#cccccc"] + [_qual_cmap(i / max(_n - 2, 1)) for i in range(_n - 1)]
            _disc_cmap = _plt_ListedColormap(_colors)

            _bounds = [v - 0.5 for v in _sorted] + [_sorted[-1] + 0.5]
            _disc_norm = _plt_BoundaryNorm(_bounds, _disc_cmap.N)
            scttr = ax.scatter(x=xs, y=ys, c=c, norm=_disc_norm, cmap=_disc_cmap, rasterized=True, linewidths=0.3, **scatter_kwargs)
            cbar = fig.colorbar(scttr, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_ticks(_sorted)
            cbar.set_ticklabels([str(v) for v in _sorted])
        else:
            vmin = plot_kwargs.get('vmin', c.min())
            vmax = plot_kwargs.get('vmax', c.max())
            norm = plot_kwargs.get('norm', _plt_LogNorm(vmin=vmin, vmax=vmax) if vmin > 0 else _plt_Normalize(vmin=vmin, vmax=vmax))
            scttr = ax.scatter(x=xs, y=ys, c=c, norm=norm, cmap=cmap, rasterized=True, linewidths=0.3, **scatter_kwargs)
            fig.colorbar(scttr, ax=ax, fraction=0.046, pad=0.04)
        plot_polygon(poly=non_valid_area, ax=ax, facecolor="none", edgecolor='black', linewidth=sample_area_lw)
        set_map_frame(ax=ax, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)
        _col_r = _col_meta.get(colname, {}).get('r', None)
        if _col_r is not None:
            _axes_with_r.append((ax, _col_r))

    if not fig is None:
        if _axes_with_r:
            fig.canvas.draw()
            for _indicator_ax, _indicator_r in _axes_with_r:
                draw_radius_indicator(fig, _indicator_ax, _indicator_r, xmin, xmax, ymin, ymax,
                                      placement='y')
        if filename:
            fig.savefig(filename, **save_kwargs)
        if not show:
            _plt_close(fig)
    return fig
    #
#
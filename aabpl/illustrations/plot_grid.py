from numpy import (
    array as _np_array,
    zeros as _np_zeros,
    unique as _np_unique,
    spacing as _np_spacing,
    linspace as _np_linspace,
)
from matplotlib.pyplot import subplots as _plt_subplots, colorbar as _plt_colorbar
from matplotlib.pyplot import get_cmap as _plt_get_cmap
from matplotlib.pyplot import close as _plt_close
from matplotlib.patches import (Rectangle as _plt_Rectangle, Polygon as _plt_Polygon, Circle as _plt_Circle)
from matplotlib.collections import PatchCollection as _plt_PatchCollection
from matplotlib.colors import LogNorm as _plt_LogNorm, Normalize as _plt_Normalize
import matplotlib.colors as _plt_colors
from mpl_toolkits.axes_grid1 import make_axes_locatable as _make_axes_locatable
from aabpl.illustrations.plot_utils import truncate_colormap, map_2D_to_rgb, get_2D_rgb_colobar_kwargs, add_color_bar_ax, set_map_frame, draw_radius_indicator, format_col_title
from aabpl.illustrations.plot_pt_vars import create_plots_for_vars

class GridPlots(object):
    """
    Plot methods attached to a Grid object after running ``radius_search``,
    ``detect_cluster_pts``, or ``detect_cluster_cells``.

    Access via ``grid.plot.<method>()``.

    All methods share these common parameters
    ----------------------------------------
    filename : str
        File path to save the figure (e.g. ``'output/my_plot.png'``).
        Leave empty (default ``''``) to skip saving.
    show : bool
        Whether to display the figure inline (Jupyter) or on screen.
        Set ``False`` to save without showing (default ``True``).
    fig / ax / axs
        Pass an existing matplotlib Figure and Axes to draw into them
        instead of creating new ones.
    display_dpi : int
        Resolution used when rendering the figure on screen / in Jupyter
        (default ``100``). Lower values render faster. Has no effect when
        ``show=False``.
    save_kwargs : dict
        Kwargs forwarded to ``fig.savefig()``.
        Defaults: ``{'dpi': 300, 'bbox_inches': 'tight'}``.
        Override dpi for saved files independently:
        ``save_kwargs={'dpi': 150, 'format': 'svg'}``.
    **plot_kwargs
        Additional keyword arguments forwarded to the underlying
        matplotlib call. Supported keys vary per method — see each
        method's docstring.

    Methods
    -------
    clusters       : Detected cluster polygons overlaid on aggregated cell values.
    cell_aggregates: Aggregated indicator values per grid cell (heatmap).
    cluster_vars   : Scatter of source points coloured by aggregated value.
    rand_dist      : Comparison of random vs. observed radius-sum distribution.
    cluster_pts    : Source points coloured by cluster membership.
    study_area     : Study area used for null-distribution random points.
    grid_ids       : Grid cell row/col indices and point counts (diagnostic).

    Examples
    --------
    fig = grid.plot.clusters(filename='output/clusters.png')
    fig = grid.plot.clusters(filename='output/clusters.png', show=False)
    fig = grid.plot.clusters(figsize=(20, 10), cmap='Blues')
    """
    def __init__(self, grid):
        self.grid = grid

    def study_area(self, filename:str='', show:bool=True, display_dpi:int=100,
                    save_kwargs:dict={}, show_grid_bounds:bool=False, **plot_kwargs):
        """
        Plot the study area used for drawing null-distribution random points.

        Parameters
        ----------
        filename : str
            Save path. Empty string skips saving (default ``''``).
        show : bool
            Display the figure (default ``True``).
        display_dpi : int
            Resolution for inline/screen display (default ``100``).
        save_kwargs : dict
            Forwarded to ``fig.savefig()``.
        show_grid_bounds : bool
            Draw the full grid extent as a dashed rectangle (default ``False``).
        **plot_kwargs
            figsize : tuple, default ``(10, 8)``
            s : float — scatter marker size
            fig, ax — existing Figure/Axes to draw into

        Returns
        -------
        matplotlib.figure.Figure
        """
        from aabpl.illustrations.plot_study_area import plot_study_area as _plot_sa
        return _plot_sa(
            grid=self.grid,
            filename=filename, show=show, display_dpi=display_dpi,
            save_kwargs=save_kwargs, show_grid_bounds=show_grid_bounds,
            **plot_kwargs,
        )

    # Backward-compat alias
    sample_area = study_area

    def vars(self, colnames=None, filename='', show:bool=True, display_dpi:int=100, save_kwargs:dict={}, **plot_kwargs):
        """
        Scatter plot of source points coloured by their radius-sum value.
        One subplot per indicator column.

        Parameters
        ----------
        colnames : array-like, optional
            Column names to plot. Defaults to all indicator columns in ``c``.
        filename : str
            Save path. Empty string skips saving (default ``''``).
        show : bool
            Display the figure (default ``True``).
        display_dpi : int
            Resolution for inline/screen display (default ``100``).
        save_kwargs : dict
            Forwarded to ``fig.savefig()``.
        **plot_kwargs
            figsize : tuple
                Figure size (default ``(10 * n_cols, 5 * n_rows)``).
            cmap : str or Colormap
                Colormap for point values (default ``'Reds'``).
            s : float
                Scatter marker size (default auto-scaled to figsize).
            vmin, vmax : float
                Colormap range (default data min/max).
            study_area_colors : list[str, str]
                Colours for [valid area, excluded area]
                (default ``['#ffffff', '#bedbe6']``).
            study_area_linewidth : float
                Border linewidth of the valid-area polygon (default ``0.2``).

        Returns
        -------
        matplotlib.figure.Figure

        Examples
        --------
        fig = grid.plot.vars()
        fig = grid.plot.vars(cmap='viridis', s=2, figsize=(14, 10))
        fig = grid.plot.vars(filename='vars.png', show=False)
        """
        from aabpl.illustrations.plot_pt_vars import create_plots_for_vars
        import numpy as np
        if colnames is None:
            _sc = self.grid._search_class
            colnames = np.array(_sc.target.c)
        return create_plots_for_vars(grid=self.grid, colnames=colnames, filename=filename,
                                     show=show, display_dpi=display_dpi, save_kwargs=save_kwargs,
                                     plot_kwargs=plot_kwargs)

    def cell_aggregates(
        self,
        filename:str='',
        fig=None,
        axs=None,
        show:bool=True,
        display_dpi:int=100,
        save_kwargs:dict={},
        **plot_kwargs
    ):
        """
        Plot the aggregated indicator value per grid cell as a heatmap.
        One subplot is created per column in ``c``.

        Parameters
        ----------
        filename : str
            Save path. Empty string skips saving (default ``''``).
        show : bool
            Display the figure (default ``True``).
        display_dpi : int
            Resolution for inline/screen display (default ``100``).
            Use ``save_kwargs={'dpi': 300}`` to control saved-file resolution.
        fig, axs
            Existing Figure / Axes array to draw into.
        save_kwargs : dict
            Forwarded to ``fig.savefig()``.
        **plot_kwargs
            figsize : tuple
                Figure size (default ``(12, 10)``).
            cmap : str or Colormap
                Colormap for cell values (default ``'Reds'``). Accepts any
                matplotlib colormap name or object.

        Returns
        -------
        matplotlib.figure.Figure

        Examples
        --------
        fig = grid.plot.cell_aggregates()
        fig = grid.plot.cell_aggregates(cmap='Blues', figsize=(14, 10))
        fig = grid.plot.cell_aggregates(filename='cells.png', show=False)
        """
        save_kwargs = {'dpi': 300, 'bbox_inches': 'tight', **save_kwargs}
        figsize = plot_kwargs.pop('figsize', (12, 10))
        cmap_arg = plot_kwargs.pop('cmap', 'Reds')
        _sc = self.grid._search_class
        if axs is None:
            fig, axs = _plt_subplots(ncols=len(_sc.target.c), figsize=figsize, dpi=display_dpi)

        # Output grid + cached per-output-cell aggregates are built lazily here.
        self.grid.update_spacing()
        # Draw one coloured square per NON-EMPTY output cell (from the cached
        # grid.cell_aggregates) as a PatchCollection — no dense full-grid array,
        # so memory scales with populated cells, not the bounding-box area. The axes
        # still span the full output extent, so the plot looks the same as a raster.
        sx, sy = self.grid.cell_size, self.grid.cell_size_y
        ox, oy = self.grid.x_steps_bounds[0], self.grid.y_steps_bounds[0]
        extent = [self.grid.x_steps_bounds[0], self.grid.x_steps_bounds[-1],
                  self.grid.y_steps_bounds[-1], self.grid.y_steps_bounds[0]]
        out_sums = self.grid.cell_aggregates
        for i,column in enumerate(_sc.target.c):
            ax = axs if len(_sc.target.c)==1 else axs.flat[i]
            rects, vals = [], []
            for (row, col), v in out_sums.items():
                val = v[i]
                if val == 0:
                    continue
                rects.append(_plt_Rectangle((ox + int(col) * sx, oy + int(row) * sy), sx, sy))
                vals.append(val)
            if not vals:
                continue
            vals = _np_array(vals)
            vmin = vals[vals != 0].min()
            vmax = vals.max()
            norm = _plt_LogNorm(vmin=vmin,vmax=vmax,clip=False) if vmin>0 else _plt_Normalize(vmin=vmin,vmax=vmax,clip=False)
            cmap = truncate_colormap(cmap_arg if hasattr(cmap_arg, 'N') else _plt_get_cmap(cmap_arg), 0.3, 1)
            cmap.set_under('#ffffff00')
            cmap.set_bad('#ffffff00')
            pc = _plt_PatchCollection(rects, cmap=cmap, norm=norm)
            pc.set_array(vals)
            ax.add_collection(pc)
            cb = _plt_colorbar(pc, cax=add_color_bar_ax(fig,ax))
            ax.set_xlabel('x/lon')
            ax.set_ylabel('y/lat')
            _col_meta_entry = getattr(self.grid, '_aabpl_col_meta', {}).get(str(column), {})
            ax.title.set_text(format_col_title(str(column), _col_meta_entry))
            _col_meta  = getattr(self.grid, '_aabpl_col_meta', {})
            _r_agg     = (_col_meta.get(str(column), {}) or {}).get('r', None)
            _xmin_agg, _xmax_agg = extent[0], extent[1]
            _ymin_agg, _ymax_agg = extent[3], extent[2]
            set_map_frame(ax=ax, xmin=_xmin_agg, xmax=_xmax_agg, ymin=_ymin_agg, ymax=_ymax_agg)
        if not fig is None:
            if filename:
                if _r_agg is not None:
                    fig.canvas.draw()
                    draw_radius_indicator(fig, ax, _r_agg, _xmin_agg, _xmax_agg, _ymin_agg, _ymax_agg)
                fig.savefig(filename, **save_kwargs)
            if not show:
                _plt_close(fig)
        return fig

    #

    def rand_dist(self, filename='', show=True, display_dpi=100, save_kwargs={}, **plot_kwargs):
        """
        Observed radius-sum distribution vs. null distribution (cumulative plot).

        Requires ``detect_cluster_pts`` to have been run on this grid first.
        The threshold line shows the k-th percentile of the null distribution.
        """
        if not hasattr(self.grid, '_cluster_result'):
            raise RuntimeError(
                "grid.plot.rand_dist() is only available on a grid returned by "
                "detect_cluster_pts() or detect_cluster_cells(). "
                "The grid you are using was created with radius_search(), which does not "
                "run the null distribution or cluster detection steps."
            )
        from aabpl.illustrations.distribution_plot import create_distribution_plot
        result = self.grid._cluster_result
        sc = self.grid._search_class.source
        create_distribution_plot(
            filename=filename, plot_kwargs=plot_kwargs,
            pts=sc.pts, x=sc.x, y=sc.y,
            radius_sum_columns=result['aggregate_cols'],
            grid=self.grid,
            rndm_pts=self.grid.null_distribution,
            cluster_threshold_values=list(result['thresholds'].values()),
            k_th_percentile=result['k_th_percentiles'],
            r=result['display_radius'],
            show=show, display_dpi=display_dpi, save_kwargs=save_kwargs,
        )

    def cluster_pts(self, filename='', show=True, display_dpi=100, save_kwargs={}, **plot_kwargs):
        """
        Source points coloured by cluster membership and radius-sum value.

        Requires ``detect_cluster_pts`` to have been run on this grid first.
        """
        if not hasattr(self.grid, '_cluster_result'):
            raise RuntimeError(
                "grid.plot.cluster_pts() is only available on a grid returned by "
                "detect_cluster_pts() or detect_cluster_cells(). "
                "The grid you are using was created with radius_search(), which does not "
                "run the null distribution or cluster detection steps."
            )
        from aabpl.illustrations.plot_pt_vars import create_plots_for_vars
        result = self.grid._cluster_result
        return create_plots_for_vars(
            grid=self.grid,
            colnames=result['plot_colnames'],
            filename=filename, show=show, display_dpi=display_dpi,
            save_kwargs=save_kwargs, plot_kwargs=plot_kwargs,
        )

    def clusters(self, filename:str='', fig=None, axs=None, show:bool=True, display_dpi:int=100, save_kwargs:dict={}, cluster_columns=None, **plot_kwargs):
        """
        Plot detected cluster polygons overlaid on aggregated cell values.
        One subplot per cluster column (one per radius/band for multi-radius runs).
        Cluster outlines are drawn in red with hatching, labelled by cluster id.

        Parameters
        ----------
        filename : str
            Save path. Empty string skips saving (default ``''``).
        show : bool
            Display the figure (default ``True``).
        display_dpi : int
            Resolution for inline/screen display (default ``100``).
            Use ``save_kwargs={'dpi': 300}`` to control saved-file resolution.
        fig, axs
            Existing Figure / Axes array to draw into.
        save_kwargs : dict
            Forwarded to ``fig.savefig()``.
        cluster_columns : list[str] or None
            Subset of cluster column names to plot.  ``None`` (default) plots all
            cluster columns, which gives one subplot per radius/band.
        **plot_kwargs
            figsize : tuple, default ``(10, 10 * n_cluster_cols)``
            cmap : str or Colormap, default ``'binary'``
            cluster_color : str, default ``'red'``
                Edge colour of the cluster polygon outline.
            cluster_hatch : str, default ``'////'``
                Hatch pattern for the cluster polygon fill.
                Any matplotlib hatch string e.g. ``'////'``, ``'xxxx'``, ``'....'``, ``''``.
            cluster_alpha : float, default ``0.0``
                Face alpha of the cluster polygon (0 = transparent).

        Returns
        -------
        matplotlib.figure.Figure

        Examples
        --------
        fig = grid.plot.clusters()
        fig = grid.plot.clusters(filename='clusters.png', show=False)
        fig = grid.plot.clusters(figsize=(20, 15), cmap='Greys')
        fig = grid.plot.clusters(cluster_color='#2255cc', cluster_hatch='xxxx')
        fig = grid.plot.clusters(cluster_columns=['employment_sum_10000_cluster'])
        """
        if len(self.grid.clustering.by_column) == 0:
            print("No clustering performed. Run detect_cell_clusters or grid.create_clusters first.")
            return
        save_kwargs = {'dpi': 300, 'bbox_inches': 'tight', **save_kwargs}
        _sc2 = self.grid._search_class

        # Determine which cluster columns to plot
        all_cluster_cols = list(self.grid.clustering.by_column.keys())
        if cluster_columns is None:
            columns_to_plot = all_cluster_cols
        else:
            columns_to_plot = [c for c in cluster_columns if c in self.grid.clustering.by_column]
        n = len(columns_to_plot)
        if n == 0:
            print("No matching cluster columns to plot.")
            return

        # Build a map from cluster column → index in cell_aggregates (= index in target.c).
        # Multi-radius uses _cluster_col_map {cluster_col: orig_col}; single-radius
        # can be derived from _aabpl_col_meta which stores the originating column.
        input_cols = list(_sc2.target.c) if hasattr(_sc2, 'target') else []
        input_col_to_agg_index = {col: idx for idx, col in enumerate(input_cols)}
        col_meta = getattr(self.grid, '_aabpl_col_meta', {})
        cluster_col_map = getattr(self.grid, '_cluster_col_map', {})

        def _agg_index_for_cluster_col(cluster_col):
            # Try _cluster_col_map first (multi-radius path)
            if cluster_col in cluster_col_map:
                orig = cluster_col_map[cluster_col]
                return input_col_to_agg_index.get(orig, 0)
            # Fallback: look in _aabpl_col_meta
            orig = col_meta.get(cluster_col, {}).get('c', None)
            if orig is not None:
                return input_col_to_agg_index.get(orig, 0)
            # Last resort: 0
            return 0

        figsize       = plot_kwargs.pop('figsize', (10, 5 * n))
        cmap_arg      = plot_kwargs.pop('cmap', 'binary')
        cluster_color = plot_kwargs.pop('cluster_color', 'red')
        cluster_hatch = plot_kwargs.pop('cluster_hatch', '////')
        cluster_alpha = plot_kwargs.pop('cluster_alpha', 0.0)
        if axs is None:
            fig, axs = _plt_subplots(nrows=n, ncols=1, figsize=figsize, dpi=display_dpi,
                                     constrained_layout=True,
                                     sharex=(n > 1))

        # Output grid + cached per-output-cell aggregates built lazily here; the
        # background matches grid.plot.cell_aggregates. Cluster outlines are drawn in
        # coordinate space below and are unaffected by the grid choice.
        self.grid.update_spacing()
        sx, sy = self.grid.cell_size, self.grid.cell_size_y
        ox, oy = self.grid.x_steps_bounds[0], self.grid.y_steps_bounds[0]
        out_sums = self.grid.cell_aggregates

        _xmin_cl = self.grid._search_internals.x_steps.min()
        _xmax_cl = self.grid._search_internals.x_steps.max()
        _ymin_cl = self.grid._search_internals.y_steps.min()
        _ymax_cl = self.grid._search_internals.y_steps.max()

        axes_with_r = []  # (ax, r_spec) pairs for radius indicator

        for i, cluster_column in enumerate(columns_to_plot):
            clusters_for_column = self.grid.clustering.by_column[cluster_column]
            ax = axs.flat[i] if n > 1 else axs
            ax.set_xlabel('x/lon ' + str(self.grid.proj_crs))
            ax.set_ylabel('y/lat ' + str(self.grid.proj_crs))
            clusters = clusters_for_column.clusters
            n_clusters = len(clusters)
            cluster_label = format_col_title(
                cluster_column,
                col_meta.get(cluster_column, {}),
            )
            # format_col_title already prepends 'Clusters - '; strip and re-add count
            cluster_label_base = cluster_label.replace('Clusters - ', '', 1)
            plural = 's' if n_clusters != 1 else ''
            ax.title.set_text(
                str(n_clusters) + ' cluster' + plural + ' - ' + cluster_label_base
            )

            # cell background: use the aggregated values for this cluster column's
            # originating input column (not the enumerate index i)
            agg_val_idx = _agg_index_for_cluster_col(cluster_column)
            rects, vals = [], []
            for (row, col), v in out_sums.items():
                val = v[agg_val_idx] if agg_val_idx < len(v) else 0
                if val == 0:
                    continue
                rects.append(_plt_Rectangle((ox + int(col) * sx, oy + int(row) * sy), sx, sy))
                vals.append(val)
            cmap = cmap_arg if hasattr(cmap_arg, 'N') else _plt_get_cmap(cmap_arg)
            cmap = truncate_colormap(cmap, 0.1, 1)
            cmap.set_under('#fff0')
            if vals:
                vals = _np_array(vals)
                vmin, vmax = vals[vals != 0].min(), vals.max()
                norm = (_plt_LogNorm(vmin=vmin, vmax=vmax, clip=False) if vmin > 0
                        else _plt_Normalize(vmin=vmin, vmax=vmax, clip=False))
                pc = _plt_PatchCollection(rects, cmap=cmap, norm=norm)
                pc.set_array(vals)
                ax.add_collection(pc)
            for cluster in clusters:
                geoms = ([cluster.geometry] if hasattr(cluster.geometry, 'exterior')
                         else cluster.geometry.geoms)
                for geom in geoms:
                    ax.add_patch(_plt_Polygon(
                        xy=geom.exterior.coords,
                        hatch=cluster_hatch,
                        facecolor=(*_plt_colors.to_rgba(cluster_color)[:3], cluster_alpha),
                        edgecolor=cluster_color,
                    ))
                ax.annotate(cluster.id, xy=cluster.centroid, fontsize=15,
                            weight='bold', color=cluster_color)

            set_map_frame(ax=ax, xmin=_xmin_cl, xmax=_xmax_cl, ymin=_ymin_cl, ymax=_ymax_cl)
            if len(vals) > 0:
                _divider = _make_axes_locatable(ax)
                _cax = _divider.append_axes("right", size="3%", pad=0.05)
                _plt_colorbar(pc, cax=_cax)

            # Collect per-axis radius: prefer column-level meta, fall back to search class
            r_for_ax = col_meta.get(cluster_column, {}).get('r', None)
            if r_for_ax is None:
                r_for_ax = getattr(getattr(self.grid, '_search_class', None), 'r', None)
            if r_for_ax is not None:
                axes_with_r.append((ax, r_for_ax))

        if fig is not None:
            if axes_with_r:
                fig.canvas.draw()
                for _indicator_ax, _indicator_r in axes_with_r:
                    draw_radius_indicator(fig, _indicator_ax, _indicator_r,
                                          _xmin_cl, _xmax_cl, _ymin_cl, _ymax_cl,
                                          placement='y')
            if filename:
                fig.savefig(filename, **save_kwargs)
            if not show:
                _plt_close(fig)
        return fig

    def cluster_vars(
            self,
            filename:str='',
            save_kwargs:dict={},
            show:bool=True,
            display_dpi:int=100,
            **plot_kwargs,
        ):
        """
        Scatter plot of source points coloured by their aggregated radius-sum
        value, alongside the cluster membership column. One subplot per
        indicator column.

        Parameters
        ----------
        filename : str
            Save path. Empty string skips saving (default ``''``).
        show : bool
            Display the figure (default ``True``).
        display_dpi : int
            Resolution for inline/screen display (default ``100``).
            Use ``save_kwargs={'dpi': 300}`` to control saved-file resolution.
        save_kwargs : dict
            Forwarded to ``fig.savefig()``.
        **plot_kwargs
            figsize : tuple
                Figure size (default ``(10 * n_cols, 8 * n_rows)``).
            cmap : str or Colormap
                Colormap for point values (default ``'Reds'``).
            s : float
                Scatter marker size (default auto-scaled to figsize).
            vmin, vmax : float
                Colormap range (default data min/max).
            study_area_colors : list[str, str]
                Colours for [valid area, excluded area]
                (default ``['#ffffff', '#bedbe6']``).
            study_area_linewidth : float
                Border linewidth of the valid-area polygon (default ``0.5``).

        Returns
        -------
        matplotlib.figure.Figure

        Examples
        --------
        fig = grid.plot.cluster_vars()
        fig = grid.plot.cluster_vars(filename='vars.png', show=False, cmap='viridis')
        fig = grid.plot.cluster_vars(figsize=(20, 10), s=2)
        """
        return create_plots_for_vars(
            grid=self.grid,
            colnames=_np_array([(_sc3 := (self.grid._search_class)).target.c, _sc3.source.aggregate_columns]),
            filename=filename,
            show=show,
            save_kwargs=save_kwargs,
            plot_kwargs=plot_kwargs,
            display_dpi=display_dpi,
        )
    #

    def grid_ids(self, fig=None, ax=None, filename:str='', show:bool=True, display_dpi:int=100, save_kwargs:dict={}, **plot_kwargs):
        """
        Diagnostic plot showing grid cell coordinates, row/column indices,
        and point counts per cell (three side-by-side panels).

        Parameters
        ----------
        filename : str
            Save path. Empty string skips saving (default ``''``).
        show : bool
            Display the figure (default ``True``).
        display_dpi : int
            Resolution for inline/screen display (default ``100``).
            Use ``save_kwargs={'dpi': 300}`` to control saved-file resolution.
        fig, ax
            Existing Figure / Axes to draw into.
        save_kwargs : dict
            Forwarded to ``fig.savefig()``.
        **plot_kwargs
            figsize : tuple, default ``(15, 10)``

        Returns
        -------
        matplotlib.figure.Figure
        """
        save_kwargs = {'dpi': 300, 'bbox_inches': 'tight', **save_kwargs}
        figsize = plot_kwargs.pop('figsize', (15, 10))
        if ax is None:
            fig, ax = _plt_subplots(ncols=3, figsize=figsize, dpi=display_dpi)
        imshow_kwargs = {
            'xmin':self.grid._search_internals.x_steps.min(),
            'ymin':self.grid._search_internals.y_steps.min(),
            'xmax':self.grid._search_internals.x_steps.max(),
            'ymax':self.grid._search_internals.y_steps.max(),
        }
        extent=[imshow_kwargs['xmin'],imshow_kwargs['xmax'],imshow_kwargs['ymax'],imshow_kwargs['ymin']]
        X = _np_array([[map_2D_to_rgb(x,y, **imshow_kwargs) for x in  self.grid._search_internals.x_steps[:-1]] for y in reversed(self.grid._search_internals.y_steps[:-1])])
        # ax.flat[0].imshow(X=X, interpolation='none', extent=extent)
        # ax.flat[0].pcolormesh([self.grid._search_internals.x_steps, self.grid._search_internals.y_steps], X)
        # ax.flat[0].pcolormesh(X, edgecolor="black", linewidth=.1/max([len(self.grid._search_internals.col_ids), len(self.grid._search_internals.row_ids)]))
        ax.flat[0].imshow(X=X, interpolation='none', extent=extent)
        # ax.flat[0].set_aspect(2)
        colorbar_kwargs = get_2D_rgb_colobar_kwargs(**imshow_kwargs)
        cb = _plt_colorbar(**colorbar_kwargs[2], ax=ax.flat[0])
        cb.ax.set_xlabel("diagonal")
        cb = _plt_colorbar(**colorbar_kwargs[0], ax=ax.flat[0])
        cb.ax.set_xlabel("x/lon")
        cb = _plt_colorbar(**colorbar_kwargs[1], ax=ax.flat[0])
        cb.ax.set_xlabel("y/lat") 
        ax.flat[0].set_xlabel('x/lon') 
        ax.flat[0].set_ylabel('y/lat') 
        ax.flat[0].title.set_text("Grid lat / lon coordinates")

        imshow_kwargs = {
            'xmin':self.grid._search_internals.col_ids.min(),
            'ymin':self.grid._search_internals.row_ids.min(),
            'xmax':self.grid._search_internals.col_ids.max(),
            'ymax':self.grid._search_internals.row_ids.max(),
        }
        extent=[imshow_kwargs['xmin'],imshow_kwargs['xmax'],imshow_kwargs['ymax'],imshow_kwargs['ymin']]

        X = _np_array([[map_2D_to_rgb(x,y, **imshow_kwargs) for x in  self.grid._search_internals.col_ids] for y in reversed(self.grid._search_internals.row_ids)])
        ax.flat[1].imshow(X=X, interpolation='none', extent=extent)
        # ax.flat[1].set_aspect(2)
        colorbar_kwargs = get_2D_rgb_colobar_kwargs(**imshow_kwargs)
        # cb = _plt_colorbar(**colorbar_kwargs[2], ax=ax.flat[1])
        cb = _plt_colorbar(**colorbar_kwargs[2], cax=add_color_bar_ax(fig,ax.flat[1]))
        cb.ax.set_xlabel("diagonal")
        # cb = _plt_colorbar(**colorbar_kwargs[0], ax=ax.flat[1])
        cb = _plt_colorbar(**colorbar_kwargs[0], cax=add_color_bar_ax(fig,ax.flat[1]))
        cb.ax.set_xlabel("col nr")
        cb = _plt_colorbar(**colorbar_kwargs[1], ax=ax.flat[1])
        cb = _plt_colorbar(**colorbar_kwargs[1], cax=add_color_bar_ax(fig,ax.flat[1]))
        cb.ax.set_xlabel("row nr") 
        ax.flat[1].set_xlabel('row nr') 
        ax.flat[1].set_ylabel('col nr') 
        ax.flat[1].title.set_text("Grid row / col indices")
        
        from aabpl.search.point_assignment import cell_count as _cell_count
        X = _np_array([[_cell_count(self.grid, row_id, col_id) for col_id in self.grid._search_internals.col_ids] for row_id in reversed(self.grid._search_internals.row_ids)])
        # p = ax.flat[2].pcolormesh(X, cmap='Reds')
        p = ax.flat[2].imshow(X=X, interpolation='none', extent=extent, cmap='Reds')
        _plt_colorbar(p, cax=add_color_bar_ax(fig,ax.flat[2]))
        ax.flat[2].set_xlabel('row nr') 
        ax.flat[2].set_ylabel('col nr') 
        if not fig is None:
            if filename:
                fig.savefig(filename, **save_kwargs)
            if not show:
                _plt_close(fig)
        return fig
    #
#
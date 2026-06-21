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
from aabpl.illustrations.plot_utils import truncate_colormap, map_2D_to_rgb, get_2D_rgb_colobar_kwargs, add_color_bar_ax, set_map_frame
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
    grid_ids       : Grid cell row/col indices and point counts (diagnostic).

    Examples
    --------
    fig = grid.plot.clusters(filename='output/clusters.png')
    fig = grid.plot.clusters(filename='output/clusters.png', show=False)
    fig = grid.plot.clusters(figsize=(20, 10), cmap='Blues')
    """
    def __init__(self, grid):
        self.grid = grid

    def vars(self, colnames=None, filename='', **plot_kwargs):
        from aabpl.illustrations.plot_pt_vars import create_plots_for_vars
        import numpy as np
        if colnames is None:
            colnames = np.array(self.grid.search.target.c)
        return create_plots_for_vars(grid=self.grid, colnames=colnames, filename=filename, plot_kwargs=plot_kwargs)

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
            figsize : tuple, default ``(12, 10)``
            cmap : str, default ``'Reds'``

        Returns
        -------
        matplotlib.figure.Figure
        """
        save_kwargs = {'dpi': 300, 'bbox_inches': 'tight', **save_kwargs}
        figsize = plot_kwargs.pop('figsize', (12, 10))
        cmap_name = plot_kwargs.pop('cmap', 'Reds')
        if axs is None:
            fig, axs = _plt_subplots(ncols=len(self.grid.search.target.c), figsize=figsize, dpi=display_dpi)

        # Output grid + cached per-output-cell aggregates are built lazily here.
        self.grid.update_spacing()
        # Draw one coloured square per NON-EMPTY output cell (from the cached
        # grid.output_id_to_sums) as a PatchCollection — no dense full-grid array,
        # so memory scales with populated cells, not the bounding-box area. The axes
        # still span the full output extent, so the plot looks the same as a raster.
        sx, sy = self.grid.output_spacing, self.grid.output_spacing_y
        ox, oy = self.grid.output_x_steps[0], self.grid.output_y_steps[0]
        extent = [self.grid.output_x_steps[0], self.grid.output_x_steps[-1],
                  self.grid.output_y_steps[-1], self.grid.output_y_steps[0]]
        out_sums = self.grid.output_id_to_sums
        for i,column in enumerate(self.grid.search.target.c):
            ax = axs if len(self.grid.search.target.c)==1 else axs.flat[i]
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
            cmap = truncate_colormap(_plt_get_cmap(cmap_name), 0.3, 1)
            cmap.set_under('#ffffff00')
            cmap.set_bad('#ffffff00')
            pc = _plt_PatchCollection(rects, cmap=cmap, norm=norm)
            pc.set_array(vals)
            ax.add_collection(pc)
            cb = _plt_colorbar(pc, cax=add_color_bar_ax(fig,ax))
            ax.set_xlabel('x/lon')
            ax.set_ylabel('y/lat')
            ax.title.set_text('Aggregated value per cell for '+str(column))
            set_map_frame(ax=ax, xmin=extent[0], xmax=extent[1], ymin=extent[3], ymax=extent[2])
        if not fig is None:
            if filename:
                fig.savefig(filename, **save_kwargs)
            if not show:
                _plt_close(fig)
        return fig

    #

    def clusters(self, filename:str='', fig=None, axs=None, show:bool=True, display_dpi:int=100, save_kwargs:dict={}, **plot_kwargs):
        """
        Plot detected cluster polygons overlaid on aggregated cell values.
        One subplot per column in ``c``. Cluster outlines are drawn in red
        with hatching, labelled by cluster id.

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
            figsize : tuple, default ``(10, 10 * n_indicators)``
            cmap : str, default ``'binary'``

        Returns
        -------
        matplotlib.figure.Figure

        Examples
        --------
        fig = grid.plot.clusters()
        fig = grid.plot.clusters(filename='clusters.png', show=False)
        fig = grid.plot.clusters(figsize=(20, 15), cmap='Greys')
        """
        if len(self.grid.clustering.by_column)==0:
            print("No clustering performed. Run detect_cell_clusters or grid.create_clusters first.")
            return
        save_kwargs = {'dpi': 300, 'bbox_inches': 'tight', **save_kwargs}
        n = len(self.grid.search.target.c)
        figsize = plot_kwargs.pop('figsize', (10, 10 * n))
        cmap_name = plot_kwargs.pop('cmap', 'binary')
        if axs is None:
            fig, axs = _plt_subplots(ncols=n, figsize=figsize, dpi=display_dpi)

        # Output grid + cached per-output-cell aggregates built lazily here; the
        # background matches grid.plot.cell_aggregates. Cluster outlines are drawn in
        # coordinate space below and are unaffected by the grid choice.
        self.grid.update_spacing()
        sx, sy = self.grid.output_spacing, self.grid.output_spacing_y
        ox, oy = self.grid.output_x_steps[0], self.grid.output_y_steps[0]
        extent = [self.grid.output_x_steps[0], self.grid.output_x_steps[-1],
                  self.grid.output_y_steps[-1], self.grid.output_y_steps[0]]
        out_sums = self.grid.output_id_to_sums

        for i, (cluster_column, clusters_for_column) in enumerate(self.grid.clustering.by_column.items()):
            ax = axs.flat[i] if n > 1 else axs
            ax.set_xlabel('x/lon '+str(self.grid.local_crs))
            ax.set_ylabel('y/lat '+str(self.grid.local_crs))
            clusters = clusters_for_column.clusters
            ax.title.set_text(str(len(clusters))+' cluster'+ ('s' if len(clusters)!=1 else '') +' for '+str(cluster_column))

            # one coloured square per non-empty output cell (sparse, no dense raster)
            rects, vals = [], []
            for (row, col), v in out_sums.items():
                val = v[i]
                if val == 0:
                    continue
                rects.append(_plt_Rectangle((ox + int(col) * sx, oy + int(row) * sy), sx, sy))
                vals.append(val)
            cmap = _plt_get_cmap(cmap_name)
            cmap = truncate_colormap(cmap, 0.1, 1)
            cmap.set_under('#fff0')
            if vals:
                vals = _np_array(vals)
                vmin, vmax = vals[vals != 0].min(), vals.max()
                norm = _plt_LogNorm(vmin=vmin,vmax=vmax,clip=False) if vmin>0 else _plt_Normalize(vmin=vmin,vmax=vmax,clip=False)
                pc = _plt_PatchCollection(rects, cmap=cmap, norm=norm)
                pc.set_array(vals)
                ax.add_collection(pc)
                cb = _plt_colorbar(pc, cax=add_color_bar_ax(fig,ax))
            for cluster in clusters:
                geoms = [cluster.geometry] if hasattr(cluster.geometry, 'exterior') else cluster.geometry.geoms
                for geom in geoms:
                    ax.add_patch(_plt_Polygon(xy=geom.exterior.coords, hatch='////', facecolor='#f000', edgecolor='#f00'))
                ax.annotate(cluster.id, xy=cluster.centroid, fontsize=15, weight='bold', color='red')

            set_map_frame(ax=ax,xmin=self.grid._search_x_steps.min(),xmax=self.grid._search_x_steps.max(),ymin=self.grid._search_y_steps.min(),ymax=self.grid._search_y_steps.max())

        if not fig is None:
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
            figsize : tuple, default ``(10 * n_cols, 8 * n_rows)``
            cmap : str, default ``'Reds'``
            s : float  — marker size (default auto-scaled to figsize)
            vmin, vmax : float — colormap range
            sample_area_colors : list[str, str]
                Two hex colours for [valid area, excluded area]
                (default ``['#ffffff', '#bedbe6']``)

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
            colnames=_np_array([self.grid.search.target.c, self.grid.search.source.aggregate_columns]),
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
            'xmin':self.grid._search_x_steps.min(),
            'ymin':self.grid._search_y_steps.min(),
            'xmax':self.grid._search_x_steps.max(),
            'ymax':self.grid._search_y_steps.max(),
        }
        extent=[imshow_kwargs['xmin'],imshow_kwargs['xmax'],imshow_kwargs['ymax'],imshow_kwargs['ymin']]
        X = _np_array([[map_2D_to_rgb(x,y, **imshow_kwargs) for x in  self.grid._search_x_steps[:-1]] for y in reversed(self.grid._search_y_steps[:-1])])
        # ax.flat[0].imshow(X=X, interpolation='none', extent=extent)
        # ax.flat[0].pcolormesh([self.grid._search_x_steps, self.grid._search_y_steps], X)
        # ax.flat[0].pcolormesh(X, edgecolor="black", linewidth=.1/max([len(self.grid._search_col_ids), len(self.grid._search_row_ids)]))
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
            'xmin':self.grid._search_col_ids.min(),
            'ymin':self.grid._search_row_ids.min(),
            'xmax':self.grid._search_col_ids.max(),
            'ymax':self.grid._search_row_ids.max(),
        }
        extent=[imshow_kwargs['xmin'],imshow_kwargs['xmax'],imshow_kwargs['ymax'],imshow_kwargs['ymin']]

        X = _np_array([[map_2D_to_rgb(x,y, **imshow_kwargs) for x in  self.grid._search_col_ids] for y in reversed(self.grid._search_row_ids)])
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
        
        from aabpl.radius_search.point_grid_assignment import cell_count as _cell_count
        X = _np_array([[_cell_count(self.grid, row_id, col_id) for col_id in self.grid._search_col_ids] for row_id in reversed(self.grid._search_row_ids)])
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
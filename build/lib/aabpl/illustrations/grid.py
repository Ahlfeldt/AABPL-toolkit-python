from numpy import (
    array as _np_array, 
    unique as _np_unique,
    spacing as _np_spacing,
    linspace as _np_linspace,
)
from matplotlib.pyplot import subplots as _plt_subplots, colorbar as _plt_colorbar
from matplotlib.pyplot import get_cmap as _plt_get_cmap
from matplotlib.pyplot import close as _plt_close
from matplotlib.patches import (Rectangle as _plt_Rectangle, Polygon as _plt_Polygon, Circle as _plt_Circle)
from matplotlib.colors import LogNorm as _plt_LogNorm, Normalize as _plt_Normalize
from aabpl.illustrations.plot_utils import truncate_colormap, map_2D_to_rgb, get_2D_rgb_colobar_kwargs, add_color_bar_ax, set_map_frame
from aabpl.illustrations.plot_pt_vars import create_plots_for_vars

class GridPlots(object):
    """ 
    Methods:

    """
    def __init__(
        self,
        grid
    ):
        """
        bind plot functions to grid 
        """
        self.grid = grid

    def cell_aggregates(
        self,
        filename:str='',
        fig=None,
        ax=None, close_plot:bool=False, 
        save_kwargs={}, **plot_kwargs
    ):
        
        """
        plot aggregated value per cell for each cluster indicator
        """
        if ax is None:
            fig, axs = _plt_subplots(ncols=len(self.grid.search.target.c), figsize=(12,10))
        
        id_to_sums = self.grid.id_to_sums
        imshow_kwargs = {
            'xmin':self.grid.x_steps.min(),
            'ymin':self.grid.y_steps.min(),
            'xmax':self.grid.x_steps.max(),
            'ymax':self.grid.y_steps.max(),
        }    
        extent=[imshow_kwargs['xmin'],imshow_kwargs['xmax'],imshow_kwargs['ymax'],imshow_kwargs['ymin']]
        # cmap = _plt_get_cmap('Reds')
        # cmap.set_under('#ccc')
        for i,column in enumerate(self.grid.search.target.c):
            ax = axs if len(self.grid.search.target.c)==1 else axs.flat[i]
            max_sum = max([vals[i] for vals in id_to_sums.values()])
            X = _np_array([[id_to_sums[(row,col)][i] if ((row,col)) in id_to_sums else 0 for col in  self.grid.col_ids] for row in reversed(self.grid.row_ids)])
            ux = _np_unique(X)
            vmin = ux[ux!=0].min()
            vmax = X.max()
            norm = _plt_LogNorm(vmin=vmin,vmax=vmax,clip=False) if vmin>=0 else _plt_Normalize(vmin=vmin,vmax=vmax,clip=False)
            cmap = truncate_colormap(_plt_get_cmap('Reds'), 0.3, 1)
            cmap.set_under('#fff0')
            cmap.set_under('#cfcd')
            p = ax.imshow(X=X, interpolation='none', cmap=cmap, norm=norm, extent=extent)
            cb = _plt_colorbar(p, cax=add_color_bar_ax(fig,ax))
            ax.set_xlabel('x/lon') 
            ax.set_ylabel('y/lat') 
            ax.title.set_text('Aggregated value per cell for '+str(column))
            set_map_frame(ax=ax,xmin=self.grid.x_steps.min(),xmax=self.grid.x_steps.max(),ymin=self.grid.y_steps.min(),ymax=self.grid.y_steps.max())
        if not fig is None:
            if filename:
                fig.savefig(filename, dpi=300, bbox_inches="tight")
            if close_plot:
                _plt_close(fig)
        return fig

    #

    def clusters(self, filename:str='', fig=None, axs=None, close_plot:bool=False, save_kwargs={}, **plot_kwargs):
        """
        Plot cell clusters (for each clusterindicator)
        """
        
        if axs is None:
            fig, axs = _plt_subplots(ncols=len(self.grid.search.target.c), figsize=(10,10*len(self.grid.search.target.c)))
            
        id_to_sums = self.grid.id_to_sums
        
        for i, (cluster_column, clusters_for_column) in enumerate(self.grid.clustering.by_column.items()):
            ax = axs.flat[i] if len(self.grid.search.target.c)>1 else axs
            ax.set_xlabel('x/lon '+str(self.grid.local_crs)) 
            ax.set_ylabel('y/lat '+str(self.grid.local_crs)) 
            clusters = clusters_for_column.clusters
            ax.title.set_text(str(len(clusters))+' cluster'+ ('s' if len(clusters)!=1 else '') +' for '+str(cluster_column))
            imshow_kwargs = {
                'xmin':self.grid.x_steps.min(),
                'ymin':self.grid.y_steps.min(),
                'xmax':self.grid.x_steps.max(),
                'ymax':self.grid.y_steps.max(),
            }
            
            extent=[imshow_kwargs['xmin'],imshow_kwargs['xmax'],imshow_kwargs['ymax'],imshow_kwargs['ymin']]

            X = _np_array([[id_to_sums[(row,col)][i] if ((row,col)) in id_to_sums else 0 for col in  self.grid.col_ids] for row in (self.grid.row_ids)])
            X_flat = X.flat
            cmap = _plt_get_cmap('binary')
            vmin, vmax = (X.flat[X_flat != 0]).min(), X.max()
            norm = _plt_LogNorm(vmin=vmin,vmax=vmax,clip=False) if vmin>=0 else _plt_Normalize(vmin=vmin,vmax=vmax,clip=False)
            cmap = truncate_colormap(cmap, 0.1, 1)
            cmap.set_under('#fff0')
            
            p = ax.imshow(X=X, interpolation='none', cmap=cmap, norm=norm, extent=extent)
            cb = _plt_colorbar(p, cax=add_color_bar_ax(fig,ax))
            for cluster in clusters:
                geoms = [cluster.geometry] if hasattr(cluster.geometry, 'exterior') else cluster.geometry.geoms
                for geom in geoms:
                    ax.add_patch(_plt_Polygon(xy=geom.exterior.coords, hatch='////', facecolor='#f000', edgecolor='#f00'))
                ax.annotate(cluster.id, xy=cluster.centroid, fontsize=15, weight='bold', color='red')
            
            set_map_frame(ax=ax,xmin=self.grid.x_steps.min(),xmax=self.grid.x_steps.max(),ymin=self.grid.y_steps.min(),ymax=self.grid.y_steps.max())
            
            
        if not fig is None:
            if filename:
                fig.savefig(filename, dpi=300, bbox_inches="tight")
            if close_plot:
                _plt_close(fig)
        return fig

    def cluster_vars(
            self,
            filename:str='',
            save_kwargs:dict={}, close_plot:bool=False,
            **plot_kwargs,
        ):
        return create_plots_for_vars(
            grid=self.grid,
            colnames=_np_array([self.grid.search.target.c, self.grid.search.source.aggregate_columns]),
            filename=filename,
            close_plot=close_plot,
            save_kwargs=save_kwargs,
            plot_kwargs=plot_kwargs,
        )
    #

    def grid_ids(self, fig=None, ax=None, filename:str='', close_plot:bool=False, save_kwargs={}, **plot_kwargs):
        """
        Illustrate row and column ids of self.grid cells.
        """
        if ax is None:
            fig, ax = _plt_subplots(ncols=3, figsize=(15,10))
        imshow_kwargs = {
            'xmin':self.grid.x_steps.min(),
            'ymin':self.grid.y_steps.min(),
            'xmax':self.grid.x_steps.max(),
            'ymax':self.grid.y_steps.max(),
        }
        extent=[imshow_kwargs['xmin'],imshow_kwargs['xmax'],imshow_kwargs['ymax'],imshow_kwargs['ymin']]
        X = _np_array([[map_2D_to_rgb(x,y, **imshow_kwargs) for x in  self.grid.x_steps[:-1]] for y in reversed(self.grid.y_steps[:-1])])
        # ax.flat[0].imshow(X=X, interpolation='none', extent=extent)
        # ax.flat[0].pcolormesh([self.grid.x_steps, self.grid.y_steps], X)
        # ax.flat[0].pcolormesh(X, edgecolor="black", linewidth=.1/max([len(self.grid.col_ids), len(self.grid.row_ids)]))
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
            'xmin':self.grid.col_ids.min(),
            'ymin':self.grid.row_ids.min(),
            'xmax':self.grid.col_ids.max(),
            'ymax':self.grid.row_ids.max(),
        }
        extent=[imshow_kwargs['xmin'],imshow_kwargs['xmax'],imshow_kwargs['ymax'],imshow_kwargs['ymin']]

        X = _np_array([[map_2D_to_rgb(x,y, **imshow_kwargs) for x in  self.grid.col_ids] for y in reversed(self.grid.row_ids)])
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
        
        X = _np_array([[len(self.grid.id_to_pt_ids[(row_id, col_id)]) if (row_id, col_id) in self.grid.id_to_pt_ids else 0 for col_id in self.grid.col_ids] for row_id in reversed(self.grid.row_ids)])
        # p = ax.flat[2].pcolormesh(X, cmap='Reds')
        p = ax.flat[2].imshow(X=X, interpolation='none', extent=extent, cmap='Reds')
        _plt_colorbar(p, cax=add_color_bar_ax(fig,ax.flat[2]))
        ax.flat[2].set_xlabel('row nr') 
        ax.flat[2].set_ylabel('col nr') 
        if not fig is None:
            if filename:
                fig.savefig(filename, dpi=300, bbox_inches="tight")
            if close_plot:
                _plt_close(fig)
        return fig
    #
#
from numpy import (
    array as _np_array, 
    unique as _np_unique,
)
from aabpl.illustrations.plot_utils import map_2D_to_rgb, get_2D_rgb_colobar_kwargs
from matplotlib.pyplot import subplots as _plt_subplots, colorbar as _plt_colorbar
from matplotlib.pyplot import get_cmap as _plt_get_cmap


def plot_cell_sums(
        self,
        fig=None,
        ax=None,
        filename:str=''
    ):
    """
    plot aggregated value per cell for each cluster indicator
    """
    if ax is None:
        fig, axs = _plt_subplots(ncols=len(self.clusters), figsize=(12,10))
    
    id_to_sums = self.id_to_sums
    imshow_kwargs = {
        'xmin':self.x_steps.min(),
        'ymin':self.y_steps.min(),
        'xmax':self.x_steps.max(),
        'ymax':self.y_steps.max(),
    }    
    extent=[imshow_kwargs['xmin'],imshow_kwargs['xmax'],imshow_kwargs['ymax'],imshow_kwargs['ymin']]
    cmap = _plt_get_cmap('Reds')
    cmap.set_under('#ccc')
    for i in range(len(self.clusters)):
        ax = axs if len(self.clusters)==1 else axs.flat[i]
        max_sum = max(list(id_to_sums.values()))
        X = _np_array([[id_to_sums[(row,col)][i] if ((row,col)) in id_to_sums else 0 for col in  self.col_ids] for row in self.row_ids])
        ux = _np_unique(X)
        minX = min(ux[ux!=0])
        print("min X", min(X), max(X))
        # p = ax.imshow(X=X, interpolation='none', cmap=cmap, vmin=1e-5,vmax=max_sum, extent=extent)
        p = ax.pcolormesh(X, cmap=cmap, vmin=minX/2,vmax=max_sum)
        cb = _plt_colorbar(p)
        ax.set_xlabel('x/lon') 
        ax.set_ylabel('y/lat') 
        ax.title.set_text('Aggregated value per cell for indicator '+str(i))
    if filename:
        fig.savefig(filename, dpi=300, bbox_inches="tight")



def plot_grid_ids(self, fig=None, ax=None, filename:str='',):
    """
    Illustrate row and column ids of grid cells.
    """
    if ax is None:
        fig, ax = _plt_subplots(ncols=3, figsize=(15,10))
    imshow_kwargs = {
        'xmin':self.x_steps.min(),
        'ymin':self.y_steps.min(),
        'xmax':self.x_steps.max(),
        'ymax':self.y_steps.max(),
    }
    extent=[imshow_kwargs['xmin'],imshow_kwargs['xmax'],imshow_kwargs['ymax'],imshow_kwargs['ymin']]
    X = _np_array([[map_2D_to_rgb(x,y, **imshow_kwargs) for x in  self.x_steps[:-1]] for y in self.y_steps[:-1]])
    # ax.flat[0].imshow(X=X, interpolation='none', extent=extent)
    # ax.flat[0].pcolormesh([self.x_steps, self.y_steps], X)
    ax.flat[0].pcolormesh(X, edgecolor="black", linewidth=1/max([self.n_x_steps, self.n_y_steps])/1.35)
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
        'xmin':self.col_ids.min(),
        'ymin':self.row_ids.min(),
        'xmax':self.col_ids.max(),
        'ymax':self.row_ids.max(),
    }
    extent=[imshow_kwargs['xmin'],imshow_kwargs['xmax'],imshow_kwargs['ymax'],imshow_kwargs['ymin']]

    X = _np_array([[map_2D_to_rgb(x,y, **imshow_kwargs) for x in  self.col_ids] for y in self.row_ids])
    ax.flat[1].imshow(X=X, interpolation='none', extent=extent)
    # ax.flat[1].set_aspect(2)
    colorbar_kwargs = get_2D_rgb_colobar_kwargs(**imshow_kwargs)
    cb = _plt_colorbar(**colorbar_kwargs[2], ax=ax.flat[1])
    cb.ax.set_xlabel("diagonal")
    cb = _plt_colorbar(**colorbar_kwargs[0], ax=ax.flat[1])
    cb.ax.set_xlabel("col nr")
    cb = _plt_colorbar(**colorbar_kwargs[1], ax=ax.flat[1])
    cb.ax.set_xlabel("row nr") 
    ax.flat[1].set_xlabel('row nr') 
    ax.flat[1].set_ylabel('col nr') 
    ax.flat[1].title.set_text("Grid row / col indices")
    
    X = _np_array([[len(self.id_to_pt_ids[(row_id, col_id)]) if (row_id, col_id) in self.id_to_pt_ids else 0 for col_id in self.col_ids] for row_id in self.row_ids])
    p = ax.flat[2].pcolormesh(X, cmap='Reds')
    ax.flat[2].set_xlabel('row nr') 
    ax.flat[2].set_ylabel('col nr') 
    _plt_colorbar(p)
    if filename:
        fig.savefig(filename, dpi=300, bbox_inches="tight")

def plot_clusters(self, fig=None, axs=None, filename:str=''):
    """
    Plot cell clusters (for each clusterindicator)
    """
    if axs is None:
        fig, axs = _plt_subplots(ncols=len(self.clusters), figsize=(10,10*len(self.clusters)))

    for i, cluster_column in enumerate(self.clusters):
        ax = axs.flat[i] if len(self.clusters)>1 else axs
        ax.set_xlabel('x/lon '+str(self.local_crs)) 
        ax.set_ylabel('y/lat '+str(self.local_crs)) 
        clusters = self.clusters[cluster_column]
        ax.title.set_text(str(len(clusters['prime_locs']))+' clusters for '+str(cluster_column))
        cell_to_cluster = clusters['cell_to_cluster']
        max_cluster_id = 1e10 if len(cell_to_cluster)==0 else max(list(cell_to_cluster.values()))
        imshow_kwargs = {
            'xmin':self.x_steps.min(),
            'ymin':self.y_steps.min(),
            'xmax':self.x_steps.max(),
            'ymax':self.y_steps.max(),
        }
        
        extent=[imshow_kwargs['xmin'],imshow_kwargs['xmax'],imshow_kwargs['ymax'],imshow_kwargs['ymin']]
        cmap = _plt_get_cmap('Reds')
        cmap.set_under('#ccc')
        X = _np_array([[cell_to_cluster[(row,col)] if (row,col) in cell_to_cluster else 0 for col in  self.col_ids] for row in self.row_ids])
        p = ax.imshow(X=X, interpolation='none', cmap=cmap, vmin=1e-5,vmax=max_cluster_id, extent=extent)
        # p = ax.pcolormesh(X, cmap=cmap, edgecolor="black", linewidth=1/max([self.n_x_steps, self.n_y_steps])/1.35)
        # cb = _plt_colorbar(p)
        # TODO zoom in
        for cluster in clusters['prime_locs']:
            ax.annotate(cluster['id'], xy=cluster['centroid'], fontsize=10)
    if filename:
        fig.savefig(filename, dpi=300, bbox_inches="tight")
from pandas import DataFrame as _pd_DataFrame
from numpy import (
    array as _np_array,
    linspace as _np_linspace,
    searchsorted as _np_searchsorted,
)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.pyplot import close as _plt_close
from matplotlib.colors import LogNorm as _plt_LogNorm, Normalize as _plt_Normalize, LinearSegmentedColormap as _plt_LinearSegmentedColormap, ListedColormap as _plt_ListedColormap
from matplotlib.pyplot import (subplots as _plt_subplots, colorbar as _plt_colorbar, get_cmap as _plt_get_cmap)
from matplotlib.patches import Patch as _plt_Patch
from aabpl.illustrations.plot_utils import add_color_bar_ax, set_map_frame, truncate_colormap, plot_polygon
from shapely.geometry import Polygon as _shapely_Polygon
from aabpl.illustrations.plot_utils import truncate_colormap
from shapely.geometry import Polygon as _shapely_Polygon, MultiPoint as _shapely_MultiPoint

def plot_sample_area(
        grid:dict,
        pts:_pd_DataFrame,
        x:str='lon',
        y:str='lat',
        filename:str='',
        plot_kwargs:dict={},
        close_plot:bool=False,
):
    """
    TODO Descripiton
    """
    x_coord_name, y_coord_name = x,y
    # specify default plot kwargs and add defaults
    default_kwargs = {
        's':0.8,
        'color':'#eaa',
        'figsize': (10,10),
        'fig':None,
        'ax':None,
        'hlines':{'color':'red', 'linewidth':1},
        'vlines':{'color':'red', 'linewidth':1},
    }
    kwargs = {}
    for k in plot_kwargs:
        if k in [k for k,v in default_kwargs.items() if type(v)==dict]:
            kwargs[k] = {**default_kwargs.pop(k), **plot_kwargs.pop(k)}
    kwargs.update(default_kwargs)
    kwargs.update(plot_kwargs)
    figsize = kwargs.pop('figsize')
    fig = kwargs.pop('fig')
    ax = kwargs.pop('ax')
    for k in ['fig', 'ax', 'figsize']:
        plot_kwargs.pop(k,None)

    if ax is None:
        fig, ax = plt.subplots(1,1,figsize=figsize)

    non_valid_area = _shapely_Polygon([
        (grid.sample_grid_bounds[0], grid.sample_grid_bounds[1]),
        (grid.sample_grid_bounds[2], grid.sample_grid_bounds[1]),
        (grid.sample_grid_bounds[2], grid.sample_grid_bounds[3]),
        (grid.sample_grid_bounds[0], grid.sample_grid_bounds[3])
        ]).difference(grid.sample_area) 
    
    non_valid_area_color = "#88b9cc"
    non_valid_cell_color = "#bedbe6"
    sample_area_color = "#ffffff"
    color_cluster = "#2a07ee"
    cmap_scatter = _plt_get_cmap('Reds')
    minval = 0.2
    color_under = cmap_scatter(minval/2)
    cmap_scatter = truncate_colormap(cmap=cmap_scatter, minval=minval, maxval=1.0, n=100)
    cmap_scatter.set_under(color_under)
    cmap_scatter.set_bad(color_under)
    cmap_scatter.set_over(color_cluster)
    s = 0.2*figsize[0]/10
    

    # SCATTER POINTS
    ax.set_facecolor(sample_area_color)
    # SET TITLEd
    ax.set_title("Sample area", fontdict={'fontsize':6})
    # ADD DISTRIUBTION PLOT
    # grey out cells that are not used for sampling
    cells_rndm_sample = grid.cells_rndm_sample
    col_min = int(round((grid.sample_grid_bounds[0]-grid.total_bounds.xmin)/grid.spacing,0))
    row_min = int(round((grid.sample_grid_bounds[1]-grid.total_bounds.ymin)/grid.spacing,0))
    col_max = int(round((grid.sample_grid_bounds[2]-grid.total_bounds.xmin)/grid.spacing-1,0))
    row_max = int(round((grid.sample_grid_bounds[3]-grid.total_bounds.ymin)/grid.spacing-1,0))
    X = _np_array([[not (cells_rndm_sample if type(cells_rndm_sample) == bool else (row,col) in cells_rndm_sample) for col in  range(col_min,col_max+1)] for row in range(row_min,row_max+1)[::-1]])
    cmap_binary = _plt_ListedColormap([sample_area_color, non_valid_cell_color])
    extent = [grid.sample_grid_bounds[0],grid.sample_grid_bounds[2],grid.sample_grid_bounds[1],grid.sample_grid_bounds[3]]
    p = ax.imshow(X=X, interpolation='none', cmap=cmap_binary, extent=extent)#, To-Do the extent is imprecise as it does not cover the full grid only its points
    non_valid_patch = _plt_Patch(facecolor=non_valid_area_color, label='Non-valid area', edgecolor='black')
    sample_patch = _plt_Patch(facecolor=sample_area_color, label='Sample area', edgecolor='black')
    ax.legend(handles=[non_valid_patch, sample_patch], loc='best')
    # plot valid area borders
    plot_polygon(ax=ax, poly=non_valid_area, facecolor=non_valid_area_color, edgecolor='black')
    # SCATTER POINTS
    if not pts is None:
        sc = ax.scatter(x=pts[x_coord_name],y=pts[y_coord_name],c='black', s=s, marker='.')
        # add borders of polygon
        # plot_polygon(ax=ax, poly=grid.sample_area, facecolor="none", edgecolor='black')
        # SET LIMITS
        set_map_frame(ax=ax,xmin=pts[x_coord_name].min(),xmax=pts[x_coord_name].max(),ymin=pts[y_coord_name].min(),ymax=pts[y_coord_name].max())
        ax.set_xticks([]), ax.set_yticks([])
        _plt_colorbar(sc, extend='both', cax=add_color_bar_ax(fig,ax))
    ax.set_aspect('equal')

    if filename:
        fig.savefig(filename, dpi=300, bbox_inches="tight")
    if close_plot:
        _plt_close(fig)
    return fig
    #
#
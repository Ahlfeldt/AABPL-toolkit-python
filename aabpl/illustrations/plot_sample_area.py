from pandas import DataFrame as _pd_DataFrame
from numpy import (
    array as _np_array,
    zeros as _np_zeros,
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

def plot_sample_area(
        grid:dict,
        pts:_pd_DataFrame,
        x:str='lon',
        y:str='lat',
        filename:str='',
        plot_kwargs:dict={},
        show:bool=True,
        display_dpi:int=100,
        save_kwargs:dict={},
):
    """
    Plot the sample area used for drawing random points, with source points
    overlaid and excluded cells shaded. Called via ``grid.plot.sample_area()``.

    Parameters
    ----------
    grid
        Grid object with ``sample_area`` and ``cells_rndm_sample`` set.
    pts : DataFrame
        Source points to scatter. Pass ``None`` to omit points.
    x, y : str
        Coordinate column names in ``pts``.
    filename : str
        Save path. Empty string skips saving (default ``''``).
    plot_kwargs : dict
        Supported keys:
        - ``figsize`` : tuple (default ``(10, 10)``)
        - ``s`` : float — scatter marker size (default ``0.8``)
        - ``fig``, ``ax`` — existing Figure/Axes to draw into
    show : bool
        Display the figure (default ``True``).
    display_dpi : int
        Resolution for inline/screen display (default ``100``).
        Use ``fig.savefig(filename, dpi=300)`` to control saved-file resolution.

    Returns
    -------
    matplotlib.figure.Figure
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
        fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=display_dpi)

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
    cells_rndm_sample = grid._search_internals.cells_rndm_sample
    _si = grid._search_internals
    col_min = int(round((grid.sample_grid_bounds[0]-_si.bounds.xmin)/_si.spacing,0))
    row_min = int(round((grid.sample_grid_bounds[1]-_si.bounds.ymin)/_si.spacing,0))
    col_max = int(round((grid.sample_grid_bounds[2]-_si.bounds.xmin)/_si.spacing-1,0))
    row_max = int(round((grid.sample_grid_bounds[3]-_si.bounds.ymin)/_si.spacing-1,0))
    n_rows = row_max - row_min + 1
    n_cols = col_max - col_min + 1
    if type(cells_rndm_sample) == bool:
        X = _np_zeros((n_rows, n_cols), dtype=bool)
        if not cells_rndm_sample:
            X[:] = True
    else:
        X = _np_zeros((n_rows, n_cols), dtype=bool)
        X[:] = True  # all excluded by default; mark valid cells False
        for (row, col) in cells_rndm_sample:
            ri = row_max - row
            ci = col - col_min
            if 0 <= ri < n_rows and 0 <= ci < n_cols:
                X[ri, ci] = False
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
        fig.savefig(filename, **{'dpi': 300, 'bbox_inches': 'tight', **save_kwargs})
    if not show:
        _plt_close(fig)
    return fig
    #
#
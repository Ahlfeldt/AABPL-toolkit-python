from matplotlib.pyplot import (subplots as _plt_subplots)
from numpy import array as _np_array

def create_plots_for_vars(
        grid,
        colnames:_np_array,
        filename:str="",
        plot_kwargs:dict={},
):
    """
    TODO Descripiton
    """
    nrows = colnames.shape[0]
    ncols = 1 if len(colnames.shape)==1 else colnames.shape[1]
    # specify default plot kwargs and add defaults
    s = 1 if not 'figsize' in plot_kwargs else 0.1*plot_kwargs['figsize'][0]
    plot_kwargs = {
        'fig': None,
        'axs': None,
        's': s,
        'cmap': 'Reds',
        'figsize': (12*ncols,12*nrows),
        'additional_varnames':[],
        **plot_kwargs
    }
    plot_kwargs.pop('color', None)
    figsize = plot_kwargs.pop('figsize')
    fig = plot_kwargs.pop('fig')
    axs = plot_kwargs.pop('axs')
    additional_varnames = plot_kwargs.pop('additional_varnames')
    if len(additional_varnames)>nrows:
        # TODO this is not compelted
        nrows = len(additional_varnames)
    
    if fig is None or axs is None:
        fig, axs = _plt_subplots(nrows,1, figsize=figsize)

    xmin, xmax, ymin, ymax = grid.total_bounds.xmin, grid.total_bounds.xmax, grid.total_bounds.ymin, grid.total_bounds.ymax,
    xs = grid.search.source.pts_df[grid.search.source.x]
    ys = grid.search.source.pts_df[grid.search.source.y]
    for i, colname in enumerate(colnames.flat):
        # SELECT AX (IF MULTIPLE)
        ax = axs.flat[i] if nrows > 1 else axs
        
        # SET TITLE
        ax_title = (colname)
        ax.set_title(ax_title)
        
        # ADD DISTRIUBTION PLOT
        c = grid.search.source.pts_df[colname]# vmin=c.min(), vmax=c.max(),
        scttr = ax.scatter(x=xs,y=ys,c=c, norm="log", **plot_kwargs)
        
        fig.colorbar(scttr, ax=ax)

        # SET LIMITS
        pad_x, pad_y = (xmax-xmin)/50, (ymax-ymin)/50
        ax.set_xlim([xmin-pad_x,xmax+pad_x])
        ax.set_ylim([ymin-pad_y,ymax+pad_y])
    
    if filename:
        fig.savefig(filename, dpi=300, bbox_inches="tight")
    #
#
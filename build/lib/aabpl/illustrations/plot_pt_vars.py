from matplotlib.pyplot import (subplots as _plt_subplots)
from matplotlib.pyplot import get_cmap as _plt_get_cmap
from matplotlib.pyplot import close as _plt_close
from matplotlib.colors import LogNorm as _plt_LogNorm, Normalize as _plt_Normalize, LinearSegmentedColormap as _plt_LinearSegmentedColormap, ListedColormap as _plt_ListedColormap
from matplotlib.patches import Patch as _plt_Patch
from numpy import array as _np_array
from aabpl.illustrations.plot_utils import truncate_colormap, set_map_frame, plot_polygon
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
        close_plot:bool=False,
):
    """
    TODO Descripiton
    """
    nrows = colnames.shape[0]
    ncols = 1 if len(colnames.shape)==1 else colnames.shape[1]
    # specify default plot kwargs and add defaults
    s = 1 if not 'figsize' in plot_kwargs else 0.1*plot_kwargs['figsize'][0]
    plot_kwargs = handle_plot_kwargs(default_kwargs={
        'fig': None,
        'axs': None,
        's': s,
        'cmap': 'Reds',
        'figsize': (10*ncols,8*nrows),
        'additional_varnames':[],
        'sample_area_colors': ["#ffffff",  "#bedbe6"]
    }, **plot_kwargs)
    save_kwargs = {'dpi':300, 'bbox_inches':"tight", **save_kwargs}
    
    plot_kwargs.pop('color', None)
    figsize = plot_kwargs.pop('figsize')
    fig = plot_kwargs.pop('fig')
    axs = plot_kwargs.pop('axs')
    
    sample_area_color, non_valid_area_color = plot_kwargs.pop('sample_area_colors')
    
    additional_varnames = plot_kwargs.pop('additional_varnames')
    if len(additional_varnames)>nrows:
        # TODO this is not compelted
        nrows = len(additional_varnames)
    if fig is None or axs is None:
        fig, axs = _plt_subplots(nrows,ncols, figsize=figsize)
    if not grid.sample_grid_bounds is None and not grid.sample_area is None:
        non_valid_area = _shapely_Polygon([
            (grid.sample_grid_bounds[0], grid.sample_grid_bounds[1]),
            (grid.sample_grid_bounds[2], grid.sample_grid_bounds[1]),
            (grid.sample_grid_bounds[2], grid.sample_grid_bounds[3]),
            (grid.sample_grid_bounds[0], grid.sample_grid_bounds[3])
            ]).difference(grid.sample_area) 
        xmin, xmax, ymin, ymax = grid.total_bounds.xmin, grid.total_bounds.xmax, grid.total_bounds.ymin, grid.total_bounds.ymax,
    else:    
        xmin, xmax, ymin, ymax = grid.total_bounds.xmin, grid.total_bounds.xmax, grid.total_bounds.ymin, grid.total_bounds.ymax,
    xs = grid.search.source.pts[grid.search.source.x]
    ys = grid.search.source.pts[grid.search.source.y]
    for i, colname in enumerate(colnames.flat):
        # SELECT AX (IF MULTIPLE)
        ax = axs.flat[i] if nrows > 1 else axs
        
        # SET TITLE
        ax_title = (colname)
        ax.set_title(ax_title)
        # CPOLOR NON SAMPLE AREA
        if not grid.sample_grid_bounds is None and not grid.sample_area is None:
            ax.set_facecolor(sample_area_color)
            cells_rndm_sample = grid.cells_rndm_sample
            col_min = int(round((grid.sample_grid_bounds[0]-grid.total_bounds.xmin)/grid.spacing,0))
            row_min = int(round((grid.sample_grid_bounds[1]-grid.total_bounds.ymin)/grid.spacing,0))
            col_max = int(round((grid.sample_grid_bounds[2]-grid.total_bounds.xmin)/grid.spacing-1,0))
            row_max = int(round((grid.sample_grid_bounds[3]-grid.total_bounds.ymin)/grid.spacing-1,0))
            # row_min, row_max = min([row for row,col in grid.cells_rndm_sample]), max([row for row,col in grid.cells_rndm_sample])
            # col_min, col_max = min([col for row,col in grid.cells_rndm_sample]), max([col for row,col in grid.cells_rndm_sample])
            X = _np_array([[not (cells_rndm_sample if type(cells_rndm_sample)==bool else (row,col) in cells_rndm_sample ) for col in  range(col_min,col_max+1)] for row in range(row_min,row_max+1)[::-1]])
            cmap_binary = _plt_ListedColormap([sample_area_color, non_valid_area_color])
            extent = [grid.sample_grid_bounds[0],grid.sample_grid_bounds[2],grid.sample_grid_bounds[1],grid.sample_grid_bounds[3]]
            p = ax.imshow(X=X, interpolation='none', cmap=cmap_binary, extent=extent)#, To-Do the extent is imprecise as it does not cover the full grid only its points
            non_valid_patch = _plt_Patch(facecolor=non_valid_area_color, label='Non-valid area', edgecolor='black')
            sample_patch = _plt_Patch(facecolor=sample_area_color, label='Sample area', edgecolor='black')
            ax.legend(handles=[non_valid_patch, sample_patch], loc='best')
            plot_polygon(poly=non_valid_area, ax=ax, facecolor=non_valid_area_color, edgecolor='black')
        # ADD DISTRIUBTION PLOT
        c = grid.search.source.pts[colname]
        vmin=plot_kwargs['vmin'] if 'vmin' in plot_kwargs else c.min()
        vmax=plot_kwargs['vmax'] if 'vmax' in plot_kwargs else c.max(),
        # norm = plot_kwargs['norm'] if 'norm' in plot_kwargs else _plt_LogNorm(vmin=c.min(),vmax=c.max()) if (c.min() > 0) else _plt_LogNorm()
        norm = plot_kwargs['norm'] if 'norm' in plot_kwargs else _plt_LogNorm(vmin=c.min(),vmax=c.max()) if (c.min() > 0) else 'linear'
        plot_kwargs['cmap'] = truncate_colormap(_plt_get_cmap(plot_kwargs['cmap']),0.1,1)
        scttr = ax.scatter(x=xs,y=ys,c=c, norm=norm, **plot_kwargs)
        plot_polygon(poly=non_valid_area, ax=ax, facecolor="none", edgecolor='black')
        fig.colorbar(scttr, ax=ax)
        set_map_frame(ax=ax,xmin=xmin,xmax=xmax,ymin=ymin,ymax=ymax)

    if not fig is None:
        if filename:
            fig.savefig(filename, **save_kwargs)
        if close_plot:
            _plt_close(fig)
    return fig
    #
#
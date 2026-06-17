from pandas import DataFrame as _pd_DataFrame
from numpy import (
    array as _np_array,
    zeros as _np_zeros,
    linspace as _np_linspace,
    searchsorted as _np_searchsorted,
    sort as _np_sort,
)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.pyplot import close as _plt_close
from matplotlib.colors import LogNorm as _plt_LogNorm, Normalize as _plt_Normalize, LinearSegmentedColormap as _plt_LinearSegmentedColormap, ListedColormap as _plt_ListedColormap
from matplotlib.pyplot import (subplots as _plt_subplots, colorbar as _plt_colorbar, get_cmap as _plt_get_cmap)
from matplotlib.patches import Patch as _plt_Patch
from aabpl.illustrations.plot_utils import add_color_bar_ax, set_map_frame, truncate_colormap, plot_polygon
from shapely.geometry import Polygon as _shapely_Polygon

def create_distribution_plot(
        pts:_pd_DataFrame,
        rndm_pts:_pd_DataFrame,
        cluster_threshold_values:list,
        k_th_percentile:list,
        radius_sum_columns:list,
        grid:dict=None,
        r:float=None,
        x:str='lon',
        y:str='lat',
        filename:str='',
        plot_kwargs:dict={},
        show:bool=True,
        display_dpi:int=100,
):
    """
    Four-panel plot comparing the random-point null distribution to the
    observed radius-sum distribution. Called via ``grid.plot.rand_dist()``.

    Parameters
    ----------
    pts : DataFrame
        Source points with radius-sum columns already appended.
    rndm_pts : DataFrame
        Random points with radius-sum columns appended by ``compute_null_distribution``.
    cluster_threshold_values : list[float]
        One threshold per column — the k-th percentile cutoff value.
    k_th_percentile : list[float]
        Percentile values used to derive ``cluster_threshold_values``.
    radius_sum_columns : list[str]
        Column names of the radius sums to plot.
    grid
        Grid object (used to draw sample-area overlay).
    r : float
        Search radius in metres (used for the plot title).
    x, y : str
        Coordinate column names in ``pts`` and ``rndm_pts``.
    filename : str
        Save path. Empty string skips saving (default ``''``).
    plot_kwargs : dict
        Supported keys:
        - ``figsize`` : tuple (default ``(10, 10)``)
        - ``s`` : float — scatter marker size (default ``0.8``)
        - ``color`` : str — marker colour (default ``'#eaa'``)
        - ``hlines`` : dict — kwargs for ``ax.hlines`` threshold line
        - ``vlines`` : dict — kwargs for ``ax.vlines`` percentile line
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
    x_coord_name, y_coord_name = x,y
    disk_sums_for_random_points = rndm_pts[radius_sum_columns]
    (n_random_points, ncols) = disk_sums_for_random_points.shape
    # specify default plot kwargs and add defaults
    default_kwargs = {
        's':0.8,
        'color':'#eaa',

        'figsize': (10,10),
        'fig':None,
        'axs':None,

        'title': None,           # override the auto suptitle; use {auto} to embed it: "My prefix: {auto}"
        'suptitle': {},          # extra kwargs forwarded to fig.suptitle(), e.g. fontsize, fontweight, y
        'ax_titles': {},         # dict keyed by panel name to override individual panel titles
                                 # keys: 'rand_dist', 'pts_dist', 'rand_map', 'pts_map'
        'hlines':{'color':'red', 'linewidth':1},
        'vlines':{'color':'red', 'linewidth':1},
    }
    kwargs = {}
    for k in list(plot_kwargs.keys()):
        if k in [k for k,v in default_kwargs.items() if type(v)==dict]:
            kwargs[k] = {**default_kwargs.pop(k), **plot_kwargs.pop(k)}
    kwargs.update(default_kwargs)
    kwargs.update(plot_kwargs)
    figsize = kwargs.pop('figsize')
    fig = kwargs.pop('fig')
    axs = kwargs.pop('axs')
    title_override = kwargs.pop('title')
    suptitle_kwargs = kwargs.pop('suptitle')
    ax_titles = kwargs.pop('ax_titles')
    for k in ['fig', 'axs', 'figsize', 'title', 'suptitle', 'ax_titles']:
        plot_kwargs.pop(k,None)

    if fig is None or axs is None:
        fig = plt.figure(figsize=figsize, dpi=display_dpi, constrained_layout=True)
        outer = gridspec.GridSpec(ncols, 1, figure=fig, hspace=0.1)

    _auto_title = "Aggregate for indicator" + ("" if ncols==1 else "s") + " within " + str(r) + " meters"
    _suptitle = title_override.replace('{auto}', _auto_title) if title_override else _auto_title
    fig.suptitle(_suptitle, **suptitle_kwargs)
    non_valid_area = _shapely_Polygon([
        (grid.sample_grid_bounds[0], grid.sample_grid_bounds[1]),
        (grid.sample_grid_bounds[2], grid.sample_grid_bounds[1]),
        (grid.sample_grid_bounds[2], grid.sample_grid_bounds[3]),
        (grid.sample_grid_bounds[0], grid.sample_grid_bounds[3])
        ]).difference(grid.sample_area) 
    
    xmin, xmax = 0, 100
    xs_random_pts = _np_linspace(xmin,xmax,n_random_points)
    xs_pts = _np_linspace(xmin,xmax,len(pts))
    random_vals = disk_sums_for_random_points.values
    pts_vals = pts[radius_sum_columns].values
    
    for (i, colname, cluster_threshold_value, k) in zip(
        range(ncols), radius_sum_columns, cluster_threshold_values, k_th_percentile):
        columns = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[i], wspace=0.25, hspace=0.0, width_ratios=[3, 7])
        left_col = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=columns[0], hspace=0.3)
        right_col = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=columns[1], hspace=0.15)
        
        # CUMULATIVE DISTRIBUTION RANDOM POINTS
        ys = _np_sort(random_vals[:,i])
        ymin, ymax = min(ys), max(ys)
        # round percentile value as far as necessary only 
        # e.g. threshold value is 10.5328... and next smaller/larger in distribution are 10.51..., 10.6... 
        # rounding to threshold value to firrst digit s.t. thath it lies between those value is sufficient (e.g. 10.53) 
        idx = _np_searchsorted(ys, cluster_threshold_value)
        next_smaller_val, next_larger_val = ys[max([0,idx-1])], ys[idx]
        sufficient_digits = next((
            i for i in range(100) if (
                (
                    (next_smaller_val == next_larger_val or cluster_threshold_value==next_smaller_val) and 
                    round(next_smaller_val,i)==next_smaller_val
                ) or (
                    round(next_larger_val, i) != round(cluster_threshold_value, i) and 
                    round(next_smaller_val, i) != round(cluster_threshold_value, i)
                )
        )),100)
        
        # SET TITLE
        ax = plt.Subplot(fig, left_col[0])
        ax_title = ax_titles.get('rand_dist', "Random points: {col} {k}th-percentile: {thresh}")
        ax_title = ax_title.format(col=colname, k=k, thresh=round(cluster_threshold_value, sufficient_digits), r=r, n=len(rndm_pts))
        ax.set_title(ax_title, fontdict={'fontsize':6})
        ax.set_ylabel("Randomly Drawn Points", fontsize=7, fontweight='bold')
        # SET TICKS
        xtick_steps, ytick_steps = 5, 5
        xticks = _np_array(sorted(
           [x for x in _np_linspace(xmin,xmax,xtick_steps) if abs(x-k) > (xmax-xmin)/(xtick_steps*2)] + 
           [k]
        ))
        ax.set_xticks(xticks, labels=xticks)
        yticks = _np_array(sorted([y for y in _np_linspace(ymin,ymax,ytick_steps) if abs(cluster_threshold_value-y)>(ymax-ymin)/(ytick_steps*10)] + [cluster_threshold_value]))
        ax.set_yticks(yticks, labels=[round(t, sufficient_digits) for t in yticks])
        # ADD CUTOFF LINES
        ax.hlines(y=cluster_threshold_value, xmin=xmin,xmax=xmax, **kwargs['hlines'])
        ax.vlines(x=k, ymin=ymin,ymax=ymax, **kwargs['vlines'])
        # ADD DISTRIUBTION PLOT
        ax.scatter(x=xs_random_pts,y=ys, **plot_kwargs)
        # SET LIMITS
        ax.set_xlim([xmin,xmax])
        if ymin != ymax:
            ax.set_ylim([ymin,ymax])
        
        fig.add_subplot(ax)


        # CUMULATIVE DISTRIBUTION ORIGNAL POINTS
        ys = _np_sort(pts_vals[:,i])
        ymin, ymax = min(ys), max(ys)
        # SELECT AX (IF MULTIPLE)
        ax = plt.Subplot(fig, left_col[1])
        # SET TITLE
        ax.set_title(ax_titles.get('pts_dist', "Distribution for {n} points").format(col=colname, k=k, thresh=round(cluster_threshold_value, sufficient_digits), r=r, n=len(pts)), fontdict={'fontsize':6})
        ax.set_ylabel("Points from Dataset", fontsize=7, fontweight='bold')
        # SET TICKS
        xtick_steps, ytick_steps = 5, 5
        xticks = _np_array(sorted(
           [x for x in _np_linspace(xmin,xmax,xtick_steps) if abs(x-k) > (xmax-xmin)/(xtick_steps*2)] + 
           [k]
        ))
        ax.set_xticks(xticks, labels=xticks)
        yticks = _np_array(sorted([y for y in _np_linspace(ymin,ymax,ytick_steps) if abs(cluster_threshold_value-y)>(ymax-ymin)/(ytick_steps*10)] + [cluster_threshold_value]))
        ax.set_yticks(yticks, labels=[round(t, sufficient_digits) for t in yticks])
        # # ADD CUTOFF LINES
        # ax.hlines(y=cluster_threshold_value, xmin=xmin,xmax=xmax, **kwargs.pop('hlines'))
        # ax.vlines(x=k, ymin=ymin,ymax=ymax, **kwargs.pop('vlines'))
        ax.hlines(y=cluster_threshold_value, xmin=xmin,xmax=xmax, **kwargs['hlines'])
        ax.vlines(x=k, ymin=ymin,ymax=ymax, **kwargs['vlines'])
        # ADD DISTRIUBTION PLOT
        ax.scatter(x=xs_pts,y=ys, **plot_kwargs)
        # SET LIMITS
        ax.set_xlim([xmin,xmax])
        if ymin != ymax:
            ax.set_ylim([ymin,ymax])
        fig.add_subplot(ax)

        # combine them and build a new colormap
        xmin, xmax =  min([pts[x_coord_name].min(), rndm_pts[x_coord_name].min()]), max([pts[x_coord_name].max(), rndm_pts[x_coord_name].max()])
        ymin, ymax =  min([pts[y_coord_name].min(), rndm_pts[y_coord_name].min()]), max([pts[y_coord_name].max(), rndm_pts[y_coord_name].max()])
        
        _pts_nonzero = pts_vals[:,i][pts_vals[:,i]!=0]
        _rnd_nonzero = random_vals[:,i][random_vals[:,i]!=0]
        _has_range = len(_pts_nonzero) > 0 and len(_rnd_nonzero) > 0
        if _has_range:
            vmin = min([_pts_nonzero.min(), _rnd_nonzero.min()])
            vmax = max([pts_vals[:,i].max(), random_vals[:,i].max()])
            _has_range = vmin < vmax and vmin > 0
        if not _has_range:
            vmin = 1.0
            vmax = max(float(cluster_threshold_value), 2.0)

        non_valid_area_color = "#bedbe6"
        sample_area_color = "#ffffff"
        color_cluster = "#2a07ee"
        cmap_scatter = _plt_get_cmap('Reds')
        minval = 0.2
        color_under = cmap_scatter(minval/2)
        cmap_scatter = truncate_colormap(cmap=cmap_scatter, minval=minval, maxval=1.0, n=100)
        cmap_scatter.set_under(color_under)
        cmap_scatter.set_bad(color_under)
        cmap_scatter.set_over(color_cluster)
        _norm_vmax = float(cluster_threshold_value) if cluster_threshold_value > vmin else vmax
        norm = _plt_LogNorm(vmin=vmin,vmax=_norm_vmax,clip=False) if vmin>0 else _plt_Normalize(vmin=vmin,vmax=_norm_vmax,clip=False)
        s = 0.2*figsize[0]/10
        
        # ADD DISTRIUBTION PLOT
        ax.set_facecolor(sample_area_color)
        ax = plt.Subplot(fig, right_col[0])
        # SET TITLE
        ax.set_title(ax_titles.get('rand_map', "Total within radius — {n} random points").format(col=colname, k=k, thresh=round(cluster_threshold_value, sufficient_digits), r=r, n=len(rndm_pts)), fontdict={'fontsize':6})
        # remove cells that are not used for sampling
        if not grid is None:
            cells_rndm_sample = grid.cells_rndm_sample
            col_min = int(round((grid.sample_grid_bounds[0]-grid.total_bounds.xmin)/grid.spacing,0))
            row_min = int(round((grid.sample_grid_bounds[1]-grid.total_bounds.ymin)/grid.spacing,0))
            col_max = int(round((grid.sample_grid_bounds[2]-grid.total_bounds.xmin)/grid.spacing-1,0))
            row_max = int(round((grid.sample_grid_bounds[3]-grid.total_bounds.ymin)/grid.spacing-1,0))
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
            extent = [grid.sample_grid_bounds[0],grid.sample_grid_bounds[2],grid.sample_grid_bounds[1],grid.sample_grid_bounds[3]]
            p = ax.imshow(X=X, interpolation='none', cmap=cmap_binary, extent=extent)#, To-Do the extent is imprecise as it does not cover the full grid only its points
            non_valid_patch = _plt_Patch(facecolor=non_valid_area_color, label='Non-valid area', edgecolor='black')
            sample_patch = _plt_Patch(facecolor=sample_area_color, label='Sample area', edgecolor='black')
            ax.legend(handles=[non_valid_patch, sample_patch], loc='best')
        # plot valid area borders
        plot_polygon(ax=ax, poly=non_valid_area, facecolor=non_valid_area_color, edgecolor='black', linewidth=0.5)
        # SCATTER RANDOM POINTS
        sc = ax.scatter(x=rndm_pts[x_coord_name],y=rndm_pts[y_coord_name],c=random_vals[:,i], s=s, marker='.', norm=norm, cmap=cmap_scatter, linewidths=0.3)
        # add borders of polygon
        plot_polygon(ax=ax, poly=grid.sample_area, facecolor="none", edgecolor='black', linewidth=0.5)
        set_map_frame(ax=ax, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)
        fig.add_subplot(ax)
        fig.colorbar(sc, ax=ax, extend='neither', fraction=0.046, pad=0.04)

        # SCATTER POINTS
        ax = plt.Subplot(fig, right_col[1])
        ax.set_facecolor(sample_area_color)
        # SET TITLEd
        ax.set_title(ax_titles.get('pts_map', "Total within radius — {n} points").format(col=colname, k=k, thresh=round(cluster_threshold_value, sufficient_digits), r=r, n=len(pts)), fontdict={'fontsize':6})
        # ADD DISTRIUBTION PLOT
        # grey out cells that are not used for sampling
        if not grid is None:
            p = ax.imshow(X=X, interpolation='none', cmap=cmap_binary, extent=extent)#, extent=extent
        # plot valid area borders
        plot_polygon(ax=ax, poly=non_valid_area, facecolor=non_valid_area_color, edgecolor='black', linewidth=0.5)
        # SCATTER POINTS
        sc = ax.scatter(x=pts[x_coord_name],y=pts[y_coord_name],c=pts_vals[:,i], s=s, marker='.', norm=norm, cmap=cmap_scatter, linewidths=0.3)
        # add borders of polygon
        # plot_polygon(ax=ax, poly=grid.sample_area, facecolor="none", edgecolor='black')
        set_map_frame(ax=ax, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)
        fig.add_subplot(ax)
        fig.colorbar(sc, ax=ax, extend='neither', fraction=0.046, pad=0.04)

    if filename:
        fig.savefig(filename, dpi=300, bbox_inches="tight")
    if not show:
        _plt_close(fig)
    return fig
    #
#
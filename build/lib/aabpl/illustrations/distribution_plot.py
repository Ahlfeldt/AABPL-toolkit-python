from pandas import DataFrame as _pd_DataFrame
from numpy import (
    array as _np_array,
    linspace as _np_linspace,
    searchsorted as _np_searchsorted,
    spacing as _np_spacing,
    vstack as _np_vstack,
)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.pyplot import close as _plt_close
from matplotlib.colors import LogNorm as _plt_LogNorm, Normalize as _plt_Normalize, LinearSegmentedColormap as _plt_LinearSegmentedColormap
from matplotlib.pyplot import (subplots as _plt_subplots, colorbar as _plt_colorbar, get_cmap as _plt_get_cmap)
from aabpl.illustrations.plot_utils import add_color_bar_ax, set_map_frame, truncate_colormap

def create_distribution_plot(
        pts:_pd_DataFrame,
        rndm_pts:_pd_DataFrame,
        cluster_threshold_values:list,
        k_th_percentile:list,
        radius_sum_columns,
        r:float=None,
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
    disk_sums_for_random_points = rndm_pts[radius_sum_columns]
    (n_random_points, ncols) = disk_sums_for_random_points.shape
    # specify default plot kwargs and add defaults

    default_kwargs = {
        's':0.8,
        'color':'#eaa',

        'figsize': (10,10),
        'fig':None,
        'axs':None,
        
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
    axs = kwargs.pop('axs')

    if fig is None or axs is None:

        fig = plt.figure(figsize=figsize)
        outer = gridspec.GridSpec(ncols, 1, wspace=0.0, hspace=0.05)
    
    fig.suptitle(
        "Aggregate for indicator" + ("" if ncols==1 else "s") + 
        " within "+ str(r) +" meters"
    )

    xmin, xmax = 0, 100
    xs_random_pts = _np_linspace(xmin,xmax,n_random_points)
    xs_pts = _np_linspace(xmin,xmax,len(pts))
    random_vals = disk_sums_for_random_points.values
    pts_vals = pts[radius_sum_columns].values

    for (i, colname, cluster_threshold_value, k) in zip(
        range(ncols), radius_sum_columns, cluster_threshold_values, k_th_percentile):
        columns = gridspec.GridSpecFromSubplotSpec(1, 2,
                            subplot_spec=outer[i], wspace=0.15, hspace=0.0)
        
        left_col = gridspec.GridSpecFromSubplotSpec(2, 1,
                            subplot_spec=columns[0], wspace=0.0, hspace=0.15)
        right_col = gridspec.GridSpecFromSubplotSpec(2, 1,
                            subplot_spec=columns[1], wspace=0.0, hspace=0.15)
        
        # CUMULATIVE DISTRIBUTION RANDOM POINTS
        ys = sorted(random_vals[:,i])
        ymin, ymax = min(ys), max(ys)
        # round percentile value as far as necessary only 
        # e.g. threshold value is 10.5328... and next smaller/larger in distribution are 10.51..., 10.6... 
        # rounding to threshold value to firrst digit s.t. thath it lies between those value is sufficient (e.g. 10.53) 
        idx = _np_searchsorted(ys, cluster_threshold_value)
        next_smaller_val, next_larger_val = ys[max([0,idx-1])], ys[idx]
        sufficient_digits = next((
            i for i in range(100) if (
                (
                    (next_smaller_val == next_larger_val or cluster_threshold_values==next_smaller_val) and 
                    round(next_smaller_val,i)==next_smaller_val
                ) or (
                    round(next_larger_val, i) != round(cluster_threshold_value, i) and 
                    round(next_smaller_val, i) != round(cluster_threshold_value, i)
                )
        )),100)
        
        
        ax = plt.Subplot(fig, left_col[0])
        # SET TITLE
        ax_title = (
            colname+" "+str(k)+"th-percentile: "+
            str(round(cluster_threshold_value, sufficient_digits))
        )
        ax.set_title(ax_title, fontdict={'fontsize':6})

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
        ax.set_ylim([ymin,ymax])
        fig.add_subplot(ax)


        # CUMULATIVE DISTRIBUTION ORIGNAL POINTS
        ys = sorted(pts_vals[:,i])
        ymin, ymax = min(ys), max(ys)
        # SELECT AX (IF MULTIPLE)
        ax = plt.Subplot(fig, left_col[1])
        # SET TITLE
        ax.set_title("Distribution for points", fontdict={'fontsize':6})
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
        # ax.set_facecolor('#ccc')
        ax.scatter(x=xs_pts,y=ys, **plot_kwargs)
        # SET LIMITS
        ax.set_xlim([xmin,xmax])
        ax.set_ylim([ymin,ymax])
        fig.add_subplot(ax)

        # combine them and build a new colormap
        xmin, xmax =  min([pts[x_coord_name].min(), rndm_pts[x_coord_name].min()]), max([pts[x_coord_name].max(), rndm_pts[x_coord_name].max()])
        ymin, ymax =  min([pts[y_coord_name].min(), rndm_pts[y_coord_name].min()]), max([pts[y_coord_name].max(), rndm_pts[y_coord_name].max()])
        vmin = min([(pts_vals[:,i][pts_vals[:,i]!=0]).min(), (random_vals[:,i][random_vals[:,i]!=0]).min()])
        vmax = max([pts_vals[:,i].max(), random_vals[:,i].max()])
        cmap = _plt_get_cmap('Reds')
        cmap.set_under('#fff')
        cmap.set_over('blue')
        norm = _plt_LogNorm(vmin=vmin,vmax=cluster_threshold_value,clip=False) if vmin>0 else _plt_Normalize(vmin=vmin,vmax=cluster_threshold_value,clip=False)
        # SCATTER RANDOM POINTS
        ax = plt.Subplot(fig, right_col[0])
        # SET TITLE
        ax.set_title("Total within radius. "+str(len(rndm_pts))+" random points:", fontdict={'fontsize':6})
        # ADD DISTRIUBTION PLOT
        ax.set_facecolor('#ccc')
        sc = ax.scatter(x=rndm_pts[x_coord_name],y=rndm_pts[y_coord_name],c=random_vals[:,i], s=0.12, marker='.', norm=norm, cmap=cmap)
        # SET LIMITS
        set_map_frame(ax=ax,xmin=xmin,xmax=xmax,ymin=ymin,ymax=ymax)
        ax.set_xticks([]), ax.set_yticks([]), ax.set_aspect('equal')
        _plt_colorbar(sc, extend='both', cax=add_color_bar_ax(fig,ax))
        fig.add_subplot(ax)

        # SCATTER POINTS
        ax = plt.Subplot(fig, right_col[1])
        # SET TITLE
        ax.set_title("For "+str(len(pts))+" points:", fontdict={'fontsize':6})
        # ADD DISTRIUBTION PLOT
        ax.set_facecolor('#ccc')
        sc = ax.scatter(x=pts[x_coord_name],y=pts[y_coord_name],c=pts_vals[:,i], s=0.12, marker='.', norm=norm, cmap=cmap)
        # SET LIMITS
        set_map_frame(ax=ax,xmin=xmin,xmax=xmax,ymin=ymin,ymax=ymax)
        ax.set_xticks([]), ax.set_yticks([]), ax.set_aspect('equal')
        _plt_colorbar(sc, extend='both', cax=add_color_bar_ax(fig,ax))
        fig.add_subplot(ax)

    if filename:
        fig.savefig(filename, dpi=300, bbox_inches="tight")
    if close_plot:
        _plt_close(fig)
    return fig
    #
#
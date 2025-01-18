from pandas import DataFrame as _pd_DataFrame
from numpy import (
    array as _np_array,
    linspace as _np_linspace,
    searchsorted as _np_searchsorted,
    spacing as _np_spacing
)
from matplotlib.pyplot import (subplots as _plt_subplots, colorbar as _plt_colorbar, get_cmap as _plt_get_cmap)

def create_distribution_plot(
        pts:_pd_DataFrame,
        rndm_pts:_pd_DataFrame,
        cluster_threshold_values:list,
        k_th_percentiles:list,
        sum_radius_names,
        r=None,
        x:str='lon',
        y:str='lat',
        filename:str='',
        plot_kwargs:dict={},
):
    """
    TODO Descripiton
    """
    x_coord_name, y_coord_name = x,y
    disk_sums_for_random_points = rndm_pts[sum_radius_names]
    (n_random_points, ncols) = disk_sums_for_random_points.shape
    # specify default plot kwargs and add defaults
    default_kwargs = {
        's':0.8,
        'color':'#eaa',

        'figsize': (5*ncols,20),
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
        fig, axs = _plt_subplots(4,ncols, figsize=figsize)
    
    fig.suptitle(
        "Values for indicator" + ("" if ncols==1 else "s") + 
        ("" if r is None else (" within "+ str(r) +" distance")) +
        " around " + str(n_random_points)+ " randomly points drawn within valid area."
    )

    xmin, xmax = 0, 100
    xs_random_pts = _np_linspace(xmin,xmax,n_random_points)
    xs_pts = _np_linspace(xmin,xmax,len(pts))
    random_vals = disk_sums_for_random_points.values
    pts_vals = pts[sum_radius_names].values

    for (i, colname, cluster_threshold_value, k_th_percentile) in zip(
        range(ncols), sum_radius_names, cluster_threshold_values, k_th_percentiles):
        

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
        
        
        ax = axs.flat[i]
        
        # SET TITLE
        ax_title = (
            "Threshold value for "+str(k_th_percentile)+"th-percentile is "+
            str(round(cluster_threshold_value, sufficient_digits)) + " for "+ colname
        )
        ax.set_title(ax_title)

        # SET TICKS
        xtick_steps, ytick_steps = 5, 5
        xticks = _np_array(sorted(
           [x for x in _np_linspace(xmin,xmax,xtick_steps) if abs(x-k_th_percentile) > (xmax-xmin)/(xtick_steps*2)] + 
           [k_th_percentile]
        ))
        ax.set_xticks(xticks, labels=xticks)
        yticks = _np_array(sorted([y for y in _np_linspace(ymin,ymax,ytick_steps) if abs(cluster_threshold_value-y)>(ymax-ymin)/(ytick_steps*10)] + [cluster_threshold_value]))
        ax.set_yticks(yticks, labels=[round(t, sufficient_digits) for t in yticks])
        # ADD CUTOFF LINES
        ax.hlines(y=cluster_threshold_value, xmin=xmin,xmax=xmax, **kwargs['hlines'])
        ax.vlines(x=k_th_percentile, ymin=ymin,ymax=ymax, **kwargs['vlines'])
        # ADD DISTRIUBTION PLOT
        ax.scatter(x=xs_random_pts,y=ys, **plot_kwargs)
        # SET LIMITS
        ax.set_xlim([xmin,xmax])
        ax.set_ylim([ymin,ymax])



        # CUMULATIVE DISTRIBUTION ORIGNAL POINTS
        ys = sorted(pts_vals[:,i])
        ymin, ymax = min(ys), max(ys)
        # SELECT AX (IF MULTIPLE)
        ax = axs.flat[ncols+i]
        # SET TITLE
        ax.set_title("Distribution for point data for "+ colname)
        # SET TICKS
        xtick_steps, ytick_steps = 5, 5
        xticks = _np_array(sorted(
           [x for x in _np_linspace(xmin,xmax,xtick_steps) if abs(x-k_th_percentile) > (xmax-xmin)/(xtick_steps*2)] + 
           [k_th_percentile]
        ))
        ax.set_xticks(xticks, labels=xticks)
        yticks = _np_array(sorted([y for y in _np_linspace(ymin,ymax,ytick_steps) if abs(cluster_threshold_value-y)>(ymax-ymin)/(ytick_steps*10)] + [cluster_threshold_value]))
        ax.set_yticks(yticks, labels=[round(t, sufficient_digits) for t in yticks])
        # # ADD CUTOFF LINES
        # ax.hlines(y=cluster_threshold_value, xmin=xmin,xmax=xmax, **kwargs.pop('hlines'))
        # ax.vlines(x=k_th_percentile, ymin=ymin,ymax=ymax, **kwargs.pop('vlines'))
        ax.hlines(y=cluster_threshold_value, xmin=xmin,xmax=xmax, **kwargs['hlines'])
        ax.vlines(x=k_th_percentile, ymin=ymin,ymax=ymax, **kwargs['vlines'])
        # ADD DISTRIUBTION PLOT
        ax.scatter(x=xs_pts,y=ys, **plot_kwargs)
        # SET LIMITS

        ax.set_xlim([xmin,xmax])
        ax.set_ylim([ymin,ymax])
        
        
        xmin, xmax =  min([pts[x_coord_name].min(), rndm_pts[x_coord_name].min()]), max([pts[x_coord_name].max(), rndm_pts[x_coord_name].max()])
        ymin, ymax =  min([pts[y_coord_name].min(), rndm_pts[y_coord_name].min()]), max([pts[y_coord_name].max(), rndm_pts[y_coord_name].max()])
        vmin, vmax = min([pts_vals[:,i].min(), random_vals[:,i].min()]), max([pts_vals[:,i].max(), random_vals[:,i].max()])
        cmap = _plt_get_cmap('Reds')
        cmap.set_under('#ccc')
        # SCATTER RANDOM POINTS
        ax = axs.flat[2*ncols+i]
        # SET TITLE
        ax.set_title("Radius sums for random points for " + colname)
        # ADD DISTRIUBTION PLOT
        ax.set_facecolor('#ccc')
        sc = ax.scatter(x=rndm_pts[x_coord_name],y=rndm_pts[y_coord_name],c=random_vals[:,i], s=0.01, vmin=_np_spacing(0.0), vmax=vmax, cmap=cmap)
        _plt_colorbar(sc, extend='min')
        # SET LIMITS
        ax.set_xlim([xmin,xmax])
        ax.set_ylim([ymin,ymax])

        # SCATTER POINTS
        ax = axs.flat[3*ncols+i]
        # SET TITLE
        ax.set_title("Radius sums for points for " + colname)
        # ADD DISTRIUBTION PLOT
        ax.set_facecolor('#ccc')
        sc = ax.scatter(x=pts[x_coord_name],y=pts[y_coord_name],c=pts_vals[:,i], s=0.01, vmin=_np_spacing(0.0), vmax=vmax, cmap=cmap)
        _plt_colorbar(sc, extend='min')
        # SET LIMITS
        ax.set_xlim([xmin,xmax])
        ax.set_ylim([ymin,ymax])

    if filename:
        fig.savefig(filename, dpi=300, bbox_inches="tight")

    #
#
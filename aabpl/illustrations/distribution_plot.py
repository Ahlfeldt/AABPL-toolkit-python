from pandas import DataFrame as _pd_DataFrame
from numpy import (
    array as _np_array,
    linspace as _np_linspace,
    searchsorted as _np_searchsorted
)
from matplotlib.pyplot import (subplots as _plt_subplots)

def create_distribution_plot(
        sums_df:_pd_DataFrame,
        cluster_threshold_values:list,
        k_th_percentiles:list,
        radius=None,
        plot_kwargs:dict={},
):
    """
    TODO Descripiton
    """
    (n_random_points, ncols) = sums_df.shape
    # specify default plot kwargs and add defaults
    default_kwargs = {
        's':0.8,
        'color':'#eaa',

        'figsize': (5*ncols,5),
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
        fig, axs = _plt_subplots(1,ncols, figsize=figsize)
    
    fig.suptitle(
        "Values for indicator" + ("" if ncols==1 else "s") + 
        ("" if radius is None else (" within "+ str(radius) +" distance")) +
        " around " + str(n_random_points)+ " randomly points drawn within valid area."
    )

    xmin, xmax = 0, 100
    xs = _np_linspace(xmin,xmax,n_random_points)
    vals = sums_df.values
    colnames = sums_df.columns

    for (i, colname, cluster_threshold_value, k_th_percentile) in zip(
        range(ncols), colnames, cluster_threshold_values, k_th_percentiles):
        ys = vals[:,i]
        ys.sort()
        ymin, ymax = ys.min(),ys.max()
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
        
        ax_title = (
            "Threshold value for "+str(k_th_percentile)+"th-percentile is "+
            str(round(cluster_threshold_value, sufficient_digits)) + " for "+ colname
        )
        
        # SELECT AX (IF MULTIPLE)
        ax = axs.flat[i] if ncols > 1 else axs
        
        # SET TITLE
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
        ax.hlines(y=cluster_threshold_value, xmin=xmin,xmax=xmax, **kwargs.pop('hlines'))
        ax.vlines(x=k_th_percentile, ymin=ymin,ymax=ymax, **kwargs.pop('vlines'))
        
        # ADD DISTRIUBTION PLOT
        ax.scatter(x=xs,y=ys, **plot_kwargs)
        
        # SET LIMITS
        ax.set_xlim([xmin,xmax])
        ax.set_ylim([ymin,ymax])
    #
#
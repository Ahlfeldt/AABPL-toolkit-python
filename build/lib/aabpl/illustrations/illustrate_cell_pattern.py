from numpy import (
    array as _np_array, 
    unique as _np_unique, 
    linspace, invert, flip, transpose, 
    concatenate, 
    sign as _np_sign, 
    zeros, min, max, equal, where, 
    logical_or, logical_and, all, newaxis
)
from math import ceil as _math_ceil
from pandas import DataFrame as _pd_DataFrame
from matplotlib.pyplot import (subplots as _plt_subplots, figure as _plt_figure)
from matplotlib.patches import Rectangle as _plt_Rectangle
from ..utils.general import ( flatten_list, )


def plot_cell_pattern(
    contained_cells, 
    overlapped_cells, 
    all_cells,
    radius:float,
    grid_spacing:float, 
    add_idxs:bool=True,
    **plot_kwargs,
):

    """
    Illustrate method
    """
    # specify default plot kwargs and add defaults
    plot_kwargs = {
        'fig':None,
        'ax':None,
        's':0.8,
        'color':'#eaa',
        'figsize': (20,30),
        **plot_kwargs
    }
    figsize = plot_kwargs.pop('figsize')
    fig = plot_kwargs.pop('fig')
    ax = plot_kwargs.pop('ax')
    ###### initialize plot  ######################

    if ax is None:
        fig, ax = _plt_subplots(1,1, figsize=figsize)
    ################################################################################################################
    colors = ['#ccc', 'green', 'red']
    for (row, col), color in flatten_list(
        [[((row, col), color) for row, col in cells] for cells,color in zip(
        [all_cells, contained_cells, overlapped_cells],
        colors)]):
        ax.add_patch(_plt_Rectangle(
            xy = (col-0.5, row-0.5), 
            width=1, height=1, 
            linewidth=.7, facecolor=color, edgecolor=color, alpha=0.3
        ))
        if add_idxs and color == colors[0]:
            ax.annotate(text=str(row)+","+str(col), xy=(col,row),horizontalalignment='center')

    ratio = radius/grid_spacing
    cell_steps_max = _math_ceil(ratio+1.5)

    ax.set_xlim((-cell_steps_max,+cell_steps_max))
    ax.set_ylim((-cell_steps_max,+cell_steps_max))
    ax.set_aspect('equal', adjustable='box')
    #
#

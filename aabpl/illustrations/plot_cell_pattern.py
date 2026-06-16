from math import ceil as _math_ceil
from matplotlib.pyplot import (subplots as _plt_subplots, figure as _plt_figure)
from matplotlib.patches import Circle as _plt_Circle, Rectangle as _plt_Rectangle, Polygon as _plt_Polygon
from ..utils.misc import flatten_list


def plot_cell_pattern(
    cntd_cells, 
    ovlpd_cells, 
    all_cells,
    r:float,
    grid_spacing:float,
    nested_cntd_cells=[],
    nested_ovlpd_cells=[],
    region_coords:list=None, 
    add_idxs:bool=True,
    hatch='',
    **plot_kwargs,
):

    """
    Illustrate method
    region_coords: list of tuples marking coords for which to draw a circle around
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
    # add circle patches
    for region_coord in region_coords:
        ax.add_patch(
            _plt_Circle(
                xy=region_coord, radius=r,
                facecolor=('#000000'+(str(int(60/len(region_coords))) if int(60/len(region_coords))>=10 else '0'+str(int(60/len(region_coords))))),
                edgecolor='#0006', linewidth=0.25))
    colors = ['#ccc', "#9ac991ff", "#fdaba3ff"]
    if len(nested_cntd_cells)+len(nested_ovlpd_cells)>0:
        ovlpd_cells = []
    [[((lvl,(row, col)), color) for lvl,(row, col) in cells] for cells,color in zip(
        [all_cells, cntd_cells, ovlpd_cells],
        colors)]
    for (lvl,(row, col)), color in flatten_list(
        [[((lvl,(row, col)), color) for lvl,(row, col) in cells] for cells,color in zip(
        [all_cells, cntd_cells, ovlpd_cells],
        colors)]):
        lvl=lvl
        ax.add_patch(_plt_Rectangle(
            xy = (float(col)-2**-lvl, float(row)-2**-lvl), 
            width=2**(1-lvl), height=2**(1-lvl), 
            linewidth=.7, facecolor=color, edgecolor=color, alpha=1, 
            **({} if hatch =='' else {'hatch':hatch})
        ))
        
        if add_idxs and color == colors[0]:
            ax.annotate(text=str(row)+","+str(col), xy=(col,row),horizontalalignment='center')
    
    colors = ["#1d802a", "#a51414"]
    for (lvl,(row, col)), color in flatten_list(
        [[((lvl,(row, col)), color) for lvl, (row, col)  in cells] for cells,color in zip(
        [nested_cntd_cells, nested_ovlpd_cells],
        colors)]):
        xy = (float(col)-2**(-lvl-1), float(row)-2**(-lvl-1))
        ax.add_patch(_plt_Rectangle(
            xy = xy, 
            width=2**(-lvl), height=2**(-lvl), 
            linewidth=.7, facecolor=color, edgecolor=color, alpha=0.3, 
            **({} if hatch =='' else {'hatch':hatch})
        ))
    ax.add_patch(_plt_Rectangle(
            xy = (-.5,-.5), 
            width=1, height=1, 
            linewidth=.7, facecolor='white', edgecolor='black', alpha=0.3, 
            **({} if hatch =='' else {'hatch':hatch})
        ))
    
    
    # add region polygon
    ax.add_patch(
        _plt_Polygon(
            xy=region_coords,
            facecolor="#FFFFFF",
            edgecolor='#000', linewidth=0.25))
        
    ratio = r/grid_spacing
    cell_steps_max = _math_ceil(ratio+1.5)

    ax.set_xlim((-cell_steps_max,+cell_steps_max))
    ax.set_ylim((-cell_steps_max,+cell_steps_max))
    ax.set_aspect('equal', adjustable='box')
    #
#

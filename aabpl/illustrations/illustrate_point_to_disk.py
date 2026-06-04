from numpy import (
    array as _np_array, 
    unique as _np_unique, 
    linspace, invert, flip, transpose, 
    concatenate, 
    sign as _np_sign, 
    zeros, min, max, equal, where, 
    logical_or, logical_and, all, newaxis
)
from pandas import DataFrame as _pd_DataFrame
from matplotlib.pyplot import (subplots as _plt_subplots, figure as _plt_figure)
from matplotlib.patches import Circle as _plt_Circle, Rectangle as _plt_Rectangle, Polygon as _plt_Polygon
from matplotlib.figure import Figure as _plt_Figure
from matplotlib.axes._axes import Axes as _plt_Axes
from aabpl.utils.general import ( flatten_list, )


def illustrate_point_disk(
    grid:dict,
    shared_contained_cells:list,
    shared_overlapped_cells:list,
    distinct_contained_cells:list,
    distinct_overlapped_cells:list,
    pts_xy_in_cells_overlapped_by_pt_region:_np_array,
    pts_xy_in_radius:_np_array,
    pts_xy_in_cell_contained_by_pt_region:list,
    pts_source:_pd_DataFrame,
    pts_target:_pd_DataFrame,
    region_id,
    home_cell:tuple,
    r:float=750,
    sum_names:list=['employment'],
    y:str='proj_lat',
    x:str='proj_lon',
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
    pt_id = plot_kwargs.pop('pt_id')
    # print("row:",pts_source.loc[pt_id])
    pt_x, pt_y = pts_source.loc[pt_id,[x,y]]
    # home_cell = grid.pt_id_to_row_col[pt_id]
    home_cell_centroid = grid.get_cell_centroid(*home_cell)
    hc_x,hc_y = home_cell_centroid
    ###### initialize plot  ######################

    if fig is None:
        fig, axs = _plt_subplots(1,1, figsize=figsize)
    elif type(fig) != _plt_Figure:
        raise TypeError
    # [(cells_overlapped_by_pt_region, pt_in_radius) for pt_maybe_in_radius, pt_in_radius in zip(cells_overlapped_by_pt_region, pts_in_radius)]
    # grid.search.target.
    ################################################################################################################
    ax = axs#[0]
    # print("(pt_x, pt_y)",(pt_x, pt_y))
    # print("cells", [ ((c-.5)*grid.spacing+grid.total_bounds.xmin, (row-.5)*grid.spacing+grid.total_bounds.ymin) for row,c in cells_cntd_by_pt_cell])
    # print(
    #     [[(grid.get_cell_centroid(row,col), color) for lvl,(row,col) in cells] for cells,color in zip(
    #     [cells_cntd_by_pt_cell, cells_contained_by_pt_region, cells_overlapped_by_pt_region],
    #     ['blue','green', 'red'])])
    # print(flatten_list(
    #     [[(grid.get_cell_centroid(row,col), color) for lvl,(row,col) in cells] for cells,color in zip(
    #     [cells_cntd_by_pt_cell, cells_contained_by_pt_region, cells_overlapped_by_pt_region],
    #     ['blue','green', 'red'])]))
    # print("distinct_overlapped_cells",distinct_overlapped_cells)
    for (lvl,(cntrd_x,cntrd_y)), color, hatch in flatten_list(
        [[((lvl, grid.get_cell_centroid(row,col)), color, hatch) for lvl,(row,col) in cells] for cells,color,hatch in zip(
        [shared_contained_cells, distinct_contained_cells, shared_overlapped_cells,distinct_overlapped_cells],
        ['blue','green','orange', 'red'], range(4))]):
        # print("cntrd",cntrd)
        # print("grid.spacing",grid.spacing)
        # print("cntrd -( .5) * grid.spacing",(cntrd[0] -( .5) * grid.spacing, cntrd[1] -( .5) * grid.spacing))
        xy = (cntrd_x-2**(-lvl-1)*grid.spacing, cntrd_y-2**(-lvl-1)*grid.spacing)
        # print("+xy",xy)
        hatches = ['*', '\\', '-', '/', '+', 'x', 'o', 'O', '.', '*']
        
        
        ax.add_patch(_plt_Rectangle(
            xy = xy, 
            hatch=hatches[(hatch+3*lvl)%len(hatches)],
            width=2**(-lvl)*grid.spacing, height=2**(-lvl)*grid.spacing, 
            linewidth=.7, facecolor=color, edgecolor=color, alpha=0.3
        ))
        ax.add_patch(_plt_Rectangle(
            xy = xy, 
            width=2**(-lvl)*grid.spacing, height=2**(-lvl)*grid.spacing, 
            linewidth=.7, facecolor='None', edgecolor=color, alpha=0.8
        ))
        # if lvl==1:
        #     ax.annotate(text=str((
        #     float(round((cntrd_x-grid.total_bounds.xmin)/grid.spacing-home_cell[1],5)),
        #     float(round((cntrd_y-grid.total_bounds.ymin)/grid.spacing-home_cell[0],5)),
        #     )), xy=xy, 
        #         horizontalalignment='center',
        #         backgroundcolor="#ffffff88",)
    cntrd_color = flatten_list(
        [[(grid.get_cell_centroid(row,col), color, lvl) for lvl,(row,col) in cells] for cells,color in zip(
        [shared_contained_cells, distinct_contained_cells, shared_overlapped_cells,distinct_overlapped_cells],
        ['blue','green','orange', 'red'])])
    ax.scatter(
        x=[cntrd[0] for cntrd,color, lvl in cntrd_color],
        y =[cntrd[1] for cntrd,color, lvl in cntrd_color],
        s=[fig.get_figheight()*300*(2**-lvl) for cntrd,color, lvl in cntrd_color], 
        linewidths=4,
    c=[color for cntrd,color,lvl in cntrd_color], marker='+', alpha=0.1)
    # ax.scatter(
    #     x=[cntrd[0] for cntrd,color, lvl in cntrd_color],
    #     y =[cntrd[1] for cntrd,color, lvl in cntrd_color],
    #     s=fig.get_figheight()*500, c=[color for cntrd,color,lvl in cntrd_color], marker='+', alpha=0.1)
    # flat_list = flatten_list(
    #     [[(grid.get_cell_centroid(row,col), color) for lvl,(row,col) in cells] for cells,color in zip(
    #     [nested_cells_contained_by_pt_region, nested_cells_overlapped_by_pt_region],
    #     ["#1d802a", "#a51414"])])
    # print("flat_list",flat_list)

    # add_grid_cell_rectangles_by_color(
    #     [cells_cntd_by_pt_cell, cells_contained_by_pt_region, cells_overlapped_by_pt_region],
    #     ['blue','green', 'red'],
    #     ax=ax, grid_spacing=grid.spacing,
    #     x_off=grid.total_bounds.xmin+grid.spacing/2,
    #     y_off=grid.total_bounds.ymin+grid.spacing/2,
    # )
    offset_region = grid.id_to_offset_regions[region_id]
    
    region_coords = offset_region.get_plot_coords()
    for region_x, region_y in region_coords:
        ax.add_patch(
            _plt_Circle(
                xy=(region_x*grid.spacing+hc_x, region_y*grid.spacing + hc_y), radius=r,
                facecolor="#000000"+(str(int(60/len(region_coords))) if int(60/len(region_coords))>=10 else '0'+str(int(60/len(region_coords)))),
                edgecolor='#0006', linewidth=0.25))
    ax.add_patch(_plt_Polygon(
        [(region_x*grid.spacing+hc_x, region_y*grid.spacing + hc_y) for region_x, region_y in region_coords], 
        facecolor="#000000",))
    ax.add_patch(_plt_Circle(xy=(pt_x, pt_y), radius=r, facecolor="#0000ff16",edgecolor='#00f',linewidth=2,))
    ax.add_patch(_plt_Circle(xy=(pt_x, pt_y), radius=r/40, alpha=0.6))
    # ax.add_patch(create_buffered_square_patch(side_length=grid.spacing, r=r, x_off=hc_x, y_off=hc_y))
    # ax.add_patch(create_debuffered_square_patch(side_length=grid.spacing, r=r, linewidth=2, x_off=hc_x, y_off=hc_y ))
    
    # ax.add_patch(create_trgl1_patch(side_length=grid.spacing/2, linewidth=2, x_off=hc_x, y_off=hc_y ))
    # ax.add_patch(create_buffered_trgl1_patch(side_length=grid.spacing/2, linewidth=2, x_off=hc_x, y_off=hc_y ))
    # ax.add_patch(create_debuffered_trgl1_patch(side_length=grid.spacing/2, linewidth=2, x_off=hc_x, y_off=hc_y ))
    # print('+++++', [(cells,color) for cells,color in zip(
    #     [cells_cntd_by_pt_cell, cells_contained_by_pt_region, cells_overlapped_by_pt_region],
    #     ['blue','green', 'red'])])
    # print('+++++', [[(grid.get_cell_centroid(row,col), color) for (row,col) in cells] for cells,color in zip(
    #     [cells_cntd_by_pt_cell, cells_contained_by_pt_region, cells_overlapped_by_pt_region],
    #     ['blue','green', 'red'])])

    # all pts
    ax.scatter(
        x=pts_target[x],
        y =pts_target[y],
        # c=pts_target['sc_nr'],
        # s=fig.get_figheight()/.2, cmap='viridis', marker='x')
        s=fig.get_figheight()/1, color='#777', marker='x')
    # pts in contained cells
    ax.scatter(
        x=pts_xy_in_cell_contained_by_pt_region[:,0],
        y =pts_xy_in_cell_contained_by_pt_region[:,1],
        s=fig.get_figheight()/2, color='yellow', marker='o')
    # pts in overlapped cells
    ax.scatter(
        x=pts_xy_in_cells_overlapped_by_pt_region[:,0],
        y =pts_xy_in_cells_overlapped_by_pt_region[:,1],
        s=fig.get_figheight()/2, color='red', marker='+')
    # pts in overlapped cells inside r
    ax.scatter(
        x=pts_xy_in_radius[:,0],
        y =pts_xy_in_radius[:,1],
        s=fig.get_figheight()/2, color='black', marker='o')
    
    # for (i, ax) in enumerate(axs):
    ax.set_xlim(pt_x-1.35*r,pt_x+1.35*r)
    ax.set_ylim(pt_y-1.35*r,pt_y+1.35*r)
    ax.set_aspect('equal', adjustable='box')
    #
#

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
from matplotlib.patches import Circle as _plt_circle, Rectangle as _plt_Rectangle
from matplotlib.figure import Figure as _plt_Figure
from matplotlib.axes._axes import Axes as _plt_Axes
from .plot_utils import (
    create_grid_cell_patches,  create_grid_cell_patches_by_type,  create_grid_cell_rectangles, 
    create_trgl1_patch,  create_buffered_trgl1_patch,  create_buffered_square_patch, create_debuffered_square_patch, create_debuffered_trgl1_patch, dual_circle_union_patch,
)
from .illustrate_point_to_cell_region_assignment import (add_grid_cell_rectangles_by_color,add_circle_patches,)
from aabpl.utils.distances_to_cell import ( get_cells_relevant_for_disk_by_type, get_cell_farthest_vertex_to_point,  )
from aabpl.utils.general import ( flatten_list, )


def illustrate_point_disk(
    grid:dict,
    cells_cntd_by_pt_cell:list,
    cells_contained_by_pt_region:list,
    cells_overlapped_by_pt_region:list,
    pts_in_cells_overlapped_by_pt_region:list,
    pts_in_radius:list,
    pts_in_cell_contained_by_pt_region:list,
    pts_df_search_source:_pd_DataFrame,
    pts_df_search_target:_pd_DataFrame,
    radius:float=750,
    sum_names:list=['employment'],
    y_coord_name:str='lat',
    x_coord_name:str='lon',
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
    print("row:",pts_df_search_source.loc[pt_id])
    print("cells_cntd_by_pt_cell", cells_cntd_by_pt_cell)
    print("cells_contained_by_pt_region", cells_contained_by_pt_region)
    print("cells_overlapped_by_pt_region", cells_overlapped_by_pt_region)
    pt_x, pt_y = pts_df_search_source.loc[pt_id,[x_coord_name,y_coord_name]]
    home_cell = grid.pt_id_to_row_col[pt_id]
    home_cell_centroid = grid.row_col_to_centroid[home_cell]
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
    # print("cells", [ ((c-.5)*grid.spacing+grid.total_bounds.xmin, (r-.5)*grid.spacing+grid.total_bounds.ymin) for r,c in cells_cntd_by_pt_cell])
    # print(
    #     [[(grid.row_col_to_centroid[cell], color) for cell in cells] for cells,color in zip(
    #     [cells_cntd_by_pt_cell, cells_contained_by_pt_region, cells_overlapped_by_pt_region],
    #     ['blue','green', 'red'])])
    # print(flatten_list(
    #     [[(grid.row_col_to_centroid[cell], color) for cell in cells] for cells,color in zip(
    #     [cells_cntd_by_pt_cell, cells_contained_by_pt_region, cells_overlapped_by_pt_region],
    #     ['blue','green', 'red'])]))
    for cntrd, color in flatten_list(
        [[(grid.row_col_to_centroid[cell], color) for cell in cells] for cells,color in zip(
        [cells_cntd_by_pt_cell, cells_contained_by_pt_region, cells_overlapped_by_pt_region],
        ['blue','green', 'red'])]):
        # print("cntrd",cntrd)
        # print("grid.spacing",grid.spacing)
        # print("cntrd -( .5) * grid.spacing",(cntrd[0] -( .5) * grid.spacing, cntrd[1] -( .5) * grid.spacing))
        ax.add_patch(_plt_Rectangle(
            xy = (cntrd[0] -( .5) * grid.spacing, cntrd[1] -( .5) * grid.spacing), 
            width=grid.spacing, height=grid.spacing, 
            linewidth=.7, facecolor=color, edgecolor=color, alpha=0.3
        ))
    
    # add_grid_cell_rectangles_by_color(
    #     [cells_cntd_by_pt_cell, cells_contained_by_pt_region, cells_overlapped_by_pt_region],
    #     ['blue','green', 'red'],
    #     ax=ax, grid_spacing=grid.spacing,
    #     x_off=grid.total_bounds.xmin+grid.spacing/2,
    #     y_off=grid.total_bounds.ymin+grid.spacing/2,
    # )
    ax.add_patch(_plt_circle(xy=(pt_x, pt_y), radius=radius, facecolor='#00f3',edgecolor='#00f',linewidth=2,))
    ax.add_patch(_plt_circle(xy=(pt_x, pt_y), radius=radius/20, alpha=1))
    # ax.add_patch(create_buffered_square_patch(side_length=grid.spacing, radius=radius, x_off=hc_x, y_off=hc_y))
    # ax.add_patch(create_debuffered_square_patch(side_length=grid.spacing, radius=radius, linewidth=2, x_off=hc_x, y_off=hc_y ))
    
    # ax.add_patch(create_trgl1_patch(side_length=grid.spacing/2, linewidth=2, x_off=hc_x, y_off=hc_y ))
    # ax.add_patch(create_buffered_trgl1_patch(side_length=grid.spacing/2, linewidth=2, x_off=hc_x, y_off=hc_y ))
    # ax.add_patch(create_debuffered_trgl1_patch(side_length=grid.spacing/2, linewidth=2, x_off=hc_x, y_off=hc_y ))
    cntrd_color = flatten_list(
        [[(grid.row_col_to_centroid[cell], color) for cell in cells] for cells,color in zip(
        [cells_cntd_by_pt_cell, cells_contained_by_pt_region, cells_overlapped_by_pt_region],
        ['blue','green', 'red'])])
    ax.scatter(
        x=[cntrd[0] for cntrd,color in cntrd_color],
        y =[cntrd[1] for cntrd,color in cntrd_color],
        s=fig.get_figheight()*500, c=[color for cntrd,color in cntrd_color], marker='+', alpha=0.1)
    
    # all pts
    ax.scatter(
        x=pts_df_search_target[x_coord_name],
        y =pts_df_search_target[y_coord_name],
        s=fig.get_figheight()/1, color='#777', marker='x')
    # pts in contained cells
    ax.scatter(
        x=pts_df_search_target.loc[pts_in_cell_contained_by_pt_region, x_coord_name],
        y =pts_df_search_target.loc[pts_in_cell_contained_by_pt_region, y_coord_name],
        s=fig.get_figheight()/2, color='yellow', marker='o')
    # pts in overlapped cells
    ax.scatter(
        x=pts_df_search_target.loc[pts_in_cells_overlapped_by_pt_region, x_coord_name],
        y =pts_df_search_target.loc[pts_in_cells_overlapped_by_pt_region, y_coord_name],
        s=fig.get_figheight()/2, color='red', marker='+')
    # pts in overlapped cells inside radius
    ax.scatter(
        x=pts_df_search_target.loc[pts_in_radius, x_coord_name],
        y =pts_df_search_target.loc[pts_in_radius, y_coord_name],
        s=fig.get_figheight()/2, color='black', marker='o')
    
    # for (i, ax) in enumerate(axs):
    ax.set_xlim(pt_x-1.35*radius,pt_x+1.35*radius)
    ax.set_ylim(pt_y-1.35*radius,pt_y+1.35*radius)
    ax.set_aspect('equal', adjustable='box')
    #
#

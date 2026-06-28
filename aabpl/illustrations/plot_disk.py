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
from aabpl.utils.misc import ( flatten_list, )


def illustrate_point_disk(
    grid:dict,
    shared_cntd_cells:list,
    shared_ovlpd_cells:list,
    distinct_cntd_cells:list,
    distinct_ovlpd_cells:list,
    pts_xy_in_cells_ovlpd_by_pt_region:_np_array,
    pts_xy_in_radius:_np_array,
    pts_xy_in_cell_cntd_by_pt_region:list,
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
    sums_within_disk = plot_kwargs.pop('sums_within_disk', None)
    display_cell_region_id = plot_kwargs.pop('cell_region_id', region_id)
    # optional output path (accepts savevig / savefig / filename); popped so it
    # does not leak into the scatter/patch kwargs below.
    save_path = (plot_kwargs.pop('savevig', None) or plot_kwargs.pop('savefig', None)
                 or plot_kwargs.pop('filename', None))
    plot_kwargs.pop('nest_depth', None)
    # print("row:",pts_source.loc[pt_id])
    pt_x, pt_y = pts_source.loc[pt_id,[x,y]]
    # home_cell = grid.pt_id_to_row_col[pt_id]
    home_cell_centroid = grid._search_internals.cell_centroid(*home_cell)
    hc_x,hc_y = home_cell_centroid
    ###### initialize plot  ######################
    colors = ["#1625ff","#12b400","#ff8c00", '#ff0000']
    if fig is None:
        fig, axs = _plt_subplots(1,1, figsize=figsize)
    elif type(fig) != _plt_Figure:
        raise TypeError
    # [(cells_ovlpd_by_pt_region, pt_in_radius) for pt_maybe_in_radius, pt_in_radius in zip(cells_ovlpd_by_pt_region, pts_in_radius)]
    # grid.search.target.
    ################################################################################################################
    ax = axs#[0]
    # print("(pt_x, pt_y)",(pt_x, pt_y))
    # print("cells", [ ((c-.5)*grid._search_internals.spacing+grid.total_bounds.xmin, (row-.5)*grid._search_internals.spacing+grid.total_bounds.ymin) for row,c in cells_cntd_by_pt_cell])
    # print(
    #     [[(grid.get_cell_centroid(row,col), color) for lvl,(row,col) in cells] for cells,color in zip(
    #     [cells_cntd_by_pt_cell, cells_cntd_by_pt_region, cells_ovlpd_by_pt_region],
    #     ['blue','green', 'red'])])
    # print(flatten_list(
    #     [[(grid.get_cell_centroid(row,col), color) for lvl,(row,col) in cells] for cells,color in zip(
    #     [cells_cntd_by_pt_cell, cells_cntd_by_pt_region, cells_ovlpd_by_pt_region],
    #     ['blue','green', 'red'])]))
    # print("distinct_ovlpd_cells",distinct_ovlpd_cells)
    # len(shared_cntd_cells), len(distinct_cntd_cells), etc.
    def _cell_centroid(lvl, row, col):
        # All levels: centroid = xmin + (col + 0.5) * spacing.
        # lvl=0: col is an integer grid index, so +0.5 gives the cell centre.
        # lvl>0: the anchor in check_nested_subcells is set as (index - 0.5), so
        #        the stored fractional coord is also 0.5 below the true centre;
        #        adding 0.5 corrects this for all nesting levels uniformly.
        return (
            grid._search_internals.bounds.xmin + (col + 0.5) * grid._search_internals.spacing,
            grid._search_internals.bounds.ymin + (row + 0.5) * grid._search_internals.spacing,
        )

    for (lvl,(cntrd_x,cntrd_y)), color, hatch in flatten_list(
        [[((lvl, _cell_centroid(lvl, row, col)), color, hatch) for lvl,(row,col) in cells] for cells,color,hatch in zip(
        [shared_cntd_cells, distinct_cntd_cells, shared_ovlpd_cells,distinct_ovlpd_cells],
        colors, range(4))]):
        # print("cntrd",cntrd)
        # print("grid._search_internals.spacing",grid._search_internals.spacing)
        # print("cntrd -( .5) * grid._search_internals.spacing",(cntrd[0] -( .5) * grid._search_internals.spacing, cntrd[1] -( .5) * grid._search_internals.spacing))
        xy = (cntrd_x-2**(-lvl-1)*grid._search_internals.spacing, cntrd_y-2**(-lvl-1)*grid._search_internals.spacing)
        # print("+xy",xy)
        # hatches = ['*', '\\', '-', '/', '+', 'x', 'o', 'O', '.', '*']
        hatches = ['', '\\', '/', 'o', '.', 'x', '*']
        
        
        ax.add_patch(_plt_Rectangle(
            xy = xy, 
            hatch=hatches[(hatch*0+1*lvl)%len(hatches)],
            width=2**(-lvl)*grid._search_internals.spacing, height=2**(-lvl)*grid._search_internals.spacing, 
            linewidth=.7, facecolor=color, edgecolor=color, alpha=0.3
        ))
        ax.add_patch(_plt_Rectangle(
            xy = xy, 
            width=2**(-lvl)*grid._search_internals.spacing, height=2**(-lvl)*grid._search_internals.spacing, 
            linewidth=.7, facecolor='None', edgecolor=color, alpha=0.8
        ))
        # if lvl==1:
        #     ax.annotate(text=str((
        #     float(round((cntrd_x-grid.total_bounds.xmin)/grid._search_internals.spacing-home_cell[1],5)),
        #     float(round((cntrd_y-grid.total_bounds.ymin)/grid._search_internals.spacing-home_cell[0],5)),
        #     )), xy=xy, 
        #         horizontalalignment='center',
        #         backgroundcolor="#ffffff88",)
    cntrd_color = flatten_list(
        [[(_cell_centroid(lvl, row, col), color, lvl) for lvl,(row,col) in cells] for cells,color in zip(
        [shared_cntd_cells, distinct_cntd_cells, shared_ovlpd_cells,distinct_ovlpd_cells],
        colors)])
    ax.scatter(
        x=[cntrd[0] for cntrd,color, lvl in cntrd_color],
        y =[cntrd[1] for cntrd,color, lvl in cntrd_color],
        s=[fig.get_figheight()*250*(2**-lvl) for cntrd,color, lvl in cntrd_color], 
        linewidths=[2*(2**-lvl) for cntrd,color, lvl in cntrd_color],
        c=[color for cntrd,color,lvl in cntrd_color], marker='+', alpha=0.1)
    # ax.scatter(
    #     x=[cntrd[0] for cntrd,color, lvl in cntrd_color],
    #     y =[cntrd[1] for cntrd,color, lvl in cntrd_color],
    #     s=fig.get_figheight()*500, c=[color for cntrd,color,lvl in cntrd_color], marker='+', alpha=0.1)
    # flat_list = flatten_list(
    #     [[(grid.get_cell_centroid(row,col), color) for lvl,(row,col) in cells] for cells,color in zip(
    #     [nested_cells_cntd_by_pt_region, nested_cells_ovlpd_by_pt_region],
    #     ["#1d802a", "#a51414"])])
    # print("flat_list",flat_list)

    # add_grid_cell_rectangles_by_color(
    #     [cells_cntd_by_pt_cell, cells_cntd_by_pt_region, cells_ovlpd_by_pt_region],
    #     ['blue','green', 'red'],
    #     ax=ax, grid_spacing=grid._search_internals.spacing,
    #     x_off=grid.total_bounds.xmin+grid._search_internals.spacing/2,
    #     y_off=grid.total_bounds.ymin+grid._search_internals.spacing/2,
    # )
    # Offset-region overlay (the family of disk centres for the point's region).
    # region_id is keyed differently from cell_region; draw it only when available.
    offset_region = grid._search_internals.id_to_offset_regions.get(region_id) if region_id is not None else None
    if offset_region is not None:
        region_coords = offset_region.get_plot_coords()
        for region_x, region_y in region_coords:
            ax.add_patch(
                _plt_Circle(
                    xy=(region_x*grid._search_internals.spacing+hc_x, region_y*grid._search_internals.spacing + hc_y), radius=r,
                    facecolor="#000000"+(str(int(60/len(region_coords))) if int(60/len(region_coords))>=10 else '0'+str(int(60/len(region_coords)))),
                    edgecolor='#0006', linewidth=0.25))
        ax.add_patch(_plt_Polygon(
            [(region_x*grid._search_internals.spacing+hc_x, region_y*grid._search_internals.spacing + hc_y) for region_x, region_y in region_coords],
            facecolor="#000000",))
    ax.add_patch(_plt_Circle(xy=(pt_x, pt_y), radius=r, facecolor="#0000ff16",edgecolor='#00f',linewidth=2,))
    ax.add_patch(_plt_Circle(xy=(pt_x, pt_y), radius=r/40, alpha=0.6))
    # ax.add_patch(create_buffered_square_patch(side_length=grid._search_internals.spacing, r=r, x_off=hc_x, y_off=hc_y))
    # ax.add_patch(create_debuffered_square_patch(side_length=grid._search_internals.spacing, r=r, linewidth=2, x_off=hc_x, y_off=hc_y ))
    
    # ax.add_patch(create_trgl1_patch(side_length=grid._search_internals.spacing/2, linewidth=2, x_off=hc_x, y_off=hc_y ))
    # ax.add_patch(create_buffered_trgl1_patch(side_length=grid._search_internals.spacing/2, linewidth=2, x_off=hc_x, y_off=hc_y ))
    # ax.add_patch(create_debuffered_trgl1_patch(side_length=grid._search_internals.spacing/2, linewidth=2, x_off=hc_x, y_off=hc_y ))
    # print('+++++', [(cells,color) for cells,color in zip(
    #     [cells_cntd_by_pt_cell, cells_cntd_by_pt_region, cells_ovlpd_by_pt_region],
    #     ['blue','green', 'red'])])
    # print('+++++', [[(grid.get_cell_centroid(row,col), color) for (row,col) in cells] for cells,color in zip(
    #     [cells_cntd_by_pt_cell, cells_cntd_by_pt_region, cells_ovlpd_by_pt_region],
    #     ['blue','green', 'red'])])

    # all pts
    ax.scatter(
        x=pts_target[x],
        y =pts_target[y],
        # c=pts_target['sc_nr'],
        # s=fig.get_figheight()/.2, cmap='viridis', marker='x')
        s=fig.get_figheight()/1, color='#777', marker='x')
    # pts in cntd cells
    ax.scatter(
        x=pts_xy_in_cell_cntd_by_pt_region[:,0],
        y =pts_xy_in_cell_cntd_by_pt_region[:,1],
        s=fig.get_figheight()/2, color='yellow', marker='o')
    # pts in ovlpd cells
    ax.scatter(
        x=pts_xy_in_cells_ovlpd_by_pt_region[:,0],
        y =pts_xy_in_cells_ovlpd_by_pt_region[:,1],
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

    display_lines = []
    if sums_within_disk is not None:
        for name, value in zip(sum_names, sums_within_disk):
            try:
                display_lines.append(f"{name}: {float(value):.6g}")
            except Exception:
                display_lines.append(f"{name}: {value}")
    else:
        display_lines = [str(name) for name in sum_names]

    display_lines.append(f"\ncell_region_id: {display_cell_region_id}")

    title_text = "  •  ".join(display_lines)
    ax.set_title(
        title_text,
        fontsize=40,
        pad=18,
        loc='left',
    )
    show = plot_kwargs.pop('show', True)
    display_dpi = plot_kwargs.pop('display_dpi', 100)
    save_kwargs = plot_kwargs.pop('save_kwargs', {})
    if save_path:
        if not str(save_path).lower().endswith(('.png', '.jpg', '.jpeg', '.pdf', '.svg')):
            save_path = str(save_path) + '.png'
        fig.savefig(save_path, **{'dpi': 150, 'bbox_inches': 'tight', **save_kwargs})
    if not show:
        from matplotlib.pyplot import close as _plt_close_local
        _plt_close_local(fig)
    #
#

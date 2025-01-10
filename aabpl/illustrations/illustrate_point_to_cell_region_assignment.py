from numpy import (
    array as _np_array, 
    unique as _np_unique, 
    arange as _np_arange,
    linspace, invert, flip, transpose, 
    concatenate, 
    sign as _np_sign, 
    zeros, min, max, equal, where, 
    logical_or, logical_and, all, newaxis
)
from math import ceil as _math_ceil
# from numpy.random import randint, random
from matplotlib.pyplot import (subplots as _plt_subplots, figure as _plt_figure)
# from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Circle as _plt_circle
from matplotlib.figure import Figure as _plt_Figure
from matplotlib.axes._axes import Axes as _plt_Axes
from .plot_utils import (
    create_grid_cell_patches, 
    create_grid_cell_patches_by_type, 
    create_grid_cell_rectangles,
    create_trgl1_patch, 
    create_buffered_trgl1_patch, 
    create_buffered_square_patch,
    dual_circle_union_patch,
    add_grid_cell_rectangles_by_color,
    add_circle_patches,
    create_debuffered_square_patch,
    create_debuffered_trgl1_patch,
    map_2D_to_rgb
)
from aabpl.utils.distances_to_cell import ( get_cells_relevant_for_disk_by_type, get_cell_farthest_vertex_to_point,  )


# Plot
def illustrate_point_to_cell_region_assignment(
    grid:dict,
    triangle_ids,
    region_ids,
    offset_xy,
    transformed_offset_xy,
    radius:float=750,
    include_boundary:bool=False,
    **plot_cell_reg_assign,
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
        'example_cell_region':0,
        **plot_cell_reg_assign
    }
    figsize = plot_kwargs.pop('figsize')
    fig = plot_kwargs.pop('fig')
    ax = plot_kwargs.pop('ax')
    example_cell_region = plot_kwargs.pop('example_cell_region')
    example_trgl_region = example_cell_region//100

    # unpack vals
    grid_spacing = grid.spacing
    cell_steps_max = -int(-radius/grid_spacing)+1
    trgl_region_ids = _np_array([region_id//100 for region_id in region_ids])
    unique_region_ids = _np_unique([region_id for region_id in grid.search.region_id_to_contained_cells])
    unique_trgl_region_ids = _np_unique([region_id//100 for region_id in unique_region_ids])
    
    contain_region_ids = _np_array([region_id // grid.search.contain_region_mult for region_id in region_ids])
    overlap_region_ids = _np_array([region_id % grid.search.contain_region_mult for region_id in region_ids])
    unique_region_ids = sorted(_np_unique([region_id for region_id in grid.search.region_id_to_contained_cells]))
    
    print('unique region_ids:',len(unique_region_ids),'unique trgl_region_ids:', len(unique_trgl_region_ids))

    (cells_contained_in_all_disks, 
     cells_contained_in_all_trgl_disks, 
     cells_maybe_overlapping_a_disk, 
     cells_maybe_overlapping_a_trgl_disk) = get_cells_relevant_for_disk_by_type(
            grid_spacing=grid_spacing,
            radius=radius,
            include_boundary=include_boundary,
    )

    ###### initialize plot  ######################

    # fig, axs = _plt_subplots(3,3, figsize=(10,15))
    # flat_axs = axs.flat
    if fig is None:
        fig = _plt_figure(1, figsize=figsize)
    elif type(fig) != _plt_Figure:
        raise TypeError
    flat_axs = []

    flat_axs.append(fig.add_subplot(3,3,len(flat_axs)+1))
    flat_axs.append(fig.add_subplot(3,3,len(flat_axs)+1))
    flat_axs.append(fig.add_subplot(3,3,len(flat_axs)+1))

    flat_axs.append(fig.add_subplot(3,3,len(flat_axs)+1+3))
    flat_axs.append(fig.add_subplot(3,3,len(flat_axs)+1+3))
    flat_axs.append(fig.add_subplot(3,3,len(flat_axs)+1+3))
    
    


    ################################################################################################################
    ax = flat_axs[0]

    add_grid_cell_rectangles_by_color(
        [cells_contained_in_all_disks, cells_maybe_overlapping_a_disk],
        ['green', 'red'],
        ax=ax, grid_spacing=grid_spacing
    )
    
    ax.add_patch(create_buffered_square_patch(side_length=grid_spacing, radius=radius))
    ax.add_patch(create_debuffered_square_patch(side_length=grid_spacing, radius=radius, linewidth=2, edgecolor="red", facecolor="None"))

    ################################################################################################################
    ax = flat_axs[1]

    add_grid_cell_rectangles_by_color(
        [cells_maybe_overlapping_a_disk, cells_contained_in_all_disks,cells_contained_in_all_trgl_disks, cells_maybe_overlapping_a_trgl_disk],
        ['#ccc', 'blue', 'green', 'red'],
        ax=ax, grid_spacing=grid_spacing
    )

    ax.add_patch(create_trgl1_patch(side_length=grid_spacing/2, linewidth=2))
    ax.add_patch(create_buffered_trgl1_patch(side_length=grid_spacing/2, radius=radius, linewidth=2))
    ax.add_patch(create_debuffered_trgl1_patch(side_length=grid_spacing/2, radius=radius, linewidth=2))

    ################################################################################################################
    ax = flat_axs[2]

    add_grid_cell_rectangles_by_color(
        [
            cells_maybe_overlapping_a_disk, 
            cells_contained_in_all_disks,
            cells_contained_in_all_trgl_disks, 
            grid.search.region_id_to_contained_cells[example_cell_region],
            grid.search.weak_order_tree.cells_contained_in_no_trgl1_disk,
            grid.search.weak_order_tree.cells_overlapped_by_all_trgl1_disks,
            grid.search.region_id_to_overlapped_cells[example_cell_region]
        ],
        [
            '#ccc',
            'pink',
            'blue',
            'green',
            'black',
            'orange',
            'red'
        ],
        ax=ax, grid_spacing=grid_spacing
    )
    filter_mask = _np_arange(len(transformed_offset_xy))[trgl_region_ids==example_trgl_region]
    ax.scatter(x=transformed_offset_xy[filter_mask,0],y=transformed_offset_xy[filter_mask,1],s=fig.get_figheight()/75, color='black')
    
    ################################################################################################################
    # n_cols = len(unique_trgl_region_ids)
    # trgl_combs = sorted(list(set([x%100 for x in unique_region_ids])))
    # n_rows = len(trgl_combs)
        
    # # hundreds = set([x//100 for x in grid.search.region_id_to_contained_cells.keys()])
    # # tens = set([x%100//10 for x in grid.search.region_id_to_contained_cells.keys()])
    # # singles = set([x%10 for x in grid.search.region_id_to_contained_cells.keys()])
 
    # # for i in range(1,8+1):
    # #     tr_ids_i = [x for x in unique_region_ids if i == x%100//10] # tens
    # #     for j in range(1,8+1):
    # #         tr_ids_ij = [x for x in tr_ids_i if j == x%10] # singles
    # for i,trgl_comb in enumerate(trgl_combs):
    #     is_first_col = True
    #     for z in [z for z in sorted(unique_trgl_region_ids)]: # hundreds
    #         current_region_id = int(100*z+trgl_comb)
    #         if current_region_id in unique_region_ids:
    #             flat_axs.append(fig.add_subplot(n_rows*3-10,n_cols,n_cols*(n_rows-5+i)+z+1))
    #             ax = flat_axs[-1]

    #             add_grid_cell_rectangles_by_color(
    #                 [
    #                     cells_maybe_overlapping_a_disk, 
    #                     cells_contained_in_all_disks,
    #                     # cells_contained_in_all_trgl_disks, 
    #                     grid.search.region_id_to_contained_cells[current_region_id],
    #                     # grid.search.weak_order_tree.cells_contained_in_no_trgl1_disk,
    #                     # grid.search.weak_order_tree.cells_overlapped_by_all_trgl1_disks,
    #                     grid.search.region_id_to_overlapped_cells[current_region_id]
    #                 ],
    #                 [
    #                     '#ccc',
    #                     'blue',
    #                     # 'blue',
    #                     'green',
    #                     # 'black',
    #                     # 'orange',
    #                     'red'
    #                 ],
    #                 ax=ax, grid_spacing=grid_spacing
    #             )
    #             if is_first_col:
    #                 ax.set_ylabel(str(trgl_comb))
    #                 is_first_col = False
    #             #
    #         #
    #         if i == 0:
    #             ax.set_title(str(z))
    #         #
    #     #
    # #
    
    unique_contain_region_ids = sorted(_np_unique(contain_region_ids))
    unique_overlap_region_ids = sorted(_np_unique(overlap_region_ids))
    n_cols = _math_ceil(len(unique_region_ids)**.5)
    n_rows = n_cols
    print("n r c", n_rows,n_cols)
    # hundreds = set([x//100 for x in grid.search.region_id_to_contained_cells.keys()])
    # tens = set([x%100//10 for x in grid.search.region_id_to_contained_cells.keys()])
    # singles = set([x%10 for x in grid.search.region_id_to_contained_cells.keys()])
 
    # for i in range(1,8+1):
    #     tr_ids_i = [x for x in unique_region_ids if i == x%100//10] # tens
    #     for j in range(1,8+1):
    #         tr_ids_ij = [x for x in tr_ids_i if j == x%10] # singles
    for i,current_region_id in enumerate(unique_region_ids):
        flat_axs.append(fig.add_subplot(n_rows*3-10,n_cols,n_cols*(n_rows-5)+i+1))
        ax = flat_axs[-1]

        add_grid_cell_rectangles_by_color(
            [
                cells_maybe_overlapping_a_disk, 
                cells_contained_in_all_disks,
                # cells_contained_in_all_trgl_disks, 
                grid.search.region_id_to_contained_cells[current_region_id],
                # grid.search.weak_order_tree.cells_contained_in_no_trgl1_disk,
                # grid.search.weak_order_tree.cells_overlapped_by_all_trgl1_disks,
                grid.search.region_id_to_overlapped_cells[current_region_id]
            ],
            [
                '#ccc',
                'blue',
                # 'blue',
                'green',
                # 'black',
                # 'orange',
                'red'
            ],
            ax=ax, grid_spacing=grid_spacing
        )
        #
    #
    
    ################################################################################################################
    ax = flat_axs[3]
    
    # ax dividie into triangles
    
    
    colorcoded_triangle_1_offset = _np_array([rgb_map_2D(x,y) for (x,y) in transformed_offset_xy/grid_spacing])
    ax.scatter(x=offset_xy[:,1],y=offset_xy[:,0],s=fig.get_figheight()/25, c=colorcoded_triangle_1_offset)
    # ax.scatter(x=offset_xy[:,1],y=offset_xy[:,0],s=fig.get_figheight()/50, c=triangle_ids)
    # triangle add number 
    # pos_0, pos_1 = 0.35*grid_spacing, 0.15*grid_spacing
    # ax.annotate('1', ( pos_0,  pos_1), color='black', weight='bold',fontsize=16, ha='center', va='center')
    # ax.annotate('2', ( pos_1,  pos_0), color='black', weight='bold',fontsize=16, ha='center', va='center')
    # ax.annotate('3', (-pos_1,  pos_0), color='black', weight='bold',fontsize=16, ha='center', va='center')
    # ax.annotate('4', (-pos_0,  pos_1), color='black', weight='bold',fontsize=16, ha='center', va='center')
    # ax.annotate('5', (-pos_0, -pos_1), color='black', weight='bold',fontsize=16, ha='center', va='center')
    # ax.annotate('6', (-pos_1, -pos_0), color='black', weight='bold',fontsize=16, ha='center', va='center')
    # ax.annotate('7', ( pos_1, -pos_0), color='black', weight='bold',fontsize=16, ha='center', va='center')
    # ax.annotate('8', ( pos_0, -pos_1), color='black', weight='bold',fontsize=16, ha='center', va='center')
    
    for trgl_region_id in _np_unique(triangle_ids):
        pos=offset_xy[triangle_ids==trgl_region_id].mean(axis=0)
        ax.annotate(str(trgl_region_id), pos, color='black', weight='bold',fontsize=16, ha='center', va='center')

    ################################################################################################################
    ax = flat_axs[4]
    # ax dividie into regions
    ax.scatter(x=transformed_offset_xy[:,0],y=transformed_offset_xy[:,1],s=fig.get_figheight()/50, c=contain_region_ids, cmap='tab20')
    
    add_circle_patches(
        ax=ax,
        # list_of_cells=[cells_maybe_overlapping_a_trgl_disk,cells_maybe_overlapping_a_trgl_disk],
        list_of_cells=[cells_maybe_overlapping_a_trgl_disk,cells_maybe_overlapping_a_trgl_disk],
        list_of_edgecolors=['red','green'],
        list_of_tuples_check_farthest_closest=[[False,True], [True,False],],
        edgecolor_outside_center_cell=True,
        convex_set_boundaries= _np_array([(0.0,0.0),(0.5,0.0),(0.5,0.5)]),
        grid_spacing=grid_spacing, 
        radius=radius, 
        linewidth=2
    )

    # for trgl_region_id in _np_unique(contain_region_ids):
    #     pos=transformed_offset_xy[contain_region_ids==trgl_region_id].mean(axis=0)
    #     ax.annotate(str(trgl_region_id), pos, color='black',fontsize=12, ha='center', va='center')

    ax.vlines(x=0, ymin=0, ymax=grid_spacing/2, linewidth=2,color='black')


    ################################################################################################################
    ax = flat_axs[5]

    # ax dividie into regions
    # ax.scatter(x=offset_xy[:,1],y=offset_xy[:,0],s=fig.get_figheight()/50, c=region_ids, cmap='tab20')
    ax.scatter(x=offset_xy[:,1],y=offset_xy[:,0],s=fig.get_figheight()/50, c=contain_region_ids, cmap='tab20')
    # ax.scatter(x=offset_xy[:,1],y=offset_xy[:,0],s=fig.get_figheight()/50, c=overlap_region_ids, cmap='tab20')
    
    add_circle_patches(
        ax=ax,
        list_of_cells=[cells_maybe_overlapping_a_disk,cells_maybe_overlapping_a_disk],
        list_of_edgecolors=['red','green'],
        list_of_tuples_check_farthest_closest=[[False,True], [True,False],],
        edgecolor_outside_center_cell=True,
        convex_set_boundaries= _np_array([(-0.5,-0.5), (0.5,-0.5),(0.5,0.5),(-0.5,0.5)]),
        grid_spacing=grid_spacing, 
        radius=radius, 
        linewidth=1.5,
        linestyle='-.',
    )
        
    for region_id in _np_unique(region_ids):
        pos=offset_xy[region_ids==region_id].mean(axis=0)
        ax.annotate(str(region_id), pos, color='black',fontsize=8, ha='center', va='center')

    ################################################################################################################
    

    for (i, ax) in enumerate(flat_axs):
        # remove ticks
        if i > 5:
            ax.set_xticks([])
            ax.set_yticks([])
        
        # set limits
        if i < 3:
            ax.set_xlim(-(cell_steps_max+1)*grid_spacing,(cell_steps_max+1)*grid_spacing)
            ax.set_ylim(-(cell_steps_max+1)*grid_spacing,(cell_steps_max+1)*grid_spacing)
        elif i in [3,5]:
            ax.set_xlim(-(.6)*grid_spacing,(.6)*grid_spacing)
            ax.set_ylim(-(.6)*grid_spacing,(.6)*grid_spacing)
        elif i in [4]:
            ax.set_xlim(-(.0)*grid_spacing,(.5)*grid_spacing)
            ax.set_ylim(-(.0)*grid_spacing,(.5)*grid_spacing)
        elif i in [5]:
            ax.set_xlim(-(3.5)*grid_spacing,(3.5)*grid_spacing)
            ax.set_ylim(-(3.5)*grid_spacing,(3.5)*grid_spacing)

        if i in [3,4,5]:
            # add diagionals 
            ax.axline([-.5*grid_spacing,  .0*grid_spacing], [ .5*grid_spacing,  .0*grid_spacing], linewidth=1, color='black')
            ax.axline([ .0*grid_spacing, -.5*grid_spacing], [ .0*grid_spacing,  .5*grid_spacing], linewidth=1, color='black')
            ax.axline([-.5*grid_spacing, -.5*grid_spacing], [ .5*grid_spacing,  .5*grid_spacing], linewidth=1, color='black')
            ax.axline([-.5*grid_spacing,  .5*grid_spacing], [ .5*grid_spacing,  -.5*grid_spacing], linewidth=1, color='black')
        
        # FOR ALL AXS 
        ax.vlines(x=-grid_spacing/2, ymin=-grid_spacing/2, ymax=grid_spacing/2, linewidth=2,color='black')
        ax.vlines(x=grid_spacing/2, ymin=-grid_spacing/2, ymax=grid_spacing/2, linewidth=2,color='black')
        ax.hlines(y=-grid_spacing/2, xmin=-grid_spacing/2, xmax=grid_spacing/2, linewidth=2,color='black')
        ax.hlines(y=grid_spacing/2, xmin=-grid_spacing/2, xmax=grid_spacing/2, linewidth=2,color='black')
        ax.set_aspect('equal', adjustable='box')



#

from pandas import DataFrame as _pd_DataFrame
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

from aabpl.radius_search.offset_region_classes import OffsetRegion

def visualize_pt_to_cell_region_assignment(
    grid,
    pts:_pd_DataFrame,
    off_x:str,
    off_y:str,
    cell_region_name:str,
    offset_all_x_vals:_np_array,
    offset_all_y_vals:_np_array,
    plot_cell_reg_validation:dict={}
    ):
        """
        Plots points by their offset coordinates (reative to cell center) colored by micro region
        plots outlines of micro regions
        """
        lims = [-0.51,0.51]
    
    
        fig, ax = _plt_subplots(1,1,figsize=(15,15))
        
        # plot scatter points by offset coordinates, color by mirco region name
        ax.scatter(x=pts[off_x], y=pts[off_y], c=pts[cell_region_name], s=2.5, cmap="prism")
        

        # add cutoff lines
        cutoff_min = (-0.5-lims[0])/(lims[1]-lims[0])
        cutoff_max = (0.5-lims[0])/(lims[1]-lims[0])
        cutoff_linewidth = 0.3
        cutoff_color = 'black'
        for offset_all_x_val in offset_all_x_vals:
                ax.axvline(
                    x=offset_all_x_val, ymin=cutoff_min, ymax=cutoff_max, linewidth=cutoff_linewidth, color=cutoff_color
                )
        for offset_all_y_val in offset_all_y_vals:
                ax.axhline(
                    y=offset_all_y_val, xmin=cutoff_min, xmax=cutoff_max, linewidth=cutoff_linewidth, color=cutoff_color
                )
        
        # plot region polygon outlines
        OffsetRegion.plot_many(
                regions=list(grid.id_to_offset_regions.values()), 
                plot_edges=False, 
                add_idxs=True,
                edgecolor='black', facecolor='None', alpha=0.8, linewidth=0.4,
                ax=ax,
                )
        
        # find centroids for micro regions from point data 
        for region_id in pts[cell_region_name].unique():
                xy = pts[[off_x, off_y]][pts[cell_region_name]==region_id].mean()
                ax.annotate(
                        text=str(region_id), xy=xy, 
                        horizontalalignment='center', fontsize=10, color="red", weight="bold"
                )
        
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        
        return ax

    
    



# replace imports with from imports
from numpy import (
    array as _np_array, arange as _np_arange, unique as _np_unique, linspace as _np_linspace, sign as _np_sign,
)
from numpy.linalg import norm as _np_linalg_norm
# from numpy.random import randint, random
# from matplotlib.animation import FuncAnimation, PillowWriter#
from math import log10 as _math_log10
from matplotlib.patches import (Rectangle as _plt_Rectangle, Polygon as _plt_Polygon, Circle as _plt_Circle)
from matplotlib.colors import LinearSegmentedColormap as _plt_LinearSegmentedColormap
from matplotlib.colors import Normalize as _plt_Normalize
from matplotlib.axes._axes import Axes as _plt_Axes
from matplotlib.cm import ScalarMappable as _plt_ScalarMappable
from math import (
    sin as _math_sin,
    cos as _math_cos,
    asin as _math_asin,
    acos as _math_acos,
    atan2 as _math_atan2,
    pi as _math_pi)
from aabpl.utils.general import flatten_list, angle
from aabpl.utils.distances_to_cell import ( get_cells_relevant_for_disk_by_type, get_cell_farthest_vertex_to_point, get_cell_closest_point_to_points, )

# x = np.linspace(-6,6, num=100)
# y = np.linspace(-10,10, num=100)
# X,Y = np.meshgrid(x,y)
# Z = np.sin(X)/X+np.sin(Y)/Y

# fig, ax = plt.subplots()
# ax.contourf(X,Y,Z, alpha=.1)
# ax.contour(X,Y,Z, alpha=.4)

# class AnchoredHScaleBar(matplotlib.offsetbox.AnchoredOffsetbox):
#     """ size: length of bar in data units
#         extent : height of bar ends in axes units """
#     def __init__(self, size=1, extent = 0.03, label="", loc=2, ax=None,
#                  pad=0.4, borderpad=0.5, ppad = 0, sep=2, prop=None, 
#                  frameon=True, linekw={}, **kwargs):
#         if not ax:
#             ax = plt.gca()
#         trans = ax.get_xaxis_transform()
#         size_bar = matplotlib.offsetbox.AuxTransformBox(trans)
#         line = Line2D([0,size],[0,0], **linekw)
#         size_bar.add_patch(_plt_Rectangle(xy=(0,0), width=1, height=1, color='black'))
#         txt = matplotlib.offsetbox.TextArea(label, minimumdescent=False)
#         self.vpac = matplotlib.offsetbox.VPacker(children=[size_bar,txt],  
#                                  align="center", pad=ppad, sep=sep) 
#         matplotlib.offsetbox.AnchoredOffsetbox.__init__(self, loc, pad=pad, 
#                  borderpad=borderpad, child=self.vpac, prop=prop, frameon=frameon,
#                  **kwargs)

# ob = AnchoredHScaleBar(size=3, label="3 units", loc=4, frameon=True,
#                        pad=0.6,sep=4, linekw=dict(color="crimson"),) 
# ax.add_artist(ob)

def set_map_frame(ax, xmin:float, xmax:float, ymin:float, ymax:float):
    pad_x, pad_y = (xmax-xmin)/50, (ymax-ymin)/50
    ax.set_xlim([xmin-pad_x,xmax+pad_x])
    ax.set_ylim([ymin-pad_y,ymax+pad_y])
    xticks = [xmin, (xmin+xmax)/2, xmax]
    yticks = [ymin, (ymin+ymax)/2, ymax]
    ndigits = int(max([0, _math_log10(100/(xticks[1]-xticks[0])), _math_log10(100/(yticks[1]-yticks[0]))]))
    ax.set_xticks(xticks, labels=[int(round(t,ndigits)) if ndigits==0 else round(t,ndigits) for t in xticks])
    ax.set_yticks(yticks, labels=[int(round(t,ndigits)) if ndigits==0 else round(t,ndigits) for t in yticks])
    

# def _plt_colorbar(sc, extend='min', cax=add_color_bar_ax(fig,ax))

def add_color_bar_ax(fig,ax):
    if fig is None:
        cax = ax.inset_axes([ax.get_position().x1 + 0.01, ax.get_position().y0, 0.02 ,ax.get_position().height])
        return cax
    cax = fig.add_axes([ax.get_position().x1 + 0.01, ax.get_position().y0, 0.02 ,ax.get_position().height])
    return cax

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=200):
    new_cmap = _plt_LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(_np_linspace(minval, maxval, n)))
    return new_cmap
#

def map_2D_to_rgb (
    x=None,
    y=None,
    xmin=None,
    xmax=None,
    ymin=None,
    ymax=None,
    rgb_xmin_ymin:tuple=(  2,253,102),
    rgb_xmax_ymin:tuple=(  2,  2,253),
    rgb_xmin_ymax:tuple=(253, 23,  2),
    skewness_exponent:float=0.5,

)->tuple:
    """
    if create_colorbars is True, it will not return rgb value but instead
    """

    if xmin is None: xmin = min(x)
    if xmax is None: xmax = max(x)
    if ymin is None: ymin = min(x)
    if ymax is None: ymax = max(x)
  
    x_share = max([1e-15, (x-xmin)/(xmax-xmin)])
    y_share = max([1e-15, (y-ymin)/(ymax-ymin)])
    m_share = ((x_share+y_share)/2)**skewness_exponent
    x_ratio = x_share/(x_share+y_share)
    y_ratio = y_share/(x_share+y_share)
    rgb = tuple([
        ((1-m_share) * v_0 + m_share * (x_ratio * v_x + y_ratio * v_y))/255
        for v_0, v_x, v_y in zip(rgb_xmin_ymin, rgb_xmax_ymin, rgb_xmin_ymax)
        ])
    return rgb

def get_2D_rgb_colobar_kwargs(
    xmin,
    xmax,
    ymin,
    ymax,
    rgb_xmin_ymin:tuple=(  2,253,102),
    rgb_xmax_ymin:tuple=(  2,  2,253),
    rgb_xmin_ymax:tuple=(253, 23,  2),
    skewness_exponent:float=0.5,
):
    kwargs = locals()
    steps = 20
    return ({'mappable': _plt_ScalarMappable(
                norm = _plt_Normalize(vmin=xmin, vmax=xmax, clip=False),
                cmap = _plt_LinearSegmentedColormap.from_list('', [map_2D_to_rgb(x=v, y=ymin, **kwargs) for v in _np_linspace(xmin, xmax, steps)])
                )
            },
            {'mappable': _plt_ScalarMappable(
                norm = _plt_Normalize(vmin=ymin, vmax=ymax, clip=False),
                cmap = _plt_LinearSegmentedColormap.from_list('', [map_2D_to_rgb(x=xmin, y=v, **kwargs) for v in _np_linspace(ymin, ymax, steps)])
                )
            },
            {'mappable': _plt_ScalarMappable(
                norm = _plt_Normalize(vmin=0, vmax=1, clip=False),
                cmap = _plt_LinearSegmentedColormap.from_list('', [map_2D_to_rgb(x=x, y=y, **kwargs) for x,y in zip(_np_linspace(xmin, xmax, steps), _np_linspace(ymin, ymax, steps))])
                )
            }
            )
    

#################### RECTANGLES ########################################################

def create_grid_cell_patches(
        grid_spacing,
        ax_min,
        ax_max,
        contain_cells_row_col:_np_array,
        overlap_cells_row_col:_np_array,
        contain_triangle_cells_row_col:_np_array=_np_array([])
    )->list:
    """
    Return list of grid cell patches
    """
    gridCellPatches = []

    # choose cell_steps_max s.t. the grid fills the complete plot area
    cell_steps_max = int(ax_max/grid_spacing+2)
    for j in range(cell_steps_max):
        for k in range(cell_steps_max):
            rect_color = (
                'yellow' if ((j,k) in contain_triangle_cells_row_col) else
                'green' if ((j,k) in contain_cells_row_col or (k,j) in contain_cells_row_col) else 
                'red' if ((j,k) in overlap_cells_row_col or (k,j) in overlap_cells_row_col) else 
                'grey'
            )
            # rect_edgecolor = 'red' if j==0 and k==0 else '#000'
            for pj in [-1,1]:
                for pk in [-1,1]:
                    append_rect = False
                    if (-1 in [pj,pk]) and (j!=0) and (k!=0):
                        # add to plot if top right is within plot area
                        if (pj*j+.5)*grid_spacing>ax_min and (pk*k+.5)*grid_spacing>ax_min:
                            append_rect = True
                    else:
                        # append all rect in top right quarter 
                        append_rect = True
                    
                    if append_rect:
                        gridCellPatches.append(_plt_Rectangle(
                            ((j*pj-.5)*grid_spacing, (k*pk-.5)*grid_spacing), grid_spacing, grid_spacing, 
                            linewidth=.7, facecolor=rect_color, edgecolor='#444', alpha=.5
                            ))
    return gridCellPatches
#

def create_grid_cell_patches_by_type(
        grid_spacing,
        contain_cells_row_col:_np_array,
        overlap_cells_row_col:_np_array,
        contain_triangle_cells_row_col:_np_array=_np_array([]),
        outside_cells_row_col:_np_array=_np_array([])
    )->list:
    """
    Return list of grid cell patches
    """
    gridCellPatches = []

    for cell_by_type, color_for_type in zip(
        [contain_cells_row_col, contain_triangle_cells_row_col, overlap_cells_row_col, outside_cells_row_col],
        ['green', 'yellow', 'red', 'grey']
    ):
        for cell in cell_by_type:
            gridCellPatches.append(_plt_Rectangle(
                ((cell[1]-.5)*grid_spacing, (cell[0]-.5)*grid_spacing), grid_spacing, grid_spacing, 
                linewidth=.7, facecolor=color_for_type, edgecolor='#444', alpha=.3
                ))
    return gridCellPatches
#

def create_grid_cell_rectangles(
    cells:_np_array,
    grid_spacing,
    facecolor:str='green',
    edgecolor='#444',
    alpha=.3,
    x_off:float=0,
    y_off:float=0,
)->list:
    """
    Return list of grid cell patches
    """
    gridCellPatches = []
    for cell in cells:
        gridCellPatches.append(_plt_Rectangle(
            xy = (
                (cell[1] - .5) * grid_spacing + x_off,
                (cell[0] - .5) * grid_spacing + y_off
            ), width=grid_spacing, height=grid_spacing, 
            linewidth=.7, facecolor=facecolor, edgecolor=edgecolor, alpha=alpha
            ))
    return gridCellPatches
#

def add_grid_cell_rectangles_by_color(
    list_of_cells:list,
    list_of_facecolors:list,
    ax:_plt_Axes,
    grid_spacing:float=1,
    x_off:float=0,
    y_off:float=0,
): 
    # create patches
    for cells, facecolor in zip(list_of_cells, list_of_facecolors):
        cell_patches = create_grid_cell_rectangles(
            cells=cells,
            grid_spacing=grid_spacing, 
            facecolor=facecolor,
            x_off=x_off,
            y_off=y_off,
        )
        for cell_patch in cell_patches:
            ax.add_patch(cell_patch)
#

def create_trgl1_patch(
        side_length:float = 1,
        facecolor:str='None', 
        edgecolor:str='green',
        x_off:float=0,
        y_off:float=0,
        **kwargs,
) -> list:
    """
    
    """
    poly_coords = (
        # top right
        [(0+x_off,0+y_off), (side_length+x_off, 0+y_off), (side_length+x_off, side_length+y_off)]
    )

    return _plt_Polygon(poly_coords, facecolor=facecolor, edgecolor=edgecolor, **kwargs)
#


#################### CIRCLES ########################################################

def create_circle_arc_coords(
    pts:_np_array, 
    arc:_np_array,
    r:float,
    rotation_angles:float=None
):
    if rotation_angles is None:
        n=len(pts)
        rotation_angles=_np_arange(n) / n * 2 *_math_pi
    r=r
    poly_coords = flatten_list([[(r*_math_cos(t) + pt_x, r*_math_sin(t) + pt_y) for t in angle+arc]  for (pt_x,pt_y),angle in zip(pts, rotation_angles)])

    return poly_coords

def create_buffered_square_patch(
        side_length:float,
        r:float=750,
        nsteps:int = 25,
        facecolor:str='None', 
        edgecolor:str='green',
        x_off:float=0,
        y_off:float=0,
        **kwargs,
) -> _plt_Polygon:
    """
    
    """
    x = side_length / 2
    poly_coords = create_circle_arc_coords(pts=(
            (+x + x_off, +x + y_off), # top right
            (-x + x_off, +x + y_off), # top left
            (-x + x_off, -x + y_off), # bottom left
            (+x + x_off, -x + y_off), # bottom right
        ),
        arc = _math_pi * _np_linspace(0, .5,nsteps),
        r = r,
    )

    return _plt_Polygon(poly_coords, facecolor=facecolor, edgecolor=edgecolor, **kwargs)
#
  
def create_debuffered_square_patch(
        side_length:float,
        r:float=750,
        nsteps:int = 25,
        facecolor:str='None', 
        edgecolor:str='red',
        x_off:float=0,
        y_off:float=0,
        **kwargs,
) -> _plt_Polygon:
    """
    
    """
    x = side_length/2
    alpha = _math_acos(x/2/r)/_math_pi
    beta = 0.5-alpha
    print("ALPA",alpha)
    poly_coords = create_circle_arc_coords(pts=(
            (-x + x_off, -x + y_off), # top right
            (+x + x_off, -x + y_off), # top left
            (+x + x_off, +x + y_off), # bottom left
            (-x + x_off, +x + y_off), # bottom right
        ),
        arc=_math_pi * _np_linspace(beta, alpha,nsteps,endpoint=True),
        r=r,
    )
    return _plt_Polygon(poly_coords,facecolor=facecolor,edgecolor=edgecolor, **kwargs)
#

def create_buffered_trgl1_patch(
        side_length:float = 250,
        r:float = 750,
        nsteps:int = 25,
        facecolor:str='None', 
        edgecolor:str='yellow',
        endpoint=True,
        x_off:float=0,
        y_off:float=0,
        **kwargs,
) -> list:
    """
    
    """
    x = side_length
    poly_coords =  (
        # top right/left
        [(r*_math_cos(t) + x + x_off, r*_math_sin(t) + x + y_off) for t in _math_pi * _np_linspace(0, 0.75,nsteps)] +
        # top/bottom left 
        [(r*_math_cos(t) + x_off, r*_math_sin(t) + y_off) for t in _math_pi * _np_linspace(0.75, 1.5,nsteps)] +
        # bottom right
        [(r*_math_cos(t) + x + x_off, r*_math_sin(t) + y_off) for t in _math_pi * _np_linspace(1.5, 2,nsteps)]
    )
    return _plt_Polygon(poly_coords, facecolor=facecolor, edgecolor=edgecolor, **kwargs)
#

def create_debuffered_trgl1_patch(
        side_length:float = 250,
        r:float = 750,
        nsteps:int = 25,
        x_off:float=0,
        y_off:float=0,
        facecolor:str='None', 
        edgecolor:str='black',
        **kwargs,
) -> list:
    """
    
    """
    x = side_length
    alpha = _math_acos(x/2/r)/_math_pi
    beta = 0.5-alpha
    poly_coords = (
        # bottom/top right
        [(r*_math_cos(t) + x_off, r*_math_sin(t) + y_off) for t in _math_pi * _np_linspace(-0.25, alpha,nsteps,endpoint=True)] +
        # top left
        [(r*_math_cos(t) + x + x_off, r*_math_sin(t) + y_off) for t in _math_pi * _np_linspace(0.5+beta, 0.5+alpha,nsteps,endpoint=True)] +
        # bottom left/right 
        [(r*_math_cos(t) + x + x_off, r*_math_sin(t) + x + y_off) for t in _math_pi * _np_linspace(1+beta, 1.75,nsteps,endpoint=True)]
    )

    return _plt_Polygon(poly_coords, facecolor=facecolor, edgecolor=edgecolor, **kwargs)

def  dual_circle_union_patch(
        centroids:_np_array,
        r:float,
        nsteps:int=100,
        **kwargs,
        ):
    dist = _np_linalg_norm(centroids[1]-centroids[0])
    if dist >= r:
        print("too far apart. 2 circles are not implemented")
        return
    alpha = _math_acos(dist/2/r)
    left_x, left_y = centroids[0]
    right_x, right_y = centroids[1]
    slope_angle = angle(left_x, left_y, right_x, right_y)

    poly_coords = (
        # left half
        [(r*_math_cos(t) + left_x, r*_math_sin(t) + left_y) for t in -slope_angle + _np_linspace(alpha, 2*_math_pi-alpha,nsteps)] +
        # right half
        [(r*_math_cos(t) +right_x, r*_math_sin(t) + right_y) for t in -slope_angle + _np_linspace(_math_pi+alpha, 3*_math_pi-alpha,nsteps)]
    )
    
    return _plt_Polygon(poly_coords, **kwargs)
#

def add_circle_patches(
    ax:_plt_Axes,
    list_of_cells:list,
    list_of_edgecolors:list,
    list_of_tuples_check_farthest_closest:list,
    edgecolor_outside_center_cell=True,
    convex_set_boundaries:_np_array= _np_array([(-0.5,-0.5), (0.5,-0.5),(0.5,0.5),(-0.5,0.5)]),
    grid_spacing:float=1, 
    r:float=3, 
    **kwargs,
):
    circles_outside_center_cell = []
    circles_inside_center_cell = []
    for (cells, edgecolor, (check_farthest, check_closest)) in zip(list_of_cells, list_of_edgecolors, list_of_tuples_check_farthest_closest):
        if (
            (type(edgecolor_outside_center_cell)==bool and edgecolor_outside_center_cell == True) or 
            (type(edgecolor_outside_center_cell)==str and not edgecolor_outside_center_cell=='None')
            ):
            edgecolor_outside = edgecolor if (edgecolor_outside_center_cell == True) else edgecolor_outside_center_cell
        
        for cell in cells:
            (row,col) = cell
            farthest_points = _np_unique(
                    get_cell_farthest_vertex_to_point(
                        convex_set_boundaries,
                        cell
                ), axis=0)*grid_spacing if check_farthest else []
            if check_closest:
                if row == 0 or col == 0:
                    # 
                    if edgecolor_outside not in [False, None, 'None']:
                        circles_outside_center_cell.append(create_buffered_square_patch(
                            side_length=grid_spacing, r=r,
                            edgecolor=edgecolor_outside,
                            x_off = (col - _np_sign(col))*grid_spacing,
                            y_off = (row - _np_sign(row))*grid_spacing,
                            **kwargs
                            ))
                    # circle_patches_segment.append()
                    pass
                else:
                    closest_cell_vertex = get_cell_closest_point_to_points(
                        convex_set_boundaries,
                        cell
                    )
                    if edgecolor_outside not in [False, None, 'None']:
                        for xy in _np_unique(closest_cell_vertex,axis=0):
                            circles_outside_center_cell.append(_plt_Circle(
                                xy=xy*grid_spacing, r=r, facecolor='None',edgecolor=edgecolor_outside, **kwargs
                            ))
                    # circle_patches_segment.append()
                    pass
            if len(farthest_points)==2:
                if row != 0 or col != 0:  
                    circles_outside_center_cell.append(
                        dual_circle_union_patch(farthest_points, r=r, facecolor='None',edgecolor=edgecolor_outside, **kwargs)
                    )
                else:
                    print("WARNING THIS IS NOT IMPLEMENTED",farthest_points)
                    for farthest_point in farthest_points:
                        circles_outside_center_cell.append(_plt_Circle(
                            xy=farthest_point, r=r, facecolor='None',edgecolor=edgecolor_outside, **kwargs
                        ))        
            else:
                if len(farthest_points)>2:
                    print("WARNING THIS IS NOT IMPLEMENTED", farthest_points)
                for farthest_point in farthest_points:
                    circles_outside_center_cell.append(_plt_Circle(
                        xy=farthest_point, r=r, facecolor='None',edgecolor=edgecolor_outside, **kwargs
                    ))        
                    # circle_patches_segment.append()
            #
        # 
    # 
                
    # circles_inside_center_cell.append(...)
    for patch_to_add in circles_outside_center_cell+circles_inside_center_cell:
        ax.add_patch(patch_to_add)
#

def create_circle_patches(
        grid_spacing:float,
        r:float=750,
        x_off:float=0,
        y_off:float=0,
        nsteps:int = 25,
        facecolor:str='None', 
        edgecolor:str='green',
        linewidth:float=3,
) -> list:
    """
    
    """
    outer_poly_patch = create_buffered_square_patch(
        side_length=grid_spacing,
        r=r,
        nsteps=nsteps,
        facecolor=facecolor, 
        edgecolor='green',
        linewidth=3,
        x_off=x_off,
        y_off=y_off,
    )
    inner_poly_patch = create_debuffered_square_patch(
        side_length=grid_spacing,
        r=r,
        nsteps=nsteps,
        facecolor=facecolor, 
        edgecolor='green',
        linewidth=3,
        x_off=x_off,
        y_off=y_off,
    )
    return [outer_poly_patch, inner_poly_patch]
#
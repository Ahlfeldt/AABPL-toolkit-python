# replace imports with from imports
from numpy import (
    array as _np_array, 
    asarray as _np_asarray,
    arange as _np_arange, unique as _np_unique, linspace as _np_linspace, sign as _np_sign,
)
from numpy.linalg import norm as _np_linalg_norm
# from numpy.random import randint, random
# from matplotlib.animation import FuncAnimation, PillowWriter#
from matplotlib.patches import (Rectangle as _plt_Rectangle, Polygon as _plt_Polygon, Circle as _plt_Circle)
from matplotlib.colors import LinearSegmentedColormap as _plt_LinearSegmentedColormap
from matplotlib.colors import Normalize as _plt_Normalize
from matplotlib.axes._axes import Axes as _plt_Axes
from matplotlib.cm import ScalarMappable as _plt_ScalarMappable
from matplotlib.pyplot import subplots as _plt_subplots
from matplotlib.ticker import FuncFormatter as _plt_FuncFormatter
from math import (
    log10 as _math_log10,
    sin as _math_sin,
    cos as _math_cos,
    asin as _math_asin,
    acos as _math_acos,
    atan2 as _math_atan2,
    pi as _math_pi)
from aabpl.utils.misc import flatten_list, angle
from aabpl.utils.cell_geometry import ( get_cell_farthest_vertex_to_point, get_cell_closest_point_to_points, )
from matplotlib.path import Path as _plt_Path
from matplotlib.patches import PathPatch as _plt_PathPatch
from matplotlib.collections import PatchCollection as _plt_PatchCollection
from shapely.geometry import (Polygon as _shapely_Polygon, MultiPolygon as _shapely_MultiPolygon)

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

def _fmt_radius_label(v):
    """Format a single radius value as a compact distance string (e.g. '15km', '500m')."""
    if v >= 1000:
        scaled = v / 1000.0
        suffix = 'km'
    else:
        scaled = float(v)
        suffix = 'm'
    # Use a plain integer when the value is whole, otherwise up to 4 significant figures.
    if scaled == int(scaled):
        return str(int(scaled)) + suffix
    text = f'{scaled:.4g}'
    return text + suffix


_STAT_VERB = {
    'sum':   'Sum of',
    'mean':  'Mean',
    'count': 'Count of',
    'max':   'Max',
    'min':   'Min',
}


def format_col_title(colname, meta=None):
    """
    Convert a raw column name to a human-readable plot title.

    Uses ``meta`` (a dict from ``grid._aabpl_col_meta``) when available.  Falls
    back to a space-separated capitalised version of ``colname`` when no meta is
    supplied.

    Examples::

        'employment'                   no meta     -> 'Employment'
        'employment_sum_15000'         r=15000     -> 'Sum of employment (r = 15km)'
        'employment_sum_0_15000'        r=(0,15000) -> 'Sum of employment (within 15km)'
        'employment_sum_5000_15000'     r=(5k,15k)  -> 'Sum of employment (5km-15km band)'
        'employment_sum_wgt'           r=[...]     -> 'Weighted sum of employment'
        'employment_sum_15000_cluster' r=15000     -> 'Clusters - sum of employment (r = 15km)'
    """
    if meta is None:
        meta = {}

    orig_col = meta.get('c', None)
    stat     = meta.get('stat', None)
    r_spec   = meta.get('r', None)

    is_cluster_col = colname.endswith('_cluster')

    if orig_col is None or stat is None:
        name = colname[:-len('_cluster')] if is_cluster_col else colname
        return ('Clusters - ' + name) if is_cluster_col else name

    stat_verb = _STAT_VERB.get(stat, stat.capitalize())
    base = stat_verb + ' ' + orig_col

    if r_spec is None:
        r_annotation = ''
    elif isinstance(r_spec, (int, float)):
        r_annotation = 'r = ' + _fmt_radius_label(r_spec)
    elif isinstance(r_spec, tuple) and len(r_spec) == 2:
        r_inner, r_outer = float(r_spec[0]), float(r_spec[1])
        if r_inner == 0:
            r_annotation = 'within ' + _fmt_radius_label(r_outer)
        else:
            r_annotation = _fmt_radius_label(r_inner) + '-' + _fmt_radius_label(r_outer) + ' band'
    elif isinstance(r_spec, list) and r_spec:
        first = r_spec[0]
        if isinstance(first, (int, float)):
            r_annotation = ' / '.join(_fmt_radius_label(v) for v in r_spec)
        elif len(first) == 3:
            r_annotation = None  # weighted bands handled separately below
        else:
            r_inner_all = float(r_spec[0][0])
            r_outer_all = float(r_spec[-1][1])
            if r_inner_all == 0:
                r_annotation = 'within ' + _fmt_radius_label(r_outer_all)
            else:
                r_annotation = (_fmt_radius_label(r_inner_all) + '-' +
                                _fmt_radius_label(r_outer_all) + ' band')
    else:
        r_annotation = ''

    is_wbands = (
        isinstance(r_spec, list) and r_spec and
        isinstance(r_spec[0], tuple) and len(r_spec[0]) == 3
    )
    if is_wbands:
        label = 'Weighted ' + stat_verb.lower() + ' ' + orig_col
    elif r_annotation:
        label = base + ' (' + r_annotation + ')'
    else:
        label = base

    if is_cluster_col:
        return 'Clusters - ' + label
    return label


def _fmt_coord(v, _=None):
    a = abs(v)
    if a >= 1e6:
        s = f'{v/1e6:.3f}'.rstrip('0').rstrip('.')
        return s + 'M'
    if a >= 1e3:
        s = f'{v/1e3:.3f}'.rstrip('0').rstrip('.')
        return s + 'k'
    return f'{v:.3f}'.rstrip('0').rstrip('.')

def set_map_frame(ax, xmin:float, xmax:float, ymin:float, ymax:float, r:float=None, padding_frac:float=0.02):
    pad_x, pad_y = (xmax-xmin)*padding_frac, (ymax-ymin)*padding_frac
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim([xmin-pad_x, xmax+pad_x])
    ax.set_ylim([ymin-pad_y, ymax+pad_y])
    xticks = [xmin, (xmin+xmax)/2, xmax]
    yticks = [ymin, (ymin+ymax)/2, ymax]
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.xaxis.set_major_formatter(_plt_FuncFormatter(_fmt_coord))
    ax.yaxis.set_major_formatter(_plt_FuncFormatter(_fmt_coord))
    # r= indicator is drawn separately via draw_radius_indicator() after canvas.draw()


def draw_radius_indicator(fig, ax, r, xmin, xmax, ymin, ymax, placement='x'):
    """
    Draw a circle/ring scale indicator in the x-axis tick-label zone.

    ``r`` may be:
    - a scalar ``15000``                         → single outline circle
    - a list of scalars ``[10000, 15000]``        → concentric rings, one per radius
    - a band tuple ``(r_inner, r_outer)``         → grey donut between inner and outer
    - a list of band tuples
        ``[(r_inner, r_outer), ...]``             → stacked grey donuts
    - a list of weighted band tuples
        ``[(r_inner, r_outer, weight), ...]``     → donuts shaded grey with alpha
                                                    proportional to normalised weight;
                                                    inner boundaries drawn thinner

    Must be called after ``fig.canvas.draw()`` so tick-label bounding boxes are
    finalised.  All geometry uses figure-fraction coordinates (DPI-independent).
    Circles are drawn as Ellipses to compensate for non-square figure dimensions.

    Clipping rules (driven by the outermost radius)
    ------------------------------------------------
    - Fits fully  → full circle(s), no dot, ``r=`` inside outer if large enough
    - Too tall    → crop top/bottom to spine-to-figure-bottom zone, center dot
    - Too wide    → left semicircle(s) (flat edge at cx), dot, ``r=`` to the right
    """
    from matplotlib.patches import Ellipse as _Ellipse, Rectangle as _Rectangle

    # ── Classify r into one of three canonical forms ──────────────────────────
    # Returns (mode, circles, bands, wbands) where only one list is filled.

    def _classify(r_arg):
        if isinstance(r_arg, (int, float)):
            return 'circles', [float(r_arg)], None, None
        if isinstance(r_arg, tuple):
            if len(r_arg) == 3:
                return 'wbands', None, None, [(float(r_arg[0]), float(r_arg[1]), float(r_arg[2]))]
            return 'bands', None, [(float(r_arg[0]), float(r_arg[1]))], None
        # List — detect contents by examining the first element
        if not r_arg:
            return 'circles', [], None, None
        first = r_arg[0]
        if isinstance(first, (int, float)):
            return 'circles', sorted(float(v) for v in r_arg), None, None
        if isinstance(first, tuple) and len(first) == 3:
            return 'wbands', None, None, [(float(a), float(b), float(c)) for a, b, c in r_arg]
        return 'bands', None, [(float(a), float(b)) for a, b, *_ in r_arg], None

    mode, circles, bands, wbands = _classify(r)

    if mode == 'circles':
        r_outer = circles[-1] if circles else 0
    elif mode == 'bands':
        r_outer = max(b[1] for b in bands)
    else:
        r_outer = max(b[1] for b in wbands)

    if r_outer == 0:
        return

    renderer = fig.canvas.get_renderer()
    trans    = ax.transData

    # ── Step 1: display-pixel geometry ───────────────────────────────────────
    fig_bb = fig.get_window_extent(renderer)
    fw, fh = fig_bb.width, fig_bb.height

    _pt = lambda x, y: trans.transform((x, y))

    xmin_disp = _pt(xmin, ymin)[0]
    xmax_disp = _pt(xmax, ymin)[0]

    data_to_disp = (xmax_disp - xmin_disp) / (xmax - xmin)
    r_outer_disp = r_outer * data_to_disp

    ax_bb = ax.get_window_extent(renderer)

    if placement == 'y':
        # ── y-axis zone (left margin) ─────────────────────────────────────────
        spine_x   = ax_bb.x0
        ytick_bbs = [
            tick.label1.get_window_extent(renderer)
            for tick in ax.yaxis.get_major_ticks()
            if tick.label1.get_visible() and tick.label1.get_text()
        ]
        cx_disp      = (sum((b.x0 + b.x1) / 2 for b in ytick_bbs) / len(ytick_bbs)
                        if ytick_bbs else spine_x - 15)
        cy_disp      = _pt(xmin, (ymin + 3 * ymax) / 4)[1]
        avail_w_disp = max(spine_x - fig_bb.x0, 12)
        avail_h_disp = ax_bb.height / 2
    else:
        # ── x-axis zone (bottom margin) ───────────────────────────────────────
        xmid_disp = _pt((xmin + xmax) / 2, ymin)[0]
        cx_disp   = _pt((xmin + 3 * xmax) / 4, ymin)[0]
        spine_y   = ax_bb.y0
        xtick_bbs = [
            tick.label1.get_window_extent(renderer)
            for tick in ax.xaxis.get_major_ticks()
            if tick.label1.get_visible() and tick.label1.get_text()
        ]
        cy_disp      = (sum((b.y0 + b.y1) / 2 for b in xtick_bbs) / len(xtick_bbs)
                        if xtick_bbs else spine_y - 15)
        avail_w_disp = xmax_disp - xmid_disp
        avail_h_disp = max(spine_y - fig_bb.y0, 12)

    too_wide   = 2 * r_outer_disp > avail_w_disp
    too_tall   = 2 * r_outer_disp > avail_h_disp
    is_clipped = too_wide or too_tall

    # ── Step 2: figure-fraction helpers ──────────────────────────────────────
    def _f(xd, yd):
        return (xd - fig_bb.x0) / fw, (yd - fig_bb.y0) / fh

    cx_f, cy_f = _f(cx_disp, cy_disp)
    T           = fig.transFigure
    bg_color    = fig.get_facecolor()

    px_x       = 1.0 / fw
    px_y       = 1.0 / fh
    clip_pad_x = 0.0 if too_wide else px_x
    clip_pad_y = 0.0 if too_tall else px_y

    if placement == 'y':
        clip_h_disp = r_outer_disp if too_tall else 2 * r_outer_disp
        clip_x_f = (cx_disp - avail_w_disp / 2 - fig_bb.x0) / fw - clip_pad_x
        clip_y_f = (cy_disp - clip_h_disp / 2 - fig_bb.y0) / fh - clip_pad_y
        clip_w_f = avail_w_disp / fw + 2 * clip_pad_x
        clip_h_f = clip_h_disp / fh + 2 * clip_pad_y
    else:
        clip_w_disp = r_outer_disp if too_wide else 2 * r_outer_disp
        clip_x_f = (cx_disp - r_outer_disp - fig_bb.x0) / fw - clip_pad_x
        clip_y_f = (cy_disp - avail_h_disp / 2 - fig_bb.y0) / fh - clip_pad_y
        clip_w_f = clip_w_disp / fw + 2 * clip_pad_x
        clip_h_f = avail_h_disp / fh + 2 * clip_pad_y

    clip_rect = _Rectangle((clip_x_f, clip_y_f), clip_w_f, clip_h_f, transform=T)

    def _add_ellipse(radius_data, *, facecolor='none', edgecolor='black',
                     linewidth=0.8, alpha=1.0, zorder=10):
        rd   = radius_data * data_to_disp
        rx_f = rd / fw
        ry_f = rd / fh
        ell  = _Ellipse(
            (cx_f, cy_f), 2 * rx_f, 2 * ry_f,
            facecolor=facecolor, edgecolor=edgecolor,
            linewidth=linewidth, alpha=alpha,
            transform=T, clip_on=True, zorder=zorder,
        )
        ell.set_clip_path(clip_rect)
        fig.add_artist(ell)

    # ── Step 3: draw geometry depending on mode ───────────────────────────────

    if mode == 'circles':
        for i, rad in enumerate(reversed(circles)):
            is_outermost = (i == 0)
            _add_ellipse(rad, linewidth=0.8 if is_outermost else 0.3)

    elif mode == 'bands':
        for r_inner, r_outer_band in sorted(bands, key=lambda b: b[1], reverse=True):
            _add_ellipse(r_outer_band, facecolor='grey', edgecolor='grey',
                         linewidth=0.5, alpha=0.35)
            if r_inner > 0:
                # Punch out the inner area with background colour.
                _add_ellipse(r_inner, facecolor=bg_color, edgecolor='black',
                             linewidth=0.3, zorder=11)
        _add_ellipse(r_outer, facecolor='none', edgecolor='black', linewidth=0.8, zorder=12)

    else:  # wbands
        total_weight = sum(w for _, _, w in wbands) or 1.0
        for r_inner, r_outer_band, weight in sorted(wbands, key=lambda b: b[1], reverse=True):
            band_alpha = 0.15 + 0.55 * (weight / total_weight)
            _add_ellipse(r_outer_band, facecolor='grey', edgecolor='grey',
                         linewidth=0.5, alpha=band_alpha)
            if r_inner > 0:
                _add_ellipse(r_inner, facecolor=bg_color, edgecolor='black',
                             linewidth=0.3, zorder=11)
        _add_ellipse(r_outer, facecolor='none', edgecolor='black', linewidth=0.8, zorder=12)

    # ── Step 4: center dot when clipped ──────────────────────────────────────
    if is_clipped:
        fig.add_artist(_Ellipse(
            (cx_f, cy_f), 2 * 2.5 / fw, 2 * 2.5 / fh,
            color='black', transform=T, zorder=13, clip_on=False,
        ))

    # ── Step 5: r= label (outermost radius, formatted as km/m) ───────────────
    def _fmt_r_label(v):
        if v >= 1000:
            return 'r=' + f'{v/1000:.3f}'.rstrip('0').rstrip('.') + 'km'
        return 'r=' + f'{v:.3f}'.rstrip('0').rstrip('.') + 'm'

    r_str = _fmt_r_label(r_outer)
    _fs   = 6
    if not is_clipped and 2 * r_outer_disp > 28:
        fig.text(cx_f, cy_f, r_str, fontsize=_fs, ha='center', va='center', zorder=14)
    elif placement == 'y':
        if too_wide:
            # no horizontal room: text above the circle (still in the y-axis margin)
            lx_f, ly_f = _f(cx_disp, cy_disp + r_outer_disp + 2)
            fig.text(lx_f, ly_f, r_str, fontsize=_fs, ha='center', va='bottom', zorder=14)
        else:
            # too tall: text below the circle, centered in the y-axis margin
            lx_f, ly_f = _f(cx_disp, cy_disp - r_outer_disp - 2)
            fig.text(lx_f, ly_f, r_str, fontsize=_fs, ha='center', va='top', zorder=14)
    elif too_wide:
        lx_f, ly_f = _f(cx_disp + 4, cy_disp)
        fig.text(lx_f, ly_f, r_str, fontsize=_fs, ha='left', va='center', zorder=14)
    else:
        lx_f, ly_f = _f(cx_disp, cy_disp - r_outer_disp - 2)
        fig.text(lx_f, ly_f, r_str, fontsize=_fs, ha='center', va='top', zorder=14)


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

def plot_polygon(poly, ax=None, **kwargs):
    if ax is None:
        fig, ax = _plt_subplots()

    geoms = [poly] if type(poly) == _shapely_Polygon else list(poly.geoms)
    for geom in geoms:
        path = _plt_Path.make_compound_path(
            _plt_Path(_np_asarray(geom.exterior.coords)[:, :2]),
            *[_plt_Path(_np_asarray(ring.coords)[:, :2]) for ring in geom.interiors]
            )

        patch = _plt_PathPatch(path, **kwargs)
        collection = _plt_PatchCollection([patch], **kwargs)
        
        ax.add_collection(collection, autolim=True)
        ax.autoscale_view()
    return collection
#
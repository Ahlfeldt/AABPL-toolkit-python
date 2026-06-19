# intersection of two circles with same radius
from numpy import array as _np_array, sign as _np_sign, arange as _np_arange, invert as _np_invert, zeros as _np_zeros
from numpy.linalg import norm as _np_linalg_norm
import numpy as _np
from aabpl import config as _config
from aabpl.utils.progress import _OUTER_PROGRESS as _outer_progress
from math import log10 as _math_log10, pi as _math_pi, floor as _math_floor
from matplotlib import pyplot as plt
from aabpl.utils.misc import make_bins_from_vals, get_vals_from_bins
from aabpl.utils.cell_geometry import (get_cell_closest_point_to_point, get_cell_farthest_vertex_to_point,
 cells_never_contain, cells_always_overlap,
 classify_disk_cells, classify_disk_cells_by_level)
from .region_classes import OffsetRegion, Vertex, LineSegment, Circle, Edge
from aabpl.illustrations.plot_cell_pattern import plot_cell_pattern
from aabpl.testing.test_performance import time_func_perf
from matplotlib.patches import (Rectangle as _plt_Rectangle, Polygon as _plt_Polygon, Circle as _plt_Circle)
from matplotlib.pyplot import (step, subplots as _plt_subplots, colorbar as _plt_colorbar, get_cmap as _plt_get_cmap)
from shapely.geometry import Polygon, LineString, Point
import inspect # remove after testing
from random import shuffle as _random_shuffle 
from matplotlib.pyplot import close as _plt_close

class _ArcCheckStrict:
    __slots__ = ('center', 'r')
    def __init__(self, x, y, r):
        self.center = _np_array([x, y])
        self.r = r
    def __call__(self, pts):
        return _np_linalg_norm(pts - self.center, axis=1) < self.r

class _ArcCheckBoundary:
    __slots__ = ('center', 'r')
    def __init__(self, x, y, r):
        self.center = _np_array([x, y])
        self.r = r
    def __call__(self, pts):
        return _np_linalg_norm(pts - self.center, axis=1) <= self.r

class _LineCheckStrict:
    __slots__ = ('col_index', 'val')
    def __init__(self, col_index, val):
        self.col_index = col_index
        self.val = val
    def __call__(self, pts):
        return pts[:, self.col_index] > self.val

class _LineCheckBoundary:
    __slots__ = ('col_index', 'val')
    def __init__(self, col_index, val):
        self.col_index = col_index
        self.val = val
    def __call__(self, pts):
        return pts[:, self.col_index] >= self.val

class _ConstRegionCheck:
    __slots__ = ('region_id',)
    def __init__(self, region_id):
        self.region_id = region_id
    def __call__(self, pts):
        return _np_zeros(len(pts), int) + self.region_id

class _TreeCheck:
    __slots__ = ('check_tree',)
    def __init__(self, check_tree):
        self.check_tree = check_tree
    def __call__(self, pts):
        return eval_region_tree(pts=pts, check_tree=self.check_tree)


def init_triangle1_region(
        clear_all:bool=True,
        convex_set_coordiantes:list=[(0.,0.), (0.5,0.), (0.5,0.5)]
) -> dict:
    """ Creates OffsetRegion for trianle 1: [(0.,0.), (0.5,0.), (0.5,0.5)].
    Delete all Regions, Edges, Vertices from dicts. Create new vertices, edges and region for triangle 1
    """
    if clear_all: 
        all_regions = dict()
        all_edges = dict()
        all_vtx = dict()
    
    vertices_set = [Vertex(x=x, y=y, all_vtx=all_vtx) for (x,y) in convex_set_coordiantes]
    edges_set = [LineSegment(vtx1=v1, vtx2=v2, all_edges=all_edges) for v1,v2 in zip(vertices_set, vertices_set[1:]+vertices_set[:1])]
    OffsetRegion(edges=edges_set, checks=[], all_regions=all_regions)
    
    return all_regions
#


def register_arc_check(
        cell:_np_array,
        nev_cn:bool,
        alw_ov:bool,
        check_dict,
        r:float,
        grid_spacing:float,
):
    """
    Register arc-based boundary checks for a diagonal cell (row != 0 and col != 0).

    For cells that are not always-ovlpd, adds a Circle check keyed on the cell's
    closest point to the triangle reference point (0.25, 0.125): a point passes if its
    distance to that anchor is < r/grid_spacing (overlap test).
    For cells that are not never-cntd, adds a Circle check keyed on the cell's
    farthest vertex from the reference point: a point passes if its distance is < r/grid_spacing
    (containment test).
    Both checks are stored in check_dict and linked to their respective cell via
    'overlaps' / 'contains' attributes on the Circle split_edge object.
    """
    trgl_pt = (0.25,0.125)
    point_in_triangle1 = _np_array([(0.25,0.125)]) # TODO remove wraping list
    
    if not alw_ov:
        closest_pt = tuple([float(v) for v in get_cell_closest_point_to_point(trgl_pt, cell)])
        if closest_pt not in check_dict:
            check_dict[closest_pt] = {'split_edge': Circle(center=closest_pt, r=r/grid_spacing)}
        check_dict[closest_pt]['overlaps'] = cell
        check_dict[closest_pt]['split_edge'].overlaps = tuple([*cell])
    #

    if not nev_cn:
        farthest_pt = tuple([float(v) for v in get_cell_farthest_vertex_to_point(point_in_triangle1, cell)[0]])
        if farthest_pt not in check_dict:
            check_dict[farthest_pt] = {'split_edge': Circle(center=farthest_pt, r=r/grid_spacing)}

        if hasattr(check_dict[farthest_pt]['split_edge'], 'contains'):
            raise ValueError("\n\n2check_dict[farthest_pt]['split_edge'].contains\n\n",check_dict[farthest_pt]['split_edge'].contains)
        check_dict[farthest_pt]['contains'] = cell
        check_dict[farthest_pt]['split_edge'].contains = tuple([*cell])
        if not hasattr(check_dict[farthest_pt]['split_edge'], 'contains'):
            raise NotImplementedError("\n\n2check_dict[farthest_pt]['split_edge'].contains\n\n",check_dict[farthest_pt]['split_edge'].contains)
        
    #
#

def register_line_check(
        cell:_np_array,
        nev_cn:bool,
        alw_ov:bool,
        check_dict,
        all_regions:dict,
        r:float,
        grid_spacing:float,
):
    """
    Register axis-aligned line checks for an on-axis cell (row == 0 or col == 0).

    For cells not always-ovlpd, creates a horizontal or vertical LineSegment
    whose position is derived from the cell's distance from the origin along the non-zero
    axis and the search radius r. A point passes the overlap check if it lies on the
    same side as the triangle interior (sign convention: left of the vector → True).
    For cells not never-cntd, adds a Circle check on the cell's farthest corner
    (containment test, same logic as register_arc_check).
    Raises NotImplementedError when row == 0 and col == 0 (origin cell), which requires
    grid_spacing > r / sqrt(2).
    """
        
    row, col = [int(c) for c in cell]
    
    an_edge  = list(all_regions.values())[-1].edges[-1]
    all_edges = an_edge.all_edges
    all_vtx = an_edge.vtx1.all_vtx
    
    assert not (row == 0 and col == 0), (
        "register_line_check called for origin cell (0,0) — "
        "build_boundary_checks should have routed it to register_arc_check."
    )
    
    if not alw_ov:
        if col == 0.:
            split_edge = LineSegment(
                vtx1=Vertex(0.0-0.001 if row > 0 else 0.5+0.001,  (row - (r - .5*-1) * _np_sign(row)), all_vtx),
                vtx2=Vertex(0.0-0.001 if row < 0 else 0.5+0.001,  (row - (r - .5*-1) * _np_sign(row)), all_vtx), 
                all_edges=all_edges
            )
        else:
            split_edge = LineSegment(
                vtx1=Vertex((col - (r - .5*-1) * _np_sign(col)), 0.5+0.001 if col > 0 else 0.0-0.001, all_vtx),
                vtx2=Vertex((col - (r - .5*-1) * _np_sign(col)), 0.5+0.001 if col < 0 else 0.0-0.001, all_vtx), 
                all_edges=all_edges
            )
        split_edge.overlaps = (row, col)
        check_dict[split_edge] = {'split_edge': split_edge, 'overlaps': (row,col)}
    
    if not nev_cn:
        farthest_pt = (
            -0.5 if col == 0 else col + .5 * _np_sign(col),
            -0.5 if row == 0 else row + .5 * _np_sign(row)
        )
        if farthest_pt not in check_dict:
            check_dict[farthest_pt] = {'split_edge': Circle(center=farthest_pt, r=r/grid_spacing)}
        if hasattr(check_dict[farthest_pt]['split_edge'], 'contains'):
            raise ValueError("\n\ncheck_dict[farthest_pt]['split_edge'].contains\n\n",check_dict[farthest_pt]['split_edge'].contains)
        check_dict[farthest_pt]['split_edge'].contains = (row, col)
        if hasattr(check_dict[farthest_pt], 'contains'):
            raise NotImplementedError("\n\ncheck_dict[farthest_pt]['split_edge'].contains\n\n",check_dict[farthest_pt]['split_edge'].contains)
        check_dict[farthest_pt]['contains'] = cell
        # # THESE RESULTS WILL BE CHECKED ANYWAYS. AT THE END YOU CAN REQUEST THOSE RESULTS
        # check_dict[(x,y)] = {'cells_to_overlap': cells_to_overlap, 'contains': (x,y)}
    #
#

def build_boundary_checks(
    cells_to_check,
    all_regions:dict,
    r:float,
    grid_spacing:float=1,
    include_boundary: bool = False        
):
    """
    Build a dict of boundary checks for all cells that potentially overlap or are cntd
    by the disk of radius r centred anywhere inside triangle 1 [(0,0),(0.5,0),(0.5,0.5)].

    For each candidate cell the function first determines whether it is always-ovlpd
    (overlaps every disk in the triangle) and/or never-cntd (never fully inside any disk).
    Cells where both flags are True are always-only-ovlpd and require no further checks.
    For the remaining cells, axis-aligned cells (row==0 or col==0) get line checks via
    register_line_check; diagonal cells get arc (circle) checks via register_arc_check.
    Returns the populated check_dict and the subset of cells_to_check that are always-ovlpd.
    """
    check_dict = dict()
    triangle_1_vertices = _np_array([[0, 0], [0.5, 0], [0.5, 0.5]])

    cells_always_ovlpd = cells_always_overlap(
        cells=cells_to_check,
        convex_set_vertices=triangle_1_vertices,
        r=r,
        grid_spacing=grid_spacing,
        vertex_is_inside_convex_set=True,
        include_boundary=include_boundary,
    )

    cells_never_cntd = cells_never_contain(
        cells=cells_to_check,
        convex_set_vertices=triangle_1_vertices,
        r=r,
        grid_spacing=grid_spacing,
        vertex_is_inside_convex_set=True,
        include_boundary=include_boundary,
    )

    cells_alw_only_ovlpd = []

    for cell, alw_ov, nev_cn in zip(cells_to_check, cells_always_ovlpd, cells_never_cntd):
        if alw_ov and nev_cn:
            cells_alw_only_ovlpd.append(cell) # TODO this can be removed - no longer necessary to store those.
            continue  # no boundary check needed: always overlapped, never contained
        row, col = int(cell[0]), int(cell[1])
        if row == 0 and col == 0:
            # Origin cell: source point is always inside it (always overlapped) and its
            # farthest vertex from any triangle-1 point is the diagonal corner (-0.5,-0.5).
            # Use an arc check for the containment boundary — line checks don't apply here.
            register_arc_check(cell=cell, nev_cn=nev_cn, alw_ov=alw_ov, check_dict=check_dict, r=r, grid_spacing=grid_spacing,)
        elif 0 in cell: # cell in same column or row
            register_line_check(cell=cell, nev_cn=nev_cn, alw_ov=alw_ov, check_dict=check_dict, all_regions=all_regions, r=r, grid_spacing=grid_spacing,)
        else:
            register_arc_check(cell=cell, nev_cn=nev_cn, alw_ov=alw_ov, check_dict=check_dict, r=r, grid_spacing=grid_spacing,)
        #
    #
    return check_dict, cells_to_check[cells_always_ovlpd]
#

def split_regions_by_checks(
        check_dict,
        trgl_regions:dict,
        r:float,
        plot_offset_checks:dict=None,
        axs = None
    ):
    """
    Iteratively split triangle-1 regions by applying each check in check_dict.

    Starts with the single initial region covering triangle 1. For each boundary check
    (arc or line), calls region.split_with_edge(), which subdivides any region that straddles
    the check boundary into two new sub-regions and records the check result on each.
    After all checks are applied every region holds a complete list of True/False results
    that together uniquely identify which sub-region of the disk pattern it corresponds to.
    Raises ValueError if the number of recorded checks differs between regions (sanity guard).
    Optionally renders each split step to the supplied axes via plot_offset_checks.
    """
    if not plot_offset_checks is None:
        if axs is None:
            nrows = int(len(check_dict)**.5)
            ncols = -int(-len(check_dict.items())//nrows) 
            fig,axs = plt.subplots(nrows, ncols, figsize=(ncols*5,nrows*5))

    for i, (key, check) in enumerate(check_dict.items()):
        colors=[]
        split_edge = check['split_edge']
        regions = list(trgl_regions.values())
        # split each region with edge
        for region in regions:
            # check if any pt is within readius
            color = region.split_with_edge(split_edge, check, plot_split=False, r=r)
            colors.append(color)
        
        if not plot_offset_checks is None:
            ax = axs.flat[i]
            OffsetRegion.plot_many(regions=regions, ax=ax, add_idxs=False, facecolor=colors, edgecolor='black', alpha=0.8, plot_edges=False)
            split_edge.plot_single(ax=ax, radial_lines=False, full_circle=True, linewidth=2, facecolor='None', edgecolor='black')
    
    if not plot_offset_checks is None and 'savefig' in plot_offset_checks:
        if type(plot_offset_checks['savefig'])==dict:
            plot_offset_checks['savefig'] = {}
        savefig_kwargs = {'fname':''+str(int(r))+"_"+str(r%1)[2:]+".png", 'dpi':100, 'bbox_inches':"tight", **plot_offset_checks['savefig']}
        fig.savefig(**savefig_kwargs)
        _plt_close(fig)
    
    # Check if all checks are performaned on each, else throw error 
    n_checks = [len(region.checks) for region in list(trgl_regions.values())]
    if not n_checks.count(n_checks[0]) == len(n_checks):
        raise ValueError("The number of checks performed differ among micro regions. They are expected to be all of the same Length.", n_checks)
    #
#

def classify_subcell_quadrants(
        r:float,
        grid_spacing:float,
        vtx_ovlpd:dict,
        vtx_cntd:dict,
        region,
        q_anchor_row=0,
        q_anchor_col=0,
        lvl:int=1,
        nest_depth:int=0,
        include_boundary:bool=False,
    ):
    """
    function that recursivey splits cell into quadrants until max nest_depth is reached or the cell/subquadrant is fully cntd or not ovlpd.
    """
    quarterstep = 2**(-lvl-1)
    halfstep = 2**(-lvl)
    fullstep = 2**(1-lvl)
    
    vtx_pts = [
        ( q_anchor_col,              q_anchor_row            ), #0 (  0%,   0%)   lef-bot
        ( q_anchor_col + halfstep,   q_anchor_row            ), #1 ( 50%,   0%)   cen-bot
        ( q_anchor_col,              q_anchor_row + halfstep ), #2 (  0%,  50%)   lef-mid
        ( q_anchor_col + halfstep,   q_anchor_row + halfstep ), #3 ( 50%,  50%)   cen-mid
        ( q_anchor_col + fullstep,   q_anchor_row            ), #4 (100%,   0%)   rig-bot
        ( q_anchor_col,              q_anchor_row + fullstep ), #5 (  0%, 100%)   lef-top
        ( q_anchor_col + fullstep,   q_anchor_row + halfstep ), #6 (100%,  50%)   rig-mid
        ( q_anchor_col + halfstep,   q_anchor_row + fullstep ), #7 ( 50%, 100%)   cen-top
        ( q_anchor_col + fullstep,   q_anchor_row + fullstep ), #8 (100%, 100%)   rig-top
    ]

    q0 = [0,1,3,2] # (  0%,   0%)  ( 50%,   0%)  ( 50%,  50%)  (  0%,  50%) lef-bot
    q1 = [1,4,6,3] # ( 50%,   0%)  (100%,   0%)  (100%,  50%)  ( 50%,  50%) rig-bot 
    q2 = [2,3,7,5] # (  0%,  50%)  ( 50%,  50%)  ( 50%, 100%)  (  0%, 100%) lef-top
    q3 = [3,6,8,7] # ( 50%,  50%)  (100%,  50%)  (100%, 100%)  ( 50%, 100%) rig-top
    
    #                   lef-bot,    rig-bot,    lef-top,        rig-top
    quadrants =        [  q0,         q1,         q2,             q3              ]
    # quadrant_offsets = [(0, 0), (0, halfstep), (halfstep, 0), (halfstep, halfstep)]
    quadrant_offsets = [(0, 0), (halfstep, 0), (0, halfstep), (halfstep, halfstep)]
 

    for vtx_pt in vtx_pts:
        
        if vtx_pt not in vtx_cntd:
            max_dist_to_vtx = 0
            # loop over edges until an edge is found with a max distance greate than r. 
            for edge in region.edges:
                # if edge.is_within_dist_to_pt(vtx_pt,r):
                if edge.calc_max_dist_to_pt(vtx_pt) > r / grid_spacing:
                    # update max dist with arbitrary value > r/grid_spacing.
                    max_dist_to_vtx = r / grid_spacing + 1
                    break
            if max_dist_to_vtx <= r / grid_spacing:
                vtx_cntd[vtx_pt] = True
                vtx_ovlpd[vtx_pt] = True
            else:
                vtx_cntd[vtx_pt] = False
        if vtx_pt not in vtx_ovlpd:
            # if edge.pt_is_within_r(vtx_pt,r):
            min_dist_to_vtx = region.calc_min_dist_to_pt(vtx_pt)
            vtx_ovlpd[vtx_pt] = min_dist_to_vtx < r / grid_spacing if not include_boundary else min_dist_to_vtx <= r / grid_spacing

   
    cntd_subcells, ovlpd_subcells, discarded_subcells = [], [], []
    for q, (a_c, a_r) in zip(quadrants, quadrant_offsets):
        row_centroid = q_anchor_row + a_r + quarterstep
        col_centroid = q_anchor_col + a_c + quarterstep
        # if all vertices are cntd
        if all([vtx_cntd[vtx_pts[i]] for i in q]):
            cntd_subcells.append((lvl,(row_centroid,col_centroid)))
        # if any vertex is ovlpd nest deeper unless limit is reached
        elif any([vtx_ovlpd[vtx_pts[i]] for i in q]):
            if lvl < nest_depth:
                # --- MINIMALLY INVASIVE FIX 2: Pass down the updated column/row anchors correctly ---
                sub_c, sub_o, sub_d = classify_subcell_quadrants(
                    r=r,
                    grid_spacing=grid_spacing,
                    vtx_ovlpd=vtx_ovlpd,
                    vtx_cntd=vtx_cntd,
                    region=region,
                    q_anchor_row=q_anchor_row + a_r,
                    q_anchor_col=q_anchor_col + a_c,
                    lvl=lvl+1,
                    nest_depth=nest_depth,
                    include_boundary=include_boundary,
                )
                cntd_subcells.extend(sub_c)
                ovlpd_subcells.extend(sub_o)
                discarded_subcells.extend(sub_d)
            else:
                ovlpd_subcells.append((lvl,(row_centroid,col_centroid)))
        # subcell is outside the disk entirely — discard but track for cross-region merge
        else:
            discarded_subcells.append((lvl,(row_centroid,col_centroid)))
    return cntd_subcells, ovlpd_subcells, discarded_subcells


# ---------------------------------------------------------------------------
# Vectorised variant — drop-in replacement for classify_subcell_quadrants
# Toggle via config.USE_VEC_VERTEX_CHECKS (aabpl/config.py, default True).
# ---------------------------------------------------------------------------


def _build_edge_arrays(region):
    """
    Pre-extract edge endpoint coordinates into numpy arrays and cache them on the
    region object.  Called once per region; subsequent calls are free.
    """
    if not hasattr(region, '_edge_p1_arr'):
        region._edge_p1_arr = _np.array(
            [[e.vtx1.x, e.vtx1.y] for e in region.edges], dtype=float)
        region._edge_p2_arr = _np.array(
            [[e.vtx2.x, e.vtx2.y] for e in region.edges], dtype=float)
    return region._edge_p1_arr, region._edge_p2_arr


def _batch_vtx_checks(vtx_pts, vtx_cntd, vtx_ovlpd,
                      r_over_s, include_boundary, edge_p1, edge_p2):
    """
    Batch-evaluate containment and overlap for all vtx_pts not already cached.
    Updates vtx_cntd and vtx_ovlpd in-place; semantics match the scalar loop in
    classify_subcell_quadrants exactly.

    Containment (vtx_cntd):
        A vertex v is contained iff for EVERY source offset p in the region
        ||v - p|| <= r.  Since region is convex, the worst-case p is at a polygon
        vertex, which is an edge endpoint.  So:
            contained  iff  max over all edge endpoints of dist(endpoint, v) <= r

    Overlap (vtx_ovlpd):
        A vertex v is overlapped iff there EXISTS a source offset p in the region
        with ||v - p|| < r  (or <= r if include_boundary).
        Equivalently: min distance from the region (including interior) to v < r.
        For a convex polygon the minimum distance is 0 when v is inside.
    """
    # --- containment ---
    need_cntd = [vp for vp in vtx_pts if vp not in vtx_cntd]
    if need_cntd:
        V  = _np.array(need_cntd, dtype=float)          # (N, 2)
        d1 = _np.linalg.norm(
            V[:, None, :] - edge_p1[None, :, :], axis=-1)   # (N, M)
        d2 = _np.linalg.norm(
            V[:, None, :] - edge_p2[None, :, :], axis=-1)   # (N, M)
        contained = _np.all(_np.maximum(d1, d2) <= r_over_s, axis=1)  # (N,)
        for i, vp in enumerate(need_cntd):
            if bool(contained[i]):
                vtx_cntd[vp] = True
                vtx_ovlpd[vp] = True   # contained → always overlapped
            else:
                vtx_cntd[vp] = False

    # --- overlap (only for vertices not already resolved by containment) ---
    need_ovlpd = [vp for vp in vtx_pts if vp not in vtx_ovlpd]
    if need_ovlpd:
        V    = _np.array(need_ovlpd, dtype=float)       # (N, 2)
        seg  = edge_p2 - edge_p1                         # (M, 2)
        len2 = _np.sum(seg ** 2, axis=1)                 # (M,)
        safe = _np.where(len2 > 0, len2, 1.0)

        diff = V[:, None, :] - edge_p1[None, :, :]      # (N, M, 2)
        t    = _np.clip(
            _np.sum(diff * seg[None, :, :], axis=-1) / safe,  # (N, M)
            0.0, 1.0)
        proj     = edge_p1[None, :, :] + t[:, :, None] * seg[None, :, :]  # (N, M, 2)
        min_dist = _np.sqrt(_np.min(
            _np.sum((V[:, None, :] - proj) ** 2, axis=-1), axis=1))       # (N,)

        # Points inside the convex polygon have distance 0 to the region.
        # Detect interior: all cross-products (edge_dir × (v - edge_start)) same sign.
        cross = ((edge_p2[None,:,0] - edge_p1[None,:,0]) * (V[:,None,1] - edge_p1[None,:,1])
               - (edge_p2[None,:,1] - edge_p1[None,:,1]) * (V[:,None,0] - edge_p1[None,:,0]))
        inside = (_np.all(cross >= -1e-10, axis=1) |
                  _np.all(cross <=  1e-10, axis=1))
        min_dist = _np.where(inside, 0.0, min_dist)

        op = _np.less_equal if include_boundary else _np.less
        overlapped = op(min_dist, r_over_s)
        for i, vp in enumerate(need_ovlpd):
            vtx_ovlpd[vp] = bool(overlapped[i])


def classify_subcell_quadrants_vec(
        r: float,
        grid_spacing: float,
        vtx_ovlpd: dict,
        vtx_cntd: dict,
        region,
        q_anchor_row=0,
        q_anchor_col=0,
        lvl: int = 1,
        nest_depth: int = 0,
        include_boundary: bool = False,
):
    """
    Vectorised drop-in replacement for classify_subcell_quadrants.

    The recursion structure and return value are identical.  The only difference
    is that all 9 vertex checks for a quadrant level are evaluated in one numpy
    batch call (_batch_vtx_checks) instead of per-vertex Python loops over edges.

    Edge endpoint arrays are pre-extracted once per region via _build_edge_arrays
    and cached on the region object, so the extraction cost is paid only on the
    first call per region.
    """
    quarterstep = 2**(-lvl - 1)
    halfstep    = 2**(-lvl)
    fullstep    = 2**(1 - lvl)

    vtx_pts = [
        (q_anchor_col,             q_anchor_row            ),  # 0 lef-bot
        (q_anchor_col + halfstep,  q_anchor_row            ),  # 1 cen-bot
        (q_anchor_col,             q_anchor_row + halfstep ),  # 2 lef-mid
        (q_anchor_col + halfstep,  q_anchor_row + halfstep ),  # 3 cen-mid
        (q_anchor_col + fullstep,  q_anchor_row            ),  # 4 rig-bot
        (q_anchor_col,             q_anchor_row + fullstep ),  # 5 lef-top
        (q_anchor_col + fullstep,  q_anchor_row + halfstep ),  # 6 rig-mid
        (q_anchor_col + halfstep,  q_anchor_row + fullstep ),  # 7 cen-top
        (q_anchor_col + fullstep,  q_anchor_row + fullstep ),  # 8 rig-top
    ]

    edge_p1, edge_p2 = _build_edge_arrays(region)
    _batch_vtx_checks(vtx_pts, vtx_cntd, vtx_ovlpd,
                      r / grid_spacing, include_boundary, edge_p1, edge_p2)

    q0 = [0, 1, 3, 2]
    q1 = [1, 4, 6, 3]
    q2 = [2, 3, 7, 5]
    q3 = [3, 6, 8, 7]
    quadrants       = [q0, q1, q2, q3]
    quadrant_offsets = [(0, 0), (halfstep, 0), (0, halfstep), (halfstep, halfstep)]

    cntd_subcells, ovlpd_subcells, discarded_subcells = [], [], []
    for q, (a_c, a_r) in zip(quadrants, quadrant_offsets):
        row_centroid = q_anchor_row + a_r + quarterstep
        col_centroid = q_anchor_col + a_c + quarterstep
        if all(vtx_cntd[vtx_pts[i]] for i in q):
            cntd_subcells.append((lvl, (row_centroid, col_centroid)))
        elif any(vtx_ovlpd[vtx_pts[i]] for i in q):
            if lvl < nest_depth:
                sub_c, sub_o, sub_d = classify_subcell_quadrants_vec(
                    r=r, grid_spacing=grid_spacing,
                    vtx_ovlpd=vtx_ovlpd, vtx_cntd=vtx_cntd,
                    region=region,
                    q_anchor_row=q_anchor_row + a_r,
                    q_anchor_col=q_anchor_col + a_c,
                    lvl=lvl + 1, nest_depth=nest_depth,
                    include_boundary=include_boundary,
                )
                cntd_subcells.extend(sub_c)
                ovlpd_subcells.extend(sub_o)
                discarded_subcells.extend(sub_d)
            else:
                ovlpd_subcells.append((lvl, (row_centroid, col_centroid)))
        else:
            discarded_subcells.append((lvl, (row_centroid, col_centroid)))

    return cntd_subcells, ovlpd_subcells, discarded_subcells


def merge_nested_cells_bottom_up(cntd_cells, ovlpd_cells):
    """
    Bottom-up pass: from most disaggregated to most aggregated level, merge
    4 siblings into their parent key whenever all 4 fall in the same category
    (all cntd or all ovlpd) and none were discarded.
    Returns updated (cntd_cells, ovlpd_cells) lists.
    """
    cntd = set(cntd_cells)
    ovlpd = set(ovlpd_cells)

    if not cntd and not ovlpd:
        return list(cntd), list(ovlpd)

    from collections import defaultdict
    max_lvl = max(lvl for lvl, _ in (cntd | ovlpd))

    for lvl in range(max_lvl, 0, -1):
        step = 2.0 ** (-lvl)
        parent_step = 2.0 * step  # = 2^(-(lvl-1))

        def parent_key(r, c, _step=step, _pstep=parent_step, _lvl=lvl):
            if _lvl - 1 == 0:
                # Level-0 centres are integers; anchors are half-integers so the
                # general formula breaks — use round-half-up directly.
                return (0, (int(_math_floor(r + 0.5)), int(_math_floor(c + 0.5))))
            pr = r - (r % _pstep) + _step
            pc = c - (c % _pstep) + _step
            return (_lvl - 1, (pr, pc))

        # group all level-lvl cells by their parent
        groups = defaultdict(lambda: {'c': [], 'o': []})
        for key in list(cntd):
            if key[0] == lvl:
                groups[parent_key(*key[1])]['c'].append(key)
        for key in list(ovlpd):
            if key[0] == lvl:
                groups[parent_key(*key[1])]['o'].append(key)

        for pkey, members in groups.items():
            c_members, o_members = members['c'], members['o']
            total = len(c_members) + len(o_members)
            if total != 4:
                continue  # some siblings were discarded → no merge
            if len(c_members) == 4:
                for k in c_members:
                    cntd.discard(k)
                cntd.add(pkey)
            elif len(o_members) == 4:
                for k in o_members:
                    ovlpd.discard(k)
                ovlpd.add(pkey)
            # mixed categories → no merge

    return list(cntd), list(ovlpd)


def get_children(pkey):
    """Return the 4 level-(k+1) child keys of a level-k cell."""
    lvl, (pr, pc) = pkey
    delta = 2.0 ** (-(lvl + 2))
    child_lvl = lvl + 1
    return [
        (child_lvl, (pr - delta, pc - delta)),
        (child_lvl, (pr - delta, pc + delta)),
        (child_lvl, (pr + delta, pc - delta)),
        (child_lvl, (pr + delta, pc + delta)),
    ]


def merge_nested_cells_consistent(regions_raw, nest_depth):
    """
    Cross-region bottom-up merge.

    regions_raw : list of (c_set, o_set, d_set) — one tuple per region.
    Returns      : list of (c_set, o_set, d_set) with the same length,
                   where cells are merged to a parent only when all four
                   siblings agree on the same category in every region.
    """
    from collections import defaultdict

    n_regions = len(regions_raw)
    # Work with mutable sets; use 'c', 'o', 'd' category labels
    cats = [{'c': set(c), 'o': set(o), 'd': set(d)}
            for c, o, d in regions_raw]

    # ------------------------------------------------------------------ #
    # Phase 1 – expand to common granularity                              #
    # A cell at level k in region A must be split into its 4 children     #
    # wherever another region has those children explicitly listed.        #
    # ------------------------------------------------------------------ #
    for lvl in range(1, nest_depth + 1):
        child_lvl = lvl + 1
        if child_lvl > nest_depth:
            continue
        # Collect all child-level keys that exist in any region
        all_child_keys = set()
        for cat_dict in cats:
            for cat in ('c', 'o', 'd'):
                for key in cat_dict[cat]:
                    if key[0] == child_lvl:
                        all_child_keys.add(key)
        if not all_child_keys:
            continue
        # Derive the set of parents that have at least one explicit child
        parents_with_children = set()
        for child_key in all_child_keys:
            child_lvl_k, (cr, cc) = child_key
            # Compute parent key (same formula as merge_nested_cells_bottom_up)
            step = 2.0 ** (-child_lvl_k)
            parent_step = 2.0 * step
            if child_lvl_k - 1 == 0:
                pk = (0, (int(_math_floor(cr + 0.5)), int(_math_floor(cc + 0.5))))
            else:
                pr = cr - (cr % parent_step) + step
                pc = cc - (cc % parent_step) + step
                pk = (child_lvl_k - 1, (pr, pc))
            parents_with_children.add(pk)
        # For each region, expand parent keys that appear as aggregated
        for cat_dict in cats:
            for cat in ('c', 'o', 'd'):
                to_expand = parents_with_children & cat_dict[cat]
                for pkey in to_expand:
                    cat_dict[cat].discard(pkey)
                    for ck in get_children(pkey):
                        cat_dict[cat].add(ck)

    # ------------------------------------------------------------------ #
    # Phase 2 – cross-region bottom-up merge                              #
    # ------------------------------------------------------------------ #
    # Find the maximum level present across all regions
    all_keys = set()
    for cat_dict in cats:
        for cat in ('c', 'o', 'd'):
            all_keys |= cat_dict[cat]
    if not all_keys:
        return [(c['c'], c['o'], c['d']) for c in cats]

    max_lvl = max(key[0] for key in all_keys)

    for lvl in range(max_lvl, 0, -1):
        step = 2.0 ** (-lvl)
        parent_step = 2.0 * step

        def _parent_key(r, c, _step=step, _pstep=parent_step, _lvl=lvl):
            if _lvl - 1 == 0:
                return (0, (int(_math_floor(r + 0.5)), int(_math_floor(c + 0.5))))
            pr = r - (r % _pstep) + _step
            pc = c - (c % _pstep) + _step
            return (_lvl - 1, (pr, pc))

        # Collect all level-lvl keys across all regions grouped by parent
        groups = defaultdict(lambda: defaultdict(set))  # pkey -> region_idx -> {keys}
        for ri, cat_dict in enumerate(cats):
            for cat in ('c', 'o', 'd'):
                for key in cat_dict[cat]:
                    if key[0] == lvl:
                        pk = _parent_key(*key[1])
                        groups[pk][ri].add(key)

        for pkey, region_map in groups.items():
            # Check: each region must have exactly 4 children for this parent
            # and all must agree on the same single category
            merge_cat = None
            can_merge = True
            for ri in range(n_regions):
                children_in_region = region_map.get(ri, set())
                if len(children_in_region) != 4:
                    can_merge = False
                    break
                # determine category in this region
                reg_cat = None
                for cat in ('c', 'o', 'd'):
                    if children_in_region <= cats[ri][cat]:
                        reg_cat = cat
                        break
                if reg_cat is None:
                    # children split across categories in this region
                    can_merge = False
                    break
                if merge_cat is None:
                    merge_cat = reg_cat
                elif reg_cat != merge_cat:
                    can_merge = False
                    break

            if can_merge and merge_cat is not None:
                for ri in range(n_regions):
                    for ck in region_map[ri]:
                        cats[ri][merge_cat].discard(ck)
                    cats[ri][merge_cat].add(pkey)

    return [(c['c'], c['o'], c['d']) for c in cats]


def finalize_region_cells(
        trgl_regions:dict, 
        cells_cntd_in_all_trgl_disks:_np_array,
        cells_always_ovlpd:list,
        cells_cntd_in_all_disks:list,
        all_cells:_np_array,
        grid_spacing:float,
        r:float, 
        include_boundary:bool=False,
        nest_depth:int=0,
        plot_offset_regions:dict=None):
    """
    Finalise each triangle-1 region's cntd/ovlpd cell lists and compute nested sub-cells.

    Reads each region's check results to build region.cntd_cells and region.ovlpd_cells
    (level-0 grid cells). Cells always cntd/ovlpd across all disks in the triangle are
    merged in unconditionally. When nest_depth > 0, each ovlpd cell is recursively split into
    sub-cell quadrants via classify_subcell_quadrants; the resulting raw classifications are then made
    consistent across all regions by merge_nested_cells_consistent, which ensures that a parent cell
    is only aggregated back when all four siblings agree in every region.
    Also sets region.shared_along_vert and region.shared_along_diag flags used later by
    expand_regions_to_all_sectors to decide how many of the 8 symmetry sectors
    can share a single merged region object.
    """
    cells_cntd_in_all_disks = [(0,(int(row),int(col))) for row,col in cells_cntd_in_all_disks]
    cells_cntd_in_all_trgl_disks = [(0,(int(row),int(col))) for row,col in cells_cntd_in_all_trgl_disks]
    cells_always_ovlpd = [(0,(int(row),int(col))) for row,col in cells_always_ovlpd]
    # print("N always ovlpd",len(cells_always_ovlpd))
    # now all checks are added to regions
    # ensure that each region.checks has the same length!
    if not plot_offset_regions is None:
        _por = plot_offset_regions
        fig, axs = plt.subplots(nrows=len(trgl_regions), ncols=2, figsize=_por.get('figsize', (8, 4*len(trgl_regions))))
    n_regs, n_edges = 0,0
    for n, region in enumerate(list(trgl_regions.values())):
        n_regs+=1

        n_edges+=len(region.edges)
        # print(n_edges, len(region.all_edges))
        region.cntd_cells = []
        region.ovlpd_cells = []
        for check in region.checks:
            if not check['result'] == True:
                continue
            if 'contains' in check:
                region.cntd_cells.append((0,tuple([int(v) for v in check['contains']])))
            if 'overlaps' in check:
                region.ovlpd_cells.append((0,tuple([int(v) for v in check['overlaps']])))
            #
        #
        # add all cells cntd in cell for triangle 1 (not including cells that are cntd for any pt inside cell)
        region.cntd_cells = tuple(sorted(set(
            [(lvl,(row,col)) for (lvl,(row,col)) in cells_cntd_in_all_disks] +
            [(lvl,(row,col)) for (lvl,(row,col)) in cells_cntd_in_all_trgl_disks] + 
            [(lvl,(row,col)) for (lvl,(row,col)) in region.cntd_cells]
        )))

        # add all cells that are always at least ovlpd and not cntd in this one
        region.ovlpd_cells = tuple(sorted(set(
            [(lvl,(row,col)) for (lvl,(row,col)) in cells_always_ovlpd if not ( (lvl,(row,col)) in region.cntd_cells )] +
            [(lvl,(row,col)) for (lvl,(row,col)) in region.ovlpd_cells if not ( (lvl,(row,col)) in region.cntd_cells )]
        )))

        region.nested_cntd_cells = tuple(list(region.cntd_cells))

        # Collect raw subcell classifications; cross-region merge happens after all regions
        if nest_depth > 0:
            raw_c, raw_o, raw_d = [], [], []
            # Shared across all ovlpd cells in this region: adjacent boundary cells share
            # edge vertices, so reusing the cache avoids recomputing the same point twice.
            _vtx_cntd: dict = {}
            _vtx_ovlpd: dict = {}
            _classify = (classify_subcell_quadrants_vec if _config.USE_VEC_VERTEX_CHECKS
                         else classify_subcell_quadrants)
            for _lvl, (_row, _col) in region.ovlpd_cells:
                sc_c, sc_o, sc_d = _classify(
                    r=r,
                    grid_spacing=grid_spacing,
                    vtx_ovlpd=_vtx_ovlpd,
                    vtx_cntd=_vtx_cntd,
                    region=region,
                    q_anchor_row=float(_row)-0.5,
                    q_anchor_col=float(_col)-0.5,
                    lvl=1,
                    nest_depth=nest_depth,
                    include_boundary=include_boundary,
                )
                raw_c.extend(sc_c)
                raw_o.extend(sc_o)
                raw_d.extend(sc_d)
            region._raw_nested = (set(raw_c), set(raw_o), set(raw_d))
        else:
            region._raw_nested = None
            region.nested_ovlpd_cells = tuple(list(region.ovlpd_cells))

        if 0.5-1e-15 <= r%1 <= 0.5+1e-15:
            # Circle boundary passes exactly through vertical grid edge midpoints.
            # The vertex check (vtx.y==0) is unreliable here — force False.
            region.shared_along_vert = False
        else:
            region.shared_along_vert = any([edge.vtx1.y==0 and edge.vtx2.y==0  for edge in region.edges])
        region.shared_along_diag = any([edge.vtx1.x==edge.vtx1.y and edge.vtx2.x==edge.vtx2.y for edge in region.edges])
        #
    #

    # Cross-region consistent merge: produce final nested cells for all regions
    if nest_depth > 0:
        regions_list = list(trgl_regions.values())
        if True:
            raw_data = [reg._raw_nested for reg in regions_list]
            merged = merge_nested_cells_consistent(raw_data, nest_depth)
            for reg, (c_set, o_set, _d_set) in zip(regions_list, merged):
                reg.nested_cntd_cells = tuple(sorted(set(
                    list(reg.cntd_cells) + list(c_set)
                )))
                reg.nested_ovlpd_cells = tuple(sorted(o_set))
                # Per-region pass: merge any group of 4 sibling subcells that all ended up
                # cntd within this region — cross-region merge misses these when other
                # regions classify the same siblings differently. Safe because:
                # - total==4 only when no siblings were discarded (discarded cells are absent)
                # - merging to parent level-0 is equivalent to summing all 4 subcell entries
                merged_c, merged_o = merge_nested_cells_bottom_up(
                    list(reg.nested_cntd_cells), list(reg.nested_ovlpd_cells)
                )
                reg.nested_cntd_cells = tuple(sorted(merged_c))
                reg.nested_ovlpd_cells = tuple(sorted(merged_o))
        else:
            for reg in regions_list:
                c_set, o_set, _d_set = reg._raw_nested
                reg.nested_cntd_cells = tuple(sorted(set(
                    list(reg.cntd_cells) + list(c_set)
                )))
                reg.nested_ovlpd_cells = tuple(sorted(o_set))
                merged_c, merged_o = merge_nested_cells_bottom_up(
                    list(reg.nested_cntd_cells), list(reg.nested_ovlpd_cells)
                )
                reg.nested_cntd_cells = tuple(sorted(merged_c))
                reg.nested_ovlpd_cells = tuple(sorted(merged_o))

    if not plot_offset_regions is None:
        for n, region in enumerate(list(trgl_regions.values())):
            region.plot_many(regions=list(trgl_regions.values()), ax=axs.flat[n*2], alpha=0.1, add_idxs=False, title=str(n)+"/"+str(len(trgl_regions)))
            region.plot_single(ax=axs.flat[n*2], alpha=0.4, add_idx_edges=False)
            region.plot_single(ax=axs.flat[n*2+1], alpha=1, plot_edges=False, add_idx_edges=False)
            plot_cell_pattern(
                cntd_cells=[],#list(set([(lvl, (row, col)) for lvl, (row, col) in region.cntd_cells] + [(lvl,(row,col)) for lvl,(row,col) in cells_cntd_in_all_disks])),
                ovlpd_cells=[],#[(lvl, (row, col)) for lvl, (row, col) in region.ovlpd_cells],
                nested_cntd_cells=region.nested_cntd_cells,
                nested_ovlpd_cells=region.nested_ovlpd_cells,
                region_coords=[edge_coords[0] for edge_coords in region.coords],
                all_cells=[],#[(0,(int(row),int(col))) for (row,col) in all_cells],
                ax=axs.flat[n*2 + 1],
                r=r,
                grid_spacing=grid_spacing,
            )
            axs.flat[n*2].set_xlim([-0.02,0.52])
            axs.flat[n*2].set_ylim([-0.02,0.52])
            axs.flat[n*2].set_aspect('equal')
        if _por.get('filename'):
            fig.savefig(_por['filename'], dpi=_por.get('dpi', 300), bbox_inches='tight')
        if not _por.get('show', True):
            _plt_close(fig)
    # print("n_regs, n_edges", n_regs, n_edges)


def expand_regions_to_all_sectors(trgl_regions, r=0, nest_depth=0):
    """
    Rotate/reflect triangle-1 regions into all 8 symmetry sectors and deduplicate.

    For each triangle-1 region, generates 8 transformed copies (one per sector 1-8) via
    region.transform_to_trgl(i). Regions whose geometry is symmetric along the vertical axis
    (shared_along_vert) or the diagonal (shared_along_diag) are merged into fewer, shared
    region objects to avoid redundant lookup tables.
    All resulting regions are assigned a composite integer id encoding their unique
    (cntd_cells, ovlpd_cells) pair. Regions that collide on this id (same base-level
    cell pattern but different geometry) have their nested sub-cell lists merged conservatively:
    intersection for cntd cells, union + demoted cells for ovlpd cells.
    A second authoritative cross-region merge pass (merge_nested_cells_consistent) is applied
    at the end when nest_depth > 0.
    Returns id_to_offset_regions, translate_reg_nr_to_reg_id, and contain_region_mult.
    """
    regions = list(trgl_regions.values())
    unique_cntd_cells = dict()
    unique_ovlpd_cells = dict()
    translate_trgl_reg_nr_to_reg_nr = dict()
    # create new regions for rotation if not similar
    condensed_regions = []
    all_regions = []
    n_goal = 0
    for nr, region in enumerate(regions):
        n_goal += (
            1 if region.shared_along_vert and region.shared_along_diag else 
            4 if region.shared_along_vert or region.shared_along_diag else 8
        ) 
        region.nr = nr*10+1
        region_offsprings = []

        for i in [1,2,3,4,5,6,7,8]:
            # create new region!
            rotated_region = region.transform_to_trgl(i)
            rotated_region.nr = nr*10+i
            region_offsprings.append(rotated_region)
        
        # if region.nr == 1281: 
            
            
        #     unique_cntd_cells2 = dict()
        #     unique_ovlpd_cells2 = dict()
        #     for r in region_offsprings:
        #         # TODO adjust this to also work with subcell quadrants
        #         if not r.cntd_cells in unique_cntd_cells2:
        #             unique_cntd_cells2[r.cntd_cells] = len(unique_cntd_cells2)
        #         if not r.ovlpd_cells in unique_ovlpd_cells2:
        #             unique_ovlpd_cells2[r.ovlpd_cells] = len(unique_ovlpd_cells2)
        
        #     contain_region_mult = 10**(int(_math_log10(len(unique_ovlpd_cells2)))+1)
        #     id_to_offset_regions = dict()
        #     trgl_reg_nr_to_id = dict() 
        #     for r in region_offsprings:
        #         # TODO adjust this to also work with subcell quadrants
        #         r.id = unique_cntd_cells2[r.cntd_cells] * contain_region_mult + unique_ovlpd_cells2[r.ovlpd_cells]
                
        #         if not r.id in id_to_offset_regions:
        #             id_to_offset_regions[r.id] = r
                
        #         for i in range(0, 8 + 1 - r.trgl_nr):
        #             trgl_reg_nr_to_id[r.nr + i] = r.id
        #         r.contain_id = unique_cntd_cells2[r.cntd_cells]
        #         r.overlap_id = unique_ovlpd_cells2[r.ovlpd_cells]
        #     ax=OffsetRegion.plot_many(
        #         region_offsprings,x_lim=None,y_lim=None, alpha=0.5, add_centroids=False,
        #         add_idxs={'text':lambda r: str(r.nr)+"<"+str(r.id)+">"+str(r.contain_id)+"."+str(r.overlap_id)}, figsize=(12,12)
        #         )
        #
        all_regions.extend(region_offsprings)
        
        if region.shared_along_vert and region.shared_along_diag:
            condensed_regions.append(OffsetRegion.merge_regions(region_offsprings))
            translate_trgl_reg_nr_to_reg_nr.update({nr*10+j: nr*10+1 for j in [1,2,3,4,5,6,7,8]})
            pass
        elif region.shared_along_vert:
            condensed_regions.append(OffsetRegion.merge_regions([region_offsprings[-1], region_offsprings[0]]))
            condensed_regions.append(OffsetRegion.merge_regions(region_offsprings[1:3]))
            condensed_regions.append(OffsetRegion.merge_regions(region_offsprings[3:5]))
            condensed_regions.append(OffsetRegion.merge_regions(region_offsprings[5:7]))
            translate_trgl_reg_nr_to_reg_nr.update({
                nr*10+8: nr*10+1, nr*10+1: nr*10+1,
                nr*10+2: nr*10+3, nr*10+3: nr*10+3,
                nr*10+4: nr*10+5, nr*10+5: nr*10+5,
                nr*10+6: nr*10+7, nr*10+7: nr*10+7,
                })
            pass
        elif region.shared_along_diag:
            condensed_regions.append(OffsetRegion.merge_regions(region_offsprings[0:2]))
            condensed_regions.append(OffsetRegion.merge_regions(region_offsprings[2:4]))
            condensed_regions.append(OffsetRegion.merge_regions(region_offsprings[4:6]))
            condensed_regions.append(OffsetRegion.merge_regions(region_offsprings[6:8]))
            translate_trgl_reg_nr_to_reg_nr.update({
                nr*10+1: nr*10+1, nr*10+2: nr*10+1,
                nr*10+3: nr*10+3, nr*10+4: nr*10+3,
                nr*10+5: nr*10+5, nr*10+6: nr*10+5,
                nr*10+7: nr*10+7, nr*10+8: nr*10+7,
                })
        else:
            condensed_regions.extend(region_offsprings)
            translate_trgl_reg_nr_to_reg_nr.update({nr*10+j: nr*10+j for j in [1,2,3,4,5,6,7,8]})
    
    all_regions = condensed_regions
    trgl_reg_nr_to_id = dict() 
    all_regions.sort(key=lambda reg: (reg.trgl_nr))
    
    for region in all_regions:
        # TODO adjust this to also work with subcell quadrants
        if not region.cntd_cells in unique_cntd_cells:
            unique_cntd_cells[region.cntd_cells] = len(unique_cntd_cells)
        if not region.ovlpd_cells in unique_ovlpd_cells:
            unique_ovlpd_cells[region.ovlpd_cells] = len(unique_ovlpd_cells)
    contain_region_mult = 10**(int(_math_log10(len(unique_ovlpd_cells)))+1)
    id_to_offset_regions = dict()
    
    for region in all_regions:

        # TODO adjust this to also work with subcell quadrants
        region.id = unique_cntd_cells[region.cntd_cells] * contain_region_mult + unique_ovlpd_cells[region.ovlpd_cells]
        
        if not region.id in id_to_offset_regions:
            id_to_offset_regions[region.id] = region
        else:
            sibling = id_to_offset_regions[region.id]
            # Two geometrically distinct condensed regions share the same base-level
            # (cntd_cells, ovlpd_cells) pattern → same ID. Both map to this
            # ID, so points in either geometry use the sibling's nested cell data.
            # Merge the nested cells conservatively: intersection for cntd,
            # union + demoted cells for ovlpd.
            if nest_depth > 0:
                nc_sib = set(sibling.nested_cntd_cells)
                no_sib = set(sibling.nested_ovlpd_cells)
                nc_reg = set(region.nested_cntd_cells)
                no_reg = set(region.nested_ovlpd_cells)
                demoted = nc_sib.difference(nc_reg)
                nc_merged = nc_sib.intersection(nc_reg)
                no_merged = no_sib.union(no_reg).union(demoted)
                # Remove any parent from nested_ovlpd when its children are
                # already represented (cntd or ovlpd).
                changed = True
                while changed:
                    changed = False
                    to_remove = set()
                    to_add = set()
                    for (lvl, (row, col)) in no_merged:
                        half = 2 ** -(lvl + 2)
                        children = [
                            (lvl + 1, (row + dr * half, col + dc * half))
                            for dr, dc in [(-1, -1), (-1, +1), (+1, -1), (+1, +1)]
                        ]
                        cntd_ch = [c for c in children if c in nc_merged]
                        ovlpd_ch = [c for c in children if c in no_merged]
                        if cntd_ch or ovlpd_ch:
                            to_remove.add((lvl, (row, col)))
                            for c in children:
                                if c not in nc_merged and c not in no_merged:
                                    to_add.add(c)
                            changed = True
                    no_merged -= to_remove
                    no_merged |= to_add
                sibling.nested_cntd_cells = tuple(sorted(nc_merged))
                sibling.nested_ovlpd_cells = tuple(sorted(no_merged))

            # Propagate per-triangle nested cells from the colliding region to the sibling.
            if hasattr(region, 'nested_cells_by_trgl'):
                if not hasattr(sibling, 'nested_cells_by_trgl'):
                    sibling.nested_cells_by_trgl = {}
                sibling.nested_cells_by_trgl.update(region.nested_cells_by_trgl)

        for i in range(0, 8 + 1 - region.trgl_nr):
            trgl_reg_nr_to_id[region.nr + i] = region.id
        
    translate_reg_nr_to_reg_id = {k: trgl_reg_nr_to_id[v] for k,v in translate_trgl_reg_nr_to_reg_nr.items()}
    # region = id_to_offset_regions[110128]
    # print("region",region.nr,region.shared_along_vert,region.shared_along_diag)
    # region_offsprings = [id_to_offset_regions[translate_reg_nr_to_reg_id[region.nr//10*10+i]] for i in [1,2,3,4,5,6,7,8]]
    # print("region_offsprings", [r.id for r in region_offsprings])
    # print("s a v o d", (region.shared_along_vert, region.shared_along_diag))
    # hex_codes = []
    # for hex_i in '0123456789abcedf':
    #     for hex_j in '0123456789abcedf':
    #         hex_codes.append(hex_i+hex_j)
    # hex_codes_8 = ['#'+(hex_codes[int(len(hex_codes)*(i-.5)/8)])*3 for i in [1,2,3,4,5,6,7,8]]
    # hex_codes_8 = [hex_codes_8[i-1] for i in [1,5,2,6,3,7,4,8]]
    # ax=None
    # for i in [1,2,3,4,5,6,7,8]:
    #     subset_regions = [r for r in all_regions if r.nr%10==i]
    #     print("hex_codes_8[i-1]",hex_codes_8[i-1])
    #     ax=OffsetRegion.plot_many(
    #         subset_regions,x_lim=[-.51,.51],y_lim=[-.51,.51], alpha=0.5,
    #         add_idxs=False, facecolor=hex_codes_8[i-1], edgecolor='#ffffff',ax=ax, figsize=(20,20))
    # OffsetRegion.plot_many(region_offsprings,x_lim=[-.51,.51],y_lim=[-.51,.51],ax=ax, alpha=0.5, add_idxs={}, facecolor='None',edgecolor='red')
    # ax.plot([-1,1],[-1,1],color='red',linewidth=0.4)
    # ax.plot([-1,1],[1,-1],color='red',linewidth=0.4)
    # ax.vlines(x=0, ymin=-1, ymax=1,color='red',linewidth=0.4)
    # ax.hlines(y=0, xmin=-1, xmax=1,color='red',linewidth=0.4)


    # ax=OffsetRegion.plot_many(all_regions,x_lim=None,y_lim=None, facecolor='#ddd', edgecolor='#bbb',add_centroids=False, figsize=(20,20))
    # OffsetRegion.plot_many(region_offsprings,x_lim=None,y_lim=None,ax=ax, alpha=0.5, add_idxs={})
    
    # ax.plot([-1,1],[-1,1],color='red',linewidth=0.4)
    # ax.plot([-1,1],[1,-1],color='red',linewidth=0.4)
    # ax.vlines(x=0, ymin=-1, ymax=1,color='red',linewidth=0.4)
    # ax.hlines(y=0, xmin=-1, xmax=1,color='red',linewidth=0.4)


    # region_offspring_problems = [r for r in region_offsprings if r.nr%10 in [2,4,6,8]]
    
    # ax=OffsetRegion.plot_many(all_regions,x_lim=None,y_lim=None, facecolor='#ddd', edgecolor='#bbb',add_centroids=False, figsize=(20,20))
    # print("len(region_offspring_problems)",len(region_offspring_problems), region.nr, region.nr%10, [r.nr for r in region_offsprings])
    # OffsetRegion.plot_many(region_offspring_problems,x_lim=None,y_lim=None,ax=ax, alpha=0.5, add_idxs={})
    # ax.plot([-1,1],[-1,1],color='red',linewidth=0.4)
    # ax.plot([-1,1],[1,-1],color='red',linewidth=0.4)
    # ax.vlines(x=0, ymin=-1, ymax=1,color='red',linewidth=0.4)
    # ax.hlines(y=0, xmin=-1, xmax=1,color='red',linewidth=0.4)
    # print("len(condensed_regions),",len(condensed_regions),"len(id_to_offset_regions),",len(id_to_offset_regions), )

    # Second (authoritative) cross-region merge pass on the final condensed regions.
    # The triangle-1 pass in finalize_region_cells is a local pre-pass only;
    # merge_regions can invalidate that consistency, so we redo it here on all final regions.
    if nest_depth > 0:
        def _build_d_set(c_set, o_set, ovlpd_cells_lvl0, nest_depth):
            """
            Reconstruct discarded sub-cells: enumerate the full sub-cell tree
            for each level-0 ovlpd cell, then subtract c_set and o_set.
            This prevents the expansion phase of merge_nested_cells_consistent
            from promoting discarded (outside-disk) cells into 'ovlpd'.
            """
            d_set = set()
            frontier = list(ovlpd_cells_lvl0)
            for _ in range(nest_depth):
                next_frontier = []
                for pkey in frontier:
                    children = get_children(pkey)
                    for ck in children:
                        if ck not in c_set and ck not in o_set:
                            d_set.add(ck)
                        next_frontier.append(ck)
                frontier = next_frontier
            return d_set

        raw_data = []
        for r in all_regions:
            c_set = set(r.nested_cntd_cells)
            o_set = set(r.nested_ovlpd_cells)
            lvl0_ovlpd = [(lvl, rc) for (lvl, rc) in r.ovlpd_cells if lvl == 0]
            d_set = _build_d_set(c_set, o_set, lvl0_ovlpd, nest_depth)
            raw_data.append((c_set, o_set, d_set))
        merged = merge_nested_cells_consistent(raw_data, nest_depth)
        for reg, (c_set, o_set, _) in zip(all_regions, merged):
            reg.nested_cntd_cells = tuple(sorted(c_set))
            reg.nested_ovlpd_cells = tuple(sorted(o_set))
            merged_c, merged_o = merge_nested_cells_bottom_up(
                list(reg.nested_cntd_cells), list(reg.nested_ovlpd_cells)
            )
            reg.nested_cntd_cells = tuple(sorted(merged_c))
            reg.nested_ovlpd_cells = tuple(sorted(merged_o))

    return id_to_offset_regions, translate_reg_nr_to_reg_id, contain_region_mult
    
#

def extract_shared_cells(
        id_to_offset_regions:dict,
):
    """
    Modifies regions by removing those nested_subcells that are shared by all regions
    return shared_cntd_cells and shared_ovlpd_cells
    """
    regions = list(id_to_offset_regions.values())
    reg_0 = regions[0]

    shared_cntd_cells = set(reg_0.nested_cntd_cells)
    # print("shared_cntd_cells",len(shared_cntd_cells))
    i = 1
    while i  < len(id_to_offset_regions) and len(shared_cntd_cells)>0:
        shared_cntd_cells.intersection_update(set(regions[i].nested_cntd_cells))
        i += 1
    #

    n_cntd_ovlpd_cells = []
    for region in id_to_offset_regions.values():
        region.distinct_cntd_cells = [
            nested_cell for nested_cell in region.nested_cntd_cells 
            if not nested_cell in shared_cntd_cells
        ]
        n_cntd_ovlpd_cells.append(len(set(region.distinct_cntd_cells)))
            
        #
    #

    for region in id_to_offset_regions.values():
        region.distinct_ovlpd_cells = list(region.nested_ovlpd_cells)

    return shared_cntd_cells

# TODO ensure that row,col x,y is not mixed up

def assign_pts_to_offset_region(
        pts,
        potential_regions,
):
    [[pt for reg in potential_regions] for pt in pts]
    pass


def create_raster_plot(
        regions:list,
        raster_cell_to_regions:dict,
        offset_x_bins,
        offset_y_bins,
        lims=[-0.05, 0.55],
        add_raster_labels:bool=True

        ):
        """
        Diagnostic plot showing how many offset regions overlap each raster cell.

        Renders three side-by-side panels: (0) region outlines only, (1) raster cells
        coloured by the number of candidate regions they contain (viridis scale) with an
        integer count label per cell, (2) raster cells coloured by their unique region-id
        combination (tab20 scale) with a combination-index label. Used for visual inspection
        of raster resolution — ideally most cells should contain exactly one region.
        """
        fig, axs = plt.subplots(1,3,figsize=(50,30))
        region_comb_dict = dict()
        for l, region_comb in sorted([(len(v), tuple([r.id for r in v])) for v in raster_cell_to_regions.values()]):
            if not region_comb in region_comb_dict:
                region_comb_dict[region_comb] = len(region_comb_dict)
        lens = [len(v) for v in raster_cell_to_regions.values()]
        region_comb_nrs = [region_comb_dict[tuple([r.id for r in v])] for v in raster_cell_to_regions.values()]
        my_cmap=_plt_get_cmap('viridis')
        my_cmap2=_plt_get_cmap('tab20')
        

        for (ix,iy),v in raster_cell_to_regions.items():
            poly_coords = [(_np_sign(ix)*x, _np_sign(iy)*y) for x,y in [
                (offset_x_bins[abs(ix)-1][0], offset_y_bins[abs(iy)-1][0]), (offset_x_bins[abs(ix)-1][1], offset_y_bins[abs(iy)-1][0]),
                (offset_x_bins[abs(ix)-1][1], offset_y_bins[abs(iy)-1][1]), (offset_x_bins[abs(ix)-1][0], offset_y_bins[abs(iy)-1][1])
                ]]
            
            l_relative = len(v)/max(lens)
            region_comb_nr = region_comb_dict[tuple([r.id for r in v])]
            region_comb_nr_relative = 0.05*region_comb_nrs.index(region_comb_nr) %1

            axs.flat[0].add_patch(_plt_Polygon(poly_coords,facecolor=my_cmap(l_relative), edgecolor='#000', alpha=0.7, linewidth=0.15))
            axs.flat[1].add_patch(_plt_Polygon(poly_coords,facecolor=my_cmap(l_relative), edgecolor='#000', alpha=0.7, linewidth=0.15))
            axs.flat[2].add_patch(_plt_Polygon(poly_coords,facecolor=my_cmap2(region_comb_nr_relative), edgecolor='#000', alpha=0.7, linewidth=0.15))
        
        
        if add_raster_labels:
            for (ix,iy), v in raster_cell_to_regions.items():
                (x_low, x_up), (y_low, y_up) = [_np_sign(ix)*x for x in offset_x_bins[abs(ix)-1]], [_np_sign(iy)*y for y in offset_y_bins[abs(iy)-1]]
                # axs.flat[1].annotate(text=".".join([str(v0) for v0 in v]), xy=((x_low+x_up)/2, (y_low+y_up)/2), horizontalalignment='center', fontsize=5)
                # axs.flat[2].annotate(text=".".join([str(v0) for v0 in v]), xy=((x_low+x_up)/2, (y_low+y_up)/2), horizontalalignment='center', fontsize=5)
                axs.flat[1].annotate(text=str(len(v)), xy=((x_low+x_up)/2, (y_low+y_up)/2), horizontalalignment='center', fontsize=8)
                axs.flat[2].annotate(text=str(region_comb_dict[tuple([r.id for r in v])]), xy=((x_low+x_up)/2, (y_low+y_up)/2), horizontalalignment='center', fontsize=8)
        else:
            for (ix,iy), v in raster_cell_to_regions.items():
                (x_low, x_up), (y_low, y_up) = [_np_sign(ix)*x for x in offset_x_bins[abs(ix)-1]], [_np_sign(iy)*y for y in offset_y_bins[abs(iy)-1]]
                text = str(len(v))#str(ix)#+"."+str(iy)
                # text = "+"
                axs.flat[1].annotate(text=text, xy=((x_low+x_up)/2, (y_low+y_up)/2), horizontalalignment='center', fontsize=7)
        
        OffsetRegion.plot_many(regions=regions, plot_edges=False, edgecolor='black', ax=axs.flat[0], facecolor='None', alpha=0.8, linewidth=0.4)
        OffsetRegion.plot_many(regions=regions, plot_edges=False, edgecolor='black', ax=axs.flat[1], facecolor='None', alpha=0.8, linewidth=0.4, add_idxs=False)
        OffsetRegion.plot_many(regions=regions, plot_edges=False, edgecolor='black', ax=axs.flat[2], facecolor='None', alpha=0.8, linewidth=0.4, add_idxs=False)
        for ax in axs.flat:
            ax.set_xlim(lims)
            ax.set_ylim(lims)
        axs.flat[-1].legend((i for i in sorted(set(lens))),loc='upper right')# To-Do legend not added


def build_region_raster(
    trgl_regions:dict,
    r:float,
    plot_offset_raster:dict=None,
):
    """
    Assign each region in triangle 1 to one or more raster cells covering [0, 0.5]².

    Builds a uniform grid of axis-aligned raster cells whose breakpoints are derived from
    the x/y vertex coordinates of all triangle-1 and triangle-2 regions, ensuring that no
    region boundary cuts through the interior of a raster cell. For each raster cell the
    function collects all regions whose bounding box overlaps it.
    The result is used downstream to narrow the region-lookup to only the 1-3 candidate
    regions that can contain a given point offset, instead of checking all regions.
    Returns raster_cell_to_regions, offset_x_bins, offset_y_bins, and
    unique_reg_id_combs_to_raster_cells.
    """
    # assign each cell to a region
    # start with triangle only
    regions_to_check = [reg for reg in trgl_regions.values() if reg.trgl_nr in [1,2]]
    regions_to_check.sort(key=lambda reg: (reg.xmin, reg.xmax, reg.ymin, reg.ymax))
    x_vals, y_vals = [], []
    for region in regions_to_check:
        if region.xmin >= region.xmax:
            print("xmin>=xmax", region.coords)
        for vtx in region.vertices:
            # if vtx.x >= 0:
            x_vals.append(vtx.x)
            # if vtx.y >= 0:
            y_vals.append(vtx.y)
        x_vals.append(region.xmin)
        x_vals.append(region.xmax)
        y_vals.append(region.ymin)
        y_vals.append(region.ymax)
    
    x_vals = [x for x in x_vals if 0<=x<=0.5]
    y_vals = [y for y in y_vals if 0<=y<=0.5]

    if False:
        unique_x_vals = sorted(set([vtx.x for vtx in vertices if vtx.x >= 0]))
        unique_y_vals = sorted(set([vtx.y for vtx in vertices if vtx.y >= 0]))
        make_bins_from_vals, get_vals_from_bins
        offset_x_bins = make_bins_from_vals(unique_x_vals) 
        offset_y_bins = make_bins_from_vals(unique_y_vals) 
    else:
        offset_x_bins = make_bins_from_vals(x_vals+y_vals)
        offset_y_bins = make_bins_from_vals(x_vals+y_vals)
    # print("bins", [float(s) for s,e in offset_y_bins]+[offset_y_bins[-1][1]])
    # remember regions are not always convex - but maybe i guess they will be once splitted along lines
    raster_cell_to_regions = dict()
    unique_reg_id_combs_to_raster_cells = dict()
    i, j = 0, 0
    regions_to_check_at_x = regions_to_check
    for ix, (x_low, x_up) in zip(range(1, len(offset_x_bins)+1), offset_x_bins):
        if True:
            # look for leftmost region that overlaps x_low  
            i = next((ix for ix, reg in enumerate(regions_to_check_at_x) if reg.xmax > x_low),-1)
            regions_to_check_at_x = regions_to_check_at_x[i:]
            
            if len(regions_to_check_at_x)==0:
                print("Break, no regions to check", x_low, x_up )
                break
        if False:
            if j != -1:
                j = next((jx for jx, reg in enumerate(regions_to_check_at_x) if reg.xmin >= x_up),-1)
                # j = next((jx for jx, reg in enumerate(regions_to_check_at_x) if reg.xmin > x_up),len(regions_to_check_at_x))
            else:
                print("-------------- j != -1 --------------",[((reg.xmin,reg.xmax),(reg.ymin,reg.ymax)) for reg in regions_to_check_at_x], x_low, x_up)

            regions_to_check_at_xy = regions_to_check_at_x[:j]
            # regions_to_check_at_xy.sort(key=lambda reg: (reg.ymax, reg.ymin))
        for iy, (y_low, y_up) in zip(range(1, len(offset_y_bins)+1), offset_y_bins):
            # if y_low == x_up:
            #     print() 
            if y_low >= x_up: 
                break
            
            regions_at_raster_cell = []
            region_ids_at_raster_cell = set()
            for reg in regions_to_check_at_x:
                if reg.xmin<x_up and x_low<reg.xmax and reg.ymin<y_up and y_low<reg.ymax:
                    region_ids_at_raster_cell.add(reg.id)
                    regions_at_raster_cell.append(reg)

            if False:
                # look for downmost region that overlaps y_low  
                n = len(regions_to_check_at_xy)
                regions_to_check_at_xy = [reg for reg in regions_to_check_at_xy if reg.ymax > y_low]
                
                # TODO leverage this performance gain later on. 
                # # extract all regions that within raster_cell
                # def check_if_raster_cell_overlaps_region(reg:OffsetRegion,x_low:float, y_low:float, x_up:float, y_up:float):
                #     raster_vertices = ((x_low, y_low), (x_up, y_low), (x_up, y_up), (x_low, y_up))
                #     for (x,y) in list(reg.get_coords()):
                #         for vx,vy in raster_vertices:
                #             if (vx)
                #     return False
                # region_nrs_at_raster_cell = set()
                regions_at_raster_cell2 = []
                region_ids_at_raster_cell2 = set()
           
                for reg in regions_to_check_at_xy:
                    if reg.ymin < y_up and not reg.id in region_ids_at_raster_cell:
                        # Check if any of raster cell corners / the centroid are within the region
                        any_vtx_is_inside = False#
                        # while not any_vtx_is_inside:
                            
                        for vtx in [(x_low, y_low), (x_up, y_low), (x_up, y_up), (x_low, y_up), ((x_low+x_up)/2,(y_low+y_up)/2)]:
                            vtx_x, vtx_y = vtx
                            vtx_is_inside = False
                            for edge in reg.edges:
                                    
                                check_is_true = (not hasattr(edge, 'contains') or (0,edge.contains) in reg.cntd_cells
                                ) and (
                                    not hasattr(edge, 'overlaps') or (0,edge.overlaps) in reg.ovlpd_cells
                                )
                                # if not check_is_true:
                                #     print("not check_is_true")
                                # else:
                                #     print("check_is_true")

                                if edge.type != 'LineSegment':
                                    check_res = ((vtx_x -edge.center[0])**2+(vtx_y -edge.center[1])**2)**.5 < edge.r
                                else:
                                    col_index = int(edge.vtx1.y == edge.vtx2.y)
                                    check_res = abs((vtx_x, vtx_y)[col_index] - edge.vtx1.xy[col_index]) < r
                                vtx_is_inside = check_res if check_is_true else not check_res
                                    
                                if not vtx_is_inside:
                                    break
                            if vtx_is_inside:
                                any_vtx_is_inside = True
                                break
                    
                    # region_nrs_at_raster_cell.add(reg.nr)
                    region_ids_at_raster_cell2.add(reg.id)
                    regions_at_raster_cell2.append(reg)
            
            # if len(regions_at_raster_cell) == 0:
            #     print("x_low,x_up",x_low,x_up,"y_low,y_up",y_low,y_up)
            #     print("regions_to_check_at_xy",regions_to_check_at_xy)
            #     print("regions_to_check_at_x",[((r.xmin,r.xmax),(r.ymin,r.ymax)) for r in regions_to_check_at_x])
            #     print([((r.xmin,r.xmax),(r.ymin,r.ymax)) for r in regions_to_check if r.xmin<x_up and x_low<r.xmax and r.ymin<y_up and y_low<r.ymax])
            # if region_ids_at_raster_cell2!=region_ids_at_raster_cell:
            #     pass
                # print("regions_at_raster_cell",[((r.xmin,r.xmax),(r.ymin,r.ymax)) for r in regions_at_raster_cell])
                # print("regions_at_raster_cell2",[((r.xmin,r.xmax),(r.ymin,r.ymax)) for r in regions_at_raster_cell2])
                # print("lens",len(regions_at_raster_cell), len(regions_at_raster_cell2))
            # region_nrs_at_raster_cell = tuple(sorted(region_nrs_at_raster_cell))
            # if len(region_nrs_at_raster_cell)==0:
            #     print("(x_low, x_up)", (x_low, x_up), "(y_low, y_up)", (y_low, y_up), 'N', n, 'n', len(regions_to_check_at_xy), 'Nx', len(regions_to_check_at_x))
            
            region_ids_at_raster_cell = tuple(sorted(region_ids_at_raster_cell))
            
            if not region_ids_at_raster_cell in unique_reg_id_combs_to_raster_cells:
                unique_reg_id_combs_to_raster_cells[region_ids_at_raster_cell] = []
            
            unique_reg_id_combs_to_raster_cells[region_ids_at_raster_cell].append((ix,iy))
            raster_cell_to_regions[(ix, iy)] = regions_at_raster_cell
        #
    #
    if not plot_offset_raster is None:
        create_raster_plot(regions=list(trgl_regions.values()), raster_cell_to_regions=raster_cell_to_regions, offset_x_bins=offset_x_bins, offset_y_bins=offset_y_bins)
    
    return raster_cell_to_regions, offset_x_bins, offset_y_bins, unique_reg_id_combs_to_raster_cells
#

def make_arc_check(edge, r:float, include_boundary:bool):
    """
    Returns a callable that checks whether pts are within radius r of the arc center.
    Returns a boolean array of length n.
    """
    (x, y) = edge.center
    if include_boundary:
        return _ArcCheckBoundary(x, y, r)
    return _ArcCheckStrict(x, y, r)

def make_line_check(edge, r:float, include_boundary:bool):
    """
    Returns a callable that checks whether pts are on the >= side of a horizontal
    or vertical line. Returns a boolean array of length n.
    """
    col_index = int(edge.vtx1.y == edge.vtx2.y)
    val = edge.vtx1.xy[col_index]
    if include_boundary:
        return _LineCheckBoundary(col_index, val)
    return _LineCheckStrict(col_index, val)

#

def make_edge_check(edge, r:float, include_boundary:bool=False):
    """
    returns lambda function
    - that returns a boolean array (of length n where n is number of points) whether pts are within distance <= r. 
    """
    if edge.type == 'Arc':
        return make_arc_check(edge, r, include_boundary)
    return make_line_check(edge, r, include_boundary)
#

_coord_match_count = {'found': 0, 'not_found': 0}

def edge_is_shared_with_region_id(
        reversed_edge_coords,
        remaining_regions,
):
    """
    Return the id of the first region in remaining_regions that has an edge whose coordinate
    tuple matches reversed_edge_coords (i.e. the same edge traversed in the opposite direction).
    Returns None if no match is found. Used to identify shared boundaries between adjacent regions.
    """
    for reg in remaining_regions:
        for e in reg.edges:
            if reversed_edge_coords == e.coords:
                _coord_match_count['found'] += 1
                return reg.id
    _coord_match_count['not_found'] += 1
    return None
#

def edge_is_shared_with_region_id2(
        edge,
        remaining_regions,
):
    """
    First checks whether the edge is a 'contains' or 'overlaps' check
    Then loops through remaining regions looking through the region edges for an edge that 
    has the same 'contains' or 'overaps' attribute and then checks whether this attribute is 
    the same as from the edge. If its the same return the region id where the edge was in  
    """
    attr_to_check = 'contains' if hasattr(edge, 'contains') else 'overlaps'
    if not hasattr(edge, attr_to_check):
        return None

    for reg in remaining_regions:
        for e in reg.edges:
            if hasattr(e, attr_to_check) and getattr(edge, attr_to_check) == getattr(e, attr_to_check):
                return reg.id
    return None
#

def edge_is_shared_with_region_id3(
        edge,
        remaining_regions,
):
    """
    First checks whether the edge is a 'contains' or 'overlaps' check
    Then loops through remaining regions looking through the region edges for an edge that 
    has the same 'contains' or 'overaps' attribute and then checks whether this attribute is 
    the same as from the edge. If its the same return the region id where the edge was in  
    """
    attr_to_check = 'contains' if hasattr(edge, 'contains') else 'overlaps'
    if not hasattr(edge, attr_to_check):
        return []

    reg_ids = []
    for reg in remaining_regions:
        for e in reg.edges:
            if hasattr(e, attr_to_check) and getattr(edge, attr_to_check) == getattr(e, attr_to_check):
                reg_ids.append(reg.id)
    return list(set(reg_ids))
#

def vertex_is_shared_with_region(
        remaining_regions,
):
    """
    to-do this only returns the first edge that shares a vertex and the region that contains in. THink about what happens if there are more than 2 regions.
    """
    
    # finds a vertex that a pair of edges among two (?Does this work for three and more?!) regions share 
    edges_current_region = remaining_regions[0].edges
    for edge in edges_current_region:
        for reg in remaining_regions:
            for e in reg.edges:
                if any([coord in edge.coords for coord in e.coords]):
                    return (edge, reg.id)
                
        return None
#
# create (potentially recursive checks:)
def build_decision_tree(
        tree_pos,
        checks,
        remaining_regions,
        r:float=1.0,
        include_boundary:bool=False,
        _rebuilt:bool=False,
):
    """
    Recursively build a binary decision tree that assigns each point to one of the remaining regions.

    At each node, selects the check (arc or line boundary) that cleanly separates the most regions,
    preferring checks where no region straddles the boundary (determined analytically from vertex
    distances, with centroid fallback). The chosen check is stored at tree_pos['check']; the True
    and False branches recurse until only one region remains, at which point its id is stored as a
    leaf. If all pre-built checks are exhausted before a unique assignment is reached, extra checks
    are synthesised from all edges of the remaining regions (_rebuilt=True pass).
    The resulting tree is later evaluated per point via eval_region_tree.
    """
    if len(remaining_regions)<=1:
        raise ValueError("remaining_regions", len(remaining_regions), len(checks))
    if len(checks)==0:
        if _rebuilt:
            _fid = remaining_regions[0].id
            tree_pos['check'] = lambda pts: _np_zeros(len(pts), dtype=bool) == 0
            tree_pos[True] = _fid
            tree_pos[False] = _fid
            return
        # All pre-built checks have been consumed but multiple regions remain.
        # Rebuild checks from all edges of the remaining regions.
        seen_coords = set()
        extra_checks = []
        for reg in remaining_regions:
            for edge in reg.edges:
                if edge.coords not in seen_coords:
                    seen_coords.add(edge.coords)
                    extra_checks.append((edge, make_edge_check(edge, r=r, include_boundary=include_boundary)))
        if extra_checks:
            build_decision_tree(tree_pos, extra_checks, remaining_regions, r=r, include_boundary=include_boundary, _rebuilt=True)
        else:
            _fid = remaining_regions[0].id
            tree_pos['check'] = lambda pts: _np_zeros(len(pts), dtype=bool) == 0
            tree_pos[True] = _fid
            tree_pos[False] = _fid
        return
    # TODO check if this also works with subcell quadrants

    def _classify_region_for_edge(reg, edge):
        """Analytically classify a region against an arc or line edge.

        For Arc edges: uses vertex distances vs arc radius with epsilon tolerance.
          - True  = all vertex distances <= r + eps  (region is inside or on boundary)
          - False = all vertex distances >= r - eps  (region is outside or on boundary)
          - None  = some vertices inside, some outside (region genuinely straddles)

        For Line edges: uses centroid (line checks use strict distance < r from line,
        so vertices on the line have distance 0 which is always < r for positive r).
        """
        vtx_coords = _np_array([list(vtx.xy) for vtx in reg.vertices])
        if edge.type == 'Arc':
            ex, ey = edge.center
            # Compute arc radius from the edge's own vertex (which lies exactly on the arc)
            arc_r = _np_linalg_norm(_np_array([edge.vtx1.x - ex, edge.vtx1.y - ey]))
            # eps must absorb floating-point error in vertex positions (chained ops can
            # give errors ~1e-8 to 1e-6), but stay small enough that genuinely-straddling
            # junction vertices (which are clearly inside OR outside by ~0.1*r) are unaffected.
            eps = arc_r * 1e-4
            # Include centroid in the distance array for robustness
            cx, cy = reg.centroid
            all_coords = _np_array(list(vtx_coords) + [[cx, cy]])
            dists = _np_linalg_norm(all_coords - (ex, ey), axis=1)
            max_d, min_d = dists.max(), dists.min()
            if max_d <= arc_r + eps:
                return True   # entirely inside (or touching boundary)
            elif min_d >= arc_r - eps:
                return False  # entirely outside (or touching boundary)
            else:
                return None   # genuinely straddles the arc
        else:
            # Line check: centroid-based (line boundaries don't have the distance=r problem)
            cx, cy = reg.centroid
            col_index = int(edge.vtx1.y == edge.vtx2.y)
            val = edge.vtx1.xy[col_index]
            cx_val = [cx, cy][col_index]
            # Determine which side of the line the centroid is on
            # (we don't know r here, but centroid should be cleanly on one side)
            vtx_vals = vtx_coords[:, col_index]
            if all(vtx_vals <= val):
                return False
            elif all(vtx_vals >= val):
                return True
            return None

    # Find the first check that cleanly separates all regions (no region straddles it)
    # and actually splits the set (not all on same side)
    chosen_edge, chosen_check = checks[0]
    chosen_classifications = None
    for candidate_edge, candidate_check in checks:
        classifications = [_classify_region_for_edge(reg, candidate_edge) for reg in remaining_regions]
        if None not in classifications:
            true_count = sum(1 for c in classifications if c)
            false_count = sum(1 for c in classifications if not c)
            if true_count > 0 and false_count > 0:
                chosen_edge, chosen_check = candidate_edge, candidate_check
                chosen_classifications = classifications
                break

    if chosen_classifications is None:
        # No check cleanly separates all regions analytically.
        # Try centroid evaluation on each check until one actually splits the set.
        for candidate_edge, candidate_check in checks:
            cent_cls = [candidate_check(_np_array([list(reg.centroid)]))[0] for reg in remaining_regions]
            if any(cent_cls) and not all(cent_cls):
                chosen_edge, chosen_check = candidate_edge, candidate_check
                chosen_classifications = cent_cls
                break
        if chosen_classifications is None:
            _fid = remaining_regions[0].id
            tree_pos['check'] = lambda pts: _np_zeros(len(pts), dtype=bool) == 0
            tree_pos[True] = _fid
            tree_pos[False] = _fid
            return

    edge, check = chosen_edge, chosen_check
    tree_pos['check'] = check

    regions_if_true = [
        reg for reg, cls in zip(remaining_regions, chosen_classifications)
        if cls
    ]

    ids_if_true = [reg.id for reg in regions_if_true]
    regions_if_false = [reg for reg in remaining_regions if reg.id not in ids_if_true]
    # print("TREE_NODE reg_ids=",[r.id for r in remaining_regions], ...)
    if False and len(remaining_regions)>1 and len(checks)<2:
        # print("contains",hasattr(edge,'contains') and edge.contains)
        # print("overlaps",hasattr(edge,'overlaps') and edge.overlaps)
        # print("r contains", [r.cntd_cells for r in remaining_regions])
        # print("r overlaps", [r.ovlpd_cells for r in remaining_regions])
        # print("r contains", [None if not hasattr(edge, 'contains') else (0,edge.contains) in r.cntd_cells for r in remaining_regions])
        # print("r overlaps", [None if not hasattr(edge, 'overlaps') else (0,edge.overlaps) in r.ovlpd_cells for r in remaining_regions])
        fig, ax = plt.subplots(1,1,figsize=(6,6))
        for reg in regions_if_false:
            reg.plot_single(ax=ax, plot_edges=False, facecolor='#ccc', edgecolor="#f808ec", hatch="//")
        for reg in regions_if_true:
            reg.plot_single(ax=ax, plot_edges=False, facecolor='#ccc', edgecolor="#07e22b", hatch="o")
        OffsetRegion.plot_many(
            regions=remaining_regions, plot_vertices=False, #x_lim=(-.52,.52), y_lim=(-.52,.52),
            title=str(check_counter['count'])+'. T='+str(len(regions_if_true))+"regF="+str(len(regions_if_false))+'. '+str(edge.type),
            ax=ax)
        edge.plot_single(ax=ax, linewidth=6, full_circle='#ccc', edgecolor='black', linestyle="dotted", alpha=1)
        edge.plot_single(ax=ax, linewidth=3, edgecolor='black', alpha=1)
        # if edge.type == 'Arc':
        #     print("++++++",edge.angle_min,edge.angle_max,edge.vtx1, edge.vtx2, edge.get_plot_coords(arc_steps_per_degree=5))
        ax.set_xlim([min([r.xmin-0.01 for r in remaining_regions]),max([r.xmax+0.01 for r in remaining_regions])])
        ax.set_ylim([min([r.ymin-0.01 for r in remaining_regions]),max([r.ymax+0.01 for r in remaining_regions])])
        # print("x",[min([r.xmin-0.01 for r in remaining_regions]),max([r.xmax+0.01 for r in remaining_regions])])
        # print("y",[min([r.ymin-0.01 for r in remaining_regions]),max([r.ymax+0.01 for r in remaining_regions])])
    
    if len(regions_if_true) == 0:
        pass
    elif len(regions_if_true) == 1:
        id_if_true = regions_if_true[0].id
        tree_pos[True] = id_if_true
    else:
        if len(regions_if_false) > 0:
            tree_pos[True] = {}
            build_decision_tree(tree_pos[True], checks[1:], regions_if_true, r=r, include_boundary=include_boundary)
        else:
            build_decision_tree(tree_pos, checks[1:], regions_if_true, r=r, include_boundary=include_boundary)

    if len(regions_if_false) == 0:
        pass
    elif len(regions_if_false)==1:
        id_if_false = regions_if_false[0].id
        tree_pos[False] = id_if_false
    else:
        if len(regions_if_true) > 0:
            tree_pos[False] = {}
            build_decision_tree(tree_pos[False], checks[1:], regions_if_false, r=r, include_boundary=include_boundary)
        else:
            build_decision_tree(tree_pos, checks[1:], regions_if_false, r=r, include_boundary=include_boundary)

    if len(regions_if_true)+len(regions_if_false) == 0:
        raise ValueError('remaining_regions', remaining_regions, 'checks', checks)
#

def eval_region_tree(
        pts:_np_array,
        check_tree
    ):
    """
    Recursively evaluate a check tree to assign each point in pts to an offset region id.

    If check_tree is an integer leaf, all points receive that region id. Otherwise the tree's
    'check' lambda is applied to pts, splitting points into True/False subsets that are
    recursed into check_tree[True] and check_tree[False] respectively. Returns an integer
    array of region ids with the same length as pts.
    """
    if type(check_tree) != dict:
        return _np_zeros(len(pts), int) + check_tree
    res = _np_zeros(len(pts), int)
    check_res = check_tree['check'](pts)
    res[_np_arange(len(pts))[check_res]] = eval_region_tree(pts=pts[_np_arange(len(pts))[check_res]], check_tree=check_tree[True])
    res[_np_arange(len(pts))[_np_invert(check_res)]] = eval_region_tree(pts=pts[_np_arange(len(pts))[_np_invert(check_res)]], check_tree=check_tree[False])
    
    return res
#

def build_raster_checks(
        id_to_offset_regions:dict,
        unique_reg_id_combs_to_raster_cells:dict,
        r:float, 
        include_boundary:bool,
    ) -> dict:
    """
    Creates a dict that maps each offeset region combination to a check tree that resolves in which of the offset regions a point falls
    
    """
    offset_reg_id_comb_to_check = dict()
    # print("unique_reg_id_combs_to_raster_cells",unique_reg_id_combs_to_raster_cells)
    # print("len(unique_reg_id_combs_to_raster_cells)",len(unique_reg_id_combs_to_raster_cells))
    # print("rlll", sum([int(len(region_comb)==0) for region_comb in unique_reg_id_combs_to_raster_cells]), len(unique_reg_id_combs_to_raster_cells))
    for region_comb in unique_reg_id_combs_to_raster_cells:
        # print("region_comb",region_comb)
        regions = [id_to_offset_regions[id] for id in sorted(set(region_comb))]
        # print("regions",regions)

        if len(regions)==0:
            print("r",r)
            raise ValueError("PP", region_comb, unique_reg_id_combs_to_raster_cells)
        if len(region_comb) == 1:
            region_id = regions[0].id
            offset_reg_id_comb_to_check[region_comb] = _ConstRegionCheck(region_id)
            continue
        
        check_tree = dict()
        check_edges_for_regions = []
        remaining_check_edges_for_regions = []
        # To-Do: Does this method of picking any shared edge work reliably for raster cells where one needs to distinguish between multiple (n>2) regions?
        
        for i, region in enumerate(regions[:-1]):
            for edge in region.edges:
                reversed_coords = (edge.coords[1], edge.coords[0])
                shared_with_reg_id = edge_is_shared_with_region_id(reversed_coords, regions[i+1:])
                if shared_with_reg_id is None:
                    shared_with_reg_id = edge_is_shared_with_region_id2(edge, regions[i+1:])
                shared_with_reg_ids = edge_is_shared_with_region_id3(edge, regions[i+1:])
                
                if not shared_with_reg_id is None:
                    edge_check = make_edge_check(edge, r=r, include_boundary=include_boundary)
                    check_edges_for_regions.append((edge, edge_check))
                    
                    
                if len(shared_with_reg_ids) > 0:
                    edge_check = make_edge_check(edge, r=r, include_boundary=include_boundary)
                    check_edges_for_regions.append((edge, edge_check))
                
                if shared_with_reg_id is None and len(shared_with_reg_ids) == 0:
                    edge_check = make_edge_check(edge, r=r, include_boundary=include_boundary)
                    remaining_check_edges_for_regions.append((edge, edge_check))

        check_edges_for_regions = check_edges_for_regions + remaining_check_edges_for_regions
                
                #maybe the following steps needs to be applied always (that if theres not shared edge a shared vertex needs to be found)
        if len(check_edges_for_regions)==0:
            print("FOUND ZERO")
            if len(regions) > 2:
                print("WARNING ENSURE IF THIS METHOD IS RELIABLE WITH MORE THAN 2 REGIONS. n=",len(regions))
            # regions dont share an edge but they might share a vertex. thus an edge needs to be chosen (arbitrarily) 
            (edge_at_vertex, reg_at_vertex) = vertex_is_shared_with_region(regions[i:])
            edge_check = make_edge_check(edge_at_vertex, r=r, include_boundary=include_boundary)
            check_edges_for_regions.append((edge_at_vertex, edge_check))
        build_decision_tree(tree_pos=check_tree, checks=check_edges_for_regions, remaining_regions=regions, r=r, include_boundary=include_boundary)
        # try:
        #     build_decision_tree(tree_pos=check_tree, checks=check_edges_for_regions, remaining_regions=regions)
        #     # OffsetRegion.plot_many(regions=regions, plot_vertices=False, x_lim=(-.52,.52), y_lim=(-.52,.52))
        #     pass
        # except:
        #     print("S2")

        #     # OffsetRegion.plot_many(regions=regions, plot_vertices=False, x_lim=(-.52,.52), y_lim=(-.52,.52))
        #     ax = OffsetRegion.plot_many(regions=regions, plot_vertices=False, x_lim=(-.52,.52), y_lim=(-.52,.52))
        #     Edge.plot_many(edges={e.coords:e for e,c in check_edges_for_regions}, ax=ax)
        
        #     raise ValueError("trgl_nrs", [(reg.trgl_nr, reg.id, reg.trgl_nrs if hasattr(reg, 'trgl_nrs') else None) for reg in regions],"n regions", len(regions), "check edges", len(check_edges_for_regions), "checktree", check_tree)
        
        # def determine_offset_region_for_pts(
        #     pts: _np_array   
        # ): 
            
        #     return eval_region_tree(pts=pts, check_tree=check_tree)
        # offset_reg_id_comb_to_check[region_comb] = determine_offset_region_for_pts
        # if type(check_tree) != dict:
        #     raise ValueError("NOT A DICT")
        offset_reg_id_comb_to_check[region_comb] = _TreeCheck(check_tree)
        if region_comb == (1001,9010,19026):
            # print("region_comb",region_comb)
            # print("check_tree:", check_tree)
            def repr_check_tree(check_tree, ct_r={}):
                if 'check' in check_tree.keys():
                    ct_r['check'] = inspect.getsource(check_tree['check'])
                if True in check_tree.keys():
                    ct_r[True] = check_tree[True] if type(check_tree[True])==int else repr_check_tree(check_tree[True])
                if False in check_tree.keys():
                    ct_r[False] = check_tree[False] if type(check_tree[False])==int else repr_check_tree(check_tree[False])
                return ct_r
            # print("repr_check_tree(check_tree)", repr_check_tree(check_tree))
            # print("lambda:",offset_reg_id_comb_to_check[region_comb])
    # print("COORD_MATCH_STATS:", _coord_match_count)
    # print(offset_reg_id_comb_to_check.keys())
    # print("offset_reg_id_comb_to_check[(1001,9010,19026)]",offset_reg_id_comb_to_check[(1001,9010,19026)])
    # print("offset_reg_id_comb_to_check[(10,7)]",offset_reg_id_comb_to_check[(10,7)])
    return offset_reg_id_comb_to_check 
#

def prune_raster_candidates(
        raster_cell_to_regions,
        offset_reg_id_comb_to_check,
        offset_x_bins:list,
        offset_y_bins:list,
        id_to_offset_regions:dict=None,
        plot_offset_raster:bool=False,
):
    """
    Prune each raster cell's candidate region list to only those regions reachable from within the cell.

    For raster cells with more than one candidate, evaluates the cell's 8 sample points (4 corners,
    4 edge midpoints, centroid) through the region-id check function and retains only the regions
    that actually appear in the result. Raster cells where any region vertex lies strictly inside
    or on the boundary of the cell are kept with all candidates intact, because the 8-point sample
    may miss a thin wedge touching that vertex.
    This pass reduces the average number of candidate regions per raster cell without risk of
    excluding a region that a real data point could fall into.
    """
    precise_raster = dict()

    offset_x_bins = offset_x_bins
    offset_y_bins = offset_y_bins
    # lenchanges = set()
    # all_precise_ids = set()

    for (ix, iy), regions_at_raster_cell in raster_cell_to_regions.items():
        if len(regions_at_raster_cell)>1:
            xmin, xmax = offset_x_bins[ix-1]
            ymin, ymax = offset_y_bins[iy-1]

            # If any region vertex lies strictly inside the cell (or on its boundary),
            # the 4-corner test is unreliable: the wedge of that region touching the
            # vertex may not reach any of the other corners, causing that region to be
            # filtered out and its points wrongly assigned to a neighbour.
            # In that case keep all candidate regions so the check function decides at
            # point-assignment time.
            reg_ids_with_vertex_inside = set(
                reg.id
                for reg in regions_at_raster_cell
                for edge in reg.edges
                for (vx, vy) in edge.coords
                if xmin <= vx <= xmax and ymin <= vy <= ymax
            )
            if len(reg_ids_with_vertex_inside)>0:
                precise_raster[(ix, iy)] = regions_at_raster_cell
                continue

            reg_ids = tuple(sorted([reg.id for reg in regions_at_raster_cell]))
            check = offset_reg_id_comb_to_check[reg_ids]
            xmid = (xmin + xmax) / 2
            ymid = (ymin + ymax) / 2
            pts = _np_array([
                (xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax),
                (xmid, ymin), (xmax, ymid), (xmid, ymax), (xmin, ymid),
                (xmid, ymid),
            ])
            check_res = check(pts=pts)
            precise_ids = set(check_res).union(reg_ids_with_vertex_inside)

            preceise_regions_at_raster = [reg for reg in regions_at_raster_cell if reg.id in precise_ids]
            if len(preceise_regions_at_raster) == 0:
                OffsetRegion.plot_many([reg for reg in regions_at_raster_cell]+[id_to_offset_regions[id] for id in precise_ids], plot_vertices=False)
                raise ValueError("regions_at_raster_cell", len(regions_at_raster_cell), [reg.id for reg in regions_at_raster_cell], "checkres", check_res, "precise_ids", precise_ids)

            precise_raster[(ix, iy)] = preceise_regions_at_raster
        else:
            precise_raster[(ix, iy)] = regions_at_raster_cell

    # if not plot_offset_raster is None:
    #     create_raster_plot(regions=list(id_to_offset_regions.values()), raster_cell_to_regions=precise_raster, offset_x_bins=offset_x_bins, offset_y_bins=offset_y_bins)

    return precise_raster
#

def get_raster_cell_stats(precise_raster, offset_x_bins, offset_y_bins) -> tuple[dict, dict]:
        """
        Gets stats on how many regions are at each raster cell and what area of the triangle is covered by cells with a certain number of regions. This can be used to determine how much precision is needed for the raster to ensure that most cells only contain one region.
        """
        # TODO move this to after its extended to remainging triangles

        # print("precise_raster",precise_raster)
        n_counts = {}
        n_area = {}
        print(
            len(offset_x_bins), 
            len(offset_y_bins), 
            min([ix for ix, iy in precise_raster.keys()]), 
            max([ix for ix, iy in precise_raster.keys()]),
            min([iy for ix, iy in precise_raster.keys()]), 
            max([iy for ix, iy in precise_raster.keys()]),
            
            
            )
        print("offset_x_bins", offset_x_bins)
        for (ix, iy), regions_at_raster_cell in precise_raster.items():
            n = len(regions_at_raster_cell)
            n_counts[n] = n_counts.get(n, 0) + 1
            # weight half if on diagonal
            factor = 8 - 4*int(
                offset_x_bins[ix][0]==offset_y_bins[iy][0] and offset_x_bins[ix][1]==offset_y_bins[iy][1]
            )
            area = factor * (offset_x_bins[ix][1]-offset_x_bins[ix][0]) * (offset_y_bins[iy][1]-offset_y_bins[iy][0])
            n_area[n] = n_area.get(n, 0) + area
        total_area = sum(n_area.values())
        total_count = sum(n_counts.values())
        n_area = {n: area/total_area for n, area in n_area.items()}  
        n_counts = {n: count/total_count for n, count in n_counts.items()}     
        print("n_counts", n_counts)
        print("n_area", n_area)
        return (n_area, n_counts)

def expand_raster_to_all_sectors(
        id_to_offset_regions:dict,
        raster_cell_to_regions:dict,
        offset_x_bins:list,
        offset_y_bins:list,
        translate_reg_nr_to_reg_id:dict,
        plot_offset_raster:dict=None
    ):
    """
    Extend the triangle-1 raster to all 8 symmetry sectors covering the full unit cell [-0.5, 0.5]².

    Iterates over all combinations of (sign_x, sign_y, sorting) that define the 8 triangular
    sectors. For each raster cell (ix, iy) in the triangle-1 raster it computes the corresponding
    (x, y) key in the extended raster and maps the triangle-1 region numbers to their rotated/
    reflected counterparts via translate_reg_nr_to_reg_id. Diagonal cells (|x|==|y|) may receive
    contributions from two sectors; their region lists are merged rather than overwritten.
    Also derives offset_all_x_vals and offset_all_y_vals: the full symmetric breakpoint arrays
    used by pts_to_offset_regions to bin each point's fractional cell offset.
    Returns the extended raster_cell_to_regions, offset_all_x_vals, offset_all_y_vals, and
    unique_reg_id_combs_to_raster_cells rebuilt from the merged raster.
    """
    raster_cell_id_to_bounds = dict()
    unique_reg_id_combs_to_raster_cells = dict()
    
    for sign_x in [-1, 1]:
        for sign_y in [-1, 1]:
            # sorting 1 if abs(x)>abs(y) else -1
            for sorting in [1, -1]:
                triangle_nr = (
                    (
                        1 if sorting>0 else 2
                    ) if sign_y>0 else (
                        8 if sorting>0 else 7
                    )
                ) if sign_x>0 else (
                    (
                        4 if sorting>0 else 3
                    ) if sign_y>0 else (
                        5 if sorting>0 else 6 
                    )
                )

                for ix, (x_low, x_up) in zip(range(1,len(offset_x_bins)+1), offset_x_bins):
                    for iy, (y_low, y_up) in zip(range(1,len(offset_y_bins)+1), offset_y_bins):
                        if iy > ix:
                            break 
                        regions_at_raster_cell = raster_cell_to_regions[(ix, iy)]
                        # raster_cell_to_regions[(ix*sign_x, iy*sign_y) if sorting>0 else (iy*sign_y, ix*sign_x)] = list(set([translate_reg_nr_to_reg_id[region_nr] for region_nr in region_nrs]))
                        i_min, i_max = (iy, ix) if sorting > 0 else (ix, iy)
                        x, y = (i_max*sign_x, i_min*sign_y)
                        raster_cell_id_to_bounds[(x,y)] = ((x_low, x_up), (y_low, y_up))
                        region_nrs = [reg.nr for reg in regions_at_raster_cell] # if abs(reg.trgl_nr - triangle_nr)%7 <= 1 else reg.nr+1 
                        # try:
                        # region_ids = []
                        # for region_nr in region_nrs:
                        #     new_nr = region_nr//10*10+triangle_nr
                        #     new_reg = id_to_offset_regions[translate_reg_nr_to_reg_id[new_nr]]
                        #     if abs(new_reg.trgl_nr  - triangle_nr)%7 <= 1:
                        #         pass
                        #         if hasattr(new_reg, 'trgl_nrs'):
                        #             pass
                        #             # print("ATTT", triangle_nr,  new_reg.trgl_nr, new_reg.trgl_nrs)
                        #     else:
                        #         old_new_nr = new_nr
                        #         old_reg = new_reg
                        #         new_nr += 1 if triangle_nr < 8 else -7
                        #         new_reg = id_to_offset_regions[translate_reg_nr_to_reg_id[new_nr]]
                        #         # print(
                        #         #     "TN", triangle_nr, "new",new_nr%10, "old",old_new_nr%10,  
                        #         #     "\nnew:", (new_reg.trgl_nr, new_reg.trgl_nrs if hasattr(new_reg,'trgl_nrs') else ''),
                        #         #     "\nold:", (old_reg.trgl_nr, old_reg.trgl_nrs if hasattr(old_reg,'trgl_nrs') else ''))
                        #     region_ids.append(translate_reg_nr_to_reg_id[new_nr])
                        # region_ids = tuple(region_ids)

                        mapped = [translate_reg_nr_to_reg_id[region_nr//10*10+triangle_nr] for region_nr in region_nrs]
                        region_ids = tuple(sorted(set(mapped)))
                        # except:
                        #     print("translate_reg_nr_to_reg_id", translate_reg_nr_to_reg_id)
                        #     raise ValueError("region_nrs", region_nrs, "A",[region_nr//10*10+triangle_nr for region_nr in region_nrs])
                        new_regions = [id_to_offset_regions[id] for id in region_ids]
                        if (x, y) in raster_cell_to_regions:
                            # Diagonal cell (|x|==|y|): two triangles produce the same key.
                            # Merge instead of overwrite so both triangles' regions are candidates.
                            existing_ids = {r.id for r in raster_cell_to_regions[(x, y)]}
                            extra = [r for r in new_regions if r.id not in existing_ids]
                            if extra:
                                raster_cell_to_regions[(x, y)] = raster_cell_to_regions[(x, y)] + extra
                        else:
                            raster_cell_to_regions[(x, y)] = new_regions

    # Rebuild unique_reg_id_combs_to_raster_cells from the final merged raster_cell_to_regions
    # (diagonal cells may have accumulated regions from two triangles)
    unique_reg_id_combs_to_raster_cells = {}
    for (cx, cy), regions in raster_cell_to_regions.items():
        comb = tuple(sorted(r.id for r in regions))
        if comb not in unique_reg_id_combs_to_raster_cells:
            unique_reg_id_combs_to_raster_cells[comb] = []
        unique_reg_id_combs_to_raster_cells[comb].append((cx, cy))

    offset_x_vals = get_vals_from_bins(offset_x_bins) 
    offset_y_vals = get_vals_from_bins(offset_y_bins) 
    offset_all_x_vals = [-x for x in reversed(offset_x_vals[1:])] + offset_x_vals
    offset_all_y_vals = [-y for y in reversed(offset_y_vals[1:])] + offset_y_vals

    if not plot_offset_raster is None:
        create_raster_plot(
            regions=list(id_to_offset_regions.values()),
            raster_cell_to_regions=raster_cell_to_regions,
            offset_x_bins=offset_x_bins,
            offset_y_bins=offset_y_bins,
            lims=[-.55,.55], add_raster_labels=True)

    return raster_cell_to_regions, offset_all_x_vals, offset_all_y_vals, unique_reg_id_combs_to_raster_cells
#


def build_check_index(
        raster_cell_to_regions,
        offset_reg_id_comb_to_check,
):
    """
    Build integer-keyed lookup tables for fast raster-cell → check dispatch.

    Assigns a compact integer index to each unique combination of region ids that appears
    in raster_cell_to_regions. Returns:
      - raster_cell_to_region_comb_nr: maps (ix, iy) → combination index (integer)
      - offset_region_comb_nr_to_check: maps combination index → check lambda
    This indirection avoids repeated tuple key lookups in the inner search loop.
    """
     
    # region_comb_to_nr = dict()
    # if not region_ids in region_comb_to_nr:
    #     region_comb_to_nr[region_ids] = len(region_comb_to_nr)
    raster_cell_to_region_comb_nr = dict()
    offset_region_comb_nr_to_check = dict()
    region_comb_to_nr = dict()
    for key, regions in raster_cell_to_regions.items():
        region_ids = tuple(sorted(set(reg.id for reg in regions)))
        if not region_ids in region_comb_to_nr:
            region_comb_to_nr[region_ids] = len(region_comb_to_nr)
            offset_region_comb_nr_to_check[region_comb_to_nr[region_ids]] = offset_reg_id_comb_to_check[region_ids]
        #
        raster_cell_to_region_comb_nr[key] = region_comb_to_nr[region_ids]
    return raster_cell_to_region_comb_nr, offset_region_comb_nr_to_check
#

_REGION_AND_TRGL_MULT = 10  # triangles 1..8; multiplier must exceed max triangle index (8)

def build_region_cell_lookups(
        id_to_offset_regions: dict,
        shared_cntd_cells: set,
) -> dict:
    """
    Build all region-cell lookup dicts that the radius-search loop reads at runtime.

    Returns a dict whose keys map directly to grid.search attributes:
      - region_id_to_{cntd,ovlpd,nested_cntd,nested_ovlpd,
                       distinct_cntd,distinct_ovlpd}_cells:
            region_id → list of (lvl, (row, col)) tuples at the base (level-0) and nested
            sub-cell levels respectively.
      - region_and_trgl_id_to_{nested_cntd,nested_ovlpd,
                               distinct_cntd,distinct_ovlpd}_cells:
            region_and_trgl_id (= region_id * 10 + triangle_nr) → list of (lvl, (row, col)).
            For symmetry-merged regions each of the 8 triangles may have its own
            nested cell list (stored in reg.nested_cells_by_trgl); all others share
            the region's standard list.
      - region_and_trgl_mult: the multiplier used to encode region_and_trgl_id (always 10).
      - shared_cntd_cells: passed through unchanged so the caller only needs to store the returned dict.
    """
    attr = {
        'region_id_to_cntd_cells':                 {rid: list(reg.cntd_cells)          for rid, reg in id_to_offset_regions.items()},
        'region_id_to_ovlpd_cells':                {rid: list(reg.ovlpd_cells)         for rid, reg in id_to_offset_regions.items()},
        'region_id_to_area':                       {rid: reg.calc_area()               for rid, reg in id_to_offset_regions.items()},
        'shared_cntd_cells':                       shared_cntd_cells,
        'region_and_trgl_mult':                    _REGION_AND_TRGL_MULT,
    }

    region_and_trgl_to_nested_cntd_cells:   dict = {}
    region_and_trgl_to_distinct_cntd_cells: dict = {}
    region_and_trgl_to_distinct_ovlpd_cells: dict = {}
    for reg_id, reg in id_to_offset_regions.items():
        nc_std = list(reg.nested_cntd_cells)
        no_std = list(reg.nested_ovlpd_cells)
        dc_std = list(reg.distinct_cntd_cells)
        do_std = list(reg.distinct_ovlpd_cells)
        by_trgl = getattr(reg, 'nested_cells_by_trgl', {})
        for trgl_nr in range(1, 9):
            key = reg_id * _REGION_AND_TRGL_MULT + trgl_nr
            if trgl_nr in by_trgl:
                nc, no = by_trgl[trgl_nr]
                region_and_trgl_to_nested_cntd_cells[key]   = list(nc)

                region_and_trgl_to_distinct_cntd_cells[key] = [(lvl, (r, c)) for lvl, (r, c) in nc if (lvl, (r, c)) not in shared_cntd_cells]
                region_and_trgl_to_distinct_ovlpd_cells[key] = list(no)
            else:
                region_and_trgl_to_nested_cntd_cells[key]   = nc_std

                region_and_trgl_to_distinct_cntd_cells[key] = dc_std
                region_and_trgl_to_distinct_ovlpd_cells[key] = do_std

    attr['region_and_trgl_id_to_nested_cntd_cells']   = region_and_trgl_to_nested_cntd_cells
    attr['region_and_trgl_id_to_distinct_cntd_cells'] = region_and_trgl_to_distinct_cntd_cells
    attr['region_and_trgl_id_to_distinct_ovlpd_cells']= region_and_trgl_to_distinct_ovlpd_cells
    return attr



from aabpl.utils.progress import DiskRegionProgress as _DiskRegionProgress

# Maximum number of (r/spacing, nest_depth, include_boundary) configurations kept
# in memory simultaneously.  Each entry holds the full region-geometry graph
# (~5–20 MB of Python objects).  Raise this if you sweep many distinct
# configurations in one session; lower it on memory-constrained machines.


@time_func_perf
def build_disk_region_lookups(
        grid:dict,
        grid_spacing:float,
        r:float,
        include_boundary:bool=False,
        nest_depth:int=0,
        plot_offset_checks:dict=None,
        plot_offset_regions:dict=None,
        plot_offset_raster:dict=None,
        silent:bool=True,
):
    """
    Pre-compute all offset-region lookup structures for a given (r, grid_spacing, nest_depth) triple.

    Results are cached in-process keyed on (r/grid_spacing, nest_depth, include_boundary).
    Repeated calls with the same geometry (e.g. during a parameter sweep) reuse the cached
    result at no cost. The cache persists for the lifetime of the Python session.

    Orchestrates the full pipeline:
      1. Determine which grid cells can overlap or be cntd by the search disk.
      2. Create triangle-1 regions and split them with arc/line boundary checks.
      3. Classify cntd/ovlpd cells per region; optionally subdivide boundary cells
         into nested sub-cell quadrants up to nest_depth levels.
      4. Rotate/reflect triangle-1 regions into all 8 symmetry sectors and deduplicate.
      5. Extract cells shared by every region (computed once per grid cell, not per point).
      6. Build the raster that maps fractional point offsets to candidate regions.
      7. Construct per-raster-cell check trees for assigning a point to its exact region.

    Returns a dict with keys:
      raster_cell_to_region_comb_nr, offset_region_comb_nr_to_check,
      offset_all_x_vals, offset_all_y_vals, id_to_offset_regions,
      contain_region_mult, plus all keys from region_cell_lookups.
    """
    _cache_key = (round(r / grid_spacing, 8), nest_depth, include_boundary)
    if _cache_key in _config.disk_region_cache and plot_offset_checks is None and plot_offset_regions is None and plot_offset_raster is None:
        if not silent and _outer_progress.get() is None:
            print("Reusing cached disk region lookups for r/spacing="+str(round(r/grid_spacing, 6))+" nest_depth="+str(nest_depth))
        # LRU: mark this entry most-recently-used so eviction (popitem(last=False))
        # drops the genuinely least-recently-used config, not the oldest-inserted.
        _config.disk_region_cache.move_to_end(_cache_key)
        return {**_config.disk_region_cache[_cache_key], 'id_to_offset_regions': {}}

    _prog = _DiskRegionProgress(silent=silent, r_over_spacing=r/grid_spacing, nest_depth=nest_depth)
    _prog.start()

    _prog.step("Classify cells")
    (cells_cntd_in_all_disks,
     cells_cntd_in_all_trgl_disks,
     cells_maybe_overlapping_a_disk,
     cells_maybe_overlapping_a_trgl_disk
    ) = classify_disk_cells(grid_spacing=grid_spacing, r=r, include_boundary=False)

    (cells_cntd_in_all_disks,
     cells_cntd_in_all_trgl_disks,
     cells_maybe_overlapping_a_disk,
     cells_maybe_overlapping_a_trgl_disk
    ) = classify_disk_cells_by_level(grid_spacing=grid_spacing, r=r, include_boundary=False, nest_depth=nest_depth)

    _prog.step("Init triangle + boundary checks")
    trgl_regions = init_triangle1_region()
    check_dict, cells_always_ovlpd = build_boundary_checks(cells_maybe_overlapping_a_trgl_disk, trgl_regions, r=r)

    _prog.step("Split regions")
    split_regions_by_checks(check_dict=check_dict, trgl_regions=trgl_regions, r=r, plot_offset_checks=plot_offset_checks)

    _prog.step("Finalize region cells")
    finalize_region_cells(
        trgl_regions=trgl_regions,
        cells_cntd_in_all_trgl_disks=cells_cntd_in_all_trgl_disks,
        cells_always_ovlpd=cells_always_ovlpd,
        all_cells=cells_maybe_overlapping_a_trgl_disk,
        cells_cntd_in_all_disks=cells_cntd_in_all_disks,
        plot_offset_regions=plot_offset_regions,
        r=r,
        include_boundary=include_boundary,
        grid_spacing=grid_spacing,
        nest_depth=nest_depth,
    )
    # print("trgl_regions",trgl_regions)
    # print([type(r) for r in trgl_regions.values()])
    # ax=OffsetRegion.plot_many([r for r in trgl_regions.values() if r.shared_along_vert and r.shared_along_diag ], add_idxs=False, edgecolor='black',facecolor="orange")
    # OffsetRegion.plot_many([r for r in trgl_regions.values() if r.shared_along_vert and not r.shared_along_diag], add_idxs=False, edgecolor='black',facecolor="red",ax=ax)
    # OffsetRegion.plot_many([r for r in trgl_regions.values() if not r.shared_along_vert and r.shared_along_diag], add_idxs=False, edgecolor='black',facecolor="yellow",ax=ax)
    # OffsetRegion.plot_many([r for r in trgl_regions.values() if not r.shared_along_vert and not r.shared_along_diag], add_idxs=False, edgecolor='black',facecolor="grey",ax=ax)
    # print("_3", "area of triangle:0.125=", sum([r.calc_area() for r in trgl_regions.values()]), all([r.is_closed for r in trgl_regions.values()]))
    _prog.step("Expand to 8 sectors")
    (id_to_offset_regions,
     translate_reg_nr_to_reg_id,
     contain_region_mult
    ) = expand_regions_to_all_sectors(trgl_regions=trgl_regions, r=r, nest_depth=nest_depth)
    shared_cntd_cells = extract_shared_cells(id_to_offset_regions)
    
    # print("_4", sum([r.calc_area() for r in id_to_offset_regions.values()]), all([r.is_closed for r in id_to_offset_regions.values()]))
    if False:
        print("--------r",r,"--------")
        fig,ax = plt.subplots(1, 1, figsize=(25,25))
        ax = OffsetRegion.plot_many(
            id_to_offset_regions.values(),
            ax=ax,  cmap='viridis',
            add_idxs={} if len(id_to_offset_regions)<150 else False, 
            edgecolor='black', alpha=1,
            x_lim=[-.501,.501], y_lim=[-.501,.501],
            )
        ax.set_facecolor('red')
        ax.plot([-1,1],[-1,1],color='white',linewidth=1.5)
        ax.plot([-1,1],[1,-1],color='white',linewidth=1.5)
        ax.vlines(x=0, ymin=-1, ymax=1,color='white',linewidth=1.5)
        ax.hlines(y=0, xmin=-1, xmax=1,color='white',linewidth=1.5)
        ax.set_title("r="+str(r)+". "+str(len(id_to_offset_regions))+" regions.")
        fig.savefig('plots/regions_r_'+str(r)[0]+'_'+str(r)[2:]+".png", dpi=100, bbox_inches="tight")
        _plt_close(fig)
    # ax=OffsetRegion.plot_many(id_to_offset_regions.values(),facecolor='red', add_idxs=False, edgecolor='None', x_lim=[-.501,.501], y_lim=[-.501,.501])
    # ax.plot([-1,1],[-1,1],color='black',linewidth=0.1)
    # ax.plot([-1,1],[1,-1],color='black',linewidth=0.1)
    # ax.vlines(x=0, ymin=-1, ymax=1,color='black',linewidth=0.1)
    # ax.hlines(y=0, xmin=-1, xmax=1,color='black',linewidth=0.1)
    _prog.step("Build region raster")
    (trgl_raster_cell_to_region,
     offset_x_bins,
     offset_y_bins,
     unique_reg_id_combs_in_trgl_raster_cells,
     ) = build_region_raster(trgl_regions=id_to_offset_regions, r=r, plot_offset_raster=plot_offset_raster)
    offset_reg_id_comb_to_check = build_raster_checks(
        id_to_offset_regions=id_to_offset_regions,
        unique_reg_id_combs_to_raster_cells=unique_reg_id_combs_in_trgl_raster_cells,
        r=r,
        include_boundary=True,
    )
    trgl_precise_raster_cell = trgl_raster_cell_to_region
    
    

    _prog.step("Expand raster + check structures")
    (raster_cell_to_regions,
     offset_all_x_vals,
     offset_all_y_vals,
     unique_reg_id_combs_to_raster_cells
     ) = expand_raster_to_all_sectors(
         id_to_offset_regions=id_to_offset_regions,
         raster_cell_to_regions=trgl_precise_raster_cell,
         translate_reg_nr_to_reg_id=translate_reg_nr_to_reg_id,
         offset_x_bins=offset_x_bins,
         offset_y_bins=offset_y_bins,
         plot_offset_raster=plot_offset_raster)
    offset_reg_id_comb_to_check = build_raster_checks(
        id_to_offset_regions=id_to_offset_regions,
        unique_reg_id_combs_to_raster_cells=unique_reg_id_combs_to_raster_cells,
        r=r,
        include_boundary=include_boundary,
    )

    _prog.step("Build lookup tables")
    raster_cell_to_region_comb_nr, offset_region_comb_nr_to_check = build_check_index(
        raster_cell_to_regions=raster_cell_to_regions,
        offset_reg_id_comb_to_check=offset_reg_id_comb_to_check
    )

    regions = list(id_to_offset_regions.values())

    n_cntd = len(shared_cntd_cells)
    n_ovlpd = 0
    area_ctnd_by_all = area_cntd = sum([2**(-2*lvl) for lvl, cell in shared_cntd_cells])
    area_ovlpd = 0
    total_reg_area = 0
    for region in regions:
        region_area = region.calc_area()
        total_reg_area+=region_area
        n_cntd += region_area * (len(region.distinct_cntd_cells)) 
        n_ovlpd += region_area * (len(region.distinct_ovlpd_cells))
        area_cntd += region_area * (sum([2**(-2*lvl) for lvl, cell in region.distinct_cntd_cells])) 
        area_ovlpd += region_area * (sum([2**(-2*lvl) for lvl, cell in region.distinct_ovlpd_cells])) 
    # print("cn", round(area_cntd/(_math_pi *r**2),4), 
    #       "ov", round(area_ovlpd/(_math_pi *r**2),4),
    #       "tot", round((area_ovlpd+area_cntd)/(_math_pi *r**2),4),
    #       "sh_cn", round(area_ctnd_by_all/(_math_pi *r**2),4),
    #       "sh_ov",  round(area_olvpd_by_all/(_math_pi *r**2),4))
    
    # print("n_cn", n_cntd, 
    #       "n_ov", n_ovlpd, 
    #       'sh_cn',len(shared_cntd_cells), 
    #       'sh_ov', len(shared_ovlpd_cells))
    
    _region_cell_lookups = build_region_cell_lookups(id_to_offset_regions, shared_cntd_cells)
    _prog.done()
    _result = {
        'raster_cell_to_region_comb_nr':  raster_cell_to_region_comb_nr,
        'offset_region_comb_nr_to_check': offset_region_comb_nr_to_check,
        'offset_all_x_vals':              offset_all_x_vals,
        'offset_all_y_vals':              offset_all_y_vals,
        'contain_region_mult':            contain_region_mult,
        **_region_cell_lookups,
        'id_to_offset_regions':           id_to_offset_regions,
    }
    _cached = {k: v for k, v in _result.items() if k != 'id_to_offset_regions'}
    _config.disk_region_cache[_cache_key] = _cached
    _evict_disk_region_cache()
    return _result


def _evict_disk_region_cache():
    """Trim disk_region_cache to DISK_REGION_CACHE_MAXSIZE.

    Eviction is LRU, but protects the deepest entry per (spacing_ratio,
    include_boundary): for each such group the entry with the largest nest_depth
    is kept as long as possible (it is the one a shallower nest_depth can be
    derived from cheaply). Only when every remaining entry is the deepest of its
    group do we fall back to evicting the plain least-recently-used one.
    """
    cache = _config.disk_region_cache
    while len(cache) > _config.DISK_REGION_CACHE_MAXSIZE:
        # deepest nest_depth per (spacing_ratio, include_boundary) -> protected keys
        deepest = {}
        for (ratio, nd, incl) in cache:
            grp = (ratio, incl)
            if grp not in deepest or nd > deepest[grp][1]:
                deepest[grp] = (ratio, nd, incl)
        protected = set(deepest.values())
        # evict the least-recently-used unprotected entry; if all are protected,
        # evict the plain LRU (oldest) so we never loop forever.
        victim = next((k for k in cache if k not in protected), None)
        if victim is None:
            cache.popitem(last=False)
        else:
            del cache[victim]


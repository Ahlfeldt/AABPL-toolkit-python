# intersection of two circles with same r
from numpy import (linspace as _np_linspace, array as _np_array, abs as _np_abs, 
                   dot as _np_dot, roll as _np_roll, cross as _np_cross, average as _np_average)
from math import sin as _math_sin, cos as _math_cos, pi as _math_pi
from matplotlib import pyplot as plt
from aabpl.utils.misc import angles_to_origin, angle_to, pt_is_left_of_vector
from aabpl.utils.rotations import transform_cell_pattern, transform_coord, transform_cell
from aabpl.utils.intersections import circle_line_segment_intersection, line_intersection, intersections_pts_arc_to_circle, arc_line_segment_intersection
from matplotlib.patches import (Rectangle as _plt_Rectangle, Polygon as _plt_Polygon, Circle as _plt_Circle)
from matplotlib.pyplot import (subplots as _plt_subplots, colorbar as _plt_colorbar, get_cmap as _plt_get_cmap)
from shapely.geometry import Polygon as _shapely_Polygon, Point as _shapely_Point
from mpmath import mp
from aabpl.illustrations.plot_utils import plot_polygon
from matplotlib.pyplot import close as _plt_close

# from shapely.geometry import Polygon as _shapely_Polygon, LineString, Point


# for each circle remember the meaning of the check (contains / overlaps)
# for each intersection point save what circle it comes from
# for each circle get intersection points with home cell
# for each circle-intersection point check whether its in triangle 1
_PRECISION_ = 15
TANGENT_TOL_CIRCLE = 1e-16
TANGENT_TOL_LINE = 1e-16
MATCH_PTS_TOL = TANGENT_TOL_CIRCLE*10
MATCH_TO_AXIS_TOL = TANGENT_TOL_CIRCLE*10
MATCH_TO_45_TOL = TANGENT_TOL_CIRCLE*10
RESCALE_FACTOR = 100000000
RESCALE_FACTOR = 1000000
RESCALE_FACTOR = 1
# DECIMALS = None
DECIMALS = 117
# DECIMALS = 98


def calculate_polygon_centroid(coords):
    return _shapely_Polygon(coords).centroid.coords[0]


def _to_shapely_polygon(region, arc_steps_per_degree:float=20):
    """
    Build a Shapely polygon from the actual region boundary.
    Uses the plotted boundary coordinates, which is essential for arc edges
    and for tiny split regions where only the endpoint list would be degenerate.
    """
    try:
        coords = region.get_plot_coords(arc_steps_per_degree=arc_steps_per_degree)
    except Exception:
        coords = [vtx.xy for vtx in region.vertices]

    if len(coords) < 4:
        return None

    if coords[0] != coords[-1]:
        coords = [*coords, coords[0]]

    return _shapely_Polygon(coords)


def _signed_polygon_area(coords):
    """
    Signed shoelace area of a polygonal chain.
    Positive -> CCW, negative -> CW.
    """
    if len(coords) < 3:
        return 0.0

    area = 0.0
    for (x1, y1), (x2, y2) in zip(coords, coords[1:] + coords[:1]):
        area += x1 * y2 - x2 * y1
    return area / 2.0


def _normalize_region_winding(region, arc_steps_per_degree:float=50):
    """
    Force a region boundary to be stored in CCW order.

    The polygon winding is determined from the actual plotted coordinates, and the
    boundary is reversed by using each edge's own reverse() implementation. This
    keeps line segments and arcs consistent with the region orientation.
    """
    try:
        plot_coords = region.get_plot_coords(arc_steps_per_degree=arc_steps_per_degree)
    except Exception:
        plot_coords = [vtx.xy for vtx in region.vertices]
        if region.is_closed and plot_coords and plot_coords[0] != plot_coords[-1]:
            plot_coords.append(plot_coords[0])

    if len(plot_coords) < 3:
        return region

    if _signed_polygon_area(plot_coords) >= 0:
        region.is_clockwise = False
        return region

    # Reverse the whole boundary in a way that preserves edge semantics.
    # Prefer the existing edge.reverse() path so arcs are handled consistently.
    region.edges = [edge.reverse() for edge in reversed(region.edges)]
    region.vertices = list(reversed(region.vertices))
    region.coords = tuple(edge.coords for edge in region.edges)
    region.centroid = calculate_polygon_centroid(region.get_plot_coords(arc_steps_per_degree=arc_steps_per_degree))
    region.is_clockwise = False
    return region
    # import warnings

    # def fxn():
    #     warnings.warn("deprecated", DeprecationWarning)

    # with warnings.catch_warnings(record=True) as w:
    #     # Cause all warnings to always be triggered.
    #     warnings.simplefilter("always")
    #     # Trigger a warning.
    #     fxn()
    #     # Verify some things
    #     assert len(w) == 1
    #     assert issubclass(w[-1].category, DeprecationWarning)
    #     assert "deprecated" in str(w[-1].message)
    # coords = _np_array(coords)
    # coords_1 = _np_roll(coords, -1, axis=0)
    # # Compute signed area of each triangle
    # signed_areas = 0.5 * _np_cross(coords, coords_1)
    # # Compute centroid of each triangle
    # centroids = (coords + coords_1) / 3.0
    # # # Get average of those centroids, weighted by the signed areas.
    # # centroid = _np_average(centroids, axis=0, weights=signed_areas)
    # try:
    #     centroid = _np_average(centroids, axis=0, weights=signed_areas)
    # except:
    #     centroid = _shapely_Polygon(coords).centroid.coords[0]
    # centroid = tuple([float(c) for c in centroid])
    # return centroid

class Vertex(object):
    """
    2D coordinate
    """

    def __init__(self,x,y,all_vtx):
        """
        create Vertex at coordinate
        """
        self.x = x
        self.y = y
        self.xy = (x,y)
        # self.outgoing_edges = []
        # self.incoming_edges = []
        
        self.regions = []
        self.all_vtx = all_vtx
        if not self.xy in all_vtx:
            all_vtx[self.xy] = self
        else:
            pass
            # self = all_vtx[self.xy]
    
    @staticmethod
    def clear(all_vtx):
        """
        clear all_vtx
        """
        all_vtx.clear()
    #

    @staticmethod
    def plot_many(vertices:dict=None, ax=None):
        """
        plot vertices at list of coordinates. if no list provided instead plots all stored in all_vtx
        """
        if ax is None:
          fig, ax = plt.subplots()

        vertices = list(vertices.values()) if type(vertices)==dict else vertices
        if len(vertices):
            ax.scatter(x=[vtx[0] if type(vtx)==tuple else vtx.x for vtx in vertices], 
                       y=[vtx[1] if type(vtx)==tuple else vtx.y for vtx in vertices], 
                       marker='*', color='black'
                       )
        #
        return ax

    #
    
    def __repr__(self):
        props_not_to_print = ['all_vtx', 'outgoing_edges', 'incoming_edges', 'regions']
        return str(tuple([round(a,5) for a in self.xy]))
        # return str({key: val if type(val) != float else round(val,5) for key, val in self.__dict__.items() if key not in props_not_to_print})
    #

    def delete(self):
        """
        Remove vertex from all_vtx
        """
        self.all_vtx.pop(self.xy, None)
    #
#


class Edge(object):
    
    def __init__(
            self,
            vtx1:Vertex,
            vtx2:Vertex,
            all_edges:dict,
            contains:tuple=None,
            overlaps:tuple=None,
            enforce_int:bool=True
            ):
        """
        vertices are order counter clockwise
        enforce_int ensures that contains and overlaps are tuples of int() not numpy.int
        """
        self.vtx1 = vtx1
        self.vtx2 = vtx2
        self.coords = (
            vtx1.xy if (vtx1 is not None) else (None, None), 
            vtx2.xy if (vtx2 is not None) else (None, None),
            )
        self.regions = []
        self.plot_coords = None
        # vtx1.outgoing_edges.append(self)
        # vtx2.incoming_edges.append(self)
        self.all_edges = all_edges
        self.all_vtx = vtx1.all_vtx
        
        if not self.coords in all_edges:
            all_edges[self.coords] = self

        if not contains is None:
            self.contains  = tuple([int(z) for z in contains]) if enforce_int else contains
        if not overlaps is None:
            self.overlaps  = tuple([int(z) for z in overlaps]) if enforce_int else overlaps
    #

    @staticmethod
    def clear(all_edges):
        all_edges.clear()
    #

    def __repr__(self):
        props_not_to_print = ['all_edges', 'all_vtx', 'regions','coords', 'angle_min', 'angle_max', 'plot_coords']
        return str({key: val if type(val) != float else round(val,5) for key, val in self.__dict__.items() if key not in props_not_to_print})
    #
    
    def delete(self):
        self.all_edges.pop(self.coords, None)
    #
    def get_plot_coords(self, arc_steps_per_degree:float=5):
        if not self.plot_coords is None:
            return self.plot_coords
        self.plot_coords = [self.vtx1.xy, self.vtx2.xy]
        return self.plot_coords
    #
    @staticmethod
    def plot_many(
        edges:dict,
        ax=None,
        color_dict={'Arc':'blue', 'LineSegment':'green','Circle':'orange', 'other':'red'},
        **kwargs
        ):
        if ax is None:
          fig, ax = plt.subplots()
        #
        for coords,edge in edges.items():
            edge.plot_single(ax=ax, color_dict=color_dict, **kwargs)
        return ax
        #
    #

    def plot_single(
            self,
            ax=None,
            radial_lines:bool=True,
            full_circle:str=None,
            add_arrow:str=None,
            color_dict:dict={'Arc':'blue', 'LineSegment':'green','Circle':'orange', 'other':'red'},
            alpha:float=0.3,
            facecolor='None',
            edgecolor=None,
            **kwargs):
        if ax is None:
          fig, ax = plt.subplots()
        #
        plot_coords = self.get_plot_coords()
        if self.type == 'Arc':
            # ax.plot([0, self.center[0]], [0, self.center[1]], marker='.', color="black")
            rotation = angle_to((0,0), self.center)
            if radial_lines:
                ax.plot(
                    [self.center[0], self.center[0]+self.r*_math_cos((rotation+self.angle_vtx1)/360*2*_math_pi)],
                    [self.center[1], self.center[1]+self.r*_math_sin((rotation+self.angle_vtx1)/360*2*_math_pi)],
                    marker='.', color='red',alpha=0.3,)
                ax.plot(
                    [self.center[0], self.center[0]+self.r*_math_cos((rotation+self.angle_vtx2)/360*2*_math_pi)],
                    [self.center[1], self.center[1]+self.r*_math_sin((rotation+self.angle_vtx2)/360*2*_math_pi)],
                    marker='.', color='green',alpha=0.3,)
            
            # if next((True for c in x_coords if abs(c)>2),False):
            #     print("x_coords",x_coords)
            # if next((True for c in y_coords if abs(c)>2),False):
            #     print("y_coords",y_coords)
            if not full_circle is None:
                ax.add_patch(_plt_Circle(xy=self.center, radius=self.r, facecolor=facecolor, edgecolor=full_circle))
        x_coords = [x for x,y in plot_coords]
        y_coords = [y for x,y in plot_coords]
        ax.plot(x_coords, y_coords, marker='o', color=edgecolor or color_dict[self.type],alpha=alpha, **kwargs)
        edge_len = ((x_coords[-1]-x_coords[0])**2+(y_coords[-1]-y_coords[0])**2)**.5
        if add_arrow:
            ax.arrow(
                x=x_coords[0], 
                y=y_coords[0], 
                dx=(x_coords[-1]-x_coords[0]), 
                dy=(y_coords[-1]-y_coords[0]), 
                width=edge_len/20,
                length_includes_head=True,
                facecolor=add_arrow if type(add_arrow)==str else 'pink',
                edgecolor='black',
                alpha=alpha**.5,                  
                )
        ax.set_aspect('equal')
        return ax
    #
#

class LineSegment(Edge):
    """LineSegment between two 2D coordinates"""
    def __init__(
            self,
            vtx1:Vertex,
            vtx2:Vertex,
            all_edges:dict,
            contains:tuple=None,
            overlaps:tuple=None
        ):
        super().__init__(vtx1, vtx2, all_edges, contains=contains, overlaps=overlaps)
        self.type = 'LineSegment'
    #
    def reverse(self):
        self.vtx1, self.vtx2 = self.vtx2, self.vtx1
        self.coords = (self.vtx1.xy, self.vtx2.xy)
        return self

    def intersection(
            self,
            edge,
            TANGENT_TOL:float = 1e-16,
            RESCALE_FACTOR:int = 1,
            DECIMALS:int = 117,
            ):
        """
        
        """
        if edge.type == 'LineSegment':
            res = line_intersection(
                self.coords, 
                edge.coords, 
                tangent_tol=TANGENT_TOL, 
                rescale=RESCALE_FACTOR, 
                decimals=DECIMALS)
            # if len (res)<1:
            #     print('itx',res,"sc",self.coords,"ec", edge.coords)
            #     ax = self.plot_single(edgecolor='red')
            #     ax = edge.plot_single(edgecolor='blue',ax=ax)
            return res
        # print("-L--", edge.center, edge.r, self.vtx1.xy, self.vtx2.xy)
        if edge.type == 'Circle':
            return circle_line_segment_intersection(
                circle_center=edge.center,
                circle_radius=edge.r,
                pt1=self.vtx1.xy,
                pt2=self.vtx2.xy,
                full_line=False,
                tangent_tol=TANGENT_TOL,
                rescale=RESCALE_FACTOR, 
                decimals=DECIMALS,
            )
        # edge.type == 'Arc':
        return arc_line_segment_intersection(
            circle_center=self.center,
            circle_radius=self.r,
            pt1=edge.vtx1.xy,
            pt2=edge.vtx2.xy,
            angle_min=self.angle_min,
            angle_max=self.angle_max,
            full_line=False,
            tangent_tol=TANGENT_TOL,
            rescale=RESCALE_FACTOR, 
            decimals=DECIMALS,
        )
        
    def split(self, new_vtx):
        """Split at point if point is not start or end point of line segment"""
        if new_vtx.xy in self.coords:
            return [self]
        self.all_edges.pop(self.coords, None)
        line_kwargs = {'all_edges': self.all_edges, **{key: getattr(self, key) for key in ['contains', 'overlaps'] if hasattr(self, key)}}
        return [LineSegment(vtx1=self.vtx1, vtx2=new_vtx, **line_kwargs), LineSegment(vtx1=new_vtx, vtx2=self.vtx2, **line_kwargs)]
    #

    def calc_min_dist_to_pt(self, pt):
        """
        returns smallest distance between edge (a finite line segement) and pt
        """
        px = self.vtx2.x-self.vtx1.x
        py = self.vtx2.y-self.vtx1.y
        u =  ((pt[0] - self.vtx1.x) * px + (pt[1] - self.vtx1.y) * py) / float(px**2 + py**2)
        if u > 1:
            u = 1
        elif u < 0:
            u = 0
        return (((self.vtx1.x + u * px) - pt[0])**2 + ((self.vtx1.y + u * py) - pt[1])**2)**.5
    #

    def calc_max_dist_to_pt(self, pt):
        """
        returns largest distance between point and edge (=finite line segment)
        """
        return max([
            ((self.vtx1.x-pt[0])**2+(self.vtx1.y-pt[1])**2)**.5,
            ((self.vtx2.x-pt[0])**2+(self.vtx2.y-pt[1])**2)**.5
        ])
    #

    def transform_to_trgl(self, i:int):
        """
        
        """
        if i == 1: return self 
        
        new_vtx1 = Vertex(*transform_coord(self.vtx1.xy, i), all_vtx=self.all_vtx)
        new_vtx2 = Vertex(*transform_coord(self.vtx2.xy, i), all_vtx=self.all_vtx)
        
        edge_kwargs = {key: tuple(transform_cell(cell=_np_array(getattr(self, key)), i=i)) for key in ['contains', 'overlaps'] if hasattr(self, key)}
        if i%2 == 1: return LineSegment(vtx1=new_vtx1, vtx2=new_vtx2, all_edges=self.all_edges, **edge_kwargs)
        # flip order for regions 2,4,6,8
        return LineSegment(vtx1=new_vtx2, vtx2=new_vtx1, all_edges=self.all_edges, **edge_kwargs)
#


class Arc(Edge):
    """Arc around center (limited by two points on circle if supplied) in the current definition the edge of arc must be <180 degrees """
    def __init__(
            self,
            center:tuple,
            r:float,
            all_edges:dict,
            vtx1:Vertex=None,
            vtx2:Vertex=None,
            is_clockwise:bool=None,
            contains:tuple=None,
            overlaps:tuple=None,
            printme:str="",
            ):
        super().__init__(vtx1, vtx2, all_edges, contains=contains, overlaps=overlaps)
        self.type = "Arc"
        self.center = center
        self.r = r
        self.vtx1 = vtx1
        self.vtx2 = vtx2
        self.angle_vtx1, self.angle_vtx2  = angles_to_origin((vtx1.xy, vtx2.xy), center) if not None in [vtx1,vtx2] else (0,360)
        self.abs_angle_vtx1, self.abs_angle_vtx2 = (angle_to(center, vtx1.xy), angle_to(center, vtx2.xy)) if not None in [vtx1,vtx2] else (0,360)
        self.angle_min, self.angle_max = sorted((self.angle_vtx1, self.angle_vtx2))
        
        # to-do. if radius/spacing is ever smaller 1/(2**.5) then angles can possibly be above 180 degrees
        # if not is_clockwise is None and is_clockwise != ((self.angle_vtx2-self.angle_vtx1)%360>=180):
        #     print("conflict: "+printme, is_clockwise,"v1",self.angle_vtx1,"v2", self.angle_vtx2)
        self.is_clockwise = (self.angle_vtx2-self.angle_vtx1)%360>=180 if is_clockwise is None else is_clockwise
        
        if not self.is_clockwise:
            if self.abs_angle_vtx1 < self.abs_angle_vtx2:
                self.first_angle = self.abs_angle_vtx1
                self.second_angle = self.abs_angle_vtx2
            else:
                self.first_angle = self.abs_angle_vtx1
                self.second_angle = self.abs_angle_vtx2 + 360
        else:
            if self.abs_angle_vtx1 > self.abs_angle_vtx2:
                self.first_angle = self.abs_angle_vtx1
                self.second_angle = self.abs_angle_vtx2
            else:
                self.first_angle = self.abs_angle_vtx1 + 360
                self.second_angle = self.abs_angle_vtx2
        self.counter_clockwise_vtx1, self.counter_clockwise_vtx2 = (self.vtx1,self.vtx2) if not self.is_clockwise else (self.vtx2,self.vtx1)
    #

    def reverse(self):
        self.vtx1, self.vtx2 = self.vtx2, self.vtx1
        self.coords = (self.vtx1.xy, self.vtx2.xy)
        self.first_angle, self.second_angle = self.second_angle, self.first_angle
        self.angle_vtx1, self.angle_vtx2 = self.angle_vtx2, self.angle_vtx1
        self.abs_angle_vtx1, self.abs_angle_vtx2 = self.abs_angle_vtx2, self.abs_angle_vtx1
        self.is_clockwise = not self.is_clockwise
        self.plot_coords = None  # invalidate cache — n_steps is unchanged after swap so stale cache would pass the length check
        return self
    
    def intersection(
            self, 
            edge,
            TANGENT_TOL:float = 1e-16,
            RESCALE_FACTOR:int = 1,
            DECIMALS:int = 117,
            ):
        """
        returns a list of intersection coords with length 0, 1 or 2
        """

  
        if edge.type == 'Arc':
            return intersections_pts_arc_to_circle(
                circle_center=edge.center,
                arc_center=self.center,
                arc_angle_min=self.angle_min,
                arc_angle_max=self.angle_max,
                r=self.r,
                tangent_tol=TANGENT_TOL,
                rescale=RESCALE_FACTOR, decimals=DECIMALS,
            )
        elif edge.type == 'Circle':
            return intersections_pts_arc_to_circle(
                circle_center=edge.center,
                arc_center=self.center,
                arc_angle_min=self.angle_min,
                arc_angle_max=self.angle_max,
                r=self.r,
                tangent_tol=TANGENT_TOL,
                rescale=RESCALE_FACTOR, decimals=DECIMALS,
            )
        
        return arc_line_segment_intersection(
            circle_center=self.center,
            circle_radius=self.r,
            pt1=edge.vtx1.xy,
            pt2=edge.vtx2.xy,
            angle_min=self.angle_min,
            angle_max=self.angle_max,
            full_line=False,
            tangent_tol=TANGENT_TOL,
            rescale=RESCALE_FACTOR, decimals=DECIMALS,
        )
    
    def split(self, new_vtx):
        """Split at point if point is not start or end point of arc segment"""
        if new_vtx.xy in self.coords:
            return [self]
        self.all_edges.pop(self.coords, None)
        arc_kwargs = {
            'center': self.center,
            'r': self.r,
            'all_edges': self.all_edges,
            'is_clockwise': self.is_clockwise,
            **{key: getattr(self, key) for key in ['contains', 'overlaps'] if hasattr(self, key)}
            }

        return [
            Arc(vtx1=self.vtx1, vtx2=new_vtx, printme="split1",**arc_kwargs), 
            Arc(vtx1=new_vtx, vtx2=self.vtx2, printme="split2",**arc_kwargs)
            ]
    #
    
    def _angle_on_arc(self, theta):
        """
        Returns True if the arc point at absolute angle theta (degrees, CCW from +x axis)
        lies on this arc (as opposed to the complementary arc of the same circle).

        Uses the rotation-invariant span check:
          CCW arc: (theta - vtx1_angle) % 360 <= (vtx2_angle - vtx1_angle) % 360
          CW  arc: (vtx1_angle - theta) % 360 <= (vtx1_angle - vtx2_angle) % 360
        This handles wrap-around (0/360 boundary) without any special cases.
        """
        a1 = self.abs_angle_vtx1
        a2 = self.abs_angle_vtx2
        if not self.is_clockwise:
            return (theta - a1) % 360 <= (a2 - a1) % 360
        else:
            return (a1 - theta) % 360 <= (a1 - a2) % 360

    def calc_min_dist_to_pt(self, pt):
        """
        returns smallest distance between edge (an arc, not full circle) and pt.

        The closest point on the arc is in the radial direction from center toward pt
        (at distance |dist(center,pt) - r|) if that direction falls on the arc;
        otherwise the closest point is one of the two arc endpoints.
        """
        abs_angle_pt = angle_to(self.center, pt)
        if self._angle_on_arc(abs_angle_pt):
            dist_center_pt = ((pt[0]-self.center[0])**2 + (pt[1]-self.center[1])**2)**.5
            return abs(dist_center_pt - self.r)
        return min(
            ((self.vtx1.x-pt[0])**2+(self.vtx1.y-pt[1])**2)**.5,
            ((self.vtx2.x-pt[0])**2+(self.vtx2.y-pt[1])**2)**.5,
        )
    #

    def calc_max_dist_to_pt(self, pt):
        """
        Calculates the maximum distance to an arc from ANY query point.
        Safely utilizes the internal _angle_on_arc helper.
        """
        # 1. Base candidates: The two physical endpoints are always valid
        dist_vtx1 = ((self.vtx1.x - pt[0])**2 + (self.vtx1.y - pt[1])**2)**0.5
        dist_vtx2 = ((self.vtx2.x - pt[0])**2 + (self.vtx2.y - pt[1])**2)**0.5
        max_dist = max(dist_vtx1, dist_vtx2)
        
        # return max_dist # TODO remove this.

        # 2. Distance from the circle center to the query point
        dx = pt[0] - self.center[0]
        dy = pt[1] - self.center[1]
        dist_center_pt = (dx**2 + dy**2)**0.5
        
        # Quick edge case: If point is exactly on the center, 
        # all points on the arc are exactly radius 'r' away.
        if dist_center_pt == 0:
            return self.r
            
        # 3. Determine the query point's absolute angle
        abs_angle_pt = angle_to(self.center, pt)
        
        # 4. Branches based on point placement relative to the radius
        if dist_center_pt >= self.r:
            # POINT IS OUTSIDE OR ON THE CIRCLE:
            # The furthest point on a full circle is the antipodal angle
            antipodal_angle = (abs_angle_pt + 180) % 360
            
            if self._angle_on_arc(antipodal_angle):
                candidate_dist = dist_center_pt + self.r
                max_dist = max(max_dist, candidate_dist)
        else:
            # POINT IS INSIDE THE CIRCLE:
            # The furthest point on a full circle is still the antipodal (distance = r + d).
            antipodal_angle = (abs_angle_pt + 180) % 360
            if self._angle_on_arc(antipodal_angle):
                candidate_dist = dist_center_pt + self.r
                max_dist = max(max_dist, candidate_dist)
                
        return max_dist
    
    def transform_to_trgl(self, i:int):
        """
        
        """
        if i == 1: return self 
        new_center = transform_coord(self.center, i)
        new_vtx1 = Vertex(*transform_coord(self.vtx1.xy, i), all_vtx=self.all_vtx)
        new_vtx2 = Vertex(*transform_coord(self.vtx2.xy, i), all_vtx=self.all_vtx)
        # flip order for regions 2,4,6,8
        edge_kwargs = {key: tuple(transform_cell(cell=_np_array(getattr(self, key)), i=i)) for key in ['contains', 'overlaps'] if hasattr(self, key)}
        if i%2 == 1: return Arc(
            new_center,
            r=self.r,
            all_edges=self.all_edges,
            vtx1=new_vtx1,
            vtx2=new_vtx2,
            is_clockwise=self.is_clockwise,
            printme="turn1",
            **edge_kwargs
            )
        return Arc(
            new_center,
            r=self.r,
            all_edges=self.all_edges,
            vtx1=new_vtx2,
            vtx2=new_vtx1,
            is_clockwise=self.is_clockwise,
            printme="turn2",
            **edge_kwargs
            )
    #
    def get_plot_coords(self, arc_steps_per_degree:float=5):
        # return [self.vtx1.xy, self.vtx2.xy]
        # total_angle = abs(self.angle_max - self.angle_min)
        # n_steps = max(3,-int(-(total_angle * arc_steps_per_degree)))
        total_angle = abs(self.first_angle - self.second_angle)
        n_steps = max(3,-int(-(total_angle * arc_steps_per_degree)))
        
        if not self.plot_coords is None and len(self.plot_coords)==n_steps+1:
            return self.plot_coords
        
        # if abs(self.abs_angle_vtx2-self.abs_angle_vtx1) > abs(self.abs_angle_vtx2-(self.abs_angle_vtx1+360)):
        #     first_angle = 0
        #     second_angle = 0
        # elif abs(self.abs_angle_vtx2-self.abs_angle_vtx1) > abs(self.abs_angle_vtx2-(self.abs_angle_vtx1+360)):
        #     0
        # else:
        #     first_angle = self.abs_angle_vtx1
        #     second_angle = self.abs_angle_vtx2
        cx, cy = self.center
        r = self.r
        coords = []
        # for angle_step in _np_linspace(0, self.abs_angle_vtx2-self.abs_angle_vtx1, n_steps)[::(-1 if False else 1)]:
        for angle_step in _np_linspace(self.first_angle, self.second_angle, n_steps):
            x = cx + r * _math_cos((angle_step)/360*2*_math_pi)#rotation+
            y = cy + r * _math_sin((angle_step)/360*2*_math_pi)#rotation+
            coords.append((float(x),float(y)))
            
        # if counter-clockwise angle_vtx1<angle_vtx2
        # if clockwise angle_vtx1>angle_vtx2
        # total_angle = (self.angle_vtx1 - self.angle_vtx2)%360*-1 if self.is_clockwise else (self.angle_vtx2 - self.angle_vtx1)%360
        # if abs(total_angle)>=180: # TOOD this is a dirty fix. ensure this works
        #     print(self.is_clockwise, self.angle_vtx1, self.angle_vtx2, total_angle)
        #     total_angle = (self.angle_vtx1 - self.angle_vtx2)%360*-1 if not self.is_clockwise else (self.angle_vtx2 - self.angle_vtx1)%360
        # for angle_step in _np_linspace(0, total_angle, n_steps):
        #     x = cx + r * _math_cos((rotation+self.angle_vtx1+angle_step)/360*2*_math_pi)
        #     y = cy + r * _math_sin((rotation+self.angle_vtx1+angle_step)/360*2*_math_pi)
        #     coords.append((float(x),float(y)))
        # coords = [self.vtx1.xy,self.vtx2.xy] #if not self.is_clockwise else [self.vtx2.xy,self.vtx1.xy] 
        self.plot_coords = coords
        # if len(coords)<2:
        #     print("coords",coords)
        return coords
    #
#

class Circle(object):
    """TODO potentially add as subclass of Edge?"""
    def __init__(self, center, r):
        self.type = "Circle"
        self.center = center
        self.r = r
        self.angle_min, self.angle_max = (0,360)
    #
    def __repr__(self):
        props_not_to_print = ['angle_min', 'angle_max']
        return str({key: val if type(val) != float else round(val,5) for key, val in self.__dict__.items() if key not in props_not_to_print})
    #
    def plot_single(
            self, 
            ax=None, 
            facecolor="#00000011", 
            edgecolor="green", 
            linewidth=1,
            radial_lines=False,
            full_circle=None, 
            add_arrow=None, 
            **kwargs):
        if ax is None:
          fig, ax = plt.subplots()
        #
        ax.add_patch(_plt_Circle(xy=self.center, radius=self.r, facecolor=facecolor, edgecolor=edgecolor, linewidth=linewidth, **kwargs))
        return ax
    #
#

class OffsetRegion(object):
    """OffsetRegion bounded by line segments and arcs"""

    def __init__(self, edges, checks, all_regions:dict, trgl_nr:int=1, printme=""):
        self.id = -1
        self.edges = edges
        self.vertices = []
        self.trgl_nr = trgl_nr
        xs, ys = [], []
        self.is_closed = True
        for n,edge in enumerate(edges):
            if n>0 and self.is_closed and edge.vtx1.xy != edges[n-1].vtx2.xy:
                self.is_closed = False
            edge.regions.append(self)
            for vtx in (edge.vtx1, edge.vtx2):
                if vtx not in self.vertices:
                    self.vertices.append(vtx)
                    vtx.regions.append(self)
            # check if most extreme x/y values are not at the vertices but between
            if edge.type == 'Arc':
                cx, cy = edge.center
                if edge._angle_on_arc(0):   xs.append(cx + edge.r)
                if edge._angle_on_arc(90):  ys.append(cy + edge.r)
                if edge._angle_on_arc(180): xs.append(cx - edge.r)
                if edge._angle_on_arc(270): ys.append(cy - edge.r)
        # if len(xs):
        #     print("xs",xs,min([vtx.x for vtx in self.vertices]),max([vtx.x for vtx in self.vertices]))
        # if len(ys):
        #     print(angles_to_origin([(1,1),(1,0),(0,-1)],(0,0)))
        #     print("ys",ys, min([vtx.y for vtx in self.vertices]),max([vtx.y for vtx in self.vertices]))
        # To-Do: For edges that have an arc it   might be that a point between the vertices is 
        # the max/min x/y value. Probably only the case  
        xs = [vtx.x for vtx in self.vertices] + xs
        ys = [vtx.y for vtx in self.vertices] + ys
        if not xs or not ys:
            raise ValueError(
                f"OffsetRegion constructed with no vertices "
                f"(trgl_nr={trgl_nr}, edges={edges}). "
                "This indicates an empty edge list was passed to the constructor."
            )
        self.xmin = float(min(xs))
        self.xmax = float(max(xs))
        self.ymin = float(min(ys))
        self.ymax = float(max(ys))
        self.checks = checks
        self.coords = tuple([edge.coords for edge in edges])
        self.all_regions = all_regions
        self.all_edges = edges[-1].all_edges
        self.all_vtx = edges[-1].vtx1.all_vtx
        self.is_clockwise = 0<sum([(vtx2.x-vtx1.x)*(vtx2.y+vtx1.y) for vtx1,vtx2 in zip(self.vertices[:-1], self.vertices[1:])])
        # print([(e.vtx1.xy,e.vtx2.xy) for e in self.edges])
        self.centroid = calculate_polygon_centroid(self.get_plot_coords(arc_steps_per_degree=50))
        _normalize_region_winding(self, arc_steps_per_degree=50)
        self.coords = tuple([edge.coords for edge in self.edges])
        if self.coords not in all_regions:
            all_regions[self.coords] = self
        if not all([e0.vtx2.xy == e1.vtx1.xy for e0, e1 in zip(self.edges, self.edges[1:]+self.edges[:1])]):
            pass  # open/non-closed region — edge connectivity mismatch; skipped for non-closed regions
       
        self.is_closed = self.is_closed or edge.vtx1.xy == edges[n-1].vtx2.xy
    #
    
    @staticmethod
    def delete_all(all_regions):
        all_regions.clear()
    #

    def __repr__(self):
        props_not_to_print = ['all_regions', 'all_edges', 'all_vtx', 'plot_coords']
        rnd = 5
        return str({key: [{e.type:[(round(e.vtx1.x,rnd),round(e.vtx1.y,rnd)),(round(e.vtx2.x,rnd),round(e.vtx2.y,rnd))]} for e in val] if key in ['edges'] else (val if type(val) != float else round(val,rnd)) for key, val in self.__dict__.items() if key not in props_not_to_print})
    #

    def delete(self):
        self.all_regions.pop(self.coords, None)
    #

    def get_vertex_coords(self):
        coords = []
        for edge_start_coord, edge_end_coord in self.coords:
            if len(coords)==0 or edge_start_coord != coords[-1]:
                coords.append(edge_start_coord)
            if edge_end_coord != coords[-1]:
                coords.append(edge_end_coord)
            # if edge_start_coord not in coords:
            #     coords.append(edge_start_coord)
            # if edge_end_coord not in coords:
            #     coords.append(edge_end_coord)
        self.vertex_coords = coords
        return coords
    #

    def get_plot_coords(self, arc_steps_per_degree:float=5):
        coords = []
        for edge in self.edges:
            coords += edge.get_plot_coords(arc_steps_per_degree=arc_steps_per_degree)

        self.plot_coords = coords
        return coords
    #

    def calc_area(self, arc_steps_per_degree:float=100):
        # xs = []
        # ys = []
        # for ((x,y), edge_end_coord), edge in zip(self.coords, self.edges):
        #     xs.append(x)  
        #     ys.append(y)  
        #     if edge.type == 'Arc':
        #         total_angle = abs(edge.angle_max - edge.angle_min)
        #         n_steps = max(3,-int(-(total_angle * arc_steps_per_degree)))
        #         cx, cy = edge.center
        #         r = edge.r
        #         rotation = angle_to((0,0), edge.center)
        #         for angle_step in _np_linspace(edge.angle_min, edge.angle_max, n_steps):
        #             xs.append(cx + r * _math_cos((rotation+angle_step)/360*2*_math_pi))
        #             ys.append(cy + r * _math_sin((rotation+angle_step)/360*2*_math_pi))
        # xs =_np_array(xs)
        # ys =_np_array(ys)
        plot_coords = _np_array(self.get_plot_coords(arc_steps_per_degree=arc_steps_per_degree))
        xs = plot_coords[:,0]
        ys = plot_coords[:,1]
        self.area = 0.5*_np_abs(_np_dot(xs,_np_roll(ys,1))-_np_dot(ys,_np_roll(xs,1))) 
        return self.area

    
    def plot_single(
            self,
            facecolor='green',
            edgecolor='black',
            alpha=0.70, 
            plot_edges:dict={},
            plot_vertices:dict=False,
            add_idx_edges:dict=False,
            add_text:dict=False,
            add_arrow:str=None,
            add_centroid:dict={},
            full_circle:str=None,
            radial_lines:bool=False,
            arc_steps_per_degree:float=5,
            ax=None,
            figsize:tuple=None,
            x_lim:tuple=None,
            y_lim:tuple=None,
            **kwargs):
        
        if ax is None:
            fig, ax = plt.subplots(1,1,**({'figsize':figsize} if not figsize is None else {}))

        kwargs = {'facecolor': facecolor, 'alpha': alpha, 'edgecolor': edgecolor if not plot_edges else 'None',  **kwargs }
        
        if plot_edges==True: plot_edges = {}
        if plot_vertices==True: plot_vertices = {}
        if add_idx_edges==True: add_idx_edges = {}
        if add_text==True: add_text = {}
        if add_arrow==True: add_arrow = {}
        if add_centroid==True: add_centroid = {}


        ax.add_patch(_plt_Polygon(self.get_plot_coords(arc_steps_per_degree=arc_steps_per_degree),  **kwargs))
        if type(plot_edges) == dict:
            for edge in self.edges:
                edge.plot_single(ax=ax, radial_lines=radial_lines, edgecolor=edgecolor, full_circle=full_circle)
        if type(plot_vertices) == dict:
            Vertex.plot_many(vertices=[vtx.xy for vtx in self.vertices], ax=ax)
        if type(add_idx_edges) == dict:
            for i, edge in enumerate(self.edges):
                plot_coords = edge.get_plot_coords()
                ax.annotate(text=str(i), xy=(sum([x/len(plot_coords) for x,y in plot_coords]), sum([y/len(plot_coords) for x,y in plot_coords]))) 
        if type(add_arrow) == dict:
            for i, edge in enumerate(self.edges):
                edge.plot_single(ax=ax,edgecolor="None",add_arrow=add_arrow, radial_lines=False)
        if type(add_centroid) == dict:
                ax.scatter(
                    *self.centroid, **{
                        **{'marker':'*', 'facecolor':"black", 'edgecolor':'black', 's':100, 'alpha':alpha},
                        **add_centroid
                        })
        if type(add_text) == dict:
            ax.annotate(
                **{k:v if type(v)!=type(lambda x:x) else v(self) for k,v in {
                    **{
                        'text':lambda r: str(i if not hasattr(r, 'id') else r.id),
                        'xy': lambda r: r.centroid,
                        'horizontalalignment':'center',
                    },
                    **add_text
                }.items()}
            )
                
        padding_share = .1
        if x_lim is None:
            ax.set_xlim([self.xmin - padding_share*(self.xmax-self.xmin), self.xmax + padding_share*(self.xmax-self.xmin)])
        elif type(x_lim) in [list, tuple, _np_array]:
            ax.set_xlim(x_lim)
        if y_lim is None:
            ax.set_ylim([self.ymin - padding_share*(self.ymax-self.ymin), self.ymax + padding_share*(self.ymax-self.ymin)])
        elif type(x_lim) in [list, tuple, _np_array]:
            ax.set_ylim(y_lim)
        return ax
    #

    @staticmethod
    def plot_many(
        regions:list, 
        plot_edges:bool=False, 
        plot_vertices:bool=False, 
        add_idxs:dict=False, 
        add_arrows:dict=False,
        x_lim:tuple=(-0.02,0.52), 
        y_lim:tuple=(-0.02,0.52),
        arc_steps_per_degree:float=5,
        add_centroids={},
        cmap='tab20', 
        alpha=0.3, 
        title:str="", 
        ax=None, 
        figsize:tuple=None,
        **kwargs):

        if plot_edges==True: plot_edges = {}
        if plot_vertices==True: plot_vertices = {}
        if add_idxs==True: add_idxs = {}
        if add_arrows==True: add_arrows = {}
        if add_centroids==True: add_centroids = {}

        if ax is None:
            fig, ax = plt.subplots(1,1,**({'figsize':figsize} if not figsize is None else {}))
        
        if len(regions) > 0:
            # get color for region
            color_keys = ['c', 'color', 'facecolor']
            if not any([key in color_keys for key in kwargs]):
                my_cmap = _plt_get_cmap(cmap)
                colors = [my_cmap(x) for x in _np_linspace(0,1,len(regions))]
                color_key = 'facecolor'
            else:
                color_key = color_keys[next((i for i,ck in enumerate(color_keys) if ck in kwargs),-1)]
                if color_key in kwargs:
                    if type(kwargs[color_key]) == list:
                        while len(kwargs[color_key]) < len(regions):
                            kwargs[color_key] = kwargs[color_key]+kwargs[color_key][:len(regions)-len(kwargs[color_key])]
                    else:
                        kwargs[color_key] = [kwargs[color_key] for r in regions]
                colors = kwargs[color_key]
                kwargs.pop('c',None)
                kwargs.pop('color',None)
                kwargs.pop('facecolor',None)
            
            for reg,color in zip(regions,colors):
                reg.plot_single(
                    ax=ax,
                    plot_edges=plot_edges,
                    plot_vertices=plot_vertices,
                    add_idx_edges=False,
                    add_text=add_idxs,
                    x_lim=False,
                    y_lim=False,
                    facecolor=color,
                    add_arrow=add_arrows,
                    add_centroid={**{'s':min(100,max(0,100/len(regions)))},**add_centroids} if type(add_centroids) == dict else add_centroids,
                    full_circle=False,
                    radial_lines=False,
                    arc_steps_per_degree=arc_steps_per_degree,
                    **{**{'alpha':alpha, 'edgecolor':None},**kwargs}
                )
            #
        #
        ax.set_aspect('equal')
        padding_share = .1
        if x_lim is None and len(regions)>0:
            xmin = min([r.xmin for r in regions])
            xmax = max([r.xmax for r in regions])
            ax.set_xlim([xmin - padding_share*(xmax-xmin), xmax + padding_share*(xmax-xmin)])
        elif type(x_lim) in [list, tuple, _np_array]:
            ax.set_xlim(x_lim)
        if y_lim is None and len(regions)>0:
            ymin = min([r.ymin for r in regions])
            ymax = max([r.ymax for r in regions])
            ax.set_ylim([ymin - padding_share*(ymax-ymin), ymax + padding_share*(ymax-ymin)])
        elif type(x_lim) in [list, tuple, _np_array]:
            ax.set_ylim(y_lim)
        ax.set_title(title or (str(len(regions)) + " regions."))
        return ax

    #
    def get_split_pts(
            self,
            intersection_edge,
            TANGENT_TOL:float = 1e-16,
            MATCH_PTS_TOL:float = 1e-15,
            RESCALE_FACTOR:int = 1,
            DECIMALS:int = 117,
            ):
        splits, vertices_intersected, edges_of_new_vtx = [], [], []
        shared_along_vert = any([edge.vtx1.y==0 and edge.vtx2.y==0  for edge in self.edges]) # always false if there is a linecheck on x-axis
        shared_along_diag = any([edge.vtx1.x==edge.vtx1.y and edge.vtx2.x==edge.vtx2.y for edge in self.edges])
        # print("shared_along_vert",shared_along_vert,"shared_along_diag", shared_along_diag,  [v.xy for v in self.vertices])
        edges_to_skip = {}
        for i, edge in enumerate(self.edges):
            # print(i,"splits",splits)
            xy_adjusted, (x,y) = (None,None), (None,None)
            if i not in edges_to_skip:
                # print("don skip",xy_adjusted)
                edge_itxs = edge.intersection(
                    intersection_edge,
                    TANGENT_TOL=TANGENT_TOL,
                    RESCALE_FACTOR=RESCALE_FACTOR,
                    DECIMALS=DECIMALS,
                    ) # to do here we maybe need to control for the rounding errors
                # print("edge_itxs",edge_itxs)
                # match edge_itxs points to vertices if within precision limit
                
                vertices_to_match = ([edge.vtx1.xy, edge.vtx2.xy])#+vertices_intersected)
                edge_itxs_adj = []
                itx_adj_type = []
                edges_links = []
                itxs_ids = []
                for (x,y) in edge_itxs:
                    # plot_coords = self.get_plot_coords(20)
                    # if plot_coords[0]!=plot_coords[-1]:
                    #    print("plot_coords",plot_coords)
                       
                    #    plot_coords.append(plot_coords[0])
                    #    print("UEQUAL")  
                    # poly = _shapely_Polygon(plot_coords)
                    # print("poly.is_valid",poly.is_valid)
                    # for v in self.vertices:
                    #     pass
                        # print("|V|",v.xy, ((v.x-x)**2+(v.y-y)**2)**.5)
                    # if not poly.buffer(0.001*100).contains(_shapely_Point(x,y)):
                    #     fig,ax=_plt_subplots()
                    #     plot_polygon(_shapely_Polygon(self.get_plot_coords(20)).buffer(0.001*100), ax=ax, color='red')
                    #     plot_polygon(_shapely_Polygon(self.get_plot_coords(20)), ax=ax, color='pink')
                    #     # ax.scatter(x,y,
                    #     #        marker='*',
                    #     #        facecolor='black',
                    #     #        edgecolor='black',
                    #     #        alpha=0.4,
                    #     #        linewidths=3,
                    #     #        s=800,)
                        
                    #     ax.set_xlim([min(self.xmin,x)-.05,max(self.xmax,x)+.05])
                    #     ax.set_ylim([min(self.ymin,y)-.05,max(self.ymax,y)+.05])

                    #     ax=self.plot_single()
                    #     ax.scatter(x,y,
                    #            marker='*',
                    #            facecolor='red',
                    #            edgecolor='black',
                    #            alpha=0.4,
                    #            linewidths=3,
                    #            s=800,)
                        
                    #     ax.set_xlim([min(self.xmin,x)-.02,max(self.xmax,x)+.02])
                    #     ax.set_ylim([min(self.ymin,y)-.02,max(self.ymax,y)+.02])
                    #     ax.set_title('itx_out'+str((x,y)))
                        
                    #     raise ValueError('itx point outside region')
                    # print("e",xy_adjusted, (x,y))
                    edges_links_itx = [edge]
                    itx_ids = [i]
                    adjust_type = ""

                    # ensure that coordinates on x-y axis are rounded onto it (y axis doesnt matter for triangle regions though) 
                    xy_adjusted = (
                            0.0 if x<MATCH_PTS_TOL else 0.5 if .5-MATCH_PTS_TOL<x else x,
                            0.0 if y<MATCH_PTS_TOL else 0.5 if .5-MATCH_PTS_TOL<y else y
                            )
                    if (x,y) != xy_adjusted:
                        adjust_type = 'ze'+str(int((x,y)!=xy_adjusted))
                    
                    # if any existing vertex is close enough: match it towards it if within precision limit. Otherwise return original point
                    previous_instances_of_coord = sorted([
                        (((xy_adjusted[0]-vtx_x)**2+(xy_adjusted[1]-vtx_y)**2)**.5, (vtx_x,vtx_y)) 
                        for vtx_x,vtx_y in vertices_to_match
                        if ((xy_adjusted[0]-vtx_x)**2+(xy_adjusted[1]-vtx_y)**2)**.5 < MATCH_PTS_TOL
                        ]) # DON'T DO THIS MATCH IF INTERSECTED EDGE IS ANOTHER ONE 
                    if len(previous_instances_of_coord)>0:
                        xy_adjusted = previous_instances_of_coord[0][1] # take the closest one
                        adjust_type += 'p_i'+str(int(xy_adjusted in [edge.vtx1.xy, edge.vtx2.xy]))+"-"+str(int((x,y)==xy_adjusted))
                        xy_adjusted = (
                            0.0 if xy_adjusted[0]<MATCH_PTS_TOL else 0.5 if .5-MATCH_PTS_TOL<xy_adjusted[0] else xy_adjusted[0],
                            0.0 if xy_adjusted[1]<MATCH_PTS_TOL else 0.5 if .5-MATCH_PTS_TOL<xy_adjusted[1] else xy_adjusted[1]
                            )
                    else:
                        
                        
                        # ensure to match coordinates on the 45 deg line onto it
                        # to have the consistency the distance to the 45 degree line is calculated.
                        if abs(x-y)/(2**.5) < MATCH_TO_45_TOL:
                            # print("abs(x-y)/(2**.5)",abs(x-y)/(2**.5))
                            # check if any v from x,y is included 
                            x_or_ys_to_match = [vx for vx,vy in vertices_to_match if x==vx] + [vy for vx,vy in vertices_to_match if y==vy]
                            if len(x_or_ys_to_match)>0:# TODO PROBLEM HERE
                                # print("match this",len(x_or_ys_to_match), x_or_ys_to_match)
                                xy_adjusted = (sum(x_or_ys_to_match)/len(x_or_ys_to_match), sum(x_or_ys_to_match)/len(x_or_ys_to_match))
                                adjust_type += ('45d'+
                                                        str(int(xy_adjusted[0] in [vx for vx,vy in vertices_to_match]))+
                                                        str(int(xy_adjusted[0] in [vy for vx,vy in vertices_to_match]))
                                                        )
                            else:
                                # print("prev",xy_adjusted)
                                xy_adjusted = (sum(xy_adjusted)/2, sum(xy_adjusted)/2)
                                adjust_type += '45d_av'
                                # print("post",xy_adjusted)
                        else:
                            adjust_type += 'none'
                        #
                    #
                    xy_adjusted = (
                            0.0 if xy_adjusted[0]<MATCH_PTS_TOL else 0.5 if .5-MATCH_PTS_TOL<xy_adjusted[0] else xy_adjusted[0],
                            0.0 if xy_adjusted[1]<MATCH_PTS_TOL else 0.5 if .5-MATCH_PTS_TOL<xy_adjusted[1] else xy_adjusted[1]
                            )
                    j = (i-1)%len(self.edges) if (xy_adjusted == edge.vtx1.xy and i==0) else (i+1)%len(self.edges)
                    if (xy_adjusted == edge.vtx1.xy and i==0) or (xy_adjusted == edge.vtx2.xy and i != len(self.edges)-1):
                        # jth edge (that hasn't been tried for split points) will be skipped and marked as also be touching the vertex
                        edges_to_skip[j] = xy_adjusted
                    if (xy_adjusted == edge.vtx1.xy and i!=0) or (xy_adjusted == edge.vtx2.xy and i == len(self.edges)-1):
                        # jth edge (that has already been tried) 
                        # if close enough itx pt is found then this point is replaced with current.
                        # otherwise the other edge will be added 
                        # ensure that previous edge is touching the vertex
                        found_prev_itx = False
                        for split in splits:
                            if 'ids' not in split:
                                continue
                            if not j in split['ids']:
                                continue
                            # check whether current point is close enough to that point
                            if ((x-split['xy'][0])**2+(y-split['xy'][1])**2)**.5<MATCH_PTS_TOL*1:
                                split['xy'] = xy_adjusted
                                found_prev_itx = True
                                break
                        if not found_prev_itx:
                            adjust_type += '+'
                            reverse = (1 if xy_adjusted == edge.vtx1.xy else -1)
                            edges_links_itx = ([self.edges[j]] + edges_links_itx)[::reverse]
                            itx_ids = ([j] + itx_ids)[::reverse]
                    

                    edge_itxs_adj.append(xy_adjusted)
                    itx_adj_type.append(adjust_type)
                    edges_links.append(edges_links_itx)
                    itxs_ids.append(itx_ids)
                    
                    edges_of_new_vtx.append(edge)
                    # if type(intersection_edge) == LineSegment:
                    #     print("adj_t", adjust_type, (x,y) == xy_adjusted, (x,y), xy_adjusted)
                    # TODO SOMEWHERE THE xy_adjusted gets passed an inplausible value ' TODO SEARCH FOR WHERE
                    # if not _shapely_Polygon(self.get_plot_coords(20)).buffer(0.001).contains(_shapely_Point(*xy_adjusted)):
                    #     ax=self.plot_single()
                    #     ax.scatter(*xy_adjusted,
                    #             marker='*',
                    #             facecolor='red',
                    #             edgecolor='black',
                    #             alpha=0.4,
                    #             linewidths=3,
                    #             s=800,)
                    #     ax.set_xlim([min(self.xmin,xy_adjusted[0])-.02,max(self.xmax,xy_adjusted[0])+.02])
                    #     ax.set_ylim([min(self.ymin,xy_adjusted[1])-.02,max(self.ymax,xy_adjusted[1])+.02])
                    #     ax.set_title('itx_out'+str(xy_adjusted))
                    #     raise ValueError('itx pt outs reg', itx_adj_type, xy_adjusted, xy_adjusted in [v.xy for v in self.vertices])
                #
            #
            else: #
                xy_adjusted = edges_to_skip[i]
                edge_itxs_adj = [xy_adjusted]
                adjust_type = 'copied'
                itx_adj_type = [adjust_type]
                edges_links.append([edge])
                itxs_ids.append([i])
            
            # check if itx is in edge
            if len(edge_itxs_adj) == 0:
                continue 
            # print('edge_itxs_adj',edge_itxs_adj)
            # else:
            #     if itx_adj_type[0] != 'none':
            #         print("adj", itx_adj_type,"|", xy_adjusted==(x,y),"|", xy_adjusted, (x,y))
            #

            if len(edge_itxs_adj) == 1:
                edge_itx = edge_itxs_adj[0]
                edges_links_itx = edges_links[0]
                itx_ids = itxs_ids[0]
                # TODO check if vertex already exists: then its clear
                # new_vtx = Vertex(*itx[0])
                if not edge_itx in vertices_intersected:
                    splits.append({
                        'i': i, 
                        'xy': edge_itx, 
                        'edge': edge, 
                        'edges': edges_links_itx, 
                        'touches': edge_itx in edge.coords, 
                        'ids': itx_ids, 
                        'adj_t':itx_adj_type}
                        )# [edge1, itx[0], edge2], 
                else:
                    for split in splits:
                        if 'xy' not in split or 'ids' not in split:
                            continue
                        if split['xy'] == edge_itx:
                            if (i + 1)%len(self.edges) in split['ids']:
                                split['ids'] = [i]+split['ids']
                                split['edges'] = [edge] + split['edges']
                                split['adj_t'] = itx_adj_type + split['adj_t']
                            else:
                                split['ids'].append(i)
                                split['edges'].append(edge)
                                split['adj_t'] = split['adj_t'] + itx_adj_type
                            break
                # if not edge_itx in edge.coords:
                vertices_intersected.append(edge_itx)
                #
                continue
            #

            if len(edge_itxs_adj) == 2:
                if edge_itxs_adj[0] == edge_itxs_adj[1]:
                    # Degenerate: both intersection points coincide — treat as single.
                    edge_itx = edge_itxs_adj[0]
                    if edge_itx not in vertices_intersected:
                        splits.append({
                            'i': i,
                            'xy': edge_itx,
                            'edge': edge,
                            'edges': edges_links[0] if edges_links else [edge],
                            'touches': edge_itx in edge.coords,
                            'ids': itxs_ids[0] if itxs_ids else [i],
                            'adj_t': itx_adj_type[:1],
                        })
                    vertices_intersected.append(edge_itx)
                else:
                    splits.append({'i': i, 'pts': edge_itxs_adj, 'edge': edge, 'touches': any(p in edge.coords for p in edge_itxs_adj)})
                continue
            #

            raise ValueError("TOO MANY (",len(edge_itxs_adj),") INTERSECTIONS!", edge_itxs_adj)
        #
        return splits, vertices_intersected
    #


    
    def add_check_result_no_intersection(self, intersection_edge, check):
        """
        
        """

        # mean_x = sum([c[0][0] for c in self.coords]) / len(self.coords)
        # mean_y = sum([c[0][1] for c in self.coords]) / len(self.coords)
        fallback_pt_in_self = ((self.vertices[0].x+self.vertices[1].x)/2, (self.vertices[0].y+self.vertices[1].y)/2)
        fallback_pt_in_self = self.centroid
        vtx_not_on_intersection_edge = next(
            (coord for coord in [startcoord for startcoord, endcoord in self.coords] if (
            hasattr(intersection_edge, 'vtx1') and coord != intersection_edge.vtx1.xy and coord != intersection_edge.vtx2.xy)),
            fallback_pt_in_self)
        if intersection_edge.type == 'LineSegment':
            # TODO this must store better on which side of the line implies check success
            result = pt_is_left_of_vector(*self.centroid, *intersection_edge.vtx1.xy, *intersection_edge.vtx2.xy)
        else:
            x,y=vtx_not_on_intersection_edge
            # TODO as polygon is not necessarily convex centroid may lay outide, s.t this check may produce wrong results.  
            # BETTER APPROACH. Chose a point that is not on the current intersection edge. 
            # print('r', intersection_edge.r, ((mean_x-intersection_edge.center[0])**2 + (mean_y-intersection_edge.center[1])**2)**.5)
            result = intersection_edge.r**2 > ((x - intersection_edge.center[0])**2 + (y - intersection_edge.center[1])**2)
        self.checks.append({**check, 'result': result})
    #

    def split_with_edge(
            self,
            intersection_edge,
            check,
            plot_split:bool=False,
            r=0, 
            TANGENT_TOL:float = 1e-16,
            MATCH_PTS_TOL:float = 1e-15,
            RESCALE_FACTOR:int = 1,
            DECIMALS:int = 117,
        ):
        """
        check if intersected
        split edges
        add vertices
        split region 
        TODO rework this logic. Too convoluted!
        TODO THIS FUNCTION IS WAY TOO LONG
        """
        splits, vertices_intersected = self.get_split_pts(
            intersection_edge=intersection_edge,
            TANGENT_TOL=TANGENT_TOL,
            MATCH_PTS_TOL=MATCH_PTS_TOL,
            RESCALE_FACTOR=RESCALE_FACTOR,
            DECIMALS=DECIMALS,
            )
        # if any([e1.vtx2.xy != e2.vtx1.xy for e1,e2 in zip(self.edges,self.edges[1:]+self.edges[:1])]):
        #     ax = self.plot_single()
        #     intersection_edge.plot_single(ax=ax, edgecolor="black")
        #     for pt in splits:
        #         print("pt",pt)
        #         ax.add_patch(_plt_Circle(xy=pt['xy'], radius=0.02, facecolor='red' if pt['touches'] else "#d40dc4",alpha=0.8))
        #         ax.annotate(
        #                     text=('L' if pt['edge'].type=='LineSegment' else 'A'), xy=pt['xy'], #("T" if pt['touches'] else "F") + '-' + 
        #                     horizontalalignment='center', fontsize=10, color="black", weight="bold"
        #             )
        #     ax.set_title(str(len(splits))+":"+str([tuple([round(c,4) for c in pt['xy']]) for pt in splits]))
        vertices_intersected, split_pts = [], []
        
        for split in splits:
            if 'xy' in split:
                if not split['xy'] in vertices_intersected:
                    split_pts.append(split)
                vertices_intersected.append(split['xy'])
            else:
                split_pts.append(split)
        
        # splits=split_pts
        # print("Length after filter", len(splits), splits)
        if len(splits) == 0:
            self.add_check_result_no_intersection(intersection_edge=intersection_edge, check=check)
            return 'green' if self.checks[-1]['result'] else 'red'
        
        if len(splits) > 2:
            # increase the tolerance within which pts are matched onto another until only 2 or less points are left
            return self.split_with_edge(
                intersection_edge=intersection_edge,
                check=check,
                plot_split=plot_split,
                r=r, 
                TANGENT_TOL=TANGENT_TOL*1,
                MATCH_PTS_TOL=MATCH_PTS_TOL*2,
                RESCALE_FACTOR=RESCALE_FACTOR*1,
                DECIMALS=DECIMALS+10,
            )   
            ax = self.plot_single(
                plot_edges=False, plot_vertices=True, add_idx_edges=True,add_arrow=True, 
                x_lim=None, y_lim=None
                )
            intersection_edge.plot_single(ax=ax, edgecolor='black')
            vertex_coords = self.get_vertex_coords()
            print(self)
            # if intersection_edge.type == 'Arc':
            print('intersection_edge:',intersection_edge.r, intersection_edge.center)
            ax.set_title("")
            markers = ['P','X','*','o']
            colors = ['red', 'orange', 'blue', '#cc22ee']
            
            for split in splits:
                marker = markers[0]
                color = colors[0]
                markers = markers[1:]+markers[:1]
                colors = colors[1:]+colors[:1]
                for edge in split['edges']:
                    edge.plot_single(ax=ax, edgecolor=color, linewidth=5, alpha=0.5, radial_lines=False)
                ax.scatter(*split['xy'],
                               marker=marker,
                               facecolor=color,
                               edgecolor='black',
                               alpha=0.4,
                               linewidths=3,
                               s=800)
                print(split['xy'], 
                      "adj_t",split['adj_t'], 
                      "ids",split['ids'], 
                      "npts",len(splits),
                      'ne',len(split['edges']),
                      'split=vtx:', any([vtx.xy==split['xy'] for vtx in self.vertices]), 
                      [{e.type:(e.vtx1.xy,e.vtx2.xy)} for e in split['edges']])
            print(
                "p dis", 
                [[((sj['xy'][0]-si['xy'][0])**2+(sj['xy'][1]-si['xy'][1])**2)**.5 for j,sj in enumerate(splits) if i<j] for i,si in enumerate(splits)])
            raise ValueError("TODO IMPLEMENT MULTIPLE INTERSECTIONS", splits, intersection_edge.center)
        
        line_kwargs = {
            'all_edges': self.all_edges, 
            **{key: getattr(intersection_edge, key) for key in ['contains', 'overlaps'] if hasattr(intersection_edge, key)}
            }
        if type(intersection_edge) != LineSegment:
            arc_kwargs = {'center': intersection_edge.center, 'r': intersection_edge.r, **line_kwargs}

        if len(splits) == 1:
            # print("------SPLIT SINGLE",splits[0])
            # print("intersection_edge1",intersection_edge)
            # print("'edges to split",[(e.vtx1.xy,e.vtx2.xy) for e in splits[0]['edges']])
            # print("region to split",self.is_closed, [v.xy for v in self.vertices])
            if not 'pts' in splits[0]:
                # this means its only touching. thus return no intersection.
                # print("NO INTERSECTION")
                self.add_check_result_no_intersection(intersection_edge=intersection_edge, check=check)
                return 'green' if self.checks[-1]['result'] else 'red'
            
            if len(splits[0]['pts']) != 2:
                raise ValueError("Unexpected number of itx", splits)
            print("OPTION A")
            start_pt, end_pt = splits[0]['pts']
            old_edge = splits[0]['edge']
            old_edge.delete()
            pos = splits[0]['i']
            start_vtx_existed = start_pt not in self.all_vtx
            end_vtx_existed = end_pt not in self.all_vtx

            start_new_vtx = Vertex(*start_pt, self.all_vtx) if start_vtx_existed else self.all_vtx[start_pt]
            end_new_vtx = Vertex(*end_pt, self.all_vtx) if end_vtx_existed else self.all_vtx[end_pt] 
            
            # if  type(intersection_edge) == Arc:
            if type(intersection_edge) != LineSegment:
                new_edge_start =               [Arc(vtx1=old_edge.vtx1, vtx2=start_new_vtx, is_clockwise=old_edge.is_clockwise, **arc_kwargs)] if start_vtx_existed else []
                new_edge_end =                 [Arc(vtx1=end_new_vtx,   vtx2=old_edge.vtx2, is_clockwise=old_edge.is_clockwise, **arc_kwargs)] if end_vtx_existed else []
                new_edge_middle =              [Arc(vtx1=start_new_vtx, vtx2=end_new_vtx, is_clockwise=old_edge.is_clockwise,   **arc_kwargs)]
                new_edge_intersection =         Arc(vtx1=start_new_vtx, vtx2=end_new_vtx, is_clockwise=None,   **arc_kwargs)#is_clockwise=True or False
                new_edge_intersection_reverse = Arc(vtx1=end_new_vtx,   vtx2=start_new_vtx, is_clockwise=not new_edge_intersection.is_clockwise, **arc_kwargs)
                print("new old is_clockwise",new_edge_intersection.is_clockwise, new_edge_intersection_reverse.is_clockwise)
            else:
                new_edge_start =               [LineSegment(vtx1=old_edge.vtx1, vtx2=start_new_vtx, **line_kwargs)] if start_vtx_existed else []
                new_edge_end =                 [LineSegment(vtx1=end_new_vtx,   vtx2=old_edge.vtx2, **line_kwargs)] if end_vtx_existed else []
                new_edge_middle =              [LineSegment(vtx1=end_new_vtx,   vtx2=old_edge.vtx2, **line_kwargs)]
                new_edge_intersection =         LineSegment(vtx1=start_new_vtx, vtx2=end_new_vtx,   **line_kwargs)
                new_edge_intersection_reverse = LineSegment(vtx1=end_new_vtx,   vtx2=start_new_vtx, **line_kwargs)
            
            if type(intersection_edge) != LineSegment:
                first_region_is_within_radius = pt_is_left_of_vector(*first_region_edge.center, *first_region_edge.vtx1.xy, *first_region_edge.vtx2.xy)
                mid_angle = (
                    (first_region_edge.angle_vtx1+first_region_edge.angle_vtx2)/2 
                    if first_region_edge.angle_vtx1<first_region_edge.angle_vtx2 else
                    ((first_region_edge.angle_vtx1+first_region_edge.angle_vtx2+360))%360
                )
                    # if not self.is_clockwise (first_region_edge.angle_vtx1+first_region_edge.angle_vtx2)/2
                
                cx, cy = first_region_edge.center
                r = first_region_edge.r
                rotation = angle_to((0,0), first_region_edge.center)
                mp.precision = 16
                x_to_check_reg1 = cx + (r-0.02) * float(mp.cos((mid_angle+rotation)/360*2*mp.pi))#rotation+
                y_to_check_reg1 = cy + (r-0.02) * float(mp.sin((mid_angle+rotation)/360*2*mp.pi))#rotation+
            else:
                x_to_check_reg1 = sum([e.vtx1.x for e in edges_reg1])/len(edges_reg1)
                y_to_check_reg1 = sum([e.vtx1.y for e in edges_reg1])/len(edges_reg1)
            first_region_is_within_radius = pt_is_left_of_vector(
                x_to_check_reg1,
                y_to_check_reg1,
                *first_region_edge.vtx1.xy, *first_region_edge.vtx2.xy
            )
            # region without intersection_edge
            reg1 = OffsetRegion(
                edges = self.edges[:pos] + new_edge_start + [new_edge_intersection] + new_edge_end + self.edges[pos+1:], 
                checks = self.checks+[{**check, 'result': first_region_is_within_radius}], all_regions = self.all_regions, printme="without_itx1"
            ) 
            # new intersection region 
            reg2 = OffsetRegion(
                edges = new_edge_middle + [new_edge_intersection_reverse], 
                checks = self.checks+[{**check, 'result': not first_region_is_within_radius}], all_regions = self.all_regions, printme="with_itx1"
            )

            print("PLOT THIS")
            fig, axs = plt.subplots(1,3, figsize=(12,3))
            for i in [0,1,2]:
                ax = axs.flat[i]
                coords = self.get_plot_coords()
                x_coords = [x for x,y in coords]
                y_coords = [y for x,y in coords]
                x_min, x_max = min(x_coords), max(x_coords)
                y_min, y_max = min(y_coords), max(y_coords)
                x_dist, y_dist = x_max-x_min, y_max-y_min 
                self.plot_single(
                    ax=ax, facecolor='#ccc', plot_edges=False, x_lim=[x_min-x_dist/10, x_max+x_dist/10], y_lim=[y_min-y_dist/10, y_max+y_dist/10], 
                    add_arrow=i==0,
                    plot_vertices=i==0, add_idx_edges=i==0)
                if i == 0:
                    first_region_edge.plot_single(ax=ax, radial_lines=False, linewidth=4, add_arrow='blue')
                    edge_start_before[0].plot_single(ax=ax, radial_lines=False, linewidth=5, linestyle='dashed', edgecolor="#ff0000")
                    if len(edge_start_after): edge_start_after[0].plot_single(ax=ax, radial_lines=False, linewidth=5, linestyle='dashed', edgecolor="#db07f7")
                    edge_end_before[0].plot_single(ax=ax, radial_lines=False, linewidth=5, linestyle='dashed', edgecolor="#28c986")
                    if len(edge_end_after): edge_end_after[0].plot_single(ax=ax, radial_lines=False, linewidth=5, linestyle='dashed', edgecolor="#91ff00")
                if i == 1:
                    reg1.plot_single(ax=ax, facecolor='green' if first_region_is_within_radius else 'red', alpha=0.5, plot_edges=False, x_lim=None, y_lim=None, plot_vertices=True, add_idx_edges=True,add_arrow=True)
                if i == 2:
                    reg2.plot_single(ax=ax, facecolor='red' if first_region_is_within_radius else 'green', alpha=0.5, plot_edges=False, x_lim=None, y_lim=None, plot_vertices=True, add_idx_edges=True,add_arrow=True)
                if type(intersection_edge) != LineSegment:
                    ax.add_patch(_plt_Circle(xy=intersection_edge.center, radius=intersection_edge.r, edgecolor='black', facecolor='None'))
                else:
                    if intersection_edge.vtx1.x == intersection_edge.vtx2.x:
                        ymin, ymax = sorted([intersection_edge.vtx1.y, intersection_edge.vtx2.y])
                        ax.vlines(x=intersection_edge.vtx1.x, ymin=ymin, ymax=ymax, color='black',linewidth=1)
                    else:
                        xmin, xmax = sorted([intersection_edge.vtx1.x, intersection_edge.vtx2.x])
                        ax.hlines(y=intersection_edge.vtx1.y, xmin=xmin, xmax=xmax, color='black',linewidth=1)

                if i == 0:
                    ax.set_title("Ts"+str(int(start['touches']))+"e:"+str(int(end['touches']))+ 's_i:'+str(start_edge_i)+'e_i:'+str(end_edge_i)+
                                    'n_es'+str(len(self.edges))+'+'+str(not_include_start_edge_in_without_itx) +"-"+ 
                                    str(int(first_region_edge.vtx2.xy == self.edges[start_edge_i].vtx1.xy)) +
                                    str(int(first_region_edge.vtx1.xy == self.edges[start_edge_i].vtx1.xy)) +
                                    str(int(first_region_edge.vtx2.xy == self.edges[start_edge_i].vtx2.xy)) +
                                    str(int(first_region_edge.vtx1.xy == self.edges[start_edge_i].vtx2.xy))
                                    )
                elif i == 1:
                    ax.set_title(
                        str(start['ids'])+str(end['ids'])+
                        "n_v:"+str(len(reg1.vertices)) + 
                        'n_e'+str(len(reg1.edges)) +
                        str(reg1.is_closed) +
                        "")
                        
                elif i == 2:
                    ax.set_title(
                        "n_v:"+str(len(reg2.vertices))+ 
                        'n_e'+str(len(reg2.edges))+
                        str(reg2.is_closed) +
                        'ef'+str(exclude_first_in_itx)+
                        "")
                ax.scatter(x_to_check_reg1,y_to_check_reg1,
                            marker='P'if first_region_is_within_radius else 'X',
                            facecolor="#a6ff02" if first_region_is_within_radius else "#ff0278",
                            edgecolor='black',
                            s=100)
                ax.scatter(*start['xy'], marker='o',facecolor="black",edgecolor='black', s=100)
                ax.scatter(*end['xy'],marker='^',facecolor="black",edgecolor='black', s=100)
                ax.scatter(x=[e.vtx1.x for e in self.edges], y=[e.vtx1.y for e in self.edges], 
                            marker='X',facecolor="#d400ff",edgecolor="#09ff00", s=200)
                ax.scatter(x=[e.vtx2.x for e in self.edges], y=[e.vtx2.y for e in self.edges], 
                            marker='X',facecolor="#d400ff",edgecolor="#09ff00", s=200)
                ax.scatter(x=[v.x for v in self.vertices], y=[v.y for v in self.vertices], 
                            marker='X',facecolor="#d400ff",edgecolor="#09ff00", s=200)
                ax.set_xlim([min([x_to_check_reg1,self.xmin-0.01]),max([x_to_check_reg1, self.xmax+0.01])])
                ax.set_ylim([min([y_to_check_reg1,self.ymin-0.01]),max([y_to_check_reg1, self.ymax+0.01])])
                ax.set_xlim([self.xmin-(self.xmax-self.xmin)/20-.18,self.xmax+(self.xmax-self.xmin)/20+.013])
                ax.set_ylim([self.ymin-(self.ymax-self.ymin)/20-.18,self.ymax+(self.ymax-self.ymin)/20+.013])
            if plot_split:
                fig, ax = plt.subplots()
                self.plot_single(ax=ax, facecolor='#caa', plot_edges=False)
                reg1.plot_single(ax=ax, facecolor='green', alpha=0.5, hatch='/', plot_edges=False)
                reg2.plot_single(ax=ax, facecolor='red', alpha=0.5, hatch='\\', plot_edges=False)
                ax.add_patch(_plt_Circle(intersection_edge.center, intersection_edge.r, edgecolor='black', facecolor='None'))
                ax.set_title("1")
            self.delete()

            return "#0692e4"
            return 'yellow'
        
        if len(splits) == 2:
            start, end = splits
            if any(['pts' in d for d in splits]):
                raise ValueError("multiple double intersections", splits)
            start_pt, end_pt = start['xy'], end['xy'] 
            
            if start['touches'] and end['touches'] and (
                (start_pt, end_pt) in self.coords or (end_pt, start_pt) in self.coords or start_pt==end_pt
            ):
                # Intersection edge coincides with an existing boundary or is a
                # tangent (start == end). No split — record the check result as-is.
                self.add_check_result_no_intersection(intersection_edge=intersection_edge, check=check)
                return 'pink'
            
            start_new_vtx = Vertex(*start_pt, self.all_vtx) if start_pt not in self.all_vtx else self.all_vtx[start_pt]
            end_new_vtx = Vertex(*end_pt, self.all_vtx) if end_pt not in self.all_vtx else self.all_vtx[end_pt] 

            # check the type of the edge here to construct a new instance of it
            # if type(intersection_edge) != LineSegment:
            #     new_edge1 = Arc(vtx1=start_new_vtx, vtx2=end_new_vtx, is_clockwise=None,printme="itx e1", **arc_kwargs)
            #     new_edge2 = Arc(vtx1=end_new_vtx, vtx2=start_new_vtx, is_clockwise=not new_edge1.is_clockwise,printme="itx e2", **arc_kwargs)
            # else:
            #     new_edge1 = LineSegment(vtx1=start_new_vtx, vtx2=end_new_vtx, **line_kwargs)
            #     new_edge2 = LineSegment(vtx1=end_new_vtx, vtx2=start_new_vtx, **line_kwargs)

            if type(intersection_edge) != LineSegment:
                new_edge1 = Arc(vtx1=end_new_vtx, vtx2=start_new_vtx, is_clockwise=None,printme="itx e1", **arc_kwargs)
                new_edge2 = Arc(vtx1=start_new_vtx, vtx2=end_new_vtx, is_clockwise=not new_edge1.is_clockwise,printme="itx e2", **arc_kwargs)
            else:
                new_edge1 = LineSegment(vtx1=end_new_vtx, vtx2=start_new_vtx, **line_kwargs)
                new_edge2 = LineSegment(vtx1=start_new_vtx, vtx2=end_new_vtx, **line_kwargs)

            if not start['touches']:
                edge_start_before, edge_start_after = [[edge] for edge in start['edges'][0].split(start_new_vtx)]
            else: 
                edge_start_before = [start['edges'][0]]
                edge_start_after = [] #if len(start['edges'])<2 else [start['edges'][-1]]
            if new_edge1.vtx2.xy != edge_start_before[0].vtx2.xy:
                start_edge_i = start['ids'][-1]
                if not start['touches']:
                    edge_start_before, edge_start_after = [[edge] for edge in start['edges'][-1].split(start_new_vtx)]
                else:
                    if len(start['edges'])>=2:
                        edge_start_before = [start['edges'][-1]]
                        edge_start_after = []#start['edges'][-1]
                    else:
                        edge_start_before = [start['edges'][0]]
                        edge_start_after = []
                        
            else:
                start_edge_i = start['ids'][-1]
            
            if not end['touches']:
                edge_end_before, edge_end_after = [[edge] for edge in end['edges'][0].split(end_new_vtx)]
            else: 
                edge_end_before = [end['edges'][0]]
                edge_end_after = [] if len(end['edges'])<2 else [end['edges'][-1]]
            if new_edge1.vtx1.xy != edge_end_before[0].vtx2.xy:
                end_edge_i = end['ids'][-1]
                if not end['touches']:
                    edge_end_before, edge_end_after = [[edge] for edge in end['edges'][-1].split(end_new_vtx)]
                else:
                    if len(end['edges'])>=2:
                        edge_end_before = [end['edges'][0]]
                        edge_end_after = [end['edges'][-1]]#
                    else:
                        edge_end_before = [end['edges'][0]]
                        edge_end_after = []
            else:
                end_edge_i = end['ids'][-1]
                # end_edge_i = end['ids'][0]
            # first_region_edge = new_edge1 if new_edge1.vtx1.xy == edge_end_before[0].vtx2.xy else new_edge2
            # second_region_edge = new_edge1 if new_edge1.vtx1.xy == edge_start_before[0].vtx2.xy else new_edge2
            # TODO CONTINUE HERE. 
            if new_edge1.vtx2.xy != edge_end_before[0].vtx2.xy:
                # edge_start_before, edge_start_after = [[edge] for edge in start['edges'][-1].split(start_new_vtx)] + ([] if not start['touches'] else [[]])
                # edge_end_before, edge_end_after = [[edge] for edge in end['edges'][-1].split(end_new_vtx)] + ([] if not end['touches'] else [[]])
                first_region_edge, second_region_edge = new_edge1, new_edge2
            else:
                raise NotImplementedError("Arrived here. Dev Error.")
                first_region_edge, second_region_edge = new_edge2, new_edge1  
            # not_include_start_edge_in_without_itx = (0 if start['touches'] and (start_edge_i+1==end_edge_i or first_region_edge.vtx2.xy==self.edges[start_edge_i].vtx2.xy) else 1)
            exclude_first_in_itx = int(start['touches'] and self.edges[start_edge_i].vtx1.xy == first_region_edge.vtx2.xy)
            include_second_in_without_itx = (0 if start['touches'] and ((start_edge_i+1)%len(self.edges)==end_edge_i or first_region_edge.vtx2.xy==self.edges[start_edge_i].vtx1.xy) else 1)
            
            not_include_start_edge_in_without_itx = (0 if start['touches'] and ((start_edge_i+1)%len(self.edges)==end_edge_i or first_region_edge.vtx2.xy==self.edges[start_edge_i].vtx1.xy) else 1)
            
            # create edges
            edges_reg1 = edge_start_after
            if end_edge_i > start_edge_i:
                edges_reg1 += self.edges[start_edge_i+not_include_start_edge_in_without_itx:end_edge_i]
            else:
                edges_reg1 += self.edges[start_edge_i+not_include_start_edge_in_without_itx:] + self.edges[:end_edge_i]
            if len(edge_end_before)>0 and edges_reg1[-1].vtx2.xy == edge_end_before[0].vtx1.xy:
                edges_reg1 += edge_end_before 
            edges_reg1 += [first_region_edge]
            
            # intersection
            edges_reg2 = edge_end_after
            if end_edge_i > start_edge_i:
                edges_reg2 += self.edges[end_edge_i+1:] +  self.edges[:start_edge_i] 
            else:
                edges_reg2 += self.edges[end_edge_i+1:start_edge_i]
            if not start['touches'] and self.edges[start_edge_i].vtx1.xy != second_region_edge.vtx1.xy:
                edges_reg2 += edge_start_before
            edges_reg2.append(second_region_edge) 

            # check if a region is so small that its vertices are so close to one another that they are considered the same.
            # in this case the original region should not be modified.
            if (len(set([e.vtx1.xy for e in edges_reg1]+[e.vtx2.xy for e in edges_reg1])) <= 2 or
                len(set([e.vtx1.xy for e in edges_reg2]+[e.vtx2.xy for e in edges_reg2])) <= 2 or
                len(set(set([e.vtx1.x for e in edges_reg2]+[e.vtx2.x for e in edges_reg2]))) <= 1 or
                len(set(set([e.vtx1.y for e in edges_reg2]+[e.vtx2.y for e in edges_reg2]))) <= 1
                ):
                # if region is invalid due to precision issue keep region unchanged and add check and result to it
                # print("++b+++++++++++++++++++++Region lost due to precison", splits)
                # print([(e.vtx1.xy, e.vtx2.xy) for e in edges_reg1])
                # print("check",check)
                # print("start_pt, end_pt", start_pt, end_pt)
                # print("start['ids']",start['ids'], "end['ids']",end['ids'])
                # print("end_edge_i",end_edge_i, "start_edge_i",start_edge_i)
                if type(intersection_edge) != LineSegment:
                    cx,cy = self.centroid
                    ie_cx, ie_cy = intersection_edge.center
                    res = ((ie_cx-cx)**2+(ie_cy-cy)**2)**.5 <= intersection_edge.r
                else:
                    cx,cy = self.centroid
                    res = pt_is_left_of_vector(cx, cy, *self.edges[0].vtx1.xy, *self.edges[0].vtx2.xy)
                    res = pt_is_left_of_vector(cx, cy, *intersection_edge.vtx1.xy, *intersection_edge.vtx2.xy)

                self.checks = self.checks+[{**check, 'result': res}]
                # create arc
                return "#7eee22" if res else "#e74505"

            # CHECK FOR WHICH OF THE TWO REGIONS THE CHECK IS TRUE 
            if type(intersection_edge) != LineSegment:
                # first_region_is_within_radius = pt_is_left_of_vector(*first_region_edge.center, *first_region_edge.vtx1.xy, *first_region_edge.vtx2.xy)
                # mid_angle = (first_region_edge.angle_vtx1+first_region_edge.angle_vtx2)/2
                
                # cx, cy = first_region_edge.center
                # r = first_region_edge.r
                # rotation = angle_to((0,0), first_region_edge.center)
                # mp.precision = 16
                # x_to_check_reg1 = cx + (r-0.02) * float(mp.cos((mid_angle+rotation)/360*2*mp.pi))#rotation+
                # y_to_check_reg1 = cy + (r-0.02) * float(mp.sin((mid_angle+rotation)/360*2*mp.pi))#rotation+
                x_to_check_reg1, y_to_check_reg1 = edges_reg1[0].vtx2.xy
                cx, cy = intersection_edge.center
                first_region_is_within_radius = ((cx-x_to_check_reg1)**2+(cy-y_to_check_reg1)**2)**.5 <= intersection_edge.r
            else:
                # edges = edge_start_after + self.edges[start_edge_i+not_include_start_edge_in_without_itx:end_edge_i] + edge_end_before + [first_region_edge]
                # mean_x, mean_y = sum([e.vtx1.x for e in edges])/len(edges), sum([e.vtx1.y for e in edges])/len(edges)
                # get coordinates that comparable to circle center meaning that they lean towards their cell. By construction of the line edge we know that following its direction this point is to the left
                # x_to_check_reg1 = (intersection_edge.vtx2.x + intersection_edge.vtx1.x)/2 + 2*(intersection_edge.vtx2.y - intersection_edge.vtx1.y)
                # y_to_check_reg1 = (intersection_edge.vtx2.y + intersection_edge.vtx1.y)/2 + 2*(intersection_edge.vtx2.x - intersection_edge.vtx1.x)
                # x_to_check_reg1, y_to_check_reg1 = edges_reg1[0].vtx2.xy
                # first_region_is_within_radius = pt_is_left_of_vector(
                #     x_to_check_reg1,
                #     y_to_check_reg1,
                #     *first_region_edge.vtx1.xy, *first_region_edge.vtx2.xy
                # )
                x_to_check_reg1 = sum([e.vtx1.x for e in edges_reg1])/len(edges_reg1)
                y_to_check_reg1 = sum([e.vtx1.y for e in edges_reg1])/len(edges_reg1)
                first_region_is_within_radius = pt_is_left_of_vector(
                    x_to_check_reg1,
                    y_to_check_reg1,
                    *intersection_edge.vtx1.xy, *intersection_edge.vtx2.xy
                )

            reg1 = OffsetRegion(
                edges = edges_reg1, 
                # checks = self.checks+[{**check, 'result': True}],
                checks = self.checks+[{**check, 'result': first_region_is_within_radius}], 
                all_regions = self.all_regions, printme="without_itx2"
            ) 
     
            reg2 = OffsetRegion(
                edges = edges_reg2, 
                # checks = self.checks+[{**check, 'result': True}],
                checks = self.checks+[{**check, 'result': not first_region_is_within_radius}], 
                all_regions = self.all_regions, printme="with_itx2"
            ) # Todo condition for when true
            

            if False and not new_edge1.vtx1.xy == edge_end_before[0].vtx2.xy and not new_edge1.vtx1.xy == edge_start_before[0].vtx2.xy:
                
                # if any([e1.vtx2.xy != e2.vtx1.xy for e1,e2 in zip(reg2.edges,reg2.edges[1:]+reg2.edges[:1])]):
                print("edge_start_before[0]",edge_start_before[0].vtx2.xy, edge_start_before[0].vtx1.xy)
                print("edge_end_before[0]",edge_end_before[0].vtx2.xy, edge_end_before[0].vtx1.xy)
                print("new_edge1.vtx1.xy",new_edge1.vtx1.xy)
                print("new_edge1.vtx2.xy",new_edge1.vtx2.xy)
                print("new_edge2.vtx1.xy",new_edge2.vtx1.xy)
                print("new_edge2.vtx2.xy",new_edge2.vtx2.xy)
                ax=self.plot_single(facecolor="#ACACAC", plot_edges=False)
                ax=reg2.plot_single(ax=ax)
                ax.set_title("reg2:"+str(int(start['touches']))+"-"+str(int(end['touches'])))
                intersection_edge.plot_single(ax=ax, edgecolor="black", linestyle='dotted',linewidth=3)
                
                ax.set_xlim([self.xmin-(self.xmax-self.xmin)/20, self.xmax+(self.xmax-self.xmin)/20])
                ax.set_ylim([self.ymin-(self.ymax-self.ymin)/20, self.ymax+(self.ymax-self.ymin)/20])
                
                # for split in splits:
                #     # print("split",split)
                #     ax.add_patch(_plt_Circle(xy=split['xy'], radius=0.02, facecolor='red' if split['touches'] else "#d40dc4",alpha=0.8))
                #     ax.annotate(
                #                 text=('L' if split['edge'].type=='LineSegment' else 'A'), xy=split['xy'], #("T" if split['touches'] else "F") + '-' + 
                #                 horizontalalignment='center', fontsize=10, color="black", weight="bold"
                #         )
                    
                ax=None
                for edge in self.edges:
                    if edge.vtx1.xy in vertices_intersected or edge.vtx2.xy in vertices_intersected:
                        ax=edge.plot_single(
                            ax=ax, color="#ff726d" if len(edge.intersection(intersection_edge))>0 else '#000',
                            radial_lines=False, linewidth=10)
                ax=edge_end_before[0].plot_single(ax=ax,radial_lines=False, color="#edfd07")
                ax=edge_start_before[0].plot_single(ax=ax,radial_lines=False, color="#0717fd")
                if len(edge_end_after): ax=edge_end_after[0].plot_single(ax=ax,radial_lines=False, color="#0a0009")
                if len(edge_start_after): ax=edge_start_after[0].plot_single(ax=ax,radial_lines=False, color="#16f829")
                
                ax.set_xlim([self.xmin-(self.xmax-self.xmin)/20, self.xmax+(self.xmax-self.xmin)/20])
                ax.set_ylim([self.ymin-(self.ymax-self.ymin)/20, self.ymax+(self.ymax-self.ymin)/20])

                ax.scatter(x=[v[0] for v in vertices_intersected], y=[v[1] for v in vertices_intersected], marker="+",s=10, color="black")
                print("vertices_intersected", vertices_intersected)
                print("start['touches'], end['touches']",start['touches'], end['touches'])
                print("edge_start_before",edge_start_before)
                print("edge_start_after",edge_start_after)
                print("edge_end_before",edge_end_before)
                print("edge_end_after",edge_end_after)
                print("not_include_start_edge_in_without_itx",not_include_start_edge_in_without_itx)
                print("first_region_is_within_radius",first_region_is_within_radius)
                print("intersection_edge",intersection_edge)
                print([(pt['xy'],any([vtx.xy==pt['xy'] for vtx in self.vertices]),[type(e) for e in pt['edges']]) for pt in splits])
                # if any([e1.vtx2.xy != e2.vtx1.xy for e1,e2 in zip(reg1.edges,reg1.edges[1:]+reg1.edges[:1])]):
                ax=self.plot_single(facecolor="#B1B1B1")
                ax=reg1.plot_single(ax=ax)
                ax.set_title("reg1:"+str(int(start['touches']))+"-"+str(int(end['touches'])))
                intersection_edge.plot_single(ax=ax, edgecolor="black", linestyle='dotted',linewidth=3)
                ax.set_xlim([self.xmin-(self.xmax-self.xmin)/20, self.xmax+(self.xmax-self.xmin)/20])
                ax.set_ylim([self.ymin-(self.ymax-self.ymin)/20, self.ymax+(self.ymax-self.ymin)/20])
                # for pt in splits:
                #     # print("pt",pt)
                #     ax.add_patch(_plt_Circle(xy=pt['xy'], radius=0.02, facecolor='red' if pt['touches'] else "#d40dc4",alpha=0.8))
                #     ax.annotate(
                #                 text=('L' if pt['edge'].type=='LineSegment' else 'A'), xy=pt['xy'], #("T" if pt['touches'] else "F") + '-' + 
                #                 horizontalalignment='center', fontsize=10, color="black", weight="bold"
                #         )
                raise ValueError("PROBLEM!")
         
            reg1_poly = _to_shapely_polygon(reg1, arc_steps_per_degree=20)
            reg2_poly = _to_shapely_polygon(reg2, arc_steps_per_degree=20)
            reg1_is_valid = reg1_poly.is_valid if reg1_poly is not None else False
            reg2_is_valid = reg2_poly.is_valid if reg2_poly is not None else False
            if plot_split or (
                len(reg1.edges) != len(reg1.vertices) or 
                len(reg2.edges) != len(reg2.vertices) or 
                not reg1.is_closed or
                not reg2.is_closed or not reg1_is_valid or not reg2_is_valid
                # or type(intersection_edge) == LineSegment
                ):
                # print("p dis", [[((sj['xy'][0]-si['xy'][0])**2+(sj['xy'][1]-si['xy'][1])**2)**.5 for j,sj in enumerate(splits) if i<j] for i,si in enumerate(splits)])
                fig, axs = plt.subplots(1,3, figsize=(12,3))
                for i in [0,1,2]:
                    ax = axs.flat[i]
                    coords = self.get_plot_coords()
                    x_coords = [x for x,y in coords]
                    y_coords = [y for x,y in coords]
                    x_min, x_max = min(x_coords), max(x_coords)
                    y_min, y_max = min(y_coords), max(y_coords)
                    x_dist, y_dist = x_max-x_min, y_max-y_min 
                    self.plot_single(
                        ax=ax, facecolor='#ccc', 
                        plot_edges=False, x_lim=[x_min-x_dist/10, x_max+x_dist/10], y_lim=[y_min-y_dist/10, y_max+y_dist/10], 
                        add_arrow=i==0,
                        plot_vertices=i==0, add_idx_edges=i==0)
                    if i == 0:
                        if type(intersection_edge) == LineSegment:
                            ax.scatter(x_to_check_reg1, y_to_check_reg1, marker='P',facecolor="black",edgecolor='black', s=300)
                            
                            intersection_edge.plot_single(ax=ax, add_arrow='red')
                        first_region_edge.plot_single(ax=ax, radial_lines=False, linewidth=4, add_arrow='blue')
                        edge_start_before[0].plot_single(ax=ax, radial_lines=False, linewidth=5, linestyle='dashed', edgecolor="#ff0000")
                        if len(edge_start_after): edge_start_after[0].plot_single(ax=ax, radial_lines=False, linewidth=5, linestyle='dashed', edgecolor="#db07f7")
                        edge_end_before[0].plot_single(ax=ax, radial_lines=False, linewidth=5, linestyle='dashed', edgecolor="#28c986")
                        if len(edge_end_after): edge_end_after[0].plot_single(ax=ax, radial_lines=False, linewidth=5, linestyle='dashed', edgecolor="#91ff00")
                    if i == 1:
                        reg1.plot_single(ax=ax, facecolor='green' if first_region_is_within_radius else 'red', alpha=0.5, plot_edges=False, x_lim=None, y_lim=None, plot_vertices=True, add_idx_edges=True,add_arrow=True)
                    if i == 2:
                        reg2.plot_single(ax=ax, facecolor='red' if first_region_is_within_radius else 'green', alpha=0.5, plot_edges=False, x_lim=None, y_lim=None, plot_vertices=True, add_idx_edges=True,add_arrow=True)
                    if type(intersection_edge) != LineSegment:
                        ax.add_patch(_plt_Circle(xy=intersection_edge.center, radius=intersection_edge.r, edgecolor='black', facecolor='None'))
                    else:
                        if intersection_edge.vtx1.x == intersection_edge.vtx2.x:
                            ymin, ymax = sorted([intersection_edge.vtx1.y, intersection_edge.vtx2.y])
                            ax.vlines(x=intersection_edge.vtx1.x, ymin=ymin, ymax=ymax, color='black',linewidth=1)
                        else:
                            xmin, xmax = sorted([intersection_edge.vtx1.x, intersection_edge.vtx2.x])
                            ax.hlines(y=intersection_edge.vtx1.y, xmin=xmin, xmax=xmax, color='black',linewidth=1)

                    if i == 0:
                        ax.set_title("Ts"+str(int(start['touches']))+"e:"+str(int(end['touches']))+ 's_i:'+str(start_edge_i)+'e_i:'+str(end_edge_i)+
                                     'n_es'+str(len(self.edges))+'+'+str(not_include_start_edge_in_without_itx) +"-"+ 
                                     str(int(first_region_edge.vtx2.xy == self.edges[start_edge_i].vtx1.xy)) +
                                     str(int(first_region_edge.vtx1.xy == self.edges[start_edge_i].vtx1.xy)) +
                                     str(int(first_region_edge.vtx2.xy == self.edges[start_edge_i].vtx2.xy)) +
                                     str(int(first_region_edge.vtx1.xy == self.edges[start_edge_i].vtx2.xy))
                                     )
                    elif i == 1:
                        ax.set_title(
                            str(start['ids'])+str(end['ids'])+
                            "n_v:"+str(len(reg1.vertices)) + 
                            'n_e'+str(len(reg1.edges)) +
                            str(reg1.is_closed) +
                            "")
                            
                    elif i == 2:
                        ax.set_title(
                            "n_v:"+str(len(reg2.vertices))+ 
                            'n_e'+str(len(reg2.edges))+
                            str(reg2.is_closed) +
                            'ef'+str(exclude_first_in_itx)+
                            "")
                    ax.scatter(x_to_check_reg1,y_to_check_reg1,
                               marker='P'if first_region_is_within_radius else 'X',
                               facecolor="#a6ff02" if first_region_is_within_radius else "#ff0278",
                               edgecolor='black',
                               s=100)
                    ax.scatter(*start['xy'], marker='o',facecolor="black",edgecolor='black', s=100)
                    ax.scatter(*end['xy'],marker='^',facecolor="black",edgecolor='black', s=100)
                    ax.scatter(x=[e.vtx1.x for e in self.edges], y=[e.vtx1.y for e in self.edges], 
                               marker='X',facecolor="#d400ff",edgecolor="#09ff00", s=200)
                    ax.scatter(x=[e.vtx2.x for e in self.edges], y=[e.vtx2.y for e in self.edges], 
                               marker='X',facecolor="#d400ff",edgecolor="#09ff00", s=200)
                    ax.scatter(x=[v.x for v in self.vertices], y=[v.y for v in self.vertices], 
                               marker='X',facecolor="#d400ff",edgecolor="#09ff00", s=200)
                    ax.set_xlim([min([x_to_check_reg1,self.xmin-0.01]),max([x_to_check_reg1, self.xmax+0.01])])
                    ax.set_ylim([min([y_to_check_reg1,self.ymin-0.01]),max([y_to_check_reg1, self.ymax+0.01])])
                    ax.set_xlim([self.xmin-(self.xmax-self.xmin)/20-.05,self.xmax+(self.xmax-self.xmin)/20+.01])
                    ax.set_ylim([self.ymin-(self.ymax-self.ymin)/20-.05,self.ymax+(self.ymax-self.ymin)/20+.01])
                    
                    fig.savefig('plots/splits_r_'+str(int(r))+'_'+str(r%1)[2:]+".png", dpi=300, bbox_inches="tight")
                    _plt_close(fig)
            
            if len(reg1.edges) != len(reg1.vertices):
                print(start['xy'], end['xy'], [v.xy for v in self.vertices])
                print("!=reg1!", len(reg1.edges), len(reg1.vertices),[(e.vtx1.xy, e.vtx2.xy) for e in reg1.edges], [v.xy for v in reg1.vertices])
                print("??reg2        !", len(reg2.edges), len(reg2.vertices),[(e.vtx1.xy, e.vtx2.xy) for e in reg2.edges], [v.xy for v in reg2.vertices])
                raise ValueError("!=reg1!", len(reg1.edges), len(reg1.vertices),len(reg2.edges), len(reg2.vertices),[(e.vtx1.xy, e.vtx2.xy) for e in reg1.edges], [v.xy for v in reg1.vertices])
            if len(reg2.edges) != len(reg2.vertices):
                print(start['xy'], end['xy'], [v.xy for v in self.vertices])
                print("==reg1!", len(reg1.edges), len(reg1.vertices),[(e.vtx1.xy, e.vtx2.xy) for e in reg1.edges], [v.xy for v in reg1.vertices])
                print("!=reg2        !", len(reg2.edges), len(reg2.vertices),[(e.vtx1.xy, e.vtx2.xy) for e in reg2.edges], [v.xy for v in reg2.vertices])
                raise ValueError("!=reg2        !",[(e.vtx1.xy, e.vtx2.xy) for e in reg2.edges], [v.xy for v in reg2.vertices])
            if not reg1.is_closed:
                raise ValueError("not closedr1r2",[reg1],reg2)
            if not reg2.is_closed:
                raise ValueError("not closedr2r1",reg2,reg1)
            self.delete()

        # create arc
        return 'orange'
    #

    def calc_min_dist_to_pt(self, pt):
        """
        returns minimum distance of region to pt
        """
        return min([
            edge.calc_min_dist_to_pt(pt) for edge in self.edges
        ])
    #
    
    def calc_max_dist_to_pt(self, pt):
        """
        returns maximum distance of region to pt
        """
        return max([
            edge.calc_max_dist_to_pt(pt) for edge in self.edges
        ])
    #
    
    def transform_to_trgl(self, i:int):
        """
        Transform region from triangle into triangle i
        """
        if i == 1: return self 
        new_edges = []
        for edge in self.edges:
            new_edges.append(edge.transform_to_trgl(i=i))
        #
        
        if i % 2 == 1: 
            rotated_region = OffsetRegion(new_edges, checks=[], trgl_nr=i, all_regions=self.all_regions, printme="tgl"+str(i))
        else: 
            # reverse order for regions 2,4,6,8
            rotated_region = OffsetRegion(new_edges[::-1], checks=[], trgl_nr=i, all_regions=self.all_regions, printme="tgl"+str(i))
        #
        
        rotated_region.cntd_cells = tuple(sorted(
            [(lvl, (type(_x)(x),type(_y)(y))) for (lvl,(_x,_y)), (x,y) in zip(
                [(lvl,(x,y)) for lvl,(x,y) in self.cntd_cells], 
                transform_cell_pattern([xy for lvl,xy in self.cntd_cells], i)
            )
        ]))
        
        rotated_region.ovlpd_cells = tuple(sorted(
            [(lvl, (type(_x)(x),type(_y)(y))) for (lvl,(_x,_y)), (x,y) in zip(
            [(lvl,(x,y)) for lvl,(x,y) in self.ovlpd_cells], 
            transform_cell_pattern([xy for lvl,xy in self.ovlpd_cells], i))
        ]))
        
        rotated_region.nested_ovlpd_cells = tuple(sorted(
            [(lvl, (type(_x)(x),type(_y)(y))) for (lvl,(_x,_y)), (x,y) in zip(
            [(lvl,(x,y)) for lvl,(x,y) in self.nested_ovlpd_cells], 
            transform_cell_pattern([xy for lvl,xy in self.nested_ovlpd_cells], i))
        ]))

        rotated_region.nested_cntd_cells = tuple(sorted(
            [(lvl, (type(_x)(x),type(_y)(y))) for (lvl,(_x,_y)), (x,y) in zip(
            [(lvl,(x,y)) for lvl,(x,y) in self.nested_cntd_cells], 
            transform_cell_pattern([xy for lvl,xy in self.nested_cntd_cells], i))
        ]))
        
        return rotated_region
    #
    
    @staticmethod
    def merge_regions(regions:list, keep_old:bool=True, add_new:bool=False,):
        """
        Regions must are assumed to shared at least 1 edge, each point that is a combination of two points in regions to merged is assumed to lie within merge region 
        """
        # regions = list(all_regions.values())
        duplicated_edge_coords = set()
        all_edge_coords = set()
        all_edges = []
        for region in regions:
            for edge in region.edges:
                coords = edge.coords
                coords_rev = (coords[1], coords[0])
                
                if coords in all_edge_coords:
                    duplicated_edge_coords.update([coords, coords_rev])
                
                all_edge_coords.update([coords, coords_rev])
        
            all_edges.extend(region.edges)
            if not keep_old:
                region.delete()
        first_last_tpl_list = []
        edges_merged_region = []
        for region in regions:
            edges = region.edges
            duplicated_ids = [i for i,edge in enumerate(edges) if edge.coords in duplicated_edge_coords]
            non_duplicated_ids = [i for i,edge in enumerate(edges) if not edge.coords in duplicated_edge_coords]
            if len(non_duplicated_ids) == len(edges):
                raise ValueError("WEIRD°!")
            if len(non_duplicated_ids) == 0:
                continue
            start = next((i for i in range(min(non_duplicated_ids), max(non_duplicated_ids)+1) if (i-1)%len(edges) not in non_duplicated_ids),None)
            end = next((i for i in range(min(non_duplicated_ids), max(non_duplicated_ids)+1) if (i+1)%len(edges) not in non_duplicated_ids), None)
            if start <= end:
                edges_merged_region.extend(edges[start:end+1])
            else:
                edges_merged_region.extend(edges[start:])
                edges_merged_region.extend(edges[:end+1])
            # check whether duplicate ids wrap around

        if not edges_merged_region:
            # Every region's edges were all shared — this can happen when merging
            # regions that together form a completely closed shape (all internal).
            # Fall back to the outer boundary of the first region.
            edges_merged_region = regions[0].edges
        else:
            # Re-order edges collected from multiple regions into a connected chain.
            # Runs from different regions are appended in arbitrary order; we need
            # each edge[n].vtx2 == edge[n+1].vtx1 to hold throughout.
            ordered = [edges_merged_region[0]]
            remaining = list(edges_merged_region[1:])
            while remaining:
                last_xy = ordered[-1].vtx2.xy
                matched = False
                for k, e in enumerate(remaining):
                    if e.vtx1.xy == last_xy:
                        ordered.append(e)
                        remaining.pop(k)
                        matched = True
                        break
                if not matched:
                    # Gap — cannot reconnect; append the rest as-is to avoid
                    # losing edges (mismatch will be detected in __init__).
                    ordered.extend(remaining)
                    break
            edges_merged_region = ordered

        merged_region = OffsetRegion(edges=edges_merged_region, checks=regions[0].checks, all_regions=region.all_regions, trgl_nr=min([reg.trgl_nr for reg in regions]), printme="merged")
        if not add_new:
            merged_region.delete()
        merged_region.cntd_cells = set(regions[0].cntd_cells)
        merged_region.ovlpd_cells = set(regions[0].ovlpd_cells)
        merged_region.nested_cntd_cells = set(regions[0].nested_cntd_cells)
        merged_region.nested_ovlpd_cells = set(regions[0].nested_ovlpd_cells)
        # Store per-triangle nested cells before the conservative cross-frame intersection
        # so the search can use triangle-specific cells for each source point.
        merged_region.nested_cells_by_trgl = {
            reg.trgl_nr: (tuple(reg.nested_cntd_cells), tuple(reg.nested_ovlpd_cells))
            for reg in regions
        }
        cells_no_longer_cntd = set()
        nested_cells_no_longer_cntd = set()
        for i in range(1,len(regions)):
            
            cells_no_longer_cntd.update(
                merged_region.cntd_cells.difference(set(regions[i].cntd_cells)))
            merged_region.cntd_cells = merged_region.cntd_cells.intersection(
                set(regions[i].cntd_cells))
            merged_region.ovlpd_cells = merged_region.ovlpd_cells.union(
                set(regions[i].ovlpd_cells))
            
            nested_cells_no_longer_cntd.update(
                merged_region.nested_cntd_cells.difference(set(regions[i].nested_cntd_cells)))
            merged_region.nested_cntd_cells = merged_region.nested_cntd_cells.intersection(
                set(regions[i].nested_cntd_cells))
            merged_region.nested_ovlpd_cells = merged_region.nested_ovlpd_cells.union(
                set(regions[i].nested_ovlpd_cells))

        #
        merged_region.ovlpd_cells.update(cells_no_longer_cntd)
        merged_region.nested_ovlpd_cells.update(nested_cells_no_longer_cntd)

        # A parent key in nested_ovlpd may cover children that are in nested_cntd
        # or nested_ovlpd (because different sub-regions merged to different granularities).
        # Replace any such parent key with its explicit children to prevent double-counting.
        # Iterate until stable — expanding one level may reveal further conflicts deeper down.
        changed = True
        while changed:
            changed = False
            to_remove = set()
            to_add = set()
            for (lvl, (row, col)) in merged_region.nested_ovlpd_cells:
                half = 2 ** -(lvl + 2)
                children = [
                    (lvl + 1, (row + dr * half, col + dc * half))
                    for dr, dc in [(-1, -1), (-1, +1), (+1, -1), (+1, +1)]
                ]
                cntd_children = [c for c in children if c in merged_region.nested_cntd_cells]
                ovlpd_children = [c for c in children if c in merged_region.nested_ovlpd_cells]
                if cntd_children or ovlpd_children:
                    to_remove.add((lvl, (row, col)))
                    for c in children:
                        if c not in merged_region.nested_cntd_cells and c not in merged_region.nested_ovlpd_cells:
                            to_add.add(c)
                    changed = True
            merged_region.nested_ovlpd_cells -= to_remove
            merged_region.nested_ovlpd_cells |= to_add

        merged_region.ovlpd_cells = tuple(sorted(merged_region.ovlpd_cells))
        merged_region.cntd_cells = tuple(sorted(merged_region.cntd_cells))

        merged_region.nested_ovlpd_cells = tuple(sorted(merged_region.nested_ovlpd_cells))
        merged_region.nested_cntd_cells = tuple(sorted(merged_region.nested_cntd_cells))
        # if len(regions[0].nested_cntd_cells) != len(merged_region.nested_cntd_cells):
        #     print("Difference in nested_cntd_cells:",
        #           len(set(regions[0].nested_cntd_cells).difference(set(merged_region.nested_cntd_cells))), len(merged_region.nested_cntd_cells))
        # else:
        #     print("merged cntd",len(merged_region.nested_cntd_cells))
        # if len(regions[0].nested_ovlpd_cells) != len(merged_region.nested_ovlpd_cells):
        #     print("Difference in nested_ovlpd_cells:",
        #           len(set(regions[0].nested_ovlpd_cells).difference(set(merged_region.nested_ovlpd_cells))), len(merged_region.nested_ovlpd_cells))
        # else:
        #     print("merged ovlpd",len(merged_region.nested_ovlpd_cells))
        # TODO THINK ABOUT WHETHER TO MERGE ONLY THE DIAGONAL REGIONS TO NOT LOSE AS MUCH OF THE NEST DEPTH
        merged_region.nr = min([reg.nr for reg in regions])
        merged_region.trgl_nrs = [reg.trgl_nr for reg in regions]
        return merged_region
    #
#












# intersection of two circles with same radius
from numpy import (linspace as _np_linspace, array as _np_array, sign as _np_sign, abs as _np_abs, dot as _np_dot, roll as _np_roll)
from math import sin as _math_sin, cos as _math_cos, atan2 as _math_atan2, pi as _math_pi, acos as _math_acos , sin as _math_asin, log10 as _math_log10
from matplotlib import pyplot as plt
from matplotlib.patches import (Rectangle as _plt_Rectangle, Polygon as _plt_Polygon, Circle as _plt_Circle)
from primelocations.utils.general import angle, angles_to_origin, angle_to, pt_is_left_of_vector
from primelocations.utils.distances_to_cell import (get_cell_closest_point_to_point, get_cell_farthest_vertex_to_point,
 check_if_never_contains_convex_set, check_if_always_overlaps_full_convex_set,check_if_always_overlaps_convex_set, get_cells_relevant_for_disk_by_type)
from primelocations.utils.rotations import transform_cell_pattern, transform_coord, transform_cell
from primelocations.utils.intersections import circle_line_segment_intersection, line_intersection, intersections_pts_arc_to_circle
from shapely.geometry import Polygon, LineString, Point
from geopandas import GeoSeries, GeoDataFrame
# from .radius_search.pts_to_offset_regions import (transform_cell_pattern)


# for each circle remember the meaning of the check (contains / overlaps)
# for each intersection point save what circle it comes from
# for each circle get intersection points with home cell
# for each circle-intersection point check whether its in triangle 1


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

        vertices = list(vertices.values())
        if len(vertices):
            ax.scatter(x=[vtx.x for vtx in vertices], y=[vtx.y for vtx in vertices], marker='x', color='black')
        #
    #
    
    def __repr__(self):
        props_not_to_print = ['all_vtx', 'outgoing_edges', 'incoming_edges', 'regions']
        return str({key: val if type(val) != float else round(val,5) for key, val in self.__dict__.items() if key not in props_not_to_print})
    #

    def delete(self):
        """
        Remove vertex from all_vtx
        """
        self.all_vtx.pop(self.xy, None)
    #
#


class Edge(object):
    
    def __init__(self, vtx1:Vertex, vtx2:Vertex, all_edges:dict, contains:tuple=None, overlaps:tuple=None):
        """
        vertices are order counter clockwise
        """
        self.vtx1 = vtx1
        self.vtx2 = vtx2
        self.coords = (
            vtx1.xy if (vtx1 is not None) else (None, None), 
            vtx2.xy if (vtx2 is not None) else (None, None),
            )
        self.regions = []
        # vtx1.outgoing_edges.append(self)
        # vtx2.incoming_edges.append(self)
        self.all_edges = all_edges
        self.all_vtx = vtx1.all_vtx
        
        if not self.coords in all_edges:
            all_edges[self.coords] = self

        if not contains is None:
            self.contains  = contains
        if not overlaps is None:
            self.overlaps  = overlaps
    #

    @staticmethod
    def clear(all_edges):
        all_edges.clear()
    #

    def __repr__(self):
        props_not_to_print = ['all_edges', 'all_vtx', 'regions']
        return str({key: val if type(val) != float else round(val,5) for key, val in self.__dict__.items() if key not in props_not_to_print})
    #
    
    def delete(self):
        self.all_edges.pop(self.coords, None)
    #

    @staticmethod
    def plot_many(edges:dict, ax=None, color_dict={'Arc':'blue', 'LineSegment':'green','Circle':'orange', 'other':'red'}, **kwargs):
        if ax is None:
          fig, ax = plt.subplots()
        #
        arc_polys = []
        for coords,edge in edges.items():
            edge.plot_single(ax=ax, color_dict=color_dict, **kwargs)
        if len(arc_polys) > 0:
            GeoSeries(arc_polys).plot(cmap='viridis', alpha=0.01, edgecolor='black', ax=ax)
        #
    #

    def plot_single(self, ax=None, color_dict={'Arc':'blue', 'LineSegment':'green','Circle':'orange', 'other':'red'}, **kwargs):
        if ax is None:
          fig, ax = plt.subplots()
        #
        arc_polys=[]
        if self.type == 'Arc':
            arc_polys.append(Polygon([self.center, self.coords[0], self.coords[1]]))

            # ax.plot([0, self.center[0]], [0, self.center[1]], marker='.', color="black")
            rotation = angle_to((0,0), self.center)
            
            # ax.plot(
            #     [self.center[0], self.center[0]+self.radius*_math_cos((rotation+self.angle_min)/360*2*_math_pi)],
            #     [self.center[1], self.center[1]+self.radius*_math_sin((rotation+self.angle_min)/360*2*_math_pi)],
            #     marker='.', color='red',alpha=0.3,)
            # ax.plot(
            #     [self.center[0], self.center[0]+self.radius*_math_cos((rotation+self.angle_max)/360*2*_math_pi)],
            #     [self.center[1], self.center[1]+self.radius*_math_sin((rotation+self.angle_max)/360*2*_math_pi)],
            #     marker='.', color='green',alpha=0.3,)
            ax.plot(
                [self.center[0], self.center[0]+self.radius*_math_cos((rotation+self.angle_vtx1)/360*2*_math_pi)],
                [self.center[1], self.center[1]+self.radius*_math_sin((rotation+self.angle_vtx1)/360*2*_math_pi)],
                marker='.', color='red',alpha=0.3,)
            ax.plot(
                [self.center[0], self.center[0]+self.radius*_math_cos((rotation+self.angle_vtx2)/360*2*_math_pi)],
                [self.center[1], self.center[1]+self.radius*_math_sin((rotation+self.angle_vtx2)/360*2*_math_pi)],
                marker='.', color='green',alpha=0.3,)
            x_coords = [x for x,y in self.coords]
            y_coords = [y for x,y in self.coords]
            ax.plot(x_coords, y_coords, marker='o', color=color_dict[self.type],alpha=0.3, **kwargs)
        else:
            GeoSeries([LineString((self.vtx1.xy, self.vtx2.xy))]).plot(ax=ax, color='pink')
        return arc_polys
    #
    
    def intersection(self,edge):
        if self.type == 'LineSegment' and edge.type == 'LineSegment':
            return line_intersection(self.coords, edge.coords)
        
        if self.type == 'Arc': 
            if edge.type == 'Arc':
                return intersections_pts_arc_to_circle(
                    circle_center=edge.center,
                    arc_center=self.center,
                    arc_angle_min=self.angle_min,
                    arc_angle_max=self.angle_max,
                    radius=self.radius,
                )
            elif edge.type == 'Circle':
                return intersections_pts_arc_to_circle(
                    circle_center=edge.center,
                    arc_center=self.center,
                    arc_angle_min=self.angle_min,
                    arc_angle_max=self.angle_max,
                    radius=self.radius,
                )
        if self.type not in ['Arc','LineSegment']:
            print("Types",self.type, edge.type)
            raise ValueError("Unexpected edge type(s)", self.type, edge.type)
        
        line, arc = (self,edge) if self.type == 'LineSegment' else (edge, self)
        
        return circle_line_segment_intersection(
            circle_center=arc.center,
            circle_radius=arc.radius,
            pt1=line.vtx1.xy,
            pt2=line.vtx2.xy,
            full_line=False,
        )
    #
#

class LineSegment(Edge):
    """LineSegment between two 2D coordinates"""
    def __init__(self, vtx1, vtx2, all_edges:dict, contains:tuple=None, overlaps:tuple=None):
        super().__init__(vtx1, vtx2, all_edges, contains=contains, overlaps=overlaps)
        self.type = 'LineSegment'
    #
        
    def split(self, new_vtx):
        """Split at point if point is not start or end point of line segment"""
        if new_vtx.xy in self.coords:
            return [self]
        self.all_edges.pop(self.coords, None)
        line_kwargs = {'all_edges': self.all_edges, **{key: getattr(self, key) for key in ['contains', 'overlaps'] if hasattr(self, key)}}
        return [LineSegment(vtx1=self.vtx1, vtx2=new_vtx, **line_kwargs), LineSegment(vtx1=new_vtx, vtx2=self.vtx2, **line_kwargs)]
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
    """Arc around center (limited by two points on circle if supplied) """
    def __init__(self, center, radius, all_edges:dict, vtx1=None, vtx2=None, outside_angle:bool=False, contains:tuple=None, overlaps:tuple=None):
        super().__init__(vtx1, vtx2, all_edges, contains=contains, overlaps=overlaps)
        self.type = "Arc"
        self.center = center
        self.radius = radius
        self.vtx1 = vtx1
        self.vtx2 = vtx2

        self.angle_vtx1, self.angle_vtx2  = angles_to_origin((vtx1.xy, vtx2.xy), center) if not None in [vtx1,vtx2] else (0,360)
        self.angle_min, self.angle_max = sorted(angles_to_origin((vtx1.xy, vtx2.xy), center)) if not None in [vtx1,vtx2] else (0,360)
        if outside_angle and self.angle_max == 0:
            self.angle_max=360
    #
    
    def split(self,new_vtx):
        """Split at point if point is not start or end point of arc segment"""
        if new_vtx.xy in self.coords:
            return [self]
        self.all_edges.pop(self.coords, None)
        arc_kwargs = {'center': self.center, 'radius': self.radius, 'all_edges': self.all_edges, **{key: getattr(self, key) for key in ['contains', 'overlaps'] if hasattr(self, key)}}

        return [Arc(vtx1=self.vtx1, vtx2=new_vtx, **arc_kwargs), Arc(vtx1=new_vtx, vtx2=self.vtx2, **arc_kwargs)]
    #

    def transform_to_trgl(self, i:int):
        """
        
        """
        if i == 1: return self 
        new_center = transform_coord(self.center, i)
        new_vtx1 = Vertex(*transform_coord(self.vtx1.xy, i), all_vtx=self.all_vtx)
        new_vtx2 = Vertex(*transform_coord(self.vtx2.xy, i), all_vtx=self.all_vtx)
        # flip order for regions 2,4,6,8
        edge_kwargs = {key: tuple(transform_cell(cell=_np_array(getattr(self, key)), i=i)) for key in ['contains', 'overlaps'] if hasattr(self, key)}
        if i%2 == 1: return Arc(new_center, radius=self.radius, all_edges=self.all_edges, vtx1=new_vtx1, vtx2=new_vtx2, **edge_kwargs)
        return Arc(new_center, radius=self.radius, all_edges=self.all_edges, vtx1=new_vtx2, vtx2=new_vtx1, **edge_kwargs)
#

class Circle(object):
    """TODO potentially add as subclass of Edge?"""
    def __init__(self, center, radius):
        self.type = "Circle"
        self.center = center
        self.radius = radius
        self.angle_min, self.angle_max = (0,360)
    #
#

class OffsetRegion(object):
    """OffsetRegion bounded by line segments and arcs"""

    def __init__(self, edges, checks, all_regions:dict, trgl_nr:int=1):
        self.edges = edges
        self.vertices = []
        self.trgl_nr = trgl_nr
        for edge in edges:
            edge.regions.append(self)
            for vtx in (edge.vtx1, edge.vtx2):
                if vtx not in self.vertices:
                    self.vertices.append(vtx)
                    vtx.regions.append(self)
        xs = [vtx.x for vtx in self.vertices]
        ys = [vtx.y for vtx in self.vertices]
        self.xmin = min(xs)
        self.xmax = max(xs)
        self.ymin = min(ys)
        self.ymax = max(ys)
        self.checks = checks
        self.coords = tuple([edge.coords for edge in edges])
        self.all_regions = all_regions
        self.all_edges = edges[-1].all_edges
        self.all_vtx = edges[-1].vtx1.all_vtx

        if not self.coords in all_regions:
            all_regions[self.coords] = self
    #
    
    @staticmethod
    def delete_all(all_regions):
        all_regions.clear()
    #

    def __repr__(self):
        props_not_to_print = ['all_regions', 'all_edges', 'all_vtx']
        return str({key: val if type(val) != float else round(val,5) for key, val in self.__dict__.items() if key not in props_not_to_print})
    #

    def delete(self):
        self.all_regions.pop(self.coords, None)
    #

    def get_coords(self):
        coords = []
        for edge_start_coord, edge_end_coord in self.coords:
            if edge_start_coord not in coords:
                coords.append(edge_start_coord)
            if edge_start_coord not in coords:
                coords.append(edge_start_coord)
        self.vertex_coords = coords
        return coords
    #

    def get_plot_coords(self, arc_steps_per_degree:float=50):
        coords = []
        for (edge_start_coord, edge_end_coord), edge in zip(self.coords, self.edges):
            if edge_start_coord not in coords:
                coords.append(edge_start_coord)
            if edge.type == 'Arc':
                total_angle = abs(edge.angle_max - edge.angle_min)
                n_steps = -int(-(total_angle * arc_steps_per_degree))
                cx, cy = edge.center
                r = edge.radius
                rotation = angle_to((0,0), edge.center)
                for angle_step in _np_linspace(edge.angle_min, edge.angle_max, n_steps):
                    x = cx + r * _math_cos((rotation+angle_step)/360*2*_math_pi)
                    y = cy + r * _math_sin((rotation+angle_step)/360*2*_math_pi)
                    coords.append((float(x),float(y)))
            if edge_end_coord not in coords:
                coords.append(edge_start_coord)
        self.plot_coords = coords
        return coords
    #

    def calc_area(self, arc_steps_per_degree:float=100):
        xs = []
        ys = []
        for ((x,y), edge_end_coord), edge in zip(self.coords, self.edges):
            xs.append(x)  
            ys.append(y)  
            if edge.type == 'Arc':
                total_angle = abs(edge.angle_max - edge.angle_min)
                n_steps = -int(-(total_angle * arc_steps_per_degree))
                cx, cy = edge.center
                r = edge.radius
                rotation = angle_to((0,0), edge.center)
                for angle_step in _np_linspace(edge.angle_min, edge.angle_max, n_steps):
                    xs.append(cx + r * _math_cos((rotation+angle_step)/360*2*_math_pi))
                    ys.append(cy + r * _math_sin((rotation+angle_step)/360*2*_math_pi))
        xs =_np_array(xs)
        ys =_np_array(ys)
        self.area = 0.5*_np_abs(_np_dot(xs,_np_roll(ys,1))-_np_dot(ys,_np_roll(xs,1))) 
        return self.area

    @staticmethod
    def plot_many(
        regions:list, 
        plot_edges:bool=False, plot_vertices:bool=True, add_idxs:bool=True, x_lim:tuple=(-0.02,0.52), y_lim:tuple=(-0.02,0.52), 
        ax=None, cmap='tab20', alpha=0.3, title:str="", **kwargs):

        if ax is None:
            fig, ax = plt.subplots()
        
        if len(regions) > 0:
            color_keys = ['c', 'color']
            if not any([key in color_keys for key in kwargs]):
                kwargs.update({'cmap': cmap})
            
            
            try:
                # GeoSeries([Polygon(reg.get_plot_coords()) for reg in regions]).plot(ax=ax, alpha=alpha, **{'edgecolor':'black', **kwargs})
                GeoSeries([Polygon(reg.get_plot_coords()) for reg in regions]).plot(ax=ax, alpha=alpha, **kwargs)
            except:
                raise ValueError("kwargs", kwargs, "polys", [Polygon(reg.get_plot_coords()) for reg in regions],"ax", type(ax), ax, "alpha", alpha)
            if add_idxs:
                for i, region in enumerate(regions):
                    coords = region.get_coords()
                    ax.annotate(
                        text=str(i if not hasattr(region, 'id') else region.id),
                        xy=(sum([x for x,y in coords])/len(coords), sum([y for x,y in coords])/len(coords)),
                        horizontalalignment='center')
                
        if plot_edges:
            Edge.plot_many(edges=regions[-1].all_edges, ax=ax,markersize=0.2)
        if plot_vertices:
            # for reg in regions.values():
            #     Vertex.plot_many(all_vtx=[vtx.xy for vtx in reg.vertices], ax=ax)
            Vertex.plot_many(vertices=regions[-1].all_vtx, ax=ax)
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        ax.set_title(str(len(regions)) + " regions.")
    #

    def plot_single(self, ax=None, plot_edges:bool=True, plot_vertices:bool=False, add_idx_edges:bool=False, x_lim:tuple=(-0.02,0.52), y_lim:tuple=(-0.02,0.52), color='green', edgecolor='black', alpha=0.70, **kwargs):
        if ax is None:
            fig, ax = plt.subplots()
        kwargs = {'color': color, 'alpha': alpha, 'edgecolor': edgecolor,  **kwargs }
        GeoSeries([Polygon(self.get_plot_coords())]).plot(ax=ax, **kwargs)
        if plot_edges:
            for edge in self.edges:
                edge.plot_single(ax=ax)
        if plot_vertices:
            Vertex.plot_many(vertices=[vtx.xy for vtx in self.vertices], ax=ax)
        if add_idx_edges:
            for i, edge in enumerate(self.edges):
                ax.annotate(text=str(i), xy=edge.vtx1.xy) 
        if not x_lim is None:
            ax.set_xlim(x_lim)
        if not y_lim is None:
            ax.set_ylim(y_lim)
    #


    def get_split_pts(self, intersection_edge):
        pts_to_split_at, vertices_intersected = [], []

        for i, edge in enumerate(self.edges):
            itx = edge.intersection(intersection_edge)
            # print(i,itx, edge)
            # check if itx is in edge
            if len(itx) == 0:
                continue 
            #

            if len(itx) == 1:
                pt = itx[0]
                # TODO check if vertex already exists: then its clear
                # new_vtx = Vertex(*itx[0])
                if not pt in vertices_intersected:
                    pts_to_split_at.append({'i': i,'pt': pt,'edge': edge, 'touches': pt in edge.coords, 'ids':[i]})# [edge1, itx[0], edge2], 
                else:
                    for pt_to_split_at in pts_to_split_at:
                        if pt_to_split_at['pt'] == pt:
                            pt_to_split_at['pt']['ids'].append(i)
                            print("pt_to_split_at['pt']['ids']",pt_to_split_at['pt']['ids'])
                            break
                if not pt in edge.coords:
                    vertices_intersected.append(pt)
                #
                continue
            #

            if len(itx) == 2:
                pts_to_split_at.append({'i': i,'pts': itx,'edge': edge, 'touches': pt in edge.coords})
                print("TWO INTERSECTIONS!", itx)
                continue
            #

            raise ValueError("TOO MANY INTERSECTIONS!", itx)
        #
        return pts_to_split_at
    #


    
    def add_check_result_no_intersection(self, intersection_edge, check):
        """
        
        """

        # mean_x = sum([c[0][0] for c in self.coords]) / len(self.coords)
        # mean_y = sum([c[0][1] for c in self.coords]) / len(self.coords)
        vtx_not_on_intersection_edge = next(
            (coord for coord in [startcoord for startcoord, endcoord in self.coords] if (
            hasattr(intersection_edge, 'vtx1') and coord != intersection_edge.vtx1.xy and coord != intersection_edge.vtx2.xy)),
            self.coords[0][0])
        if intersection_edge.type == 'LineSegment':
            # TODO this must store better on which side of the line implies check success
            result = pt_is_left_of_vector(*vtx_not_on_intersection_edge, *intersection_edge.vtx1.xy, *intersection_edge.vtx2.xy)
        else:
            x,y=vtx_not_on_intersection_edge
            # TODO as polygon is not necessarily convex centroid may lay outide, s.t this check may produce wrong results.  
            # BETTER APPROACH. Chose a point that is not on the current intersection edge. 
            # print('radius', intersection_edge.radius, ((mean_x-intersection_edge.center[0])**2 + (mean_y-intersection_edge.center[1])**2)**.5)
            result = intersection_edge.radius**2 > ((x - intersection_edge.center[0])**2 + (y - intersection_edge.center[1])**2)
        self.checks.append({**check, 'result': result})
    #

    def split_with_circle(self, intersection_edge, check, plot_split:bool=False):
        """
        check if intersected
        split edges
        add vertices
        split region 
        TODO rework this logic. Too convoluted!
        TODO THIS FUNCTION IS WAY TOO LONG
        """
        pts_to_split_at = self.get_split_pts(intersection_edge=intersection_edge)
        # print(intersection_edge.type, "Length before filter", len(pts_to_split_at), pts_to_split_at)
        vertices_intersected, split_pts = [], []

        for split_at in pts_to_split_at:
            if 'pt' in split_at:
                if not split_at['pt'] in vertices_intersected:
                    split_pts.append(split_at)
                vertices_intersected.append(split_at['pt'])
            else:
                split_pts.append(split_at)
        
        pts_to_split_at=split_pts
        # print("Length after filter", len(pts_to_split_at), pts_to_split_at)
        if len(pts_to_split_at) == 0:
            self.add_check_result_no_intersection(intersection_edge=intersection_edge, check=check)
            return 'green' if self.checks[-1]['result'] else 'red'
        
        if len(pts_to_split_at) > 2:
            raise ValueError("TODO IMPLEMENT MULTIPLE INTERSECTIONS", pts_to_split_at, intersection_edge.center)
        
        line_kwargs = {'all_edges':self.all_edges, **{key: getattr(intersection_edge, key) for key in ['contains', 'overlaps'] if hasattr(intersection_edge, key)}}
        arc_kwargs = {'center': intersection_edge.center, 'radius': intersection_edge.radius, **line_kwargs}

        if len(pts_to_split_at) == 1:
            if not 'pts' in pts_to_split_at[0]:
                self.add_check_result_no_intersection(intersection_edge=intersection_edge, check=check)
                return 'green' if self.checks[-1]['result'] else 'red'
            
            if len(pts_to_split_at[0]['pts']) != 2:
                raise ValueError("Unexpected number of itx", pts_to_split_at)
            print("OPTION A")
            start_pt, end_pt = pts_to_split_at[0]['pts']
            old_edge = pts_to_split_at[0]['edge']
            old_edge.delete()
            pos = pts_to_split_at[0]['i']
            start_vtx_existed = start_pt not in self.all_vtx
            end_vtx_existed = end_pt not in self.all_vtx

            start_new_vtx = Vertex(*start_pt, self.all_vtx) if start_vtx_existed else self.all_vtx[start_pt]
            end_new_vtx = Vertex(*end_pt, self.all_vtx) if end_vtx_existed else self.all_vtx[end_pt] 
            
            if type(intersection_edge) != LineSegment:
                new_edge_start =               [Arc(vtx1=old_edge.vtx1, vtx2=start_new_vtx, **arc_kwargs)] if start_vtx_existed else []
                new_edge_end =                 [Arc(vtx1=end_new_vtx,   vtx2=old_edge.vtx2, **arc_kwargs)] if end_vtx_existed else []
                new_edge_middle =              [Arc(vtx1=start_new_vtx, vtx2=end_new_vtx,   **arc_kwargs)]
                new_edge_intersection =         Arc(vtx1=start_new_vtx, vtx2=end_new_vtx,   **arc_kwargs)
                new_edge_intersection_reverse = Arc(vtx1=end_new_vtx,   vtx2=start_new_vtx, **arc_kwargs)
            else:
                new_edge_start =               [LineSegment(vtx1=old_edge.vtx1, vtx2=start_new_vtx, **line_kwargs)] if start_vtx_existed else []
                new_edge_end =                 [LineSegment(vtx1=end_new_vtx,   vtx2=old_edge.vtx2, **line_kwargs)] if end_vtx_existed else []
                new_edge_middle =              [LineSegment(vtx1=end_new_vtx,   vtx2=old_edge.vtx2, **line_kwargs)]
                new_edge_intersection =         LineSegment(vtx1=start_new_vtx, vtx2=end_new_vtx,   **line_kwargs)
                new_edge_intersection_reverse = LineSegment(vtx1=end_new_vtx,   vtx2=start_new_vtx, **line_kwargs)
            
            # TODO MAKE THIS HANDLE LINESEGEMENTS ASWELL
            first_region_is_within_radius = pt_is_left_of_vector(*new_edge_intersection.center, *new_edge_intersection.vtx1.xy, *new_edge_intersection.vtx2.xy)
            # region without intersection_edge
            reg_without_itx = OffsetRegion(
                edges = self.edges[:pos] + new_edge_start + [new_edge_intersection] + new_edge_end + self.edges[pos+1:], 
                checks = self.checks+[{**check, 'result': first_region_is_within_radius}], all_regions = self.all_regions
            ) 
            # new intersection region 
            reg_itx = OffsetRegion(
                edges = new_edge_middle + [new_edge_intersection_reverse], 
                checks = self.checks+[{**check, 'result': not first_region_is_within_radius}], all_regions = self.all_regions
            )
            if plot_split:
                fig, ax = plt.subplots()
                self.plot_single(ax=ax, color='#caa', plot_edges=False)
                reg_without_itx.plot_single(ax=ax, color='green', alpha=0.5, hatch='/', plot_edges=False)
                reg_itx.plot_single(ax=ax, color='red', alpha=0.5, hatch='\\', plot_edges=False)
                GeoSeries([Point(intersection_edge.center).buffer(intersection_edge.radius, 40)]).plot(ax=ax, edgecolor='black', facecolor='None')
                ax.set_title("1")
            self.delete()

            return 'yellow'
        
        if len(pts_to_split_at) == 2:
            start, end = pts_to_split_at
            if any(['pts' in d for d in pts_to_split_at]):
                raise ValueError("multiple double intersections", pts_to_split_at)
            start_pt, end_pt = start['pt'], end['pt'] 
            
            if start['touches'] and end['touches'] and (
                (start_pt, end_pt) in self.coords or (end_pt, start_pt) in self.coords or start_pt==end_pt
            ):
                indexes = []
                if (start_pt, end_pt) in self.coords:
                    indexes.append(self.coords.index((start_pt, end_pt)))
                if (end_pt, start_pt) in self.coords:
                    indexes.append(self.coords.index((end_pt, start_pt)))
                if any([type(edge) != type(intersection_edge) for i,edge in enumerate(self.edges) if i in indexes]):
                    print("TOOOOOOOOOOOODDDDDDDDDDOOOOOOOOOO")
                return 'pink'
            
            start_new_vtx = Vertex(*start_pt, self.all_vtx) if start_pt not in self.all_vtx else self.all_vtx[start_pt]
            end_new_vtx = Vertex(*end_pt, self.all_vtx) if end_pt not in self.all_vtx else self.all_vtx[end_pt] 
            
            if type(intersection_edge) != LineSegment:
                new_edge1 = Arc(vtx1=end_new_vtx, vtx2=start_new_vtx, **arc_kwargs)
                new_edge2 = Arc(vtx1=start_new_vtx, vtx2=end_new_vtx, **arc_kwargs)
            else:
                new_edge1 = LineSegment(vtx1=end_new_vtx, vtx2=start_new_vtx, **line_kwargs)
                new_edge2 = LineSegment(vtx1=start_new_vtx, vtx2=end_new_vtx, **line_kwargs)

            # check the type of the edge here to construct a new instance of it
            edge_start_before, edge_start_after = [[edge] for edge in start['edge'].split(start_new_vtx)] + ([] if not start['touches'] else [[]])
            edge_end_before, edge_end_after = [[edge] for edge in end['edge'].split(end_new_vtx)] + ([] if not end['touches'] else [[]])
            first_region_edge = new_edge1 if new_edge1.vtx1.xy == edge_end_before[0].vtx2.xy else new_edge2
            second_region_edge = new_edge1 if new_edge1.vtx1.xy == edge_start_before[0].vtx2.xy else new_edge2
            
            include_first_in_without_itx = (0 if start['touches'] and (start['i']+1==end['i'] or first_region_edge.vtx2.xy==self.edges[start['i']].vtx1.xy) else 1)
            # TODO MAKE THIS HANDLE LINESEGEMENTS ASWELL
            first_region_is_within_radius = pt_is_left_of_vector(*first_region_edge.center, *first_region_edge.vtx1.xy, *first_region_edge.vtx2.xy)
            
            reg_without_itx = OffsetRegion(
                edges = edge_start_after + self.edges[start['i']+include_first_in_without_itx:end['i']] + edge_end_before + [first_region_edge], 
                # checks = self.checks+[{**check, 'result': True}], all_regions = self.all_regions
                checks = self.checks+[{**check, 'result': first_region_is_within_radius}], all_regions = self.all_regions
            ) 
            # intersection
            exclude_first_in_itx = int(start['touches'] and self.edges[start['i']].vtx1.xy == first_region_edge.vtx2.xy)
            reg_itx = OffsetRegion(
                edges = edge_end_after + self.edges[end['i']+1:] +  self.edges[:start['i']] + (
                    edge_start_before if not (start['touches'] and self.edges[start['i']].vtx1.xy == first_region_edge.vtx2.xy) else []
                    ) + [second_region_edge], 
                # checks = self.checks+[{**check, 'result': True}], all_regions = self.all_regions
                checks = self.checks+[{**check, 'result': not first_region_is_within_radius}], all_regions = self.all_regions
            ) # Todo condition for when true
            
            if plot_split:
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
                        ax=ax, color='#ccc', plot_edges=False, x_lim=[x_min-x_dist/10, x_max+x_dist/10], y_lim=[y_min-y_dist/10, y_max+y_dist/10], plot_vertices=i==0, add_idx_edges=i==0,)
                    if i == 0:
                        first_region_edge.plot_single(ax=ax, linewidth=4)
                        ax.annotate(text='A', xy=first_region_edge.vtx1.xy)
                        ax.annotate(text='B', xy=first_region_edge.vtx2.xy)
                    if i == 1:
                        reg_without_itx.plot_single(ax=ax, color='green', alpha=0.5, hatch='/', plot_edges=False, x_lim=None, y_lim=None, plot_vertices=True)
                    if i == 2:
                        reg_itx.plot_single(ax=ax, color='red', alpha=0.5, plot_edges=False, x_lim=None, y_lim=None, plot_vertices=True)
                    GeoSeries([Point(intersection_edge.center).buffer(intersection_edge.radius, 40)]).plot(ax=ax, edgecolor='black', facecolor='None')
                    if i == 0:
                        ax.set_title("T st:"+str(start['touches'])+", end:"+str(end['touches'])+ 'i st:'+str(start['i'])+' i end:'+str(end['i'])+
                                     ' nedges'+str(len(self.edges))+'+'+str(include_first_in_without_itx) + str(first_region_edge.vtx2.xy == self.edges[start['i']].vtx1.xy) )
                    elif i == 1:
                        ax.set_title("n vtcs:"+str(len(reg_without_itx.vertices)))
                    elif i == 2:
                        ax.set_title("n vtcs:"+str(len(reg_itx.vertices))+'ef'+str(exclude_first_in_itx))

            self.delete()

        # create arc
        return 'orange'
   #

    def transform_to_trgl(self, i:int):
        """
        Transform region from triangle into triangle i
        """
        if i == 1: return self 
        new_edges = []
        for edge in self.edges:
            new_edges.append(edge.transform_to_trgl(i=i))
        
        
        if i % 2 == 1: 
            rotated_region = OffsetRegion(new_edges, checks=[], trgl_nr=i, all_regions=self.all_regions)
        else: 
            # reverse order for regions 2,4,6,8
            rotated_region = OffsetRegion(list(reversed(new_edges)), checks=[], trgl_nr=i, all_regions=self.all_regions)

        rotated_region.contained_cells = tuple(sorted([(x,y) for x,y in transform_cell_pattern(self.contained_cells, i)]))
        rotated_region.overlapped_cells = tuple(sorted([(x,y) for x,y in transform_cell_pattern(self.overlapped_cells, i)]))
        return rotated_region
    
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
            start = next((i for i in range(min(non_duplicated_ids), max(non_duplicated_ids)+1) if (i-1)%len(edges) not in non_duplicated_ids),None)
            end = next((i for i in range(min(non_duplicated_ids), max(non_duplicated_ids)+1) if (i+1)%len(edges) not in non_duplicated_ids), None)
            if start <= end:
                edges_merged_region.extend(edges[start:end+1])
            else:
                edges_merged_region.extend(edges[start:])
                edges_merged_region.extend(edges[:end+1])
            # check whether duplicate ids wrap around 
            
        merged_region = OffsetRegion(edges=edges_merged_region, checks=regions[0].checks, all_regions=region.all_regions, trgl_nr=min([reg.trgl_nr for reg in regions]))
        if not add_new:
            merged_region.delete()
        merged_region.contained_cells = regions[0].contained_cells
        merged_region.overlapped_cells = regions[0].overlapped_cells
        merged_region.nr = min([reg.nr for reg in regions])
        merged_region.trgl_nrs = [reg.trgl_nr for reg in regions]
        return merged_region
    #
#

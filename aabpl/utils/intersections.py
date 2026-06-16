# intersection of two circles with same radius
from numpy import (linspace as _np_linspace, array as _np_array, sign as _np_sign)
from math import (
    sin as _math_sin, cos as _math_cos, atan2 as _math_atan2, pi as _math_pi,
    acos as _math_acos, log10 as _math_log10,
    factorial as _math_factorial
)
from ..utils.misc import angle, angles_to_origin, angle_to
from decimal import Decimal as _decimal_Decimal, getcontext as _decimal_getcontext
# from mpmath import mp

def closest_pt_on_line_to_pt(line, pt,decimals:int=None):
    if decimals is None:
        (x1, y1), (x2, y2), (x3, y3) = *line, pt
    else:
        _decimal_getcontext().prec = decimals
        (x1, y1), (x2, y2), (x3, y3) = [(_decimal_Decimal(x),_decimal_Decimal(y)) for x,y in [*line, pt]]
    dx, dy = x2-x1, y2-y1
    det = dx*dx + dy*dy
    a = (dy*(y3-y1)+dx*(x3-x1))/det
    return x1+a*dx, y1+a*dy

def closest_pt_on_linesegment_to_pt(
        line, 
        pt,
        decimals:int=None, 
        tangent_tol=0.0,
        match_pt_on_segment:bool=False,
        match_pt_off_segment:bool=True,
        ):
    """
        
    """
    if decimals is None:
        (x1, y1), (x2, y2) = line
        e2 = .5
    else:
        _decimal_getcontext().prec = decimals
        (x1, y1), (x2, y2) = [(_decimal_Decimal(x),_decimal_Decimal(y)) for x,y in line]
        e2 = _decimal_Decimal(.5)
    
    x,y = closest_pt_on_line_to_pt(line, pt, decimals)

    # determine where on the line the point lies. use the axis on which there is more variation. 
    if abs(x2-x1)>abs(y2-y1):
        fraction_along_segment = (x-x1)/(x2-x1)
    else:
        fraction_along_segment = (y-y1)/(y2-y1)
    
    # if selected it will try to match the closest pt to the closest line segement endpoint if within distance tolerance
    if (match_pt_on_segment and 0<=fraction_along_segment<=1) or (match_pt_off_segment and fraction_along_segment<=0<1<=fraction_along_segment):
        if ((x-x1)**2+(y-y1)**2)**e2<=tangent_tol:
            if ((x-x2)**2+(y-y2)**2)**e2<((x-x1)**2+(y-y1)**2)**e2:
                return x2,y2
            return x1,y1
        if ((x-x2)**2+(y-y2)**2)**e2<=tangent_tol:
            return x2,y2
    
    # return closest pt that lies on the line segement
    if 0<=fraction_along_segment<=1:
        return x,y
    
    # return closest line segement endpoint
    if ((x-x2)**2+(y-y2)**2)**e2<((x-x1)**2+(y-y1)**2)**e2:
        return x2,y2
    return x1,y1
    

def enusure_zero_type(nums:tuple, tp:type=float):
    return type(nums)([tp(num) if num != 0 else tp(0.0) for num in nums])

def circle_line_segment_intersection(
        circle_center,
        circle_radius,
        pt1,
        pt2,
        full_line=False,
        tangent_tol:float=1e-14,
        rescale:int=1000000,
        stretch_factor:int=None,
        decimals:int=90,
        return_decimals:bool=False,
        ):
    """ Find the points at which a circle intersects a line-segment.  This can happen at 0, 1, or 2 points.

    :param circle_center: The (x, y) location of the circle center
    :param circle_radius: The radius of the circle
    :param pt1: The (x, y) location of the first point of the segment
    :param pt2: The (x, y) location of the second point of the segment
    :param full_line: True to find intersections along full line - not just in the segment.  False will just return intersections within the segment.
    :param tangent_tol: Numerical tolerance at which we decide the intersections are close enough to consider it a tangent
    :return Sequence[Tuple[float, float]]: A list of length 0, 1, or 2, where each element is a point at which the circle intercepts a line segment.

    Note: We follow: http://mathworld.wolfram.com/Circle-LineIntersection.html
    TAKEN FROM: https://stackoverflow.com/a/59582674
    """
    if not decimals is None:
        _decimal_getcontext().prec = decimals
        circle_center = [_decimal_Decimal(c) for c in circle_center]
        circle_radius = _decimal_Decimal(circle_radius)
        pt1 = [_decimal_Decimal(c) for c in pt1]
        pt2 = [_decimal_Decimal(c) for c in pt2]
        e2 = _decimal_Decimal(.5)
    else:
        e2 = .5 # this is needed as Decimal()**.5 will throw an error

    circle_radius = circle_radius*rescale
    (_p1x, _p1y) = [v*rescale for v in pt1]
    (_p2x, _p2y) = [v*rescale for v in pt2]
    (cx, cy) = [v*rescale for v in circle_center]
    
    # check if linesegment is only touching circle:
    x,y = closest_pt_on_linesegment_to_pt(
        line=((_p1x, _p1y),(_p2x, _p2y)), pt=(cx, cy), decimals=decimals, tangent_tol=tangent_tol
    )
    closest_pt_dist = ((x-cx)**2+(y-cy)**2)**e2
    if closest_pt_dist == circle_radius:
        intersections = [(x,y)]
        if decimals is None or return_decimals:
            return [(x/rescale,y/rescale) for x,y in intersections]
        return [(float(x/rescale),float(y/rescale)) for x,y in intersections]


    if stretch_factor is None:
        stretch_factor = -int(3*-circle_radius/((_p2x-_p1x)**2+(_p2y-_p1y)**2)**e2)
        
    (p1x, p1y) = (_p1x-stretch_factor*(_p2x-_p1x), _p1y-stretch_factor*(_p2y-_p1y))
    (p2x, p2y) = (_p2x-stretch_factor*(_p1x-_p2x), _p2y-stretch_factor*(_p1y-_p2y))
    (x1, y1), (x2, y2) = (p1x - cx, p1y - cy), (p2x - cx, p2y - cy)
    dx, dy = (x2 - x1), (y2 - y1)
    dr = (dx ** 2 + dy ** 2)**e2
    big_d = (x1 * y2 - x2 * y1)
    discriminant = (circle_radius ** 2 * dr ** 2 - big_d ** 2)

    
    if discriminant < 0:  # No intersection between circle and line

        if closest_pt_dist-circle_radius<=0:# tangent_tol
            intersections = [(x/rescale,y/rescale)]
            if decimals is None or return_decimals:
                return intersections
            return [(float(x),float(y)) for x,y in intersections]
        return []

    else:  # There may be 0, 1, or 2 intersections with the segment
        intersections = [
            (cx + (big_d * dy + sign * (-1 if dy < 0 else 1) * dx * discriminant**e2) / dr ** 2,
             cy + (-big_d * dx + sign * abs(dy) * discriminant**e2) / dr ** 2)
            for sign in ((1, -1) if dy < 0 else (-1, 1))]  # This makes sure the order along the segment is correct
        if not full_line:  # If only considering the segment, filter out intersections that do not fall within the segment
            # but allow for pts that are slightly outside of the line segment but within decimal limit to be matched to the start and end points of the line segment
            fraction_along_segment = [(xi - p1x) / dx if abs(dx) > abs(dy) else (yi - p1y) / dy for xi, yi in intersections]
            frac_lower, frac_upper = stretch_factor/(1+2*stretch_factor),(stretch_factor+1)/(1+2*stretch_factor)
            intersections = [
                # if p1 close enough to itx and closer than p2 to itx move itx onto p1
                (p1x,p1y) if (
                    ((pt[0]-p1x)**2+(pt[1]-p1y)**2)**e2 < tangent_tol*rescale*10 and 
                    ((pt[0]-p1x)**2+(pt[1]-p1y)**2)<((pt[0]-p2x)**2+(pt[1]-p2y)**2)
                    # else move itx onto p2 if p2 within distance tolerance
                ) else (p2x,p2y) if (
                    ((pt[0]-p2x)**2+(pt[1]-p2y)**2)**e2 < tangent_tol*rescale*10 
                ) else pt
                #   if 0 <= frac <= 1 
                #   else (p1x,p1y) if (
                    # ((pt[0]-p1x)**2+(pt[1]-p1y)**2) < ((pt[0]-p2x)**2+(pt[1]-p2y)**2) # if closer to x1,y1 than x2,y2
                # ) else (p2x,p2y)
                  for pt, frac 
                in zip(intersections, fraction_along_segment) if 
                    frac_lower <= frac <= frac_upper or # if on line segement/(1+2*stretch_factor)
                    ((pt[0]-_p1x)**2+(pt[1]-_p1y)**2)**e2 < tangent_tol*rescale*10 or # if close enough to x1,y1
                    ((pt[0]-_p2x)**2+(pt[1]-_p2y)**2)**e2 < tangent_tol*rescale*10  # if close enough to x2,y2
                ]
        
        intersections = [enusure_zero_type((x/rescale, y/rescale),type(x)) for x,y in intersections]
        
        if len(intersections) == 2 and abs(discriminant) <= tangent_tol*rescale:  # If line is tangent to circle, return just one point (as both intersections have same location)
            intersections = [((intersections[0][0]+intersections[1][0])/2, (intersections[0][1]+intersections[1][1])/2)]
        
        if decimals is None or return_decimals:
            return intersections
        return [(float(x),float(y)) for x,y in intersections]
#

def arc_line_segment_intersection(
        circle_center,
        circle_radius,
        pt1,
        pt2,
        angle_max,
        angle_min,
        full_line=False,
        tangent_tol:float=1e-14,
        rescale:int=1000000,
        decimals:int=90,
        return_decimals:bool=False,
):
    return filter_pts_on_arc(
        pts=circle_line_segment_intersection(
            circle_center=circle_center,
            circle_radius=circle_radius,
            pt1=pt1,
            pt2=pt2,
            full_line=full_line,
            tangent_tol=tangent_tol,
            rescale=rescale,
            decimals=decimals,
            return_decimals=return_decimals,
        ),
        arc_center=circle_center,
        arc_angle_min=angle_min,
        arc_angle_max=angle_max,
        tangent_tol=tangent_tol,
    )

def isBetween(a, b, c, tangent_tol:int=1e-15, rescale:float=1000000,):
    crossproduct = (
        (c[1]*rescale - a[1]*rescale) * (b[0]*rescale - a[0]*rescale) - 
        (c[0]*rescale - a[0]*rescale) * (b[1]*rescale - a[1]*rescale)
    )
    # compare versus epsilon for floating point values, or != 0 if using integers
    if abs(crossproduct) > tangent_tol*rescale:
        return False

    dotproduct = (
        (c[0]*rescale - a[0]*rescale) * (b[0]*rescale - a[0]*rescale) + 
        (c[1]*rescale - a[1]*rescale) * (b[1]*rescale - a[1]*rescale)
    )
    if dotproduct < 0:
        return False

    squaredlengthba = (
        (b[0]*rescale - a[0]*rescale) * (b[0]*rescale - a[0]*rescale) + 
        (b[1]*rescale - a[1]*rescale) * (b[1]*rescale - a[1]*rescale)
    )
    if dotproduct > squaredlengthba:
        return False

    return True
#

def det(a, b):
    return a[0] * b[1] - a[1] * b[0]
#

def line_intersection(line1, line2, tangent_tol:float=1e-14, rescale:int=1000000,decimals:int=None, return_decimals:bool=False):
    """
    Returns list of length zero if two input line segments do not intersect
    otherwise returns a list of length one containing the coordinate-tuple where lines intersect  
    """
    line1, line2 = [[(x*rescale, y*rescale) for x,y in line] for line in [line1, line2]]
    if not decimals is None:
        _decimal_getcontext().prec = decimals
        line1 = [(_decimal_Decimal(x), _decimal_Decimal(y)) for x,y in line1]
        line2 = [(_decimal_Decimal(x), _decimal_Decimal(y)) for x,y in line2]
        e2 = _decimal_Decimal(.5)
    else:
        e2 = .5
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    div = det(xdiff, ydiff)
    if div == 0:
       return []
    d = (det(*line1), det(*line2))
    x,y = (det(d, xdiff) / div, det(d, ydiff) / div)
    if x==0:
        x = type(x)(0.)
    if y==0:
        y = type(y)(0.)

    if not isBetween(line1[0], line1[1], (x,y), tangent_tol=tangent_tol):
        return []
    
    # match intersections on to line segment endpoints if within precision tolerance
    tolerance = tangent_tol*rescale
    if (((x-line1[0][0])**2+(y-line1[0][1])**2)**e2) < tolerance:
        # print("match",x,y,line1[0])
        x,y = line1[0]
    elif (((x-line1[1][0])**2+(y-line1[1][1])**2)**e2) < tolerance:
        # print("match",x,y,line1[1])
        x,y = line1[1]
    elif (((x-line2[0][0])**2+(y-line2[0][1])**2)**e2) < tolerance:
        # print("match",x,y,line2[0])
        x,y = line2[0]
    elif (((x-line2[1][0])**2+(y-line2[1][1])**2)**e2) < tolerance:
        # print("match",x,y,line2[1])
        x,y = line1[1]
    
    if decimals is None or return_decimals:
        return [(x/rescale,y/rescale)]
    return [(float(x/rescale),float(y/rescale))]
    
    # return [(round(x,precision),round(y,precision))]
#

def filter_pts_on_arc(
        pts,
        arc_center,
        arc_angle_min:float=0,
        arc_angle_max:float=360,
        tangent_tol:float=1e-14,
        rescale:int=1000000,
):
    """
    Filter points on circle to those within angle_min and angle_max 
    """
    arc_center= [v*rescale for v in arc_center]
    pts = [(x*rescale,y*rescale) for x,y in pts]
    return [
        (x/rescale,y/rescale) for (x,y), ngl in zip(pts, angles_to_origin(pts, arc_center)) 
        if arc_angle_min - tangent_tol <= ngl and ngl <= arc_angle_max + tangent_tol # maybe use tangent tol for distance between pts at arc edge and these pts
    ]
#

def  intersections_pts_two_circles_same_radius(
        center_1,
        center_2,
        r:float,
        # tangent_tol:int=13,
        rescale:int=1000000,
        decimals:int=None, 
        return_decimals:bool=False,
        ):
    """
    Returns list intersections points of two circle with same radius
    """
    if not decimals is None:
        _decimal_getcontext().prec = decimals
        circle_1_x, circle_1_y = [_decimal_Decimal(c) for c in center_1]
        circle_2_x, circle_2_y = [_decimal_Decimal(c) for c in center_2]
        r = _decimal_Decimal(r)
        e2 = _decimal_Decimal(.5)
    else:
        circle_1_x, circle_1_y = center_1
        circle_2_x, circle_2_y = center_2
        e2 = .5
    r = r*rescale
    circle_1_x = circle_1_x*rescale
    circle_1_y = circle_1_y*rescale
    circle_2_x = circle_2_x*rescale
    circle_2_y = circle_2_y*rescale

    dist = ((circle_1_x-circle_2_x)**2 + (circle_1_y-circle_2_y)**2)**e2
    if dist > 2*r:
        return []
    if dist == 2*r:
        itx_coords = [(
            (circle_1_x + circle_2_x) / 2, 
            (circle_1_y + circle_2_y) / 2,
            )]
    else:
        if False:
            pass
            # mp.dps = 16
            
            # alpha = _math_acos(dist/2/r)
            # slope_angle = angle(circle_1_x, circle_1_y, circle_2_x, circle_2_y)
            # angle_itx1 = alpha-slope_angle
            # angle_itx2 = mp.pi*2-slope_angle-alpha
            # itx_coords = [
            #     (
            #         float(r*mp.cos(angle_itx1) + circle_1_x),
            #         float(r*mp.sin(angle_itx1)+ circle_1_y),
            #     ), (
            #             float(r*mp.cos(angle_itx2) + circle_1_x),
            #             float(r*mp.sin(angle_itx2) + circle_1_y),
            #     )
            # ]
        else:
            if not decimals is None:
                alpha = _decimal_Decimal(_math_acos(dist/2/r))
                slope_angle = _decimal_Decimal(angle(circle_1_x, circle_1_y, circle_2_x, circle_2_y))
                angle_itx1 = alpha-slope_angle
                angle_itx2 = _decimal_Decimal(_math_pi*2)-slope_angle-alpha
                itx_coords = [
                    (
                        r*_decimal_Decimal(_math_cos(angle_itx1)) + circle_1_x,
                        r*_decimal_Decimal(_math_sin(angle_itx1)) + circle_1_y,
                    ), (
                            r*_decimal_Decimal(_math_cos(angle_itx2)) + circle_1_x,
                            r*_decimal_Decimal(_math_sin(angle_itx2)) + circle_1_y,
                    )
                ]
            else:
                alpha = _math_acos(dist/2/r)
                slope_angle = angle(circle_1_x, circle_1_y, circle_2_x, circle_2_y)
                angle_itx1 = alpha-slope_angle
                angle_itx2 = _math_pi*2-slope_angle-alpha
                itx_coords = [
                    (
                        r*_math_cos(angle_itx1) + circle_1_x,
                        r*_math_sin(angle_itx1)+ circle_1_y,
                    ), (
                            r*_math_cos(angle_itx2) + circle_1_x,
                            r*_math_sin(angle_itx2) + circle_1_y,
                    )
                ]
    itx_coords = [tuple([e/rescale if e != 0 else 0. for e in el]) for el in itx_coords] # ensure -0.0 is +0.0
    
    if decimals is None or return_decimals:
        return itx_coords
    return [(float(x),float(y)) for x,y  in itx_coords]
# 

def intersections_pts_arc_to_circle(
        circle_center,
        arc_center,
        r:float,
        arc_angle_min:float=0,
        arc_angle_max:float=360,
        tangent_tol:int=13,
        rescale=1000000,
        decimals:int=None, 
        return_decimals:bool=False,
        ):
    """
    Get intersections points of arc and circle (filtering out points that are not on arc)
    """
    # pts = intersections_pts_two_circles_same_radius(
    #         center_1=circle_center,
    #         center_2=arc_center,
    #         r=r,
    #     )
    # filtered_pts = filter_pts_on_arc(
    #     pts=pts,
    #     arc_center=arc_center,
    #     arc_angle_min=arc_angle_min,
    #     arc_angle_max=arc_angle_max,
    #     tangent_tol=tangent_tol,
    # )
    # if len(pts) != len(filtered_pts):
    #     pass
    #     print("FILTER",len(pts), len(filtered_pts), (arc_angle_min, arc_angle_max, ), pts)
    
    return filter_pts_on_arc(
        pts=intersections_pts_two_circles_same_radius(
            center_1=circle_center,
            center_2=arc_center,
            r=r,
            rescale=rescale,
            decimals=decimals, 
            return_decimals=return_decimals,
        ),
        arc_center=arc_center,
        arc_angle_min=arc_angle_min,
        arc_angle_max=arc_angle_max,
        tangent_tol=tangent_tol,
        rescale=rescale,
    )
#
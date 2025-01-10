# intersection of two circles with same radius
from numpy import (linspace as _np_linspace, array as _np_array, sign as _np_sign)
from math import sin as _math_sin, cos as _math_cos, atan2 as _math_atan2, pi as _math_pi, acos as _math_acos , sin as _math_asin, log10 as _math_log10
from primelocations.utils.general import angle, angles_to_origin, angle_to



def enusure_zero_type(nums:tuple, tp:type=float):
    return type(nums)([num if num != 0 else tp(num) for num in nums])

def circle_line_segment_intersection(circle_center, circle_radius, pt1, pt2, full_line=True, precision:int=13):
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
    tangent_tol = 10**-(precision)
    (p1x, p1y), (p2x, p2y), (cx, cy) = pt1, pt2, circle_center
    (x1, y1), (x2, y2) = (p1x - cx, p1y - cy), (p2x - cx, p2y - cy)
    dx, dy = (x2 - x1), (y2 - y1)
    dr = (dx ** 2 + dy ** 2)**.5
    big_d = (x1 * y2 - x2 * y1)
    discriminant = (circle_radius ** 2 * dr ** 2 - big_d ** 2)

    if discriminant < 0:  # No intersection between circle and line
        return []
    else:  # There may be 0, 1, or 2 intersections with the segment
        intersections = [
            (cx + (big_d * dy + sign * (-1 if dy < 0 else 1) * dx * discriminant**.5) / dr ** 2,
             cy + (-big_d * dx + sign * abs(dy) * discriminant**.5) / dr ** 2)
            for sign in ((1, -1) if dy < 0 else (-1, 1))]  # This makes sure the order along the segment is correct
        if not full_line:  # If only considering the segment, filter out intersections that do not fall within the segment
            fraction_along_segment = [(xi - p1x) / dx if abs(dx) > abs(dy) else (yi - p1y) / dy for xi, yi in intersections]
            intersections = [pt for pt, frac in zip(intersections, fraction_along_segment) if 0 <= frac <= 1]
        intersections = [enusure_zero_type((round(x, precision), round(y, precision)),float) for x,y in intersections]
        if len(intersections) == 2 and abs(discriminant) <= tangent_tol:  # If line is tangent to circle, return just one point (as both intersections have same location)
            return [intersections[0]]
        else:
            return intersections
#

def isBetween(a, b, c, precision:int=13):
    crossproduct = (c[1] - a[1]) * (b[0] - a[0]) - (c[0] - a[0]) * (b[1] - a[1])

    # compare versus epsilon for floating point values, or != 0 if using integers
    if abs(crossproduct) > 10**-precision:
        return False

    dotproduct = (c[0] - a[0]) * (b[0] - a[0]) + (c[1] - a[1])*(b[1] - a[1])
    if dotproduct < 0:
        return False

    squaredlengthba = (b[0] - a[0])*(b[0] - a[0]) + (b[1] - a[1])*(b[1] - a[1])
    if dotproduct > squaredlengthba:
        return False

    return True
#

def det(a, b):
    return a[0] * b[1] - a[1] * b[0]
#

def line_intersection(line1, line2, precision:int=13):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    div = det(xdiff, ydiff)
    if div == 0:
       return []
    d = (det(*line1), det(*line2))
    x,y = (det(d, xdiff) / div, det(d, ydiff) / div)
    if x==0:
        x = 0.
    if y==0:
        y = 0.
    if not isBetween(line1[0], line1[1], (x,y), precision=precision):
        return []
    
    return [(x,y)]
#

def filter_pts_on_arc(
        pts,
        arc_center,
        arc_angle_min:float=0,
        arc_angle_max:float=360,
        precision:int=13,
):
    """
    Filter points on circle to those within angle_min and angle_max 
    """
    return [
        pt for pt, ngl in zip(pts, angles_to_origin(pts, arc_center)) 
        if arc_angle_min - 1*10**-(precision + 0) <= ngl and ngl <= arc_angle_max + 1*10**-(precision + 0)
    ]
#

def  intersections_pts_two_circles_same_radius(
        center_1,
        center_2,
        radius:float,
        precision:int=13
        ):
    """
    Returns list intersections points of two circle with same radius
    """
    circle_1_x, circle_1_y = center_1
    circle_2_x, circle_2_y = center_2

    dist = ((circle_1_x-circle_2_x)**2 + (circle_1_y-circle_2_y)**2)**.5
    if dist > 2*radius:
        return []
    if dist == 2*radius:
        return [(
            round((circle_1_x + circle_2_x) / 2, precision), 
            round((circle_1_y + circle_2_y) / 2, precision),
            )]

    r = radius
    alpha = _math_acos(dist/2/r)
    slope_angle = angle(circle_1_x, circle_1_y, circle_2_x, circle_2_y)
    angle_itx1 = alpha-slope_angle
    angle_itx2 = _math_pi*2-slope_angle-alpha
    itx_coords = [
        (
            round(r*_math_cos(angle_itx1) + circle_1_x, precision),
            round(r*_math_sin(angle_itx1)+ circle_1_y, precision),
        ), (
            round(r*_math_cos(angle_itx2) + circle_1_x, precision),
            round(r*_math_sin(angle_itx2) + circle_1_y, precision),
        )
    ]
    itx_coords = [tuple([e if e != 0 else 0. for e in el]) for el in itx_coords] # ensure -0.0 is +0.0
    
    return itx_coords
# 

def intersections_pts_arc_to_circle(
        circle_center,
        arc_center,
        radius:float,
        arc_angle_min:float=0,
        arc_angle_max:float=360,
        precision:int=10,
        ):
    """
    Get intersections points of arc and circle (filtering out points that are not on arc)
    """
    # pts = intersections_pts_two_circles_same_radius(
    #         center_1=circle_center,
    #         center_2=arc_center,
    #         radius=radius,
    #     )
    # filtered_pts = filter_pts_on_arc(
    #     pts=pts,
    #     arc_center=arc_center,
    #     arc_angle_min=arc_angle_min,
    #     arc_angle_max=arc_angle_max,
    #     precision=precision,
    # )
    # if len(pts) != len(filtered_pts):
    #     pass
    #     print("FILTER",len(pts), len(filtered_pts), (arc_angle_min, arc_angle_max, ), pts)
    
    return filter_pts_on_arc(
        pts=intersections_pts_two_circles_same_radius(
            center_1=circle_center,
            center_2=arc_center,
            radius=radius,
            precision=13,
        ),
        arc_center=arc_center,
        arc_angle_min=arc_angle_min,
        arc_angle_max=arc_angle_max,
        precision=precision,
    )
#
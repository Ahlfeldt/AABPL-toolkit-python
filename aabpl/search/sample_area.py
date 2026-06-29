from numpy import (
    array as _np_array, arange as _np_arange,
    exp as _np_exp,
    unique, linspace, invert, flip, transpose, concatenate, sign, zeros,
    min as _np_min, max as _np_max, equal, where, logical_or, logical_and, all, newaxis,
    mgrid as _np_mgrid)
from math import inf as _math_inf, pi as _math_pi, acos as _math_acos, sin as _math_sin, log2 as _math_log2
from shapely.geometry import (
    Polygon as _shapely_Polygon,
    MultiPoint as _shapely_MultiPoint,
    MultiPolygon as _shapely_MultiPolygon,
    GeometryCollection as _shapely_GeometryCollection
)
from shapely.ops import unary_union as _shapely_unary_union
from pandas import DataFrame as _pd_DataFrame
from concave_hull import concave_hull
from matplotlib import pyplot as plt
from matplotlib.patches import (Rectangle as _plt_Rectangle, Polygon as _plt_Polygon, Circle as _plt_Circle)
from aabpl.illustrations.plot_utils import plot_polygon
from aabpl.utils.misc import count_polygon_interiors
from aabpl.utils.crs_transformation import convert_pts_to_crs as _convert_pts_to_crs
from aabpl.utils.intersections import circle_line_segment_intersection
from aabpl.utils.cell_geometry import (classify_disk_cells, get_cells_by_lvl_ovlpd_by_cell_buffer)
from aabpl.testing.test_performance import time_func_perf
from aabpl.utils.progress import progress_print
# TODO make individual function for each hull type?
# TODO make clever use of buffered cells s.t. polygon check is only performed in edge cases

# Define the Polygon and the cutting line

def cell_to_polygon(
        x_steps:list=[],
        y_steps:list=[],
):
    return _shapely_Polygon([
                (x_steps[0], y_steps[0]),
                (x_steps[-1], y_steps[0]),
                (x_steps[-1], y_steps[-1]),
                (x_steps[0], y_steps[-1]),
            ])

def clip_polygon_to_grid(
        poly, 
        cell_to_poly:dict={},
        lvls_to_store:list=True, 
        min_steps:int=1,
        lvl:int=None,
        x_steps:list=[],
        y_steps:list=[],
        store_full_poly:bool=True,
        row:int=0,
        col:int=0
        ) -> dict:
    """
    Split (Multi-)Polygon along grid. Store splitted polygons in dict that is returned.
    Dict contains only cells that contain a part of polygon
    """
    if lvl is None:
        # crop initial shape to grid extent
        if hasattr(poly, 'intersection'):
            poly=poly.intersection(cell_to_polygon(x_steps, y_steps)) 
    #
        lvl = min(int((-_math_log2(len(x_steps)-1)//1)), int((-_math_log2(len(y_steps)-1)//1)))

    if type(poly) is bool and poly==True:
        pass
    elif type(poly) in [_shapely_MultiPolygon, _shapely_GeometryCollection]:
        for p in poly.geoms:
            # print("type p",type(p),lvl)
            clip_polygon_to_grid(
                poly=p, 
                cell_to_poly=cell_to_poly,
                lvls_to_store=lvls_to_store,
                min_steps=min_steps, 
                lvl=lvl, 
                x_steps=x_steps, 
                y_steps=y_steps, 
                store_full_poly=store_full_poly,
                row=row, 
                col=col
                )
        return cell_to_poly
    elif type(poly) is not _shapely_Polygon:
        return cell_to_poly
    elif poly.is_empty:
        return cell_to_poly
    # elif poly.area >= spacing**2*2**-(2*lvl): split_full_cells_along_grid()
    # print("len x_steps",len(x_steps), "len y_steps", len(y_steps), "split_x",split_x,"split_y",split_y,"lvl",lvl,"row",row,"col",col)
    minx, maxx = x_steps[0], x_steps[-1]
    miny, maxy = y_steps[0], y_steps[-1]
    
    lvl_x = int((-_math_log2(len(x_steps)-1)//1))
    lvl_y = int((-_math_log2(len(y_steps)-1)//1))
    split_x = int(len(x_steps)>min_steps+1 and lvl_x <= lvl_y and lvl_x <= lvl)
    split_y = int(len(y_steps)>min_steps+1 and lvl_y <= lvl_x and lvl_y <= lvl)
    # x_mid_id = (len(x_steps)//2)
    # y_mid_id = (len(y_steps)//2)
    if split_x:
        x_mid_id = 2**(-int((-_math_log2(len(x_steps)-1)//1))-1)
        midx = x_steps[x_mid_id]
    if split_y:
        y_mid_id = 2**(-int((-_math_log2(len(y_steps)-1)//1))-1)
        midy = y_steps[y_mid_id]
    
    # also dont split_x if len(y_steps)>2*len(x_steps) is 
    
    if split_x or split_y:
        if split_x:
            x_starts, x_stops = [0,x_mid_id], [x_mid_id+1, len(x_steps)]
            x_mins, x_maxs = [minx, midx], [midx, maxx]
        else:
            x_starts, x_stops = [0], [len(x_steps)]
            x_mins, x_maxs = [minx], [maxx]
        if split_y:
            y_starts, y_stops = [0,y_mid_id], [y_mid_id+1, len(y_steps)]
            y_mins, y_maxs = [miny, midy], [midy, maxy]
        else:
            y_starts, y_stops = [0], [len(y_steps)]
            y_mins, y_maxs = [miny], [maxy]

        for x_start, x_stop, x_min, x_max in zip(x_starts, x_stops, x_mins, x_maxs):
            for y_start, y_stop, y_min, y_max in zip(y_starts, y_stops, y_mins, y_maxs):
                if type(poly) is bool and poly==True:
                    intersection_poly = poly
                else:
                    cell_poly = cell_to_polygon([x_min,x_max], [y_min, y_max])
                    intersection_poly = cell_poly.intersection(poly)
                    if cell_poly.area <= intersection_poly.area:
                        intersection_poly = True
                if not store_full_poly and type(intersection_poly) is bool and intersection_poly is True:
                    cell_to_poly[(lvl+1,(row+y_start,col+x_start))] = [intersection_poly]
                    continue
                
                clip_polygon_to_grid(
                    poly=intersection_poly,
                    cell_to_poly=cell_to_poly,
                    min_steps=min_steps,
                    lvl=lvl+1, 
                    store_full_poly=store_full_poly,
                    lvls_to_store=lvls_to_store,
                    x_steps=x_steps[x_start:x_stop],
                    y_steps=y_steps[y_start:y_stop],
                    row=row+y_start,
                    col=col+x_start
                )
                

        lvl_i = min(lvl_x, lvl_y, lvl)
        if ((type(lvls_to_store)==bool and lvls_to_store) or 
            (type(lvls_to_store) in [list,tuple] and lvl_i in lvls_to_store)):
            if store_full_poly and type(poly) is bool and poly is True:
                poly = cell_to_polygon(x_steps, y_steps)
            if (lvl,(row,col)) not in cell_to_poly:
                cell_to_poly[(lvl_i,(row,col))] = [poly]
            else: 
                cell_to_poly[(lvl_i,(row,col))].append(poly)

    # elif min(lvl_x,lvl_y) <= lvl:
        # pass

    else:
        # for lvl_i in [lvl]:
        if store_full_poly and type(poly) is bool and poly is True:
            poly = cell_to_polygon(x_steps, y_steps)

        for lvl_i in range(lvl, max([lvl_x, lvl_y, lvl])+1):
            if (lvl_i,(row,col)) not in cell_to_poly:
                cell_to_poly[(lvl_i,(row,col))] = [poly]
            else: 
                cell_to_poly[(lvl_i,(row,col))].append(poly)
            #
        #
        if min(lvl_x,lvl_y) > lvl:
            clip_polygon_to_grid(
                    poly=poly,
                    cell_to_poly=cell_to_poly,
                    min_steps=min_steps,
                    lvl=lvl+1, 
                    store_full_poly=store_full_poly,
                    lvls_to_store=lvls_to_store,
                    x_steps=x_steps,
                    y_steps=y_steps,
                    row=row,
                    col=col
                )
    #
    
    return cell_to_poly

@time_func_perf
def intersect_polygon_with_grid(
        grid,
        rel_tol = 0.00000,

):
    _si = grid._search_internals
    grid.sample_col_min = sample_col_min = int((grid.sample_area.bounds[0] - _si.bounds.xmin) // _si.spacing)
    grid.sample_row_min = sample_row_min = int((grid.sample_area.bounds[1] - _si.bounds.ymin) // _si.spacing)
    grid.sample_col_max = sample_col_max = int((grid.sample_area.bounds[2] - _si.bounds.xmin) // _si.spacing)
    grid.sample_row_max = sample_row_max = int((grid.sample_area.bounds[3] - _si.bounds.ymin) // _si.spacing)

    col_min = int(_si.col_ids.min())
    row_min = int(_si.row_ids.min())
    col_max = int(_si.col_ids.max())
    row_max = int(_si.row_ids.max())
    _sp = _si.spacing
    _x0, _y0 = _si.bounds.xmin, _si.bounds.ymin
    _x1, _y1 = _si.bounds.xmax, _si.bounds.ymax
    _col_lo, _col_hi = col_min, col_max
    _row_lo, _row_hi = row_min, row_max
    grid.sample_x_steps = concatenate([
        _np_arange(_x0 + (sample_col_min - _col_lo) * _sp, _x0, _sp) if sample_col_min < _col_lo else _np_array([]),
        _si.x_steps,
        _np_arange(_x1 + _sp, _x0 + (sample_col_max + 1) * _sp + _sp * 0.5, _sp) if sample_col_max > _col_hi else _np_array([]),
    ])
    grid.sample_y_steps = concatenate([
        _np_arange(_y0 + (sample_row_min - _row_lo) * _sp, _y0, _sp) if sample_row_min < _row_lo else _np_array([]),
        _si.y_steps,
        _np_arange(_y1 + _sp, _y0 + (sample_row_max + 1) * _sp + _sp * 0.5, _sp) if sample_row_max > _row_hi else _np_array([]),
    ])
    grid.update_spacing(recompute=True)
    # Classify output cells now that the grid has been extended to cover the
    # sample area.  Using x_steps_bounds[0] as the anchor here (post-extension)
    # ensures null_distribution uses the same anchor as this classification.
    _ox = grid.x_steps_bounds[0]
    _oy = grid.y_steps_bounds[0]
    _sx = grid.cell_size
    _sy = grid.cell_size_y
    _bx0, _by0, _bx1, _by1 = grid.sample_area.bounds
    from math import floor as _mfloor, ceil as _mceil
    _ocmin = max(0, _mfloor((_bx0 - _ox) / _sx))
    _ormin = max(0, _mfloor((_by0 - _oy) / _sy))
    _ocmax = _mceil((_bx1 - _ox) / _sx)
    _ormax = _mceil((_by1 - _oy) / _sy)
    _out_fully = set()
    _out_partly = set()
    try:
        from shapely import box as _shapely_box
        from shapely import contains_properly as _shapely_contains_properly
        from shapely import intersects as _shapely_intersects
        _R, _C = _np_mgrid[_ormin:_ormax+1, _ocmin:_ocmax+1]
        _rf, _cf = _R.ravel(), _C.ravel()
        _CHUNK = 500_000
        for _i in range(0, len(_rf), _CHUNK):
            _sl = slice(_i, _i + _CHUNK)
            _rf_c, _cf_c = _rf[_sl], _cf[_sl]
            _boxes = _shapely_box(
                _ox + _cf_c * _sx, _oy + _rf_c * _sy,
                _ox + (_cf_c + 1) * _sx, _oy + (_rf_c + 1) * _sy,
            )
            _fully = _shapely_contains_properly(grid.sample_area, _boxes)
            _partly = ~_fully & _shapely_intersects(grid.sample_area, _boxes)
            _out_fully.update(zip(_rf_c[_fully].tolist(), _cf_c[_fully].tolist()))
            _out_partly.update(zip(_rf_c[_partly].tolist(), _cf_c[_partly].tolist()))
    except ImportError:
        from shapely.prepared import prep as _prep
        _prep_area = _prep(grid.sample_area)
        for _r in range(_ormin, _ormax + 1):
            for _c in range(_ocmin, _ocmax + 1):
                _cp = _shapely_Polygon([
                    (_ox + _c * _sx,       _oy + _r * _sy),
                    (_ox + (_c+1) * _sx,   _oy + _r * _sy),
                    (_ox + (_c+1) * _sx,   _oy + (_r+1) * _sy),
                    (_ox + _c * _sx,       _oy + (_r+1) * _sy),
                ])
                if _prep_area.contains(_cp):
                    _out_fully.add((_r, _c))
                elif _prep_area.intersects(_cp):
                    _out_partly.add((_r, _c))
    grid.output_cells_fully_valid = _out_fully
    grid.output_cells_partly_valid = _out_partly
    grid._output_sample_spacing_x = _sx
    grid._output_sample_spacing_y = _sy
    # grid.sample_x_steps = _np_array(
    #     [grid.total_bounds.xmin - (col_min-col)*grid._search_spacing for col in range(sample_col_min, col_min)] +  
    #     list(grid._search_x_steps) + 
    #     [grid.total_bounds.xmax + (col-col_max)*grid._search_spacing for col in range(max(grid._search_col_ids),col_max+1)]
    # )
    # grid.sample_y_steps = _np_array(
    #     [grid.total_bounds.ymin - (row_min-row)*grid._search_spacing for row in range(sample_row_min, row_min)] +  
    #     list(grid._search_y_steps) + 
    #     [grid.total_bounds.ymax + (row-row_max)*grid._search_spacing for row in range(max(grid._search_row_ids),row_max+1)]
    # )
    grid.sample_col_ids = range(col_min,col_max+1)
    grid.sample_row_ids = range(row_min,row_max+1)
    grid.sample_grid_bounds = [
        _si.bounds.xmin + sample_col_min * _si.spacing,
        _si.bounds.ymin + sample_row_min * _si.spacing,
        _si.bounds.xmin + (sample_col_max+1) * _si.spacing,
        _si.bounds.ymin + (sample_row_max+1) * _si.spacing,
    ]
    
    

    lvls_to_store = True

    cell_to_poly = clip_polygon_to_grid(
        poly=grid.sample_area,
        cell_to_poly={},
        lvls_to_store=lvls_to_store,
        min_steps=1, 
        # lvl=None, 
        store_full_poly=False, # change to False. 
        x_steps=grid.sample_x_steps, 
        y_steps=grid.sample_y_steps, 
        row=grid.sample_row_min,
        col=grid.sample_col_min,
    )
    #clip_polygon_to_grid + split into subcells? 
    # unary union of polygons
    
    used_poly_lvls = sorted(set(lvl for lvl,(row,col) in cell_to_poly if lvls_to_store==True or lvl in lvls_to_store))
    min_lvl = int(min(used_poly_lvls))
    max_lvl = int(max(used_poly_lvls))
    abs_tol = rel_tol * (grid._search_internals.spacing*2**-max_lvl)**2
    # method to obtain poky for fully valid cells from lvl, row, col
    def get_fully_valid_poly(
            lvl:int,
            row,
            col, 
            sample_x_steps=grid.sample_x_steps, 
            sample_y_steps=grid.sample_y_steps, 
            max_lvl:int=max_lvl, 
            sample_row_min=grid.sample_row_min, 
            sample_col_min=grid.sample_col_min) -> dict:
        """
        Creates fully valid poly for given cell by lvl, row, col
        """
        return cell_to_polygon(
            x_steps=sample_x_steps[col-sample_col_min:col-sample_col_min+2**(max_lvl-lvl)+1],
            y_steps=sample_y_steps[row-sample_row_min:row-sample_row_min+2**(max_lvl-lvl)+1],
        )
    grid.get_fully_valid_poly = get_fully_valid_poly

    # create unary union for cells with multiple polys
    cell_to_poly = {
        k:_shapely_unary_union(v) if len(v)!=1 else v[0]
        for k,v in cell_to_poly.items()
        }
    # extract fully valid and partly valid cells
    cells_fully_valid  ={
        (lvl, (row,col)) for (lvl, (row,col)), poly in cell_to_poly.items() 
        if lvl in used_poly_lvls and (
        (type(poly) is bool and poly is True) or
        poly.area >= (grid._search_internals.spacing*2**-lvl)**2 - abs_tol
        )}
    cells_partly_valid_max_lvl = {
        (lvl, (row,col)) for (lvl, (row,col)), poly in cell_to_poly.items() 
        if lvl == max_lvl and not type(poly) is bool and
        (grid._search_internals.spacing*2**-lvl)**2 - abs_tol > poly.area > abs_tol
        } 
    #

    # function to get cell_to_poly with fully valid polys instead of True for fully valid cells
    def get_cell_to_poly(grid=grid, bind_to_grid:bool=True):
        """
        get cell_to_poly dict with fully valid polys instead of True for fully valid cells
        
        :param grid: Grid
        :param bind_to_grid: whether to bind the result to the grid object
        """
        cell_to_poly = {
            (lvl,(row,col)): 
                get_fully_valid_poly(lvl,row,col) if type(v) is bool and v is True else v 
            for (lvl,(row,col)),v in grid._search_internals.cell_to_poly.items()
            }
        if bind_to_grid:
            grid._search_internals.cell_to_poly = cell_to_poly
        return cell_to_poly
    grid.get_cell_to_poly = get_cell_to_poly
    #

    # ensure that no subcell is part of sample_cells if parent cell is part of sample_cells
    
    
    # if store_full_poly==False then this will only remove cells that are now considered fully valid after removing precision issues.
    sample_cells = cells_fully_valid.union(cells_partly_valid_max_lvl) 
    # work from most detailed level to most aggregated
    for lvl in range(min_lvl, max_lvl+1)[::-1]:
        cell_parent = (min_lvl,(0,0))
        for row,col in [(row,col) for lvl_i,(row,col) in sample_cells if lvl_i==lvl]:
            cell_parent_included = False

            row_rounded, col_rounded = int(round(row)), int(round(col))
            row_remainder, col_remainder = (row+.5)%1, (col+.5)%1
            # for lvl_j in range(min_lvl, lvl)[::-1]:
            #     if lvl_j < 0:
            #         row_lvl_lookup, col_lvl_lookup = row_rounded//(2**lvl_j), col_rounded//(2**lvl_j)
            #     elif lvl_j == 0:
            #         row_lvl_lookup, col_lvl_lookup = row_rounded, col_rounded
            #     else:
            #         row_remainder = (row_remainder//2**-lvl //2 *2+1)*2**-lvl
            #         col_remainder = (col_remainder//2**-lvl //2 *2+1)*2**-lvl
            #         row_lvl_lookup, col_lvl_lookup = row_rounded+row_remainder-.5, col_rounded+col_remainder-.5
            #     cell_parent = (lvl_j, (row_lvl_lookup, col_lvl_lookup))
            #     cell_parent_in_fully_valid = cell_parent in cells_fully_valid
            #     cell_parent_in_partly_valid = cell_parent in cells_partly_valid
            #     cell_parent_included = cell_parent in sample_cells
            #     if cell_parent_included:
            #         break
            if lvl-1 < 0:
                # find next row, col in parent lvl which means the next smaller value out of all possible parent cell values
                parent_lvl_row_ids = [x for x in range(grid.sample_row_min, grid.sample_row_max+1) if (x-grid.sample_row_min)%(2**-(lvl-1))==0][::-1]
                parent_lvl_col_ids = [x for x in range(grid.sample_col_min, grid.sample_col_max+1) if (x-grid.sample_col_min)%(2**-(lvl-1))==0][::-1]
                row_lvl_lookup  = next((x for x in parent_lvl_row_ids if x<=row_rounded), parent_lvl_row_ids[0])
                col_lvl_lookup  = next((x for x in parent_lvl_col_ids if x<=col_rounded), parent_lvl_col_ids[0])               
                # row_lvl_lookup, col_lvl_lookup = int(row_rounded//(2**(lvl-1))), int(col_rounded//(2**(lvl-1)))
            elif (lvl-1) == 0:
                row_lvl_lookup, col_lvl_lookup = row_rounded, col_rounded
            else:
                row_remainder = (row_remainder//2**-lvl //2 *2+1)*2**-lvl
                col_remainder = (col_remainder//2**-lvl //2 *2+1)*2**-lvl
                row_lvl_lookup, col_lvl_lookup = row_rounded+row_remainder-.5, col_rounded+col_remainder-.5
            cell_parent = ((lvl-1), (row_lvl_lookup, col_lvl_lookup))
            cell_parent_in_fully_valid = cell_parent in cells_fully_valid
            cell_parent_in_partly_valid = cell_parent in cells_partly_valid_max_lvl
            cell_parent_included = cell_parent in sample_cells
         
            if cell_parent_in_fully_valid:
                if (lvl, (row, col)) in cells_fully_valid:
                    cells_fully_valid.remove((lvl, (row, col)))
                if (lvl, (row, col)) in cells_partly_valid_max_lvl:
                    cells_partly_valid_max_lvl.discard((lvl, (row, col)))
            
            # if cell_parent_in_partly_valid:
            #     cells_partly_valid.remove(cell_parent)

            # else keep cell


    grid._search_internals.cell_to_poly = cell_to_poly
    grid._search_internals.cell_to_poly_partly_valid = {cell:poly for cell, poly in cell_to_poly.items() if cell in cells_partly_valid_max_lvl}
    grid._search_internals.cells_fully_valid = cells_fully_valid
    grid._search_internals.cells_partly_valid_max_lvl = set(cells_partly_valid_max_lvl)
    
    # if area_complexity>500:
    #     print('WARNING: The Polygon defining the valid area is complex (n edges=' +
    #           str(area_complexity) +'). Consider simplifying it with a higher tolerance: (p=p.simplify(tolerance=...)) ')

    

def infer_sample_area_from_pts(
        pts:_pd_DataFrame=None,
        grid=None,
        hull_type:str=['buff_non_empty_cells', 'buff_cells', 'buff_pts', 'concave','convex','bounding_box','grid'][0],
        concavity:float=1,
        buffer:float=None,
        tolerance:float=None,
        x:str='lon',
        y:str='lat',
        crs:str=None,
        proj_crs:str='auto',
        min_pts_to_sample_cell:int=1,
        plot_sample_area:dict=None,
) -> _shapely_Polygon:
    """Creates and returns a polygon containing all points which can be used to draw random points within

    Args:
    -------
    pts (pandas.DataFrame):
        DataFrame of points for which a search for other points within the specified radius shall be performed
    hull_type (str):
        Must be one of ['concave','convex','bounding_box','grid']. 
            - 'buff_non_empty_cells': each non-empty cell plus buffer around them
            - 'buff_cells': each cell with at least min_pts_to_sample_cell points plus buffer around them
            - 'buff_pts': a buffer will be drawn around each point. Warning: extremely be slow for large number of points
            - 'concave': a concave hull will be drawn around points. 
            - 'convex': a convex hull will be drawn around points.
            - 'bounding_box': a bounding box around points will be drawn
            - 'grid': a box covering full grid will be drawn
    concavity (float):
        will only be used when hull_type=='concave'. Value must be in (0,Inf]. Small values results in "very concave"(=fuzzy) hull. Inf results in convex hull (default=1)
    buffer (float):
        Size of the buffer that shall be applied on the hull. If None then it will be set equal to radius (r) (default=None)
    tolerance (float):
        Tolerance>=0 used to simplify geometry using Douglas-Peucker specifying maximum allowed geometry displacement. Chosing a parameter that is too small might result in performance issues. (default=None)
    x (str):
        column name of x-coordinate (=longtitude) in pts_source (default='lon')
    y (str):
        column name of y-coordinate (=lattitude) in pts_source (default='lat')
    min_pts_to_sample_cell (int):
        will only be used when hull_type=='buff_cells'. Minimum number of points that need to be present in a cell so that the cell is included in the sample area (default=0)
    
    Returns:
    -------
    sample_poly (shapely.geometry.Polygon):
        a grid covering all points (custom class containing 
    """
    # Reproject pts to a metric CRS so that buffer is in metres and hull is computed
    # in projected space. Temporary columns are dropped before returning.
    _proj_cols_to_drop = []
    if crs is not None:
        x, y, _proj_crs = _convert_pts_to_crs(pts=pts, x=x, y=y, initial_crs=crs, target_crs=proj_crs)
        _proj_cols_to_drop = [x, y]

    # To-Do maybe add minumum observations per cell to be kept
    if tolerance is None:
        tolerance = buffer

    area_missing_from_hull = 1
    
    if hull_type=='buffer':
        progress_print("sample_area hull_type 'buffer' deprectated. Use 'buff_pts' instead.")
        hull_type='buff_pts'
    elif hull_type == 'buffered_cells':
        progress_print("sample_area hull_type 'buffered_cells' deprecated. Use 'buff_cells' instead.")
        hull_type='buff_cells'
    
    if hull_type == 'bounding_box':
    
        min_x, min_y = pts[[x,y]].values.min(axis=0)
        max_x, max_y = pts[[x,y]].values.max(axis=0)
        hull_coordinates = [
            (min_x,min_y),
            (max_x,min_y),
            (max_x,max_y),
            (min_x,max_y),
            ]
        
    elif hull_type == 'grid':
        if grid is None:
            raise ValueError('In order to use the grid bounds as valid area, a grid needs to be supplied as function input: infer_sample_area_from_pts(grid=...)')
        # hull_coordinates = [
        #     (grid._search_internals.bounds.xmin,grid._search_internals.bounds.ymin),
        #     (grid._search_internals.bounds.xmax,grid._search_internals.bounds.ymin),
        #     (grid._search_internals.bounds.xmax,grid._search_internals.bounds.ymax),
        #     (grid._search_internals.bounds.xmin,grid._search_internals.bounds.ymax),
        #     ]
        hull_coordinates = [
            (grid.x_steps[0],  grid.y_steps[0]),   # Links unten
            (grid.x_steps[-1], grid.y_steps[0]),   # Rechts unten
            (grid.x_steps[-1], grid.y_steps[-1]),  # Rechts oben
            (grid.x_steps[0],  grid.y_steps[-1]),  # Links oben
        ]

    
    elif hull_type in ['concave', 'convex']:
    
        hull_coordinates = concave_hull(
            points=pts[[x,y]].values,
            concavity=concavity if hull_type=='concave' else _math_inf,
        )
    elif hull_type == 'buff_pts':
        if len(pts)>10000:
            progress_print("WARNING: creating a buffer around each point might cause long computation times for "+str(len(pts))+" points. Consider using hull_type='buff_non_empty_cells' or hull_type='concave' as more efficient method.")
        
        q=max(1,-(-2*buffer/tolerance)//1)
        sample_poly = _shapely_MultiPoint(pts[[x,y]].values).buffer(distance=buffer, quad_segs=q).simplify(tolerance)
        # don't simplify this shape
        area_missing_from_hull = 0

    elif hull_type in ['buff_non_empty_cells', 'buff_cells']:
        # TODO this should no longer refer to the search grid!
        # grid_xmin = grid._search_internals.bounds.xmin
        # grid_ymin = grid._search_internals.bounds.ymin
        # grid_xmax = grid._search_internals.bounds.xmax
        # grid_ymax = grid._search_internals.bounds.ymax
        grid_xmin = grid.x_steps[0]
        grid_ymin = grid.y_steps[0]
        grid_xmax = grid.x_steps[-1]
        grid_ymax = grid.y_steps[-1]
        if min_pts_to_sample_cell == 0 and hull_type != 'buff_non_empty_cells':
            sample_poly = _shapely_Polygon([
                (grid_xmin, grid_ymin),
                (grid_xmax, grid_ymin),
                (grid_xmax, grid_ymax),
                (grid_xmin, grid_ymax),
                ])
            
            # THIS SECTION IS REDUNDANT BECAUSE IT IS HANDLED IN intersect_polygon_with_grid
            # (cntd_cells, ovlpd_cells
            # ) = get_cells_by_lvl_ovlpd_by_cell_buffer(grid_spacing=grid._search_spacing, r=buffer, nest_depth=0)
            # cells_fully_valid = set(grid.get_all_ids())
            # cells_partly_valid = set()

            # cntd_cells = set([(lvl,(abs(row),abs(col))) for lvl,(row,col) in cntd_cells])
            # ovlpd_cells = set([(lvl,(abs(row),abs(col))) for lvl,(row,col) in ovlpd_cells])
            # min_row_id = min(grid._search_row_ids)
            # max_row_id = max(grid._search_row_ids)
            # min_col_id = min(grid._search_col_ids)
            # max_col_id = max(grid._search_col_ids)
            # for row_i in grid._search_row_ids:
            #     for lvl,(row_j, col_j) in cntd_cells:
            #         cells_fully_valid.add((lvl,(row_i-row_j,min_col_id-col_j)))
            #         cells_fully_valid.add((lvl,(row_i-row_j,max_col_id+col_j)))
            #         cells_fully_valid.add((lvl,(row_i+row_j,min_col_id-col_j)))
            #         cells_fully_valid.add((lvl,(row_i+row_j,max_col_id+col_j)))
            #     for lvl,(row_j, col_j) in ovlpd_cells:
            #         cells_partly_valid.add((lvl,(row_i-row_j,min_col_id-col_j)))
            #         cells_partly_valid.add((lvl,(row_i-row_j,max_col_id+col_j)))
            #         cells_partly_valid.add((lvl,(row_i+row_j,min_col_id-col_j)))
            #         cells_partly_valid.add((lvl,(row_i+row_j,max_col_id+col_j)))
            
            # for col_i in grid._search_col_ids:
            #     for lvl,(row_j, col_j) in cntd_cells:
            #         cells_fully_valid.add((lvl,(min_row_id-row_j,col_i-col_j)))
            #         cells_fully_valid.add((lvl,(min_row_id-row_j,col_i+col_j)))
            #         cells_fully_valid.add((lvl,(max_row_id+row_j,col_i-col_j)))
            #         cells_fully_valid.add((lvl,(max_row_id+row_j,col_i+col_j)))
            #     for lvl,(row_j, col_j) in ovlpd_cells:
            #         cells_partly_valid.add((lvl,(min_row_id-row_j,col_i-col_j)))
            #         cells_partly_valid.add((lvl,(min_row_id-row_j,col_i+col_j)))
            #         cells_partly_valid.add((lvl,(max_row_id+row_j,col_i-col_j)))
            #         cells_partly_valid.add((lvl,(max_row_id+row_j,col_i+col_j)))
      
            # cells_partly_valid.difference_update(cells_fully_valid)

            # grid.cells_fully_valid = cells_fully_valid
            # grid.cells_partly_valid = cells_partly_valid

        else:
            if hull_type == 'buff_non_empty_cells':
                if min_pts_to_sample_cell != 1:
                    progress_print("sample_area hull_type 'buff_non_empty_cells' used together with min_pts_to_sample_cell != 1. min_pts_to_sample_cell will be set = 1. Use 'buff_cells' to specify different value.")
                min_pts_to_sample_cell = 1
            # Build the valid-area footprint from OUTPUT grid cells, then intersect
            # with the search grid in intersect_polygon_with_grid below.
            # Note: the intersection step could be skipped when weight_valid_area is
            # False, since partial cell coverage only matters for area weighting.
            from numpy import floor as _np_floor
            from collections import Counter as _Counter
            sx = grid.cell_size
            sy = grid.cell_size_y
            # ox = grid.x_steps_bounds[0]
            # oy = grid.y_steps_bounds[0]
            ox = grid.x_steps[0]
            oy = grid.y_steps[0]
            _cols = _np_floor((pts[x].values - ox) / sx).astype(int)
            _rows = _np_floor((pts[y].values - oy) / sy).astype(int)
            _counts = _Counter(zip(_rows.tolist(), _cols.tolist()))
            _cells = [rc for rc, n in _counts.items() if n >= min_pts_to_sample_cell]

            def _out_cell_poly(row, col, ox=ox, oy=oy, sx=sx, sy=sy):
                # Return the bare 4-corner ring (matching grid.get_cell_poly), NOT
                # wrapped in a list — the outer [ ... ] is added by the caller below.
                x0 = ox + col * sx; x1 = x0 + sx
                y0 = oy + row * sy; y1 = y0 + sy
                return ((x0, y0), (x1, y0), (x1, y1), (x0, y1))

            polygons = [[_out_cell_poly(row, col)] for row, col in _cells]
            
            # THIS SECTION IS REDUNDANT BECAUSE IT IS HANDLED IN intersect_polygon_with_grid
            # buffer/spacing
            # maybe create one np array arround with coords of buffered cell around (0,0) and then add centroids to it 
            # (cntd_cells, ovlpd_cells
            # ) = get_cells_by_lvl_ovlpd_by_cell_buffer(grid_spacing=grid._search_spacing, r=buffer, nest_depth=0)
            # cells_fully_valid = set([(0,(row,col)) for row,col in id_to_pt_ids])
            
            # cells_partly_valid = set()
            # for row,col in id_to_pt_ids:
            #     for lvl,(d_row,d_col) in cntd_cells:
            #         tp = int if lvl==0 else float
            #         cells_fully_valid.add( (lvl, (tp(row+d_row), tp(col+d_col))) )
            #     for lvl,(d_row,d_col) in ovlpd_cells:
            #         tp = int if lvl==0 else float
            #         cells_partly_valid.add( (lvl, (tp(row+d_row), tp(col+d_col))) )        
            # cells_partly_valid.difference_update(cells_fully_valid)
            # grid.cells_fully_valid = cells_fully_valid
            # grid.cells_partly_valid = cells_partly_valid

            sample_poly = _shapely_MultiPolygon(polygons).buffer(buffer, quad_segs=3)
            # don't simplify this shape
        area_missing_from_hull = 0
           
    else:
        raise ValueError("hull_type to infere sample area for random points must be in ['buff_non_empty_cells', 'buff_cells', 'concave','convex','buff_pts', 'bounding_box', 'grid']. Value provided:",hull_type)
    #
    if area_missing_from_hull > 0:
        hull_poly = _shapely_Polygon(hull_coordinates).buffer(distance=0,quad_segs=1)
        # plot_polygon(poly=hull_poly)
        while area_missing_from_hull > 0:
            sample_poly = hull_poly
            q=max(1,-(-2*buffer/tolerance)//1)
            sample_poly = sample_poly.buffer(distance=buffer, quad_segs=q)
            
            sample_poly = sample_poly.simplify(tolerance)

            area_missing_from_hull = hull_poly.difference(
                sample_poly.intersection(hull_poly)
            ).area
            tolerance = tolerance*0.8
        #

    if _proj_cols_to_drop:
        pts.drop(columns=_proj_cols_to_drop, inplace=True, errors='ignore')

    # plot_polygon(poly=sample_poly)
    if not plot_sample_area is None:
        fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(8,8))
        ax.scatter(x=pts[x], y=pts[y], color="#51da58", s=0.3)
        plot_polygon(ax=ax, poly=sample_poly, facecolor="#06047640", edgecolor='red')
    # shoot warning if polygon is getting complex

    return sample_poly
#

def subtract_invalid_area(
        sample_poly:_shapely_Polygon,
        invalid_areas:_shapely_Polygon,
):
    valid_area_poly = sample_poly.difference(invalid_areas)

    return valid_area_poly
#

def apply_invalid_area_on_grid(
        
):
    return
#

def compute_disk_cell_overlap(
    disk_center_pt:_np_array,      
    row_col:tuple=(1,2),
    grid_spacing:float=0.0075,
    r:float=0.0075,
    silent = False,
    return_n_itx=False,
) -> float:
    """
    note this does not handle the case where the point lies within cell bounds

    Calculates intersection area of cell and search-circle (0,grid_spacing**2)
    Case for no intersection will be handled before (fully included or fully excluded).
    Case 1: two intersection points (more than half of square are within radius) - 3 vertices are within circle 
    Case 2: two intersection points (more than half of square are within radius) - 1 vertex is within circle
    Case 3: two intersection points (less than half of square within radius) - 0 vertices within circle (same row or col)
    Case 4: two intersection points (unclear wheter more or less than half) - 2 vertices within circle (same row or col)
    Case 5: four intersection points (more than half of circle is included) - 2 vertices within circle (same row or col)
    
    TODO: if grid_spacing/2 is greater than radius there will be weird instances 

    This can also be done already as a function of the point offset
    and is also symmetrical towards the triangle
    the intersection area only needs to be computed for those cases where excluded cells are intersected 

    """
    
    (xmin,ymin),(xmax,ymax) = ((row_col[1]-.5)*grid_spacing, (row_col[0]-.5)*grid_spacing), ((row_col[1]+.5)*grid_spacing, (row_col[0]+.5)*grid_spacing)
    if grid_spacing is None:
        rectangle_area = (abs(xmax-xmin)*abs(ymax-ymin))
    else:
        rectangle_area = grid_spacing**2
    calculated_area = 0
    precision = 13
    if not silent:
        fig,ax=plt.subplots()
        ax.add_patch(_plt_Circle(xy=disk_center_pt, radius=r, alpha=0.4))
        ax.add_patch(_plt_Rectangle(xy=(xmin,ymin),width=(xmax-xmin),height=(ymax-ymin), alpha=0.4))
        ax.autoscale_view()
    
    vtx_coords = (
        (xmin,ymin),
        (xmax,ymin),
        (xmax,ymax),
        (xmin,ymax),
        )
    vtx_dis_to_c = (
        ((disk_center_pt[0]-vtx_coords[0][0])**2+(disk_center_pt[1]-vtx_coords[0][1])**2)**.5,
        ((disk_center_pt[0]-vtx_coords[1][0])**2+(disk_center_pt[1]-vtx_coords[1][1])**2)**.5,
        ((disk_center_pt[0]-vtx_coords[2][0])**2+(disk_center_pt[1]-vtx_coords[2][1])**2)**.5,
        ((disk_center_pt[0]-vtx_coords[3][0])**2+(disk_center_pt[1]-vtx_coords[3][1])**2)**.5
    )
    vtx_in_r = [dis<=r for dis in vtx_dis_to_c]
    
    if sum(vtx_in_r)==4:
        if return_n_itx:
            return (rectangle_area, sum(vtx_in_r)) 
        else:
            return rectangle_area
    #
    if sum(vtx_in_r)==0:
        # circular segment 
        # say vertices B and C are closest to circle center. Then circle intersects BC 
        # (twice unless only touching or not intersecting at all)
        v_closest1, v_secondclosest1 = [i for d,i in sorted([(d,i) for i,d in enumerate(vtx_dis_to_c)])][:2]
        segment1 = (vtx_coords[v_closest1], vtx_coords[v_secondclosest1])
        itx_pts = circle_line_segment_intersection(
            circle_center=disk_center_pt,
            circle_radius=r,
            pt1=segment1[0],
            pt2=segment1[1],
            full_line=False,
            decimals=precision,
        )
        if len(itx_pts)<2:
            if return_n_itx:
                return (0., sum(vtx_in_r))
            else:
                return 0.
        
        itx_pt1, itx_pt2 = itx_pts
        #             
    elif sum(vtx_in_r)==1:
        # circular segment + triangle
        # say vertex B is closest to circle center then circle intersect AB and BC
        v_closest1 = vtx_dis_to_c.index(min(vtx_dis_to_c))
        segment1 = (vtx_coords[(v_closest1-1)%4], vtx_coords[v_closest1])
        segment2 = (vtx_coords[v_closest1], vtx_coords[(v_closest1+1)%4])
        itx_pt1 = circle_line_segment_intersection(
            circle_center=disk_center_pt,
            circle_radius=r,
            pt1=segment1[0],
            pt2=segment1[1],
            full_line=False,
            decimals=precision,
        )
        itx_pt2 = circle_line_segment_intersection(
            circle_center=disk_center_pt,
            circle_radius=r,
            pt1=segment2[0],
            pt2=segment2[1],
            full_line=False,
            decimals=precision,
        )
        if len(itx_pt1) != 1 or len(itx_pt2) != 1:
            raise ValueError("Unexpected number of intersections",itx_pt1, itx_pt2)
        itx_pt1, itx_pt2 = itx_pt1[0], itx_pt2[0]
        triangle = 1/2 * abs(itx_pt1[0]-itx_pt2[0])*abs(itx_pt1[1]-itx_pt2[1])
        calculated_area += triangle    
    elif sum(vtx_in_r)==2:
        # circular segement + triangle + triangle
        # say vertex B and C are closest to circle center then circle intersect AB and CD
        v_closest1, v_secondclosest1 = [i for d,i in sorted([(d,i) for i,d in enumerate(vtx_dis_to_c)])][:2]
        offset = 1 if (v_closest1-1)%4 != v_secondclosest1 else -1 
        segment1 = (vtx_coords[(v_closest1-offset)%4], vtx_coords[v_closest1])
        segment2 = (vtx_coords[v_secondclosest1], vtx_coords[(v_secondclosest1+offset)%4])
        itx_pt1 = circle_line_segment_intersection(
            circle_center=disk_center_pt,
            circle_radius=r,
            pt1=segment1[0],
            pt2=segment1[1],
            full_line=False,
            decimals=precision,
        )
        itx_pt2 = circle_line_segment_intersection(
            circle_center=disk_center_pt,
            circle_radius=r,
            pt1=segment2[0],
            pt2=segment2[1],
            full_line=False,
            decimals=precision,
        )
        if len(itx_pt1) != 1 or len(itx_pt2) != 1:
            raise ValueError("Unexpected number of intersections",itx_pt1, itx_pt2)
        itx_pt1, itx_pt2 = itx_pt1[0], itx_pt2[0]
        # match: which intersection point is aligned to which vertex?
        # align_closesty_to_pt1y = (
        #     0 if vtx_coords[v_closest1][0]==itx_pt1[0] else # x closest = x pt1 
        #     1 if vtx_coords[v_closest1][1]==itx_pt1[1] else # y closest = y pt1
        #     -2 if vtx_coords[v_closest1][0]==itx_pt2[0] else # x closest = x pt2
        #     -1 #if vtx_coords[v_closest1][0]==itx_pt1[0] # y closest = y pt2
        # )
        # # triangle 1: closest1,secondclosest1,pt2
        # pt_A, pt_B = (itx_pt1, itx_pt2) if align_closesty_to_pt1y in [0,1] else (itx_pt2, itx_pt1)
        # x_is_aligned = int(align_closesty_to_pt1y%2==0)
        # # if x is aligned take y difference
        # if vtx_coords[v_closest1][x_is_aligned]!=vtx_coords[v_secondclosest1][x_is_aligned]:
        #     print("x_is_aligned",x_is_aligned,"closest1",vtx_coords[v_closest1],"secondclosest1",vtx_coords[v_secondclosest1])
        # triangles = 1/2 * grid_spacing * (
        #     abs(vtx_coords[v_closest1][x_is_aligned]-pt_B[x_is_aligned]) +
        #     abs(vtx_coords[v_secondclosest1][x_is_aligned]-pt_A[x_is_aligned])
        # ) 
       
        x_is_aligned = 1 if vtx_coords[v_closest1][0]==itx_pt1[0] or vtx_coords[v_closest1][0]==itx_pt2[0] else 0
        triangles = 1/2 * grid_spacing * (
            abs(vtx_coords[v_closest1][x_is_aligned]-itx_pt1[x_is_aligned]) +
            abs(vtx_coords[v_closest1][x_is_aligned]-itx_pt2[x_is_aligned])
        )

        calculated_area += triangles

        # triangle 2: closest1,pt2,pt1
        # triangle1 = 1/2 * abs(
        #     vtx_coords[v_closest1][(align_closesty_to_pt1y+1)%2]-itx_pt2[(align_closesty_to_pt1y+1)%2]) * abs(
        #         vtx_coords[v_closest1][(align_closesty_to_pt1y+0)%2]-itx_pt2[(align_closesty_to_pt1y+0)%2])
        # # triangle 2: closest1,pt2,pt1
        # triangle2 = 1/2 * abs(
        #     vtx_coords[v_closest1][(align_closesty_to_pt1y+1)%2]-itx_pt1[(align_closesty_to_pt1y+1)%2]) * abs(
        #     vtx_coords[v_closest1][(align_closesty_to_pt1y+0)%2]-itx_pt2[(align_closesty_to_pt1y+0)%2]
        # )
        # # if triangle1==0.0 or triangle2==0.0:
        # #     print("triangle1",triangle1, "triangle2", triangle2)
        # # else:
        # #     print("both,,,")
        # calculated_area += triangle1 + triangle2
    elif sum(vtx_in_r)==3:
        # rectangle - triangle + circular segement
        # say vertex B is most distant from circle center: then circle intersects line segments AB and BC.
        v_farthest1 = vtx_dis_to_c.index(max(vtx_dis_to_c))
        segment1 = (vtx_coords[(v_farthest1-1)%4], vtx_coords[v_farthest1])
        segment2 = (vtx_coords[v_farthest1], vtx_coords[(v_farthest1+1)%4])
        
        itx_pt1 = circle_line_segment_intersection(
            circle_center=disk_center_pt,
            circle_radius=r,
            pt1=segment1[0],
            pt2=segment1[1],
            full_line=False,
            decimals=precision,
        )
        itx_pt2 = circle_line_segment_intersection(
            circle_center=disk_center_pt,
            circle_radius=r,
            pt1=segment2[0],
            pt2=segment2[1],
            full_line=False,
            decimals=precision,
        )
        itx_pt1, itx_pt2 = itx_pt1[0], itx_pt2[0]
        triangle = 1/2 * abs(itx_pt1[0]-itx_pt2[0])*abs(itx_pt1[1]-itx_pt2[1])
        calculated_area = rectangle_area - triangle 
    
    # calcualte area of circle segement
    len_pt0_pt1 = ((itx_pt1[0]-itx_pt2[0])**2+(itx_pt1[1]-itx_pt2[1])**2)**.5
    angle_rad = _math_pi-2*_math_acos(((len_pt0_pt1/2)/r))
    circle_segment_area = 0.5 * r**2 * (angle_rad - _math_sin(angle_rad))
    calculated_area += circle_segment_area

    # return  (calculated_area, sum(vtx_in_r))
    if return_n_itx:
        return min(calculated_area, rectangle_area),sum(vtx_in_r)
    else:
        return min(calculated_area, rectangle_area)

#

def estimate_disk_cell_overlap(
    disk_center_pt:_np_array,      
    centroid:tuple=(0,0),
    grid_spacing:float=0.0075,
    r:float=0.0075,
    a:float=0,
    b:float=0,
        
):
    d = ((disk_center_pt[0]-centroid[0])**2+(disk_center_pt[0]-centroid[0])**2)**.5
    d_r = d/r
    s_r = grid_spacing/r

    # b and c are the same for all points
    b = 1 / (0.70628102 + _np_exp(0.57266908 * (s_r - 2))) # b
    c = 1 / (-0.21443453 + _np_exp(0.76899004 * (s_r - 2))) # c
    a = 1 - 1 / (1.0 + b * _np_exp(-c * (d_r - 1)))
    area = a * grid_spacing**2
    return area

def compute_valid_area_weights(
        pts,
        grid,
        r:float,
        invalid_grid_cells,
        x:str='lon',
        y:str='lat',
        invalid_area_geometry=None,
        return_percentage:bool=True,
):
    """Calculates valid
    return_percentage if True returns share from [0,1] otherwise returns area in units of projection (meters)



    """
    full_circle = 2*_math_pi*r**2
    valid_area = full_circle
    # sort points by cell - cell_region - xy 
    # for cell: common contained cells sum invalid area
    # check if ovlpd cells have an invalid area
    # if not jump to next cell
    # for cell-cell-region sum common cntd area
    # check if ovlpd cell have an invalid area
    # if not jump to next cell-region
    # for point check ovlpd cells for invalid area

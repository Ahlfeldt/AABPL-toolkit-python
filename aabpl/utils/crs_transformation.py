from pandas import DataFrame as _pd_DataFrame
from math import floor as _math_floor
from pyproj import Transformer as _pyproj_Transformer
from shapely.ops import transform as _shapely_transform

def convert_wgs_to_utm(lon: float, lat: float):
    """Based on lat and lng, return best utm epsg-code"""
    # https://gis.stackexchange.com/a/269552
    # convert_wgs_to_utm function, see https://stackoverflow.com/a/40140326/4556479
    # see https://gis.stackexchange.com/a/127432/33092
    utm_band = str((_math_floor((lon + 180) / 6 ) % 60) + 1)
    if len(utm_band) == 1:
        utm_band = '0'+utm_band
    if lat >= 0:
        epsg_code = '326' + utm_band
        return epsg_code
    epsg_code = '327' + utm_band
    return epsg_code
#

def convert_MultiPolygon_crs(
        multipoly,
        initial_crs:str="EPSG:4326",
        target_crs:str="EPSG:4326",
):
    """Reprojects (Mulit-)Polygon from initial crs to target crs.

    """
    if initial_crs == target_crs:
        return multipoly
    project = _pyproj_Transformer.from_crs(crs_from=initial_crs, crs_to=target_crs, always_xy=True).transform
    try: 
        multipoly_transformed = _shapely_transform(project, multipoly)
    except:
        from aabpl.utils.progress import progress_print
        progress_print("ERROR in reprojecting sample_area "+str(type(multipoly))+" from "+initial_crs+" to "+target_crs+". Ensure that both crs are valid for coordinates of the sample_area.")
        multipoly_transformed = _shapely_transform(project, multipoly)
    return multipoly_transformed

def convert_coords_to_local_crs(
        pts,
        x:str='lon',
        y:str='lat',
        proj_x:str='proj_lon',
        proj_y:str='proj_lat',
        initial_crs:str="EPSG:4326",
        target_crs:str='auto',
        silent:bool=False,
) -> str:
    """Reprojects coordinates into target crs. Modifies DataFrame and returns string of local_crs. If non specified it chooses best crs based on the mean coordinate.

    """
    tgt_was_auto = target_crs == 'auto'
    if tgt_was_auto:
        if initial_crs != "EPSG:4326":
            # Convert to WGS84 first to determine the best UTM zone, then pick local_crs
            transformer = _pyproj_Transformer.from_crs(crs_from=initial_crs, crs_to="EPSG:4326", always_xy=True)
            x_wgs,y_wgs = transformer.transform(pts[x], pts[y])
            local_crs = 'EPSG:'+str(convert_wgs_to_utm(sum(x_wgs)/len(x_wgs), sum(y_wgs)/len(y_wgs)))
        else:
            local_crs = 'EPSG:'+str(convert_wgs_to_utm(*pts[[x,y]].mean(axis=0)))
    else:
        local_crs = target_crs
    transformer = _pyproj_Transformer.from_crs(crs_from=initial_crs, crs_to=local_crs, always_xy=True)
    pts[proj_x],pts[proj_y] = transformer.transform(pts[x], pts[y])
    if initial_crs != local_crs and (silent==False or tgt_was_auto):
        from aabpl.utils.progress import progress_print
        progress_print("Reproject from " +str(initial_crs)+' to '+local_crs)
    return local_crs
#

def _find_matching_col(pts, values):
    """Return the name of the first column in pts whose float64 values exactly match `values`, or None."""
    import numpy as _np
    for col in pts.columns:
        try:
            if pts[col].values.dtype.kind in ('f', 'i', 'u') and _np.array_equal(pts[col].values.astype(float), values):
                return col
        except Exception:
            pass
    return None

def convert_pts_to_crs(
    pts:_pd_DataFrame=None,
    x:str='lon',
    y:str='lat',
    initial_crs:str='EPSG:4326',
    target_crs:str='auto',
    silent:bool=False,
):
    if target_crs is None:
        return x, y, initial_crs

    # Determine local_crs without writing to pts yet.
    tgt_was_auto = target_crs == 'auto'
    if tgt_was_auto:
        if initial_crs != "EPSG:4326":
            transformer_wgs = _pyproj_Transformer.from_crs(crs_from=initial_crs, crs_to="EPSG:4326", always_xy=True)
            x_wgs, y_wgs = transformer_wgs.transform(pts[x], pts[y])
            local_crs = 'EPSG:' + str(convert_wgs_to_utm(sum(x_wgs)/len(x_wgs), sum(y_wgs)/len(y_wgs)))
        else:
            local_crs = 'EPSG:' + str(convert_wgs_to_utm(*pts[[x, y]].mean(axis=0)))
    else:
        local_crs = target_crs

    if local_crs == initial_crs:
        # Coords are already in the target CRS — x/y are usable as-is.
        return x, y, local_crs

    # Compute projected values so we can check whether they already exist in pts.
    from numpy import array as _np_array
    transformer = _pyproj_Transformer.from_crs(crs_from=initial_crs, crs_to=local_crs, always_xy=True)
    x_proj, y_proj = transformer.transform(pts[x].values, pts[y].values)
    x_proj = _np_array(x_proj, dtype=float)
    y_proj = _np_array(y_proj, dtype=float)

    # Reuse an existing column if it already contains the exact projected values.
    existing_x = _find_matching_col(pts, x_proj)
    existing_y = _find_matching_col(pts, y_proj)

    if existing_x is None:
        proj_x = next(('proj_x' + str(i) for i in [''] + list(range(len(pts.columns))) if 'proj_x' + str(i) not in pts.columns))
        pts[proj_x] = x_proj
    else:
        proj_x = existing_x

    if existing_y is None:
        proj_y = next(('proj_y' + str(i) for i in [''] + list(range(len(pts.columns))) if 'proj_y' + str(i) not in pts.columns))
        pts[proj_y] = y_proj
    else:
        proj_y = existing_y

    if not silent or tgt_was_auto:
        from aabpl.utils.progress import progress_print
        progress_print("Reproject from " + str(initial_crs) + ' to ' + local_crs)

    return proj_x, proj_y, local_crs
#

def convert_bounds_to_local_crs(
        xmin:float,
        xmax:float,
        ymin:float,
        ymax:float,
        initial_crs:str="EPSG:4326",
        target_crs:str='auto',
        silent:bool=False,
) -> tuple:
    """Reprojects coordinates into target crs. Modifies DataFrame and returns string of local_crs. If non specified it chooses best crs based on the mean coordinate.
    
    """
    bounds_corners_x = [xmin,xmax,xmax,xmin]
    bounds_corners_y = [ymin,ymax,ymax,ymin]
    
    bounds_corners_x = []
    bounds_corners_y = []
    for x in [xmin, (xmin + xmax)/2, xmax]:
        for y in [ymin, (ymin + ymax)/2, ymax]:
            bounds_corners_x.append(x)
            bounds_corners_y.append(y)
    if target_crs == 'auto':
        if initial_crs != "EPSG:4326":
            transformer = _pyproj_Transformer.from_crs(crs_from=initial_crs, crs_to="EPSG:4326", always_xy=True)
            x_wgs,y_wgs = transformer.transform(bounds_corners_x, bounds_corners_y)
            local_crs = 'EPSG:'+str(convert_wgs_to_utm(sum(x_wgs)/len(x_wgs), sum(y_wgs)/len(y_wgs)))
        else:
            local_crs = 'EPSG:'+str(convert_wgs_to_utm(
                sum(bounds_corners_x)/len(bounds_corners_x), 
                sum(bounds_corners_y)/len(bounds_corners_y))
                )
    else:
        local_crs = target_crs
    transformer = _pyproj_Transformer.from_crs(crs_from=initial_crs, crs_to=local_crs, always_xy=True)
    xs_local,ys_local = transformer.transform(bounds_corners_x, bounds_corners_y)
    return local_crs, (min(xs_local), max(xs_local),min(ys_local), max(ys_local))
#
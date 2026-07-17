import numpy as _np
from pandas import DataFrame as _pd_DataFrame
from math import floor as _math_floor
from pyproj import Transformer as _pyproj_Transformer
from shapely import transform as _shapely_modern_transform
from shapely.ops import transform as _shapely_legacy_transform


def convert_wgs_to_utm(lon: float, lat: float) -> str:
    """Based on lat and lng, return best utm epsg-code"""
    # https://gis.stackexchange.com/a/269552
    # convert_wgs_to_utm function, see https://stackoverflow.com/a/40140326/4556479
    # see https://gis.stackexchange.com/a/127432/33092
    utm_band = str((_math_floor((lon + 180) / 6 ) % 60) + 1)
    if len(utm_band) == 1:
        utm_band = '0' + utm_band
    if lat >= 0:
        return '326' + utm_band
    return '327' + utm_band

def convert_MultiPolygon_crs(
        multipoly,
        initial_crs: str = "EPSG:4326",
        target_crs: str = "EPSG:4326",
):
    """Reprojects (Multi-)Polygon from initial crs to target crs using vectorized modern Shapely 2.0+ api."""
    if initial_crs == target_crs:
        return multipoly

    transformer = _pyproj_Transformer.from_crs(crs_from=initial_crs, crs_to=target_crs, always_xy=True)

    try:
        # Vectorized transform (Shapely 2.0+)
        return _shapely_modern_transform(multipoly, transformer.transform)
    except Exception:
        # Fallback for older environments (Shapely < 2.0)
        try:
            return _shapely_legacy_transform(transformer.transform, multipoly)
        except Exception:
            from aabpl.utils.progress import progress_print
            progress_print(f"ERROR in reprojecting study_area {type(multipoly)} from {initial_crs} to {target_crs}. Ensure that both crs are valid for coordinates of the study_area.")
            raise


def convert_coords_to_local_crs(
        pts: _pd_DataFrame,
        x: str = 'lon',
        y: str = 'lat',
        proj_x: str = 'proj_lon',
        proj_y: str = 'proj_lat',
        initial_crs: str = "EPSG:4326",
        target_crs: str = 'auto',
        silent: bool = False,
) -> str:
    """Reprojects coordinates into target crs. Modifies DataFrame and returns string of local_crs.

    If target_crs is not specified it chooses the best crs based on the mean coordinate.
    """
    tgt_was_auto = target_crs == 'auto'

    if tgt_was_auto:
        if initial_crs != "EPSG:4326":
            transformer_sample = _pyproj_Transformer.from_crs(crs_from=initial_crs, crs_to="EPSG:4326", always_xy=True)
            x_wgs, y_wgs = transformer_sample.transform(pts[x].values, pts[y].values)
            local_crs = 'EPSG:' + str(convert_wgs_to_utm(_np.mean(x_wgs), _np.mean(y_wgs)))
        else:
            local_crs = 'EPSG:' + str(convert_wgs_to_utm(pts[x].mean(), pts[y].mean()))
    else:
        local_crs = target_crs

    transformer = _pyproj_Transformer.from_crs(crs_from=initial_crs, crs_to=local_crs, always_xy=True)
    pts[proj_x], pts[proj_y] = transformer.transform(pts[x].values, pts[y].values)

    if initial_crs != local_crs and (not silent or tgt_was_auto):
        from aabpl.utils.progress import progress_print
        progress_print(f"Reproject from {initial_crs} to {local_crs}")

    return local_crs

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
            x_wgs, y_wgs = transformer_wgs.transform(pts[x].values, pts[y].values)
            local_crs = 'EPSG:' + str(convert_wgs_to_utm(_np.mean(x_wgs), _np.mean(y_wgs)))
        else:
            local_crs = 'EPSG:' + str(convert_wgs_to_utm(*pts[[x, y]].mean(axis=0)))
    else:
        local_crs = target_crs

    if local_crs == initial_crs:
        # Coords are already in the target CRS — x/y are usable as-is.
        return x, y, local_crs

    # Compute projected values so we can check whether they already exist in pts.
    transformer = _pyproj_Transformer.from_crs(crs_from=initial_crs, crs_to=local_crs, always_xy=True)
    x_proj, y_proj = transformer.transform(pts[x].values, pts[y].values)
    x_proj = _np.array(x_proj, dtype=float)
    y_proj = _np.array(y_proj, dtype=float)

    # Reuse an existing column if it already contains the exact projected values.
    # Cheap summary-stat pre-checks (mean/min/max) reject most columns before
    # paying for a full elementwise np.array_equal comparison.
    existing_x = None
    existing_y = None
    x_mean, x_min, x_max = _np.mean(x_proj), _np.min(x_proj), _np.max(x_proj)
    y_mean, y_min, y_max = _np.mean(y_proj), _np.min(y_proj), _np.max(y_proj)
    proj_len = len(x_proj)
    columns_set = set(pts.columns)

    for col in pts.columns:
        try:
            col_series = pts[col]
            if col_series.dtype.kind in ('f', 'i', 'u') and len(col_series) == proj_len:
                arr = col_series.values
                arr_float = arr if arr.dtype == _np.float64 else arr.astype(float)

                if existing_x is None:
                    if _np.isclose(_np.mean(arr_float), x_mean) and _np.isclose(_np.min(arr_float), x_min) and _np.isclose(_np.max(arr_float), x_max):
                        if _np.array_equal(arr_float, x_proj):
                            existing_x = col
                            if existing_y is not None: break
                            continue

                if existing_y is None:
                    if _np.isclose(_np.mean(arr_float), y_mean) and _np.isclose(_np.min(arr_float), y_min) and _np.isclose(_np.max(arr_float), y_max):
                        if _np.array_equal(arr_float, y_proj):
                            existing_y = col
                            if existing_x is not None: break
                            continue
        except Exception:
            pass

    if existing_x is None:
        if 'proj_x' not in columns_set:
            proj_x = 'proj_x'
        else:
            i = 0
            while f'proj_x{i}' in columns_set:
                i += 1
            proj_x = f'proj_x{i}'
        pts[proj_x] = x_proj
    else:
        proj_x = existing_x

    if existing_y is None:
        if 'proj_y' not in columns_set:
            proj_y = 'proj_y'
        else:
            i = 0
            while f'proj_y{i}' in columns_set:
                i += 1
            proj_y = f'proj_y{i}'
        pts[proj_y] = y_proj
    else:
        proj_y = existing_y

    if not silent or tgt_was_auto:
        from aabpl.utils.progress import progress_print
        progress_print("Reproject from " + str(initial_crs) + ' to ' + local_crs)

    return proj_x, proj_y, local_crs

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

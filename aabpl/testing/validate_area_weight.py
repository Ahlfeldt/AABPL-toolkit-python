"""
Post-hoc accuracy check for area_weight (valid_area_share) computation.

Called automatically when config.VALIDATE_AREA=True after any radius_search
that uses area_weight.  For each decile of the computed valid_area_share
distribution, picks 5 source points closest to that decile value, computes
the exact circle ∩ study_area fraction via Shapely, and returns a list of
11 (computed_mean, exact_mean) tuples.
"""
import math
import numpy as np
from shapely.geometry import Point

from aabpl.utils.progress import progress_print


def validate_area_shares(pts, share_col, study_area_poly, r, x_col, y_col):
    """
    Compare computed valid_area_share against exact Shapely intersections.

    Parameters
    ----------
    pts : pd.DataFrame
        Source points with projected x/y columns and the share_col already set.
    share_col : str
        Column name of the computed valid_area_share (e.g. 'valid_area_share_750').
    study_area_poly : shapely.Polygon | shapely.MultiPolygon
        Study-area polygon in the same projected CRS as pts[x_col/y_col].
    r : float
        Search radius in projected units (metres).
    x_col, y_col : str
        Column names of projected x and y coordinates.

    Returns
    -------
    list of tuple
        11 (computed_mean, exact_mean) pairs, one per decile 0 % … 100 %.
        Values are rounded to 4 decimal places.
    """
    full_disk_area = math.pi * r * r
    shares  = pts[share_col].values.astype(float)
    xs      = pts[x_col].values.astype(float)
    ys      = pts[y_col].values.astype(float)
    n       = len(shares)
    n_per   = min(5, n)

    percentile_levels = np.linspace(0, 100, 11)   # 0, 10, 20, …, 100
    result = []

    for p in percentile_levels:
        target = float(np.percentile(shares, p))
        diffs  = np.abs(shares - target)
        idx5   = np.argsort(diffs)[:n_per]

        computed_mean = float(np.mean(shares[idx5]))

        exact_vals = []
        for i in idx5:
            circle = Point(xs[i], ys[i]).buffer(r)
            isect  = circle.intersection(study_area_poly)
            exact_vals.append(isect.area / full_disk_area)
        exact_mean = float(np.mean(exact_vals))

        result.append((round(computed_mean, 4), round(exact_mean, 4)))

    _fmt = "  ".join(f"({c:.3f},{e:.3f})" for c, e in result)
    progress_print(
        f"VALIDATE_AREA [{share_col}]  (computed, exact) per decile:\n  {_fmt}"
    )
    return result

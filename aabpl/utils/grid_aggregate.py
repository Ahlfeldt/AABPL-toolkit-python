"""
Aggregate per-point radius_search results onto a regular output grid.
"""

from __future__ import annotations
from typing import Union, Sequence, Tuple, Optional

import numpy as np
import pandas as pd


def aggregate_to_grid(
    pts: pd.DataFrame,
    val_cols: Union[str, Sequence[str]],
    output_spacing: float,
    x: str = "lon",
    y: str = "lat",
    agg: Union[str, dict] = "sum",
    bounds: Optional[Tuple[float, float, float, float]] = None,
) -> pd.DataFrame:
    """
    Bin per-point ``radius_search`` results into a regular grid of output cells.

    Each source point is assigned to the cell that contains it; the value
    columns are then aggregated within each cell.  The returned DataFrame has
    one row per occupied cell with the cell centroid coordinates and the
    requested aggregate values.

    Parameters
    ----------
    pts : pd.DataFrame
        Source points with coordinate columns and the aggregated value columns
        produced by ``radius_search``.  The original coordinate columns (e.g.
        ``lon`` / ``lat``) are always present unless ``keep_cols=False`` was
        used *and* the columns were not in the original DataFrame — which is
        not the case for the search-origin coordinates.
    val_cols : str or sequence of str
        Column name(s) in *pts* to aggregate per output cell.  Typically the
        ``radius_search`` result columns (e.g. ``"employment_750m"``).
    output_spacing : float
        Side length of each output cell, in the same coordinate units as the
        *x* / *y* columns.  Use metres when working in a projected CRS.
    x : str, optional
        Name of the x-coordinate column in *pts*.  Default ``"lon"``.
    y : str, optional
        Name of the y-coordinate column in *pts*.  Default ``"lat"``.
    agg : str or dict, optional
        Aggregation applied within each cell.  Any value accepted by
        ``pandas.DataFrame.groupby().agg()`` works — ``"sum"``, ``"mean"``,
        ``"median"``, or a per-column dict such as
        ``{"employment_750m": "sum", "n_pts": "count"}``.  Default ``"sum"``.
    bounds : (xmin, ymin, xmax, ymax) or None, optional
        Grid origin and extent.  When *None* the bounding box is inferred from
        the data, which means cells may shift slightly between calls on
        different subsets.  Pass explicit bounds (e.g. from ``grid.total_bounds``)
        for a reproducible, comparable grid layout.

    Returns
    -------
    pd.DataFrame
        One row per occupied output cell with columns:

        * ``cell_x``, ``cell_y`` — cell centroid in the same CRS as *x* / *y*
        * ``cell_col``, ``cell_row`` — integer cell indices (useful for joins)
        * all columns in *val_cols* after aggregation
    """
    if isinstance(val_cols, str):
        val_cols = [val_cols]
    val_cols = list(val_cols)

    missing = [c for c in val_cols if c not in pts.columns]
    if missing:
        raise KeyError(f"aggregate_to_grid: columns not found in pts: {missing}")
    for coord in (x, y):
        if coord not in pts.columns:
            raise KeyError(f"aggregate_to_grid: coordinate column '{coord}' not found in pts")

    if bounds is not None:
        xmin, ymin, xmax, ymax = bounds
    else:
        xmin, ymin = pts[x].min(), pts[y].min()
        xmax, ymax = pts[x].max(), pts[y].max()

    cell_col = ((pts[x] - xmin) / output_spacing).apply(np.floor).astype(int)
    cell_row = ((pts[y] - ymin) / output_spacing).apply(np.floor).astype(int)

    tmp = pts[val_cols].copy()
    tmp["_cell_col"] = cell_col
    tmp["_cell_row"] = cell_row

    grp = tmp.groupby(["_cell_col", "_cell_row"])[val_cols].agg(agg).reset_index()
    grp.rename(columns={"_cell_col": "cell_col", "_cell_row": "cell_row"}, inplace=True)

    grp["cell_x"] = xmin + (grp["cell_col"] + 0.5) * output_spacing
    grp["cell_y"] = ymin + (grp["cell_row"] + 0.5) * output_spacing

    cols = ["cell_x", "cell_y", "cell_col", "cell_row"] + val_cols
    return grp[cols].reset_index(drop=True)

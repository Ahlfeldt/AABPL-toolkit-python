"""
Quick sanity check: run radius_search on 20 pts with a tight study area,
compare computed valid_area_share against exact Shapely for each variant.
"""
import sys
sys.path.insert(0, 'Z:/Algorithm/PL_python/AABPL-toolkit-python')
[sys.modules.pop(m) for m in list(sys.modules.keys()) if m.startswith('aabpl')]

import math
import numpy as np
import pandas as pd
from shapely.geometry import Point
import aabpl
import aabpl.config as config
import aabpl.search.algorithm.disk_search as _ds
from aabpl.search.algorithm.disk_aggregation_chunk import search_and_aggregate as _chunk_saa
_ds.search_and_aggregate = _chunk_saa   # force chunk path

config.VALIDATE_AREA = False

# 20 real pts from plants_10180
_CSV = 'Z:/Algorithm/cbsa_sample_data/plants_10180.txt'
_raw = pd.read_csv(_CSV, sep=',', header=None)
_raw.columns = ['eid', 'employment', 'industry', 'lat', 'lon', 'moved']
pts_all = _raw[['lat', 'lon', 'employment']].head(20).copy()

R = 4000
CRS = 'EPSG:4326'
# tight buffer: many pts will have partial disks outside study area
SA = f'cells,m=1,b={int(0.1 * R)}'

VARIANTS = ['exact', 'logit', 'flat', 'binary']

print(f"n_pts=20  r={R}  study_area='{SA}'")
print(f"{'variant':8s}  {'share_col_mean':>16s}  {'exact_shapely_mean':>18s}  {'MAD':>8s}  {'max_diff':>10s}")

for wva in VARIANTS:
    pts = pts_all.copy()
    grid = aabpl.radius_search(pts, crs=CRS, r=R, c=['employment'],
                               x='lon', y='lat',
                               study_area=SA, area_weight=wva, silent=True)
    share_col = f'valid_area_share_{R}'
    if share_col not in pts.columns:
        print(f"  {wva:8s}  (no share column)")
        continue

    computed = pts[share_col].values.astype(float)

    # exact Shapely: need projected coords
    x_col = [c for c in pts.columns if 'proj' in c.lower() and 'lon' in c.lower()]
    y_col = [c for c in pts.columns if 'proj' in c.lower() and 'lat' in c.lower()]
    if not x_col or not y_col:
        # fallback: find proj columns
        x_col = [c for c in pts.columns if c not in ('lat','lon','employment') and 'x' in c.lower()]
        y_col = [c for c in pts.columns if c not in ('lat','lon','employment') and 'y' in c.lower()]

    sa_poly = getattr(grid, 'study_area', None)
    full_disk = math.pi * R * R
    if sa_poly is not None and x_col and y_col:
        xs = pts[x_col[0]].values
        ys = pts[y_col[0]].values
        exact = np.array([
            Point(xs[i], ys[i]).buffer(R).intersection(sa_poly).area / full_disk
            for i in range(len(pts))
        ])
        diffs = np.abs(computed - exact)
        print(f"  {wva:8s}  {computed.mean():>16.4f}  {exact.mean():>18.4f}  {diffs.mean():>8.4f}  {diffs.max():>10.4f}")
    else:
        print(f"  {wva:8s}  {computed.mean():>16.4f}  (no study_area/proj cols for exact check)")

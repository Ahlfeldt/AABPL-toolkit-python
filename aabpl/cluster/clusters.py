from numpy import (
    array as _np_array, 
    zeros as _np_zeros,
)
from pyproj import Transformer
from pandas import DataFrame as _pd_DataFrame
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union
from aabpl.utils.misc import find_column_name, arr_to_tpls
from aabpl.utils.cell_geometry import min_possible_dist_cells_to_cell

def merge_condition_queen_contingency(max_steps:int=1)->bool:
    """
    check if clusters are queen contigent (horizontally, vertically, or diagonally). If max_steps>=2 it allow for that many minus 1 gaps to count as contingent.
    """
    def check_if_merge(cluster_a, cluster_b, max_steps=max_steps):
        cluster_small, cluster_large = (cluster_a, cluster_b) if cluster_a.n_cells < cluster_b.n_cells else (cluster_b, cluster_a)
        for row_a, col_a in cluster_small.cells:
            for row_offset in range(-max_steps, max_steps + 1):
                for col_offset in range(-max_steps, max_steps + 1):
                    if (row_a+row_offset, col_a+col_offset) in cluster_large.cells:
                        return True
        return False
    return check_if_merge

def merge_condition_rook_contingency(max_steps:int=1)->bool:
    """
    check if clusters are rook contigent (horizontally, vertically). If max_steps>=2 it allow for that many minus 1 gaps to count as contingent.
    """
    def check_if_merge(cluster_a, cluster_b, max_steps=max_steps):
        cluster_small, cluster_large = (cluster_a, cluster_b) if cluster_a.n_cells < cluster_b.n_cells else (cluster_b, cluster_a)
        for row_a, col_a in cluster_small.cells:
            for row_offset in range(-max_steps, max_steps + 1):
                if row_offset == 0: continue
                for col_offset in range(-max_steps, max_steps + 1):
                    if col_offset == 0: continue
                    if (row_a+row_offset, col_a+col_offset) in cluster_large.cells:
                        return True
        return False
    return check_if_merge

def merge_condition_centroid_distance(max_centroid_dist:float)->bool:
    def check_if_merge(cluster_a, cluster_b, max_centroid_dist=max_centroid_dist):
        if ((cluster_a.centroid[0]-cluster_b.centroid[0])**2+(cluster_a.centroid[1]-cluster_b.centroid[1])**2)**.5 <= max_centroid_dist:
            return True
        return False
    return check_if_merge

def merge_condition_border_distance(max_border_dist:float, spacing:float)->bool:
    def check_if_merge(cluster_a, cluster_b, spacing=spacing):
        # distance = geopy_distance(cluster_a.centroid, cluster_b.centroid).meters
        cluster_small, cluster_large = (cluster_a, cluster_b) if cluster_a.n_cells < cluster_b.n_cells else (cluster_b, cluster_a)
        cluster_small_cells = _np_array(cluster_small.cells)
        cluster_large_cells = _np_array(cluster_large.cells)
        for cell in cluster_small_cells:
            if min(min_possible_dist_cells_to_cell(cluster_large_cells, cell)) * spacing <= max_border_dist:
                return True
        return False
    return check_if_merge

def merge_condition_distance_based(max_centroid_dist:float, max_border_dist:float, spacing:float)->bool:
    def check_if_merge(cluster_a, cluster_b, max_centroid_dist=max_centroid_dist, max_border_dist=max_border_dist, spacing=spacing):
        # distance = geopy_distance(cluster_a.centroid, cluster_b.centroid).meters
        if ((cluster_a.centroid[0]-cluster_b.centroid[0])**2+(cluster_a.centroid[1]-cluster_b.centroid[1])**2)**.5 <= max_centroid_dist:
            cluster_small, cluster_large = (cluster_a, cluster_b) if cluster_a.n_cells < cluster_b.n_cells else (cluster_b, cluster_a)
            cluster_small_cells = _np_array(cluster_small.cells)
            cluster_large_cells = _np_array(cluster_large.cells)
            for cell in cluster_small_cells:
                if min(min_possible_dist_cells_to_cell(cluster_large_cells, cell)) * spacing <= max_border_dist:
                    return True
        return False
    return check_if_merge

def merge_condition_distance_based_dnf(conditions: list, spacing: float) -> bool:
    """
    DNF (Disjunctive Normal Form) merge condition for a list of (centroid_dist, border_dist) tuples.

    Merge is triggered when ANY condition tuple is fully satisfied (OR between tuples).
    Within each tuple both thresholds must hold (AND within tuple).

    Parameters
    ----------
    conditions : list of (float, float)
        Each element is ``(max_centroid_dist, max_border_dist)``.
        A ``None`` threshold in a pair disables that measure for that condition:
        ``(25000, None)`` = centroid only; ``(None, 10000)`` = border only.
    spacing : float
        Grid cell size in the same unit as the distance thresholds.

    Examples
    --------
    ``[(25000, 10000)]``
        Equivalent to the original AND behaviour.
    ``[(25000, 10000), (15000, 20000)]``
        Merge if (centroid ≤ 25 km AND border ≤ 10 km) OR (centroid ≤ 15 km AND border ≤ 20 km).
    """
    def check_if_merge(cluster_a, cluster_b, conditions=conditions, spacing=spacing):
        centroid_dist = (
            (cluster_a.centroid[0] - cluster_b.centroid[0]) ** 2 +
            (cluster_a.centroid[1] - cluster_b.centroid[1]) ** 2
        ) ** 0.5
        # Compute border distance lazily — only when at least one condition needs it.
        _border_dist = None
        def _get_border_dist():
            nonlocal _border_dist
            if _border_dist is None:
                cluster_small, cluster_large = (
                    (cluster_a, cluster_b) if cluster_a.n_cells < cluster_b.n_cells
                    else (cluster_b, cluster_a)
                )
                small_cells = _np_array(cluster_small.cells)
                large_cells = _np_array(cluster_large.cells)
                _border_dist = min(
                    min(min_possible_dist_cells_to_cell(large_cells, cell))
                    for cell in small_cells
                ) * spacing
            return _border_dist

        for (max_c, max_b) in conditions:
            c_ok = (max_c is None) or (centroid_dist <= max_c)
            if not c_ok:
                continue
            b_ok = (max_b is None) or (_get_border_dist() <= max_b)
            if b_ok:
                return True
        return False
    return check_if_merge

def get_val_if_not_already_scalar(val_or_vals, index:int):
    if hasattr(val_or_vals, '__len__'):
        return float(val_or_vals[index])
    return float(val_or_vals)

class Clustering(object):
    """ 
    Methods:
    
    """
    def __init__(
        self,
        grid
    ):
        """
        bind Clusters object to grid 
        """
        self.grid = grid
        self.by_column = {}

    @property
    def thresholds(self):
        """Dict mapping cluster column name to {k_percentile: threshold_value}."""
        return {
            col: {getattr(cfc, 'k', None): getattr(cfc, 'threshold', None)}
            for col, cfc in self.by_column.items()
        }

    def create_clusters(
        self,
        pts:_pd_DataFrame,
        c:str=['employment'],
        cluster_c:str=['employment_cluster'],
        queen_contingency:int=1,
        rook_contingency:int=0,
        centroid_dist_threshold:float=None,
        border_dist_threshold:float=None,
        min_cluster_share_after_contingency:float=0.05,
        min_cluster_share_after_centroid_dist:float=0.00,
        min_cluster_share_after_convex:float=0.00,
        make_convex:bool=True,
        row_name:str='id_y',
        col_name:str='id_x',
        cluster_suffix:str='_cluster',
    ):
        """
        Detects all grid cells containing a point that is labeled as clusters. those cells are then labeled as cluster cells. 
        If a contingency parameter is set it then merges clustered cells to clusters based on contingency.
        Then it drops all clusters of which the total is smaller than min_cluster_share_after_contingency of the largest cluster. 
        Then it merges cluster based on centroid distance and/or border distance a parameter is set. Then it drops clusters smaller than min_cluster_share_after_centroid_dist of the largest cluster.
        If make_convex it applies a convex hull on the cluster ands all cells within that hull to the cluster. Then it drops all cluster smaller than min_cluster_share_after_convex of the largest cluster.

        Args:
        -------
        pts (pandas.DataFrame):
            DataFrame of points for which a search for other points within the specified radius shall be performed
        cluster_c (str or list):
            column(s) in DataFrame with boolean values indicating whether a point is part of cluster for column or not. To create this column you may first run aaabpl.detect_cluster_pts(...).
        c (str or list):
            column(s) in DataFrame for which data within search radius shall be aggregated. If None provided it will simply count the points within the radius. 
        queen_contingency (int):
            if contigent (vertical, horizontal, diagonal) cells that are also classified as clustered shall be part of the same cluster. If set to a value>=2 then it also adds non-contingent cells that are that many steps away to the same cluster. (default=1) 
        rook_contingency (int):
            if contigent (vertical, horizontal, diagonal) cells that are also classified as clustered shall be part of the same cluster. Ignored if queen_contingency is set to a higher value. If set to a value>=2 then it also adds non-contingent cells that are that many steps away to the same cluster. (default=1) 
        centroid_dist_threshold (float):
            maximum distance between centroids of clusters to be merged into a single cluster. If None clusters won't be merged based on centroid distance. (default=None)
        border_dist_threshold (float):
            maximum distance between borders of clusters to be merged into a single cluster. If None clusters won't be merged based on boundary distance (default=None)
        min_cluster_share_after_contingency (float):
            minimum share of cluster of total to not be dropped after cells are merged to clusters based on contingency
        min_cluster_share_after_centroid_dist (float):
            minimum share of cluster of total to not be dropped after clusters are merged based on centroid
        min_cluster_share_after_convex (float):
            minimum share of cluster of total to not be dropped after clusters are made convex by adding cells within its convex hull
        make_convex (bool):
            Whether all cells within the convex hull of a cluster shall be added to it (default=True)
        """
        # TODO raise error when not columns contains col for which no radius search has been performed
        centroid_dist_threshold = centroid_dist_threshold if type(centroid_dist_threshold) in [list, _np_array] else [centroid_dist_threshold for n in c]
        border_dist_threshold = border_dist_threshold if type(border_dist_threshold) in [list, _np_array] else [border_dist_threshold for n in c]
        for i, (column, max_centroid_dist, max_border_dist) in enumerate(zip(c, centroid_dist_threshold, border_dist_threshold)):
            cluster_column = column + cluster_suffix
            cells_with_cluster = (pts[[row_name, col_name]][pts[cluster_column]]).values
            clusters_for_column = Clustering.ClustersForColumn(
                self.grid,
                clustered_cells=cells_with_cluster,
                column=column,
                column_id=i,
            )
            self.by_column[cluster_column] = clusters_for_column
            largest_cluster = max([0]+[cluster.total for cluster in clusters_for_column.clusters])
            if queen_contingency:
                clusters_for_column.merge_clusters(check_if_merge=merge_condition_queen_contingency(queen_contingency))
            if rook_contingency > queen_contingency:
                clusters_for_column.merge_clusters(check_if_merge=merge_condition_rook_contingency(rook_contingency))
            if min_cluster_share_after_contingency > 0:
                clusters_for_column.drop_small_cluster(largest_cluster * min_cluster_share_after_contingency)
            clusters_for_column.clusters_pre_merge = 0
            # merge_dist_conditions is a normalised list of (centroid, border) tuples
            # (set by _unpack_merge_dist when caller passes merge_dist=).
            # Fall back to the legacy two-scalar path when called directly.
            _conditions = getattr(self, '_merge_dist_conditions_for_col', {}).get(column)
            if _conditions is not None:
                if _conditions:
                    clusters_for_column.merge_clusters(
                        check_if_merge=merge_condition_distance_based_dnf(_conditions, self.grid.cell_size)
                    )
                clusters_for_column.merge_rules = {'contingency': queen_contingency or rook_contingency, 'merge_dist': _conditions}
            elif not centroid_dist_threshold is None and not border_dist_threshold is None:
                clusters_for_column.merge_clusters(check_if_merge=merge_condition_distance_based(max_centroid_dist, max_border_dist, self.grid.cell_size))
                clusters_for_column.merge_rules = {'contingency': queen_contingency or rook_contingency, 'centroid_dist': max_centroid_dist, 'border_dist': max_border_dist}
            elif not centroid_dist_threshold is None:
                clusters_for_column.merge_clusters(check_if_merge=merge_condition_centroid_distance(centroid_dist_threshold))
                clusters_for_column.merge_rules = {'contingency': queen_contingency or rook_contingency, 'centroid_dist': centroid_dist_threshold}
            elif not border_dist_threshold is None:
                clusters_for_column.merge_clusters(check_if_merge=merge_condition_border_distance(max_border_dist, self.grid.cell_size))
                clusters_for_column.merge_rules = {'contingency': queen_contingency or rook_contingency, 'border_dist': max_border_dist}
            else:
                clusters_for_column.merge_rules = {'contingency': queen_contingency or rook_contingency}
            if min_cluster_share_after_centroid_dist > 0:
                clusters_for_column.drop_small_cluster(largest_cluster * min_cluster_share_after_centroid_dist)
            if make_convex:
                clusters_for_column.make_clusters_convex()
            if min_cluster_share_after_convex > 0:
                clusters_for_column.drop_small_cluster(largest_cluster * min_cluster_share_after_convex)
            clusters_for_column.match_cell_to_cluster_id()
            clusters_for_column.add_geom_to_clusters()
            clusters_for_column.add_area_to_clusters()
            clusters_for_column.add_cluster_id_to_pts(column, cluster_column)
    
    def make_cluster_orthogonally_convex(
            self
        ):
        """
        ensure all cells between (=orthogononally, not diagonally) two cluster cells are also part of the cluster
        exception: a cell is part of another cluster already
        """
        id_to_sums = self.grid.cell_aggregates
        grid_xmin = self.grid.x_steps_bounds[0]
        grid_ymin = self.grid.y_steps_bounds[0]
        cell_size = self.grid.cell_size
        cell_size_y = self.grid.cell_size_y
        for (cluster_column, clusters) in self.by_column.items():
            all_clustered_cells = set()
            for cluster in clusters['prime_locs']:
                all_clustered_cells.update(cluster.cells)
            
            for cluster in clusters['prime_locs']:
                cells_from_other_clusters = all_clustered_cells.difference(cluster.cells)
                n_last_it = -1
                while len(cluster.cells) != n_last_it:
                    cells = cluster.cells
                    cells_in_convex_cluster = set(cells)
                    row_ids = sorted(set([row for row,col in cells]))
                    col_ids = sorted(set([col for row,col in cells]))
                    row_range = range(min(row_ids), max(row_ids)+1)
                    col_range = range(min(col_ids), max(col_ids)+1)
                    for r in row_range:
                        cells_to_left = [col for row, col in cells if row<r]
                        cells_to_right = [col for row, col in cells if row>r]
                        cells_same_col = [col for row, col in cells if row==r]
                        max_left, min_left, max_right, min_right, max_same, min_same = None, None, None, None, None, None
                        if len(cells_to_left) > 0:
                            min_left = min(cells_to_left)
                            max_left = max(cells_to_left)
                        if len(cells_to_right) > 0:
                            min_right = min(cells_to_right)
                            max_right = max(cells_to_right)
                        if len(cells_same_col) > 0:
                            min_same = min(cells_same_col)
                            max_same = max(cells_same_col)
                        max_other = max_right if max_left is None else max_left if max_right is None else max([min_left, min_right]) 
                        min_other = min_right if min_left is None else min_left if min_right is None else min([min_left, min_right])
                        max_all = max_other if max_same is None else max_same if max_other is None else min([min_same, min_other])
                        min_all = min_other if min_same is None else min_same if min_other is None else max([min_same, min_other])
                        cells_in_convex_cluster.update([(r,c) for c in range(min_all, max_all+1)])
                    #

                    for c in col_range:
                        cells_to_left = [row for row, col in cells if col<c]
                        cells_to_right = [row for row, col in cells if col>c]
                        cells_same_col = [row for row, col in cells if col==c]
                        max_left, min_left, max_right, min_right, max_same, min_same = None, None, None, None, None, None
                        if len(cells_to_left) > 0:
                            min_left = min(cells_to_left)
                            max_left = max(cells_to_left)
                        if len(cells_to_right) > 0:
                            min_right = min(cells_to_right)
                            max_right = max(cells_to_right)
                        if len(cells_same_col) > 0:
                            min_same = min(cells_same_col)
                            max_same = max(cells_same_col)
                        # max_other = max_right if max_left is None else max_left if max_right is None or max_left < max_right else max_right 
                        # min_other = min_right if min_left is None else min_left if min_right is None or min_left > min_right else min_right
                        min_other = None if max_left is None or max_right is None else max([min_left, min_right])
                        max_other = None if max_left is None or max_right is None else min([min_left, min_right])
                        min_all = min_other if min_same is None else min_same if min_other is None else min([min_same, min_other])
                        max_all = max_other if max_same is None else max_same if max_other is None else max([min_same, min_other])
                        cells_in_convex_cluster.update([(r,c) for r in range(min_all, max_all+1)])
                    #

                    cells_in_convex_cluster.difference_update(cells_from_other_clusters)
                    cluster.cells = sorted(cells_in_convex_cluster)
                    n_last_it = len(cluster.cells)
                
                cluster.total = sum([id_to_sums[cell] for cell in cells_in_convex_cluster if cell in id_to_sums])
                cluster.centroid = _np_array([(grid_xmin+(col+.5)*cell_size, grid_ymin+(row+.5)*cell_size_y) for row,col in cells_in_convex_cluster]).sum(axis=0)/len(cells_in_convex_cluster)
            #
        #
    #
    class Cluster(object):
        """ 
        Methods:
        
        """
        def __init__(
                self,
                id:int,
                cell_in_cluster:list,
                centroid:tuple,
                total:float,
                get_cell_centroid,
        ):
            """
            bind Clusters object to grid 
            """
            self.id = id
            self.cells = [cell_in_cluster]
            self.centroid = centroid
            self.total = total
            self.n_cells = 1
            self.get_cell_centroid = get_cell_centroid

        
        def annex_cluster(self, cluster_to_annex):
            n_current, n_neighbor = self.n_cells, cluster_to_annex.n_cells
            self.cells = self.cells + cluster_to_annex.cells
            self.total += cluster_to_annex.total
            n_cells = n_current + n_neighbor
            self.centroid = (
                (self.centroid[0]*n_current + cluster_to_annex.centroid[0]*n_neighbor)/n_cells,
                (self.centroid[1]*n_current + cluster_to_annex.centroid[1]*n_neighbor)/n_cells
            )
            self.n_cells = n_cells

        def update_id(self,new_id):
            self.id = new_id
            return self
        
        def add_cells_to_cluster(
                self,
                cells_to_add:set,
                id_to_sums:dict,
                column_id:int,
        ):
            """
            Add cells to cluster and update total, centroid, and n_cells
            """
            if len(cells_to_add) == 0:
                return
            self.total += sum([id_to_sums[cell][column_id] for cell in cells_to_add if cell in id_to_sums])
            n_cells_to_add = len(cells_to_add)
            n_cells = self.n_cells + n_cells_to_add
            centroids_of_cells_to_add = _np_array([self.get_cell_centroid(row, col) for row, col in cells_to_add])
            self.centroid = (
                (self.centroid[0]*self.n_cells + centroids_of_cells_to_add[:,0].sum()) / n_cells,
                (self.centroid[1]*self.n_cells + centroids_of_cells_to_add[:,1].sum()) / n_cells
            )
      
            self.n_cells = n_cells
            self.cells = self.cells + list(cells_to_add)
        
        def add_area(
                self,
                spacing:float,
        ):
            """Add area attribute as product of number of cells and square grid spacing"""
            self.area = self.n_cells * spacing**2
        #
        def add_geometry(
                self,
                grid_xmin:float,
                grid_ymin:float,
                spacing:float,
                spacing_y:float=None,
        ):
            """add shapely polygon unaray union geometry"""
            if spacing_y is None:
                spacing_y = spacing
            self.geometry = unary_union([
                Polygon([
                    (grid_xmin+col*spacing,   grid_ymin+row*spacing_y),
                    (grid_xmin+(col+1)*spacing, grid_ymin+row*spacing_y),
                    (grid_xmin+(col+1)*spacing, grid_ymin+(row+1)*spacing_y),
                    (grid_xmin+col*spacing,   grid_ymin+(row+1)*spacing_y)
                    ])
                for row,col in self.cells]
            )
            # self.geometry = unary_union(
            #     [[Polygon(((xmin,ymin),(xmax,ymin),(xmax,ymax),(xmin,ymax)))
            #     for (xmin,ymin),(xmax,ymax) in [row_col_to_bounds[cell]]][0]
            #     for cell in self.cells]
            # )
    #


    class ClustersForColumn(object):
        """ 
        Methods:
        
        """
        def __init__(
                self,
                grid,
                clustered_cells:_np_array,
                column:str='employment',
                column_id:int=None,
        ):
            """
            bind Clusters object to grid 
            """
            self.grid = grid
            self.column = column
            self.column_id = column_id
            self.clustered_cells = set(arr_to_tpls(clustered_cells, int))
            id_to_sums = self.grid.cell_aggregates
            get_cell_centroid = grid.get_cell_centroid  # output-grid centroid: (xmin + (col+0.5)*cell_size, ...)
            self.clusters = [Clustering.Cluster(
                id=i,
                cell_in_cluster=(row, col),
                centroid=get_cell_centroid(row, col),
                total=get_val_if_not_already_scalar(id_to_sums[(row, col)], column_id),
                get_cell_centroid=get_cell_centroid,
            ) for i, (row, col) in enumerate(self.clustered_cells)]
            self.by_id = {cluster.id: cluster for cluster in self.clusters}
        #

        def update_ids(self):
            self.clusters.sort(key=lambda c: -c.total)
            for i, cluster in enumerate(self.clusters):
                cluster.update_id(i+1)
            #
        #
        
        
        def merge_clusters(
                self,
                check_if_merge,
                # distance_threshold:float
            ):
            
            def find_next_merge(clusters, check_if_merge, annexed_cluster_ids:set):
                for i, current_cluster in enumerate(clusters):
                    current_cluster_id = current_cluster.id
                    if current_cluster_id in annexed_cluster_ids:
                        continue
                    for neighbor_cluster in clusters[i+1:]:
                        neighbor_cluster_id = neighbor_cluster.id
                        if neighbor_cluster_id == current_cluster_id or neighbor_cluster_id in annexed_cluster_ids:
                            continue  # Skip unclustered cells and self-comparison
                        # distance = ((current_centroid[0]-neighbor_cluster.centroid[0])**2+(current_centroid[1]-neighbor_cluster.centroid[1])**2)**.5
                        # if distance < distance_threshold:
                        if check_if_merge(current_cluster, neighbor_cluster):
                            current_cluster.annex_cluster(neighbor_cluster)
                            return neighbor_cluster_id
                        #
                    #

            annexed_cluster_ids = set()
            while True:
                self.clusters = [c for c in self.clusters if not c.id in annexed_cluster_ids]
                self.clusters.sort(key=lambda c: (-c.total, -c.n_cells))
                neighbor_cluster_id = find_next_merge(self.clusters, check_if_merge, annexed_cluster_ids)
                if neighbor_cluster_id is None:
                    break
                else:
                    annexed_cluster_ids.add(neighbor_cluster_id)
                #
            # assign ids starting at 1 from biggest (according to sum value) to largest cluster 
            self.update_ids()
        #
        
        def drop_small_cluster(self, min_value):
            self.clusters = [cluster for cluster in self.clusters if cluster.total >= min_value]
            clustered_cells = set()
            for cluster in self.clusters:
                clustered_cells.update(cluster.cells)
            self.clustered_cells = clustered_cells
        #

        def make_clusters_convex(
                self
        ):
            set_clustered_cells = self.clustered_cells
            id_to_sums = self.grid.cell_aggregates
            cell_size = self.grid.cell_size
            cell_size_y = self.grid.cell_size_y
            grid_xmin = self.grid.x_steps_bounds[0]
            grid_ymin = self.grid.y_steps_bounds[0]

            for cluster in self.clusters:
                cells = cluster.cells
                cells_to_add = set()
                hull_poly = unary_union([
                    Polygon([
                        (grid_xmin + col      * cell_size,   grid_ymin + row      * cell_size_y),
                        (grid_xmin + (col+1)  * cell_size,   grid_ymin + row      * cell_size_y),
                        (grid_xmin + (col+1)  * cell_size,   grid_ymin + (row+1)  * cell_size_y),
                        (grid_xmin + col      * cell_size,   grid_ymin + (row+1)  * cell_size_y),
                    ])
                    for row, col in cells
                ]).convex_hull

                row_ids = [row for row, col in cells]
                col_ids = [col for row, col in cells]
                for r in range(min(row_ids), max(row_ids) + 1):
                    for c in range(min(col_ids), max(col_ids) + 1):
                        if (r, c) not in set_clustered_cells:
                            cx = grid_xmin + (c + 0.5) * cell_size
                            cy = grid_ymin + (r + 0.5) * cell_size_y
                            if hull_poly.contains(Point(cx, cy)):
                                set_clustered_cells.add((r, c))
                                cells_to_add.add((r, c))
                cluster.add_cells_to_cluster(cells_to_add=cells_to_add, id_to_sums=id_to_sums, column_id=self.column_id)
            #
            self.update_ids()
        #
        
        def match_cell_to_cluster_id(self):
            cell_to_cluster = {}
            for cluster in self.clusters:
                cell_to_cluster.update({cell: cluster.id for cell in cluster.cells})
            self.cell_to_cluster_id = cell_to_cluster
        #

        def add_geom_to_clusters(self):
            grid_xmin = self.grid.x_steps_bounds[0]
            grid_ymin = self.grid.y_steps_bounds[0]
            out_s  = self.grid.cell_size
            out_sy = self.grid.cell_size_y
            for cluster in self.clusters:
                # cluster.cells already holds output-grid (row, col) indices.
                out_cells = set(cluster.cells)
                cluster.geometry = unary_union([
                    Polygon([
                        (grid_xmin +  col      * out_s,  grid_ymin +  row      * out_sy),
                        (grid_xmin + (col + 1) * out_s,  grid_ymin +  row      * out_sy),
                        (grid_xmin + (col + 1) * out_s,  grid_ymin + (row + 1) * out_sy),
                        (grid_xmin +  col      * out_s,  grid_ymin + (row + 1) * out_sy),
                    ])
                    for row, col in out_cells
                ])
            
        def add_area_to_clusters(self):
            for cluster in self.clusters:
                cluster.add_area(spacing=self.grid.cell_size)
        #

        def add_cluster_id_to_pts(self, column, cluster_column):
            # cluster_column stays as the strict per-point boolean (sum > threshold).
            # Cluster polygon membership (which may include convex hull / contingency cells)
            # is written to a separate cluster_id column so the two semantics stay distinct.
            cell_to_cluster = self.cell_to_cluster_id
            _sc = self.grid._search_class
            pts = _sc.source.pts
            vals = _np_zeros(len(pts), int)
            for i,(row,col) in enumerate(pts[[
                _sc.source.row_name,
                _sc.source.col_name,
            ]].values):
                if (row, col) in cell_to_cluster:
                    vals[i] = cell_to_cluster[(row, col)]
            # derive cluster_id column name from cluster_column
            # e.g. employment_cluster_sum_5000  → employment_cluster_id_sum_5000  (_cluster_ present)
            #      employment_sum_0_15000_cluster  → employment_sum_0_15000_cluster_id  (ends with _cluster)
            if '_cluster_' in cluster_column:
                cluster_id_column = cluster_column.replace('_cluster_', '_cluster_id_', 1)
            else:
                cluster_id_column = cluster_column + '_id'
            pts[cluster_id_column] = vals
        #
    #
#

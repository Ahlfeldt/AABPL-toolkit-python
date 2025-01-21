from numpy import (
    array as _np_array, 
    zeros as _np_zeros,
)
from pyproj import Transformer
from pandas import DataFrame as _pd_DataFrame
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union
from aabpl.utils.general import find_column_name, arr_to_tpls
from geopandas import GeoDataFrame as _gpd_GeoDataFrame

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
    
    def create_clusters(
        self,
        pts:_pd_DataFrame,
        columns:list=['employment'],
        distance_thresholds=2500,
        make_convex:bool=True,
        row_name:str='id_y',
        col_name:str='id_x',
        cluster_suffix:str='_750m',
        plot_cluster_cells:dict=None,
    ):
        distance_thresholds = distance_thresholds if type(distance_thresholds) in [list, _np_array] else [distance_thresholds for n in columns]
        for i, (column, distance_threshold) in enumerate(zip(columns, distance_thresholds)):
            cells_with_cluster = (pts[[row_name, col_name]][pts[column + cluster_suffix]]).values
            clusters_for_column = Clustering.ClustersForColumn(
                self.grid,
                clustered_cells=cells_with_cluster,
                column=column,
                column_id=i,
            )
            self.by_column[column] = clusters_for_column
        
            clusters_for_column.merge_clusters(distance_threshold=distance_threshold)
            if make_convex:
                clusters_for_column.make_clusters_convex()
        
            clusters_for_column.match_cell_to_cluster_id()
            clusters_for_column.add_geom_to_clusters()
            clusters_for_column.add_area_to_clusters()
            clusters_for_column.add_cluster_id_to_pts(column, cluster_suffix)

        
        if not plot_cluster_cells is None:
            self.plot_clusters()

    
    # def make_cluster_orthogonally_convex(
    #         self
    #     ):
    #     """
    #     ensure all cells between (=orthogononally, not diagonally) two cluster cells are also part of the cluster
    #     exception: a cell is part of another cluster already
    #     """
    #     id_to_sums = self.grid.id_to_sums
    #     row_col_to_centroid = self.grid.row_col_to_centroid
    #     for (cluster_column, clusters) in self.by_column.items():
    #         all_clustered_cells = set()
    #         for cluster in clusters['prime_locs']:
    #             all_clustered_cells.update(cluster['cells'])
            
    #         for cluster in clusters['prime_locs']:
    #             cells_from_other_clusters = all_clustered_cells.difference(cluster['cells'])
    #             n_last_it = -1
    #             while len(cluster['cells']) != n_last_it:
    #                 cells = cluster['cells']
    #                 cells_in_convex_cluster = set(cells)
    #                 row_ids = sorted(set([row for row,col in cells]))
    #                 col_ids = sorted(set([col for row,col in cells]))
    #                 row_range = range(min(row_ids), max(row_ids)+1)
    #                 col_range = range(min(col_ids), max(col_ids)+1)
    #                 for r in row_range:
    #                     cells_to_left = [col for row, col in cells if row<r]
    #                     cells_to_right = [col for row, col in cells if row>r]
    #                     cells_same_col = [col for row, col in cells if row==r]
    #                     max_left, min_left, max_right, min_right, max_same, min_same = None, None, None, None, None, None
    #                     if len(cells_to_left) > 0:
    #                         min_left = min(cells_to_left)
    #                         max_left = max(cells_to_left)
    #                     if len(cells_to_right) > 0:
    #                         min_right = min(cells_to_right)
    #                         max_right = max(cells_to_right)
    #                     if len(cells_same_col) > 0:
    #                         min_same = min(cells_same_col)
    #                         max_same = max(cells_same_col)
    #                     max_other = max_right if max_left is None else max_left if max_right is None else max([min_left, min_right]) 
    #                     min_other = min_right if min_left is None else min_left if min_right is None else min([min_left, min_right])
    #                     max_all = max_other if max_same is None else max_same if max_other is None else min([min_same, min_other])
    #                     min_all = min_other if min_same is None else min_same if min_other is None else max([min_same, min_other])
    #                     cells_in_convex_cluster.update([(r,c) for c in range(min_all, max_all+1)])
    #                 #

    #                 for c in col_range:
    #                     cells_to_left = [row for row, col in cells if col<c]
    #                     cells_to_right = [row for row, col in cells if col>c]
    #                     cells_same_col = [row for row, col in cells if col==c]
    #                     max_left, min_left, max_right, min_right, max_same, min_same = None, None, None, None, None, None
    #                     if len(cells_to_left) > 0:
    #                         min_left = min(cells_to_left)
    #                         max_left = max(cells_to_left)
    #                     if len(cells_to_right) > 0:
    #                         min_right = min(cells_to_right)
    #                         max_right = max(cells_to_right)
    #                     if len(cells_same_col) > 0:
    #                         min_same = min(cells_same_col)
    #                         max_same = max(cells_same_col)
    #                     # max_other = max_right if max_left is None else max_left if max_right is None or max_left < max_right else max_right 
    #                     # min_other = min_right if min_left is None else min_left if min_right is None or min_left > min_right else min_right
    #                     min_other = None if max_left is None or max_right is None else max([min_left, min_right])
    #                     max_other = None if max_left is None or max_right is None else min([min_left, min_right])
    #                     min_all = min_other if min_same is None else min_same if min_other is None else min([min_same, min_other])
    #                     max_all = max_other if max_same is None else max_same if max_other is None else max([min_same, min_other])
    #                     cells_in_convex_cluster.update([(r,c) for r in range(min_all, max_all+1)])
    #                 #

    #                 cells_in_convex_cluster.difference_update(cells_from_other_clusters)
    #                 cluster['cells'] = sorted(cells_in_convex_cluster)
    #                 n_last_it = len(cluster['cells'])
                
    #             cluster['aggregate_vals'] = sum([id_to_sums[cell] for cell in cells_in_convex_cluster if cell in id_to_sums])
    #             cluster['centroid'] = _np_array([row_col_to_centroid[cell] for cell in cells_in_convex_cluster]).sum(axis=0)/len(cells_in_convex_cluster)
    #         #
    #     #
    # #
    class Cluster(object):
        """ 
        Methods:
        
        """
        def __init__(
                self,
                id:int,
                cell_in_cluster:list,
                column_id:int,
                row_col_to_centroid:dict,
                id_to_sums:dict,
        ):
            """
            bind Clusters object to grid 
            """
            self.id = id
            self.cells = [cell_in_cluster]
            self.centroid = row_col_to_centroid[cell_in_cluster]
            self.aggregate = id_to_sums[cell_in_cluster][column_id]
            self.n_cells = 1

        
        def annex_cluster(self,cluster_to_annex):
            n_current, n_neighbor = self.n_cells, cluster_to_annex.n_cells
            self.cells = self.cells + cluster_to_annex.cells
            self.aggregate += cluster_to_annex.aggregate
            n_cells = n_current + n_neighbor
            self.centroid = (
                (self.centroid[0]*n_current + cluster_to_annex.centroid[0]*n_neighbor)/n_cells,
                (self.centroid[1]*n_current + cluster_to_annex.centroid[1]*n_neighbor)/n_cells
            )
            self.n_cells = n_cells

        def change_id(self,new_id):
            self.id = new_id
            return self
        
        def add_cells_to_cluster(
                self,
                cells_to_add:set,
                row_col_to_centroid:dict,
                id_to_sums:dict,
                column_id:int,
        ):
            self.aggregate += sum([id_to_sums[cell][column_id] for cell in cells_to_add if cell in id_to_sums])
            n_cells_to_add = len(cells_to_add)
            n_cells = self.n_cells + n_cells_to_add
            self.centroid = (
                (self.centroid[0]*self.n_cells + sum([row_col_to_centroid[cell][0] for cell in cells_to_add])*n_cells_to_add)/n_cells,
                (self.centroid[1]*self.n_cells + sum([row_col_to_centroid[cell][0] for cell in cells_to_add])*n_cells_to_add)/n_cells
            )
            self.n_cells = n_cells
        
        def add_area(
                self,
                spacing:float,
        ):
            """Add area attribute as product of number of cells and square grid spacing"""
            self.area = self.n_cells * spacing**2
        #
        def add_geometry(
                self,
                row_col_to_bounds:dict,
        ):
            """add shapely polygon unaray union geometry"""
            # there are more efficient methods
            self.geometry = unary_union(
                [[Polygon(((xmin,ymin),(xmax,ymin),(xmax,ymax),(xmin,ymax)))
                for (xmin,ymin),(xmax,ymax) in [row_col_to_bounds[cell]]][0]
                for cell in self.cells]
            )
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
            self.clustered_cells = sorted(set(arr_to_tpls(clustered_cells, int)))
            
            id_to_sums = self.grid.id_to_sums    
            row_col_to_centroid = self.grid.row_col_to_centroid

            self.clusters = [Clustering.Cluster(
                id=i,
                cell_in_cluster=cell,
                column_id=column_id,
                row_col_to_centroid=row_col_to_centroid,
                id_to_sums=id_to_sums,
                ) for i, cell in enumerate(self.clustered_cells)]
            self.by_id = {cluster.id: cluster for cluster in self.clusters}
        #

        def merge_clusters(
                self,
                distance_threshold:float
            ):
            
            annexed_cluster_ids = set()
            clusters_to_delete = set([-1])
            def find_next_merge(clusters):
                for i, current_cluster in enumerate(clusters):
                    current_cluster_id = current_cluster.id
                    # Check if current cluster has any remaining cells
                    current_centroid = current_cluster.centroid
                    for neighbor_cluster in clusters[i+1:]:
                        neighbor_cluster_id = neighbor_cluster.id
                        if neighbor_cluster_id == current_cluster_id or neighbor_cluster_id in clusters_to_delete:
                            continue  # Skip unclustered cells and self-comparison
                        
                        # Compute distance between centroids
                        # distance = geopy_distance(current_centroid, neighbor_cluster.centroid).meters
                        distance = ((current_centroid[0]-neighbor_cluster.centroid[0])**2+(current_centroid[1]-neighbor_cluster.centroid[1])**2)**.5

                        if distance < distance_threshold:
                            current_cluster.annex_cluster(neighbor_cluster)
                            return neighbor_cluster_id
                        #
                    #
            while True:
                clusters = [c for c in self.clusters if not c.id in annexed_cluster_ids]
                clusters.sort(key=lambda c: (-c.n_cells, -c.aggregate))
                neighbor_cluster_id = find_next_merge(clusters)
                if neighbor_cluster_id is None:
                    break
                else:
                    annexed_cluster_ids.add(neighbor_cluster_id)
                #
            # assign ids starting at 1 from biggest (according to sum value) to largest cluster 
            clusters = [v for k,v in self.by_id.items() if not k in annexed_cluster_ids]
            clusters.sort(key=lambda c: -c.aggregate)
            self.clusters = [cluster.change_id(i+1) for i, cluster in enumerate(clusters)]
        #

        def make_clusters_convex(
                self
        ):  
            new_clusters_dict = {}
            set_clustered_cells = set(self.clustered_cells) 
            id_to_sums = self.grid.id_to_sums
            row_col_to_centroid = self.grid.row_col_to_centroid
            row_col_to_bounds = self.grid.row_col_to_bounds
            for cluster in self.clusters:
                cells = cluster.cells
                cells_in_convex_cluster = set(cells)
                cells_to_add = set()
                convex_hull = unary_union(
                    [[Polygon(((xmin,ymin),(xmax,ymin),(xmax,ymax),(xmin,ymax))) for (xmin,ymin),(xmax,ymax) in [row_col_to_bounds[cell]]][0] for cell in cells]
                ).convex_hull
                
                row_ids = sorted(set([row for row,col in cells]))
                col_ids = sorted(set([col for row,col in cells]))
                row_range = range(min(row_ids), max(row_ids)+1)
                col_range = range(min(col_ids), max(col_ids)+1)
                for r in row_range:
                    for c in col_range:
                        if not (r,c) in set_clustered_cells and convex_hull.contains(Point(row_col_to_centroid[(r,c)])):
                            cells_in_convex_cluster.add((r,c))
                            set_clustered_cells.add((r,c))
                            cells_to_add.add((r,c))
                cluster.add_cells_to_cluster(cells_to_add, row_col_to_centroid=row_col_to_centroid, id_to_sums=id_to_sums)
            
            self.clusters.sort(key=lambda c: -c.aggregate)
            for i, cluster in enumerate(self.clusters):
                cluster.change_id(i+1)

            # self.by_column = new_clusters_dict
            #
        
        def match_cell_to_cluster_id(self):
            cell_to_cluster = {}
            for cluster in self.clusters:
                cell_to_cluster.update({cell: cluster.id for cell in cluster.cells})
            self.cell_to_cluster_id = cell_to_cluster
        #

        def add_geom_to_clusters(self):
            row_col_to_bounds = self.grid.row_col_to_bounds
            for cluster in self.clusters:
                cluster.add_geometry(row_col_to_bounds)
            
        def add_area_to_clusters(self):
            for cluster in self.clusters:
                cluster.add_area(spacing=self.grid.spacing)
        #

        def add_cluster_id_to_pts(self, column, cluster_suffix):
            cluster_column = column + cluster_suffix
            cell_to_cluster = self.cell_to_cluster_id
            pts = self.grid.search.source.pts
            vals = _np_zeros(len(pts),int)#-1
            for i,(row,col) in enumerate(pts[[
                self.grid.search.source.row_name,
                self.grid.search.source.col_name,
            ]].values):
                if (row, col) in cell_to_cluster: 
                    vals[i] = cell_to_cluster[(row, col)]
            pts[cluster_column] = vals
        #
    #
#

from numpy import (
    array as _np_array, 
    linspace as _np_linspace,
    stack as _np_stack,
    arange as _np_arange, 
    unique as _np_unique,
    zeros as _np_zeros,
)
from pyproj import Transformer
from pandas import DataFrame as _pd_DataFrame
from math import log10 as _math_log10
from aabpl.utils.general import flatten_list, arr_to_tpls
from aabpl.illustrations.plot_utils import map_2D_to_rgb, get_2D_rgb_colobar_kwargs
from .radius_search_class import (
    aggregate_point_data_to_cells,
    aggreagate_point_data_to_disks_vectorized
)
from .pts_to_offset_regions import assign_points_to_cell_regions
from aabpl.valid_area import disk_cell_intersection_area
from aabpl.testing.test_performance import time_func_perf
import matplotlib.pyplot as plt
from matplotlib.pyplot import get_cmap as _plt_get_cmap
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union
from geopandas import GeoDataFrame as _gpd_GeoDataFrame

class Bounds(object):
    __slots__ = ('xmin', 'xmax', 'ymin', 'ymax', 'np_array_of_bounds') # use this syntax to save some memory. also only create vars that are really neccessary
    def __init__(self, xmin:float, xmax:float, ymin:float, ymax:float):
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
    #
#

# TODO potentially add nestes GridCell classtype the 
class GridCell(object):
    # __slots__ = ('id', 'row_col_nr', ...) # use this syntax to save some memory. also only create vars that are really neccessary
    def __init__(self, row_nr, col_nr, y_steps, x_steps):
        self.id = (row_nr,col_nr)
        self.centroid = (y_steps[row_nr:row_nr+2].sum()/2,x_steps[col_nr:col_nr+2].sum()/2)
        self.bounds = Bounds(xmin=x_steps[col_nr], ymin=y_steps[row_nr], xmax=x_steps[col_nr+1], ymax=y_steps[row_nr+1])
        self.pt_ids = []
        self.excluded = None

    def add_pt_id(self, pt_id):
        self.pt_ids = [*self.pt_ids, pt_id]
        
    def add_pt_ids(self, pt_ids):
        self.pt_ids = [*self.pt_ids, *pt_ids]
    
    def set_excluded_area(self,excluded_area):
        pass

    def make_sparse(self):
        # self.pt_ids # make _np_array or tuple
        # tighten bounds
        pass            
    def tighten_bounds(self, pt_ids):
        self.pt_ids = [*self.pt_ids, *pt_ids]
    
    def plot(self, facecolor, edgecolor, ):
        pass
#


class Grid(object):
    """

    """
    @time_func_perf
    def __init__(
        self,
        xmin:float,
        xmax:float,
        ymin:float,
        ymax:float,
        initial_crs:str,
        local_crs:str,
        set_fixed_spacing:float=None,
        r:float=750,
        n_points:int=10000,
        silent = False,
        ):

        """

        """
        if set_fixed_spacing:
            spacing = set_fixed_spacing
        else:
            # find optimal spacing TODO
            print("TODO find optimal spacing for",r, n_points)
            spacing = 1.
        self.spacing = spacing

        # TODO total_bounds should also contain excluded area if not contained 
        # min(points.total_bounds+r, max(points.total_bounds, excluded_area_total_bound))  
        self.initial_crs = initial_crs
        self.local_crs = local_crs
        x_padding = ((xmin-xmax) % spacing)/2
        y_padding = ((ymin-ymax) % spacing)/2
        self.total_bounds = total_bounds = Bounds(xmin=xmin-x_padding,xmax=xmax+x_padding,ymin=ymin-y_padding,ymax=ymax+y_padding)
        self.n_x_steps = n_x_steps = -int((self.total_bounds.xmin-self.total_bounds.xmax)/spacing)+1 # round up
        self.n_y_steps = n_y_steps = -int((self.total_bounds.ymin-self.total_bounds.ymax)/spacing)+1 # round up 
        
        self.x_steps = x_steps = _np_linspace(total_bounds.xmin, total_bounds.xmax, n_x_steps)
        self.y_steps = y_steps = _np_linspace(total_bounds.ymin, total_bounds.ymax, n_y_steps)
        self.id_y_mult = id_y_mult = 10**(int(_math_log10(n_x_steps))+1)
        
        self.row_ids = _np_arange(n_y_steps-1)
        self.col_ids = _np_arange(n_x_steps-1)
        
        self.ids = tuple(flatten_list([[(row_id, col_id) for col_id in range(n_x_steps-1)] for row_id in range(n_y_steps-1)]))
        class CellDict(dict):
            def __init__(self, x_steps, y_steps, id_y_mult):
                self.x_steps = x_steps
                self.y_steps = y_steps
                self.id_y_mult = id_y_mult

            def id_to_row_col(self,id:int): 
                return (id // self.id_y_mult, id % self.id_y_mult)
            
            def row_col_to_id(self,row_nr:int,col_nr:int): 
                return row_nr * self.id_y_mult + col_nr
            
            def add_new(self,row,col):
                setattr(self, str(self.row_col_to_id(row,col)), GridCell(
                    row, col, self.x_steps, self.y_steps
                ))
            
            def get_by_row_col(self,row,col):
                return getattr(self, str(self.row_col_to_id(row,col)))
            
            def get_by_id(self,id):
                return getattr(self, str(id))
            
            def get_or_create(self,row,col):
                id = str(self.row_col_to_id(row,col))
                if not hasattr(self, id):
                    self.add_new(row,col)
                return self.get_by_id(id)
            
            def add_pts(self, pts, row, col):
                cell = self.get_or_create(row,col)
                cell.add_pts(pts)
            
            def add_pt(self, pt, row, col):
                cell = self.get_or_create(row,col)
                cell.add_pt(pt)
            
        self.row_col_stack = _np_stack([
                    _np_array([[row_id for col_id in range(n_x_steps-1)] for row_id in range(n_y_steps-1)]).flatten(),
                    _np_array([[col_id for col_id in range(n_x_steps-1)] for row_id in range(n_y_steps-1)]).flatten(),
                ])
        self.cells = CellDict(x_steps=x_steps, y_steps=y_steps, id_y_mult=id_y_mult,)

        self.cells = _np_array([[
            GridCell(row_id, col_id, y_steps=y_steps, x_steps=x_steps) for col_id in range(n_x_steps-1)
            ] for row_id in range(n_y_steps-1)]).flatten()

        self.row_col_to_centroid = {g_row_col:centroid for (g_row_col,centroid) in flatten_list([
                [((row_id,col_id),(x_steps[col_id:col_id+2].mean(), y_steps[row_id:row_id+2].mean())) for col_id in range(n_x_steps-1)] 
                for row_id in range(n_y_steps-1)]
                )}
        self.centroids = _np_array([centroid for centroid in flatten_list([
                [(x_steps[col_id:col_id+2].mean(), y_steps[row_id:row_id+2].mean()) for col_id in range(n_x_steps-1)] 
                for row_id in range(n_y_steps-1)]
                )])
        self.id_to_bounds = {row_id*id_y_mult+col_id: bounds for ((row_id,col_id),bounds) in flatten_list([
                [((row_id,col_id),((x_steps[col_id], y_steps[row_id]), (x_steps[col_id+1], y_steps[row_id+1]))) for col_id in range(n_x_steps-1)] 
                for row_id in range(n_y_steps-1)]
                )}
        self.row_col_to_bounds = {(row_id,col_id): bounds for ((row_id,col_id),bounds) in flatten_list([
                [((row_id,col_id),((x_steps[col_id], y_steps[row_id]), (x_steps[col_id+1], y_steps[row_id+1]))) for col_id in range(n_x_steps-1)] 
                for row_id in range(n_y_steps-1)]
                )}
        self.bounds = flatten_list([
                [((x_steps[col_id], y_steps[row_id]), (x_steps[col_id+1], y_steps[row_id+1])) for col_id in range(n_x_steps-1)] 
                for row_id in range(n_y_steps-1)])
        self.cell_dict = dict()
        self.clusters = dict()
        if not silent:
            print('Create grid with '+str(n_y_steps-1)+'x'+str(n_x_steps-1)+'='+str((n_y_steps-1)*(n_x_steps-1)))
        #
    #
    # add functions
    aggregate_point_data_to_cells = aggregate_point_data_to_cells
    # aggregate_point_data_to_nested_cells = aggregate_point_data_to_nested_cells
    assign_points_to_cell_regions = assign_points_to_cell_regions
    aggreagate_point_data_to_disks_vectorized = aggreagate_point_data_to_disks_vectorized
    # aggreagate_point_data_to_disks_vectorized_nested = aggreagate_point_data_to_disks_vectorized_nested
    # append a variable to pts that indicates the share of valid area float[0,1] 
    disk_cell_intersection_area = disk_cell_intersection_area

    def add_cluster_tags_to_cells(
            self,
            cells_with_cluster:_np_array,
            cluster_tag:str='employment',
    ):
        self.clusters[cluster_tag] = {
            'tag': cluster_tag,
            'cells': sorted(set(arr_to_tpls(cells_with_cluster, int)))
            }

    def merge_clusters(
            self,
            distance_thresholds:float
        ):

        for (n, (cluster_column, clusters)), distance_threshold in zip(enumerate(self.clusters.items()), distance_thresholds):
            id_to_sums = self.id_to_sums    
            row_col_to_centroid = self.row_col_to_centroid

            prime_locs = [{
                'id':i,
                'cells': [cell],
                'centroid': row_col_to_centroid[cell],
                'sum': id_to_sums[cell][n] # TODO select correct column
                } for i, cell in enumerate(clusters['cells'])]
            prime_locs_by_id = {pl['id']: pl for pl in prime_locs}
            deleted_clusters = set()
            clusters_to_delete = set([-1])
            def find_next_merge(prime_locs):
                for i, current_cluster_id in enumerate([pl['id'] for pl in prime_locs]):
                    current_cluster = prime_locs_by_id[current_cluster_id]
                    
                    # Check if current cluster has any remaining cells
                    current_cluster_cells = current_cluster['cells']
                    current_centroid = current_cluster['centroid']

                    for neighbor_cluster_id in [pl['id'] for pl in prime_locs[i+1:]]:
                        if neighbor_cluster_id == current_cluster_id or neighbor_cluster_id in clusters_to_delete:
                            continue  # Skip unclustered cells and self-comparison
                        
                        neighbor_cluster = prime_locs_by_id[neighbor_cluster_id]

                        neighbor_centroid = neighbor_cluster['centroid']

                        # Compute distance between centroids
                        # distance = geopy_distance(current_centroid, neighbor_centroid).meters
                        distance = ((current_centroid[0]-neighbor_centroid[0])**2+(current_centroid[1]-neighbor_centroid[1])**2)**.5

                        if distance < distance_threshold:
                            n_current, n_neighbor = len(current_cluster_cells), len(neighbor_cluster['cells'])
                            current_cluster['cells'] = current_cluster_cells + neighbor_cluster['cells']
                            current_cluster['sum'] += neighbor_cluster['sum']
                            current_cluster['centroid'] = (
                                (current_centroid[0]*n_current + neighbor_centroid[0]*n_neighbor)/(n_current + n_neighbor),
                                (current_centroid[1]*n_current + neighbor_centroid[1]*n_neighbor)/(n_current + n_neighbor)
                            )
                            return neighbor_cluster_id
                        #
                    #
            while len(clusters_to_delete) > 0:
                prime_locs = [c for c in prime_locs if not c['id'] in clusters_to_delete]
                prime_locs.sort(key=lambda c: (-len(c['cells']), -c['sum']))
                neighbor_cluster_id = find_next_merge(prime_locs)
                clusters_to_delete = set()
                if not neighbor_cluster_id is None:
                    clusters_to_delete.add(neighbor_cluster_id)
                    deleted_clusters.add(neighbor_cluster_id)
                    #
                #
            # assign ids starting at 1 from biggest (according to sum value) to largest cluster 
            prime_locs = [v for k,v in prime_locs_by_id.items() if not k in deleted_clusters]
            prime_locs.sort(key=lambda c: -c['sum'])
            prime_locs = [{**c, 'id':i+1} for i, c in enumerate(prime_locs)]
            clusters['prime_locs'] = prime_locs
        #
    #

    def make_cluster_convex(
            self
    ):  
        new_clusters_dict = {}
        for n, cluster_column in enumerate(self.clusters):
            clusters = self.clusters[cluster_column]
            all_clustered_cells = set()
            for cluster in clusters['prime_locs']:
                all_clustered_cells.update(cluster['cells'])
            new_prime_locs = []
            for cluster in clusters['prime_locs']:
                id_to_sums = self.id_to_sums
                row_col_to_centroid = self.row_col_to_centroid
                row_col_to_bounds = self.row_col_to_bounds
                cells = cluster['cells']
                cells_in_convex_cluster = set(cells)
                convex_hull = unary_union(
                    [[Polygon(((xmin,ymin),(xmax,ymin),(xmax,ymax),(xmin,ymax))) for (xmin,ymin),(xmax,ymax) in [self.row_col_to_bounds[cell]]][0] for cell in cells]
                ).convex_hull
                
                row_ids = sorted(set([row for row,col in cells]))
                col_ids = sorted(set([col for row,col in cells]))
                row_range = range(min(row_ids), max(row_ids)+1)
                col_range = range(min(col_ids), max(col_ids)+1)
                for r in row_range:
                    for c in col_range:
                        if not (r,c) in all_clustered_cells and convex_hull.contains(Point(row_col_to_centroid[(r,c)])):
                            cells_in_convex_cluster.add((r,c))
                            all_clustered_cells.add((r,c))
                new_prime_locs.append(
                    {**cluster,
                    'cells': sorted(cells_in_convex_cluster),
                    'sum': sum([id_to_sums[cell][n] for cell in cells_in_convex_cluster if cell in id_to_sums]),
                    'centroid': _np_array([row_col_to_centroid[cell] for cell in cells_in_convex_cluster]).sum(axis=0)/len(cells_in_convex_cluster)
                })
            new_clusters_dict[cluster_column] = {**clusters, 'prime_locs': new_prime_locs}
        
        self.clusters = new_clusters_dict
        #
    #
    def make_cluster_orthogonally_convex(
            self
        ):
        """
        ensure all cells between (=orthogononally, not diagonally) two cluster cells are also part of the cluster
        exception: a cell is part of another cluster already
        """
        id_to_sums = self.id_to_sums
        row_col_to_centroid = self.row_col_to_centroid
        for (cluster_column, clusters) in self.clusters.items():
            all_clustered_cells = set()
            for cluster in clusters['prime_locs']:
                all_clustered_cells.update(cluster['cells'])
            
            for cluster in clusters['prime_locs']:
                cells_from_other_clusters = all_clustered_cells.difference(cluster['cells'])
                n_last_it = -1
                while len(cluster['cells']) != n_last_it:
                    cells = cluster['cells']
                    cells_in_convex_cluster = set(cells)
                    row_ids = sorted(set([row for row,col in cells]))
                    col_ids = sorted(set([col for row,col in cells]))
                    # row_to_cols = {row: [c for r,c in cells if r==row] for row in row_ids}
                    # col_to_rows = {col: [r for r,c in cells if c==col] for col in col_ids}
                    # row_to_min_col = {k:min([min(v), max([
                    #     min([col for row, col in cells if row<k]),
                    #     min([col for row, col in cells if row>k])
                    # ])]) for k,v in row_to_cols.items()}
                    
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
                    cluster['cells'] = sorted(cells_in_convex_cluster)
                    n_last_it = len(cluster['cells'])
                
                cluster['aggregate_vals'] = sum([id_to_sums[cell] for cell in cells_in_convex_cluster if cell in id_to_sums])
                cluster['centroid'] = _np_array([row_col_to_centroid[cell] for cell in cells_in_convex_cluster]).sum(axis=0)/len(cells_in_convex_cluster)
            #
        #
    #
    def connect_cells_to_clusters(self):
        for (cluster_column, clusters) in self.clusters.items():
            clusters['cell_to_cluster'] = {}
            for cluster in clusters['prime_locs']:
                clusters['cell_to_cluster'].update({cell: cluster['id'] for cell in cluster['cells']})
    
    def add_geom_to_cluster(self):
        for (cluster_column, clusters) in self.clusters.items():
            for cluster in clusters['prime_locs']:
                cluster['geometry'] = unary_union(
                    [[Polygon(((xmin,ymin),(xmax,ymin),(xmax,ymax),(xmin,ymax))) for (xmin,ymin),(xmax,ymax) in [self.row_col_to_bounds[cell]]][0] for cell in cluster['cells']]
                )
                cluster['area'] = len(cluster['cells']) * self.spacing**2

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
        for column in columns:
            cells_with_cluster = (pts[[row_name, col_name]][pts[column + cluster_suffix]]).values
            self.add_cluster_tags_to_cells(
                cells_with_cluster=cells_with_cluster,
                cluster_tag=column,
            )
        
        distance_thresholds = distance_thresholds if type(distance_thresholds) in [list, _np_array] else [distance_thresholds for n in columns]
        self.merge_clusters(distance_thresholds=distance_thresholds)
        if make_convex:
            self.make_cluster_convex()
            # self.make_cluster_orthogonally_convex()
        
        self.connect_cells_to_clusters()
        self.add_geom_to_cluster()

        for column in columns:
            cluster_column = column + cluster_suffix
            cell_to_cluster = self.clusters[column]['cell_to_cluster']
            vals = _np_zeros(len(pts),int)#-1
            for i,(row,col) in enumerate(pts[[row_name, col_name]].values):
                if (row, col) in cell_to_cluster: 
                    vals[i] = cell_to_cluster[(row, col)]
            pts[cluster_column] = vals
        
        if not plot_cluster_cells is None:
            self.plot_all_clusters()

                    

    def create_sparse_grid(self):
        return
        # TODO shall this be another class? A new instane of the grid? Or create a sparse copy within itself? Or overwrite itself with an own c?
        print("Make this grid sparse")
    
    def create_nested_grid(self):
        print("Create nested grid")
    #

    
    def save_full_grid(
            self,
            filename:str="full_grid",
            file_format:str=['shp','csv'][0],
            # target_crs:str="EPSG:4326"
        ):
        for (cluster_column, clusters) in self.clusters.items():
            
            cell_to_cluster = self.clusters[cluster_column]['cell_to_cluster']
            c_ids = _np_zeros(len(self.ids),int)#-1
            sums = _np_zeros(len(self.ids),int)
            polys = []
            id_to_sums = self.id_to_sums
            centroids = _np_array(list(self.row_col_to_centroid.values()))
            transformer = Transformer.from_crs(crs_from=self.local_crs, crs_to=self.initial_crs, always_xy=True)
            centroids_x, centroids_y = transformer.transform(centroids[:,0], centroids[:,1])
            
            for (i, row_col), ((xmin,ymin),(xmax,ymax)) in zip(enumerate(self.ids), self.id_to_bounds.values()):
                if row_col in cell_to_cluster: 
                    c_ids[i] = cell_to_cluster[row_col]
                if row_col in id_to_sums: 
                    sums[i] = id_to_sums[row_col]
                polys.append(Polygon(((xmin,ymin),(xmax,ymin),(xmax,ymax),(xmin,ymax))))
            df = _gpd_GeoDataFrame({
                'centroid_x': centroids_x,
                'centroid_y': centroids_y,
                'cluster_id': c_ids,
                'sum': sums,},
                geometry=polys,
                crs=self.local_crs
                )
            df.to_crs(self.initial_crs, inplace=True)
            # save
            filename = filename + (("_"+cluster_column) if len(self.clusters) > 1 else '') + '.'+file_format
            if file_format == 'shp':
                df.to_file(filename, driver="ESRI Shapefile", index=False)
            elif file_format == 'csv':
                df.to_csv(filename, index=False)
            else:
                raise ValueError('Unknown file_format:',file_format,'Choose on out of shp, csv')
        #
    
    def save_sparse_grid(
            self,
            filename:str="sparse_grid",
            file_format:str=['shp','csv'][0],
            # target_crs:str="EPSG:4326"
        ):
        """
        saving all grid cell filled with
        """
        for (cluster_column, clusters) in self.clusters.items():
            
            cell_to_cluster = self.clusters[cluster_column]['cell_to_cluster']
            id_to_sums = self.id_to_sums
            polys = []
            c_ids = _np_zeros(len(id_to_sums),int)
            sums = _np_zeros(len(id_to_sums),float)
            centroids = _np_array(list(self.row_col_to_centroid.values()))
            transformer = Transformer.from_crs(crs_from=self.local_crs, crs_to=self.initial_crs, always_xy=True)
            centroids_x_full, centroids_y_full = transformer.transform(centroids[:,0], centroids[:,1])
            centroids_x = _np_zeros(len(id_to_sums),float)
            centroids_y = _np_zeros(len(id_to_sums),float)
            i = 0
            for row_col, ((xmin,ymin),(xmax,ymax)), c_x, c_y in zip(self.ids, self.id_to_bounds.values(), centroids_x_full, centroids_y_full):
                if row_col in id_to_sums: 
                    if row_col in cell_to_cluster: 
                        c_ids[i] = cell_to_cluster[row_col]
                    sums[i] = id_to_sums[row_col]
                    centroids_x[i] = c_x 
                    centroids_y[i] = c_y 
                    polys.append(Polygon(((xmin,ymin),(xmax,ymin),(xmax,ymax),(xmin,ymax))))
                    i += 1
                #
            #
            df = _gpd_GeoDataFrame({
                'centroid_x': centroids_x,
                'centroid_y': centroids_y,
                'cluster_id': c_ids,
                'sum': sums,},
                geometry=polys,
                crs=self.local_crs
                )
            df.to_crs(self.initial_crs, inplace=True)
            # save
            filename = filename + (("_"+cluster_column) if len(self.clusters) > 1 else '') + '.'+file_format
            if file_format == 'shp':
                df.to_file(filename, driver="ESRI Shapefile", index=False)
            elif file_format == 'csv':
                df.to_csv(filename, index=False)
            else:
                raise ValueError('Unknown file_format:',file_format,'Choose on out of shp, csv')
        #
    
    def save_grid_clusters(
            self,
            filename:str="grid_clusters",
            file_format:str=['shp','csv'][0],
            # target_crs:str="EPSG:4326"
        ):
        for (cluster_column, clusters) in self.clusters.items():
            df = _gpd_GeoDataFrame({
                'centroid_x': [pl['centroid'][0] for pl in clusters['prime_locs']],
                'centroid_y': [pl['centroid'][1] for pl in clusters['prime_locs']],
                'cluster_id': [pl['id'] for pl in clusters['prime_locs']],
                'sum': [pl['sum'] for pl in clusters['prime_locs']],
                "n_cells": [len(pl['cells']) for pl in clusters['prime_locs']],
                'area': [pl['area'] for pl in clusters['prime_locs']],
                },
                geometry = [pl['geometry'] for pl in clusters['prime_locs']],
                crs=self.local_crs)
            transformer = Transformer.from_crs(crs_from=self.local_crs, crs_to=self.initial_crs, always_xy=True)
            df['centroid_x'], df['centroid_y'] = transformer.transform(df['centroid_x'], df['centroid_y'])
            df.to_crs(self.initial_crs, inplace=True)
            # save
            filename = filename + (("_"+cluster_column) if len(self.clusters) > 1 else '') + '.'+file_format
            if file_format == 'shp':
                df.to_file(filename, driver="ESRI Shapefile", index=False)
            elif file_format == 'csv':
                df.to_csv(filename, index=False)
            else:
                raise ValueError('Unknown file_format:',file_format,'Choose on out of shp, csv')
        #
    #

    def plot_cell_sums(
            self, fig=None, ax=None, filename:str=''):
        if ax is None:
            fig, ax = plt.subplots(nrows=len(self.clusters), figsize=(10,10))
        
        id_to_sums = self.id_to_sums
        imshow_kwargs = {
            'xmin':self.x_steps.min(),
            'ymin':self.y_steps.min(),
            'xmax':self.x_steps.max(),
            'ymax':self.y_steps.max(),
        }
        
        extent=[imshow_kwargs['xmin'],imshow_kwargs['xmax'],imshow_kwargs['ymax'],imshow_kwargs['ymin']]
        cmap = _plt_get_cmap('Reds')
        cmap.set_under('#ccc')
        max_sum = max(list(id_to_sums.values()))
        X = _np_array([[id_to_sums[(row,col)][0] if ((row,col)) in id_to_sums else 0 for col in  self.col_ids] for row in self.row_ids])
        ux = _np_unique(X)
        minX = min(ux[ux!=0])
        print("min X", min(X), max(X))
        # p = ax.imshow(X=X, interpolation='none', cmap=cmap, vmin=1e-5,vmax=max_sum, extent=extent)
        p = ax.pcolormesh(X, cmap=cmap, vmin=minX/2,vmax=max_sum)
        cb = plt.colorbar(p)
        ax.set_xlabel('x/lon') 
        ax.set_ylabel('y/lat') 
        ax.title.set_text('Sums in grid cells')
        if filename:
            fig.savefig(filename, dpi=300, bbox_inches="tight")
    


    def plot_grid_ids(self, fig=None, ax=None, filename:str='',):
        if ax is None:
            fig, ax = plt.subplots(nrows=3, figsize=(15,25))
        imshow_kwargs = {
            'xmin':self.x_steps.min(),
            'ymin':self.y_steps.min(),
            'xmax':self.x_steps.max(),
            'ymax':self.y_steps.max(),
        }
        extent=[imshow_kwargs['xmin'],imshow_kwargs['xmax'],imshow_kwargs['ymax'],imshow_kwargs['ymin']]
        X = _np_array([[map_2D_to_rgb(x,y, **imshow_kwargs) for x in  self.x_steps[:-1]] for y in self.y_steps[:-1]])
        # ax.flat[0].imshow(X=X, interpolation='none', extent=extent)
        # ax.flat[0].pcolormesh([self.x_steps, self.y_steps], X)
        ax.flat[0].pcolormesh(X, edgecolor="black", linewidth=1/max([self.n_x_steps, self.n_y_steps])/1.35)
        # ax.flat[0].set_aspect(2)
        colorbar_kwargs = get_2D_rgb_colobar_kwargs(**imshow_kwargs)
        cb = plt.colorbar(**colorbar_kwargs[2], ax=ax.flat[0])
        cb.ax.set_xlabel("diagonal")
        cb = plt.colorbar(**colorbar_kwargs[0], ax=ax.flat[0])
        cb.ax.set_xlabel("x/lon")
        cb = plt.colorbar(**colorbar_kwargs[1], ax=ax.flat[0])
        cb.ax.set_xlabel("y/lat") 
        ax.flat[0].set_xlabel('x/lon') 
        ax.flat[0].set_ylabel('y/lat') 
        ax.flat[0].title.set_text("Grid lat / lon coordinates")
        # ax.flat[0].set_xticks(self.x_steps, minor=True)
        # ax.flat[0].set_yticks(self.y_steps, minor=True)
        # ax.flat[0].grid(which='minor', color='w', linestyle='-', linewidth=0.002)

        imshow_kwargs = {
            'xmin':self.col_ids.min(),
            'ymin':self.row_ids.min(),
            'xmax':self.col_ids.max(),
            'ymax':self.row_ids.max(),
        }
        extent=[imshow_kwargs['xmin'],imshow_kwargs['xmax'],imshow_kwargs['ymax'],imshow_kwargs['ymin']]

        X = _np_array([[map_2D_to_rgb(x,y, **imshow_kwargs) for x in  self.col_ids] for y in self.row_ids])
        ax.flat[1].imshow(X=X, interpolation='none', extent=extent)
        # ax.flat[1].set_aspect(2)
        colorbar_kwargs = get_2D_rgb_colobar_kwargs(**imshow_kwargs)
        cb = plt.colorbar(**colorbar_kwargs[2], ax=ax.flat[1])
        cb.ax.set_xlabel("diagonal")
        cb = plt.colorbar(**colorbar_kwargs[0], ax=ax.flat[1])
        cb.ax.set_xlabel("col nr")
        cb = plt.colorbar(**colorbar_kwargs[1], ax=ax.flat[1])
        cb.ax.set_xlabel("row nr") 
        ax.flat[1].set_xlabel('row nr') 
        ax.flat[1].set_ylabel('col nr') 
        ax.flat[1].title.set_text("Grid row / col indices")
        # ax.flat[1].set_xticks(self.col_ids, minor=True)
        # ax.flat[1].set_yticks(self.row_ids, minor=True)
        # ax.flat[1].grid(which='minor', color='black', linestyle='-', linewidth=0.003)
        
        X = _np_array([[len(self.id_to_pt_ids[(row_id, col_id)]) if (row_id, col_id) in self.id_to_pt_ids else 0 for col_id in self.col_ids] for row_id in self.row_ids])
        p = ax.flat[2].pcolormesh(X, cmap='Reds')
        ax.flat[2].set_xlabel('row nr') 
        ax.flat[2].set_ylabel('col nr') 
        plt.colorbar(p)
        if filename:
            fig.savefig(filename, dpi=300, bbox_inches="tight")
    
    def plot_clusters(self, cluster_column:str, fig=None, ax=None, filename:str='',):
        if ax is None:
            fig, ax = plt.subplots(nrows=len(self.clusters), figsize=(15,25))
        
        clusters = self.clusters[cluster_column]
        cell_to_cluster = clusters['cell_to_cluster']
        if len(cell_to_cluster)==0:
            print("No clusters detected.")
            return
        max_cluster_id = max(list(cell_to_cluster.values()))
        imshow_kwargs = {
            'xmin':self.x_steps.min(),
            'ymin':self.y_steps.min(),
            'xmax':self.x_steps.max(),
            'ymax':self.y_steps.max(),
        }
        
        extent=[imshow_kwargs['xmin'],imshow_kwargs['xmax'],imshow_kwargs['ymax'],imshow_kwargs['ymin']]
        cmap = _plt_get_cmap('Reds')
        cmap.set_under('#ccc')
        X = _np_array([[cell_to_cluster[(row,col)]/max_cluster_id if (row,col) in cell_to_cluster else 0 for col in  self.col_ids] for row in self.row_ids])
        p = ax.imshow(X=X, interpolation='none', cmap=cmap, vmin=.5/max_cluster_id,vmax=1.0, extent=extent)
        # p = ax.pcolormesh(X, cmap=cmap, edgecolor="black", linewidth=1/max([self.n_x_steps, self.n_y_steps])/1.35)
        # cb = plt.colorbar(p)
        for cluster in clusters['prime_locs']:
            ax.annotate(cluster['id'], xy=cluster['centroid'], fontsize=10)
        ax.set_xlabel('x/lon') 
        ax.set_ylabel('y/lat') 
        ax.title.set_text(str(len(clusters['prime_locs']))+' clusters for '+str(cluster_column))
    
    def plot_all_clusters(self, fig=None, axs=None, filename:str=''):
        
        if axs is None:
            fig, axs = plt.subplots(nrows=len(self.clusters), figsize=(10,10*len(self.clusters)))

        for i, cluster_column in enumerate(self.clusters):
            ax = axs if not hasattr(axs, 'flat') else axs.flat[i]
            self.plot_clusters(cluster_column=cluster_column, ax=ax)
        if filename:
            fig.savefig(filename, dpi=300, bbox_inches="tight")
#

class ExludedArea:
    def __init__(self,excluded_area_geometry_or_list, grid:Grid):
        # recursively split exluded area geometry along grid 
        # then sort it into grid cell
        
        pass
#



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


def add_cluster_tags_to_cells(
        self,
        cells_with_cluster:_np_array,
        cluster_tag:str='employment',
):
    self.clusters_by_column[cluster_tag] = {
        'tag': cluster_tag,
        'cells': sorted(set(arr_to_tpls(cells_with_cluster, int)))
        }

def merge_clusters(
        self,
        distance_thresholds:float
    ):

    for (n, (cluster_column, clusters)), distance_threshold in zip(enumerate(self.clusters_by_column.items()), distance_thresholds):
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
    for n, cluster_column in enumerate(self.clusters_by_column):
        clusters = self.clusters_by_column[cluster_column]
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
                [[Polygon(((xmin,ymin),(xmax,ymin),(xmax,ymax),(xmin,ymax))) for (xmin,ymin),(xmax,ymax) in [row_col_to_bounds[cell]]][0] for cell in cells]
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
    
    self.clusters_by_column = new_clusters_dict
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
    for (cluster_column, clusters) in self.clusters_by_column.items():
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
    for (cluster_column, clusters) in self.clusters_by_column.items():
        clusters['cell_to_cluster'] = {}
        for cluster in clusters['prime_locs']:
            clusters['cell_to_cluster'].update({cell: cluster['id'] for cell in cluster['cells']})

def add_geom_to_cluster(self):
    for (cluster_column, clusters) in self.clusters_by_column.items():
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
        cell_to_cluster = self.clusters_by_column[column]['cell_to_cluster']
        vals = _np_zeros(len(pts),int)#-1
        for i,(row,col) in enumerate(pts[[row_name, col_name]].values):
            if (row, col) in cell_to_cluster: 
                vals[i] = cell_to_cluster[(row, col)]
        pts[cluster_column] = vals
    
    if not plot_cluster_cells is None:
        self.plot_clusters()
def save_full_grid(
        self,
        filename:str="full_grid",
        file_format:str=['shp','csv'][0],
):
    """save each grid cell with attributes on their Polygon, centroid, sum of indicator(s), and cluster id
    
    Args:
    filename (str):
        name of the output file excluding file format extension. It can contain full path like 'output_folder/fname'
    file_format (str):
        format in which the file shall be saved. Currently available options are 'shp' and 'csv'. Extension will be appended to filename.
    """
    c_ids = _np_zeros((len(self.ids), len(self.clusters_by_column)),int)#-1
    sums = _np_zeros((len(self.ids), len(self.clusters_by_column)),int)
    polys = []
    id_to_sums = self.id_to_sums
    centroids = _np_array(list(self.row_col_to_centroid.values()))
    transformer = Transformer.from_crs(crs_from=self.local_crs, crs_to=self.initial_crs, always_xy=True)
    centroids_x, centroids_y = transformer.transform(centroids[:,0], centroids[:,1])
    clusters_for_columns = list(self.clusters_by_column.values())
    for (i, row_col), ((xmin,ymin),(xmax,ymax)) in zip(enumerate(self.ids), self.bounds):
        for clusters_for_column in clusters_for_columns:
            if row_col in clusters_for_column['cell_to_cluster']: 
                c_ids[i] = clusters_for_column['cell_to_cluster'][row_col]
        if row_col in id_to_sums: 
            sums[i] = id_to_sums[row_col]
        polys.append(Polygon(((xmin,ymin),(xmax,ymin),(xmax,ymax),(xmin,ymax))))
    df = _gpd_GeoDataFrame({
        'centroid_x': centroids_x,
        'centroid_y': centroids_y,
        }, geometry=polys,
        crs=self.local_crs
        )
    if len(self.clusters_by_column)<=1:
        df['cluster_id'] = c_ids
        df['sum'] = sums
    else:
        for j, column in enumerate(self.clusters_by_column):
            c_id_colname = find_column_name("cluster_id", column, df.columns, 10 if file_format=='shp' else 20)
            agg_colname = find_column_name("sum_radius", column, df.columns, 10 if file_format=='shp' else 20)
            df[c_id_colname] = c_ids[:,j]
            df[agg_colname] = sums[:,j]
    df.to_crs(self.initial_crs, inplace=True)
    # save
    filename = filename +'.'+file_format
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
    ):
    """
    save each grid cell with attributes on their Polygon, centroid, sum of indicator(s), and cluster id
    
    Args:
    filename (str):
        name of the output file excluding file format extension. It can contain full path like 'output_folder/fname'
    file_format (str):
        format in which the file shall be saved. Currently available options are 'shp' and 'csv'. Extension will be appended to filename.
    
    """
    
        
    id_to_sums = self.id_to_sums
    x_steps = self.x_steps
    y_steps = self.y_steps
    col_ids = self.col_ids
    polys = []
    transformer = Transformer.from_crs(crs_from=self.local_crs, crs_to=self.initial_crs, always_xy=True)
    centroids_x_full, centroids_y_full = transformer.transform(self.centroids[:,0], self.centroids[:,1])
    centroids_x = _np_zeros(self.n_cells,float)
    centroids_y = _np_zeros(self.n_cells,float)
    sums = _np_zeros((self.n_cells, len(self.clusters_by_column)),float)
    c_ids = _np_zeros((self.n_cells, len(self.clusters_by_column)),int)
    i = 0
    js_clusters_for_columns = [x for x in enumerate(self.clusters_by_column.values())]
    for row in self.row_ids:
        (ymin,ymax) = (y_steps[row], y_steps[row+1])
        for col in col_ids:
            cell_in_a_cluster = False
            for j, clusters_for_column in js_clusters_for_columns:
                if (row,col) in clusters_for_column['cell_to_cluster']: 
                    c_ids[i] = clusters_for_column['cell_to_cluster'][(row,col)]
                    cell_in_a_cluster = True
            
            if (row,col) in id_to_sums: 
                sums[i] = id_to_sums[(row,col)]
            elif not cell_in_a_cluster:
                continue
            (xmin, xmax) = (x_steps[col], x_steps[col+1])
            centroids_x[i] = centroids_x_full[row*col+col] 
            centroids_y[i] = centroids_y_full[row*col+col]
            polys.append(Polygon(((xmin,ymin),(xmax,ymin),(xmax,ymax),(xmin,ymax))))
            i += 1
        #
    #
    df = _gpd_GeoDataFrame({
        'centroid_x': centroids_x[:i],
        'centroid_y': centroids_y[:i],
        }, geometry=polys,
        crs=self.local_crs
        )
    if len(self.clusters_by_column)<=1:
        df['cluster_id'] = c_ids[:i]
        df['sum'] = sums[:i]
    else:
        for j, column in enumerate(self.clusters_by_column):
            c_id_colname = find_column_name("cluster_id", column, df.columns, 10 if file_format=='shp' else 20)
            agg_colname = find_column_name("sum_radius", column, df.columns, 10 if file_format=='shp' else 20)
            df[c_id_colname] = c_ids[:i,j]
            df[agg_colname] = sums[:i,j]
    df.to_crs(self.initial_crs, inplace=True)
    # save
    filename = filename +'.'+file_format
    if file_format == 'shp':
        df.to_file(filename, driver="ESRI Shapefile", index=False)
    elif file_format == 'csv':
        df.to_csv(filename, index=False)
    else:
        raise ValueError('Unknown file_format:',file_format,'Choose on out of shp, csv')
    #

def save_cell_clusters(
        self,
        filename:str="grid_clusters",
        file_format:str=['shp','csv'][0],
    ):
    for (cluster_column, clusters) in self.clusters_by_column.items():
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
        filename = filename + (("_"+cluster_column) if len(self.clusters_by_column) > 1 else '') + '.'+file_format
        if file_format == 'shp':
            df.to_file(filename, driver="ESRI Shapefile", index=False)
        elif file_format == 'csv':
            df.to_csv(filename, index=False)
        else:
            raise ValueError('Unknown file_format:',file_format,'Choose on out of shp, csv')
    #
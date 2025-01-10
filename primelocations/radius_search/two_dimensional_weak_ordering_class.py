from numpy import (
    array as _np_array, invert as _np_invert, hstack as _np_hstack, zeros as _np_zeros, ones as _np_ones, all as _np_all,  any as _np_any,
)
from ..utils.distances_to_cell import ( min_dist_points_to_cell, max_dist_points_to_cell,)
from ..testing.test_performance import time_func_perf, func_timer_dict

# This script provides methods to sort points according to 
# their distance to an undefined point within a convex polygon
# (in this case this Polygon is a triangle defined by [(0,0),(.5,0),(.5,.5)])

def _visualize(weak_order_branch, indent=0): 
    """
    Print weak ordering tree structure
    """
    for person in weak_order_branch:
        print(" - "*indent + str(person.id)+":"+
              '(oc'+ str(person.check_if_overlapped_nr if hasattr(person, "check_if_overlapped_nr") else None)+","+
              'cc'+str(person.check_if_contained_nr if hasattr(person, "check_if_contained_nr") else None)+')')
        _visualize(person.children, indent+1)

@time_func_perf
def document_ancestry(
          person:int,
          ancestor_to_descandants:dict,
          parent_to_children:dict,
          child_to_parent:dict,
          child_to_parents:dict,
          person_to_family_tree_path:dict,
          family_tree:dict,
):
     """"
     updates dicts 
     finds and returns children defined as 
     subsets of descendants that is not included in any descendants' descendants set
     """

     # add to path
     family_tree_next_generation = {}
     if person in child_to_parents and len(child_to_parents[person])>0:
          grand_parents = child_to_parents[person]
          for grand_parent in grand_parents:
               grand_parent_path = person_to_family_tree_path[grand_parent] 
               grand_parent_path.update({person:family_tree_next_generation})
          person_to_family_tree_path[person] = grand_parent_path[person]
     else:
          family_tree[person]=family_tree_next_generation
          person_to_family_tree_path[person] = family_tree[person]
     
     
     descendants_of_children = set()
     descendants = ancestor_to_descandants[person]
     for descendant in descendants:
          descendants_of_children.update(list(ancestor_to_descandants[descendant]))
     children = [descendant for descendant in descendants if not descendant in descendants_of_children]

     parent_to_children[person]=children
     child_to_parent.update({child: person for child in children})
     for child in children:
          child_to_parents[child].append(person)
     return children
#

@time_func_perf
def create_bilateral_comp_mx_smaller_than(
        sort_vars:_np_array,
):
    """
    Creates and returns a boolean bilateral comparison matrix
    Matrix entry value for element i-th row,j-th col
    - True:
        iff for ALL variables of i-th cell are smaller or equal 
        to value of the j-th cell (indicated by column).
    - False
        otherwise

    Args:
      sort_vars (numpy.array):
        Numeric variables which determining ordering of cells.
        Shape=(n_cell, n_vars)
    Returns:
      bilater_comp_matrix_smaller (numpy.array):
        Boolean comparison if ALL values for i-th cell are weakly 
        smaller than j-th cell. Shape=(n_cells, n_cells)
    """
    # For each cell: 
    # compare the cell minimum distance and maximum distance to 
    # each of the three trianlge points to distance of other cell
    # and check whether the distance is always smaller or equal.

    # If the distance is always smaller or equal it can be said that
    # Each point within the triangle will always be closer to any 
    # point in this cell than  any point within the other cell   
    return _np_array([
        _np_all(cell_ordering_vars <= sort_vars, axis=1) 
        for cell_ordering_vars in sort_vars
    ])
#
# TODO remove unecessary parts
# TODO turn this into a Class asweel

class WeakOrderTreeBranch(object):
    def __init__(
        self,
        id,
        root,
        # ancestor_to_descandants:dict,
        # parent_to_children:dict,
        # child_to_parent:dict,
        # child_to_parents:dict,
        # family_tree:dict
    ):
        """"
        updates dicts 
        finds and returns children defined as 
        subsets of descendants that is not included in any descendants' descendants set
        """

        # initialize dict add to path
        self.id = id
        self.children = []
        self.parent = root

    def add_sub_branch():
        pass
    #

    def add_child(self,child):
        self.children.append(child)
    #

    def add_parent(self,parent):
        self.parent = parent
    #
    def add_attributes_to_branch(
                self,
                cell_to_all_int,
                cell_to_nev_cnt,
                tree,
    ):
        self.is_overlapped_by_all_trgl1_disks = cell_to_all_int[self.id]
        self.is_never_contained_in_trgl1_disks = cell_to_nev_cnt[self.id]
        
        if not self.is_overlapped_by_all_trgl1_disks: 
            self.check_if_overlapped_nr = tree.check_if_overlapped_nrs.pop(0)
            tree.check_if_overlapped_order.append(self.id)
        if not self.is_never_contained_in_trgl1_disks: 
            self.check_if_contained_nr = tree.check_if_contained_nrs.pop(0)
            tree.check_if_contained_order.append(self.id)
        #
        for child in self.children:
            child.add_attributes_to_branch(
                cell_to_all_int = cell_to_all_int,
                cell_to_nev_cnt = cell_to_nev_cnt,
                tree=tree,
            )
    #
    def get_nest_structure_as_list(
            self,
            nested_structure_as_list
    ):
        nested_structure_as_list.append(self.id)
        for child in self.children:
            child.get_nest_structure_as_list(nested_structure_as_list)
#
def member_similarity(
        id1:tuple, id2:tuple
):
    """
    Member
    """
    res = 0
    for i in range(len(id1)):
        res += abs(id1[i]-id2[i])
    return res
#
def choose_parent(child_id, parent_ids):
    # if multiple parents choose most similar one
    chosen_parent = min([(member_similarity(parent_id, child_id), parent_id) for parent_id in parent_ids])[1]
    return chosen_parent
#


class WeakOrderTree(object):
    def __init__(
            self,
            cells:_np_array,
            sort_vars:_np_array,
    ):
        bl_comp_mx_smaller_than = create_bilateral_comp_mx_smaller_than(
            sort_vars=sort_vars
        )

        # convert to list of tuples to facilitate dict creation    
        cells = [tuple([int(row),int(col)]) for row, col in cells]
        # read out weak ordering and store properties from bilateral_comparisons 
        ancestor_to_descandants = {}
        descandant_to_ancestors = {cell:[] for cell in cells}
        for cell_i,check_if_closer_than in zip(cells, bl_comp_mx_smaller_than):
            closer_cells = set([
                cell 
                for cell, closer in zip(cells, check_if_closer_than) 
                if closer and cell_i != cell
            ])
            ancestor_to_descandants[cell_i] = closer_cells
            descandant_to_ancestors.update(
                {f_o_id: descandant_to_ancestors[f_o_id]+[cell_i] for f_o_id in closer_cells}
            )
        #
        # Now 
        
        orphans = cells
        n_ancestors = _np_array([len(descandant_to_ancestors[descandant]) for descandant in orphans])
        # chose those as parents that are not children of any other 
        current_gen = [orphan for orphan, n_a in zip(orphans, n_ancestors) if n_a == 0]
        self.root = []
        
        self.members = members = {orphan: WeakOrderTreeBranch(id=orphan, root=self.root) for orphan in orphans}
        self.root.extend([members[parent] for parent in current_gen])
        
        orphans = [orphan for orphan in orphans if not orphan in current_gen]
        
        while len(orphans) > 0:
            orphans = [orphan for orphan in orphans if not orphan in current_gen]

            # From all remaining family member - get all their descendants
            next_gen_descendants = set()
            for orphan in orphans:
                next_gen_descendants = next_gen_descendants.union(ancestor_to_descandants[orphan])
            
            # Remove all descendants of next generation from next generaration (as they will be part of subsequent generation)
            parent_to_childrens = {}
            for parent in current_gen:
                descandants = ancestor_to_descandants[parent]
                parent_to_childrens[parent] = descandants.difference(next_gen_descendants)
            
            next_gen = set(orphans).difference(next_gen_descendants)
            # handle cases where one child could be of multiple parents and assign single parent only.
            next_gen_child_to_parent = {}
            for child in next_gen:
                child_parents = [parent for parent in current_gen if child in parent_to_childrens[parent]]
                if len(child_parents) == 1:
                    parent = child_parents[0]
                else:
                    parent = choose_parent(child, child_parents)
                
                next_gen_child_to_parent[child] = parent
            #
            
            for child, parent in next_gen_child_to_parent.items():
                members[parent].add_child(members[child])
                members[child].add_parent(members[parent])

            # remove from orphans
            orphans = [orphan for orphan in orphans if not orphan in next_gen]

            current_gen = next_gen
        #
        
            self.nest_structure_as_list = []
            for person in self.root:
                person.get_nest_structure_as_list(self.nest_structure_as_list)
    #
        
    def visualize(self):
        _visualize(self.root)
    #

    @time_func_perf
    def add_attributes_to_tree(
        self, 
        cells,
        trgl_always_overlaps_cells,
        trgl_never_contains_cells,
    ):
        """
        Modifies flat_family_tree
        """
        self.n_checks_if_overlapped = len(trgl_always_overlaps_cells)-sum(trgl_always_overlaps_cells)
        self.n_checks_if_contained = len(trgl_never_contains_cells)-sum(trgl_never_contains_cells)

        self.cells_overlapped_by_all_trgl1_disks = [(row,col) for ((row,col),all_vtx_int) in zip(cells, trgl_always_overlaps_cells) if all_vtx_int]
        self.cells_contained_in_no_trgl1_disk = [(row,col) for ((row,col),no_vtx_cnt) in zip(cells, trgl_never_contains_cells) if no_vtx_cnt]
                
        cell_to_all_int = {(row,col): all_vtx_int for ((row,col), all_vtx_int) in zip(cells, trgl_always_overlaps_cells)}
        cell_to_nev_cnt = {(row,col): no_vtx_cnt for ((row,col), no_vtx_cnt) in zip(cells, trgl_never_contains_cells)}
        
        self.check_if_overlapped_order = []
        self.check_if_contained_order = []
        self.check_if_overlapped_nrs = list(range(self.n_checks_if_overlapped))
        self.check_if_contained_nrs = list(range(self.n_checks_if_contained))
        for child in self.root:
            child.add_attributes_to_branch(
                cell_to_all_int = cell_to_all_int, 
                cell_to_nev_cnt = cell_to_nev_cnt,
                tree=self,
            )
        
        self.check_if_overlapped_order = _np_array(self.check_if_overlapped_order)
        self.check_if_contained_order = _np_array(self.check_if_contained_order)
    #  
#
        # find start point: not implied by another 
        # then work through list until none remain
        # at dead end start with any left overs
        # if multiple values are implied at once 
        # continue with the one value that is implied only by values that have been dealt with
        # otherwise check whether any of those values imply 'uncles'
        # data structure shall be nested tree at the end of which all candidate cells are categorized
        # are there any parallel structures possible?

        # finally it should be check what the best starting point of tree would
        # in the triangle there might be cells that are never overlapped and cells that are alwas included. those cells never have to be checked again.  
        # prepare and store in dict
        # flip(a,1)*[-10,100]

        # Question is it possible that nephews are older than uncles?
        # a->b, a->c, b~?~c, b->d, 

        # function needs to return all overlapped cells and all included cells
        # so its best go from inside to outside

    
    

@time_func_perf
def gen_weak_order_rel_to_convex_set(
        cells:_np_array,
        convex_set_vertices:_np_array = _np_array([[0,0],[.5,0],[.5,.5]]),
        vertex_is_inside_convex_set=True,
        radius:float=0.00750,
        grid_spacing:float=0.00250,
        include_boundary:bool=False,
)->tuple:
    """
    TODO:

    Assume two points A and B outiside of the convex set. If the distance of
     point A to all vertices of the convex set is smaller [larger] than the
     respective distance of point B, then A will be closer [farther] to any
     arbitrray point within the set than B will be. 
     
    Args:
      TODO (TODO):
        TODO
      TODO (TODO):
        TODO
      convex_set_vertices (numpy.array):
        vertices defining a two dimensional convex set
    Returns:
      weak_order_relative_to_convex_set (TODO): 
      - hierarchical tree like structure which contains information
        
    """
    if type(vertex_is_inside_convex_set)==bool:
        if vertex_is_inside_convex_set:
            vertex_is_inside_convex_set = _np_ones(len(convex_set_vertices), dtype=bool)
        else:
            vertex_is_inside_convex_set = _np_zeros(len(convex_set_vertices), dtype=bool)
    
    # smallest possible distance of cell to convex set
    dist_lower_bounds = grid_spacing * _np_array([
        min_dist_points_to_cell(convex_set_vertices,cell) 
        for cell in cells]
    )
    # largest possible distance of cell to convex set
    dist_upper_bounds = grid_spacing * _np_array([
        max_dist_points_to_cell(convex_set_vertices,cell)
        for cell in cells]
    )

    # CREATE WEAK ORDERING FROM HERE
    sort_vars = _np_hstack([dist_lower_bounds, dist_upper_bounds])
    
    weak_order_tree = WeakOrderTree(
        cells=cells,
        sort_vars=sort_vars
    )
    # TODO FIX THIS SECTION

    # store if for some cells all the complete set always or never fullfills a condition
    trgl_always_overlaps_cells = (
        _np_all(dist_lower_bounds <= radius, axis=1)
    ) if include_boundary else (
        _np_all(dist_lower_bounds < radius, axis=1)
    )
    trgl_always_overlaps_cells = _np_all([
            (dist_lower_bounds[:,i] <= radius) if include_boundary and point_in_set else (dist_lower_bounds[:,i] < radius) 
            for i,point_in_set in zip(range(dist_lower_bounds.shape[1]), vertex_is_inside_convex_set)
        ], axis=0)
    
    trgl_maybe_contains_cells = _np_any(dist_upper_bounds <= radius, axis=1)
    trgl_maybe_contains_cells = _np_any([
            (dist_upper_bounds[:,i] <= radius) if include_boundary and point_in_set else (dist_upper_bounds[:,i] < radius) 
            for i,point_in_set in zip(range(dist_upper_bounds.shape[1]), vertex_is_inside_convex_set)
        ], axis=0)

    # trgl_always_overlaps_cells = _np_zeros(len(dist_lower_bounds),bool)
    # trgl_maybe_contains_cells = _np_ones(len(dist_lower_bounds),bool)

    trgl_never_contains_cells = _np_invert(trgl_maybe_contains_cells)
    
    weak_order_tree.add_attributes_to_tree(
            cells=cells,
            trgl_always_overlaps_cells=trgl_always_overlaps_cells,
            trgl_never_contains_cells=trgl_never_contains_cells,
    )
    weak_order_tree.trgl_always_overlaps_cells = trgl_always_overlaps_cells
    weak_order_tree.trgl_never_contains_cells = trgl_never_contains_cells

    return weak_order_tree
#


######################## PERFORM_SERACH #####################
@time_func_perf
def recursive_cell_region_inference(
        grid,
        transformed_offset_xy:_np_array,
        reference_ids:_np_array,
        disks_by_cells_contains:_np_array,
        disks_by_cells_overlaps:_np_array,
        family_tree_pos:WeakOrderTree,
        grid_spacing:float=200,
        radius:float=750,
        include_boundary:bool=False,
):
    """
    Recursively performance checks on nested structure of grid cells. 
    Checks whether closest vertex of cell is within disk of specified radius, i.e. cell is at least partly contained in disk. 
    For all points for which this hold true, it checks whether furthest vertex is also in disk, i.e. cell is fully contained in disk.
    Result are stored in a 2D-matrix (number of points x number of cell to be checked) for whether cell in fully contained in disk [disks_by_cells_contains]. 
    If cell is NOT fully contained but partly respective values in partly overlap 2D-matrix are set to true [disks_by_cells_overlaps].
    The right column of both matrix is determined by first element fo remaining check_nrs.  

    For the subset of pts of which their disk have at least partly contained the checked cell the function will be called recursively to test the next cell in the hierarchy.   
    The family_tree containes the hierarchy of cells. The hierarchy is done s.t. if cell is not at least party included in hierarchy, 
    neither will be any of the cells that are ranked strictly lower (i.e. nested below it in dictionary family_tree). 
    Each cell occurs only once in the family_tree. 
    Each cell in the first hierarchy level of the family_tree is within disk of any pts satifying P(px,py), px>=py, px,py in [0,grid_spacing/2]

    
    """
    for person in family_tree_pos:
        # print("person",person.id)
        # if tuple([person.id[0],person.id[1]])==(3,3):
        #     print(person.id,"CELL 3,3: ",person.is_overlapped_by_all_trgl1_disks, person.is_never_contained_in_trgl1_disks, len(reference_ids))
        # indicates column where result need to be stored into 
        # in the top level of tree all points are at least partly contained by definition of the tree. thus this step will be skipped.
        if not person.is_overlapped_by_all_trgl1_disks: 
            # check whether closest vertex is contained in search disk
            check_overlap = (
                grid_spacing * min_dist_points_to_cell(
                    pts_xy=transformed_offset_xy, 
                    cell=_np_array(person.id)
                ) <= radius
            ) if include_boundary else (
                grid_spacing * min_dist_points_to_cell(
                    pts_xy=transformed_offset_xy, 
                    cell=_np_array(person.id)
                ) < radius
            )
            grid.search.overlap_checks.append((person.id, len(reference_ids), sum(check_overlap)))
            # if tuple([person.id[0],person.id[1]])==(3,3):
            #     print("CELL 3,3: ",len(check_overlap), sum(check_overlap))
            # discard all points that are not overlap for the following nest levels
            reference_ids=reference_ids[check_overlap]
            transformed_offset_xy=transformed_offset_xy[check_overlap,:]
        #
        
        if not person.is_never_contained_in_trgl1_disks: # potential speed looking up in full family_tree (where children may have >1 parents). If no children then cell will never be fully contains
            # check whether farthest vertex of cell is in disk. if true cell is fully contained in disk.
            check_contains = grid_spacing * max_dist_points_to_cell(
                pts_xy=transformed_offset_xy, 
                cell=_np_array(person.id)
            ) <= radius
            grid.search.contain_checks.append((person.id, len(reference_ids), sum(check_contains)))
            
            # store result: set value to true in contains regions _np_array
            disks_by_cells_contains[reference_ids[check_contains], person.check_if_contained_nr] = True
            # mark 
            if not person.is_overlapped_by_all_trgl1_disks:
                disks_by_cells_overlaps[reference_ids[_np_invert(check_contains)], person.check_if_overlapped_nr] = True
        else:
            if not person.is_overlapped_by_all_trgl1_disks:
                disks_by_cells_overlaps[reference_ids, person.check_if_overlapped_nr] = True
        #   

        # call function children on if check_contains or check_overlap = True  
        recursive_cell_region_inference(
            grid=grid,
            transformed_offset_xy=transformed_offset_xy,
            reference_ids=reference_ids,
            disks_by_cells_contains=disks_by_cells_contains,
            disks_by_cells_overlaps=disks_by_cells_overlaps,
            family_tree_pos=person.children,
            grid_spacing=grid_spacing,
            radius=radius,
            include_boundary=include_boundary,
        )
        #
    #
    return
#
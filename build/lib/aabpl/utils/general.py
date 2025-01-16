from datetime import datetime
from numpy import (
    array as _np_array,
    sign as _np_sign,
)
from pandas import DataFrame as _pd_DataFrame
from math import (
    atan2 as _math_atan2,
    pi as _math_pi)
from math import degrees, atan2

def angle_to(p1, p2, rotation=0, clockwise=False):
    """
    rotation clockwise
    """
    angle = degrees(atan2(p2[1] - p1[1], p2[0] - p1[0])) - rotation
    if clockwise:
        angle = -angle
    return angle % 360
#

def angles_to_origin(pts,center):
    """
    pts: list of 2D coordinates
    2D center coordiates 
    """
    rotation = angle_to((0,0), center)
    return [angle_to(center, pt, rotation=rotation) for pt in pts]
#

def angle(x1, y1, x2, y2):
    """
    TODO
    """
    return _math_atan2(-y2+y1, x2-x1) % (2*_math_pi)
#

def pt_is_left_of_vector(ptx:float, pty:float, startx:float, starty:float, endx:float, endy:float) -> bool:
    """
    returns True if point is on the leftside of vector. False if to the right or on the vector
    """
    if (startx - endx)*(pty- endy) - (starty - endy)*(ptx - endx)  == 0:
        raise ValueError('Point is on vector!',locals())
    return (startx - endx)*(pty- endy) - (starty - endy)*(ptx - endx) < 0
#


def flatten_list(nestedList:list)->list:
    """
    Flatten list of lists once: [[1,2,],[3,[4,5]]] -> [1,2,3,[4,5]]
    """
    return [item for sublist in nestedList for item in sublist] 
#


def make_bins_from_vals(vals:list) -> list:
    """
    returns list of tuples of upper and lower bounds for bins crated from sorted set of vals
    """
    vals = sorted(set(vals))
    return [(lower,upper) for lower,upper in zip(vals[:-1], vals[1:])]

def get_vals_from_bins(bins:list) -> list:
    """
    return sorted unique vals retrieved from bins upper and lower bounds
    """
    return sorted(set([v[0] for v in bins] + [v[1] for v in bins])) 


def sort_2D_array_by_rows(
        arr:_np_array
):
    for i in range(arr.shape[1])[::-1]:
        arr=arr[arr[:,i].argsort()]
    return arr
#

# copied helpefer funs 
def depth(d):
    if isinstance(d, dict):
        return 1 + (max(map(depth, d.values())) if d else 0)
    return 0
#

def visualize(root, indent=0):
    if type(root) == dict:
        for k, v in root.items():
            print(" - "*indent + f"{k}:")
            visualize(v, indent+1)
    else:
        print(" - "*indent + repr(root))
#

def visualize(root:dict, indent:int,single_nest:str):
    for i,(k, v) in enumerate(root.items()):
        k_s = ' '*(2-len(str(k)))+str(k)
        if i==0:
            if len(v.items())>0:
                visualize(v, indent+0, single_nest + k_s+' --> ')
            else:
                print('    '*indent+single_nest + k_s+'')
        else:
            print(''.join([('    ' if j<indent-1 else '|--> ') for j in range(indent)])+ k_s+'')
            visualize(v, indent+1,'')
    #
#

def list_dict_keys(d:dict):
    res = []#list(d.keys())
    for key,val in d.items():
       res += [key]+list_dict_keys(val) 
    return res
#

def arr_to_tpls(arr:_np_array,tgt_type:type=None):
    """
    transform 2d numpy array to list of tuples and convert to type if specified
    """
    if tgt_type is None:
        return list([tuple(el for el in row) for row in arr])
    return list([tuple(tgt_type(el) for el in row) for row in arr])
#

class DataFrameRelation(object):
    """
    
    """
    EQUAL = 'equal'
    SUBSET = 'subset'
    SUPERSET = 'superset'
    OVERLAP = 'overlap'
    DISJOINT = 'disjoint'
    
    @staticmethod
    def get_bilateral_relation_type(
        a:_pd_DataFrame,b:_pd_DataFrame, force_same_columns:bool=False, silent:bool=False
        )->str:
        """
        returns relations of `a` towards `b` for shared columns.
        Args:
        a : pandas.DataFrame
        any DataFrame to check its relation towards `b` 
        b : pandas.DataFrame
        any DataFrame to check relation of `a` towards `b`
        force_same_columns : bool
        if True will raise an error if DataFrames don't share all columns
        silent : bool
        if True will surpress warning if DataFrames don't share all columns
        
        Returns:
        relation : str
        - 'equal': `a` equals `b`
        - 'subset': `a` is a subset of `b`
        - 'superset': `a` is a superset of `b`
        - 'overlap': `a` and `b` have intersection but neither is a subset of the other
        - 'disjoint': `a` and `b` have no intersection
        """
        if type(a) != _pd_DataFrame or type(b) != _pd_DataFrame:
            raise TypeError("search source and target both have be of type pandas.DataFrame. Types supplied:",type(a),type(b))
        if a is b: return DataFrameRelation.EQUAL
        if a.equals(b): return DataFrameRelation.EQUAL
        
        a_columns = set(a.columns)
        b_columns = set(b.columns)
        if (force_same_columns or not silent) and len(a_columns.difference(b_columns)) > 0:
            msg = ("Checking relation of two DataFrames with columns that are not shared: " + 
                str(a_columns.difference(b_columns))+". Shared columns:"+str(a_columns.intersection(b_columns))
                )
            if force_same_columns:
                raise ValueError(msg)
            elif not silent:
                print(msg)
        
        # drop duplicates to check if intersection includes all
        a_no_dup = a.drop_duplicates()
        b_no_dup = b.drop_duplicates()
        len_df_intersection = len(a_no_dup.merge(b_no_dup, how='inner'))
        
        if len_df_intersection == 0: return DataFrameRelation.DISJOINT
        if len_df_intersection < len(a_no_dup) and len_df_intersection < len(b_no_dup): return DataFrameRelation.OVERLAP
        if len_df_intersection < len(a_no_dup): return DataFrameRelation.SUPERSET
        if len_df_intersection < len(b_no_dup): return DataFrameRelation.SUBSET 
        return DataFrameRelation.EQUAL
    #

    @staticmethod
    def check_if_df_is_contained(
            a:_pd_DataFrame,b:_pd_DataFrame, force_same_columns:bool=False, silent:bool=False
    ) -> bool:
        """
        returns whether `a` is equal to `b` or a subset of `b`for shared columns.
        Args:
        a : pandas.DataFrame
        any DataFrame to check its relation towards `b` 
        b : pandas.DataFrame
        any DataFrame to check relation of `a` towards `b`
        force_same_columns : bool
        if True will raise an error if DataFrames don't share all columns
        silent : bool
        if True will surpress warning if DataFrames don't share all columns
        
        Returns:
        is_contained : bool
          True if a equals b or a is subset of b. False otherwise
        """
        is_contained = DataFrameRelation.get_bilateral_relation_type(
            a=a,b=b,force_same_columns=force_same_columns,silent=silent
            ) in (DataFrameRelation.EQUAL, DataFrameRelation.SUBSET)
        return is_contained
    #
#

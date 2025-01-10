# intersection of two circles with same radius
from numpy import (linspace as _np_linspace, array as _np_array, sign as _np_sign)
from math import sin as _math_sin, cos as _math_cos, atan2 as _math_atan2, pi as _math_pi, acos as _math_acos , sin as _math_asin, log10 as _math_log10
from matplotlib.patches import (Rectangle as _plt_Rectangle, Polygon as _plt_Polygon, Circle as _plt_Circle)

################ transform_cell_pattern ######################################################################################
def transform_cell_pattern(
    cells:_np_array,
    i:int,
)->_np_array:
    """
    Transform a pattern of cells (row_nr, col_nr) around origin cell (0,0) to match a 
    transformation of point from Triangle 1 into any other Triangle 1-8.

    Transformation from Traingle 1 to Triangle X: cell(row, col): explanation
    - 1: ( r, c): keep unchanged
    - 2: ( c, r): mirror along +45deg line (y=x)
    - 3: ( c,-r): rotate by 90° counter clockwise
    - 4: ( r,-c): mirror along +45deg line (y=x). rotate by 90° counter clockwise
    - 5: (-r,-c): rotate by 180° counter clockwise
    - 6: (-c,-r): mirror along +45deg line (y=x). rotate by 180° counter clockwise
    - 7: (-c, r): rotate by 270° counter clockwise
    - 8: (-r, c): mirror along +45deg line (y=x). rotate by 270° counter clockwise

    Args:
        cells (numpy.array):
        Array of cells that will be rotated. each cell has (row_nr, col_nr). shape=(n,2)
        i (int):
        Tranform a cell pattern relative to Triangle 1 such that is has the same properties relative
        to Triangle i   
    Returns:
        rotated_cells (numpy.array):
        cell rotated from Triangle to Triangle [i].
    """
    if i not in [1,2,3,4,5,6,7,8]:
        raise ValueError("Triangle number must be an integer between 1 and 8")
    if type(cells) != _np_array:
        cells = _np_array(cells)
    else:
        cells = cells.copy()

    if i == 1: return cells[:,:] # no change *array([1,1]) # Triangle 1
    if i == 2: return cells[:,::-1]#* _np_array([ 1, 1])   # Triangle 2
    if i == 3: return cells[:,::-1] * _np_array([ 1,-1])   # Triangle 3
    if i == 4: return cells[:,:]    * _np_array([ 1,-1])   # Triangle 4
    if i == 5: return cells[:,:]    * _np_array([-1,-1])   # Triangle 5
    if i == 6: return cells[:,::-1] * _np_array([-1,-1])   # Triangle 6
    if i == 7: return cells[:,::-1] * _np_array([-1, 1])   # Triangle 7
    if i == 8: return cells[:,:]    * _np_array([-1, 1])   # Triangle 8
#

def transform_cell(
    cell:_np_array,
    i:int,
)->_np_array:
    """
    Transform a pattern of cells (row_nr, col_nr) around origin cell (0,0) to match a 
    transformation of point from Triangle 1 into any other Triangle 1-8.

    Transformation from Traingle 1 to Triangle X: cell(row, col): explanation
    - 1: ( r, c): keep unchanged
    - 2: ( c, r): mirror along +45deg line (y=x)
    - 3: ( c,-r): rotate by 90° counter clockwise
    - 4: ( r,-c): mirror along +45deg line (y=x). rotate by 90° counter clockwise
    - 5: (-r,-c): rotate by 180° counter clockwise
    - 6: (-c,-r): mirror along +45deg line (y=x). rotate by 180° counter clockwise
    - 7: (-c, r): rotate by 270° counter clockwise
    - 8: (-r, c): mirror along +45deg line (y=x). rotate by 270° counter clockwise

    Args:
        cells (numpy.array):
        Array of cells that will be rotated. each cell has (row_nr, col_nr). shape=(n,2)
        i (int):
        Tranform a cell pattern relative to Triangle 1 such that is has the same properties relative
        to Triangle i   
    Returns:
        rotated_cells (numpy.array):
        cell rotated from Triangle to Triangle [i].
    """
    if i not in [1,2,3,4,5,6,7,8]:
        raise ValueError("Triangle number must be an integer between 1 and 8")
    cell = cell.copy()

    if i == 1: return cell # no change *array([1,1]) # Triangle 1
    if i == 2: return cell[::-1]#* _np_array([ 1, 1])   # Triangle 2
    if i == 3: return cell[::-1] * _np_array([ 1,-1])   # Triangle 3
    if i == 4: return cell[:]    * _np_array([ 1,-1])   # Triangle 4
    if i == 5: return cell[:]    * _np_array([-1,-1])   # Triangle 5
    if i == 6: return cell[::-1] * _np_array([-1,-1])   # Triangle 6
    if i == 7: return cell[::-1] * _np_array([-1, 1])   # Triangle 7
    if i == 8: return cell[:]    * _np_array([-1, 1])   # Triangle 8
#


def transform_coord(
    coord:_np_array,
    i:int,
)->_np_array:
    """
    Transform a pattern of cells (row_nr, col_nr) around origin cell (0,0) to match a 
    transformation of point from Triangle 1 into any other Triangle 1-8.

    Transformation from Traingle 1 to Triangle X: cell(row, col): explanation
    - 1: ( x, y): keep unchanged
    - 2: ( y, x): mirror along +45deg line (y=x)
    - 3: (-y, x): rotate by 90° counter clockwise
    - 4: (-x, y): mirror along +45deg line (y=x). rotate by 90° counter clockwise
    - 5: (-x,-y): rotate by 180° counter clockwise
    - 6: (-y,-x): mirror along +45deg line (y=x). rotate by 180° counter clockwise
    - 7: ( y,-x): rotate by 270° counter clockwise
    - 8: ( x,-y): mirror along +45deg line (y=x). rotate by 270° counter clockwise

    Args:
        cells (numpy.array):
        Array of cells that will be rotated. each cell has (row_nr, col_nr). shape=(n,2)
        i (int):
        Tranform a cell pattern relative to Triangle 1 such that is has the same properties relative
        to Triangle i   
    Returns:
        rotated_cells (numpy.array):
        cell rotated from Triangle to Triangle [i].
    """
    if i not in [1,2,3,4,5,6,7,8]:
        raise ValueError("Triangle number must be an integer between 1 and 8")
    if type(coord) != _np_array:
        coord = _np_array(coord)
    else:
        coord = coord.copy()

    if i == 1: return coord # no change *array([1,1]) # Triangle 1
    if i == 2: return coord[::-1]#* _np_array([ 1, 1])   # Triangle 2
    if i == 3: return coord[::-1] * _np_array([-1, 1])   # Triangle 3
    if i == 4: return coord[:]    * _np_array([-1, 1])   # Triangle 4
    if i == 5: return coord[:]    * _np_array([-1,-1])   # Triangle 5
    if i == 6: return coord[::-1] * _np_array([-1,-1])   # Triangle 6
    if i == 7: return coord[::-1] * _np_array([ 1,-1])   # Triangle 7
    if i == 8: return coord[:]    * _np_array([ 1,-1])   # Triangle 8
#
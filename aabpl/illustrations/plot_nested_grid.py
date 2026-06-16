from numpy import (unique as _np_unique, ones as _np_ones,)
from matplotlib.pyplot import (get_cmap as _plt_get_cmap, subplots as _plt_subplots)
from matplotlib.patches import Rectangle as _plt_Rectangle

def illustrate_nested_grid(
            grid:dict,
            pts_lat_lon,
            cell_ids,
            y_min:float=None,
            y_max:float=None,
            x_min:float=None,
            x_max:float=None,
):

    """
    Illustrate method
    """
    # unpack vals
    grid_spacing = grid.spacing
    nested_grid = grid['nested']

    # filter pts to be inside bounds:
    filter_mask = _np_ones(len(pts_lat_lon),dtype=bool) 
    if y_min != None:
        filter_mask = filter_mask * (pts_lat_lon[:,0]>=y_min)
    if y_max != None:
        filter_mask = filter_mask * (pts_lat_lon[:,0]<=y_max)
    if x_min != None:
        filter_mask = filter_mask * (pts_lat_lon[:,1]>=x_min)
    if x_max != None:
        filter_mask = filter_mask * (pts_lat_lon[:,1]<=x_max)
    
    pts_lat_lon = pts_lat_lon[filter_mask] 
    cell_ids = cell_ids[filter_mask]
    print('N Pts:',len(cell_ids), 'in', len(_np_unique(cell_ids)),'cells.')

    def unpackNestedGrid(cell_dict:dict, nest_lvl:int=0):
        """
        TODO Check to move it outside as independent functions
        """
        res = []
        if 'quadrants' in cell_dict:
            for quadrant in cell_dict['quadrants']:
                res += unpackNestedGrid(cell_dict=quadrant,nest_lvl=nest_lvl+1) 
        res += [(nest_lvl, tuple([*tuple([*cell_dict['bounds']])]))]
        return res
    
    flat_nested_grid = []
    for cell_id in _np_unique(cell_ids):
        flat_nested_grid += unpackNestedGrid(nested_grid[cell_id],nest_lvl=0)

    flat_nested_grid = sorted(flat_nested_grid)
    max_nest_level = flat_nested_grid[-1][0]
    print('max_nest_level',max_nest_level)
    cmp = _plt_get_cmap("YlOrBr",max_nest_level)
    # print(flat_nested_grid)
    nested_patches = [_plt_Rectangle(
                # x         y           width           height
                (x_min, y_min), x_max-x_min, y_max-y_min, 
                linewidth=.7, facecolor=(cmp(nest_lvl) if max_nest_level>0 else 'grey'), edgecolor='#444', alpha=1
                ) for nest_lvl, ((y_min, x_min), (y_max, x_max) ) in flat_nested_grid]
    
        
    fig, ax = _plt_subplots(figsize=(15,10))

    for patchToAdd in nested_patches:
        ax.add_patch(patchToAdd)

    ax.scatter(x=pts_lat_lon[:,1], y=pts_lat_lon[:,0], s=0.2,color='black')
    x_steps = grid.x_steps
    y_steps = grid.y_steps
    ax.set_xticks(x_steps)
    ax.set_yticks(y_steps)
    ax.set_xlim(left=x_min, right=x_max)
    ax.set_ylim(bottom=y_min, top=y_max)
    ax.set_aspect('equal', adjustable='box')
    ax.grid()
#
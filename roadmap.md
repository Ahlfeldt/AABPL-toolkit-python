# documentation:
[ ] emphasize that its important to select local crs to minimize projection error and that thus the methods are less precise the larger the area gets (good on city level, less accurat at country leveel)

# notes on final form:
## step one: determine optimal grid size
[ ] based on bounding_box, n_points, (maybe distribution / centrality measure of points), radius
[ ] filter list of row col indexes s.t. they are within [xmin=0,xmax=ncols] and [ymin=0,ymax=nrows]

## we also want have a function that gives us the share of intersected area 
[ ] for expected value of relevant cells 
[ ] for undevelopable cell adjustment 

# TODO:
[ ] add polygon 2 patches (as on sketch)
[ ] validate inputs

## add plot of center cell with areas colored in 
[ ] number of interesected cells 
[ ] number of overlapped cells 

[ ] to do double check those distance checks for cells. 

[ ] add excluded property to grid cells
[ ] think about radius_search class
[ ] make point Triangle 1 not contain points x=y except for 0,0 in recursive checks

[ ] replace recursive cell region inference with linear segmentation
[ ] either follow up with more granular linear segmentation
[ ] or rather do distance check to either one or two points another cell

## Performance
[ ] use decorator function for performance testing
[ ] create super cells 3x3 etc 2x1 or whatever is optimal to minimize cell dict lookups when using smaller gridsize
[ ] make linebased comparisons insted of bilateral pt comps

Maybe lower precision by:
- if the grid is fine enough you could think about not computing bilateral point distances but instead use the share of overlap of the cell to weight the cells sum
- convert circles to line segments


BUG HUNTING
[x] not introduced from early returns is_always_overlapped and never_contained. BUT maybe something additionally unexpected
[ ] scaling with grid_spacing
[ ] mixed up rows/cols, x/y (lon/lat) somewhere 
    [x] min_possible_dist_cells_to_cell
    [x] max_possible_dist_cells_to_cell
    [x] min_possible_dist_trgl1_to_cell
    [x] max_possible_dist_trgl1_to_cell
    [ ] min_dist_points_to_cell
    [ ] max_dist_points_to_cell
[ ] 
[ ] 

- potentially define area boundaries for each region and get the areas to then weight the nuber of intersected grid cells by the area. to better determine opt grid size.
- also pot be smarter on empty cells so that summing over them doesnt take as much space
- think if the pattern can applied on vector of points to repplace ppart of the loop and make filter out empty grid ells easier
- maybe cell intersection function takes to much space 

[ ] aggreagate_point_data_to_disks_vectorized removed vectorized from name
[ ] shorten long functions
[ ] add silent option to skip prints
[ ] add set random seed
[ ] ensure x,y lon,lat row,col is always ordered correctly! Convention: use (x,y) instead of (y,x)
- np.newaxis to shift patterns 

- bundle logic in single place
    - determine whether cells are included/intersected/outside
    - which cell vertices are closest / farthest and need to be checked

- Create comments, rename variables and functions

- think about classes -> makes sense for grid
    - does it make sense to have additional Class for 
        - SparseGrid
        - NestedGrid
            - Idea: it could make the algorithm more flexible to have different methods for the disksearch defined under the classes, s.t. that the grid could be a mix of nested and unnested grid cells. 
        - GridCells



# DONE:
## for optimal gridsize obtain list of relative positon of  
[x] always fully contained cells
[x] (potentially) intersected cels

[x] create function (based on thresholds?) that uses relative position of point within grid cell:
[x] and further spereates potentialy intersected cells into 
    [x] actually fully contained cells
    [x] actually not intersected cells
    [x] actually interesected cells





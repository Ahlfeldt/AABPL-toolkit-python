from numpy import (
    array as _np_array, 
    linspace as _np_linspace,
    stack as _np_stack,
    arange, unique, invert, flip, transpose, concatenate, sign, zeros, 
    min, max, equal, where, logical_or, logical_and, all, newaxis
)
# from numpy.linalg import norm
# from numpy.random import randint, random
from pandas import DataFrame as _pd_DataFrame
from math import pi#,ceil,asin,acos
# from primelocations.utils.general import ( flatten_list, visualize, depth)
from aabpl.illustrations.illustrate_optimal_grid_spacing import ( create_optimal_grid_spacing_gif, )
from aabpl.utils.distances_to_cell import ( get_always_contained_potentially_overlapped_cells,)
# from primelocations.valid_area import disk_cell_intersection_area


def get_next_relevant_grid_spacing(
        relevantGridSizes:_pd_DataFrame,
        # allGridSizes:_pd_DataFrame,
        # current:float=750,
        upper_bound:float=400,
        lower_bound:float=80,
        precision:float=0.1,
        radius:float=750,
        smallest_contain_count:int=0, 
        smallest_overlap_count:int=0
)->tuple:
    """
    
    """
    if len(relevantGridSizes) == 0:
        return upper_bound
    
    it = 0
    current = (upper_bound * 9 + lower_bound*1) / 10
    changeDetected = False
    while upper_bound - lower_bound > precision:
        it += 1
        early_it = int(it<20)+int(it<12)+int(it<6)+int(it<3)
        current = (upper_bound * (5+early_it) + lower_bound*(5-early_it)) / 10
        
        
        contain_count, overlap_count = get_always_contained_potentially_overlapped_cells(grid_spacing=current, radius=radius, countOnly=True)
        #
        changeDetected = (
            relevantGridSizes.iloc[-1]['contain_count'] != contain_count or
            relevantGridSizes.iloc[-1]['overlap_count'] != overlap_count
            )
        
        if changeDetected:
            lower_bound = current
        elif contain_count==smallest_contain_count and smallest_overlap_count==overlap_count:
            # 
            return lower_bound
        else:
            upper_bound = current
        #
    # the current differs in overlapping / intersecting cells return current
    if changeDetected:
        return current
    # else return lower_bound which is <precison close to current
    return lower_bound
#

def find_relevant_grid_spacings(
    largest:float=400,
    smallest:float=80,
    precision:float=.1,
    radius:float=750,
    plot_opt_spacing:dict={},     
    GIF:bool=True,     
)->_pd_DataFrame:
    """
    Function that iteratively decreases from 'largest' grid_spacing to 'smallest' in steps of 'stepsize'
    Whenever decreasing the function
    """
    relevantGridSizes = _pd_DataFrame(
        columns=[
            'grid_spacing', 
            'contain_count', 
            'overlap_count',
            'cell_steps_max', 
            'contain_ids', 
            'overlap_ids'
            ]
        )

    current = largest
    print('Start searching for relevant gridsizes.')
    
    smallest_contain_count, smallest_overlap_count = get_always_contained_potentially_overlapped_cells(
        grid_spacing=smallest, radius=radius, countOnly=True)

    while current > smallest:
        current = get_next_relevant_grid_spacing(
            relevantGridSizes=relevantGridSizes,
            upper_bound = current,
            lower_bound = smallest,
            precision=precision,
            radius=radius,
            smallest_contain_count=smallest_contain_count, 
            smallest_overlap_count=smallest_overlap_count
        )

        contain_count, overlap_count, contain_ids, overlap_ids, cell_steps_max = get_always_contained_potentially_overlapped_cells(
            grid_spacing=current, radius=radius, countOnly=False)

        relevantGridSizes.loc[current]= {
                    'grid_spacing': current,
                    'contain_count':contain_count,
                    'overlap_count':overlap_count,
                    'cell_steps_max':cell_steps_max, 
                    'contain_ids':contain_ids, 
                    'overlap_ids':overlap_ids
                    }      

    #

    relevantGridSizes['share_contain'] = (relevantGridSizes['contain_count']*relevantGridSizes['grid_spacing']**2)/(pi*radius**2)
    relevantGridSizes['share_overlap'] = (relevantGridSizes['overlap_count']*relevantGridSizes['grid_spacing']**2)/(pi*radius**2)
    
    print('Found '+str(len(relevantGridSizes))+' relevant Gridsizes.')

    return relevantGridSizes
#

def fill_in_all_grid_spacing_steps(
        relevantGridSizes:_pd_DataFrame,
        largest:float=400,
        smallest:float=80,
        stepsize:float=.1,
        radius:float=750
    ) -> _pd_DataFrame:
    """
    TODO Currently not used
    Takes relevantGridSizes and fills in all missing steps (subject to stepsizes) and return new extended dataframe
    """
    # dec_places = max_decimal_places([smallest, largest, stepsize])
    # allGridSizesVec = [i for i in arange(smallest, largest+stepsize, stepsize)[::-1]]
    allGridSizesVec = [item for sublist in 
                       [[i0]+list(_np_linspace(i0,i1,int((i0-i1)/stepsize)+1)[1:-1]) for (i0,i1) in zip(
                           relevantGridSizes.index[:-1], relevantGridSizes.index[1:]
                           )]
                       for item in sublist]  + [smallest]
    allGridSizes = relevantGridSizes.iloc[:0].copy()
    relevantGridSizesVec_temp = relevantGridSizes.index #['grid_spacing']

    for i in allGridSizesVec:
        selected_ids = relevantGridSizes[relevantGridSizesVec_temp>=i]
        allGridSizes.loc[i] = selected_ids.iloc[-1]
        if i != selected_ids['grid_spacing'].iloc[-1]:
            allGridSizes.loc[i,'share_contain'] = (allGridSizes.loc[i, 'contain_count']*i**2)/(pi*radius**2)
            allGridSizes.loc[i,'share_overlap'] = (allGridSizes.loc[i, 'overlap_count']*i**2)/(pi*radius**2)
        allGridSizes.loc[i, 'grid_spacing'] = i

    allGridSizes = allGridSizes.astype(relevantGridSizes.dtypes.to_dict()) # ensure correct data types
    print('Extended to allGridSizes returning DataFrame for '+str(len(allGridSizes))+' Gridsizes.')  

    return allGridSizes      
#

def wrap_optimal_grid_fun(
    largest:float=400,
    smallest:float=80,
    stepsize:float=.5,
    precision:float=.01,
    radius:float=750,
    plot_opt_spacing:dict={},     
    GIF:bool=True,
    frames_between_relevant:int=6,
    FuncAnimation_interval:int=500,
    FuncAnimation_repeat:bool=True,
    FuncAnimation_repeat_delay:int=1500,
    FuncAnimation_cache_frame_data:bool=True,
    dpi:int=50,
    PillowWriterFps:int=10,
    output_dir:str='./opt_grid_'     
)->_pd_DataFrame:
    """
    Wrapper
    """

    relevantGridSizes = find_relevant_grid_spacings(
        largest=largest,
        smallest=smallest,
        precision=precision,
        radius=radius
    )
    
    allGridSizes = fill_in_all_grid_spacing_steps(
        relevantGridSizes=relevantGridSizes,
        largest=largest,
        smallest=smallest,
        stepsize=stepsize,
        radius=radius,
    )
        
    if GIF:
        create_optimal_grid_spacing_gif(
            relevantGridSizes=relevantGridSizes,
            allGridSizes=allGridSizes,
            largest=largest,
            smallest=smallest,
            radius=radius,
            frames_between_relevant=frames_between_relevant,
            FuncAnimation_interval=FuncAnimation_interval,
            FuncAnimation_repeat=FuncAnimation_repeat,
            FuncAnimation_repeat_delay=FuncAnimation_repeat_delay,
            FuncAnimation_cache_frame_data=FuncAnimation_cache_frame_data,
            dpi=dpi,
            PillowWriterFps=PillowWriterFps,
            output_dir=output_dir
        ) 
    
    return (relevantGridSizes, allGridSizes)
#

def select_optimal_grid_spacing(
        x_min:float=0.,
        x_max:float=11*100.,
        y_min:float=0.,
        y_max:float=2*100.,
        radius:float=750,
        n_points:int=1e6,
        n_queries:int=1,
)->float:
    """
    
    """
    grid_spacing = 1.
    print('Selected optimal grid_spacing of: '+str(grid_spacing)+'.')
    relevantGridSizes, allGridsizes = wrap_optimal_grid_fun(
        largest=350,
        smallest=60,
        stepsize=.2,
        precision=1e-13,
        radius=radius,
        plot_opt_spacing={},     
        GIF=False,
        frames_between_relevant=20,
        FuncAnimation_interval=100,
        FuncAnimation_repeat=True,
        FuncAnimation_repeat_delay=0,
        FuncAnimation_cache_frame_data=False,
        dpi=100,
        PillowWriterFps=8,
        output_dir='./optimal_grid/plots/opt_grid_'     
        )
    
    relevantGridSizes['comp_time_est'] = relevantGridSizes['grid_spacing']


    return grid_spacing
#




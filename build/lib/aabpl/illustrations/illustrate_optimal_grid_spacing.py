from pandas import DataFrame as _pd_DataFrame
from matplotlib import pyplot as plt
from matplotlib.animation import (FuncAnimation as _plt_FuncAnimation, PillowWriter as _plt_PillowWriter)
from matplotlib.figure import Figure as _plt_Figure
from matplotlib.axes._axes import Axes as _plt_Axes
from .plot_utils import (
    create_grid_cell_patches, 
    create_circle_patches
)

## Animation / GIF for varying grid spacings

def set_animation_frames( 
    relevantGridSizes:_pd_DataFrame,
    allGridSizes:_pd_DataFrame,
    largest:float=400,
    frames_between_relevant:int = 6
    )->list:
    """
    Returns vector of gridsizes -> each grid_spacing will be an iteration frame
    """
    # define GIF steps
    animation_frames = [largest]
    allGridSizes_indexes = list(allGridSizes.index)

    for gridsize_0,gridsize_1 in zip(relevantGridSizes.index[:-1], relevantGridSizes.index[1:]):
        
        pos_0, pos_1 = allGridSizes_indexes.index(gridsize_0), allGridSizes_indexes.index(gridsize_1)
        
        #
        if (pos_1-pos_0) > frames_between_relevant:
            # number of position that should moved along per allGridSizes_indexes frame 
            stepsize_factor = (pos_1-pos_0)//frames_between_relevant
            # move an aditional positon for the n<stepsize_adjust frames
            stepsize_adjust = (pos_1-pos_0)%frames_between_relevant
            
            animation_frames += [
                allGridSizes_indexes[pos_0+s*stepsize_factor+int(stepsize_adjust<s)] 
                for s in range(frames_between_relevant)
                ]
        else:
            #
            animation_frames += allGridSizes_indexes[pos_0+1:pos_1+1]
            
    print('Animate GIF with '+str(len(animation_frames))+' frames.')

    return animation_frames


def add_coverage_plot(
        fig:_plt_Figure,
        allGridSizes:_pd_DataFrame,
        ymax:float=3,
        yticks:list=[.5,1]
    ) -> _plt_Axes:
    """
    adds new axis to fig and create a stackplot on it that shows:
     - percent of area within grid cells that are completly within
     - stacked percent of area within grid cells that are potetially intersected
     - 
    Returns: axes 
    """
    ax_coverage = fig.add_axes([0.6, 0.63, 0.3, 0.25])
        
    ax_coverage.stackplot(
        allGridSizes['grid_spacing'],allGridSizes['share_contain']*100,allGridSizes['share_overlap']*100,
        colors=['green', 'red']
        )
    ax_coverage.plot(allGridSizes['grid_spacing'], allGridSizes['share_overlap']*100, color='#000')
    min_x, max_x = min(allGridSizes['grid_spacing']),max(allGridSizes['grid_spacing'])
    ax_coverage.set_xlim(min_x,max_x)
    ax_coverage.set_ylim(0,100*ymax)
    
    ax_coverage.set_yticks([100*x for x in sorted(yticks+[round(ymax)])])
    
    # add hline at 100%
    ax_coverage.hlines(y=100,xmin=min_x,xmax=max_x,linewidth=2,color='#000')
    
    return ax_coverage
#

def add_cell_counter_plot(
        fig:_plt_Figure,
        allGridSizes:_pd_DataFrame,
        ymax:float=3,
        yticks:list=[.5,1]
    ) -> _plt_Axes:
    """
    adds new axis to fig and create a step plot on that counts number of cells:
    - which are fully contained s.t. the sum of emplyoment will be requested
    - which (might be) intersected s.t. all points within will need to be checked whether they are within r  
    Returns: axes 
    """
    ax_counter = fig.add_axes([0.6, 0.43, 0.3, 0.25])
        
    ax_counter.plot(
        allGridSizes['grid_spacing'], allGridSizes['contain_count'], color='green'
        )
    ax_counter.plot(
        allGridSizes['grid_spacing'], allGridSizes['overlap_count'], color='red'
        )
    ax_counter.plot(allGridSizes['grid_spacing'], allGridSizes['overlap_count'], color='#000')
    minx, maxx = min(allGridSizes['grid_spacing']),max(allGridSizes['grid_spacing'])
    ax_counter.set_xlim(minx,maxx)
    ax_counter.set_ylim(0,max(allGridSizes['contain_count']+allGridSizes['overlap_count']))
    
    ax_counter.set_yticks(yticks)
    
    return ax_counter
#

def update_fig_title(fig, grid_spacing:float,it:dict):
    """
    updates fig.suptitle with rounded grid_spacing of current frame
    """
    # display grid_spacing as title
    gridsize_rnd = round(grid_spacing,2) 
    fig.suptitle(
        (' ' if gridsize_rnd >= 100 else '') +
        str(gridsize_rnd)[:5+int(gridsize_rnd >= 100)] +
        '0'*max(0,len(str(gridsize_rnd))-int(gridsize_rnd >= 100)-3) +
        '. Frame: '+str(it)
    )
#

def update_main_axis_frame(
        allGridSizes:_pd_DataFrame, 
        grid_spacing:float, 
        ax_main,
        ax_min:float=0, 
        ax_max:float=1,
        r:float=750,
    ) -> list:
    # clear previous picture
    ax_main.clear()
    
    listOfPatches = []
    # create patches for grid cell: green=overlapped, red= (potentially) intersected, grey=outside 
    listOfPatches += create_grid_cell_patches( 
        grid_spacing=grid_spacing, 
        ax_min=ax_min, 
        ax_max=ax_max,
        contain_cells_row_col=allGridSizes.loc[grid_spacing, 'contain_ids'],
        overlap_cells_row_col=allGridSizes.loc[grid_spacing, 'overlap_ids'],
        )
    
    # create patches for circle around bottom left and top right corner of center grid cell
    listOfPatches += create_circle_patches(
        grid_spacing=grid_spacing,
        r=r
    )

    for patchToAdd in listOfPatches:
        ax_main.add_patch(patchToAdd)

    points = [p[0] for p in [
        ax_main.plot(0, 0, marker='x', color='black'),
    ]]

    # set tickmarks and limits
    ax_main.set_yticks([r-grid_spacing/2, r, r+grid_spacing/2]) 
    ax_main.set_xticks([r-grid_spacing/2, r, r+grid_spacing/2]) 
    ax_main.set_xlim(ax_min,ax_max)
    ax_main.set_ylim(ax_min,ax_max)
    ax_main.set_aspect('equal', adjustable='box')

    return *listOfPatches, *points, #, point1,
#

def update_coverage_axis_frame(
    grid_spacing:float,
    max_gridsize:float,
    min_gridsize:float,
    ax_coverage,
    vline,
    ) -> tuple:
    """
    
    """
    # update vline
    if vline != None:
            vline.remove()
    vline = ax_coverage.vlines(x=grid_spacing, ymin=0, ymax=3)
    ax_coverage.set_xticks([round(x,0) for x in set([min_gridsize, grid_spacing, max_gridsize])])
    return (vline,)
#

def create_optimal_grid_spacing_gif (
        relevantGridSizes:_pd_DataFrame,
        allGridSizes:_pd_DataFrame,
        largest:float=400,
        smallest:float=80,
        r:float=750,
        frames_between_relevant:int = 6,
        FuncAnimation_interval:int=500,
        FuncAnimation_repeat:bool=True,
        FuncAnimation_repeat_delay:int=1500,
        FuncAnimation_cache_frame_data:bool=True,
        dpi:int=50,
        PillowWriterFps:int=10,
        output_dir:str='./opt_grid_'
        )->tuple:
    
    print('Create GIF')
    # initialise plot and main axis
    fig,ax_main = plt.subplots()

    # define ax limits for main axis 
    min_gridsize = min(relevantGridSizes['grid_spacing'])
    max_gridsize = max(relevantGridSizes['grid_spacing'])
    max_relevant_cells = max(relevantGridSizes['contain_count']+relevantGridSizes['overlap_count'])
    ax_min=-max_gridsize/2
    ax_max= max_gridsize*(max(relevantGridSizes['cell_steps_max'])-.5)    
    
    # add stackplot to top right corner
    ax_coverage = add_coverage_plot(
        fig=fig, allGridSizes=allGridSizes,
        ymax=1.01*max(relevantGridSizes['share_contain']+relevantGridSizes['share_overlap']),
        yticks=[.5,1])

    # ax_counter = add_cell_counter_plot(
    #     fig=fig, 
    #     allGridSizes=allGridSizes,
    #     ymax=max_relevant_cells,
    #     yticks=[int(x*max_relevant_cells/4) for x in range(1,5)]
    #     )
    
    # 
    an_dct={
        'it':0,
        'vline':None
        }

    def animate(grid_spacing):
        """
        function to pass into FuncAnimation to create GIF. 
        Will be supplied with list of gridsizes to iterate over.
        """
        an_dct['it']=an_dct['it']+1
        
        update_fig_title(fig=fig,grid_spacing=grid_spacing,it=an_dct['it'])
        
        # update main axis:
        artistObjects = update_main_axis_frame(
            allGridSizes=allGridSizes, 
            grid_spacing=grid_spacing,
            ax_main=ax_main, 
            ax_min=ax_min, 
            ax_max=ax_max,
            r=r,
        )
        
        #
        an_dct['vline'], = update_coverage_axis_frame(
            grid_spacing=grid_spacing,
            max_gridsize=max_gridsize,
            min_gridsize=min_gridsize,
            ax_coverage=ax_coverage,
            vline=an_dct['vline']
            ) 

        return artistObjects
    #

    animation_frames = set_animation_frames(
        relevantGridSizes=relevantGridSizes,
        allGridSizes=allGridSizes,
        largest=largest,
        frames_between_relevant=frames_between_relevant
        )

    ani = _plt_FuncAnimation(
        fig, 
        animate, 
        interval=FuncAnimation_interval, 
        blit=True, 
        repeat=FuncAnimation_repeat, repeat_delay=FuncAnimation_repeat_delay, 
        cache_frame_data=FuncAnimation_cache_frame_data,
        frames=animation_frames
        )    

    print('Save GIF')
    ani.save(
        output_dir+str(largest)+"_"+str(smallest)+".gif", 
        dpi=dpi, 
        writer=_plt_PillowWriter(fps=PillowWriterFps)
        )

    return fig,ax_main,ax_coverage,#ax_counter,
#
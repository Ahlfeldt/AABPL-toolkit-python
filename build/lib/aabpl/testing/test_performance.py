from time import process_time
from pandas import DataFrame as _pd_DataFrame, concat as _pd_concat
import matplotlib.pyplot as plt
from numpy import (array as _np_array, arange as _np_arange, linspace as _np_linspace)
from functools import wraps

func_timer_dict = {
    'times': [],
    'overhead': None
}
#

def time_func_perf(func):
    @wraps(func)
    def wrapper(*args,**kwargs):
        start_time = process_time()
        pos = len(func_timer_dict['times'])
        func_timer_dict['times'].append({'func_name':func.__name__, 'start_time':start_time})
        result = func(*args,**kwargs)
        func_timer_dict['times'][pos].update({'end_time':process_time()})
        return result
    return wrapper
#

@time_func_perf
def time_single_perform_overhead():
    pass
#
def time_single_perform_no_overhead():
    pass
#

def overhead_per_perf_timing(iters:int=10000, ):
    """
    times the overhead introduced due to performance timing to later exclude it from performance analysis
    """
    
    start_time = process_time()
    for i in range(iters):
        time_single_perform_overhead()
    time_with_overhead = (process_time() - start_time)
    
    start_time = process_time()
    for i in range(iters):
        time_single_perform_no_overhead()
    time_without_overhead = (process_time() - start_time)
    
    delay_per_iteration = (time_with_overhead - time_without_overhead) / iters
    
    # drop rows create for this speed test
    func_timer_dict['times'] = func_timer_dict['times'][:-iters]
    
    return delay_per_iteration
#

def reset_perf_times(
        # func_timer_dict:dict = func_timer_dict,
        ):
    """
    reset performance times
    """
    func_timer_dict['times'] = []
#

def analyze_func_perf(
        # func_timer_dict:dict = func_timer_dict,
        plot=False,
        iters:int=10000,
        threshold:float=0.01,
    ):
    # todo put plot ints its on function single_performance_plot()
    # allow for threshold value under which regions are aggregated to other and displayed as such
    if func_timer_dict['overhead'] is None:
        func_timer_dict['overhead'] = overhead_per_perf_timing(iters)
    overhead = func_timer_dict['overhead']

    # first create hierarchy. 
    func_name_to_cat = dict()
    concat_name_to_cat = dict()
    open_funcs = []
    grp_dicts = []
    func_dicts = []
    nest_level_max = 0
    for t_dict in func_timer_dict['times']:

        func_name = t_dict['func_name']
        start_time, end_time = t_dict['start_time'], t_dict['end_time']
        time_elapsed = (end_time-start_time)
        process_time = max([0., (end_time-start_time)-overhead])
        if func_name not in func_name_to_cat:
            func_name_to_cat[func_name] = str(len(func_name_to_cat))
        #
        cat = func_name_to_cat[func_name]
        
        
        t_dict.update({
                'cat': cat, 
                'time_elapsed': time_elapsed,
                'process_time': process_time,
                'n_calls': 1,
        })

        if len(open_funcs) > 0 and open_funcs[-1]['end_time'] <= start_time:
                # close the last one maybe save this now to grp dicts?
                open_funcs.pop()
        #

        if not any([func['func_name'] == func_name for func in open_funcs]):
            # nest function call
            if len(open_funcs) > 0:
                open_funcs[-1]['process_time'] = max([0., open_funcs[-1]['process_time'] - time_elapsed])  
            #
            
            open_funcs.append(t_dict)
            grp_dicts.append(t_dict)
            nest_level = len(open_funcs)-1
            nest_level_max = max([nest_level_max, nest_level])
            t_dict.update({
                'nest_level': nest_level,
                'path_cat': "-".join([func['cat'] for func in open_funcs]),
                'path_name': "-".join([func['func_name'] for func in open_funcs]),
                **{'path_cat_'+str(i): "-".join([func['cat'] for func in open_funcs[:i+1]]) for i, open_func in enumerate(open_funcs)},
                **{'path_name_'+str(i): "-".join([func['func_name'] for func in open_funcs[:i+1]]) for i, open_func in enumerate(open_funcs)},
            })
            concat_name_to_cat[t_dict['path_name']] = t_dict['path_cat'] 
            for open_func in open_funcs:
                t_dict.update({
                })    
            #
        else:
            # dont nest function: either its recursively calling itself func1(func1())
            # computing time of func1 should not be substracted from func1
            # or being called by a child func1(func2(func1()))
            # computing time of func1 should not be substracted from func1 but from 
            for i, open_func in enumerate(open_funcs[::-1]):
                if open_func == func_name:
                    open_func['n_calls'] += 1 
                    if i > 0:
                        open_func['process_time'] += process_time
                    else: 
                        # if the function is nested within itself immediately only deduct overhead
                        open_func['process_time'] = max([0., process_time - overhead])
                    #
                    break
                #
            #
        #
    #
    
    grp_df = _pd_DataFrame(grp_dicts)
    if len(grp_df['path_cat_0'].unique())==1 and nest_level_max > 0:
        grp_df = grp_df[grp_df['path_cat'] != grp_df['path_cat_0']]
        grp_df.drop(columns=['path_cat_0', 'path_name_0'], inplace=True)
        grp_df.rename(columns={'path_cat_'+str(i):'path_cat_'+str(i-1) for i in range(1, nest_level_max+1)}, inplace=True) 
        grp_df.rename(columns={'path_name_'+str(i):'path_name_'+str(i-1) for i in range(1, nest_level_max+1)}, inplace=True) 
        concat_name_to_cat = {"-".join(key.split("-")[1:]): "-".join(val.split("-")[1:]) for key, val in concat_name_to_cat.items()}
        grp_df['path_cat'] = grp_df['path_cat'].str.split('-', n=1).str.get(-1)
        grp_df['path_name'] = grp_df['path_name'].str.split('-', n=1).str.get(-1)
        nest_level_max -= 1
        for i in range(nest_level_max+1):
            grp_df['path_cat_'+str(i)] = grp_df['path_cat_'+str(i)].str.split('-', n=1).str.get(-1)
            grp_df['path_name_'+str(i)] = grp_df['path_name_'+str(i)].str.split('-', n=1).str.get(-1)
    
    # intial_sort = grp_df.index
    # grp_df.sort_values(['path_cat'])
    # grp_df['path_int'] = range(len(grp_df))

    for i in range(1, nest_level_max+1):
        grp_df.fillna({'path_cat_'+str(i): grp_df['path_cat_'+str(i-1)]}, inplace=True)
        grp_df.fillna({'path_name_'+str(i): grp_df['path_name_'+str(i-1)]}, inplace=True)
    
    if plot:
        # plot by time per function call
        # plot by share of total computing time
        # group colors 
        fig, ax = plt.subplots(figsize=(10,10))
        # vals = _np_array([[60., 32.], [37., 40.], [29., 10.]])
        space_dict = dict()
        n_cats = len(grp_df['path_cat'].unique())
        for i in range(0, nest_level_max+1)[::-1]:
            #[grp_df['nest_level']>=i]
            cat_to_unique_cats = sorted(list(grp_df.groupby(['path_cat_'+str(i)])['path_cat'].unique().items()))
            cum_n_at_path = [sum([len(val) for key, val in list(cat_to_unique_cats)[:n]])/n_cats for n in range(len(cat_to_unique_cats))]
            # update only dicts that are at current or lower nest levels
            updt = {
                key:cum for (key,val), cum in 
                zip(cat_to_unique_cats, cum_n_at_path) if key.count("-") == i
            }
            space_dict.update(updt)
        
        total_process_time = grp_df['process_time'].sum()
        used_cmap = plt.get_cmap('jet')
        other_name, other_color = 'other', '#ccc'
        used_names = []
        used_colors = []
        for i, r in zip(range(nest_level_max)[::-1], _np_linspace(0.0, 1, nest_level_max+1)[1:][::-1]):
            group_col = ['path_cat_'+str(i)]
            xs = grp_df.groupby(group_col)['process_time'].sum()
            below_threshold = (xs/total_process_time) <= threshold
            # x[below_threshold] = x[below_threshold]

            labels = [label if (x/total_process_time)>threshold else other_name for x, label in zip(xs, grp_df.groupby(group_col).groups.keys())]
            names = [name if (x/total_process_time)>threshold else other_name for x, name in zip(xs, grp_df.groupby(['path_name_'+str(i)]).groups.keys())]
            colors = [used_cmap(space_dict[key]) if (x/total_process_time)>threshold else other_color for x, key in zip(xs, labels)]
            for name, color, xs0 in zip(names, colors, xs):
                if name not in used_names:
                    used_names.append(name)
                    used_colors.append(color)
                else:
                    used_colors[used_names.index(name)] = color

            # colors = [used_cmap(cmap_dict[key]) for key in labels]
            patches, text = ax.pie(
                x=xs,
                radius=r,
                colors=colors,
                labels=['' if name == other_name else (n.split('-')[-1] if n.split('-')[-1] != '__init__' else n.split('-')[-2]) if xs/total_process_time>0.000 else '' for n,xs in zip(names,xs)],
                labeldistance=.999,
                textprops={'fontsize': 7},
                wedgeprops={'edgecolor':'black'}, # TODO maybe its possible to pass a vector of bordercolors in here(remove if same slice)
                )
            func_timer_dict['cmap_dict']=space_dict
        # title_str = "Effective time: "+ str(grp_df['total_no_sub'].sum()) + ". Total time:"+str(max(grp_df['end_time'])-min(grp_df['start_time']))
        # print("title_str",title_str)
        title = 'Total process time:'+str(round(total_process_time,1))+'s.'
        ax.set(aspect="equal", title=title)
        handles = [
            plt.Rectangle((0, 0), 0, 0, color=color if name != other_name else other_color, label=name.split('-')[-1]) # used_cmap(space_dict[concat_name_to_cat[name]])
            for name, color in zip(used_names, used_colors)]
        ax.legend(handles=handles, framealpha=0., bbox_to_anchor=(0.85,1.025), loc="upper left")# bbox_to_anchor=(0.2, 1.1),
        plt.show()
    func_timer_dict['grp_df'] = grp_df
    
    return func_timer_dict
#

def compare_performace(list_of_kwargs, func):
    for kwargs in list_of_kwargs:
        func(**kwargs)
    # write function to compare perfomrance across different param inputs
    pass
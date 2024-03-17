#Module to hold functions from data processing and plotting
#avoiding excessive repeated code

import pandas as pd
import matplotlib.cm as cm
import numpy as np
import matplotlib.pyplot as plt

#processes dataframe for stacked bars - commute vs all trips
#input: dataframe of all trips, flag to insert missing "Other" row
#output: formatted df of work trips and all trips with counts and percentages for modes
def format_for_mode_bars(trip_df, dataset, insert_other=False):
  all_trips_df = trip_df.copy()

  all_trips_df.loc[all_trips_df['Mode_confirm']=='Personal Micromobility', 'Mode_confirm'] = 'Other'
  all_trips_df.loc[all_trips_df['Mode_confirm']=='Shared Micromobility', 'Mode_confirm'] = 'Other'  

  all_trips = all_trips_df.groupby(['Mode_confirm'], as_index=False).count()[['Mode_confirm','distance_miles']]
  all_trips['proportion'] = all_trips['distance_miles'] / np.sum(all_trips.distance_miles)
  all_trips['trip_type'] = 'All Trips'

  work_trips = all_trips_df[all_trips_df['Trip_purpose']=='Work'].copy()
  work_trips = work_trips.groupby(['Mode_confirm'], as_index=False).count()[['Mode_confirm','distance_miles']]
  work_trips['proportion'] = work_trips['distance_miles'] / np.sum(work_trips.distance_miles)
  work_trips['trip_type'] = 'Work Trips'

  if insert_other:
    work_trips.loc[1.5] = 'Other', 0, 0, 'Work Trips'
    work_trips = work_trips.sort_index().reset_index(drop=True)

  formatted_df = pd.concat([all_trips,work_trips])
  formatted_df['Dataset'] = dataset
  formatted_df.columns = ['Mode','Count','Proportion','Trip Type', "Dataset"]

  return formatted_df

#input: dataframe formatted for plotting, dimension (Mode or Purpose)
def make_mini_vs_full(plot_data, dimension):
    width = 0.8
    fig, ax = plt.subplots(2,1, figsize=(20,10))
    plt.rcParams.update({'font.size': 30}) 
    running_total_mini = [0,0]
    running_total_long = [0,0]
    fig_data_mini = plot_data[plot_data['Dataset']=='Minipilot']
    fig_data_long = plot_data[plot_data['Dataset']=='Long Term']

    for mode in pd.unique(fig_data_mini[dimension]):
        mini = fig_data_mini[fig_data_mini[dimension]==mode]
        long = fig_data_long[fig_data_long[dimension]==mode]

        labels = mini['Trip Type']
        vals = mini['Proportion']*100
        bar_labels = mini['Count']
        vals_str = [f'{y:.1f} %\n({x:,})' if y>5 else '' for x, y in zip(bar_labels, vals)]
        bar = ax[0].barh(labels, vals, width, left=running_total_mini, label=mode)
        ax[0].bar_label(bar, label_type='center', labels=vals_str, rotation=90, fontsize=22)
        running_total_mini[0] = running_total_mini[0]+vals.iloc[0]
        running_total_mini[1] = running_total_mini[1]+vals.iloc[1]

        labels = long['Trip Type']
        vals = long['Proportion']*100
        bar_labels = long['Count']
        vals_str = [f'{y:.1f} %\n({x:,})' if y>5 else '' for x, y in zip(bar_labels, vals)]
        bar = ax[1].barh(labels, vals, width, left=running_total_long, label=mode)
        ax[1].bar_label(bar, label_type='center', labels=vals_str, rotation=90, fontsize=22)
        running_total_long[0] = running_total_long[0]+vals.iloc[0]
        running_total_long[1] = running_total_long[1]+vals.iloc[1]

    ax[0].set_title('Minipilot', fontsize=25)
    ax[1].set_title('All Programs', fontsize=25)
    ax[0].legend(bbox_to_anchor=(1,1), fancybox=True, shadow=True, fontsize=25)
    plt.subplots_adjust(bottom=0.20)
    fig.tight_layout()
    plt.show()
    
def format_mode_by_program(df, program_list, work = False):
    mode_data = df.copy()
    subset_plot_data = []
    for program in program_list:
        program_data = mode_data[mode_data.program == program]

        if work:
            program_data = program_data[program_data['Trip_purpose']=='Work'].copy()
            
        formatted_df = program_data.groupby(['Mode_confirm'], as_index=False).count()[['Mode_confirm','distance_miles']]
        formatted_df[program] = (formatted_df['distance_miles'] / np.sum(formatted_df.distance_miles)) * 100
        formatted_df = formatted_df.set_index('Mode_confirm')
        formatted_df = formatted_df.drop(columns = ['distance_miles'])
        subset_plot_data.append(formatted_df)


    formatted_trips = pd.concat(subset_plot_data, axis = 1)
    formatted_trips = formatted_trips.transpose()
    formatted_trips['program'] = formatted_trips.index
    formatted_trips = formatted_trips.replace({'4c': 'Four Corners\n(Durango)', 
                                   'cc': 'Comunity Cycles\n(Boulder)',
                                   'sc': 'Smart Commute\n(Denver North)',
                                   'pc':'Pueblo',
                                   'vail':'Vail',
                                   'fc':'Fort Collins'})
    formatted_trips = formatted_trips.set_index('program')

    return formatted_trips


#from https://stackoverflow.com/questions/22787209/how-to-have-clusters-of-stacked-bars
def plot_clustered_stacked(dfall, labels=None, title="multiple stacked bar plot",  H="/", **kwargs):
    """Given a list of dataframes, with identical columns and index, create a clustered stacked bar plot. 
labels is a list of the names of the dataframe, used for the legend
title is a string for the title of the plot
H is the hatch used for identification of the different dataframe"""

    n_df = len(dfall)
    n_col = len(dfall[0].columns) 
    n_ind = len(dfall[0].index)
    axe = plt.subplot(111)

    for df in dfall : # for each data frame
        axe = df.plot(kind="bar",
                      stacked=True,
                      ax=axe,
                      legend=False,
                      grid=False,
                      **kwargs)  # make bar plots

    h,l = axe.get_legend_handles_labels() # get the handles we want to modify
    for i in range(0, n_df * n_col, n_col): # len(h) = n_col * n_df
        for j, pa in enumerate(h[i:i+n_col]):
            for rect in pa.patches: # for each index
                rect.set_x(rect.get_x() + 1 / float(n_df + 1) * i / float(n_col))
                rect.set_hatch(H * int(i / n_col)) #edited part     
                rect.set_width(1 / float(n_df + 1))

    axe.set_xticks((np.arange(0, 2 * n_ind, 2) + 1 / float(n_df + 1)) / 2.)
    axe.set_xticklabels(df.index, rotation = 0)
    axe.set_title(title, fontsize = 22)

    # Add invisible data to add another legend
    n=[]        
    for i in range(n_df):
        n.append(axe.bar(0, 0, color="gray", hatch=H * i))

    l1 = axe.legend(h[:n_col], l[:n_col], loc=[1.01, 0.5])
    if labels is not None:
        l2 = plt.legend(n, labels, loc=[1.01, 0.1]) 
    axe.add_artist(l1)
    return axe

def format_purpose_bars(trip_df, dataset):
    trips = trip_df.copy()
    trips = trips[~trips['Trip_purpose'].isin(['No travel'])]
    trips.loc[trips['Trip_purpose']=='Religious', 'Trip_purpose'] = 'Other'
    trips.loc[trips['Trip_purpose']=='School', 'Trip_purpose'] = 'Other'

    all_trips = trips.groupby(['Trip_purpose'], as_index=False).count()[['Trip_purpose','distance_miles']]
    all_trips['proportion'] = all_trips['distance_miles'] / np.sum(all_trips.distance_miles)
    all_trips['trip type'] = 'All Trips'

    ebike_trips = trips[trips['Mode_confirm']=='E-bike'].copy()
    ebike_trips = ebike_trips.groupby(['Trip_purpose'], as_index=False).count()[['Trip_purpose','distance_miles']]
    ebike_trips['proportion'] = ebike_trips['distance_miles'] / np.sum(ebike_trips.distance_miles)
    ebike_trips['trip type'] = 'E-Bike Trips'
    
    if dataset == "Long Term": #full dataset needs pickup/dropoff added
        all_trips.loc[len(all_trips.index)] = ['Pick-up/Drop off', 0, 0, 'All Trips']
        ebike_trips.loc[len(ebike_trips.index)] = ['Pick-up/Drop off', 0, 0, 'E-Bike Trips']

    formatted_trips = pd.concat([all_trips, ebike_trips])
    formatted_trips['Dataset'] = dataset
    formatted_trips.columns = ['Purpose','Count','Proportion','Trip Type', "Dataset"]

    return formatted_trips

def make_occupation_chart(df, plot_title, filename):
    plot_data = df.copy()

    mode_dist_by_user = plot_data.groupby(['user_id','Mode_confirm'], as_index=False).count()[['user_id','Mode_confirm','distance_miles']]
    mode_dist_by_user['distance_miles'].fillna(0, inplace=True)
    distance_by_user = plot_data.groupby(['user_id'], as_index=False).count()[['user_id','distance_miles']]
    plot_data = mode_dist_by_user.merge(distance_by_user, on='user_id')
    plot_data['proportion'] = plot_data['distance_miles_x'] / plot_data['distance_miles_y']
    plot_data['proportion'].fillna(0, inplace=True)
    occupation_users = data.copy().groupby(['occupation_cat','user_id'], as_index=False).nth(0)[['occupation_cat','user_id']]

    plot_data = plot_data[plot_data['Mode_confirm']=='E-bike']
    plot_data = plot_data.merge(occupation_users, on='user_id')

    ylab='Occupation Category'
    fig, ax = plt.subplots(figsize=(8,6))
    sns.barplot(data= plot_data, x='proportion' , y='occupation_cat', estimator=np.mean).set(title=plot_title,xlabel='Proportion of Total Trips',ylabel=ylab)
    plt.xticks(rotation=35, ha='right')
    plt.subplots_adjust(bottom=0.25)

    plt.savefig(filename, bbox_inches='tight')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append('/Users/mallen2/alternate_branches/eval-compatible-server/e-mission-server')

import emission.storage.timeseries.abstract_timeseries as esta
import emission.storage.decorations.trip_queries as esdtq



def get_expanded_labeled_trips(user_list):
    confirmed_trip_df_map = {}
    labeled_trip_df_map = {}
    expanded_labeled_trip_df_map = {}
    for u in user_list:
        ts = esta.TimeSeries.get_time_series(u)
        ct_df = ts.get_data_df("analysis/confirmed_trip")

        confirmed_trip_df_map[u] = ct_df
        labeled_trip_df_map[u] = esdtq.filter_labeled_trips(ct_df)
        expanded_labeled_trip_df_map[u] = esdtq.expand_userinputs(
            labeled_trip_df_map[u])

    return pd.concat(expanded_labeled_trip_df_map.values(), ignore_index=True)

def relative_error(m,t):
    # measured minus true over true
    return (m-t)/t


def drop_unwanted_trips(df,drop_not_a_trip):

    print('Dropping user labeled air trips and trips with no OS.')
    df = df.copy()
    df = df.drop(df[df.mode_confirm == 'air'].index)
    df = df[df['os'].notna()]  # several stage trips have nan operating systems.

    if drop_not_a_trip == True: 
        print("Also dropping trips labeled as not a trip and trips with mode_confirm of nan.")
        df = df.drop(df[df.mode_confirm.isin(['not_a_trip','no_travel'])].index)
        df = df[df['mode_confirm'].notna()]

    #for i,ct in df.iterrows():

        # To look at only trips where the user label is in ['drove_alone','bike','bus','walk'] and the predicted mode is in
        # ["car","walking","bicycling","no_sensed","bus"]
        '''    if ct['mode_confirm'] not in ['drove_alone','bike','bus','walk']:
            expanded_labeled_trips = expanded_labeled_trips.drop(index = i)
        elif any(x not in ["car","walking","bicycling","no_sensed","bus"] for x in ct['section_modes']):
            # if any section mode is not in the above list, drop the trip.
            expanded_labeled_trips = expanded_labeled_trips.drop(index = i)'''

        # This code is to look at correctly labeled trips.
        # ie the sensed mode maps to the same energy intensity (when there is no confusion) as the user labeled mode maps to.
        # This will also include shared rides and use a load factor.
        '''    if ct['mode_confirm'] not in MODE_MAPPING_DICT:
            expanded_labeled_trips = expanded_labeled_trips.drop(index = i)
            continue
        if not (MODE_MAPPING_DICT[ct['mode_confirm']]==MODE_MAPPING_DICT[ct['primary_mode']]):
            # one last check to make sure we don't drop shared rides that were labeled as car.
            if not ((ct['mode_confirm'] == 'shared_ride') and (ct['primary_mode'] == 'car')):
                expanded_labeled_trips = expanded_labeled_trips.drop(index = i)
                continue'''
    return df

def get_ratios_for_dataset(df):
    # Take a confirmed trips dataframe.
    # return:
    #   the ratio of drove alone distance to shared ride distance.
    #   the proportion of car distance
    #   the proportion of ebike distance
    mode_distances = df.groupby('mode_confirm').sum()['distance']

    drove_alone_distance = 0
    shared_ride_distance = 0
    all_modes_distance = 0
    for mode in mode_distances.index:
        if mode == np.nan or type(mode) == float: continue
        elif (('car' in mode) & ('alone' in mode)) or (mode == 'drove_alone'):
            drove_alone_distance += mode_distances[mode]
        elif (('car' in mode) & ('with others' in mode)) or mode == 'shared_ride':
            shared_ride_distance += mode_distances[mode]    
        all_modes_distance += mode_distances[mode]  # should nan trip distance be included?

    ebike_distance = mode_distances.loc['pilot_ebike']
    walk_distance = mode_distances.loc['walk']

    #print(f"Distance in drove alone, shared ride (m): {drove_alone_distance:.1f}, {shared_ride_distance:.1f}")

    r = drove_alone_distance/shared_ride_distance
    car_proportion = (drove_alone_distance + shared_ride_distance)/all_modes_distance
    ebike_proportion = ebike_distance/all_modes_distance
    walk_proportion = walk_distance/all_modes_distance
    drove_alone_proportion = drove_alone_distance/all_modes_distance
    shared_ride_proportion = shared_ride_distance/all_modes_distance

    proportion_dict = {
        "r": r, "car_proportion": car_proportion, "ebike_proportion": ebike_proportion,
        "walk_proportion": walk_proportion,
        "drove_alone_proportion": drove_alone_proportion,
        "shared_ride_proportion": shared_ride_proportion,
        "drove_alone_distance": drove_alone_distance,
        "shared_ride_distance": shared_ride_distance
    }

    return  proportion_dict

def get_primary_modes(df,energy_dict,MODE_MAPPING_DICT):
    # Add primary mode and length columns to expanded labeled trips
    df = df.copy()
    primary_modes = []
    primary_lengths = []

    no_sections_count = 0
    for i,ct in df.iterrows():
        # Get primary mode
        if len(ct["section_distances"]) == 0: # for data up to 5-9-2022, there are 63 stage trips with no sensed sections.
            # maybe I should instead set primary mode to na 
            # and use an estimated energy consumption of 0 for these trips.
            df = df.drop(index = i)    
            no_sections_count += 1
        else:
            longest_section = max(ct["section_distances"])
            primary_mode = ct["section_modes"][ct["section_distances"]==longest_section]

            # in case there are ever tied longest sections.
            # pick the most energy intensive mode.
            if isinstance(primary_mode,list): 
                mini_energy_dict = {x:energy_dict[MODE_MAPPING_DICT[x]] for x in primary_mode}
                primary_mode = max(mini_energy_dict, key=mini_energy_dict.get)

            primary_modes.append(primary_mode)
            primary_lengths.append(longest_section)

    df['primary_mode'] = primary_modes
    df['primary_length'] = primary_lengths
    print(f"Dropped {no_sections_count} trips with no sensed sections.")
    return df

def get_outliers(df, column_of_interest, u_percentile, l_percentile):
    '''takes a dataframe and returns a subsetted dataframe with the outlier values for the column of interest.
    u_percentile: upper percentile
    l_percentile: lower percentile
    outliers will be the values above the upper percentile and below the lower percentile.
    '''
    quantiles = [u_percentile,l_percentile] 
    upper, lower = np.percentile(df[column_of_interest], quantiles)
    print(f"{quantiles[0]:.2f} and {quantiles[1]:.2f} percentiles: {upper:.2f},{lower:.2f}")

    outliers = df[(df[column_of_interest] < lower) | (df[column_of_interest] > upper)]
    return outliers

def plot_energy_consumption_by_mode(energy_consumption_df,program_name):
    df = energy_consumption_df.copy()
    main_mode_labels = ['drove_alone','shared_ride','walk','pilot_ebike','bus','bike','train','taxi','free_shuttle']
    program_main_mode_labels = [x for x in main_mode_labels if x in df.mode_confirm.unique()] # 4c doesn't have train before May 2022.

    program_main_modes_EC = df.groupby('mode_confirm').sum().loc[program_main_mode_labels]
    program_main_modes_EC = program_main_modes_EC[['predicted','expected','user_labeled']]

    program_main_modes_EC.plot(kind='barh')
    program_percent_error_expected = 100*relative_error(df.expected.sum(),df.user_labeled.sum())
    plt.xlabel('Energy consumption (kWH)')
    plt.title(f"Energy consumption by mode for {program_name} (full % error for expected: {program_percent_error_expected:.2f})")
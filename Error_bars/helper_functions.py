import pandas as pd
import numpy as np

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
    return (m-t)/t


def drop_unwanted_trips(df):
    df = df.copy()
    df = df.drop(
        df[df.mode_confirm == 'air'].index
    )


    for i,ct in df.iterrows():

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

        # dropping air
        if 'air_or_hsr' in ct['section_modes']:
            #print(f"Sensed {ct['section_modes']}, user label was {ct['mode_confirm']}") 
            df= df.drop(index = i)
        elif type(ct['os']) == float:  # several stage trips have nan operating systems.
            df = df.drop(index = i)

        elif type(ct['mode_confirm']) == float: 
            df = df.drop(index = i)

        elif (ct['mode_confirm'] == 'not_a_trip') or (ct['mode_confirm'] == 'no_travel'):
            #print(f"Sensed {ct['section_modes']}, user label was {ct['mode_confirm']}") 
            df = df.drop(index = i)
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
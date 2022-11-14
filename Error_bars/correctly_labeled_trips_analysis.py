import argparse
import pandas as pd
import numpy as np
from uuid import UUID
import pickle
import os

import matplotlib.pyplot as plt

import sys
sys.path.append('/Users/mallen2/alternate_branches/eval-compatible-server/e-mission-server')

import helper_functions
# Covered by helper_functions
#import emission.storage.timeseries.abstract_timeseries as esta
#import emission.storage.decorations.trip_queries as esdtq

import emission.core.wrapper.user as ecwu


import sklearn.model_selection as skm
import confusion_matrix_handling as cm_handling
from confusion_matrix_handling import MODE_MAPPING_DICT
import get_EC

import emission.core.get_database as edb

METERS_TO_MILES = 0.000621371 # 1 meter = 0.000621371 miles

def relative_error(m,t):
    return (m-t)/t

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("chosen_program", type=str,
                        help="the program you want to run sensitivity analysis for")
    parser.add_argument("which_r", type=str,
                    help="Select the ratio of drove alone trips to shared ride trips. Options are dataset (use the ratio found in the dataset)\n \
                    or TEDB (use the ratio that sets Shankari\'s load factor to 1.5 (r = 1)")

    args = parser.parse_args()
    chosen_program = args.chosen_program
    which_r = args.which_r

    df_EI = pd.read_csv(r'Public_Dashboard/auxiliary_files/energy_intensity.csv') # r stands for raw string, only matters if the path is on Windows

    # Get error related info
    unit_dist_MCS_df = pd.read_csv("unit_distance_MCS.csv").set_index("moment")
    android_EI_moments_df = pd.read_csv("android_EI_moments.csv").set_index("mode")
    ios_EI_moments_df = pd.read_csv("ios_EI_moments.csv").set_index("mode")

    # Dictionary of energy intensities in kWH/PMT
    energy_dict = cm_handling.get_energy_dict(df_EI)

    # Fetch the pickle file of trips. I used place_all_trips_in_pkl.py to generate it.
    df = pd.read_pickle("/Users/mallen2/OpenPATH_Data/Sensing_sensitivity_analysis/expanded_labeled_trips.pickle")

    if chosen_program != 'all':
        expanded_labeled_trips = df[df['program'] == chosen_program].copy()
    else:
        expanded_labeled_trips = df.copy()

    # Add primary mode and length columns to expanded labeled trips
    primary_modes = []
    primary_lengths = []

    for i,ct in expanded_labeled_trips.iterrows():
        # Get primary mode
        if len(ct["section_distances"]) == 0: # for data up to 5-9-2022, there are 63 stage trips with no sensed sections.
            expanded_labeled_trips = expanded_labeled_trips.drop(index = i) 
            print("dropped")
            continue
        longest_section = max(ct["section_distances"])
        primary_mode = ct["section_modes"][ct["section_distances"]==longest_section]

        # in case there are ever tied longest sections.
        # pick the most energy intensive mode.
        if isinstance(primary_mode,list): 
            mini_energy_dict = {x:energy_dict[MODE_MAPPING_DICT[x]] for x in primary_mode}
            primary_mode = max(mini_energy_dict, key=mini_energy_dict.get)

        primary_modes.append(primary_mode)
        primary_lengths.append(longest_section)

    expanded_labeled_trips['primary_mode'] = primary_modes
    expanded_labeled_trips['primary_length'] = primary_lengths

    # drop air
    expanded_labeled_trips = expanded_labeled_trips.drop(
        expanded_labeled_trips[expanded_labeled_trips.mode_confirm == 'air'].index
        )

    print('Dropping air trips, trips labeled as not a trip, and trips with no OS.')
    for i,ct in expanded_labeled_trips.iterrows():

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
        if ct['mode_confirm'] not in MODE_MAPPING_DICT:
            expanded_labeled_trips = expanded_labeled_trips.drop(index = i)
            continue
        if not (MODE_MAPPING_DICT[ct['mode_confirm']]==MODE_MAPPING_DICT[ct['primary_mode']]):
            # one last check to make sure we don't drop shared rides that were labeled as car.
            if not ((ct['mode_confirm'] == 'shared_ride') and (ct['primary_mode'] == 'car')):
                expanded_labeled_trips = expanded_labeled_trips.drop(index = i)
                continue

        # dropping air
        if 'air_or_hsr' in ct['section_modes']:
            #print(f"Sensed {ct['section_modes']}, user label was {ct['mode_confirm']}") 
            expanded_labeled_trips = expanded_labeled_trips.drop(index = i)
        elif type(ct['os']) == float:  # several stage trips have nan operating systems.
            expanded_labeled_trips = expanded_labeled_trips.drop(index = i)

        elif type(ct['mode_confirm']) == float: 
            expanded_labeled_trips = expanded_labeled_trips.drop(index = i)

        elif (ct['mode_confirm'] == 'not_a_trip') or (ct['mode_confirm'] == 'no_travel'):
            #print(f"Sensed {ct['section_modes']}, user label was {ct['mode_confirm']}") 
            expanded_labeled_trips = expanded_labeled_trips.drop(index = i)

    print(expanded_labeled_trips['mode_confirm'].value_counts())

    # find the ratio of shared ride to drove alone:
    mode_distances = expanded_labeled_trips.groupby('mode_confirm').sum()['distance']

    drove_alone_distance = 0
    shared_ride_distance = 0
    for mode in mode_distances.index:
        if mode == np.nan or type(mode) == float: continue
        elif (('car' in mode) & ('alone' in mode)) or (mode == 'drove_alone'):
            drove_alone_distance += mode_distances[mode]
        elif (('car' in mode) & ('with others' in mode)) or mode == 'shared_ride':
            shared_ride_distance += mode_distances[mode]

    # for pc, if I include the labels below, r is 0.714. Otherwise r is 0.710

        '''    if (('car' in mode) & ('alone' in mode)) or (mode == 'drove_alone'):
            if 'with others' in mode:
                drove_alone_distance += mode_distances[mode]/2
                shared_ride_distance += mode_distances[mode]/2
                print('Drove alone and shared ride:')
            else:
                drove_alone_distance += mode_distances[mode]
                print('Drove Alone')
            print(f"{mode}")
        elif (('car' in mode) & ('with others' in mode)) or mode == 'shared_ride':
            shared_ride_distance += mode_distances[mode]
            print("Shared ride:")
            print(f"{mode}")'''

    r_for_dataset = drove_alone_distance/shared_ride_distance
    print(f"r for {chosen_program} is {r_for_dataset}")

    # for each trip, predict energy consumption with either the expectation or the prediction. compare it to the actual energy consumption.

    #android_EI_moments_df = pd.read_csv("android_EI_moments_corrected_load.csv").set_index("mode")
    #ios_EI_moments_df = pd.read_csv("ios_EI_moments_corrected_load.csv").set_index("mode")

    print("Computing trip level energy consumptions")
    new_car_precision = 0.83  # 0.739 for pc, 0.83 for vail.
    android_confusion = pd.read_csv("android_confusion.csv").set_index('gt_mode')
    ios_confusion = pd.read_csv("ios_confusion.csv").set_index('gt_mode')

    #new_android_cm = cm_handling.change_precision(android_confusion,'car',new_car_precision)
    #new_ios_cm = cm_handling.change_precision(ios_confusion,'car',new_car_precision)

    new_android_cm = android_confusion #cm_handling.drop_rows_and_columns(android_confusion,['Train','Pilot ebike','Scooter share'],['subway','train'])
    new_ios_cm = ios_confusion #cm_handling.drop_rows_and_columns(ios_confusion,['Train','Pilot ebike','Scooter share'],['subway','train'])

    r = 1 if args.which_r == 'TEDB' else r_for_dataset
    car_load_factor = (r+1)/(r+0.5)
    drove_alone_EI = energy_dict["Gas Car, drove alone"]
    energy_dict.update({"Gas Car, sensed": drove_alone_EI/car_load_factor})

    android_EI_moments_df = cm_handling.get_conditional_EI_expectation_and_variance(new_android_cm,energy_dict)
    ios_EI_moments_df = cm_handling.get_conditional_EI_expectation_and_variance(new_ios_cm,energy_dict)

    expected = []
    predicted = []
    user_labeled = []

    confusion_based_variance = []
    user_based_variance = []

    predicted_dict = {}
    expected_dict = {}

    expected_error_list = []
    prediction_error_list = []

    for i,ct in expanded_labeled_trips.iterrows():

        # Calculate expected energy consumption
        trip_expected, trip_confusion_based_variance = get_EC.get_expected_EC_for_one_trip(ct,unit_dist_MCS_df,android_EI_moments_df,ios_EI_moments_df)

        # Calculate predicted energy consumption
        trip_predicted = get_EC.get_predicted_EC_for_one_trip(ct,unit_dist_MCS_df,energy_dict)[0]
        
        # Calculate user labeled energy consumption
        trip_user_labeled, trip_user_based_variance = get_EC.get_user_labeled_EC_for_one_trip(ct,unit_dist_MCS_df,energy_dict)

        expected.append(trip_expected)
        predicted.append(trip_predicted)
        user_labeled.append(trip_user_labeled)

        confusion_based_variance.append(trip_confusion_based_variance)
        user_based_variance.append(trip_user_based_variance)

        user_mode = ct['mode_confirm']
        if user_mode not in predicted_dict: predicted_dict[user_mode] = []
        if user_mode not in expected_dict: expected_dict[user_mode] = []

        prediction_error = trip_predicted - trip_user_labeled
        expected_error = trip_expected - trip_user_labeled

        expected_error_list.append(expected_error)
        prediction_error_list.append(prediction_error)

        if abs(expected_error) < 100: 

            predicted_dict[user_mode].append(prediction_error)
            expected_dict[user_mode].append(expected_error)
        else:
            print(f"Large EC error: EC user labeled, EC expected: {trip_user_labeled, trip_expected}")
            print(f"\tTrip info: mode_confirm,sensed,distance (mi): {ct['mode_confirm'],ct['section_modes'],ct['distance']*METERS_TO_MILES}")

    def relative_error(m,t):
        return (m-t)/t
    total_expected = sum(expected)
    total_predicted = sum(predicted)
    total_user_labeled = sum(user_labeled)
    print(f"Total EC: expected, predicted, user labeled: {total_expected:.2f}, {total_predicted:.2f}, {total_user_labeled:.2f}")
    print(f"standard deviation for expected: {np.sqrt(sum(confusion_based_variance)):.2f}")
    print(f"Percent error: {relative_error(sum(expected),sum(user_labeled))*100:.3f}")

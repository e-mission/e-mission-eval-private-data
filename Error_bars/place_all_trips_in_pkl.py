import pandas as pd

import sys
sys.path.append('/Users/mallen2/alternate_branches/eval-compatible-server/e-mission-server')
import pickle

import helper_functions
# Covered by helper_functions
#import emission.storage.timeseries.abstract_timeseries as esta
#import emission.storage.decorations.trip_queries as esdtq

import emission.core.wrapper.user as ecwu

import emission.core.get_database as edb

##############################
# This script generates a dataframe of all confirmed trips, including OS, program, and expanded user inputs.
# It then saves it to a pickle file.
##############################
if __name__ == '__main__':

    # Get the list of users for each program.

    # Split UUIDs by program
    # ue stands for user_email
    all_user_list = []

    uuid_program_map = {}
    for ue in edb.get_uuid_db().find():
        program = ue['user_email'].split("_")[0]
        uuid = ue['uuid']
        uuid_program_map[uuid] = program
        all_user_list.append(uuid)


    # Get the OS for each user.
    print(f"Finding OS for each user in the database.")
    os_map = {}
    for u in all_user_list:
        profile = ecwu.User(u).getProfile()
        if 'curr_platform' in profile:
            os_map[u] = profile['curr_platform']
        else:
            print("Removed a user who had no OS information.")
            all_user_list.remove(u) # Note: this removes u from programs_uuid_map[chosen_program] as well.

    confirmed_trip_df_map = {}
    labeled_trip_df_map = {}
    expanded_labeled_trip_df_map = {}

    # Get all user labeled trips and expand the user inputs.
    print("Placing all confirmed trips into dataframe.")
    expanded_labeled_trips = helper_functions.get_expanded_labeled_trips(all_user_list)

    # Make an operating system column.
    expanded_labeled_trips['os'] = expanded_labeled_trips['user_id'].map(os_map)
    expanded_labeled_trips['program'] = expanded_labeled_trips['user_id'].map(uuid_program_map)

    output_path = "/Users/mallen2/OpenPATH_Data/Sensing_sensitivity_analysis/"
    pkl_file_name = output_path + "expanded_labeled_trips.pickle"

    # Drop info we don't need right now.
    expanded_labeled_trips = expanded_labeled_trips.drop(labels = ['source', 'end_ts', 'end_fmt_time', 'end_loc', 'raw_trip', 'start_ts',
       'start_fmt_time', 'start_loc','start_local_dt_year', 'start_local_dt_month', 'start_local_dt_day',
       'start_local_dt_hour', 'start_local_dt_minute', 'start_local_dt_second',
       'start_local_dt_weekday', 'start_local_dt_timezone',
       'end_local_dt_year', 'end_local_dt_month', 'end_local_dt_day',
       'end_local_dt_hour', 'end_local_dt_minute', 'end_local_dt_second',
       'end_local_dt_weekday', 'end_local_dt_timezone'], axis = 1)

    print("Saving pickle file.")
    with open(pkl_file_name, 'wb') as f:
        pickle.dump(expanded_labeled_trips, f)
    print("Done.")
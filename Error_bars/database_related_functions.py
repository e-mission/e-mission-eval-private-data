import sys
sys.path.append('/Users/mallen2/alternate_branches/eval-compatible-server/e-mission-server')  
# maybe make a config file to specify each path to emission server depending on what functions you need

import emission.storage.timeseries.abstract_timeseries as esta
import emission.storage.decorations.trip_queries as esdtq
import emission.core.get_database as edb
import emission.core.wrapper.user as ecwu

import time
import pandas as pd

def get_participants_programs_and_operating_systems():
    '''
    Returns: user_list: list of user ids
        os_map: dictionary by user id of operating systems.
        uuid_program_map: dictionary by user id of programs.
    '''
    user_list = []
    os_map = {}
    programs_all = {} # for printing number of users in each program.
    uuid_program_map = {}
    # This info is in the Stage_uuids collection of the database
    for u in edb.get_uuid_db().find():         # add users to proper locations in programs 
        uuid = u["uuid"]

        # Append the user to the list.
        user_list.append(uuid)

        # Get operating system info.
        profile = ecwu.User(uuid).getProfile()

        # Check whether they have OS information listed.
        if 'curr_platform' in profile:
            os_map[uuid] = profile['curr_platform']

            # Get program info.
            program = u['user_email'].split("_")[0]
            uuid_program_map[uuid] = program

            if program not in programs_all.keys(): programs_all[program] = []
            programs_all[program].append(uuid)

        else:
            print("Removed a user who had no OS information.")
            user_list.remove(uuid)

    print('Number of participants with operating system information in each program:')
    print({program: len(programs_all[program]) for program in programs_all})

    return user_list, os_map, uuid_program_map


def get_expanded_labeled_trips(user_list):
    '''
    Fetches labeled trips for each user in user_list from the database.

    user_list: list of uuid objects
    Returns a dataframe of labeled trips with expanded user inputs (1 column for each user input.)
    '''
    labeled_trip_df_map = {}

    n_all_trips = 0
    for u in user_list:
        ts = esta.TimeSeries.get_time_series(u)
        ct_df = ts.get_data_df("analysis/confirmed_trip")
        n_all_trips += len(ct_df)
        del ts
        labeled_trip_df_map[u] = esdtq.filter_labeled_trips(ct_df)
        del ct_df

    labeled_trips_df = pd.concat(labeled_trip_df_map.values(), ignore_index=True)
    expanded_labeled_trips = esdtq.expand_userinputs(labeled_trips_df)

    print(f"Labeling percent: {100*len(expanded_labeled_trips)/n_all_trips}")

    return expanded_labeled_trips

def get_confirmed_trips(user_list):
    '''
    Fetches confirmed trips for each user in user_list from the database.

    user_list: list of uuid objects
    Returns a dataframe of confirmed trips
    '''
    ct_df_map = {}
    for u in user_list:
        ts = esta.TimeSeries.get_time_series(u)
        ct_df_map[u] = ts.get_data_df("analysis/confirmed_trip")

    confirmed_trips_df = pd.concat(ct_df_map.values(), ignore_index=True)

    return confirmed_trips_df
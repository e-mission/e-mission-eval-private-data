'''
This script goes through every confirmed trip for each user and finds the sensed sections for that trip. 
The sections are added to the "data" field of the confirmed trip jsons. 
Input argument: location in the user list that you want to start with.
    The script sometimes fails after running out of memory, so I added this argument to start where I left off.
@author Michael Allen
'''


# from multiprocessing import Pool
import sys
import time
import pandas as pd
from turtle import update

import argparse

sys.path.insert(0,"/Users/mallen2/alternate_branches/eval-compatible-server/e-mission-server") 
#input("Enter your path to the emission server: ") )   # maybe I could try an emission import and then do this as a catch?
import emission.storage.decorations.trip_queries as esdtq
import emission.storage.timeseries.abstract_timeseries as esta
import emission.storage.timeseries.builtin_timeseries as estbt
import emission.core.wrapper.entry as ecwe


# How I ran it:
# source $HOME/OpenPATH_Data/e-mission-server/setup/activate.sh # just to start the emission environment
# cd OpenPATH_Data/e-mission-eval-private-data/Error_bars/
# source $HOME/OpenPATH_Data/e-mission-server/e-mission-py.bash add_sensed_sections_to_confirmed_trips.py

# Get the Stage_analysis_timeseries collection
import emission.core.get_database as edb

def update_trip_with_sections(ct, gis_sensed_modes):
    segments = esdtq.get_sections_for_trip(key = "analysis/inferred_section", user_id = ct["user_id"], trip_id = ct['data']['cleaned_trip'])
    modes = []
    distances = []
    for s in segments:
        # These are sorted in ascending time order.
        # the sensed mode is a number in the database, so I'm relabeling it as a string.
        modes.append(gis_sensed_modes[s['data']['sensed_mode']])
        distances.append(s["data"]["distance"])

    # Add the mode:duration dictionary to the confirmed trip data as sensed_mode.
    ct["data"]["section_modes"], ct["data"]["section_distances"] = modes, distances

    # Update the corresponding confirmed_trip entry in the database.
    estbt.BuiltinTimeSeries.update(ecwe.Entry(ct))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("start_participant", type=int,
                        help="the int location in user_list that you want to start updating from.")
    args = parser.parse_args()
    
    print("Fetching user ids.")
    user_list = []
    # This info is in the Stage_uuids collection of the database
    for u in edb.get_uuid_db().find():         # add users to proper locations in programs 
        uuid = u["uuid"]
        # Append the user to the list.
        user_list.append(uuid)

    gis_sensed_modes = {0 : 'no_sensed',    # UNKNOWN  #NOTE: this is important info to mention.
        1 : 'walking',    # WALKING
        2 : 'bicycling',    # BICYCLING
        3 : 'bus',        # BUS
        4 : 'train',      # TRAIN
        5 : 'car',        # CAR
        6 : 'air_or_hsr', # AIR_OR_HSR
        7 : 'subway',      # SUBWAY
        8 : 'train',      # TRAM
        9 : 'train',      # LIGHT_RAIL
    }

    print("Fetching confirmed trips for each user.")
    t1 = time.time()
    t_first = time.time()
    confirmed_trip_df_map = {}

    # This will likely take 30 to 60 minutes.
    i = args.start_participant
    n_failures = 0
    n_updates = 0
    for u in user_list[args.start_participant:]:
        ts = esta.TimeSeries.get_time_series(u)
        result_it = ts.find_entries(["analysis/confirmed_trip"],time_query = None, geo_query = None,
                    extra_query_list=None)
        # ct_df = ts.get_data_df("analysis/confirmed_trip")
        # get list of sections.
        # get list of trips.
        # if they have the same, then merge.
        for ct in result_it:
            # Only query and update if we need to.
            if not set(['section_modes', 'section_distances']).issubset(ct['data'].keys()):
                try:
                    update_trip_with_sections(ct, gis_sensed_modes)
                    n_updates += 1
                except:
                    print(f"Updating a trip in {i} failed")
                    n_failures += 1
                
        print(f"{i} df found and updated in {time.time()-t_first} seconds.")
        print(f"n_failures: {n_failures}")
        i += 1
    print(f"Fetching and updating {n_updates} trips took {time.time()-t1} seconds.")


    # Below is the error I got with my attempt to parallelize this.  
    # /opt/anaconda3/envs/emission/lib/python3.7/site-packages/pymongo/topology.py:162: 
    # UserWarning: MongoClient opened before fork. Create MongoClient only after forking. 
    # See PyMongo's documentation for details: https://pymongo.readthedocs.io/en/stable/faq.html#is-pymongo-fork-safe

    '''    # For each trip, find its segments and get consolidated label assist labels.
    # update these trips in the database
    print("Updating trip documents")
    t1 = time.time()
    with Pool(8) as p:
        print(p.map(update_trip,confirmed_trips))
        p.close()
        p.terminate()
        p.join()'''
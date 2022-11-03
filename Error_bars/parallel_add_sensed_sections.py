from multiprocessing import Pool
import sys
import time
from turtle import update
sys.path.insert(0,"/Users/mallen2/alternate_branches/eval-compatible-server/e-mission-server") 
#input("Enter your path to the emission server: ") )   # maybe I could try an emission import and then do this as a catch?
import emission.storage.decorations.trip_queries as esdtq
import emission.storage.timeseries.builtin_timeseries as estbt
import emission.core.wrapper.entry as ecwe

# Get the Stage_analysis_timeseries collection
import emission.core.get_database as edb

def get_section_modes_and_distances(ct):

    segments = esdtq.get_sections_for_trip(key = "analysis/inferred_section", user_id = ct["user_id"], trip_id = ct["data"]['cleaned_trip'])

    modes = []
    distances = []
    for s in segments:
        # These are sorted in ascending time order.
        # the sensed mode is a number in the database, so I'm relabeling it as a string.
        modes.append(gis_sensed_modes[s['data']['sensed_mode']])
        distances.append(s["data"]["distance"])
    
    return modes, distances

def update_trip(ct):
    # Add the mode:duration dictionary to the confirmed trip data as sensed_mode.
    ct["data"]["section_modes"], ct["data"]["section_distances"] = get_section_modes_and_distances(ct)

    # Update the corresponding confirmed_trip entry in the database.
    estbt.BuiltinTimeSeries.update(ecwe.Entry(ct))

if __name__ == '__main__':
    Stage_analysis_timeseries = edb.get_analysis_timeseries_db()

    # Get all confirmed trips
    print("Finding all confirmed trips")
    confirmed_trips = [doc for doc in Stage_analysis_timeseries.find({"metadata.key":"analysis/confirmed_trip"})]

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

    print("Updating trip documents")
    t1 = time.time()
    for ct in confirmed_trips:
        update_trip(ct)

    print(f"Updated {len(confirmed_trips)} total trips in {time.time()-t1} seconds.")

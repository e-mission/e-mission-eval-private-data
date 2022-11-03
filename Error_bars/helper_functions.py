import pandas as pd

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
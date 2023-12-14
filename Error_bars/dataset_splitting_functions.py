import pandas as pd
import scipy
import numpy as np
from itertools import chain

# There are multiple crossvalidation splitters in sklearn but all of them split into one training and one test set at a time
# If you get this error: "object of type 'int' has no len()", just lower the number of splits per round
def get_set_splits(df, n_rounds = 50, n_splits_per_round=10):
    '''
    Splits data into n_rounds * n_splits_per_round sets.
    n_splits_per_round controls the size of the resulting data subsets. 
    To get lots of datasets without shrinking the size too much, we use multiple rounds of splits.

    Returns numpy array of arrays of data indices.
    '''
    df = df.copy()
    from numpy.random import default_rng
    large_size_splits = []
    for round in range(n_rounds):
        rng = default_rng()
        trip_index = np.array(df.index.copy())
        rng.shuffle(trip_index)
        # print(energy_consumption_df.index, trip_index)

        # splits is a list of numpy arrays of trip indices
        splits = np.array_split(trip_index, n_splits_per_round)

        large_size_splits.append(splits)
    
    unnested_large_size_splits = list(chain.from_iterable(large_size_splits))
    print(f"Subset lengths: {len(unnested_large_size_splits[0])}. Number of subsets: {len(unnested_large_size_splits)}")
    #print([len(s) for s in large_size_splits])

    return unnested_large_size_splits

def get_split_results(df, splits):
    
    df = df.copy()
    CAR_LIKE_MODES = ['drove_alone', 'shared_ride', 'taxi']
    NON_CAR_MOTORIZED_MODES = ['bus', 'free_shuttle', 'train']
    split_result_list = []
    for s in splits:
        ERROR_COLS = ['error_for_confusion',
           'error_for_prediction', 'expected', 'predicted', 'user_labeled', 'distance_miles', 'distance', 'duration']
        curr_split_trips = df.loc[s]
        curr_split_result = {'count': len(s)}
        for e in ERROR_COLS:
            curr_split_result[e] = curr_split_trips[e].sum()
        curr_split_result['drove_alone_2_shared_ride'] = curr_split_trips.query('mode_confirm == "drove_alone"').distance.sum() / curr_split_trips.query('mode_confirm == "shared_ride"').distance.sum()
        curr_split_result['no_sensed_ratio'] = curr_split_trips.query('primary_mode == "no_sensed"').distance.sum() / curr_split_trips.distance.sum()
        curr_split_result['car_like_ratio'] = curr_split_trips.query('mode_confirm == @CAR_LIKE_MODES').distance.sum() / curr_split_trips.distance.sum()        
        curr_split_result['e_bike_ratio'] = curr_split_trips.query('mode_confirm == "pilot_ebike"').distance.sum() / curr_split_trips.distance.sum()
        curr_split_result['not_a_trip_ratio'] = curr_split_trips.query('mode_confirm == "not_a_trip"').distance.sum() / curr_split_trips.distance.sum()
        
        # car_like_as_not_car: the fraction of car trips that were wrongly labeled as not car. 
        curr_split_result['car_like_as_not_car'] = curr_split_trips.query('mode_confirm == @CAR_LIKE_MODES & primary_mode != "car"').distance.sum() / curr_split_trips.query('mode_confirm == @CAR_LIKE_MODES').distance.sum()
        curr_split_result['e_bike_as_car'] = curr_split_trips.query('mode_confirm == "pilot_ebike" & primary_mode == "car"').distance.sum() / curr_split_trips.query('mode_confirm == "pilot_ebike"').distance.sum()
        curr_split_result['e_bike_as_not_car_bike'] = curr_split_trips.query('mode_confirm == "pilot_ebike" & primary_mode != ["car", "bicycling"]').distance.sum() / curr_split_trips.query('mode_confirm == "pilot_ebike"').distance.sum()

        curr_split_result['non_car_2_car_user_label'] = curr_split_trips.query('mode_confirm == @NON_CAR_MOTORIZED_MODES').distance.sum() / curr_split_trips.query('mode_confirm == @CAR_LIKE_MODES').distance.sum()
        curr_split_result['non_car_2_car_sensed'] = curr_split_trips.query('primary_mode == ["bus", "train"]').distance.sum() / curr_split_trips.query('primary_mode == "car"').distance.sum()
        curr_split_result['mispredicted_as_walk'] = curr_split_trips.query('mode_confirm != "walk" & primary_mode == "walking"').distance.sum() / curr_split_trips.distance.sum()
        curr_split_result['mispredicted_as_car'] = curr_split_trips.query('mode_confirm != @CAR_LIKE_MODES & primary_mode == "car"').distance.sum() / curr_split_trips.distance.sum()
    
        # if curr_split_result['drove_alone_2_shared_ride'] > 0.5:
            # print(f"CHECK: drove_alone %s, shared_ride %s" % (curr_split_trips.query('mode_confirm == "drove_alone"').distance_miles.sum(),
            #                                                   curr_split_trips.query('mode_confirm == "shared_ride"').distance_miles.sum()))
        # print(curr_split_result)
        # print(f"CHECK user_labeled {energy_consumption_df.loc[s].user_labeled.sum()}")
        # print(f"CHECK error_for_confusion {energy_consumption_df.loc[s].error_for_confusion.sum()}")
        split_result_list.append(curr_split_result)
    split_results = pd.DataFrame(split_result_list)
    split_results['error_pct_for_confusion'] = (split_results.error_for_confusion / split_results.user_labeled ) * 100
    split_results['error_pct_for_prediction'] = (split_results.error_for_prediction / split_results.user_labeled) * 100
    return split_results

def find_correlations(split_results, IND_VAR, DEP_VAR):
    '''
    Find each correlation between dep_var and the variables found in ind_var, using the calculated split characteristics found in split results.
    '''

    ind_var_correlation_df = pd.DataFrame(columns=["Independent Variable", "Correlation", "p-value"])
    for iv in IND_VAR:
        corr, p = scipy.stats.pearsonr(split_results[iv],split_results[DEP_VAR])
        ind_var_correlation_df = ind_var_correlation_df.append({"Independent Variable": iv, "Correlation": corr, "p-value": p}, ignore_index=True)
    return ind_var_correlation_df.set_index("Independent Variable")

def get_splits_and_correlations(df, ind_var, dep_var, n_rounds = 50, n_splits_per_round=10):
    df = df.copy()
    splits = get_set_splits(df, n_rounds, n_splits_per_round)
    split_results = get_split_results(df, splits)
    ind_var_correlation_df = find_correlations(split_results, ind_var, dep_var)
    return ind_var_correlation_df
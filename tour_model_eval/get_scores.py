import pandas as pd
import pandas.testing as pdt
import label_processing as label_pro
import sklearn.metrics as skm
import itertools


# compare the trip orders in bin_trips with those in filter_trips above cutoff
def compare_trip_orders(bins,bin_trips,filter_trips):
    bin_trips_ts = pd.DataFrame(data=[trip["data"]["start_ts"] for trip in bin_trips])
    bin_ls = list(itertools.chain(*bins))
    bins_ts = pd.DataFrame(data=[filter_trips[i]["data"]["start_ts"] for i in bin_ls])
    # compare two data frames, the program will continue to score calculation if two data frames are the same
    pdt.assert_frame_equal(bins_ts, bin_trips_ts)


# This function is to get homogeneity score after the first/second round of clustering
# It is based on bin_trips, which are common trips. bin_trips are collected according to the indices in bins
# More info about bin_trips is in similarity.py (delete_bins)
def score(bin_trips, labels_pred):
    bin_trips_user_input_df = pd.DataFrame(data=[trip["data"]["user_input"] for trip in bin_trips])
    bin_trips_user_input_df = label_pro.map_labels(bin_trips_user_input_df)

    # turn all user_input into list without binning
    bin_trips_user_input_ls = bin_trips_user_input_df.values.tolist()
    # drop duplicate user_input
    no_dup_df = bin_trips_user_input_df.drop_duplicates()
    # turn non-duplicate user_input into list
    no_dup_list = no_dup_df.values.tolist()

    # collect labels_true based on user_input
    # To compute labels_true, we need to find out non-duplicate user labels, and use the index of the unique user label
    # to label the whole trips
    # if user labels are [home, work, home, home, work, exercise], the unique label list is [0,1,2],
    # labels_true will be [0,1,0,0,1,2]
    # labels_pred is the flattened list of labels of all common trips, e.g.[1,1,11,12,13,22,23]
    labels_true = []
    for userinput_dict in bin_trips_user_input_ls:
        if userinput_dict in no_dup_list:
            labels_true.append(no_dup_list.index(userinput_dict))

    labels_pred = labels_pred
    homo_score = skm.homogeneity_score(labels_true, labels_pred)
    return homo_score

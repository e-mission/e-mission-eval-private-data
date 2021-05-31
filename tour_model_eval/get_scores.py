import pandas as pd
from pandas.testing import assert_frame_equal
import label_processing as label_pro
from sklearn import metrics
import itertools


# compare the trip orders in bin_trips with those in filter_trips above cutoff
def compare_trip_orders(bins,bin_trips,filter_trips):
    bin_trips_ts = pd.DataFrame(data=[trip["data"]["start_ts"] for trip in bin_trips])
    bin_ls = list(itertools.chain(*bins))
    bins_ts = pd.DataFrame(data=[filter_trips[i]["data"]["start_ts"] for i in bin_ls])
    # compare two data frames, the program will continue to score calculation if two data frames are the same
    assert_frame_equal(bins_ts, bin_trips_ts)


# This function is to get homogeneity score after the first/second round of clustering
# It is based on bin_trips, which are common trips. bin_trips are collected according to the indices of the trips
# in bins above cutoff
# More info about bin_trips is in similarity.py (delete_bins)
# For mapping the labels, in one cluster, we do have trips with user labels.
# There are only two situations - trips are grouped in one cluster or the single trip in one cluster.
# Ideally, we should have labels [11,11,11] for trips in one cluster after the second round.
# In such a cluster, we should already have 1 or 2 trips that have user labels, the third one is predicted.
# Then we label it with the same user labels as the previous two.
# We don't have 1.0 homogeneity score for most of the cluster. It can also be [11,11,12].
# Then for the third one, we will still label it with the same user labels as the previous two
# (since they are in one cluster).
# At this point, we didn't make user_input have same labels for labels_true and labels_pred.
# For example, in the second round, user labels are [("home", "ebike", "bus"),("home", "walk", "bus"),
# ("home", "ebike", "bus")], the labels_pred can be [0,1,0], or [1,0,1] or represented by other numeric labels.
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
    # If user labels are [(purpose, confirmed_mode, replaced_mode)]
    # e.g.,[("home","ebike","bus"),("work","walk","bike"),("home","ebike","bus"),("home","ebike","bus"),
    # ("work","walk","bike"),("exercise","ebike","walk")],
    # the unique label list is [0,1,2], labels_true will be [0,1,0,0,1,2]
    # labels_pred is the flattened list of labels of all common trips, e.g.[1,1,11,12,13,22,23]
    labels_true = []
    for trip in bin_trips_user_input_ls:
        if trip in no_dup_list:
            labels_true.append(no_dup_list.index(trip))

    labels_pred = labels_pred
    homo_score = metrics.homogeneity_score(labels_true, labels_pred)
    return homo_score

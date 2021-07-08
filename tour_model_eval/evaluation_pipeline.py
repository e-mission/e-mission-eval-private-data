import emission.analysis.modelling.tour_model.similarity as similarity
import numpy as np
import emission.analysis.modelling.tour_model.get_request_percentage as grp
import emission.analysis.modelling.tour_model.get_scores as gs
import emission.analysis.modelling.tour_model.label_processing as lp
import emission.analysis.modelling.tour_model.data_preprocessing as preprocess

def second_round(first_label_set,first_labels,bin_trips,filter_trips,low,dist_pct,sim,new_labels,track):
    for l in first_label_set:
        # store second round trips data
        second_round_trips = []
        # create a track to store indices and labels for the second round
        second_round_idx_labels = []
        for index, first_label in enumerate(first_labels):
            if first_label == l:
                second_round_trips.append(bin_trips[index])
                second_round_idx_labels.append([index, first_label])
        x = preprocess.extract_features(second_round_trips)

        # We choose single-linkage clustering.
        # See examples and explanations at https://en.wikipedia.org/wiki/Single-linkage_clustering
        # It is based on grouping clusters in bottom-up fashion (agglomerative clustering),
        # at each step combining two clusters that contain the closest pair of elements not yet belonging
        # to the same cluster as each other.
        method = 'single'
        # get the second label from the second round of clustering using hierarchical clustering
        second_labels = lp.get_second_labels(x, method, low, dist_pct)
        # concatenate the first label (label from the first round) and the second label (label
        # from the second round) (e.g.first label[1,1,1], second label[1,2,3], new_labels is [11,12,13]
        new_labels = lp.get_new_labels(second_labels, second_round_idx_labels, new_labels)
        # change the labels in track with new_labels
        track = lp.change_track_labels(track, new_labels)

    # get request percentage for the subset for the second round
    percentage_second = grp.get_req_pct(new_labels, track, filter_trips, sim)

    # get homogeneity score for the second round
    homo_second = gs.score(bin_trips, new_labels)
    return percentage_second,homo_second


# we use functions in similarity to build the first round of clustering
def first_round(data,radius):
    sim = similarity.similarity(data, radius)
    filter_trips = sim.data
    sim.bin_data()
    sim.delete_bins()
    bins = sim.bins
    bin_trips = sim.newdata
    return sim, bins, bin_trips, filter_trips


def get_first_label(bins):
    # get first round labels
    # the labels from the first round are the indices of bins
    # e.g. in bin 0 [trip1, trip2, trip3], the labels of this bin is [0,0,0]
    first_labels = []
    for b in range(len(bins)):
        for trip in bins[b]:
            first_labels.append(b)
    return first_labels


def get_track(bins, first_labels):
    # create a list idx_labels_track to store indices and labels
    # the indices of the items will be the same in the new label list after the second round clustering
    # item[0] is the original index of the trip in filter_trips
    # item[1] is the label from the first round of clustering
    idx_labels_track = []
    for bin in bins:
        for ori_idx in bin:
            idx_labels_track.append([ori_idx])
    # store first round labels in idx_labels_track list
    for i in range(len(first_labels)):
        idx_labels_track[i].append(first_labels[i])

    return idx_labels_track


def tune(data,radius):
    sim, bins, bin_trips, filter_trips = first_round(data, radius)
    # it is possible that we don't have common trips for tuning or testing
    # bins contain common trips indices
    if len(bins) is not 0:
        gs.compare_trip_orders(bins, bin_trips, filter_trips)
        first_labels = get_first_label(bins)
        # new_labels temporary stores the labels from the first round, but later the labels in new_labels will be
        # updated with the labels after two rounds of clustering.
        new_labels = first_labels.copy()
        first_label_set = list(set(first_labels))
        track = get_track(bins, first_labels)
        # collect tuning scores and parameters
        tune_score = {}
        for dist_pct in np.arange(0.15, 0.6, 0.02):
            for low in range(250, 600):
                percentage_second, homo_second = second_round(first_label_set, first_labels, bin_trips, filter_trips,
                                                              low, dist_pct,
                                                              sim, new_labels, track)

                curr_score = gs.get_score(homo_second, percentage_second)
                if curr_score not in tune_score:
                    tune_score[curr_score] = (low, dist_pct, homo_second, percentage_second)

        best_score = max(tune_score)
        sel_tradeoffs = tune_score[best_score][0:2]
    else:
        sel_tradeoffs = (0,0)

    return sel_tradeoffs


def test(data,radius,low,dist_pct):
    sim, bins, bin_trips, filter_trips = first_round(data, radius)
    # it is possible that we don't have common trips for tuning or testing
    # bins contain common trips indices
    if len(bins) is not 0:
        gs.compare_trip_orders(bins, bin_trips, filter_trips)
        first_labels = get_first_label(bins)
        # new_labels temporary stores the labels from the first round, but later the labels in new_labels will be
        # updated with the labels after two rounds of clustering.
        new_labels = first_labels.copy()
        first_label_set = list(set(first_labels))
        track = get_track(bins, first_labels)
        # get request percentage for the subset for the first round
        percentage_first = grp.get_req_pct(new_labels, track, filter_trips, sim)
        # get homogeneity score for the subset for the first round
        homo_first = gs.score(bin_trips, first_labels)
        percentage_second, homo_second = second_round(first_label_set, first_labels, bin_trips, filter_trips, low,
                                                      dist_pct, sim, new_labels, track)
    else:
        percentage_first = 1
        homo_first = 1
        percentage_second = 1
        homo_second = 1
    scores = gs.get_score(homo_second, percentage_second)
    return homo_first,percentage_first,homo_second,percentage_second,scores


def main(uuid = None):
    user = uuid
    radius = 100
    trips = preprocess.read_data(user)
    filter_trips = preprocess.filter_data(trips, radius)
    tune_idx, test_idx = preprocess.split_data(filter_trips)
    tune_data = preprocess.get_subdata(filter_trips, test_idx)
    test_data = preprocess.get_subdata(filter_trips, tune_idx)
    pct_collect_first = []
    homo_collect_first = []
    pct_collect_second = []
    homo_collect_second = []
    coll_score = []
    coll_tradeoffs = []

    # tune data
    for j in range(len(tune_data)):
        tuning_parameters = tune(tune_data[j], radius)
        coll_tradeoffs.append(tuning_parameters)

    # testing
    for k in range(len(test_data)):
        tradoffs = coll_tradeoffs[k]
        low = tradoffs[0]
        dist_pct = tradoffs[1]
        homo_first, percentage_first, homo_second, percentage_second, scores = test(tune_data[k],radius,low,dist_pct)
        pct_collect_first.append(percentage_first)
        homo_collect_first.append(homo_first)
        pct_collect_second.append(percentage_second)
        homo_collect_second.append(homo_second)
        coll_score.append(scores)


if __name__ == '__main__':
    main(uuid=None)

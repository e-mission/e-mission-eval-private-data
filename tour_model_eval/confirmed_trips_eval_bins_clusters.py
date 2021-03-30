import logging

# Our imports
import emission.core.get_database as edb
import emission.analysis.modelling.tour_model.cluster_pipeline as pipeline
import emission.analysis.modelling.tour_model.similarity as similarity
import emission.analysis.modelling.tour_model.featurization as featurization
import emission.analysis.modelling.tour_model.representatives as representatives
import emission.storage.decorations.analysis_timeseries_queries as esda
import pandas as pd
from numpy import *
from sklearn import metrics
from pandas.testing import assert_frame_equal


def get_user_ls(all_users,radius):
    user_ls = []
    valid_user_ls = []
    for i in range(len(all_users)):
        curr_user = 'user' + str(i + 1)
        user = all_users[i]
        filter_trips,sim,trips = filter_data(user,radius)
        if valid_user(filter_trips,trips):
            valid_user_ls.append(curr_user)
            user_ls.append(curr_user)
        else:
            user_ls.append(curr_user)
            continue
    return user_ls,valid_user_ls


def filter_data(user,radius):
    trips = pipeline.read_data(uuid=user, key=esda.CONFIRMED_TRIP_KEY)
    non_empty_trips = [t for t in trips if t["data"]["user_input"] != {}]
    non_empty_trips_df = pd.DataFrame(t["data"]["user_input"] for t in non_empty_trips)
    valid_trips_df = non_empty_trips_df.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)
    valid_trips_idx_ls = valid_trips_df.index.tolist()
    valid_trips = [non_empty_trips[i]for i in valid_trips_idx_ls]

    sim = similarity.similarity(valid_trips, radius)
    filter_trips = sim.data
    return filter_trips,sim,trips

def valid_user(filter_trips,trips):
    valid = False
    if len(filter_trips) >= 10 and len(filter_trips) / len(trips) >= 0.5:
        valid = True
    return valid

def map_labels(user_input_df,sp2en,cvt_pur_mo):
    # Spanish words to English
    span_eng_dict = {'revisado_bike': 'test ride with bike', 'placas_de carro': 'car plates', 'aseguranza': 'insurance',
                     'iglesia': 'church', 'curso': 'course',
                     'mi_hija recién aliviada': 'my daughter just had a new baby',
                     'servicio_comunitario': 'community service', 'pago_de aseguranza': 'insurance payment',
                     'grupo_comunitario': 'community group', 'caminata_comunitaria': 'community walk'}

    # Convert purpose
    map_pur_dict = {'course': 'school', 'work_- lunch break': 'lunch_break', 'on_the way home': 'home',
                    'insurance_payment': 'insurance'}

    if sp2en:
        # change language
        user_input_df = user_input_df.replace(span_eng_dict)
    elif cvt_pur_mo:
        # change language first
        user_input_df = user_input_df.replace(span_eng_dict)
        # convert purpose
        user_input_df = user_input_df.replace(map_pur_dict)
        # convert mode
        for a in range(len(user_input_df)):
            if user_input_df.iloc[a]["replaced_mode"] == "same_mode":
                # to see which row will be converted
                logging.debug("The following rows will be changed: %s", user_input_df.iloc[a])
                user_input_df.iloc[a]["replaced_mode"] = user_input_df.iloc[a]['mode_confirm']
    return user_input_df


def valid_user_check(filter_trips,trips,homo_score,comp_score,v_score):
    if not valid_user(filter_trips, trips):
        homo_score.append(NaN)
        comp_score.append(NaN)
        v_score.append(NaN)
        skip = True
    else:
        skip = False
    return homo_score,comp_score,v_score,skip


def compute_score(labels_true,labels_pred,homo_score,comp_score,v_score):
    homo = metrics.homogeneity_score(labels_true, labels_pred)
    homo_score.append(float('%.3f' % homo))
    comp = metrics.completeness_score(labels_true, labels_pred)
    comp_score.append(float('%.3f' % comp))
    v = metrics.v_measure_score(labels_true, labels_pred)
    v_score.append(float('%.3f' % v))
    return homo_score,comp_score,v_score


# v_measure_bins takes 5 parameters
# - sp2en=True: change Spanish to English
# - cvt_pur_mo=True: convert purposes and replaced mode
# - cutoff=True: choose to analyze bins above cutoff
# - cutoff=None: analyze all bins
# Note: for sp2en and cvt_pur_mo, set either one to be True as needed. cvt_pur_mo will change language first
def v_measure_bins(all_users,radius,sp2en=None,cvt_pur_mo=None,cutoff=None):
    homo_score = []
    comp_score = []
    v_score = []
    for i in range(len(all_users)):
        user = all_users[i]
        filter_trips,sim,trips = filter_data(user,radius)

        homo_score,comp_score,v_score,skip = valid_user_check(filter_trips,trips,homo_score,comp_score,v_score)
        if skip:
            continue

        sim.bin_data()
        if cutoff is None:
            trip_index_ls = []
            bins = sim.bins
            for bin in bins:
                for index in bin:
                    trip_index_ls.append(index)
            bin_trips = [filter_trips[num] for num in trip_index_ls]

        elif cutoff:
            sim.delete_bins()
            bin_trips = sim.newdata
            bins = sim.bins

        bin_trips_user_input_df = pd.DataFrame(data=[trip["data"]["user_input"] for trip in bin_trips])
        bin_trips_user_input_df = map_labels(bin_trips_user_input_df, sp2en, cvt_pur_mo)

        # turn all user_input into list without binning
        bin_trips_user_input_ls = bin_trips_user_input_df.values.tolist()
        # drop duplicate user_input
        no_dup_df = bin_trips_user_input_df.drop_duplicates()
        # turn non-duplicate user_input into list
        no_dup_list = no_dup_df.values.tolist()

        # collect labels_true based on user_input
        labels_true = []
        for trip in bin_trips_user_input_ls:
            if trip in no_dup_list:
                labels_true.append(no_dup_list.index(trip))

        # collect labels_pred based on bins
        labels_pred = []
        for b in range(len(bins)):
            for trip in bins[b]:
                labels_pred.append(b)

        # compare the trips order in bins and those in valid_trips using timestamp
        bin_trips_ts = pd.DataFrame(data=[trip["data"]["start_ts"] for trip in bin_trips])
        bin_ls = []
        for bin in bins:
            for index in bin:
                bin_ls.append(index)
        bins_ts = pd.DataFrame(data=[filter_trips[i]["data"]["start_ts"] for i in bin_ls])
        # compare two data frames, the program will continue to score calculation if two data frames are the same
        assert_frame_equal(bins_ts, bin_trips_ts)
        homo_score, comp_score, v_score = compute_score(labels_true, labels_pred, homo_score, comp_score, v_score)

    return homo_score, comp_score, v_score


# - sp2en=True: change Spanish to English
# - cvt_pur_mo=True: convert purposes and replaced mode
# - cutoff=True: choose to analyze bins above cutoff
# - cutoff=None: analyze all bins
# Note: for sp2en and cvt_pur_mo, set either one to be True as needed. cvt_pur_mo will change language first
def v_measure_clusters(all_users,radius,sp2en=None,cvt_pur_mo=None):
    homo_score = []
    comp_score = []
    v_score = []
    for i in range(len(all_users)):
        user = all_users[i]
        filter_trips,sim,trips = filter_data(user,radius)

        homo_score,comp_score,v_score,skip = valid_user_check(filter_trips,trips,homo_score,comp_score,v_score)
        if skip:
            continue

        sim.bin_data()
        sim.delete_bins()
        bin_trips = sim.newdata
        bins = sim.bins

        # clustering the data only based on sil score (min_cluster = 0) instead of bins number (len(bins))
        feat = featurization.featurization(bin_trips)
        min = 0
        max = int(math.ceil(1.5 * len(bins)))
        feat.cluster(min_clusters=min, max_clusters=max)
        cluster_trips = feat.data
        cluster_user_input_df = pd.DataFrame(data=[i["data"]["user_input"] for i in cluster_trips])
        cluster_user_input_df = map_labels(cluster_user_input_df, sp2en, cvt_pur_mo)
        # turn cluster_trips to list without any changes
        cluster_user_input_ls = cluster_user_input_df.values.tolist()
        # drop duplicate user_input
        no_dup_df = cluster_user_input_df.drop_duplicates()
        # turn non-duplicate user_input into list
        no_dup_list = no_dup_df.values.tolist()
        # collect labels_true based on user_input
        labels_true = []
        for trip in cluster_user_input_ls:
            if trip in no_dup_list:
                labels_true.append(no_dup_list.index(trip))
        labels_pred = feat.labels

        # compare the points in cluster_trips and those in feat.points, the program will continue to score calculation
        # if the frames are the same
        cluster_ps = []
        for trip in cluster_trips:
            cluster_ps.append([trip["data"]["start_loc"]["coordinates"][0],
                               trip["data"]["start_loc"]["coordinates"][1],
                               trip["data"]["end_loc"]["coordinates"][0],
                               trip["data"]["end_loc"]["coordinates"][1]])
        cluster_ps_df = pd.DataFrame(data=cluster_ps)
        label_ps_df = pd.DataFrame(data=feat.points)
        assert_frame_equal(cluster_ps_df, label_ps_df)
        homo_score, comp_score, v_score = compute_score(labels_true, labels_pred, homo_score, comp_score, v_score)

    return homo_score, comp_score, v_score





















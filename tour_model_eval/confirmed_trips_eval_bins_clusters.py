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


# - user_ls: a list of all users
# - valid_user_ls: a list of valid users
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


# - trips: all trips read from database
# - filter_trips: valid trips that have user labels and are not points
def filter_data(user,radius):
    trips = pipeline.read_data(uuid=user, key=esda.CONFIRMED_TRIP_KEY)
    non_empty_trips = [t for t in trips if t["data"]["user_input"] != {}]
    non_empty_trips_df = pd.DataFrame(t["data"]["user_input"] for t in non_empty_trips)
    valid_trips_df = non_empty_trips_df.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)
    valid_trips_idx_ls = valid_trips_df.index.tolist()
    valid_trips = [non_empty_trips[i]for i in valid_trips_idx_ls]

    # similarity codes can filter out trips that are points in valid_trips
    sim = similarity.similarity(valid_trips, radius)
    filter_trips = sim.data
    return filter_trips,sim,trips


# to determine if the user is valid:
# valid user should have >= 10 trips for further analysis and the proportion of filter_trips is >=50%
def valid_user(filter_trips,trips):
    valid = False
    if len(filter_trips) >= 10 and len(filter_trips) / len(trips) >= 0.5:
        valid = True
    return valid


# to map the user labels
# - user_input_df: pass in original user input dataframe, return changed user input dataframe
# - sp2en: change Spanish to English
# - cvt_pur_mo: convert purposes and replaced mode
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


# check if the user is valid
# append NaN to the score lists when the user invalid
def valid_user_check(filter_trips,trips,homo_score,comp_score,v_score):
    if not valid_user(filter_trips, trips):
        homo_score.append(NaN)
        comp_score.append(NaN)
        v_score.append(NaN)
        skip = True
    else:
        skip = False
    return homo_score,comp_score,v_score,skip


#  This function is to get homogeneity score, complete score, and v-score
def compute_score(labels_true,labels_pred,homo_score,comp_score,v_score):
    homo = metrics.homogeneity_score(labels_true, labels_pred)
    homo_score.append(float('%.3f' % homo))
    comp = metrics.completeness_score(labels_true, labels_pred)
    comp_score.append(float('%.3f' % comp))
    v = metrics.v_measure_score(labels_true, labels_pred)
    v_score.append(float('%.3f' % v))
    return homo_score,comp_score,v_score


# This function is to compare a trip with a group of trips to see if they happened in a same day
def match_day(trip,bin,filter_trips):
    if bin:
        t = filter_trips[bin[0]]
        if trip['data']['start_local_dt']['year']==t['data']['start_local_dt']['year']\
                and trip['data']['start_local_dt']['month']==t['data']['start_local_dt']['month']\
                and trip['data']['start_local_dt']['day']==t['data']['start_local_dt']['day']:
            return True
    return False


# This function is to compare a trip with a group of trips to see if they happened in a same month
def match_month(trip,bin,filter_trips):
    if bin:
        t = filter_trips[bin[0]]
        if trip['data']['start_local_dt']['year']==t['data']['start_local_dt']['year']\
                and trip['data']['start_local_dt']['month']==t['data']['start_local_dt']['month']:
            return True
    return False


# This function bins trips according to ['start_local_dt']
def bin_date(trip_ls,filter_trips,day=None,month=None):
    bin_date = []
    for trip_index in trip_ls:
        added = False
        trip = filter_trips[trip_index]

        for bin in bin_date:
            if day:
                if match_day(trip,bin,filter_trips):
                    bin.append(trip_index)
                    added = True
                    break
            if month:
                if match_month(trip,bin,filter_trips):
                    bin.append(trip_index)
                    added = True
                    break

        if not added:
            bin_date.append([trip_index])

    return bin_date


# compare the trip orders in bin_trips with those in filter_trips above cutoff
def compare_trip_orders(bins,bin_trips,filter_trips):
    # compare the trips order in bins and those in valid_trips using timestamp
    bin_trips_ts = pd.DataFrame(data=[trip["data"]["start_ts"] for trip in bin_trips])
    bin_ls = []
    for bin in bins:
        for index in bin:
            bin_ls.append(index)
    bins_ts = pd.DataFrame(data=[filter_trips[i]["data"]["start_ts"] for i in bin_ls])
    # compare two data frames, the program will continue to score calculation if two data frames are the same
    assert_frame_equal(bins_ts, bin_trips_ts)


def find_first_trip(filter_trips,bin):
    early_trip = filter_trips[bin[0]]
    index = 0
    for i in range(1,len(bin)):
        compare_trip = filter_trips[bin[i]]
        if early_trip['data']["start_ts"] > compare_trip['data']["start_ts"]:
            early_trip = compare_trip
            index = i
    early_trip_index = bin[index]
    return early_trip_index, index






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





















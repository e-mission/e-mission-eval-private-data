import logging
from scipy.cluster.hierarchy import linkage, dendrogram,fcluster



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


# use hierarchical clustering to get labels of the second round
def get_second_labels(x,method,low,dist_pct):
    z = linkage(x, method=method, metric='euclidean')
    last_d = z[-1][2]
    clusters = []
    if last_d < low:
        for i in range(len(x)):
            clusters.append(0)
    else:
        max_d = last_d * dist_pct
        clusters = fcluster(z, max_d, criterion='distance')
    return clusters


# this function includes hierarchical clustering and changing labels to get appropriate labels for
# the second round of clustering
# appropriate labels are label from the first round concatenate label from the second round
# (e.g. label from first round is 1, label from second round is 2, the new label will be 12)
def get_new_labels(x,low,dist_pct,second_round_idx_labels,new_labels,method=None):
    idx_label = second_round_idx_labels.copy()
    second_labels = get_second_labels(x,method,low,dist_pct)
    for i in range(len(second_labels)):
        index = idx_label[i][0]
        new_label = idx_label[i][1]
        # concatenate labels from two rounds
        new_label = int(str(new_label) + str(second_labels[i]))
        for k in range(len(new_labels)):
            if k == index:
                new_labels[k] = new_label
                break
    return new_labels


# group similar trips according to new_labels, store the original indices of the trips
def group_similar_trips(new_labels,track):
    bin_sim_trips = []
    for trip_index,label in enumerate(new_labels):
        added = False
        for bin in bin_sim_trips:
            if label == new_labels[bin[0]]:
                bin.append(trip_index)
                added = True
                break
        if not added:
            bin_sim_trips.append([trip_index])
    # using track to replace the current indices with original indicies
    for bin in bin_sim_trips:
        for i in range(len(bin)):
            bin[i] = track[bin[i]][0]
    return bin_sim_trips


# replace the first round labels with new labels
def change_track_labels(track,new_labels):
    for i in range(len(new_labels)):
        track[i][1] = new_labels[i]
    return track




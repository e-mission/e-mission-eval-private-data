import label_processing as label_pro

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


# collect requested trips and common trips(no need to request) indices above cutoff
def requested_trips_ab_cutoff(new_bins, filter_trips):
    # collect requested trip indices above cutoff
    ab_trip_ls = []
    # collect common trip indices above cutoff
    no_req_trip_ls = []
    for bin in new_bins:
        early_trip_index, index = find_first_trip(filter_trips, bin)
        ab_trip_ls.append(early_trip_index)

        for k in range(len(bin)):
            if k != index:
                no_req_trip_idx = bin[k]
                no_req_trip_ls.append(no_req_trip_idx)
    return ab_trip_ls, no_req_trip_ls


# collect requested trips indices below cutoff
def requested_trips_bl_cutoff(sim):
    # bins below cutoff
    bl_bins = sim.below_cutoff

    # collect requested trips indices below cutoff
    bl_trip_ls = []
    for bin in bl_bins:
        for trip_index in bin:
            bl_trip_ls.append(trip_index)
    return bl_trip_ls


# a list of all requested trips indices
def get_requested_trips(new_bins,filter_trips,sim):
    ab_trip_ls,no_req_trip_ls = requested_trips_ab_cutoff(new_bins,filter_trips)
    bl_trip_ls = requested_trips_bl_cutoff(sim)
    req_trips_ls=ab_trip_ls+bl_trip_ls
    return req_trips_ls


# get request percentage based on the number of requested trips and the total number of trips
def get_req_pct(new_labels,track,filter_trips):
    # - new_bins: bins with original indices of similar trips
    new_bins = label_pro.group_similar_trips(new_labels,track)
    req_trips = get_requested_trips(new_bins,filter_trips,sim)
    pct = len(req_trips)/len(filter_trips)
    return pct

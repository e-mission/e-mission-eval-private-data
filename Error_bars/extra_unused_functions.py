import numpy as np
import helper_functions

import sys
sys.path.append('/path/to/e-mission-server')
import emission.core.wrapper.user as ecwu

def lagged_auto_cov(Xi,k):
    """
    For series of values x_i, length N, compute the empirical auto-covariance with lag k
    defined as: 1/(N-1) * \sum_{i=0}^{N-t} ( x_i - x_s ) * ( x_{i+t} - x_s )
    From https://stackoverflow.com/questions/20110590/how-to-calculate-auto-covariance-in-python, by RGWinston.

    Returns auto_cov (float), the autocovariance of the series at lag k.
    """
    N = len(Xi)
    if k >= N: return 0

    # use sample mean estimate from whole series
    Xs = np.mean(Xi)

    # construct copies of series shifted relative to each other, 
    # with mean subtracted from values
    end_padded_series = np.zeros(N+k)
    end_padded_series[:N] = Xi - Xs
    start_padded_series = np.zeros(N+k)
    start_padded_series[k:] = Xi - Xs

    auto_cov = 1./(N-1) * np.dot( start_padded_series, end_padded_series )
    return auto_cov

def get_totals_and_errors_with_temporal_autocovariance(df, os_EI_moments_map, unit_dist_MCS_df, include_autocovariance = False):
    # df in this case should be an energy consumption dataframe for one user.

    expected, predicted, actual = sum(df['expected']), sum(df['predicted']), sum(df['user_labeled'])
    n_trips = len(df)

    cov_sum = 0
    if include_autocovariance == True:
        for u in df.user_id.unique():
            auto_cov = df[df.user_id == u].auto_cov.iloc[0] # switch to just passing in a user -> autocov map?
            n_trips = len(df[df.user_id == u])
            if n_trips >= 50: # not calculating autocov if we do not have a large timeseries sample
                coeffs = 2*np.array([n_trips - 1, n_trips - 2])
                cov_sum += np.dot(coeffs,auto_cov)
            # used median values of the autocovariance across users that I found. Maybe I should just ignore autocovariance for small sets of trips.
            elif n_trips > 2:
                cov_sum += (n_trips - 1)*2*5.5 + (n_trips - 2)*2*0.95

    #print(f"sum of the autocov terms for this dataframe: {cov_sum}")
    aggregate_variance = df.confusion_var.sum() + cov_sum
    # Calculate aggregate standard deviation. Including autocovariance is only appropriate for trips from the same user.
    if aggregate_variance < 0: print(df.confusion_var.sum(), cov_sum)
    final_variance = aggregate_variance if aggregate_variance > 0 else df.confusion_var.sum()

    sd = np.sqrt(final_variance)
    error_for_expected, error_for_predicted = helper_functions.relative_error(expected,actual)*100, helper_functions.relative_error(predicted,actual)*100

    signed_error = expected - actual
    error_over_sd = abs(signed_error/sd)

    return {"total_expected": expected, "total_user_labeled": actual, 
        "total_predicted": predicted, 
        "aggregate_sd": sd, 
        "user_sd": np.sqrt(df.user_var.sum()),
        "aggregate_var": final_variance,
        "percent_error_for_expected": error_for_expected, 
        "percent_error_for_predicted": error_for_predicted,
        "signed_error": signed_error,
        "error_over_sd": error_over_sd,
        "autocov_sum": cov_sum}


#######
###### Aggregate EC and related functions.
def investigate_sensing_for_one_user_label(expanded_labeled_trips_df, label):
    one_label = expanded_labeled_trips_df[expanded_labeled_trips_df['mode_confirm'] == label]

    print(get_aggregate_EC(one_label,True, unit_dist_MCS_df, android_EI_moments_df, ios_EI_moments_df, gis_sensed_modes,energy_dict))
    label_sensed = []
    label_dist = []
    for _,ct in one_label.iterrows():
        label_sensed += [m for m in ct['section_modes']]
        label_dist += [d for d in ct['section_distances']]
    return pd.DataFrame({"mode": label_sensed, "dist": label_dist})

def get_aggregate_EC(trips_df, only_sensing, unit_dist_MCS_df, android_EI_moments, ios_EI_moments,energy_dict):

    mean_EC_agg = 0    # aggregate energy consumption
    var_EC_agg = 0

    for  _,ct in trips_df.iterrows():
        # Get operating system
        u = ecwu.User(ct.user_id)
        os = u.getProfile()["curr_platform"]

        # Get OS specific trip length info.
        mean_for_unit_L = unit_dist_MCS_df[os]["mean"]
        var_for_unit_L = unit_dist_MCS_df[os]["var"]


        # Get trip mode info.
        # Later the condition will be whether the model chosen is sensing.
        if only_sensing == True:
            # Get segments for the trip.
            n_sections = len(ct["section_modes"])
            section_modes = ct["section_modes"]
            sections_lengths = np.array(ct["section_distances"])*METERS_TO_MILES   # 1 meter = 0.000621371 miles

            mean_L = sections_lengths*mean_for_unit_L
            var_L = sections_lengths**2 * var_for_unit_L
            
            
            for s in range(0,n_sections):
                # EI mean and variance.
                # Perhaps it would be better to keep the moments in the same file?

                if section_modes[s] == "air_or_hsr": continue

                # Later: switch to a map style function.
                mean_EI, var_EI = get_EI_moments_for_trip(section_modes[s],os,android_EI_moments,ios_EI_moments)

                # Propagate variance for the trip
                mean_EC = mean_L[s]*mean_EI
                var_EC = var_EI*mean_L[s]**2 + var_L[s]*mean_EI**2

                # Add to total - follows from assumed independence of section errors.
                mean_EC_agg += mean_EC
                var_EC_agg += var_EC
        
        # use user labels.
        else:
            mode = ct["mode_confirm"]  # need to make sure you convert it to an appropriate energy intensity.

            if mode not in MODE_MAPPING_DICT or mode == np.nan: continue
            if MODE_MAPPING_DICT[mode] == "Air": continue
            EI = energy_dict[MODE_MAPPING_DICT[mode]]

            length = ct["distance"]*METERS_TO_MILES
            mean_L = length* mean_for_unit_L  
            var_L = length**2 * var_for_unit_L

            mean_EC_agg += EI*mean_L
            var_EC_agg +=  EI*var_L

    #print(air_count)
    return mean_EC_agg, var_EC_agg

def get_primary_mode_aggregate_EC(trips_df, only_sensing, unit_dist_MCS_df, 
    android_EI_moments, ios_EI_moments,energy_dict, using_predictions, only_primary_section):

    mean_EC_agg = 0    # aggregate energy consumption
    var_EC_agg = 0
    primary_mode_dict = {}

    for  _,ct in trips_df.iterrows():
        # Get operating system
        u = ecwu.User(ct.user_id)
        os = u.getProfile()["curr_platform"]

        # Get OS specific trip length info.
        mean_for_unit_L = unit_dist_MCS_df[os]["mean"]
        var_for_unit_L = unit_dist_MCS_df[os]["var"]

        # Get primary mode
        longest_section = max(ct["section_distances"])
        primary_mode = ct["section_modes"][ct["section_distances"]==longest_section]

        # in case there are ever tied longest sections.
        # pick the most energy intensive mode.
        if isinstance(primary_mode,list): 
            mini_energy_dict = {x:energy_dict[MODE_MAPPING_DICT[x]] for x in primary_mode}
            primary_mode = max(mini_energy_dict, key=mini_energy_dict.get)

        if primary_mode == 'air_or_hsr': continue

        # Get the trip length or primary section length.
        if only_primary_section == True:
            trip_length = longest_section*METERS_TO_MILES
        else:
            trip_length = ct['distance']*METERS_TO_MILES   # 1 meter = 0.000621371 miles

        # Get trip mode info.
        # Later the condition will be whether the model chosen is sensing.
        if only_sensing == True:

            # save trip length sums for each mode
            if primary_mode not in primary_mode_dict: primary_mode_dict[primary_mode] = 0
            primary_mode_dict[primary_mode] +=  trip_length

            mean_L = trip_length*mean_for_unit_L
            var_L = trip_length**2 * var_for_unit_L




            if using_predictions:
                mean_EI = energy_dict[MODE_MAPPING_DICT[primary_mode]]
                var_EI = 0
            else:
                # Later: switch to a map style function.
                mean_EI, var_EI = get_EI_moments_for_trip(primary_mode,os,android_EI_moments,ios_EI_moments)

            # Propagate variance for the trip
            mean_EC = mean_L*mean_EI
            var_EC = var_EI*mean_L**2 + var_L*mean_EI**2

            # Add to total - follows from assumed independence of section errors.
            mean_EC_agg += mean_EC
            var_EC_agg += var_EC
        
        # use user labels.
        else:
            mode = ct["mode_confirm"]  # need to make sure you convert it to an appropriate energy intensity.

            if mode not in MODE_MAPPING_DICT or mode == np.nan: continue
            if MODE_MAPPING_DICT[mode] == "Air": continue
            EI = energy_dict[MODE_MAPPING_DICT[mode]]

            length = trip_length
            mean_L = length* mean_for_unit_L  
            var_L = length**2 * var_for_unit_L

            mean_EC_agg += EI*mean_L
            var_EC_agg +=  EI*var_L

    #print(air_count)
    return mean_EC_agg, var_EC_agg, primary_mode_dict

def get_aggregate_EC_with_extras(trips_df, only_sensing, unit_dist_MCS_df, 
                                android_EI_moments, 
                                ios_EI_moments,
                                energy_dict, use_naive_sensing_prediction=False, car_load_factor=1):
    # requires the trips dataframe to have expanded labeled trips

    # The load factor here only updates the predicted mode EI, not the confusion EI. See store_errors.ipynb to save EI_moments dataframes with different load factors.
    drove_alone_EI = energy_dict["Gas Car, drove alone"]
    energy_dict.update({"Car, sensed": drove_alone_EI/car_load_factor})

    mean_EC_agg = 0    # aggregate energy consumption
    var_EC_agg = 0  
    sum_sensed_mean_EI = 0
    N_sections = 0

    sum_labeled_mean_EI = 0
    n_trips = 0

    ios_count = 0
    android_count = 0

    for  _,ct in trips_df.iterrows():
        # Get operating system
        u = ecwu.User(ct.user_id)
        os = u.getProfile()["curr_platform"]

        if os == "ios": 
            ios_count+=1 
        else: 
            android_count += 1

        # Get OS specific trip length info.
        mean_for_unit_L = unit_dist_MCS_df[os]["mean"]
        var_for_unit_L = unit_dist_MCS_df[os]["var"]


        # Get trip mode info.
        # Later the condition will be whether the model chosen is sensing.
        if only_sensing == True:
            # Get segments for the trip.
            n_sections = len(ct["section_modes"])
            section_modes = ct["section_modes"]
            sections_lengths = np.array(ct["section_distances"])*METERS_TO_MILES   # 1 meter = 0.000621371 miles

            mean_L = sections_lengths*mean_for_unit_L
            var_L = sections_lengths**2 * var_for_unit_L
            
            for s in range(0,n_sections):
                # EI mean and variance.
                # Perhaps it would be better to keep the moments in the same file?

                if section_modes[s] == "air_or_hsr": continue

                if use_naive_sensing_prediction:
                    #mean_EI = energy_dict[MODE_MAPPING_DICT[section_modes[s]]]
                    if section_modes[s] == 'car':
                        mean_EI = energy_dict['Car, sensed']
                    else:
                        mean_EI = energy_dict[MODE_MAPPING_DICT[section_modes[s]]]
                    var_EI = 0
                else:
                    # Later: switch to a map style function.
                    mean_EI, var_EI = get_EI_moments_for_trip(section_modes[s],os,android_EI_moments,ios_EI_moments)

                sum_sensed_mean_EI += mean_EI
                N_sections += 1

                # Propagate variance for the trip
                mean_EC = mean_L[s]*mean_EI
                var_EC = var_EI*mean_L[s]**2 + var_L[s]*mean_EI**2

                # Add to total - follows from assumed independence of section errors.
                mean_EC_agg += mean_EC
                var_EC_agg += var_EC
        
        # use user labels.
        else:
            mode = ct["mode_confirm"]  # need to make sure you convert it to an appropriate energy intensity.

            if mode not in MODE_MAPPING_DICT or mode == np.nan: continue
            if MODE_MAPPING_DICT[mode] == "Air": continue
            EI = energy_dict[MODE_MAPPING_DICT[mode]]

            sum_labeled_mean_EI += EI
            n_trips += 1

            length = ct["distance"]*METERS_TO_MILES
            mean_L = length* mean_for_unit_L  
            var_L = length**2 * var_for_unit_L

            mean_EC_agg += EI*mean_L
            var_EC_agg +=  EI*var_L

        avg_EI = sum_sensed_mean_EI/N_sections if only_sensing == True else sum_labeled_mean_EI/n_trips

    #print(f"ios vs android trip count: {ios_count,android_count}")
    #print(f"Sum of EIs (sensed, user labeled): {sum_sensed_mean_EI,sum_labeled_mean_EI}")   # could weight by distance
    #print(f"number of sections or trips: {N_sections,n_trips}")
    #print(air_count)
    return mean_EC_agg, var_EC_agg, avg_EI


################################################################################################
################################################################################################
################################################################################################
################################################################################################ 
## What happens when we ignore length errors?

def get_user_labeled_EC_for_one_trip_no_length_error(ct,unit_dist_MCS_df,energy_dict):

    length = ct["distance"]*METERS_TO_MILES
    mean_L = length
    var_L = 0
    
    mode = ct["mode_confirm"]  # need to make sure you convert it to an appropriate energy intensity.

    # watch out for nan and air.
    # could also search for 'drove' and 'friend' within string

    # Check for nan first or you'll get the error
    # argument of type 'float' is not iterable
    if mode == np.nan or type(mode) == float: 
        return 0,0
    elif (('car' in mode) & ('alone' in mode)) or (mode == 'drove_alone'):
        # later we can think about mixed shared and drove alone.
        #if 'with others' in mode:
        ''' EI_sr = energy_dict[MODE_MAPPING_DICT['shared_ride']]
            EI_da = energy_dict[MODE_MAPPING_DICT['drove_alone']]
            trip_user_EC = (EI_sr + EI_da)*mean_L/2'''
        EI = energy_dict[MODE_MAPPING_DICT['drove_alone']]

    elif (('car' in mode) & ('with others' in mode)) or mode == 'shared_ride':
        EI = energy_dict[MODE_MAPPING_DICT['shared_ride']]
    elif mode not in MODE_MAPPING_DICT:
        return 0,0  # might want to think about how to handle this differently.
    else:
        EI = energy_dict[MODE_MAPPING_DICT[mode]]
    
    trip_user_EC = EI*mean_L
    var_user_EC =  var_L*EI**2
    return trip_user_EC, var_user_EC


def get_predicted_EC_for_one_trip_no_length_error(ct, unit_dist_MCS_df, energy_dict):
    # currently requires that ct is not sensed as air
    #Initilize trip energy consumption

    trip_mean_EC = 0
    trip_var_EC = 0

    # Get trip mode info.
    # Get segments for the trip.
    n_sections = len(ct["section_modes"])
    section_modes = ct["section_modes"]
    sections_lengths = np.array(ct["section_distances"])*METERS_TO_MILES   # 1 meter = 0.000621371 miles

    mean_L = sections_lengths
    var_L = 0  
        
    for s in range(0,n_sections):

        if section_modes[s] == 'car':
            mean_EI = energy_dict['Car, sensed']
        elif section_modes[s] == 'air_or_hsr':
            mean_EI = energy_dict['Train']
        else:
            mean_EI = energy_dict[MODE_MAPPING_DICT[section_modes[s]]]
        var_EI = 0  # ignoring the EI variance since we're only looking at the prediction's performance

        # Propagate variance for the trip
        mean_EC = mean_L[s]*mean_EI
        var_EC = var_EI*mean_L[s]**2 + var_L*mean_EI**2

        # Add to total - follows from assumed independence of section errors.
        trip_mean_EC += mean_EC
        trip_var_EC += var_EC

    return trip_mean_EC, trip_var_EC

def get_expected_EC_for_one_trip_no_length_error(ct, unit_dist_MCS_df,android_EI_moments, ios_EI_moments):
    # currently requires that ct is not sensed as air

    # Get operating system
    os = ct['os']

    #Initialize trip energy consumption
    trip_mean_EC = 0
    trip_var_EC = 0

    # Get trip mode info.
    # Get segments for the trip.
    n_sections = len(ct["section_modes"])
    section_modes = ct["section_modes"]
    sections_lengths = np.array(ct["section_distances"])*METERS_TO_MILES   # 1 meter = 0.000621371 miles

    mean_L = sections_lengths
    var_L = 0  
        
    for s in range(0,n_sections):
        # EI mean and variance.
        # Perhaps it would be better to keep the moments in the same file?

        # Later: switch to a map style function.
        mean_EI, var_EI = get_EI_moments_for_trip(section_modes[s],os,android_EI_moments,ios_EI_moments)

        # Propagate variance for the trip
        mean_EC = mean_L[s]*mean_EI
        var_EC = var_EI*mean_L[s]**2 + var_L*mean_EI**2 #+ 

        # Add to total - follows from assumed independence of section errors.
        trip_mean_EC += mean_EC
        trip_var_EC += var_EC

    return trip_mean_EC, trip_var_EC

def compute_EC_for_all_trips_no_length_error():
    print("Computing energy consumption for each trip.")
    expected = []
    predicted = []
    user_labeled = []

    confusion_based_variance = []
    user_based_variance = []

    expected_error_list = []
    prediction_error_list = []

    for _,ct in df.iterrows():

        # Calculate expected energy consumption
        trip_expected, trip_confusion_based_variance = get_expected_EC_for_one_trip_no_length_error(ct,unit_dist_MCS_df,android_EI_moments_df,ios_EI_moments_df)

        # Calculate predicted energy consumption
        trip_predicted = get_predicted_EC_for_one_trip_no_length_error(ct,unit_dist_MCS_df,energy_dict)[0]
        
        # Calculate user labeled energy consumption
        trip_user_labeled, trip_user_based_variance = get_user_labeled_EC_for_one_trip_no_length_error(ct,unit_dist_MCS_df,energy_dict)

        expected.append(trip_expected)
        predicted.append(trip_predicted)
        user_labeled.append(trip_user_labeled)

        confusion_based_variance.append(trip_confusion_based_variance)
        user_based_variance.append(trip_user_based_variance)

        prediction_error = trip_predicted - trip_user_labeled
        expected_error = trip_expected - trip_user_labeled

        expected_error_list.append(expected_error)
        prediction_error_list.append(prediction_error)

        if abs(expected_error) > 100: 
            print(f"Large EC error: EC user labeled, EC expected: {trip_user_labeled:.2f}, {trip_expected:.2f}")
            print(f"\tTrip info: mode_confirm,sensed,distance (mi): {ct['mode_confirm'],ct['section_modes']},{ct['distance']*METERS_TO_MILES:.2f}")


    total_expected = sum(expected)
    total_predicted = sum(predicted)
    total_user_labeled = sum(user_labeled)
    print(f"Total EC: expected, predicted, user labeled: {total_expected:.2f}, {total_predicted:.2f}, {total_user_labeled:.2f}")
    print(f"standard deviation for expected: {np.sqrt(sum(confusion_based_variance)):.2f}")

    percent_error_expected = helper_functions.relative_error(total_expected,total_user_labeled)*100
    percent_error_predicted = helper_functions.relative_error(total_predicted,total_user_labeled)*100
    print(f"Percent errors for expected and for predicted, including outliers: {percent_error_expected:.2f}, {percent_error_predicted:.2f}")

    # Append the values to expanded_labeled_trips
    elt_with_errors = df.copy()  # elt: expanded labeled trips
    elt_with_errors['error_for_confusion'] = expected_error_list
    elt_with_errors['error_for_prediction'] = prediction_error_list
    elt_with_errors['expected'] = expected
    elt_with_errors['predicted'] = predicted
    elt_with_errors['user_labeled'] = user_labeled

    # Append variances
    elt_with_errors['confusion_var'] = confusion_based_variance
    elt_with_errors['user_var'] = user_based_variance
    elt_with_errors['confusion_sd'] = np.sqrt(np.array(confusion_based_variance))
    elt_with_errors['user_sd'] = np.sqrt(np.array(user_based_variance))

    return elt_with_errors
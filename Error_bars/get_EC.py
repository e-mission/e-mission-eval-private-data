import pandas as pd
import numpy as np

from confusion_matrix_handling import MODE_MAPPING_DICT

import helper_functions

METERS_TO_MILES = 0.000621371 # 1 meter = 0.000621371 miles

def get_EI_moments_for_trip(mode,os,android_EI_moments,ios_EI_moments):
    '''
    Returns the mean and variance of energy intensity for a trip or section.

    mode: string of the sensed mode for the trip or section. 
        Must be one of the possible sensed modes. air_or_hsr gets replaced with 'train'.
    os: string of the operating system that collected the trip data. 'ios' or 'android'
    android_EI_moments: dataframe of energy intensity mean and variance for each mode sensed with android.
    ios_EI_moments: dataframe of energy intensity mean and variance for each mode sensed with ios.
    '''
    # also works for a section.
    if mode == 'air_or_hsr': mode = 'train'
    if os == "android":
        mean_EI = android_EI_moments["mean(EI)"][mode]
        var_EI = android_EI_moments["variance(EI)"][mode]  # variance given the inferred mode is <mode>
    elif os == "ios":
        mean_EI = ios_EI_moments["mean(EI)"][mode]
        var_EI = ios_EI_moments["variance(EI)"][mode]

    return mean_EI, var_EI

def get_user_labeled_EC_for_one_trip(ct,unit_dist_MCS_df,energy_dict):
    '''
    Finds the mean energy consumption and variance for a single user labeled trip. The variance is all from trip length error.

    ct: confirmed trip. A row of a labeled trips dataframe.
    unit_dist_MCS_df: dataframe containing the mean and variance of trip length for a 1 unit long trip, for both operating systems.
    energy_dict: dictionary by mode of energy intensities in kWH.

    Returns the energy consumption mean and variance as a tuple of floats.
    '''
    
    # Get operating system
    os = ct['os']

    # Get OS specific trip length info.
    mean_for_unit_L = unit_dist_MCS_df[os]["mean"]
    var_for_unit_L = unit_dist_MCS_df[os]["var"]

    length = ct["distance"]*METERS_TO_MILES
    mean_L = length* mean_for_unit_L  
    var_L = length**2 * var_for_unit_L
    
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

def get_predicted_EC_for_one_trip(ct, unit_dist_MCS_df, energy_dict):
    '''
    Finds the energy consumption and variance based on predicted mode. The variance is all from trip length error.

    ct: confirmed trip. A row of a labeled trips dataframe.
    unit_dist_MCS_df: dataframe containing the mean and variance of trip length for a 1 unit long trip, for both operating systems.
    energy_dict: dictionary by mode of energy intensities in kWH.

    Returns the energy consumption mean and variance as a tuple of floats.
    '''
    # currently requires that ct is not sensed as air
    #Initilize trip energy consumption

    trip_mean_EC = 0
    trip_var_EC = 0

    # Get operating system
    os = ct['os']

    # Get OS specific trip length info.
    mean_for_unit_L = unit_dist_MCS_df[os]["mean"]
    var_for_unit_L = unit_dist_MCS_df[os]["var"]

    # Get trip mode info.
    # Get segments for the trip.
    n_sections = len(ct["section_modes"])
    section_modes = ct["section_modes"]
    sections_lengths = np.array(ct["section_distances"])*METERS_TO_MILES   # 1 meter = 0.000621371 miles

    mean_L = sections_lengths*mean_for_unit_L
    var_L = sections_lengths**2 * var_for_unit_L  
        
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
        var_EC = var_EI*mean_L[s]**2 + var_L[s]*mean_EI**2

        # Add to total - follows from assumed independence of section errors.
        trip_mean_EC += mean_EC
        trip_var_EC += var_EC

    return trip_mean_EC, trip_var_EC

def get_expected_EC_for_one_trip(ct, unit_dist_MCS_df,android_EI_moments, ios_EI_moments, EI_length_covariance):
    '''
    Finds the expected mean energy consumption and variance for a single trip.
    The variance is calculated with variance propagation of the energy intensity variance and the trip length variance.

    ct:                     confirmed trip. A row of a labeled trips dataframe.
    unit_dist_MCS_df:       dataframe containing the mean and variance of trip length for a 1 unit long trip, for both operating systems.
    energy_dict:            dictionary by mode of energy intensities in kWH.
    android_EI_moments:     dataframe of energy intensity mean and variance for each mode sensed with android.
    ios_EI_moments:         dataframe of energy intensity mean and variance for each mode sensed with ios.

    EI_length_covariance:   (assumed to be 0). covariance between trip energy intensity and trip length.
        To use this, we would need to either find a value based on past user labels or estimate this with sensed energy consumption.
        I'm not sure whether this should be different for different sensed modes (ie, use a covariance conditional on sensed mode), 
        since knowing the sensed mode tells us more information about the energy consumption than if we had no knowledge.

        With all CEO + stage user labels, I estimated EI_length covariance as 1.29.
        You might also need to add the covariance to each trip energy consumption estimate since E[XY] = E[X]E[Y] + cov(X,Y), 
        but this might overestimate energy consumption if we use a covariance of 1.2 for every trip, 
        which would be similar to assigning short trips to drove alone or a higher intensity mode.
    
    Returns the expected energy consumption mean and variance as a tuple of floats: trip_mean_EC, trip_var_EC.
    '''
    #Initialize trip energy consumption
    trip_mean_EC = 0
    trip_var_EC = 0

    # Get operating system
    os = ct['os']

    # Get OS specific trip length info.
    mean_for_unit_L = unit_dist_MCS_df[os]["mean"]
    var_for_unit_L = unit_dist_MCS_df[os]["var"]

    # Get trip mode info.
    # Get segments for the trip.
    n_sections = len(ct["section_modes"])
    section_modes = ct["section_modes"]
    sections_lengths = np.array(ct["section_distances"])*METERS_TO_MILES   # 1 meter = 0.000621371 miles

    mean_L = sections_lengths*mean_for_unit_L
    var_L = sections_lengths**2 * var_for_unit_L  
        
    for current_section in range(0,n_sections):
        # EI mean and variance.
        # Perhaps it would be better to keep the moments in the same file?

        # Later: switch to a map style function.
        mean_EI, var_EI = get_EI_moments_for_trip(section_modes[current_section],os,android_EI_moments,ios_EI_moments)

        # Propagate variance for the trip
        mean_EC = mean_L[current_section]*mean_EI
        var_EC = var_EI*mean_L[current_section]**2 + var_L[current_section]*mean_EI**2 + 2*EI_length_covariance*mean_EI*mean_L[current_section]

        # Add to total - follows from assumed independence of section errors.  # Might want to consider dependence between sections.
        trip_mean_EC += mean_EC
        trip_var_EC += var_EC

    return trip_mean_EC, trip_var_EC

def compute_aggregate_variance_by_primary_mode(df, os_EI_moments_map, unit_dist_MCS_df):
    '''
    Finds total distances in each predicted mode and uses those totals in the final aggregate variance calculation.

    df: trips dataframe with a primary_mode column.
    os_EI_moments_map: dictionary by operating system of energy intensity moments dataframes, which store mean and variance of energy intensity
        for each predicted mode.
    unit_dist_MCS_df: mean and variance estimates for unit distance trips.

    Returns the aggregate variance (var_total)
    '''
    # an idea for vectorizing:
    # Goal: get a vector of expected energy intensities and a vector of lengths.
    #cleaned_modes = [x if x != 'air_or_hsr' else 'train' for x in df.primary_mode.unique()]
    var_total = 0

    for os in df.os.unique():
        single_os_trips = df[df.os == os].copy()
        trips_grouped_by_primary_mode = single_os_trips.groupby('primary_mode').sum()

        # Get OS specific trip length info.
        mean_for_unit_L = unit_dist_MCS_df[os]["mean"]
        var_for_unit_L = unit_dist_MCS_df[os]["var"]

        for primary_mode in trips_grouped_by_primary_mode.index:

            # Fetch the distance in miles for trips that had the current primary mode.
            # This is done before renaming air_or_hsr to train because air_or_hsr has it's own associated distance.
            primary_mode_distance_traveled = trips_grouped_by_primary_mode['distance_miles'][primary_mode]
            mean_L = primary_mode_distance_traveled*mean_for_unit_L
            var_L = primary_mode_distance_traveled**2 * var_for_unit_L  

            primary_mode = 'train' if primary_mode == 'air_or_hsr' else primary_mode

            mean_EI = os_EI_moments_map[os]["mean(EI)"][primary_mode] 
            var_EI = os_EI_moments_map[os]["variance(EI)"][primary_mode] 

            var_total += var_EI*mean_L**2 + var_L*mean_EI**2 #+ 2*1.2*mean_EI*mean_L  if covariance
        
        # If you want to include covariance between lengths in each mode. The difference is tiny (<5 kWH).
        '''        
        n_modes = len(trips_grouped_by_primary_mode.index)
        p_mode = 1/n_modes
        cov_term = 0
        cov_between_lengths = trips_grouped_by_primary_mode.distance_miles.sum()*p_mode**2

        for primary_mode_i in trips_grouped_by_primary_mode.index:
            primary_mode_i = 'train' if primary_mode_i == 'air_or_hsr' else primary_mode_i
            for primary_mode_j in trips_grouped_by_primary_mode.index:
                primary_mode_j = 'train' if primary_mode_j == 'air_or_hsr' else primary_mode_j
                if primary_mode_i == primary_mode_j: continue
                var_total -= os_EI_moments_map[os]["mean(EI)"][primary_mode_i]*os_EI_moments_map[os]["mean(EI)"][primary_mode_j]*cov_between_lengths 
                #cov_term += os_EI_moments_map[os]["mean(EI)"][primary_mode_i]*os_EI_moments_map[os]["mean(EI)"][primary_mode_j]*cov_between_lengths
        #print(f"length cov: {cov_between_lengths:.2f}, cov_term: {cov_term}")
        '''

    return var_total

def spatial_autocov_based_on_clusters(df, col_of_interest, print_statistics=False):
    '''
    Outputs a loosely defined spatial covariance for the col_of_interest variable based off Moran's I.
    See https://en.wikipedia.org/wiki/Moran%27s_I for details. 
    The weights are based on cluster membership. If a trip is in the neighborhood of another (aka in the same cluster),
    the spatial weight is 1. Otherwise it is 0. Cluster membership is seen in the cluster_id or in the trip_neighbors columns.

    It might be possible to instead create a pysal weights object from a dictionary by trip_id of neighbors and use esda.moran.Moran and multiply by var(col_of_interest).

    energy_consumption_df: a trips dataframe that already has assigned clusters for each trip id.
    col_of_interest: a string label for the column to find spatial covariance from.
        eg, use 'expected' for sensing expected energy consumption estimates.

    Returns: a float representing the spatial autocovariance of the variable. 
        (I think) it should be between -1*v and 1*v, where v is the variance of the variable.
    '''
    n = len(df)
    xbar = np.mean(df[col_of_interest])
    var_x = np.var(df[col_of_interest])
    cov_sum = 0  # the final value of cov_sum will be sum_{i=1:n} (sum_{j=1:n} w_{ij} (x_i - xbar)(x_j - xbar))
    W_sum = 0 


    for i,trip in df.iterrows():
        #if trip['cluster_size'] == 1: continue
        #neighbor_list = list(trip['trip_neighbors']) 
        # sometimes neighbor list will have neighbors that are not in the timeframe of interest, but neighbors_df will only have the trips of interest.
        neighbors_df = df[(df.cluster_id == trip['cluster_id']) & (df._id != trip['_id'])]#df[df['_id'].isin(neighbor_list)] 

        # for each neighbor, multiply trip i's deviation from the mean by the neighbor's deviation from the mean.
        # then sum:  ( sum_{j=1:n} w_{ij} (x_i - xbar)(x_j - xbar) )
        trip_i_deviation = trip[col_of_interest] - xbar   # scalar
        neighbor_deviations = neighbors_df[col_of_interest] - xbar  # array
        #print(trip_i_deviation)
        #print(neighbor_deviations)
        cov_sum += sum(trip_i_deviation*neighbor_deviations)  # implicitly multiplied by w_ij = 1 for two trips in the same cluster

        # Note: since some trips have no neighbors, the "weights matrix" has rows with all zero entries and thus cannot be row standardized
        # This makes the maximum value for my I calculation greater than 1. 
        # See Whuber's answer: https://stats.stackexchange.com/questions/160459/why-is-morans-i-coming-out-greater-than-1
        W_sum += len(neighbors_df)

    # To get a covariance instead of a correlation, I don't divide by the variance
    autocov = cov_sum/W_sum  if W_sum > 0 else 0
    Morans_I = autocov/var_x
    if print_statistics:
        print(f"Moran's I for {col_of_interest}: {Morans_I:.3f}")
        print(f"spatial autocovariance: {autocov:.3f}")
    return autocov, Morans_I

def get_user_spatial_cov_map(df, estimation_method):
    '''
    Finds a spatial covariance of trip level energy consumption specific to each user.

    df: dataframe with trip expected energy consumption estimates.
    estimation_method: string, either 'expected' (confusion matrix based) or 'predicted'.

    Returns: 
        user_spatial_cov_map: dictionary by user id of trip estimated energy consumption spatial covariance
        user_Morans_I_map: dictionary by user id of trip expected energy consumption Moran's I
    '''
    user_spatial_cov_map = {}
    user_morans_I_map = {}
    df = df.copy()
    for user in df.user_id.unique():
        user_df = df[df.user_id == user].copy()
        if len(user_df) < 2: 
            user_spatial_cov_map[user] = 0
            morans_I = 0
        else:
            user_spatial_cov_map[user], morans_I = spatial_autocov_based_on_clusters(user_df, estimation_method)
        user_morans_I_map[user] = morans_I
    return user_spatial_cov_map, user_morans_I_map

def compute_variance_including_spatial_cov_for_trips_dataframe(energy_consumption_df,user_spatial_cov_map):
    '''
    energy_consumption_df: a dataframe with expected/confusion based energy consumptions and variances for each trip.
        also needs a "cluster_id" column.
    user_spatial_cov_map: a dictionary by user id of spatial covariances between trip energy consumption estimates.

    Returns variance of aggregate energy consumption for energy_consumption_df.
    '''
    # for each user, add to the sum of covariances.
    cov_sum = 0
    for user in energy_consumption_df.user_id.unique():
        # Get the trips associated with this user.
        user_df = energy_consumption_df[energy_consumption_df.user_id == user].copy()
        for cluster_id in user_df.cluster_id.unique():
            cluster_size = len(user_df[user_df.cluster_id == cluster_id])
            if cluster_size > 1:
                # To find the variance of the sum of trip EC in a cluster, the term with the covariances
                # should (I think) be 2*sum_{1<=i<j<=n_trips_in_cluster} cov(trip_i,trip_j)
                # Since we treat the covariance for the user as a constant, we need to find the number of pairs of trips in the cluster: n_trips_in_clusteer choose 2
                cov_sum += user_spatial_cov_map[user]*(cluster_size**2 - cluster_size)
    full_variance = energy_consumption_df.confusion_var.sum() + cov_sum
    return full_variance

def get_totals_and_errors(df, os_EI_moments_map, unit_dist_MCS_df):
    '''
    Finds total distances in each predicted mode and uses those totals in the final aggregate variance calculation.

    df: trips dataframe with a primary_mode column and expected energy consumptions.
    os_EI_moments_map: dictionary by operating system of energy intensity moments dataframes, which store mean and variance of energy intensity
        for each predicted mode.
    unit_dist_MCS_df: mean and variance estimates for unit distance trips.

    Returns a dictionary of various numbers.
        total_expected: the aggregate expected energy consumption
        total_user_labeled: the actual aggregate energy consumption
        total_predicted: the aggregate energy consumption based on predicted modes without using the confusion matrix.
        aggregate_sd: the standard deviation for total expected.
        user_sd: the standard deviation for total_user_labeled
        aggregate_var: the variance for total_expected
        percent_error_for_expected
        percent_error_for_predicted
        signed_error: total_expected - total_user_labeled
        error_over_sd: abs(signed_error)/sd. Tells you the number of standard deviations the expected value is from the user labeled value.
    '''

    expected, predicted, actual = sum(df['expected']), sum(df['predicted']), sum(df['user_labeled'])

    # Note: use this OR the spatial covariance function. Not both.
    final_variance = compute_aggregate_variance_by_primary_mode(df, os_EI_moments_map, unit_dist_MCS_df)

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
            "error_over_sd": error_over_sd}


def compute_all_EC_values(df, unit_dist_MCS_df,energy_dict, android_EI_moments_df,ios_EI_moments_df, EI_length_covariance = 0, print_info = True):
    '''
    Calculates trip level energy consuption (EC) in three ways: 
        expected: uses conditional energy intensity means based on the confusion matrix to calculate energy consumption
        predicted: uses energy intensity based solely on the predicted mode to calculate energy consumption.
        user labeled: calculates energy consumption based on the mode_confirm column.

    df:                     a dataframe with labeled trips and expanded user inputs.
    android_EI_moments:     dataframe of energy intensity mean and variance for each mode sensed with android.
    ios_EI_moments:         dataframe of energy intensity mean and variance for each mode sensed with ios.
    unit_dist_MCS_df:       dataframe of mean and variance estimates for unit distance trips.
    EI_length_covariance:   covariance between energy intensity and trip length. See get_expected_EC_for_one_trip for a full description.
    print_info:             boolean to decide whether to print information about large errors and total expected EC.

    Returns a copy of df with extra columns. Each value in the column is for 1 trip:     
        'error_for_confusion': exoected - user labeled EC
        'error_for_prediction' = predicted - user labeled EC
        'expected' = expected EC
        'predicted' = predicted EC
        'user_labeled' = user_labeled
        'confusion_var' = variance for expected EC
        'user_var' = variance for user labeled EC
        'confusion_sd' = standard deviation for expected EC
        'user_sd' = standard deviation for user labeled EC
    '''

    print("Computing energy consumption for each trip.")
    print(f"Using EI length covariance = {EI_length_covariance}.")
    expected = []
    predicted = []
    user_labeled = []

    confusion_based_variance = []
    user_based_variance = []

    expected_error_list = []
    prediction_error_list = []

    for _,ct in df.iterrows():

        # Calculate expected energy consumption
        trip_expected, trip_confusion_based_variance = get_expected_EC_for_one_trip(ct,unit_dist_MCS_df,android_EI_moments_df,ios_EI_moments_df,\
                                                            EI_length_covariance)

        # Calculate predicted energy consumption
        trip_predicted = get_predicted_EC_for_one_trip(ct,unit_dist_MCS_df,energy_dict)[0]
        
        # Calculate user labeled energy consumption
        trip_user_labeled, trip_user_based_variance = get_user_labeled_EC_for_one_trip(ct,unit_dist_MCS_df,energy_dict)

        expected.append(trip_expected)
        predicted.append(trip_predicted)
        user_labeled.append(trip_user_labeled)

        confusion_based_variance.append(trip_confusion_based_variance)
        user_based_variance.append(trip_user_based_variance)

        prediction_error = trip_predicted - trip_user_labeled
        expected_error = trip_expected - trip_user_labeled

        expected_error_list.append(expected_error)
        prediction_error_list.append(prediction_error)

        if (abs(expected_error) > 100) and (print_info == True): 
            print(f"Large EC error: EC user labeled, EC expected: {trip_user_labeled:.2f}, {trip_expected:.2f}")
            print(f"\tTrip info: mode_confirm,sensed,distance (mi): {ct['mode_confirm'],ct['section_modes']},{ct['distance']*METERS_TO_MILES:.2f}")


    total_expected = sum(expected)
    total_predicted = sum(predicted)
    total_user_labeled = sum(user_labeled)


    percent_error_expected = helper_functions.relative_error(total_expected,total_user_labeled)*100
    percent_error_predicted = helper_functions.relative_error(total_predicted,total_user_labeled)*100

    if print_info == True:
        print(f"Percent errors for expected and for predicted, including outliers: {percent_error_expected:.2f}, {percent_error_predicted:.2f}")
        print(f"Total EC: expected, predicted, user labeled: {total_expected:.2f}, {total_predicted:.2f}, {total_user_labeled:.2f}")
        print(f"standard deviation for expected: {np.sqrt(sum(confusion_based_variance)):.2f}")

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

def compute_expected_EC_values(df, unit_dist_MCS_df,android_EI_moments_df,ios_EI_moments_df, EI_length_covariance = 0):
    '''
    Calculates trip level expected energy consuption (EC) and adds them as a column to the dataframe.
        expected: uses conditional energy intensity means based on the confusion matrix to calculate energy consumption

    df:                     a dataframe with "section_modes" and "section_distances" columns.
    unit_dist_MCS_df:       dataframe of mean and variance estimates for unit distance trips.
    android_EI_moments:     dataframe of energy intensity mean and variance for each mode sensed with android.
    ios_EI_moments:         dataframe of energy intensity mean and variance for each mode sensed with ios.
    EI_length_covariance:   covariance between energy intensity and trip length. See get_expected_EC_for_one_trip for a full description.
    print_info:             boolean to decide whether to print information about large errors and total expected EC.

    Returns a copy of df with extra columns. Each value in the column is for 1 trip:     
        'expected' = expected EC
        'confusion_var' = variance for expected EC
        'confusion_sd' = standard deviation for expected EC
    '''

    print("Computing energy consumption for each trip.")
    print(f"Using EI length covariance = {EI_length_covariance}.")
    expected = []
    confusion_based_variance = []
    for _,ct in df.iterrows():

        # Calculate expected energy consumption
        trip_expected, trip_confusion_based_variance = get_expected_EC_for_one_trip(ct,unit_dist_MCS_df,android_EI_moments_df,ios_EI_moments_df,\
                                                            EI_length_covariance)
        expected.append(trip_expected)

        confusion_based_variance.append(trip_confusion_based_variance)

    # elt: expanded labeled trips
    # Append the values to expanded_labeled_trips
    elt_with_errors = df.copy() 
    elt_with_errors['expected'] = expected

    # Append variances
    elt_with_errors['confusion_var'] = confusion_based_variance
    elt_with_errors['confusion_sd'] = np.sqrt(np.array(confusion_based_variance))

    return elt_with_errors

def get_expected_EC_based_on_primary_mode_for_one_trip(ct, unit_dist_MCS_df, android_EI_moments, ios_EI_moments, energy_dict):
    '''
    Finds the expected mean energy consumption and variance for a single trip.
    It uses the primary mode and the trip distance to calculate the mean, rather than using section modes and distances.
    The variance is calculated with variance propagation of the energy intensity variance and the trip length variance.

    ct:                     confirmed trip. A row of a labeled trips dataframe.
    unit_dist_MCS_df:       dataframe containing the mean and variance of trip length for a 1 unit long trip, for both operating systems.
    energy_dict:            dictionary by mode of energy intensities in kWH.
    android_EI_moments:     dataframe of energy intensity mean and variance for each mode sensed with android.
    ios_EI_moments:         dataframe of energy intensity mean and variance for each mode sensed with ios.
    energy_dict:            dictionary by mode of energy intensity in kWH

    EI_length_covariance:   (assumed to be 0). covariance between trip energy intensity and trip length.
        To use this, we would need to either find a value based on past user labels or estimate this with sensed energy consumption.
        I'm not sure whether this should be different for different sensed modes (ie, use a covariance conditional on sensed mode), 
        since knowing the sensed mode tells us more information about the energy consumption than if we had no knowledge.

        With all CEO + stage user labels, I estimated EI_length covariance as 1.29.
        You might also need to add the covariance to each trip energy consumption estimate since E[XY] = E[X]E[Y] + cov(X,Y), 
        but this might overestimate energy consumption if we use a covariance of 1.2 for every trip, 
        which would be similar to assigning short trips to drove alone or a higher intensity mode.
    
    Returns the expected energy consumption mean.
    '''
    # Get operating system
    os = ct['os']

    # Get OS specific trip length info.
    mean_for_unit_L = unit_dist_MCS_df[os]["mean"]
    var_for_unit_L = unit_dist_MCS_df[os]["var"]

    # Get primary mode
    longest_section_distance = max(ct["section_distances"])*METERS_TO_MILES
    primary_mode = ct["section_modes"][ct["section_distances"]==longest_section_distance]

    # in case there are ever tied longest sections.
    # pick the most energy intensive mode.
    if isinstance(primary_mode,list): 
        mini_energy_dict = {x: energy_dict[MODE_MAPPING_DICT[x]] for x in primary_mode}
        primary_mode = max(mini_energy_dict, key=mini_energy_dict.get)
        print(f"found a tie for longest section. Choosing {primary_mode}")

    mean_EI, var_EI = get_EI_moments_for_trip(primary_mode,os,android_EI_moments,ios_EI_moments)

    # use longest section distance or use trip distance?
    # mean_EC = longest_section_distance*mean_for_unit_L*mean_EI
    mean_L = ct["distance_miles"]*mean_for_unit_L
    mean_EC = mean_L*mean_EI

    return mean_EC
    
def compute_all_EC_values_from_primary_mode(df, unit_dist_MCS_df,energy_dict, android_EI_moments_df,ios_EI_moments_df):

    print("Computing energy consumption for each trip.")
    expected = []
    user_labeled = []
    user_var = []

    for _,ct in df.iterrows():
        # Calculate expected energy consumption
        trip_expected = get_expected_EC_based_on_primary_mode_for_one_trip(ct,unit_dist_MCS_df,android_EI_moments_df,ios_EI_moments_df, energy_dict)
        expected.append(trip_expected)

        trip_user_labeled, trip_user_var = get_user_labeled_EC_for_one_trip(ct, unit_dist_MCS_df, energy_dict)
        user_labeled.append(trip_user_labeled)
        user_var.append(trip_user_var)

    # Append the values to expanded_labeled_trips
    elt = df.copy()  # elt: expanded labeled trips
    elt['expected'] = expected
    elt['user_labeled'] = user_labeled
    elt['user_var'] = user_var

    return elt


def compute_aggregate_variance_with_total_distance_from_sections(df, os_EI_moments_map, unit_dist_MCS_df):
    '''
    Finds total distances in each predicted mode and uses those totals in the final aggregate variance calculation.

    df: trips dataframe with a primary_mode column.
    os_EI_moments_map: dictionary by operating system of energy intensity moments dataframes, which store mean and variance of energy intensity
        for each predicted mode.
    unit_dist_MCS_df: mean and variance estimates for unit distance trips.

    Returns the aggregate variance (var_total) and a dictionary by OS of dictionaries by mode of total section distances 
    '''
    var_total = 0

    distance_in_mode = {}
    for os in df.os.unique():

        single_os_trips = df[df.os == os].copy()

        # Get OS specific trip length info.
        mean_for_unit_L = unit_dist_MCS_df[os]["mean"]
        var_for_unit_L = unit_dist_MCS_df[os]["var"]

        sensed_mode_distance_map = {}
        for _,ct in single_os_trips.iterrows():
            sections_lengths = np.array(ct["section_distances"])*METERS_TO_MILES 
            for i, mode in enumerate(ct["section_modes"]):
                if mode not in sensed_mode_distance_map.keys():
                    sensed_mode_distance_map[mode] = 0
                # Add to the total distance traveled in this mode.
                sensed_mode_distance_map[mode] += sections_lengths[i]
        
        distance_in_mode[os] = sensed_mode_distance_map

        for mode in sensed_mode_distance_map.keys():
            mean_L = sensed_mode_distance_map[mode]*mean_for_unit_L
            var_L = sensed_mode_distance_map[mode]**2 * var_for_unit_L  
            mode = 'train' if mode == 'air_or_hsr' else mode

            mean_EI = os_EI_moments_map[os]["mean(EI)"][mode] 
            var_EI = os_EI_moments_map[os]["variance(EI)"][mode] 

            var_total += var_EI*mean_L**2 + var_L*mean_EI**2 #+ 2*cov(EI,L)*mean_EI*mean_L  if including covariance

    return var_total, distance_in_mode
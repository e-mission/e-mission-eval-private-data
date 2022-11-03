import argparse
import pandas as pd
import numpy as np
from uuid import UUID
import pickle
import os

import matplotlib.pyplot as plt

import sys
sys.path.append('/Users/mallen2/alternate_branches/eval-compatible-server/e-mission-server')

import helper_functions
# Covered by helper_functions
#import emission.storage.timeseries.abstract_timeseries as esta
#import emission.storage.decorations.trip_queries as esdtq

import emission.core.wrapper.user as ecwu


import sklearn.model_selection as skm
import confusion_matrix_handling as cm_handling
from confusion_matrix_handling import MODE_MAPPING_DICT
import get_EC

import emission.core.get_database as edb

METERS_TO_MILES = 0.000621371 # 1 meter = 0.000621371 miles

def relative_error(m,t):
    return (m-t)/t

# Energy Consumption errors by mode:
def plot_error_hists_by_mode(df):
    n_plots = len(df.mode_confirm.unique())
    fig,axs = plt.subplots(n_plots,2)
    fig.set_figwidth(15)
    fig.set_figheight(4*n_plots)
    fig.suptitle(f"Energy consumption errors by mode for {chosen_program}")
    i = 0

    for mode in df.mode_confirm.unique():
        if mode == 'combination_football game, dinner, drove friend home': continue

        mode_expected_error = df[df.mode_confirm == mode]['error_for_confusion']
        mode_prediction_error = df[df.mode_confirm == mode]['error_for_prediction']


        if type(mode) == float: mode = 'nan'
        axs[i,0].hist(mode_expected_error,bins=30)
        axs[i,0].set_xlabel(mode + ' EC confusion based error')

        axs[i,1].hist(mode_prediction_error,bins=30)
        axs[i,1].set_xlabel(mode + ' EC prediction based error')
        i+=1
    fig_file = output_path+chosen_program+"_EC_mode_errors_with_"+which_car_precision+ "_for_car_precision_info"+ "_r_from_"+which_r+ "_" +remove_outliers + "_remove_outliers"+".png"
    fig.savefig(fig_file)
    plt.close(fig)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("chosen_program", type=str,
                        help="the program you want to run sensitivity analysis for")
    parser.add_argument("which_r", type=str,
                        help="Select the ratio of drove alone trips to shared ride trips. Options are dataset (use the ratio found in the dataset)\n \
                        or TEDB (use the ratio that sets Shankari\'s load factor to 1.5 (r = 1)")
    parser.add_argument("which_car_precision", type=str,
                        help="Select how to pick car precision. Options are dataset (use the precision found in the dataset)\n \
                        or MobilityNet (stick to the original confusion matrix values)")

    parser.add_argument("remove_outliers",type=str,
                        help= "String indicating whether to drop the trips with outlier EC error. yes or no, case sensitive.")

    parser.add_argument("drop_not_a_trip",type=str, default=False,  # tried type= bool but always returned true.
                        help= "signifies whether you want to drop trips labeled by the user as \'not a trip\'. Also will drop nans if true.")
    parser.add_argument("analysis_name",type=str, default='',
                        help= "extra name to identify the analysis so that you don't overwrite past analysis outputs.")

    args = parser.parse_args()
    chosen_program = args.chosen_program
    which_r = args.which_r
    which_car_precision = args.which_car_precision
    remove_outliers = args.remove_outliers
    drop_not_a_trip = True if args.drop_not_a_trip.upper() == 'TRUE' else False 
    analysis_name = args.analysis_name

    # Create program specific folder
    output_path = "/Users/mallen2/OpenPATH_Data/Sensing_sensitivity_analysis/"+chosen_program+"_"+analysis_name+"/"
    if not os.path.isdir(output_path):
        os.mkdir(output_path)
    

    df_EI = pd.read_csv(r'Public_Dashboard/auxiliary_files/energy_intensity.csv') # r stands for raw string, only matters if the path is on Windows

    # Get the list of users for each program.

    # Split UUIDs by program
    # ue stands for user_email
    program_uuid_map = {}
    all_user_list = []
    for ue in edb.get_uuid_db().find():
        program = ue['user_email'].split("_")[0]
        uuid = ue['uuid']  # didn't convert to string since ecwu.User(u).getProfile() returns none if u is a string.
        if program in program_uuid_map.keys():
            program_uuid_map[program].append(uuid)
        else:
            print(f"Found new program {program}, creating new list")
            program_uuid_map[program] = []
            program_uuid_map[program].append(uuid)
        all_user_list.append(uuid)

    # Only look look at the chosen program's data.
    user_list = all_user_list if chosen_program == 'all' else program_uuid_map[chosen_program]

    # Get the OS for each user.
    print(f"Finding OS for each user in {chosen_program}.")
    os_map = {}
    for u in user_list:
        profile = ecwu.User(u).getProfile()
        if 'curr_platform' in profile:
            os_map[u] = profile['curr_platform']
        else:
            print("Removed a user who had no OS information.")
            user_list.remove(u) # Note: this removes u from programs_uuid_map[chosen_program] as well.

    confirmed_trip_df_map = {}
    labeled_trip_df_map = {}
    expanded_labeled_trip_df_map = {}

    # Get all user labeled trips and expand the user inputs.
    expanded_labeled_trips = helper_functions.get_expanded_labeled_trips(user_list)

    # Get error related info
    unit_dist_MCS_df = pd.read_csv("unit_distance_MCS.csv").set_index("moment")
    android_EI_moments_df = pd.read_csv("android_EI_moments.csv").set_index("mode")
    ios_EI_moments_df = pd.read_csv("ios_EI_moments.csv").set_index("mode")

    # Dictionary of energy intensities in kWH/PMT
    energy_dict = cm_handling.get_energy_dict(df_EI)

    # Make an operating system column.
    expanded_labeled_trips['os'] = expanded_labeled_trips['user_id'].map(os_map)

    #print("Counts of each mode before dropping trips.")
    #print(expanded_labeled_trips['mode_confirm'].value_counts())

    ################################################################################################
    # Drop trips that we don't want to include.
    ################################################################################################
    expanded_labeled_trips = expanded_labeled_trips.drop(
        expanded_labeled_trips[expanded_labeled_trips.mode_confirm == 'air'].index
        )
    print('Dropping air trips, and trips with no OS.')
    if drop_not_a_trip == True: print("Also dropping trips labeled as not a trip and trips with mode_confirm of nan.")
    for i,ct in expanded_labeled_trips.iterrows():

        if 'air_or_hsr' in ct['section_modes']:
            #print(f"Sensed {ct['section_modes']}, user label was {ct['mode_confirm']}") 
            expanded_labeled_trips = expanded_labeled_trips.drop(index = i)
        elif type(ct['os']) == float:  # several stage trips have nan operating systems.
            expanded_labeled_trips = expanded_labeled_trips.drop(index = i)

        elif (type(ct['mode_confirm']) == float) and (drop_not_a_trip == True): 
            expanded_labeled_trips = expanded_labeled_trips.drop(index = i)

        elif ((ct['mode_confirm'] == 'not_a_trip') or (ct['mode_confirm'] == 'no_travel')) and (drop_not_a_trip == True):
            #print(f"Sensed {ct['section_modes']}, user label was {ct['mode_confirm']}") 
            expanded_labeled_trips = expanded_labeled_trips.drop(index = i)
    print("Finished dropping trips.")

    #print("Counts of each mode after dropping trips.")
    #print(expanded_labeled_trips['mode_confirm'].value_counts())

    ################################################################################################
    # Find the ratio of shared ride to drove alone distance, and car to other
    ################################################################################################
    mode_distances = expanded_labeled_trips.groupby('mode_confirm').sum()['distance']

    drove_alone_distance = 0
    shared_ride_distance = 0
    other_modes_distance = 0
    for mode in mode_distances.index:
        if mode == np.nan or type(mode) == float: continue
        elif (('car' in mode) & ('alone' in mode)) or (mode == 'drove_alone'):
            drove_alone_distance += mode_distances[mode]
        elif (('car' in mode) & ('with others' in mode)) or mode == 'shared_ride':
            shared_ride_distance += mode_distances[mode]
        else:
            other_modes_distance += mode_distances[mode]

    car_to_other = (drove_alone_distance + shared_ride_distance)/other_modes_distance

    r_for_dataset = drove_alone_distance/shared_ride_distance
    print(f"r for {chosen_program} is {r_for_dataset:.3f}")

    r = 1 if args.which_r == 'TEDB' else r_for_dataset
    print(f"Using r={r:.3f} from {args.which_r}.")

    # for each trip, predict energy consumption with either the expectation or the prediction. compare it to the actual energy consumption.

    ################################################################################################
    # Calculate car precision for this dataset.
    ################################################################################################
    print(f'Calculating car precision for {chosen_program}.')
    car_user_sensing_match = 0
    primary_cars = 0

    for i,ct in expanded_labeled_trips.iterrows():
        if len(ct["section_distances"]) == 0: # for data up to 5-9-2022, there are 63 stage trips with no sensed sections.
            print('Dropped a trip with no sensed sections.')
            expanded_labeled_trips = expanded_labeled_trips.drop(index = i) 
        else:
            longest_section = max(ct["section_distances"])
            primary_mode = ct["section_modes"][ct["section_distances"]==longest_section]

            # in case there are ever tied longest sections.
            # pick the most energy intensive mode.
            if isinstance(primary_mode,list): 
                mini_energy_dict = {x:energy_dict[MODE_MAPPING_DICT[x]] for x in primary_mode}
                primary_mode = max(mini_energy_dict, key=mini_energy_dict.get)

            if primary_mode == 'car': 
                primary_cars += 1
                mode = ct["mode_confirm"]
                if type(mode) is float: continue
                if ('car' in mode) or (mode == 'drove_alone') or (mode == 'shared_ride'): #ct["mode_confirm"] in ["drove_alone","shared_ride","car"]:
                    car_user_sensing_match +=1

    # Calculate precision for car.
    car_precision = car_user_sensing_match/primary_cars  #P(userlabel = car| predict car) = P(predict and ground truth car)/P(predict car)
    print(f"Sensed car primary mode precision for {chosen_program}: {car_precision:.3f}")   # 83% for vail, 73.9% for pueblo county
    print(f"For {chosen_program}, using car precision found in {which_car_precision}")

    android_confusion = pd.read_csv("android_confusion.csv").set_index('gt_mode')
    ios_confusion = pd.read_csv("ios_confusion.csv").set_index('gt_mode')

    if which_car_precision == 'dataset':
        new_car_precision = car_precision
        new_android_cm = cm_handling.change_precision(android_confusion,'car',new_car_precision)
        new_ios_cm = cm_handling.change_precision(ios_confusion,'car',new_car_precision)
    else:
        new_android_cm = android_confusion
        new_ios_cm = ios_confusion

    car_EI_load_divider = (r+1)/(r+0.5)  # aka Michael's definition of load factor.
    drove_alone_EI = energy_dict["Gas Car, drove alone"]
    energy_dict.update({"Gas Car, sensed": drove_alone_EI/car_EI_load_divider})

    android_EI_moments_df = cm_handling.get_conditional_EI_expectation_and_variance(new_android_cm,energy_dict)
    ios_EI_moments_df = cm_handling.get_conditional_EI_expectation_and_variance(new_ios_cm,energy_dict)

    ################################################################################################
    # Calculate energy consumption for each trip in three ways: 
    # expected/confusion based, predicted, and user labeled
    ################################################################################################
    print("Computing energy consumption for each trip.")
    expected = []
    predicted = []
    user_labeled = []

    confusion_based_variance = []
    user_based_variance = []

    predicted_dict = {}
    expected_dict = {}

    expected_error_list = []
    prediction_error_list = []

    for i,ct in expanded_labeled_trips.iterrows():

        # Calculate expected energy consumption
        trip_expected, trip_confusion_based_variance = get_EC.get_expected_EC_for_one_trip(ct,unit_dist_MCS_df,android_EI_moments_df,ios_EI_moments_df)

        # Calculate predicted energy consumption
        trip_predicted = get_EC.get_predicted_EC_for_one_trip(ct,unit_dist_MCS_df,energy_dict)[0]
        
        # Calculate user labeled energy consumption
        trip_user_labeled, trip_user_based_variance = get_EC.get_user_labeled_EC_for_one_trip(ct,unit_dist_MCS_df,energy_dict)

        expected.append(trip_expected)
        predicted.append(trip_predicted)
        user_labeled.append(trip_user_labeled)

        confusion_based_variance.append(trip_confusion_based_variance)
        user_based_variance.append(trip_user_based_variance)

        user_mode = ct['mode_confirm']
        if user_mode not in predicted_dict: predicted_dict[user_mode] = []
        if user_mode not in expected_dict: expected_dict[user_mode] = []

        prediction_error = trip_predicted - trip_user_labeled
        expected_error = trip_expected - trip_user_labeled

        expected_error_list.append(expected_error)
        prediction_error_list.append(prediction_error)

        if abs(expected_error) < 100: 

            predicted_dict[user_mode].append(prediction_error)
            expected_dict[user_mode].append(expected_error)
        else:
            print(f"Large EC error: EC user labeled, EC expected: {trip_user_labeled:.2f}, {trip_expected:.2f}")
            print(f"\tTrip info: mode_confirm,sensed,distance (mi): {ct['mode_confirm'],ct['section_modes']},{ct['distance']*METERS_TO_MILES:.2f}")


    total_expected = sum(expected)
    total_predicted = sum(predicted)
    total_user_labeled = sum(user_labeled)
    print(f"Total EC: expected, predicted, user labeled {total_expected:.2f},{total_predicted:.2f},{total_user_labeled:.2f}")
    print(f"standard deviation for expected: {np.sqrt(sum(confusion_based_variance)):.2f}")

    print(f"Percent errors for expected and for predicted, including outliers: {relative_error(total_expected,total_user_labeled)*100:.2f}, {relative_error(total_predicted,total_user_labeled)*100:.2f}")

    # Append the values to expanded_labeled_trips
    elt_with_errors = expanded_labeled_trips.copy()  # elt: expanded labeled trips
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

    quantiles = [99.9,0.1]       ######################INPUT PARAMETER
    upper, lower = np.percentile(expected_error_list, quantiles)
    print(f"{quantiles[0]:.2f} and {quantiles[1]:.2f} percentiles: {upper:.2f},{lower:.2f}")

    if remove_outliers == 'yes':
        # | (elt_with_errors.errors_from_confusion > upper)
        xlow_outliers = elt_with_errors[(elt_with_errors.error_for_confusion < lower) | (elt_with_errors.error_for_confusion > upper)]

        # Drop outliers below the 0.1 percentile.
        elt_with_errors_preprocessed = elt_with_errors.drop(xlow_outliers.index)
    else:
        elt_with_errors_preprocessed = elt_with_errors


    ########################################################################################
    # Sensitivity Analysis
    ########################################################################################
    print("Performing sensitivity analysis.")
    # Calculate the mean and sd for all user labeled and for all sensed:
    mean_EC_all_sensing = sum(elt_with_errors_preprocessed['expected'])
    predicted_EC_all_sensing = sum(elt_with_errors_preprocessed['predicted'])
    mean_EC_all_user_labeled = sum(elt_with_errors_preprocessed['user_labeled'])

    print(f"Percent errors for expected and predicted, {remove_outliers} to removing outliers: {relative_error(mean_EC_all_sensing,mean_EC_all_user_labeled)*100:.2f}, {relative_error(predicted_EC_all_sensing,mean_EC_all_user_labeled)*100:.2f}")

    sd_sensed = np.sqrt(sum(elt_with_errors_preprocessed['confusion_var']))
    sd_users = np.sqrt(sum(elt_with_errors_preprocessed['user_var']))

    # Now calculate for various random splits of the data
    # 10^3 NMC takes 10 seconds on vail to create all 4 splits.
    proportion_sensed = [0.2,0.4,0.6,0.8]
    NMC = 100#**2#**3

    summary_df_map = {}
    for ps in proportion_sensed:
        
        mean_EC_agg = []
        var_EC_agg = []
        error_EC_agg = []
        for j in range(0,NMC):
            rand_state = np.random.RandomState(1+j)

            # Split the labeled trips into a user labeled dataframe and a sensed dataframe
            user_labeled,sensed  = skm.train_test_split(elt_with_errors_preprocessed, 
                                                        test_size = ps, # sensed
                                                        train_size = 1-ps,  # user_labeled
                                                        random_state= rand_state)
            mean_EC_sensed, var_EC_sensed = sum(sensed['expected']), sum(sensed['confusion_var'])
            
            mean_EC_user_labeled, var_EC_user_labeled = sum(user_labeled['user_labeled']), sum(user_labeled['user_var'])

            ########################################################################################## Save these
            # Get the total mean and variance for the current iteration and add it to a list.
            current_aggregate_EC = mean_EC_sensed + mean_EC_user_labeled
            mean_EC_agg.append(current_aggregate_EC)
            var_EC_agg.append(var_EC_sensed + var_EC_user_labeled)
            error_EC_agg.append(current_aggregate_EC - mean_EC_all_user_labeled)

            sd_EC_agg = np.sqrt(np.array(var_EC_agg))

        summary_df_map[ps] = pd.DataFrame({"mean": mean_EC_agg, "sd": sd_EC_agg, 'error': error_EC_agg})
    
    summary_df_map[0] = pd.DataFrame({"mean": mean_EC_all_user_labeled, "sd": sd_sensed},index= [0])
    summary_df_map[1] = pd.DataFrame({"mean": mean_EC_all_sensing , "sd": sd_sensed, 'error': mean_EC_all_sensing - mean_EC_all_user_labeled},index= [0])

    pkl_file_name = output_path+chosen_program+"_sensitivity_summary_with_"+which_car_precision+ "_for_car_precision_info"+ "_r_from_"+which_r+ "_" +remove_outliers + "_remove_outliers"+".pickle"
    with open(pkl_file_name, 'wb') as f:
        pickle.dump(summary_df_map, f)
            # prop var sensed
            # prop var user labeled
    average_summaries = {}
    for ps in proportion_sensed:
        average_across_splits_mean = np.mean(summary_df_map[ps]["mean"])
        average_across_splits_sd = np.mean(summary_df_map[ps]["sd"])
        average_summaries[ps] = {"mean": average_across_splits_mean, "sd": average_across_splits_sd}

    def get_interval(mean,sd):
        return [mean -sd, mean,mean + sd]

    interval_sensed_vail = get_interval(mean_EC_all_sensing,sd_sensed)
    interval_users_vail = get_interval(mean_EC_all_user_labeled,sd_users)

    #######################
    # Plot sensitvity analysis results
    #######################
    fig,ax = plt.subplots()
    fig.set_figheight(6)

    print(f"Prop = {0}: mean, sd: {mean_EC_all_user_labeled:.2f},{sd_users:.2f}")

    ax.plot([0]*3,interval_users_vail,'bo') 
    j = 1
    for ps in proportion_sensed:
        summary = average_summaries[ps]

        print(f"Prop = {ps}: mean, sd: {summary['mean']:.2f},{summary['sd']:.2f}")
        x = [ps]*3
        y = get_interval(summary["mean"],summary["sd"])
        ax.plot(x,y,'bo')
        j+=1

    
    print(f"Prop = {1}: mean, sd: {mean_EC_all_sensing:.2f},{sd_sensed:.2f}")
    ax.plot([1]*3,interval_sensed_vail,'bo')
    #ax.set_ylim([7000,11000])#([7000,11000]) [40000,70000]
    ax.set_xlabel("Proportion of trips using sensing as opposed to user labels")
    ax.set_ylabel("Energy consumption (kWH)")

    fig.suptitle(f"{chosen_program} energy consumption mean +- 1 sd as percent of sensed trips increases")

    ################ Save Figure
    fig_file = output_path+chosen_program+"_EC_sensitivity_with_"+which_car_precision+ "_for_car_precision_info"+ "_r_from_"+which_r+ "_" +remove_outliers + "_remove_outliers"+".png"
    fig.savefig(fig_file)
    plt.close(fig)

    # How often is the magnitude of the aggregate error less than z standard deviations?
    z = 2
    print(f"What percent of the time is the error magnitude within {z} standard deviations of the mean?")
    for ps in proportion_sensed:
        ps0x = summary_df_map[ps]   # proportion sensed = 0.x
        print(f"{ps} proportion sensed: {sum(z*ps0x['sd'] > abs(ps0x['error']))/len(ps0x)}")

    plot_error_hists_by_mode(elt_with_errors_preprocessed);

    print("Summary:")
    print(f"r for {chosen_program} is {r_for_dataset:.3f}")
    print(f"car:other ratio for {chosen_program} is {car_to_other:.3f}")
    print(f"Sensed car primary mode precision for {chosen_program}: {car_precision:.3f}")
    print(f"Percent errors for expected and predicted, {remove_outliers} to removing outliers: {relative_error(mean_EC_all_sensing,mean_EC_all_user_labeled)*100:.2f}, {relative_error(predicted_EC_all_sensing,mean_EC_all_user_labeled)*100:.2f}")
    print(f"Total EC:\n expected: {total_expected:.2f}\n predicted: {total_predicted:.2f}\n user labeled: {total_user_labeled:.2f}")

    print(f"Input arguments used:\n{args}")
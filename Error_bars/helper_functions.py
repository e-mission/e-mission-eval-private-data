import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import get_EC
import confusion_matrix_handling as cm_handling
from numpy.random import default_rng

def relative_error(m,t):
    '''Compute the relative error. m is the measured value, t is the true value.'''
    return (m-t)/t if t != 0 else np.nan

def find_sensed_car_energy_intensity(energy_dict, electric_car_proportion, drove_alone_to_shared_ride_ratio):
    '''
    Finds an energy intensity to use for all trips sensed as car, 
    taking electric cars and shared rides into consideration.

    Returns a float of energy intensity for a trip sensed as car.
    
    '''
    # here I'm referring to car_load_factor: the number that we divide the drove alone energy intensity by
    # for r = 1, car_load_factor is 4/3.
    gas_car_drove_alone_EI = energy_dict["Gas Car, drove alone"]
    e_car_drove_alone_EI = energy_dict["E-car, drove alone"]
    # NOTE: MODE_MAPPING_DICT (seen in confusion_matrix_handling.py) is currently mapping 'drove_alone' 
    # (from before the OpenPATH update that distinguished E-car and gas car) to 'Gas Car, drove alone.'
    # MODE_MAPPING_DICT = {'drove_alone': 'Gas Car, drove alone', ...

    # Include the chance of electric car in the sensed energy intensity.
    sensed_car_drove_alone_EI = electric_car_proportion*e_car_drove_alone_EI + (1-electric_car_proportion)*gas_car_drove_alone_EI

    # Include the chance that a sensed car trip is shared ride.
    r = drove_alone_to_shared_ride_ratio
    car_load_factor = (r+1)/(r+0.5)     
    sensed_car_EI = sensed_car_drove_alone_EI/car_load_factor
    return sensed_car_EI

def drop_unwanted_trips(df,drop_not_a_trip):
    '''
    Drops trips labeled by the user as 'air'.\
    Optionally drops trips labeled as "not_a_trip" or "no_travel" or with mode_confirm== nan.

    df: pandas dataframe with participant trip mode labels.
    drop_not_a_trip: boolean representing whether to leave out trips with the user label "not_a_trip" or "no_travel" or with mode_confirm== nan.

    Returns a dataframe copy of df with fewer trips.
    '''

    print('Dropping user labeled AIR trips and trips with no OS.')
    df = df.copy()
    df = df.drop(df[df.mode_confirm == 'air'].index)
    df = df[df['os'].notna()]  # several stage trips have nan operating systems.

    if drop_not_a_trip == True: 
        print("Also dropping trips labeled as not a trip and trips with mode_confirm of nan.")
        df = df.drop(df[df.mode_confirm.isin(['not_a_trip','no_travel'])].index)
        df = df[df['mode_confirm'].notna()]

    # Old code:
    #for i,ct in df.iterrows():

        # To look at only trips where the user label is in ['drove_alone','bike','bus','walk'] and the predicted mode is in
        # ["car","walking","bicycling","no_sensed","bus"]
        '''    if ct['mode_confirm'] not in ['drove_alone','bike','bus','walk']:
            expanded_labeled_trips = expanded_labeled_trips.drop(index = i)
        elif any(x not in ["car","walking","bicycling","no_sensed","bus"] for x in ct['section_modes']):
            # if any section mode is not in the above list, drop the trip.
            expanded_labeled_trips = expanded_labeled_trips.drop(index = i)'''

        # This code is to look at correctly labeled trips.
        # ie the sensed mode maps to the same energy intensity (when there is no confusion) as the user labeled mode maps to.
        # This will also include shared rides and use a load factor.
        '''    if ct['mode_confirm'] not in MODE_MAPPING_DICT:
            expanded_labeled_trips = expanded_labeled_trips.drop(index = i)
            continue
        if not (MODE_MAPPING_DICT[ct['mode_confirm']]==MODE_MAPPING_DICT[ct['primary_mode']]):
            # one last check to make sure we don't drop shared rides that were labeled as car.
            if not ((ct['mode_confirm'] == 'shared_ride') and (ct['primary_mode'] == 'car')):
                expanded_labeled_trips = expanded_labeled_trips.drop(index = i)
                continue'''
    return df

def drop_custom_labels(df, MODE_MAPPING_DICT):
    ''' Removes all trips for which the mode_confirm is not in MODE_MAPPING_DICT 
    and car and alone/with others is not in the mode_confirm string'''
    df = df.copy()

    custom_mode_trip_list = []
    for i,ct in df.iterrows():
        mode = ct['mode_confirm']
        if mode == np.nan or type(mode) == float: 
            continue
        elif (('car' in mode) & ('alone' in mode)) or (mode == 'drove_alone'):
            continue
        elif (('car' in mode) & ('with others' in mode)) or mode == 'shared_ride':
            continue
        elif mode not in MODE_MAPPING_DICT:
            custom_mode_trip_list.append(ct['_id'])

    # drop the indices where the _id is in custom_mode_trip_list
    df = df.drop(df[df._id.isin(custom_mode_trip_list)].index)
    return df




def get_ratios_for_dataset(df):
    '''
    Finds various ratios and proportions of trip modes for df.
    df: pandas dataframe with participant trip mode labels.

    You may need to update the criteria for drove alone and shared ride to include electric car trips.

    Returns a dictionary containing the ratios and proportions.
    Currently they are:
        Based on user labels:
            r: ratio of drove alone distance to shared ride distance
            car_proportion: distance traveled in the mode 'car' over the total distance traveled.
            ebike_proportion: distance traveled in the mode 'ebike' over the total distance traveled.
            walk_proportion: distance traveled in the mode 'walk' over the total distance traveled.
            drove_alone_proportion: distance traveled in the mode 'drove_alone' over the total distance traveled.
            shared_ride_proportion: distance traveled in the mode 'shared_ride' over the total distance traveled.
            drove_alone_distance: distance traveled in the mode 'drove_alone'.
            shared_ride_distance: distance traveled in the mode 'shared_ride'.
            car_to_other: ratio of car distance to distance traveled in non car modes.
            non_moto_to_moto: ratio of distance traveled in non-motorized modes to distance traveled in motorized modes.

    '''     
    mode_distances = df.groupby('mode_confirm').sum()['distance']

    drove_alone_distance = 0
    shared_ride_distance = 0
    all_modes_distance = 0
    non_car_motorized = 0
    for mode in mode_distances.index:
        if mode == np.nan or type(mode) == float: continue
        elif (('car' in mode) & ('alone' in mode)) or (mode == 'drove_alone'): 
            drove_alone_distance += mode_distances[mode]
        elif (('car' in mode) & ('with others' in mode)) or mode == 'shared_ride':
            shared_ride_distance += mode_distances[mode]    
        elif mode in ['bus','taxi','free_shuttle','train']: # should ebike be considered motorized?
            non_car_motorized += mode_distances[mode]
        all_modes_distance += mode_distances[mode]  # should nan trip distance be included?

    car_distance = drove_alone_distance + shared_ride_distance
    #not_a_trip_distance = mode_distances.loc["not_a_trip"] if 'not_a_trip' in mode_distances.index else 0
    motorized = car_distance + non_car_motorized
    non_motorized = all_modes_distance - motorized # this will include not a trip.

    non_moto_to_moto = non_motorized/motorized

    primary_mode_percents = df.groupby('primary_mode').sum().distance/df.distance.sum()

    #no_sensed_distance = df.groupby('primary_mode').sum()['distance']

    other_distance = all_modes_distance - car_distance
    ebike_distance = mode_distances.loc['pilot_ebike'] if 'pilot_ebike' in mode_distances.index else 0
    walk_distance = mode_distances.loc['walk'] if 'walk' in mode_distances.index else 0

    #print(f"Distance in drove alone, shared ride (m): {drove_alone_distance:.1f}, {shared_ride_distance:.1f}")

    r = drove_alone_distance/shared_ride_distance
    car_proportion = car_distance/all_modes_distance
    car_to_other = car_distance/other_distance
    ebike_proportion = ebike_distance/all_modes_distance
    walk_proportion = walk_distance/all_modes_distance
    drove_alone_proportion = drove_alone_distance/all_modes_distance
    shared_ride_proportion = shared_ride_distance/all_modes_distance

    proportion_dict = {
        "r": r, "car_proportion": car_proportion, "ebike_proportion": ebike_proportion,
        "walk_proportion": walk_proportion,
        "drove_alone_proportion": drove_alone_proportion,
        "shared_ride_proportion": shared_ride_proportion,
        "drove_alone_distance": drove_alone_distance,
        "shared_ride_distance": shared_ride_distance,
        "car_to_other": car_to_other,
        "non_moto_to_moto": non_moto_to_moto
    }

    return  proportion_dict

def get_primary_modes(df,energy_dict,MODE_MAPPING_DICT):
    '''
    Finds the primary mode for each trip and appends it to df as a columnn.
    df: pandas dataframe with participant trip mode labels.
    energy_dict: dictionary by mode of energy intensities in kWH. Used to decide between modes if there are multiple primary modes.
    MODE_MAPPING_DICT: dictionary that maps possible mode labels to mode labels that are consistent with those found in energy dict.

    Returns a copy of df with a primary mode and the length of the primary mode section. 
    '''
    # Add primary mode and length columns to expanded labeled trips
    df = df.copy()
    primary_modes = []
    primary_lengths = []

    no_sections_count = 0
    for i,ct in df.iterrows():
        # Get primary mode
        if len(ct["section_distances"]) == 0: # for data up to 5-9-2022, there are 63 stage trips with no sensed sections.
            # maybe I should instead set primary mode to na 
            # and use an estimated energy consumption of 0 for these trips.
            df = df.drop(index = i)    
            no_sections_count += 1
        else:
            longest_section = max(ct["section_distances"])
            primary_mode = ct["section_modes"][ct["section_distances"]==longest_section]

            # in case there are ever tied longest sections.
            # pick the most energy intensive mode.
            if isinstance(primary_mode,list): 
                mini_energy_dict = {x:energy_dict[MODE_MAPPING_DICT[x]] for x in primary_mode}
                primary_mode = max(mini_energy_dict, key=mini_energy_dict.get)

            primary_modes.append(primary_mode)
            primary_lengths.append(longest_section)

    df['primary_mode'] = primary_modes
    df['primary_length'] = primary_lengths
    print(f"Dropped {no_sections_count} trips with no sensed sections.")
    return df

def get_outliers(df, column_of_interest, u_percentile, l_percentile):
    '''takes a dataframe and returns a subsetted dataframe with the outlier values for the column of interest.
    u_percentile: upper percentile
    l_percentile: lower percentile
    outliers will be the values above the upper percentile and below the lower percentile.
    '''
    quantiles = [u_percentile,l_percentile] 
    upper, lower = np.percentile(df[column_of_interest], quantiles)
    print(f"{quantiles[0]:.2f} and {quantiles[1]:.2f} percentiles: {upper:.2f},{lower:.2f}")

    outliers = df[(df[column_of_interest] < lower) | (df[column_of_interest] > upper)]
    return outliers

def plot_energy_consumption_by_mode(energy_consumption_df,program_name, main_mode_labels = ['drove_alone','shared_ride','walk','pilot_ebike','bus','bike','train','taxi','free_shuttle']):
    '''
    Prints a bar chart showing the expected energy consumption (EC) and the user labeled energy consumption for each user labeled mode in main_mode_labels

    energy_consumption_df: pandas dataframe with trip level estimated and actual energy consumptions
    program_name: string to label the program that energy_consumption_df is based on.
    main_mode_labels: the user labels you want to look at energy consumption estimates for.
    '''
    
    df = energy_consumption_df.copy()
    program_main_mode_labels = [x for x in main_mode_labels if x in df.mode_confirm.unique()] # 4c doesn't have train before May 2022.
    # if you want all of the labels: program_main_mode_labels = df.mode_confirm.unique()

    program_main_modes_EC = df.groupby('mode_confirm').sum().loc[program_main_mode_labels]
    program_main_modes_EC = program_main_modes_EC[['user_labeled', 'expected']] # 'predicted',

    program_main_modes_EC.plot(kind='barh')
    program_percent_error_expected = 100*relative_error(df.expected.sum(),df.user_labeled.sum())
    plt.xlabel('Energy Consumption (MWH)', fontsize=14)
    plt.legend(['user labeled', 'inferred'])
    plt.ylabel('Actual Travel Mode', fontsize=14)
    plt.title(f"Estimated energy consumption by actual mode for {program_name}",fontsize=14)# (full % error for expected: {program_percent_error_expected:.2f})")

def plot_error_by_primary_mode(df,chosen_program, r_for_dataset, r, percent_error_expected, percent_error_predicted, mean_EC_all_user_labeled, output_path):
    '''
    Prints a bar chart showing total error by sensed primary mode.

    df: pandas dataframe with trip level estimated and actual energy consumptions
    chosen_program: string of the program name that df is based on.
    r_for_dataset: the ratio of drove alone distance to shared ride distance in the dataset, based on user labels.
    r: the assumed value of r that we use for calculations. (we plan to use r = 1)
    percent_error_expected: the percent error of the expected aggregate energy consumption for this dataset.
    percent_error_predicted: the percent error of the predicted aggregate energy consumption for this dataset.
    mean_EC_all_user_labeled: the actual total energy consumption in this dataset.
    output_path: the place to store the plot.
    '''
    # Plot error totals by mode:
    mode_expected_errors = {}
    mode_predicted_errors = {}

    for mode in df.primary_mode.unique():
        if type(mode) == float: continue
        user_labeled_total = sum(df[df.primary_mode == mode]['user_labeled'])
        error_for_expected = sum(df[df.primary_mode == mode]['expected']) - user_labeled_total
        error_for_predicted = sum(df[df.primary_mode == mode]['predicted']) - user_labeled_total

        mode_expected_errors[mode] = error_for_expected
        mode_predicted_errors[mode] = error_for_predicted

    mode_expected_errors['Total'] = sum(mode_expected_errors.values())
    mode_predicted_errors['Total'] = sum(mode_predicted_errors.values())
    all_modes = list(mode_expected_errors.keys())

    fig,axs = plt.subplots(1,2)
    fig.set_figwidth(15)
    fig.set_figheight(8)

    title = f"Total energy consumption errors by mode for {chosen_program}. Dataset r = {r_for_dataset:.2f}, used r = {r:.2f}, percent errors: expected: {percent_error_expected:.2f} predicted: {percent_error_predicted:.2f}\
    \nuser labeled EC: {mean_EC_all_user_labeled:.2f}"
    fig.suptitle(title)

    axs[0].grid(axis='x')
    axs[1].grid(axis='x')

    axs[0].barh(all_modes,[mode_expected_errors[x] for x in all_modes],height=0.5)
    axs[0].set_title("Confusion based error share by primary mode")
    axs[1].barh(all_modes,[mode_predicted_errors[x] for x in all_modes],height=0.5)
    axs[1].set_title("Prediction error share by primary mode")

    #fig_file = output_path+chosen_program+"_EC_mode_total_errors_"+which_car_precision+ "_for_car_precision_info"+ "_r_from_"+which_r+ "_" +remove_outliers + "_remove_outliers"+".png"

    fig_file = output_path+chosen_program+"_EC_primary_mode_total_errors_"+"Mobilitynet_precision"+"r_from_dataset"+"keep_outliers.png"
    fig.savefig(fig_file)
    plt.close(fig)

def show_bootstrap(df,program,os_EI_moments_map,unit_dist_MCS_df):
    '''
    Prints a histogram of the bootstrap energy consumption estimates of the dataset.
    Also prints the 1 standard deviation interval based on aggregate distance with the original data.

    df: pandas dataframe with trip level estimated and actual energy consumptions
    program: string to label the program that df is based on.
    os_EI_moments_map: dictionary by operating system of EI moments dataframes. (EI moments (mean and variance) are different for android vs ios)
    unit_dist_MCS_df: dataframe containing the mean and variance of trip length for a 1 unit long trip, for both operating systems.
    '''
    print(program)
    NB = 1000
    df = df.copy()
    aggregate_EC_estimates = []
    for j in range(0,NB):
        bootstrap_sample = np.random.choice(df.expected,len(df),replace=True)
        aggregate_EC_estimates.append(sum(bootstrap_sample))
    plt.hist(aggregate_EC_estimates)
    plt.show()

    aggregate_EC_estimates = np.array(aggregate_EC_estimates)
    totals_and_errors = get_EC.get_totals_and_errors(df, os_EI_moments_map, unit_dist_MCS_df, include_autocovariance=False)
    total_expected = totals_and_errors['total_expected']
    boot_mean = np.mean(aggregate_EC_estimates)
    print(f'our estimate: {total_expected:.2f}\nTrue value: {totals_and_errors["total_user_labeled"]:.2f}\nMean of bootstrap estimates: {boot_mean:.2f}')
    sd = totals_and_errors["aggregate_sd"]
    boot_sd = np.sqrt(np.var(aggregate_EC_estimates))
    print(f'our 1 sd interval: {total_expected - sd:.2f},{total_expected + sd:.2f}')
    print(f'bootstrap 1 sd interval: {boot_mean - boot_sd:.2f},{boot_mean + boot_sd:.2f}')
    print(f'bootstrap 2 sd interval: {boot_mean - 2*boot_sd:.2f},{boot_mean + 2*boot_sd:.2f}')

def get_interval(mean,sd):
    '''Returns a list that includes the mean and the mean plus or minus one standard deviation.'''
    return [mean -sd, mean,mean + sd]

def plot_estimates_with_sd_by_program(df, os_EI_moments_map, unit_dist_MCS_df, variance_method, user_spatial_cov_map = {}):
    '''
    Prints a plot showing the user labeled aggregate EC +- 1 standard deviation and the expected aggregate EC +- 1 standard deviation.
    Allows for different methods of computing the aggregate variance.

    df: pandas dataframe with trip level estimated and actual energy consumptions
    os_EI_moments_map: dictionary by operating system of EI moments dataframes. (EI moments (mean and variance) are different for android vs ios)
    unit_dist_MCS_df: dataframe containing the mean and variance of trip length for a 1 unit long trip, for both operating systems.
    variance_method: 
        'aggregate_section_distances': calculate variance by finding total distance in each mode using sections distances
        'aggregate_primary_mode_distances': calculate variance by finding total distance in each mode using primary mode distances
        'spatial_cov': calculate variance by summing individual trip variances and adding a spatial covariance term.
        any other string: calculate variance by summing individual trip variances.
    title_extension: string to append to the title.
    user_spatial_cov_map: dictionary by user id of spatial covariance of trip level energy consumption.
    Returns the number of standard deviations the expected aggregate EC is from the truth. 
    '''
    
    df = df.copy()
    fig,axs = plt.subplots(3,3)
    fig.set_figwidth(25)
    fig.set_figheight(15)
    j = 0
    ax = axs.ravel() # flatten the axs object to 1D
    program_n_sd_map = {}

    if (variance_method == 'spatial_cov') and (len(user_spatial_cov_map) != len(df.user_id.unique())):
        print('Invalid spatial covariance map.')
        return

    # Each program is plotted in the loop, then afterwards we plot for all programs.
    for program in df.program.unique():
        program_df = df[df.program == program]

        if variance_method == 'aggregate_section_distances':
            aggregate_var, _ = get_EC.compute_aggregate_variance_with_total_distance_from_sections(program_df, os_EI_moments_map, unit_dist_MCS_df)
            aggregate_sd = np.sqrt(aggregate_var)
        elif variance_method == 'aggregate_primary_mode_distances':
            aggregate_var = get_EC.compute_aggregate_variance_by_primary_mode(program_df, os_EI_moments_map, unit_dist_MCS_df)
            aggregate_sd = np.sqrt(aggregate_var)
        elif variance_method == 'spatial_cov':
            # Treat individual trips separately and as spatially dependent.
            aggregate_sd = np.sqrt(get_EC.compute_variance_including_spatial_cov_for_trips_dataframe(program_df,user_spatial_cov_map))
        else:
            # Treat individual trips separately and as independent.
            aggregate_sd = np.sqrt(program_df.confusion_var.sum())
        
        user_sd = np.sqrt(program_df.user_var.sum())
        user_labeled = program_df.user_labeled.sum()
        expected = program_df.expected.sum()

        x = [0,0,0]
        y = get_interval(user_labeled,user_sd)
        ax[j].scatter(x,y, c = 'tab:blue', marker= 'o')
        
        n_sd = abs(program_df.expected.sum() - program_df.user_labeled.sum())/aggregate_sd
        program_n_sd_map[program] = round(n_sd, 2)

        x = [1,1,1]
        y = get_interval(expected,aggregate_sd)
        ax[j].scatter(x,y, c = 'tab:orange', marker= 'o')
        ax[j].set_ylim([0,max(1.1*user_labeled,1.1*max(y))]) # make sure not to cut off the top of the plot.

        ax[j].set_xlabel(program, fontsize = 15)
        ax[j].set_ylabel('Energy consumption (kWH)', fontsize = 10)
        j+=1

    program = 'all'
    program_df = df.copy()
    if variance_method == 'aggregate_section_distances':
        aggregate_var, _ = get_EC.compute_aggregate_variance_with_total_distance_from_sections(program_df, os_EI_moments_map, unit_dist_MCS_df)
        aggregate_sd = np.sqrt(aggregate_var)
        title_extension = '\nVariance computed by aggregating section distances by sensed mode.'
    elif variance_method == 'aggregate_primary_mode_distances':
        aggregate_var = get_EC.compute_aggregate_variance_by_primary_mode(program_df, os_EI_moments_map, unit_dist_MCS_df)
        aggregate_sd = np.sqrt(aggregate_var)
        title_extension = '\nVariance computed by aggregating trip distances by sensed primary mode.'
    elif variance_method == 'spatial_cov':
        # Treat individual trips separately and as spatially dependent.
        aggregate_sd = np.sqrt(get_EC.compute_variance_including_spatial_cov_for_trips_dataframe(program_df,user_spatial_cov_map))
        title_extension = '\nVariance computed as sum of individual trip variances and a spatial covariance term.'
    else:
        # Treat individual trips separately and as independent.
        aggregate_sd = np.sqrt(program_df.confusion_var.sum())
        title_extension = '\nVariance computed as sum of individual trip variances.'
    
    fig.suptitle(f'Total energy consumption by program based on user label (left) or expected value (right).\nDisplayed as mean +- 1 standard deviation.{title_extension}', fontsize=20)

    user_sd = np.sqrt(program_df.user_var.sum())
    user_labeled = program_df.user_labeled.sum()
    expected = program_df.expected.sum()

    x = [0,0,0]
    y = get_interval(user_labeled,user_sd)
    ax[j].scatter(x,y, c = 'tab:blue', marker= 'o')
    
    n_sd = abs(program_df.expected.sum() - program_df.user_labeled.sum())/aggregate_sd
    program_n_sd_map[program] = round(n_sd, 2)

    x = [1,1,1]
    y = get_interval(expected,aggregate_sd)
    ax[j].scatter(x,y, c = 'tab:orange', marker= 'o')

    ax[j].set_ylim([0,max(1.1*user_labeled,1.1*max(y))])  # make sure not to cut off the top of the plot.

    ax[j].set_xlabel(program, fontsize = 15)
    ax[j].set_ylabel('Energy consumption (kWH)', fontsize = 10)

    return program_n_sd_map

def get_program_percent_error_map(df, estimate_type):
    '''
    Computes the aggregate energy consumption expected value percent error for each program

    df: pandas dataframe with trip level estimated and actual energy consumptions
    estimate_type: string, either 'expected' or 'predicted'

    Returns a dictionary by program of percent errors.
    '''
    df = df.copy()
    program_percent_error_map = {}

    for program in df.program.unique():
        program_df = df[df.program == program]

        user_labeled = program_df.user_labeled.sum()
        expected = program_df[estimate_type].sum()
        program_percent_error_map[program] = 100*relative_error(expected,user_labeled)

    program = 'all'
    program_df = df.copy()
    user_labeled = program_df.user_labeled.sum()
    expected = program_df[estimate_type].sum()
    program_percent_error_map[program] = round(100*relative_error(expected,user_labeled),2)

    return program_percent_error_map

def plot_aggregate_EC_bar_chart(df):
    '''
    Prints a bar chart of total expected and actual energy consumption for each program in df.

    df: pandas dataframe with trip level estimated and actual energy consumptions
    '''
    fig,axs = plt.subplots(2,4)
    fig.set_figwidth(25)
    fig.set_figheight(15)
    fig.suptitle('Total energy consumption in kWH by program based on user labels (left) or expected value (right)', fontsize=30)
    j = 0
    ax = axs.ravel() # flatten the axs object to 1D
    x_labels = ['user labeled', 'expected']
    bar_colors = ['tab:green','tab:blue']

    for program in df.program.unique():
        program_df = df[df.program == program]

        user_labeled = program_df.user_labeled.sum()
        expected = program_df.expected.sum()

        heights = [user_labeled,expected]
        ax[j].bar(x_labels,heights, color = bar_colors, width = 0.5)

        ax[j].set_xlabel(program, fontsize=16)
        #ax[j].set_ylabel('Energy consumption (kWH)', fontsize=16)
        j+=1

    program = 'all'
    user_labeled = df.user_labeled.sum()
    expected = df.expected.sum()

    heights =[user_labeled,expected]
    ax[j].bar(x_labels,heights, color = bar_colors, width = 0.5)

    ax[j].set_xlabel(program, fontsize=16)
    #ax[j].set_ylabel('Energy consumption (kWH)', fontsize=16)
    plt.show()

def construct_mostly_ebike_df(df):
    df = df.copy()
    all_ebike_trips = df[df.mode_confirm == 'pilot_ebike'].copy()
    not_ebike_trips = df[df.mode_confirm != 'pilot_ebike'].copy()
    n_trips_over_2 = int(np.floor(len(df)/2))
    ebike_trip_list = np.random.choice(all_ebike_trips._id,len(df))
    other_trip_list = np.random.choice(not_ebike_trips._id,n_trips_over_2)
    half_ebike_trips = np.concatenate((ebike_trip_list, other_trip_list))

    # Construct a dataframe the size of ceo with 50% of the trips being ebike
    half_ebike_trips_idx = df[df._id.isin(half_ebike_trips)].index
    return df.loc[half_ebike_trips_idx]

def prior_mode_distribution_sensitivity_analysis(df, prior_mode_distributions_map, android_confusion, ios_confusion, unit_dist_MCS_df, energy_dict, EI_length_cov=0):
    '''
    df: an expanded labeled trips dataframe with primary modes
    prior_mode_distributions: dictionary by name you assign to the prior of {dictionaries representing assumed prior probabilities of each mode}.
    android_confusion: the confusion matrix associated with sensing on android phones
    ios_confusion: the confusion matrix associated with sensing on ios phones
    unit_dist_MCS_df: dataframe containing the mean and variance of trip length for a 1 unit long trip, for both operating systems.
    energy_dict: dictionary by mode of energy intensities in kWH.

    Returns: prior_and_error_dataframe: dataframe with name of the prior, percent error, and number of standard deviations the error is composed of for `df` when fully sensed
        prior_name_energy_dataframe_map: an energy consumption dataframe for each named prior.
    '''
    df = df.copy()
    prior_name_energy_dataframe_map = {}
    prior_and_error_dataframe = pd.DataFrame(columns=["Prior Name", "Percent Error", "Estimated Standard Deviation (SD)",
            "Number of Standard Deviations to Truth"])

    for prior_name in prior_mode_distributions_map.keys():

        # construct EI moments df.
        if prior_name == 'MobilityNet Specific to OS':
            prior_probs_android = android_confusion.sum(axis=1)/android_confusion.sum().sum()
            prior_probs_ios = ios_confusion.sum(axis=1)/ios_confusion.sum().sum()

            android_EI_moments_df = cm_handling.get_Bayesian_conditional_EI_expectation_and_variance(android_confusion,energy_dict, prior_probs_android)
            ios_EI_moments_df = cm_handling.get_Bayesian_conditional_EI_expectation_and_variance(ios_confusion,energy_dict, prior_probs_ios)
        elif prior_name == 'No Bayes Update':
            android_EI_moments_df = cm_handling.get_conditional_EI_expectation_and_variance(android_confusion,energy_dict)
            ios_EI_moments_df = cm_handling.get_conditional_EI_expectation_and_variance(ios_confusion,energy_dict)
        else:
            prior_probs = prior_mode_distributions_map[prior_name]
            android_EI_moments_df = cm_handling.get_Bayesian_conditional_EI_expectation_and_variance(android_confusion,energy_dict, prior_probs)
            ios_EI_moments_df = cm_handling.get_Bayesian_conditional_EI_expectation_and_variance(ios_confusion,energy_dict, prior_probs)

        # calculate energy consumption.
        print(f"{prior_name}")
        energy_consumption_df = get_EC.compute_all_EC_values(df,
            unit_dist_MCS_df, 
            energy_dict,
            android_EI_moments_df,
            ios_EI_moments_df, 
            EI_length_cov, print_info=False)

        prior_name_energy_dataframe_map[prior_name] = energy_consumption_df 

        # calculate error for the current energy consumption df.
        os_EI_moments_map = {'ios': ios_EI_moments_df, 'android': android_EI_moments_df}
        aggregate_var, _ = get_EC.compute_aggregate_variance_with_total_distance_from_sections(energy_consumption_df, os_EI_moments_map, unit_dist_MCS_df)
        aggregate_sd = np.sqrt(aggregate_var)
        total_expected = energy_consumption_df.expected.sum()
        total_user_labeled = energy_consumption_df.user_labeled.sum()
        error_over_sd = abs(total_expected - total_user_labeled)/aggregate_sd
        percent_error = 100*relative_error(total_expected, total_user_labeled)

        prior_and_error_dataframe = prior_and_error_dataframe.append({"Prior Name": prior_name, "Percent Error": percent_error, "Estimated Standard Deviation (SD)": aggregate_sd,
            "Number of Standard Deviations to Truth": error_over_sd}, ignore_index = True)
        
    return prior_and_error_dataframe, prior_name_energy_dataframe_map

def construct_prior_dict(prior_probs_prespecified, available_ground_truth_modes):
    '''
    Constructs a map of ground truth modes and their assumed prior probabilities.

    prior_probs_prespecified: dictionary by mode of the prior probability that the mode occurs. 
        This can be shorter than the number of ground truth modes. 
        The remaining modes that you did not specify prior probabilities for will be assigned equal shares of the remaining probability.
    available_ground_truth_modes: list of modes that can be found in the confusion matrix.
    
    Returns prior_probs, a dictionary by mode of prior probabilities for all modes found in `available_ground_truth_modes`
    '''

    n_other_modes = len(available_ground_truth_modes) - len(prior_probs_prespecified)

    if n_other_modes < 0:
        print("Error: More mode probabilities were specified than the number of available ground truth modes.")
        return
    elif n_other_modes == 0:
        prior_probs = prior_probs_prespecified
    elif len(prior_probs_prespecified) > 0:
        prior_probs = prior_probs_prespecified.copy()
        probability_remaining = 1 - sum(prior_probs_prespecified.values())
        prior_probs.update({x: probability_remaining/n_other_modes for x in available_ground_truth_modes if x not in prior_probs_prespecified.keys()})
    else:
        prior_probs = {x: 1/n_other_modes for x in available_ground_truth_modes}

    return prior_probs

def print_top_mode_confirm_proportions(expanded_labeled_trips):
    all_mode_distances = expanded_labeled_trips.groupby('mode_confirm').sum().distance_miles
    all_mode_distance_proportions = all_mode_distances.divide(sum(expanded_labeled_trips.distance_miles))
    print(all_mode_distance_proportions.sort_values(ascending=False)[0:10].round(4).to_latex())

# this version of show_bootstrap shows the distribution of errors rather than of expected values.
def show_bootstrap_error_distribution(df,program,os_EI_moments_map,unit_dist_MCS_df, print_results):
    print(program)
    NB = 300
    df = df.copy()
    df = df.set_index("_id")
    aggregate_EC_estimates = []
    aggregate_EC_actual = []
    sd_list = []
    for j in range(0,NB):
        bootstrap_idx = np.random.choice(df.index,len(df),replace=True)
        bootstrap_sample = df.loc[bootstrap_idx]
        aggregate_EC_estimates.append(sum(bootstrap_sample.expected))
        aggregate_EC_actual.append(sum(bootstrap_sample.user_labeled))
        sd_list.append(get_EC.get_totals_and_errors(df, os_EI_moments_map, unit_dist_MCS_df, include_autocovariance=False)['aggregate_sd'])

    aggregate_EC_estimates = np.array(aggregate_EC_estimates)
    aggregate_EC_actual = np.array(aggregate_EC_actual)
    errors = aggregate_EC_estimates - aggregate_EC_actual

    totals_and_errors = get_EC.get_totals_and_errors(df, os_EI_moments_map, unit_dist_MCS_df, include_autocovariance=False)
    total_expected = totals_and_errors['total_expected']
    boot_mean = np.mean(aggregate_EC_estimates)
    sd = totals_and_errors["aggregate_sd"]
    boot_sd = np.sqrt(np.var(aggregate_EC_estimates))

    if print_results == True:
        plt.hist(errors)

        print(f'our estimate: {total_expected:.2f}\nTrue value: {totals_and_errors["total_user_labeled"]:.2f}\nMean of bootstrap estimates: {boot_mean:.2f}')
        print(f'our error: {sum(energy_consumption_df.expected - energy_consumption_df.user_labeled):.2f}')

        print(f'our 1 sd interval: {total_expected - sd:.2f},{total_expected + sd:.2f}')
        print(f'bootstrap 1 sd interval: {boot_mean - boot_sd:.2f},{boot_mean + boot_sd:.2f}')
        print(f'bootstrap 2 sd interval: {boot_mean - 2*boot_sd:.2f},{boot_mean + 2*boot_sd:.2f}')

    # I want to know: how does the error compare to the standard deviation each time?
    return abs(errors)/np.array(sd_list)

#show_bootstrap(energy_consumption_df,'all', os_EI_moments_map, unit_dist_MCS_df)
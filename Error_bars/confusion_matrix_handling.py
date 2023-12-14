import numpy as np
import pandas as pd


def drop_rows_and_columns(df,row_list,column_list):
    '''
    Drops the rows/columns with labels in row_list/column_list from df.
    Args: df, row_list, column_list
    '''
    df = df.copy()
    df = df.drop(labels = row_list, axis = 0)
    df = df.drop(labels = column_list, axis = 1)
    return df

def get_energy_dict(energy_intensity_dataframe, units='kWH'):
    '''
    energy_intensity_dataframe: dataframe based on energy_intensity.csv, after converting to kilowatt hours.
    units: (string) kWH or MWH 

    Returns a dictionary by mode of energy intensity in kWH/PMT
    '''
    if units== 'kWH':
        scaling_factor = 1  
    elif units=='MWH':
        scaling_factor = 1e-3
    else:
        print("Error: That choice of units is not supported yet.")
        return

    energy_dict = {}
    for _,row in energy_intensity_dataframe.iterrows():
        # Convert to kWH/PMT if the energy intensity is not already in kWH/PMT.
        energy_intensity_kWH = row["energy_intensity_factor"] * 0.000293071 if row["fuel"] not in ["electric","human_powered"] else row["energy_intensity_factor"]
        energy_dict[row['mode']] = energy_intensity_kWH*scaling_factor

    # Add 'no_gt'
    energy_dict['no_gt'] = 0
    return energy_dict

def collapse_confusion_matrix(df, 
        rows_to_collapse = {"no_gt": ["no_gt_start","no_gt_middle", "no_gt_end"]},
        columns_to_collapse = {"no_sensed": ["no_sensed_start","no_sensed_middle","no_sensed_end"]}
    ):
    '''
    Merges rows or columns of the confusion matrix by addition. 
    Used to combine the no sensed columns and the no ground truth columns into 1.

    rows_to_collapse: dictionary where the key is the row name you want to assign 
        to the row that results when you combine the rows associated with that key.
    columns_to_collapse: dictionary where the key is the column name you want to assign 
        to the column that results when you combine the columns associated with that key.
    
    Returns a copy of the confusion matrix with fewer rows or columns.
    '''
    # Other ideas:
    # add a multi-index and group by that.
    # leave the confusion as is and just compute expected value and variance anyway.

    df = df.copy()

    # Add together the rows we want in one row and drop the original split rows
    for combined_row in rows_to_collapse:
        #df.loc[combined_row] = sum([df.loc[x] for x in rows_to_collapse[combined_row]])
        if len(rows_to_collapse[combined_row]) > 1:
            temp = sum([df.loc[x] for x in rows_to_collapse[combined_row]])
        else: 
            temp = df.loc[rows_to_collapse[combined_row]].sum()
        df = df.drop(labels = rows_to_collapse[combined_row], axis = 0)
        df.loc[combined_row] = temp

    # Add together the cols we want in one col and drop the original split cols
    for combined_col in columns_to_collapse:
        # eg t['no_sensed'] = sum([t[x] for x in ["no_start","no_middle","no_end"]])
        df[combined_col] = sum([df[x] for x in columns_to_collapse[combined_col]])
        df = df.drop(labels = columns_to_collapse[combined_col], axis = 1)
    return df


def expectation(probs,values):
    # Takes two lists, probabilities and values, and returns the expected value.
    return sum(probs*values)

def get_Bayesian_conditional_EI_expectation_and_variance(collapsed_confusion_matrix, energy_dict, prior_mode_probs):
    '''
    Finds the probability of each ground truth mode conditional on the predicted mode by updating from 
    the prior probability of each ground truth mode and using P(predicted | ground truth) from the confusion matrix.

    collapsed_confusion_matrix: confusion matrix dataframe where train, no_gt, and no_sensed are placed in 1 row or column.
        ground truth modes are the rows, predicted modes are the columns.
    energy_dict: dictionary by mode of energy intensity in kWH/PMT. The keys have to match with the rows of the confusion matrix.
    prior_mode_probs: the assumed prior probabilities of a trip being in each mode.

    Returns a dataframe with mean and variance of energy intensity as columns and predicted mode as row labels.
    '''
    # Divide each row by its row sum.
    p_predicted_given_actual = collapsed_confusion_matrix.divide(collapsed_confusion_matrix.sum(axis=1), axis='rows')

    # temporary until I have the actual confusion matrix.
    #p_predicted_given_actual.loc['Pilot ebike','car'] = 0.40
    #p_predicted_given_actual.loc['Pilot ebike','bicycling'] -= 0.40

    # Find the numerator of Bayes rule for each (ground truth, inferred) confusion matrix entry
    likelihood_times_priors = p_predicted_given_actual.multiply(pd.Series(prior_mode_probs), axis='rows')

    normalizing_constants = likelihood_times_priors.sum(axis='rows')
    prob_actual_given_predicted_df = likelihood_times_priors.divide(normalizing_constants, axis='columns').copy()

    #print(prob_actual_given_predicted_df)
    energy_intensities = np.array([energy_dict[mode] for mode in prob_actual_given_predicted_df.index]) # this will place each intensity in the same order as it appears in the confusion matrix.

    # Compute expected energy intensities given predicted mode. X stands for energy intensity.
    # Find an expected energy intensity for each column (predicted mode)
    E_X = np.array([expectation(prob_actual_given_predicted_df[col], energy_intensities) for col in prob_actual_given_predicted_df.columns]) 

    # Compute variances
    sqr_EIs = energy_intensities**2
    E_X2 = np.array([expectation(prob_actual_given_predicted_df[col], sqr_EIs) for col in prob_actual_given_predicted_df.columns])
    V_X = E_X2 - E_X**2   # Var(X) = E[X^2] - (E[X])^2. Here this is an element-wise difference of lists.

    # Place these into a dataframe.
    EI_expectations_and_vars = pd.DataFrame({"mean(EI)": E_X, "variance(EI)": V_X})
    return EI_expectations_and_vars.set_index(keys = prob_actual_given_predicted_df.columns)

def get_conditional_EI_expectation_and_variance(df, energy_dict):
    '''
    Finds the probability of each ground truth mode conditional on the predicted mode without a Bayes update using the confusion matrix.

    collapsed_confusion_matrix: confusion matrix dataframe where train, no_gt, and no_sensed are placed in 1 row or column.
        ground truth modes are the rows, predicted modes are the columns.
    energy_dict: dictionary by mode of energy intensity in kWH/PMT. The keys have to match with the rows of the confusion matrix.

    Returns a dataframe with mean and variance of energy intensity as columns and predicted mode as row labels.
    '''

    df = df.copy()

    # Experimenting with ebike mispredicted as car accounted for, since MobilityNet doesn't have cases of it as of Feb 16, 2023.
    #duration_sensed_as_car_given_actual_ebike = 0.4*df.loc['Pilot ebike'].sum()
    #df.at['Pilot ebike','bicycling'] -=duration_sensed_as_car_given_actual_ebike
    #df.at['Pilot ebike','car'] += duration_sensed_as_car_given_actual_ebike
    column_normd_matrix = df/df.sum(axis=0) # divide the entries in each column by the corresponding column sum
    energy_intensities = np.array([energy_dict[mode] for mode in df.index]) # this will place each intensity in the same order as it appears in the confusion matrix.

    # Compute expected energy intensities given predicted mode. X stands for energy intensity.
    E_X = np.array([expectation(column_normd_matrix[col], energy_intensities) for col in df.columns])  # gives an expected energy intensity given each predicted mode.

    # Compute variances
    sqr_EIs = energy_intensities**2
    E_X2 = np.array([expectation(column_normd_matrix[col], sqr_EIs) for col in df.columns])
    V_X = E_X2 - E_X**2   # Var(X) = E[X^2] - (E[X])^2. Here this is an element-wise difference of lists.

    # Place these into a dataframe.
    EI_expectations_and_vars = pd.DataFrame({"mean(EI)": E_X, "variance(EI)": V_X})
    return EI_expectations_and_vars.set_index(keys = df.columns)


def change_precision(confusion_mat,mode,new_precision):
    '''
    This was for experimenting with what happens with different precisions (P(ground truth car|predict car))
    No longer in use.
    '''
    cm = confusion_mat.copy()
    mode_prediction_total = cm[mode].sum()

    # Let px = the new precision for mode x.
    # Let i be the row index within a column.
    # The new values in the mode_x_column should be:
    # If i = x: px
    # If i != x: (1-px)*mode_x_column[i]/sum_{k != x}(mode_x_column[k])

    # for sensing, column modes are listed as they are labeled when sensed.
    # row modes are listed as modes for which we have an energy intensity.
    mode_col = cm[mode]     
    predicted_mode_row_name = MODE_MAPPING_DICT[mode] if mode != 'car' else 'Car, sensed'

    # Zero out the row entry for the chosen mode.               
    mode_col[predicted_mode_row_name] = 0

    # Find the proportions that the other modes contribute.
    mode_col = mode_col/sum(mode_col)
    mode_col = (1-new_precision)*mode_col

    # New conditional probabilities
    mode_col[predicted_mode_row_name] = new_precision

    # Update the confusion matrix, reverting to counts or durations, 
    # since get_conditional_EI_expectation_and_variance expects counts or durations in the confusion matrix.
    cm[mode] = mode_prediction_total*mode_col

    return cm


# See the end of store_errors.ipynb for how this is created.
MODE_MAPPING_DICT = {'drove_alone': 'Gas Car, drove alone',
 'e_car_drove_alone': 'E-car, drove alone',
 'work_vehicle': 'Gas Car, drove alone',
 'bus': 'Bus',
 'train': 'Train',
 'free_shuttle': 'Free Shuttle',
 'train,_bus and walk': 'Train',
 'train_and pilot e-bike': 'Train',
 'taxi': 'Taxi/Uber/Lyft',
 'friend_picked me up': 'Gas Car, with others',
 'carpool_w/ friend to work': 'Gas Car, with others',
 'friend_carpool to work': 'Gas Car, with others',
 'carpool_to work': 'Gas Car, with others',
 'friend/co_worker carpool': 'Gas Car, with others',
 'carpool_to lunch': 'Gas Car, with others',
 'carpool': 'Gas Car, with others',
 'carpool_for lunch': 'Gas Car, with others',
 'carpool_lunch': 'Gas Car, with others',
 'shared_ride': 'Gas Car, with others',
 'e_car_shared_ride': 'E-car, with others',
 'bikeshare': 'Bikeshare',
 'scootershare': 'Scooter share',
 'pilot_ebike': 'Pilot ebike',
 'e-bike': 'Pilot ebike',
 'walk': 'Walk',
 'skateboard': 'Skate board',
 'bike': 'Regular Bike',
 'the_friend who drives us to work was running errands after the shift before dropping me off. not a trip of mine.': 'Not a Trip',
 'not_a_trip': 'Not a Trip',
 'no_travel': 'No Travel',
 'same_mode': 'Same Mode',
 'Bike': 'Regular Bike',
 'Drove Alone': 'Gas Car, drove alone',
 'Shared Ride': 'Gas Car, with others',
 'Air': 'Air',
 'Gas Car, drove alone': 'Gas Car, drove alone',
 'Gas Car, with others': 'Gas Car, with others',
 'E-car, drove alone': 'E-car, drove alone',
 'E-car, with others': 'E-car, with others',
 'Taxi/Uber/Lyft': 'Taxi/Uber/Lyft',
 'Bus': 'Bus',
 'Free Shuttle': 'Free Shuttle',
 'Train': 'Train',
 'Scooter share': 'Scooter share',
 'Pilot ebike': 'Pilot ebike',
 'Bikeshare': 'Bikeshare',
 'Walk': 'Walk',
 'Skate board': 'Skate board',
 'Regular Bike': 'Regular Bike',
 'Not a Trip': 'Not a Trip',
 'No Travel': 'No Travel',
 'air': 'Air',
 'car': 'Gas Car, drove alone',
 'electric_vehicle': 'E-car, drove alone',
 'skiing': 'Walk',
 'snowboarding': 'Walk',
 'subway': 'Train',
 'walking': 'Walk',
 'bicycling': 'Regular Bike',
 'escooter': 'Scooter share',
 'ebike': 'Pilot ebike',
 'e_bike': 'Pilot ebike',
 'light_rail': 'Train',
 'no_gt': 'no_gt',
 'air_or_hsr': 'Train',
 'no_sensed': 'Not a Trip',
 'sensed_car': 'Car, sensed'}

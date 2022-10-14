import numpy as np
import pandas as pd


def get_energy_dict(energy_intensity_dataframe):
    # Takes the energy intensity.csv dataframe.
    # Returns a dictionary of energy intensity for each mode in kWH/PMT
    energy_dict = {}
    for _,row in energy_intensity_dataframe.iterrows():
        # Convert to kWH if needed. Should I round down to 4 sig figs?
        energy_intensity_kWH = row["energy_intensity_factor"] * 0.000293071 if row["fuel"] not in ["electric","human_powered"] else row["energy_intensity_factor"]
        energy_dict[row['mode']] = energy_intensity_kWH

    # Add 'no_gt'
    energy_dict['no_gt'] = 0
    return energy_dict

def collapse_confusion_matrix(df, 
        rows_to_collapse = {"no_gt": ["no_gt_start","no_gt_middle", "no_gt_end"]},
        columns_to_collapse = {"no_sensed": ["no_start","no_middle","no_end"]}
    ):
    # Other ideas:
    # add a multi-index and group by that.
    # leave the confusion as is and just compute expected value and variance anyway.

    df = df.copy()

    # Add together the rows we want in one row and drop the original split rows
    for combined_row in rows_to_collapse:
        df.loc[combined_row] = sum([df.loc[x] for x in rows_to_collapse[combined_row]])
        df = df.drop(labels = rows_to_collapse[combined_row], axis = 0)

    # Add together the cols we want in one col and drop the original split cols
    for combined_col in columns_to_collapse:
        # eg t['no_sensed'] = sum([t[x] for x in ["no_start","no_middle","no_end"]])
        df[combined_col] = sum([df[x] for x in columns_to_collapse[combined_col]])
        df = df.drop(labels = columns_to_collapse[combined_col], axis = 1)
    return df



def rename_ground_truth(confusion_matrix, mode_dict):
    #t.rename(mapper= dic_mode, axis="index")
    return True

def expectation(probs,values):
    # Takes two lists, probabilities and values, and returns the expected value.
    return sum(probs*values)

def get_conditional_EI_expectation_and_variance(df, energy_dict):
    df = df.copy()
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



MODE_MAPPING_DICT = {
    'drove_alone': 'Gas Car, drove alone',
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
    'pilot_ebike': 'E-bike',
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
    'light_rail': 'Train',
    'no_gt': 'no_gt'
 }

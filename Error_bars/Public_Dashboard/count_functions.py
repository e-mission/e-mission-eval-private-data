# Final mode functions
def sensed_mode(mode):
    # When selecting a mode, use the value accepted by the energy intensity dataframe
    # Below is from dic_mode
    '''{'drove_alone': 'Car, drove alone',
             'work_vehicle': 'Car, drove alone',
             'bus': 'Bus',
             'train': 'Train',
             'free_shuttle': 'Free Shuttle',
             'train,_bus and walk': 'Train',
             'train_and pilot e-bike': 'Train',
             'taxi': 'Taxi/Uber/Lyft',
             'friend_picked me up': 'Car, with others',
             'carpool_w/ friend to work': 'Car, with others',
             'friend_carpool to work': 'Car, with others',
             'carpool_to work': 'Car, with others',
             'friend/co_worker carpool': 'Car, with others',
             'carpool_to lunch': 'Car, with others',
             'carpool': 'Car, with others',
             'carpool_for lunch': 'Car, with others',
             'carpool_lunch': 'Car, with others',
             'shared_ride': 'Car, with others',
             'bikeshare': 'Bikeshare',
             'scootershare': 'Scooter share',
             'pilot_ebike': 'Pilot ebike',
             'walk': 'Walk',
             'skateboard': 'Skate board',
             'bike': 'Regular Bike',
             'the_friend who drives us to work was running errands after the shift before dropping me off. not a trip of mine.': 'Not a Trip',
             'not_a_trip': 'Not a Trip',
             'no_travel': 'No Travel',
             'same_mode': 'Same Mode',
             nan: 'nan'})'''

    if mode == "unknown" or mode == "UNKNOWN":       # How should unknown and other be handled?
        return "Not a trip"
    elif mode == "walking" or mode == "WALKING": 
        return "Walk"
    elif mode == "Regular Bike" or mode == "CYCLING":   # expand to bike, ebike, escooter
        return "Bike"
    elif mode ==  "bus" or mode == "BUS": 
        return "Bus"
    elif mode == "train" or mode == "TRAIN":     # collapsed with subway, tram, light rail because we lack intensities for those
        return "Train"
    elif mode ==  "car" or mode == "CAR":      # expand to Car, drove alone and Car, with others
        return "Car, drove alone"
    elif mode ==  "air_or_hsr" or mode == "AIR_OR_HSR":   # keep as air for now
        return "air"
    elif mode == "subway" or mode == "SUBWAY": 
        return "Train"
    elif mode == "tram" or mode == "TRAM": 
        return "Train"
    elif mode == "light_rail" or mode == "LIGHT_RAIL":
        return "Train"
    else:
        Warning("Sensed mode had a different label than expected")

    '''sensed_mode_types = {0: "unknown", 1: "walking",2: "bicycling",
                    3: "bus", 4: "train", 5: "car", 6: "air_or_hsr",
                    7: "subway", 8: "tram", 9: "light_rail"}'''

    '''UNKNOWN = 0 Other
    WALKING = 1  Walk
    BICYCLING = 2 Bike
    BUS = 3       Bus 
    TRAIN = 4     Train
    CAR = 5       Drove Alone
    AIR_OR_HSR = 6 Air
    SUBWAY = 7     Train
    TRAM = 8        Train
    LIGHT_RAIL = 9  Train'''

def get_final_mode(trip):

    if not trip['Mode_confirm'] == 'nan':
        return trip["Mode_confirm"]

    if  trip["algorithm_chosen"] == "sensing" and len(trip["sensed_mode"]) > 0:
        sensed_label = max(trip["sensed_mode"], key=trip["sensed_mode"].get)
        final_mode = sensed_mode(sensed_label)
    else:  
        final_mode = trip["la_mode"]

    return final_mode

def get_footprints_by_mode(df,footprint_col_name,mode_column):
    '''Returns a dictionary of total footprint values for each mode. Also includes an overall total.'''
    footprints_by_mode = {}
    for mode in df[mode_column].unique():
        single_mode_df = df[df[mode_column] == mode]
        footprints_by_mode[mode] = single_mode_df[footprint_col_name].sum()

    total = sum(footprints_by_mode.values())
    footprints_by_mode["Total"] = total
    return footprints_by_mode

def get_inferred_footprint_intervals(footprint_dict,footprint_rel_errors):
    '''
    Calculate confidence intervals for footprint metrics
    footprint_dict: a dictionary by mode of footprint values (carbon or energy)
    footprint_rel_errors: a dictionary by mode containing upper and lower relative errors.\
        it is a subdictionary in the Error Rates collection document, under the name 'carbon' or 'energy'
    '''
    inferred_footprint_intervals = {}
    for mode in footprint_dict:
        if mode == "Other": continue

        estimate = footprint_dict[mode]
        if estimate != 0:
            lower_rel_error = footprint_rel_errors[mode][0]
            upper_rel_error = footprint_rel_errors[mode][1]
            footprint_interval = [estimate*(1 + lower_rel_error), estimate*(1+upper_rel_error)]
        else:
            footprint_interval = [0,0]

        inferred_footprint_intervals[mode]= {"estimate": estimate, "lower": footprint_interval[0], "upper": footprint_interval[1]}

    return inferred_footprint_intervals

def make_rel_errors(max_size):
    import numpy as np
    upper= np.random.rand(1)*max_size
    lower = -np.random.rand(1)*max_size
    return  [float(lower),float(upper)]

def construct_mock_errors():
    
    import emission.core.get_database as edb

    Error_Rates = edb.get_Error_Rates_db()
    # mode count intervals, (purpose count intervals later), carbon intervals, energy intervals
    # Maybe eventually provide the bootstrap distributions?
    modes = ['Bikeshare','Walk', 'Regular Bike', 'Pilot ebike', 'Scooter share', 'Car, drove alone', 
            'Car, with others', 'Taxi/Uber/Lyft', 'Bus', 'Train', 'Free Shuttle', 
            'Air', 'Other', 'Not a Trip','Total']
    mode_rel_errors = {}
    carbon_rel_errors = {}
    energy_rel_errors = {}
    for m in modes:
        if m is not 'Total':
            mode_rel_errors[m] = make_rel_errors(0.15)
        carbon_rel_errors[m] = make_rel_errors(0.2)
        energy_rel_errors[m] = make_rel_errors(0.2)

    mode_rel_errors['Not a Trip'] = make_rel_errors(0.15)

    error_dictionary = {}
    error_dictionary["counts"] = {}
    error_dictionary["counts"]["mode_confirm"] = mode_rel_errors
    error_dictionary["carbon"] = carbon_rel_errors
    error_dictionary["energy"] = energy_rel_errors
    error_dictionary["distance"] = {"interval": make_rel_errors(0.05), "mean": 0.01, "variance":0.02}
    Error_Rates.drop()
    Error_Rates.insert_one(error_dictionary)
## This folder contains notebooks used to test the energy consumption mean and variance estimation.

0. Get access to and mongorestore OpenPATH data. See the emission readme for more instructions.
1. Run add_sensed_sections_to_confirmed_trips.py 
2. Run preliminary notebooks.


### First notebooks and scripts to run:
##### add_sensed_sections_to_confirmed_trips.py 
    adds "section_modes" and "section_distances" fields to confirmed trip documents in the database.
##### check_for_sections.ipynb
    checks whether we've added section_modes and section_distances to all confirmed trips in the database

#### If you need to make changes to the confusion matrices or distance errors you use:
Otherwise, the confusion matrices and distance mean and variance are in:
- android_confusion.csv, ios_confusion.csv, unit_dist_MCS.csv
##### From mobilitynet: classification_analysis.ipynb and trajectory_distance_eval
    Use these to %store the error characteristics for mode and distance.
##### store_errors.ipynb
    Preps errors for use in this folder.
##### store_expanded_labeled_trips.ipynb
Takes 6-14 minutes on the full CEO dataset + stage + prepilot. Every other notebook below uses expanded_labeled_trips. 
This notebook finds labeled trips for all participants in the database, places them in a dataframe, and expands user inputs to their own columns.

### Files with important functions
    helper_functions.py
    get_EC.py
    confusion_matrix_handling.py
    database_related_functions.py
    dataset_splitting_functions.py (currently only used in the correlation related notebooks)



### Analyses done in this folder:

##### sensing_mean_and_variance_by_program.ipynb: 
    Calculate and plot expected energy consumption and variance for each program. 
    Investigate using primary mode vs section modes. 
    Look at different methods: expected from sections, expected from primary mode
        - Makes a percent error table comparing these
    Also shows proportions of distance traveled in each mode and an estimate of accuracy for sensing
    The program specific and actual mode specifc bar charts come from this notebook.

##### using_spatial_covariance.ipynb:
    This notebook is where the variance calculation methods are compared.
        - looks at how many standard deviations the sensed value is from the user labeled value.
    Calculate variance by summing trip level variances and including a spatial covariance term based on trip clusters.

##### factors_contributing_to_error.ipynb
    Looks at correlations between dataset characteristics and percent error.

    I think a better focus would be targeting and improving specific trip level mode errors eg ebike misclassified as car.\
    - *To see specific mmisprediction rates, look at the cells with print_actual_percents_given_prediction() and print_prediction_percents_given_actual. I think the latter is more important.
    - It might be useful to make a confusion matrix based on CanBikeCO data and compare it with the mobilitynet confusion matrices.*

    Split datasets into subsets to get different shares of modes \
    Calculate subset wide characteristics (eg, proportion of trips that use a car/taxi).\
    Are these characteristics correlated with the percent error for expected value based on the sensed modes?

##### proportion_sensed_sensitivity_analysis.ipynb:
    Plot expected energy consumption for various proportions of sensed vs user labeled data.
    We want to be able to rely on both sensed and user labeled data, \
    so we simulate not knowing some of the user labels and use sensed labels instead

##### prior_mode_distribution_sensitivity_analysis.ipynb:
    Calculate expected energy consumption assuming different prevalances of each mode.

##### Correlation_demonstration_with_fake_programs.ipynb
    Splits all ceo into subsets that serve as mock programs and then splits those further to calculate correlations 
    between dataset characteristics and percent error for expected. 
    The correlations found using each mock program as a starting point were different.

##### EC_over_time_sensitivity_analysis.ipynb
    Looks at cumulative energy consumption at various time points within a study.
    Is the uncertainty reasonable at each point? Looks like yes. Possibly a bit too wide.

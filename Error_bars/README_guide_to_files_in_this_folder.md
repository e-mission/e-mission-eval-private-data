## This folder contains notebooks used to test the energy consumption mean and variance estimation.

A note on pushing to Github: always ensure that no locations are present in any outputs before pushing notebooks. This is to prevent leaking sensitive personal info.

0. Get access to and mongorestore OpenPATH data. See the emission readme for more instructions.
    If using Docker, you will need to increase your docker disk image size to at least 120 GB and possibly increase the memory in Docker Desktop > Settings > Resources.
1. Clone the emission server eval-private-data-compatibility branch:
    - https://github.com/hlu109/e-mission-server/tree/eval-private-data-compatibility
    - check with Shankari about this. another server branch that might work is below:
    - gis-based-mode-detection branch: https://github.com/e-mission/e-mission-server/tree/gis-based-mode-detection
2. Change paths to fit with your setup
    - in e-mission-server/emission/core/get_database.py, edit the first except so config_file can be found:
        - config_file = open(‘path/to/server/e-mission-server/conf/storage/db.conf.sample')
    - if you are working with data from Durham or later, the database in emission.core.get_database should be changed to:
    - openpath_prod_<program_name> (eg openpath_prod_durham) instead of Stage_database.
    - In database_related_functions.py, edit the path to the server in this line:
        - sys.path.append('/path/to/e-mission-server')  

1. If your data does not have sensed sections in the trips, run add_sensed_sections_to_confirmed_trips.py 
    NOTE: this will be replaced soon with Shankari's more efficient implementation and an altered data structure.
2. Run the preliminary notebooks below


### Preliminary notebooks and scripts to run:
#### add_sensed_sections_to_confirmed_trips.py 
    adds "section_modes" and "section_distances" fields to confirmed trip documents in the database.
#### check_for_sections.ipynb
    checks whether we've added section_modes and section_distances to all confirmed trips in the database
    alternatively, you could check the expanded_labeled_trips dataframe after running store_expanded_labeled_trips.ipynb
    to see if section_modes and section_distances are all nonempty.

#### From mobilitynet: classification_analysis.ipynb and trajectory_distance_eval
    Use classification_analysis.ipynb and trajectory_distance_eval to %store the error characteristics for mode and distance.

#### If you need to make changes to the confusion matrices or distance errors you use:
    Run store_errors.ipynb - Preps errors for use in this folder.
    Otherwise, the confusion matrices and distance mean and variance are in:
        - android_confusion.csv, ios_confusion.csv, unit_dist_MCS.csv

#### store_expanded_labeled_trips.ipynb
    Takes 6-14 minutes on the full CEO dataset + stage + prepilot. Every other notebook below uses expanded_labeled_trips. 
    This notebook finds labeled trips for all participants in the database, places them in a dataframe, and expands user inputs to their own columns.

### Files with important functions:
    helper_functions.py - plotting and miscellaneous functions
    get_EC.py - calculate energy consumption mean and variance
    confusion_matrix_handling.py - handle the confusion matrices, calculate EI expecation and variance
    database_related_functions.py
    dataset_splitting_functions.py (currently only used in the correlation related notebooks)

### Creating the plots in the Count Every Trip paper (Energy consumption):
figure: notebook(s) used
    Table 5 (relative length error): 
    mobilitynet-analysis-scripts/trajectory_distance_eval.ipynb, e-mission-eval-private-data/Error_bars/store_errors.ipynb

    Table 6 (MobilityNet proportion of duration spent in each ground truth mode): 
    sensing_mean_and_variance_by_program, under “What are the proportions of each mode in mobilitynet?"

    Table 7 (Proportion of user labeled distance):
    sensing_mean_and_variance_by_program, search for the cell with all_mode_distance_proportions

    Figure 5 (energy consumption bar chart by program)
    sensing_mean_and_variance_by_program, under “Bar chart version of mean plus or minus 1 standard deviation”

    Figure 6 (EC by actual mode):
    sensing_mean_and_variance_by_program, under “Energy consumption by actual mode”

    Table 8 (labeled trip counts by program):
    sensing_mean_and_variance_by_program, under “Labeled trip counts by program”

    Table 9 (percent error by program): 
    sensing_mean_and_variance_by_program, under percent error table

    Table 10 (number of sd’s from truth by program for variance methods):
    using_spatial_covariance.ipynb, under compare standard errors
    first requires you to find clusters: run e-mission-server/bin/build_label_model.py -a 

    Figrue 7 (EC over time):
    EC_over_time_sensitivity_analysis.ipynb

    Figure 8 (Histogram of user standard errors):
    EC_over_time_sensitivity_analysis.ipynb

    Figure 9a/b/c (vary proportion sensed):
    proportion_sensed_sensitivity_analysis.ipynb, under Vary proportion sensed with random splits

    Table 11 (prior sensitivity analysis):
    prior_mode_distribution_sensitivity_analysis.ipynb, under Priors for paper

    Table 14 (dataset characteristics and error):
    factors_contributing_to_error.ipynb, under Correlations between data characteristics and error

    Table 15 (correlations with mock programs):
    Correlation_demonstration_with_fake_programs.ipynb, under Table for paper

    Table 16 (relative length error summary):
    mobilitynet-analysis-scripts/trajectory_distance_eval.ipynb

    Figure 10 (confusion matrices): 
    mobilitynet-analysis-scripts/classification_analysis.ipynb

    Figure 11 (KDE of relative length error):
    mobilitynet-analysis-scripts/trajectory_distance_eval.ipynb

### Analyses done in this folder:

#### sensing_mean_and_variance_by_program.ipynb: 
    Energy units (kWH or MWH) are chosen in this line: energy_dict = cm_handling.get_energy_dict(df_EI, units='MWH')
    Calculate and plot expected energy consumption and variance for each program. 
    Investigate using primary mode vs section modes. 
    Look at different methods: expected from sections, expected from primary mode
        - Makes a percent error table comparing these
    Also shows proportions of distance traveled in each mode and an estimate of accuracy for sensing
    The program specific and actual mode specifc bar charts come from this notebook.

#### using_spatial_covariance.ipynb:
    This notebook is where the variance calculation methods are compared.
        - looks at how many standard deviations the sensed value is from the user labeled value.
    Calculate variance by summing trip level variances and including a spatial covariance term based on trip clusters.

#### factors_contributing_to_error.ipynb
    Looks at correlations between dataset characteristics and percent error.

    NOTE: for future work with functions related to get_set_splits(), might want to update them in a separate file. See dataset_splitting_functions.py

    I think a better focus would be targeting and improving specific trip level mode errors eg ebike misclassified as car.
    - To see specific mmisprediction rates, look at the cells with print_actual_percents_given_prediction() and print_prediction_percents_given_actual. I think the latter is more important.
    - It might be useful to make a confusion matrix based on CanBikeCO data and compare it with the mobilitynet confusion matrices.

    Split datasets into subsets to get different shares of modes
    Calculate subset wide characteristics (eg, proportion of trips that use a car/taxi).
    Are these characteristics correlated with the percent error for expected value based on the sensed modes?

#### proportion_sensed_sensitivity_analysis.ipynb:
    Plot expected energy consumption for various proportions of sensed vs user labeled data.
    We want to be able to rely on both sensed and user labeled data, 
    so we simulate not knowing some of the user labels and use sensed labels instead

#### prior_mode_distribution_sensitivity_analysis.ipynb:
    Calculate expected energy consumption assuming different prevalances of each mode.

#### Correlation_demonstration_with_fake_programs.ipynb
    Splits all ceo into subsets that serve as mock programs and then splits those further to calculate correlations 
    between dataset characteristics and percent error for expected. 
    The correlations found using each mock program as a starting point were different.

#### EC_over_time_sensitivity_analysis.ipynb
    Looks at cumulative energy consumption at various time points within a study.
    Is the uncertainty reasonable at each point? Looks like yes. Possibly a bit too wide.

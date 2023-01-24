## This folder contains notebooks used to test the energy consumption mean and variance estimation.

0. Get access to and mongorestore OpenPATH data. See the emission readme.
1. Run preliminary notebooks.
The first notebook to run is store_errors.ipynb, unless you have access to more updated versions of the confusion matrices than those seen in store_errors.ipynb.

The next notebook to run is store_expanded_labeled_trips.ipynb. This finds labeled trips for all participants in the database, places them in a dataframe, and expands user inputs to their own columns.

### Files with important functions
    helper_functions.py
    get_EC.py
    confusion_matrix_handling.py

### Analyses done in this folder:

##### sensing_mean_and_variance_by_program.ipynb: 
    Calculate and plot expected energy consumption and variance for each program. 
    Investigate using primary mode vs section modes. 
##### proportion_sensed_sensitivity_analysis.ipynb:
    Plot expected energy consumption for various proportions of sensed vs user labeled data.
##### prior_mode_distribution_sensitivity_analysis:
    Calculate expected energy consumption assuming different prevalances of each mode.
##### using_spatial_covariance.ipynb:
    Calculate variance by summing trip level variances and including a spatial covariance term based on trip clusters.
##### factors_contributing_to_error.ipynb
    Split datasets into subsets to get different shares of modes
    Calculate subset wide characteristics (eg, proportion of trips that use a car/taxi).
    Are these characteristics correlated with the percent error for expected value based on the sensed modes?
##### Correlation_demonstration_with_fake_programs.ipynb
    Splits all ceo into subsets that serve as mock programs and then splits those further to calculate correlations 
    between dataset characteristics and percent error for expected. 
    The correlations found using each mock program as a starting point were different.

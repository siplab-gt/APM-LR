# Approximate Posterior Matching for Logistic Regression (APM-LR)
This code accompanies the paper "Feedback Coding for Active Learning" (AISTATS 2021).

## Requirements
- python3
- scipy
- numpy
- sklearn
- pandas
- matplotlib

## Code structure
- `APMexp.py` - main file to run experiments
- `BayesLR.py` - object for Bayesian logistic regression
- `datasets.py` - object for dataset generation
- `LR_active_selectors.py` - functions for active example selection

## Working example
The following command runs an experiment with 5 trials, 50 queries per trial, preprocessed datasets (zero-mean, unit-variance), testing methods APM-LR and Random on the Vehicle Recognition data with superclasses {saab,opel} and {bus,van}, with debugging turned on:  
  
`python APMexp.py --ntrials 5 --nqueries 50 --preprocess --methods APMLR RANDOM --methods_type 0 0 --methods_plot 0 0 --dataset_type VEHICLE --classlist0 saab opel --classlist1 bus van --verbose`
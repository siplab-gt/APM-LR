# Approximate Posterior Matching for Logistic Regression (APM-LR)
This code accompanies the paper "Feedback Coding for Active Learning" by Greg Canal, Matthieu Bloch, and Chris Rozell (AISTATS 2021).

Please send correspondence to Greg Canal (gregory.canal@gatech.edu).

## Requirements
Versions used for experiments listed next to each package:
- Python 3 (3.7.7)
- scipy (1.5.0)
- numpy (1.19.1)
- scikit-learn (0.23.1)
- pandas (1.1.0)
- matplotlib (3.3.1)

## Code structure

### Core functions
- `BayesLR.py`: object for Bayesian logistic regression. When run as main, runs a demonstration of logistic regression in two dimensions.
- `datasets.py`: object for dataset generation. When run as main, runs a demonstration of loading train and test data. Dataset files can be acquired at the following URLs:
    - Vehicle Silhouettes: https://archive.ics.uci.edu/ml/datasets/Statlog+%28Vehicle+Silhouettes%29
    - Letter Recognition: https://archive.ics.uci.edu/ml/datasets/Letter+Recognition
    - Australian Credit Approval: https://archive.ics.uci.edu/ml/datasets/Statlog+%28Australian+Credit+Approval%29
    - Breast Cancer Wisconsin (Diagnostic): https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29

    To use, create a datasets folder and specify the path as the DATAFOLDER constant. Each of the above dataset files should be placed in the following folder paths:
        - Vehicle Silhouettes: DATAFOLDER/UCI/vehicle
        - Letter Recognition: DATAFOLDER/UCI/letter
        - Australian Credit Approval: DATAFOLDER/UCI/austra
        - Breast Cancer Wisconsin (Diagnostic): DATAFOLDER/UCI/wdbc

- `LR_active_selectors.py`: acquisition functions for active example selection, including APM-LR.

### Meta functions
- `APMexp.py`: main file to run a single experiment with specified arguments. Argument options viewable with `python -h APMexp.py`.

    As an example, the following command runs an experiment with 5 trials, 50 queries per trial, preprocessed datasets (zero-mean, unit-variance), testing methods APM-LR and Random on the Vehicle Recognition dataset with superclasses {saab,opel} and {bus,van}, with debug printing turned on at the query level:  
  
    `python APMexp.py --ntrials 5 --nqueries 50 --preprocess --methods APMLR RANDOM --methods_type 0 0 --methods_plot 0 0 --dataset_type VEHICLE --classlist0 saab opel --classlist1 bus van --verbose --query_verbose`

- `jobs.txt`: array of APMexp.py calls to generate paper data, with random number generator seeds included. Paper data was generated by utilizing multiple compute nodes, each with a separate python call and associated save file (saved data files not included in repository; available upon request).

- `analysis.py`: main file to analyze data and recreate results figures, along with computational cost tables. Data is loaded from saved data files generated by `jobs.txt` (data files not included in repository; available upon request). Dataset metadata is structured as a dictionary &mdash; see `main` method for file loading path structure.
# Import necessary libraries
import os  # Library for interacting with the operating system
import pickle  # Library for serializing and deserializing Python objects
import numpy as np  # Library for numerical operations
import pandas as pd  # Library for data manipulation and analysis
from data_loader import read_data_frame_from_pickle  # Function to read data frames from pickle files
from pyod.models.ecod import ECOD  # ECOD model from the PyOD library for anomaly detection

def generate_real_data(method=1):
    """
    Generate real data with anomaly scores using the ECOD model.

    Args:
    method (int): Method to generate anomaly scores (0, 1, or 2).

    Method 1:
    - Calculate anomaly scores using ECOD for the entire dataset.
    - Normalize scores and save the dataset with true thresholds.

    Method 2:
    - Split the dataset into training and testing sets.
    - Train ECOD on the training set and calculate scores for both sets.
    - Normalize scores and save the dataset with a test flag.

    Method 0:
    - Perform both Method 1 and Method 2.

    The results are saved as pickle files in the output path.
    """
    output_path = "../Data/FirewallTestingData/"  # Path to save the output files
    input_path = "../Data/Datasets/"  # Path to the input dataset files

    for filename in os.listdir(input_path):
        dataset = pd.DataFrame()
        file = os.path.join(input_path, str(filename))
        if os.path.isfile(file):
            dataset = read_data_frame_from_pickle(file)

        # Uncomment the next line to sample 10% of the dataset
        # dataset = dataset.sample(round(0.1 * dataset.shape[0]))
        X = dataset.drop('label', axis=1)  # Features
        y = dataset['label'].copy()  # Labels

        if method == 1 or method == 0:
            anomaly_dataset = pd.DataFrame()
            anomaly_dataset['Label'] = y.values

            clf = ECOD()
            clf.fit(X)
            y_scores = clf.decision_scores_
            anomaly_scores = (y_scores - y_scores.min()) / (y_scores.max() - y_scores.min())
            anomaly_dataset['Score'] = anomaly_scores

            normal_count = anomaly_dataset.loc[anomaly_dataset['Label'] == 0].shape[0]
            anomaly_count = anomaly_dataset.loc[anomaly_dataset['Label'] == 1].shape[0]
            contamination = anomaly_count / normal_count
            if contamination == 0.0:
                anomaly_dataset['True_threshold'] = 1.0
            else:
                clf_true = ECOD(contamination=contamination)
                clf_true.fit(X)
                true_anomaly_scores = clf_true.decision_scores_
                true_threshold = clf_true.threshold_
                true_threshold = (true_threshold - true_anomaly_scores.min()) / (
                    true_anomaly_scores.max() - true_anomaly_scores.min())
                anomaly_dataset['True_threshold'] = true_threshold

            with open(output_path + filename.split('.')[0] + "_anomaly_scores_m1.pickle", 'wb') as f:
                pickle.dump(anomaly_dataset, f)

        if method == 2 or method == 0:
            anomaly_dataset = pd.DataFrame()
            y_train = pd.DataFrame()
            y_test = pd.DataFrame()
            baseline = 0.8
            X_train = X[:round(baseline * len(X.index))]
            X_test = X[round(baseline * len(X.index)):]
            y_train['label'] = y[:round(baseline * len(X.index))]
            y_test['label'] = y[round(baseline * len(X.index)):]

            # Removing anomalies from train data
            abnormal_y_train = y_train.loc[dataset['label'] == 1]
            X_train = X_train.drop(abnormal_y_train.index.values)
            y_train = y_train.drop(abnormal_y_train.index.values)

            # Fitting ECOD on train data, and calculating anomaly scores for train and test set
            clf = ECOD()
            clf.fit(X_train)
            y_train_scores = clf.decision_scores_
            y_train_scores = (y_train_scores - y_train_scores.min()) / (y_train_scores.max() - y_train_scores.min())
            y_test_scores = clf.decision_function(X_test)
            y_test_scores = (y_test_scores - y_test_scores.min()) / (y_test_scores.max() - y_test_scores.min())

            # Making result dataset
            anomaly_scores = np.copy(y_train_scores)
            anomaly_scores = np.concatenate((anomaly_scores, y_test_scores))
            anomaly_dataset['Score'] = anomaly_scores
            labels = np.copy(y_train)
            labels = np.concatenate((labels, y_test))
            anomaly_dataset['Label'] = labels
            test_flags = np.zeros((len(X_train.index),), dtype=int)
            test_flags_ones = np.ones((len(X_test.index),), dtype=int)
            test_flags = np.concatenate((test_flags, test_flags_ones))
            anomaly_dataset['Test_flag'] = test_flags

            with open(output_path + filename.split('.')[0] + "_anomaly_scores_m2.pickle", 'wb') as f:
                pickle.dump(anomaly_dataset, f)

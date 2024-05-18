import os
import pickle
import pandas as pd
from Scripts.data_descriptor import make_data_description
from Scripts.data_loader import read_data_frame_from_pickle
from test_data_generator import generate_real_data
from threshold_generator import generate_static_threshold, generate_optimal_threshold, generate_dynamic_threshold_and_predict, generate_supervised_threshold
from utils import predict_y, plot_anomaly_scores, evaluate_prediction
import warnings
import multiprocessing

# Set TensorFlow log level to suppress warnings and errors
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Suppress warnings
warnings.filterwarnings('ignore')

# List of dynamic threshold methods
dynamic_threshold_methods = ['SPOT', 'SAX', 'LOF',
                             'Rolling_3', 'Rolling_5',
                             'EWM_3', 'EWM_5', 'Windowing_3', 'Windowing_5',
                             'Standard_deviation']

# List of supervised threshold methods
supervised_threshold_methods = ["PR", "EER", "Youden", "zero_miss", "alpha_FPR"]

# List of all static threshold selection methods implemented (acronyms)
# To all static methods you can append "_Windowing" (e.g., STD_5_Windowing)
all_static_methods = ['MAX', 'STD_3', 'STD_4', 'STD_5', 'STD_6', 'PERCENTIL_97', 'PERCENTIL_98', "PERCENTIL_99", 'IQR',
                      'DBScan', 'KMeans', 'POT_1', 'POT_2', 'OTSU', 'Density_distance', 'ECDF', 'LOF',
                      'AUCP', 'CHAU', 'CPD', 'DECOMP', 'DSN', 'EB', 'FGD', 'FILTER', 'FWFM', 'GESD', 'HIST', 'KARCH',
                      'MAD', 'MOLL', 'MTT', 'REGR', 'VAE', 'WIND', 'YJ',
                      'Turning_point_DIFF', 'Turning_point_KNEE', 'Turning_point_PERC', 'ECDF', 'Standard_deviations',
                      'KDE_gaussian', 'KDE_tophat', 'KDE_epanechnikov', 'KDE_exponential', 'KDE_linear', 'KDE_cosine',
                      'Distribution_KS_3', 'Distribution_KS_5', 'Distribution_AD_3', 'Distribution_AD_5',
                      'Distribution_CVM_3', 'Distribution_CVM_5',
                      'Optimal_F1', 'Optimal_F2']

# List of fast methods for quick evaluation
fast_methods = ['CHAU', 'HIST', 'EB', 'FILTER', 'GESD', 'MAD', 'MOLL', 'MTT', 'REGR',
                'DBScan', 'Density_distance', 'SAX', 'KMeans', 'OTSU', 'LOF',
                'Optimal_F1', 'Optimal_F2', 'POT_1', 'POT_2',
                'STD_3', 'STD_4', 'STD_5', 'STD_6', 'PERCENTIL_97', 'PERCENTIL_98', 'PERCENTIL_99', 'IQR', 'MAX',
                'Windowing_3', 'Windowing_5', 'Rolling_3', 'Rolling_5', 'EWM_3', 'EWM_5',
                'Turning_point_DIFF', 'Turning_point_KNEE', 'Turning_point_PERC', 'ECDF', 'Standard_deviations',
                'KDE_gaussian', 'KDE_tophat', 'KDE_epanechnikov', 'KDE_exponential', 'KDE_linear', 'KDE_cosine']

def generate_datasets():
    """
    Generate testing datasets with anomaly scores and labels, and initialize the results DataFrame.
    """
    print("Generating test data")
    generate_real_data(method=0)
    print("Test data generated")


def run(threshold_selection_methods, data_path, generation_method, plot, timeout_seconds):
    """
    Run the evaluation of threshold selection methods on generated test data.

    Parameters:
    threshold_selection_methods (list): List of threshold selection methods to evaluate.
    data_path (str): Path to the directory containing test data files.
    generation_method (int): Method of data generation (0 for both, 1 for method 1, 2 for method 2).
    plot (bool): Whether to plot anomaly scores and labels.
    timeout_seconds (int): Timeout in seconds for each threshold generation method.

    Returns:
    None
    """

    if generation_method == 1 or generation_method == 0:
        print("TESTING BY GENERATION METHOD 1")
        results = pd.DataFrame(
            columns=['Method', 'Test_Data', 'Threshold', 'MCC', 'F1_Score', 'F2_Score', 'MCC_Adj', 'F1_Adj', 'F2_Adj',
                     'Execution_Time'])

        # Read all generated test data
        for filename in os.listdir(data_path):
            test_data_name = filename.split('.')[0]
            if "_m1" in test_data_name:
                print(
                    "-----------------------------------------------------------------------------------------------------")
                print("Testing data: " + test_data_name)
                print(
                    "-----------------------------------------------------------------------------------------------------")
                file = os.path.join(data_path, filename)
                if os.path.isfile(file):
                    with open(file, 'rb') as f:
                        test_data = pickle.load(f)
                        anomaly_scores = test_data['Score'].values
                        true_y = test_data['Label'].values
                        best_threshold = test_data['True_threshold'].values[0]
                        test_index = int(len(anomaly_scores) * 0.2)
                        if plot:
                            plot_anomaly_scores(test_data['Score'].values, test_data['Label'].values, filename)

                # Testing all threshold selection methods provided
                for selection_method in threshold_selection_methods:
                    # Making multiprocess to handle timeout of threshold generation method
                    manager = multiprocessing.Manager()
                    return_dict = manager.dict()
                    if selection_method == "Optimal_F1" or selection_method == "Optimal_F2" or selection_method == "Optimal_MCC":
                        p = multiprocessing.Process(target=generate_optimal_threshold,
                                                    args=(anomaly_scores, true_y, selection_method, return_dict,))
                    elif selection_method in supervised_threshold_methods:
                        p = multiprocessing.Process(target=generate_supervised_threshold,
                                                    args=(anomaly_scores, true_y, selection_method, return_dict,))
                    elif selection_method in dynamic_threshold_methods:
                        p = multiprocessing.Process(target=generate_dynamic_threshold_and_predict,
                                                    args=(anomaly_scores, test_index, selection_method, return_dict,))
                    else:
                        p = multiprocessing.Process(target=generate_static_threshold,
                                                    args=(anomaly_scores, selection_method, return_dict,))

                    p.start()
                    p.join(timeout_seconds)
                    try:
                        # Trying to fetch return values from process function
                        if selection_method in dynamic_threshold_methods:
                            predicted_y = return_dict['predicted_y']
                            threshold = 0
                        else:
                            threshold = return_dict['threshold']
                        elapsed_time = return_dict['elapsed_time']
                    except:
                        pass
                    if p.is_alive():
                        # Timeout has happened
                        p.kill()
                        p.join()
                        print(
                            "Timeout for threshold selection method: " + str(selection_method) + " with time = " + str(
                                timeout_seconds) + " seconds.")
                        if "Distribution" in selection_method:
                            # Get last found result
                            threshold = return_dict['threshold']
                            elapsed_time = timeout_seconds
                            if threshold == 0:
                                # Save timeout result entry
                                results_entry = {"Method": selection_method, "Test_Data": test_data_name,
                                                 "Threshold": str(0),
                                                 "MCC": str(0), "F1_Score": str(0), "F2_Score": str(0),
                                                 "MCC_Adj": str(-1), "F1_Adj": str(-1), "F2_Adj": str(-1),
                                                 "Execution_Time": str(timeout_seconds)}
                                results.loc[len(results)] = results_entry
                                continue
                        else:
                            # Save timeout result entry
                            results_entry = {"Method": selection_method, "Test_Data": test_data_name,
                                             "Threshold": str(0),
                                             "MCC": str(0), "F1_Score": str(0), "F2_Score": str(0),
                                             "MCC_Adj": str(-1), "F1_Adj": str(-1), "F2_Adj": str(-1),
                                             "Execution_Time": str(timeout_seconds)}
                            results.loc[len(results)] = results_entry
                            continue

                    # Predict labels based on threshold selected and calculate performance measures
                    if selection_method in dynamic_threshold_methods:
                        print("Generated thresholds and predicted data with method " + str(selection_method) +
                              ", in time = " + str(elapsed_time) + " seconds.")
                    else:
                        print("Generated threshold with method " + str(selection_method) + " is = " + str(
                            threshold) + ", in time = " + str(elapsed_time) + " seconds.")
                        predicted_y = predict_y(anomaly_scores, threshold)
                    mcc, f2_score, f1_score = evaluate_prediction(true_y, predicted_y)
                    best_predicted_y = predict_y(anomaly_scores, best_threshold)
                    mcc_best, f2_score_best, f1_score_best = evaluate_prediction(true_y, best_predicted_y)
                    mcc_adj = round(mcc - mcc_best, 4)
                    f1_adj = round(f1_score - f1_score_best, 4)
                    f2_adj = round(f2_score - f2_score_best, 4)

                    # Save results
                    results_entry = {"Method": selection_method, "Test_Data": test_data_name,
                                     "Threshold": str(threshold),
                                     "MCC": str(mcc), "F1_Score": str(f1_score), "F2_Score": str(f2_score),
                                     "MCC_Adj": str(mcc_adj), "F1_Adj": str(f1_adj), "F2_Adj": str(f2_adj),
                                     "Execution_Time": str(elapsed_time)}

                    results.loc[len(results)] = results_entry

        results.to_excel("../Results/results_m1.xlsx", index=False)

    if generation_method == 2 or generation_method == 0:
        print("TESTING BY GENERATION METHOD 2")
        results = pd.DataFrame(
            columns=['Method', 'Test_Data', 'Threshold', 'Train_MCC', 'Train_F1_Score', 'Train_F2_Score', 'Test_MCC',
                     'Test_F1_Score', 'Test_F2_Score', 'Execution_Time'])

        # Read all generated test data
        for filename in os.listdir(data_path):
            test_data_name = filename.split('.')[0]
            if "_m2" in test_data_name:
                print(
                    "-----------------------------------------------------------------------------------------------------")
                print("Testing data: " + test_data_name)
                print(
                    "-----------------------------------------------------------------------------------------------------")
                file = os.path.join(data_path, filename)
                if os.path.isfile(file):
                    with open(file, 'rb') as f:
                        data = pickle.load(f)
                        anomaly_scores = data['Score'].values
                        true_y = data['Label'].values
                        train_dataset = data.loc[data['Test_flag'] == 0]
                        test_dataset = data.loc[data['Test_flag'] == 1]
                        test_index = test_dataset.index[0]
                        train_anomaly_scores = train_dataset['Score'].values
                        test_anomaly_scores = test_dataset['Score'].values
                        train_true_y = train_dataset['Label'].values
                        test_true_y = test_dataset['Label'].values

                        if plot:
                            plot_anomaly_scores(data['Score'].values, data['Label'].values, filename)

                        # Testing all threshold selection methods provided on train dataset, if optimal threshold then on test set
                        for selection_method in threshold_selection_methods:
                            # Making multiprocess to handle timeout of threshold generation method
                            manager = multiprocessing.Manager()
                            return_dict = manager.dict()
                            if selection_method == "Optimal_F1" or selection_method == "Optimal_F2" or selection_method == "Optimal_MCC":
                                p = multiprocessing.Process(target=generate_optimal_threshold, args=(
                                    test_anomaly_scores, test_true_y, selection_method, return_dict,))
                            elif selection_method in supervised_threshold_methods:
                                p = multiprocessing.Process(target=generate_supervised_threshold,
                                                            args=(
                                                            anomaly_scores, true_y, selection_method, return_dict,))
                            elif selection_method in dynamic_threshold_methods:
                                p = multiprocessing.Process(target=generate_dynamic_threshold_and_predict,
                                                            args=(
                                                                anomaly_scores, test_index, selection_method,
                                                                return_dict,))
                            else:
                                p = multiprocessing.Process(target=generate_static_threshold,
                                                            args=(train_anomaly_scores, selection_method, return_dict,))

                            p.start()
                            p.join(timeout_seconds)
                            try:
                                # Trying to fetch return values from process function
                                elapsed_time = return_dict['elapsed_time']
                                if selection_method in dynamic_threshold_methods:
                                    predicted_y = return_dict['predicted_y']
                                    train_predicted_y = predicted_y[0:test_index]
                                    test_predicted_y = predicted_y[test_index:len(anomaly_scores)]
                                    threshold = 0
                                else:
                                    threshold = return_dict['threshold']
                            except:
                                pass
                            if p.is_alive():
                                # Timeout has happened
                                p.kill()
                                p.join()
                                print("Timeout for threshold selection method: " + str(
                                    selection_method) + " with time = " + str(timeout_seconds) + " seconds.")
                                if "Distribution" in selection_method:
                                    # Get last found result
                                    threshold = return_dict['threshold']
                                    elapsed_time = timeout_seconds
                                    if threshold == 0:
                                        # Save timeout result entry
                                        results_entry = {"Method": selection_method, "Test_Data": test_data_name,
                                                         "Threshold": str(0.0),
                                                         "Train_MCC": str(0.0), "Train_F1_Score": str(0.0),
                                                         "Train_F2_Score": str(0.0),
                                                         "Test_MCC": str(0.0), "Test_F1_Score": str(0.0),
                                                         "Test_F2_Score": str(0.0),
                                                         "Execution_Time": str(timeout_seconds)}
                                        results.loc[len(results)] = results_entry
                                        continue
                                else:
                                    # Save timeout result entry
                                    results_entry = {"Method": selection_method, "Test_Data": test_data_name,
                                                     "Threshold": str(0.0),
                                                     "Train_MCC": str(0.0), "Train_F1_Score": str(0.0),
                                                     "Train_F2_Score": str(0.0),
                                                     "Test_MCC": str(0.0), "Test_F1_Score": str(0.0),
                                                     "Test_F2_Score": str(0.0),
                                                     "Execution_Time": str(timeout_seconds)}
                                    results.loc[len(results)] = results_entry
                                    continue

                            # Predict train set labels based on threshold selected and calculate performance measures
                            if selection_method in dynamic_threshold_methods:
                                print("Generated threshold with method " + str(selection_method) + " , in time = " +
                                      str(elapsed_time) + " seconds.")
                            else:
                                print("Generated threshold with method " + str(selection_method) + " is = " + str(
                                    threshold) + ", in time = " + str(elapsed_time) + " seconds.")
                                train_predicted_y = predict_y(train_anomaly_scores, threshold)
                            train_mcc, train_f2_score, train_f1_score = evaluate_prediction(train_true_y,
                                                                                            train_predicted_y)

                            # Predict test set labels based on threshold selected and calculate performance measures
                            if selection_method not in dynamic_threshold_methods:
                                test_predicted_y = predict_y(test_anomaly_scores, threshold)
                            test_mcc, test_f2_score, test_f1_score = evaluate_prediction(test_true_y,
                                                                                         test_predicted_y)

                            # Save results
                            results_entry = {"Method": selection_method, "Test_Data": test_data_name,
                                             "Threshold": str(threshold),
                                             "Train_MCC": str(train_mcc), "Train_F1_Score": str(train_f1_score),
                                             "Train_F2_Score": str(train_f2_score),
                                             "Test_MCC": str(test_mcc), "Test_F1_Score": str(test_f1_score),
                                             "Test_F2_Score": str(test_f2_score),
                                             "Execution_Time": str(elapsed_time)}
                            results.loc[len(results)] = results_entry

        results.to_excel("../Results/results_m2.xlsx", index=False)


if __name__ == '__main__':
    """
    Main entry point for the script.
    Uncomment the dataset generation and data description lines if needed.
    Runs the threshold selection evaluation process.
    """

    # Uncomment the lines below to generate datasets and create data descriptions
    # generate_datasets()
    # make_data_description()

    # Run the threshold selection evaluation

    run(threshold_selection_methods=['MAX', 'STD_3', 'STD_4', 'STD_5', 'STD_6'],
        data_path='../Data/FirewallTestingData/',
        generation_method=1,
        plot=False,
        timeout_seconds=20000)


    print("-------------------------------------------SCRIPTS DONE--------------------------------------------")

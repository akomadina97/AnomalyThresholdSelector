import math
import time
import numpy as np
import statistics
import pandas as pd
from pythresh.thresholds.aucp import AUCP
from pythresh.thresholds.chau import CHAU
from pythresh.thresholds.cpd import CPD
from pythresh.thresholds.decomp import DECOMP
from pythresh.thresholds.eb import EB
from pythresh.thresholds.fgd import FGD
from pythresh.thresholds.filter import FILTER
from pythresh.thresholds.fwfm import FWFM
from pythresh.thresholds.gesd import GESD
from pythresh.thresholds.hist import HIST
from pythresh.thresholds.karch import KARCH
from pythresh.thresholds.mad import MAD
from pythresh.thresholds.mcst import MCST
from pythresh.thresholds.moll import MOLL
from pythresh.thresholds.mtt import MTT
from pythresh.thresholds.regr import REGR
from pythresh.thresholds.vae import VAE
from pythresh.thresholds.wind import WIND
from pythresh.thresholds.yj import YJ
from scipy.signal import find_peaks
from sklearn.cluster import KMeans
from sklearn.metrics import precision_recall_curve, f1_score, matthews_corrcoef, roc_curve
from dbscan1d.core import DBSCAN1D
import random
from Scripts.utils import calculate_autocorrelation
from kneed import KneeLocator
from pyts.approximation import SymbolicAggregateApproximation
from sklearn.neighbors import KernelDensity, LocalOutlierFactor
from scipy import stats


def generate_static_threshold(anomaly_scores, method, return_dict):
    """
    Generate a static threshold based on the selected method and optional windowing.

    Parameters:
    anomaly_scores (list): List of anomaly scores.
    method (str): String specifying the method to use for threshold generation.
    return_dict (dict): Dictionary to store the results.

    Returns:
    tuple: Calculated threshold value and time taken to compute the threshold.
    """
    # Record the start time for measuring elapsed time
    start_time = time.time()

    # Initialize list to hold local thresholds and set initial threshold to 0
    local_thresholds = []
    threshold = 0
    return_dict['threshold'] = threshold

    # Check if windowing is specified in the method
    if "Windowing" in method:
        # Calculate window size using autocorrelation
        window_size = calculate_autocorrelation(anomaly_scores)

        # Create a DataFrame to hold anomaly scores
        scores_df = pd.DataFrame()
        scores_df['Score'] = anomaly_scores

        # Calculate the number of splits based on window size
        number_of_splits = int(scores_df.shape[0] / window_size)

        # Split the DataFrame into smaller DataFrames based on the number of splits
        df_split = np.array_split(scores_df, number_of_splits)

        # Iterate through each split to calculate local thresholds
        for i in range(number_of_splits):
            local_df = df_split[i]
            local_anomaly_scores = local_df['Score'].values

            # Generate threshold for the local anomaly scores
            local_threshold = generate_threshold_by_method(local_anomaly_scores, method, return_dict)
            local_thresholds.append(local_threshold)

        # Calculate the overall threshold as the mean of local thresholds
        threshold = np.mean(local_thresholds)
    else:
        # Generate threshold directly if windowing is not specified
        threshold = generate_threshold_by_method(anomaly_scores, method, return_dict)

    # Record the end time and calculate the elapsed time
    end_time = time.time()
    elapsed_time = round(end_time - start_time, 4)

    # Store the final threshold and elapsed time in the return dictionary
    return_dict['threshold'] = threshold
    return_dict['elapsed_time'] = elapsed_time

    return threshold, elapsed_time


def generate_threshold_by_method(anomaly_scores, method, return_dict):
    """
    Generate a threshold using various methods specified in the method parameter.

    Parameters:
    anomaly_scores (list): List of anomaly scores.
    method (str): String specifying the method to use for threshold generation.
    return_dict (dict): Dictionary to store the results.

    Returns:
    float: Calculated threshold value.
    """
    threshold = 0
    if method == "MAX":
        threshold = np.max(anomaly_scores)
    elif "STD" in method:
        method_parts = method.split("_")
        k = float(method_parts[1])
        mean = np.mean(anomaly_scores)
        std_dev = statistics.stdev(anomaly_scores)
        threshold = mean + k * std_dev
    elif "PERCENTIL" in method:
        method_parts = method.split("_")
        perc = float(method_parts[1])
        threshold = np.percentile(anomaly_scores, perc)
    elif method == "IQR":
        q1 = np.percentile(anomaly_scores, 25)
        q3 = np.percentile(anomaly_scores, 75)
        IQR = abs(q3 - q1)
        threshold = q3 + 1.5 * IQR
    elif method == "AUCP":
        function = AUCP()
        function.eval(anomaly_scores)
        threshold = function.thresh_
    elif method == "CHAU":
        function = CHAU()
        function.eval(anomaly_scores)
        threshold = function.thresh_
    elif method == "CPD":
        function = CPD()
        function.eval(anomaly_scores)
        threshold = function.thresh_
    elif method == "DECOMP":
        function = DECOMP()
        function.eval(anomaly_scores)
        threshold = function.thresh_
    elif method == "EB":
        function = EB()
        function.eval(anomaly_scores)
        threshold = function.thresh_
    elif method == "FGD":
        function = FGD()
        function.eval(anomaly_scores)
        threshold = function.thresh_
    elif method == "FILTER":
        function = FILTER()
        function.eval(anomaly_scores)
        threshold = function.thresh_
    elif method == "FWFM":
        function = FWFM()
        function.eval(anomaly_scores)
        threshold = function.thresh_[0]
    elif method == "GESD":
        function = GESD()
        function.eval(anomaly_scores)
        threshold = function.thresh_
    elif method == "HIST":
        function = HIST()
        function.eval(anomaly_scores)
        threshold = function.thresh_
    elif method == "KARCH":
        function = KARCH()
        function.eval(anomaly_scores)
        threshold = function.thresh_
    elif method == "MAD":
        function = MAD()
        function.eval(anomaly_scores)
        threshold = function.thresh_
    elif method == "MCST":
        function = MCST()
        function.eval(anomaly_scores)
        threshold = function.thresh_
    elif method == "MOLL":
        function = MOLL()
        function.eval(anomaly_scores)
        threshold = function.thresh_
    elif method == "MTT":
        function = MTT()
        function.eval(anomaly_scores)
        threshold = function.thresh_
    elif method == "REGR":
        function = REGR()
        function.eval(anomaly_scores)
        threshold = function.thresh_
    elif method == "VAE":
        function = VAE()
        function.eval(anomaly_scores)
        threshold = function.thresh_
    elif method == "WIND":
        function = WIND()
        function.eval(anomaly_scores)
        threshold = function.thresh_
    elif method == "YJ":
        function = YJ()
        function.eval(anomaly_scores)
        threshold = function.thresh_
    elif method == "DBScan":
        X = anomaly_scores.reshape(-1, 1)
        dbs = DBSCAN1D(eps=0.05, min_samples=15)
        labels = dbs.fit_predict(X)
        if len(set(labels)) == 1:
            max_value = max(anomaly_scores)
            std_dev = statistics.stdev(anomaly_scores)
            threshold = max_value + std_dev
        elif len(set(labels)) == 2:
            unique_labels = list(set(labels))
            clustered_df = pd.DataFrame()
            clustered_df['Score'] = anomaly_scores
            clustered_df['Cluster'] = labels
            df1 = clustered_df.loc[clustered_df['Cluster'] == unique_labels[0]]
            df2 = clustered_df.loc[clustered_df['Cluster'] == unique_labels[1]]
            min_1 = min(df1['Score'].values)
            max_1 = max(df1['Score'].values)
            min_2 = min(df2['Score'].values)
            max_2 = max(df2['Score'].values)

            if min_1 < min_2:
                d = abs((min_2 - max_1) / 2)
                threshold = max_1 + d
            else:
                d = abs((min_1 - max_2) / 2)
                threshold = max_2 + d
        else:
            threshold = 0
    elif method == "KMeans":
        center_1 = 0
        center_2 = np.mean(anomaly_scores)
        center_3 = np.max(anomaly_scores)
        X = anomaly_scores.reshape(-1, 1)
        cluster_centers = np.array([[center_1], [center_2], [center_3]])
        kmeans = KMeans(n_clusters=3, init=cluster_centers, n_init=1, random_state=1)
        kmeans.fit(cluster_centers)
        labels = kmeans.predict(X)
        cluser_df = pd.DataFrame()
        cluser_df['score'] = anomaly_scores
        cluser_df['label'] = labels
        initial_threshold = min(cluser_df[cluser_df['label'] == 2]['score'].values)
        maximum_threshold = np.max(anomaly_scores)
        step = 0.01
        scores_df = pd.DataFrame()
        scores_df['scores'] = anomaly_scores
        std_X = statistics.stdev(anomaly_scores)
        measure_df = pd.DataFrame(columns=['T', "Measure"])
        T = initial_threshold
        while T <= maximum_threshold:
            X_l = scores_df[scores_df['scores'] <= T]['scores'].values
            X_h = scores_df[scores_df['scores'] > T]['scores'].values
            std_l = statistics.stdev(X_l)
            n_h = len(X_h)
            measure = (std_X) / (std_l * n_h)
            measure_entry = {"T": T, "Measure": measure}
            measure_df.loc[len(measure_df)] = measure_entry
            T += step
        measure_df = measure_df.sort_values(by=['Measure'], ascending=False)
        threshold = measure_df['T'].values[0]

    elif method == "POT_1":
        percentile = 98
        q = 7 * pow(10, -4)
        n = len(anomaly_scores)
        t = np.percentile(anomaly_scores, percentile)
        Yt = anomaly_scores[np.argwhere(anomaly_scores > t)].reshape(1, -1)[0]
        Yt = Yt - t
        Nt = len(Yt)
        params = stats.genpareto.fit(Yt)
        gamma = params[0]  # shape
        mi = params[1]  # location
        sigma = params[2]  # scale
        zq = t + (sigma / gamma) * (math.pow(((q * n) / Nt), -gamma) - 1)
        threshold = zq
    elif method == "POT_2":
        percentile = 98
        q = 3 * pow(10, -4)
        n = len(anomaly_scores)
        t = np.percentile(anomaly_scores, percentile)
        Yt = anomaly_scores[np.argwhere(anomaly_scores > t)].reshape(1, -1)[0]
        Yt = Yt - t
        Nt = len(Yt)
        params = stats.genpareto.fit(Yt)
        gamma = params[0]  # shape
        mi = params[1]  # location
        sigma = params[2]  # scale
        zq = t + (sigma / gamma) * (math.pow(((q * n) / Nt), -gamma) - 1)
        threshold = zq
    elif "Distribution" in method:
        method_parts = method.split("_")
        test_name = method_parts[1]
        k = float(method_parts[2])
        statistics_df = pd.DataFrame(columns=["Distribution", "Statistic", "P_value", "Mean", "Stddev"])
        all_distribution_names = []
        for distribution_name, distribution_params in stats._distr_params.distcont:
            all_distribution_names.append(distribution_name)
        random.shuffle(all_distribution_names)

        for distribution_name in all_distribution_names:
            fitted_distribution = getattr(stats, distribution_name)
            params = fitted_distribution.fit(anomaly_scores)
            cdf = getattr(stats, distribution_name)(*params).cdf

            mean, variance = getattr(stats, distribution_name)(*params).stats(moments='mv')
            stddev = math.sqrt(abs(variance))

            # Calculate KS statistic
            if test_name == "KS":
                try:
                    statistic_ks, p_value_ks = stats.ks_1samp(anomaly_scores, cdf)
                    statistics_entry = {"Distribution": distribution_name, "Statistic": statistic_ks,
                                        "P_value": p_value_ks, "Mean": mean, "Stddev": stddev}
                    statistics_df.loc[len(statistics_df)] = statistics_entry

                except:
                    pass
            elif test_name == "AD":
                # Calculate AD statistic
                if distribution_name in ['norm', 'expon', 'logistic', 'gumbel', 'gumbel_l', 'gumbel_r', 'extreme1']:
                    try:
                        AD, crit, sig = stats.anderson(anomaly_scores, dist=distribution_name)
                        statistic_ad = AD * (1 + (.75 / 50) + 2.25 / (50 ** 2))
                        if statistic_ad >= .6:
                            p_value_ad = math.exp(1.2937 - 5.709 * statistic_ad - .0186 * (statistic_ad ** 2))
                        elif statistic_ad >= .34:
                            p_value_ad = math.exp(.9177 - 4.279 * statistic_ad - 1.38 * (statistic_ad ** 2))
                        elif statistic_ad > .2:
                            p_value_ad = 1 - math.exp(-8.318 + 42.796 * statistic_ad - 59.938 * (statistic_ad ** 2))
                        else:
                            p_value_ad = 1 - math.exp(-13.436 + 101.14 * statistic_ad - 223.73 * (statistic_ad ** 2))
                        statistics_entry = {"Distribution": distribution_name, "Statistic": statistic_ad,
                                            "P_value": p_value_ad, "Mean": mean, "Stddev": stddev}
                        statistics_df.loc[len(statistics_df)] = statistics_entry
                    except:
                        pass

            elif test_name == "CVM":
                # Calculate CVM statistic
                try:
                    cvm_res = stats.cramervonmises(anomaly_scores, cdf)
                    statistic_cvm = cvm_res.statistic
                    p_value_cvm = cvm_res.pvalue
                    statistics_entry = {"Distribution": distribution_name, "Statistic": statistic_cvm,
                                        "P_value": p_value_cvm, "Mean": mean, "Stddev": stddev}
                    statistics_df.loc[len(statistics_df)] = statistics_entry
                except:
                    pass

            statistics_df.replace([np.inf, -np.inf], np.nan, inplace=True)
            statistics_df.dropna(inplace=True)
            statistics_df = statistics_df.sort_values(['P_value', 'Statistic'], ascending=[False, True])
            statistics_df = statistics_df.reset_index(drop=True)
            try:
                mean = statistics_df['Mean'].iloc[0]
                stddev = statistics_df['Stddev'].iloc[0]
                threshold = mean + k * stddev
            except:
                threshold = 0
            return_dict['threshold'] = threshold

    elif method == "Density_distance":
        def distance(set1, set2):
            return (min(set2) - max(set1)) / (min(set2) + max(set1) - 2 * min(set1))

        df = pd.DataFrame()
        df['Score'] = anomaly_scores

        initial_threshold = np.mean(df['Score'])
        final_threshold = initial_threshold
        step = 0.01
        max_threshold = max(df['Score'])
        threshold_list = []

        while final_threshold <= max_threshold:
            set1 = df[df['Score'] <= final_threshold]['Score']
            set2 = df[df['Score'] > final_threshold]['Score']

            threshold_list.append([final_threshold, distance(set1, set2)])

            final_threshold = final_threshold + step

        max_d = max(threshold_list, key=lambda x: x[1])[1]

        for pair in threshold_list:
            if pair[1] == max_d:
                threshold = pair[0]
                break


    elif method == 'OTSU':
        final_threshold = np.mean(anomaly_scores)
        step = 0.001
        max_threshold = max(anomaly_scores)
        n = len(anomaly_scores)

        min_var = np.inf
        best_threshold = final_threshold
        while final_threshold <= max_threshold:
            thresholded_anomaly_scores = np.zeros(anomaly_scores.shape)
            thresholded_anomaly_scores[anomaly_scores >= final_threshold] = 1

            m = np.count_nonzero(thresholded_anomaly_scores)
            weight1 = m / n
            weight0 = 1 - weight1

            val1 = anomaly_scores[thresholded_anomaly_scores == 1]
            val0 = anomaly_scores[thresholded_anomaly_scores == 0]

            var0 = np.var(val0) if len(val0) > 0 else 0
            var1 = np.var(val1) if len(val1) > 0 else 0

            measure = weight0 * var0 + weight1 * var1
            if measure < min_var:
                min_var = measure
                best_threshold = final_threshold

            final_threshold = final_threshold + step

        threshold = best_threshold


    elif method == "ECDF":
        alpha = 0.0005
        X = np.sort(anomaly_scores)
        Y = np.zeros((len(X),))
        for i in range(1, len(X) + 1):
            Y[i - 1] = i / len(X)
        df = pd.DataFrame()
        df['x'] = X
        df['y'] = Y
        df_closest = df.iloc[(df['y'] - (1 - alpha)).abs().argsort()[:1]]
        threshold = df_closest['x'].values[0]


    elif "Turning_point" in method:
        anomaly_scores_df = pd.DataFrame()
        anomaly_scores_df['Score'] = anomaly_scores
        anomaly_scores_df = anomaly_scores_df.sort_values(by=['Score'], ignore_index=True)
        x = anomaly_scores_df.index.values
        y = anomaly_scores_df['Score'].values
        if "KNEE" in method:
            kl = KneeLocator(x, y, curve="convex", S=20)
            threshold = kl.knee_y
        elif "PERC" in method:
            perc = 99.5
            step = int(len(anomaly_scores) * 0.1)

            # Calculate differences between sorted anomaly scores with step size
            differences = []
            for i in range(0, step):
                differences.append(np.nan)
            for i in range(0, len(y) - step):
                last_index = i
                next_index = i + step
                local_difference = y[next_index] - y[last_index]
                differences.append(local_difference)
            differences_df = pd.DataFrame()
            differences_df['Score'] = y
            differences_df['Diff'] = differences

            # Find peaks from calculated differences
            peaks, _ = find_peaks(differences)

            # Find peak with biggest peak value
            biggest_peak_value = 0
            biggest_peak = peaks[0]
            for peak in peaks:
                if differences[peak] >= biggest_peak_value:
                    biggest_peak_value = differences[peak]
                    biggest_peak = peak

            # Generate threshold candidates depending on biggest peak and step size
            threshold_candidates = []
            for i in range(biggest_peak - step, biggest_peak):
                threshold_candidates.append(i)

            # Get threshold from candidates based on the 99.5 percentile
            threshold_candidates_values = y[threshold_candidates]
            threshold = np.percentile(threshold_candidates_values, perc)
        elif "DIFF" in method:
            step = 20

            # Calculate differences between sorted anomaly scores with step size
            differences = []
            for i in range(0, step):
                differences.append(np.nan)
            for i in range(0, len(y) - step):
                last_index = i
                next_index = i + step
                local_difference = y[next_index] - y[last_index]
                differences.append(local_difference)
            differences_df = pd.DataFrame()
            differences_df['Score'] = y
            differences_df['Diff'] = differences

            # Find peaks from calculated differences
            peaks, _ = find_peaks(differences)

            # Find peak with biggest peak value
            biggest_peak_value = 0
            biggest_peak = peaks[0]
            for peak in peaks:
                if differences[peak] >= biggest_peak_value:
                    biggest_peak_value = differences[peak]
                    biggest_peak = peak

            # Generate threshold candidates depending on biggest peak and step size
            threshold_candidates = []
            for i in range(biggest_peak - step, biggest_peak):
                threshold_candidates.append(i)

            # Get threshold from threshold candidates based on difference between each
            threshold_candidate_diffs = np.diff(y[threshold_candidates], 1)
            biggest_threshold_candidate = threshold_candidates[np.argmax(threshold_candidate_diffs)]
            threshold = y[biggest_threshold_candidate]
    elif "KDE" in method:
        def make_probs(density):
            cumsum = np.cumsum(density)
            probs = cumsum / cumsum[-1]
            return probs

        method_parts = method.split("_")
        kernel = method_parts[1]
        d = 1
        m = len(anomaly_scores)
        n1 = (m * (d + 2)) / 4
        n2 = -1 / (d + 4)
        h = pow(n1, n2)
        alpha_dict = {
            "gaussian": 0.01,
            "tophat": 0.002,
            "epanechnikov": 0.002,
            "exponential": 0.02,
            "linear": 0.001,
            "cosine": 0.001
        }
        alpha = alpha_dict[kernel]

        x = anomaly_scores
        eval_points = np.linspace(np.min(x), np.max(x), 1000)
        kde_sk = KernelDensity(bandwidth=h, kernel=kernel)
        kde_sk.fit(x.reshape([-1, 1]))
        pdf_sk = np.exp(kde_sk.score_samples(eval_points.reshape(-1, 1)))
        df = pd.DataFrame()
        df['score'] = eval_points
        df['p'] = make_probs(pdf_sk)
        df_closest = df.iloc[(df['p'] - (1 - alpha)).abs().argsort()[:1]]
        threshold = df_closest['score'].values[0]

    elif method == "Standard_deviations":
        perc = 0.01
        target_len = int(perc * len(anomaly_scores))
        standard_deviations = np.array([])
        scores_df = pd.DataFrame()
        scores_df['score'] = anomaly_scores
        scores_df = scores_df.sort_values(by=['score'], ascending=False)
        scores_df = scores_df.reset_index(drop=True)

        for i in range(0, target_len):
            std = np.std(scores_df['score'])
            standard_deviations = np.append(standard_deviations, std)
            scores_df = scores_df.iloc[1:, :]

        scores_df = pd.DataFrame()
        scores_df['score'] = anomaly_scores
        scores_df = scores_df.sort_values(by=['score'], ascending=False)
        scores_df = scores_df.reset_index(drop=True)

        kl = KneeLocator(range(0, len(standard_deviations)), standard_deviations, S=50, curve="convex",
                         direction="decreasing")
        if kl.knee is not None:
            threshold_df = scores_df.iloc[[kl.knee, kl.knee + 1]]
            threshold_df = threshold_df.reset_index(drop=True)
            threshold = np.mean(threshold_df['score'].values)
        else:
            threshold = 0

    elif method == "NDT":
        mean = np.mean(anomaly_scores)
        std = np.std(anomaly_scores)
        Z = range(1, 11)
        measure_df = pd.DataFrame(columns=['threshold', 'measure'])
        for z in Z:
            t = mean + z*std
            df = pd.DataFrame()
            df['score'] = anomaly_scores
            df['threshold'] = t
            df['label'] = df.apply(lambda x: 1 if x['score'] > x['threshold'] else 0, axis=1)
            df_1 = df[df['label'] == 1]
            e_a = df_1['score'].values
            df_0 = df[df['label'] == 0]
            e_s = df_0['score'].values
            delta_mean = mean - np.mean(e_s)
            delta_std = std - np.std(e_s)
            measure = ((delta_mean/mean)+(delta_std/delta_std) ) / (len(e_a)+pow(len(anomaly_scores), 2))
            entry_df = {"threshold": t, "measure": measure}
            measure_df.loc[len(measure_df)] = entry_df
        measure_df = measure_df.sort_values(by=['measure'], ascending=False)
        threshold = measure_df['threshold'].iloc[0]

    return threshold


def generate_dynamic_threshold_and_predict(anomaly_scores, test_index, method, return_dict):
    """
    Generate a dynamic threshold and predict anomalies based on the selected method.

    Parameters:
    anomaly_scores (list): List of anomaly scores.
    test_index (int): Index to split training and test data.
    method (str): String specifying the method to use for threshold generation and prediction.
    return_dict (dict): Dictionary to store the results.

    Returns:
    tuple: Calculated threshold value, list of predicted labels (0 or 1), and time taken to compute the threshold and predictions.
    """
    # Record the start time for measuring elapsed time
    start_time = time.time()

    # Initialize predicted labels and threshold
    predicted_y = []
    threshold = 0

    if method == "SPOT":
        # Initial parameters and structures
        anomaly_scores_n = anomaly_scores[:test_index]
        n = len(anomaly_scores_n)
        N = len(anomaly_scores)
        percentile = 98
        q = 5 * pow(10, -5)
        anomalies = []
        anomalies_indexes = []

        # Calculate initial zq on n observations
        t = np.percentile(anomaly_scores_n, percentile)
        Yt = anomaly_scores_n[np.argwhere(anomaly_scores_n > t)].reshape(1, -1)[0]
        Yt = Yt - t
        Yt = Yt.tolist()
        Nt = len(Yt)
        params = stats.genpareto.fit(Yt)
        gamma = params[0]
        sigma = params[2]
        zq = t + (sigma / gamma) * (math.pow(((q * n) / Nt), -gamma) - 1)
        threshold = zq

        k = n
        for i in range(n, N):
            if anomaly_scores[i] > zq:
                # Anomaly
                anomalies.append(anomaly_scores[i])
                anomalies_indexes.append(i)
            elif anomaly_scores[i] > t:
                # Real peak
                Yi = anomaly_scores[i] - t
                Yt.append(Yi)
                Nt += 1
                k += 1
                params = stats.genpareto.fit(Yt)
                zq = t + (params[2] / params[0]) * (math.pow(((q * k) / Nt), -params[0]) - 1)
                threshold = zq
            else:
                # Normal case
                k += 1

        # Create the predicted labels array
        predicted_y = np.zeros(N)
        predicted_y[anomalies_indexes] = 1.0

    elif "Rolling" in method:
        # Rolling window method
        method_parts = method.split("_")
        k = float(method_parts[1])
        window_size_factor = 10.0
        window_size = calculate_autocorrelation(anomaly_scores)
        window_size = window_size * window_size_factor
        if window_size >= len(anomaly_scores):
            window_size = len(anomaly_scores)
        window_size = int(window_size)

        thresholds = []
        for i in range(len(anomaly_scores)):
            mean = np.inf
            std = np.inf
            if i != 0:
                start_index = i - window_size
                if start_index < 0:
                    start_index = 0
                window = anomaly_scores[start_index:i]
                mean = np.mean(window)
                std = np.std(window)
            t = mean + k * std
            thresholds.append(t)

        df = pd.DataFrame()
        df['score'] = anomaly_scores
        df['threshold'] = thresholds
        df['predicted'] = df.apply(lambda x: 1 if x['score'] > x['threshold'] else 0, axis=1)
        predicted_y = df['predicted'].values

    elif "EWM" in method:
        # Exponentially Weighted Moving Average method
        method_parts = method.split("_")
        k = float(method_parts[1])
        l = 0.25
        if k == 3.0:
            l = 0.2
        elif k == 5.0:
            l = 0.25
        means = []
        thresholds = []
        var = np.var(anomaly_scores)
        for i in range(len(anomaly_scores)):
            xi = anomaly_scores[i]
            if i == 0:
                mean = l * xi
            else:
                mean = l * xi + (1 - l) * means[i - 1]
            means.append(mean)
            variance = var * (l / (2 - l)) * (1 - pow((1 - l), 2 * i))
            threshold = mean + k * np.sqrt(variance)
            thresholds.append(threshold)
        df = pd.DataFrame(columns=['score', 'threshold'])
        df['score'] = anomaly_scores
        df['threshold'] = thresholds
        df['predicted'] = df.apply(lambda x: 1 if x['score'] > x['threshold'] else 0, axis=1)
        predicted_y = df['predicted'].values

    elif "SAX" in method:
        # Symbolic Aggregate approXimation (SAX) method
        sax = SymbolicAggregateApproximation(n_bins=2, strategy='normal')
        global_scores = anomaly_scores.reshape(1, -1)
        labels = sax.fit_transform(global_scores)[0]
        values, counts = np.unique(labels, return_counts=True)
        predicted_y = []
        for label in labels:
            if label == values[-1]:
                predicted_y.append(0)
            else:
                predicted_y.append(1)

    elif "Windowing" in method:
        # Windowing method with standard deviation
        method_parts = method.split("_")
        k = float(method_parts[1])
        window_size_factor = 20.0
        window_size = calculate_autocorrelation(anomaly_scores)
        window_size = window_size * window_size_factor
        if window_size >= len(anomaly_scores):
            window_size = len(anomaly_scores)
        window_size = int(window_size)
        scores_df = pd.DataFrame()
        scores_df['Score'] = anomaly_scores
        number_of_splits = int(scores_df.shape[0] / window_size)
        df_split = np.array_split(scores_df, number_of_splits)
        predicted_y = []
        for i in range(number_of_splits):
            local_df = df_split[i]
            mean = np.mean(local_df['Score'].values)
            stddev = statistics.stdev(local_df['Score'].values)
            local_threshold = mean + k * stddev
            local_df['Threshold'] = local_threshold
            local_df['Predicted'] = local_df.apply(lambda x: 1 if x['Score'] > x['Threshold'] else 0, axis=1)
            predicted_y.extend(local_df['Predicted'].values)

    elif method == "LOF":
        # Local Outlier Factor (LOF) method
        neighbors = 500  # empirically determined
        clf = LocalOutlierFactor(n_neighbors=neighbors, n_jobs=-1)
        X = anomaly_scores.reshape(-1, 1)
        labels = clf.fit_predict(X)
        predicted_y = (labels + 2) % 3

    # Record the end time and calculate the elapsed time
    end_time = time.time()
    elapsed_time = round(end_time - start_time, 4)

    # Store the results in the return dictionary
    return_dict['elapsed_time'] = elapsed_time
    return_dict['threshold'] = threshold
    return_dict['predicted_y'] = predicted_y

    return threshold, predicted_y, elapsed_time


def generate_optimal_threshold(anomaly_scores, y_true, method, return_dict):
    """
    Generate an optimal threshold based on the provided true labels and selected method.

    Parameters:
    anomaly_scores (list): List of anomaly scores.
    y_true (list): List of true labels.
    method (str): String specifying the method to use for optimal threshold generation.
    return_dict (dict): Dictionary to store the results.

    Returns:
    tuple: Calculated optimal threshold value and time taken to compute the optimal threshold.
    """
    # Record the start time for measuring elapsed time
    start_time = time.time()

    # Initialize threshold variable
    threshold = 0

    if method == "Optimal_F2":
        # Calculate precision-recall curve
        precision, recall, thresholds = precision_recall_curve(y_true, anomaly_scores)

        # Calculate F2 score
        f2score = 5 / ((4 / precision) + (1 / recall))

        # Handle NaN values
        f2score = [0 if math.isnan(x) else x for x in f2score]

        # Find the threshold that maximizes F2 score
        index = np.argmax(f2score)
        threshold = thresholds[index]

    elif method == "Optimal_F1":
        # Calculate precision-recall curve
        precision, recall, thresholds = precision_recall_curve(y_true, anomaly_scores)

        # Calculate F1 score
        f1score = (2 * precision * recall) / (precision + recall)

        # Handle NaN values
        f1score = [0 if math.isnan(x) else x for x in f1score]

        # Find the threshold that maximizes F1 score
        index = np.argmax(f1score)
        threshold = thresholds[index]

    elif method == "Optimal_MCC":
        # Calculate precision-recall curve
        precision, recall, thresholds = precision_recall_curve(y_true, anomaly_scores)

        # Initialize variables to find the best MCC
        best_mcc = -1
        best_threshold = thresholds[0]
        best_index = 0
        step_size = 500

        # Iterate through thresholds with step size to find the best MCC
        for i in range(0, len(thresholds), step_size):
            y_predicted = np.where(anomaly_scores > thresholds[i], 1, 0)
            mcc = matthews_corrcoef(y_true, y_predicted)
            if mcc > best_mcc:
                best_mcc = mcc
                best_threshold = thresholds[i]
                best_index = i

        # Refine search around the best threshold found
        start_index = best_index - step_size
        end_index = best_index + step_size
        if best_index < step_size:
            start_index = 0
        if best_index > len(thresholds) - step_size:
            end_index = len(thresholds)

        for i in range(start_index, end_index, 1):
            y_predicted = np.where(anomaly_scores > thresholds[i], 1, 0)
            mcc = matthews_corrcoef(y_true, y_predicted)
            if mcc > best_mcc:
                best_mcc = mcc
                best_threshold = thresholds[i]

        threshold = best_threshold

    # Record the end time and calculate the elapsed time
    end_time = time.time()
    elapsed_time = round(end_time - start_time, 4)

    # Store the results in the return dictionary
    return_dict['threshold'] = threshold
    return_dict['elapsed_time'] = elapsed_time

    return threshold, elapsed_time


def generate_supervised_threshold(anomaly_scores, y_true, method, return_dict):
    """
    Generate a supervised threshold based on the provided true labels and selected method.

    Parameters:
    anomaly_scores (list): List of anomaly scores.
    y_true (list): List of true labels.
    method (str): String specifying the method to use for supervised threshold generation.
    return_dict (dict): Dictionary to store the results.

    Returns:
    tuple: Calculated supervised threshold value and time taken to compute the supervised threshold.
    """
    # Record the start time for measuring elapsed time
    start_time = time.time()

    # Initialize threshold variable
    threshold = 0

    if method == "PR":
        # Calculate precision-recall curve
        precision, recall, thresholds = precision_recall_curve(y_true, anomaly_scores)

        # Calculate F1 score
        f1score = (2 * precision * recall) / (precision + recall)

        # Handle NaN values in F1 score
        f1score = [0 if math.isnan(x) else x for x in f1score]

        # Find the threshold that maximizes F1 score
        index = np.argmax(f1score)
        threshold = thresholds[index]

    elif method == "Youden":
        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(y_true, anomaly_scores, pos_label=1)

        # Exclude the first threshold (0)
        thresholds = thresholds[1:]
        fpr = fpr[1:]
        tpr = tpr[1:]

        # Calculate Youden's J statistic
        j = tpr - fpr

        # Find the threshold that maximizes Youden's J statistic
        index = np.argmax(j)
        threshold = thresholds[index]

    elif method == "zero_miss":
        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(y_true, anomaly_scores, pos_label=1)

        # Exclude the first threshold (0)
        thresholds = thresholds[1:]
        fpr = fpr[1:]
        tpr = tpr[1:]

        # Initialize minimum false positive rate and index
        min_fpr = 1
        min_fpr_index = 0

        # Find the threshold with zero false negatives
        for i in range(len(thresholds)):
            if tpr[i] == 1 and fpr[i] < min_fpr:
                min_fpr = fpr[i]
                min_fpr_index = i
        threshold = thresholds[min_fpr_index]

    elif method == "alpha_FPR":
        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(y_true, anomaly_scores, pos_label=1)

        # Exclude the first threshold (0)
        thresholds = thresholds[1:]
        fpr = fpr[1:]
        tpr = tpr[1:]

        # Initialize alpha value and maximum true positive rate
        alpha = 0.018
        max_tpr = 0
        max_tpr_index = 0

        # Find the threshold with the highest TPR for a given alpha FPR
        for i in range(len(thresholds)):
            if fpr[i] <= alpha and tpr[i] > max_tpr:
                max_tpr = tpr[i]
                max_tpr_index = i
        threshold = thresholds[max_tpr_index]

    elif method == "EER":
        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(y_true, anomaly_scores, pos_label=1)

        # Calculate FNR (false negative rate)
        fnr = 1 - tpr

        # Find the threshold where FNR equals FPR (equal error rate)
        threshold = thresholds[np.nanargmin(np.abs(fnr - fpr))]

    # Record the end time and calculate the elapsed time
    end_time = time.time()
    elapsed_time = round(end_time - start_time, 4)

    # Store the results in the return dictionary
    return_dict['threshold'] = threshold
    return_dict['elapsed_time'] = elapsed_time

    return threshold, elapsed_time


def calculate_pot_parameters(anomaly_scores, optimal_threshold, dataset_name, return_df):
    """
    Calculate POT (Peak Over Threshold) parameters and evaluate their performance against the optimal threshold.

    Parameters:
    anomaly_scores (list): List of anomaly scores.
    optimal_threshold (float): The optimal threshold value for comparison.
    dataset_name (str): The name of the dataset.
    return_df (pd.DataFrame): DataFrame to store the results.

    Returns:
    pd.DataFrame: Updated DataFrame with POT parameters and their performance.
    """
    # Define percentiles and q values to evaluate
    percentiles = [98, 99]
    qs = np.linspace(pow(10, -4), pow(10, -2), num=100)
    optimal = optimal_threshold

    # Loop through each percentile and q value
    for percentile in percentiles:
        for q in qs:
            # Number of anomaly scores
            n = len(anomaly_scores)

            # Calculate initial threshold based on the percentile
            initial_threshold = np.percentile(anomaly_scores, percentile)

            # Extract scores above the initial threshold and normalize them
            Yt = anomaly_scores[np.argwhere(anomaly_scores > initial_threshold)].reshape(1, -1)[0]
            Yt = Yt - initial_threshold
            Nt = len(Yt)

            # Fit the Generalized Pareto Distribution (GPD) to the normalized scores
            params = stats.genpareto.fit(Yt)
            gamma = params[0]
            sigma = params[2]

            try:
                # Calculate the POT threshold
                threshold = initial_threshold + (sigma / gamma) * (math.pow(((q * n) / Nt), -gamma) - 1)

                # Calculate the distance between the calculated threshold and the optimal threshold
                dist = abs(optimal - threshold)

                # Create a result entry and add it to the return DataFrame
                result_entry = {
                    "Dataset": dataset_name,
                    "Percentile": percentile,
                    "q": q,
                    "Threshold": threshold,
                    "Optimal_threshold": optimal_threshold,
                    "Distance": dist
                }
                return_df.loc[len(return_df)] = result_entry
            except:
                pass

    return return_df


def evaluate_spot_parameters(anomaly_scores, test_index, true_y, dataset_name, return_df):
    """
    Evaluate SPOT (Streaming POT) parameters and their performance against the true labels.

    Parameters:
    anomaly_scores (list): List of anomaly scores.
    test_index (int): Index to split training and test data.
    true_y (list): List of true labels.
    dataset_name (str): The name of the dataset.
    return_df (pd.DataFrame): DataFrame to store the results.

    Returns:
    pd.DataFrame: Updated DataFrame with SPOT parameters and their performance.
    """
    # Define q values to evaluate
    qs = [pow(10, -2), 5 * pow(10, -3), pow(10, -3), 5 * pow(10, -4), pow(10, -4), 5 * pow(10, -5), pow(10, -5),
          5 * pow(10, -6), pow(10, -6)]

    # Loop through each q value
    for q in qs:
        # Initial parameters and structures
        anomaly_scores_n = anomaly_scores[:test_index]
        n = len(anomaly_scores_n)
        N = len(anomaly_scores)
        percentile = 98
        anomalies = []
        anomalies_indexes = []
        true_test_y = true_y[:N]

        # Calculate initial zq on n observations
        t = np.percentile(anomaly_scores_n, percentile)
        Yt = anomaly_scores_n[np.argwhere(anomaly_scores_n > t)].reshape(1, -1)[0]
        Yt = Yt - t
        Yt = Yt.tolist()
        Nt = len(Yt)
        params = stats.genpareto.fit(Yt)
        gamma = params[0]
        sigma = params[2]
        zq = t + (sigma / gamma) * (math.pow(((q * n) / Nt), -gamma) - 1)

        # Process remaining anomaly scores
        k = n
        for i in range(n, N):
            if anomaly_scores[i] > zq:
                # Anomaly detected
                anomalies.append(anomaly_scores[i])
                anomalies_indexes.append(i)
            elif anomaly_scores[i] > t:
                # Real peak, update parameters
                Yi = anomaly_scores[i] - t
                Yt.append(Yi)
                Nt += 1
                k += 1
                params = stats.genpareto.fit(Yt)
                zq = t + (params[2] / params[0]) * (math.pow(((q * k) / Nt), -params[0]) - 1)
            else:
                # Normal case, increment k
                k += 1

        # Generate predicted labels
        generated_y = np.zeros(N)
        df = pd.DataFrame()

        generated_y[anomalies_indexes] = 1.0
        df['Generated'] = generated_y
        df['True'] = true_test_y
        df = df.astype({'True': 'float'})

        y_true = df['True'].values
        y_pred = df['Generated'].values

        # Calculate performance metrics
        f1 = f1_score(y_true, y_pred, average='macro')
        mcc = matthews_corrcoef(y_true, y_pred)

        # Create a result entry and add it to the return DataFrame
        result_entry = {
            "Dataset": dataset_name,
            "Percentile": percentile,
            "q": q,
            "F1": f1,
            "MCC": mcc
        }
        return_df.loc[len(return_df)] = result_entry

    return return_df


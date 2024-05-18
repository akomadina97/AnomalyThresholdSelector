# Import necessary libraries
import numpy as np  # Library for numerical operations
import pandas as pd  # Library for data manipulation and analysis
import seaborn  # Library for statistical data visualization
from matplotlib import pyplot as plt  # Library for creating static, animated, and interactive visualizations
from matplotlib.colors import ListedColormap  # Library for custom color maps in Matplotlib
from scipy.signal import find_peaks  # Function to find peaks in data
from sklearn.metrics import matthews_corrcoef, fbeta_score, f1_score  # Functions for evaluating classification performance
import scipy  # Library for scientific and technical computing

# Function to predict label values (0/1) based on anomaly scores and a selected threshold
def predict_y(anomaly_scores, threshold):
    """
    Predict labels based on anomaly scores and a threshold.

    Args:
    anomaly_scores (array-like): Array of anomaly scores.
    threshold (float): Threshold value for classifying anomalies.

    Returns:
    np.ndarray: Array of predicted labels (0 or 1).
    """
    predicted_y = np.where(anomaly_scores > threshold, 1, 0)
    return predicted_y

# Function to calculate performance measures (TP, FP, TN, FN)
def calculate_performance_measures(y_true, y_predicted):
    """
    Calculate performance measures (True Positives, False Positives, True Negatives, False Negatives).

    Args:
    y_true (array-like): Array of true labels.
    y_predicted (array-like): Array of predicted labels.

    Returns:
    tuple: A tuple containing TP, FP, TN, FN.
    """
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_predicted)):
        if y_true[i] == y_predicted[i] == 1:
            TP += 1
        if y_predicted[i] == 1 and y_true[i] != y_predicted[i]:
            FP += 1
        if y_true[i] == y_predicted[i] == 0:
            TN += 1
        if y_predicted[i] == 0 and y_true[i] != y_predicted[i]:
            FN += 1

    return (TP, FP, TN, FN)

# Function to plot a confusion matrix based on true and predicted labels
def plot_confusion_matrix(true_y, predicted_y):
    """
    Plot a confusion matrix based on true and predicted labels.

    Args:
    true_y (array-like): Array of true labels.
    predicted_y (array-like): Array of predicted labels.
    """
    TP, FP, TN, FN = calculate_performance_measures(true_y, predicted_y)
    conf_matrix = [[TN, FP], [FN, TP]]
    confusion_matrix_df = pd.DataFrame(conf_matrix, columns=["0", "1"], index=["0", "1"])
    seaborn.heatmap(confusion_matrix_df, annot=True, fmt='g', cmap=create_color_map(), linewidths=1, annot_kws={
        'fontsize': 40,
        'fontweight': 'bold',
        'fontfamily': 'fantasy',
        'fontstretch': 'expanded'
    })
    plt.title("Confusion matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

# Function to plot a simple histogram for anomaly scores
def plot_anomaly_scores_simple(y_true, y_scores):
    """
    Plot a simple histogram for anomaly scores.

    Args:
    y_true (array-like): Array of true labels.
    y_scores (array-like): Array of anomaly scores.
    """
    plt.rcParams["figure.figsize"] = (18, 9)
    histogram_data_frame = pd.DataFrame()
    histogram_data_frame['label'] = y_true.copy()
    histogram_data_frame['score'] = y_scores.copy()
    histogram_data_frame.pivot(columns='label', values='score').plot.hist(bins=1000, color=['#002E65', '#FD7813'])
    plt.show()

# Function to plot anomaly scores colored by clusters, with an optional threshold line
def plot_anomaly_scores(anomaly_scores, clusters, title, threshold=None):
    """
    Plot anomaly scores colored by clusters, with an optional threshold line.

    Args:
    anomaly_scores (array-like): Array of anomaly scores.
    clusters (array-like): Array of cluster labels.
    title (str): Title of the plot.
    threshold (float, optional): Threshold value for classifying anomalies.
    """
    number_of_bins = int(len(anomaly_scores) / 10)
    histogram_data_frame = pd.DataFrame()
    histogram_data_frame['label'] = clusters
    histogram_data_frame['score'] = anomaly_scores
    plt.rcParams['figure.figsize'] = [18, 8]
    ax = histogram_data_frame.pivot(columns='label', values='score').plot.hist(bins=number_of_bins)
    plt.xlabel('Anomaly score')
    plt.ylabel('Number of points')
    plt.title(title)

    if threshold is not None:
        y_max = ax.get_ylim()[1]
        ax.plot([threshold, threshold], [0, int(y_max) + 1], 'k--')
    plt.show()

# Function to create a line plot of (x, y) data
def plot_data(x, y):
    """
    Create a line plot of (x, y) data.

    Args:
    x (array-like): Array of x values.
    y (array-like): Array of y values.
    """
    plt.plot(x, y)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()

# Function to evaluate prediction based on Matthews Correlation Coefficient and F2 Score
def evaluate_prediction(true_y, predicted_y):
    """
    Evaluate prediction based on Matthews Correlation Coefficient (MCC) and F2 Score.

    Args:
    true_y (array-like): Array of true labels.
    predicted_y (array-like): Array of predicted labels.

    Returns:
    tuple: A tuple containing MCC, F2 score, and F1 score.
    """
    mmc = round(matthews_corrcoef(true_y, predicted_y), 4)
    f2_score = round(fbeta_score(true_y, predicted_y, average='macro', beta=0.5), 4)
    f1 = round(f1_score(true_y, predicted_y, average='macro'), 4)
    return mmc, f2_score, f1

# Function to create a custom color map
def create_color_map(color1hex='#002E65', color2hex='#FD7813'):
    """
    Create a custom color map.

    Args:
    color1hex (str): Hex code for the first color.
    color2hex (str): Hex code for the second color.

    Returns:
    ListedColormap: A custom color map.
    """
    N = 256

    color1 = np.ones((N, 4))
    color1[:, 0] = np.linspace(int(color1hex[1:3], 16) / 256, 1, N)
    color1[:, 1] = np.linspace(int(color1hex[3:5], 16) / 256, 1, N)
    color1[:, 2] = np.linspace(int(color1hex[5:7], 16) / 256, 1, N)
    color1_cmp = ListedColormap(color1)

    color2 = np.ones((N, 4))
    color2[:, 0] = np.linspace(int(color2hex[1:3], 16) / 256, 1, N)
    color2[:, 1] = np.linspace(int(color2hex[3:5], 16) / 256, 1, N)
    color2[:, 2] = np.linspace(int(color2hex[5:7], 16) / 256, 1, N)
    color2_cmp = ListedColormap(color2)

    newcolors2 = np.vstack((color1_cmp(np.linspace(0, 1, 128)),
                            color2_cmp(np.linspace(1, 0, 128))))
    double = ListedColormap(newcolors2, name='double')

    return double

# Function to calculate the optimal period by autocorrelation
def calculate_autocorrelation(data):
    """
    Calculate the optimal period by autocorrelation.

    Args:
    data (array-like): Array of data values.

    Returns:
    int: The optimal period based on autocorrelation.
    """
    signal_df = pd.DataFrame()
    signal_df['y'] = data
    signal_df.reset_index()
    signal_df['x'] = signal_df.index
    xs = signal_df['x'].values
    ys = signal_df['y'].values
    acf = scipy.signal.correlate(ys, ys, 'full')[-len(xs):]

    peaks, _ = find_peaks(acf, width=50)
    biggest_peak_value = 0
    biggest_peak = peaks[0]
    for peak in peaks:
        peak_value = acf[peak]
        if (peak_value > biggest_peak_value):
            biggest_peak_value = peak_value
            biggest_peak = peak

    return biggest_peak

# Import necessary libraries
import os  # Library for interacting with the operating system
from matplotlib import pyplot as plt  # Library for creating static, animated, and interactive visualizations
from Scripts.data_loader import read_data_frame_from_pickle  # Custom function to read data from a pickle file
from matplotlib import rcParams  # Module for configuring Matplotlib settings

# Set font weight for Matplotlib plots
rcParams['font.weight'] = 'bold'

# Define the input path where data files are stored
input_path = "../Data/FirewallTestingData/"

# Define positions for description text in the plots
description_text_positions = [[0.249, 0.869], [0.576, 0.869], [0.903, 0.869],
                              [0.249, 0.542], [0.576, 0.542], [0.903, 0.542],
                              [0.249, 0.215], [0.576, 0.215], [0.903, 0.215]]


def parse_month(number):
    """
    Convert a month number to its corresponding month name.

    Args:
    number (str): Month number as a string (e.g., "10" for October).

    Returns:
    str: Month name corresponding to the input number.
    """
    if number == "10":
        return "October"
    if number == "11":
        return "November"
    if number == "12":
        return "December"


def make_data_description():
    """
    Generate and display data descriptions and histograms for anomaly scores from multiple datasets.

    This function reads multiple datasets from the specified input path, processes them to extract anomaly scores,
    and generates histograms and statistical summaries for each dataset. The results are displayed in a 3x3 grid of plots.
    """
    all_anomaly_scores = []  # List to store anomaly scores from all datasets
    all_anomaly_scores_dfs = []  # List to store DataFrame of scores and labels from all datasets
    summaries = []  # List to store statistical summaries of the anomaly scores
    all_anomaly_scores_names = []  # List to store the names of the datasets

    # Iterate over all files in the input directory
    for filename in os.listdir(input_path):
        if "m1" in filename and '02' not in filename:  # Filter files based on specific criteria
            file = os.path.join(input_path, str(filename))
            if os.path.isfile(file):  # Check if the path is a file
                dataset = read_data_frame_from_pickle(file)  # Read data from the pickle file
                all_anomaly_scores.append(dataset['Score'].values)  # Extract and store anomaly scores
                anomaly_scores_df = dataset[['Score', 'Label']]  # Create a DataFrame with scores and labels
                all_anomaly_scores_dfs.append(anomaly_scores_df)

                # Calculate and print the number of normal and anomaly instances
                all_number = anomaly_scores_df.shape[0]
                normal_number = anomaly_scores_df.loc[anomaly_scores_df['Label'] == 0].shape[0]
                anomaly_number = anomaly_scores_df.loc[anomaly_scores_df['Label'] == 1].shape[0]
                normal_ratio = (normal_number / all_number) * 100
                anomaly_ratio = (anomaly_number / all_number) * 100
                print(all_number, normal_number, anomaly_number)
                print("Normal ratio: " + str(normal_ratio) + " %")
                print("Anomaly ratio: " + str(anomaly_ratio) + " %")

                # Generate and store a statistical summary of the anomaly scores
                summary = dataset['Score'].describe().round(4)
                summaries.append(summary)
                all_anomaly_scores_names.append(filename)  # Store the dataset name

    # Create a 3x3 grid of subplots
    figure, axis = plt.subplots(3, 3)

    # Iterate over all anomaly scores and plot histograms
    for i in range(len(all_anomaly_scores)):
        anomaly_scores = all_anomaly_scores[i]
        anomaly_scores_name = all_anomaly_scores_names[i]
        x = int(i / 3)
        y = i % 3
        splits = anomaly_scores_name.split("_")
        plot_name = "Month: " + parse_month(splits[1]) + ", Anomaly type: " + splits[2]

        # Plot histogram of the anomaly scores
        axis[x, y].hist(anomaly_scores, bins=20000)
        axis[x, y].set_xlim(-0.02, 1)
        axis[x, y].set_title(plot_name, fontsize=9, fontweight='bold')

        # Add statistical summary text to the plot
        plt.figtext(description_text_positions[i][0], description_text_positions[i][1], summaries[i].to_string(),
                    {'fontsize': 8})

    # Adjust layout of the subplots
    plt.subplots_adjust(left=0.04, bottom=0.04, right=0.97, top=0.97, wspace=0.15, hspace=0.18)
    plt.show()  # Display the plots

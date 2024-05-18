# Import necessary libraries
import time  # Library for time-related functions
import pandas as pd  # Library for data manipulation and analysis

# Set pandas display options to show all columns and set the display width
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 400)


# Function to load firewall data from a CSV file into a pandas DataFrame
def load_data_frames(path):
    """
    Load firewall data from a CSV file into a pandas DataFrame.

    Args:
    path (str): The path to the CSV file.

    Returns:
    pd.DataFrame: A pandas DataFrame containing the firewall data.
    """
    start_time = time.time()  # Record the start time
    print("Loading data from: " + path)
    csv_file = open(path)  # Open the CSV file

    # Define the data types for each column
    dtype = {0: "string", 1: "string", 2: "string", 3: "string", 4: "string", 5: "string", 6: "string", 7: "string",
             8: "string", 9: "string", 10: "string", 11: "string", 12: "string", 13: "string", 14: "string",
             15: "string", 16: "string", 17: "string", 18: "string"}

    # Load the CSV file into a pandas DataFrame
    data_frame = pd.read_csv(csv_file, header=None,
                             usecols=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
                             dtype=dtype, encoding='utf-8', on_bad_lines='skip', nrows=1000000)

    # Rename the columns
    data_frame = data_frame.rename(
        columns={0: "timestamp",
                 1: "source_ip", 2: "source_port", 3: "source_range", 4: "source_subnet",
                 5: "source_area", 6: "source_object", 7: "source_object_type",
                 8: "destination_ip", 9: "destination_port", 10: "destination_range", 11: "destination_subnet",
                 12: "destination_area", 13: "destination_object", 14: "destination_object_type",
                 15: "protocol", 16: "action",
                 17: "origin", 18: "log_type"})

    # Fill blank cells with "unknown" value
    data_frame = data_frame.fillna("unknown")

    # Remove rows with "unknown" values in specified columns
    for column_name in ['source_ip', 'destination_ip', 'destination_port', 'protocol']:
        data_frame = data_frame[data_frame[column_name] != "unknown"]

    # Convert destination_port to integer
    data_frame["destination_port"] = data_frame["destination_port"].astype(int)

    # Convert timestamp from string to datetime and remove timezone information
    data_frame["timestamp"] = pd.to_datetime(data_frame["timestamp"])
    data_frame["timestamp"] = data_frame["timestamp"].dt.tz_localize(None)

    end_time = time.time()  # Record the end time
    elapsed_time = round(end_time - start_time, 2)  # Calculate elapsed time
    print("Data loaded in " + str(elapsed_time) + " seconds")

    return data_frame


# Function to load firewall data from a CSV file and save it as a pickle file
def convert_data_into_pickle(path, output_path):
    """
    Load firewall data from a CSV file and save it as a pickle file.

    Args:
    path (str): The path to the CSV file.
    output_path (str): The path to save the pickle file.
    """
    start_time = time.time()  # Record the start time
    print("Loading data from: " + path)
    csv_file = open(path)  # Open the CSV file

    # Define the data types for each column
    dtype = {0: "string", 1: "string", 2: "string", 3: "string", 4: "string", 5: "string", 6: "string", 7: "string",
             8: "string", 9: "string", 10: "string", 11: "string", 12: "string", 13: "string", 14: "string",
             15: "string", 16: "string", 17: "string", 18: "string"}

    # Load the CSV file into a pandas DataFrame
    data_frame = pd.read_csv(csv_file, header=None,
                             usecols=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
                             dtype=dtype, encoding='utf-8', on_bad_lines='skip')

    # Rename the columns
    data_frame = data_frame.rename(
        columns={0: "timestamp",
                 1: "source_ip", 2: "source_port", 3: "source_range", 4: "source_subnet",
                 5: "source_area", 6: "source_object", 7: "source_object_type",
                 8: "destination_ip", 9: "destination_port", 10: "destination_range", 11: "destination_subnet",
                 12: "destination_area", 13: "destination_object", 14: "destination_object_type",
                 15: "protocol", 16: "action",
                 17: "origin", 18: "log_type"})

    # Fill blank cells with "unknown" value
    data_frame = data_frame.fillna("unknown")

    # Remove rows with "unknown" values in specified columns
    for column_name in ['source_ip', 'destination_ip', 'destination_port', 'protocol']:
        data_frame = data_frame[data_frame[column_name] != "unknown"]

    # Convert timestamp from string to datetime and remove timezone information
    data_frame["timestamp"] = pd.to_datetime(data_frame["timestamp"])
    data_frame["timestamp"] = data_frame["timestamp"].dt.tz_localize(None)

    # Save DataFrame to pickle file
    splits = path.split("/")
    input_name = splits[-1].split(".")[0]
    data_frame.to_pickle(output_path + input_name + ".pickle")

    end_time = time.time()  # Record the end time
    elapsed_time = round(end_time - start_time, 2)  # Calculate elapsed time
    print("Data converted in " + str(elapsed_time) + " seconds")


# Function to load anomaly data from a CSV file into a pandas DataFrame
def load_anomaly_data(path):
    """
    Load anomaly data from a CSV file into a pandas DataFrame.

    Args:
    path (str): The path to the CSV file.

    Returns:
    pd.DataFrame: A pandas DataFrame containing the anomaly data.
    """
    csv_file = open(path)  # Open the CSV file

    # Define the data types for each column
    dtype = {0: "string", 1: "string", 2: "string", 3: "string", 4: "string", 5: "string"}

    # Load the CSV file into a pandas DataFrame
    data_frame = pd.read_csv(csv_file, header=None,
                             usecols=[0, 1, 2, 3, 4, 5],
                             dtype=dtype, encoding='utf-8')

    # Rename the columns
    data_frame = data_frame.rename(
        columns={0: "timestamp",
                 1: "source_ip", 2: "source_port", 3: "destination_ip", 4: "destination_port",
                 5: "protocol"})

    # Fill blank cells with "unknown" value
    data_frame = data_frame.fillna("unknown")

    # Remove rows with "unknown" values in specified columns
    for column_name in ['source_ip', 'destination_ip', 'protocol']:
        data_frame = data_frame[data_frame[column_name] != "unknown"]

    # Convert timestamp from string to datetime
    data_frame["timestamp"] = pd.to_datetime(data_frame["timestamp"])

    return data_frame


# Function to load anomaly data from a CSV file and save it as a pickle file
def convert_anomaly_data_into_pickle(path, output_path):
    """
    Load anomaly data from a CSV file and save it as a pickle file.

    Args:
    path (str): The path to the CSV file.
    output_path (str): The path to save the pickle file.
    """
    csv_file = open(path)  # Open the CSV file

    # Define the data types for each column
    dtype = {0: "string", 1: "string", 2: "string", 3: "string", 4: "string", 5: "string"}

    # Load the CSV file into a pandas DataFrame
    data_frame = pd.read_csv(csv_file, header=None,
                             usecols=[0, 1, 2, 3, 4, 5],
                             dtype=dtype, encoding='utf-8')

    # Rename the columns
    data_frame = data_frame.rename(
        columns={0: "timestamp",
                 1: "source_ip", 2: "source_port", 3: "destination_ip", 4: "destination_port",
                 5: "protocol"})

    # Fill blank cells with "unknown" value
    data_frame = data_frame.fillna("unknown")

    # Remove rows with "unknown" values in specified columns
    for column_name in ['source_ip', 'destination_ip', 'protocol']:
        data_frame = data_frame[data_frame[column_name] != "unknown"]

    # Convert timestamp from string to datetime
    data_frame["timestamp"] = pd.to_datetime(data_frame["timestamp"])

    # Save DataFrame to pickle file
    splits = path.split("/")
    input_name = splits[-1].split(".")[0]
    data_frame.to_pickle(output_path + input_name + ".pickle")


# Function to load firewall logs CSV into a pandas DataFrame, split it by hours, and save each hour as a separate pickle file
def load_continuous_data_frame(data_path, output_path):
    """
    Load firewall logs CSV into a pandas DataFrame, split it by hours, and save each hour as a separate pickle file.

    Args:
    data_path (str): The path to the CSV file.
    output_path (str): The path to save the pickle files.
    """
    start_time = time.time()  # Record the start time
    print("Loading data from: " + data_path)
    csv_file = open(data_path)  # Open the CSV file

    # Define the data types for each column
    dtype = {0: "string", 1: "string", 2: "string", 3: "string", 4: "string", 5: "string", 6: "string", 7: "string",
             8: "string", 9: "string", 10: "string", 11: "string", 12: "string", 13: "string", 14: "string",
             15: "string", 16: "string", 17: "string", 18: "string"}

    # Load the CSV file into a pandas DataFrame
    data_frame = pd.read_csv(csv_file, header=None,
                             usecols=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
                             dtype=dtype, encoding='utf-8', on_bad_lines='skip')

    # Rename the columns
    data_frame = data_frame.rename(
        columns={0: "timestamp",
                 1: "source_ip", 2: "source_port", 3: "source_range", 4: "source_subnet",
                 5: "source_area", 6: "source_object", 7: "source_object_type",
                 8: "destination_ip", 9: "destination_port", 10: "destination_range", 11: "destination_subnet",
                 12: "destination_area", 13: "destination_object", 14: "destination_object_type",
                 15: "protocol", 16: "action",
                 17: "origin", 18: "log_type"})

    # Fill blank cells with "unknown" value
    data_frame = data_frame.fillna("unknown")

    # Remove rows with "unknown" values in specified columns
    for column_name in ['source_ip', 'destination_ip', 'protocol']:
        data_frame = data_frame[data_frame[column_name] != "unknown"]

    # Convert timestamp from string to datetime and remove timezone information
    data_frame["timestamp"] = pd.to_datetime(data_frame["timestamp"])
    data_frame["timestamp"] = data_frame["timestamp"].dt.tz_localize(None)

    # Add an hour column to the DataFrame
    data_frame['hour'] = data_frame['timestamp'].dt.hour
    data_frame['hour'] = data_frame['hour'].astype(int)

    # Split the DataFrame into partial DataFrames by hour and save each as a pickle file
    partial_data_frames = []
    splits = data_path.split("/")
    input_name = splits[-1].split(".")[0]
    for i in range(0, 24):
        temp_df = data_frame.loc[data_frame['hour'] == i]
        temp_df = temp_df.drop(['hour'], axis=1)
        partial_data_frames.append(temp_df)
        temp_df.to_pickle(output_path + input_name + "_" + str(i) + "h.pickle")

    end_time = time.time()  # Record the end time
    elapsed_time = round(end_time - start_time, 2)  # Calculate elapsed time
    print("Data loaded and split in " + str(elapsed_time) + " seconds")


# Function to read a DataFrame from a pickle file
def read_data_frame_from_pickle(data_path):
    """
    Read a DataFrame from a pickle file.

    Args:
    data_path (str): The path to the pickle file.

    Returns:
    pd.DataFrame: A pandas DataFrame loaded from the pickle file.
    """
    print("Loading data from: " + data_path)
    data_frame = pd.read_pickle(data_path)
    return data_frame


# Function to sample a DataFrame from a pickle file
def sample_dataframe_from_pickle(path, fraction, output_path):
    """
    Sample a fraction of a DataFrame from a pickle file and optionally save it as a new pickle file.

    Args:
    path (str): The path to the pickle file.
    fraction (float): The fraction of the DataFrame to sample.
    output_path (str): The path to save the sampled pickle file (optional).

    Returns:
    pd.DataFrame: The sampled DataFrame if output_path is None.
    """
    print("Loading data from: " + path)
    data_frame = pd.read_pickle(path)
    sample_data_frame = data_frame.sample(frac=fraction)
    splits = path.split("/")
    input_name = splits[-1].split(".")[0]
    percentage = fraction * 100
    if output_path is None:
        return sample_data_frame
    else:
        sample_data_frame.to_pickle(output_path + input_name + "_" + str(percentage) + "%.pickle")

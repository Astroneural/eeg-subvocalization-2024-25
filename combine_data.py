import pandas as pd

def preprocess(file_path):
    df = pd.read_csv(file_path)

    # Drop the specified columns
    df = df.drop(columns=["timestamps", "Timestamp", "Counter", "Interpolate", "HardwareMarker", "Markers", "Marker"])

    # Rename the columns
    df = df.rename(columns={"words": 0, "terms": 1, "AF3": 2, "T7": 3, "Pz": 4, "T8": 5, "AF4": 6})

    # Subtract 1 from the "words" column (now column 0)
    # df.iloc[:, 0] -= 1

    # Replace "A", "B", and "C" with 0, 1, and 2, respectively
    df.replace({"yes": 0, "no": 1, "help": 2, "sun": 3, "water": 4}, inplace=True)

    # Save the formatted DataFrame to a new CSV file
    #df.to_csv("formatted_data_2-4_6.csv", index=False)

    max_word = df.iloc[:, 0].max()

    # Filter out rows with the maximum word value
    df = df[df.iloc[:, 0] != max_word]

    return df

def combine_csv_files(*file_paths, output_file):
    """Combines any number of CSV files, ensuring unique indices across all files."""

    # Initialize a counter to track the next unique index
    next_index = 0
    dfs = []

    # Read each file and increment indices accordingly
    for file_path in file_paths:
        df = preprocess(file_path)
        #df = pd.read_csv(file_path)
        max_index = df.iloc[:, 0].max()
        df.iloc[:, 0] += next_index
        next_index += max_index + 1  # Update the counter for the next file
        dfs.append(df)

    # Concatenate the DataFrames
    combined_df = pd.concat(dfs, ignore_index=True)

    # Write the combined DataFrame to a new CSV file
    combined_df.to_csv(output_file, index=False)

# Example usage (combining any number of files):
'''combine_csv_files("/Users/shaum/eeg-stuffs/data/ut4-2/data_ut_2024-11-25-19.48.02.csv", "/Users/shaum/eeg-stuffs/data/ut4-2/data_ut_2024-11-25-19.52.03.csv", "/Users/shaum/eeg-stuffs/data/ut4-2/data_ut_2024-11-25-19.55.24.csv", "/Users/shaum/eeg-stuffs/data/ut4-2/data_ut_2024-11-25-19.58.38.csv", output_file="ut4_train.csv") # adding /Users/shaum/eeg-stuffs/data/ut3-2/data_ut_2024-11-25-17.42.21.csv breaks the data_pca.py code somehow
combine_csv_files("/Users/shaum/eeg-stuffs/data/ut4-2/data_ut_2024-11-25-20.01.52.csv", "/Users/shaum/eeg-stuffs/data/ut4-2/data_ut_2024-11-25-20.05.16.csv", output_file="ut4_val.csv")
combine_csv_files("/Users/shaum/eeg-stuffs/data/ut4-2/data_ut_2024-11-25-20.08.54.csv", output_file="ut4_test.csv")'''
#combine_csv_files("preprocessing_data/combined_data_wrec3.csv", "combined_1-3.csv", output_file="combined_1-23_1-3.csv")

combine_csv_files("/Users/shaum/eeg-stuffs/data/ut3/data_ut_2024-11-25-17.30.27.csv", "data/ut3-2/data_ut_2024-11-25-17.35.30.csv", "/Users/shaum/eeg-stuffs/data/ut3-2/data_ut_2024-11-25-17.38.59.csv", output_file="ut3_train.csv") # adding /Users/shaum/eeg-stuffs/data/ut3-2/data_ut_2024-11-25-17.42.21.csv breaks the data_pca.py code somehow
combine_csv_files("/Users/shaum/eeg-stuffs/data/ut3-2/data_ut_2024-11-25-17.45.55.csv", "/Users/shaum/eeg-stuffs/data/ut3-2/data_ut_2024-11-25-17.49.11.csv", output_file="ut3_val.csv")
combine_csv_files("/Users/shaum/eeg-stuffs/data/ut3-2/data_ut_2024-11-25-17.52.25.csv", output_file="ut3_test.csv")

'''combine_csv_files("/Users/shaum/eeg-stuffs/data/ut5-2/data_ut_2024-11-26-16.32.52.csv", "/Users/shaum/eeg-stuffs/data/ut5-2/data_ut_2024-11-26-16.37.08.csv", "/Users/shaum/eeg-stuffs/data/ut5-2/data_ut_2024-11-26-16.40.41.csv", "/Users/shaum/eeg-stuffs/data/ut5-2/data_ut_2024-11-26-16.43.59.csv", output_file="ut5_train.csv") # adding /Users/shaum/eeg-stuffs/data/ut3-2/data_ut_2024-11-25-17.42.21.csv breaks the data_pca.py code somehow
combine_csv_files("/Users/shaum/eeg-stuffs/data/ut5-2/data_ut_2024-11-26-16.47.25.csv", "/Users/shaum/eeg-stuffs/data/ut5-2/data_ut_2024-11-26-16.50.43.csv", "/Users/shaum/eeg-stuffs/data/ut5-2/data_ut_2024-11-26-16.54.02.csv", output_file="ut5_val.csv")
combine_csv_files("/Users/shaum/eeg-stuffs/data/ut5-2/data_ut_2024-11-26-16.57.24.csv", "/Users/shaum/eeg-stuffs/data/ut5-2/data_ut_2024-11-26-17.00.59.csv", output_file="ut5_test.csv")'''

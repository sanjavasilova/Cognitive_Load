import os
import pandas as pd
import matplotlib.pyplot as plt

participant_directory = "../data/participant-division"
time_window_sizes = []


def get_minimal_time_window_size(data_frame):
    unique_counts = {
        column: data_frame[column].nunique()
        for column in data_frame.columns
        if "empatica" in column or "samsung" in column
    }

    print(f"Unique counts: {unique_counts}")

    non_zero_counts = [count for count in unique_counts.values() if count > 0]

    if len(non_zero_counts) == 0:
        print("All columns have zero unique values. Using default minimal time window size.")
        return 0.001

    min_unique_count = min(non_zero_counts)

    print(f"Minimum non-zero unique count: {min_unique_count}")

    return 1.4 / min_unique_count


def process_data(data_frame, cognitive_load):
    data_frame.sort_values(
        by=["empatica_bvp_time", "empatica_eda_time", "empatica_temp_time", "samsung_bvp_time"],
        inplace=True
    )

    time_window_size = get_minimal_time_window_size(data_frame)

    if time_window_size is None:
        return None

    participant_id = data_frame["participant_id"].iloc[0]
    time_window_sizes.append((participant_id, cognitive_load, time_window_size))
    print(
        f"Participant ID: {data_frame['participant_id'].iloc[0]}, CL: {cl}, Time Window Size: {time_window_size:.4f}"
    )

    data_frame["time"] = data_frame[
        ["empatica_bvp_time", "empatica_eda_time", "empatica_temp_time", "samsung_bvp_time"]].max(axis=1)

    minimum_time = data_frame["time"].min()
    maximum_time = data_frame["time"].max()
    num_intervals = int((maximum_time - minimum_time) / time_window_size)
    time_intervals = [minimum_time + i * time_window_size for i in range(num_intervals + 1)]

    mean_seq_df = pd.DataFrame(
        columns=["ID", "empatica_bvp", "empatica_eda", "empatica_temp", "samsung_bvp", "time", "CL"])

    for i in range(num_intervals):
        start_time = time_intervals[i]
        end_time = time_intervals[i + 1]

        interval_rows = data_frame[(data_frame["time"] >= start_time) & (data_frame["time"] < end_time)]

        mean_values = interval_rows.mean()

        new_row = pd.DataFrame({
            "ID": [mean_values["participant_id"]],
            "empatica_bvp": [mean_values["empatica_bvp"]],
            "empatica_eda": [mean_values["empatica_eda"]],
            "empatica_temp": [mean_values["empatica_temp"]],
            "samsung_bvp": [mean_values["samsung_bvp"]],
            "time": [end_time],
            "CL": [cognitive_load],
        })

        mean_seq_df = pd.concat([mean_seq_df, new_row], ignore_index=True)

    return mean_seq_df


for participant_id in os.listdir(participant_directory):
    participant_folder = os.path.join(participant_directory, participant_id)

    for cl in ["LOW_CL", "HIGH_CL"]:
        input_file_name = f"{participant_id}_{cl}.csv"
        output_file_name = f"{participant_id}_{cl}_SEQ.csv"
        data_frame = pd.read_csv(os.path.join(participant_folder, input_file_name))

        mean_seq_df = process_data(data_frame, 1 if cl == "HIGH_CL" else 0)

        if mean_seq_df is not None:
            mean_seq_df.to_csv(os.path.join(participant_folder, output_file_name), index=False)

time_window_sizes_df = pd.DataFrame(time_window_sizes, columns=["ID", "CL", "Size"])

time_window_sizes_df = time_window_sizes_df.groupby(["ID", "CL"])["Size"].mean().unstack()

time_window_sizes_df = time_window_sizes_df.sort_values(by="ID")
print(f"The time windows sizes ID-CL wise:\n{time_window_sizes_df}")

time_window_sizes_df.plot(kind="bar", figsize=(12, 6))
plt.xlabel("Participant ID")
plt.ylabel("Time Window Size")
plt.title("Time Window Size per Participant ID and CL")
plt.legend(title="CL")
plt.grid(True)
plt.tight_layout()
plt.show()

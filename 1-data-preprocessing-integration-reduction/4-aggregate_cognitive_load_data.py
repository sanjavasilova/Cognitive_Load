import os
import pandas as pd


def ensure_directory_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def aggregate_data(participant_dir, cl_termination, aggregated_df):
    for participant_id in os.listdir(participant_dir):
        file_path = os.path.join(participant_dir, participant_id, f"{participant_id}{cl_termination}")
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            aggregated_df = pd.concat([aggregated_df, df], ignore_index=True)
    return aggregated_df


participant_directory = "../data/participant-division"
low_cl_output_path = "../data/high-low-separation/LOW_CL_AGG.csv"
high_cl_output_path = "../data/high-low-separation/HIGH_CL_AGG.csv"

output_directory = os.path.dirname(low_cl_output_path)
ensure_directory_exists(output_directory)

low_cl_df = pd.DataFrame()
high_cl_df = pd.DataFrame()

low_cl_df = aggregate_data(participant_directory, "_LOW_CL_SEQ.csv", low_cl_df)
high_cl_df = aggregate_data(participant_directory, "_HIGH_CL_SEQ.csv", high_cl_df)

low_cl_df.to_csv(low_cl_output_path, index=False)
high_cl_df.to_csv(high_cl_output_path, index=False)

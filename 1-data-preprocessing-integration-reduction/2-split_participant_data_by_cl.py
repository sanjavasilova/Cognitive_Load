import os
import pandas as pd


def save_participant_data(participant_data, cl_label, output_dir="data/participant-division"):
    for participant_id, df in participant_data.items():
        participant_dir = os.path.join(output_dir, str(participant_id))
        os.makedirs(participant_dir, exist_ok=True)

        file_name = f"{participant_id}_{cl_label}_CL.csv"
        file_path = os.path.join(participant_dir, file_name)

        df.to_csv(file_path, index=False)


def split_by_cognitive_load(df):
    low_cl_df = df[df["CL"] == 0]
    high_cl_df = df[df["CL"] == 1]

    low_cl_data = {participant_id: group for participant_id, group in low_cl_df.groupby("participant_id")}
    high_cl_data = {participant_id: group for participant_id, group in high_cl_df.groupby("participant_id")}

    return low_cl_data, high_cl_data


if __name__ == "__main__":
    df = pd.read_csv("../data/csv-aggregation/cogwear-agg.csv")

    low_cl_data, high_cl_data = split_by_cognitive_load(df)

    save_participant_data(low_cl_data, "LOW")
    save_participant_data(high_cl_data, "HIGH")

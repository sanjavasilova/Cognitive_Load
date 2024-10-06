import os
import pandas as pd

high_cl_path = "../data/high-low-separation/HIGH_CL_AGG.csv"
low_cl_path = "../data/high-low-separation/LOW_CL_AGG.csv"
final_dataset_path = "../data/pre-processed-dataset/processed_dataset.csv"


def merge_and_format_cl_data(high_cl_file, low_cl_file, output_file, decimal_places=6, id_limit=11):
    high_cl_df = pd.read_csv(high_cl_file)
    low_cl_df = pd.read_csv(low_cl_file)

    combined_df = pd.concat([high_cl_df, low_cl_df], ignore_index=True)

    float_columns = combined_df.select_dtypes(include=["float"]).columns
    columns_to_format = [col for col in float_columns if col not in ["CL", "ID"]]
    combined_df[columns_to_format] = combined_df[columns_to_format].map(
        lambda x: f"{x:.{decimal_places}f}"
    )

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    combined_df.to_csv(output_file, index=False)
    print(f"Data merged and saved to: {output_file}")

    for cl in range(2):
        for participant_id in range(id_limit):
            sample_count = len(combined_df[(combined_df["ID"] == participant_id) & (combined_df["CL"] == cl)])
            print(f"Samples for ID {participant_id}, CL {cl}: {sample_count}")


merge_and_format_cl_data(high_cl_path, low_cl_path, final_dataset_path)

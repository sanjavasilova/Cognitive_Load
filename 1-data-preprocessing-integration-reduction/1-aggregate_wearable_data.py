import os
import pandas as pd


def process_file(file_path, file, cognitive_load, participant_id):
    data_frame = pd.read_csv(file_path)
    processed_data = []

    for _, row in data_frame.iterrows():
        processed_data.append({
            "participant_id": participant_id,
            "empatica_bvp": row.get("bvp"),
            "empatica_bvp_time": row.get("time") if "empatica_bvp" in file else None,
            "empatica_eda": row.get("eda"),
            "empatica_eda_time": row.get("time") if "empatica_eda" in file else None,
            "empatica_temp": row.get("temp"),
            "empatica_temp_time": row.get("time") if "empatica_temp" in file else None,
            "samsung_bvp": row.get("PPG GREEN"),
            "samsung_bvp_time": row.get("time") if "samsung_bvp" in file else None,
            "CL": cognitive_load
        })

    return processed_data


def process_directory(subdirectory_path, cognitive_load, participant_id):
    aggregated_data = []
    for file in os.listdir(subdirectory_path):
        if file.endswith(".csv") and file.startswith(("empatica_bvp", "empatica_eda", "empatica_temp", "samsung_bvp")):
            file_path = os.path.join(subdirectory_path, file)
            aggregated_data.extend(process_file(file_path, file, cognitive_load, participant_id))
    return aggregated_data


def create_directory(root_directory, output_file):
    aggregated_data = []
    for directory in os.listdir(root_directory):
        participant_id = directory
        for subdirectory in ["baseline", "cognitive_load"]:
            subdirectory_path = os.path.join(root_directory, directory, subdirectory)
            if os.path.isdir(subdirectory_path):
                cognitive_load = 0 if subdirectory == "baseline" else 1
                aggregated_data.extend(process_directory(subdirectory_path, cognitive_load, participant_id))

    aggregated_data_frame = pd.DataFrame(aggregated_data)
    save_to_csv(aggregated_data_frame, output_file)


def save_to_csv(aggregated_data_frame, output_file):
    output_dir = os.path.dirname(output_file)
    os.makedirs(output_dir, exist_ok=True)
    aggregated_data_frame.to_csv(output_file, index=False)


if __name__ == "__main__":
    root_directory = "../data/cogwear/pilot"
    output_file = "../data/csv-aggregation/cogwear-agg.csv"
    create_directory(root_directory, output_file)

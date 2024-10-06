import pandas as pd

dataset_path = "../data/pre-processed-dataset/processed_dataset.csv"
filtered_dataset_path = "../data/pre-processed-dataset/filtered_dataset.csv"

df = pd.read_csv(dataset_path).dropna()
df.to_csv(filtered_dataset_path, index=False)

print(f"Filtered dataset saved at {filtered_dataset_path}")

id_upper_limit = 11
cl_classes = [0, 1]

for cl in cl_classes:
    for participant_id in range(id_upper_limit):
        sample_count = len(df[(df["ID"] == participant_id) & (df["CL"] == cl)])
        print(f"Number of samples for ID {participant_id} and CL {cl}: {sample_count}")

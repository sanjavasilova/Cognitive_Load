import pandas as pd

dataset_path = "../data/data-balancing/dataset.csv"
reduced_path = "../data/data-balancing/reduced_dataset.csv"

data_frame = pd.read_csv(dataset_path)
data_frame.drop(columns=["time"], inplace=True)

id_upper_limit = 11
cl_classes = [0, 1]
num_digits = 6

float_columns = data_frame.select_dtypes(include=["float"]).columns
float_columns = [col for col in float_columns if col not in ["CL", "ID"]]

for column in float_columns:
    data_frame[column] = data_frame[column].map(lambda x: f"{x:.{num_digits}f}")

data_frame.to_csv(reduced_path, index=True)
print(f"Final pre-processed dataset saved to {reduced_path}")

for cl in cl_classes:
    for id in range(id_upper_limit):
        samples = data_frame[(data_frame["ID"] == id) & (data_frame["CL"] == cl)]
        if not samples.empty:
            print(f"Number of samples for ID {id} and CL {cl}: {len(samples)}")

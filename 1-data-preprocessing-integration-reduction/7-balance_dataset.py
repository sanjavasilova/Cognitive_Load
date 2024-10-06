import random
import pandas as pd

dataset_path = "../data/pre-processed-dataset/filtered_dataset.csv"
balanced_dataset = "../data/data-balancing/dataset.csv"

data_frame = pd.read_csv(dataset_path)

class_counts = data_frame["CL"].value_counts()
for class_value, count in class_counts.items():
    print(f"The class {class_value} has {count} entries")

minority_sample_distribution = 100000
id_upper_limit = 11
cl_classes = [0, 1]

for cl in cl_classes:
    for id in range(id_upper_limit):
        samples = data_frame[(data_frame["ID"] == id) & (data_frame["CL"] == cl)]
        print(f"Number of samples for ID {id} and CL {cl}: {len(samples)}")
        if len(samples) < minority_sample_distribution and len(samples) > 0:
            minority_sample_distribution = len(samples)

print(f"The minority_sample_distribution value is {minority_sample_distribution}")

for cl in cl_classes:
    for id in range(id_upper_limit):
        samples = data_frame[(data_frame["ID"] == id) & (data_frame["CL"] == cl)]
        if len(samples) > minority_sample_distribution:
            indexes_to_remove = random.sample(samples.index.tolist(), len(samples) - minority_sample_distribution)
            data_frame = data_frame.drop(indexes_to_remove)

data_frame["ID"] = data_frame["ID"].astype(int)
data_frame["CL"] = data_frame["CL"].astype(int)
data_frame.to_csv(balanced_dataset, index=False)
print(f"Balanced dataset saved at {balanced_dataset}")

for cl in cl_classes:
    for id in range(id_upper_limit):
        samples = data_frame[(data_frame["ID"] == id) & (data_frame["CL"] == cl)]
        if len(samples) > 0:
            print(f"Number of samples for ID {id} and CL {cl}: {len(samples)}")

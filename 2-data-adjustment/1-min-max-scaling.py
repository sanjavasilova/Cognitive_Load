import pandas as pd

balanced_dataset_path = "../data/data-balancing/reduced_dataset.csv"
normalized_dataset_path = "../data/data-scaling/min_max_normalization.csv"

data_frame = pd.read_csv(balanced_dataset_path)

features_to_normalize = ["empatica_bvp", "empatica_eda", "empatica_temp", "samsung_bvp"]


def min_max_scaling(x):
    min_value = x.min()
    max_value = x.max()
    return ((x - min_value) / (max_value - min_value)).apply(lambda x: f"{x:.10f}")


for feature in features_to_normalize:
    data_frame[feature] = min_max_scaling(data_frame[feature])

data_frame.to_csv(normalized_dataset_path, index=False)

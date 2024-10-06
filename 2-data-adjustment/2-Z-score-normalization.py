import pandas as pd

balanced_dataset_path = "../data/data-balancing/reduced_dataset.csv"
z_score_normalized_path = "../data/data-scaling/z-score-standardization.csv"

data_frame = pd.read_csv(balanced_dataset_path)

features_to_standardize = ["empatica_bvp", "empatica_eda", "empatica_temp", "samsung_bvp"]


def z_score_standardization(x):
    mean_value = x.mean()
    std_deviation = x.std()
    return ((x - mean_value) / std_deviation).apply(lambda x: f"{x:.10f}")


for feature in features_to_standardize:
    data_frame[feature] = z_score_standardization(data_frame[feature])

data_frame.to_csv(z_score_normalized_path, index=False)

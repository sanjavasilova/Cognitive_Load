import pandas as pd

datasets = {
    "min_max": {
        "data_path": "../data/data-scaling/min_max_normalization.csv",
        "train_path": "../data/data-split/min-max/training.csv",
        "test_path": "../data/data-split/min-max/testing.csv"
    },
    "z_score": {
        "data_path": "../data/data-scaling/z-score-standardization.csv",
        "train_path": "../data/data-split/z-score/training.csv",
        "test_path": "../data/data-split/z-score/testing.csv"
    }
}

train_ids = [0, 2, 4, 6, 8, 9]
test_id = 10

for scaling_method, paths in datasets.items():
    data_frame = pd.read_csv(paths["data_path"])

    train_data_frame = data_frame[data_frame["ID"].isin(train_ids)].reset_index(drop=True)
    test_data_frame = data_frame[data_frame["ID"] == test_id].reset_index(drop=True)

    train_data_frame.to_csv(paths["train_path"], index=False)
    test_data_frame.to_csv(paths["test_path"], index=False)

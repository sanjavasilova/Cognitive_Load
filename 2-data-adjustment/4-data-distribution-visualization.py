import pandas as pd
import matplotlib.pyplot as plt

min_max_path = "../data/data-scaling/min_max_normalization.csv"
z_score_path = "../data/data-scaling/z-score-standardization.csv"

min_max_df = pd.read_csv(min_max_path)
z_score_df = pd.read_csv(z_score_path)


def plot_features(data, title):
    features = [col for col in data.columns if col not in ["Unnamed: 0", "ID", "CL"]]
    plt.figure(figsize=(12, 6))
    for feature in features:
        plt.plot(data.index, data[feature], label=feature)

    plt.title(title)
    plt.xlabel("Samples")
    plt.ylabel("Feature Values")
    plt.legend()
    plt.grid(True)
    plt.show()


print("Min-Max Normalized Dataset Preview:")
print(min_max_df.head(), "\n")
plot_features(min_max_df, "Min-Max Normalization Feature Distribution")

print("Z-Score Standardized Dataset Preview:")
print(z_score_df.head(), "\n")
plot_features(z_score_df, "Z-Score Standardization Feature Distribution")

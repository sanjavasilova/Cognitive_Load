import pandas as pd
import matplotlib.pyplot as plt

dataset_path = "../data/data-balancing/reduced_dataset.csv"
data_frame = pd.read_csv(dataset_path)

low_cl_samples = []
high_cl_samples = []

num_participants = 11

for participant_id in range(num_participants):
    low_cl_count = len(data_frame[(data_frame["ID"] == participant_id) & (data_frame["CL"] == 0)])
    high_cl_count = len(data_frame[(data_frame["ID"] == participant_id) & (data_frame["CL"] == 1)])

    if low_cl_count > 0 and high_cl_count > 1:
        low_cl_samples.append(low_cl_count)
        high_cl_samples.append(high_cl_count)

participant_ids = data_frame["ID"].unique()

fig, ax = plt.subplots(figsize=(12, 6))
bar_width = 0.35

x_pos = participant_ids

low_cl_bars = ax.bar(
    x_pos - bar_width / 2, low_cl_samples, bar_width, color="pink", label="Low CL (0)"
)
high_cl_bars = ax.bar(
    x_pos + bar_width / 2, high_cl_samples, bar_width, color="purple", label="High CL (1)"
)

ax.set_xlabel("Participant ID")
ax.set_ylabel("Sample Count")
ax.set_title("Samples per Participant by Cognitive Load Category")
ax.set_xticks(x_pos)
ax.legend()


def add_labels(bars):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f'{height}',
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center", va="bottom"
        )


add_labels(low_cl_bars)
add_labels(high_cl_bars)

plt.tight_layout()
plt.show()

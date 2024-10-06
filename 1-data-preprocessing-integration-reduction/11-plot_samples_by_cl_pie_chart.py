import pandas as pd
import matplotlib.pyplot as plt

dataset_path = "../data/data-balancing/reduced_dataset.csv"
data_frame = pd.read_csv(dataset_path)

cognitive_loads = [0, 1]
participant_count = 11

for cl in cognitive_loads:
    for participant_id in range(participant_count):
        samples = data_frame[(data_frame["ID"] == participant_id) & (data_frame["CL"] == cl)]
        if len(samples) > 0:
            print(f"Samples for ID {participant_id} and CL {cl}: {len(samples)}")

total_samples_cl_0 = sum(
    len(data_frame[(data_frame["ID"] == participant_id) & (data_frame["CL"] == 0)])
    for participant_id in range(participant_count)
)
total_samples_cl_1 = sum(
    len(data_frame[(data_frame["ID"] == participant_id) & (data_frame["CL"] == 1)])
    for participant_id in range(participant_count)
)

load_labels = ["Low CL (0)", "High CL (1)"]
sample_sizes = [total_samples_cl_0, total_samples_cl_1]
load_colors = ["pink", "purple"]

plt.figure(figsize=(6, 6))
plt.pie(sample_sizes, labels=load_labels, colors=load_colors, autopct="%1.0f%%", startangle=90)
plt.title("Distribution of Samples by Cognitive Load")
plt.axis("auto")
plt.show()

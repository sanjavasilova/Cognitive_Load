import pandas as pd
import matplotlib.pyplot as plt

reduced_dataset = "../data/data-balancing/reduced_dataset.csv"

data_frame = pd.read_csv(reduced_dataset)

participants = 11
cognitive_loads = [0, 1]

for cognitive_load in cognitive_loads:
    for participant in range(participants):
        filtered_samples = data_frame[(data_frame["ID"] == participant) & (data_frame["CL"] == cognitive_load)]
        if not filtered_samples.empty:
            print(f"Samples for Participant {participant} and CL {cognitive_load}: {len(filtered_samples)}")

low_cognitive_samples = []
high_cognitive_samples = []

for participant in range(participants):
    low_cl_count = data_frame[(data_frame["ID"] == participant) & (data_frame["CL"] == 0)].shape[0]
    high_cl_count = data_frame[(data_frame["ID"] == participant) & (data_frame["CL"] == 1)].shape[0]

    low_cognitive_samples.append(low_cl_count)
    high_cognitive_samples.append(high_cl_count)

participant_ids = range(participants)

plt.figure(figsize=(12, 4))
plt.bar(participant_ids, low_cognitive_samples, color="pink", label="Low Cognitive Load (CL 0)")
plt.bar(participant_ids, high_cognitive_samples, bottom=low_cognitive_samples, color="purple", label="High Cognitive Load (CL 1)")

plt.xlabel("Participant ID")
plt.ylabel("Sample Count")
plt.title("Sample Distribution per Participant and Cognitive Load")
plt.legend()
plt.xticks(participant_ids)
plt.show()

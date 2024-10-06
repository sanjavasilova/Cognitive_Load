import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

train_df = pd.read_csv("../data/data-split/z-score/training.csv")
test_df = pd.read_csv("../data/data-split/z-score/testing.csv")

x_train_empatica, y_train_empatica = train_df[["empatica_bvp", "empatica_eda", "empatica_temp"]], train_df["CL"]
x_train_samsung, y_train_samsung = train_df[["samsung_bvp"]], train_df["CL"]
x_test_empatica, y_test_empatica = test_df[["empatica_bvp", "empatica_eda", "empatica_temp"]], test_df["CL"]
x_test_samsung, y_test_samsung = test_df[["samsung_bvp"]], test_df["CL"]

model_empatica = LogisticRegression(random_state=42)
model_samsung = LogisticRegression(random_state=42)

model_empatica.fit(x_train_empatica, y_train_empatica)
model_samsung.fit(x_train_samsung, y_train_samsung)

y_pred_empatica = model_empatica.predict(x_test_empatica)
y_pred_samsung = model_samsung.predict(x_test_samsung)

accuracy_empatica = accuracy_score(y_test_empatica, y_pred_empatica)
accuracy_samsung = accuracy_score(y_test_samsung, y_pred_samsung)

print("Empatica Accuracy:", accuracy_empatica)
print("Samsung Accuracy:", accuracy_samsung)

print("Empatica Classification Report:\n", classification_report(y_test_empatica, y_pred_empatica, zero_division=1))
print("Samsung Classification Report:\n", classification_report(y_test_samsung, y_pred_samsung, zero_division=1))

conf_mat_empatica = confusion_matrix(y_test_empatica, y_pred_empatica)
conf_mat_samsung = confusion_matrix(y_test_samsung, y_pred_samsung)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].imshow(conf_mat_empatica, cmap="OrRd")
axes[0].set_title("Empatica Confusion Matrix")
axes[0].set_xticks([0, 1])
axes[0].set_yticks([0, 1])
for i in range(conf_mat_empatica.shape[0]):
    for j in range(conf_mat_empatica.shape[1]):
        axes[0].text(j, i, str(conf_mat_empatica[i, j]), ha="center", va="center", color="black")

axes[1].imshow(conf_mat_samsung, cmap="OrRd")
axes[1].set_title("Samsung Confusion Matrix")
axes[1].set_xticks([0, 1])
axes[1].set_yticks([0, 1])
for i in range(conf_mat_samsung.shape[0]):
    for j in range(conf_mat_samsung.shape[1]):
        axes[1].text(j, i, str(conf_mat_samsung[i, j]), ha="center", va="center", color="black")

plt.tight_layout()
plt.show()

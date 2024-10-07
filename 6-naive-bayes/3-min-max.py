import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

train_data = pd.read_csv("../data/data-split/min-max/training.csv")
test_data = pd.read_csv("../data/data-split/min-max/testing.csv")

features_empatica = train_data[["empatica_bvp", "empatica_eda", "empatica_temp"]]
target_empatica = train_data["CL"]

features_samsung = train_data[["samsung_bvp"]]
target_samsung = train_data["CL"]

test_features_empatica = test_data[["empatica_bvp", "empatica_eda", "empatica_temp"]]
test_target_empatica = test_data["CL"]

test_features_samsung = test_data[["samsung_bvp"]]
test_target_samsung = test_data["CL"]

model_empatica = GaussianNB()
model_samsung = GaussianNB()

model_empatica.fit(features_empatica, target_empatica)
model_samsung.fit(features_samsung, target_samsung)

predictions_empatica = model_empatica.predict(test_features_empatica)
predictions_samsung = model_samsung.predict(test_features_samsung)

accuracy_empatica = accuracy_score(test_target_empatica, predictions_empatica)
accuracy_samsung = accuracy_score(test_target_samsung, predictions_samsung)

print(f"Empatica Model Accuracy: {accuracy_empatica:.4f}")
print(f"Samsung Model Accuracy: {accuracy_samsung:.4f}")

print("\nEmpatica Classification Report:\n",
      classification_report(test_target_empatica, predictions_empatica, zero_division=1))
print("\nSamsung Classification Report:\n",
      classification_report(test_target_samsung, predictions_samsung, zero_division=1))

confusion_empatica = confusion_matrix(test_target_empatica, predictions_empatica)
confusion_samsung = confusion_matrix(test_target_samsung, predictions_samsung)


def plot_confusion_matrix(cm, title):
    plt.figure(figsize=(8, 4))
    plt.imshow(cm, interpolation='nearest', cmap='OrRd')
    plt.title(title)
    plt.colorbar()
    plt.xticks([0, 1], ["0", "1"])
    plt.yticks([0, 1], ["0", "1"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha='center', va='center', color='white' if cm[i, j] > cm.max() / 2 else 'black')
    plt.show()


plot_confusion_matrix(confusion_empatica, "Naive Bayes - Min-Max - Empatica")
plot_confusion_matrix(confusion_samsung, "Naive Bayes - Min-Max - Samsung")

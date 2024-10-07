import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

train_data = pd.read_csv("../data/data-split/min-max/training.csv")
test_data = pd.read_csv("../data/data-split/min-max/testing.csv")

x_train_empatica = train_data[["empatica_bvp", "empatica_eda", "empatica_temp"]]
y_train_empatica = train_data["CL"]

x_train_samsung = train_data[["samsung_bvp"]]
y_train_samsung = train_data["CL"]

x_test_empatica = test_data[["empatica_bvp", "empatica_eda", "empatica_temp"]]
y_test_empatica = test_data["CL"]

x_test_samsung = test_data[["samsung_bvp"]]
y_test_samsung = test_data["CL"]

N_FOLDS = 5


def cross_validate(model, X, y):
    kf = KFold(n_splits=N_FOLDS)
    best_model = None
    max_accuracy = 0

    for train_idx, val_idx in kf.split(X):
        x_train_fold, x_val_fold = X.iloc[train_idx], X.iloc[val_idx]
        y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]

        model.fit(x_train_fold, y_train_fold)
        preds = model.predict(x_val_fold)
        accuracy = accuracy_score(y_val_fold, preds)

        if accuracy > max_accuracy:
            max_accuracy = accuracy
            best_model = model

    return best_model


model_empatica = GaussianNB()
model_samsung = GaussianNB()

model_empatica = cross_validate(model_empatica, x_train_empatica, y_train_empatica)
model_samsung = cross_validate(model_samsung, x_train_samsung, y_train_samsung)

predictions_empatica = model_empatica.predict(x_test_empatica)
predictions_samsung = model_samsung.predict(x_test_samsung)

accuracy_empatica = accuracy_score(y_test_empatica, predictions_empatica)
accuracy_samsung = accuracy_score(y_test_samsung, predictions_samsung)

print(f"Empatica Model Accuracy: {accuracy_empatica:.4f}")
print(f"Samsung Model Accuracy: {accuracy_samsung:.4f}")

print("\nEmpatica Classification Report:\n",
      classification_report(y_test_empatica, predictions_empatica, zero_division=1))
print("\nSamsung Classification Report:\n", classification_report(y_test_samsung, predictions_samsung, zero_division=1))

conf_matrix_empatica = confusion_matrix(y_test_empatica, predictions_empatica)
conf_matrix_samsung = confusion_matrix(y_test_samsung, predictions_samsung)


def display_confusion_matrix(cm, title):
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


display_confusion_matrix(conf_matrix_empatica, "Naive Bayes - Cross-Validation - Min-Max - Empatica")
display_confusion_matrix(conf_matrix_samsung, "Naive Bayes - Cross-Validation - Min-Max - Samsung")

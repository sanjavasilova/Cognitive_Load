import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

train_path = "../data/data-split/z-score/training.csv"
test_path = "../data/data-split/z-score/testing.csv"

train_data = pd.read_csv(train_path)
test_data = pd.read_csv(test_path)

X_train_empatica = train_data[["empatica_bvp", "empatica_eda", "empatica_temp"]]
y_train_empatica = train_data["CL"]

X_train_samsung = train_data[["samsung_bvp"]]
y_train_samsung = train_data["CL"]

X_test_empatica = test_data[["empatica_bvp", "empatica_eda", "empatica_temp"]]
y_test_empatica = test_data["CL"]

X_test_samsung = test_data[["samsung_bvp"]]
y_test_samsung = test_data["CL"]

n_splits = 5


def cross_validate_and_select_best_model(clf, x, y):
    kfold = KFold(n_splits=n_splits)
    optimal_model = None
    max_accuracy = 0

    for train_idx, val_idx in kfold.split(x):
        x_train_split, x_val_split = x.iloc[train_idx], x.iloc[val_idx]
        y_train_split, y_val_split = y.iloc[train_idx], y.iloc[val_idx]

        clf.fit(x_train_split, y_train_split)
        predictions = clf.predict(x_val_split)
        fold_accuracy = accuracy_score(y_val_split, predictions)

        if fold_accuracy > max_accuracy:
            max_accuracy = fold_accuracy
            optimal_model = clf

    return optimal_model


nb_empatica = GaussianNB()
nb_samsung = GaussianNB()

best_empatica_model = cross_validate_and_select_best_model(nb_empatica, X_train_empatica, y_train_empatica)
best_samsung_model = cross_validate_and_select_best_model(nb_samsung, X_train_samsung, y_train_samsung)

empatica_predictions = best_empatica_model.predict(X_test_empatica)
samsung_predictions = best_samsung_model.predict(X_test_samsung)

empatica_accuracy = accuracy_score(y_test_empatica, empatica_predictions)
samsung_accuracy = accuracy_score(y_test_samsung, samsung_predictions)

print(f"Accuracy (Empatica Model): {empatica_accuracy}")
print(f"Accuracy (Samsung Model): {samsung_accuracy}")

print("\nEmpatica Model Classification Report:\n",
      classification_report(y_test_empatica, empatica_predictions, zero_division=1))
print("\nSamsung Model Classification Report:\n",
      classification_report(y_test_samsung, samsung_predictions, zero_division=1))

empatica_conf_matrix = confusion_matrix(y_test_empatica, empatica_predictions)
samsung_conf_matrix = confusion_matrix(y_test_samsung, samsung_predictions)

print("\nConfusion Matrix (Empatica):\n", empatica_conf_matrix)
print("\nConfusion Matrix (Samsung):\n", samsung_conf_matrix)

plt.figure(figsize=(10, 5))
plt.imshow(empatica_conf_matrix, cmap="OrRd")
plt.title("Empatica Model - Naive Bayes - Cross-Validation - Z-Score")
plt.colorbar()
plt.xticks([0, 1], labels=["0", "1"])
plt.yticks([0, 1], labels=["0", "1"])
for i in range(empatica_conf_matrix.shape[0]):
    for j in range(empatica_conf_matrix.shape[1]):
        plt.text(j, i, empatica_conf_matrix[i, j], ha="center", va="center")
plt.show()

plt.figure(figsize=(10, 5))
plt.imshow(samsung_conf_matrix, cmap="OrRd")
plt.title("Samsung Model - Naive Bayes - Cross-Validation - Z-Score")
plt.colorbar()
plt.xticks([0, 1], labels=["0", "1"])
plt.yticks([0, 1], labels=["0", "1"])
for i in range(samsung_conf_matrix.shape[0]):
    for j in range(samsung_conf_matrix.shape[1]):
        plt.text(j, i, samsung_conf_matrix[i, j], ha="center", va="center")
plt.show()

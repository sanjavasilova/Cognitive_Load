import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt


def load_data(train_path, test_path):
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    return train_data, test_data


def split_features_targets(data, empatica_cols, samsung_col, target_col):
    x_empatica = data[empatica_cols]
    x_samsung = data[[samsung_col]]
    y = data[target_col]
    return x_empatica, x_samsung, y


def perform_grid_search(x, y, param_grid):
    grid_search = GridSearchCV(LogisticRegression(random_state=0), param_grid, cv=5)
    grid_search.fit(x, y)
    return grid_search.best_estimator_, grid_search.best_params_


def evaluate_model(model, x_test, y_test):
    predictions = model.predict(x_test)
    acc = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions)
    confusion = confusion_matrix(y_test, predictions)
    return acc, report, confusion


def plot_confusion_matrices(conf_matrix1, conf_matrix2, title1, title2):
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for matrix, ax_idx, title in zip([conf_matrix1, conf_matrix2], [0, 1], [title1, title2]):
        ax[ax_idx].imshow(matrix, cmap="OrRd")
        ax[ax_idx].set_title(title)
        ax[ax_idx].set_xticks([0, 1])
        ax[ax_idx].set_yticks([0, 1])
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ax[ax_idx].text(j, i, str(matrix[i, j]), ha="center", va="center", color="black")
    plt.tight_layout()
    plt.show()


train_file = "../data/data-split/z-score/training.csv"
test_file = "../data/data-split/z-score/testing.csv"

empatica_features = ["empatica_bvp", "empatica_eda", "empatica_temp"]
samsung_feature = "samsung_bvp"
target_column = "CL"

train_data, test_data = load_data(train_file, test_file)
x_train_empatica, x_train_samsung, y_train = split_features_targets(train_data, empatica_features, samsung_feature,
                                                                    target_column)
x_test_empatica, x_test_samsung, y_test = split_features_targets(test_data, empatica_features, samsung_feature,
                                                                 target_column)

param_grid = {
    "C": [0.001, 0.01, 0.1, 1, 10, 100],
    "solver": ["liblinear", "lbfgs", "newton-cg", "sag", "saga"],
    "class_weight": [None, "balanced"],
}

best_empatica_model, empatica_params = perform_grid_search(x_train_empatica, y_train, param_grid)
best_samsung_model, samsung_params = perform_grid_search(x_train_samsung, y_train, param_grid)

print(f"Best Params for Empatica Model: {empatica_params}")
print(f"Best Params for Samsung Model: {samsung_params}")

empatica_acc, empatica_report, empatica_conf_matrix = evaluate_model(best_empatica_model, x_test_empatica, y_test)
samsung_acc, samsung_report, samsung_conf_matrix = evaluate_model(best_samsung_model, x_test_samsung, y_test)

print(f"Empatica Model Accuracy: {empatica_acc}\n")
print(f"Empatica Model Classification Report:\n{empatica_report}\n")
print(f"Empatica Model Confusion Matrix:\n{empatica_conf_matrix}\n")

print(f"Samsung Model Accuracy: {samsung_acc}\n")
print(f"Samsung Model Classification Report:\n{samsung_report}\n")
print(f"Samsung Model Confusion Matrix:\n{samsung_conf_matrix}\n")

plot_confusion_matrices(empatica_conf_matrix, samsung_conf_matrix, "Empatica Confusion Matrix",
                        "Samsung Confusion Matrix")

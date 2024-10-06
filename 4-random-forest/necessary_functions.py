import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

ALGORITHM_NAME = "Random Forest"
scaling_method = "z-score"

train_path = f"../data/data-split/{scaling_method}/training.csv"
test_path = f"../data/data-split/{scaling_method}/testing.csv"


def load_data(train_path, test_path):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df


def split_data(df, features, target):
    x = df[features]
    y = df[target]
    return x, y


def flatten_features(x):
    return np.hstack([x[col].values[:, np.newaxis] for col in x.columns])


def evaluate_model(model, x_test, y_test):
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    conf_mat = confusion_matrix(y_test, y_pred)
    return accuracy, report, conf_mat


def visualize_confusion_matrices(cm1, cm2, params, scaling):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))

    ax[0].imshow(cm1, cmap="OrRd")
    ax[0].set_title(f"{ALGORITHM_NAME} ({params}) - {scaling} - Empatica")
    ax[0].set_xlabel("Predicted")
    ax[0].set_ylabel("Actual")
    ax[0].set_xticks([0, 1])
    ax[0].set_yticks([0, 1])
    ax[0].set_xticklabels(['0', '1'])
    ax[0].set_yticklabels(['0', '1'])

    for i in range(cm1.shape[0]):
        for j in range(cm1.shape[1]):
            ax[0].text(j, i, cm1[i, j], ha="center", va="center", color="black")

    ax[1].imshow(cm2, cmap="OrRd")
    ax[1].set_title(f"{ALGORITHM_NAME} ({params}) - {scaling} - Samsung")
    ax[1].set_xlabel("Predicted")
    ax[1].set_ylabel("Actual")
    ax[1].set_xticks([0, 1])
    ax[1].set_yticks([0, 1])
    ax[1].set_xticklabels(['0', '1'])
    ax[1].set_yticklabels(['0', '1'])

    for i in range(cm2.shape[0]):
        for j in range(cm2.shape[1]):
            ax[1].text(j, i, cm2[i, j], ha="center", va="center", color="black")

    plt.tight_layout()
    plt.show()

train_df, test_df = load_data(train_path, test_path)

empatica_features = ["empatica_bvp", "empatica_eda", "empatica_temp"]
samsung_features = ["samsung_bvp"]
target = "CL"

x_train_empatica, y_train_empatica = split_data(train_df, empatica_features, target)
x_test_empatica, y_test_empatica = split_data(test_df, empatica_features, target)

x_train_samsung, y_train_samsung = split_data(train_df, samsung_features, target)
x_test_samsung, y_test_samsung = split_data(test_df, samsung_features, target)

x_train_empatica_flat = flatten_features(x_train_empatica)
x_test_empatica_flat = flatten_features(x_test_empatica)

import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

devices = ["empatica", "samsung"]
kernel_type = "SVM"
random_state = 11


def train_and_evaluate(model, x_train, y_train, x_test, y_test, device_name):
    y_train = y_train.values.ravel()
    y_test = y_test.values.ravel()

    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"{device_name.capitalize()} Accuracy: {accuracy:.4f}")
    print(f"\nClassification Report for {device_name.capitalize()}:\n",
          classification_report(y_test, y_pred, zero_division=1))

    return y_pred


def visualize_confusion_matrices(conf_mat_empatica, conf_mat_samsung, kernel, scaling):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].imshow(conf_mat_empatica, cmap="OrsRd")
    axes[0].set_title(f"{kernel_type} ({kernel}) - {scaling} - Empatica")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Actual")

    axes[0].set_xticks([0, 1])
    axes[0].set_xticklabels([0, 1])
    axes[0].set_yticks([0, 1])
    axes[0].set_yticklabels([0, 1])

    for i in range(conf_mat_empatica.shape[0]):
        for j in range(conf_mat_empatica.shape[1]):
            axes[0].text(j, i, str(conf_mat_empatica[i, j]), ha="center", va="center")

    axes[1].imshow(conf_mat_samsung, cmap="OrRd")
    axes[1].set_title(f"{kernel_type} ({kernel}) - {scaling} - Samsung")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("Actual")

    axes[1].set_xticks([0, 1])
    axes[1].set_xticklabels([0, 1])
    axes[1].set_yticks([0, 1])
    axes[1].set_yticklabels([0, 1])

    for i in range(conf_mat_samsung.shape[0]):
        for j in range(conf_mat_samsung.shape[1]):
            axes[1].text(j, i, str(conf_mat_samsung[i, j]), ha="center", va="center")

    plt.tight_layout()
    plt.show()


def run_svm_model(kernel, scaling, train_path, test_path):
    empatica_features = ["empatica_bvp", "empatica_eda", "empatica_temp"]
    samsung_features = ["samsung_bvp"]
    target_column = ["CL"]

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    x_train_empatica = train_df[empatica_features]
    y_train_empatica = train_df[target_column]
    x_test_empatica = test_df[empatica_features]
    y_test_empatica = test_df[target_column]

    x_train_samsung = train_df[samsung_features]
    y_train_samsung = train_df[target_column]
    x_test_samsung = test_df[samsung_features]
    y_test_samsung = test_df[target_column]

    model_empatica = SVC(kernel=kernel, random_state=random_state)
    model_samsung = SVC(kernel=kernel, random_state=random_state)

    y_pred_empatica = train_and_evaluate(model_empatica, x_train_empatica, y_train_empatica, x_test_empatica,
                                         y_test_empatica, "Empatica")
    y_pred_samsung = train_and_evaluate(model_samsung, x_train_samsung, y_train_samsung, x_test_samsung, y_test_samsung,
                                        "Samsung")

    conf_matrix_empatica = confusion_matrix(y_test_empatica, y_pred_empatica)
    conf_matrix_samsung = confusion_matrix(y_test_samsung, y_pred_samsung)

    visualize_confusion_matrices(conf_matrix_empatica, conf_matrix_samsung, kernel, scaling)


if __name__ == "__main__":
    SCALINGS = ["z-score", "min-max"]

    KERNELS = ["rbf", "linear", "poly", "sigmoid"]

    for kernel in KERNELS:
        for scaling in SCALINGS:
            print(f"SVM ({kernel}) - {scaling}")
            TRAINING_FILE_PATH = f"../data/data-split/{scaling}/training.csv"
            TESTING_FILE_PATH = f"../data/data-split/{scaling}/testing.csv"

            run_svm_model(
                kernel, scaling, TRAINING_FILE_PATH, TESTING_FILE_PATH
            )


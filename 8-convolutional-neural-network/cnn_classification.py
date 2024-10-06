import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neural_network import MLPClassifier

feature_columns_empatica = ["empatica_bvp", "empatica_eda", "empatica_temp"]
feature_columns_samsung = ["samsung_bvp"]


def cnn_classification_using_hidden_layer(hidden_layer_empatica, hidden_layer_samsung, scaling):
    training_df = pd.read_csv(f"../data/data-split/{scaling}/training.csv")
    testing_df = pd.read_csv(f"../data/data-split/{scaling}/testing.csv")

    x_empatica = training_df[feature_columns_empatica]
    y_empatica = training_df["CL"]

    x_samsung = training_df[feature_columns_samsung]
    y_samsung = training_df["CL"]

    cnn_empatica = MLPClassifier(hidden_layer_sizes=hidden_layer_empatica, max_iter=500,
                                 random_state=1234)
    cnn_empatica.fit(x_empatica, y_empatica)

    cnn_samsung = MLPClassifier(hidden_layer_sizes=hidden_layer_samsung, max_iter=500,
                                random_state=1234)
    cnn_samsung.fit(x_samsung, y_samsung)

    y_pred_empatica = cnn_empatica.predict(testing_df[feature_columns_empatica])
    y_pred_samsung = cnn_samsung.predict(testing_df[feature_columns_samsung])

    print("\nClassification Report (Empatica Model CNN):\n", classification_report(testing_df["CL"], y_pred_empatica))
    print("\nClassification Report (Samsung Model CNN):\n", classification_report(testing_df["CL"], y_pred_samsung))

    conf_mat_empatica = confusion_matrix(testing_df["CL"], y_pred_empatica)
    conf_mat_samsung = confusion_matrix(testing_df["CL"], y_pred_samsung)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].imshow(conf_mat_empatica, cmap="OrRd")
    axes[0].set_title(f"Empatica Model CNN {scaling.title()}")
    axes[0].set_xticks([0, 1])
    axes[0].set_yticks([0, 1])
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Actual")
    for i in range(conf_mat_empatica.shape[0]):
        for j in range(conf_mat_empatica.shape[1]):
            axes[0].text(j, i, str(conf_mat_empatica[i, j]), ha="center", va="center", color="black")

    axes[1].imshow(conf_mat_samsung, cmap="OrRd")
    axes[1].set_title(f"Samsung Model CNN {scaling.title()}")
    axes[1].set_xticks([0, 1])
    axes[1].set_yticks([0, 1])
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("Actual")
    for i in range(conf_mat_samsung.shape[0]):
        for j in range(conf_mat_samsung.shape[1]):
            axes[1].text(j, i, str(conf_mat_samsung[i, j]), ha="center", va="center", color="black")

    plt.tight_layout()
    plt.show()

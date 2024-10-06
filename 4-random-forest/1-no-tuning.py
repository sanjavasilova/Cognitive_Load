from sklearn.ensemble import RandomForestClassifier

from necessary_functions import *


def train_model(x, y, depth=20, random_state=42):
    model = RandomForestClassifier(max_depth=depth, random_state=random_state)
    model.fit(x, y)
    return model


if __name__ == "__main__":
    empatica_model = train_model(x_train_empatica_flat, y_train_empatica)
    samsung_model = train_model(x_train_samsung, y_train_samsung)

    empatica_acc, empatica_report, empatica_conf_mat = evaluate_model(empatica_model, x_test_empatica_flat,
                                                                      y_test_empatica)
    samsung_acc, samsung_report, samsung_conf_mat = evaluate_model(samsung_model, x_test_samsung, y_test_samsung)

    print(f"Empatica Model (RF) Accuracy: {empatica_acc}")
    print(f"Samsung Model (RF) Accuracy: {samsung_acc}")

    print(f"\nClassification Report (Empatica Model RF):\n{empatica_report}")
    print(f"\nClassification Report (Samsung Model RF):\n{samsung_report}")

    visualize_confusion_matrices(empatica_conf_mat, samsung_conf_mat, "no tuning", scaling_method)

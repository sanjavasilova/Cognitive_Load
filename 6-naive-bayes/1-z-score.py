from sklearn.naive_bayes import GaussianNB
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt


def load_data(train_file, test_file):
    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)
    return train_df, test_df


def prepare_features_targets(df, empatica_columns, samsung_column, target_column):
    x_empatica = df[empatica_columns]
    x_samsung = df[[samsung_column]]
    y = df[target_column]
    return x_empatica, x_samsung, y


def train_model(x, y):
    model = GaussianNB()
    model.fit(x, y)
    return model


def evaluate_model(model, x_test, y_test):
    predictions = model.predict(x_test)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions, zero_division=1)
    confusion_mat = confusion_matrix(y_test, predictions)
    return accuracy, report, confusion_mat


def display_confusion_matrix(conf_matrix, model_name):
    plt.imshow(conf_matrix, cmap="OrRd")
    plt.title(f"Naive Bayes - Z-Score - {model_name}")
    plt.colorbar()
    plt.xticks([0, 1], ["0", "1"])
    plt.yticks([0, 1], ["0", "1"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            plt.text(j, i, str(conf_matrix[i, j]), ha="center", va="center", color="black")
    plt.show()


train_path = "../data/data-split/z-score/training.csv"
test_path = "../data/data-split/z-score/testing.csv"

empatica_cols = ["empatica_bvp", "empatica_eda", "empatica_temp"]
samsung_col = "samsung_bvp"
target_col = "CL"

train_data, test_data = load_data(train_path, test_path)

x_train_empatica, x_train_samsung, y_train = prepare_features_targets(train_data, empatica_cols, samsung_col,
                                                                      target_col)
x_test_empatica, x_test_samsung, y_test = prepare_features_targets(test_data, empatica_cols, samsung_col, target_col)

empatica_nb_model = train_model(x_train_empatica, y_train)
samsung_nb_model = train_model(x_train_samsung, y_train)

empatica_acc, empatica_report, empatica_conf_matrix = evaluate_model(empatica_nb_model, x_test_empatica, y_test)
print(f"Empatica Model (NB) Accuracy: {empatica_acc}")
print(f"\nClassification Report (Empatica NB):\n{empatica_report}")
print(f"\nConfusion Matrix (Empatica NB):\n{empatica_conf_matrix}")

samsung_acc, samsung_report, samsung_conf_matrix = evaluate_model(samsung_nb_model, x_test_samsung, y_test)
print(f"Samsung Model (NB) Accuracy: {samsung_acc}")
print(f"\nClassification Report (Samsung NB):\n{samsung_report}")
print(f"\nConfusion Matrix (Samsung NB):\n{samsung_conf_matrix}")

display_confusion_matrix(empatica_conf_matrix, "Empatica Model")
display_confusion_matrix(samsung_conf_matrix, "Samsung Model")

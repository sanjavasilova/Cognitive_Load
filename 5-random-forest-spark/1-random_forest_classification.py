import dask.dataframe as dd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler


def load_data(file_path):
    return dd.read_csv(file_path)


def assemble_features(data, feature_columns, label_column):
    x = data[feature_columns].compute()
    y = data[label_column].compute().astype(int)
    return x, y


def train_random_forest(x_train, y_train, max_depth):
    rf_model = RandomForestClassifier(max_depth=max_depth, random_state=42)
    return rf_model.fit(x_train, y_train)


def make_predictions(model, x_test):
    return model.predict(x_test)


def plot_confusion_matrix(conf_matrix, title, ax):
    ax.imshow(conf_matrix, cmap="OrRd")
    ax.set_title(title)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            ax.text(j, i, str(int(conf_matrix[i, j])), ha="center", va="center", color="black")


train_file = "../data/data-split/z-score/training.csv"
test_file = "../data/data-split/z-score/testing.csv"

train_df = load_data(train_file)
test_df = load_data(test_file)

empatica_features = ["empatica_bvp", "empatica_eda", "empatica_temp"]
samsung_features = ["samsung_bvp"]

x_train_empatica, y_train_empatica = assemble_features(train_df, empatica_features, "CL")
x_train_samsung, y_train_samsung = assemble_features(train_df, samsung_features, "CL")

x_test_empatica, y_test_empatica = assemble_features(test_df, empatica_features, "CL")
x_test_samsung, y_test_samsung = assemble_features(test_df, samsung_features, "CL")

scaler_empatica = StandardScaler()
scaler_samsung = StandardScaler()

x_train_empatica = scaler_empatica.fit_transform(x_train_empatica)
x_test_empatica = scaler_empatica.transform(x_test_empatica)

x_train_samsung = scaler_samsung.fit_transform(x_train_samsung)
x_test_samsung = scaler_samsung.transform(x_test_samsung)

model_empatica = train_random_forest(x_train_empatica, y_train_empatica, max_depth=20)
model_samsung = train_random_forest(x_train_samsung, y_train_samsung, max_depth=20)

y_pred_empatica = make_predictions(model_empatica, x_test_empatica)
y_pred_samsung = make_predictions(model_samsung, x_test_samsung)

conf_matrix_empatica = confusion_matrix(y_test_empatica, y_pred_empatica)
conf_matrix_samsung = confusion_matrix(y_test_samsung, y_pred_samsung)

print("\nEmpatica Model Classification Report:\n", classification_report(y_test_empatica, y_pred_empatica))
print("\nSamsung Model Classification Report:\n", classification_report(y_test_samsung, y_pred_samsung))

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
plot_confusion_matrix(conf_matrix_empatica, "Empatica Confusion Matrix", axes[0])
plot_confusion_matrix(conf_matrix_samsung, "Samsung Confusion Matrix", axes[1])

plt.tight_layout()
plt.show()

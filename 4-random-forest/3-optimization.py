from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from necessary_functions import *

params_empatica = {
    "max_depth": [10, 20, 30],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2"],
    "n_estimators": [100, 200, 300],
}

params_samsung = {
    "max_depth": [10, 20, 30],
    "min_samples_leaf": [1, 2, 4],
    "n_estimators": [100, 200, 300],
}

models = {
    "Empatica": RandomForestClassifier(random_state=42),
    "Samsung": RandomForestClassifier(random_state=42),
}


grid_search_results = {}
for model_name, model in models.items():
    params = params_empatica if model_name == "Empatica" else params_samsung
    x_train = x_train_empatica if model_name == "Empatica" else x_train_samsung
    y_train = y_train_empatica if model_name == "Empatica" else y_train_samsung

    grid_search = GridSearchCV(estimator=model, param_grid=params, cv=5)
    grid_search.fit(x_train, y_train)

    grid_search_results[model_name] = {
        "best_params": grid_search.best_params_,
        "model": RandomForestClassifier(random_state=42, **grid_search.best_params_).fit(x_train, y_train)
    }

for model_name, result in grid_search_results.items():
    x_test = x_test_empatica if model_name == "Empatica" else x_test_samsung
    y_test = y_test_empatica if model_name == "Empatica" else y_test_samsung

    y_pred = result["model"].predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Best params for {model_name}:", result["best_params"])
    print(f"Accuracy for {model_name} model:", accuracy, "\n")

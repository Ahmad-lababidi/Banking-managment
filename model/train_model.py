import json

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from .config import METRICS_FILE, MODEL_DIR, MODEL_FILE, RANDOM_STATE, TARGET, TEST_FILE, TRAIN_FILE
from .prepare_data import prepare_data
from .utils import add_features, get_logger, make_preprocessor


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def scores(model, x, y):
    predicted = model.predict(x)
    labels = sorted(pd.Series(y).unique())
    matrix = confusion_matrix(y, predicted, labels=labels)
    report = classification_report(y, predicted, labels=labels, output_dict=True, zero_division=0)
    per_class_errors = {}
    for index, label in enumerate(labels):
        row = matrix[index]
        errors_by_prediction = {str(predicted_label): int(row[predicted_index])
                                for predicted_index, predicted_label in enumerate(labels)
                                if predicted_index != index and row[predicted_index]}
        most_common = max(errors_by_prediction, key=errors_by_prediction.get) if errors_by_prediction else None
        per_class_errors[str(label)] = {
            "support": int(row.sum()),
            "correct": int(matrix[index, index]),
            "misclassified": int(row.sum() - matrix[index, index]),
            "error_rate": round(float((row.sum() - matrix[index, index]) / row.sum()), 4) if row.sum() else 0.0,
            "most_common_misclassification": most_common,
            "misclassified_as": errors_by_prediction,
        }
    return {"accuracy": round(float(accuracy_score(y, predicted)), 4),
            "weighted_f1": round(float(f1_score(y, predicted, average="weighted")), 4),
            "report": report,
            "class_labels": [str(label) for label in labels],
            "confusion_matrix": matrix.tolist(),
            "per_class_errors": per_class_errors}


def main():
    log = get_logger("training")
    log.info("Loading data")
    prepare_data()
    train, test = pd.read_csv(TRAIN_FILE), pd.read_csv(TEST_FILE)
    X, y = add_features(train), train[TARGET]
    X_test, y_test = add_features(test), test[TARGET]
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)

    models = {
        "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "decision_tree": DecisionTreeClassifier(max_depth=16, class_weight="balanced", random_state=RANDOM_STATE),
        "random_forest": GridSearchCV(
            Pipeline([("preprocess", make_preprocessor(X)),
                      ("classifier", RandomForestClassifier(class_weight="balanced", random_state=RANDOM_STATE,
                                                             n_estimators=60, n_jobs=-1))]),
            {"classifier__max_depth": [12, None]},
            cv=StratifiedKFold(3, shuffle=True, random_state=RANDOM_STATE), scoring="f1_weighted", n_jobs=-1),
    }
    fitted, results = {}, {"validation_size": len(X_val), "test_size": len(X_test), "models": {}}
    for name, classifier in models.items():
        log.info("Training %s", name)
        if name == "random_forest":
            classifier.fit(X_train, y_train)
            model = classifier.best_estimator_
            results["models"][name] = {"cross_validation_weighted_f1": round(float(classifier.best_score_), 4),
                                        "best_parameters": classifier.best_params_}
        else:
            model = Pipeline([("preprocess", make_preprocessor(X_train)), ("classifier", classifier)])
            model.fit(X_train, y_train)
            results["models"][name] = {}
        results["models"][name]["validation"] = scores(model, X_val, y_val)
        results["models"][name]["test"] = scores(model, X_test, y_test)
        fitted[name] = model
        log.info("%s test weighted F1: %.4f", name, results["models"][name]["test"]["weighted_f1"])

    best = max(fitted, key=lambda n: results["models"][n]["validation"]["weighted_f1"])
    results["selected_model"] = best
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"models": fitted, "selected_model": best, "feature_columns": X.columns.tolist()}, MODEL_FILE)
    METRICS_FILE.write_text(json.dumps(json_safe(results), indent=2), encoding="utf-8")
    log.info("Selected %s using validation weighted F1", best)
    print(f"Selected model: {best}")
    print(f"Metrics saved to {METRICS_FILE}")


if __name__ == "__main__":
    main()

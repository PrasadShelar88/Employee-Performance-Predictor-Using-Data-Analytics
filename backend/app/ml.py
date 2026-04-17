from __future__ import annotations

import json

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler

from .config import (
    CONFUSION_MATRIX_PATH,
    DATASET_PATH,
    FEATURE_IMPORTANCE_PATH,
    METRICS_PATH,
    MODEL_PATH,
    MODELS_DIR,
    NUMERIC_CORRELATION_PATH,
    OUTPUTS_DIR,
    PERFORMANCE_DISTRIBUTION_PATH,
    PREDICTIONS_PATH,
)
from .data_generator import save_dataset

TARGET_COL = "perf_band_next"
DROP_COLS = ["employee_id"]


def load_or_create_dataset() -> pd.DataFrame:
    if DATASET_PATH.exists():
        return pd.read_csv(DATASET_PATH)
    return save_dataset()


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    num_cols = X.select_dtypes(include=["number", "int64", "float64"]).columns.tolist()

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", RobustScaler()),
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols),
    ])


def build_models(preprocessor: ColumnTransformer) -> tuple[Pipeline, Pipeline, dict]:
    logistic = Pipeline([
        ("pre", preprocessor),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])
    rf = Pipeline([
        ("pre", preprocessor),
        ("clf", RandomForestClassifier(class_weight="balanced", random_state=13)),
    ])
    param_grid = {
        "clf__n_estimators": [300, 500],
        "clf__max_depth": [10, None],
        "clf__min_samples_leaf": [1, 3],
    }
    return logistic, rf, param_grid


def save_output_artifacts(df: pd.DataFrame, X_test: pd.DataFrame, y_test: pd.Series, y_pred, y_proba, best_model: Pipeline) -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    prediction_df = X_test.copy()
    prediction_df["actual_band"] = y_test.values
    prediction_df["predicted_band"] = y_pred
    class_to_idx = {label: idx for idx, label in enumerate(best_model.classes_)}
    for label in ["High", "Medium", "Low"]:
        prediction_df[f"prob_{label.lower()}"] = y_proba[:, class_to_idx[label]] if label in class_to_idx else 0.0
    prediction_df.head(50).to_csv(PREDICTIONS_PATH, index=False)

    perm = permutation_importance(best_model, X_test, y_test, scoring="f1_macro", n_repeats=3, random_state=13, n_jobs=1)
    feature_importance = pd.DataFrame({"feature": X_test.columns, "importance": perm.importances_mean}).sort_values("importance", ascending=False)
    feature_importance.to_csv(FEATURE_IMPORTANCE_PATH, index=False)

    cm = pd.DataFrame(
        confusion_matrix(y_test, y_pred, labels=["Low", "Medium", "High"]),
        index=["Low", "Medium", "High"],
        columns=["Low", "Medium", "High"],
    )
    cm.to_csv(CONFUSION_MATRIX_PATH.with_suffix(".csv"))

    perf_dist = df[TARGET_COL].value_counts().rename_axis("band").reset_index(name="count")
    perf_dist.to_csv(PERFORMANCE_DISTRIBUTION_PATH.with_suffix(".csv"), index=False)

    corr = df.select_dtypes(include=["number"]).corr(numeric_only=True)
    corr.to_csv(NUMERIC_CORRELATION_PATH.with_suffix(".csv"))


def train_and_save() -> dict:
    df = load_or_create_dataset()
    X = df.drop(columns=[TARGET_COL] + DROP_COLS)
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=13
    )

    preprocessor = build_preprocessor(X_train)
    logistic, rf, param_grid = build_models(preprocessor)

    logistic.fit(X_train, y_train)
    logistic_pred = logistic.predict(X_test)
    logistic_macro_f1 = f1_score(y_test, logistic_pred, average="macro")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=13)
    gs = GridSearchCV(rf, param_grid=param_grid, scoring="f1_macro", cv=cv, n_jobs=1, verbose=0)
    gs.fit(X_train, y_train)

    best_model = gs.best_estimator_
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)
    save_output_artifacts(df, X_test, y_test, y_pred, y_proba, best_model)

    metrics = {
        "dataset_path": str(DATASET_PATH),
        "model_path": str(MODEL_PATH),
        "accuracy": round(float(accuracy), 4),
        "macro_f1": round(float(macro_f1), 4),
        "baseline_logistic_macro_f1": round(float(logistic_macro_f1), 4),
        "best_params": gs.best_params_,
        "classification_report": report,
    }
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return metrics


def load_model() -> Pipeline:
    if not MODEL_PATH.exists():
        train_and_save()
    return joblib.load(MODEL_PATH)

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


DATA_PATH = Path("final_dataset_6000.csv")
ARTIFACT_DIR = Path("artifacts")
MODEL_PATH = ARTIFACT_DIR / "models.joblib"
META_PATH = ARTIFACT_DIR / "metadata.json"

DROP_COLS = [
    "release",
    "date_parsed",
    "full_name",
    "owner",
    "repo_name",
    "updated_at",
    "pushed_at",
    "id",
    "id_repo",
]

TARGET_CLASS = "defect_prone"
TARGET_REG = "effort_next_lines_added"
UNUSED_TARGET = "effort_next_n_auth"
RANDOM_STATE = 42

# Best params from notebook GridSearchCV results.
QDA_REG_PARAM = 0.001
KRR_ALPHA = 0.1
KRR_GAMMA = 0.0001


@dataclass
class ModelArtifacts:
    qda_model: Pipeline
    krr_model: Pipeline
    feature_cols: list[str]
    numeric_cols: list[str]
    categorical_cols: list[str]
    numeric_defaults: dict[str, float]
    categorical_defaults: dict[str, str]
    categorical_options: dict[str, list[str]]
    qda_importance: pd.DataFrame
    krr_importance: pd.DataFrame


def _prepare_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    y_class = df[TARGET_CLASS]
    y_reg = df[TARGET_REG]
    x = df.drop(columns=[TARGET_CLASS, TARGET_REG, UNUSED_TARGET], errors="ignore")
    x = x.drop(columns=[c for c in DROP_COLS if c in x.columns], errors="ignore")
    return x, y_class, y_reg


def _build_preprocessor(x_train: pd.DataFrame) -> tuple[ColumnTransformer, list[str], list[str]]:
    numeric_cols = x_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = x_train.select_dtypes(include=["object", "string"]).columns.tolist()

    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        [
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ]
    )
    return preprocessor, numeric_cols, categorical_cols


def _train_from_scratch() -> ModelArtifacts:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    x, y_class, y_reg = _prepare_xy(df)

    x_train, x_test, y_class_train, y_class_test, y_reg_train, y_reg_test = train_test_split(
        x,
        y_class,
        y_reg,
        test_size=0.2,
        stratify=y_class,
        random_state=RANDOM_STATE,
    )

    preprocessor, numeric_cols, categorical_cols = _build_preprocessor(x_train)

    qda_model = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("model", QuadraticDiscriminantAnalysis(reg_param=QDA_REG_PARAM)),
        ]
    )
    qda_model.fit(x_train, y_class_train)

    krr_model = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("model", KernelRidge(kernel="rbf", alpha=KRR_ALPHA, gamma=KRR_GAMMA)),
        ]
    )
    krr_model.fit(x_train, y_reg_train)

    qda_perm = permutation_importance(
        qda_model,
        x_test,
        y_class_test,
        n_repeats=10,
        random_state=RANDOM_STATE,
        scoring="f1",
        n_jobs=-1,
    )
    qda_importance = (
        pd.DataFrame(
            {
                "feature": x_test.columns,
                "importance_mean": qda_perm.importances_mean,
                "importance_std": qda_perm.importances_std,
            }
        )
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )

    krr_perm = permutation_importance(
        krr_model,
        x_test,
        y_reg_test,
        n_repeats=10,
        random_state=RANDOM_STATE,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
    )
    krr_importance = (
        pd.DataFrame(
            {
                "feature": x_test.columns,
                "importance_mean": krr_perm.importances_mean,
                "importance_std": krr_perm.importances_std,
            }
        )
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )

    numeric_defaults = x_train[numeric_cols].median().to_dict()
    categorical_defaults = {}
    categorical_options = {}
    for col in categorical_cols:
        mode_values = x_train[col].mode(dropna=True)
        categorical_defaults[col] = str(mode_values.iloc[0]) if not mode_values.empty else ""
        top_vals = (
            x_train[col]
            .dropna()
            .astype(str)
            .value_counts()
            .head(20)
            .index.tolist()
        )
        categorical_options[col] = top_vals

    return ModelArtifacts(
        qda_model=qda_model,
        krr_model=krr_model,
        feature_cols=x_train.columns.tolist(),
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        numeric_defaults={k: float(v) for k, v in numeric_defaults.items()},
        categorical_defaults=categorical_defaults,
        categorical_options=categorical_options,
        qda_importance=qda_importance,
        krr_importance=krr_importance,
    )


def _save_artifacts(artifacts: ModelArtifacts) -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "qda_model": artifacts.qda_model,
            "krr_model": artifacts.krr_model,
            "qda_importance": artifacts.qda_importance,
            "krr_importance": artifacts.krr_importance,
        },
        MODEL_PATH,
    )

    metadata = {
        "feature_cols": artifacts.feature_cols,
        "numeric_cols": artifacts.numeric_cols,
        "categorical_cols": artifacts.categorical_cols,
        "numeric_defaults": artifacts.numeric_defaults,
        "categorical_defaults": artifacts.categorical_defaults,
        "categorical_options": artifacts.categorical_options,
    }
    META_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def _load_artifacts() -> ModelArtifacts:
    models: dict[str, Any] = joblib.load(MODEL_PATH)
    metadata = json.loads(META_PATH.read_text(encoding="utf-8"))
    return ModelArtifacts(
        qda_model=models["qda_model"],
        krr_model=models["krr_model"],
        feature_cols=metadata["feature_cols"],
        numeric_cols=metadata["numeric_cols"],
        categorical_cols=metadata["categorical_cols"],
        numeric_defaults=metadata["numeric_defaults"],
        categorical_defaults=metadata["categorical_defaults"],
        categorical_options=metadata["categorical_options"],
        qda_importance=models["qda_importance"],
        krr_importance=models["krr_importance"],
    )


def load_or_train_artifacts() -> ModelArtifacts:
    if MODEL_PATH.exists() and META_PATH.exists():
        try:
            return _load_artifacts()
        except Exception:
            # Recover from incompatible serialized artifacts (e.g., pandas/sklearn version mismatch).
            try:
                MODEL_PATH.unlink(missing_ok=True)
                META_PATH.unlink(missing_ok=True)
            except OSError:
                pass

    artifacts = _train_from_scratch()
    _save_artifacts(artifacts)
    return artifacts


def build_input_frame(raw_input: dict[str, Any], artifacts: ModelArtifacts) -> pd.DataFrame:
    row = {}
    for col in artifacts.feature_cols:
        if col in artifacts.numeric_cols:
            val = raw_input.get(col, artifacts.numeric_defaults.get(col, 0.0))
            row[col] = float(val)
        elif col in artifacts.categorical_cols:
            val = raw_input.get(col, artifacts.categorical_defaults.get(col, ""))
            row[col] = str(val)
        else:
            row[col] = raw_input.get(col)
    return pd.DataFrame([row], columns=artifacts.feature_cols)


def predict_single(raw_input: dict[str, Any], artifacts: ModelArtifacts) -> dict[str, Any]:
    input_df = build_input_frame(raw_input, artifacts)

    class_pred = int(artifacts.qda_model.predict(input_df)[0])
    class_prob = float(artifacts.qda_model.predict_proba(input_df)[0][1])
    reg_pred = float(artifacts.krr_model.predict(input_df)[0])

    return {
        "defect_prone_prediction": class_pred,
        "defect_probability": class_prob,
        "effort_next_lines_added_prediction": reg_pred,
    }


def predict_batch(input_df: pd.DataFrame, artifacts: ModelArtifacts) -> pd.DataFrame:
    df = input_df.copy()

    for col in artifacts.feature_cols:
        if col not in df.columns:
            if col in artifacts.numeric_cols:
                df[col] = artifacts.numeric_defaults.get(col, 0.0)
            elif col in artifacts.categorical_cols:
                df[col] = artifacts.categorical_defaults.get(col, "")
            else:
                df[col] = np.nan

    df = df[artifacts.feature_cols].copy()

    for col in artifacts.numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(artifacts.numeric_defaults.get(col, 0.0))
    for col in artifacts.categorical_cols:
        df[col] = df[col].astype(str)

    class_preds = artifacts.qda_model.predict(df)
    class_probs = artifacts.qda_model.predict_proba(df)[:, 1]
    reg_preds = artifacts.krr_model.predict(df)

    out = input_df.copy()
    out["defect_prone_prediction"] = class_preds.astype(int)
    out["defect_probability"] = class_probs.astype(float)
    out["effort_next_lines_added_prediction"] = reg_preds.astype(float)
    return out

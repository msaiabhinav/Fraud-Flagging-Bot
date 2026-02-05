# pipeline.py
# Train + inference pipeline for Fraud Triage Bot
# Artifacts saved/loaded from artifacts/:
#   - model.joblib
#   - preprocessor.joblib
#   - threshold.json

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, List

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split


# ----------------------------
# Paths
# ----------------------------
BASE_DIR = os.path.dirname(__file__)
ARTIFACT_DIR = os.path.join(BASE_DIR, "artifacts")

MODEL_PATH = os.path.join(ARTIFACT_DIR, "model.joblib")
PREPROCESSOR_PATH = os.path.join(ARTIFACT_DIR, "preprocessor.joblib")
THRESHOLD_PATH = os.path.join(ARTIFACT_DIR, "threshold.json")

DATASET_PATH = os.path.join(BASE_DIR, "insurance_claims.csv")


# ----------------------------
# Results dataclass
# ----------------------------
@dataclass
class PredictionResult:
    label: int
    proba: float
    threshold: float
    decision: str
    model_name: str


# ----------------------------
# Data loader
# ----------------------------
def load_dataset(csv_path: Optional[str] = None) -> pd.DataFrame:
    path = csv_path or DATASET_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(f"Could not find dataset at: {path}")
    return pd.read_csv(path)


# ----------------------------
# Feature engineering (your logic, stabilized)
# ----------------------------
def build_features(raw_df: pd.DataFrame) -> pd.DataFrame:
    data = raw_df.copy()

    # Fill missing values for incident-related categorical columns
    if "property_damage" in data.columns:
        data["property_damage"] = data["property_damage"].fillna("Unknown")
    if "collision_type" in data.columns:
        data["collision_type"] = data["collision_type"].fillna("Unknown")

    if "police_report_available" in data.columns:
        data["police_report_available"] = data["police_report_available"].fillna("Unknown")
    if "authorities_contacted" in data.columns:
        data["authorities_contacted"] = data["authorities_contacted"].fillna("Unknown")

    # customer_maturity
    if "age" in data.columns and "months_as_customer" in data.columns:
        data["customer_maturity"] = (data["age"] - 18) / (data["months_as_customer"] / 12)
        data["customer_maturity"] = data["customer_maturity"].replace([np.inf, -np.inf], np.nan)

    # capital_gain_loss
    if "capital-gains" in data.columns and "capital-loss" in data.columns:
        data["capital_gain_loss"] = data["capital-gains"] + data["capital-loss"]

    # Ratios using total_claim_amount
    if "total_claim_amount" in data.columns:
        denom = data["total_claim_amount"].replace(0, np.nan)
        for raw_col, new_col in [
            ("injury_claim", "injury_claim_ratio"),
            ("property_claim", "property_claim_ratio"),
            ("vehicle_claim", "vehicle_claim_ratio"),
        ]:
            if raw_col in data.columns:
                data[new_col] = (data[raw_col] / denom).replace([np.inf, -np.inf], 0).fillna(0)
            else:
                data[new_col] = 0.0

    # Parse dates
    if "incident_date" in data.columns:
        data["incident_date"] = pd.to_datetime(data["incident_date"], errors="coerce")
        data["incident_week_no"] = data["incident_date"].dt.isocalendar().week.astype("Int64")
    else:
        data["incident_week_no"] = pd.Series([pd.NA] * len(data), dtype="Int64")

    if "policy_bind_date" in data.columns:
        data["policy_bind_date"] = pd.to_datetime(data["policy_bind_date"], errors="coerce")

    # responsible_days
    if "incident_date" in data.columns and "policy_bind_date" in data.columns:
        data["responsible_days"] = (data["incident_date"] - data["policy_bind_date"]).dt.days
    else:
        data["responsible_days"] = np.nan

    # incident_hour_bucket
    if "incident_hour_of_the_day" in data.columns:
        data["incident_hour_bucket"] = pd.cut(
            data["incident_hour_of_the_day"],
            bins=[0, 6, 12, 18, 24],
            labels=["morning", "afternoon", "evening", "night"],
            right=False,
            include_lowest=True,
        ).astype("object").fillna("unknown")
    else:
        data["incident_hour_bucket"] = "unknown"

    # vehicle_age
    if "incident_date" in data.columns and "auto_year" in data.columns:
        data["vehicle_age"] = (data["incident_date"].dt.year - data["auto_year"]).clip(lower=0)
    else:
        data["vehicle_age"] = np.nan

    # Drop raw + leakage + IDs + fields you decided to exclude
    drop_cols = [
        "incident_date",
        "policy_bind_date",
        "incident_hour_of_the_day",
        "injury_claim",
        "property_claim",
        "vehicle_claim",
        "months_as_customer",
        "age",
        "auto_make",
        "auto_model",
        "auto_year",
        "incident_city",
        "incident_state",
        "insured_zip",
        "policy_state",
        "policy_number",
        "capital-gains",
        "capital-loss",
    ]
    df_mod = data.drop(columns=[c for c in drop_cols if c in data.columns]).copy()

    # Target mapping (robust)
    if "fraud_reported" in df_mod.columns:
        df_mod["fraud_reported"] = (
            df_mod["fraud_reported"]
            .astype(str)
            .str.strip()
            .str.upper()
            .replace({"Y": 1, "N": 0})
        )

    return df_mod


# ----------------------------
# Preprocessor
# ----------------------------
def make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    cat_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    num_cols = [c for c in X.columns if c not in cat_cols]

    num_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", MinMaxScaler()),
        ]
    )

    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


# ----------------------------
# Threshold search (F1 on val)
# ----------------------------
def find_best_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    thresholds = np.linspace(0.05, 0.95, 91)
    best_t, best_f1 = 0.5, -1.0
    for t in thresholds:
        f1 = f1_score(y_true, (y_prob >= t).astype(int))
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t


# ----------------------------
# Train + save artifacts
# ----------------------------
def train_and_save(
    csv_path: Optional[str] = None,
    target_col: str = "fraud_reported",
    random_state: int = 42,
) -> Dict[str, Any]:
    df = load_dataset(csv_path)
    df_mod = build_features(df)

    if target_col not in df_mod.columns:
        raise ValueError(f"Target column '{target_col}' missing after feature engineering.")

    df_mod = df_mod.dropna(subset=[target_col]).copy()

    # Force numeric target safely
    df_mod[target_col] = pd.to_numeric(df_mod[target_col], errors="coerce")
    df_mod = df_mod.dropna(subset=[target_col]).copy()

    y = df_mod[target_col].astype(int)
    X = df_mod.drop(columns=[target_col])

    # 65/10/25 split
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.25, random_state=random_state, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=(10 / 75), random_state=random_state, stratify=y_temp
    )

    preprocessor = make_preprocessor(X_train)
    X_train_t = preprocessor.fit_transform(X_train)
    X_val_t = preprocessor.transform(X_val)
    X_test_t = preprocessor.transform(X_test)

    model = LogisticRegression(max_iter=4000, class_weight="balanced", solver="lbfgs")
    model.fit(X_train_t, y_train)

    val_prob = model.predict_proba(X_val_t)[:, 1]
    test_prob = model.predict_proba(X_test_t)[:, 1]

    best_threshold = find_best_threshold(y_val.to_numpy(), val_prob)

    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(preprocessor, PREPROCESSOR_PATH)
    with open(THRESHOLD_PATH, "w", encoding="utf-8") as f:
        json.dump({"threshold": best_threshold}, f, indent=2)

    return {
        "val_auc": float(roc_auc_score(y_val, val_prob)),
        "test_auc": float(roc_auc_score(y_test, test_prob)),
        "best_threshold": float(best_threshold),
        "val_f1": float(f1_score(y_val, (val_prob >= best_threshold).astype(int))),
        "test_f1": float(f1_score(y_test, (test_prob >= best_threshold).astype(int))),
        "train_shape": tuple(X_train.shape),
        "val_shape": tuple(X_val.shape),
        "test_shape": tuple(X_test.shape),
    }


# ----------------------------
# Load artifacts
# ----------------------------
def _load_threshold(path: str) -> float:
    if not os.path.exists(path):
        return 0.5
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if isinstance(obj, (int, float)):
        return float(obj)
    if isinstance(obj, dict) and "threshold" in obj:
        return float(obj["threshold"])
    return 0.5


def load_artifacts():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Missing model artifact: {MODEL_PATH}")
    if not os.path.exists(PREPROCESSOR_PATH):
        raise FileNotFoundError(f"Missing preprocessor artifact: {PREPROCESSOR_PATH}")

    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    threshold = _load_threshold(THRESHOLD_PATH)
    return model, preprocessor, threshold


# ----------------------------
# Inference helpers
# ----------------------------
def _expected_feature_names(preprocessor) -> Optional[list]:
    names = getattr(preprocessor, "feature_names_in_", None)
    return list(names) if names is not None else None


def _coerce_input_to_df(user_input: Dict[str, Any], expected_cols: Optional[list]) -> pd.DataFrame:
    if not isinstance(user_input, dict) or len(user_input) == 0:
        raise ValueError("user_input must be a non-empty dict of feature_name -> value")

    df = pd.DataFrame([user_input])

    if expected_cols is None:
        return df

    for c in expected_cols:
        if c not in df.columns:
            df[c] = np.nan

    return df[expected_cols]


# ----------------------------
# Predict
# ----------------------------
def predict(
    user_input: Dict[str, Any],
    model=None,
    preprocessor=None,
    threshold: Optional[float] = None,
) -> PredictionResult:
    if model is None or preprocessor is None or threshold is None:
        model, preprocessor, loaded_threshold = load_artifacts()
        if threshold is None:
            threshold = loaded_threshold

    # Ensure inference uses engineered features too
    df_raw = pd.DataFrame([user_input])
    df_eng = build_features(df_raw)
    user_input_eng = df_eng.iloc[0].to_dict()

    expected_cols = _expected_feature_names(preprocessor)
    X_df = _coerce_input_to_df(user_input_eng, expected_cols)

    X_t = preprocessor.transform(X_df)

    if hasattr(model, "predict_proba"):
        proba = float(model.predict_proba(X_t)[0, 1])
    else:
        pred = int(model.predict(X_t)[0])
        proba = float(pred)

    thr = float(threshold)
    label = int(proba >= thr)
    decision = "FLAG_FRAUD" if label == 1 else "NO_FRAUD"

    return PredictionResult(
        label=label,
        proba=proba,
        threshold=thr,
        decision=decision,
        model_name=type(model).__name__,
    )


# ----------------------------
# Explain prediction (LogisticRegression contributions)
# ----------------------------
def explain_prediction(
    user_input: Dict[str, Any],
    top_k: int = 8,
    model=None,
    preprocessor=None,
    threshold: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Returns probability + logit + top feature contributions for LogisticRegression.

    Contributions are computed in the transformed feature space:
      contribution_i = x_i * coef_i
      logit = intercept + sum(contribution_i)
      probability = sigmoid(logit)
    """
    if model is None or preprocessor is None or threshold is None:
        model, preprocessor, loaded_threshold = load_artifacts()
        if threshold is None:
            threshold = loaded_threshold

    # engineer features
    df_raw = pd.DataFrame([user_input])
    df_eng = build_features(df_raw)
    user_input_eng = df_eng.iloc[0].to_dict()

    expected_cols = _expected_feature_names(preprocessor)
    X_df = _coerce_input_to_df(user_input_eng, expected_cols)
    X_t = preprocessor.transform(X_df)

    # feature names after preprocessing (includes OHE categories)
    try:
        feat_names = preprocessor.get_feature_names_out()
        feat_names = [str(x) for x in feat_names]
    except Exception:
        feat_names = [f"f_{i}" for i in range(X_t.shape[1])]

    # if not logistic regression, just return prediction
    if not hasattr(model, "coef_") or not hasattr(model, "intercept_"):
        pr = predict(user_input, model=model, preprocessor=preprocessor, threshold=threshold)
        return {
            "probability": float(pr.proba),
            "threshold": float(pr.threshold),
            "decision": pr.decision,
            "note": "Model is not LogisticRegression; coefficient-based explanation unavailable.",
            "engineered_input": user_input_eng,
        }

    coef = np.asarray(model.coef_).ravel()
    intercept = float(np.asarray(model.intercept_).ravel()[0])

    # handle sparse matrix safely
    if hasattr(X_t, "toarray"):
        x = X_t.toarray().ravel()
    else:
        x = np.asarray(X_t).ravel()

    contrib = x * coef
    logit = intercept + float(contrib.sum())
    proba = float(1.0 / (1.0 + np.exp(-logit)))

    # top positive and negative contributions
    idx_sorted = np.argsort(contrib)
    neg_idx = idx_sorted[:top_k]        # most negative
    pos_idx = idx_sorted[::-1][:top_k]  # most positive

    positives = [
        {"feature": feat_names[i], "contribution": float(contrib[i]), "value": float(x[i])}
        for i in pos_idx if contrib[i] > 0
    ]
    negatives = [
        {"feature": feat_names[i], "contribution": float(contrib[i]), "value": float(x[i])}
        for i in neg_idx if contrib[i] < 0
    ]

    thr = float(threshold)
    decision = "FLAG_FRAUD" if proba >= thr else "NO_FRAUD"

    return {
        "probability": proba,
        "logit": float(logit),
        "threshold": thr,
        "decision": decision,
        "top_positive": positives,
        "top_negative": negatives,
        "engineered_input": user_input_eng,
    }


# ----------------------------
# Utility: sample row -> dict
# ----------------------------
def sample_input_from_row(
    df: pd.DataFrame, row_idx: int = 0, target_col: str = "fraud_reported"
) -> Dict[str, Any]:
    row = df.iloc[row_idx].to_dict()
    if target_col in row:
        row.pop(target_col)
    return row

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from bire.evaluation.metrics import evaluate_multiple_splits


def patient_level_split(df, random_state: int = 42):
    """
    Patient-level train/val/test split to avoid leakage across patients.
    """
    patients = df["patient_id"].dropna().unique()

    train_p, temp_p = train_test_split(
        patients,
        test_size=0.4,
        random_state=random_state,
    )
    val_p, test_p = train_test_split(
        temp_p,
        test_size=0.5,
        random_state=random_state,
    )

    train_df = df[df["patient_id"].isin(train_p)].copy()
    val_df = df[df["patient_id"].isin(val_p)].copy()
    test_df = df[df["patient_id"].isin(test_p)].copy()

    return train_df, val_df, test_df


def build_logistic_model(random_state: int = 42):
    return LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=random_state,
    )


def apply_alert_logic(df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    """
    Add risk score, risk band, and alert columns from predicted probabilities.
    Expects `pred_proba` to already exist.
    """
    out = df.copy()

    out["risk_score"] = out["pred_proba"]
    out["risk_band"] = pd.cut(
        out["risk_score"],
        bins=[-float("inf"), 0.25, 0.5, float("inf")],
        labels=["low", "moderate", "high"],
    )
    out["alert"] = out["risk_score"] >= threshold

    return out


def run_bire_modeling(df, feature_cols, threshold=0.5, random_state: int = 42):
    print("Inside modeling function")
    print("target in input df:", "target" in df.columns)

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing features: {missing}")

    optional_keep_cols = [c for c in ["timestamp", "event_now"] if c in df.columns]
    required_cols = ["patient_id", "target"] + optional_keep_cols + feature_cols

    work_df = df[required_cols].copy()
    work_df = work_df.dropna(subset=["target"])

    train_df, val_df, test_df = patient_level_split(
        work_df,
        random_state=random_state,
    )

    print("target in train_df:", "target" in train_df.columns)
    print("target in val_df:", "target" in val_df.columns)
    print("target in test_df:", "target" in test_df.columns)

    X_train = train_df[feature_cols]
    y_train = train_df["target"]

    X_val = val_df[feature_cols]
    y_val = val_df["target"]

    X_test = test_df[feature_cols]
    y_test = test_df["target"]

    imputer = SimpleImputer(strategy="median")

    X_train_imp = pd.DataFrame(
        imputer.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index,
    )
    X_val_imp = pd.DataFrame(
        imputer.transform(X_val),
        columns=X_val.columns,
        index=X_val.index,
    )
    X_test_imp = pd.DataFrame(
        imputer.transform(X_test),
        columns=X_test.columns,
        index=X_test.index,
    )

    model = build_logistic_model(random_state=random_state)
    model.fit(X_train_imp, y_train)

    results_df, probas = evaluate_multiple_splits(
        model,
        {
            "val": (X_val_imp, y_val),
            "test": (X_test_imp, y_test),
        },
    )

    test_scored_df = test_df.copy()
    test_scored_df["pred_proba"] = model.predict_proba(X_test_imp)[:, 1]
    test_scored_df = apply_alert_logic(test_scored_df, threshold=threshold)

    return model, test_scored_df, results_df


def run_xgb_modeling(
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    scale_pos_weight,
    random_state: int = 42,
):
    """
    Train and evaluate XGBoost on train/val/test splits.
    Returns fitted model, imputer, split metrics, split probabilities,
    and imputed feature matrices.
    """
    imputer = SimpleImputer(strategy="median")

    X_train_imp = pd.DataFrame(
        imputer.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index,
    )

    X_val_imp = pd.DataFrame(
        imputer.transform(X_val),
        columns=X_val.columns,
        index=X_val.index,
    )

    X_test_imp = pd.DataFrame(
        imputer.transform(X_test),
        columns=X_test.columns,
        index=X_test.index,
    )

    xgb_model = XGBClassifier(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=random_state,
        scale_pos_weight=scale_pos_weight,
    )

    xgb_model.fit(X_train_imp, y_train)

    results_df, probas = evaluate_multiple_splits(
        xgb_model,
        {
            "val": (X_val_imp, y_val),
            "test": (X_test_imp, y_test),
        },
    )

    artifacts = {
        "X_train_imp": X_train_imp,
        "X_val_imp": X_val_imp,
        "X_test_imp": X_test_imp,
    }

    return xgb_model, imputer, results_df, probas, artifacts


def format_model_results(results_df: pd.DataFrame, model_name: str) -> pd.DataFrame:
    formatted = results_df.copy()
    formatted["model"] = model_name
    return formatted


def compare_split_metrics(results_by_model: dict) -> pd.DataFrame:
    frames = []
    for model_name, df in results_by_model.items():
        this_df = df.copy()
        this_df["model"] = model_name
        frames.append(this_df)

    return pd.concat(frames, ignore_index=True)[
        ["model", "split", "n_rows", "positive_rate", "roc_auc", "pr_auc"]
    ]

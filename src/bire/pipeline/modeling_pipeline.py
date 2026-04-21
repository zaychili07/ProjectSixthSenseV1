import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from bire.evaluation.metrics import evaluate_multiple_splits


def patient_level_split(df, random_state: int = 42):
    """
    Patient-level train/val/test split with stratification on patient-level
    target presence to avoid leakage and reduce risk of single-class splits.
    """
    required_cols = {"patient_id", "target"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"patient_level_split missing required columns: {sorted(missing)}")

    patient_summary = (
        df.groupby("patient_id", as_index=False)["target"]
        .max()
        .rename(columns={"target": "any_target"})
    )

    if patient_summary.empty:
        raise ValueError("No patients available for splitting.")

    if patient_summary["any_target"].nunique() < 2:
        raise ValueError(
            "Patient-level target summary has only one class. "
            f"Counts: {patient_summary['any_target'].value_counts(dropna=False).to_dict()}"
        )

    patient_class_counts = patient_summary["any_target"].value_counts()
    min_class_count = patient_class_counts.min()

    if len(patient_summary) < 3:
        raise ValueError(
            "Need at least 3 patients to create train/val/test splits."
        )

    train_stratify = patient_summary["any_target"] if min_class_count >= 2 else None

    train_p, temp_p = train_test_split(
        patient_summary["patient_id"],
        test_size=0.4,
        random_state=random_state,
        stratify=train_stratify,
    )

    temp_summary = patient_summary[patient_summary["patient_id"].isin(temp_p)].copy()

    if temp_summary["any_target"].nunique() < 2:
        raise ValueError(
            "Temp patient split has only one class. "
            "Increase positive patients or adjust split strategy."
        )

    temp_class_counts = temp_summary["any_target"].value_counts()
    temp_min_class_count = temp_class_counts.min()
    temp_stratify = temp_summary["any_target"] if temp_min_class_count >= 2 else None

    val_p, test_p = train_test_split(
        temp_summary["patient_id"],
        test_size=0.5,
        random_state=random_state,
        stratify=temp_stratify,
    )

    train_df = df[df["patient_id"].isin(train_p)].copy()
    val_df = df[df["patient_id"].isin(val_p)].copy()
    test_df = df[df["patient_id"].isin(test_p)].copy()

    for split_name, split_df in {
        "train": train_df,
        "val": val_df,
        "test": test_df,
    }.items():
        if split_df.empty:
            raise ValueError(f"{split_name} split is empty.")

    return train_df, val_df, test_df


def build_logistic_model(random_state: int = 42):
    return LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        solver="liblinear",
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


def _validate_binary_target(series: pd.Series, name: str):
    non_null = series.dropna()
    if non_null.empty:
        raise ValueError(f"{name} is empty after dropping nulls.")

    unique_vals = sorted(non_null.unique().tolist())
    if len(unique_vals) < 2:
        raise ValueError(
            f"{name} has only one class: {non_null.value_counts(dropna=False).to_dict()}"
        )


def run_bire_modeling(df, feature_cols, threshold=0.5, random_state: int = 42):
    print("Inside modeling function")
    print("target in input df:", "target" in df.columns)

    if "patient_id" not in df.columns:
        raise ValueError("Input dataframe must contain 'patient_id'.")
    if "target" not in df.columns:
        raise ValueError("Input dataframe must contain 'target'.")

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing features: {missing}")

    optional_keep_cols = [c for c in ["timestamp", "event_now"] if c in df.columns]
    required_cols = ["patient_id", "target"] + optional_keep_cols + feature_cols

    work_df = df[required_cols].copy()
    work_df = work_df.dropna(subset=["target"])

    if work_df.empty:
        raise ValueError("No rows remain after dropping null targets.")

    _validate_binary_target(work_df["target"], "Full dataframe target")

    print("Full row-level target counts:", work_df["target"].value_counts(dropna=False).to_dict())
    print(
        "Full patient-level any_target counts:",
        work_df.groupby("patient_id")["target"].max().value_counts(dropna=False).to_dict()
    )

    train_df, val_df, test_df = patient_level_split(
        work_df,
        random_state=random_state,
    )

    print("target in train_df:", "target" in train_df.columns)
    print("target in val_df:", "target" in val_df.columns)
    print("target in test_df:", "target" in test_df.columns)

    print("Train row-level target counts:", train_df["target"].value_counts(dropna=False).to_dict())
    print("Val row-level target counts:", val_df["target"].value_counts(dropna=False).to_dict())
    print("Test row-level target counts:", test_df["target"].value_counts(dropna=False).to_dict())

    print(
        "Train patient-level any_target counts:",
        train_df.groupby("patient_id")["target"].max().value_counts(dropna=False).to_dict()
    )
    print(
        "Val patient-level any_target counts:",
        val_df.groupby("patient_id")["target"].max().value_counts(dropna=False).to_dict()
    )
    print(
        "Test patient-level any_target counts:",
        test_df.groupby("patient_id")["target"].max().value_counts(dropna=False).to_dict()
    )

    X_train = train_df[feature_cols].copy()
    y_train = train_df["target"].copy()

    X_val = val_df[feature_cols].copy()
    y_val = val_df["target"].copy()

    X_test = test_df[feature_cols].copy()
    y_test = test_df["target"].copy()

    _validate_binary_target(y_train, "Training target")
    _validate_binary_target(y_val, "Validation target")
    _validate_binary_target(y_test, "Test target")

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

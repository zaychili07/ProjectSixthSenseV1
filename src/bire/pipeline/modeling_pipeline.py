import pandas as pd
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier

from bire.evaluation.metrics import evaluate_multiple_splits


def run_bire_modeling(df, feature_cols, threshold=0.5, window=3):
    print("Inside modeling function")
    print("target in input df:", "target" in df.columns)

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing features: {missing}")

    train_df, test_df = time_aware_patient_split(df)

    print("target in train_df:", "target" in train_df.columns)
    print("target in test_df:", "target" in test_df.columns)

    X_train = train_df[feature_cols]
    y_train = train_df["target"]

    X_test = test_df[feature_cols]

    model = build_logistic_model()
    model.fit(X_train, y_train)

    test_df = test_df.copy()
    test_df["pred_proba"] = model.predict_proba(X_test)[:, 1]
    test_df = apply_alert_logic(test_df, threshold=threshold, window=window)

    return model, train_df, test_df


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

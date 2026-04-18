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

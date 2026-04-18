def validate_target_columns(df):
    if "target" not in df.columns:
        raise ValueError("Missing 'target' column")

    if "event_now" not in df.columns:
        raise ValueError("Missing 'event_now' column")

    if df["target"].sum() == 0:
        print("Warning: No positive target events detected")

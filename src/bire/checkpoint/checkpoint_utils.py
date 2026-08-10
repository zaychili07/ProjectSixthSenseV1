import json
from pathlib import Path
from datetime import datetime
import pandas as pd

# this function ensures that any columns in the dataframe that contain lists, dicts, tuples, or sets are converted to JSON strings
# before saving to parquet. This is necessary because parquet does not support these data types natively.
def parquet_safe(df: pd.DataFrame) -> pd.DataFrame:
    safe_df = df.copy()

    for col in safe_df.columns:
        if safe_df[col].dtype == "object":
            safe_df[col] = safe_df[col].apply(
                lambda x: json.dumps(x)
                if isinstance(x, (list, dict, tuple, set))
                else x
            )

    return safe_df


def make_checkpoint_dir(checkpoint_dir: str | Path) -> Path:
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return checkpoint_dir


def build_manifest(checkpoint_name: str) -> dict:
    return {
        "checkpoint_name": checkpoint_name,
        "created_at": datetime.now().isoformat(),
        "saved_dataframes": {},
        "missing_or_failed": {},
    }
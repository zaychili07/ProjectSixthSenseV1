import json
from pathlib import Path
import pandas as pd

from bire.checkpoint.checkpoint_utils import (
    parquet_safe,
    make_checkpoint_dir,
    build_manifest,
)

# This function saves the progress of the BIRE OS recovery process by saving the provided dataframes to parquet files
#  in a specified checkpoint directory.
def save_checkpoint(
    dataframes: dict[str, pd.DataFrame],
    checkpoint_dir: str | Path = "outputs/checkpoints/bire_os_recovery_checkpoint",
    checkpoint_name: str = "bire_os_recovery_checkpoint",
) -> dict:
    checkpoint_dir = make_checkpoint_dir(checkpoint_dir)
    manifest = build_manifest(checkpoint_name)

    for name, df in dataframes.items():
        try:
            if not isinstance(df, pd.DataFrame):
                raise TypeError(f"{name} is not a pandas DataFrame")

            path = checkpoint_dir / f"{name}.parquet"
            safe_df = parquet_safe(df)
            safe_df.to_parquet(path, index=False)

            manifest["saved_dataframes"][name] = {
                "rows": int(df.shape[0]),
                "columns": int(df.shape[1]),
                "path": str(path),
            }

        except Exception as e:
            manifest["missing_or_failed"][name] = str(e)

    manifest_path = checkpoint_dir / "checkpoint_manifest.json"

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=4)

    print(f"✅ Saved {len(manifest['saved_dataframes'])} checkpoint tables")
    print(f"⚠️ Failed/Missing: {len(manifest['missing_or_failed'])}")

    return manifest
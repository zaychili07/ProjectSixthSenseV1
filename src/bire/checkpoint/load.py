import json
from pathlib import Path
import pandas as pd

# This function loads the progress of the BIRE OS recovery process by reading the saved dataframes from parquet files
def load_checkpoint(
    checkpoint_dir: str | Path = "outputs/checkpoints/bire_os_recovery_checkpoint",
) -> dict[str, pd.DataFrame]:
    checkpoint_dir = Path(checkpoint_dir)
    manifest_path = checkpoint_dir / "checkpoint_manifest.json"

    if not manifest_path.exists():
        raise FileNotFoundError(f"Checkpoint manifest not found: {manifest_path}")

    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    loaded = {}

    for name, info in manifest["saved_dataframes"].items():
        loaded[name] = pd.read_parquet(info["path"])

    print(f"✅ Reloaded {len(loaded)} checkpoint tables")

    return loaded
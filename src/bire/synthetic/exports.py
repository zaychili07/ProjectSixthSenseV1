"""
BIRE OS Synthetic Ecosystem Export Pipeline

Chapter 53 doctrine:
No more babying BIRE OS.

Purpose:
Export, preserve, document, and prepare the Chapter 53 synthetic healthcare
ecosystem for audit, replay, DuckDB ingestion, Parquet migration, dashboarding,
and resilient intelligence benchmarking.

This module is not only a CSV saver.

It is the ecosystem preservation layer.

We Detect What Others Miss.
"""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd


# =========================================================
# Utilities
# =========================================================

def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def _json_default(obj):
    if hasattr(obj, "item"):
        return obj.item()
    if pd.isna(obj):
        return None
    return str(obj)


def _safe_write_json(data, path):
    with open(path, "w") as f:
        json.dump(
            data,
            f,
            indent=4,
            default=_json_default,
        )


def _save_csv(df, path):
    df.to_csv(path, index=False)


def _save_parquet_if_enabled(df, path, enabled=False):
    if not enabled:
        return None

    df.to_parquet(path, index=False)
    return str(path)


def _file_sha256(path):
    path = Path(path)

    if not path.exists():
        return None

    sha = hashlib.sha256()

    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            sha.update(block)

    return sha.hexdigest()


def _dataframe_profile(df, name):
    return {
        "name": name,
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "column_names": list(df.columns),
        "memory_mb": round(float(df.memory_usage(deep=True).sum() / 1_000_000), 4),
    }


def _column_intersection(df, columns):
    return [col for col in columns if col in df.columns]


def _safe_subset(df, columns):
    existing = _column_intersection(df, columns)
    if not existing:
        return pd.DataFrame()
    return df[existing].copy()


def _safe_filter(df, col, value=True):
    if col not in df.columns:
        return pd.DataFrame()
    return df[df[col] == value].copy()


def _safe_threshold_filter(df, col, threshold):
    if col not in df.columns:
        return pd.DataFrame()
    return df[pd.to_numeric(df[col], errors="coerce") >= threshold].copy()


def _explode_pipe_values(series):
    return (
        series
        .fillna("none")
        .astype(str)
        .str.split(" | ")
        .explode()
        .str.strip()
    )


# =========================================================
# Learning / upgrade summaries
# =========================================================

def build_learning_signal_summary(outcomes_df):
    if "bire_learning_signal" not in outcomes_df.columns:
        return pd.DataFrame(columns=["learning_signal", "count"])

    exploded = _explode_pipe_values(outcomes_df["bire_learning_signal"])

    summary = exploded.value_counts().reset_index()
    summary.columns = ["learning_signal", "count"]

    return summary


def build_upgrade_area_summary(outcomes_df):
    if "recommended_system_upgrade_area" not in outcomes_df.columns:
        return pd.DataFrame(columns=["upgrade_area", "count"])

    exploded = (
        outcomes_df["recommended_system_upgrade_area"]
        .apply(lambda x: x if isinstance(x, list) else [str(x)])
        .explode()
        .astype(str)
        .str.strip()
    )

    summary = exploded.value_counts().reset_index()
    summary.columns = ["upgrade_area", "count"]

    return summary


def build_outcome_teaching_summary(outcomes_df):
    if "outcome_teaching_statement" not in outcomes_df.columns:
        return pd.DataFrame(columns=["outcome_teaching_statement", "count"])

    summary = (
        outcomes_df["outcome_teaching_statement"]
        .astype(str)
        .value_counts()
        .reset_index()
    )

    summary.columns = ["outcome_teaching_statement", "count"]

    return summary


# =========================================================
# Replay bundle builders
# =========================================================

def build_replay_bundles(outcomes_df):
    replay = {}

    replay["false_reassurance_cases"] = _safe_filter(
        outcomes_df,
        "false_reassurance_outcome_flag",
        True,
    )

    replay["self_doubt_cases"] = _safe_filter(
        outcomes_df,
        "bire_self_doubt_required_flag",
        True,
    )

    replay["synthetic_regret_cases"] = _safe_filter(
        outcomes_df,
        "synthetic_regret_signal",
        True,
    )

    replay["missed_hidden_deterioration_cases"] = _safe_filter(
        outcomes_df,
        "missed_hidden_deterioration_signal",
        True,
    )

    replay["stabilized_but_not_safe_cases"] = _safe_filter(
        outcomes_df,
        "stabilized_but_not_safe",
        True,
    )

    replay["early_discharge_instability_cases"] = _safe_filter(
        outcomes_df,
        "early_discharge_instability",
        True,
    )

    replay["masked_physiology_failure_cases"] = _safe_filter(
        outcomes_df,
        "masked_physiology_failure",
        True,
    )

    replay["operationally_amplified_harm_cases"] = _safe_filter(
        outcomes_df,
        "operationally_amplified_harm",
        True,
    )

    replay["systemically_missed_decline_cases"] = _safe_filter(
        outcomes_df,
        "systemically_missed_decline",
        True,
    )

    replay["high_complexity_cases"] = _safe_threshold_filter(
        outcomes_df,
        "longitudinal_complexity_score",
        0.65,
    )

    return replay


# =========================================================
# Main export
# =========================================================

def export_synthetic_ecosystem(
    patient_master_df,
    encounter_df,
    encounter_lifecycle_df,
    context_df,
    labs_df,
    imaging_df,
    medication_df,
    diagnosis_df,
    intervention_df,
    handoff_df,
    operations_df,
    outcomes_df,
    ecosystem_outputs,
    output_dir="outputs/chapter_53_exports",
    save_parquet=False,
):
    """
    Export the complete Chapter 53 synthetic healthcare ecosystem.

    Parameters
    ----------
    save_parquet:
        If True, also writes parquet copies for future DuckDB workflows.
        Keep False by default until infrastructure phase begins.
    """

    base_dir = Path(output_dir)

    stage_dir = base_dir / "stages"
    parquet_dir = base_dir / "parquet"
    core_dir = base_dir / "core"
    clinical_dir = base_dir / "clinical"
    operational_dir = base_dir / "operational"
    outcomes_dir = base_dir / "outcomes"
    learning_dir = base_dir / "learning"
    audit_dir = base_dir / "audits"
    schema_dir = base_dir / "schema"
    high_risk_dir = base_dir / "high_risk"
    uncertainty_dir = base_dir / "high_uncertainty"
    replay_dir = base_dir / "replay_bundles"
    manifest_dir = base_dir / "manifest"
    snapshot_dir = base_dir / "snapshots"
    integrity_dir = base_dir / "integrity"
    duckdb_ready_dir = base_dir / "duckdb_ready"

    for d in [
        stage_dir,
        parquet_dir,
        core_dir,
        clinical_dir,
        operational_dir,
        outcomes_dir,
        learning_dir,
        audit_dir,
        schema_dir,
        high_risk_dir,
        uncertainty_dir,
        replay_dir,
        manifest_dir,
        snapshot_dir,
        integrity_dir,
        duckdb_ready_dir,
    ]:
        ensure_dir(d)

    export_timestamp_utc = datetime.now(timezone.utc).isoformat()

    # -----------------------------
    # Stage-by-stage exports
    # -----------------------------

    stage_exports = {
        "01_patient_master": patient_master_df,
        "02_encounters": encounter_df,
        "03_encounter_lifecycle": encounter_lifecycle_df,
        "04_context": context_df,
        "05_labs": labs_df,
        "06_imaging": imaging_df,
        "07_medications": medication_df,
        "08_diagnoses": diagnosis_df,
        "09_interventions": intervention_df,
        "10_handoffs": handoff_df,
        "11_operations": operations_df,
        "12_outcomes": outcomes_df,
    }

    file_registry = []

    for name, df in stage_exports.items():
        csv_path = stage_dir / f"{name}.csv"
        _save_csv(df, csv_path)

        file_registry.append(
            {
                "artifact_type": "stage_csv",
                "name": name,
                "path": str(csv_path),
                "sha256": _file_sha256(csv_path),
                "rows": int(len(df)),
                "columns": int(len(df.columns)),
            }
        )

        parquet_path = parquet_dir / f"{name}.parquet"
        parquet_written = _save_parquet_if_enabled(
            df,
            parquet_path,
            enabled=save_parquet,
        )

        if parquet_written:
            file_registry.append(
                {
                    "artifact_type": "stage_parquet",
                    "name": name,
                    "path": str(parquet_path),
                    "sha256": _file_sha256(parquet_path),
                    "rows": int(len(df)),
                    "columns": int(len(df.columns)),
                }
            )

    # -----------------------------
    # Thematic exports
    # -----------------------------

    thematic_exports = {
        core_dir / "patient_master.csv": patient_master_df,
        core_dir / "encounters.csv": encounter_df,
        core_dir / "encounter_lifecycle.csv": encounter_lifecycle_df,
        clinical_dir / "context.csv": context_df,
        clinical_dir / "labs.csv": labs_df,
        clinical_dir / "imaging.csv": imaging_df,
        clinical_dir / "medications.csv": medication_df,
        clinical_dir / "diagnoses.csv": diagnosis_df,
        clinical_dir / "interventions.csv": intervention_df,
        operational_dir / "handoffs.csv": handoff_df,
        operational_dir / "operations.csv": operations_df,
        outcomes_dir / "outcomes.csv": outcomes_df,
    }

    for path, df in thematic_exports.items():
        _save_csv(df, path)

    # -----------------------------
    # Full snapshot
    # -----------------------------

    snapshot_path = snapshot_dir / "full_synthetic_world_snapshot.csv"
    _save_csv(outcomes_df, snapshot_path)

    file_registry.append(
        {
            "artifact_type": "full_snapshot",
            "name": "full_synthetic_world_snapshot",
            "path": str(snapshot_path),
            "sha256": _file_sha256(snapshot_path),
            "rows": int(len(outcomes_df)),
            "columns": int(len(outcomes_df.columns)),
        }
    )

    if save_parquet:
        snapshot_parquet_path = parquet_dir / "full_synthetic_world_snapshot.parquet"
        outcomes_df.to_parquet(snapshot_parquet_path, index=False)

        file_registry.append(
            {
                "artifact_type": "full_snapshot_parquet",
                "name": "full_synthetic_world_snapshot",
                "path": str(snapshot_parquet_path),
                "sha256": _file_sha256(snapshot_parquet_path),
                "rows": int(len(outcomes_df)),
                "columns": int(len(outcomes_df.columns)),
            }
        )

    # -----------------------------
    # Audit artifacts
    # -----------------------------

    ecosystem_audit = ecosystem_outputs["audit"]

    audit_path = audit_dir / "ecosystem_audit.json"
    _safe_write_json(ecosystem_audit, audit_path)

    ecosystem_outputs["schema_df"].to_csv(
        schema_dir / "schema_report.csv",
        index=False,
    )

    audit_artifact_map = {
        "contradiction_cases.csv": "contradiction_df",
        "high_risk_cases.csv": "high_risk_df",
        "high_uncertainty_cases.csv": "high_uncertainty_df",
        "high_regret_cases.csv": "high_regret_df",
        "high_self_doubt_cases.csv": "high_self_doubt_df",
        "false_reassurance_cases.csv": "false_reassurance_df",
        "upgrade_pressure_cases.csv": "upgrade_pressure_df",
    }

    for filename, key in audit_artifact_map.items():
        if key in ecosystem_outputs:
            ecosystem_outputs[key].to_csv(
                audit_dir / filename,
                index=False,
            )

    # -----------------------------
    # Learning artifacts
    # -----------------------------

    learning_summary_df = build_learning_signal_summary(outcomes_df)
    upgrade_summary_df = build_upgrade_area_summary(outcomes_df)
    teaching_summary_df = build_outcome_teaching_summary(outcomes_df)

    learning_summary_df.to_csv(
        learning_dir / "learning_signal_summary.csv",
        index=False,
    )

    upgrade_summary_df.to_csv(
        learning_dir / "recommended_upgrade_area_summary.csv",
        index=False,
    )

    teaching_summary_df.to_csv(
        learning_dir / "outcome_teaching_statement_summary.csv",
        index=False,
    )

    # -----------------------------
    # Replay bundles
    # -----------------------------

    replay_bundles = build_replay_bundles(outcomes_df)

    replay_manifest = {}

    for name, replay_df in replay_bundles.items():
        replay_path = replay_dir / f"{name}.csv"
        replay_df.to_csv(replay_path, index=False)

        replay_manifest[name] = {
            "path": str(replay_path),
            "rows": int(len(replay_df)),
            "columns": int(len(replay_df.columns)),
            "purpose": "Replay focused synthetic cases for BIRE OS stress testing.",
        }

    _safe_write_json(
        replay_manifest,
        replay_dir / "replay_manifest.json",
    )

    # -----------------------------
    # DuckDB / Parquet readiness notes
    # -----------------------------

    duckdb_bootstrap_sql = """
-- BIRE OS Chapter 53 DuckDB Bootstrap
-- Run after parquet exports are enabled.

CREATE OR REPLACE VIEW patient_master AS
SELECT * FROM read_parquet('parquet/01_patient_master.parquet');

CREATE OR REPLACE VIEW encounters AS
SELECT * FROM read_parquet('parquet/02_encounters.parquet');

CREATE OR REPLACE VIEW encounter_lifecycle AS
SELECT * FROM read_parquet('parquet/03_encounter_lifecycle.parquet');

CREATE OR REPLACE VIEW context AS
SELECT * FROM read_parquet('parquet/04_context.parquet');

CREATE OR REPLACE VIEW labs AS
SELECT * FROM read_parquet('parquet/05_labs.parquet');

CREATE OR REPLACE VIEW imaging AS
SELECT * FROM read_parquet('parquet/06_imaging.parquet');

CREATE OR REPLACE VIEW medications AS
SELECT * FROM read_parquet('parquet/07_medications.parquet');

CREATE OR REPLACE VIEW diagnoses AS
SELECT * FROM read_parquet('parquet/08_diagnoses.parquet');

CREATE OR REPLACE VIEW interventions AS
SELECT * FROM read_parquet('parquet/09_interventions.parquet');

CREATE OR REPLACE VIEW handoffs AS
SELECT * FROM read_parquet('parquet/10_handoffs.parquet');

CREATE OR REPLACE VIEW operations AS
SELECT * FROM read_parquet('parquet/11_operations.parquet');

CREATE OR REPLACE VIEW outcomes AS
SELECT * FROM read_parquet('parquet/12_outcomes.parquet');
"""

    with open(duckdb_ready_dir / "duckdb_bootstrap.sql", "w") as f:
        f.write(duckdb_bootstrap_sql.strip() + "\n")

    # -----------------------------
    # Manifest
    # -----------------------------

    manifest = ecosystem_audit.get("manifest", {})

    export_manifest = {
        "ecosystem_name": "BIRE_OS_SYNTHETIC_HEALTHCARE_WORLD",
        "chapter": "53",
        "export_layer": "53.17",
        "export_timestamp_utc": export_timestamp_utc,
        "synthetic_doctrine": "No more babying BIRE OS.",
        "mission": "Break it now, so humans do not pay later.",
        "motto": "We Detect What Others Miss.",
        "source_manifest": manifest,
        "stage_order": list(stage_exports.keys()),
        "final_snapshot": "snapshots/full_synthetic_world_snapshot.csv",
        "parquet_enabled": bool(save_parquet),
        "duckdb_ready": bool(save_parquet),
        "audit_available": True,
        "replay_bundles_available": True,
        "learning_artifacts_available": True,
    }

    _safe_write_json(
        export_manifest,
        manifest_dir / "ecosystem_export_manifest.json",
    )

    # -----------------------------
    # Profiles
    # -----------------------------

    dataframe_profiles = [
        _dataframe_profile(df, name)
        for name, df in stage_exports.items()
    ]

    _safe_write_json(
        dataframe_profiles,
        manifest_dir / "dataframe_profiles.json",
    )

    # -----------------------------
    # Integrity registry
    # -----------------------------

    registry_df = pd.DataFrame(file_registry)
    registry_df.to_csv(
        integrity_dir / "file_registry.csv",
        index=False,
    )

    _safe_write_json(
        file_registry,
        integrity_dir / "file_registry.json",
    )

    # -----------------------------
    # Export summary
    # -----------------------------

    export_summary = {
        "base_dir": str(base_dir),
        "export_timestamp_utc": export_timestamp_utc,
        "stage_exports": len(stage_exports),
        "core_exports": 3,
        "clinical_exports": 6,
        "operational_exports": 2,
        "outcome_exports": 1,
        "audit_exports": len(audit_artifact_map) + 1,
        "schema_exports": 1,
        "learning_exports": 3,
        "replay_bundle_exports": len(replay_bundles),
        "snapshot_exports": 1,
        "manifest_exports": 2,
        "integrity_exports": 2,
        "parquet_enabled": bool(save_parquet),
        "total_rows_final_snapshot": int(len(outcomes_df)),
        "total_columns_final_snapshot": int(len(outcomes_df.columns)),
        "ecosystem_pressure_grade": ecosystem_audit.get(
            "ecosystem_pressure_scorecard",
            {},
        ).get("ecosystem_pressure_grade", "unknown"),
        "ecosystem_pressure_score": ecosystem_audit.get(
            "ecosystem_pressure_scorecard",
            {},
        ).get("ecosystem_pressure_score", None),
    }

    _safe_write_json(
        export_summary,
        base_dir / "export_summary.json",
    )

    return export_summary
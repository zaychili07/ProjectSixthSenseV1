from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

CORE_EXPORTS = [
    "raw_df",
    "cycle1_df",
    "v2_df",
    "test_df",
    "v2_scored_test_df",
    "xgb_scored_df",
]

INTELLIGENCE_EXPORTS = [
    "gss_df",
    "ibpip_gss_df",
    "gss_ve_df",
    "episodes_df",
    "episode_timing_df",
    "bms_episodes_df",
    "bire_output_df",
    "bire_monitor_df",
    "bire_re_df",
    "bire_critical_df",
    "bire_fi_df",
    "bire_fi_ibpip_df",
    "forecast_bridge_df",
]

DASHBOARD_EXPORTS = [
    "dashboard_df",
    "patient_chart_df",
    "episode_view",
    "reasoning_view",
    "cohort_view",
    "psr_breakdown_view",
    "psr_operational_queue",
    "human_queue_view",
    "queue_state_view",
    "summary_queue_view",
]

AUDIT_EXPORTS = [
    "event_patient_audit",
    "gss_v3_timing_audit",
    "gss_v3_warning_audit",
    "warning_event_audit",
    "audit_source",
]

SUMMARY_EXPORTS = [
    "threshold_df",
    "xgb_results_df",
    "v2_metrics_df",
    "brier_summary_df",
    "alert_summary_df",
    "timing_summary_df",
    "policy_metrics_df",
    "episode_summary_df",
    "episode_patient_summary",
    "episode_duration_summary",
    "episode_burden_stats",
    "lead_time_stats",
    "watch_override_impact",
    "final_watch_policy_impact",
]


EXPORT_GROUPS = {
    "core": CORE_EXPORTS,
    "intelligence": INTELLIGENCE_EXPORTS,
    "dashboard": DASHBOARD_EXPORTS,
    "audit": AUDIT_EXPORTS,
    "summary": SUMMARY_EXPORTS,
}


def collect_bire_artifacts(global_vars: dict) -> dict:
    """
    Collect only approved BIRE export artifacts
    from notebook globals().
    """

    artifacts = {}

    for group_name, export_names in EXPORT_GROUPS.items():

        for name in export_names:

            if name not in global_vars:
                continue

            obj = global_vars[name]

            if not isinstance(obj, pd.DataFrame):
                continue

            artifacts[name] = obj

    return artifacts


def make_export_dir(
    base_dir: str | Path = "outputs",
    run_name: Optional[str] = None,
) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = run_name or f"bire_run_{timestamp}"

    export_dir = Path(base_dir) / run_name
    export_dir.mkdir(parents=True, exist_ok=True)

    (export_dir / "csv").mkdir(exist_ok=True)
    (export_dir / "logs").mkdir(exist_ok=True)
    (export_dir / "audits").mkdir(exist_ok=True)

    return export_dir


def export_csv(
    df: pd.DataFrame,
    export_dir: str | Path,
    filename: str,
    subfolder: str = "csv",
    index: bool = False,
) -> Path:
    export_path = Path(export_dir) / subfolder / filename
    export_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(export_path, index=index)
    return export_path


def write_json_log(
    payload: dict,
    export_dir: str | Path,
    filename: str,
    subfolder: str = "logs",
) -> Path:
    log_path = Path(export_dir) / subfolder / filename
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4, default=str)

    return log_path


def export_bire_artifacts(
    artifacts: Dict[str, pd.DataFrame],
    base_dir: str | Path = "outputs",
    run_name: Optional[str] = None,
    metadata: Optional[dict] = None,
) -> dict:
    export_dir = make_export_dir(base_dir=base_dir, run_name=run_name)

    exported_files = {}

    for name, df in artifacts.items():
        if df is None:
            continue

        if not isinstance(df, pd.DataFrame):
            continue

        filename = f"{name}.csv"
        exported_files[name] = str(
            export_csv(df=df, export_dir=export_dir, filename=filename)
        )

    run_log = {
        "export_time": datetime.now().isoformat(),
        "run_name": run_name,
        "export_dir": str(export_dir),
        "num_artifacts": len(exported_files),
        "exported_files": exported_files,
        "metadata": metadata or {},
    }

    run_log_path = write_json_log(
        payload=run_log,
        export_dir=export_dir,
        filename="run_log.json",
    )

    exported_files["run_log"] = str(run_log_path)

    validation_df = validate_exported_files(exported_files)

    validation_path = export_csv(
        df=validation_df,
        export_dir=export_dir,
        filename="export_validation.csv",
        subfolder="logs",
    )

    manifest = build_run_manifest(
        artifacts=artifacts,
        exported_files=exported_files,
        metadata=metadata,
    )

    manifest_path = write_json_log(
        payload=manifest,
        export_dir=export_dir,
        filename="manifest.json",
        subfolder="logs",
    )

    exported_files["export_validation"] = str(validation_path)
    exported_files["manifest"] = str(manifest_path)


    # Audit-ready outputs
    audit_source = None

    for candidate in [
        "bire_fi_ibpip_df",
        "forecast_bridge_df",
        "bire_output_df",
        "bire_critical_df",
        "bire_re_df",
        "bire_monitor_df",
        "gss_ve_df",
        "ibpip_gss_df",
        "gss_df",
    ]:
        if candidate in artifacts:
            audit_source = artifacts[candidate]
            break

    if audit_source is not None:
        system_decision_audit = build_system_decision_audit(audit_source)
        suppression_audit = build_suppression_audit(audit_source)
        escalation_audit = build_escalation_audit(audit_source)

        exported_files["system_decision_audit"] = str(
            export_csv(
                df=system_decision_audit,
                export_dir=export_dir,
                filename="system_decision_audit.csv",
                subfolder="audits",
            )
        )

        exported_files["suppression_audit"] = str(
            export_csv(
                df=suppression_audit,
                export_dir=export_dir,
                filename="suppression_audit.csv",
                subfolder="audits",
            )
        )

        exported_files["escalation_audit"] = str(
            export_csv(
                df=escalation_audit,
                export_dir=export_dir,
                filename="escalation_audit.csv",
                subfolder="audits",
            )
        )

    system_event_log = build_system_event_log(
        exported_files=exported_files,
        metadata=metadata,
    )

    exported_files["system_event_log"] = str(
        export_csv(
            df=system_event_log,
            export_dir=export_dir,
            filename="system_event_log.csv",
            subfolder="logs",
        )
    )

    return {
        "export_dir": str(export_dir),
        "exported_files": exported_files,
    }


def build_decision_audit(
    df: pd.DataFrame,
    patient_col: str = "patient_id",
    time_col: str = "timestamp",
) -> pd.DataFrame:
    cols = [
        patient_col,
        time_col,
        "bire_state",
        "timing_category",
        "gss_alert",
        "gss_suppressed",
        "gss_ve_reason",
        "bire_fi_state",
        "ibpip_state",
        "combined_psr_score",
        "event_now",
        "target",
    ]

    existing_cols = [c for c in cols if c in df.columns]

    audit_df = df[existing_cols].copy()

    audit_df["audit_created_at"] = datetime.now().isoformat()

    return audit_df

def validate_exported_files(exported_files: dict) -> pd.DataFrame:
    """
    Validate that exported artifact files exist on disk.

    Parameters
    ----------
    exported_files : dict
        Dictionary of artifact names mapped to file paths.

    Returns
    -------
    pd.DataFrame
        Validation report with file existence and file size.
    """

    records = []

    for artifact_name, file_path in exported_files.items():
        path = Path(file_path)

        records.append(
            {
                "artifact_name": artifact_name,
                "file_path": str(path),
                "exists": path.exists(),
                "file_size_bytes": path.stat().st_size if path.exists() else None,
            }
        )

    return pd.DataFrame(records)


def build_run_manifest(
    artifacts: Dict[str, pd.DataFrame],
    exported_files: dict,
    metadata: Optional[dict] = None,
) -> dict:
    """
    Build a structured run manifest for exported BIRE artifacts.
    """

    artifact_summary = {}

    for name, df in artifacts.items():
        if isinstance(df, pd.DataFrame):
            artifact_summary[name] = {
                "rows": int(df.shape[0]),
                "columns": int(df.shape[1]),
                "column_names": list(df.columns),
            }

    manifest = {
        "manifest_created_at": datetime.now().isoformat(),
        "num_artifacts": len(artifact_summary),
        "artifact_summary": artifact_summary,
        "exported_files": exported_files,
        "metadata": metadata or {},
    }

    return manifest

def build_system_decision_audit(
    df: pd.DataFrame,
    patient_col: str = "patient_id",
    time_col: str = "timestamp",
) -> pd.DataFrame:
    """
    Build a lightweight system decision audit dataframe.

    This preserves key BIRE operational decision fields for later review.
    """

    preferred_cols = [
        patient_col,
        time_col,

        # Risk / prediction
        "pred_proba",
        "risk_15",
        "risk_30",
        "risk_60",

        # Target / event context
        "event_now",
        "target",
        "target_15",
        "target_30",
        "target_60",

        # GSS / suppression
        "gss_alert",
        "gss_suppressed",
        "gss_escalation_reason",
        "gss_ve_reason",

        # BMS / mode
        "bire_mode",
        "care_mode",
        "bms_mode",
        "esi_level",

        # IBPIP
        "ibpip_score",
        "ibpip_state",
        "ibpip_baseline_ready",
        "ibpip_n_abnormal_signals",

        # BIRE-FI / lifecycle
        "bire_fi_score",
        "bire_fi_state",
        "bire_state",
        "bire_final_tier",
        "timing_category",

        # Episodes / PSR
        "episode_id",
        "alert_episode_flag",
        "combined_psr_score",
        "attention_band",
        "operational_priority",

        # Reasoning
        "bire_decision_reason",
        "watch_override_reason",
        "monitor_reason",
        "escalation_reason",
    ]

    existing_cols = [c for c in preferred_cols if c in df.columns]

    audit_df = df[existing_cols].copy()
    audit_df["audit_created_at"] = datetime.now().isoformat()

    return audit_df


def build_suppression_audit(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a suppression-focused audit table.
    """

    suppression_cols = [
        "patient_id",
        "timestamp",
        "pred_proba",
        "risk_15",
        "risk_30",
        "risk_60",
        "event_now",
        "gss_alert",
        "gss_suppressed",
        "gss_escalation_reason",
        "gss_ve_reason",
        "watch_override_reason",
        "bire_state",
        "bire_final_tier",
        "timing_category",
    ]

    existing_cols = [c for c in suppression_cols if c in df.columns]
    audit_df = df[existing_cols].copy()

    if "gss_suppressed" in audit_df.columns:
        audit_df = audit_df[audit_df["gss_suppressed"] == True].copy()

    audit_df["audit_type"] = "suppression_audit"
    audit_df["audit_created_at"] = datetime.now().isoformat()

    return audit_df


def build_escalation_audit(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build an escalation-focused audit table.
    """

    escalation_cols = [
        "patient_id",
        "timestamp",
        "pred_proba",
        "risk_15",
        "risk_30",
        "risk_60",
        "event_now",
        "gss_alert",
        "gss_suppressed",
        "gss_escalation_reason",
        "gss_ve_reason",
        "bire_fi_score",
        "bire_fi_state",
        "bire_state",
        "bire_final_tier",
        "timing_category",
        "episode_id",
        "alert_episode_flag",
        "combined_psr_score",
        "attention_band",
        "operational_priority",
        "escalation_reason",
    ]

    existing_cols = [c for c in escalation_cols if c in df.columns]
    audit_df = df[existing_cols].copy()

    escalation_indicators = []

    for col in ["gss_alert", "alert_episode_flag"]:
        if col in audit_df.columns:
            escalation_indicators.append(audit_df[col] == True)

    for col in ["bire_state", "bire_final_tier", "bire_fi_state"]:
        if col in audit_df.columns:
            escalation_indicators.append(
                audit_df[col].astype(str).str.upper().isin(["WATCH", "ESCALATE", "URGENT", "RE-ESCALATE", "CRITICAL"])
            )

    if escalation_indicators:
        mask = escalation_indicators[0]
        for indicator in escalation_indicators[1:]:
            mask = mask | indicator
        audit_df = audit_df[mask].copy()

    audit_df["audit_type"] = "escalation_audit"
    audit_df["audit_created_at"] = datetime.now().isoformat()

    return audit_df


def build_system_event_log(exported_files: dict, metadata: Optional[dict] = None) -> pd.DataFrame:
    """
    Build a lightweight internal system event log for the export run.
    """

    now = datetime.now().isoformat()

    records = [
        {
            "event_time": now,
            "event_type": "EXPORT_RUN_COMPLETED",
            "event_source": "bire.io.exports",
            "message": "BIRE artifact export run completed.",
            "num_exported_files": len(exported_files),
            "metadata": json.dumps(metadata or {}, default=str),
        }
    ]

    for artifact_name, file_path in exported_files.items():
        records.append(
            {
                "event_time": now,
                "event_type": "ARTIFACT_EXPORTED",
                "event_source": "bire.io.exports",
                "message": f"Artifact exported: {artifact_name}",
                "num_exported_files": len(exported_files),
                "metadata": json.dumps(
                    {
                        "artifact_name": artifact_name,
                        "file_path": file_path,
                    },
                    default=str,
                ),
            }
        )

    return pd.DataFrame(records)

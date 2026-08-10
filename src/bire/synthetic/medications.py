"""
BIRE OS Synthetic Therapeutic Systems Intelligence Engine

Chapter 53 doctrine:
No more babying BIRE OS.

Purpose:
Generate longitudinal therapeutic behavior under:
- operational pressure
- treatment masking
- therapy escalation
- support dependence
- secondary deterioration
- rebound instability
- false stabilization
- iatrogenic complications
- medication chaos

This is NOT medication logging.

This is:
Therapeutic Systems Intelligence.

Doctrine:
We Detect What Others Miss.
"""

import numpy as np
import pandas as pd

from bire.synthetic.config import (
    SYNTHETIC_ECOSYSTEM_CONFIG,
)


MEDICATION_CLASSES = {
    "oxygen": "respiratory_support",
    "high_flow_oxygen": "advanced_respiratory_support",
    "bipap": "noninvasive_ventilation",
    "iv_fluids": "hemodynamic_support",
    "blood_transfusion": "circulatory_support",
    "insulin": "metabolic_management",
    "dextrose": "metabolic_rescue",
    "antibiotics": "infection_management",
    "broad_spectrum_antibiotics": "critical_infection_management",
    "vasopressors": "critical_hemodynamic_support",
    "antiarrhythmic": "cardiac_management",
    "anticoagulant": "thrombotic_prevention",
    "diuretic": "fluid_management",
    "steroids": "anti_inflammatory",
    "bronchodilator": "respiratory_management",
    "morphine": "opioid_pain_management",
    "dilaudid": "opioid_pain_management",
    "sedative": "critical_care_support",
    "antipyretic": "symptom_suppression",
}


ALL_MEDICATIONS = list(
    MEDICATION_CLASSES.keys()
)


THERAPEUTIC_TRAJECTORY_STATES = [
    "TRUE_RECOVERY_RESPONSE",
    "PARTIAL_STABILIZATION",
    "THERAPY_DEPENDENT_STABILITY",
    "FALSE_RECOVERY_RESPONSE",
    "MASKED_DECLINE",
    "SECONDARY_COMPLICATION_EVOLUTION",
    "ESCALATING_SUPPORT_REQUIREMENT",
    "REBOUND_AFTER_DEESCALATION",
    "PROGRESSIVE_TREATMENT_FAILURE",
]


SECONDARY_DETERIORATION_TYPES = [
    "none",
    "opioid_respiratory_decline",
    "vasopressor_perfusion_mismatch",
    "fluid_overload_respiratory_failure",
    "steroid_hyperglycemic_instability",
    "anticoagulant_bleeding_instability",
    "sedation_masked_decline",
    "renal_dosing_failure",
    "electrolyte_destabilization",
    "therapy_rebound_instability",
]


def _clip(value, low, high):
    return float(np.clip(value, low, high))


def generate_therapeutic_intelligence(
    df,
    random_seed=None,
):

    if random_seed is None:
        random_seed = (
            SYNTHETIC_ECOSYSTEM_CONFIG[
                "random_seed"
            ]
        )

    rng = np.random.default_rng(
        random_seed
    )

    medication_rows = []

    for _, row in df.iterrows():

        care_mode = row["care_mode"]

        condition_profile = row[
            "condition_profile"
        ]

        fragility = float(
            row["fragility_score"]
        )

        hidden_instability = float(
            row.get(
                "silent_collapse_pressure",
                0.0,
            )
        )

        deceptive_stability = float(
            row.get(
                "deceptive_stability_index",
                0.0,
            )
        )

        overload_pressure = float(
            row.get(
                "hospital_overload_pressure",
                0.0,
            )
        )

        has_event = bool(
            row["has_event"]
        )

        post_event_state = row[
            "post_event_state"
        ]

        allergy_type = row.get(
            "allergy_type",
            "none",
        )

        # ==========================================
        # Medication count
        # ==========================================

        if care_mode == "OUTPATIENT":
            med_count = rng.integers(
                0,
                4,
            )

        elif "ER" in care_mode:
            med_count = rng.integers(
                2,
                7,
            )

        elif care_mode == "INPATIENT":
            med_count = rng.integers(
                4,
                10,
            )

        elif care_mode == "ICU":
            med_count = rng.integers(
                7,
                16,
            )

        med_count += int(
            fragility * 4
        )

        # ==========================================
        # Medication pool
        # ==========================================

        medication_pool = (
            ALL_MEDICATIONS.copy()
        )

        if (
            "copd"
            in condition_profile
        ):
            medication_pool.extend(
                [
                    "oxygen",
                    "high_flow_oxygen",
                    "bipap",
                    "bronchodilator",
                    "steroids",
                ]
            )

        if (
            "diabetes"
            in condition_profile
        ):
            medication_pool.extend(
                [
                    "insulin",
                    "dextrose",
                ]
            )

        if (
            "arrhythmia"
            in condition_profile
        ):
            medication_pool.extend(
                [
                    "antiarrhythmic",
                ]
            )

        if (
            "sepsis"
            in condition_profile
        ):
            medication_pool.extend(
                [
                    "broad_spectrum_antibiotics",
                    "vasopressors",
                    "iv_fluids",
                ]
            )

        if care_mode == "ICU":

            medication_pool.extend(
                [
                    "vasopressors",
                    "sedative",
                    "high_flow_oxygen",
                ]
            )

        selected_meds = list(
            rng.choice(
                medication_pool,
                size=min(
                    med_count,
                    len(
                        set(
                            medication_pool
                        )
                    ),
                ),
                replace=False,
            )
        )

        medication_classes = [
            MEDICATION_CLASSES.get(
                med,
                "unknown",
            )
            for med in selected_meds
        ]

        # ==========================================
        # Treatment masking pressure
        # ==========================================

        treatment_masking_risk = 0

        if (
            "oxygen"
            in selected_meds
            or "high_flow_oxygen"
            in selected_meds
            or "bipap"
            in selected_meds
        ):
            treatment_masking_risk += (
                0.18
            )

        if (
            "vasopressors"
            in selected_meds
        ):
            treatment_masking_risk += (
                0.22
            )

        if (
            "sedative"
            in selected_meds
        ):
            treatment_masking_risk += (
                0.20
            )

        if (
            "morphine"
            in selected_meds
            or "dilaudid"
            in selected_meds
        ):
            treatment_masking_risk += (
                0.16
            )

        if (
            "antipyretic"
            in selected_meds
        ):
            treatment_masking_risk += (
                0.08
            )

        treatment_masking_risk += (
            deceptive_stability
            * 0.25
        )

        treatment_masking_risk = (
            _clip(
                treatment_masking_risk,
                0,
                1,
            )
        )

        # ==========================================
        # Therapy dependency
        # ==========================================

        therapy_dependency_state = (
            "NONE"
        )

        if (
            "vasopressors"
            in selected_meds
        ):
            therapy_dependency_state = (
                "HEMODYNAMIC_SUPPORT_DEPENDENT"
            )

        elif (
            "high_flow_oxygen"
            in selected_meds
            or "bipap"
            in selected_meds
        ):
            therapy_dependency_state = (
                "RESPIRATORY_SUPPORT_DEPENDENT"
            )

        elif (
            "sedative"
            in selected_meds
        ):
            therapy_dependency_state = (
                "SEDATION_DEPENDENT_STABILITY"
            )

        # ==========================================
        # Secondary deterioration
        # ==========================================

        secondary_deterioration_triggered = (
            False
        )

        secondary_deterioration_type = (
            "none"
        )

        secondary_risk = (
            0.02
            + fragility * 0.08
            + hidden_instability * 0.10
        )

        if (
            treatment_masking_risk
            >= 0.40
        ):
            secondary_risk += 0.12

        if (
            len(selected_meds)
            >= 8
        ):
            secondary_risk += 0.08

        if (
            rng.random()
            < secondary_risk
        ):

            secondary_deterioration_triggered = (
                True
            )

            if (
                "morphine"
                in selected_meds
                or "dilaudid"
                in selected_meds
            ):
                secondary_deterioration_type = (
                    "opioid_respiratory_decline"
                )

            elif (
                "vasopressors"
                in selected_meds
            ):
                secondary_deterioration_type = (
                    "vasopressor_perfusion_mismatch"
                )

            elif (
                "iv_fluids"
                in selected_meds
            ):
                secondary_deterioration_type = (
                    "fluid_overload_respiratory_failure"
                )

            elif (
                "steroids"
                in selected_meds
            ):
                secondary_deterioration_type = (
                    "steroid_hyperglycemic_instability"
                )

            elif (
                "anticoagulant"
                in selected_meds
            ):
                secondary_deterioration_type = (
                    "anticoagulant_bleeding_instability"
                )

            elif (
                "sedative"
                in selected_meds
            ):
                secondary_deterioration_type = (
                    "sedation_masked_decline"
                )

            else:

                secondary_deterioration_type = (
                    rng.choice(
                        SECONDARY_DETERIORATION_TYPES[1:]
                    )
                )

        # ==========================================
        # Stabilization durability
        # ==========================================

        stabilization_durability_score = (
            0.70
        )

        stabilization_durability_score -= (
            treatment_masking_risk
            * 0.35
        )

        stabilization_durability_score -= (
            hidden_instability
            * 0.25
        )

        stabilization_durability_score -= (
            overload_pressure
            * 0.18
        )

        if secondary_deterioration_triggered:
            stabilization_durability_score -= (
                0.25
            )

        stabilization_durability_score = (
            _clip(
                stabilization_durability_score,
                0,
                1,
            )
        )

        # ==========================================
        # Rebound instability
        # ==========================================

        rebound_instability_risk = (
            0.06
            + (
                1
                - stabilization_durability_score
            )
            * 0.45
        )

        if (
            therapy_dependency_state
            != "NONE"
        ):
            rebound_instability_risk += (
                0.18
            )

        rebound_instability_risk = (
            _clip(
                rebound_instability_risk,
                0,
                1,
            )
        )

        # ==========================================
        # Therapeutic trajectory
        # ==========================================

        if (
            stabilization_durability_score
            >= 0.75
            and not secondary_deterioration_triggered
        ):

            therapeutic_trajectory_state = (
                "TRUE_RECOVERY_RESPONSE"
            )

        elif (
            treatment_masking_risk
            >= 0.45
            and hidden_instability
            >= 0.40
        ):

            therapeutic_trajectory_state = (
                "MASKED_DECLINE"
            )

        elif (
            secondary_deterioration_triggered
        ):

            therapeutic_trajectory_state = (
                "SECONDARY_COMPLICATION_EVOLUTION"
            )

        elif (
            rebound_instability_risk
            >= 0.45
        ):

            therapeutic_trajectory_state = (
                "REBOUND_AFTER_DEESCALATION"
            )

        elif (
            therapy_dependency_state
            != "NONE"
        ):

            therapeutic_trajectory_state = (
                "THERAPY_DEPENDENT_STABILITY"
            )

        else:

            therapeutic_trajectory_state = (
                rng.choice(
                    THERAPEUTIC_TRAJECTORY_STATES
                )
            )

        # ==========================================
        # Operational medication chaos
        # ==========================================

        pharmacy_overload_delay = (
            rng.random()
            < (
                0.04
                + overload_pressure
                * 0.10
            )
        )

        medication_charting_lag = (
            rng.random()
            < (
                0.05
                + overload_pressure
                * 0.12
            )
        )

        missed_dose_window = (
            rng.random()
            < (
                0.03
                + overload_pressure
                * 0.08
            )
        )

        duplicate_administration = (
            rng.random()
            < (
                0.01
                + overload_pressure
                * 0.03
            )
        )

        interrupted_therapy = (
            rng.random()
            < (
                0.03
                + hidden_instability
                * 0.08
            )
        )

        medication_reconciliation_failure = (
            rng.random()
            < (
                0.02
                + fragility
                * 0.06
            )
        )

        # ==========================================
        # Medication burden pressure
        # ==========================================

        medication_burden_score = (
            len(selected_meds)
        )

        medication_burden_score += int(
            treatment_masking_risk
            * 5
        )

        medication_burden_score += int(
            rebound_instability_risk
            * 4
        )

        if secondary_deterioration_triggered:
            medication_burden_score += 4

        # ==========================================
        # Therapeutic intelligence pressure
        # ==========================================

        therapeutic_system_pressure_score = (
            _clip(
                (
                    treatment_masking_risk
                    * 0.30
                )
                + (
                    rebound_instability_risk
                    * 0.25
                )
                + (
                    hidden_instability
                    * 0.20
                )
                + (
                    secondary_deterioration_triggered
                    * 0.15
                )
                + (
                    overload_pressure
                    * 0.10
                ),
                0,
                1,
            )
        )

        medication_rows.append(
            {
                **row.to_dict(),

                "medications_administered":
                    selected_meds,

                "medication_classes":
                    medication_classes,

                "medication_count":
                    len(selected_meds),

                "treatment_masking_risk":
                    round(
                        treatment_masking_risk,
                        4,
                    ),

                "therapy_dependency_state":
                    therapy_dependency_state,

                "secondary_deterioration_triggered":
                    secondary_deterioration_triggered,

                "secondary_deterioration_type":
                    secondary_deterioration_type,

                "stabilization_durability_score":
                    round(
                        stabilization_durability_score,
                        4,
                    ),

                "rebound_instability_risk":
                    round(
                        rebound_instability_risk,
                        4,
                    ),

                "therapeutic_trajectory_state":
                    therapeutic_trajectory_state,

                "pharmacy_overload_delay":
                    pharmacy_overload_delay,

                "medication_charting_lag":
                    medication_charting_lag,

                "missed_dose_window":
                    missed_dose_window,

                "duplicate_administration":
                    duplicate_administration,

                "interrupted_therapy":
                    interrupted_therapy,

                "medication_reconciliation_failure":
                    medication_reconciliation_failure,

                "medication_burden_score":
                    medication_burden_score,

                "therapeutic_system_pressure_score":
                    round(
                        therapeutic_system_pressure_score,
                        4,
                    ),

                "therapeutic_attention_required_flag":
                    (
                        therapeutic_system_pressure_score
                        >= 0.45
                    )
                    or secondary_deterioration_triggered,
            }
        )

    medication_df = pd.DataFrame(
        medication_rows
    )

    return medication_df
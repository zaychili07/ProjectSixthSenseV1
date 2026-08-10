"""
BIRE OS Synthetic Healthcare World Orchestrator

Chapter 53 doctrine:
No more babying BIRE OS.

Purpose:
Coordinate the complete Chapter 53 synthetic healthcare ecosystem.

This module transforms independent subsystem generators into:
- one connected synthetic world
- one operational ecosystem
- one longitudinal proving ground for BIRE OS

The orchestrator coordinates:
- patient generation
- encounters
- lifecycle progression
- context generation
- labs
- imaging
- medications
- diagnoses
- interventions
- handoffs
- operations
- outcomes
- audits
- exports
- vitals

This is the ecosystem conductor.

We Detect What Others Miss.
"""

from __future__ import annotations

from bire.synthetic import (
    patients,
    encounters,
    lifecycle,
    context,
    labs,
    imaging,
    medications,
    diagnoses,
    interventions,
    handoffs,
    operations,
    outcomes,
    audits,
    exports,
    vitals
)



def build_synthetic_healthcare_world(
    n_patients=30000,
    random_seed=42,
    export_world=True,
    export_dir="outputs/chapter_53_exports",
    save_parquet=False,
):
    """
    Build the full Chapter 53 synthetic healthcare ecosystem.
    """

    # =====================================================
    # Patient master
    # =====================================================

    patient_master_df = patients.generate_patient_master_table(
        n_patients=n_patients,
        random_seed=random_seed,
    )

    # =====================================================
    # Encounters
    # =====================================================

    encounter_df = encounters.generate_longitudinal_encounters(
        patient_df=patient_master_df,
        random_seed=random_seed,
)

    # =====================================================
    # Lifecycle
    # =====================================================

    encounter_lifecycle_df = lifecycle.assign_operational_phases(
        encounter_df,
        random_seed=random_seed,
)
    # =====================================================
    # Context
    # =====================================================

    context_df = context.generate_contextual_operational_data(
        lifecycle_df=encounter_lifecycle_df,
        random_seed=random_seed,
)

    # =====================================================
    # Vitals
    #====================================================

    vitals_df = vitals.generate_vitals_with_hidden_signals(
        df=context_df,
        patient_col="patient_id",
        time_col="encounter_start",
        random_seed=random_seed,
        include_temporal=True,
)

    # =====================================================
    # Labs
    # =====================================================

    labs_df = labs.generate_numeric_labs(
        df=vitals_df,
        random_seed=random_seed,
)

    # =====================================================
    # Imaging
    # =====================================================

    imaging_df = imaging.generate_imaging_evidence(
        df=labs_df,
        random_seed=random_seed,
)


    # =====================================================
    # Medications
    # =====================================================

    medication_df = medications.generate_therapeutic_intelligence(
        df=imaging_df,
        random_seed=random_seed,
)

    # =====================================================
    # Diagnoses
    # =====================================================

    diagnosis_df = diagnoses.generate_diagnosis_timeline(
        df=medication_df,
        random_seed=random_seed,
)

    # =====================================================
    # Interventions
    # =====================================================

    intervention_df = interventions.apply_intervention_effects(
        df=diagnosis_df,
        random_seed=random_seed,
)

    # =====================================================
    # Handoffs
    # =====================================================

    handoff_df = handoffs.generate_handoff_degradation(
        df=intervention_df,
        random_seed=random_seed,
)

    # =====================================================
    # Operations
    # =====================================================

    operations_df = operations.generate_operational_pressure(
        df=handoff_df,
        random_seed=random_seed,
)

    # =====================================================
    # Outcomes
    # =====================================================

    outcomes_df = outcomes.generate_longitudinal_outcomes(
        df=operations_df,
        random_seed=random_seed,
)

    # =====================================================
    # Audits
    # =====================================================

    ecosystem_outputs = audits.generate_ecosystem_audit(
        outcomes_df
    )

    # =====================================================
    # Exports
    # =====================================================

    export_summary = None

    if export_world:

        export_summary = exports.export_synthetic_ecosystem(
            patient_master_df=patient_master_df,
            encounter_df=encounter_df,
            encounter_lifecycle_df=encounter_lifecycle_df,
            context_df=context_df,
            labs_df=labs_df,
            imaging_df=imaging_df,
            medication_df=medication_df,
            diagnosis_df=diagnosis_df,
            intervention_df=intervention_df,
            handoff_df=handoff_df,
            operations_df=operations_df,
            outcomes_df=outcomes_df,
            ecosystem_outputs=ecosystem_outputs,
            output_dir=export_dir,
            save_parquet=save_parquet,
        )

    # =====================================================
    # Return ecosystem
    # =====================================================

    return {
        "patient_master_df": patient_master_df,
        "encounter_df": encounter_df,
        "encounter_lifecycle_df": encounter_lifecycle_df,
        "context_df": context_df,
        "vitals_df": vitals_df,
        "labs_df": labs_df,
        "imaging_df": imaging_df,
        "medication_df": medication_df,
        "diagnosis_df": diagnosis_df,
        "intervention_df": intervention_df,
        "handoff_df": handoff_df,
        "operations_df": operations_df,
        "outcomes_df": outcomes_df,
        "ecosystem_outputs": ecosystem_outputs,
        "export_summary": export_summary,
    }
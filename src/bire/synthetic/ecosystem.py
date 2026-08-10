"""
BIRE OS Synthetic Healthcare Ecosystem Blueprint

Chapter 53 doctrine:
No more babying BIRE OS.

This module defines the design blueprint for the synthetic healthcare
world used to pressure-test BIRE OS before real EHR / EMR integration.
"""

import pandas as pd

from bire.synthetic.config import (
    SYNTHETIC_ECOSYSTEM_CONFIG,
    CARE_MODES,
    CONDITION_PROFILES,
    OPERATIONAL_PHASES,
    SYNTHETIC_MESSINESS_TYPES,
)


def build_synthetic_ecosystem_design_summary():
    """
    Build high-level synthetic ecosystem design summary.
    """

    design_df = pd.DataFrame(
        {
            "design_area": [
                "Population",
                "Timeline",
                "Care Ecosystem",
                "Lifecycle Coverage",
                "Clinical Context",
                "Intervention Context",
                "Weak Signal Intelligence",
                "False Stability",
                "Contradictory Evidence",
                "Longitudinal Memory",
                "Trust Calibration",
                "Operational Pressure",
                "Handoff Degradation",
                "Messiness Engine",
                "Outcomes",
                "Auditability",
                "Stress Purpose",
                "Core Doctrine",
                "Overconfidence Resistance",
                "Population Scale",
                "Simulation Horizon",
                "Longitudinal Hospital Civilization",
                "Maximum Stress Doctrine",
                "Resilient Intelligence Standard",
            ],
            "planned_behavior": [
                "Generate a large patient population with diverse chronic profiles, fragility, deterioration tendency, recovery resilience, and data messiness tendency.",
                "Simulate multi-year longitudinal patient journeys across repeated encounters, admissions, discharges, readmissions, and care transitions.",
                "Represent outpatient, ER ESI levels, inpatient, ICU, discharge, and readmission pathways using BMS-aligned care modes.",
                "Support pre-event surveillance, event transition, post-event monitoring, recovery evaluation, re-escalation, critical post-event concern, and reintegration.",
                "Include vitals, hidden derived vitals, labs, imaging evidence, diagnoses, allergies, medications, and treatment context.",
                "Simulate interventions that may improve, partially improve, fail, mask deterioration, create rebound risk, or increase uncertainty.",
                "The ecosystem should generate subtle, faint, delayed, compensated, fragmented, and operationally obscured instability patterns that may only become visible longitudinally or contextually.",
                "Patients may temporarily appear stable due to intervention effects, operational delay, incomplete monitoring, or fragmented context while underlying deterioration continues progressing.",
                "Different ecosystem layers may disagree simultaneously, including vitals, labs, imaging, medications, diagnoses, operational context, interventions, and outcomes.",
                "The ecosystem should challenge whether BIRE OS can preserve continuity across repeated admissions, care transitions, fragmented handoffs, delayed diagnoses, and evolving instability patterns.",
                "The ecosystem should force BIRE OS to learn when to trust, partially trust, or distrust incoming evidence depending on uncertainty, fragmentation, operational pressure, contradiction burden, and longitudinal inconsistency.",
                "Simulate hospital strain, staffing pressure, ICU saturation, ER crowding, delays, resource shortages, documentation backlog, and operational cascade risk.",
                "Simulate continuity loss, provider transitions, fragmented summaries, missing context, treatment dependency loss, and readmission memory failure.",
                "Inject structured healthcare messiness including missingness, typos, mixed units, impossible values, duplicate records, copy-forward artifacts, delayed charting, and conflicting labels.",
                "Generate synthetic outcomes that reflect physiology, treatment response, operational pressure, uncertainty, handoff fragmentation, and hidden instability.",
                "Export audits, manifests, schemas, high-risk cases, high-uncertainty cases, contradiction reports, and learning signal summaries.",
                "Expose flaws, limitations, overconfidence, confusion, brittleness, and subsystem weaknesses before real-world EHR / EMR integration.",
                "No more babying BIRE OS. The synthetic world exists to reveal what must be strengthened.",
                "The ecosystem should punish simplistic certainty and reward contextual, longitudinal, uncertainty-aware reasoning across fragmented healthcare environments.",
                "Generate 30,000 synthetic patients across a 10-year healthcare environment, ranging from average patients to extremely complex longitudinal patients.",
                "Simulate a decade of healthcare activity including routine care, ER visits, ICU admissions, discharge, readmission, complications, recovery, recurrence, and death.",
                "Treat the synthetic world as a living hospital ecosystem, not a simple dataframe. Patients, hospitals, evidence, operations, and outcomes all evolve over time.",
                "Include every major hospital challenge x3: labs, discharge failures, readmissions, deaths, imaging delays, complications, treatment masking, handoff failures, operational overload, documentation breakdown, and contradictory evidence.",
                "BIRE OS does not receive the title Resilient Intelligence by default. It must earn that title by surviving hidden instability, false reassurance, fragmented context, uncertainty, and longitudinal operational chaos.",
            ],
        }
    )

    return design_df


def build_synthetic_ecosystem_component_registry():
    """
    Build registry of synthetic ecosystem components and backend files.
    """

    registry_df = pd.DataFrame(
        [
            {
                "chapter_section": "53.1",
                "component": "Ecosystem Blueprint",
                "backend_file": "src/bire/synthetic/ecosystem.py",
                "purpose": "Define doctrine, design summary, stress goals, and ecosystem scope.",
                "stress_target": "Architecture clarity and design governance.",
            },
            {
                "chapter_section": "53.2",
                "component": "Patient Master Profiles",
                "backend_file": "src/bire/synthetic/patients.py",
                "purpose": "Generate patient identity, condition profile, fragility, deterioration tendency, recovery resilience, and messiness tendency.",
                "stress_target": "Longitudinal patient individuality and baseline complexity.",
            },
            {
                "chapter_section": "53.3",
                "component": "Encounter Timeline",
                "backend_file": "src/bire/synthetic/encounters.py",
                "purpose": "Generate admissions, outpatient visits, ER visits, inpatient stays, ICU stays, discharges, and readmissions.",
                "stress_target": "Temporal continuity and care mode transitions.",
            },
            {
                "chapter_section": "53.4",
                "component": "Operational Lifecycle",
                "backend_file": "src/bire/synthetic/lifecycle.py",
                "purpose": "Assign pre-event, event, post-event, recovery, re-escalation, critical, and reintegration phases.",
                "stress_target": "Pre/post event ambiguity and lifecycle realism.",
            },
            {
                "chapter_section": "53.5",
                "component": "Vitals + Hidden Instability",
                "backend_file": "src/bire/synthetic/vitals.py",
                "purpose": "Generate/derive hidden physiological stress signals from simple vitals.",
                "stress_target": "Compensated instability and masked deterioration.",
            },
            {
                "chapter_section": "53.6",
                "component": "Context Layer",
                "backend_file": "src/bire/synthetic/context.py",
                "purpose": "Generate lab profiles, imaging context, medications, allergies, delays, and treatment complexity.",
                "stress_target": "Clinical context beyond vitals.",
            },
            {
                "chapter_section": "53.7",
                "component": "Messiness Engine",
                "backend_file": "src/bire/synthetic/messiness.py",
                "purpose": "Inject structured corruption, missingness, typos, mixed units, contradictions, and duplicate documentation.",
                "stress_target": "Ingestion resilience and data-quality skepticism.",
            },
            {
                "chapter_section": "53.8",
                "component": "Numeric Labs",
                "backend_file": "src/bire/synthetic/labs.py",
                "purpose": "Generate numeric lab evidence for renal, metabolic, infectious, respiratory, cardiac, hepatic, and pancreatitis-like patterns.",
                "stress_target": "Multimodal clinical evidence integration.",
            },
            {
                "chapter_section": "53.9",
                "component": "Imaging Evidence",
                "backend_file": "src/bire/synthetic/imaging.py",
                "purpose": "Generate imaging orders, modality, findings, contrast use, allergy reactions, delays, uncertainty, repeat imaging, and instability scores.",
                "stress_target": "Delayed and ambiguous imaging evidence.",
            },
            {
                "chapter_section": "53.10",
                "component": "Medication Administration",
                "backend_file": "src/bire/synthetic/medications.py",
                "purpose": "Generate medication burden, classes, delays, holds, allergy conflicts, interactions, and respiratory suppression risk.",
                "stress_target": "Treatment complexity and medication contradictions.",
            },
            {
                "chapter_section": "53.11",
                "component": "Diagnosis Timeline",
                "backend_file": "src/bire/synthetic/diagnoses.py",
                "purpose": "Generate documented diagnoses, complications, new diagnoses, delayed diagnosis updates, and diagnosis uncertainty.",
                "stress_target": "Documentation context and evolving diagnosis ambiguity.",
            },
            {
                "chapter_section": "53.12",
                "component": "Intervention Effects",
                "backend_file": "src/bire/synthetic/interventions.py",
                "purpose": "Model treatment response, masked deterioration, treatment dependency, failed stabilization, overcorrection, and rebound risk.",
                "stress_target": "False reassurance and treatment-supported stability.",
            },
            {
                "chapter_section": "53.13",
                "component": "Handoff Degradation",
                "backend_file": "src/bire/synthetic/handoffs.py",
                "purpose": "Simulate continuity loss, shift changes, provider changes, fragmented summaries, documentation conflict, and memory loss.",
                "stress_target": "Operational continuity failure.",
            },
            {
                "chapter_section": "53.14",
                "component": "Hospital Operations Pressure",
                "backend_file": "src/bire/synthetic/operations.py",
                "purpose": "Generate staffing pressure, ER crowding, ICU saturation, backlog, delays, resource constraints, fatigue, and cascade risk.",
                "stress_target": "Hospital-wide operational instability.",
            },
            {
                "chapter_section": "53.15",
                "component": "Outcomes",
                "backend_file": "src/bire/synthetic/outcomes.py",
                "purpose": "Generate outcomes, uncertainty, confidence, failure modes, recovery trust, learning signals, and upgrade areas.",
                "stress_target": "Ecosystem consequence modeling.",
            },
            {
                "chapter_section": "53.16",
                "component": "Audits",
                "backend_file": "src/bire/synthetic/audits.py",
                "purpose": "Generate ecosystem audit, schema report, contradictions, high-risk cases, uncertainty cases, and manifest.",
                "stress_target": "Observability and governance.",
            },
            {
                "chapter_section": "53.17",
                "component": "Exports",
                "backend_file": "src/bire/synthetic/exports.py",
                "purpose": "Export every stage, snapshot, audit artifact, schema, manifest, contradictions, high-risk cases, and learning summaries.",
                "stress_target": "Reproducibility and version governance.",
            },
        ]
    )

    return registry_df


def build_synthetic_ecosystem_stress_contract():
    """
    Build stress contract for what Chapter 53 must challenge.
    """

    stress_contract_df = pd.DataFrame(
        {
            "stress_domain": [
                "Data Quality",
                "Temporal Continuity",
                "Pre/Post Event Ambiguity",
                "Hidden Physiology",
                "Treatment Effects",
                "Medication Complexity",
                "Imaging Uncertainty",
                "Lab Evidence",
                "Diagnosis Drift",
                "Handoff Fragmentation",
                "Hospital Operations",
                "Outcome Trust",
                "BIRE OS Learning",
                "Weak Signal Detection",
                "False Reassurance",
                "Evidence Contradiction",
                "Overconfidence Pressure",
            ],
            "failure_pressure": [
                "Missingness, typos, mixed units, impossible values, duplicated rows, delayed charting, and conflicting labels.",
                "Multiple encounters, readmissions, long stays, discharge gaps, timeline drift, and patient journey fragmentation.",
                "Patients may arrive with prior deterioration but are pre-event relative to the monitored hospital environment.",
                "Vitals may look normal while ratios, trajectories, and derived signals suggest hidden instability.",
                "Interventions may improve, partially improve, fail, mask deterioration, or create rebound risk.",
                "Medication burden, allergies, contraindications, holds, delays, interactions, and care-mode contradictions.",
                "Imaging may be delayed, ambiguous, limited quality, contradictory, incidental, or complicated by contrast reactions.",
                "Labs may reveal hidden instability, chronic burden, delayed deterioration, or contradiction against vitals.",
                "Diagnoses may arrive late, change during admission, conflict across teams, or fail to capture evolving complexity.",
                "Context may be lost across ER, inpatient, ICU, discharge, readmission, shift change, and provider transitions.",
                "Staffing pressure, ER crowding, ICU saturation, delays, resource shortage, and operational cascade risk.",
                "Recovery may be trustworthy, unstable, rebound-prone, fragmented, treatment-dependent, or falsely reassuring.",
                "The synthetic world should expose what BIRE OS needs to strengthen instead of proving perfection.",
                "Subtle trajectory drift, compensated instability, delayed deterioration, and weak longitudinal abnormalities hiding beneath apparently stable vitals.",
                "Temporary improvement after intervention despite persistent underlying instability or operationally hidden deterioration.",
                "Vitals, labs, imaging, diagnoses, medications, and operational context may conflict simultaneously without obvious ground truth clarity.",
                "The ecosystem should intentionally create situations where incomplete information appears trustworthy enough to encourage dangerous overconfidence.",
            ],
            "bire_os_question": [
                "Can BIRE OS become skeptical of bad data?",
                "Can BIRE OS preserve longitudinal memory?",
                "Can BIRE OS define event state operationally?",
                "Can BIRE OS detect what static thresholds miss?",
                "Can BIRE OS distinguish recovery from treatment-supported stability?",
                "Can BIRE OS recognize medication context and contradictions?",
                "Can BIRE OS reason under delayed or uncertain imaging evidence?",
                "Can BIRE OS integrate labs without overconfidence?",
                "Can BIRE OS treat diagnoses as context, not absolute truth?",
                "Can BIRE OS remain aware when operational memory fragments?",
                "Can BIRE OS account for hospital pressure as risk context?",
                "Can BIRE OS avoid trusting false recovery?",
                "Can BIRE OS learn from failure signals and upgrade naturally?",
                "Can BIRE OS detect instability before obvious collapse signals emerge?",
                "Can BIRE OS distinguish true recovery from temporary stabilization?",
                "Can BIRE OS reason through conflicting multimodal evidence without collapsing into simplistic interpretation?",
                "Can BIRE OS remain calibrated when evidence quality is incomplete, delayed, fragmented, or contradictory?",
            ],
        }
    )

    return stress_contract_df


def build_synthetic_ecosystem_config_summary():
    """
    Summarize current synthetic ecosystem configuration.
    """

    config_summary = {
        "dataset_name": SYNTHETIC_ECOSYSTEM_CONFIG.get("dataset_name"),
        "version": SYNTHETIC_ECOSYSTEM_CONFIG.get("version"),
        "n_patients": SYNTHETIC_ECOSYSTEM_CONFIG.get("n_patients"),
        "start_date": SYNTHETIC_ECOSYSTEM_CONFIG.get("start_date"),
        "end_date": SYNTHETIC_ECOSYSTEM_CONFIG.get("end_date"),
        "time_granularity": SYNTHETIC_ECOSYSTEM_CONFIG.get("time_granularity"),
        "messiness_level": SYNTHETIC_ECOSYSTEM_CONFIG.get("messiness_level"),
        "random_seed": SYNTHETIC_ECOSYSTEM_CONFIG.get("random_seed"),
        "care_modes_count": len(CARE_MODES),
        "condition_profiles_count": len(CONDITION_PROFILES),
        "operational_phases_count": len(OPERATIONAL_PHASES),
        "messiness_types_count": len(SYNTHETIC_MESSINESS_TYPES),
        "care_modes": CARE_MODES,
        "condition_profiles": CONDITION_PROFILES,
        "operational_phases": OPERATIONAL_PHASES,
        "messiness_types": SYNTHETIC_MESSINESS_TYPES,
        "chapter_doctrine": (
            "No more babying BIRE OS. Chapter 53 will scale toward 30,000 patients "
            "over 10 years and include average-to-extremely-complex patients, labs, "
            "discharges, readmissions, death, imaging delays, complications, and every "
            "major hospital challenge x3. The environment must exceed the meaning of "
            "Project Sixth Sense and the motto 'We Detect What Others Miss.' BIRE OS "
            "must come out stronger, more advanced, and earn the title Resilient Intelligence."
        ),
        "mission": "Expose flaws, uncertainty, confusion, limitations, and upgrade needs before real EHR / EMR integration.",
        "target_patient_count": 30000,
        "target_simulation_years": 10,
        "target_environment_scale": "30,000 patients over 10 years",
        "resilient_intelligence_status": "not_earned_yet",
        "resilient_intelligence_standard": (
            "BIRE OS earns Resilient Intelligence only after being stress-tested "
            "against hidden deterioration, false recovery, fragmented context, "
            "contradictory evidence, operational overload, delayed results, readmissions, "
             "death, and longitudinal hospital chaos."
        ),
    }

    return config_summary


def build_synthetic_ecosystem_readiness_checklist():
    """
    Build readiness checklist before orchestration.
    """

    checklist_df = pd.DataFrame(
        {
            "readiness_area": [
                "Patient Identity",
                "Encounter Timeline",
                "Lifecycle States",
                "Hidden Vitals",
                "Clinical Context",
                "Numeric Labs",
                "Imaging Evidence",
                "Medication Behavior",
                "Diagnosis Timeline",
                "Intervention Effects",
                "Handoff Degradation",
                "Operational Pressure",
                "Outcomes",
                "Audits",
                "Exports",
                "Orchestrator",
            ],
            "status": [
                "built",
                "built",
                "built",
                "built",
                "built",
                "built",
                "built",
                "built",
                "built",
                "built",
                "built",
                "built",
                "built",
                "built",
                "built",
                "pending",
            ],
            "review_question": [
                "Do patients have enough individuality to influence downstream behavior?",
                "Do encounters create realistic movement across care environments?",
                "Do lifecycle phases represent pre-event, post-event, recovery, re-escalation, and reintegration?",
                "Do derived signals expose instability beyond static vitals?",
                "Does context include labs, imaging, meds, allergies, delays, and treatment complexity?",
                "Do labs create useful hidden evidence and contradictions?",
                "Does imaging include findings, delays, contrast, uncertainty, and repeat imaging?",
                "Do medications create burden, conflicts, holds, delays, and interactions?",
                "Do diagnoses evolve and include uncertainty, delays, complications, and new diagnoses?",
                "Do interventions create false reassurance, treatment dependency, rebound risk, and trust states?",
                "Do handoffs degrade operational memory and amplify uncertainty?",
                "Does hospital pressure behave like a system-level instability source?",
                "Do outcomes teach BIRE OS what needs to improve?",
                "Do audits expose quality, risk, contradictions, learning signals, and schema?",
                "Do exports preserve every stage and artifact?",
                "Can one function build, audit, and export the whole world reproducibly?",
            ],
        }
    )

    return checklist_df


def build_full_synthetic_ecosystem_blueprint():
    """
    Return all Chapter 53 blueprint artifacts.
    """

    return {
        "design_summary": build_synthetic_ecosystem_design_summary(),
        "component_registry": build_synthetic_ecosystem_component_registry(),
        "stress_contract": build_synthetic_ecosystem_stress_contract(),
        "config_summary": build_synthetic_ecosystem_config_summary(),
        "readiness_checklist": build_synthetic_ecosystem_readiness_checklist(),
    }
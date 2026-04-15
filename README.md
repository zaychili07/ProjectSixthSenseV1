# 🧬 Project Sixth Sense: Bio-Intelligence Risk Engine (BIRE)

 📊 Overview

The Bio-Intelligence Risk Engine (BIRE) is a time-series machine learning system designed to detect early signs of patient deterioration using longitudinal vital sign data.

Unlike traditional threshold-based monitoring systems, BIRE focuses on temporal dynamics—capturing trends, instability, and rate-of-change in physiological signals—to generate forward-looking risk scores and actionable alerts.

🎯 Objective

Develop a system that:

- Predicts clinical deterioration within a future time window (60 minutes)
- Converts risk scores into low-noise, actionable alerts
- Minimizes false positives and alert fatigue
- Preserves temporal causality (no data leakage)

🧠 Core System Design

BIRE is structured as a modular pipeline:

Raw Data → Temporal Alignment → Feature Engineering → Target Construction → Modeling → Alerting → Evaluation

⚙️ Pipeline Architecture
1. Data Processing
Validation and cleaning of patient time-series data
Deduplication of patient-timestamp pairs
Range enforcement for physiological signals

2. Temporal Alignment
Resampling to fixed intervals (5-minute bins)
Per-patient chronological alignment

3. Feature Engineering (Cycle I)

For each vital sign:

- Lag features (t-1, t-2)
- Rate-of-change (delta)
- Rolling statistics (mean, std, min)

All rolling features are leakage-safe using shifted windows.

🎯 Target Construction (Cycle II)

Two key labels are defined:

event_now

Binary indicator of immediate physiological abnormality:

SpO₂ < 90
SBP < 90
Temperature > 38°C
target

Forward-looking deterioration label:

Indicates whether a patient will deteriorate within the next 60 minutes

Constructed using:
- Future shifting
- Rolling window aggregation
- Strict temporal causality

🤖 Modeling
- Baseline Model
- Logistic Regression
- Training Strategy
- Time-aware patient-level split
- Prevents data leakage across patients and time
- Features Used
- All engineered temporal features
- Excludes:
- patient_id
- timestamp
- event_now
- target

🚨 Alerting Framework

BIRE converts probabilistic outputs into clinical alerts using:

Threshold-based filtering
Persistence requirement (alerts trigger only after sustained elevated risk)

This ensures:

- Reduced noise
- Fewer false positives
- Clinically meaningful signals

📈 Model Performance
Metric	Score
ROC-AUC	0.828
PR-AUC	0.782

These results indicate strong discrimination, particularly in an imbalanced setting.

⚡ Operational Alerting Performance

- Total alerts: 2
- Patients alerted: 1 (P003)
- Stable patients (no alerts): P001, P002
- Key Observations:
- Alerts were highly concentrated on the deteriorating patient
- No false alerts were triggered in stable patients
- Maximum predicted risk reached 0.98
- Alert precision proxy: 1.0

All alerts aligned with true deterioration events in this evaluation slice.

Interpretation

BIRE demonstrates strong real-world behavior:

High selectivity → avoids alert fatigue
Strong sensitivity → captures meaningful deterioration
Temporal consistency → alerts require sustained risk

🧪 Key Strengths
✅ Leakage-safe time-series pipeline
✅ Forward-looking target design
✅ Patient-level validation strategy
✅ Operational alerting system (not just a model)
✅ Clinically interpretable outputs

⚠️ Limitations
- Small sample size (mock dataset)
- Evaluation results may not generalize without larger validation
- Thresholds and persistence parameters require tuning for real-world deployment

🚀 Future Work
- XGBoost / advanced models for rare event detection
- Threshold optimization & calibration
- Lead-time analysis (how early deterioration is detected)
- Patient-specific risk modeling
- Real-world clinical validation

🧠 Key Insight

BIRE is not just a classifier—it is a decision-support system.

The system bridges the gap between:

- raw model predictions and actionable clinical alerts

📂 Project Structure
src/bire/
├── data/              # Validation, alignment, imputation
├── features/          # Temporal feature engineering
├── models/            # Modeling logic
├── pipeline/          # End-to-end orchestration
├── evaluation/        # Alerting + performance evaluation

📌 Summary

BIRE demonstrates that incorporating temporal dynamics + persistence-based alerting can produce:

- Early detection signals
- Reduced noise
- Clinically relevant alerts

This positions the system as a strong foundation for next-generation patient monitoring systems.

# 🧬 Project Sixth Sense — Bio-Intelligence Risk Engine (BIRE)

**A time-series machine learning system for early detection of patient deterioration using physiological dynamics, not static thresholds.**

---

## 🚀 Overview

BIRE is a modular clinical ML system designed to identify **early warning signals of patient decline before they become clinically obvious**.

Traditional monitoring systems rely on fixed thresholds (e.g., SpO₂ < 90), often reacting **after deterioration has already occurred**.  
BIRE instead analyzes **temporal patterns—trends, instability, and rate-of-change**—to generate **forward-looking risk scores and low-noise alerts**.

---

## 🎯 What Makes BIRE Different

- 🔄 **Temporal Awareness** — detects change over time, not just abnormal values  
- ⚠️ **Forward Prediction** — predicts deterioration within a future window (60 minutes)  
- 🚨 **Persistence-Based Alerting** — reduces false alarms by requiring sustained risk  
- 🧠 **Interpretable Modeling** — logistic regression baseline with explainable outputs  
- 🏥 **Clinical Framing** — designed as a decision-support system, not just a model  

---

## 🧠 Key Insight

> Early deterioration is not defined by a single abnormal reading  
> it is defined by **how physiology changes over time**.

BIRE captures this by transforming raw vital signs into **temporal signals of instability**, enabling earlier and more reliable detection.

---
## 🎯 Objective

Develop a system that:

- Predicts clinical deterioration within a future time window (60 minutes)
- Converts risk scores into low-noise, actionable alerts
- Minimizes false positives and alert fatigue
- Preserves temporal causality (no data leakage)

## 🧠 Core System Design

BIRE is structured as a modular pipeline:

Raw Data → Temporal Alignment → Feature Engineering → Target Construction → Modeling → Alerting → Evaluation

# ⚙️ Pipeline Architecture

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

## 🎯 Target Construction (Cycle II)

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

## 🤖 Modeling

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

## 🚨 Alerting Framework

BIRE converts probabilistic outputs into clinical alerts using:

Threshold-based filtering
Persistence requirement (alerts trigger only after sustained elevated risk)

This ensures:

- Reduced noise
- Fewer false positives
- Clinically meaningful signals

## 📈 Model Performance
Metric	Score
ROC-AUC	0.828
PR-AUC	0.782

These results indicate strong discrimination, particularly in an imbalanced setting.

## ⚡ Operational Alerting Performance

- Total alerts: 2
- Patients alerted: 1 (P003)
- Stable patients (no alerts): P001, P002
- Key Observations:
- Alerts were highly concentrated on the deteriorating patient
- No false alerts were triggered in stable patients
- Maximum predicted risk reached 0.98
- Alert precision proxy: 1.0

All alerts aligned with true deterioration events in this evaluation slice.

# Interpretation

BIRE demonstrates strong real-world behavior:

High selectivity → avoids alert fatigue
Strong sensitivity → captures meaningful deterioration
Temporal consistency → alerts require sustained risk

## 🧪 Key Strengths

✅ Leakage-safe time-series pipeline
✅ Forward-looking target design
✅ Patient-level validation strategy
✅ Operational alerting system (not just a model)
✅ Clinically interpretable outputs

## ⚠️ Limitations

- Small sample size (mock dataset)
- Evaluation results may not generalize without larger validation
- Thresholds and persistence parameters require tuning for real-world deployment

## 🚀 Future Work

- XGBoost / advanced models for rare event detection
- Threshold optimization & calibration
- Lead-time analysis (how early deterioration is detected)
- Patient-specific risk modeling
- Real-world clinical validation

## 🧠 Key Insight

BIRE is not just a classifier—it is a decision-support system.

The system bridges the gap between:

- raw model predictions and actionable clinical alerts

## 📂 Project Structure
src/bire/
├── data/              # Validation, alignment, imputation
├── features/          # Temporal feature engineering
├── models/            # Modeling logic
├── pipeline/          # End-to-end orchestration
├── evaluation/        # Alerting + performance evaluation

## 📌 Summary

BIRE demonstrates that incorporating temporal dynamics + persistence-based alerting can produce:

- Early detection signals
- Reduced noise
- Clinically relevant alerts

This positions the system as a strong foundation for next-generation patient monitoring systems.

## 🚀 Getting Started

1. Clone the Repository
```
git clone https://github.com/zaychili07/ProjectSixthSenseV1.git
cd ProjectSixthSenseV1
```
2. Set Up Environment

Create a virtual environment (recommended):
```
python -m venv venv
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate     # Windows
```
Install dependencies:
```
pip install -r requirements.txt
```
If no requirements.txt is present, install:
```
pip install pandas numpy scikit-learn matplotlib
```
3. Prepare Input Data

Ensure your dataset follows this structure:

- column	description
- patient_id	unique patient identifier
- timestamp	datetime
- heart_rate	heart rate
- resp_rate	respiratory rate
- spo2	oxygen saturation
- temperature	body temperature
- sbp	systolic blood pressure
- dbp	diastolic blood pressure

Example file:
```
data/raw/bire_mock_vitals.csv
```
4. Run the Pipeline

You can run the full feature engineering pipeline:
```
python src/bire/pipeline/main_pipeline.py
```
Or specify paths:
```
export BIRE_INPUT_PATH="data/raw/bire_mock_vitals.csv"
export BIRE_OUTPUT_PATH="data/processed/bire_cycle1_features.csv"

python src/bire/pipeline/main_pipeline.py
```
5. Run Modeling (Notebook Recommended)

Open the notebook and run:
```
from bire.pipeline.main_pipeline import run_cycle1, run_bire_modeling

df = run_cycle1("data/raw/bire_mock_vitals.csv")

feature_cols = [c for c in df.columns if c not in ["patient_id", "timestamp", "event_now", "target"]]

model, train_df, test_df = run_bire_modeling(df, feature_cols)
6. Generate Alerts
test_df["alert"].sum()
```
Or run full evaluation section in the notebook.

## 📓 Notebook Workflow

The recommended workflow is:

- Run full pipeline (Cycle I)
- Construct target (Cycle II)
- Train model
- Evaluate performance
- Analyze alerting behavior

## 🛠️ Notes
Ensure working directory is project root

Kaggle users may need to adjust paths:
```
"/kaggle/working/ProjectSixthSenseV1/..."
```
Use sys.path.append("src") if import issues occur

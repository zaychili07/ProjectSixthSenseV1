# 🧬 Project Sixth Sense — Bio-Intelligence Risk Engine (BIRE)

AI-powered time-series system for early detection of patient deterioration using physiological signals and temporal modeling.

---

##  Overview

BIRE is a modular machine learning pipeline designed to detect **early warning signals in patient physiology before clinical deterioration becomes apparent**.

It shifts monitoring from:

**Reactive thresholds → Proactive intelligence**

---

##  Key Capabilities

- Time-aware validation (no data leakage)
- Temporal feature engineering (lags, rolling stats, deltas)
- Interpretable risk modeling (logistic regression baseline)
- Persistence-based alert logic (reduces false alarms)
- Patient-level risk trajectory analysis
- Modular, production-style pipeline

---

### Quick Start (Run the System)

### 1. Generate features from raw data

```python
from bire.pipeline.main_pipeline import run_cycle1

df = run_cycle1("/path/to/input.csv")

from bire.pipeline.main_pipeline import run_bire_modeling

feature_cols = [c for c in df.columns if c not in ["patient_id", "timestamp", "target"]]

model, train_df, test_df = run_bire_modeling(
    df,
    feature_cols,
    threshold=0.5,
    window=3,
)
```
### System Architechure

Raw Data → Validation → Temporal Alignment → Feature Engineering  
→ Model → Risk Scores → Alert Logic → Trajectory Analysis

### Key Findings
- Temporal validation exposed weaknesses in static threshold systems
- Persistence-based alerting significantly reduced noise
- Logistic regression produced more stable and interpretable signals than XGBoost
- Risk trajectories showed meaningful temporal structure across patients

### Project Structure
src/bire/
├── data/
├── features/
├── models/
├── evaluation/
├── pipeline/

### Feature Engineering Strategy

For each physiological signal:

Lag features (t-1, t-2)
Delta (rate of change)
Rolling mean, std, min, max

### All features:

- computed per patient
- strictly use past data
- prevent leakage via .shift(1)

### Data Leakage Prevention
- Patient-level grouping
- Strict timestamp ordering
- No future-aware imputation
- Rolling windows exclude current timestep
- Time-aware train/test split

### Modeling Approach
- Supervised Model
- Logistic Regression (baseline)
- Balanced class weighting

### Outputs
- pred_proba → risk score
- alert → persistence-based signal
### Evaluation Strategy
- Time-aware split (simulates real-world deployment)
- Patient-level separation

### Metrics:
- AUROC
- Precision / Recall
- Alert behavior analysis

### ⚠️ Limitations
- Small dataset
- Limited positive events
- Lead-time estimation constrained

### Next Steps
- Larger datasets
- Probability calibration
- Real-time deployment pipeline
- Multi-signal fusion

### 🔐 Responsible Use
- Research / educational purposes only
- No PII
- Not a clinical decision system

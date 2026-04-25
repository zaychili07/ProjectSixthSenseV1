🧬 Project Sixth Sense — Bio-Intelligence Risk Engine (BIRE)

"Because physiology changes before it fails."

A time-series machine learning system for early detection of patient deterioration
using temporal dynamics, not static thresholds.

🚀 Overview

BIRE is a clinical ML system designed to detect early warning signals of patient decline before they become clinically obvious.

Traditional monitoring systems rely on fixed thresholds (e.g., SpO₂ < 90), reacting after deterioration has already occurred.

BIRE instead analyzes:

trends
instability
rate-of-change

to generate forward-looking risk scores and low-noise alert episodes.

📊 Results Snapshot
Metric	Value
Event Detection Rate	96.69%
Median Lead Time	405 minutes
Mean Alert Burden	0.372 alerts/hr
Max Alert Burden	1.708 alerts/hr
📈 Event-Level Detection

⏱️ Lead-Time Distribution

🚨 Alert Burden

🔝 High-Burden Patients

🧠 Key Insight

Early deterioration is not defined by a single abnormal reading
but by how physiology changes over time.

BIRE captures this by transforming raw vitals into temporal signals of instability, enabling earlier and more reliable detection.

🎯 What Makes BIRE Different
🔄 Temporal Awareness — learns change over time, not just abnormal values
🔮 Forward Prediction — predicts deterioration within a 60-minute window
🚨 Persistence-Based Alerting — reduces false alarms via sustained risk
🧠 Interpretable Modeling — explainable baseline with logistic regression
🏥 Clinical Framing — built as a decision-support system
⚙️ Pipeline Architecture
Raw Data 
→ Temporal Alignment 
→ Feature Engineering 
→ Target Construction 
→ Modeling 
→ Alerting 
→ Evaluation
🧠 Core System Design
1. Data Processing
Validation & cleaning
Deduplication
Physiological range enforcement
2. Temporal Alignment
5-minute resampling
Per-patient chronological ordering
3. Feature Engineering

For each vital:

Lag features (t-1, t-2)
Rate-of-change (delta)
Rolling stats (mean, std, min)

All features are leakage-safe.

🎯 Target Construction

Two labels:

event_now

Immediate physiological abnormality:

SpO₂ < 90
SBP < 90
Temp > 38°C
target

Forward-looking deterioration within 60 minutes using:

future shift
rolling aggregation
strict temporal causality
🤖 Modeling
Logistic Regression baseline
Patient-level time-aware split
Leakage prevention across time + patients
🚨 Alerting Framework
Threshold-based filtering
Persistence requirement

This produces:

fewer false positives
clinically meaningful alert episodes
⚡ Operational Performance
Alerts triggered: 2
Patients alerted: 1
False alerts: 0
Observations:
Alerts concentrated on deteriorating patient
No noise in stable patients
Max risk reached 0.98
📊 Evaluation Philosophy

BIRE is evaluated beyond standard metrics:

✔️ Event detection (not just row classification)
✔️ Lead-time before deterioration
✔️ Alert burden (operational cost)
✔️ False alert rate

This makes it closer to a real clinical system.

🧪 Strengths
✅ Leakage-safe pipeline
✅ Temporal modeling
✅ Forward-looking targets
✅ Event-level evaluation
✅ Operational alerting system
⚠️ Limitations
Small mock dataset
Requires larger-scale validation
Threshold tuning needed for deployment
🚀 Future Work
XGBoost / advanced models
Threshold optimization
Lead-time refinement
Patient-specific modeling
Clinical validation
📂 Project Structure
src/bire/
├── data/
├── features/
├── models/
├── pipeline/
├── evaluation/
🚀 Getting Started
git clone https://github.com/zaychili07/ProjectSixthSenseV1.git
cd ProjectSixthSenseV1
Install dependencies
pip install pandas numpy scikit-learn matplotlib
Run pipeline
python src/bire/pipeline/main_pipeline.py

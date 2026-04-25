## 🧬 Project Sixth Sense — Bio-Intelligence Risk Engine (BIRE)

***"Because physiology changes before it fails."***

A **time-series machine learning system** for early detection of patient deterioration
using **temporal dynamics, not static thresholds**.

---

##  Overview

BIRE is a clinical ML system designed to detect **early warning signals of patient decline before they become clinically obvious**.

Traditional monitoring systems rely on fixed thresholds (e.g., SpO₂ < 90), reacting **after deterioration has already occurred**.

BIRE instead analyzes:

* trends
* instability
* rate-of-change

to generate **forward-looking risk scores and low-noise alert episodes**.

---

##  System Architecture

![System Diagram](outputs/system_diagram.png)

##  Results Snapshot

| Metric               |               Value |
| -------------------- | ------------------: |
| Event Detection Rate |          **96.69%** |
| Median Lead Time     |     **405 minutes** |
| Mean Alert Burden    | **0.372 alerts/hr** |
| Max Alert Burden     | **1.708 alerts/hr** |

---

##  Event-Level Detection

![Event Detection](outputs/figures/event_detection_summary.png)

---

##  Lead-Time Distribution

![Lead Time](outputs/figures/leadtime_distribution.png)

---

##  Alert Burden

![Alert Burden](outputs/figures/alert_burden_distribution.png)

---

##  High-Burden Patients

![Top Burden](outputs/figures/top_alert_burden_patients.png)

---

## 🏥 Why This Matters Clinically

In real clinical settings, deterioration rarely happens suddenly—it develops gradually through subtle physiological changes. Traditional monitoring systems rely on fixed thresholds (e.g., SpO₂ < 90), which often trigger **after a patient has already begun to decline**.

This leads to two major problems:

*  **Delayed intervention** — clinicians react after deterioration has already occurred
*  **Alert fatigue** — excessive false alarms reduce trust in monitoring systems

---

###  How BIRE Improves This

BIRE reframes monitoring from **reactive thresholds → proactive intelligence**.

Instead of asking:

> “Is this value abnormal right now?”

BIRE asks:

> “Is this patient *trending toward deterioration*?”

---

### ⏱️ Earlier Detection

By modeling **temporal dynamics (trends, instability, rate-of-change)**, BIRE identifies deterioration **before clinical thresholds are crossed**, providing meaningful lead time for intervention.

---

### 🚨 Reduced Alert Fatigue

BIRE uses **persistence-based alerting**, meaning alerts only trigger when elevated risk is sustained—not from isolated spikes.

This results in:

* fewer false positives
* more trustworthy alerts
* better clinician adoption

---

### ⚖️ Balanced Decision Support

BIRE explicitly evaluates:

* ✔️ Detection of true deterioration events
* ✔️ Lead time before events
* ✔️ Alert burden (alerts per patient-hour)
* ✔️ False alert rate

This ensures the system is not just accurate—but **operationally usable**.

---

###  Real-World Impact

If deployed in a clinical environment, a system like BIRE could:

* Enable **earlier interventions** (e.g., oxygen therapy, fluids, escalation of care)
* Reduce **ICU transfers and adverse events**
* Improve **workflow efficiency** by minimizing unnecessary alerts
* Provide clinicians with **interpretable, actionable signals**

---

###  Key Takeaway

BIRE shifts patient monitoring from:

> **“Detecting when a patient is already deteriorating”**

to

> **“Predicting when a patient is about to deteriorate—and acting in time.”**


##  Key Insight

> Early deterioration is not defined by a single abnormal reading
> but by **how physiology changes over time**.

BIRE captures this by transforming raw vitals into **temporal signals of instability**, enabling earlier and more reliable detection.

---

##  What Makes BIRE Different

*  **Temporal Awareness** — learns change over time, not just abnormal values
*  **Forward Prediction** — predicts deterioration within a 60-minute window
*  **Persistence-Based Alerting** — reduces false alarms via sustained risk
*  **Interpretable Modeling** — explainable baseline with logistic regression
*  **Clinical Framing** — built as a decision-support system

---

##  Pipeline Architecture

Raw Data → Temporal Alignment → Feature Engineering → Target Construction → Modeling → Alerting → Evaluation

---

##  Core System Design

### 1. Data Processing

* Validation & cleaning
* Deduplication
* Physiological range enforcement

### 2. Temporal Alignment

* 5-minute resampling
* Per-patient chronological ordering

### 3. Feature Engineering

For each vital:

* Lag features (t-1, t-2)
* Rate-of-change (delta)
* Rolling stats (mean, std, min)

All features are **leakage-safe**.

---

##  Target Construction

Two labels:

### `event_now`

Immediate physiological abnormality:

* SpO₂ < 90
* SBP < 90
* Temp > 38°C

### `target`

Forward-looking deterioration within 60 minutes using:

* future shift
* rolling aggregation
* strict temporal causality

---

##  Modeling

* Logistic Regression baseline
* Patient-level time-aware split
* Leakage prevention across time + patients

---

##  Alerting Framework

* Threshold-based filtering
* Persistence requirement

This produces:

* fewer false positives
* clinically meaningful alert episodes

---

##  Operational Performance

* Alerts triggered: **2**
* Patients alerted: **1**
* False alerts: **0**

### Observations:

* Alerts concentrated on deteriorating patient
* No noise in stable patients
* Max risk reached **0.98**

---

##  Evaluation Philosophy

BIRE is evaluated beyond standard metrics:

* ✔️ Event detection (not just row classification)
* ✔️ Lead-time before deterioration
* ✔️ Alert burden (operational cost)
* ✔️ False alert rate

This makes it closer to a **real clinical system**.

---

##  Strengths

* ✅ Leakage-safe pipeline
* ✅ Temporal modeling
* ✅ Forward-looking targets
* ✅ Event-level evaluation
* ✅ Operational alerting system

---

##  Limitations

* Small mock dataset
* Requires larger-scale validation
* Threshold tuning needed for deployment

---

##  Future Work

* XGBoost / advanced models
* Threshold optimization
* Lead-time refinement
* Patient-specific modeling
* Clinical validation

---

##  Project Structure

src/bire/
├── data/
├── features/
├── models/
├── pipeline/
├── evaluation/

---

## 🚀 Getting Started

```bash
git clone https://github.com/zaychili07/ProjectSixthSenseV1.git
cd ProjectSixthSenseV1
```

### Install dependencies

```bash
pip install pandas numpy scikit-learn matplotlib
```

### Run pipeline

```bash
python src/bire/pipeline/main_pipeline.py
```

---

##  Final Takeaway

BIRE is not just a classifier.

It is a **clinical decision-support system** that bridges:

> raw physiological data → actionable early warnings

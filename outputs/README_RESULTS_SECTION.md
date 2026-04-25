
## BIRE Evaluation Results

BIRE was evaluated as a forward-looking patient deterioration risk engine using time-series vital signs and episode-level alerting. The system does not simply flag isolated threshold crossings; it converts model risk scores into alert episodes and evaluates whether those alerts occur before deterioration events.

### Key Evaluation Metrics

| Metric | Value |
|---|---:|
| Total deterioration events | 181 |
| Events detected before deterioration | 175 |
| Event detection rate | 96.69% |
| Median first-alert lead time | 405.0 minutes |
| Mean alert burden | 0.372 alerts / patient-hour |
| Max alert burden | 1.708 alerts / patient-hour |

### Event-Level Detection

![Event Detection Summary](outputs/figures/event_detection_summary.png)

### Lead-Time Performance

![Lead-Time Distribution](outputs/figures/leadtime_distribution.png)

### Alert Burden

![Alert Burden Distribution](outputs/figures/alert_burden_distribution.png)

### Highest Alert-Burden Patients

![Top Alert Burden Patients](outputs/figures/top_alert_burden_patients.png)

### Interpretation

These results show that BIRE is moving beyond standard classification metrics into clinically relevant evaluation. The system is assessed not only by whether it predicts deterioration, but also by whether alerts occur before deterioration, how much warning time the system provides, and whether alert burden is operationally reasonable.

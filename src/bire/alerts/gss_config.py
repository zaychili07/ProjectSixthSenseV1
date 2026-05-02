# src/bire/alerts/gss_config.py

MODE_AWARE_GSS_POLICY = {
    "ICU": {
        "risk_threshold": 0.985,
        "cooldown_steps": 3,
        "escalation_delta": 0.03,
        "allow_post_event_escalation": True,
    },
    "ER_ESI_1": {
        "risk_threshold": 0.985,
        "cooldown_steps": 3,
        "escalation_delta": 0.035,
        "allow_post_event_escalation": True,
    },
    "ER_ESI_2": {
        "risk_threshold": 0.990,
        "cooldown_steps": 4,
        "escalation_delta": 0.04,
        "allow_post_event_escalation": True,
    },
    "ER_ESI_3": {
        "risk_threshold": 0.992,
        "cooldown_steps": 5,
        "escalation_delta": 0.05,
        "allow_post_event_escalation": True,
    },
    "ER_ESI_4": {
        "risk_threshold": 0.995,
        "cooldown_steps": 6,
        "escalation_delta": 0.06,
        "allow_post_event_escalation": False,
    },
    "ER_ESI_5": {
        "risk_threshold": 0.996,
        "cooldown_steps": 8,
        "escalation_delta": 0.07,
        "allow_post_event_escalation": False,
    },
    "Inpatient": {
        "risk_threshold": 0.990,
        "cooldown_steps": 5,
        "escalation_delta": 0.05,
        "allow_post_event_escalation": True,
    },
    "Outpatient": {
        "risk_threshold": 0.997,
        "cooldown_steps": 10,
        "escalation_delta": 0.08,
        "allow_post_event_escalation": False,
    },
}

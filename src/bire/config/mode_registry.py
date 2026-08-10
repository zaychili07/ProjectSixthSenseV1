# src/bire/config/mode_registry.py

from __future__ import annotations

import pandas as pd
## BMS ESI Consistency Note

## BMS ESI Consistency Note

#BMS uses ER ESI levels as acuity-aware operational modes.

#ESI 1 represents the highest acuity emergency condition, while ESI 5 represents the lowest acuity emergency condition.

#To preserve operational consistency:
#- ER_ESI_1 should remain the most sensitive ER mode
#- ER_ESI_5 should remain the least sensitive ER mode
#- thresholds and cooldown behavior should increase gradually from ESI 1 to ESI 5


#backend note - this is a static registry for care modes and their associated metadata, thresholds, and queue orchestration strategies.
#added queue orchestration metadata to support future development of more sophisticated queue management
# strategies that may use care mode, IBPIP baseline deviation, PSR burden, confidence,
# uncertainty, and lifecycle activity states to determine mode-specific active queue visibility.
# This allows for more dynamic and context-aware queue management in the
# future as we develop and integrate these additional features.

MODE_REGISTRY = {

    "ICU": {
        "description": "High-acuity continuous monitoring environment.",
        "acuity": "High",
        "default_threshold": 0.990,
        "cooldown_steps": 6,

        "queue_orchestration": {
         "current_model": "GLOBAL_QUEUE_V1",
         "current_behavior": (
        "Global PSR thresholding separates ACTIVE_QUEUE from "
        "BACKGROUND_SURVEILLANCE without suppressing patients."
    ),
        "future_model": "GLOBAL_QUEUE_V2",
         "future_direction": (
        "Future BMS-aware queue orchestration may use care mode, "
        "IBPIP baseline deviation, PSR burden, confidence, uncertainty, "
        "and lifecycle activity states to determine mode-specific active "
        "queue visibility."
),

},

   },

   "ER_ESI_1": {
        "description": "Emergency department resuscitation / immediate life-saving intervention.",
        "acuity": "Critical",
        "default_threshold": 0.990,
        "cooldown_steps": 4,

        "queue_orchestration": {
         "current_model": "GLOBAL_QUEUE_V1",
         "current_behavior": (
        "Global PSR thresholding separates ACTIVE_QUEUE from "
        "BACKGROUND_SURVEILLANCE without suppressing patients."
    ),
        "future_model": "GLOBAL_QUEUE_V2",
         "future_direction": (
        "Future BMS-aware queue orchestration may use care mode, "
        "IBPIP baseline deviation, PSR burden, confidence, uncertainty, "
        "and lifecycle activity states to determine mode-specific active "
        "queue visibility."
),

},

       },

    "ER_ESI_2": {
        "description": "Emergency department high-risk urgent patient.",
        "acuity": "High",
        "default_threshold": 0.991,
        "cooldown_steps": 6,

        "queue_orchestration": {
         "current_model": "GLOBAL_QUEUE_V1",
         "current_behavior": (
        "Global PSR thresholding separates ACTIVE_QUEUE from "
        "BACKGROUND_SURVEILLANCE without suppressing patients."
    ),
        "future_model": "GLOBAL_QUEUE_V2",
         "future_direction": (
        "Future BMS-aware queue orchestration may use care mode, "
        "IBPIP baseline deviation, PSR burden, confidence, uncertainty, "
        "and lifecycle activity states to determine mode-specific active "
        "queue visibility."
),

},

},


   "ER_ESI_3": {
        "description": "Emergency department stable but requiring multiple resources.",
        "acuity": "Moderate",
        "default_threshold": 0.992,
        "cooldown_steps": 8,

        "queue_orchestration": {
         "current_model": "GLOBAL_QUEUE_V1",
         "current_behavior": (
        "Global PSR thresholding separates ACTIVE_QUEUE from "
        "BACKGROUND_SURVEILLANCE without suppressing patients."
    ),
        "future_model": "GLOBAL_QUEUE_V2",
         "future_direction": (
        "Future BMS-aware queue orchestration may use care mode, "
        "IBPIP baseline deviation, PSR burden, confidence, uncertainty, "
        "and lifecycle activity states to determine mode-specific active "
        "queue visibility."
),

},

       },

    "ER_ESI_4": {
        "description": "Emergency department lower-acuity patient requiring minimal resources.",
        "acuity": "Lower",
        "default_threshold": 0.994,
        "cooldown_steps": 10,

        "queue_orchestration": {
         "current_model": "GLOBAL_QUEUE_V1",
         "current_behavior": (
        "Global PSR thresholding separates ACTIVE_QUEUE from "
        "BACKGROUND_SURVEILLANCE without suppressing patients."
    ),
        "future_model": "GLOBAL_QUEUE_V2",
         "future_direction": (
        "Future BMS-aware queue orchestration may use care mode, "
        "IBPIP baseline deviation, PSR burden, confidence, uncertainty, "
        "and lifecycle activity states to determine mode-specific active "
        "queue visibility."
),

},
       },

    "ER_ESI_5": {
        "description": "Emergency department lowest-acuity patient requiring minimal intervention.",
        "acuity": "Low",
        "default_threshold": 0.995,
        "cooldown_steps": 12,

        "queue_orchestration": {
         "current_model": "GLOBAL_QUEUE_V1",
         "current_behavior": (
        "Global PSR thresholding separates ACTIVE_QUEUE from "
        "BACKGROUND_SURVEILLANCE without suppressing patients."
    ),
        "future_model": "GLOBAL_QUEUE_V2",
         "future_direction": (
        "Future BMS-aware queue orchestration may use care mode, "
        "IBPIP baseline deviation, PSR burden, confidence, uncertainty, "
        "and lifecycle activity states to determine mode-specific active "
        "queue visibility."
),

},


    },

    "INPATIENT": {
        "description": "General ward or long-term inpatient surveillance.",
        "acuity": "Moderate",
        "default_threshold": 0.994,
        "cooldown_steps": 12,

        "queue_orchestration": {
         "current_model": "GLOBAL_QUEUE_V1",
         "current_behavior": (
        "Global PSR thresholding separates ACTIVE_QUEUE from "
        "BACKGROUND_SURVEILLANCE without suppressing patients."
    ),
        "future_model": "GLOBAL_QUEUE_V2",
         "future_direction": (
        "Future BMS-aware queue orchestration may use care mode, "
        "IBPIP baseline deviation, PSR burden, confidence, uncertainty, "
        "and lifecycle activity states to determine mode-specific active "
        "queue visibility."
),

},

},

    "OUTPATIENT": {
        "description": "Scheduled or ambulatory monitoring environment.",
        "acuity": "Low",
        "default_threshold": 0.995,
        "cooldown_steps": 12,

        "queue_orchestration": {
         "current_model": "GLOBAL_QUEUE_V1",
         "current_behavior": (
        "Global PSR thresholding separates ACTIVE_QUEUE from "
        "BACKGROUND_SURVEILLANCE without suppressing patients."
    ),
        "future_model": "GLOBAL_QUEUE_V2",
         "future_direction": (
        "Future BMS-aware queue orchestration may use care mode, "
        "IBPIP baseline deviation, PSR burden, confidence, uncertainty, "
        "and lifecycle activity states to determine mode-specific active "
        "queue visibility."
),

},



}

}

# Utility functions to access the mode registry

def get_mode_registry():
    return MODE_REGISTRY


def get_mode(mode_name: str):
    return MODE_REGISTRY.get(mode_name.upper())


def list_registered_modes():
    return list(MODE_REGISTRY.keys())


def mode_registry_to_dataframe():
    rows = []

    for mode, meta in MODE_REGISTRY.items():
        rows.append({
            "mode": mode,
            "description": meta["description"],
            "acuity": meta["acuity"],
            "default_threshold": meta["default_threshold"],
            "cooldown_steps": meta["cooldown_steps"],

            "queue_current_model": (
                meta.get("queue_orchestration", {}) #added queue orchestration metadata to support
                #future development of more sophisticated queue management
                .get("current_model")
            ),

            "queue_future_model": (
                meta.get("queue_orchestration", {})
                .get("future_model")
            ),
        })

    return pd.DataFrame(rows)
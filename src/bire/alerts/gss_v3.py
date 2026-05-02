from bire.alerts.gss_config import MODE_AWARE_GSS_POLICY
import pandas as pd

STATE_ORDER = {
    "SUPPRESS": 0,
    "WATCH": 1,
    "ESCALATE": 2,
}


def gss_v3_decision(
    row,
    high_short_risk: float = 0.90,
    high_mid_risk: float = 0.80,
    high_long_risk: float = 0.75,
    watch_long_risk: float = 0.45,
    convergence_threshold: float = 0.025,
    spread_threshold: float = 0.12,
    velocity_escalate: float = 0.025,
    acceleration_escalate: float = 0.0125,
    velocity_watch: float = 0.0075,
):
    """
    GSS V3.2 base decision logic.

    Produces a raw proposed state:
    - SUPPRESS
    - WATCH
    - ESCALATE

    This function does not smooth state flicker.
    Smoothing is handled by apply_state_stabilization().
    """

    r15 = row["risk_15min"]
    r30 = row["risk_30min"]
    r60 = row["risk_60min"]

    spread = row["risk_spread"]
    convergence = row["risk_convergence"]
    velocity = row["risk_velocity"]
    acceleration = row["risk_acceleration"]

    velocity = 0 if pd.isna(velocity) else velocity
    acceleration = 0 if pd.isna(acceleration) else acceleration

    horizon_confirmed_high_risk = (
        r15 >= high_short_risk
        and r30 >= high_mid_risk
    )

    converged_high_risk = (
        r60 >= high_long_risk
        and r30 >= high_mid_risk
        and convergence <= convergence_threshold
    )

    rapid_worsening_confirmed = (
        r60 >= watch_long_risk
        and velocity >= velocity_escalate
        and acceleration >= acceleration_escalate
    )

    if (
        horizon_confirmed_high_risk
        or converged_high_risk
        or rapid_worsening_confirmed
    ):
        return "ESCALATE"

    if (
        r60 >= watch_long_risk
        or spread >= spread_threshold
        or velocity >= velocity_watch
    ):
        return "WATCH"

    return "SUPPRESS"


def apply_state_stabilization(
    df: pd.DataFrame,
    patient_col: str = "patient_id",
    time_col: str = "timestamp",
    raw_state_col: str = "gss_v3_raw_state",
    stable_state_col: str = "gss_v3_state",
    persistence_steps: int = 2,
    cooldown_steps: int = 2,
) -> pd.DataFrame:
    """
    Stabilize GSS state transitions per patient.

    Purpose:
    - Reduce WATCH/SUPPRESS flicker
    - Prevent isolated spikes from immediately changing state
    - Require repeated confirmation before state escalation
    - Preserve clinically meaningful sustained escalation

    Logic:
    - A proposed higher state must persist for `persistence_steps`
      consecutive rows before being accepted.
    - After a state change, cooldown prevents immediate flipping.
    """

    if raw_state_col not in df.columns:
        raise ValueError(f"Missing required raw state column: {raw_state_col}")

    out = df.copy()
    out[time_col] = pd.to_datetime(out[time_col])
    out = out.sort_values([patient_col, time_col]).reset_index(drop=True)

    stabilized_frames = []

    for _, pdf in out.groupby(patient_col, sort=False):
        pdf = pdf.copy()

        raw_states = pdf[raw_state_col].tolist()

        stable_states = []
        current_state = "SUPPRESS"
        candidate_state = None
        candidate_count = 0
        cooldown = 0

        for proposed_state in raw_states:
            proposed_level = STATE_ORDER.get(proposed_state, 0)
            current_level = STATE_ORDER.get(current_state, 0)

            if cooldown > 0:
                stable_states.append(current_state)
                cooldown -= 1
                continue

            # Same state: stay stable
            if proposed_state == current_state:
                candidate_state = None
                candidate_count = 0
                stable_states.append(current_state)
                continue

            # Higher urgency state requires persistence
            if proposed_level > current_level:
                if candidate_state == proposed_state:
                    candidate_count += 1
                else:
                    candidate_state = proposed_state
                    candidate_count = 1

                if candidate_count >= persistence_steps:
                    current_state = proposed_state
                    candidate_state = None
                    candidate_count = 0
                    cooldown = cooldown_steps

                stable_states.append(current_state)
                continue

            # Lower urgency transitions are allowed but cooled down
            if proposed_level < current_level:
                current_state = proposed_state
                candidate_state = None
                candidate_count = 0
                cooldown = cooldown_steps
                stable_states.append(current_state)
                continue

            stable_states.append(current_state)

        pdf[stable_state_col] = stable_states
        stabilized_frames.append(pdf)

    return pd.concat(stabilized_frames, axis=0).reset_index(drop=True)


def apply_watch_persistence_escalation(
    df: pd.DataFrame,
    patient_col: str = "patient_id",
    time_col: str = "timestamp",
    state_col: str = "gss_v3_state",
    risk_col: str = "risk_60min",
    velocity_col: str = "risk_velocity",
    output_col: str = "gss_v3_state",
    watch_steps: int = 3,
    min_watch_risk: float = 0.35,
    min_velocity: float = -0.005,
) -> pd.DataFrame:
    """
    GSS V3.3 WATCH persistence escalation.

    Purpose:
    - Convert persistent WATCH states into ESCALATE when risk remains elevated.
    - Prevent BIRE from detecting early warning but waiting too long to act.
    - Preserve suppression for transient WATCH spikes.

    Logic:
    - If WATCH persists for `watch_steps` consecutive rows
    - AND long-horizon risk remains elevated
    - AND risk is not clearly improving
    - THEN convert WATCH to ESCALATE
    """

    out = df.copy()
    out[time_col] = pd.to_datetime(out[time_col])
    out = out.sort_values([patient_col, time_col]).reset_index(drop=True)

    frames = []

    for _, pdf in out.groupby(patient_col, sort=False):
        pdf = pdf.copy()

        watch_run = 0
        adjusted_states = []

        for _, row in pdf.iterrows():
            state = row[state_col]
            risk = row[risk_col]
            velocity = row[velocity_col]

            velocity = 0 if pd.isna(velocity) else velocity

            if state == "WATCH":
                watch_run += 1
            else:
                watch_run = 0

            persistent_watch = watch_run >= watch_steps
            elevated_risk = risk >= min_watch_risk
            not_improving = velocity >= min_velocity

            if persistent_watch and elevated_risk and not_improving:
                adjusted_states.append("ESCALATE")
            else:
                adjusted_states.append(state)

        pdf[output_col] = adjusted_states
        frames.append(pdf)

    return pd.concat(frames, axis=0).reset_index(drop=True)

def apply_gss_v3(
    df: pd.DataFrame,
    patient_col: str = "patient_id",
    time_col: str = "timestamp",
    persistence_steps: int = 2,
    cooldown_steps: int = 2,
) -> pd.DataFrame:
    """
    Apply GSS V3.3 production decision layer.

    Output columns:
    - gss_v3_raw_state
    - gss_v3_raw_alert
    - gss_v3_state
    - gss_v3_alert
    """

    required_cols = [
        patient_col,
        time_col,
        "risk_15min",
        "risk_30min",
        "risk_60min",
        "risk_spread",
        "risk_convergence",
        "risk_velocity",
        "risk_acceleration",
    ]

    missing = [col for col in required_cols if col not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns for GSS V3.3: {missing}")

    out = df.copy()
    out[time_col] = pd.to_datetime(out[time_col])
    out = out.sort_values([patient_col, time_col]).reset_index(drop=True)

    # Step 1: raw row-level decision
    out["gss_v3_raw_state"] = out.apply(gss_v3_decision, axis=1)
    out["gss_v3_raw_alert"] = (
        out["gss_v3_raw_state"] == "ESCALATE"
    ).astype(int)

    # Step 2: stabilize state transitions
    out = apply_state_stabilization(
        out,
        patient_col=patient_col,
        time_col=time_col,
        raw_state_col="gss_v3_raw_state",
        stable_state_col="gss_v3_state",
        persistence_steps=persistence_steps,
        cooldown_steps=cooldown_steps,
    )

    # Step 3: convert persistent WATCH into earlier ESCALATE when appropriate
    out = apply_watch_persistence_escalation(
        out,
        patient_col=patient_col,
        time_col=time_col,
        state_col="gss_v3_state",
        risk_col="risk_60min",
        velocity_col="risk_velocity",
        output_col="gss_v3_state",
        watch_steps=3,
        min_watch_risk=0.35,
        min_velocity=-0.005,
    )

    # Step 4: final stabilized alert flag
    out["gss_v3_alert"] = (
        out["gss_v3_state"] == "ESCALATE"
    ).astype(int)

    return out

def apply_gss_mode_aware(
    df,
    policy = "MODE_AWARE_GSS_POLICY",
    patient_col = "patient_id",
    time_col = "timestamp",
    risk_col = "pred_proba",
    mode_col = "bms_mode",
    base_alert_col = "bms_alert",
    event_col = "event_now",
    output_alert_col ="gss_mode_alert",
    output_suppressed_col = "gss_mode_suppressed",
    output_reason_col = "gss_mode_reason",
):
    """
    Mode-aware Gateway Suppression System (GSS)

    Applies suppression and escalation logic conditioned on BMS mode.

    Notes
    -----
    Research prototype. Not a clinical decision system.
    """

    import numpy as np

    out = df.copy()

    required_cols = [patient_col, time_col, risk_col, mode_col, base_alert_col]
    missing = [c for c in required_cols if c not in out.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    has_event = event_col in out.columns

    out = out.sort_values([patient_col, time_col]).reset_index(drop=True)

    # Initialize outputs
    out[output_alert_col] = 0
    out[output_suppressed_col] = False
    out[output_reason_col] = "no_base_alert"

    out["gss_mode_policy_threshold"] = np.nan
    out["gss_mode_cooldown_steps"] = np.nan
    out["gss_mode_risk_delta"] = 0.0

    for pid, idxs in out.groupby(patient_col).groups.items():
        idxs = list(idxs)

        last_alert_pos = None
        prev_risk = None

        for pos, idx in enumerate(idxs):
            row = out.loc[idx]

            mode = row[mode_col]
            risk = row[risk_col]
            base_alert = row[base_alert_col]

            if mode not in policy:
                raise ValueError(f"Mode {mode} not in GSS policy.")

            cfg = policy[mode]

            threshold = cfg["risk_threshold"]
            cooldown = cfg["cooldown_steps"]
            delta_thresh = cfg["escalation_delta"]
            allow_post = cfg["allow_post_event_escalation"]

            out.loc[idx, "gss_mode_policy_threshold"] = threshold
            out.loc[idx, "gss_mode_cooldown_steps"] = cooldown

            # Risk delta
            risk_delta = 0.0 if prev_risk is None else (risk - prev_risk)
            out.loc[idx, "gss_mode_risk_delta"] = risk_delta
            prev_risk = risk

            if base_alert != 1:
                out.loc[idx, output_reason_col] = "no_base_alert"
                continue

            event_now = bool(row[event_col] == 1) if has_event else False

            # Post-event suppression
            if event_now and not allow_post:
                out.loc[idx, output_suppressed_col] = True
                out.loc[idx, output_reason_col] = "post_event_suppressed"
                continue

            # Cooldown logic
            in_cooldown = (
                last_alert_pos is not None
                and (pos - last_alert_pos) < cooldown
            )

            strong = risk >= threshold
            escalating = risk_delta >= delta_thresh

            # First alert always passes
            if last_alert_pos is None:
                out.loc[idx, output_alert_col] = 1
                out.loc[idx, output_reason_col] = "first_alert"
                last_alert_pos = pos
                continue

            # Suppress during cooldown unless escalation
            if in_cooldown and not escalating:
                out.loc[idx, output_suppressed_col] = True
                out.loc[idx, output_reason_col] = "cooldown_suppressed"
                continue

            # Allow if strong or escalating
            if strong or escalating:
                out.loc[idx, output_alert_col] = 1

                if escalating:
                    out.loc[idx, output_reason_col] = "escalation_alert"
                else:
                    out.loc[idx, output_reason_col] = "threshold_alert"

                last_alert_pos = pos
            else:
                out.loc[idx, output_suppressed_col] = True
                out.loc[idx, output_reason_col] = "threshold_suppressed"

    return out

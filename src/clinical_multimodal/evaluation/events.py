"""Optional continuous-warning metrics with an explicit exposure/event contract.

One episode ends at its first event or censoring. Source event times must be
provided by the caller; they are never inferred from sampled binary targets.
"""
import numpy as np
import pandas as pd


def continuous_warning_metrics(predictions, episodes, *, horizon_hours, threshold, refractory_hours=0.):
    if not np.isfinite([horizon_hours, threshold, refractory_hours]).all() or horizon_hours <= 0 or not 0 < threshold < 1 or refractory_hours < 0:
        raise ValueError("Invalid alarm evaluation parameters")
    p, e = predictions.copy(), episodes.copy()
    p_columns = {"patient_id", "episode_id", "prediction_time", "risk"}
    e_columns = {"patient_id", "episode_id", "monitor_start", "monitor_end", "event_time"}
    if not p_columns <= set(p) or not e_columns <= set(e):
        raise ValueError("Missing prediction or exposure/event columns")
    if e.empty or e[["patient_id", "episode_id", "monitor_start", "monitor_end"]].isna().any().any():
        raise ValueError("Empty/incomplete episode exposure")
    if p[["patient_id", "episode_id", "prediction_time", "risk"]].isna().any().any():
        raise ValueError("Remove invalid-risk predictions explicitly before warning evaluation")
    if e.duplicated(["patient_id", "episode_id"]).any() or p.duplicated(["patient_id", "episode_id", "prediction_time"]).any():
        raise ValueError("Duplicate episode or prediction time")
    for column in ("monitor_start", "monitor_end", "event_time"):
        e[column] = pd.to_datetime(e[column], utc=True, errors="raise")
    p["prediction_time"] = pd.to_datetime(p["prediction_time"], utc=True, errors="raise")
    if p.duplicated(["patient_id", "episode_id", "prediction_time"]).any():
        raise ValueError("Duplicate canonical prediction time")
    risk = p["risk"].to_numpy(dtype=float)
    if not np.isfinite(risk).all() or ((risk < 0) | (risk > 1)).any():
        raise ValueError("Invalid risk probabilities")
    if (e.monitor_end <= e.monitor_start).any():
        raise ValueError("Exposure must have positive duration")
    event_rows = e.event_time.notna()
    if ((e.loc[event_rows, "event_time"] <= e.loc[event_rows, "monitor_start"]) |
        (e.loc[event_rows, "event_time"] > e.loc[event_rows, "monitor_end"])).any():
        raise ValueError("Event must lie inside the monitoring interval")
    e["end"] = e["event_time"].where(event_rows, e["monitor_end"])
    for _, group in e.groupby("patient_id"):
        intervals = group.sort_values("monitor_start")
        if any(a > b for a, b in zip(intervals["end"].iloc[:-1], intervals["monitor_start"].iloc[1:])):
            raise ValueError("Patient monitoring intervals overlap")
    allowed = set(zip(e.patient_id, e.episode_id))
    if not set(zip(p.patient_id, p.episode_id)) <= allowed:
        raise ValueError("Prediction references an unknown episode")
    horizon = pd.Timedelta(hours=horizon_hours)
    refractory = pd.Timedelta(hours=refractory_hours)
    events, detected, unassessable_events, false_alarms, censored_alarms, alarms = 0, 0, 0, 0, 0, 0
    lead_times = []
    for row in e.itertuples():
        times = p.loc[(p.patient_id == row.patient_id) & (p.episode_id == row.episode_id)].sort_values("prediction_time")
        if ((times.prediction_time < row.monitor_start) | (times.prediction_time >= row.end)).any():
            raise ValueError("Prediction falls outside pre-event monitoring exposure")
        fired, last = [], None
        for prediction in times.itertuples():
            t = prediction.prediction_time
            if prediction.risk >= threshold and (last is None or t-last >= refractory):
                fired.append(t)
                last = t
        alarms += len(fired)
        has_event = pd.notna(row.event_time)
        true_alarms = [t for t in fired if has_event and t < row.event_time <= t+horizon]
        for t in fired:
            if has_event and t < row.event_time <= t+horizon:
                continue
            if t+horizon <= row.end:
                false_alarms += 1
            else:
                censored_alarms += 1
        if has_event:
            opportunities = ((times.prediction_time < row.event_time) & (times.prediction_time+horizon >= row.event_time)).any()
            if opportunities:
                events += 1
                if true_alarms:
                    detected += 1
                    lead_times.append((row.event_time-min(true_alarms)).total_seconds()/3600)
            else:
                unassessable_events += 1
    patient_days = float((e["end"]-e["monitor_start"]).dt.total_seconds().sum()/86400)
    return {"event_detection_rate": detected/events if events else None, "evaluable_events": events,
            "detected_events": detected, "events_without_prediction_opportunity": unassessable_events,
            "first_alarm_lead_hours": lead_times,
            "median_first_alarm_lead_hours": float(np.median(lead_times)) if lead_times else None,
            "false_alarms": false_alarms, "unassessable_censored_alarms": censored_alarms,
            "total_alarms_after_refractory": alarms, "monitored_patient_days": patient_days,
            "false_alarms_per_patient_day": false_alarms/patient_days,
            "threshold": threshold, "horizon_hours": horizon_hours, "refractory_hours": refractory_hours}


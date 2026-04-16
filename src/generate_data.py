from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .utils import ensure_dir


RNG = np.random.default_rng(42)


def _clip(value: float, low: float, high: float) -> float:
    return float(np.clip(value, low, high))


def _generate_session_rows(
    athlete_id: int,
    session_id: int,
    num_steps: int,
    base_hydration: float,
    base_sleep: float,
    base_recovery: float,
    session_intensity: float,
) -> pd.DataFrame:
    # fake one workout: each loop step = one CSV row
    rows: list[dict] = []

    fatigue = float(RNG.uniform(5, 25))
    heart_rate = float(RNG.uniform(58, 85))

    base_load = float(RNG.uniform(35, 140)) * (0.5 + session_intensity / 100)

    for time_step in range(num_steps):
        progress = time_step / max(num_steps - 1, 1)

        load = base_load * (0.85 + 0.35 * progress)
        load += RNG.normal(0, 3.0)
        load = _clip(load, 20, 180)

        reps = int(_clip(round(RNG.normal(8 + 6 * session_intensity / 100, 2)), 1, 20))

        hydration_level = base_hydration - 0.35 * time_step + 0.15 * base_recovery / 100
        hydration_level += RNG.normal(0, 1.2)
        hydration_level = _clip(hydration_level, 0, 100)

        fatigue_gain = (
            0.9
            + 0.55 * (session_intensity / 100)
            + 0.010 * load
            + 0.08 * reps
            - 0.006 * base_recovery
            - 0.004 * hydration_level
            - 0.003 * base_sleep
        )
        fatigue_gain = max(0.2, fatigue_gain)
        fatigue += fatigue_gain + RNG.normal(0, 0.8)
        fatigue = _clip(fatigue, 0, 100)

        recovery_score = base_recovery - 0.25 * time_step - 0.35 * fatigue
        recovery_score += 0.10 * hydration_level + 0.08 * base_sleep
        recovery_score += RNG.normal(0, 1.5)
        recovery_score = _clip(recovery_score, 0, 100)

        heart_rate = (
            62
            + 0.45 * session_intensity
            + 0.18 * load
            + 0.35 * fatigue
            - 0.10 * recovery_score
        )
        heart_rate += RNG.normal(0, 2.2)
        heart_rate = _clip(heart_rate, 55, 200)

        power_output = (
            120
            + 1.10 * load
            + 6.0 * reps
            + 0.35 * session_intensity
            - 1.05 * fatigue
            + 0.25 * recovery_score
        )
        power_output += RNG.normal(0, 12.0)
        power_output = _clip(power_output, 40, 2200)

        # risk score (baseline 14 so early rows aren't all zero after clipping)
        injury_risk_score = (
            14.0
            + 0.48 * fatigue
            + 0.20 * session_intensity
            + 0.10 * load
            + 0.28 * reps
            - 0.28 * recovery_score
            - 0.16 * hydration_level
            - 0.10 * base_sleep
            + 0.0010 * max(0, heart_rate - 150) ** 2
        )
        injury_risk_score += RNG.normal(0, 2.0)
        injury_risk_score = _clip(injury_risk_score, 0, 100)

        rows.append(
            {
                "athlete_id": athlete_id,
                "session_id": session_id,
                "time_step": time_step,
                "heart_rate": heart_rate,
                "fatigue": fatigue,
                "load": load,
                "reps": reps,
                "recovery_score": recovery_score,
                "hydration_level": hydration_level,
                "sleep_quality": base_sleep,
                "session_intensity": session_intensity,
                "power_output": power_output,
                "injury_risk_score": injury_risk_score,
            }
        )

    return pd.DataFrame(rows)


def generate_synthetic_data(output_path: str) -> pd.DataFrame:
    # many athletes × sessions × timesteps, then save CSV
    num_athletes = int(RNG.integers(40, 81))  # 40..80 inclusive
    all_sessions: list[pd.DataFrame] = []

    global_session_id = 0
    for athlete_id in range(1, num_athletes + 1):
        num_sessions = int(RNG.integers(5, 16))  # 5..15 inclusive
        base_sleep = float(RNG.uniform(35, 95))
        base_hydration = float(RNG.uniform(40, 95))

        for _ in range(num_sessions):
            base_recovery = float(RNG.uniform(40, 95))
            session_intensity = float(RNG.uniform(25, 95))
            num_steps = int(RNG.integers(30, 61))  # 30..60 inclusive

            session_df = _generate_session_rows(
                athlete_id=athlete_id,
                session_id=global_session_id,
                num_steps=num_steps,
                base_hydration=base_hydration,
                base_sleep=base_sleep,
                base_recovery=base_recovery,
                session_intensity=session_intensity,
            )
            all_sessions.append(session_df)
            global_session_id += 1

    dataset = pd.concat(all_sessions, ignore_index=True)
    ensure_dir(Path(output_path).parent)
    dataset.to_csv(output_path, index=False)
    return dataset

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

WINDOW_SIZE = 10

# inputs for LSTM (no injury_risk_score here — that's what we predict)
FEATURE_COLUMNS = [
    "heart_rate",
    "fatigue",
    "load",
    "reps",
    "recovery_score",
    "hydration_level",
    "sleep_quality",
    "session_intensity",
    "power_output",
    "time_step_norm",
]

TARGET_COLUMN = "injury_risk_score"


def _add_time_step_norm(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["time_step_norm"] = out.groupby(["athlete_id", "session_id"])["time_step"].transform(
        lambda s: (s - s.min()) / (s.max() - s.min() + 1e-8)
    )
    return out


def _build_sequences_for_session(session_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    values = session_df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    targets = session_df[TARGET_COLUMN].to_numpy(dtype=np.float32)

    xs: list[np.ndarray] = []
    ys: list[float] = []

    # sliding window: past WINDOW_SIZE rows -> risk at row i
    for i in range(WINDOW_SIZE, len(session_df)):
        window = values[i - WINDOW_SIZE : i]
        xs.append(window)
        ys.append(float(targets[i]))

    if not xs:
        return np.zeros((0, WINDOW_SIZE, values.shape[1])), np.zeros((0,))

    return np.stack(xs, axis=0), np.array(ys, dtype=np.float32)


def load_and_preprocess(csv_path: str, test_size: float = 0.2, random_state: int = 42):
    df = pd.read_csv(csv_path)
    df = _add_time_step_norm(df)

    all_X: list[np.ndarray] = []
    all_y: list[np.ndarray] = []

    grouped = df.sort_values(["athlete_id", "session_id", "time_step"]).groupby(
        ["athlete_id", "session_id"], sort=False
    )
    for _, session_df in grouped:
        Xs, ys = _build_sequences_for_session(session_df.reset_index(drop=True))
        if len(Xs) == 0:
            continue
        all_X.append(Xs)
        all_y.append(ys)

    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, shuffle=True
    )

    n_train, timesteps, n_features = X_train.shape
    X_train_2d = X_train.reshape(-1, n_features)
    X_test_2d = X_test.reshape(-1, n_features)

    feature_scaler = MinMaxScaler()
    feature_scaler.fit(X_train_2d)

    X_train_scaled = feature_scaler.transform(X_train_2d).reshape(n_train, timesteps, n_features)
    X_test_scaled = feature_scaler.transform(X_test_2d).reshape(X_test.shape[0], timesteps, n_features)

    meta = {
        "n_features": n_features,
        "timesteps": timesteps,
        "n_train": n_train,
        "n_test": X_test.shape[0],
        "feature_columns": FEATURE_COLUMNS,
        "target_column": TARGET_COLUMN,
        "csv_path": str(Path(csv_path)),
    }
    return X_train_scaled, X_test_scaled, y_train, y_test, feature_scaler, meta

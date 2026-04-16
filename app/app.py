from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request, send_from_directory
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.preprocess import FEATURE_COLUMNS, WINDOW_SIZE

MODEL_PATH = BASE_DIR / "models" / "smart_gym_lstm.keras"
DATA_PATH = BASE_DIR / "data" / "synthetic_gym_data.csv"
RESULTS_DIR = BASE_DIR / "results"
FBX_PATH = BASE_DIR / "src" / "model" / "gym.fbx"

app = Flask(
    __name__,
    template_folder="templates",
    static_folder="static",
)

model: keras.Model | None = None
feature_scaler: MinMaxScaler | None = None


def _parse_metrics_file() -> dict[str, float]:
    metrics = {"MSE": float("nan"), "MAE": float("nan"), "R2": float("nan")}
    metrics_file = RESULTS_DIR / "metrics.txt"
    if not metrics_file.exists():
        return metrics

    for line in metrics_file.read_text(encoding="utf-8").splitlines():
        for key in metrics.keys():
            if line.startswith(f"{key}:"):
                try:
                    metrics[key] = float(line.split(":", 1)[1].strip())
                except ValueError:
                    pass
    return metrics


def _fit_or_fallback_scaler() -> MinMaxScaler:
    scaler = MinMaxScaler()

    if DATA_PATH.exists():
        df = pd.read_csv(DATA_PATH)
        if "time_step_norm" not in df.columns and "time_step" in df.columns:
            df["time_step_norm"] = df.groupby(["athlete_id", "session_id"])["time_step"].transform(
                lambda s: (s - s.min()) / (s.max() - s.min() + 1e-8)
            )

        if all(col in df.columns for col in FEATURE_COLUMNS):
            scaler.fit(df[FEATURE_COLUMNS].to_numpy(dtype=np.float32))
            return scaler

    # fallback ranges for a first demo run before data exists
    defaults = np.array(
        [
            [55, 0, 20, 1, 0, 0, 0, 0, 40, 0],
            [200, 100, 180, 20, 100, 100, 100, 100, 2200, 1],
        ],
        dtype=np.float32,
    )
    scaler.fit(defaults)
    return scaler


def _safe_float(value: object, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clamp(v: float, low: float, high: float) -> float:
    return float(np.clip(v, low, high))


def _build_feature_row(payload: dict) -> np.ndarray:
    intensity = _clamp(_safe_float(payload.get("session_intensity"), 50), 0, 100)
    fatigue = _clamp(_safe_float(payload.get("fatigue"), intensity * 0.7), 0, 100)
    recovery = _clamp(_safe_float(payload.get("recovery_score"), 70), 0, 100)
    hydration = _clamp(_safe_float(payload.get("hydration_level"), 75), 0, 100)
    sleep = _clamp(_safe_float(payload.get("sleep_quality"), 70), 0, 100)
    load = _clamp(_safe_float(payload.get("load"), 70), 20, 180)
    reps = _clamp(_safe_float(payload.get("reps"), 8), 1, 20)
    heart_rate = _clamp(_safe_float(payload.get("heart_rate"), 75 + 0.65 * intensity + 0.25 * fatigue), 55, 200)
    power_output = _clamp(
        _safe_float(payload.get("power_output"), 120 + 1.1 * load + 6 * reps + 0.35 * intensity - 1.05 * fatigue),
        40,
        2200,
    )
    time_step_norm = _clamp(_safe_float(payload.get("time_step_norm"), 0.8), 0, 1)

    row = np.array(
        [
            heart_rate,
            fatigue,
            load,
            reps,
            recovery,
            hydration,
            sleep,
            intensity,
            power_output,
            time_step_norm,
        ],
        dtype=np.float32,
    )
    return row


def _build_sequence(payload: dict) -> np.ndarray:
    # demo mode: repeat one state to create a compatible LSTM input sequence
    row = _build_feature_row(payload)
    repeated = np.repeat(row.reshape(1, -1), WINDOW_SIZE, axis=0)
    scaled = feature_scaler.transform(repeated)
    return scaled.reshape(1, WINDOW_SIZE, len(FEATURE_COLUMNS))


def _risk_label(risk_value: float) -> tuple[str, str, str, str]:
    if risk_value < 35:
        return (
            "LOW",
            "Stable short-term response",
            "Continue training with the current load.",
            "Athlete condition remains under control.",
        )
    if risk_value < 70:
        return (
            "MEDIUM",
            "Moderate stress accumulation",
            "Monitor fatigue and consider a short recovery block.",
            "Signs of fatigue are increasing but still manageable.",
        )
    return (
        "HIGH",
        "High risk response detected",
        "Reduce intensity and allow more recovery.",
        "Current load pattern may increase injury probability.",
    )


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if model is None or feature_scaler is None:
        return jsonify({"error": "Model is not available yet."}), 503

    payload = request.get_json(silent=True) or {}
    x_seq = _build_sequence(payload)

    pred = float(model.predict(x_seq, verbose=0).reshape(-1)[0])
    injury_risk = _clamp(pred, 0, 100)
    risk_level, predicted_state, recommendation, coach_insight = _risk_label(injury_risk)

    return jsonify(
        {
            "injury_risk": round(injury_risk, 2),
            "risk_level": risk_level,
            "recommendation": recommendation,
            "predicted_state": predicted_state,
            "coach_insight": coach_insight,
        }
    )


@app.route("/api/model-info")
def model_info():
    metrics = _parse_metrics_file()
    return jsonify(
        {
            "model_type": "LSTM",
            "task": "Regression",
            "target": "injury_risk_score",
            "input_type": "Time-series athlete data",
            "output": "Predicted injury risk percentage",
            "features": FEATURE_COLUMNS,
            "window_size": WINDOW_SIZE,
            "metrics": metrics,
            "model_available": MODEL_PATH.exists(),
        }
    )


@app.route("/api/metrics")
def api_metrics():
    return jsonify(_parse_metrics_file())


@app.route("/results-image/<path:filename>")
def results_image(filename: str):
    return send_from_directory(RESULTS_DIR, filename)


@app.route("/model/gym.fbx")
def gym_model():
    if FBX_PATH.exists():
        return send_from_directory(FBX_PATH.parent, FBX_PATH.name)
    return jsonify({"error": "FBX model not found. Place it at src/model/gym.fbx"}), 404


@app.route("/health")
def health():
    return jsonify(
        {
            "status": "ok",
            "model_loaded": model is not None,
            "model_path": str(MODEL_PATH),
            "fbx_found": FBX_PATH.exists(),
        }
    )


if __name__ == "__main__":
    if MODEL_PATH.exists():
        model = keras.models.load_model(MODEL_PATH)
    feature_scaler = _fit_or_fallback_scaler()
    app.run(debug=True)
else:
    if MODEL_PATH.exists():
        model = keras.models.load_model(MODEL_PATH)
    feature_scaler = _fit_or_fallback_scaler()

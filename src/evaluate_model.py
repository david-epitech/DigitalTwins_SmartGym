from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow import keras

from .utils import ensure_dir, save_figure


def _plot_training_history(history: keras.callbacks.History, output_path: str) -> None:
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs = range(1, len(loss) + 1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, loss, label="Training loss")
    ax.plot(epochs, val_loss, label="Validation loss")
    ax.set_title("Training vs validation loss (MSE)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.25)
    ax.legend()
    save_figure(fig, output_path)


def _plot_actual_vs_predicted(y_true: np.ndarray, y_pred: np.ndarray, output_path: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(y_true, y_pred, alpha=0.35, s=12)
    lims = [0, 100]
    ax.plot(lims, lims, "r--", linewidth=1, label="Perfect prediction")
    ax.set_title("Actual vs predicted injury risk score")
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.grid(True, alpha=0.25)
    ax.legend()
    save_figure(fig, output_path)


def _plot_risk_distribution(y_true: np.ndarray, y_pred: np.ndarray, output_path: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(y_true, bins=30, alpha=0.55, label="Actual", color="#1f77b4")
    ax.hist(y_pred, bins=30, alpha=0.55, label="Predicted", color="#ff7f0e")
    ax.set_title("Injury risk score distribution (test set)")
    ax.set_xlabel("Injury risk score")
    ax.set_ylabel("Count")
    ax.grid(True, alpha=0.25)
    ax.legend()
    save_figure(fig, output_path)


def evaluate_and_save(
    model: keras.Model,
    history: keras.callbacks.History,
    X_test,
    y_test,
    results_dir: str,
) -> dict[str, float]:
    ensure_dir(results_dir)

    # run on test set, then save plots + metrics.txt
    y_pred = model.predict(X_test, verbose=0).reshape(-1)

    mse = float(mean_squared_error(y_test, y_pred))
    mae = float(mean_absolute_error(y_test, y_pred))
    r2 = float(r2_score(y_test, y_pred))

    metrics_path = Path(results_dir) / "metrics.txt"
    with metrics_path.open("w", encoding="utf-8") as f:
        f.write("Smart Gym Digital Twin – evaluation metrics (test set)\n")
        f.write("=====================================================\n")
        f.write(f"MSE: {mse:.6f}\n")
        f.write(f"MAE: {mae:.6f}\n")
        f.write(f"R2:  {r2:.6f}\n")

    _plot_training_history(history, str(Path(results_dir) / "training_loss.png"))
    _plot_actual_vs_predicted(y_test, y_pred, str(Path(results_dir) / "actual_vs_predicted.png"))
    _plot_risk_distribution(y_test, y_pred, str(Path(results_dir) / "risk_distribution.png"))

    return {"mse": mse, "mae": mae, "r2": r2}

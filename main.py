from __future__ import annotations

from pathlib import Path

from src.evaluate_model import evaluate_and_save
from src.generate_data import generate_synthetic_data
from src.preprocess import load_and_preprocess
from src.train_model import train_model
from src.utils import ensure_dir


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "data" / "synthetic_gym_data.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "smart_gym_lstm.keras"
RESULTS_DIR = PROJECT_ROOT / "results"


def main() -> None:
    print("Smart Gym Digital Twin – Injury Risk Prediction (LSTM)")
    print("======================================================")

    # make sure output folders exist
    ensure_dir(PROJECT_ROOT / "data")
    ensure_dir(PROJECT_ROOT / "models")
    ensure_dir(RESULTS_DIR)

    print("\nStep 1: Generating synthetic dataset...")
    generate_synthetic_data(str(DATA_PATH))
    print("Dataset saved successfully.")

    print("\nStep 2: Preprocessing data...")
    X_train, X_test, y_train, y_test, _feature_scaler, meta = load_and_preprocess(str(DATA_PATH))
    print(f"Train sequences: {meta['n_train']} | Test sequences: {meta['n_test']}")
    print(f"Each sample shape: ({meta['timesteps']} time steps, {meta['n_features']} features)")

    print("\nStep 3: Training LSTM model...")
    model, history = train_model(
        X_train,
        y_train,
        model_output_path=str(MODEL_PATH),
    )
    print("Training complete.")
    print(f"Model saved to: {MODEL_PATH}")

    print("\nStep 4: Evaluating model...")
    metrics = evaluate_and_save(
        model=model,
        history=history,
        X_test=X_test,
        y_test=y_test,
        results_dir=str(RESULTS_DIR),
    )
    print("Evaluation complete.")
    print(f"Test MSE: {metrics['mse']:.6f}")
    print(f"Test MAE: {metrics['mae']:.6f}")
    print(f"Test R2:  {metrics['r2']:.6f}")
    print(f"Plots and metrics saved under: {RESULTS_DIR}")


if __name__ == "__main__":
    main()

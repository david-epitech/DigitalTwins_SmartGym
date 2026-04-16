from __future__ import annotations

from pathlib import Path

from tensorflow import keras
from tensorflow.keras import layers


def build_lstm_model(timesteps: int, n_features: int) -> keras.Model:
    model = keras.Sequential(
        [
            layers.Input(shape=(timesteps, n_features)),
            layers.LSTM(64, return_sequences=True),
            layers.Dropout(0.2),
            layers.LSTM(32),
            layers.Dense(1),
        ],
        name="smart_gym_lstm",
    )
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


def train_model(
    X_train,
    y_train,
    model_output_path: str,
    epochs: int = 60,
    batch_size: int = 64,
    validation_split: float = 0.15,
) -> tuple[keras.Model, keras.callbacks.History]:
    _, timesteps, n_features = X_train.shape
    model = build_lstm_model(timesteps=timesteps, n_features=n_features)

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=6,
            restore_best_weights=True,
        )
    ]

    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        callbacks=callbacks,
        verbose=1,
    )

    Path(model_output_path).parent.mkdir(parents=True, exist_ok=True)
    model.save(model_output_path)
    return model, history

# Smart Gym Digital Twin

**Injury risk prediction with an LSTM** (TensorFlow / Keras). University-style prototype: synthetic workout time series, training pipeline, saved model, metrics, and plots.

## Overview

The project generates **synthetic** per-time-step gym data (multiple athletes and sessions), builds **sliding-window sequences** for an LSTM, trains a regressor to predict **injury risk score** (0–100), and writes outputs under `data/`, `models/`, and `results/`.

It is **Python-only** (no web UI). A future HTML/CSS front-end could load `models/smart_gym_lstm.keras` once the pipeline is validated.

## Why LSTM and Digital Twin?

- **LSTM:** Heart rate, fatigue, and load evolve over time; a sequence model uses past steps as context instead of a single row.
- **Digital Twin:** A computational stand-in for an athlete is updated with streaming-style measurements; here those signals are synthetic but mimic monitoring and risk estimation.

## Requirements

- Python **3.10–3.12** recommended (TensorFlow support varies; **3.13** may not install cleanly on all platforms).
- Dependencies: see `requirements.txt` (TensorFlow, NumPy, Pandas, Matplotlib, scikit-learn).

## Setup

Clone the repo, enter the project folder, create a virtual environment, and install packages.

**Windows (PowerShell)**

```powershell
cd smart-gym-digital-twin
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**macOS / Linux**

```bash
cd smart-gym-digital-twin
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Do not commit `venv/`. A typical `.gitignore` includes `venv/`, `__pycache__/`, and optionally `data/`, `models/`, `results/` if you want a source-only repo (otherwise you can commit generated files for grading).

## Run

```bash
python main.py
```

This will:

1. Generate `data/synthetic_gym_data.csv`
2. Preprocess data, scale features, build LSTM sequences
3. Train and save `models/smart_gym_lstm.keras`
4. Evaluate and save `results/metrics.txt` plus the three PNG plots

## Repository layout

```
smart-gym-digital-twin/
├── data/                    # synthetic_gym_data.csv (generated)
├── models/                  # smart_gym_lstm.keras (generated)
├── results/                 # plots + metrics.txt (generated)
├── src/
│   ├── __init__.py
│   ├── generate_data.py
│   ├── preprocess.py
│   ├── train_model.py
│   ├── evaluate_model.py
│   └── utils.py
├── main.py
├── requirements.txt
└── README.md
```

## Technical notes

- Sequences are built **per session** so windows do not span different workouts.
- Each training sample uses the **previous 10** time steps of the selected features to predict **injury risk at the same session time index** (see `src/preprocess.py`).

## Demo Video
https://github.com/user-attachments/assets/fb9d7ee3-cf80-42b1-a7a6-a1b2c9ec502a



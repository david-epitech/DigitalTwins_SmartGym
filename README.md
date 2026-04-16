# Smart Gym Digital Twin - Injury Risk Prediction using LSTM

A complete university mini-project combining a Python AI pipeline and a web interface.
The system simulates athlete workout monitoring, predicts injury risk with an LSTM model, and displays insights in a Digital Twin dashboard.

## What this project includes

### Dashboard tab
- Real-time style monitoring cards (heart rate, fatigue, load, reps, recovery, hydration, power).
- Interactive 3D viewer in the center (Three.js + FBXLoader).
- Risk prediction panel with risk percentage, level, recommendation, and coach insight.
- Session controls (profile, exercise type, sliders, load, reps, etc.).
- Trend charts (heart rate, fatigue, injury risk).

### AI Model tab
- Model summary (LSTM, regression, target, input/output).
- Why LSTM explanation.
- Input feature list used by the model.
- Output explanation (0-100 injury risk score).
- Evaluation metrics loaded from `results/metrics.txt`.
- Generated graphs from training/evaluation results.
- Notes about the saved `.keras` model and Digital Twin concept.

## Core idea

- A virtual athlete state is monitored through workout variables.
- The backend uses the trained `.keras` LSTM model to predict injury risk.
- The frontend sends controls to Flask and updates the dashboard from JSON responses.

## Project structure

```text
smart-gym-digital-twin/
├── app/
│   ├── templates/
│   │   └── index.html
│   ├── static/
│   │   ├── css/
│   │   │   └── style.css
│   │   ├── js/
│   │   │   └── script.js
│   │   ├── assets/
│   │   └── generated/
│   └── app.py
├── src/
│   ├── model/
│   │   └── gym.fbx   # place your FBX model here
│   ├── generate_data.py
│   ├── preprocess.py
│   ├── train_model.py
│   ├── evaluate_model.py
│   └── utils.py
├── data/
├── models/
│   └── smart_gym_lstm.keras
├── results/
│   ├── training_loss.png
│   ├── actual_vs_predicted.png
│   ├── risk_distribution.png
│   └── metrics.txt
├── main.py
├── requirements.txt
└── README.md
```

## Requirements

- Python 3.10-3.12 recommended.
- Install dependencies from `requirements.txt`.

## Setup (venv + dependencies)

### Windows PowerShell

```powershell
cd smart-gym-digital-twin
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### macOS / Linux

```bash
cd smart-gym-digital-twin
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Where to place the 3D model

Put your FBX file at:

```text
app/static/model/gym.fbx
```

If `gym.fbx` is missing, the web page still opens and the 3D viewer shows a fallback message.

## Run flow (clean and explicit)

1. Create and activate virtual environment.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. If model/results are missing, generate/train first:
   ```bash
   python main.py
   ```
   This creates or refreshes:
   - `data/synthetic_gym_data.csv`
   - `models/smart_gym_lstm.keras`
   - `results/metrics.txt`
   - result charts in `results/`
4. Start Flask backend:
   ```bash
   python app/app.py
   ```
5. Open the web interface in your browser:
   - [http://127.0.0.1:5000](http://127.0.0.1:5000)

## Backend and frontend communication

- Frontend sends session values to `POST /predict`.
- Flask loads the trained `.keras` model once at startup.
- Backend builds a safe demo-compatible sequence shape for LSTM input.
- Backend returns JSON:
  - `injury_risk`
  - `risk_level`
  - `recommendation`
  - `predicted_state`
  - `coach_insight`
- Frontend updates cards, badges, insights, and trend charts.

Additional routes:
- `GET /api/model-info` for model metadata + metrics.
- `GET /results-image/<filename>` to display generated images in AI Model tab.
- `GET /static/model/gym.fbx` to load the 3D model.

## Notes

- Existing ML logic is preserved.
- Web integration does not retrain during requests.
- The `.keras` model is used through Flask backend only (not in-browser).
- Sequences are built **per session** so windows do not span different workouts.
- Each training sample uses the **previous 10** time steps of the selected features to predict **injury risk at the same session time index** (see `src/preprocess.py`).

## Demo Video
https://github.com/user-attachments/assets/fb9d7ee3-cf80-42b1-a7a6-a1b2c9ec502a

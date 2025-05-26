# 🏎️ F1 Race Result Predictor

This project uses machine learning to predict Formula 1 race outcomes using historical data, live telemetry, qualifying performance, and weather forecasts. It simulates the upcoming 2025 Grand Prix races — starting with Monaco — by predicting race times for each driver and ranking them accordingly.

## 🔍 Project Highlights

- Combines **FastF1 API**, **2024 race data**, and **2025 qualifying results**
- Integrates **live weather forecasts** via OpenWeather API
- Models include both [**Gradient Boosting**](https://github.com/mar-antaya/2025_f1_predictions)
 and **PyTorch neural networks**
- Prioritizes **low MAE** and **podium stability** across runs
- Predicts full driver race rankings and identifies the expected race winner

---

## 📊 Data Sources

| Source | Purpose |
|--------|---------|
| [FastF1](https://theoehrly.github.io/Fast-F1/) | Lap times, sector data, driver telemetry |
| 2024 Season Data | Training data for model |
| 2025 Qualifying Results | Real-time prediction input |
| OpenWeather API | Forecasts for race-time temperature and rain probability |
| Constructor Standings | Team performance weighting |
| Custom Historical Stats | Clean-air race pace, position-change metrics |

---

## 🧠 Model Pipeline

1. **Data Collection**
   - Gathers session data and qualifying results using FastF1
   - Fetches race-time weather forecast via OpenWeather

2. **Feature Engineering**
   - Aggregates mean sector times
   - Calculates team performance score from constructor points
   - Maps average position change at Monaco
   - Merges clean air pace, qualifying time, and weather into features

3. **Model Training**
   - Baseline: `GradientBoostingRegressor` from scikit-learn
   - Custom: `PyTorch` MLP with dropout, tuning, and evaluation

4. **Prediction & Evaluation**
   - Outputs predicted race times and sorts them for ranking
   - Evaluates performance using **Mean Absolute Error (MAE)**

---

## 📦 Dependencies

```bash
pip install fastf1 pandas numpy requests scikit-learn matplotlib torch

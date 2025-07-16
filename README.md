# EV Battery Usage Prediction & Convoy Recommendation API

## Project Summary

**Project Name:** `batteryconvoy-predictor-api`  
**Platform:** Python (Flask, XGBoost, Pandas)  
**Purpose:** Real-time prediction of EV battery consumption and recommendation generation for vehicle convoys based on elevation, SoC, and other trip-level telemetry inputs.

This application deploys a machine learning pipeline designed to process incoming vehicle trip data, predict battery usage, and return intelligent energy-efficiency recommendations for convoy vehicles. The predictions are informed by historical data and a trained XGBoost regression model, while recommendation logic responds dynamically to field thresholds such as elevation gain, SoC, and SoH.

---

## Key Features

- **Real-Time Prediction API**  
  Predict battery consumption (kWh) given trip and vehicle telemetry data.

- **Convoy-Based Recommendation System**  
  Evaluate trip context (such as elevation) and recommend speed changes, mode switches (for hybrids), or charging when required.

- **Telemetry Data Mapping & Resilience**  
  Intelligent normalization and default-filling for incoming JSON data fields across various formats.

- **Custom Feature Engineering Pipeline**  
  Physics-inspired and empirically-guided features like:
  - Adjusted energy cost by vehicle type
  - Trip intensity (distance over time)
  - Battery thermal stress (temperature × distance)

- **End-to-End Deployment**  
  Includes model persistence, session handling (Flask-Session), and a lightweight REST API server.

---

## API Endpoints

### `/`  
**Method:** `GET`  
**Description:** Root endpoint. Returns a simple HTML message.

---

### `/lead_trip`  
**Method:** `POST`  
**Description:**  
Stores the lead vehicle context – trip, elevation, temperature, battery status – to use for downstream convoy recommendation logic.

**Request Body Example (JSON):**
{
"total_distance": 12.5,
"route_energy_cost": 14.2,
"trip_duration": 2.0,
"max_cell_temp": 38.0,
"state_of_charge": 80,
"vehicle_type": "sedan",
"elevation_gain": 60
}


**Response:**
{
"message": "Lead trip context set.",
"lead_trip_context": {...}
}


---

### `/predict`  
**Method:** `POST`  
**Description:**  
Accepts trip and vehicle data; returns predicted battery usage (in kWh).

**Input:** JSON object with one vehicle's telemetry.

**Output:**
{
"predicted_batteryused_kwh": 6.75
}


---

### `/recommend`  
**Method:** `POST`  
**Description:**  
Generates tailored driving/charging recommendations for each convoy vehicle using the pre-set lead trip context.

**Input:**  
- Either a JSON object with a `convoy_vehicles` list  
- Or a JSON list of vehicle objects directly

**Output Sample:**
{
"timestamp": "2025-07-02 09:38:25",
"predictions": [{...}],
"convoy_recommendations": [
{
"vehicle_id": 3,
"action": "Change Speed",
"recommended_speed_kmph": 62,
"reason": "Elevation detected ahead. Maintain recommended speed to optimize battery SoH."
}
]
}


---

## Technologies Used

- **Python 3.9+**
- **Flask** + **Flask-Session**: Lightweight REST API backend
- **Pandas / NumPy**: Data manipulation
- **XGBoost**: Regression model
- **Scikit-learn**: ML utilities and pipeline composition
- **Joblib**: Model persistence
- **Datetime / Random**: Real-time timestamping and control logic

---

## Project Structure

| File / Folder       | Purpose |
|---------------------|---------|
| `pipeline.pkl`      | Trained sklearn/XGBoost pipeline |
| `battery_trips_cleaned_no_onehot.csv` | Cleaned dataset used for training |
| `main.py`           | API server and core ML logic |
| `README.md`         | Project documentation |
| `requirements.txt`  | Dependency management |

---

## Feature Engineering Highlights

- **adjusted_energy_cost**: Adjusts energy cost for vehicle efficiency  
  \[
  \text{adjusted\_energy\_cost} = \text{route\_energy\_cost} \times \text{vehicle\_efficiency}
  \]
- **intensity_score**: Average trip speed  
  \[
  \text{intensity\_score} = \frac{\text{total\_distance}}{\text{trip\_duration}}
  \]
- **temp_distance_stress**: Battery thermal stress proxy  
  \[
  \text{temp\_distance\_stress} = \text{max\_cell\_temp} \times \text{total\_distance}
  \]

---

## Setup Instructions

### 1. Clone the Repository

git clone https://github.com/arnavgupta27/Elevation-based-EV-Battery-Usage-Prediction-Convoy-Recommendation-API

cd batteryconvoy-api


### 2. Install Dependencies

> Create and activate a virtual environment if preferred.

- Flask
- Flask-Session
- pandas
- numpy
- scikit-learn
- xgboost
- joblib


### 3. Prepare Training Data and Train Model (First Time Only)

Ensure `battery_trips_cleaned_no_onehot.csv` exists in the root folder.

python app.py #Runs initial training if no pipeline.pkl found


### 4. Launch the API Server

python app.py


Server runs on: `http://0.0.0.0:8000`

---

## Notes

- Endpoint `/lead_trip` must be called first to store shared trip context.
- All incoming vehicle data is normalized via `FIELD_MAP` and filled with defaults via `MEAN_VALUES` to maintain consistency.
- Currently supports a small set of vehicle types: `sedan`, `suv`, `truck`, `hatchback`.

---

## Future Enhancements

- OTA backend integration and dashboard UI
- Vehicle location and Mapbox integration
- Dynamic charging station lookup
- Expanded diagnostics and anomaly detection

---



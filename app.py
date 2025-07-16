
import os
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, session
from flask_session import Session
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
import joblib
from datetime import datetime
import random
# combined app

# Field Mapping and Default values for missing fields. 
# I will be setting up contingencies in case ANY data is found missing in any part of the pipeline.
# These extra values will not adversely affect anything else.

# Mapping from incoming JSON field names (lowercase, no underscores) to pipeline field names
ALL_VEHICLE_TYPES = ['sedan', 'suv', 'truck', 'hatchback', 'unknown']


FIELD_MAP = {
    "totaldistance": "total_distance",
    "total_distance": "total_distance",
    "routeenergycost": "route_energy_cost",
    "route_energy_cost": "route_energy_cost",
    "tripduration": "trip_duration",
    "trip_duration": "trip_duration",
    "maxcelltemp": "max_cell_temp",
    "max_cell_temp": "max_cell_temp",
    "stateofcharge": "state_of_charge",
    "state_of_charge": "state_of_charge",
    "vehicletype": "vehicle_type",
    "vehicle_type": "vehicle_type",
    "elevationgain": "elevation_gain",
    "maxelevation": "max_elevation",
    "minelevation": "min_elevation",
    "totalascent": "total_ascent",
    "totaldescent": "total_descent",
    "mincelltemp": "min_cell_temp",
    "stateofhealth": "state_of_health",
    "vehicleid": "vehicleid",
    # Add more if needed
    "RouteEnergyCost_kWh": "route_energy_cost"
}



# Mean values for all fields (example values from your project)

MEAN_VALUES = {
    # BMS_sample_data.csv (lowercase/underscore for consistency)
    "vehicleid": 10,
    "battery_pack_id": "BP-1001",
    "manufacturer": "tesla",
    "model": "model s",
    "total_pack_voltage": 401.0,
    "total_pack_current": 122.0,
    "state_of_charge": 85.0,
    "state_of_health": 96.0,
    "max_cell_voltage": 4.18,
    "min_cell_voltage": 3.90,
    "max_cell_temp": 40.0,
    "min_cell_temp": 29.0,
    "estimated_range": 400,
    "bms_status": "ok",
    "timestamp": "2025-07-06 08:00",

    # Charging_data_sample_data.csv
    "charging_session_id": 10,
    "start_time": "2025-07-06 08:00",
    "end_time": "2025-07-06 09:00",
    "energy_consumed_kwh": 30.0,
    "charger_type": "ac",

    # GPS_sample_data.csv
    "latitude": 40.0,
    "longitude": -74.0,
    "altitude": 150,
    "speed_kmph": 40.0,
    "heading_deg": 90,
    "is_lead_vehicle": False,

    # Trip_data.csv
    "trip_id": 10,
    "start_timestamp": "2025-07-06 08:00",
    "end_timestamp": "2025-07-06 09:00",
    "start_latitude": 40.0,
    "start_longitude": -74.0,
    "end_latitude": 40.1,
    "end_longitude": -74.1,
    "total_distance": 12.0,
    "fuel_used_liters": 0.5,
    "battery_used_kwh": 7.0,
    "elevation_profile": "0:100;5:110;10:105,15:110",
    "max_elevation": 150,
    "min_elevation": 80,
    "total_ascent": 12,
    "total_descent": 10,

    # vehicle_info_sample_data.csv
    "vin": "1hgcm82633a004352",
    "make": "toyota",
    "year": 2022,
    "firmware_version": "v1.2.3",
    "vehicle_type": "sedan",   # lowercase for your pipeline

    # Model-required fields (must match RAW_FIELDS)
    "route_energy_cost": 15.0,
    "trip_duration": 2.1,

    # One-hot vehicle type (only if ever needed, not for your pipeline)
    "vehicle_type_sedan": True,
    "vehicle_type_suv": False,
    "vehicle_type_truck": False,
    "vehicle_type_hatchback": False,
}


def preprocess_input(input_json):
    """
    Maps incoming JSON keys to expected keys (using FIELD_MAP),
    fills missing fields with mean/default values from MEAN_VALUES,
    and returns a cleaned dictionary ready for prediction.
    """
    mapped = {}

    # 1. Map all input keys to expected keys (lowercase/underscore)
    for key, value in input_json.items():
        # Normalize key: lowercase and remove underscores for mapping
        norm_key = key.lower().replace("_", "")
        mapped_key = FIELD_MAP.get(norm_key, key.lower())
        mapped[mapped_key] = value

    # 2. Fill missing fields (needed for model) with mean/default values
    for field, mean_val in MEAN_VALUES.items():
        # If field is missing or empty, fill with mean/default
        if field not in mapped or mapped[field] in [None, "", []]:
            mapped[field] = mean_val

        # Type safety: cast numeric fields to float
    numeric_fields = ["total_distance", "route_energy_cost", "trip_duration", "max_cell_temp", "state_of_charge"]
    for field in numeric_fields:
        try:
            mapped[field] = float(mapped[field])
        except (ValueError, TypeError, KeyError):
            mapped[field] = float(MEAN_VALUES[field])

    # Vehicle type normalization (lowercase, valid set)
    vehicle_type = str(mapped.get("vehicle_type", "sedan")).lower()
    valid_vehicle_types = ["sedan", "suv", "truck", "hatchback"]
    if vehicle_type not in valid_vehicle_types:
        vehicle_type = "sedan"
    mapped["vehicle_type"] = vehicle_type

    return mapped




# --- Custom Transformer ---

class CustomFeatureEngineering(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        X = X.copy()
        vehicle_efficiency = {
            'sedan': 0.9,
            'suv': 1.1,
            'truck': 1.3,
            'ev': 1.0
        }
        # Normalize vehicle_type and handle unknowns
        X['vehicle_type'] = X['vehicle_type'].str.lower().fillna('unknown')
        valid_types = ['sedan', 'suv', 'truck', 'hatchback']
        X['vehicle_type'] = X['vehicle_type'].apply(lambda x: x if x in valid_types else 'unknown')

        X['adjusted_energy_cost'] = X['route_energy_cost'] * X['vehicle_type'].map(vehicle_efficiency).fillna(1.0)
        X['intensity_score'] = (X['total_distance'] / X['trip_duration']).replace([np.inf, -np.inf], 0).fillna(0)
        X['temp_distance_stress'] = X['max_cell_temp'] * X['total_distance']
        X['adjusted_energy_cost'] = X['adjusted_energy_cost'].fillna(X['route_energy_cost'])
        X['temp_distance_stress'] = X['temp_distance_stress'].fillna(0)

        # One-hot encode and ensure all columns are present
        dummies = pd.get_dummies(X['vehicle_type'], prefix='vehicle_type')
        for vt in ALL_VEHICLE_TYPES:
            col = f'vehicle_type_{vt}'
            if col not in dummies.columns:
                dummies[col] = 0
        dummies = dummies[[f'vehicle_type_{vt}' for vt in ALL_VEHICLE_TYPES]]
        X = pd.concat([X, dummies], axis=1)
        X = X.drop(columns=['vehicle_type'])
        return X

# --- Train or Load Pipeline ---
PIPELINE_PATH = 'pipeline.pkl'

if not os.path.exists(PIPELINE_PATH):
    # Load your prepared data
    df = pd.read_csv('battery_trips_cleaned_no_onehot.csv')

    # Define the features and target (all lowercase/underscore)
    basic_features = [
        'total_distance',
        'route_energy_cost',
        'trip_duration',
        'max_cell_temp',
        'state_of_charge',
        'vehicle_type'
    ]
    target = 'batteryused_kwh'

    # Drop rows with missing values in any feature or target
    model_data = df.dropna(subset=basic_features + [target])
    X = model_data[basic_features]
    y = model_data[target]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Build the pipeline (feature engineering + model)
    pipeline = Pipeline([
        ('feature_engineering', CustomFeatureEngineering()),
        ('model', XGBRegressor(n_estimators=100, random_state=42))
    ])

    # Fit and save the pipeline
    pipeline.fit(X_train, y_train)
    joblib.dump(pipeline, PIPELINE_PATH)
else:
    # Load the trained pipeline
    pipeline = joblib.load(PIPELINE_PATH)


# --- Flask App Setup ---
app = Flask(__name__)
app.secret_key = 'your_secret_key'
app.config['SESSION_TYPE'] = 'filesystem'
Session(app)

RAW_FIELDS = [
    'total_distance',
    'route_energy_cost',
    'trip_duration',
    'max_cell_temp',
    'state_of_charge',
    'vehicle_type'
]


@app.route('/')
def index():
    return "EV Battery Usage Prediction & Convoy Recommendation API (Single JSON, Professional Version)"

@app.route('/lead_trip', methods=['POST'])
def set_lead_trip():
    """
    Accepts a single JSON with all trip, BMS, and extra fields.
    Stores the context in session for use in recommendations.
    """
    data = request.get_json()
    if not data:
        return jsonify({'error': "Input JSON is required."}), 400
    cleaned_data = preprocess_input(data)
    session['lead_trip_context'] = cleaned_data

    return jsonify({'message': 'Lead trip context set.', 'lead_trip_context': data})

@app.route('/predict', methods=['POST'])
def predict():
    """
    Accepts a single JSON with all fields.
    Extracts only model-required fields and predicts battery usage.
    """
    input_json = request.get_json()
    if not input_json:
        return jsonify({'error': 'No input data provided'}), 400
    cleaned_input = preprocess_input(input_json)
    data = {k: cleaned_input.get(k) for k in RAW_FIELDS}

    df = pd.DataFrame([data])
    pred = pipeline.predict(df)[0]
    return jsonify({'predicted_batteryused_kwh': float(pred)})
@app.route('/recommend', methods=['POST'])
def recommend_for_convoy():
    """
    Accepts either:
    - A JSON object with a "convoy_vehicles" key containing a list of vehicles, or
    - A JSON list of vehicle objects directly.
    Uses the stored lead_trip_context and generates recommendations for each vehicle.
    """
    data = request.get_json()
    if isinstance(data, list):
        convoy_vehicles = data
    else:
        convoy_vehicles = data.get('convoy_vehicles', [])

    lead_trip_context = session.get('lead_trip_context', None)
    if lead_trip_context is None:
        return jsonify({'error': "Lead trip context not set. Please POST to /lead_trip first."}), 400

    # Thresholds
    HIGH_ELEVATION_GAIN = 50
    LOW_SOC_THRESHOLD = 80
    CRITICAL_HEALTH_THRESHOLD = 95
    elevation_gain = lead_trip_context.get('elevation_gain', 0)
    high_elevation = elevation_gain >= HIGH_ELEVATION_GAIN

    recommendations = []
    predictions = []

    for vehicle in convoy_vehicles:
        # Merge lead trip context and vehicle-specific fields
        merged_input = {**lead_trip_context, **vehicle}
        cleaned_input = preprocess_input(merged_input)
        data = {k: cleaned_input.get(k) for k in RAW_FIELDS}

        df = pd.DataFrame([data])
        predicted_batteryused_kwh = float(pipeline.predict(df)[0])
        predictions.append({
            "vehicle_id": vehicle.get('vehicleid'),
            "predicted_batteryused_kwh": predicted_batteryused_kwh
        })

        soc = vehicle.get('state_of_charge', 100)
        health = vehicle.get('state_of_health', 100)
        vehicle_type = str(vehicle.get('vehicle_type', '')).lower()
        recs = []

        # 1. Elevation-based recommendation
        if high_elevation:
            recs.append({
                "vehicle_id": vehicle.get('vehicleid'),
                "action": "Change Speed",
                "recommended_speed_kmph": random.randint(54, 64),
                "reason": "Elevation detected ahead. Maintain recommended speed to optimize battery SoH."
            })
            if 'hybrid' in vehicle_type:
                recs.append({
                    "vehicle_id": vehicle.get('vehicleid'),
                    "action": "Switch Modes",
                    "sub_action": "switch_to_fuel",
                    "reason": "Elevation detected ahead. Consider switching to fuel for efficiency."
                })

        # 2. SoC and health-based recommendations
        if soc <= LOW_SOC_THRESHOLD:
            recs.append({
                "vehicle_id": vehicle.get('vehicleid'),
                "action": "charge",
                "station_name": "NearestStation",
                "reason": f"Low SoC ({soc}%) - charge to maintain battery health"
            })
        if health < CRITICAL_HEALTH_THRESHOLD:
            recs.append({
                "vehicle_id": vehicle.get('vehicleid'),
                "action": "reduce_load",
                "recommended_load_kg": 50,
                "reason": f"Battery health critical ({health}%) - reduce cargo load"
            })
        if recs:
            recommendations.extend(recs)

    return jsonify({
        "timestamp": str(datetime.now()),
        "predictions": predictions,
        "convoy_recommendations": recommendations
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
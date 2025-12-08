import joblib
import pandas as pd
import json

# -----------------------------------------------------
# 1. LOAD THE TRAINED PIPELINE
# -----------------------------------------------------
full_pipeline = joblib.load(r"E:\vidyut_sanchay\backend\full_ml_pipeline_1.pkl")

# -----------------------------------------------------
# 2. PREPARE TEST DATA
# -----------------------------------------------------
test_data = pd.DataFrame([{
    "project_type": "Transmission Line",
    "region": "South",
    "soil_type": "Rocky",
    "terrain_type": "Plains",
    "voltage_kv": 132,
    "circuit_type": "Single Circuit",
    "conductor_type": "ACSR Panther",
    "Length_km": 71.79,
    "num_towers": 282
}])

# -----------------------------------------------------
# 3. MAKE PREDICTIONS
# -----------------------------------------------------
raw_pred = full_pipeline.predict(test_data)[0]

# -----------------------------------------------------
# 4. MAP OUTPUTS TO NAMES + UNITS
# -----------------------------------------------------
output_mapping = {
    "steel_tonnes": {"value": raw_pred[0], "unit": "tonnes"},
    "conductor_km": {"value": raw_pred[1], "unit": "km"},
    "insulators_unit": {"value": raw_pred[2], "unit": "units"},
    "concrete_cubic_meter": {"value": raw_pred[3], "unit": "cubic_meter"},
    "bus_reactor_count": {"value": raw_pred[4], "unit": "count"},
    "transformers_count": {"value": raw_pred[5], "unit": "count"},
    "circuit_breaker_count": {"value": raw_pred[6], "unit": "count"}
}

# -----------------------------------------------------
# 5. CONVERT TO JSON
# -----------------------------------------------------
json_output = json.dumps(output_mapping, indent=4)

print(json_output)

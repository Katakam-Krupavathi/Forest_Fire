import os
import logging
import pickle
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

application = Flask(__name__)
app = application

# Load Model and Scaler
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "ridge.pkl")
SCALER_PATH = os.path.join(os.path.dirname(__file__), "models", "scaler.pkl")

try:
    ridge_model = pickle.load(open(MODEL_PATH, "rb"))
    scaler = pickle.load(open(SCALER_PATH, "rb"))
    logger.info("Successfully loaded ridge model and scaler.")
except Exception as e:
    logger.error(f"Error loading models: {e}")
    ridge_model = None
    scaler = None

FEATURE_NAMES = ["Temperature", "RH", "Ws", "Rain", "FFMC", "DMC", "ISI", "Classes", "Region"]

# Validation rules: (min_val, max_val, description)
VALIDATION_RULES = {
    "Temperature": (-10.0, 60.0, "Temperature must be between -10°C and 60°C"),
    "RH": (0.0, 100.0, "Relative Humidity (RH) must be between 0% and 100%"),
    "Ws": (0.0, 100.0, "Wind Speed (Ws) must be between 0 and 100 km/h"),
    "Rain": (0.0, 500.0, "Rain must be 0 mm or greater"),
    "FFMC": (0.0, 105.0, "FFMC must be between 0 and 105"),
    "DMC": (0.0, 500.0, "DMC must be 0 or greater"),
    "ISI": (0.0, 100.0, "ISI must be 0 or greater"),
    "Classes": (0.0, 1.0, "Classes must be 0 (Not Fire) or 1 (Fire)"),
    "Region": (0.0, 1.0, "Region must be 0 (Bejaia) or 1 (Sidi Bel-abbes)"),
}


def get_fire_risk_level(fwi_value):
    """Categorizes the Fire Weather Index (FWI) value into standard risk levels."""
    if fwi_value < 5.0:
        return "Low", "success"
    elif fwi_value < 15.0:
        return "Moderate", "info"
    elif fwi_value < 30.0:
        return "High", "warning"
    else:
        return "Extreme", "danger"


def parse_and_validate_inputs(data_dict):
    """Parses and validates feature inputs from dictionary."""
    parsed_values = {}
    for feature in FEATURE_NAMES:
        val = data_dict.get(feature)
        if val is None or str(val).strip() == "":
            return None, f"Missing required parameter: '{feature}'"

        try:
            num_val = float(val)
        except (ValueError, TypeError):
            return None, f"Invalid value for '{feature}'. Must be a valid number."

        min_val, max_val, err_msg = VALIDATION_RULES[feature]
        if num_val < min_val or num_val > max_val:
            return None, err_msg

        if feature in ["Classes", "Region"] and num_val not in [0.0, 1.0]:
            return None, f"'{feature}' must be either 0 or 1."

        parsed_values[feature] = num_val

    return parsed_values, None


@app.route("/")
def index():
    """Renders the landing page."""
    return render_template("index.html")


@app.route("/health")
def health():
    """Health check endpoint for deployment monitoring."""
    is_ready = ridge_model is not None and scaler is not None
    return jsonify({
        "status": "healthy" if is_ready else "unhealthy",
        "model_loaded": is_ready
    }), 200 if is_ready else 503


@app.route("/predict", methods=["GET", "POST"])
def predict_data():
    """Handles both GET (render form) and POST (prediction via form or JSON)."""
    if request.method == "GET":
        return render_template("home.html")

    if ridge_model is None or scaler is None:
        error_msg = "Model or scaler is not loaded. Please check server logs."
        if request.is_json:
            return jsonify({"status": "error", "message": error_msg}), 503
        return render_template("home.html", error=error_msg), 503

    # Extract input data from JSON or Form
    is_json_req = request.is_json
    input_source = request.get_json() if is_json_req else request.form

    parsed_values, error_msg = parse_and_validate_inputs(input_source)
    if error_msg:
        if is_json_req:
            return jsonify({"status": "error", "message": error_msg}), 400
        return render_template("home.html", error=error_msg, form_data=input_source), 400

    try:
        # Construct DataFrame to maintain feature names and avoid scikit-learn warnings
        input_df = pd.DataFrame([[parsed_values[f] for f in FEATURE_NAMES]], columns=FEATURE_NAMES)
        new_data_scaled = scaler.transform(input_df)
        prediction = ridge_model.predict(new_data_scaled)
        fwi_result = round(float(prediction[0]), 2)
        risk_level, risk_class = get_fire_risk_level(fwi_result)

        if is_json_req:
            return jsonify({
                "status": "success",
                "fwi": fwi_result,
                "risk_level": risk_level,
                "inputs": parsed_values
            }), 200

        return render_template(
            "home.html",
            results=fwi_result,
            risk_level=risk_level,
            risk_class=risk_class,
            form_data=input_source
        )

    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        err_response = f"An error occurred during prediction: {str(e)}"
        if is_json_req:
            return jsonify({"status": "error", "message": err_response}), 500
        return render_template("home.html", error=err_response, form_data=input_source), 500


if __name__ == "__main__":
    debug_mode = os.environ.get("FLASK_DEBUG", "False").lower() in ["true", "1"]
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=debug_mode)
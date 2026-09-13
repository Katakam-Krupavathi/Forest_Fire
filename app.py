import io
import os
import logging
import pickle
import numpy as np
import pandas as pd
from flask import (
    Flask,
    render_template,
    request,
    jsonify,
    Response,
    send_file,
    redirect,
    url_for,
)

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
    """
    Categorizes the Fire Weather Index (FWI) value into Canadian Forest Fire Danger bands:
    - Low: < 5.2
    - Moderate: 5.2 - 11.2
    - High: 11.2 - 21.3
    - Very High: 21.3 - 38.0
    - Extreme: >= 38.0
    """
    if fwi_value < 5.2:
        return "Low", "success"
    elif fwi_value < 11.2:
        return "Moderate", "info"
    elif fwi_value < 21.3:
        return "High", "warning"
    elif fwi_value < 38.0:
        return "Very High", "danger"
    else:
        return "Extreme", "dark"


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


def compute_feature_contributions(scaled_vector):
    """Computes exact linear feature contributions: contribution_i = coef_i * scaled_x_i."""
    if ridge_model is None or not hasattr(ridge_model, "coef_"):
        return []
    contributions = []
    coefs = ridge_model.coef_
    for i, feature in enumerate(FEATURE_NAMES):
        contrib = float(coefs[i] * scaled_vector[0][i])
        contributions.append({
            "feature": feature,
            "weight": round(float(coefs[i]), 4),
            "contribution": round(contrib, 3),
            "impact": "Increases Risk" if contrib > 0 else "Decreases Risk"
        })
    # Sort by absolute contribution magnitude descending
    contributions.sort(key=lambda x: abs(x["contribution"]), reverse=True)
    return contributions


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

    is_json_req = request.is_json
    input_source = request.get_json() if is_json_req else request.form

    parsed_values, error_msg = parse_and_validate_inputs(input_source)
    if error_msg:
        if is_json_req:
            return jsonify({"status": "error", "message": error_msg}), 400
        return render_template("home.html", error=error_msg, form_data=input_source), 400

    try:
        input_df = pd.DataFrame([[parsed_values[f] for f in FEATURE_NAMES]], columns=FEATURE_NAMES)
        new_data_scaled = scaler.transform(input_df)
        prediction = ridge_model.predict(new_data_scaled)
        fwi_result = round(float(prediction[0]), 2)
        risk_level, risk_class = get_fire_risk_level(fwi_result)
        contributions = compute_feature_contributions(new_data_scaled)

        if is_json_req:
            return jsonify({
                "status": "success",
                "fwi": fwi_result,
                "risk_level": risk_level,
                "risk_class": risk_class,
                "inputs": parsed_values,
                "feature_contributions": contributions
            }), 200

        return render_template(
            "home.html",
            results=fwi_result,
            risk_level=risk_level,
            risk_class=risk_class,
            form_data=input_source,
            contributions=contributions
        )

    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        err_response = f"An error occurred during prediction: {str(e)}"
        if is_json_req:
            return jsonify({"status": "error", "message": err_response}), 500
        return render_template("home.html", error=err_response, form_data=input_source), 500


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """Dedicated JSON REST API endpoint for predictions."""
    if ridge_model is None or scaler is None:
        return jsonify({"status": "error", "message": "Model not loaded"}), 503

    if not request.is_json:
        return jsonify({
            "status": "error",
            "message": "Content-Type must be application/json"
        }), 400

    payload = request.get_json() or {}
    parsed_values, error_msg = parse_and_validate_inputs(payload)
    if error_msg:
        return jsonify({"status": "error", "message": error_msg}), 400

    try:
        input_df = pd.DataFrame([[parsed_values[f] for f in FEATURE_NAMES]], columns=FEATURE_NAMES)
        new_data_scaled = scaler.transform(input_df)
        prediction = ridge_model.predict(new_data_scaled)
        fwi_result = round(float(prediction[0]), 2)
        risk_level, risk_class = get_fire_risk_level(fwi_result)
        contributions = compute_feature_contributions(new_data_scaled)

        return jsonify({
            "status": "success",
            "fwi": fwi_result,
            "risk_level": risk_level,
            "risk_class": risk_class,
            "inputs": parsed_values,
            "feature_contributions": contributions
        }), 200
    except Exception as e:
        logger.error(f"API Error: {e}", exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/batch", methods=["GET", "POST"])
def batch_predict():
    """Handles CSV file upload for bulk FWI predictions."""
    if request.method == "GET":
        return render_template("batch.html")

    if ridge_model is None or scaler is None:
        return render_template("batch.html", error="Model not loaded. Please contact server admin."), 503

    if "file" not in request.files:
        return render_template("batch.html", error="No file selected for upload."), 400

    file = request.files["file"]
    if file.filename == "":
        return render_template("batch.html", error="Please choose a CSV file to upload."), 400

    if not file.filename.lower().endswith(".csv"):
        return render_template("batch.html", error="Only CSV files (.csv) are supported."), 400

    try:
        df = pd.read_csv(file)
        df.columns = df.columns.str.strip()

        # Check required columns
        missing_cols = [col for col in FEATURE_NAMES if col not in df.columns]
        if missing_cols:
            return render_template(
                "batch.html",
                error=f"CSV is missing required feature columns: {', '.join(missing_cols)}"
            ), 400

        # Validate numeric conversion
        for col in FEATURE_NAMES:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        if df[FEATURE_NAMES].isnull().any().any():
            return render_template(
                "batch.html",
                error="CSV contains non-numeric or missing values in feature columns."
            ), 400

        # Transform and predict
        scaled_features = scaler.transform(df[FEATURE_NAMES])
        predictions = ridge_model.predict(scaled_features)
        df["Predicted_FWI"] = [round(float(p), 2) for p in predictions]
        df["Risk_Level"] = [get_fire_risk_level(p)[0] for p in df["Predicted_FWI"]]
        df["Risk_Class"] = [get_fire_risk_level(p)[1] for p in df["Predicted_FWI"]]

        # Check if CSV download requested
        if request.form.get("download_csv") == "true":
            output = io.StringIO()
            df.to_csv(output, index=False)
            return Response(
                output.getvalue(),
                mimetype="text/csv",
                headers={"Content-Disposition": "attachment;filename=forest_fire_predictions.csv"}
            )

        results_list = df.to_dict(orient="records")
        return render_template("batch.html", results=results_list, total_count=len(results_list))

    except Exception as e:
        logger.error(f"Batch processing error: {e}", exc_info=True)
        return render_template("batch.html", error=f"Error parsing CSV file: {str(e)}"), 400


@app.route("/batch/sample")
def download_sample_csv():
    """Generates and serves a sample CSV template for batch predictions."""
    sample_data = {
        "Temperature": [32, 28, 35, 26],
        "RH": [55, 65, 40, 75],
        "Ws": [14, 18, 12, 20],
        "Rain": [0.0, 0.2, 0.0, 1.5],
        "FFMC": [86.2, 80.5, 91.0, 68.0],
        "DMC": [16.4, 10.2, 25.8, 5.0],
        "ISI": [5.8, 3.4, 10.2, 1.2],
        "Classes": [1, 0, 1, 0],
        "Region": [0, 0, 1, 1]
    }
    df = pd.DataFrame(sample_data)
    output = io.StringIO()
    df.to_csv(output, index=False)
    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment;filename=sample_weather_batch.csv"}
    )


@app.errorhandler(404)
def not_found_error(error):
    if request.is_json or request.path.startswith("/api/"):
        return jsonify({"status": "error", "message": "Resource not found (404)"}), 404
    return render_template("404.html"), 404


@app.errorhandler(500)
def internal_error(error):
    if request.is_json or request.path.startswith("/api/"):
        return jsonify({"status": "error", "message": "Internal server error (500)"}), 500
    return render_template("500.html"), 500


if __name__ == "__main__":
    debug_mode = os.environ.get("FLASK_DEBUG", "False").lower() in ["true", "1"]
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=debug_mode)
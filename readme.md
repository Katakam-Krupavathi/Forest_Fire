# 🔥 Algerian Forest Fire Weather Index (FWI) Predictor

[![Python Application CI](https://github.com/Katakam-Krupavathi/Forest_Fire/actions/workflows/ci.yml/badge.svg)](https://github.com/Katakam-Krupavathi/Forest_Fire/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A robust Machine Learning web application and API that predicts the **Forest Fire Weather Index (FWI)** using environmental and meteorological observations from the **Algerian Forest Fire Dataset**.

---

## 🚀 Key Features

- **Trained Machine Learning Model**: Uses Ridge Regression with a Standard Scaler pipeline trained on curated meteorological data.
- **Flask Web Application**: User-friendly, responsive interface built with Bootstrap 5.
- **REST API & Form Support**: Supports both HTML form submissions and JSON API endpoints (`/predict` & `/health`).
- **Comprehensive Input Validation**: Robust range and type checking with user-friendly error alerts.
- **Risk Level Categorization**: Dynamically classifies the predicted FWI into risk categories (*Low, Moderate, High, Extreme*).
- **Automated CI/CD**: Automated unit test suite run via GitHub Actions on every push/PR.
- **AWS Elastic Beanstalk Ready**: Pre-configured WSGI deployment via `.ebextensions/python.config`.

---

## 🛠️ Tech Stack

- **Core**: Python 3.10+
- **Machine Learning**: Scikit-Learn, NumPy, Pandas
- **Web Framework**: Flask, Gunicorn
- **Frontend**: HTML5, CSS3, Bootstrap 5
- **Testing & CI**: Pytest, GitHub Actions
- **Deployment**: AWS Elastic Beanstalk / WSGI

---

## 📊 Feature Parameters

| Parameter | Name | Description | Valid Range |
| :--- | :--- | :--- | :--- |
| **Temperature** | `Temperature` | Ambient temperature in Celsius | -10°C to 60°C |
| **Relative Humidity** | `RH` | Relative humidity percentage | 0% to 100% |
| **Wind Speed** | `Ws` | Wind speed in km/h | 0 to 100 km/h |
| **Rainfall** | `Rain` | 24-hour total rainfall in mm | $\ge 0$ mm |
| **FFMC** | `FFMC` | Fine Fuel Moisture Code index | 0 to 105 |
| **DMC** | `DMC` | Duff Moisture Code index | $\ge 0$ |
| **ISI** | `ISI` | Initial Spread Index | $\ge 0$ |
| **Classes** | `Classes` | Fire occurrence status | `0` (Not Fire), `1` (Fire) |
| **Region** | `Region` | Geographic study area | `0` (Bejaia), `1` (Sidi Bel-abbes) |

---

## 📂 Run Locally

### 1. Clone the Repository
```bash
git clone https://github.com/Katakam-Krupavathi/Forest_Fire.git
cd Forest_Fire
```

### 2. Set Up Virtual Environment & Dependencies
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Run Unit Tests
```bash
pytest tests/
```

### 4. Retrain Model (Optional)
```bash
python train_model.py
```

### 5. Start the Application
```bash
python app.py
```
Open your browser and navigate to `http://127.0.0.1:5000/`.

---

## 🌐 API Usage

### Health Check
```bash
curl -X GET http://127.0.0.1:5000/health
```
**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

### Predict FWI (JSON Endpoint)
```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Temperature": 32.0,
    "RH": 55.0,
    "Ws": 16.0,
    "Rain": 0.0,
    "FFMC": 88.5,
    "DMC": 18.2,
    "ISI": 6.8,
    "Classes": 1,
    "Region": 0
  }'
```
**Response:**
```json
{
  "status": "success",
  "fwi": 11.24,
  "risk_level": "Moderate",
  "inputs": { ... }
}
```

---

## ☁️ Deployment

### 1. AWS Elastic Beanstalk
The repository includes `.ebextensions/python.config` configured for WSGI deployment:
```yaml
option_settings:
    "aws:elasticbeanstalk:container:python":
        WSGIPath: app:app
```

### 2. Render / Railway / Production WSGI
To deploy on PaaS providers like Render or Railway, configure the start command:
```bash
gunicorn app:app --bind 0.0.0.0:$PORT
```

---

## 👨‍💻 Author

**Katakam Krupavathi**  
- GitHub: [@Katakam-Krupavathi](https://github.com/Katakam-Krupavathi)

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

## 🏗️ Architecture & ML Pipeline

### 1. Request Flow Architecture
The web application is powered by a Flask backend (`app.py`) that serves both user-facing HTML templates and REST API endpoints. Serialized models and scalers are loaded into memory once upon startup for high-performance, low-latency inference.

```mermaid
flowchart TD
    subgraph Client [User Client / Browser]
        UI[Web Browser / REST Client]
    end

    subgraph Server [Flask Application - app.py]
        Router{Route Handler}
        Val[Input Validator & Range Checker]
        Transform[StandardScaler Pipeline]
        Predictor[Ridge Regression Inference]
        RiskCalc[FWI Risk Level Classifier]
    end

    subgraph Storage [Serialized Artifacts]
        ScalerFile[(models/scaler.pkl)]
        ModelFile[(models/ridge.pkl)]
    end

    subgraph Views [Jinja2 Templates / Responses]
        IndexView[index.html - Welcome Page]
        FormView[home.html - Prediction Form & Results]
        JSONResp[JSON API Response]
    end

    UI -->|GET /| Router
    Router -->|Render| IndexView

    UI -->|GET /predict| Router
    Router -->|Render Empty Form| FormView

    UI -->|POST /predict + 9 Features| Router
    Router --> Val
    Val -->|Validation Error| FormView
    Val -->|Validated Data| Transform

    ScalerFile -.->|Load on Startup| Transform
    ModelFile -.->|Load on Startup| Predictor

    Transform -->|Standardized Vector| Predictor
    Predictor -->|Raw FWI Value| RiskCalc
    RiskCalc -->|HTML Request| FormView
    RiskCalc -->|JSON Request| JSONResp
```

### 2. Machine Learning Training Pipeline
The offline training workflow cleans raw meteorological observations, merges regional records, reduces multicollinearity, scales continuous features, and tests multiple regularized regression algorithms:

```mermaid
flowchart TD
    A[Algerian Forest Fire Raw Dataset - CSV] --> B[Data Cleaning & Header Standardization]
    B --> C[Region Labeling: Bejaia=0, Sidi Bel-abbes=1]
    C --> D[Exploratory Data Analysis & Correlation Heatmap]
    
    D --> E[Multicollinearity Reduction: Drop BUI & DC correlation > 0.85]
    E --> F[Selected 9 Features: Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region]
    
    F --> G[Train / Test Split - 75% Train / 25% Test, random_state=42]
    G --> H[StandardScaler - Fit on Train, Transform Test]
    
    H --> I[Model Exploration: Linear, Lasso, Ridge, ElasticNet]
    I --> J[Evaluate Metrics: R2 Score, MAE, RMSE]
    J --> K[Model Selection: Ridge Regression selected for top R2 ~ 0.984 & lowest MAE]
    
    K --> L[Serialize: models/scaler.pkl & models/ridge.pkl]
    L --> M[Production Ingestion: Loaded by Flask at Startup]
```

### 3. Why Ridge Regression?
During model exploration across candidate algorithms (Ordinary Least Squares Linear Regression, Lasso, Ridge, and ElasticNet), **Ridge Regression ($L_2$ Regularization)** was selected for several key reasons:

1. **Multicollinearity Management**: Meteorological indicators (FFMC, DMC, and ISI) have strong mutual correlation. Ordinary Least Squares is prone to inflated parameter variance when collinearity is present.
2. **Smooth Coefficient Shrinkage vs. Feature Elimination**: While Lasso ($L_1$ Regularization) forces coefficients strictly to zero, continuous meteorological variations (relative humidity, wind speed, moisture codes) all provide valuable predictive signals. Discarding features led to higher prediction error.
3. **Generalization Performance**: Ridge regression applies quadratic shrinkage on model weights ($\|\beta\|_2^2$), penalizing large coefficients without removing informative features. It achieved the highest test determination score ($R^2 \approx 0.9843$) and lowest Mean Absolute Error ($\text{MAE} \approx 0.5642$) on unseen validation data.

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

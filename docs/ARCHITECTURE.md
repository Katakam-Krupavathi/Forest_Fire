# System Architecture & Machine Learning Workflow

This document details the system design, request lifecycle, machine learning pipeline, and algorithmic rationale behind the **Algerian Forest Fire Weather Index (FWI) Predictor**.

---

## 1. Request Flow Architecture

The web application is powered by a Flask backend that serves both user-facing HTML templates and REST API endpoints. Models are loaded into memory once upon application initialization for low-latency inference.

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

---

## 2. Machine Learning Training Pipeline

The offline training workflow processes raw meteorological and fire observation records, cleans noise and structural anomalies, scales continuous features, and tests multiple regularized regression algorithms.

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

---

## 3. Why Ridge Regression?

During model exploration across multiple candidate algorithms (Ordinary Least Squares Linear Regression, Lasso, Ridge, and ElasticNet), **Ridge Regression (L2 Regularization)** was selected as the production model for several reasons:

1. **Multicollinearity Management**: The Canadian Forest Fire Weather Index (FWI) system features interrelated indicators (such as FFMC, DMC, and ISI). Ordinary Least Squares is vulnerable to inflated variance when collinearity exists among meteorological features.
2. **Smooth Coefficient Shrinkage vs. Feature Elimination**: While Lasso (L1 Regularization) forces feature coefficients to absolute zero, in this domain, subtle signals from relative humidity, wind speed, and fuel moisture indices all contribute predictive power. Setting coefficients to zero discarded informative variance and reduced test accuracy.
3. **Generalization Performance**: Ridge regression applies quadratic shrinkage on model weights ($\|\beta\|_2^2$), constraining coefficient magnitude without eliminating predictors. This balance produced the highest test coefficient of determination ($R^2 \approx 0.9843$) and the lowest Mean Absolute Error ($\text{MAE} \approx 0.5642$) on unseen validation data.

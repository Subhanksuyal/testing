# Industrial Thermal Anomaly & Risk Intelligence System (SIH)

**Real-time satellite-based thermal anomaly monitoring and industrial proximity risk intelligence.**

Built for the **Smart India Hackathon (SIH)**.

---

## 📌 1. Project Objective

The **Industrial Thermal Anomaly & Risk Intelligence System** is an end-to-end real-time geospatial ML platform designed to monitor active thermal anomalies across India using live NASA satellite telemetry, enrich them with OpenStreetMap (OSM) industrial and forest geographic infrastructure, and classify potential industrial-associated fire risks using a trained Random Forest machine learning pipeline.

---

## 🏗️ 2. System Architecture

```text
                               ┌────────────────────────┐
                               │ NASA FIRMS Satellites  │
                               │ (VIIRS 375m / MODIS)   │
                               └───────────┬────────────┘
                                           │ Live Telemetry API / Open NRT
                                           ▼
                               ┌────────────────────────┐
                               │ Data Ingestion Layer   │
                               │ (utils.py & firms_api) │
                               └───────────┬────────────┘
                                           │ Normalized Coordinates
                                           ▼
┌─────────────────────────┐    ┌────────────────────────┐
│ OpenStreetMap / Overpass│───▶│ GIS Enrichment Engine  │
│ (Industrial & Forest)   │    │ (cKDTree Spatial Index)│
└─────────────────────────┘    └───────────┬────────────┘
                                           │ Spatial Distances (km)
                                           ▼
                               ┌────────────────────────┐
                               │ Feature Engineering    │
                               │ (features.py)          │
                               └───────────┬────────────┘
                                           │ Persistence, Night Flag, Proximity Scores
                                           ▼
                               ┌────────────────────────┐
                               │ Random Forest Pipeline │
                               │ (Imputer+Scaler+Forest)│
                               └───────────┬────────────┘
                                           │ Risk Probability & Bands
                                           ▼
                               ┌────────────────────────┐
                               │ Streamlit Dashboard    │
                               │ (Folium Map + Plotly)  │
                               └────────────────────────┘
```

---

## 🛰️ 3. Data Sources & Integration

1. **NASA FIRMS (Fire Information for Resource Management System)**
   - **Satellites**: VIIRS Suomi NPP (375m resolution), VIIRS NOAA-20 (375m), VIIRS NOAA-21 (375m), and MODIS Terra/Aqua (1km).
   - **Endpoints**: Official NASA FIRMS Area and Country CSV APIs with fallback to NASA FIRMS Open Near-Real-Time South Asia feeds.
   - **Extracted Attributes**: `latitude`, `longitude`, `frp` (Fire Radiative Power in MW), `confidence`, `acq_date`, `acq_time`, `daynight`, `satellite`.

2. **OpenStreetMap / Overpass API**
   - **Infrastructure Queried**: `landuse=industrial`, `power=plant`, `industrial=*`, `landuse=forest`, `natural=wood`.
   - **Spatial Optimization**: Regional bounding-box queries, disk/memory caching, and sub-millisecond local nearest-neighbor search (`scipy.spatial.cKDTree` on 3D spherical coordinates) to eliminate 429 rate limits.

---

## 🔑 4. NASA FIRMS API Key Setup

### Obtaining a Free Key:
1. Visit the [NASA FIRMS API Key Request Portal](https://firms.modaps.eosdis.nasa.gov/api/map_key/).
2. Enter your email address to receive your unique `MAP_KEY`.

### Configuring Locally:
Create a `.env` file in the project root:
```env
FIRMS_API_KEY=your_nasa_firms_map_key_here
```

### Configuring for Streamlit Cloud:
In your Streamlit Cloud App dashboard:
1. Go to **Settings > Secrets**.
2. Enter:
```toml
FIRMS_API_KEY = "your_nasa_firms_map_key_here"
```

> **Note:** If no key is configured, the system automatically falls back to the open NASA FIRMS South Asia 24h feed without crashing.

---

## 🚀 5. Local Setup & Installation

### Prerequisites:
- Python 3.9+ (Python 3.10, 3.11, 3.12, 3.13 supported)

### Step 1: Clone or Navigate to the Repository
```bash
cd industrial_thermal_sih
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Run Automated Test Suite
```bash
python -m pytest tests/ -v
```

### Step 4: Train the Machine Learning Model
```bash
python src/train_model.py
```

### Step 5: Launch the Streamlit Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## ☁️ 6. Streamlit Cloud Deployment Guide

1. Push this repository to GitHub.
2. Log in to [Streamlit Cloud](https://share.streamlit.io/).
3. Click **"New App"** and select your repository, branch, and set main file path to `app.py` (or `industrial_thermal_sih/app.py`).
4. In **App settings > Secrets**, configure:
   ```toml
   FIRMS_API_KEY = "your_nasa_firms_map_key_here"
   ```
5. Click **"Deploy"**.

---

## 🤖 7. Machine Learning Model & Threshold Optimization

- **Algorithm**: `RandomForestClassifier` (150 estimators, balanced class weights, depth 12).
- **Preprocessing Pipeline**: Integrated `SimpleImputer(strategy='median')` and `StandardScaler()`.
- **Selected Features**:
  - `frp` (Fire Radiative Power)
  - `confidence`
  - `industry_distance_km`
  - `forest_distance_km`
  - `persistence_days`
  - `night_flag`
  - `industry_proximity_score`
  - `forest_proximity_score`
  - `thermal_intensity_score`
  - `confidence_score`
  - `persistence_score`
- **Threshold Decision (`0.30`)**:
  - For disaster warning systems, **Recall (catching actual industrial risks)** is prioritized to avoid critical false negatives.
  - Threshold `0.30` achieves **~74% Recall** with strong precision on holdout evaluation.

### Risk Categorization:
- **Low Risk:** Probability < 0.30
- **Medium Risk:** 0.30 <= Probability < 0.60
- **High Risk:** Probability >= 0.60

---

## ⚠️ 8. ML Limitations & Disclaimers

> **Important Note for SIH Evaluators:**
> This system estimates industrial risk based on thermal anomaly intensity, persistence, and geographic proximity to mapped industrial facilities and forests. It does **not** prove that a thermal event originated from an industrial facility or declare the root cause of a fire.
> Output classifications are termed **"Potential industrial-associated thermal anomalies"** for situational awareness and priority inspection.

---

## 📁 9. Project Directory Structure

```text
industrial_thermal_sih/
├── .env.example
├── .gitignore
├── README.md
├── requirements.txt
├── app.py
├── .streamlit/
│   └── secrets.toml.example
├── data/
│   └── practice_firms.csv
├── models/
│   ├── industrial_risk_model.joblib
│   └── model_metadata.json
├── src/
│   ├── __init__.py
│   ├── firms_api.py
│   ├── osm.py
│   ├── features.py
│   ├── model.py
│   ├── train_model.py
│   └── utils.py
├── cache/
│   └── .gitkeep
└── tests/
    ├── test_data_validation.py
    └── test_features.py
```

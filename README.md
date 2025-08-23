# Trip Duration Prediction

## Goal

Predict trip duration (seconds) using features like pickup/dropoff coordinates, start time, and passenger count.  
Target variable: `trip_duration` (in seconds).

## 🔰 Start Here (Open in this order)

1. This file `README.md` – 2-minute overview.  
2. EDA notebook: `notebooks/01_Trip Duration EDA.ipynb`  
   Run top-to-bottom. It loads data from `data/raw/` and performs sanity checks, profiling, and first plots.  
3. (Optional) Feature engineering notebook: `notebooks/02_feature_engineering.ipynb`  
   Uses helper functions from `src/features/` to create distance & time features, saves to `data/processed/`.  
4. (Optional) Modeling notebook: `notebooks/03_modeling.ipynb`  
   Trains baseline models and reports metrics.  

> If your data file names/paths differ, edit the path cell at the top of each notebook.

## 📦 Project Structure

trip-duration-prediction/  
├── config/  
│   └── params.yaml  
├── data/  
│   ├── raw/  
│   ├── processed/  
│   └── external/  
├── notebooks/  
│   └── 01_Trip Duration EDA.ipynb
|   └── 02_feature_engineering.ipynb
|   └── 03_modeling.ipynb
├── src/  
│   ├── data_Utils/  
│   │   └── data_helper_.py  
│   ├── features/  
│   │   ├── distances.py  
│   │   └── timeparts.py  
│   │   └── encoding.py  
│   ├── models/  
│   │   └── KNN.py  
│   │   └── LR.py  
│   │   └── Ridge.py  
│   │   └── XGBoost.py  
├── .gitignore
├── .LICENSE  
├── Makefile  
├── requirements.txt  
└── README.md  

## 📑 Dataset Schema

| Column | Meaning |
|--------|--------|
| id | Unique trip identifier |
| vendor_id | Provider code for the trip |
| pickup_datetime | When meter was engaged |
| dropoff_datetime | When meter was disengaged |
| passenger_count | Number of passengers (driver-entered) |
| pickup_longitude | Longitude at pickup |
| pickup_latitude | Latitude at pickup |
| dropoff_longitude | Longitude at dropoff |
| dropoff_latitude | Latitude at dropoff |
| store_and_fwd_flag | Y if buffered offline and forwarded later; N otherwise |
| trip_duration | Target – duration in seconds |

## 🧰 Setup

Requirements: Python 3.9+, pip, Jupyter, VS Code (optional).  

1. Create & activate a virtual environment:

```text
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

1. Place your raw CSV(s) under `data/raw/` ( `train.csv`).

## ▶️ Reproduce the EDA

1. Open VS Code at project root (`code .`).  
2. Open `notebooks/01_Trip Duration EDA.ipynb`.  
3. Edit config cell if paths differ.  
4. Run all cells to get:  
   - Data shape & schema checks  
   - Missing values/outliers scans  
   - Coordinate sanity  
   - Time-based distributions  
   - Target distribution & log-transform  
   - Initial feature ideas and so on

> Raw data stays untouched; processed outputs go to `data/processed/`.

## 🧪 Planned Feature Engineering

**Distance & geometry:** `haversine_km`, `manhattan_km`, `bearing_deg`, `pickup_dropoff_same_cell`  
**Temporal:** `pickup_hour`, `pickup_dow`, `pickup_month`, `weather`,`is_weekend`, `is_rush_hour`, `is_holiday`  
**Speed proxies:** `approx_speed_kmh = distance_km / (trip_duration_hours)`  

> Functions: `src/features/distances.py` & `src/features/timeparts.py`

## 📂 Project Steps

1. **EDA (Initial)**  
2. **Feature Engineering**  
3. **EDA (with new features)**  
4. **Modeling**  
   - Baseline: Linear Regression, Ridge, KNN, XGBoost  
   - Targets: `trip_duration` vs `log(trip_duration)`  
   - Validation: Time-aware split  
   - Metrics: RMSE / MAE / R2  
   - Error Analysis: Residuals & slice analysis  

## 🔄 Workflow

```bash
# Activate env
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # macOS/Linux

# Pull changes
git pull

# Run EDA
jupyter notebook

# Commit
git add .
git commit -m "eda: add distance features & time breakdown"
git push

# Optional feature branch
git checkout -b feature/fe-distances
git push -u origin feature/fe-distances
```

## 🔐 Data Policy

- Never commit raw/large data (`data/raw/`)  
- Commit only code, configs, small artifacts  
- External sources under `data/external/`

## 🗺️ Roadmap

- Add `02_feature_engineering.ipynb` & `03_modeling.ipynb`  
- Wire `train.py` to pipeline  
- Optional: integrate holidays/weather  
- Unit tests for distances/timeparts  

## 🙋 FAQ

**Dataset location:** `data/raw/`  
**First file to open:** `01_Trip Duration EDA.ipynb`  
**Where do features come from?** Generated via `src/features/`  
**Run without VS Code?** Yes, use Jupyter  

**Issues:** Open a GitHub Issue with steps to reproduce.

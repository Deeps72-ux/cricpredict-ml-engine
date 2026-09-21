# CricPredict: ML Match Analytics & Monte Carlo Simulation Engine

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111+-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML_Models-F7931E.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Gradient_Boosting-red.svg)](https://xgboost.readthedocs.io/)
[![Pandas](https://img.shields.io/badge/Pandas-Data_ETL-150458.svg?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/NumPy-Monte_Carlo-013243.svg?logo=numpy&logoColor=white)](https://numpy.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**CricPredict** is a machine learning and probabilistic simulation platform engineered for T20 cricket analytics. Built with **Python**, **FastAPI**, **Scikit-Learn**, and **NumPy**, it transforms ball-by-ball cricket datasets into real-time win probability curves, batter-vs-bowler historical matchup vectors, venue-adjusted run distributions, and 10,000-iteration stochastic Monte Carlo trajectory simulations.

---

## Architecture Overview

```mermaid
flowchart TD
    Data[Ball-by-Ball T20 / IPL Datasets] --> ETL[Pandas & NumPy ETL Pipeline]
    ETL --> Features[Feature Engineering: Match State, Phase, Run-Rate, Wickets, Venue Bias]
    
    subgraph Machine Learning & Inference Engine
        Features --> XGB[XGBoost Win-Probability Classifier]
        Features --> Matchup[Batter vs Bowler Historical Matrix]
        
        XGB --> LiveProb[Live Win Probability Curve Generator]
        Matchup --> ProbDist[Over-by-Over Transition Probability Distributions]
        
        ProbDist --> MonteCarlo[10,000-Iteration Monte Carlo Engine: NumPy Vectorized]
        MonteCarlo --> SimResults[Stochastic Target Reach, Wicket Distributions & Confidence Intervals]
    end

    LiveProb --> API[FastAPI Analytics Service]
    SimResults --> API
    API --> UI([Streamlit Dashboard & React Frontend])
```

---

## Key Features

- **Vectorized Monte Carlo Simulation**: Executes 10,000 parallel match simulations in under 200ms using vectorized **NumPy** operations to project target distributions, optimal batting orders, and chase confidence intervals.
- **Dynamic Win Probability (Live In-Play)**: Evaluates ball-by-ball game states (wickets remaining, required run rate, pitch par scores, match phase) via calibrated **XGBoost** models.
- **Micro-Matchup Historical Embeddings**: Models individual batter vs bowler matchups across bowling styles (e.g. left-arm pace, leg-spin, off-spin) and phases (Powerplay, Middle, Death).
- **Venue & Dew Factor Adjustments**: Factored regressions for venue dimensions, first-innings par scores, and toss decision biases across global T20 venues.
- **High-Performance FastAPI Endpoints**: RESTful microservice designed for low-latency queries during live broadcast or decision-support scenarios.

---

## Tech Stack

| Layer | Technologies |
| :--- | :--- |
| **API & Backend** | Python 3.11, FastAPI, Uvicorn, Pydantic |
| **Data Science & ETL** | Pandas, NumPy, SciPy (Optimization & Distributions) |
| **Machine Learning** | XGBoost, LightGBM, Scikit-Learn, Joblib |
| **Visualization & UI** | Streamlit, Plotly, React / Modern CSS |
| **Storage & Data** | PostgreSQL, Parquet (Fast analytical reads), Redis (Cache) |
| **DevOps** | Docker, Docker Compose, GitHub Actions |

---

## Project Structure

```text
cricpredict-ml-engine/
├── app/
│   ├── api/
│   │   ├── v1/
│   │   │   ├── predict.py           # Win probability & target projections
│   │   │   ├── simulate.py          # Monte Carlo simulation triggers
│   │   │   └── matchups.py          # Player head-to-head metrics
│   │   └── router.py
│   ├── core/
│   │   ├── config.py
│   │   └── database.py
│   ├── etl/
│   │   ├── loader.py                # Cricsheet JSON/CSV parser
│   │   └── feature_engineering.py   # Rolling averages & pressure metrics
│   ├── ml/
│   │   ├── train.py                 # XGBoost training pipeline
│   │   ├── model_loader.py          # Serialized model inference
│   │   └── monte_carlo.py           # Vectorized stochastic simulator
│   └── main.py                      # FastAPI App
├── models/                          # Pretrained .joblib weights
├── notebooks/                       # Exploratory Data Analysis & backtests
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## Quickstart

### 1. Installation

```bash
git clone https://github.com/Deeps72-ux/cricpredict-ml-engine.git
cd cricpredict-ml-engine

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Run API & Dashboard

```bash
# Start FastAPI backend
uvicorn app.main:app --port 8000 --reload

# Start Streamlit UI
streamlit run ui/dashboard.py --server.port 8501
```

---

## API Endpoints

- `POST /api/v1/predict/win-probability`: Calculate current live win probability given match state.
- `POST /api/v1/simulate/chase`: Run 10,000 Monte Carlo iterations for target chase.
- `GET /api/v1/matchup/{batter_id}/{bowler_id}`: Retrieve historical strike-rate, dot-ball %, and dismissal risk.

---

## License

MIT License.

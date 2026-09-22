"""Exploratory Data Analysis and Model Backtesting script for CricPredict."""

import pandas as pd
import numpy as np
from app.etl.loader import load_deliveries_dataset
from app.etl.feature_engineering import MatchFeatureEngineer
from app.ml.model_loader import get_model_manager

def main():
    print("Loading IPL deliveries for EDA...")
    df = load_deliveries_dataset()
    print(f"Total deliveries: {len(df):,}")
    print(f"Total matches: {df['match_id'].nunique():,}")
    print(f"Seasons: {sorted(df['season'].unique().tolist())}")
    
    print("\nTop 5 Venues:")
    print(df['venue'].value_counts().head(5))
    
    print("\nEvaluating win probability model...")
    mgr = get_model_manager()
    sample_state = {
        "target_runs": 185,
        "current_score": 110,
        "overs_completed": 12.0,
        "wickets_fallen": 2,
        "venue": "Wankhede Stadium, Mumbai"
    }
    pred = mgr.predict_win_probability(**sample_state)
    print("Sample prediction:", pred)

if __name__ == "__main__":
    main()

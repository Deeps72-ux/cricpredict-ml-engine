"""Training pipeline for XGBoost T20 win probability classifier."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from app.core.config import settings
from app.etl.feature_engineering import MatchFeatureEngineer
from app.etl.loader import load_deliveries_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def train_win_probability_model(
    model_output_path: Optional[Path] = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Train an XGBoost classifier on historical IPL 2nd innings chase deliveries.
    Saves model artifact with preprocessor and metadata to models/ directory.
    """
    output_path = model_output_path or settings.model_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Loading IPL ball-by-ball dataset for training...")
    df = load_deliveries_dataset()
    logger.info("Computing cumulative features...")
    processed_df = MatchFeatureEngineer.compute_cumulative_features(df)

    logger.info("Extracting chasing feature matrix (X) and win target (y)...")
    X, y = MatchFeatureEngineer.extract_training_dataset(processed_df)

    # Define feature groups
    numeric_features = [
        "runs_needed",
        "balls_remaining",
        "wickets_fallen",
        "wickets_remaining",
        "current_run_rate",
        "required_run_rate",
        "pressure_index",
    ]
    categorical_features = ["match_phase", "venue"]

    logger.info("Feature columns: Numeric=%s, Categorical=%s", numeric_features, categorical_features)
    logger.info("Total training samples: %d (Class 1 ratio: %.2f%%)", len(X), y.mean() * 100)

    # Train / Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Preprocessing Pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
        ]
    )

    # XGBoost Classifier
    xgb = XGBClassifier(
        n_estimators=180,
        max_depth=5,
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=0.1,
        eval_metric="logloss",
        random_state=random_state,
        n_jobs=-1,
    )

    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", xgb)])

    logger.info("Fitting XGBoost win probability pipeline...")
    pipeline.fit(X_train, y_train)

    # Evaluate
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)

    acc = float(accuracy_score(y_test, y_pred))
    loss = float(log_loss(y_test, y_pred_proba))
    roc_auc = float(roc_auc_score(y_test, y_pred_proba))
    brier = float(brier_score_loss(y_test, y_pred_proba))

    logger.info("--- Model Validation Metrics ---")
    logger.info("Accuracy:     %.4f", acc)
    logger.info("ROC-AUC:      %.4f", roc_auc)
    logger.info("Log Loss:     %.4f", loss)
    logger.info("Brier Score:  %.4f", brier)

    # Prepare artifact
    artifact = {
        "pipeline": pipeline,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "feature_names": numeric_features + categorical_features,
        "metrics": {
            "accuracy": acc,
            "roc_auc": roc_auc,
            "log_loss": loss,
            "brier_score": brier,
        },
        "trained_at": datetime.utcnow().isoformat(),
        "version": settings.VERSION,
        "classes": [0, 1],  # 0: Bowling team win, 1: Batting team win
    }

    joblib.dump(artifact, output_path)
    logger.info("Model artifact successfully saved to %s", output_path)
    return artifact


if __name__ == "__main__":
    train_win_probability_model()

"""Model loader and inference manager for CricPredict win probability classifier."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

from app.core.config import settings
from app.etl.feature_engineering import MatchFeatureEngineer

logger = logging.getLogger(__name__)


class ModelManager:
    """Manages serialized model artifact lifecycle and real-time inference."""

    _instance: Optional["ModelManager"] = None

    def __init__(self, model_path: Optional[Path] = None):
        self.model_path = model_path or settings.model_path
        self.artifact: Optional[Dict[str, Any]] = None
        self.pipeline: Optional[Any] = None
        self.load_or_train()

    @classmethod
    def get_instance(cls) -> "ModelManager":
        """Singleton accessor."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def load_or_train(self) -> None:
        """Load trained pipeline from disk, or train fresh if artifact is missing."""
        if self.model_path.exists():
            try:
                logger.info("Loading model artifact from %s", self.model_path)
                self.artifact = joblib.load(self.model_path)
                self.pipeline = self.artifact["pipeline"]
                logger.info("Model loaded successfully (Version: %s)", self.artifact.get("version"))
                return
            except Exception as e:
                logger.warning("Failed to load model from %s (%s). Re-training...", self.model_path, e)

        # Train new model
        from app.ml.train import train_win_probability_model

        logger.info("Training new win probability model...")
        self.artifact = train_win_probability_model(self.model_path)
        self.pipeline = self.artifact["pipeline"]

    def predict_win_probability(
        self,
        target_runs: int,
        current_score: int,
        overs_completed: float,
        wickets_fallen: int,
        venue: str,
        dew_factor: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Compute live win probability for chasing batting team vs defending bowling team.
        Applies deterministic cricket boundary conditions before ML evaluation.
        """
        # Feature extraction
        feature_dict = MatchFeatureEngineer.build_feature_dict(
            target_runs=target_runs,
            current_score=current_score,
            overs_completed=overs_completed,
            wickets_fallen=wickets_fallen,
            venue=venue,
        )

        runs_needed = feature_dict["runs_needed"]
        balls_remaining = feature_dict["balls_remaining"]
        wickets = feature_dict["wickets_fallen"]

        # Deterministic Boundary Conditions
        # 1. Target reached: Chase successful
        if current_score >= target_runs:
            return {
                "win_probability_batting": 1.0,
                "win_probability_bowling": 0.0,
                "feature_dict": feature_dict,
                "is_boundary_case": True,
                "boundary_reason": "Target already reached",
            }

        # 2. All 10 wickets fallen: Bowling team wins
        if wickets >= 10:
            return {
                "win_probability_batting": 0.0,
                "win_probability_bowling": 1.0,
                "feature_dict": feature_dict,
                "is_boundary_case": True,
                "boundary_reason": "All wickets fallen",
            }

        # 3. Overs exhausted:
        if balls_remaining <= 0:
            if current_score == target_runs - 1:
                # Tie match
                return {
                    "win_probability_batting": 0.5,
                    "win_probability_bowling": 0.5,
                    "feature_dict": feature_dict,
                    "is_boundary_case": True,
                    "boundary_reason": "Match tied (scores level at over limit)",
                }
            else:
                return {
                    "win_probability_batting": 0.0,
                    "win_probability_bowling": 1.0,
                    "feature_dict": feature_dict,
                    "is_boundary_case": True,
                    "boundary_reason": "Overs exhausted without reaching target",
                }

        # 4. Mathematically impossible chase: (e.g. need 37 runs off 6 balls)
        max_possible_runs = balls_remaining * 6
        if runs_needed > max_possible_runs:
            return {
                "win_probability_batting": 0.0001,
                "win_probability_bowling": 0.9999,
                "feature_dict": feature_dict,
                "is_boundary_case": True,
                "boundary_reason": "Required runs exceed theoretical boundary maximum",
            }

        # ML Model Inference
        df_input = pd.DataFrame([feature_dict])

        # Raw probabilities: class 0 = bowling win, class 1 = batting win
        proba = self.pipeline.predict_proba(df_input)[0]
        prob_bowling = float(proba[0])
        prob_batting = float(proba[1])

        # Dew factor adjustment (dew makes ball slicker, favoring chasing batters)
        if dew_factor > 0:
            boost = min(0.12, float(dew_factor) * 0.08)
            prob_batting = min(0.995, prob_batting + boost)
            prob_bowling = 1.0 - prob_batting

        # Bound check
        prob_batting = round(max(0.001, min(0.999, prob_batting)), 4)
        prob_bowling = round(1.0 - prob_batting, 4)

        return {
            "win_probability_batting": prob_batting,
            "win_probability_bowling": prob_bowling,
            "feature_dict": feature_dict,
            "is_boundary_case": False,
            "boundary_reason": None,
        }


def get_model_manager() -> ModelManager:
    """Convenience getter for singleton."""
    return ModelManager.get_instance()

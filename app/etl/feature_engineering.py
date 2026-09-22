"""Feature engineering pipeline for T20 match states and live win probability."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from app.etl.loader import normalize_venue

logger = logging.getLogger(__name__)

# Canonical phase names
PHASE_POWERPLAY = "powerplay"
PHASE_MIDDLE = "middle"
PHASE_DEATH = "death"


def get_match_phase(over: float) -> str:
    """
    Determine match phase from overs completed (0-indexed).
    - Powerplay: 0 to 5.6 (overs < 6)
    - Middle: 6 to 14.6 (6 <= overs < 15)
    - Death: 15 to 19.6 (overs >= 15)
    """
    if over < 6.0:
        return PHASE_POWERPLAY
    elif over < 15.0:
        return PHASE_MIDDLE
    else:
        return PHASE_DEATH


def calculate_crr(current_score: float, overs_completed: float) -> float:
    """Calculate Current Run Rate (CRR)."""
    if overs_completed <= 0:
        return 0.0
    return round(float(current_score) / float(overs_completed), 3)


def calculate_rrr(runs_needed: float, overs_remaining: float) -> float:
    """Calculate Required Run Rate (RRR)."""
    if overs_remaining <= 0:
        return 36.0 if runs_needed > 0 else 0.0
    return round(float(runs_needed) / float(overs_remaining), 3)


class MatchFeatureEngineer:
    """Extracts, computes, and standardizes features for match state analytics."""

    # Top venues to retain individually before bucketing into 'Other'
    TOP_VENUES = [
        "Wankhede Stadium, Mumbai",
        "M Chinnaswamy Stadium, Bengaluru",
        "Eden Gardens, Kolkata",
        "MA Chidambaram Stadium, Chennai",
        "Narendra Modi Stadium, Ahmedabad",
        "Arun Jaitley Stadium, Delhi",
        "Rajiv Gandhi International Stadium, Hyderabad",
        "PCA Stadium, Mohali",
        "Sawai Mansingh Stadium, Jaipur",
        "Dubai International Cricket Stadium",
    ]

    FEATURE_COLUMNS = [
        "runs_needed",
        "balls_remaining",
        "wickets_fallen",
        "wickets_remaining",
        "current_run_rate",
        "required_run_rate",
        "pressure_index",
        "match_phase",
        "venue",
    ]

    @classmethod
    def clean_venue(cls, venue: str) -> str:
        """Normalize venue string and group rare venues."""
        norm = normalize_venue(venue)
        if norm in cls.TOP_VENUES:
            return norm
        return "Other Venue"

    @classmethod
    def compute_cumulative_features(cls, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute rolling and cumulative match features for ball-by-ball DataFrame.
        Focuses on 2nd innings (chasing team) for win-probability training.
        """
        logger.info("Computing cumulative match features across %d deliveries...", len(df))
        df = df.copy()

        # Ensure correct sort order
        df = df.sort_values(by=["match_id", "innings", "over", "ball_in_over"]).reset_index(drop=True)

        # Standardize venue
        df["venue_clean"] = df["venue"].apply(cls.clean_venue)

        # Calculate balls completed
        df["balls_completed"] = df["over"] * 6 + df["ball_in_over"]
        df["balls_remaining"] = np.maximum(0, 120 - df["balls_completed"])
        df["overs_completed"] = (df["balls_completed"] / 6.0).round(2)
        df["overs_remaining"] = (df["balls_remaining"] / 6.0).round(2)

        # Match phase
        df["match_phase"] = df["over"].apply(get_match_phase)

        # Group by match and innings to compute cumulative runs and wickets
        grouped = df.groupby(["match_id", "innings"])
        df["current_score"] = grouped["runs_total"].cumsum()
        df["wickets_fallen"] = grouped["is_wicket"].cumsum().clip(upper=10)
        df["wickets_remaining"] = 10 - df["wickets_fallen"]

        # 2nd innings chase calculations
        df["runs_needed"] = np.maximum(0, df["target_runs"] - df["current_score"])

        # Run rates
        df["current_run_rate"] = np.where(
            df["balls_completed"] > 0,
            (df["current_score"] / df["balls_completed"]) * 6.0,
            0.0,
        ).round(2)

        df["required_run_rate"] = np.where(
            df["balls_remaining"] > 0,
            (df["runs_needed"] / df["balls_remaining"]) * 6.0,
            np.where(df["runs_needed"] > 0, 36.0, 0.0),
        ).round(2)

        # Pressure index = RRR / (CRR + 0.1)
        df["pressure_index"] = (df["required_run_rate"] / (df["current_run_rate"] + 0.1)).clip(0, 10).round(3)

        # Win label: 1 if batting_team matches match_winner
        if "match_winner" in df.columns:
            df["is_winner"] = (df["batting_team"] == df["match_winner"]).astype(int)
        else:
            df["is_winner"] = 0

        return df

    @classmethod
    def extract_training_dataset(cls, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Filter 2nd innings deliveries and return (X, y) feature matrix and labels.
        """
        # Only 2nd innings represents the chase with fixed target
        chase_df = df[df["innings"] == 2].copy()

        # Clean target: remove abandoned matches where target_runs <= 1
        chase_df = chase_df[chase_df["target_runs"] > 20]

        # Compute cumulative features if not already computed
        if "runs_needed" not in chase_df.columns:
            chase_df = cls.compute_cumulative_features(chase_df)

        X = chase_df[
            [
                "runs_needed",
                "balls_remaining",
                "wickets_fallen",
                "wickets_remaining",
                "current_run_rate",
                "required_run_rate",
                "pressure_index",
                "match_phase",
                "venue_clean",
            ]
        ].rename(columns={"venue_clean": "venue"})

        y = chase_df["is_winner"]
        return X, y

    @classmethod
    def build_feature_dict(
        cls,
        target_runs: int,
        current_score: int,
        overs_completed: float,
        wickets_fallen: int,
        venue: str,
    ) -> Dict[str, Union[int, float, str]]:
        """
        Build a single row feature dictionary from an instantaneous match state.
        Handles conversions and validations.
        """
        # Convert overs_completed (e.g. 10.3) to balls completed
        full_overs = int(overs_completed)
        balls_in_curr_over = int(round((overs_completed - full_overs) * 10))
        balls_completed = min(120, full_overs * 6 + min(6, balls_in_curr_over))
        balls_remaining = max(0, 120 - balls_completed)

        overs_comp_float = balls_completed / 6.0
        overs_rem_float = balls_remaining / 6.0

        wickets_fallen = min(10, max(0, int(wickets_fallen)))
        wickets_remaining = 10 - wickets_fallen

        runs_needed = max(0, int(target_runs) - int(current_score))
        crr = calculate_crr(current_score, overs_comp_float)
        rrr = calculate_rrr(runs_needed, overs_rem_float)
        pressure_index = round(rrr / (crr + 0.1), 3) if crr >= 0 else 1.0
        match_phase = get_match_phase(full_overs)
        cleaned_venue = cls.clean_venue(venue)

        return {
            "runs_needed": runs_needed,
            "balls_remaining": balls_remaining,
            "wickets_fallen": wickets_fallen,
            "wickets_remaining": wickets_remaining,
            "current_run_rate": crr,
            "required_run_rate": rrr,
            "pressure_index": min(10.0, pressure_index),
            "match_phase": match_phase,
            "venue": cleaned_venue,
        }

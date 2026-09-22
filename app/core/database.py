"""In-memory and cached analytical database for match data and player matchups."""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from app.etl.loader import load_deliveries_dataset, normalize_venue

logger = logging.getLogger(__name__)


def normalize_player_name(name: str) -> str:
    """Normalize player name for flexible matching (removes dots, lowercase, trim)."""
    if not name:
        return ""
    cleaned = re.sub(r"[.\-_]", " ", name).lower()
    return " ".join(cleaned.split())


class MatchAnalyticsDatabase:
    """Analytical data store managing deliveries and precomputed head-to-head matchups."""

    _instance: Optional["MatchAnalyticsDatabase"] = None

    def __init__(self, deliveries_df: Optional[pd.DataFrame] = None):
        self.df = deliveries_df if deliveries_df is not None else load_deliveries_dataset()
        self._initialize_caches()

    @classmethod
    def get_instance(cls) -> "MatchAnalyticsDatabase":
        """Singleton accessor."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _initialize_caches(self) -> None:
        """Build player lookup tables and unique entity registries."""
        logger.info("Initializing player lookup index from %d deliveries...", len(self.df))

        # Unique entities
        self.batters: List[str] = sorted(self.df["batter"].dropna().unique().tolist())
        self.bowlers: List[str] = sorted(self.df["bowler"].dropna().unique().tolist())
        self.venues: List[str] = sorted(self.df["venue"].dropna().unique().tolist())
        all_teams = set(self.df["batting_team"].dropna().unique()).union(
            set(self.df["bowling_team"].dropna().unique())
        )
        self.teams: List[str] = sorted(list(all_teams))

        # Normalized lookup maps
        self.batter_map: Dict[str, str] = {normalize_player_name(b): b for b in self.batters}
        self.bowler_map: Dict[str, str] = {normalize_player_name(b): b for b in self.bowlers}

    def resolve_batter(self, query: str) -> Optional[str]:
        """Resolve query string to official batter name."""
        norm = normalize_player_name(query)
        if norm in self.batter_map:
            return self.batter_map[norm]
        # Partial match
        for key, canonical in self.batter_map.items():
            if norm in key or key in norm:
                return canonical
        return None

    def resolve_bowler(self, query: str) -> Optional[str]:
        """Resolve query string to official bowler name."""
        norm = normalize_player_name(query)
        if norm in self.bowler_map:
            return self.bowler_map[norm]
        for key, canonical in self.bowler_map.items():
            if norm in key or key in norm:
                return canonical
        return None

    def get_matchup_stats(self, batter_query: str, bowler_query: str) -> Optional[Dict[str, Any]]:
        """
        Calculate historical head-to-head metrics for batter vs bowler.
        Returns None if no deliveries recorded between the pair.
        """
        canonical_batter = self.resolve_batter(batter_query) or batter_query
        canonical_bowler = self.resolve_bowler(bowler_query) or bowler_query

        # Filter deliveries where batter faces bowler
        sub = self.df[(self.df["batter"] == canonical_batter) & (self.df["bowler"] == canonical_bowler)].copy()

        if sub.empty:
            return None

        # Exclude wides for legal balls faced
        # In cricsheet, extras can be wides. If 'wides' column present, use it; otherwise check runs_extras
        if "wides" in sub.columns:
            legal_deliveries = sub[sub["wides"].isna() | (sub["wides"] == 0)]
        else:
            legal_deliveries = sub

        balls_faced = len(legal_deliveries)
        if balls_faced == 0:
            balls_faced = len(sub)

        runs_scored = int(sub["runs_batter"].sum())
        # Dismissals where player_dismissed is the batter and not a run-out
        dismissals_df = sub[
            (sub["player_dismissed"] == canonical_batter)
            & (~sub["dismissal_type"].isin(["run out", "retired hurt", "obstructing the field"]))
        ]
        dismissals = len(dismissals_df)

        dot_balls = int((legal_deliveries["runs_batter"] == 0).sum())
        fours = int((sub["runs_batter"] == 4).sum())
        sixes = int((sub["runs_batter"] == 6).sum())
        boundaries = fours + sixes

        strike_rate = round((runs_scored / balls_faced) * 100, 2) if balls_faced > 0 else 0.0
        dot_pct = round((dot_balls / balls_faced) * 100, 2) if balls_faced > 0 else 0.0
        boundary_pct = round((boundaries / balls_faced) * 100, 2) if balls_faced > 0 else 0.0
        batting_avg = round(runs_scored / dismissals, 2) if dismissals > 0 else None

        # Phase-by-phase breakdown
        # Over 0-5 = powerplay, 6-14 = middle, 15-19 = death
        phase_breakdown = {}
        for phase_name, over_range in [
            ("powerplay", range(0, 6)),
            ("middle", range(6, 15)),
            ("death", range(15, 20)),
        ]:
            phase_deliv = legal_deliveries[legal_deliveries["over"].isin(over_range)]
            phase_all = sub[sub["over"].isin(over_range)]
            p_balls = len(phase_deliv)
            p_runs = int(phase_all["runs_batter"].sum())
            p_dismissals = len(
                phase_all[
                    (phase_all["player_dismissed"] == canonical_batter)
                    & (~phase_all["dismissal_type"].isin(["run out"]))
                ]
            )
            p_dots = int((phase_deliv["runs_batter"] == 0).sum())
            p_sr = round((p_runs / p_balls) * 100, 2) if p_balls > 0 else 0.0

            phase_breakdown[phase_name] = {
                "balls": p_balls,
                "runs": p_runs,
                "dismissals": p_dismissals,
                "strike_rate": p_sr,
                "dot_balls": p_dots,
            }

        return {
            "batter": canonical_batter,
            "bowler": canonical_bowler,
            "balls_faced": balls_faced,
            "runs_scored": runs_scored,
            "dismissals": dismissals,
            "strike_rate": strike_rate,
            "dot_ball_percentage": dot_pct,
            "boundary_percentage": boundary_pct,
            "average": batting_avg,
            "fours": fours,
            "sixes": sixes,
            "phase_breakdown": phase_breakdown,
        }

    def get_unique_batters(self) -> List[str]:
        return self.batters

    def get_unique_bowlers(self) -> List[str]:
        return self.bowlers

    def get_venues(self) -> List[str]:
        return self.venues

    def get_teams(self) -> List[str]:
        return self.teams

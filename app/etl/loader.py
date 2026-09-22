"""ETL Pipeline for loading and normalizing IPL ball-by-ball datasets from Cricsheet."""

from __future__ import annotations

import io
import logging
import urllib.request
import zipfile
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from app.core.config import settings

logger = logging.getLogger(__name__)

# Standard venue name normalization dictionary
VENUE_MAP = {
    "Wankhede Stadium": "Wankhede Stadium, Mumbai",
    "Wankhede Stadium, Mumbai": "Wankhede Stadium, Mumbai",
    "M. Chinnaswamy Stadium": "M Chinnaswamy Stadium, Bengaluru",
    "M Chinnaswamy Stadium": "M Chinnaswamy Stadium, Bengaluru",
    "M Chinnaswamy Stadium, Bengaluru": "M Chinnaswamy Stadium, Bengaluru",
    "Eden Gardens": "Eden Gardens, Kolkata",
    "Eden Gardens, Kolkata": "Eden Gardens, Kolkata",
    "MA Chidambaram Stadium, Chepauk": "MA Chidambaram Stadium, Chennai",
    "MA Chidambaram Stadium": "MA Chidambaram Stadium, Chennai",
    "MA Chidambaram Stadium, Chepauk, Chennai": "MA Chidambaram Stadium, Chennai",
    "Narendra Modi Stadium, Ahmedabad": "Narendra Modi Stadium, Ahmedabad",
    "Narendra Modi Stadium": "Narendra Modi Stadium, Ahmedabad",
    "Arun Jaitley Stadium": "Arun Jaitley Stadium, Delhi",
    "Arun Jaitley Stadium, Delhi": "Arun Jaitley Stadium, Delhi",
    "Feroz Shah Kotla": "Arun Jaitley Stadium, Delhi",
    "Rajiv Gandhi International Stadium, Uppal": "Rajiv Gandhi International Stadium, Hyderabad",
    "Rajiv Gandhi International Stadium": "Rajiv Gandhi International Stadium, Hyderabad",
    "Punjab Cricket Association IS Bindra Stadium": "PCA Stadium, Mohali",
    "Punjab Cricket Association Stadium, Mohali": "PCA Stadium, Mohali",
    "Sawai Mansingh Stadium": "Sawai Mansingh Stadium, Jaipur",
    "Sawai Mansingh Stadium, Jaipur": "Sawai Mansingh Stadium, Jaipur",
    "Dubai International Cricket Stadium": "Dubai International Cricket Stadium",
}


def normalize_venue(raw_venue: Optional[str]) -> str:
    """Standardize venue strings across IPL seasons."""
    if not raw_venue or not isinstance(raw_venue, str):
        return "Wankhede Stadium, Mumbai"
    cleaned = raw_venue.strip()
    return VENUE_MAP.get(cleaned, cleaned)


class CricsheetLoader:
    """Manages downloading, parsing, and caching Cricsheet ball-by-ball CSV data."""

    def __init__(self, data_dir: Optional[Path] = None, download_url: Optional[str] = None):
        self.data_dir = data_dir or settings.DATA_DIR
        self.download_url = download_url or settings.CRICSHEET_IPL_URL
        self.sample_csv_path = self.data_dir / "ipl_sample.csv"
        self.parquet_path = self.data_dir / "ipl_deliveries.parquet"

    def download_and_extract(
        self,
        max_matches: Optional[int] = None,
        timeout: int = 40,
    ) -> pd.DataFrame:
        """Download Cricsheet IPL zip archive and consolidate deliveries into a DataFrame."""
        logger.info("Connecting to Cricsheet to download IPL dataset: %s", self.download_url)
        headers = {"User-Agent": "CricPredict-ML-Engine/1.0"}
        req = urllib.request.Request(self.download_url, headers=headers)

        with urllib.request.urlopen(req, timeout=timeout) as response:
            zip_bytes = response.read()

        logger.info("Downloaded %d bytes. Parsing match zip archive...", len(zip_bytes))
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
            file_list = set(z.namelist())

            # 1. Parse all match winners and metadata from info files
            winners: Dict[int, str] = {}
            toss_winners: Dict[int, str] = {}
            toss_decisions: Dict[int, str] = {}
            info_files = [f for f in file_list if f.endswith("_info.csv")]

            for info_file in info_files:
                try:
                    match_id = int(info_file.split("_")[0])
                    lines = z.open(info_file).read().decode("utf-8", errors="ignore").splitlines()
                    for line in lines:
                        parts = [p.strip() for p in line.split(",")]
                        if len(parts) >= 3 and parts[0] == "info":
                            if parts[1] == "winner":
                                winners[match_id] = parts[2]
                            elif parts[1] == "toss_winner":
                                toss_winners[match_id] = parts[2]
                            elif parts[1] == "toss_decision":
                                toss_decisions[match_id] = parts[2]
                except Exception as e:
                    logger.debug("Error parsing info file %s: %s", info_file, e)

            # 2. Read deliveries
            if "all_matches.csv" in file_list:
                df = pd.read_csv(z.open("all_matches.csv"), low_memory=False)
            else:
                match_files = [f for f in file_list if f.endswith(".csv") and not f.endswith("_info.csv")]
                df = pd.concat([pd.read_csv(z.open(f), low_memory=False) for f in match_files], ignore_index=True)

        if df.empty:
            raise ValueError("Parsed empty dataset from Cricsheet archive.")

        # Ensure match_id is integer
        df["match_id"] = pd.to_numeric(df["match_id"], errors="coerce").fillna(0).astype(int)

        # Filter to recent matches if max_matches is set
        if max_matches and max_matches < df["match_id"].nunique():
            recent_ids = df["match_id"].drop_duplicates().tail(max_matches)
            df = df[df["match_id"].isin(recent_ids)].copy()

        # Map metadata
        df["match_winner"] = df["match_id"].map(winners)
        df["toss_winner"] = df["match_id"].map(toss_winners)
        df["toss_decision"] = df["match_id"].map(toss_decisions)

        # Standardize columns
        rename_map = {
            "striker": "batter",
            "runs_off_bat": "runs_batter",
            "extras": "runs_extras",
            "start_date": "date",
            "wicket_type": "dismissal_type",
        }
        df = df.rename(columns=rename_map)

        if "runs_extras" not in df.columns:
            df["runs_extras"] = 0
        df["runs_extras"] = pd.to_numeric(df["runs_extras"], errors="coerce").fillna(0).astype(int)
        df["runs_batter"] = pd.to_numeric(df["runs_batter"], errors="coerce").fillna(0).astype(int)
        df["runs_total"] = df["runs_batter"] + df["runs_extras"]

        # Handle over and ball
        df["ball"] = pd.to_numeric(df["ball"], errors="coerce").fillna(0.0)
        df["over"] = df["ball"].astype(int)
        df["ball_in_over"] = ((df["ball"] - df["over"]) * 10).round().astype(int)

        df["venue"] = df["venue"].apply(normalize_venue)
        df["season"] = df["season"].astype(str)
        df["is_wicket"] = df["player_dismissed"].notna().astype(int)

        # Compute innings targets
        df = self._compute_innings_targets(df)

        self._save_datasets(df)
        return df

    def _compute_innings_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute 1st innings total score and assign target_runs to 2nd innings chase."""
        innings_totals = (
            df.groupby(["match_id", "innings"])["runs_total"]
            .sum()
            .reset_index()
            .rename(columns={"runs_total": "innings_score"})
        )

        first_inn = innings_totals[innings_totals["innings"] == 1].copy()
        first_inn["target_runs"] = first_inn["innings_score"] + 1
        first_inn = first_inn[["match_id", "target_runs"]]

        df = df.merge(first_inn, on="match_id", how="left")
        df["target_runs"] = df["target_runs"].fillna(0).astype(int)
        return df

    def generate_synthetic_sample(self, num_matches: int = 60) -> pd.DataFrame:
        """Generate a realistic synthetic sample IPL dataset for offline usage and tests."""
        logger.info("Generating realistic sample IPL dataset (%d matches)...", num_matches)
        np.random.seed(42)

        teams = [
            "Mumbai Indians",
            "Chennai Super Kings",
            "Royal Challengers Bengaluru",
            "Kolkata Knight Riders",
            "Gujarat Titans",
            "Rajasthan Royals",
            "Sunrisers Hyderabad",
            "Delhi Capitals",
            "Punjab Kings",
            "Lucknow Super Giants",
        ]

        venues = [
            "Wankhede Stadium, Mumbai",
            "M Chinnaswamy Stadium, Bengaluru",
            "Eden Gardens, Kolkata",
            "MA Chidambaram Stadium, Chennai",
            "Narendra Modi Stadium, Ahmedabad",
        ]

        batters = [
            "V Kohli", "RG Sharma", "SA Yadav", "MS Dhoni", "KL Rahul",
            "Shubman Gill", "RR Pant", "HH Pandya", "AD Russell", "SV Samson",
            "F du Plessis", "Q de Kock", "DA Warner", "RA Jadeja", "S Dube",
        ]

        bowlers = [
            "JJ Bumrah", "Rashid Khan", "Mohammed Shami", "YS Chahal", "AR Patel",
            "Kuldeep Yadav", "B Kumar", "TA Boult", "K Rabada", "SP Narine",
            "CV Varun", "Mohammed Siraj", "HV Patel", "Arshdeep Singh", "SN Thakur",
        ]

        rows = []
        for match_idx in range(1, num_matches + 1):
            match_id = 1400000 + match_idx
            team_a, team_b = np.random.choice(teams, size=2, replace=False)
            venue = np.random.choice(venues)
            season = str(np.random.choice([2022, 2023, 2024]))

            # Innings 1
            first_inn_runs = 0
            first_inn_wickets = 0
            first_inn_deliveries = []

            for over in range(20):
                for ball in range(1, 7):
                    if first_inn_wickets >= 10:
                        break
                    batter = np.random.choice(batters)
                    bowler = np.random.choice(bowlers)

                    p_wkt = 0.045 if over < 6 else (0.075 if over >= 15 else 0.04)
                    is_wkt = np.random.rand() < p_wkt
                    if is_wkt:
                        first_inn_wickets += 1
                        runs_bat = 0
                        extras = 0
                        dismissal = np.random.choice(["caught", "bowled", "lbw", "run out"])
                        player_dismissed = batter
                    else:
                        runs_bat = int(np.random.choice([0, 1, 2, 4, 6], p=[0.38, 0.35, 0.09, 0.12, 0.06]))
                        extras = 1 if np.random.rand() < 0.05 else 0
                        dismissal = None
                        player_dismissed = None

                    runs_tot = runs_bat + extras
                    first_inn_runs += runs_tot

                    first_inn_deliveries.append({
                        "match_id": match_id,
                        "season": season,
                        "date": f"{season}-04-15",
                        "venue": venue,
                        "innings": 1,
                        "batting_team": team_a,
                        "bowling_team": team_b,
                        "over": over,
                        "ball_in_over": ball,
                        "ball": float(f"{over}.{ball}"),
                        "batter": batter,
                        "bowler": bowler,
                        "non_striker": np.random.choice([b for b in batters if b != batter]),
                        "runs_batter": runs_bat,
                        "runs_extras": extras,
                        "runs_total": runs_tot,
                        "dismissal_type": dismissal,
                        "player_dismissed": player_dismissed,
                        "is_wicket": int(is_wkt),
                    })

            target_runs = first_inn_runs + 1

            # Innings 2 (Chase)
            chase_runs = 0
            chase_wickets = 0
            second_inn_deliveries = []

            for over in range(20):
                for ball in range(1, 7):
                    if chase_runs >= target_runs or chase_wickets >= 10:
                        break

                    batter = np.random.choice(batters)
                    bowler = np.random.choice(bowlers)

                    req_rate = (target_runs - chase_runs) / max(0.1, (120 - (over * 6 + ball)) / 6.0)
                    p_wkt = 0.05 + (0.015 if req_rate > 10 else 0.0)
                    is_wkt = np.random.rand() < p_wkt

                    if is_wkt:
                        chase_wickets += 1
                        runs_bat = 0
                        extras = 0
                        dismissal = np.random.choice(["caught", "bowled", "lbw", "run out"])
                        player_dismissed = batter
                    else:
                        runs_bat = int(np.random.choice([0, 1, 2, 4, 6], p=[0.35, 0.36, 0.10, 0.13, 0.06]))
                        extras = 1 if np.random.rand() < 0.05 else 0
                        dismissal = None
                        player_dismissed = None

                    runs_tot = runs_bat + extras
                    chase_runs += runs_tot

                    second_inn_deliveries.append({
                        "match_id": match_id,
                        "season": season,
                        "date": f"{season}-04-15",
                        "venue": venue,
                        "innings": 2,
                        "batting_team": team_b,
                        "bowling_team": team_a,
                        "over": over,
                        "ball_in_over": ball,
                        "ball": float(f"{over}.{ball}"),
                        "batter": batter,
                        "bowler": bowler,
                        "non_striker": np.random.choice([b for b in batters if b != batter]),
                        "runs_batter": runs_bat,
                        "runs_extras": extras,
                        "runs_total": runs_tot,
                        "dismissal_type": dismissal,
                        "player_dismissed": player_dismissed,
                        "is_wicket": int(is_wkt),
                    })

                if chase_runs >= target_runs or chase_wickets >= 10:
                    break

            winner = team_b if chase_runs >= target_runs else team_a

            for d in first_inn_deliveries:
                d["target_runs"] = target_runs
                d["match_winner"] = winner
                rows.append(d)

            for d in second_inn_deliveries:
                d["target_runs"] = target_runs
                d["match_winner"] = winner
                rows.append(d)

        df = pd.DataFrame(rows)
        self._save_datasets(df)
        return df

    def _save_datasets(self, df: pd.DataFrame) -> None:
        """Persist DataFrame to CSV and Parquet formats."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        # Ensure object columns like season are cast to str for pyarrow
        df["season"] = df["season"].astype(str)
        try:
            df.to_parquet(self.parquet_path, index=False)
            logger.info("Saved parquet dataset to %s", self.parquet_path)
        except Exception as e:
            logger.warning("Could not write parquet (%s). Falling back to CSV only.", e)

        df.to_csv(self.sample_csv_path, index=False)
        logger.info("Saved sample CSV to %s (%d rows)", self.sample_csv_path, len(df))

    def load_dataset(self, force_download: bool = False, max_matches: Optional[int] = None) -> pd.DataFrame:
        """
        Load deliveries dataset from parquet/CSV if available, otherwise fetch
        from Cricsheet with automatic graceful fallback to synthetic generator.
        """
        if not force_download:
            if self.parquet_path.exists():
                try:
                    logger.info("Loading deliveries from %s", self.parquet_path)
                    return pd.read_parquet(self.parquet_path)
                except Exception as e:
                    logger.warning("Error reading parquet file: %s", e)

            if self.sample_csv_path.exists():
                try:
                    logger.info("Loading deliveries from %s", self.sample_csv_path)
                    df = pd.read_csv(self.sample_csv_path, low_memory=False)
                    # Verify it has match_winner
                    if "match_winner" in df.columns and df["match_winner"].notna().mean() > 0.8:
                        return df
                except Exception as e:
                    logger.warning("Error reading CSV file: %s", e)

        # Attempt download from Cricsheet
        try:
            return self.download_and_extract(max_matches=max_matches)
        except Exception as e:
            logger.warning("Cricsheet download failed (%s). Generating realistic dataset...", e)
            return self.generate_synthetic_sample(num_matches=60)


def load_deliveries_dataset(force_download: bool = False) -> pd.DataFrame:
    """Convenience function to load cached deliveries or initialize if missing."""
    loader = CricsheetLoader()
    return loader.load_dataset(force_download=force_download)

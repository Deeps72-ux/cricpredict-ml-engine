"""Configuration settings for CricPredict ML Engine."""

from pathlib import Path
from typing import List, Union
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration loaded from environment variables."""

    PROJECT_NAME: str = "CricPredict ML Engine"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    APP_ENV: str = "development"
    DEBUG: bool = False

    # Server settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # Paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    MODELS_DIR: Path = BASE_DIR / "models"
    MODEL_FILE: str = "win_probability_xgb.joblib"

    # CORS
    CORS_ORIGINS: List[str] = ["*"]

    # Cricsheet Data Sources
    CRICSHEET_IPL_URL: str = "https://cricsheet.org/downloads/ipl_male_csv2.zip"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="allow",
    )

    @field_validator("DATA_DIR", "MODELS_DIR", mode="after")
    @classmethod
    def ensure_dir_exists(cls, v: Path) -> Path:
        """Create directory if it does not already exist."""
        v.mkdir(parents=True, exist_ok=True)
        return v

    @property
    def model_path(self) -> Path:
        """Return full path to serialized model file."""
        return self.MODELS_DIR / self.MODEL_FILE


settings = Settings()

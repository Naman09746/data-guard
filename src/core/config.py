"""
Configuration management for the Data Quality & Leakage Detection System.

This module provides centralized configuration using Pydantic Settings
with support for environment variables, .env files, and YAML configuration.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class QualitySettings(BaseSettings):
    """Settings for data quality validation."""

    # Completeness thresholds
    missing_value_threshold: float = Field(
        default=0.05,
        ge=0.0,
        le=1.0,
        description="Maximum allowed ratio of missing values per column",
    )

    # Consistency settings
    consistency_check_enabled: bool = Field(default=True)
    cross_column_validation: bool = Field(default=True)

    # Accuracy settings
    outlier_detection_method: Literal["zscore", "iqr", "isolation_forest"] = Field(
        default="zscore"
    )
    outlier_zscore_threshold: float = Field(default=3.0, gt=0.0)
    outlier_iqr_multiplier: float = Field(default=1.5, gt=0.0)

    # Timeliness settings
    max_data_age_hours: int = Field(default=24, gt=0)
    timestamp_column: str | None = Field(default=None)

    # Schema validation
    enforce_schema: bool = Field(default=True)
    allow_extra_columns: bool = Field(default=False)
    coerce_types: bool = Field(default=True)


class LeakageSettings(BaseSettings):
    """Settings for leakage detection."""

    # Train-test contamination
    duplicate_detection_enabled: bool = Field(default=True)
    near_duplicate_threshold: float = Field(
        default=0.95,
        ge=0.0,
        le=1.0,
        description="Similarity threshold for near-duplicate detection",
    )

    # Target leakage
    target_correlation_threshold: float = Field(
        default=0.95,
        ge=0.0,
        le=1.0,
        description="Correlation threshold to flag potential target leakage",
    )

    # Feature leakage
    suspicious_feature_patterns: list[str] = Field(
        default_factory=lambda: [
            r".*_future.*",
            r".*_target.*",
            r".*_label.*",
            r".*_outcome.*",
        ]
    )

    # Temporal leakage
    temporal_validation_enabled: bool = Field(default=True)
    time_column: str | None = Field(default=None)

    # General
    sample_size_for_detection: int = Field(
        default=10000,
        gt=0,
        description="Maximum sample size for expensive detection operations",
    )


class LoggingSettings(BaseSettings):
    """Logging configuration settings."""

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO"
    )
    format: Literal["json", "console"] = Field(default="json")
    include_timestamp: bool = Field(default=True)
    include_caller: bool = Field(default=True)
    log_file: Path | None = Field(default=None)
    max_log_size_mb: int = Field(default=100, gt=0)
    backup_count: int = Field(default=5, ge=0)


class APISettings(BaseSettings):
    """API server settings."""

    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000, ge=1, le=65535)
    reload: bool = Field(default=False)
    workers: int = Field(default=4, ge=1)
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])
    api_prefix: str = Field(default="/api/v1")
    docs_url: str = Field(default="/docs")
    redoc_url: str = Field(default="/redoc")
    rate_limit_per_minute: int = Field(default=100, gt=0)
    request_timeout_seconds: int = Field(default=300, gt=0)


class Settings(BaseSettings):
    """
    Main application settings.
    
    Configuration is loaded from:
    1. Environment variables (highest priority)
    2. .env file
    3. config/settings.yaml
    4. Default values (lowest priority)
    """

    model_config = SettingsConfigDict(
        env_prefix="DQ_",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Application metadata
    app_name: str = Field(default="Data Quality & Leakage Detection System")
    environment: Literal["development", "staging", "production"] = Field(
        default="development"
    )
    debug: bool = Field(default=False)

    # Paths
    config_dir: Path = Field(default=Path("config"))
    data_dir: Path = Field(default=Path("data"))
    output_dir: Path = Field(default=Path("output"))

    # Sub-settings
    quality: QualitySettings = Field(default_factory=QualitySettings)
    leakage: LeakageSettings = Field(default_factory=LeakageSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    api: APISettings = Field(default_factory=APISettings)

    # Performance
    max_workers: int = Field(default=4, ge=1)
    chunk_size: int = Field(default=10000, gt=0)
    memory_limit_mb: int = Field(default=4096, gt=0)

    @field_validator("config_dir", "data_dir", "output_dir", mode="before")
    @classmethod
    def validate_paths(cls, v: str | Path) -> Path:
        """Convert string paths to Path objects."""
        return Path(v) if isinstance(v, str) else v

    @model_validator(mode="after")
    def validate_settings(self) -> Settings:
        """Validate settings after all fields are populated."""
        # In production, debug should be disabled
        if self.environment == "production" and self.debug:
            raise ValueError("Debug mode cannot be enabled in production")

        # Ensure output directory exists in non-test environments
        if self.environment != "development":
            self.output_dir.mkdir(parents=True, exist_ok=True)

        return self

    def to_dict(self) -> dict[str, Any]:
        """Convert settings to dictionary for serialization."""
        return self.model_dump(mode="json")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Get cached application settings.
    
    Uses LRU cache to ensure settings are only loaded once.
    Call `get_settings.cache_clear()` to reload settings.
    
    Returns:
        Settings: Application settings instance.
    """
    return Settings()


def reload_settings() -> Settings:
    """
    Force reload of application settings.
    
    Clears the cache and returns fresh settings.
    
    Returns:
        Settings: Fresh application settings instance.
    """
    get_settings.cache_clear()
    return get_settings()

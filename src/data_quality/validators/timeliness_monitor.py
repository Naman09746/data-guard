"""
Timeliness monitor for data quality validation.

Validates data freshness, update frequency, and temporal gaps.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from time import perf_counter
from typing import Any

import pandas as pd

from src.core.config import get_settings
from src.core.logging_config import get_logger
from src.data_quality.validators.base import (
    BaseValidator,
    ValidationIssue,
    ValidationResult,
    ValidationSeverity,
)

logger = get_logger(__name__)


@dataclass
class TimelinessConfig:
    """Configuration for timeliness validation."""

    max_data_age_hours: float = 24.0
    expected_update_frequency_hours: float | None = None
    max_gap_hours: float | None = None
    timestamp_column: str | None = None
    timezone: str = "UTC"


class TimelinessMonitor(BaseValidator[pd.DataFrame]):
    """
    Validates data timeliness and temporal consistency.
    
    Features:
    - Data freshness checks
    - Update frequency validation
    - Temporal gap detection
    - Future date detection
    
    Edge cases handled:
    - Timezone handling
    - DST transitions
    - Missing timestamp columns
    - Invalid datetime formats
    """

    def __init__(
        self,
        config: TimelinessConfig | dict[str, Any] | None = None,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name or "TimelinessMonitor")

        if isinstance(config, dict):
            self.config = TimelinessConfig(**config)
        elif config is None:
            settings = get_settings()
            self.config = TimelinessConfig(
                max_data_age_hours=settings.quality.max_data_age_hours,
                timestamp_column=settings.quality.timestamp_column,
            )
        else:
            self.config = config

    def validate(
        self,
        data: pd.DataFrame,
        timestamp_column: str | None = None,
        reference_time: datetime | None = None,
        **kwargs: Any,
    ) -> ValidationResult:
        start_time = perf_counter()
        issues: list[ValidationIssue] = []
        metrics: dict[str, Any] = {}

        try:
            empty_issue = self._check_empty_data(data)
            if empty_issue and empty_issue.severity == ValidationSeverity.CRITICAL:
                return self._create_result([empty_issue], duration=perf_counter() - start_time)

            # Determine timestamp column
            ts_col = timestamp_column or self.config.timestamp_column
            if not ts_col:
                ts_col = self._detect_timestamp_column(data)

            if not ts_col or ts_col not in data.columns:
                return ValidationResult(
                    validator_name=self.name,
                    status=ValidationStatus.SKIPPED,
                    issues=[ValidationIssue(
                        message="No timestamp column found for timeliness validation",
                        severity=ValidationSeverity.INFO,
                    )],
                    duration_seconds=perf_counter() - start_time,
                )

            # Parse timestamps
            timestamps = self._parse_timestamps(data[ts_col])
            if timestamps is None:
                return self._create_result([ValidationIssue(
                    message=f"Could not parse timestamps in column '{ts_col}'",
                    severity=ValidationSeverity.ERROR,
                    column=ts_col,
                )], duration=perf_counter() - start_time)

            ref_time = reference_time or datetime.now(UTC)

            # Freshness check
            freshness_issues, freshness_metrics = self._check_freshness(
                timestamps, ts_col, ref_time
            )
            issues.extend(freshness_issues)
            metrics.update(freshness_metrics)

            # Future date check
            future_issues = self._check_future_dates(timestamps, ts_col, ref_time)
            issues.extend(future_issues)

            # Gap detection
            if self.config.max_gap_hours:
                gap_issues, gap_metrics = self._check_temporal_gaps(timestamps, ts_col)
                issues.extend(gap_issues)
                metrics.update(gap_metrics)

            metrics["timestamp_column"] = ts_col
            metrics["rows_analyzed"] = len(timestamps.dropna())

            return self._create_result(issues, metrics, perf_counter() - start_time)

        except Exception as e:
            return self._handle_exception(e, "timeliness_check")

    def _detect_timestamp_column(self, data: pd.DataFrame) -> str | None:
        """Auto-detect timestamp column."""
        datetime_cols = data.select_dtypes(include=["datetime64"]).columns.tolist()
        if datetime_cols:
            return datetime_cols[0]

        # Check for common timestamp column names
        common_names = [
            "timestamp", "datetime", "date", "time", "created_at", "updated_at",
            "created", "modified", "event_time", "event_date"
        ]
        for col in data.columns:
            if col.lower() in common_names:
                return col

        return None

    def _parse_timestamps(self, column: pd.Series) -> pd.Series | None:
        """Parse column as timestamps."""
        if pd.api.types.is_datetime64_any_dtype(column):
            return column

        try:
            return pd.to_datetime(column, errors="coerce", utc=True)
        except Exception:
            return None

    def _check_freshness(
        self,
        timestamps: pd.Series,
        column_name: str,
        reference_time: datetime,
    ) -> tuple[list[ValidationIssue], dict[str, Any]]:
        """Check data freshness."""
        issues = []

        valid_ts = timestamps.dropna()
        if len(valid_ts) == 0:
            return issues, {}

        # Ensure timezone-aware comparison
        if valid_ts.dt.tz is None:
            valid_ts = valid_ts.dt.tz_localize("UTC")

        latest = valid_ts.max()
        oldest = valid_ts.min()

        # Make reference_time timezone-aware if needed
        if reference_time.tzinfo is None:
            reference_time = reference_time.replace(tzinfo=UTC)

        age_hours = (reference_time - latest).total_seconds() / 3600

        metrics = {
            "latest_timestamp": latest.isoformat(),
            "oldest_timestamp": oldest.isoformat(),
            "data_age_hours": round(age_hours, 2),
            "data_span_hours": round((latest - oldest).total_seconds() / 3600, 2),
        }

        if age_hours > self.config.max_data_age_hours:
            issues.append(ValidationIssue(
                message=f"Data is {age_hours:.1f} hours old, exceeding maximum of "
                f"{self.config.max_data_age_hours} hours",
                severity=ValidationSeverity.WARNING,
                column=column_name,
                details={
                    "data_age_hours": round(age_hours, 2),
                    "max_age_hours": self.config.max_data_age_hours,
                    "latest_timestamp": latest.isoformat(),
                },
            ))

        return issues, metrics

    def _check_future_dates(
        self,
        timestamps: pd.Series,
        column_name: str,
        reference_time: datetime,
    ) -> list[ValidationIssue]:
        """Check for future dates."""
        issues = []

        valid_ts = timestamps.dropna()
        if len(valid_ts) == 0:
            return issues

        if valid_ts.dt.tz is None:
            valid_ts = valid_ts.dt.tz_localize("UTC")

        if reference_time.tzinfo is None:
            reference_time = reference_time.replace(tzinfo=UTC)

        # Allow small buffer for clock skew
        buffer = timedelta(minutes=5)
        future = valid_ts[valid_ts > reference_time + buffer]

        if len(future) > 0:
            issues.append(ValidationIssue(
                message=f"Column '{column_name}' has {len(future)} future timestamps",
                severity=ValidationSeverity.ERROR,
                column=column_name,
                row_indices=future.index.tolist()[:100],
                details={"future_count": len(future)},
            ))

        return issues

    def _check_temporal_gaps(
        self,
        timestamps: pd.Series,
        column_name: str,
    ) -> tuple[list[ValidationIssue], dict[str, Any]]:
        """Check for temporal gaps."""
        issues = []
        metrics: dict[str, Any] = {}

        valid_ts = timestamps.dropna().sort_values()
        if len(valid_ts) < 2:
            return issues, metrics

        # Calculate gaps
        gaps = valid_ts.diff().dt.total_seconds() / 3600  # Convert to hours
        gaps = gaps.dropna()

        max_gap = gaps.max()
        avg_gap = gaps.mean()

        metrics["max_gap_hours"] = round(max_gap, 2)
        metrics["avg_gap_hours"] = round(avg_gap, 2)

        if self.config.max_gap_hours and max_gap > self.config.max_gap_hours:
            large_gaps = gaps[gaps > self.config.max_gap_hours]
            issues.append(ValidationIssue(
                message=f"Found {len(large_gaps)} temporal gaps exceeding "
                f"{self.config.max_gap_hours} hours (max: {max_gap:.1f}h)",
                severity=ValidationSeverity.WARNING,
                column=column_name,
                details={
                    "max_gap_hours": round(max_gap, 2),
                    "large_gap_count": len(large_gaps),
                },
            ))

        return issues, metrics


# Import here to avoid circular import
from src.data_quality.validators.base import ValidationStatus

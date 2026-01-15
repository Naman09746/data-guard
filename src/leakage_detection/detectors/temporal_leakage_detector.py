"""
Temporal Leakage Detector.

Detects time-based data leakage in temporal datasets.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

import pandas as pd

from src.core.config import get_settings
from src.core.logging_config import get_logger
from src.leakage_detection.detectors.base import (
    BaseDetector,
    DetectionResult,
    DetectionStatus,
    LeakageIssue,
    LeakageSeverity,
)

logger = get_logger(__name__)


@dataclass
class TemporalLeakageConfig:
    """Configuration for temporal leakage detection."""

    time_column: str | None = None
    check_train_test_overlap: bool = True
    check_look_ahead: bool = True
    look_ahead_columns: list[str] | None = None
    sample_size: int = 10000


class TemporalLeakageDetector(BaseDetector[pd.DataFrame]):
    """
    Detects temporal leakage in time-series data.
    
    Features:
    - Time-based split validation
    - Look-ahead bias detection
    - Feature timestamp validation
    
    Edge cases handled:
    - Missing timestamp columns
    - Timezone issues
    - Irregular time series
    """

    def __init__(
        self,
        config: TemporalLeakageConfig | dict[str, Any] | None = None,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name or "TemporalLeakageDetector")

        if isinstance(config, dict):
            self.config = TemporalLeakageConfig(**config)
        elif config is None:
            settings = get_settings()
            self.config = TemporalLeakageConfig(
                time_column=settings.leakage.time_column,
                sample_size=settings.leakage.sample_size_for_detection,
            )
        else:
            self.config = config

    def detect(
        self,
        train_data: pd.DataFrame,
        test_data: pd.DataFrame | None = None,
        target_column: str | None = None,
        time_column: str | None = None,
        **kwargs: Any,
    ) -> DetectionResult:
        start_time = perf_counter()
        issues: list[LeakageIssue] = []
        metrics: dict[str, Any] = {}

        try:
            # Validate inputs
            empty_issue = self._check_empty_data(train_data, "train_data")
            if empty_issue:
                return self._create_result([empty_issue], duration=perf_counter() - start_time)

            # Determine time column
            ts_col = time_column or self.config.time_column or self._detect_time_column(train_data)

            if not ts_col or ts_col not in train_data.columns:
                return DetectionResult(
                    detector_name=self.name,
                    status=DetectionStatus.SKIPPED,
                    issues=[LeakageIssue(
                        message="No time column found for temporal leakage detection",
                        severity=LeakageSeverity.INFO,
                        leakage_type="skipped",
                    )],
                    duration_seconds=perf_counter() - start_time,
                )

            # Parse timestamps
            train_times = self._parse_timestamps(train_data[ts_col])
            if train_times is None:
                return self._create_result([LeakageIssue(
                    message=f"Could not parse timestamps in column '{ts_col}'",
                    severity=LeakageSeverity.CRITICAL,
                    leakage_type="error",
                )], duration=perf_counter() - start_time)

            # Check train-test temporal overlap
            if self.config.check_train_test_overlap and test_data is not None:
                if ts_col in test_data.columns:
                    test_times = self._parse_timestamps(test_data[ts_col])
                    if test_times is not None:
                        overlap_issues, overlap_metrics = self._check_temporal_overlap(
                            train_times, test_times, ts_col
                        )
                        issues.extend(overlap_issues)
                        metrics.update(overlap_metrics)

            # Check for look-ahead bias
            if self.config.check_look_ahead:
                lookahead_issues, lookahead_metrics = self._check_look_ahead_bias(
                    train_data, train_times, ts_col
                )
                issues.extend(lookahead_issues)
                metrics.update(lookahead_metrics)

            # Check temporal ordering
            order_issues, order_metrics = self._check_temporal_ordering(train_times, ts_col)
            issues.extend(order_issues)
            metrics.update(order_metrics)

            metrics["time_column"] = ts_col
            metrics["rows_analyzed"] = len(train_times.dropna())

            duration = perf_counter() - start_time
            self._logger.info(
                "temporal_leakage_detection_complete",
                issues_found=len(issues),
                duration=round(duration, 4),
            )

            return self._create_result(issues, metrics, duration)

        except Exception as e:
            return self._handle_exception(e, "temporal_leakage_detection")

    def _detect_time_column(self, data: pd.DataFrame) -> str | None:
        """Auto-detect time column."""
        # Check datetime columns
        datetime_cols = data.select_dtypes(include=["datetime64"]).columns.tolist()
        if datetime_cols:
            return datetime_cols[0]

        # Check common names
        common_names = [
            "timestamp", "datetime", "date", "time", "created_at",
            "event_time", "event_date", "ts", "dt"
        ]
        for col in data.columns:
            if str(col).lower() in common_names:
                return str(col)

        return None

    def _parse_timestamps(self, column: pd.Series) -> pd.Series | None:
        """Parse column as timestamps."""
        if pd.api.types.is_datetime64_any_dtype(column):
            return column

        try:
            return pd.to_datetime(column, errors="coerce", utc=True)
        except Exception:
            return None

    def _check_temporal_overlap(
        self,
        train_times: pd.Series,
        test_times: pd.Series,
        column_name: str,
    ) -> tuple[list[LeakageIssue], dict[str, Any]]:
        """Check for temporal overlap between train and test."""
        issues = []
        metrics: dict[str, Any] = {}

        train_valid = train_times.dropna()
        test_valid = test_times.dropna()

        if len(train_valid) == 0 or len(test_valid) == 0:
            return issues, {"temporal_overlap_check": "skipped_empty_data"}

        train_max = train_valid.max()
        train_min = train_valid.min()
        test_max = test_valid.max()
        test_min = test_valid.min()

        metrics["train_time_range"] = {
            "min": str(train_min),
            "max": str(train_max),
        }
        metrics["test_time_range"] = {
            "min": str(test_min),
            "max": str(test_max),
        }

        # Check if test contains times before train ends
        if test_min < train_max:
            overlap_count = (test_valid < train_max).sum()
            issues.append(LeakageIssue(
                message=f"Temporal overlap detected: {overlap_count} test samples before train end",
                severity=LeakageSeverity.CRITICAL,
                leakage_type="temporal_overlap",
                affected_rows=int(overlap_count),
                details={
                    "train_max": str(train_max),
                    "test_min": str(test_min),
                    "overlap_count": int(overlap_count),
                },
                recommendation="Ensure proper time-based train/test split",
            ))

        # Check if train contains times after test starts
        if train_max > test_min:
            future_in_train = (train_valid > test_min).sum()
            if future_in_train > 0:
                issues.append(LeakageIssue(
                    message=f"Future data in train: {future_in_train} train samples after test start",
                    severity=LeakageSeverity.CRITICAL,
                    leakage_type="future_in_train",
                    affected_rows=int(future_in_train),
                    details={
                        "test_min": str(test_min),
                        "future_train_count": int(future_in_train),
                    },
                    recommendation="Remove future data from training set",
                ))

        return issues, metrics

    def _check_look_ahead_bias(
        self,
        data: pd.DataFrame,
        timestamps: pd.Series,
        time_column: str,
    ) -> tuple[list[LeakageIssue], dict[str, Any]]:
        """Check for look-ahead bias in features."""
        issues = []
        metrics: dict[str, Any] = {}

        # Check for features that might contain future information
        look_ahead_columns = self.config.look_ahead_columns or []

        # Also check column names for patterns suggesting future data
        future_patterns = ["_next", "_future", "_forward", "_tomorrow", "_t+"]

        suspicious_cols = []
        for col in data.columns:
            if col == time_column:
                continue
            col_lower = str(col).lower()
            for pattern in future_patterns:
                if pattern in col_lower:
                    suspicious_cols.append(col)
                    break

        suspicious_cols.extend([c for c in look_ahead_columns if c in data.columns])
        suspicious_cols = list(set(suspicious_cols))

        if suspicious_cols:
            issues.append(LeakageIssue(
                message=f"Found {len(suspicious_cols)} column(s) with potential look-ahead bias",
                severity=LeakageSeverity.WARNING,
                leakage_type="look_ahead_bias",
                affected_features=suspicious_cols,
                recommendation="Verify these features don't contain future information",
            ))

        metrics["look_ahead_suspicious_columns"] = suspicious_cols

        return issues, metrics

    def _check_temporal_ordering(
        self,
        timestamps: pd.Series,
        column_name: str,
    ) -> tuple[list[LeakageIssue], dict[str, Any]]:
        """Check temporal ordering of data."""
        issues = []
        metrics: dict[str, Any] = {}

        valid_ts = timestamps.dropna()
        if len(valid_ts) < 2:
            return issues, {"ordering_check": "skipped_insufficient_data"}

        # Check if data is sorted by time
        is_sorted = valid_ts.is_monotonic_increasing
        is_reverse_sorted = valid_ts.is_monotonic_decreasing

        metrics["is_time_sorted"] = is_sorted or is_reverse_sorted

        if not (is_sorted or is_reverse_sorted):
            issues.append(LeakageIssue(
                message="Data is not sorted by timestamp",
                severity=LeakageSeverity.INFO,
                leakage_type="temporal_ordering",
                details={"column": column_name},
                recommendation="Consider sorting data by time for time-series analysis",
            ))

        return issues, metrics

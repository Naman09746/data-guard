"""
Train-Test Contamination Detector.

Detects duplicate and near-duplicate rows between train and test sets.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np
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
class TrainTestConfig:
    """Configuration for train-test contamination detection."""

    check_duplicates: bool = True
    check_near_duplicates: bool = True
    near_duplicate_threshold: float = 0.95
    columns_to_check: list[str] | None = None
    exclude_columns: list[str] | None = None
    sample_size: int = 10000
    hash_columns: bool = True


class TrainTestDetector(BaseDetector[pd.DataFrame]):
    """
    Detects train-test contamination.
    
    Features:
    - Exact duplicate detection
    - Near-duplicate detection using similarity
    - Column subset checking
    - Efficient hashing for large datasets
    
    Edge cases handled:
    - Empty datasets
    - Different column sets
    - Floating-point comparison
    - Hash collisions
    """

    def __init__(
        self,
        config: TrainTestConfig | dict[str, Any] | None = None,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name or "TrainTestDetector")

        if isinstance(config, dict):
            self.config = TrainTestConfig(**config)
        elif config is None:
            settings = get_settings()
            self.config = TrainTestConfig(
                near_duplicate_threshold=settings.leakage.near_duplicate_threshold,
                sample_size=settings.leakage.sample_size_for_detection,
            )
        else:
            self.config = config

    def detect(
        self,
        train_data: pd.DataFrame,
        test_data: pd.DataFrame | None = None,
        target_column: str | None = None,
        **kwargs: Any,
    ) -> DetectionResult:
        start_time = perf_counter()
        issues: list[LeakageIssue] = []
        metrics: dict[str, Any] = {}

        try:
            # Validate inputs
            train_issue = self._check_empty_data(train_data, "train_data")
            if train_issue:
                return self._create_result([train_issue], duration=perf_counter() - start_time)

            if test_data is None:
                return DetectionResult(
                    detector_name=self.name,
                    status=DetectionStatus.SKIPPED,
                    issues=[LeakageIssue(
                        message="No test data provided",
                        severity=LeakageSeverity.INFO,
                        leakage_type="skipped",
                    )],
                    duration_seconds=perf_counter() - start_time,
                )

            test_issue = self._check_empty_data(test_data, "test_data")
            if test_issue:
                return self._create_result([test_issue], duration=perf_counter() - start_time)

            # Determine columns to check
            columns = self._get_check_columns(train_data, test_data, target_column)

            # Check for exact duplicates
            if self.config.check_duplicates:
                dup_issues, dup_metrics = self._check_duplicates(
                    train_data, test_data, columns
                )
                issues.extend(dup_issues)
                metrics.update(dup_metrics)

            # Check for near-duplicates
            if self.config.check_near_duplicates and len(issues) == 0:
                near_dup_issues, near_dup_metrics = self._check_near_duplicates(
                    train_data, test_data, columns
                )
                issues.extend(near_dup_issues)
                metrics.update(near_dup_metrics)

            metrics["train_rows"] = len(train_data)
            metrics["test_rows"] = len(test_data)
            metrics["columns_checked"] = len(columns)

            duration = perf_counter() - start_time
            self._logger.info(
                "train_test_detection_complete",
                issues_found=len(issues),
                duration=round(duration, 4),
            )

            return self._create_result(issues, metrics, duration)

        except Exception as e:
            return self._handle_exception(e, "train_test_detection")

    def _get_check_columns(
        self,
        train_data: pd.DataFrame,
        test_data: pd.DataFrame,
        target_column: str | None,
    ) -> list[str]:
        """Determine which columns to use for comparison."""
        if self.config.columns_to_check:
            columns = [c for c in self.config.columns_to_check
                      if c in train_data.columns and c in test_data.columns]
        else:
            columns = list(set(train_data.columns) & set(test_data.columns))

        # Exclude specified columns
        if self.config.exclude_columns:
            columns = [c for c in columns if c not in self.config.exclude_columns]

        # Exclude target column
        if target_column and target_column in columns:
            columns.remove(target_column)

        return columns

    def _check_duplicates(
        self,
        train_data: pd.DataFrame,
        test_data: pd.DataFrame,
        columns: list[str],
    ) -> tuple[list[LeakageIssue], dict[str, Any]]:
        """Check for exact duplicates between datasets."""
        issues = []
        metrics: dict[str, Any] = {}

        if self.config.hash_columns:
            # Use hashing for efficiency
            train_hashes = self._compute_row_hashes(train_data[columns])
            test_hashes = self._compute_row_hashes(test_data[columns])

            duplicates = set(train_hashes) & set(test_hashes)
            duplicate_count = len(duplicates)

            # Find indices of duplicates in test set
            duplicate_indices = [
                i for i, h in enumerate(test_hashes) if h in duplicates
            ]
        else:
            # Direct comparison
            merged = pd.merge(
                train_data[columns].drop_duplicates().reset_index(drop=True),
                test_data[columns].reset_index().rename(columns={"index": "test_idx"}),
                on=columns,
                how="inner",
            )
            duplicate_count = len(merged)
            duplicate_indices = merged["test_idx"].tolist()

        metrics["exact_duplicates"] = duplicate_count
        metrics["duplicate_ratio"] = round(duplicate_count / len(test_data), 4) if len(test_data) > 0 else 0

        if duplicate_count > 0:
            issues.append(LeakageIssue(
                message=f"Found {duplicate_count} exact duplicate rows between train and test "
                f"({metrics['duplicate_ratio']:.2%} of test set)",
                severity=LeakageSeverity.CRITICAL,
                leakage_type="train_test_contamination",
                affected_rows=duplicate_count,
                details={
                    "duplicate_indices_sample": duplicate_indices[:20],
                    "columns_compared": len(columns),
                },
                recommendation="Remove duplicate rows from train set or use group-aware splitting",
            ))

        return issues, metrics

    def _check_near_duplicates(
        self,
        train_data: pd.DataFrame,
        test_data: pd.DataFrame,
        columns: list[str],
    ) -> tuple[list[LeakageIssue], dict[str, Any]]:
        """Check for near-duplicate rows."""
        issues = []
        metrics: dict[str, Any] = {}

        # Sample for efficiency
        train_sample = train_data[columns]
        test_sample = test_data[columns]

        if len(train_sample) > self.config.sample_size:
            train_sample = train_sample.sample(n=self.config.sample_size, random_state=42)
        if len(test_sample) > self.config.sample_size:
            test_sample = test_sample.sample(n=self.config.sample_size, random_state=42)

        # Select only numeric columns for similarity
        numeric_cols = train_sample.select_dtypes(include=[np.number]).columns.tolist()

        if len(numeric_cols) < 2:
            metrics["near_duplicate_check"] = "skipped"
            metrics["reason"] = "insufficient_numeric_columns"
            return issues, metrics

        train_numeric = train_sample[numeric_cols].fillna(0).values
        test_numeric = test_sample[numeric_cols].fillna(0).values

        # Normalize
        train_norm = np.linalg.norm(train_numeric, axis=1, keepdims=True)
        test_norm = np.linalg.norm(test_numeric, axis=1, keepdims=True)

        train_normalized = np.divide(train_numeric, train_norm,
                                    where=train_norm != 0, out=np.zeros_like(train_numeric))
        test_normalized = np.divide(test_numeric, test_norm,
                                   where=test_norm != 0, out=np.zeros_like(test_numeric))

        # Compute cosine similarity in batches
        near_dup_count = 0
        batch_size = 100

        for i in range(0, len(test_normalized), batch_size):
            batch = test_normalized[i:i + batch_size]
            similarities = batch @ train_normalized.T
            max_similarities = similarities.max(axis=1)
            near_dup_count += (max_similarities >= self.config.near_duplicate_threshold).sum()

        metrics["near_duplicates_sampled"] = int(near_dup_count)
        metrics["sample_size_train"] = len(train_sample)
        metrics["sample_size_test"] = len(test_sample)

        if near_dup_count > 0:
            estimated_total = int(near_dup_count * len(test_data) / len(test_sample))
            issues.append(LeakageIssue(
                message=f"Found approximately {estimated_total} near-duplicate rows "
                f"(similarity >= {self.config.near_duplicate_threshold})",
                severity=LeakageSeverity.WARNING,
                leakage_type="near_duplicate_contamination",
                affected_rows=estimated_total,
                details={
                    "similarity_threshold": self.config.near_duplicate_threshold,
                    "sampled_near_duplicates": int(near_dup_count),
                },
                recommendation="Review near-duplicate rows and consider more strict splitting",
            ))

        return issues, metrics

    def _compute_row_hashes(self, data: pd.DataFrame) -> list[str]:
        """Compute hash for each row."""
        hashes = []
        for idx in range(len(data)):
            row_str = data.iloc[idx].to_string()
            hash_val = hashlib.md5(row_str.encode(), usedforsecurity=False).hexdigest()
            hashes.append(hash_val)
        return hashes

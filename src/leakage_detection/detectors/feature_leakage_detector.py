"""
Feature Leakage Detector.

Detects suspicious feature patterns and potential data leakage in features.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any

import pandas as pd

from src.core.config import get_settings
from src.core.logging_config import get_logger
from src.leakage_detection.detectors.base import (
    BaseDetector,
    DetectionResult,
    LeakageIssue,
    LeakageSeverity,
)

logger = get_logger(__name__)


@dataclass
class FeatureLeakageConfig:
    """Configuration for feature leakage detection."""

    suspicious_patterns: list[str] = field(default_factory=lambda: [
        r".*_future.*",
        r".*_target.*",
        r".*_label.*",
        r".*_outcome.*",
        r".*_result.*",
        r".*_prediction.*",
        r".*_pred.*",
        r".*_actual.*",
        r".*_y_.*",
        r"^y_.*",
        r".*_next_.*",
        r".*_tomorrow.*",
    ])
    check_feature_names: bool = True
    check_feature_importance: bool = True
    importance_threshold: float = 0.5
    sample_size: int = 10000


class FeatureLeakageDetector(BaseDetector[pd.DataFrame]):
    """
    Detects potential feature leakage through pattern analysis.
    
    Features:
    - Feature name pattern matching
    - Suspicious naming detection
    - Feature importance analysis
    
    Edge cases handled:
    - Case sensitivity
    - Empty feature sets
    - Non-string column names
    """

    def __init__(
        self,
        config: FeatureLeakageConfig | dict[str, Any] | None = None,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name or "FeatureLeakageDetector")

        if isinstance(config, dict):
            self.config = FeatureLeakageConfig(**config)
        elif config is None:
            settings = get_settings()
            self.config = FeatureLeakageConfig(
                suspicious_patterns=settings.leakage.suspicious_feature_patterns,
                sample_size=settings.leakage.sample_size_for_detection,
            )
        else:
            self.config = config

        # Compile patterns
        self._compiled_patterns = [
            re.compile(p, re.IGNORECASE) for p in self.config.suspicious_patterns
        ]

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
            empty_issue = self._check_empty_data(train_data, "train_data")
            if empty_issue:
                return self._create_result([empty_issue], duration=perf_counter() - start_time)

            # Get feature columns
            feature_columns = [c for c in train_data.columns if c != target_column]

            # Check feature names for suspicious patterns
            if self.config.check_feature_names:
                name_issues, name_metrics = self._check_feature_names(feature_columns)
                issues.extend(name_issues)
                metrics.update(name_metrics)

            # Check feature importance if target is provided
            if self.config.check_feature_importance and target_column:
                imp_issues, imp_metrics = self._check_feature_importance(
                    train_data, target_column, feature_columns
                )
                issues.extend(imp_issues)
                metrics.update(imp_metrics)

            # Check for constant features that might indicate leakage
            const_issues, const_metrics = self._check_constant_features(
                train_data, feature_columns
            )
            issues.extend(const_issues)
            metrics.update(const_metrics)

            metrics["features_analyzed"] = len(feature_columns)

            duration = perf_counter() - start_time
            self._logger.info(
                "feature_leakage_detection_complete",
                issues_found=len(issues),
                duration=round(duration, 4),
            )

            return self._create_result(issues, metrics, duration)

        except Exception as e:
            return self._handle_exception(e, "feature_leakage_detection")

    def _check_feature_names(
        self,
        feature_columns: list[str],
    ) -> tuple[list[LeakageIssue], dict[str, Any]]:
        """Check feature names for suspicious patterns."""
        issues = []
        metrics: dict[str, Any] = {}

        suspicious_features: dict[str, list[str]] = {}

        for col in feature_columns:
            col_str = str(col)
            for pattern in self._compiled_patterns:
                if pattern.match(col_str):
                    pattern_str = pattern.pattern
                    if pattern_str not in suspicious_features:
                        suspicious_features[pattern_str] = []
                    suspicious_features[pattern_str].append(col_str)
                    break

        metrics["suspicious_name_patterns"] = {
            pattern: len(features)
            for pattern, features in suspicious_features.items()
        }

        all_suspicious = []
        for pattern, features in suspicious_features.items():
            all_suspicious.extend(features)

            issues.append(LeakageIssue(
                message=f"{len(features)} feature(s) match suspicious pattern '{pattern}'",
                severity=LeakageSeverity.WARNING,
                leakage_type="suspicious_feature_name",
                affected_features=features,
                details={"pattern": pattern},
                recommendation="Review these features for potential data leakage",
            ))

        return issues, metrics

    def _check_feature_importance(
        self,
        data: pd.DataFrame,
        target_column: str,
        feature_columns: list[str],
    ) -> tuple[list[LeakageIssue], dict[str, Any]]:
        """Check for unusually high feature importance."""
        issues = []
        metrics: dict[str, Any] = {}

        try:
            from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

            # Sample data
            sample = data
            if len(data) > self.config.sample_size:
                sample = data.sample(n=self.config.sample_size, random_state=42)

            # Prepare features
            numeric_cols = [c for c in feature_columns
                          if pd.api.types.is_numeric_dtype(sample[c])]

            if len(numeric_cols) < 2:
                return issues, {"feature_importance": "skipped_insufficient_features"}

            X = sample[numeric_cols].fillna(0).values
            y = sample[target_column]

            # Determine model type
            if pd.api.types.is_numeric_dtype(y) and y.nunique() > 10:
                model = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
            else:
                y = pd.factorize(y)[0]
                model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)

            model.fit(X, y)
            importances = model.feature_importances_

            # Find unusually important features
            suspicious = []
            for i, col in enumerate(numeric_cols):
                if importances[i] >= self.config.importance_threshold:
                    suspicious.append((col, float(importances[i])))

            metrics["feature_importance_analysis"] = {
                "high_importance_features": {col: round(imp, 4) for col, imp in suspicious}
            }

            for col, importance in suspicious:
                issues.append(LeakageIssue(
                    message=f"Feature '{col}' has unusually high importance ({importance:.4f})",
                    severity=LeakageSeverity.WARNING,
                    leakage_type="high_importance_feature",
                    affected_features=[col],
                    details={
                        "importance": round(importance, 4),
                        "threshold": self.config.importance_threshold,
                    },
                    recommendation=f"Investigate why '{col}' has such high predictive power",
                ))

        except ImportError:
            metrics["feature_importance"] = "skipped_sklearn_not_available"
        except Exception as e:
            metrics["feature_importance_error"] = str(e)

        return issues, metrics

    def _check_constant_features(
        self,
        data: pd.DataFrame,
        feature_columns: list[str],
    ) -> tuple[list[LeakageIssue], dict[str, Any]]:
        """Check for constant or near-constant features."""
        issues = []
        metrics: dict[str, Any] = {}

        constant_features = []
        near_constant_features = []

        for col in feature_columns:
            unique_count = data[col].nunique()
            unique_ratio = unique_count / len(data) if len(data) > 0 else 0

            if unique_count <= 1:
                constant_features.append(col)
            elif unique_ratio < 0.01 and unique_count <= 3:
                near_constant_features.append(col)

        metrics["constant_features"] = len(constant_features)
        metrics["near_constant_features"] = len(near_constant_features)

        if constant_features:
            issues.append(LeakageIssue(
                message=f"Found {len(constant_features)} constant feature(s)",
                severity=LeakageSeverity.INFO,
                leakage_type="constant_feature",
                affected_features=constant_features,
                recommendation="Consider removing constant features",
            ))

        return issues, metrics

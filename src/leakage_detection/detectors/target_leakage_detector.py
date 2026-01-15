"""
Target Leakage Detector.

Detects features that have suspiciously high correlation with the target.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

import pandas as pd
from scipy import stats

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
class TargetLeakageConfig:
    """Configuration for target leakage detection."""

    correlation_threshold: float = 0.95
    mutual_info_threshold: float = 0.9
    check_correlation: bool = True
    check_mutual_info: bool = True
    sample_size: int = 10000


class TargetLeakageDetector(BaseDetector[pd.DataFrame]):
    """
    Detects potential target leakage in features.
    
    Features:
    - Correlation analysis with target
    - Mutual information scoring
    - Proxy feature identification
    
    Edge cases handled:
    - Missing target column
    - Non-numeric features
    - Constant columns
    """

    def __init__(
        self,
        config: TargetLeakageConfig | dict[str, Any] | None = None,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name or "TargetLeakageDetector")

        if isinstance(config, dict):
            self.config = TargetLeakageConfig(**config)
        elif config is None:
            settings = get_settings()
            self.config = TargetLeakageConfig(
                correlation_threshold=settings.leakage.target_correlation_threshold,
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
            empty_issue = self._check_empty_data(train_data, "train_data")
            if empty_issue:
                return self._create_result([empty_issue], duration=perf_counter() - start_time)

            if target_column is None:
                return DetectionResult(
                    detector_name=self.name,
                    status=DetectionStatus.SKIPPED,
                    issues=[LeakageIssue(
                        message="No target column specified",
                        severity=LeakageSeverity.INFO,
                        leakage_type="skipped",
                    )],
                    duration_seconds=perf_counter() - start_time,
                )

            if target_column not in train_data.columns:
                return self._create_result([LeakageIssue(
                    message=f"Target column '{target_column}' not found",
                    severity=LeakageSeverity.CRITICAL,
                    leakage_type="error",
                )], duration=perf_counter() - start_time)

            # Sample data if needed
            data = train_data
            if len(data) > self.config.sample_size:
                data = data.sample(n=self.config.sample_size, random_state=42)

            # Get feature columns
            feature_columns = [c for c in data.columns if c != target_column]

            # Correlation analysis
            if self.config.check_correlation:
                corr_issues, corr_metrics = self._check_correlations(
                    data, target_column, feature_columns
                )
                issues.extend(corr_issues)
                metrics.update(corr_metrics)

            # Mutual information
            if self.config.check_mutual_info:
                mi_issues, mi_metrics = self._check_mutual_info(
                    data, target_column, feature_columns
                )
                issues.extend(mi_issues)
                metrics.update(mi_metrics)

            metrics["features_analyzed"] = len(feature_columns)
            metrics["sample_size"] = len(data)

            duration = perf_counter() - start_time
            self._logger.info(
                "target_leakage_detection_complete",
                issues_found=len(issues),
                duration=round(duration, 4),
            )

            return self._create_result(issues, metrics, duration)

        except Exception as e:
            return self._handle_exception(e, "target_leakage_detection")

    def _check_correlations(
        self,
        data: pd.DataFrame,
        target_column: str,
        feature_columns: list[str],
    ) -> tuple[list[LeakageIssue], dict[str, Any]]:
        """Check feature correlations with target."""
        issues = []
        metrics: dict[str, Any] = {}

        target = data[target_column]
        correlations: dict[str, float] = {}
        suspicious_features: list[str] = []

        for col in feature_columns:
            feature = data[col]

            # Skip non-numeric or constant columns
            if not pd.api.types.is_numeric_dtype(feature):
                continue
            if feature.nunique() <= 1:
                continue

            # Calculate correlation
            valid_mask = ~(feature.isna() | target.isna())
            if valid_mask.sum() < 10:
                continue

            if pd.api.types.is_numeric_dtype(target):
                corr, _ = stats.pearsonr(
                    feature[valid_mask].values,
                    target[valid_mask].values
                )
            else:
                # For categorical target, use point-biserial if binary
                unique_vals = target.dropna().unique()
                if len(unique_vals) == 2:
                    binary_target = (target == unique_vals[0]).astype(int)
                    corr, _ = stats.pointbiserialr(
                        binary_target[valid_mask].values,
                        feature[valid_mask].values
                    )
                else:
                    continue

            corr = abs(corr)
            correlations[col] = round(corr, 4)

            if corr >= self.config.correlation_threshold:
                suspicious_features.append(col)

        metrics["correlation_analysis"] = {
            "high_correlations": {
                k: v for k, v in correlations.items()
                if v >= self.config.correlation_threshold
            }
        }

        for feature in suspicious_features:
            corr = correlations[feature]
            issues.append(LeakageIssue(
                message=f"Feature '{feature}' has high correlation ({corr:.4f}) with target",
                severity=LeakageSeverity.CRITICAL,
                leakage_type="target_leakage",
                affected_features=[feature],
                details={
                    "correlation": corr,
                    "threshold": self.config.correlation_threshold,
                },
                recommendation=f"Investigate if '{feature}' contains future/target information",
            ))

        return issues, metrics

    def _check_mutual_info(
        self,
        data: pd.DataFrame,
        target_column: str,
        feature_columns: list[str],
    ) -> tuple[list[LeakageIssue], dict[str, Any]]:
        """Check mutual information between features and target."""
        issues = []
        metrics: dict[str, Any] = {}

        try:
            from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

            target = data[target_column]

            # Get numeric features only
            numeric_cols = [c for c in feature_columns
                          if pd.api.types.is_numeric_dtype(data[c])]

            if not numeric_cols:
                return issues, {"mutual_info": "skipped_no_numeric_features"}

            X = data[numeric_cols].fillna(0).values
            y = target.values

            # Determine if classification or regression
            if pd.api.types.is_numeric_dtype(target) and target.nunique() > 10:
                mi_scores = mutual_info_regression(X, y, random_state=42)
            else:
                # Encode categorical target
                y_encoded = pd.factorize(y)[0]
                mi_scores = mutual_info_classif(X, y_encoded, random_state=42)

            # Normalize scores
            max_score = mi_scores.max() if mi_scores.max() > 0 else 1
            mi_normalized = mi_scores / max_score

            suspicious = []
            for i, col in enumerate(numeric_cols):
                if mi_normalized[i] >= self.config.mutual_info_threshold:
                    suspicious.append((col, float(mi_normalized[i])))

            metrics["mutual_info_analysis"] = {
                "high_mi_features": {col: score for col, score in suspicious}
            }

            for col, score in suspicious:
                issues.append(LeakageIssue(
                    message=f"Feature '{col}' has high mutual info ({score:.4f}) with target",
                    severity=LeakageSeverity.WARNING,
                    leakage_type="potential_target_leakage",
                    affected_features=[col],
                    details={
                        "mutual_info_normalized": round(score, 4),
                        "threshold": self.config.mutual_info_threshold,
                    },
                    recommendation=f"Review feature '{col}' for potential target leakage",
                ))

        except ImportError:
            metrics["mutual_info"] = "skipped_sklearn_not_available"
        except Exception as e:
            metrics["mutual_info_error"] = str(e)

        return issues, metrics

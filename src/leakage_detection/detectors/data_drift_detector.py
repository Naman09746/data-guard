"""
Data Drift Detector.

Detects distribution shifts between training and test data.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np
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
class DataDriftConfig:
    """Configuration for data drift detection."""

    ks_threshold: float = 0.05  # KS test p-value threshold
    psi_threshold: float = 0.2  # PSI threshold
    chi2_threshold: float = 0.05  # Chi-squared test threshold
    check_numeric_drift: bool = True
    check_categorical_drift: bool = True
    sample_size: int = 10000


class DataDriftDetector(BaseDetector[pd.DataFrame]):
    """
    Detects data drift between train and test sets.
    
    Features:
    - Kolmogorov-Smirnov test for numeric features
    - Chi-squared test for categorical features
    - Population Stability Index (PSI)
    - Feature distribution comparison
    
    Edge cases handled:
    - Missing values in distributions
    - Constant columns
    - High cardinality categoricals
    """

    def __init__(
        self,
        config: DataDriftConfig | dict[str, Any] | None = None,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name or "DataDriftDetector")
        
        if isinstance(config, dict):
            self.config = DataDriftConfig(**config)
        else:
            self.config = config or DataDriftConfig()

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
                        message="No test data provided for drift detection",
                        severity=LeakageSeverity.INFO,
                        leakage_type="skipped",
                    )],
                    duration_seconds=perf_counter() - start_time,
                )
            
            test_issue = self._check_empty_data(test_data, "test_data")
            if test_issue:
                return self._create_result([test_issue], duration=perf_counter() - start_time)
            
            # Get common columns
            common_cols = list(set(train_data.columns) & set(test_data.columns))
            if target_column and target_column in common_cols:
                common_cols.remove(target_column)
            
            # Sample if needed
            train_sample = train_data[common_cols]
            test_sample = test_data[common_cols]
            
            if len(train_sample) > self.config.sample_size:
                train_sample = train_sample.sample(n=self.config.sample_size, random_state=42)
            if len(test_sample) > self.config.sample_size:
                test_sample = test_sample.sample(n=self.config.sample_size, random_state=42)
            
            drift_results: dict[str, dict[str, Any]] = {}
            
            # Check numeric features
            if self.config.check_numeric_drift:
                numeric_cols = train_sample.select_dtypes(include=[np.number]).columns
                for col in numeric_cols:
                    result = self._check_numeric_drift(train_sample[col], test_sample[col])
                    if result:
                        drift_results[col] = result
            
            # Check categorical features
            if self.config.check_categorical_drift:
                cat_cols = train_sample.select_dtypes(include=['object', 'category']).columns
                for col in cat_cols:
                    result = self._check_categorical_drift(train_sample[col], test_sample[col])
                    if result:
                        drift_results[col] = result
            
            # Create issues for significant drift
            drifted_features = []
            for col, result in drift_results.items():
                if result.get('has_drift', False):
                    drifted_features.append(col)
            
            metrics["drift_analysis"] = drift_results
            metrics["drifted_features"] = drifted_features
            metrics["features_checked"] = len(common_cols)
            
            if drifted_features:
                issues.append(LeakageIssue(
                    message=f"Significant data drift detected in {len(drifted_features)} feature(s)",
                    severity=LeakageSeverity.WARNING,
                    leakage_type="data_drift",
                    affected_features=drifted_features,
                    details={
                        "drift_details": {k: v for k, v in drift_results.items() if v.get('has_drift')},
                    },
                    recommendation="Review distribution changes and consider retraining model",
                ))
            
            duration = perf_counter() - start_time
            self._logger.info(
                "data_drift_detection_complete",
                drifted_features=len(drifted_features),
                duration=round(duration, 4),
            )
            
            return self._create_result(issues, metrics, duration)
            
        except Exception as e:
            return self._handle_exception(e, "data_drift_detection")

    def _check_numeric_drift(
        self,
        train_col: pd.Series,
        test_col: pd.Series,
    ) -> dict[str, Any] | None:
        """Check drift for numeric column using KS test."""
        train_valid = train_col.dropna()
        test_valid = test_col.dropna()
        
        if len(train_valid) < 10 or len(test_valid) < 10:
            return None
        
        # Skip constant columns
        if train_valid.nunique() <= 1 or test_valid.nunique() <= 1:
            return None
        
        # KS test
        ks_stat, ks_pvalue = stats.ks_2samp(train_valid, test_valid)
        
        # Calculate PSI
        psi = self._calculate_psi(train_valid, test_valid)
        
        has_drift = ks_pvalue < self.config.ks_threshold or psi > self.config.psi_threshold
        
        return {
            "type": "numeric",
            "ks_statistic": round(ks_stat, 4),
            "ks_pvalue": round(ks_pvalue, 4),
            "psi": round(psi, 4),
            "has_drift": has_drift,
            "train_mean": round(train_valid.mean(), 4),
            "test_mean": round(test_valid.mean(), 4),
            "train_std": round(train_valid.std(), 4),
            "test_std": round(test_valid.std(), 4),
        }

    def _check_categorical_drift(
        self,
        train_col: pd.Series,
        test_col: pd.Series,
    ) -> dict[str, Any] | None:
        """Check drift for categorical column using Chi-squared test."""
        train_valid = train_col.dropna()
        test_valid = test_col.dropna()
        
        if len(train_valid) < 10 or len(test_valid) < 10:
            return None
        
        # Get value counts
        train_counts = train_valid.value_counts()
        test_counts = test_valid.value_counts()
        
        # Skip high cardinality
        if len(train_counts) > 100:
            return None
        
        # Align categories
        all_cats = set(train_counts.index) | set(test_counts.index)
        train_aligned = pd.Series(0, index=all_cats)
        test_aligned = pd.Series(0, index=all_cats)
        train_aligned[train_counts.index] = train_counts.values
        test_aligned[test_counts.index] = test_counts.values
        
        # Chi-squared test
        try:
            chi2, pvalue = stats.chisquare(test_aligned, f_exp=train_aligned + 1)
            has_drift = pvalue < self.config.chi2_threshold
        except Exception:
            return None
        
        return {
            "type": "categorical",
            "chi2_statistic": round(float(chi2), 4),
            "chi2_pvalue": round(float(pvalue), 4),
            "has_drift": has_drift,
            "train_categories": len(train_counts),
            "test_categories": len(test_counts),
            "new_categories": len(set(test_counts.index) - set(train_counts.index)),
        }

    def _calculate_psi(
        self,
        train_col: pd.Series,
        test_col: pd.Series,
        n_bins: int = 10,
    ) -> float:
        """Calculate Population Stability Index."""
        # Create bins from training data
        bins = np.quantile(train_col, np.linspace(0, 1, n_bins + 1))
        bins = np.unique(bins)
        
        if len(bins) < 2:
            return 0.0
        
        # Calculate distributions
        train_hist, _ = np.histogram(train_col, bins=bins)
        test_hist, _ = np.histogram(test_col, bins=bins)
        
        # Normalize
        train_dist = (train_hist + 1) / (train_hist.sum() + len(train_hist))
        test_dist = (test_hist + 1) / (test_hist.sum() + len(test_hist))
        
        # Calculate PSI
        psi = np.sum((test_dist - train_dist) * np.log(test_dist / train_dist))
        
        return float(psi)

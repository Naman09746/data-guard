"""
Statistical utilities for data analysis.

Provides statistical functions used across quality and leakage detection.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


class DataStatistics:
    """Utility class for statistical analysis of data."""

    @staticmethod
    def get_summary(df: pd.DataFrame) -> dict[str, Any]:
        """
        Get comprehensive summary statistics for a DataFrame.
        
        Args:
            df: Input DataFrame.
        
        Returns:
            Dictionary with summary statistics.
        """
        summary = {
            "shape": {"rows": len(df), "columns": len(df.columns)},
            "dtypes": df.dtypes.value_counts().to_dict(),
            "memory_usage_mb": round(df.memory_usage(deep=True).sum() / 1024 / 1024, 2),
            "missing": {
                "total": int(df.isna().sum().sum()),
                "by_column": df.isna().sum().to_dict(),
            },
        }

        # Numeric statistics
        numeric_df = df.select_dtypes(include=[np.number])
        if len(numeric_df.columns) > 0:
            summary["numeric"] = {
                "columns": list(numeric_df.columns),
                "stats": numeric_df.describe().to_dict(),
            }

        # Categorical statistics
        cat_df = df.select_dtypes(include=["object", "category"])
        if len(cat_df.columns) > 0:
            summary["categorical"] = {
                "columns": list(cat_df.columns),
                "unique_counts": {col: int(cat_df[col].nunique()) for col in cat_df.columns},
            }

        return summary

    @staticmethod
    def detect_outliers(
        series: pd.Series,
        method: str = "zscore",
        threshold: float = 3.0,
    ) -> pd.Series:
        """
        Detect outliers in a numeric series.
        
        Args:
            series: Input numeric series.
            method: Detection method ('zscore', 'iqr', 'mad').
            threshold: Detection threshold.
        
        Returns:
            Boolean series indicating outliers.
        """
        non_null = series.dropna()

        if len(non_null) < 3:
            return pd.Series(False, index=series.index)

        if method == "zscore":
            z_scores = np.abs(stats.zscore(non_null))
            outlier_mask = z_scores > threshold
        elif method == "iqr":
            q1 = non_null.quantile(0.25)
            q3 = non_null.quantile(0.75)
            iqr = q3 - q1
            outlier_mask = (non_null < q1 - threshold * iqr) | (non_null > q3 + threshold * iqr)
        elif method == "mad":
            median = non_null.median()
            mad = np.median(np.abs(non_null - median))
            if mad == 0:
                return pd.Series(False, index=series.index)
            modified_z = 0.6745 * (non_null - median) / mad
            outlier_mask = np.abs(modified_z) > threshold
        else:
            raise ValueError(f"Unknown method: {method}")

        result = pd.Series(False, index=series.index)
        result.loc[non_null.index] = outlier_mask
        return result

    @staticmethod
    def calculate_correlation_matrix(
        df: pd.DataFrame,
        method: str = "pearson",
    ) -> pd.DataFrame:
        """
        Calculate correlation matrix for numeric columns.
        
        Args:
            df: Input DataFrame.
            method: Correlation method ('pearson', 'spearman', 'kendall').
        
        Returns:
            Correlation matrix DataFrame.
        """
        numeric_df = df.select_dtypes(include=[np.number])
        return numeric_df.corr(method=method)

    @staticmethod
    def calculate_feature_target_correlations(
        df: pd.DataFrame,
        target_column: str,
    ) -> dict[str, float]:
        """
        Calculate correlations between features and target.
        
        Args:
            df: Input DataFrame.
            target_column: Name of target column.
        
        Returns:
            Dictionary of feature -> correlation values.
        """
        if target_column not in df.columns:
            return {}

        target = df[target_column]
        correlations = {}

        for col in df.columns:
            if col == target_column:
                continue

            if not pd.api.types.is_numeric_dtype(df[col]):
                continue

            valid_mask = ~(df[col].isna() | target.isna())
            if valid_mask.sum() < 10:
                continue

            try:
                if pd.api.types.is_numeric_dtype(target):
                    corr, _ = stats.pearsonr(
                        df[col][valid_mask].values,
                        target[valid_mask].values
                    )
                else:
                    # Point-biserial for binary target
                    unique_vals = target.dropna().unique()
                    if len(unique_vals) == 2:
                        binary_target = (target == unique_vals[0]).astype(int)
                        corr, _ = stats.pointbiserialr(
                            binary_target[valid_mask].values,
                            df[col][valid_mask].values
                        )
                    else:
                        continue

                correlations[col] = round(abs(float(corr)), 4)
            except Exception:
                continue

        return dict(sorted(correlations.items(), key=lambda x: -x[1]))

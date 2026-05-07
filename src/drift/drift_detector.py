import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timezone
import structlog

logger = structlog.get_logger(__name__)

@dataclass
class FeatureDrift:
    feature_name: str
    drift_score: float
    method: str
    is_drifted: bool
    threshold: float
    p_value: Optional[float] = None
    distribution_current: Optional[List[float]] = None
    distribution_reference: Optional[List[float]] = None

class DriftDetector:
    """
    Detects distribution shifts between reference (training) and current (production) data.
    """
    
    def __init__(self, psi_threshold: float = 0.2, ks_threshold: float = 0.05):
        self.psi_threshold = psi_threshold
        self.ks_threshold = ks_threshold

    def calculate_psi(self, expected: np.ndarray, actual: np.ndarray, buckets: int = 10) -> float:
        """
        Calculate Population Stability Index (PSI).
        PSI < 0.1: No significant change
        PSI < 0.2: Minor change
        PSI >= 0.2: Significant change
        """

        breakpoints = np.arange(0, buckets + 1) / (buckets) * 100
        breakpoints = np.percentile(expected, breakpoints)
        breakpoints[0] = -np.inf
        breakpoints[-1] = np.inf

        expected_percents = np.histogram(expected, bins=breakpoints)[0] / len(expected)
        actual_percents = np.histogram(actual, bins=breakpoints)[0] / len(actual)

        # Handle zero divisions
        expected_percents = np.clip(expected_percents, a_min=0.0001, a_max=None)
        actual_percents = np.clip(actual_percents, a_min=0.0001, a_max=None)

        psi_value = np.sum((expected_percents - actual_percents) * np.log(expected_percents / actual_percents))
        return float(psi_value)

    def analyze_drift(self, reference_df: pd.DataFrame, current_df: pd.DataFrame) -> List[FeatureDrift]:
        """
        Compare two dataframes and return drift metrics for all common features.
        """
        drift_results = []
        common_cols = list(set(reference_df.columns) & set(current_df.columns))
        
        for col in common_cols:
            # Only analyze numeric columns for now
            if not pd.api.types.is_numeric_dtype(reference_df[col]):
                continue
                
            ref_data = reference_df[col].dropna().values
            curr_data = current_df[col].dropna().values
            
            if len(ref_data) == 0 or len(curr_data) == 0:
                continue

            # 1. KS Test (Sensitive to distribution shape)
            ks_stat, p_value = stats.ks_2samp(ref_data, curr_data)
            
            # 2. PSI (Industry standard for stability)
            psi_score = self.calculate_psi(ref_data, curr_data)
            
            is_drifted = psi_score >= self.psi_threshold or p_value <= self.ks_threshold
            
            # Generate histogram data for frontend visualization (Using same bins for comparison)
            min_val = min(np.min(ref_data), np.min(curr_data))
            max_val = max(np.max(ref_data), np.max(curr_data))
            bins = np.linspace(min_val, max_val, 21)
            
            hist_ref, _ = np.histogram(ref_data, bins=bins, density=True)
            hist_curr, _ = np.histogram(curr_data, bins=bins, density=True)

            drift_results.append(FeatureDrift(
                feature_name=col,
                drift_score=psi_score,
                method="PSI + KS-Test",
                is_drifted=is_drifted,
                threshold=self.psi_threshold,
                p_value=float(p_value),
                distribution_reference=hist_ref.tolist(),
                distribution_current=hist_curr.tolist()
            ))
            
        return drift_results

    def get_drift_summary(self, drift_results: List[FeatureDrift]) -> Dict[str, Any]:
        """Generate a summary report of drift analysis."""
        drifted_features = [f.feature_name for f in drift_results if f.is_drifted]
        return {
            "drift_detected": len(drifted_features) > 0,
            "drifted_features_count": len(drifted_features),
            "total_features_analyzed": len(drift_results),
            "drifted_features": drifted_features,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "overall_drift_score": np.mean([f.drift_score for f in drift_results]) if drift_results else 0
        }

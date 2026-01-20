"""
Leakage Risk Scoring Model - ML-based risk prediction for features.

This module transforms leakage detection from rule-based to ML-based,
providing probability-based risk scores for each feature.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

from src.core.logging_config import get_logger

logger = get_logger(__name__)


class RiskLevel(str, Enum):
    """Risk levels for leakage."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    
    @classmethod
    def from_score(cls, score: float) -> "RiskLevel":
        """Convert a score (0-1) to a risk level."""
        if score >= 0.7:
            return cls.HIGH
        elif score >= 0.4:
            return cls.MEDIUM
        return cls.LOW


@dataclass
class FeatureRiskScore:
    """Risk assessment for a single feature."""
    feature_name: str
    risk_score: float  # 0.0 to 1.0
    risk_level: RiskLevel
    risk_percentage: int  # 0 to 100
    contributing_factors: dict[str, float] = field(default_factory=dict)
    recommendations: list[str] = field(default_factory=list)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "feature_name": self.feature_name,
            "risk_score": round(self.risk_score, 4),
            "risk_level": self.risk_level.value,
            "risk_percentage": self.risk_percentage,
            "contributing_factors": {k: round(v, 4) for k, v in self.contributing_factors.items()},
            "recommendations": self.recommendations,
        }


@dataclass
class RiskScoringResult:
    """Result of risk scoring analysis."""
    feature_scores: list[FeatureRiskScore]
    overall_risk: RiskLevel
    high_risk_features: list[str]
    medium_risk_features: list[str]
    duration_seconds: float = 0.0
    model_version: str = "1.0.0"
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "overall_risk": self.overall_risk.value,
            "high_risk_count": len(self.high_risk_features),
            "medium_risk_count": len(self.medium_risk_features),
            "high_risk_features": self.high_risk_features,
            "medium_risk_features": self.medium_risk_features,
            "feature_scores": [fs.to_dict() for fs in self.feature_scores],
            "duration_seconds": round(self.duration_seconds, 4),
            "model_version": self.model_version,
        }


class LeakageRiskFeatureExtractor:
    """
    Extracts signal features for leakage risk prediction.
    
    For each feature in the dataset, extracts:
    - Correlation with target
    - Mutual information
    - Variance across train/test splits
    - Time-lag correlation (if temporal)
    - Stability metrics
    """
    
    def __init__(self, n_splits: int = 5, sample_size: int = 10000) -> None:
        self.n_splits = n_splits
        self.sample_size = sample_size
        self._logger = get_logger("risk_feature_extractor")
    
    def extract_features(
        self,
        data: pd.DataFrame,
        target_column: str,
        time_column: str | None = None,
    ) -> pd.DataFrame:
        """
        Extract leakage signal features for all columns.
        
        Args:
            data: Input DataFrame
            target_column: Name of target column
            time_column: Optional time column for temporal features
            
        Returns:
            DataFrame with one row per feature and signal columns
        """
        start_time = perf_counter()
        
        if target_column not in data.columns:
            raise ValueError(f"Target column '{target_column}' not found in data")
        
        # Sample if needed
        if len(data) > self.sample_size:
            data = data.sample(n=self.sample_size, random_state=42)
        
        # Get feature columns (exclude target and time)
        exclude_cols = {target_column}
        if time_column:
            exclude_cols.add(time_column)
        
        feature_cols = [c for c in data.columns if c not in exclude_cols]
        numeric_cols = data[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
        
        target = data[target_column]
        records = []
        
        for col in numeric_cols:
            try:
                feature_vector = self._extract_single_feature(
                    data[col], target, data, time_column
                )
                feature_vector["feature_name"] = col
                records.append(feature_vector)
            except Exception as e:
                self._logger.warning(f"Failed to extract features for {col}: {e}")
                continue
        
        result_df = pd.DataFrame(records)
        
        self._logger.info(
            "feature_extraction_complete",
            num_features=len(records),
            duration=round(perf_counter() - start_time, 4),
        )
        
        return result_df
    
    def _extract_single_feature(
        self,
        feature: pd.Series,
        target: pd.Series,
        full_data: pd.DataFrame,
        time_column: str | None,
    ) -> dict[str, float]:
        """Extract signal features for a single feature column."""
        signals: dict[str, float] = {}
        
        # 1. Correlation with target
        try:
            if target.dtype in [np.float64, np.int64, float, int]:
                correlation = feature.corr(target)
                signals["correlation_abs"] = abs(correlation) if pd.notna(correlation) else 0.0
            else:
                # For categorical targets, use point-biserial correlation
                target_numeric = pd.factorize(target)[0]
                correlation = feature.corr(pd.Series(target_numeric))
                signals["correlation_abs"] = abs(correlation) if pd.notna(correlation) else 0.0
        except Exception:
            signals["correlation_abs"] = 0.0
        
        # 2. Mutual information approximation (using binned correlation)
        try:
            # Discretize feature into bins and compute normalized mutual info
            feature_binned = pd.qcut(feature.rank(method="first"), q=10, labels=False, duplicates="drop")
            if target.dtype in [np.float64, np.int64, float, int]:
                target_binned = pd.qcut(target.rank(method="first"), q=10, labels=False, duplicates="drop")
            else:
                target_binned = pd.factorize(target)[0]
            
            # Compute contingency table chi-squared as MI proxy
            contingency = pd.crosstab(feature_binned, target_binned)
            chi2, _, _, _ = stats.chi2_contingency(contingency)
            n = len(feature)
            # Normalized by sample size
            signals["mutual_info_proxy"] = min(chi2 / (n * 10), 1.0)  # Normalize
        except Exception:
            signals["mutual_info_proxy"] = 0.0
        
        # 3. Variance across random splits (stability)
        try:
            correlations = []
            for seed in range(self.n_splits):
                np.random.seed(seed)
                mask = np.random.rand(len(feature)) > 0.5
                if mask.sum() > 10 and (~mask).sum() > 10:
                    target_numeric = target if target.dtype in [np.float64, np.int64] else pd.factorize(target)[0]
                    corr = feature[mask].corr(pd.Series(target_numeric)[mask])
                    if pd.notna(corr):
                        correlations.append(abs(corr))
            signals["correlation_variance"] = np.var(correlations) if correlations else 0.0
            signals["correlation_stability"] = 1.0 - min(np.std(correlations) * 2, 1.0) if correlations else 0.5
        except Exception:
            signals["correlation_variance"] = 0.0
            signals["correlation_stability"] = 0.5
        
        # 4. Time-lag behavior (if temporal)
        if time_column and time_column in full_data.columns:
            try:
                # Check if feature correlates better with future target
                sorted_idx = full_data[time_column].argsort()
                sorted_target = target.iloc[sorted_idx].values
                sorted_feature = feature.iloc[sorted_idx].values
                
                # Lag-1 correlation (feature vs next-period target)
                future_corr = np.corrcoef(sorted_feature[:-1], sorted_target[1:])[0, 1]
                current_corr = np.corrcoef(sorted_feature, sorted_target)[0, 1]
                
                # If future correlation is higher, suspicious
                signals["future_correlation_ratio"] = abs(future_corr) / (abs(current_corr) + 0.001)
                signals["temporal_suspicion"] = float(abs(future_corr) > abs(current_corr) * 1.1)
            except Exception:
                signals["future_correlation_ratio"] = 0.0
                signals["temporal_suspicion"] = 0.0
        else:
            signals["future_correlation_ratio"] = 0.0
            signals["temporal_suspicion"] = 0.0
        
        # 5. Feature uniqueness (high cardinality = potential ID leak)
        try:
            unique_ratio = feature.nunique() / len(feature)
            signals["unique_ratio"] = unique_ratio
            signals["high_cardinality_risk"] = float(unique_ratio > 0.9)
        except Exception:
            signals["unique_ratio"] = 0.0
            signals["high_cardinality_risk"] = 0.0
        
        # 6. Missing value pattern correlation
        try:
            missing_mask = feature.isna().astype(int)
            target_numeric = target if target.dtype in [np.float64, np.int64] else pd.factorize(target)[0]
            missing_corr = missing_mask.corr(pd.Series(target_numeric))
            signals["missing_pattern_correlation"] = abs(missing_corr) if pd.notna(missing_corr) else 0.0
        except Exception:
            signals["missing_pattern_correlation"] = 0.0
        
        return signals


class LeakageRiskScoringModel:
    """
    ML-based model for predicting leakage risk per feature.
    
    Uses a Random Forest classifier trained on synthetic leakage patterns
    to predict risk levels (LOW, MEDIUM, HIGH) for each feature.
    """
    
    MODEL_VERSION = "1.0.0"
    
    def __init__(self, model_path: str | Path | None = None) -> None:
        self.model: RandomForestClassifier | None = None
        self.scaler = StandardScaler()
        self.feature_names: list[str] = []
        self.model_path = Path(model_path) if model_path else None
        self._logger = get_logger("risk_scoring_model")
        self.extractor = LeakageRiskFeatureExtractor()
        
        # Initialize with pre-trained weights or train on synthetic data
        self._initialize_model()
    
    def _initialize_model(self) -> None:
        """Initialize the model (train on synthetic data if no saved model)."""
        if self.model_path and self.model_path.exists():
            self._load_model()
        else:
            self._train_on_synthetic_data()
    
    def _train_on_synthetic_data(self) -> None:
        """Train model on synthetic leakage scenarios."""
        self._logger.info("training_on_synthetic_data")
        
        # Generate synthetic training data representing various leakage patterns
        np.random.seed(42)
        n_samples = 1000
        
        # Feature vectors for different risk levels
        # HIGH risk: high correlation, high MI, low stability, temporal suspicion
        high_risk = pd.DataFrame({
            "correlation_abs": np.random.uniform(0.85, 1.0, n_samples // 3),
            "mutual_info_proxy": np.random.uniform(0.6, 1.0, n_samples // 3),
            "correlation_variance": np.random.uniform(0.1, 0.5, n_samples // 3),
            "correlation_stability": np.random.uniform(0.0, 0.4, n_samples // 3),
            "future_correlation_ratio": np.random.uniform(1.0, 2.0, n_samples // 3),
            "temporal_suspicion": np.random.choice([0, 1], n_samples // 3, p=[0.3, 0.7]),
            "unique_ratio": np.random.uniform(0.0, 1.0, n_samples // 3),
            "high_cardinality_risk": np.random.choice([0, 1], n_samples // 3, p=[0.7, 0.3]),
            "missing_pattern_correlation": np.random.uniform(0.0, 0.5, n_samples // 3),
            "label": 2,  # HIGH
        })
        
        # MEDIUM risk: moderate correlation, some instability
        medium_risk = pd.DataFrame({
            "correlation_abs": np.random.uniform(0.5, 0.85, n_samples // 3),
            "mutual_info_proxy": np.random.uniform(0.3, 0.6, n_samples // 3),
            "correlation_variance": np.random.uniform(0.05, 0.15, n_samples // 3),
            "correlation_stability": np.random.uniform(0.4, 0.7, n_samples // 3),
            "future_correlation_ratio": np.random.uniform(0.8, 1.2, n_samples // 3),
            "temporal_suspicion": np.random.choice([0, 1], n_samples // 3, p=[0.7, 0.3]),
            "unique_ratio": np.random.uniform(0.0, 0.8, n_samples // 3),
            "high_cardinality_risk": np.random.choice([0, 1], n_samples // 3, p=[0.9, 0.1]),
            "missing_pattern_correlation": np.random.uniform(0.0, 0.3, n_samples // 3),
            "label": 1,  # MEDIUM
        })
        
        # LOW risk: low correlation, stable, no temporal issues
        low_risk = pd.DataFrame({
            "correlation_abs": np.random.uniform(0.0, 0.5, n_samples // 3),
            "mutual_info_proxy": np.random.uniform(0.0, 0.3, n_samples // 3),
            "correlation_variance": np.random.uniform(0.0, 0.05, n_samples // 3),
            "correlation_stability": np.random.uniform(0.7, 1.0, n_samples // 3),
            "future_correlation_ratio": np.random.uniform(0.5, 1.0, n_samples // 3),
            "temporal_suspicion": np.random.choice([0, 1], n_samples // 3, p=[0.95, 0.05]),
            "unique_ratio": np.random.uniform(0.0, 0.5, n_samples // 3),
            "high_cardinality_risk": np.random.choice([0, 1], n_samples // 3, p=[0.98, 0.02]),
            "missing_pattern_correlation": np.random.uniform(0.0, 0.1, n_samples // 3),
            "label": 0,  # LOW
        })
        
        # Combine
        training_data = pd.concat([high_risk, medium_risk, low_risk], ignore_index=True)
        
        self.feature_names = [c for c in training_data.columns if c != "label"]
        X = training_data[self.feature_names]
        y = training_data["label"]
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train Random Forest
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1,
        )
        self.model.fit(X_scaled, y)
        
        # Validate
        scores = cross_val_score(self.model, X_scaled, y, cv=5)
        self._logger.info(
            "model_trained",
            cv_accuracy=round(scores.mean(), 4),
            cv_std=round(scores.std(), 4),
        )
    
    def _load_model(self) -> None:
        """Load a pre-trained model from disk."""
        # Placeholder for model loading - would use joblib in production
        self._train_on_synthetic_data()
    
    def predict_risk(
        self,
        data: pd.DataFrame,
        target_column: str,
        time_column: str | None = None,
    ) -> RiskScoringResult:
        """
        Predict leakage risk for all features in the dataset.
        
        Args:
            data: Input DataFrame
            target_column: Name of target column
            time_column: Optional time column
            
        Returns:
            RiskScoringResult with scores for all features
        """
        start_time = perf_counter()
        
        if self.model is None:
            raise RuntimeError("Model not initialized")
        
        # Extract features
        feature_df = self.extractor.extract_features(data, target_column, time_column)
        
        if len(feature_df) == 0:
            return RiskScoringResult(
                feature_scores=[],
                overall_risk=RiskLevel.LOW,
                high_risk_features=[],
                medium_risk_features=[],
                duration_seconds=perf_counter() - start_time,
            )
        
        # Prepare features for prediction
        X = feature_df[self.feature_names].fillna(0)
        X_scaled = self.scaler.transform(X)
        
        # Get predictions and probabilities
        predictions = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)
        
        # Build results
        feature_scores: list[FeatureRiskScore] = []
        high_risk: list[str] = []
        medium_risk: list[str] = []
        
        for i, row in feature_df.iterrows():
            feature_name = row["feature_name"]
            pred_class = predictions[i]
            probs = probabilities[i]
            
            # Use weighted score: probability of HIGH * 1.0 + probability of MEDIUM * 0.5
            risk_score = probs[2] * 1.0 + probs[1] * 0.5
            risk_level = RiskLevel.from_score(risk_score)
            
            # Contributing factors
            factors = {
                "correlation": row.get("correlation_abs", 0),
                "mutual_info": row.get("mutual_info_proxy", 0),
                "stability": row.get("correlation_stability", 0.5),
                "temporal_risk": row.get("temporal_suspicion", 0),
                "cardinality_risk": row.get("high_cardinality_risk", 0),
            }
            
            # Generate recommendations
            recommendations = self._generate_recommendations(row, risk_level)
            
            score = FeatureRiskScore(
                feature_name=feature_name,
                risk_score=risk_score,
                risk_level=risk_level,
                risk_percentage=int(risk_score * 100),
                contributing_factors=factors,
                recommendations=recommendations,
            )
            feature_scores.append(score)
            
            if risk_level == RiskLevel.HIGH:
                high_risk.append(feature_name)
            elif risk_level == RiskLevel.MEDIUM:
                medium_risk.append(feature_name)
        
        # Determine overall risk
        if high_risk:
            overall_risk = RiskLevel.HIGH
        elif medium_risk:
            overall_risk = RiskLevel.MEDIUM
        else:
            overall_risk = RiskLevel.LOW
        
        duration = perf_counter() - start_time
        
        self._logger.info(
            "risk_scoring_complete",
            num_features=len(feature_scores),
            high_risk_count=len(high_risk),
            medium_risk_count=len(medium_risk),
            duration=round(duration, 4),
        )
        
        return RiskScoringResult(
            feature_scores=sorted(feature_scores, key=lambda x: -x.risk_score),
            overall_risk=overall_risk,
            high_risk_features=high_risk,
            medium_risk_features=medium_risk,
            duration_seconds=duration,
            model_version=self.MODEL_VERSION,
        )
    
    def _generate_recommendations(
        self,
        feature_row: pd.Series,
        risk_level: RiskLevel,
    ) -> list[str]:
        """Generate actionable recommendations based on risk factors."""
        recommendations = []
        
        if risk_level == RiskLevel.LOW:
            return ["Feature appears safe to use"]
        
        correlation = feature_row.get("correlation_abs", 0)
        temporal = feature_row.get("temporal_suspicion", 0)
        cardinality = feature_row.get("high_cardinality_risk", 0)
        stability = feature_row.get("correlation_stability", 1)
        
        if correlation > 0.9:
            recommendations.append(
                "CRITICAL: Extremely high correlation suggests direct target leakage. "
                "Verify this feature is not derived from the target."
            )
        elif correlation > 0.7:
            recommendations.append(
                "High correlation detected. Investigate if this feature is available at prediction time."
            )
        
        if temporal > 0:
            recommendations.append(
                "Temporal leakage suspected. Feature may contain future information. "
                "Check feature engineering pipeline for time-travel bugs."
            )
        
        if cardinality > 0:
            recommendations.append(
                "High cardinality suggests potential ID or unique identifier leak. "
                "Consider removing or encoding this feature."
            )
        
        if stability < 0.5:
            recommendations.append(
                "Unstable correlation across splits. Feature may not generalize well."
            )
        
        if not recommendations:
            recommendations.append("Review feature definition and availability at prediction time.")
        
        return recommendations


# Convenience function for quick risk assessment
def assess_feature_risk(
    data: pd.DataFrame,
    target_column: str,
    time_column: str | None = None,
) -> RiskScoringResult:
    """
    Quick function to assess leakage risk for all features.
    
    Args:
        data: Input DataFrame
        target_column: Name of target column
        time_column: Optional time column
        
    Returns:
        RiskScoringResult with risk scores
    """
    model = LeakageRiskScoringModel()
    return model.predict_risk(data, target_column, time_column)

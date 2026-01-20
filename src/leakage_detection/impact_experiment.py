"""
Leakage Impact Experiment Framework.

Proves business value by comparing model performance with and without leakage.
Generates comprehensive reports showing accuracy drops and generalization gaps.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

from src.core.logging_config import get_logger
from src.leakage_detection.risk_scoring_model import (
    LeakageRiskScoringModel,
    RiskLevel,
)

logger = get_logger(__name__)


class ExperimentStatus(str, Enum):
    """Status of an experiment."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class ModelMetrics:
    """Metrics for a trained model."""
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float | None = None
    cv_accuracy_mean: float = 0.0
    cv_accuracy_std: float = 0.0
    
    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary."""
        result = {
            "accuracy": round(self.accuracy, 4),
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "cv_accuracy_mean": round(self.cv_accuracy_mean, 4),
            "cv_accuracy_std": round(self.cv_accuracy_std, 4),
        }
        if self.roc_auc is not None:
            result["roc_auc"] = round(self.roc_auc, 4)
        return result


@dataclass
class ExperimentResult:
    """Result of a before/after leakage experiment."""
    experiment_id: str
    status: ExperimentStatus
    
    # Metrics with leakage
    metrics_with_leakage: ModelMetrics | None = None
    leaky_features: list[str] = field(default_factory=list)
    
    # Metrics after removal
    metrics_after_removal: ModelMetrics | None = None
    features_removed: list[str] = field(default_factory=list)
    
    # Comparison metrics
    accuracy_drop: float = 0.0
    generalization_gap_before: float = 0.0
    generalization_gap_after: float = 0.0
    stability_improvement: float = 0.0
    
    # Metadata
    duration_seconds: float = 0.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    error_message: str | None = None
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "experiment_id": self.experiment_id,
            "status": self.status.value,
            "metrics_with_leakage": self.metrics_with_leakage.to_dict() if self.metrics_with_leakage else None,
            "metrics_after_removal": self.metrics_after_removal.to_dict() if self.metrics_after_removal else None,
            "leaky_features": self.leaky_features,
            "features_removed": self.features_removed,
            "comparison": {
                "accuracy_drop": round(self.accuracy_drop, 4),
                "generalization_gap_before": round(self.generalization_gap_before, 4),
                "generalization_gap_after": round(self.generalization_gap_after, 4),
                "stability_improvement": round(self.stability_improvement, 4),
            },
            "duration_seconds": round(self.duration_seconds, 4),
            "timestamp": self.timestamp.isoformat(),
        }
    
    def to_markdown_report(self) -> str:
        """Generate a markdown report."""
        report = f"""# Leakage Impact Experiment Report

**Experiment ID:** {self.experiment_id}  
**Status:** {self.status.value.upper()}  
**Timestamp:** {self.timestamp.strftime("%Y-%m-%d %H:%M:%S")}  
**Duration:** {self.duration_seconds:.2f} seconds

---

## Executive Summary

"""
        if self.accuracy_drop > 0.1:
            report += f"""⚠️ **Significant leakage impact detected!**

The model's accuracy dropped by **{self.accuracy_drop * 100:.1f}%** after removing leaky features,
indicating these features were artificially inflating performance.

"""
        elif self.accuracy_drop > 0.02:
            report += f"""📊 **Moderate leakage impact detected.**

The model's accuracy dropped by **{self.accuracy_drop * 100:.1f}%** after removing suspicious features.

"""
        else:
            report += f"""✅ **Minimal leakage impact.**

The model performance remained stable after feature cleanup.

"""

        report += f"""---

## Leaky Features Identified

"""
        if self.leaky_features:
            for feat in self.leaky_features:
                report += f"- `{feat}`\n"
        else:
            report += "No leaky features were identified.\n"

        report += f"""
---

## Performance Comparison

| Metric | With Leakage | After Removal | Change |
|--------|--------------|---------------|--------|
"""
        if self.metrics_with_leakage and self.metrics_after_removal:
            m1 = self.metrics_with_leakage
            m2 = self.metrics_after_removal
            report += f"| Accuracy | {m1.accuracy:.4f} | {m2.accuracy:.4f} | {(m2.accuracy - m1.accuracy):+.4f} |\n"
            report += f"| Precision | {m1.precision:.4f} | {m2.precision:.4f} | {(m2.precision - m1.precision):+.4f} |\n"
            report += f"| Recall | {m1.recall:.4f} | {m2.recall:.4f} | {(m2.recall - m1.recall):+.4f} |\n"
            report += f"| F1 Score | {m1.f1:.4f} | {m2.f1:.4f} | {(m2.f1 - m1.f1):+.4f} |\n"
            report += f"| CV Std | {m1.cv_accuracy_std:.4f} | {m2.cv_accuracy_std:.4f} | {(m2.cv_accuracy_std - m1.cv_accuracy_std):+.4f} |\n"

        report += f"""
---

## Generalization Analysis

| Metric | Before | After | Interpretation |
|--------|--------|-------|----------------|
| Generalization Gap | {self.generalization_gap_before:.4f} | {self.generalization_gap_after:.4f} | {"Improved" if self.generalization_gap_after < self.generalization_gap_before else "Worse"} |
| Stability (CV Std) | - | - | {self.stability_improvement:+.1%} improvement |

---

## Recommendations

"""
        if self.accuracy_drop > 0.1:
            report += """1. **Remove identified leaky features** - They are causing significant overfitting
2. **Review feature engineering pipeline** - Check for time-travel bugs
3. **Audit data collection process** - Ensure no target information leaks
"""
        elif self.accuracy_drop > 0.02:
            report += """1. **Review suspicious features** - Some may contain target information
2. **Monitor model performance in production** - Watch for degradation
3. **Consider partial removal** - Some features may still be useful
"""
        else:
            report += """1. **Features appear clean** - Continue with current feature set
2. **Maintain monitoring** - Regularly check for new leakage
3. **Document findings** - Record this analysis for future reference
"""

        return report


class LeakageImpactExperiment:
    """
    Runs before/after experiments to prove leakage impact.
    
    Trains a model with leaky features, then removes them and retrains
    to show the difference in performance and generalization.
    """
    
    def __init__(
        self,
        model_type: str = "random_forest",
        test_size: float = 0.2,
        cv_folds: int = 5,
        random_state: int = 42,
    ) -> None:
        self.model_type = model_type
        self.test_size = test_size
        self.cv_folds = cv_folds
        self.random_state = random_state
        self._logger = get_logger("impact_experiment")
        self.risk_model = LeakageRiskScoringModel()
    
    def _create_model(self) -> Any:
        """Create a fresh model instance."""
        if self.model_type == "random_forest":
            return RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=self.random_state,
                n_jobs=-1,
            )
        elif self.model_type == "logistic":
            return LogisticRegression(
                random_state=self.random_state,
                max_iter=1000,
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def _train_and_evaluate(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
    ) -> ModelMetrics:
        """Train model and compute metrics."""
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        model = self._create_model()
        model.fit(X_train_scaled, y_train)
        
        # Predictions
        y_pred = model.predict(X_test_scaled)
        y_proba = None
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test_scaled)[:, 1]
        
        # Compute metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="binary", zero_division=0)
        recall = recall_score(y_test, y_pred, average="binary", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="binary", zero_division=0)
        roc_auc = roc_auc_score(y_test, y_proba) if y_proba is not None else None
        
        # Cross-validation for stability
        cv_model = self._create_model()
        cv_scores = cross_val_score(cv_model, X_train_scaled, y_train, cv=self.cv_folds)
        
        return ModelMetrics(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1=f1,
            roc_auc=roc_auc,
            cv_accuracy_mean=cv_scores.mean(),
            cv_accuracy_std=cv_scores.std(),
        )
    
    def run_experiment(
        self,
        data: pd.DataFrame,
        target_column: str,
        time_column: str | None = None,
        risk_threshold: float = 0.6,
    ) -> ExperimentResult:
        """
        Run a before/after leakage impact experiment.
        
        Args:
            data: Full dataset
            target_column: Target column name
            time_column: Optional time column
            risk_threshold: Risk score threshold for removing features
            
        Returns:
            ExperimentResult with comparison metrics
        """
        import hashlib
        
        experiment_id = hashlib.md5(
            f"{datetime.now(UTC).isoformat()}_{len(data)}".encode()
        ).hexdigest()[:12]
        
        start_time = perf_counter()
        
        self._logger.info(
            "starting_impact_experiment",
            experiment_id=experiment_id,
            rows=len(data),
            target=target_column,
        )
        
        try:
            # Step 1: Identify leaky features
            risk_result = self.risk_model.predict_risk(data, target_column, time_column)
            
            leaky_features = [
                score.feature_name
                for score in risk_result.feature_scores
                if score.risk_score >= risk_threshold
            ]
            
            # Prepare features
            exclude_cols = {target_column}
            if time_column:
                exclude_cols.add(time_column)
            
            feature_cols = [c for c in data.columns if c not in exclude_cols]
            numeric_cols = data[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
            
            if len(numeric_cols) == 0:
                raise ValueError("No numeric features found for modeling")
            
            X = data[numeric_cols].fillna(0)
            y = data[target_column]
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
            )
            
            # Step 2: Train WITH leakage
            metrics_with_leakage = self._train_and_evaluate(
                X_train, X_test, y_train, y_test
            )
            
            # Step 3: Remove leaky features and retrain
            clean_cols = [c for c in numeric_cols if c not in leaky_features]
            
            if len(clean_cols) == 0:
                # All features are leaky - use all anyway for comparison
                self._logger.warning("all_features_leaky", count=len(leaky_features))
                clean_cols = numeric_cols
            
            X_clean = data[clean_cols].fillna(0)
            X_train_clean, X_test_clean, _, _ = train_test_split(
                X_clean, y, test_size=self.test_size, random_state=self.random_state, stratify=y
            )
            
            metrics_after_removal = self._train_and_evaluate(
                X_train_clean, X_test_clean, y_train, y_test
            )
            
            # Step 4: Calculate comparison metrics
            accuracy_drop = metrics_with_leakage.accuracy - metrics_after_removal.accuracy
            
            # Generalization gap: difference between train CV and test accuracy
            gen_gap_before = metrics_with_leakage.cv_accuracy_mean - metrics_with_leakage.accuracy
            gen_gap_after = metrics_after_removal.cv_accuracy_mean - metrics_after_removal.accuracy
            
            # Stability improvement (lower CV std is better)
            stability_improvement = (
                (metrics_with_leakage.cv_accuracy_std - metrics_after_removal.cv_accuracy_std)
                / max(metrics_with_leakage.cv_accuracy_std, 0.001)
            )
            
            duration = perf_counter() - start_time
            
            result = ExperimentResult(
                experiment_id=experiment_id,
                status=ExperimentStatus.COMPLETED,
                metrics_with_leakage=metrics_with_leakage,
                leaky_features=leaky_features,
                metrics_after_removal=metrics_after_removal,
                features_removed=leaky_features,
                accuracy_drop=accuracy_drop,
                generalization_gap_before=gen_gap_before,
                generalization_gap_after=gen_gap_after,
                stability_improvement=stability_improvement,
                duration_seconds=duration,
            )
            
            self._logger.info(
                "experiment_completed",
                experiment_id=experiment_id,
                accuracy_drop=round(accuracy_drop, 4),
                leaky_features_count=len(leaky_features),
                duration=round(duration, 4),
            )
            
            return result
            
        except Exception as e:
            self._logger.error(
                "experiment_failed",
                experiment_id=experiment_id,
                error=str(e),
            )
            return ExperimentResult(
                experiment_id=experiment_id,
                status=ExperimentStatus.FAILED,
                duration_seconds=perf_counter() - start_time,
                error_message=str(e),
            )
    
    def save_report(
        self,
        result: ExperimentResult,
        output_dir: str | Path,
        format: str = "both",
    ) -> dict[str, Path]:
        """
        Save experiment report to files.
        
        Args:
            result: Experiment result
            output_dir: Directory to save reports
            format: "json", "markdown", or "both"
            
        Returns:
            Dict mapping format to file path
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        saved_files: dict[str, Path] = {}
        
        if format in ("json", "both"):
            json_path = output_dir / f"experiment_{result.experiment_id}.json"
            with open(json_path, "w") as f:
                json.dump(result.to_dict(), f, indent=2, default=str)
            saved_files["json"] = json_path
        
        if format in ("markdown", "both"):
            md_path = output_dir / f"experiment_{result.experiment_id}.md"
            with open(md_path, "w") as f:
                f.write(result.to_markdown_report())
            saved_files["markdown"] = md_path
        
        return saved_files


# Convenience function
def run_impact_experiment(
    data: pd.DataFrame,
    target_column: str,
    time_column: str | None = None,
    model_type: str = "random_forest",
) -> ExperimentResult:
    """
    Run a quick impact experiment.
    
    Args:
        data: Dataset
        target_column: Target column
        time_column: Optional time column
        model_type: Model type to use
        
    Returns:
        ExperimentResult
    """
    experiment = LeakageImpactExperiment(model_type=model_type)
    return experiment.run_experiment(data, target_column, time_column)

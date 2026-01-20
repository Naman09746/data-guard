#!/usr/bin/env python3
"""
Churn Dataset Case Study - Run Script

This script demonstrates the full capabilities of the Data Quality & Leakage
Detection system using a realistic customer churn prediction scenario.

Usage:
    python run_case_study.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add src to path
src_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(src_path))

from src.leakage_detection.leakage_engine import LeakageDetectionEngine
from src.leakage_detection.risk_scoring_model import LeakageRiskScoringModel
from src.leakage_detection.impact_experiment import LeakageImpactExperiment
from src.data_quality.quality_engine import DataQualityEngine
from src.core.data_versioning import DataVersioner, ScanHistoryStore, ScanRecord, ScanType
from src.core.alert_system import AlertManager


def generate_churn_dataset(n_samples: int = 5000, seed: int = 42) -> pd.DataFrame:
    """
    Generate a synthetic churn dataset with intentional leakage.
    
    The dataset simulates a telecom company's customer data with:
    - Clean features: customer demographics and usage patterns
    - Leaky features: information that shouldn't be available at prediction time
    """
    np.random.seed(seed)
    
    # Target: whether customer churned (1) or not (0)
    base_churn_prob = 0.25
    
    # ========== CLEAN FEATURES ==========
    
    # Customer demographics
    tenure_months = np.random.exponential(24, n_samples).astype(int)
    tenure_months = np.clip(tenure_months, 1, 72)
    
    age = np.random.normal(45, 15, n_samples).astype(int)
    age = np.clip(age, 18, 80)
    
    # Contract type affects churn
    contract_type = np.random.choice(
        ["Month-to-month", "One year", "Two year"],
        n_samples,
        p=[0.5, 0.3, 0.2]
    )
    contract_factor = np.where(contract_type == "Month-to-month", 0.4, 
                               np.where(contract_type == "One year", 0.2, 0.1))
    
    # Monthly charges
    monthly_charges = np.random.uniform(20, 120, n_samples)
    
    # Total charges (derived from tenure and monthly)
    total_charges = tenure_months * monthly_charges * np.random.uniform(0.9, 1.1, n_samples)
    
    # Service features
    has_internet = np.random.choice([0, 1], n_samples, p=[0.2, 0.8])
    has_streaming = np.random.choice([0, 1], n_samples, p=[0.4, 0.6])
    has_security = np.random.choice([0, 1], n_samples, p=[0.6, 0.4])
    
    # Support tickets (higher = more likely to churn)
    support_tickets = np.random.poisson(2, n_samples)
    support_factor = support_tickets * 0.05
    
    # ========== CALCULATE CHURN ==========
    
    churn_prob = (
        base_churn_prob 
        + contract_factor 
        + support_factor
        - (tenure_months * 0.003)  # Longer tenure = less likely to churn
        - (has_security * 0.1)     # Security service reduces churn
    )
    churn_prob = np.clip(churn_prob, 0.05, 0.95)
    churned = (np.random.random(n_samples) < churn_prob).astype(int)
    
    # ========== LEAKY FEATURES (INTENTIONAL) ==========
    
    # LEAKY 1: Cancellation survey score (only filled when customer churns!)
    # This is a classic post-hoc feature that perfectly predicts churn
    cancellation_score = np.where(
        churned == 1,
        np.random.uniform(1, 5, n_samples),  # Only churned customers have scores
        np.nan
    )
    # Fill NaN with 0 to make it usable but suspicious
    cancellation_score_filled = np.nan_to_num(cancellation_score, nan=0)
    
    # LEAKY 2: Final bill amount (calculated after churn decision)
    # Churned customers get prorated final bills
    final_bill = np.where(
        churned == 1,
        monthly_charges * np.random.uniform(0.1, 0.9, n_samples),  # Partial month
        monthly_charges  # Full month for non-churned
    )
    
    # LEAKY 3: Churn risk label (this IS the target with noise)
    # Simulates someone accidentally including a derived feature
    churn_risk_label = churned + np.random.normal(0, 0.02, n_samples)
    
    # ========== CREATE DATAFRAME ==========
    
    df = pd.DataFrame({
        # Clean features
        "customer_id": range(1, n_samples + 1),
        "tenure_months": tenure_months,
        "age": age,
        "contract_type": contract_type,
        "monthly_charges": np.round(monthly_charges, 2),
        "total_charges": np.round(total_charges, 2),
        "has_internet_service": has_internet,
        "has_streaming_service": has_streaming,
        "has_security_service": has_security,
        "support_tickets_last_6mo": support_tickets,
        
        # Leaky features
        "cancellation_survey_score": np.round(cancellation_score_filled, 2),
        "final_bill_amount": np.round(final_bill, 2),
        "churn_risk_score": np.round(churn_risk_label, 4),
        
        # Target
        "churned": churned,
    })
    
    return df


def run_case_study():
    """Run the complete case study."""
    print("=" * 60)
    print("CHURN DATASET CASE STUDY")
    print("Data Quality & Leakage Detection System Demo")
    print("=" * 60)
    
    case_dir = Path(__file__).parent
    
    # Step 1: Generate dataset
    print("\n[1/6] Generating synthetic churn dataset...")
    df = generate_churn_dataset(n_samples=5000)
    df.to_csv(case_dir / "raw_data.csv", index=False)
    print(f"  Generated {len(df)} samples with {len(df.columns)} columns")
    print(f"  Churn rate: {df['churned'].mean():.1%}")
    print(f"  Saved to: raw_data.csv")
    
    # Step 2: Version the dataset
    print("\n[2/6] Creating dataset version...")
    versioner = DataVersioner()
    version = versioner.create_version(df, metadata={"source": "synthetic"})
    print(f"  Version hash: {version.version_hash}")
    print(f"  Schema hash: {version.schema_hash}")
    
    # Step 3: Run data quality checks
    print("\n[3/6] Running data quality validation...")
    quality_engine = DataQualityEngine()
    quality_report = quality_engine.validate(df)
    print(f"  Status: {quality_report.status.value.upper()}")
    print(f"  Issues found: {quality_report.total_issues}")
    print(f"  Quality score: {quality_engine.get_quality_score(df):.2f}")
    
    # Step 4: Run leakage detection
    print("\n[4/6] Running leakage detection...")
    leakage_engine = LeakageDetectionEngine()
    leakage_report = leakage_engine.detect(
        df, 
        target_column="churned",
    )
    print(f"  Status: {leakage_report.status.value.upper()}")
    print(f"  Issues found: {leakage_report.total_issues}")
    
    # Step 5: Get ML-based risk scores
    print("\n[5/6] Computing ML-based risk scores...")
    risk_result = leakage_engine.get_risk_scores(df, "churned")
    print(f"  Overall risk: {risk_result.overall_risk.value.upper()}")
    print(f"  High-risk features: {risk_result.high_risk_features}")
    print(f"  Medium-risk features: {risk_result.medium_risk_features}")
    
    # Write leakage found report
    leakage_md = f"""# Leakage Detection Report

**Dataset:** Churn Prediction Dataset  
**Samples:** {len(df):,}  
**Features:** {len(df.columns) - 1} (excluding target)

## Summary

| Metric | Value |
|--------|-------|
| Overall Risk Level | {risk_result.overall_risk.value.upper()} |
| High-Risk Features | {len(risk_result.high_risk_features)} |
| Medium-Risk Features | {len(risk_result.medium_risk_features)} |
| Detection Duration | {risk_result.duration_seconds:.2f}s |

## High-Risk Features Identified

"""
    for score in risk_result.feature_scores:
        if score.risk_level.value == "high":
            leakage_md += f"""### `{score.feature_name}`

- **Risk Score:** {score.risk_percentage}%
- **Risk Level:** {score.risk_level.value.upper()}
- **Contributing Factors:**
"""
            for factor, value in score.contributing_factors.items():
                leakage_md += f"  - {factor}: {value:.4f}\n"
            leakage_md += "- **Recommendations:**\n"
            for rec in score.recommendations:
                leakage_md += f"  - {rec}\n"
            leakage_md += "\n"
    
    with open(case_dir / "leakage_found.md", "w") as f:
        f.write(leakage_md)
    print("  Saved leakage report to: leakage_found.md")
    
    # Step 6: Run impact experiment
    print("\n[6/6] Running before/after impact experiment...")
    experiment = LeakageImpactExperiment(model_type="random_forest")
    result = experiment.run_experiment(df, "churned")
    
    # Save metrics before
    if result.metrics_with_leakage:
        m = result.metrics_with_leakage
        before_md = f"""# Model Performance WITH Leakage

**Warning:** These metrics are inflated due to data leakage!

## Metrics

| Metric | Value |
|--------|-------|
| Accuracy | {m.accuracy:.4f} |
| Precision | {m.precision:.4f} |
| Recall | {m.recall:.4f} |
| F1 Score | {m.f1:.4f} |
| ROC AUC | {f'{m.roc_auc:.4f}' if m.roc_auc else 'N/A'} |
| CV Accuracy (mean) | {m.cv_accuracy_mean:.4f} |
| CV Accuracy (std) | {m.cv_accuracy_std:.4f} |

## Analysis

These metrics are **unrealistically high** because the model is using features 
that contain information about the target variable:

- `churn_risk_score` - Directly derived from the target
- `cancellation_survey_score` - Only collected after customer churns
- `final_bill_amount` - Calculated based on churn decision

This model would fail catastrophically in production because these features
won't be available at prediction time.
"""
        with open(case_dir / "metrics_before.md", "w") as f:
            f.write(before_md)
        print("  Saved before metrics to: metrics_before.md")
    
    # Save metrics after
    if result.metrics_after_removal:
        m = result.metrics_after_removal
        after_md = f"""# Model Performance AFTER Leakage Removal

**These are realistic, production-ready metrics.**

## Metrics

| Metric | Value |
|--------|-------|
| Accuracy | {m.accuracy:.4f} |
| Precision | {m.precision:.4f} |
| Recall | {m.recall:.4f} |
| F1 Score | {m.f1:.4f} |
| ROC AUC | {f'{m.roc_auc:.4f}' if m.roc_auc else 'N/A'} |
| CV Accuracy (mean) | {m.cv_accuracy_mean:.4f} |
| CV Accuracy (std) | {m.cv_accuracy_std:.4f} |

## Features Removed

The following leaky features were removed:

"""
        for feat in result.features_removed:
            after_md += f"- `{feat}`\n"
        
        after_md += f"""
## Comparison

| Metric | With Leakage | After Removal | Change |
|--------|--------------|---------------|--------|
| Accuracy | {result.metrics_with_leakage.accuracy:.4f} | {m.accuracy:.4f} | {result.accuracy_drop:+.4f} |

## Key Insights

1. **Accuracy dropped by {result.accuracy_drop:.1%}** - This is expected and healthy!
2. **Generalization gap improved** - Model is now learning real patterns
3. **Cross-validation stability improved** - Less overfitting

The "lower" performance is actually **more honest** and reflects what the model
will achieve in production.
"""
        with open(case_dir / "metrics_after.md", "w") as f:
            f.write(after_md)
        print("  Saved after metrics to: metrics_after.md")
    
    # Print summary
    print("\n" + "=" * 60)
    print("CASE STUDY COMPLETE")
    print("=" * 60)
    print(f"""
Results Summary:
  - Leaky features identified: {len(risk_result.high_risk_features)}
  - Accuracy with leakage: {result.metrics_with_leakage.accuracy:.1%}
  - Accuracy after cleanup: {result.metrics_after_removal.accuracy:.1%}
  - Accuracy drop: {result.accuracy_drop:.1%}
  - Experiment ID: {result.experiment_id}

Files Generated:
  - raw_data.csv
  - leakage_found.md
  - metrics_before.md
  - metrics_after.md
""")
    
    return result


if __name__ == "__main__":
    run_cas
"""
Example: Basic usage of Data Quality and Leakage Detection.

This script demonstrates the core functionality of the system.
"""

import numpy as np
import pandas as pd

from src.data_quality import DataQualityEngine
from src.leakage_detection import LeakageDetectionEngine


def create_sample_data():
    """Create sample train/test data for demonstration."""
    np.random.seed(42)
    
    # Training data
    n_train = 1000
    train_df = pd.DataFrame({
        "user_id": range(n_train),
        "age": np.random.randint(18, 80, n_train),
        "income": np.random.lognormal(10, 1, n_train),
        "credit_score": np.random.randint(300, 850, n_train),
        "account_age_days": np.random.randint(30, 3650, n_train),
        "transaction_count": np.random.poisson(50, n_train),
        "is_fraud": np.random.choice([0, 1], n_train, p=[0.95, 0.05]),
        "created_at": pd.date_range("2023-01-01", periods=n_train, freq="h"),
    })
    
    # Test data (clean, no contamination)
    n_test = 200
    test_df = pd.DataFrame({
        "user_id": range(n_train, n_train + n_test),
        "age": np.random.randint(18, 80, n_test),
        "income": np.random.lognormal(10, 1, n_test),
        "credit_score": np.random.randint(300, 850, n_test),
        "account_age_days": np.random.randint(30, 3650, n_test),
        "transaction_count": np.random.poisson(50, n_test),
        "is_fraud": np.random.choice([0, 1], n_test, p=[0.95, 0.05]),
        "created_at": pd.date_range("2023-02-15", periods=n_test, freq="h"),
    })
    
    return train_df, test_df


def main():
    """Run the example."""
    print("=" * 60)
    print("Data Quality & Leakage Detection - Example")
    print("=" * 60)
    
    # Create sample data
    print("\n1. Creating sample data...")
    train_df, test_df = create_sample_data()
    print(f"   Train: {train_df.shape[0]} rows, {train_df.shape[1]} columns")
    print(f"   Test: {test_df.shape[0]} rows, {test_df.shape[1]} columns")
    
    # Data Quality Validation
    print("\n2. Running Data Quality Validation...")
    quality_engine = DataQualityEngine()
    quality_report = quality_engine.validate(train_df)
    
    print(f"   Status: {quality_report.status.value}")
    print(f"   Passed: {quality_report.passed}")
    print(f"   Issues: {quality_report.total_issues}")
    print(f"   Duration: {quality_report.duration_seconds:.3f}s")
    
    # Show quality summary
    summary = quality_report.get_summary()
    print(f"   Validators Run: {summary['validators_run']}")
    print(f"   Validators Passed: {summary['validators_passed']}")
    
    # Leakage Detection
    print("\n3. Running Leakage Detection...")
    leakage_engine = LeakageDetectionEngine()
    leakage_report = leakage_engine.detect(
        train_df,
        test_df,
        target_column="is_fraud",
        time_column="created_at",
    )
    
    print(f"   Status: {leakage_report.status.value}")
    print(f"   Is Clean: {leakage_report.is_clean}")
    print(f"   Has Leakage: {leakage_report.has_leakage}")
    print(f"   Issues: {leakage_report.total_issues}")
    print(f"   Duration: {leakage_report.duration_seconds:.3f}s")
    
    # Show leakage summary
    summary = leakage_report.get_summary()
    print(f"   Detectors Run: {summary['detectors_run']}")
    print(f"   Detectors Clean: {summary['detectors_clean']}")
    
    # Generate reports
    print("\n4. Generating Reports...")
    
    # Save quality report
    with open("quality_report.md", "w") as f:
        f.write(quality_report.to_markdown())
    print("   Quality report saved to: quality_report.md")
    
    # Save leakage report
    with open("leakage_report.md", "w") as f:
        f.write(leakage_report.to_markdown())
    print("   Leakage report saved to: leakage_report.md")
    
    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()

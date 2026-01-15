"""Test configuration and fixtures."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Create a sample DataFrame for testing."""
    np.random.seed(42)
    return pd.DataFrame({
        "id": range(100),
        "name": [f"item_{i}" for i in range(100)],
        "value": np.random.randn(100),
        "category": np.random.choice(["A", "B", "C"], 100),
        "timestamp": pd.date_range("2024-01-01", periods=100, freq="h"),
        "is_active": np.random.choice([True, False], 100),
    })


@pytest.fixture
def sample_df_with_nulls() -> pd.DataFrame:
    """Create a sample DataFrame with missing values."""
    np.random.seed(42)
    df = pd.DataFrame({
        "id": range(100),
        "name": [f"item_{i}" if i % 5 != 0 else None for i in range(100)],
        "value": [np.random.randn() if i % 3 != 0 else np.nan for i in range(100)],
        "category": np.random.choice(["A", "B", "C", None], 100),
    })
    return df


@pytest.fixture
def train_df() -> pd.DataFrame:
    """Create training DataFrame."""
    np.random.seed(42)
    n = 500
    return pd.DataFrame({
        "feature1": np.random.randn(n),
        "feature2": np.random.randn(n),
        "feature3": np.random.randn(n),
        "target": np.random.choice([0, 1], n),
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="h"),  # Ends Jan 21
    })


@pytest.fixture
def test_df() -> pd.DataFrame:
    """Create test DataFrame - starts AFTER train ends."""
    np.random.seed(456)  # Different seed for different random values
    n = 100
    return pd.DataFrame({
        "feature1": np.random.randn(n) * 2 + 5,  # Different distribution
        "feature2": np.random.randn(n) * 2 - 3,  # Different distribution
        "feature3": np.random.randn(n) * 1.5,
        "target": np.random.choice([0, 1], n),
        "timestamp": pd.date_range("2024-03-01", periods=n, freq="h"),  # Starts March 1
    })


@pytest.fixture
def contaminated_train_test() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create train/test with contamination."""
    np.random.seed(42)
    
    # Create training data
    train = pd.DataFrame({
        "feature1": np.random.randn(100),
        "feature2": np.random.randn(100),
        "target": np.random.choice([0, 1], 100),
    })
    
    # Create test with some duplicates from train
    test = pd.concat([
        train.iloc[:20].copy(),  # Duplicates
        pd.DataFrame({
            "feature1": np.random.randn(80),
            "feature2": np.random.randn(80),
            "target": np.random.choice([0, 1], 80),
        }),
    ]).reset_index(drop=True)
    
    return train, test


@pytest.fixture
def leaky_features_df() -> pd.DataFrame:
    """Create DataFrame with target leakage."""
    np.random.seed(42)
    n = 500
    target = np.random.choice([0, 1], n)
    
    return pd.DataFrame({
        "feature1": np.random.randn(n),
        "feature2": np.random.randn(n),
        "leaky_feature": target + np.random.randn(n) * 0.01,  # Highly correlated
        "target": target,
    })


@pytest.fixture
def schema_definition() -> dict:
    """Sample schema definition."""
    return {
        "columns": [
            {"name": "id", "dtype": "int", "nullable": False, "unique": True},
            {"name": "name", "dtype": "string", "nullable": False},
            {"name": "value", "dtype": "float", "nullable": True},
            {"name": "category", "dtype": "string", "allowed_values": ["A", "B", "C"]},
        ],
        "allow_extra_columns": True,
    }

import pandas as pd
from typing import Any, Dict, List

def analyze_missing_values(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Analyze missing value patterns across the dataset.
    Returns data for a heatmap or matrix visualization.
    """
    # For now, let's return the percentage of missing values per column
    # and a simplified 'sparsity' matrix (e.g., sample rows to show patterns)
    
    missing_stats = []
    for col in df.columns:
        missing_count = int(df[col].isnull().sum())
        missing_pct = float(missing_count / len(df)) if len(df) > 0 else 0.0
        
        missing_stats.append({
            "column": col,
            "count": missing_count,
            "pct": missing_pct
        })
    
    return missing_stats

def get_missing_matrix_sample(df: pd.DataFrame, sample_size: int = 250) -> List[Dict[str, Any]]:
    """
    Get a sample of the missingness matrix (True if missing, False if not).
    Used for the missingno-style matrix visualization.
    """
    if len(df) > sample_size:
        sample_df = df.sample(sample_size).sort_index()
    else:
        sample_df = df
        
    # Convert to a format easy for the frontend (e.g., a list of row indices and missing col masks)
    # Actually, let's just return a list of dicts where each dict is a column's binary missing mask
    matrix_data = []
    for col in sample_df.columns:
        matrix_data.append({
            "column": col,
            "mask": sample_df[col].isnull().tolist()
        })
        
    return matrix_data

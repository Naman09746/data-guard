import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional

def analyze_numeric_column(series: pd.Series) -> Dict[str, Any]:
    """
    Perform deep numeric analysis on a pandas series.
    """
    # Drop NaNs for statistical calculations
    data = series.dropna()
    
    if data.empty:
        return {}

    stats = {
        "mean": float(data.mean()),
        "std": float(data.std()) if len(data) > 1 else 0.0,
        "min": float(data.min()),
        "max": float(data.max()),
        "p25": float(data.quantile(0.25)),
        "p50": float(data.quantile(0.50)),
        "p75": float(data.quantile(0.75)),
        "skewness": float(data.skew()) if len(data) > 2 else 0.0,
        "kurtosis": float(data.kurt()) if len(data) > 3 else 0.0,
    }

    # Outlier detection using IQR
    q1 = stats["p25"]
    q3 = stats["p75"]
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    outliers = data[(data < lower_bound) | (data > upper_bound)]
    stats["outliers_count"] = int(len(outliers))

    # Histogram calculation
    counts, bin_edges = np.histogram(data, bins=20)
    stats["histogram"] = {
        "counts": counts.tolist(),
        "bin_edges": bin_edges.tolist()
    }

    return stats

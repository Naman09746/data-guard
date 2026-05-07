import pandas as pd
from typing import Any, Dict, List

def analyze_categorical_column(series: pd.Series, top_n: int = 10) -> Dict[str, Any]:
    """
    Perform deep categorical analysis on a pandas series.
    """
    data = series.dropna()
    
    if data.empty:
        return {}

    # Value counts
    vc = data.value_counts()
    top_values = vc.head(top_n).to_dict()
    
    # Format for schema: [{"value": x, "count": y}]
    formatted_top_values = [
        {"value": str(k), "count": int(v)} 
        for k, v in top_values.items()
    ]

    # Imbalance ratio (ratio of the most frequent to the least frequent among top values)
    if len(vc) > 1:
        imbalance_ratio = float(vc.max() / vc.min())
    else:
        imbalance_ratio = 1.0

    stats = {
        "top_values": formatted_top_values,
        "imbalance_ratio": imbalance_ratio,
    }

    # Histogram/Bar chart data (already in top_values basically)
    stats["histogram"] = {
        "counts": [int(v) for v in vc.head(20).values],
        "bin_edges": [str(k) for k in vc.head(20).index]
    }

    return stats

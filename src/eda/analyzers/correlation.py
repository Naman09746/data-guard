import pandas as pd
from typing import Dict, List, Tuple

def calculate_correlation_matrix(df: pd.DataFrame) -> Tuple[List[List[float]], List[str]]:
    """
    Calculate the correlation matrix for numeric columns in a dataframe.
    Returns (matrix, labels).
    """
    numeric_df = df.select_dtypes(include=['number'])
    
    if numeric_df.empty:
        return [], []

    corr_matrix = numeric_df.corr(method='pearson')
    
    # Fill NaNs with 0 (e.g., constant columns)
    corr_matrix = corr_matrix.fillna(0)
    
    labels = corr_matrix.columns.tolist()
    matrix = corr_matrix.values.tolist()
    
    return matrix, labels

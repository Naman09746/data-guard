import uuid
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional, Dict, Any

from src.eda.schemas import EDAReport, ColumnProfile
from src.eda.analyzers.numeric import analyze_numeric_column
from src.eda.analyzers.categorical import analyze_categorical_column
from src.eda.analyzers.correlation import calculate_correlation_matrix
from src.eda.analyzers.missing import analyze_missing_values, get_missing_matrix_sample

class EDAProfiler:
    def __init__(self):
        pass

    def run_full_profile(self, df: pd.DataFrame, dataset_name: str, target_column: Optional[str] = None) -> EDAReport:
        """
        Run a complete EDA profile on the given dataframe.
        """
        scan_id = str(uuid.uuid4())
        
        # Basic dataset stats
        shape = df.shape
        memory_mb = float(df.memory_usage(deep=True).sum() / (1024 * 1024))
        duplicate_rows = int(df.duplicated().sum())
        duplicate_pct = float(duplicate_rows / len(df)) if len(df) > 0 else 0.0
        
        # Column-by-column profiling
        column_profiles = []
        for col in df.columns:
            series = df[col]
            dtype = str(series.dtype)
            
            # Common stats
            profile_data = {
                "name": str(col),
                "type": dtype,
                "count": int(series.count()),
                "missing_count": int(series.isnull().sum()),
                "missing_pct": float(series.isnull().sum() / len(df)) if len(df) > 0 else 0.0,
                "unique_count": int(series.nunique()),
                "unique_pct": float(series.nunique() / len(df)) if len(df) > 0 else 0.0,
            }
            
            # Type-specific stats
            if pd.api.types.is_numeric_dtype(series):
                num_stats = analyze_numeric_column(series)
                profile_data.update(num_stats)
            else:
                cat_stats = analyze_categorical_column(series)
                profile_data.update(cat_stats)
                
            column_profiles.append(ColumnProfile(**profile_data))
            
        # Correlation matrix
        corr_matrix, labels = calculate_correlation_matrix(df)
        
        # Missing heatmap data
        missing_heatmap = get_missing_matrix_sample(df)
        
        # Target column analysis (class balance)
        class_balance = None
        if target_column and target_column in df.columns:
            vc = df[target_column].value_counts()
            class_balance = {
                "counts": vc.to_dict(),
                "percentages": (vc / len(df)).to_dict()
            }
            
        # Simple health score calculation (placeholder)
        health_score = self._calculate_health_score(df, duplicate_pct, column_profiles)
        
        # Top risks and recommendations (placeholders)
        top_risks, recommendations = self._generate_basic_insights(df, column_profiles, duplicate_pct)
        
        return EDAReport(
            scan_id=scan_id,
            dataset_name=dataset_name,
            shape=shape,
            memory_mb=memory_mb,
            duplicate_rows=duplicate_rows,
            duplicate_pct=duplicate_pct,
            overall_health_score=health_score,
            column_profiles=column_profiles,
            correlation_matrix=corr_matrix,
            column_labels=labels,
            missing_heatmap=missing_heatmap,
            class_balance=class_balance,
            top_risks=top_risks,
            recommendations=recommendations
        )

    def _calculate_health_score(self, df: pd.DataFrame, duplicate_pct: float, profiles: list) -> float:
        score = 100.0
        
        # Deduct for duplicates
        score -= (duplicate_pct * 100)
        
        # Deduct for missing values
        avg_missing = sum(p.missing_pct for p in profiles) / len(profiles) if profiles else 0
        score -= (avg_missing * 50)
        
        return max(0.0, min(100.0, score))

    def _generate_basic_insights(self, df: pd.DataFrame, profiles: list, duplicate_pct: float):
        risks = []
        recs = []
        
        if duplicate_pct > 0.05:
            risks.append(f"High duplication detected: {duplicate_pct:.1%}")
            recs.append("Consider dropping exact duplicates.")
            
        for p in profiles:
            if p.missing_pct > 0.3:
                risks.append(f"Column '{p.name}' has over 30% missing values.")
                recs.append(f"Evaluate if '{p.name}' can be imputed or if it should be dropped.")
                
            if p.type == 'object' and p.unique_pct > 0.95 and p.unique_count > 50:
                 risks.append(f"Column '{p.name}' appears to be a unique ID/hash (95%+ unique).")
        
        return risks, recs

import pandas as pd
import numpy as np
from typing import List, Dict, Any
from src.leakage_detection.risk_scoring_model import LeakageRiskScoringModel

class CorrelationNetworkExtractor:
    """
    Generates a network representation of feature correlations.
    Nodes = Features
    Edges = Correlation Strength (if above threshold)
    """
    
    def __init__(self, correlation_threshold: float = 0.3):
        self.correlation_threshold = correlation_threshold
        self.risk_model = LeakageRiskScoringModel()

    def extract_network(self, df: pd.DataFrame, target_column: str) -> Dict[str, Any]:
        """
        Extracts nodes and edges for a force-directed graph.
        """
        # 1. Calculate Correlation Matrix
        numeric_df = df.select_dtypes(include=[np.number])
        if target_column not in numeric_df.columns:
            # Try to factorize target if it's categorical
            numeric_df[target_column] = pd.factorize(df[target_column])[0]
            
        corr_matrix = numeric_df.corr().fillna(0)
        
        # 2. Get Risk Scores for Node Coloring
        risk_result = self.risk_model.predict_risk(df, target_column)
        risk_map = {s.feature_name: s.risk_score for s in risk_result.feature_scores}
        
        nodes = []
        edges = []
        
        # Create Nodes
        for col in corr_matrix.columns:
            is_target = col == target_column
            risk_score = risk_map.get(col, 0.0)
            
            nodes.append({
                "id": col,
                "name": col,
                "val": 10 if is_target else 5, # Node size
                "group": "target" if is_target else ("high_risk" if risk_score > 0.6 else "normal"),
                "risk_score": risk_score,
                "is_target": is_target
            })

        # Create Edges
        cols = corr_matrix.columns
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                corr = abs(corr_matrix.iloc[i, j])
                if corr >= self.correlation_threshold:
                    edges.append({
                        "source": cols[i],
                        "target": cols[j],
                        "value": corr, # Edge weight
                        "opacity": corr # For styling
                    })

        return {
            "nodes": nodes,
            "links": edges,
            "overall_risk": risk_result.overall_risk.value
        }

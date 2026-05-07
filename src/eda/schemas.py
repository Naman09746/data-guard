from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field

class ColumnProfile(BaseModel):
    name: str
    type: str
    count: int
    missing_count: int
    missing_pct: float
    unique_count: int
    unique_pct: float
    
    # Numeric stats
    mean: Optional[float] = None
    std: Optional[float] = None
    min: Optional[float] = None
    max: Optional[float] = None
    p25: Optional[float] = None
    p50: Optional[float] = None
    p75: Optional[float] = None
    skewness: Optional[float] = None
    kurtosis: Optional[float] = None
    outliers_count: Optional[int] = None
    
    # Categorical stats
    top_values: Optional[List[Dict[str, Any]]] = None  # [{"value": x, "count": y}]
    imbalance_ratio: Optional[float] = None
    
    # Common
    histogram: Optional[Dict[str, List[Any]]] = None # {"bin_edges": [], "counts": []}

class EDAReport(BaseModel):
    scan_id: Optional[str] = None
    dataset_name: str
    shape: Tuple[int, int]
    memory_mb: float
    duplicate_rows: int
    duplicate_pct: float
    overall_health_score: float = Field(..., ge=0, le=100)
    
    column_profiles: List[ColumnProfile]
    correlation_matrix: List[List[float]]
    column_labels: List[str]
    missing_heatmap: List[Dict[str, Any]]
    class_balance: Optional[Dict[str, Any]] = None
    
    top_risks: List[str] = []
    recommendations: List[str] = []
    insight: Optional[Dict[str, Any]] = None
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

class EDAScanRequest(BaseModel):
    dataset_name: str
    target_column: Optional[str] = None

"""
Automated Data Quality & Leakage Detection System

A production-grade system for validating data quality and detecting
data leakage in machine learning pipelines.
"""

__version__ = "1.0.0"
__author__ = "Data Quality Team"

from src.core.config import get_settings
from src.data_quality.quality_engine import DataQualityEngine
from src.leakage_detection.leakage_engine import LeakageDetectionEngine

__all__ = [
    "DataQualityEngine",
    "LeakageDetectionEngine",
    "__version__",
    "get_settings",
]

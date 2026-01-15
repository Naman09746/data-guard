"""
Leakage Detection Engine - Main orchestrator for leakage detection.

Coordinates all detectors and generates comprehensive leakage reports.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from time import perf_counter
from typing import Any

import pandas as pd

from src.core.config import get_settings
from src.core.logging_config import get_logger, log_execution_time
from src.leakage_detection.detectors.base import DetectionResult, DetectionStatus
from src.leakage_detection.detectors.feature_leakage_detector import FeatureLeakageDetector
from src.leakage_detection.detectors.target_leakage_detector import TargetLeakageDetector
from src.leakage_detection.detectors.temporal_leakage_detector import TemporalLeakageDetector
from src.leakage_detection.detectors.train_test_detector import TrainTestDetector
from src.leakage_detection.leakage_report import LeakageReport, LeakageStatus

logger = get_logger(__name__)


@dataclass
class LeakageCheckConfig:
    """Configuration for leakage checks."""

    run_train_test_detection: bool = True
    run_target_leakage_detection: bool = True
    run_feature_leakage_detection: bool = True
    run_temporal_leakage_detection: bool = True
    fail_fast: bool = False
    custom_detectors: list[Any] = field(default_factory=list)


class LeakageDetectionEngine:
    """
    Main orchestrator for leakage detection.
    
    Coordinates multiple detectors to perform comprehensive
    leakage checks and generates unified reports.
    """

    def __init__(
        self,
        config: LeakageCheckConfig | dict[str, Any] | None = None,
    ) -> None:
        self.settings = get_settings()

        if isinstance(config, dict):
            self.config = LeakageCheckConfig(**config)
        else:
            self.config = config or LeakageCheckConfig()

        # Initialize detectors
        self.train_test_detector = TrainTestDetector()
        self.target_leakage_detector = TargetLeakageDetector()
        self.feature_leakage_detector = FeatureLeakageDetector()
        self.temporal_leakage_detector = TemporalLeakageDetector()

        self._logger = get_logger("leakage_engine")

    @log_execution_time()
    def detect(
        self,
        train_data: pd.DataFrame,
        test_data: pd.DataFrame | None = None,
        target_column: str | None = None,
        time_column: str | None = None,
        **kwargs: Any,
    ) -> LeakageReport:
        """
        Run all configured leakage checks.
        
        Args:
            train_data: Training data.
            test_data: Test data (optional).
            target_column: Target column name.
            time_column: Time column for temporal checks.
            **kwargs: Additional parameters.
        
        Returns:
            LeakageReport with comprehensive results.
        """
        start_time = perf_counter()
        results: list[DetectionResult] = []

        self._logger.info(
            "starting_leakage_detection",
            train_rows=len(train_data),
            test_rows=len(test_data) if test_data is not None else 0,
            target=target_column,
        )

        try:
            # Train-test contamination
            if self.config.run_train_test_detection:
                result = self.train_test_detector.detect(
                    train_data, test_data, target_column, **kwargs
                )
                results.append(result)
                if self.config.fail_fast and result.has_leakage:
                    return self._create_report(results, start_time, train_data, test_data)

            # Target leakage
            if self.config.run_target_leakage_detection:
                result = self.target_leakage_detector.detect(
                    train_data, test_data, target_column, **kwargs
                )
                results.append(result)
                if self.config.fail_fast and result.has_leakage:
                    return self._create_report(results, start_time, train_data, test_data)

            # Feature leakage
            if self.config.run_feature_leakage_detection:
                result = self.feature_leakage_detector.detect(
                    train_data, test_data, target_column, **kwargs
                )
                results.append(result)
                if self.config.fail_fast and result.has_leakage:
                    return self._create_report(results, start_time, train_data, test_data)

            # Temporal leakage
            if self.config.run_temporal_leakage_detection:
                result = self.temporal_leakage_detector.detect(
                    train_data, test_data, target_column, time_column=time_column, **kwargs
                )
                results.append(result)

            # Custom detectors
            for detector in self.config.custom_detectors:
                result = detector.detect(
                    train_data, test_data, target_column, **kwargs
                )
                results.append(result)
                if self.config.fail_fast and result.has_leakage:
                    break

        except Exception as e:
            self._logger.error("leakage_detection_error", error=str(e))
            results.append(DetectionResult(
                detector_name="LeakageEngine",
                status=DetectionStatus.ERROR,
                issues=[],
            ))

        return self._create_report(results, start_time, train_data, test_data)

    def _create_report(
        self,
        results: list[DetectionResult],
        start_time: float,
        train_data: pd.DataFrame,
        test_data: pd.DataFrame | None,
    ) -> LeakageReport:
        """Create leakage report from detection results."""
        duration = perf_counter() - start_time

        # Determine overall status
        if any(r.status == DetectionStatus.DETECTED for r in results):
            status = LeakageStatus.LEAKAGE_DETECTED
        elif any(r.status == DetectionStatus.ERROR for r in results):
            status = LeakageStatus.ERROR
        elif all(r.status == DetectionStatus.SKIPPED for r in results):
            status = LeakageStatus.SKIPPED
        else:
            status = LeakageStatus.CLEAN

        # Calculate metrics
        total_issues = sum(len(r.issues) for r in results)

        report = LeakageReport(
            status=status,
            detection_results=results,
            total_issues=total_issues,
            duration_seconds=duration,
            timestamp=datetime.now(UTC),
            train_shape=(len(train_data), len(train_data.columns)),
            test_shape=(len(test_data), len(test_data.columns)) if test_data is not None else None,
        )

        self._logger.info(
            "leakage_detection_complete",
            status=status.value,
            total_issues=total_issues,
            duration=round(duration, 4),
        )

        return report

    def add_custom_detector(self, detector: Any) -> None:
        """Add a custom detector to the engine."""
        self.config.custom_detectors.append(detector)

    def quick_check(
        self,
        train_data: pd.DataFrame,
        test_data: pd.DataFrame | None = None,
        target_column: str | None = None,
    ) -> bool:
        """
        Quick check for any leakage.
        
        Returns:
            True if leakage is detected, False otherwise.
        """
        report = self.detect(train_data, test_data, target_column)
        return report.has_leakage

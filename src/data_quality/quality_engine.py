"""
Data Quality Engine - Main orchestrator for quality validation.

Coordinates all validators and generates comprehensive quality reports.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from time import perf_counter
from typing import Any

import pandas as pd

from src.core.config import get_settings
from src.core.logging_config import get_logger, log_execution_time
from src.data_quality.quality_report import QualityReport, QualityStatus
from src.data_quality.validators.accuracy_validator import AccuracyValidator
from src.data_quality.validators.base import ValidationResult, ValidationStatus
from src.data_quality.validators.completeness_checker import CompletenessChecker
from src.data_quality.validators.consistency_analyzer import ConsistencyAnalyzer
from src.data_quality.validators.schema_validator import DataFrameSchema, SchemaValidator
from src.data_quality.validators.timeliness_monitor import TimelinessMonitor

logger = get_logger(__name__)


@dataclass
class QualityCheckConfig:
    """Configuration for quality checks."""

    run_schema_validation: bool = True
    run_completeness_check: bool = True
    run_consistency_check: bool = True
    run_accuracy_check: bool = True
    run_timeliness_check: bool = True
    fail_fast: bool = False
    schema: DataFrameSchema | dict[str, Any] | None = None
    custom_validators: list[Any] = field(default_factory=list)


class DataQualityEngine:
    """
    Main orchestrator for data quality validation.
    
    Coordinates multiple validators to perform comprehensive
    data quality checks and generates unified reports.
    """

    def __init__(
        self,
        config: QualityCheckConfig | dict[str, Any] | None = None,
    ) -> None:
        self.settings = get_settings()

        if isinstance(config, dict):
            self.config = QualityCheckConfig(**config)
        else:
            self.config = config or QualityCheckConfig()

        # Initialize validators
        self.schema_validator = SchemaValidator(schema=self.config.schema)
        self.completeness_checker = CompletenessChecker()
        self.consistency_analyzer = ConsistencyAnalyzer()
        self.accuracy_validator = AccuracyValidator()
        self.timeliness_monitor = TimelinessMonitor()

        self._logger = get_logger("quality_engine")

    @log_execution_time()
    def validate(
        self,
        data: pd.DataFrame,
        schema: DataFrameSchema | dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> QualityReport:
        """
        Run all configured quality checks on the data.
        
        Args:
            data: DataFrame to validate.
            schema: Optional schema override.
            **kwargs: Additional parameters passed to validators.
        
        Returns:
            QualityReport with comprehensive results.
        """
        start_time = perf_counter()
        results: list[ValidationResult] = []

        self._logger.info(
            "starting_quality_validation",
            rows=len(data),
            columns=len(data.columns),
        )

        try:
            # Schema validation
            if self.config.run_schema_validation:
                result = self.schema_validator.validate(
                    data, schema=schema or self.config.schema
                )
                results.append(result)
                if self.config.fail_fast and not result.passed:
                    return self._create_report(results, start_time, data)

            # Completeness check
            if self.config.run_completeness_check:
                result = self.completeness_checker.validate(data, **kwargs)
                results.append(result)
                if self.config.fail_fast and not result.passed:
                    return self._create_report(results, start_time, data)

            # Consistency check
            if self.config.run_consistency_check:
                result = self.consistency_analyzer.validate(data, **kwargs)
                results.append(result)
                if self.config.fail_fast and not result.passed:
                    return self._create_report(results, start_time, data)

            # Accuracy check
            if self.config.run_accuracy_check:
                result = self.accuracy_validator.validate(data, **kwargs)
                results.append(result)
                if self.config.fail_fast and not result.passed:
                    return self._create_report(results, start_time, data)

            # Timeliness check
            if self.config.run_timeliness_check:
                result = self.timeliness_monitor.validate(data, **kwargs)
                results.append(result)

            # Custom validators
            for validator in self.config.custom_validators:
                result = validator.validate(data, **kwargs)
                results.append(result)
                if self.config.fail_fast and not result.passed:
                    break

        except Exception as e:
            self._logger.error("quality_validation_error", error=str(e))
            results.append(ValidationResult(
                validator_name="QualityEngine",
                status=ValidationStatus.FAILED,
                issues=[],
            ))

        return self._create_report(results, start_time, data)

    def _create_report(
        self,
        results: list[ValidationResult],
        start_time: float,
        data: pd.DataFrame,
    ) -> QualityReport:
        """Create quality report from validation results."""
        duration = perf_counter() - start_time

        # Determine overall status
        if any(r.status == ValidationStatus.FAILED for r in results):
            status = QualityStatus.FAILED
        elif any(r.status == ValidationStatus.WARNING for r in results):
            status = QualityStatus.WARNING
        elif all(r.status == ValidationStatus.SKIPPED for r in results):
            status = QualityStatus.SKIPPED
        else:
            status = QualityStatus.PASSED

        # Calculate metrics
        total_issues = sum(len(r.issues) for r in results)

        report = QualityReport(
            status=status,
            validation_results=results,
            total_issues=total_issues,
            duration_seconds=duration,
            timestamp=datetime.now(UTC),
            data_shape=(len(data), len(data.columns)),
        )

        self._logger.info(
            "quality_validation_complete",
            status=status.value,
            total_issues=total_issues,
            duration=round(duration, 4),
        )

        return report

    def add_custom_validator(self, validator: Any) -> None:
        """Add a custom validator to the engine."""
        self.config.custom_validators.append(validator)

    def get_quality_score(self, data: pd.DataFrame) -> float:
        """
        Calculate an overall quality score for the data.
        
        Returns:
            Score between 0.0 and 1.0.
        """
        report = self.validate(data)

        if report.status == QualityStatus.FAILED:
            return 0.0
        elif report.status == QualityStatus.WARNING:
            # Deduct based on number of warnings
            deduction = min(0.5, report.total_issues * 0.05)
            return 0.5 + (0.5 - deduction)
        else:
            return 1.0

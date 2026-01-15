"""
Accuracy validator for data quality checks.

Validates data accuracy through range checks, pattern matching,
domain validation, and statistical outlier detection.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy import stats

from src.core.config import get_settings
from src.core.logging_config import get_logger
from src.data_quality.validators.base import (
    BaseValidator,
    ValidationIssue,
    ValidationResult,
    ValidationSeverity,
)

logger = get_logger(__name__)


@dataclass
class AccuracyRule:
    """Definition of an accuracy validation rule."""

    column: str
    rule_type: Literal["range", "pattern", "domain", "custom", "outlier", "format"]
    parameters: dict[str, Any] = field(default_factory=dict)
    severity: ValidationSeverity = ValidationSeverity.ERROR
    description: str | None = None


@dataclass
class AccuracyConfig:
    """Configuration for accuracy validation."""

    outlier_method: Literal["zscore", "iqr", "mad"] = "zscore"
    zscore_threshold: float = 3.0
    iqr_multiplier: float = 1.5
    mad_threshold: float = 3.5
    case_sensitive: bool = False
    float_tolerance: float = 1e-9
    max_violations_to_report: int = 1000


class AccuracyValidator(BaseValidator[pd.DataFrame]):
    """Validates data accuracy through various validation rules."""

    FORMAT_PATTERNS: dict[str, str] = {
        "email": r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$",
        "phone_us": r"^\+?1?[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}$",
        "url": r"^https?://[^\s/$.?#].[^\s]*$",
        "uuid": r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
        "iso_date": r"^\d{4}-\d{2}-\d{2}$",
    }

    def __init__(
        self,
        rules: list[AccuracyRule | dict[str, Any]] | None = None,
        config: AccuracyConfig | dict[str, Any] | None = None,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name or "AccuracyValidator")

        self.rules: list[AccuracyRule] = []
        if rules:
            for rule in rules:
                if isinstance(rule, dict):
                    self.rules.append(AccuracyRule(**rule))
                else:
                    self.rules.append(rule)

        if isinstance(config, dict):
            self.config = AccuracyConfig(**config)
        elif config is None:
            settings = get_settings()
            self.config = AccuracyConfig(
                outlier_method=settings.quality.outlier_detection_method,  # type: ignore
                zscore_threshold=settings.quality.outlier_zscore_threshold,
                iqr_multiplier=settings.quality.outlier_iqr_multiplier,
            )
        else:
            self.config = config

    def validate(
        self,
        data: pd.DataFrame,
        rules: list[AccuracyRule | dict[str, Any]] | None = None,
        check_outliers: bool = True,
        **kwargs: Any,
    ) -> ValidationResult:
        start_time = perf_counter()
        issues: list[ValidationIssue] = []
        metrics: dict[str, Any] = {}

        try:
            empty_issue = self._check_empty_data(data)
            if empty_issue and empty_issue.severity == ValidationSeverity.CRITICAL:
                return self._create_result([empty_issue], duration=perf_counter() - start_time)

            effective_rules = []
            if rules:
                for rule in rules:
                    if isinstance(rule, dict):
                        effective_rules.append(AccuracyRule(**rule))
                    else:
                        effective_rules.append(rule)
            else:
                effective_rules = self.rules.copy()

            for rule in effective_rules:
                rule_issues = self._apply_rule(data, rule)
                issues.extend(rule_issues)

            if check_outliers:
                for col in data.select_dtypes(include=[np.number]).columns:
                    outlier_issues, _ = self._detect_outliers(data[col], col)
                    issues.extend(outlier_issues)

            metrics = {"total_rules": len(effective_rules), "issues_found": len(issues)}
            return self._create_result(issues, metrics, perf_counter() - start_time)

        except Exception as e:
            return self._handle_exception(e, "accuracy_check")

    def _apply_rule(self, data: pd.DataFrame, rule: AccuracyRule) -> list[ValidationIssue]:
        if rule.column not in data.columns:
            return [ValidationIssue(
                message=f"Column '{rule.column}' not found",
                severity=ValidationSeverity.WARNING,
                column=rule.column,
            )]

        column = data[rule.column]

        if rule.rule_type == "range":
            return self._validate_range(column, rule)
        elif rule.rule_type == "pattern":
            return self._validate_pattern(column, rule)
        elif rule.rule_type == "domain":
            return self._validate_domain(column, rule)
        return []

    def _validate_range(self, column: pd.Series, rule: AccuracyRule) -> list[ValidationIssue]:
        issues = []
        min_val = rule.parameters.get("min")
        max_val = rule.parameters.get("max")
        non_null = column.dropna()

        if min_val is not None:
            violations = non_null[non_null < min_val]
            if len(violations) > 0:
                issues.append(ValidationIssue(
                    message=f"Column '{rule.column}' has {len(violations)} values below {min_val}",
                    severity=rule.severity,
                    column=rule.column,
                    row_indices=violations.index.tolist()[:100],
                ))

        if max_val is not None:
            violations = non_null[non_null > max_val]
            if len(violations) > 0:
                issues.append(ValidationIssue(
                    message=f"Column '{rule.column}' has {len(violations)} values above {max_val}",
                    severity=rule.severity,
                    column=rule.column,
                    row_indices=violations.index.tolist()[:100],
                ))

        return issues

    def _validate_pattern(self, column: pd.Series, rule: AccuracyRule) -> list[ValidationIssue]:
        pattern = rule.parameters.get("pattern", "")
        if not pattern:
            return []

        try:
            regex = re.compile(pattern, 0 if self.config.case_sensitive else re.IGNORECASE)
            non_null = column.dropna().astype(str)
            invalid = non_null[~non_null.str.match(regex, na=False)]

            if len(invalid) > 0:
                return [ValidationIssue(
                    message=f"Column '{rule.column}' has {len(invalid)} values not matching pattern",
                    severity=rule.severity,
                    column=rule.column,
                    row_indices=invalid.index.tolist()[:100],
                )]
        except re.error:
            pass
        return []

    def _validate_domain(self, column: pd.Series, rule: AccuracyRule) -> list[ValidationIssue]:
        allowed = rule.parameters.get("values", [])
        if not allowed:
            return []

        non_null = column.dropna()
        invalid = non_null[~non_null.isin(allowed)]

        if len(invalid) > 0:
            return [ValidationIssue(
                message=f"Column '{rule.column}' has {len(invalid)} values not in allowed domain",
                severity=rule.severity,
                column=rule.column,
                row_indices=invalid.index.tolist()[:100],
            )]
        return []

    def _detect_outliers(self, column: pd.Series, name: str) -> tuple[list[ValidationIssue], dict]:
        if not pd.api.types.is_numeric_dtype(column):
            return [], {}

        non_null = column.dropna()
        if len(non_null) < 3:
            return [], {}

        if self.config.outlier_method == "zscore":
            z_scores = np.abs(stats.zscore(non_null, nan_policy="omit"))
            outlier_mask = z_scores > self.config.zscore_threshold
        else:  # iqr
            q1, q3 = non_null.quantile(0.25), non_null.quantile(0.75)
            iqr = q3 - q1
            outlier_mask = (non_null < q1 - 1.5 * iqr) | (non_null > q3 + 1.5 * iqr)

        count = outlier_mask.sum()
        if count > 0:
            return [ValidationIssue(
                message=f"Column '{name}' has {count} statistical outliers",
                severity=ValidationSeverity.WARNING,
                column=name,
                row_indices=non_null[outlier_mask].index.tolist()[:100],
            )], {"outlier_count": int(count)}
        return [], {}

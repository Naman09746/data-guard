"""
Consistency analyzer for data quality validation.

Validates cross-column relationships, referential integrity,
and business rule compliance.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Literal

import numpy as np
import pandas as pd

from src.core.logging_config import get_logger
from src.data_quality.validators.base import (
    BaseValidator,
    ValidationIssue,
    ValidationResult,
    ValidationSeverity,
)

logger = get_logger(__name__)


@dataclass
class ConsistencyRule:
    """Definition of a consistency rule."""

    name: str
    description: str
    rule_type: Literal["comparison", "expression", "custom", "referential"]
    columns: list[str]
    condition: str | Callable[[pd.DataFrame], pd.Series] | None = None
    severity: ValidationSeverity = ValidationSeverity.ERROR
    parameters: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ConsistencyRule:
        """Create ConsistencyRule from dictionary."""
        return cls(
            name=data["name"],
            description=data.get("description", ""),
            rule_type=data.get("rule_type", "expression"),
            columns=data.get("columns", []),
            condition=data.get("condition"),
            severity=ValidationSeverity(data.get("severity", "error")),
            parameters=data.get("parameters", {}),
        )


class ConsistencyAnalyzer(BaseValidator[pd.DataFrame]):
    """
    Validates data consistency across columns and rows.
    
    Features:
    - Cross-column comparisons (e.g., start_date <= end_date)
    - Expression-based rules (e.g., total == sum of parts)
    - Referential integrity checks
    - Custom validation functions
    - Built-in common consistency rules
    
    Edge cases handled:
    - Null values in comparisons
    - Type mismatches
    - Circular dependencies
    - Partial matches in referential checks
    """

    def __init__(
        self,
        rules: list[ConsistencyRule | dict[str, Any]] | None = None,
        auto_detect: bool = True,
        name: str | None = None,
    ) -> None:
        """
        Initialize the consistency analyzer.
        
        Args:
            rules: List of consistency rules to check.
            auto_detect: Whether to auto-detect common consistency patterns.
            name: Validator name.
        """
        super().__init__(name=name or "ConsistencyAnalyzer")

        self.rules: list[ConsistencyRule] = []
        if rules:
            for rule in rules:
                if isinstance(rule, dict):
                    self.rules.append(ConsistencyRule.from_dict(rule))
                else:
                    self.rules.append(rule)

        self.auto_detect = auto_detect

    def validate(
        self,
        data: pd.DataFrame,
        rules: list[ConsistencyRule | dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> ValidationResult:
        """
        Validate data consistency.
        
        Args:
            data: DataFrame to validate.
            rules: Override rules for this validation.
            **kwargs: Additional options.
        
        Returns:
            ValidationResult with consistency validation results.
        """
        start_time = perf_counter()
        issues: list[ValidationIssue] = []
        metrics: dict[str, Any] = {}

        try:
            # Check for empty data
            empty_issue = self._check_empty_data(data)
            if empty_issue:
                if empty_issue.severity == ValidationSeverity.CRITICAL:
                    return self._create_result([empty_issue], duration=perf_counter() - start_time)
                issues.append(empty_issue)
                return self._create_result(issues, metrics, perf_counter() - start_time)

            # Determine rules to apply
            effective_rules = []
            if rules:
                for rule in rules:
                    if isinstance(rule, dict):
                        effective_rules.append(ConsistencyRule.from_dict(rule))
                    else:
                        effective_rules.append(rule)
            else:
                effective_rules = self.rules.copy()

            # Auto-detect rules if enabled
            if self.auto_detect:
                detected_rules = self._detect_consistency_rules(data)
                effective_rules.extend(detected_rules)

            # Apply each rule
            rules_passed = 0
            rules_failed = 0

            for rule in effective_rules:
                rule_issues = self._apply_rule(data, rule)
                issues.extend(rule_issues)

                if rule_issues:
                    rules_failed += 1
                else:
                    rules_passed += 1

            metrics = {
                "total_rules": len(effective_rules),
                "rules_passed": rules_passed,
                "rules_failed": rules_failed,
                "rows_checked": len(data),
                "issues_found": len(issues),
            }

            duration = perf_counter() - start_time
            self._logger.info(
                "consistency_check_complete",
                rules_passed=rules_passed,
                rules_failed=rules_failed,
                duration=round(duration, 4),
            )

            return self._create_result(issues, metrics, duration)

        except Exception as e:
            return self._handle_exception(e, "consistency_check")

    def _apply_rule(
        self,
        data: pd.DataFrame,
        rule: ConsistencyRule,
    ) -> list[ValidationIssue]:
        """Apply a single consistency rule."""
        issues = []

        # Check if required columns exist
        missing_cols = [c for c in rule.columns if c not in data.columns]
        if missing_cols:
            issues.append(ValidationIssue(
                message=f"Rule '{rule.name}' requires missing columns: {missing_cols}",
                severity=ValidationSeverity.WARNING,
                details={"rule_name": rule.name, "missing_columns": missing_cols},
            ))
            return issues

        try:
            if rule.rule_type == "comparison":
                issues.extend(self._apply_comparison_rule(data, rule))
            elif rule.rule_type == "expression":
                issues.extend(self._apply_expression_rule(data, rule))
            elif rule.rule_type == "custom":
                issues.extend(self._apply_custom_rule(data, rule))
            elif rule.rule_type == "referential":
                issues.extend(self._apply_referential_rule(data, rule))
            else:
                issues.append(ValidationIssue(
                    message=f"Unknown rule type: {rule.rule_type}",
                    severity=ValidationSeverity.WARNING,
                    details={"rule_name": rule.name},
                ))
        except Exception as e:
            issues.append(ValidationIssue(
                message=f"Error applying rule '{rule.name}': {e}",
                severity=ValidationSeverity.ERROR,
                details={
                    "rule_name": rule.name,
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
            ))

        return issues

    def _apply_comparison_rule(
        self,
        data: pd.DataFrame,
        rule: ConsistencyRule,
    ) -> list[ValidationIssue]:
        """Apply a comparison rule between two columns."""
        issues = []

        if len(rule.columns) != 2:
            return [ValidationIssue(
                message=f"Comparison rule requires exactly 2 columns, got {len(rule.columns)}",
                severity=ValidationSeverity.ERROR,
                details={"rule_name": rule.name},
            )]

        col1, col2 = rule.columns
        operator = rule.parameters.get("operator", "<=")

        # Create comparison mask based on operator
        valid_mask = ~(data[col1].isna() | data[col2].isna())

        if operator == "<=":
            violations = data.loc[valid_mask, col1] > data.loc[valid_mask, col2]
        elif operator == "<":
            violations = data.loc[valid_mask, col1] >= data.loc[valid_mask, col2]
        elif operator == ">=":
            violations = data.loc[valid_mask, col1] < data.loc[valid_mask, col2]
        elif operator == ">":
            violations = data.loc[valid_mask, col1] <= data.loc[valid_mask, col2]
        elif operator == "==":
            violations = data.loc[valid_mask, col1] != data.loc[valid_mask, col2]
        elif operator == "!=":
            violations = data.loc[valid_mask, col1] == data.loc[valid_mask, col2]
        else:
            return [ValidationIssue(
                message=f"Unknown operator: {operator}",
                severity=ValidationSeverity.ERROR,
                details={"rule_name": rule.name},
            )]

        violation_count = violations.sum()
        if violation_count > 0:
            violation_indices = data.index[valid_mask][violations].tolist()
            issues.append(ValidationIssue(
                message=f"Rule '{rule.name}' violated: {col1} {operator} {col2} "
                f"failed for {violation_count} rows",
                severity=rule.severity,
                row_indices=violation_indices,
                details={
                    "rule_name": rule.name,
                    "column1": col1,
                    "column2": col2,
                    "operator": operator,
                    "violation_count": int(violation_count),
                },
            ))

        return issues

    def _apply_expression_rule(
        self,
        data: pd.DataFrame,
        rule: ConsistencyRule,
    ) -> list[ValidationIssue]:
        """Apply an expression-based rule."""
        issues = []

        if rule.condition is None:
            return [ValidationIssue(
                message=f"Expression rule '{rule.name}' has no condition",
                severity=ValidationSeverity.ERROR,
            )]

        if isinstance(rule.condition, str):
            # Evaluate string expression
            try:
                # Create a safe evaluation context
                eval_context = {col: data[col] for col in rule.columns}
                eval_context.update({
                    "pd": pd,
                    "np": np,
                    "abs": np.abs,
                    "sum": np.sum,
                    "mean": np.mean,
                    "min": np.min,
                    "max": np.max,
                })

                result = eval(rule.condition, {"__builtins__": {}}, eval_context)

                if isinstance(result, pd.Series):
                    violations = ~result
                else:
                    violations = pd.Series([not result] * len(data), index=data.index)

            except Exception as e:
                return [ValidationIssue(
                    message=f"Error evaluating expression: {e}",
                    severity=ValidationSeverity.ERROR,
                    details={"expression": rule.condition, "error": str(e)},
                )]
        else:
            # Callable condition
            try:
                result = rule.condition(data)
                violations = ~result
            except Exception as e:
                return [ValidationIssue(
                    message=f"Error executing custom condition: {e}",
                    severity=ValidationSeverity.ERROR,
                    details={"error": str(e)},
                )]

        # Filter out NaN results
        violations = violations.fillna(False)
        violation_count = violations.sum()

        if violation_count > 0:
            violation_indices = data.index[violations].tolist()
            issues.append(ValidationIssue(
                message=f"Rule '{rule.name}' violated for {violation_count} rows",
                severity=rule.severity,
                row_indices=violation_indices,
                details={
                    "rule_name": rule.name,
                    "description": rule.description,
                    "violation_count": int(violation_count),
                },
            ))

        return issues

    def _apply_custom_rule(
        self,
        data: pd.DataFrame,
        rule: ConsistencyRule,
    ) -> list[ValidationIssue]:
        """Apply a custom validation function."""
        if not callable(rule.condition):
            return [ValidationIssue(
                message=f"Custom rule '{rule.name}' condition is not callable",
                severity=ValidationSeverity.ERROR,
            )]

        return self._apply_expression_rule(data, rule)

    def _apply_referential_rule(
        self,
        data: pd.DataFrame,
        rule: ConsistencyRule,
    ) -> list[ValidationIssue]:
        """Apply referential integrity rule."""
        issues = []

        reference_values = rule.parameters.get("reference_values")
        reference_column = rule.parameters.get("reference_column")

        if reference_values is None and reference_column is None:
            return [ValidationIssue(
                message=f"Referential rule '{rule.name}' requires reference_values or reference_column",
                severity=ValidationSeverity.ERROR,
            )]

        for col in rule.columns:
            if col not in data.columns:
                continue

            values = data[col].dropna()

            if reference_values:
                valid_values = set(reference_values)
            elif reference_column and reference_column in data.columns:
                valid_values = set(data[reference_column].dropna().unique())
            else:
                continue

            invalid = values[~values.isin(valid_values)]

            if len(invalid) > 0:
                invalid_values = invalid.unique().tolist()[:10]
                issues.append(ValidationIssue(
                    message=f"Column '{col}' has {len(invalid)} values not in reference set",
                    severity=rule.severity,
                    column=col,
                    row_indices=invalid.index.tolist(),
                    details={
                        "rule_name": rule.name,
                        "invalid_values_sample": invalid_values,
                        "invalid_count": len(invalid),
                    },
                ))

        return issues

    def _detect_consistency_rules(self, data: pd.DataFrame) -> list[ConsistencyRule]:
        """Auto-detect common consistency patterns."""
        detected_rules = []

        # Detect date range columns
        date_columns = data.select_dtypes(include=["datetime64"]).columns.tolist()

        for i, col1 in enumerate(date_columns):
            for col2 in date_columns[i + 1:]:
                # Common patterns: start/end, from/to, begin/end
                col1_lower = col1.lower()
                col2_lower = col2.lower()

                is_range_pair = (
                    ("start" in col1_lower and "end" in col2_lower) or
                    ("begin" in col1_lower and "end" in col2_lower) or
                    ("from" in col1_lower and "to" in col2_lower) or
                    ("created" in col1_lower and "updated" in col2_lower)
                )

                if is_range_pair:
                    detected_rules.append(ConsistencyRule(
                        name=f"auto_{col1}_before_{col2}",
                        description=f"Auto-detected: {col1} should be <= {col2}",
                        rule_type="comparison",
                        columns=[col1, col2],
                        parameters={"operator": "<="},
                        severity=ValidationSeverity.WARNING,
                    ))

        # Detect numeric range columns
        numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()

        for i, col1 in enumerate(numeric_columns):
            for col2 in numeric_columns[i + 1:]:
                col1_lower = col1.lower()
                col2_lower = col2.lower()

                is_range_pair = (
                    ("min" in col1_lower and "max" in col2_lower) or
                    ("low" in col1_lower and "high" in col2_lower) or
                    ("lower" in col1_lower and "upper" in col2_lower)
                )

                if is_range_pair:
                    detected_rules.append(ConsistencyRule(
                        name=f"auto_{col1}_leq_{col2}",
                        description=f"Auto-detected: {col1} should be <= {col2}",
                        rule_type="comparison",
                        columns=[col1, col2],
                        parameters={"operator": "<="},
                        severity=ValidationSeverity.WARNING,
                    ))

        return detected_rules

    def add_rule(
        self,
        name: str,
        columns: list[str],
        rule_type: str = "expression",
        condition: str | Callable | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Add a consistency rule.
        
        Args:
            name: Rule name.
            columns: Columns involved in the rule.
            rule_type: Type of rule.
            condition: Rule condition.
            **kwargs: Additional parameters.
        """
        rule = ConsistencyRule(
            name=name,
            description=kwargs.get("description", ""),
            rule_type=rule_type,  # type: ignore
            columns=columns,
            condition=condition,
            severity=ValidationSeverity(kwargs.get("severity", "error")),
            parameters=kwargs.get("parameters", {}),
        )
        self.rules.append(rule)

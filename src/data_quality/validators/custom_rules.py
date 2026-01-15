"""
Custom validation rules system.

Provides extensible, configurable validation rules with YAML support.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable
from pathlib import Path

import yaml
import pandas as pd

from src.core.logging_config import get_logger
from src.data_quality.validators.base import (
    BaseValidator,
    ValidationResult,
    ValidationIssue,
    ValidationSeverity,
)

logger = get_logger(__name__)


# ==================== Built-in Format Validators ====================

EMAIL_PATTERN = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
PHONE_PATTERNS = {
    'us': r'^\+?1?[-.\s]?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}$',
    'international': r'^\+?[1-9]\d{1,14}$',
    'basic': r'^[\d\s\-\+\(\)]{7,20}$',
}
CREDIT_CARD_PATTERN = r'^(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13}|6(?:011|5[0-9]{2})[0-9]{12})$'
DATE_PATTERNS = {
    'iso': r'^\d{4}-\d{2}-\d{2}$',
    'us': r'^\d{2}/\d{2}/\d{4}$',
    'eu': r'^\d{2}-\d{2}-\d{4}$',
}
URL_PATTERN = r'^https?://(?:www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b(?:[-a-zA-Z0-9()@:%_\+.~#?&//=]*)$'
UUID_PATTERN = r'^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[1-5][0-9a-fA-F]{3}-[89abAB][0-9a-fA-F]{3}-[0-9a-fA-F]{12}$'
IP_V4_PATTERN = r'^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$'
SSN_PATTERN = r'^\d{3}-\d{2}-\d{4}$'
ZIP_CODE_PATTERN = r'^\d{5}(?:-\d{4})?$'


@dataclass
class CustomRule:
    """Definition of a custom validation rule."""
    
    name: str
    rule_type: str
    column: str
    enabled: bool = True
    severity: str = "warning"
    parameters: dict[str, Any] = field(default_factory=dict)
    description: str | None = None
    error_message: str | None = None


class CustomRulesValidator(BaseValidator[pd.DataFrame]):
    """
    Validator for custom, user-defined rules.
    
    Supports:
    - Email, phone, credit card format validation
    - Date format validation
    - Custom regex patterns
    - Range checks with dynamic bounds
    - Conditional rules (if-then)
    - Aggregate constraints
    """

    BUILT_IN_VALIDATORS: dict[str, Callable] = {}

    def __init__(
        self,
        rules: list[CustomRule] | list[dict[str, Any]] | None = None,
        rules_file: str | Path | None = None,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name or "CustomRulesValidator")
        
        self.rules: list[CustomRule] = []
        
        if rules:
            for rule in rules:
                if isinstance(rule, dict):
                    self.rules.append(CustomRule(**rule))
                else:
                    self.rules.append(rule)
        
        if rules_file:
            self._load_rules_from_file(rules_file)
        
        # Register built-in validators
        self._register_validators()

    def _register_validators(self) -> None:
        """Register built-in format validators."""
        self.BUILT_IN_VALIDATORS = {
            'email': self._validate_email,
            'phone': self._validate_phone,
            'credit_card': self._validate_credit_card,
            'date': self._validate_date,
            'url': self._validate_url,
            'uuid': self._validate_uuid,
            'ip_address': self._validate_ip,
            'ssn': self._validate_ssn,
            'zip_code': self._validate_zip,
            'pattern': self._validate_pattern,
            'range': self._validate_range,
            'in_list': self._validate_in_list,
            'not_in_list': self._validate_not_in_list,
            'conditional': self._validate_conditional,
            'unique': self._validate_unique,
            'aggregate': self._validate_aggregate,
        }

    def _load_rules_from_file(self, file_path: str | Path) -> None:
        """Load rules from YAML file."""
        path = Path(file_path)
        if not path.exists():
            self._logger.warning("rules_file_not_found", path=str(path))
            return
        
        with open(path) as f:
            data = yaml.safe_load(f)
        
        if 'rules' in data:
            for rule_data in data['rules']:
                self.rules.append(CustomRule(**rule_data))

    def validate(self, data: pd.DataFrame, **kwargs: Any) -> ValidationResult:
        from time import perf_counter
        start_time = perf_counter()
        issues: list[ValidationIssue] = []
        
        empty_issue = self._check_empty_data(data)
        if empty_issue:
            return self._create_result([empty_issue], duration=perf_counter() - start_time)
        
        enabled_rules = [r for r in self.rules if r.enabled]
        
        for rule in enabled_rules:
            if rule.column not in data.columns:
                issues.append(ValidationIssue(
                    message=f"Column '{rule.column}' for rule '{rule.name}' not found",
                    severity=ValidationSeverity.WARNING,
                    column=rule.column,
                ))
                continue
            
            validator = self.BUILT_IN_VALIDATORS.get(rule.rule_type)
            if validator:
                rule_issues = validator(data, rule)
                issues.extend(rule_issues)
            else:
                self._logger.warning("unknown_rule_type", rule_type=rule.rule_type)
        
        return self._create_result(
            issues,
            {"rules_evaluated": len(enabled_rules)},
            perf_counter() - start_time
        )

    def _validate_email(self, data: pd.DataFrame, rule: CustomRule) -> list[ValidationIssue]:
        return self._validate_pattern_internal(data, rule, EMAIL_PATTERN, "Invalid email format")

    def _validate_phone(self, data: pd.DataFrame, rule: CustomRule) -> list[ValidationIssue]:
        fmt = rule.parameters.get('format', 'basic')
        pattern = PHONE_PATTERNS.get(fmt, PHONE_PATTERNS['basic'])
        return self._validate_pattern_internal(data, rule, pattern, "Invalid phone format")

    def _validate_credit_card(self, data: pd.DataFrame, rule: CustomRule) -> list[ValidationIssue]:
        issues = []
        col = data[rule.column].astype(str).str.replace(r'[\s\-]', '', regex=True)
        invalid = ~col.str.match(CREDIT_CARD_PATTERN) & col.notna()
        
        if invalid.sum() > 0:
            # Also validate Luhn checksum
            issues.append(ValidationIssue(
                message=f"Column '{rule.column}' has {invalid.sum()} invalid credit card numbers",
                severity=ValidationSeverity[rule.severity.upper()],
                column=rule.column,
                row_indices=data[invalid].index.tolist()[:50],
            ))
        return issues

    def _validate_date(self, data: pd.DataFrame, rule: CustomRule) -> list[ValidationIssue]:
        fmt = rule.parameters.get('format', 'iso')
        pattern = DATE_PATTERNS.get(fmt, DATE_PATTERNS['iso'])
        return self._validate_pattern_internal(data, rule, pattern, f"Invalid date format (expected {fmt})")

    def _validate_url(self, data: pd.DataFrame, rule: CustomRule) -> list[ValidationIssue]:
        return self._validate_pattern_internal(data, rule, URL_PATTERN, "Invalid URL format")

    def _validate_uuid(self, data: pd.DataFrame, rule: CustomRule) -> list[ValidationIssue]:
        return self._validate_pattern_internal(data, rule, UUID_PATTERN, "Invalid UUID format")

    def _validate_ip(self, data: pd.DataFrame, rule: CustomRule) -> list[ValidationIssue]:
        return self._validate_pattern_internal(data, rule, IP_V4_PATTERN, "Invalid IPv4 address")

    def _validate_ssn(self, data: pd.DataFrame, rule: CustomRule) -> list[ValidationIssue]:
        return self._validate_pattern_internal(data, rule, SSN_PATTERN, "Invalid SSN format")

    def _validate_zip(self, data: pd.DataFrame, rule: CustomRule) -> list[ValidationIssue]:
        return self._validate_pattern_internal(data, rule, ZIP_CODE_PATTERN, "Invalid ZIP code")

    def _validate_pattern(self, data: pd.DataFrame, rule: CustomRule) -> list[ValidationIssue]:
        pattern = rule.parameters.get('pattern')
        if not pattern:
            return [ValidationIssue(
                message=f"Rule '{rule.name}' missing 'pattern' parameter",
                severity=ValidationSeverity.ERROR,
            )]
        return self._validate_pattern_internal(data, rule, pattern, rule.error_message or "Pattern mismatch")

    def _validate_pattern_internal(
        self, data: pd.DataFrame, rule: CustomRule, pattern: str, msg: str
    ) -> list[ValidationIssue]:
        issues = []
        col = data[rule.column].astype(str)
        invalid = ~col.str.match(pattern) & data[rule.column].notna()
        
        if invalid.sum() > 0:
            issues.append(ValidationIssue(
                message=f"{msg} in column '{rule.column}': {invalid.sum()} invalid values",
                severity=ValidationSeverity[rule.severity.upper()],
                column=rule.column,
                row_indices=data[invalid].index.tolist()[:50],
            ))
        return issues

    def _validate_range(self, data: pd.DataFrame, rule: CustomRule) -> list[ValidationIssue]:
        issues = []
        col = data[rule.column]
        
        min_val = rule.parameters.get('min')
        max_val = rule.parameters.get('max')
        
        violations = pd.Series(False, index=data.index)
        if min_val is not None:
            violations |= col < min_val
        if max_val is not None:
            violations |= col > max_val
        
        violations &= col.notna()
        
        if violations.sum() > 0:
            issues.append(ValidationIssue(
                message=f"Column '{rule.column}' has {violations.sum()} values outside range [{min_val}, {max_val}]",
                severity=ValidationSeverity[rule.severity.upper()],
                column=rule.column,
                row_indices=data[violations].index.tolist()[:50],
            ))
        return issues

    def _validate_in_list(self, data: pd.DataFrame, rule: CustomRule) -> list[ValidationIssue]:
        issues = []
        allowed = rule.parameters.get('values', [])
        invalid = ~data[rule.column].isin(allowed) & data[rule.column].notna()
        
        if invalid.sum() > 0:
            issues.append(ValidationIssue(
                message=f"Column '{rule.column}' has {invalid.sum()} values not in allowed list",
                severity=ValidationSeverity[rule.severity.upper()],
                column=rule.column,
                details={"allowed_values": allowed},
            ))
        return issues

    def _validate_not_in_list(self, data: pd.DataFrame, rule: CustomRule) -> list[ValidationIssue]:
        issues = []
        forbidden = rule.parameters.get('values', [])
        invalid = data[rule.column].isin(forbidden)
        
        if invalid.sum() > 0:
            issues.append(ValidationIssue(
                message=f"Column '{rule.column}' has {invalid.sum()} forbidden values",
                severity=ValidationSeverity[rule.severity.upper()],
                column=rule.column,
            ))
        return issues

    def _validate_conditional(self, data: pd.DataFrame, rule: CustomRule) -> list[ValidationIssue]:
        """Validate conditional rules: if condition_column == condition_value, then target must be valid."""
        issues = []
        condition_col = rule.parameters.get('condition_column')
        condition_val = rule.parameters.get('condition_value')
        target_col = rule.column
        target_rule = rule.parameters.get('target_rule')
        
        if not all([condition_col, condition_val is not None, target_rule]):
            return issues
        
        mask = data[condition_col] == condition_val
        subset = data[mask]
        
        if len(subset) > 0 and target_rule:
            # Create sub-rule for target
            sub_rule = CustomRule(
                name=f"{rule.name}_conditional",
                rule_type=target_rule,
                column=target_col,
                parameters=rule.parameters.get('target_parameters', {}),
                severity=rule.severity,
            )
            validator = self.BUILT_IN_VALIDATORS.get(target_rule)
            if validator:
                sub_issues = validator(subset, sub_rule)
                for issue in sub_issues:
                    issue.message = f"[Conditional: when {condition_col}={condition_val}] {issue.message}"
                issues.extend(sub_issues)
        
        return issues

    def _validate_unique(self, data: pd.DataFrame, rule: CustomRule) -> list[ValidationIssue]:
        issues = []
        duplicates = data[rule.column].duplicated()
        
        if duplicates.sum() > 0:
            issues.append(ValidationIssue(
                message=f"Column '{rule.column}' has {duplicates.sum()} duplicate values",
                severity=ValidationSeverity[rule.severity.upper()],
                column=rule.column,
            ))
        return issues

    def _validate_aggregate(self, data: pd.DataFrame, rule: CustomRule) -> list[ValidationIssue]:
        """Validate aggregate constraints (sum, mean, etc.)."""
        issues = []
        agg_type = rule.parameters.get('aggregation', 'sum')
        expected = rule.parameters.get('expected')
        tolerance = rule.parameters.get('tolerance', 0.01)
        
        if expected is None:
            return issues
        
        actual = getattr(data[rule.column], agg_type)()
        
        if abs(actual - expected) > tolerance * abs(expected):
            issues.append(ValidationIssue(
                message=f"Column '{rule.column}' {agg_type}={actual:.2f}, expected {expected:.2f}",
                severity=ValidationSeverity[rule.severity.upper()],
                column=rule.column,
                details={"actual": actual, "expected": expected, "aggregation": agg_type},
            ))
        return issues

    def add_rule(self, rule: CustomRule | dict[str, Any]) -> None:
        """Add a new rule to the validator."""
        if isinstance(rule, dict):
            self.rules.append(CustomRule(**rule))
        else:
            self.rules.append(rule)

    def save_rules(self, file_path: str | Path) -> None:
        """Save rules to YAML file."""
        path = Path(file_path)
        data = {
            'rules': [
                {
                    'name': r.name,
                    'rule_type': r.rule_type,
                    'column': r.column,
                    'enabled': r.enabled,
                    'severity': r.severity,
                    'parameters': r.parameters,
                }
                for r in self.rules
            ]
        }
        with open(path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False)

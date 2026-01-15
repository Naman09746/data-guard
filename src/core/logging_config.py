"""
Structured logging configuration for the Data Quality & Leakage Detection System.

Provides JSON-formatted structured logging with context propagation,
correlation IDs, and performance tracking.
"""

from __future__ import annotations

import logging
import sys
import uuid
from collections.abc import Callable
from contextvars import ContextVar
from datetime import UTC, datetime
from functools import wraps
from pathlib import Path
from time import perf_counter
from typing import Any, TypeVar

import structlog
from structlog.types import EventDict, WrappedLogger

from src.core.config import get_settings

# Context variable for correlation ID
correlation_id_var: ContextVar[str] = ContextVar("correlation_id", default="")

# Type variable for decorated functions
F = TypeVar("F", bound=Callable[..., Any])


def get_correlation_id() -> str:
    """Get the current correlation ID or generate a new one."""
    cid = correlation_id_var.get()
    if not cid:
        cid = str(uuid.uuid4())[:8]
        correlation_id_var.set(cid)
    return cid


def set_correlation_id(cid: str) -> None:
    """Set the correlation ID for the current context."""
    correlation_id_var.set(cid)


def add_correlation_id(
    logger: WrappedLogger, method_name: str, event_dict: EventDict
) -> EventDict:
    """Add correlation ID to log events."""
    event_dict["correlation_id"] = get_correlation_id()
    return event_dict


def add_timestamp(
    logger: WrappedLogger, method_name: str, event_dict: EventDict
) -> EventDict:
    """Add ISO timestamp to log events."""
    event_dict["timestamp"] = datetime.now(UTC).isoformat()
    return event_dict


def add_service_info(
    logger: WrappedLogger, method_name: str, event_dict: EventDict
) -> EventDict:
    """Add service information to log events."""
    settings = get_settings()
    event_dict["service"] = settings.app_name
    event_dict["environment"] = settings.environment
    return event_dict


def filter_sensitive_data(
    logger: WrappedLogger, method_name: str, event_dict: EventDict
) -> EventDict:
    """Filter sensitive data from log events."""
    sensitive_keys = {"password", "secret", "token", "api_key", "authorization"}

    def redact(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {
                k: "[REDACTED]" if k.lower() in sensitive_keys else redact(v)
                for k, v in obj.items()
            }
        if isinstance(obj, list):
            return [redact(item) for item in obj]
        return obj

    return redact(event_dict)


def configure_logging() -> None:
    """
    Configure structured logging for the application.
    
    Sets up structlog with appropriate processors based on settings.
    """
    settings = get_settings()
    log_settings = settings.logging

    # Define shared processors
    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        add_correlation_id,
        filter_sensitive_data,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    if log_settings.include_timestamp:
        shared_processors.insert(0, add_timestamp)

    if log_settings.include_caller:
        shared_processors.append(structlog.processors.CallsiteParameterAdder(
            [
                structlog.processors.CallsiteParameter.FILENAME,
                structlog.processors.CallsiteParameter.LINENO,
                structlog.processors.CallsiteParameter.FUNC_NAME,
            ]
        ))

    # Add service info in production
    if settings.environment == "production":
        shared_processors.append(add_service_info)

    # Configure format-specific processors
    if log_settings.format == "json":
        shared_processors.append(structlog.processors.format_exc_info)
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(
            colors=True,
            exception_formatter=structlog.dev.plain_traceback,
        )

    # Configure structlog
    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # Configure standard library logging
    log_level = getattr(logging, log_settings.level)

    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(console_handler)
    root_logger.setLevel(log_level)

    # File handler if configured
    if log_settings.log_file:
        log_path = Path(log_settings.log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        from logging.handlers import RotatingFileHandler

        file_handler = RotatingFileHandler(
            log_path,
            maxBytes=log_settings.max_log_size_mb * 1024 * 1024,
            backupCount=log_settings.backup_count,
        )
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    # Reduce noise from third-party libraries
    logging.getLogger("uvicorn").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """
    Get a structured logger instance.
    
    Args:
        name: Logger name (defaults to calling module).
    
    Returns:
        Configured structlog logger instance.
    """
    return structlog.get_logger(name)


def log_execution_time(
    logger: structlog.stdlib.BoundLogger | None = None,
    level: str = "info",
) -> Callable[[F], F]:
    """
    Decorator to log function execution time.
    
    Args:
        logger: Logger instance (defaults to function's module logger).
        level: Log level for the timing message.
    
    Returns:
        Decorated function.
    """
    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            log = logger or get_logger(func.__module__)
            start_time = perf_counter()

            try:
                result = func(*args, **kwargs)
                elapsed = perf_counter() - start_time
                getattr(log, level)(
                    "function_completed",
                    function=func.__name__,
                    duration_seconds=round(elapsed, 4),
                )
                return result
            except Exception as e:
                elapsed = perf_counter() - start_time
                log.error(
                    "function_failed",
                    function=func.__name__,
                    duration_seconds=round(elapsed, 4),
                    error=str(e),
                    error_type=type(e).__name__,
                )
                raise

        return wrapper  # type: ignore

    return decorator


class LogContext:
    """
    Context manager for adding temporary context to logs.
    
    Usage:
        with LogContext(user_id="123", operation="validation"):
            logger.info("Processing")  # Will include user_id and operation
    """

    def __init__(self, **context: Any) -> None:
        """Initialize with context key-value pairs."""
        self.context = context
        self.tokens: list[Any] = []

    def __enter__(self) -> LogContext:
        """Enter context and bind variables."""
        for key, value in self.context.items():
            token = structlog.contextvars.bind_contextvars(**{key: value})
            self.tokens.append((key, token))
        return self

    def __exit__(self, *args: Any) -> None:
        """Exit context and unbind variables."""
        structlog.contextvars.unbind_contextvars(*[key for key, _ in self.tokens])


# Initialize logging on module import
configure_logging()

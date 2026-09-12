"""Unified error taxonomy and structured exceptions.

Every failure inside the harness must be mapped to one of the ErrorType values
below instead of being stored as a raw exception string.
"""

from __future__ import annotations

from enum import StrEnum


class ErrorType(StrEnum):
    # Phase 0
    CONFIG_ERROR = "CONFIG_ERROR"
    MODEL_API_ERROR = "MODEL_API_ERROR"
    MODEL_OUTPUT_PARSE_ERROR = "MODEL_OUTPUT_PARSE_ERROR"
    ENVIRONMENT_INIT_ERROR = "ENVIRONMENT_INIT_ERROR"
    ACTION_EXECUTION_ERROR = "ACTION_EXECUTION_ERROR"
    OBSERVATION_ERROR = "OBSERVATION_ERROR"
    MAX_STEPS_EXCEEDED = "MAX_STEPS_EXCEEDED"
    TASK_TERMINATED = "TASK_TERMINATED"
    TASK_TRUNCATED = "TASK_TRUNCATED"
    TRACE_WRITE_ERROR = "TRACE_WRITE_ERROR"
    UNKNOWN_ERROR = "UNKNOWN_ERROR"
    # Reserved for later phases (defined now for a stable taxonomy)
    VERIFICATION_FAILED = "VERIFICATION_FAILED"
    RECOVERY_FAILED = "RECOVERY_FAILED"
    BUDGET_EXCEEDED = "BUDGET_EXCEEDED"
    CHECKPOINT_ERROR = "CHECKPOINT_ERROR"
    POLICY_BLOCKED = "POLICY_BLOCKED"
    HUMAN_REJECTED = "HUMAN_REJECTED"


class HarnessError(Exception):
    """Base class for all structured harness errors."""

    error_type = ErrorType.UNKNOWN_ERROR

    def __init__(self, message: str, *, detail: str | None = None):
        super().__init__(message)
        self.message = message
        self.detail = detail


class ConfigError(HarnessError):
    error_type = ErrorType.CONFIG_ERROR


class ModelApiError(HarnessError):
    error_type = ErrorType.MODEL_API_ERROR


class ModelOutputParseError(HarnessError):
    error_type = ErrorType.MODEL_OUTPUT_PARSE_ERROR

    def __init__(self, message: str, *, raw_text: str | None = None):
        super().__init__(message)
        self.raw_text = raw_text


class EnvironmentInitError(HarnessError):
    error_type = ErrorType.ENVIRONMENT_INIT_ERROR


class ActionExecutionError(HarnessError):
    error_type = ErrorType.ACTION_EXECUTION_ERROR


class ObservationError(HarnessError):
    error_type = ErrorType.OBSERVATION_ERROR


class TraceWriteError(HarnessError):
    error_type = ErrorType.TRACE_WRITE_ERROR

"""
Custom Exception Classes for Terminal Chatbot
Provides specific exception types for better error handling and reporting.
"""


class ChatbotError(Exception):
    """Base exception for all chatbot errors."""

    def __init__(self, message: str, details: dict = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def __str__(self):
        if self.details:
            return f"{self.message} | Details: {self.details}"
        return self.message


class ConfigurationError(ChatbotError):
    """Raised when configuration is invalid or missing required values."""

    def __init__(self, message: str, field: str = None, value=None):
        details = {}
        if field:
            details["field"] = field
        if value is not None:
            details["value"] = value
        super().__init__(message, details)
        self.field = field
        self.value = value


class RateLimitExceededError(ChatbotError):
    """Raised when the rate limit has been exceeded."""

    def __init__(
        self,
        message: str = "Rate limit exceeded",
        requests_per_minute: int = None,
        retry_after: float = None
    ):
        details = {}
        if requests_per_minute:
            details["requests_per_minute"] = requests_per_minute
        if retry_after:
            details["retry_after_seconds"] = retry_after
        super().__init__(message, details)
        self.requests_per_minute = requests_per_minute
        self.retry_after = retry_after


class CostLimitExceededError(ChatbotError):
    """Raised when the session cost limit has been reached."""

    def __init__(
        self,
        message: str = "Cost limit exceeded",
        current_cost: float = None,
        limit: float = None
    ):
        details = {}
        if current_cost is not None:
            details["current_cost"] = f"${current_cost:.4f}"
        if limit is not None:
            details["limit"] = f"${limit:.2f}"
        super().__init__(message, details)
        self.current_cost = current_cost
        self.limit = limit


class ValidationError(ChatbotError):
    """Raised when input validation fails."""

    def __init__(
        self,
        message: str,
        field: str = None,
        value=None,
        reason: str = None
    ):
        details = {}
        if field:
            details["field"] = field
        if value is not None:
            # Truncate long values for readability
            str_value = str(value)
            details["value"] = str_value[:100] + "..." if len(str_value) > 100 else str_value
        if reason:
            details["reason"] = reason
        super().__init__(message, details)
        self.field = field
        self.value = value
        self.reason = reason


class APIError(ChatbotError):
    """Raised when an API call fails after retries."""

    def __init__(
        self,
        message: str,
        status_code: int = None,
        api_error: str = None,
        retries_attempted: int = None
    ):
        details = {}
        if status_code:
            details["status_code"] = status_code
        if api_error:
            details["api_error"] = api_error
        if retries_attempted:
            details["retries_attempted"] = retries_attempted
        super().__init__(message, details)
        self.status_code = status_code
        self.api_error = api_error
        self.retries_attempted = retries_attempted


class FileProcessingError(ChatbotError):
    """Raised when file processing fails."""

    def __init__(
        self,
        message: str,
        filepath: str = None,
        reason: str = None
    ):
        details = {}
        if filepath:
            details["filepath"] = filepath
        if reason:
            details["reason"] = reason
        super().__init__(message, details)
        self.filepath = filepath
        self.reason = reason

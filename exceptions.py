"""
Custom Exception Classes for Terminal Chatbot
Provides specific exception types for better error handling and reporting.

Includes OpenAI-compatible exception classes for unified API.
"""

from typing import List, Optional


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
    """Raised when the cost limit has been reached."""

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


# =============================================================================
# OpenAI-Compatible Exception Classes (New)
# =============================================================================


class MessageValidationError(ValidationError):
    """Raised when message format is invalid."""

    def __init__(
        self,
        message: str,
        message_index: int = None,
        field: str = None
    ):
        details = {}
        if message_index is not None:
            details["message_index"] = message_index
        if field:
            details["field"] = field
        
        # Build a more descriptive error message
        full_message = message
        if message_index is not None:
            full_message = f"Message at index {message_index}: {message}"
        
        super().__init__(
            full_message,
            field=field,
            reason="message_validation_error"
        )
        self.message_index = message_index
        self.original_message = message
        
        # Add to details for complete info
        self.details.update(details)


class ModelNotSupportedError(ValidationError):
    """Raised when model is not supported."""

    def __init__(
        self,
        model: str,
        supported_models: List[str] = None
    ):
        details = {}
        if model:
            details["requested_model"] = model
        if supported_models:
            details["supported_models"] = supported_models
        
        message = f"Model '{model}' is not supported"
        super().__init__(
            message,
            field="model",
            value=model,
            reason="model_not_supported"
        )
        self.model = model
        self.supported_models = supported_models or []
        
        # Add to details for complete info
        self.details.update(details)


class ContentTypeError(ValidationError):
    """Raised when content type is invalid."""

    def __init__(
        self,
        content_type: str,
        allowed_types: List[str] = None
    ):
        details = {}
        if content_type:
            details["content_type"] = content_type
        if allowed_types:
            details["allowed_types"] = allowed_types
        
        message = f"Invalid content type: '{content_type}'"
        if allowed_types:
            message += f". Allowed types: {', '.join(allowed_types)}"
        
        super().__init__(
            message,
            field="content.type",
            value=content_type,
            reason="invalid_content_type"
        )
        self.content_type = content_type
        self.allowed_types = allowed_types or []
        
        # Add to details for complete info
        self.details.update(details)

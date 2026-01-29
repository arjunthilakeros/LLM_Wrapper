"""
OpenAI API Client with Retry Logic
Provides a wrapper around OpenAI client with timeout and exponential backoff.
"""

from functools import wraps
from typing import Callable, TypeVar

from openai import OpenAI, APIConnectionError, RateLimitError, APITimeoutError, APIError
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log
)

from logger import get_logger
from exceptions import APIError as ChatbotAPIError

# Type variable for generic return type
T = TypeVar('T')

# Default configuration
DEFAULT_TIMEOUT = 30.0  # seconds
DEFAULT_MAX_RETRIES = 3
DEFAULT_WAIT_MIN = 1  # seconds
DEFAULT_WAIT_MAX = 10  # seconds


def create_openai_client(
    api_key: str,
    timeout: float = DEFAULT_TIMEOUT,
    max_retries: int = 0  # We handle retries ourselves with tenacity
) -> OpenAI:
    """
    Create an OpenAI client with configured timeout.

    Args:
        api_key: OpenAI API key
        timeout: Request timeout in seconds
        max_retries: Built-in retries (set to 0, we use tenacity)

    Returns:
        Configured OpenAI client
    """
    return OpenAI(
        api_key=api_key,
        timeout=timeout,
        max_retries=max_retries
    )


def get_retry_decorator(
    max_retries: int = DEFAULT_MAX_RETRIES,
    wait_min: float = DEFAULT_WAIT_MIN,
    wait_max: float = DEFAULT_WAIT_MAX
):
    """
    Create a retry decorator with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts
        wait_min: Minimum wait time between retries (seconds)
        wait_max: Maximum wait time between retries (seconds)

    Returns:
        Configured retry decorator
    """
    logger = get_logger()

    return retry(
        stop=stop_after_attempt(max_retries + 1),  # +1 because first attempt counts
        wait=wait_exponential(multiplier=1, min=wait_min, max=wait_max),
        retry=retry_if_exception_type((
            APIConnectionError,
            RateLimitError,
            APITimeoutError
        )),
        before_sleep=before_sleep_log(logger, log_level=20),  # INFO level
        reraise=True
    )


# Pre-configured retry decorator for common use
api_retry = get_retry_decorator()


def with_retry(func: Callable[..., T]) -> Callable[..., T]:
    """
    Decorator to add retry logic to API calls.

    Usage:
        @with_retry
        def my_api_call():
            return client.chat.completions.create(...)
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger = get_logger()
        try:
            return api_retry(func)(*args, **kwargs)
        except (APIConnectionError, RateLimitError, APITimeoutError) as e:
            logger.error(
                f"API call failed after retries: {type(e).__name__}: {e}",
                exc_info=True
            )
            raise ChatbotAPIError(
                message=f"API call failed: {type(e).__name__}",
                api_error=str(e),
                retries_attempted=DEFAULT_MAX_RETRIES
            ) from e
        except APIError as e:
            logger.error(
                f"API error (no retry): {type(e).__name__}: {e}",
                exc_info=True
            )
            raise ChatbotAPIError(
                message=f"API error: {type(e).__name__}",
                status_code=getattr(e, 'status_code', None),
                api_error=str(e)
            ) from e

    return wrapper


class RetryableOpenAIClient:
    """
    OpenAI client wrapper with built-in retry logic.

    All API methods automatically retry on transient errors.
    """

    def __init__(
        self,
        api_key: str,
        timeout: float = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES
    ):
        self._client = create_openai_client(api_key, timeout)
        self._max_retries = max_retries
        self._retry_decorator = get_retry_decorator(max_retries)
        self._logger = get_logger()

    @property
    def conversations(self):
        """Access conversations API with retry logic."""
        return self._client.conversations

    @property
    def chat(self):
        """Access chat API with retry logic."""
        return self._client.chat

    @property
    def responses(self):
        """Access responses API with retry logic."""
        return self._client.responses

    def _execute_with_retry(self, operation: Callable[..., T], *args, **kwargs) -> T:
        """Execute an operation with retry logic."""
        @self._retry_decorator
        def _inner():
            return operation(*args, **kwargs)

        try:
            return _inner()
        except (APIConnectionError, RateLimitError, APITimeoutError) as e:
            self._logger.error(
                f"API call failed after {self._max_retries} retries: {e}",
                exc_info=True
            )
            raise ChatbotAPIError(
                message=f"API call failed: {type(e).__name__}",
                api_error=str(e),
                retries_attempted=self._max_retries
            ) from e

    def create_conversation(self, **kwargs):
        """Create a conversation with retry logic."""
        return self._execute_with_retry(
            self._client.conversations.create,
            **kwargs
        )

    def create_chat_completion(self, **kwargs):
        """Create a chat completion with retry logic."""
        return self._execute_with_retry(
            self._client.chat.completions.create,
            **kwargs
        )

    def create_response(self, **kwargs):
        """Create a response with retry logic."""
        return self._execute_with_retry(
            self._client.responses.create,
            **kwargs
        )

    def list_conversation_items(self, **kwargs):
        """List conversation items with retry logic."""
        return self._execute_with_retry(
            self._client.conversations.items.list,
            **kwargs
        )

"""
OpenAI Responses API Client - Conversations API Integration
Provides high-level interface for OpenAI Responses + Conversations API.

Key Features:
- Conversation persistence (OpenAI stores ALL message history)
- No local message storage required
- Automatic context management by OpenAI
- Streaming support
"""

from functools import wraps
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union

from openai import (
    OpenAI,
    APIConnectionError,
    RateLimitError,
    APITimeoutError,
    APIError,
)
from openai.types.responses import (
    Response,
    ResponseOutputMessage,
    ResponseOutputText,
)
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)

from logger import get_logger
from exceptions import APIError as ChatbotAPIError

T = TypeVar("T")

DEFAULT_TIMEOUT = 60.0
DEFAULT_MAX_RETRIES = 3
DEFAULT_WAIT_MIN = 1
DEFAULT_WAIT_MAX = 10

RETRYABLE_EXCEPTIONS = (APIConnectionError, RateLimitError, APITimeoutError)


def get_retry_decorator(
    max_retries: int = DEFAULT_MAX_RETRIES,
    wait_min: float = DEFAULT_WAIT_MIN,
    wait_max: float = DEFAULT_WAIT_MAX,
):
    """Create retry decorator with exponential backoff."""
    logger = get_logger()
    return retry(
        stop=stop_after_attempt(max_retries + 1),
        wait=wait_exponential(multiplier=1, min=wait_min, max=wait_max),
        retry=retry_if_exception_type(RETRYABLE_EXCEPTIONS),
        before_sleep=before_sleep_log(logger, log_level=20),
        reraise=True,
    )


class ResponsesAPIClient:
    """
    Client for OpenAI Responses API + Conversations API.

    Key principle: OpenAI stores ALL conversation history.
    We just pass conversation_id and OpenAI includes full context automatically.

    Usage:
        client = create_responses_client(api_key)

        # Create conversation (OpenAI stores it)
        conv = client.create_conversation(metadata={"user_id": "123"})

        # Send message - OpenAI auto-includes full history
        response = client.create_response(
            model="gpt-4o",
            input="My name is Arjun",
            conversation_id=conv.id  # OpenAI handles context!
        )

        # Later messages remember everything
        response = client.create_response(
            model="gpt-4o",
            input="What's my name?",
            conversation_id=conv.id  # Remembers "Arjun"!
        )

        # Get full history from OpenAI
        items = client.list_conversation_items(conv.id)
    """

    def __init__(
        self,
        api_key: str,
        timeout: float = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
    ):
        self._client = OpenAI(
            api_key=api_key,
            timeout=timeout,
            max_retries=0,  # We handle retries with tenacity
        )
        self._max_retries = max_retries
        self._retry_decorator = get_retry_decorator(max_retries)
        self._logger = get_logger()

    def _execute_with_retry(self, operation: Callable[..., T], *args, **kwargs) -> T:
        """Execute operation with retry logic."""
        @self._retry_decorator
        def _inner():
            return operation(*args, **kwargs)

        try:
            return _inner()
        except RETRYABLE_EXCEPTIONS as e:
            self._logger.error(f"API call failed after {self._max_retries} retries: {e}")
            raise ChatbotAPIError(
                message=f"API call failed: {type(e).__name__}",
                api_error=str(e),
                retries_attempted=self._max_retries,
            ) from e
        except APIError as e:
            self._logger.error(f"API error: {e}")
            raise ChatbotAPIError(
                message=f"API error: {type(e).__name__}",
                status_code=getattr(e, "status_code", None),
                api_error=str(e),
            ) from e

    # =========================================================================
    # Conversation Management (OpenAI stores everything)
    # =========================================================================

    def create_conversation(
        self,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """
        Create a new conversation in OpenAI.

        OpenAI stores the conversation and all future messages.
        Returns conversation object with .id attribute.
        """
        self._logger.debug(f"Creating conversation with metadata: {metadata}")
        kwargs = {}
        if metadata is not None:
            kwargs["metadata"] = metadata
        return self._execute_with_retry(self._client.conversations.create, **kwargs)

    def get_conversation(self, conversation_id: str) -> Any:
        """Retrieve conversation metadata from OpenAI."""
        self._logger.debug(f"Getting conversation: {conversation_id}")
        return self._execute_with_retry(
            self._client.conversations.retrieve, conversation_id
        )

    def delete_conversation(self, conversation_id: str) -> bool:
        """Delete conversation from OpenAI."""
        self._logger.debug(f"Deleting conversation: {conversation_id}")
        self._execute_with_retry(self._client.conversations.delete, conversation_id)
        return True

    def list_conversation_items(
        self,
        conversation_id: str,
        limit: int = 100,
        order: str = "asc",
        after: Optional[str] = None,
        before: Optional[str] = None,
    ) -> Any:
        """
        Get FULL conversation history from OpenAI.

        This fetches ALL messages stored by OpenAI for this conversation.
        Like viewing chat history in ChatGPT.

        Args:
            conversation_id: The OpenAI conversation ID
            limit: Max items to return (1-100)
            order: "asc" (oldest first) or "desc" (newest first)

        Returns:
            Response with .data list of conversation items
        """
        self._logger.debug(f"Listing items for conversation {conversation_id}")
        kwargs = {
            "conversation_id": conversation_id,
            "limit": limit,
            "order": order,
        }
        if after is not None:
            kwargs["after"] = after
        if before is not None:
            kwargs["before"] = before
        return self._execute_with_retry(self._client.conversations.items.list, **kwargs)

    # =========================================================================
    # Response Creation (with automatic context from conversation)
    # =========================================================================

    def create_response(
        self,
        model: str,
        input: Union[str, List[Dict[str, Any]]],
        conversation_id: Optional[str] = None,
        instructions: Optional[str] = None,
        stream: bool = False,
        max_output_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        store: Optional[bool] = None,
        user: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Response:
        """
        Create a response using OpenAI Responses API.

        CRITICAL: When conversation_id is provided, OpenAI automatically
        includes ALL previous messages as context. You don't need to
        manually build a messages array!

        Args:
            model: Model to use (e.g., "gpt-4o")
            input: User's message (string or multimodal content)
            conversation_id: OpenAI conversation ID for context persistence
            instructions: System prompt
            stream: Enable streaming
            ...other params...

        Returns:
            Response object with .output, .usage, .id, .conversation_id
        """
        self._logger.debug(f"Creating response: model={model}, conversation={conversation_id}")

        request_kwargs = {
            "model": model,
            "input": input,
            "stream": stream,
        }

        # CRITICAL: Pass conversation ID for automatic context
        if conversation_id is not None:
            request_kwargs["conversation"] = conversation_id

        if instructions is not None:
            request_kwargs["instructions"] = instructions
        if max_output_tokens is not None:
            request_kwargs["max_output_tokens"] = max_output_tokens
        if temperature is not None:
            request_kwargs["temperature"] = temperature
        if top_p is not None:
            request_kwargs["top_p"] = top_p
        if tools is not None:
            request_kwargs["tools"] = tools
        if tool_choice is not None:
            request_kwargs["tool_choice"] = tool_choice
        if store is not None:
            request_kwargs["store"] = store
        if user is not None:
            request_kwargs["user"] = user
        if metadata is not None:
            request_kwargs["metadata"] = metadata

        request_kwargs.update(kwargs)

        return self._execute_with_retry(
            self._client.responses.create, **request_kwargs
        )

    def get_response(self, response_id: str) -> Response:
        """Retrieve a stored response by ID."""
        self._logger.debug(f"Getting response: {response_id}")
        return self._execute_with_retry(self._client.responses.retrieve, response_id)

    # =========================================================================
    # Streaming Support
    # =========================================================================

    def create_response_stream_raw(
        self,
        model: str,
        input: Union[str, List[Dict[str, Any]]],
        conversation_id: Optional[str] = None,
        instructions: Optional[str] = None,
        **kwargs,
    ) -> Any:
        """
        Create streaming response, return raw stream object.

        Use this for terminal/CLI streaming where you handle events directly.
        """
        request_kwargs = {
            "model": model,
            "input": input,
            "stream": True,
        }

        if conversation_id is not None:
            request_kwargs["conversation"] = conversation_id
        if instructions is not None:
            request_kwargs["instructions"] = instructions

        request_kwargs.update(kwargs)

        return self._execute_with_retry(
            self._client.responses.create, **request_kwargs
        )


# =============================================================================
# Title Generation
# =============================================================================

TITLE_GENERATION_PROMPT = """Generate a short, descriptive title (max 50 chars) for a conversation that starts with:
"{first_message}"

Rules:
- Max 50 characters
- No quotes or special characters
- Descriptive and specific
- Just the title, nothing else"""


def generate_title(first_message: str, client: "ResponsesAPIClient", model: str = "gpt-4o-mini") -> str:
    """
    Generate a smart title for a conversation using AI.

    Args:
        first_message: The first user message in the conversation
        client: ResponsesAPIClient instance
        model: Model to use for title generation (default: gpt-4o-mini for cost)

    Returns:
        Generated title (max 50 characters)
    """
    # Truncate message if too long
    truncated_message = first_message[:500] if len(first_message) > 500 else first_message
    prompt = TITLE_GENERATION_PROMPT.format(first_message=truncated_message)

    try:
        response = client.create_response(
            model=model,
            input=prompt,
            max_output_tokens=50,
            temperature=0.3,  # Lower temperature for consistent results
            store=False,  # Don't store this helper request
        )

        title = extract_output_text(response).strip()
        # Clean up and truncate
        title = title.strip('"\'')  # Remove quotes
        title = title[:50]  # Max 50 chars
        return title if title else "New Conversation"

    except Exception as e:
        logger = get_logger()
        logger.warning(f"Failed to generate title: {e}")
        # Fallback: use first words of message
        words = first_message.split()[:5]
        fallback = " ".join(words)
        return (fallback[:47] + "...") if len(fallback) > 50 else fallback


# =============================================================================
# Helper Functions
# =============================================================================

def extract_output_text(response: Response) -> str:
    """
    Extract text content from Response object.

    Args:
        response: Response from create_response()

    Returns:
        Concatenated text from all output messages
    """
    if not response or not response.output:
        return ""

    texts = []
    for item in response.output:
        if isinstance(item, ResponseOutputMessage):
            for content in item.content:
                if isinstance(content, ResponseOutputText):
                    texts.append(content.text)
                elif hasattr(content, "text"):
                    texts.append(content.text)
        elif hasattr(item, "text"):
            texts.append(item.text)
        elif hasattr(item, "content"):
            if isinstance(item.content, str):
                texts.append(item.content)
            elif isinstance(item.content, list):
                for c in item.content:
                    if hasattr(c, "text"):
                        texts.append(c.text)
                    elif isinstance(c, str):
                        texts.append(c)

    return "".join(texts)


def extract_usage(response: Response) -> Optional[Dict[str, Any]]:
    """
    Extract token usage from Response object.

    Returns:
        Dict with input_tokens, output_tokens, total_tokens
    """
    if not response or not response.usage:
        return None

    usage = response.usage
    return {
        "input_tokens": getattr(usage, "input_tokens", 0),
        "output_tokens": getattr(usage, "output_tokens", 0),
        "total_tokens": getattr(usage, "total_tokens", 0),
    }


def create_responses_client(
    api_key: str,
    timeout: float = DEFAULT_TIMEOUT,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> ResponsesAPIClient:
    """Create a ResponsesAPIClient instance."""
    return ResponsesAPIClient(
        api_key=api_key,
        timeout=timeout,
        max_retries=max_retries,
    )

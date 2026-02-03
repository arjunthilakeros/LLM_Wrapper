"""
Input Validation and Sanitization for Terminal Chatbot
Provides security-focused validation functions.

Includes OpenAI-compatible validators for unified API.
"""

import base64
import re
import string
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from exceptions import (
    ContentTypeError,
    MessageValidationError,
    ModelNotSupportedError,
    ValidationError,
)


# Characters to remove from user input (control characters except newline/tab)
CONTROL_CHARS = ''.join(
    chr(i) for i in range(32) if chr(i) not in '\n\t\r'
)
CONTROL_CHARS_PATTERN = re.compile(f'[{re.escape(CONTROL_CHARS)}]')

# Patterns that may indicate prompt injection attempts
INJECTION_PATTERNS = [
    r'ignore\s+(all\s+)?(previous|above|prior)\s+(instructions?|prompts?)',
    r'disregard\s+(all\s+)?(previous|above|prior)',
    r'forget\s+(all\s+)?(previous|above|prior)',
    r'new\s+instructions?:',
    r'system\s*:\s*you\s+are',
    r'\[SYSTEM\]',
    r'<\|im_start\|>',
    r'<\|im_end\|>',
]
INJECTION_REGEX = re.compile(
    '|'.join(INJECTION_PATTERNS),
    re.IGNORECASE
)

# Valid conversation ID pattern (OpenAI format)
CONVERSATION_ID_PATTERN = re.compile(r'^conv_[a-zA-Z0-9]{20,}$')

# OpenAI-compatible message roles
VALID_MESSAGE_ROLES = {"system", "developer", "user", "assistant", "tool"}

# OpenAI-compatible content part types
VALID_CONTENT_TYPES = {"text", "image_url", "file"}

# Supported models (OpenAI-compatible)
VALID_MODELS = {
    "gpt-4o", "gpt-4o-mini", "gpt-4o-latest",
    "gpt-4-turbo", "gpt-4-turbo-preview",
    "gpt-4", "gpt-4-0613", "gpt-4-32k",
    "gpt-3.5-turbo", "gpt-3.5-turbo-0613", "gpt-3.5-turbo-16k",
    "o1", "o1-mini", "o1-preview",
    "o3-mini"
}

# Image detail options
VALID_IMAGE_DETAILS = {"auto", "low", "high"}

# Data URI pattern for base64 images
DATA_URI_PATTERN = re.compile(
    r'^data:image/(?P<type>[a-zA-Z+]+);base64,(?P<data>[A-Za-z0-9+/=]+)$'
)

# HTTP/HTTPS URL pattern
HTTP_URL_PATTERN = re.compile(
    r'^https?://[^\s<>"\'{}|\\^`\[\]]+$',
    re.IGNORECASE
)


def sanitize_input(
    text: str,
    max_length: int = 10000,
    remove_control_chars: bool = True,
    check_injection: bool = True
) -> Tuple[str, list]:
    """
    Sanitize user input text.

    Args:
        text: Raw input text
        max_length: Maximum allowed length
        remove_control_chars: Whether to strip control characters
        check_injection: Whether to check for prompt injection patterns

    Returns:
        Tuple of (sanitized_text, list of warnings)

    Raises:
        ValidationError: If input is invalid
    """
    if text is None:
        raise ValidationError(
            "Input cannot be None",
            field="text",
            reason="null_input"
        )

    warnings = []
    sanitized = text

    # Remove control characters (keep newlines, tabs)
    if remove_control_chars:
        original_len = len(sanitized)
        sanitized = CONTROL_CHARS_PATTERN.sub('', sanitized)
        if len(sanitized) < original_len:
            warnings.append("Removed control characters from input")

    # Truncate if too long
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length]
        warnings.append(f"Input truncated to {max_length} characters")

    # Check for potential prompt injection (warning only, don't block)
    if check_injection and INJECTION_REGEX.search(sanitized):
        warnings.append("Input contains patterns that may be prompt injection attempts")

    # Strip leading/trailing whitespace
    sanitized = sanitized.strip()

    if not sanitized:
        raise ValidationError(
            "Input cannot be empty",
            field="text",
            reason="empty_input"
        )

    return sanitized, warnings


def validate_conversation_id(conv_id: str) -> bool:
    """
    Validate a conversation ID format.

    Args:
        conv_id: Conversation ID to validate

    Returns:
        True if valid

    Raises:
        ValidationError: If format is invalid
    """
    if not conv_id:
        raise ValidationError(
            "Conversation ID cannot be empty",
            field="conversation_id",
            reason="empty"
        )

    if not isinstance(conv_id, str):
        raise ValidationError(
            "Conversation ID must be a string",
            field="conversation_id",
            value=type(conv_id).__name__,
            reason="invalid_type"
        )

    # OpenAI conversation IDs start with "conv_"
    if not CONVERSATION_ID_PATTERN.match(conv_id):
        raise ValidationError(
            "Invalid conversation ID format",
            field="conversation_id",
            value=conv_id[:20] + "..." if len(conv_id) > 20 else conv_id,
            reason="invalid_format"
        )

    return True


def validate_file_path(
    filepath: str,
    allowed_extensions: set = None,
    max_path_length: int = 500
) -> Tuple[Path, list]:
    """
    Validate a file path for security issues.

    Args:
        filepath: File path to validate
        allowed_extensions: Set of allowed file extensions (e.g., {'.txt', '.pdf'})
        max_path_length: Maximum path length

    Returns:
        Tuple of (resolved Path object, list of warnings)

    Raises:
        ValidationError: If path is invalid or potentially malicious
    """
    if not filepath:
        raise ValidationError(
            "File path cannot be empty",
            field="filepath",
            reason="empty"
        )

    warnings = []

    # Check path length
    if len(filepath) > max_path_length:
        raise ValidationError(
            f"File path exceeds maximum length of {max_path_length}",
            field="filepath",
            reason="too_long"
        )

    # Check for null bytes (path traversal attack vector)
    if '\x00' in filepath:
        raise ValidationError(
            "File path contains null bytes",
            field="filepath",
            reason="null_bytes"
        )

    # Convert to Path object
    try:
        path = Path(filepath)
    except Exception as e:
        raise ValidationError(
            f"Invalid file path: {e}",
            field="filepath",
            value=filepath,
            reason="invalid_path"
        )

    # Check for path traversal attempts
    # Resolve the path and check if it's trying to escape
    try:
        if not path.is_absolute():
            path = Path.cwd() / path

        resolved = path.resolve()

        # Check for .. components in the original path that might indicate traversal
        if '..' in filepath:
            warnings.append("Path contains relative traversal components (..)")

    except Exception as e:
        raise ValidationError(
            f"Cannot resolve file path: {e}",
            field="filepath",
            value=filepath,
            reason="resolve_failed"
        )

    # Check file extension if restrictions provided
    if allowed_extensions:
        ext = resolved.suffix.lower()
        if ext not in allowed_extensions:
            raise ValidationError(
                f"File extension '{ext}' is not allowed",
                field="filepath",
                value=ext,
                reason="invalid_extension"
            )

    return resolved, warnings


def validate_positive_number(
    value,
    field_name: str,
    min_value: float = None,
    max_value: float = None,
    allow_zero: bool = False
) -> float:
    """
    Validate that a value is a positive number.

    Args:
        value: Value to validate
        field_name: Name of the field (for error messages)
        min_value: Minimum allowed value
        max_value: Maximum allowed value
        allow_zero: Whether zero is allowed

    Returns:
        Validated number

    Raises:
        ValidationError: If validation fails
    """
    try:
        num = float(value)
    except (TypeError, ValueError):
        raise ValidationError(
            f"{field_name} must be a number",
            field=field_name,
            value=value,
            reason="not_a_number"
        )

    if not allow_zero and num == 0:
        raise ValidationError(
            f"{field_name} cannot be zero",
            field=field_name,
            value=num,
            reason="zero_not_allowed"
        )

    if num < 0 and (min_value is None or min_value >= 0):
        raise ValidationError(
            f"{field_name} must be positive",
            field=field_name,
            value=num,
            reason="negative"
        )

    if min_value is not None and num < min_value:
        raise ValidationError(
            f"{field_name} must be at least {min_value}",
            field=field_name,
            value=num,
            reason="below_minimum"
        )

    if max_value is not None and num > max_value:
        raise ValidationError(
            f"{field_name} must be at most {max_value}",
            field=field_name,
            value=num,
            reason="above_maximum"
        )

    return num


# =============================================================================
# OpenAI-Compatible Validators (New)
# =============================================================================


def validate_chat_messages(messages: List[dict]) -> Tuple[List[dict], List[str]]:
    """
    Validate OpenAI message format.

    Rules:
    - Must be non-empty list
    - Each message must have 'role' and 'content'
    - Role must be in: system, developer, user, assistant, tool
    - Content can be: string OR list of content parts

    Args:
        messages: List of message dictionaries

    Returns:
        Tuple of (validated_messages, warnings)

    Raises:
        MessageValidationError: If message format is invalid
    """
    warnings = []

    # Must be a non-empty list
    if not messages:
        raise MessageValidationError(
            "Messages must be a non-empty list",
            message_index=None,
            field="messages"
        )

    if not isinstance(messages, list):
        raise MessageValidationError(
            f"Messages must be a list, got {type(messages).__name__}",
            message_index=None,
            field="messages"
        )

    validated_messages = []

    for idx, msg in enumerate(messages):
        # Must be a dictionary
        if not isinstance(msg, dict):
            raise MessageValidationError(
                f"Message must be an object, got {type(msg).__name__}",
                message_index=idx,
                field="message"
            )

        # Must have 'role' field
        if "role" not in msg:
            raise MessageValidationError(
                "Message missing required field: 'role'",
                message_index=idx,
                field="role"
            )

        role = msg.get("role")
        if role not in VALID_MESSAGE_ROLES:
            raise MessageValidationError(
                f"Invalid role: '{role}'. Must be one of: {', '.join(sorted(VALID_MESSAGE_ROLES))}",
                message_index=idx,
                field="role"
            )

        # Must have 'content' field (can be null for assistant with tool_calls)
        if "content" not in msg:
            raise MessageValidationError(
                "Message missing required field: 'content'",
                message_index=idx,
                field="content"
            )

        content = msg.get("content")

        # Content can be:
        # 1. None (for assistant with tool_calls)
        # 2. String
        # 3. List of content parts
        if content is None:
            # Allow null content for assistant messages with tool_calls
            if role == "assistant" and "tool_calls" in msg:
                pass  # Valid case
            else:
                raise MessageValidationError(
                    "Message content cannot be null unless assistant has tool_calls",
                    message_index=idx,
                    field="content"
                )
        elif isinstance(content, str):
            # Valid: simple text content
            pass
        elif isinstance(content, list):
            # Must be non-empty array of content parts
            if len(content) == 0:
                raise MessageValidationError(
                    "Content array cannot be empty",
                    message_index=idx,
                    field="content"
                )

            # Validate each content part
            for part_idx, part in enumerate(content):
                try:
                    validate_content_part(part)
                except ContentTypeError as e:
                    raise MessageValidationError(
                        f"Invalid content part at index {part_idx}: {e.message}",
                        message_index=idx,
                        field=f"content[{part_idx}]"
                    )
        else:
            raise MessageValidationError(
                f"Content must be a string, array, or null, got {type(content).__name__}",
                message_index=idx,
                field="content"
            )

        # Validate 'name' field if present (optional)
        if "name" in msg and msg["name"] is not None:
            name = msg["name"]
            if not isinstance(name, str):
                raise MessageValidationError(
                    f"Name must be a string, got {type(name).__name__}",
                    message_index=idx,
                    field="name"
                )
            if len(name) > 64:
                warnings.append(f"Message {idx}: name exceeds 64 characters, may be truncated")

        # Validate tool-specific fields
        if role == "tool":
            if "tool_call_id" not in msg:
                raise MessageValidationError(
                    "Tool message must have 'tool_call_id'",
                    message_index=idx,
                    field="tool_call_id"
                )

        validated_messages.append(msg)

    return validated_messages, warnings


def validate_content_part(part: dict) -> dict:
    """
    Validate a content part (text or image_url).

    Rules:
    - Must have 'type' field
    - Type "text": must have 'text' field
    - Type "image_url": must have 'image_url' with 'url'
    - Type "file": must have 'file' with 'url'

    Args:
        part: Content part dictionary

    Returns:
        Validated content part (may have defaults applied)

    Raises:
        ContentTypeError: If content type is invalid
        ValidationError: If validation fails
    """
    if not isinstance(part, dict):
        raise ValidationError(
            "Content part must be an object",
            field="content_part",
            value=type(part).__name__,
            reason="invalid_type"
        )

    if "type" not in part:
        raise ContentTypeError(
            "missing",
            allowed_types=list(VALID_CONTENT_TYPES)
        )

    content_type = part.get("type")

    if content_type not in VALID_CONTENT_TYPES:
        raise ContentTypeError(
            content_type,
            allowed_types=list(VALID_CONTENT_TYPES)
        )

    validated = {"type": content_type}

    if content_type == "text":
        if "text" not in part:
            raise ValidationError(
                "Text content part must have 'text' field",
                field="text.text",
                reason="missing_field"
            )
        if not isinstance(part["text"], str):
            raise ValidationError(
                "Text content must be a string",
                field="text.text",
                value=type(part["text"]).__name__,
                reason="invalid_type"
            )
        validated["text"] = part["text"]

    elif content_type == "image_url":
        if "image_url" not in part:
            raise ValidationError(
                "Image content part must have 'image_url' field",
                field="image_url",
                reason="missing_field"
            )

        image_url_obj = part["image_url"]
        if not isinstance(image_url_obj, dict):
            raise ValidationError(
                "image_url must be an object",
                field="image_url",
                value=type(image_url_obj).__name__,
                reason="invalid_type"
            )

        if "url" not in image_url_obj:
            raise ValidationError(
                "image_url must contain 'url' field",
                field="image_url.url",
                reason="missing_field"
            )

        url = image_url_obj["url"]
        if not isinstance(url, str):
            raise ValidationError(
                "image_url.url must be a string",
                field="image_url.url",
                value=type(url).__name__,
                reason="invalid_type"
            )

        # Validate the URL format
        is_valid, error_msg = validate_image_url(url)
        if not is_valid:
            raise ValidationError(
                error_msg,
                field="image_url.url",
                value=url[:50] + "..." if len(url) > 50 else url,
                reason="invalid_url"
            )

        validated["image_url"] = {"url": url}

        # Handle detail parameter
        detail = image_url_obj.get("detail", "auto")
        if detail not in VALID_IMAGE_DETAILS:
            detail = "auto"  # Default to auto for invalid values
        validated["image_url"]["detail"] = detail

    elif content_type == "file":
        if "file" not in part:
            raise ValidationError(
                "File content part must have 'file' field",
                field="file",
                reason="missing_field"
            )

        file_obj = part["file"]
        if not isinstance(file_obj, dict):
            raise ValidationError(
                "file must be an object",
                field="file",
                value=type(file_obj).__name__,
                reason="invalid_type"
            )

        if "url" not in file_obj:
            raise ValidationError(
                "file must contain 'url' field",
                field="file.url",
                reason="missing_field"
            )

        validated["file"] = {
            "url": file_obj["url"],
            "name": file_obj.get("name", "")
        }

    return validated


def validate_model(model: str) -> str:
    """
    Validate model name against supported models.

    Args:
        model: Model name to validate

    Returns:
        Validated model name

    Raises:
        ModelNotSupportedError: If model is not supported
    """
    if not model:
        raise ModelNotSupportedError(
            model="(empty)",
            supported_models=sorted(VALID_MODELS)
        )

    if not isinstance(model, str):
        raise ModelNotSupportedError(
            model=f"{type(model).__name__}",
            supported_models=sorted(VALID_MODELS)
        )

    # Check exact match
    if model in VALID_MODELS:
        return model

    # Check if it's a dated model version (e.g., gpt-4o-2024-05-13)
    # Allow models that start with supported base models
    for valid_model in VALID_MODELS:
        if model.startswith(valid_model + "-"):
            return model

    # Model not supported
    raise ModelNotSupportedError(
        model=model,
        supported_models=sorted(VALID_MODELS)
    )


def validate_chat_request_params(params: dict) -> Tuple[dict, List[str]]:
    """
    Validate OpenAI request parameters.

    Rules:
    - temperature: 0-2, default 1
    - max_tokens: positive int or null
    - max_completion_tokens: positive int or null (preferred over max_tokens)
    - top_p: 0-1, default 1
    - presence_penalty: -2 to 2, default 0
    - frequency_penalty: -2 to 2, default 0
    - stream: boolean, default false
    - n: integer 1-128, default 1

    Args:
        params: Dictionary of request parameters

    Returns:
        Tuple of (validated_params, warnings)

    Raises:
        ValidationError: If validation fails
    """
    if not isinstance(params, dict):
        raise ValidationError(
            "Parameters must be an object",
            field="params",
            value=type(params).__name__,
            reason="invalid_type"
        )

    validated = {}
    warnings = []

    # Temperature: 0-2, default 1
    temperature = params.get("temperature", 1.0)
    if temperature is not None:
        try:
            temp_val = float(temperature)
            if temp_val < 0 or temp_val > 2:
                warnings.append(f"temperature {temp_val} out of range [0, 2], clamping")
                temp_val = max(0, min(2, temp_val))
            validated["temperature"] = temp_val
        except (TypeError, ValueError):
            raise ValidationError(
                "temperature must be a number",
                field="temperature",
                value=temperature,
                reason="invalid_type"
            )
    else:
        validated["temperature"] = 1.0

    # max_tokens / max_completion_tokens: positive int or null
    max_tokens = params.get("max_completion_tokens") or params.get("max_tokens")
    if max_tokens is not None:
        try:
            max_tok = int(max_tokens)
            if max_tok < 1:
                raise ValidationError(
                    "max_tokens must be a positive integer",
                    field="max_tokens",
                    value=max_tok,
                    reason="below_minimum"
                )
            validated["max_completion_tokens"] = max_tok
        except (TypeError, ValueError):
            raise ValidationError(
                "max_tokens must be an integer",
                field="max_tokens",
                value=max_tokens,
                reason="invalid_type"
            )
    else:
        validated["max_completion_tokens"] = None

    # Warn about deprecated max_tokens
    if "max_tokens" in params and "max_completion_tokens" not in params:
        warnings.append("max_tokens is deprecated, use max_completion_tokens instead")

    # top_p: 0-1, default 1
    top_p = params.get("top_p", 1.0)
    if top_p is not None:
        try:
            top_p_val = float(top_p)
            if top_p_val < 0 or top_p_val > 1:
                warnings.append(f"top_p {top_p_val} out of range [0, 1], clamping")
                top_p_val = max(0, min(1, top_p_val))
            validated["top_p"] = top_p_val
        except (TypeError, ValueError):
            raise ValidationError(
                "top_p must be a number",
                field="top_p",
                value=top_p,
                reason="invalid_type"
            )
    else:
        validated["top_p"] = 1.0

    # presence_penalty: -2 to 2, default 0
    presence_penalty = params.get("presence_penalty", 0)
    if presence_penalty is not None:
        try:
            pres_val = float(presence_penalty)
            if pres_val < -2 or pres_val > 2:
                warnings.append(f"presence_penalty {pres_val} out of range [-2, 2], clamping")
                pres_val = max(-2, min(2, pres_val))
            validated["presence_penalty"] = pres_val
        except (TypeError, ValueError):
            raise ValidationError(
                "presence_penalty must be a number",
                field="presence_penalty",
                value=presence_penalty,
                reason="invalid_type"
            )
    else:
        validated["presence_penalty"] = 0

    # frequency_penalty: -2 to 2, default 0
    frequency_penalty = params.get("frequency_penalty", 0)
    if frequency_penalty is not None:
        try:
            freq_val = float(frequency_penalty)
            if freq_val < -2 or freq_val > 2:
                warnings.append(f"frequency_penalty {freq_val} out of range [-2, 2], clamping")
                freq_val = max(-2, min(2, freq_val))
            validated["frequency_penalty"] = freq_val
        except (TypeError, ValueError):
            raise ValidationError(
                "frequency_penalty must be a number",
                field="frequency_penalty",
                value=frequency_penalty,
                reason="invalid_type"
            )
    else:
        validated["frequency_penalty"] = 0

    # stream: boolean, default false
    stream = params.get("stream", False)
    if stream is not None:
        if isinstance(stream, bool):
            validated["stream"] = stream
        elif isinstance(stream, str):
            validated["stream"] = stream.lower() in ("true", "1", "yes")
        else:
            validated["stream"] = bool(stream)
    else:
        validated["stream"] = False

    # n: integer 1-128, default 1
    n = params.get("n", 1)
    if n is not None:
        try:
            n_val = int(n)
            if n_val < 1 or n_val > 128:
                raise ValidationError(
                    "n must be between 1 and 128",
                    field="n",
                    value=n_val,
                    reason="out_of_range"
                )
            validated["n"] = n_val
        except (TypeError, ValueError):
            raise ValidationError(
                "n must be an integer",
                field="n",
                value=n,
                reason="invalid_type"
            )
    else:
        validated["n"] = 1

    # store: boolean, default true
    store = params.get("store", True)
    validated["store"] = bool(store) if store is not None else True

    # user_id: optional string
    user_id = params.get("user_id")
    if user_id is not None:
        if not isinstance(user_id, str):
            raise ValidationError(
                "user_id must be a string",
                field="user_id",
                value=type(user_id).__name__,
                reason="invalid_type"
            )
        validated["user_id"] = user_id

    # conversation_id: optional string
    conversation_id = params.get("conversation_id")
    if conversation_id is not None:
        if not isinstance(conversation_id, str):
            raise ValidationError(
                "conversation_id must be a string",
                field="conversation_id",
                value=type(conversation_id).__name__,
                reason="invalid_type"
            )
        validated["conversation_id"] = conversation_id

    return validated, warnings


def validate_image_url(url: str) -> Tuple[bool, str]:
    """
    Validate image URL (data URI or HTTP).

    Rules:
    - data:image/{type};base64,{data}
    - http:// or https://

    Args:
        url: URL string to validate

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not url:
        return False, "Image URL cannot be empty"

    if not isinstance(url, str):
        return False, f"Image URL must be a string, got {type(url).__name__}"

    # Check for data URI
    if url.startswith("data:"):
        match = DATA_URI_PATTERN.match(url)
        if not match:
            return False, "Invalid data URI format. Expected: data:image/{type};base64,{data}"

        # Validate base64 data
        base64_data = match.group("data")
        try:
            # Check if it's valid base64
            base64.b64decode(base64_data, validate=True)
        except Exception:
            return False, "Invalid base64 data in data URI"

        return True, ""

    # Check for HTTP/HTTPS URL
    if url.startswith(("http://", "https://")):
        if not HTTP_URL_PATTERN.match(url):
            return False, "Invalid HTTP URL format"
        return True, ""

    return False, "Image URL must be a data URI or HTTP(S) URL"

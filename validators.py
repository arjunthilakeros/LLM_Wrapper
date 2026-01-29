"""
Input Validation and Sanitization for Terminal Chatbot
Provides security-focused validation functions.
"""

import re
import string
from pathlib import Path
from typing import Tuple

from exceptions import ValidationError


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

    if num < 0:
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

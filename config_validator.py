"""
Configuration Validator for Terminal Chatbot
Validates configuration on startup and fails fast with clear error messages.
"""

from typing import Any, Dict, List, Optional, Union

from exceptions import ConfigurationError


# Configuration schema definition
CONFIG_SCHEMA = {
    # Model settings
    "model": {
        "type": str,
        "required": True,
        "allowed_values": [
            "gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-4",
            "gpt-3.5-turbo", "o1", "o1-mini", "o1-preview"
        ],
        "default": "gpt-4o"
    },
    # Conversation settings
    "max_history_items": {
        "type": int,
        "required": False,
        "min": 10,
        "max": 1000,
        "default": 100
    },

    # Rate limiting
    "rate_limit_per_minute": {
        "type": int,
        "required": False,
        "min": 1,
        "max": 100,
        "default": 10
    },
    "max_input_length": {
        "type": int,
        "required": False,
        "min": 100,
        "max": 100000,
        "default": 10000
    },

    # File upload settings
    "max_file_size_mb": {
        "type": (int, float),
        "required": False,
        "min": 1,
        "max": 100,
        "default": 20
    },
    "max_document_chars": {
        "type": int,
        "required": False,
        "min": 1000,
        "max": 500000,
        "default": 50000
    },

    # Cost control
    "warn_at_cost": {
        "type": (int, float),
        "required": False,
        "min": 0.01,
        "max": 1000.0,
        "default": 1.0
    },

    # Pricing
    "pricing": {
        "type": dict,
        "required": False,
        "default": {"input_per_1k": 0.0025, "output_per_1k": 0.01},
        "nested_schema": {
            "input_per_1k": {"type": (int, float), "min": 0, "required": True},
            "output_per_1k": {"type": (int, float), "min": 0, "required": True}
        }
    },

    # Storage directories
    "data_dir": {
        "type": str,
        "required": False,
        "default": "./data"
    },
    "export_dir": {
        "type": str,
        "required": False,
        "default": "./exports"
    },
    "upload_dir": {
        "type": str,
        "required": False,
        "default": "./uploads"
    },

    # Display options
    "show_tokens": {
        "type": bool,
        "required": False,
        "default": True
    },
    "show_cost": {
        "type": bool,
        "required": False,
        "default": True
    },
    "stream_responses": {
        "type": bool,
        "required": False,
        "default": True
    },

    # Logging settings
    "logging": {
        "type": dict,
        "required": False,
        "default": {"level": "INFO", "log_to_file": True, "log_dir": "./logs"},
        "nested_schema": {
            "level": {
                "type": str,
                "required": False,
                "allowed_values": ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                "default": "INFO"
            },
            "log_to_file": {
                "type": bool,
                "required": False,
                "default": True
            },
            "log_dir": {
                "type": str,
                "required": False,
                "default": "./logs"
            }
        }
    },

    # API settings
    "api_timeout": {
        "type": (int, float),
        "required": False,
        "min": 5,
        "max": 300,
        "default": 30
    },
    "api_max_retries": {
        "type": int,
        "required": False,
        "min": 0,
        "max": 10,
        "default": 3
    },

    # Context Management - Summary + Window strategy
    "context_management": {
        "type": dict,
        "required": False,
        "default": {
            "mode": "full",
            "window_size": 10,
            "max_context_tokens": 2500,
            "summarize_after_messages": 10,
            "summary_model": "gpt-4o-mini",
            "max_summary_tokens": 500,
            "summary_update_interval": 10
        },
        "nested_schema": {
            "mode": {
                "type": str,
                "required": False,
                "allowed_values": ["full", "summary_window"],
                "default": "full"
            },
            "window_size": {
                "type": int,
                "required": False,
                "min": 1,
                "max": 50,
                "default": 10
            },
            "max_context_tokens": {
                "type": int,
                "required": False,
                "min": 500,
                "max": 100000,
                "default": 2500
            },
            "summarize_after_messages": {
                "type": int,
                "required": False,
                "min": 5,
                "max": 100,
                "default": 10
            },
            "summary_model": {
                "type": str,
                "required": False,
                "default": "gpt-4o-mini"
            },
            "max_summary_tokens": {
                "type": int,
                "required": False,
                "min": 100,
                "max": 2000,
                "default": 500
            },
            "summary_update_interval": {
                "type": int,
                "required": False,
                "min": 5,
                "max": 50,
                "default": 10
            }
        }
    }
}


def _validate_field(
    field_name: str,
    value: Any,
    schema: Dict[str, Any],
    parent_path: str = ""
) -> List[str]:
    """
    Validate a single configuration field.

    Returns:
        List of validation error messages
    """
    errors = []
    full_path = f"{parent_path}.{field_name}" if parent_path else field_name

    # Check type
    expected_type = schema.get("type")
    if expected_type and value is not None:
        if isinstance(expected_type, tuple):
            if not isinstance(value, expected_type):
                errors.append(
                    f"'{full_path}' must be one of types {expected_type}, "
                    f"got {type(value).__name__}"
                )
        elif not isinstance(value, expected_type):
            errors.append(
                f"'{full_path}' must be type {expected_type.__name__}, "
                f"got {type(value).__name__}"
            )

    # Check allowed values
    allowed_values = schema.get("allowed_values")
    if allowed_values and value is not None:
        if value not in allowed_values:
            errors.append(
                f"'{full_path}' must be one of {allowed_values}, got '{value}'"
            )

    # Check numeric ranges
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        min_val = schema.get("min")
        max_val = schema.get("max")

        if min_val is not None and value < min_val:
            errors.append(
                f"'{full_path}' must be at least {min_val}, got {value}"
            )

        if max_val is not None and value > max_val:
            errors.append(
                f"'{full_path}' must be at most {max_val}, got {value}"
            )

    # Validate nested schema for dicts
    nested_schema = schema.get("nested_schema")
    if nested_schema and isinstance(value, dict):
        for nested_field, nested_field_schema in nested_schema.items():
            nested_value = value.get(nested_field)

            if nested_field_schema.get("required") and nested_value is None:
                errors.append(f"'{full_path}.{nested_field}' is required")
            elif nested_value is not None:
                errors.extend(_validate_field(
                    nested_field,
                    nested_value,
                    nested_field_schema,
                    full_path
                ))

    return errors


def validate_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate configuration dictionary against the schema.

    Args:
        config: Configuration dictionary to validate

    Returns:
        Validated configuration with defaults applied

    Raises:
        ConfigurationError: If validation fails
    """
    if not isinstance(config, dict):
        raise ConfigurationError(
            "Configuration must be a dictionary",
            field="config",
            value=type(config).__name__
        )

    errors = []
    validated = {}

    # Validate each schema field
    for field_name, field_schema in CONFIG_SCHEMA.items():
        value = config.get(field_name)

        # Check required fields
        if field_schema.get("required") and value is None:
            errors.append(f"Required field '{field_name}' is missing")
            continue

        # Apply default if not provided
        if value is None:
            value = field_schema.get("default")

        # Validate if value exists
        if value is not None:
            field_errors = _validate_field(field_name, value, field_schema)
            errors.extend(field_errors)

            # For nested dicts, apply nested defaults
            if field_schema.get("type") == dict and isinstance(value, dict):
                nested_schema = field_schema.get("nested_schema", {})
                for nested_field, nested_field_schema in nested_schema.items():
                    if nested_field not in value:
                        default = nested_field_schema.get("default")
                        if default is not None:
                            value[nested_field] = default

        validated[field_name] = value

    # Check for unknown fields (warning, not error)
    unknown_fields = set(config.keys()) - set(CONFIG_SCHEMA.keys())
    if unknown_fields:
        # These are warnings, not errors - allow custom fields
        pass

    # Raise if there are validation errors
    if errors:
        raise ConfigurationError(
            f"Configuration validation failed:\n  - " + "\n  - ".join(errors),
            field="config"
        )

    return validated


def get_config_with_defaults(user_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Get configuration with all defaults applied.

    Args:
        user_config: Optional user-provided configuration

    Returns:
        Complete configuration with defaults
    """
    config = {}

    # Apply defaults from schema
    for field_name, field_schema in CONFIG_SCHEMA.items():
        default = field_schema.get("default")
        if default is not None:
            config[field_name] = default

    # Override with user config
    if user_config:
        for key, value in user_config.items():
            if value is not None:
                config[key] = value

    return validate_config(config)

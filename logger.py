"""
Logging Framework for Terminal Chatbot
Provides rotating file handler with configurable levels and formats.
"""

import logging
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional


class ChatbotLogger:
    """Centralized logging with rotating file handler and console output."""

    _instance: Optional['ChatbotLogger'] = None
    _logger: Optional[logging.Logger] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def setup(
        cls,
        level: str = "INFO",
        log_to_file: bool = True,
        log_dir: str = "./logs",
        max_file_size_mb: int = 10,
        backup_count: int = 5
    ) -> logging.Logger:
        """
        Configure and return the logger instance.

        Args:
            level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_to_file: Whether to write logs to file
            log_dir: Directory for log files
            max_file_size_mb: Maximum size of each log file in MB
            backup_count: Number of backup files to keep

        Returns:
            Configured logger instance
        """
        if cls._logger is not None:
            return cls._logger

        cls._logger = logging.getLogger("terminal_chatbot")
        cls._logger.setLevel(getattr(logging, level.upper(), logging.INFO))
        cls._logger.handlers.clear()

        # Format: timestamp | level | module:function:line | message
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(module)s:%(funcName)s:%(lineno)d | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        # Console handler (only errors and above to avoid cluttering terminal)
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setLevel(logging.ERROR)
        console_handler.setFormatter(formatter)
        cls._logger.addHandler(console_handler)

        # File handler with rotation
        if log_to_file:
            log_path = Path(log_dir)
            log_path.mkdir(parents=True, exist_ok=True)

            log_filename = log_path / f"chatbot_{datetime.now().strftime('%Y%m%d')}.log"

            file_handler = RotatingFileHandler(
                filename=str(log_filename),
                maxBytes=max_file_size_mb * 1024 * 1024,
                backupCount=backup_count,
                encoding="utf-8"
            )
            file_handler.setLevel(getattr(logging, level.upper(), logging.INFO))
            file_handler.setFormatter(formatter)
            cls._logger.addHandler(file_handler)

        return cls._logger

    @classmethod
    def get_logger(cls) -> logging.Logger:
        """Get the logger instance, creating with defaults if not set up."""
        if cls._logger is None:
            return cls.setup()
        return cls._logger


def get_logger() -> logging.Logger:
    """Convenience function to get the logger instance."""
    return ChatbotLogger.get_logger()


def setup_logging(
    level: str = "INFO",
    log_to_file: bool = True,
    log_dir: str = "./logs"
) -> logging.Logger:
    """
    Convenience function to set up logging.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_to_file: Whether to write logs to file
        log_dir: Directory for log files

    Returns:
        Configured logger instance
    """
    return ChatbotLogger.setup(
        level=level,
        log_to_file=log_to_file,
        log_dir=log_dir
    )

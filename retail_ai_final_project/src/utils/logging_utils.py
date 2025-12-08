"""
Logging Utilities
Provides consistent logging configuration across the project.
"""
import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    format_string: Optional[str] = None
) -> None:
    """
    Set up logging configuration for the entire project.

    Args:
        level: Logging level (default: INFO)
        log_file: Optional file path to write logs
        format_string: Custom format string for log messages
    """
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    # Configure root logger
    handlers = [logging.StreamHandler(sys.stdout)]

    if log_file:
        # Ensure log directory exists
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=level,
        format=format_string,
        handlers=handlers
    )


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Get a configured logger instance.

    Args:
        name: Name of the logger (typically __name__)
        level: Logging level (default: INFO)

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)

    # Only configure if handlers aren't already set
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(level)

    return logger


class ProgressLogger:
    """
    Simple progress logger for long-running operations.
    """

    def __init__(self, total: int, description: str = "Processing"):
        """
        Initialize progress logger.

        Args:
            total: Total number of items to process
            description: Description of the operation
        """
        self.total = total
        self.current = 0
        self.description = description
        self.logger = get_logger(__name__)

    def update(self, n: int = 1) -> None:
        """
        Update progress by n items.

        Args:
            n: Number of items processed
        """
        self.current += n
        percentage = (self.current / self.total) * 100
        self.logger.info(f"{self.description}: {self.current}/{self.total} ({percentage:.1f}%)")

    def finish(self) -> None:
        """Mark progress as complete."""
        self.logger.info(f"{self.description}: Completed!")


# Initialize default logging configuration
setup_logging()

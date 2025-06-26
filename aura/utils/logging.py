"""
Logging utilities for AURA system.
"""
import logging
import sys
from pathlib import Path
from typing import Optional, Dict, Any

from rich.console import Console
from rich.logging import RichHandler

from aura.config import settings


# Create console for rich output
console = Console()


def configure_logging(
    level: Optional[str] = None,
    format_str: Optional[str] = None,
    log_file: Optional[str] = None
) -> None:
    """Configure logging for the application.
    
    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_str: Log format string
        log_file: Path to log file
    """
    # Use settings if parameters not provided
    level = level or settings.logging.level
    format_str = format_str or settings.logging.format
    log_file = log_file or settings.logging.file_path
    
    # Convert level string to logging level
    log_level = getattr(logging, level.upper(), logging.INFO)
    
    # Configure root logger
    logging.basicConfig(
        level=log_level,
        format=format_str,
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            RichHandler(console=console, rich_tracebacks=True)
        ]
    )
    
    # Add file handler if log file specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(format_str))
        logging.getLogger().addHandler(file_handler)
    
    # Set lower log level for some noisy libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)
    logging.getLogger("git").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the specified name.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


class LoggingContext:
    """Context manager for temporarily changing log level."""
    
    def __init__(self, logger: logging.Logger, level: int):
        """Initialize the context manager.
        
        Args:
            logger: Logger to modify
            level: Temporary log level
        """
        self.logger = logger
        self.level = level
        self.old_level = logger.level
    
    def __enter__(self) -> logging.Logger:
        """Enter the context manager."""
        self.logger.setLevel(self.level)
        return self.logger
    
    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit the context manager."""
        self.logger.setLevel(self.old_level)


# Configure logging on module import
configure_logging()

# Create module-level logger
logger = get_logger("aura")
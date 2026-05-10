"""
Structured Logger
=================
Provides per-module loggers with consistent formatting.
Logs to both file and console.
"""

import logging
import sys
from pathlib import Path

from config import LOG_DIR, LOG_FORMAT, LOG_LEVEL


def setup_logger(name: str, log_file: str = None) -> logging.Logger:
    """
    Create a named logger that writes to console + file.
    
    Args:
        name: Logger name (use __name__ of the calling module)
        log_file: Optional custom log filename. Default: {name}.log
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Prevent duplicate handlers on re-import
    if logger.handlers:
        return logger
    
    logger.setLevel(getattr(logging, LOG_LEVEL.upper(), logging.INFO))
    logger.propagate = False
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    console_formatter = logging.Formatter(LOG_FORMAT)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file is None:
        log_file = f"{name.split('.')[-1]}.log"
    file_path = LOG_DIR / log_file
    file_handler = logging.FileHandler(file_path, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(LOG_FORMAT)
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    return logger

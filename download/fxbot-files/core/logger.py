# =============================================================
# core/logger.py
# Central logging for the entire system.
# All modules use this — consistent, timestamped, color-coded.
# =============================================================

import logging
import os
from datetime import datetime

LOG_DIR = os.path.join(os.path.dirname(__file__), '..', 'logs')
os.makedirs(LOG_DIR, exist_ok=True)

def get_logger(name: str) -> logging.Logger:
    """
    Returns a named logger that writes to both console and a daily log file.
    Usage:  from core.logger import get_logger
            log = get_logger(__name__)
            log.info("Bot started")
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # Already configured

    logger.setLevel(logging.DEBUG)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    fmt = logging.Formatter('[%(asctime)s] [%(levelname)s] %(name)s: %(message)s',
                            datefmt='%H:%M:%S')
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler — new file each day
    today = datetime.now().strftime('%Y-%m-%d')
    fh = logging.FileHandler(os.path.join(LOG_DIR, f'{today}.log'), encoding='utf-8')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger

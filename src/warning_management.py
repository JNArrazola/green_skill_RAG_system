"""  
Module for managing warnings and errors in the application.
Provides functions to log warnings, errors, and critical issues.
"""

import logging

logging.basicConfig(
    format="{asctime} - {levelname} - {message}",
    style="{",
    datefmt="%Y-%m-%d %H:%M",
)

logger = logging.getLogger(__name__)

def send_warning(message: str):
    logger.warning(message)

def send_error(message: str):
    logger.error(message)

def send_critical(message: str):
    logger.critical(message)
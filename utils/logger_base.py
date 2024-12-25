"""Base logger configuration without settings dependencies."""
import logging
from pathlib import Path
from typing import Optional

def get_base_logger(
    name: Optional[str] = None,
    log_level: str = 'INFO'
) -> logging.Logger:
    """Create a basic logger without settings dependencies."""
    logger = logging.getLogger(name) if name else logging.getLogger()
    
    # Set basic log level
    level = getattr(logging, log_level.upper(), logging.INFO)
    logger.setLevel(level)
    
    # Add console handler if none exists
    if not logger.handlers:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    return logger

# Create base logger
logger = get_base_logger(__name__)

import logging
import os
from pathlib import Path
from typing import Optional

def setup_logger(
    name: Optional[str] = None,
    log_level: Optional[str] = None,
    log_file: Optional[str] = None
) -> logging.Logger:
    """
    Configure and return a logger instance with the specified settings.
    
    Args:
        name: Logger name (defaults to root logger if None)
        log_level: Logging level (defaults to environment variable LOG_LEVEL or INFO)
        log_file: Optional file path for logging to file
        
    Returns:
        Configured logger instance
    """
    # Get logger
    logger = logging.getLogger(name) if name else logging.getLogger()
    
    # Set log level from environment or parameter
    level_name = log_level or os.getenv('LOG_LEVEL', 'INFO')
    level = getattr(logging, level_name.upper(), logging.INFO)
    logger.setLevel(level)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if log_file specified)
    if log_file:
        log_dir = Path(log_file).parent
        log_dir.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

# Create default logger
logger = setup_logger(
    name=__name__,
    log_file=os.path.join(
        Path(__file__).parent.parent,
        'logs',
        'app.log'
    ) if not os.getenv('DEV_MODE', 'false').lower() == 'true' else None
)

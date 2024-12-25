import logging
from pathlib import Path
from typing import Optional
from utils.logger_base import get_base_logger
from config.settings import LOG_LEVEL, DEV_MODE, PROJECT_ROOT

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
    # Get base logger with console handler
    logger = get_base_logger(name, log_level or LOG_LEVEL)
    
    # Add file handler if specified
    if log_file:
        log_dir = Path(log_file).parent
        log_dir.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )
        logger.addHandler(file_handler)
    
    return logger

# Create default logger
logger = setup_logger(
    name=__name__,
    log_file=PROJECT_ROOT / 'logs' / 'app.log' if not DEV_MODE else None
)

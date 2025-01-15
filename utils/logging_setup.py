import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path

def setup_logging():
    """Configure logging with both file and console handlers."""
    # Create logger first
    logger = logging.getLogger('utils.logging_setup')
    logger.setLevel(logging.DEBUG)
    
    # Skip if handlers already exist
    if logger.handlers:
        return logger
        
    # Create logs directory if it doesn't exist
    log_dir = Path(__file__).parent.parent / 'logs'
    log_dir.mkdir(exist_ok=True)
    
    # File handler
    file_handler = RotatingFileHandler(
        log_dir / 'app.log',
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setLevel(logging.DEBUG)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Add filters to reduce noise
    class VerboseFilter(logging.Filter):
        def filter(self, record):
            skip_patterns = [
                "Looking up time zone info",
                "Full xAI Request Payload",
                "Full xAI Response",
                "Found existing label",
                "CSV headers",
                "Loaded timezone data",
                "Replaced placeholder",
                "Raw segmentation response"
            ]
            return not any(pattern in str(record.msg) for pattern in skip_patterns)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s\n'
        'File: [%(pathname)s:%(lineno)d]\n'
    )
    
    # Set formatters
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add filters
    file_handler.addFilter(VerboseFilter())
    console_handler.addFilter(VerboseFilter())
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    # Set levels for other loggers
    logging.getLogger('utils.gmail_integration').setLevel(logging.INFO)
    logging.getLogger('utils.xai_integration').setLevel(logging.INFO)
    logging.getLogger('services.data_gatherer_service').setLevel(logging.INFO)
    
    return logger

# Create the logger instance
logger = setup_logging()

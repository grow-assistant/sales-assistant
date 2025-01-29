import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path
import sys
from typing import Optional

class CustomFormatter(logging.Formatter):
    """Custom formatter with colors for console output and detailed file output"""
    
    # Color codes
    grey = "\x1b[38;21m"
    blue = "\x1b[38;5;39m"
    yellow = "\x1b[38;5;226m"
    red = "\x1b[38;5;196m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"

    # Different formats for different logging levels
    FORMATS = {
        logging.DEBUG: grey + "%(asctime)s - %(levelname)s - %(message)s" + reset,
        logging.INFO: blue + "%(message)s" + reset,
        logging.WARNING: yellow + "%(levelname)s - %(message)s" + reset,
        logging.ERROR: red + "%(levelname)s - %(message)s\nFile: %(pathname)s:%(lineno)d" + reset,
        logging.CRITICAL: bold_red + "%(levelname)s - %(message)s\nFile: %(pathname)s:%(lineno)d" + reset,
    }

    def format(self, record):
        # Get the appropriate format for this level
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, datefmt='%Y-%m-%d %H:%M:%S')
        return formatter.format(record)

class SmartFilter(logging.Filter):
    """Enhanced filter with more sophisticated filtering rules"""
    
    NOISE_PATTERNS = {
        'DEBUG': [
            # Timezone related
            "Looking up time zone info",
            "Loaded timezone data",
            
            # Service initialization
            "Initializing CompanyEnrichmentService",
            "Initialized DataGathererService",
            
            # HTTP related
            "Starting new HTTPS connection",
            "Starting new HTTP connection",
            "https://api.hubapi.com",
            "http://www.oceanedge.com",
            "https://www.oceanedge.com",
            "https://oauth2.googleapis.com",
            "Making request: POST",
            "URL being requested",
            
            # API related
            "Full xAI Request Payload",
            "Full xAI Response",
            "Found existing label",
            "CSV headers",
            "Replaced placeholder",
            "Raw segmentation response",
            "Starting request",
            "Completed request",
            "Processing item",
            "Fetching list memberships",
            "Found records in list",
            
            # Google API related
            "googleapiclient.discovery",
            "google.auth.transport",
            "Making request: GET",
            "Making request: POST",
            
            # Connection pool
            "urllib3.connectionpool",
            "Starting new connection",
            "GET /crm/v3/objects",
            "POST /token"
        ],
        'INFO': [
            "routine check",
            "heartbeat",
            "periodic update",
            "Adding job tentatively",
            "Follow-up scheduler initialized",
            "file_cache is only supported",
            "googleapiclient.discovery"
        ],
        'WARNING': [
            "retrying request"  # Filter out retry warnings unless they exceed a threshold
        ]
    }
    
    # Add specific loggers to always filter
    FILTERED_LOGGERS = {
        'urllib3.connectionpool',
        'tzlocal',
        'apscheduler.scheduler',
        'google.auth.transport.requests',
        'googleapiclient.discovery',
        'googleapiclient.discovery_cache'
    }
    
    def __init__(self, name: str):
        super().__init__()
        self.name = name
        self.retry_counts = {}
    
    def filter(self, record) -> bool:
        # Check if the logger name is in our filtered list
        if record.name in self.FILTERED_LOGGERS:
            return False
            
        # Always allow ERROR and CRITICAL messages
        if record.levelno >= logging.ERROR:
            return True
            
        # Check against noise patterns for the specific level
        msg = str(record.msg).lower()
        level_name = record.levelname
        
        # Special handling for retry warnings
        if level_name == 'WARNING' and 'retry' in msg:
            identifier = f"{record.pathname}:{record.lineno}"
            self.retry_counts[identifier] = self.retry_counts.get(identifier, 0) + 1
            # Only show retry warnings after multiple occurrences
            return self.retry_counts[identifier] >= 3
        
        # Check against the noise patterns for this level
        patterns = self.NOISE_PATTERNS.get(level_name, [])
        return not any(pattern.lower() in msg.lower() for pattern in patterns)

    def __repr__(self):
        return f"SmartFilter(name={self.name})"

def setup_logging(log_name: str = 'app', 
                 console_level: str = 'INFO',
                 file_level: str = 'DEBUG',
                 max_bytes: int = 10485760,  # 10MB
                 backup_count: int = 5,
                 include_console: bool = True) -> logging.Logger:
    """
    Configure logging with enhanced formatting and filtering.
    
    Args:
        log_name (str): Name of the log file (without .log extension)
        console_level (str): Logging level for console output
        file_level (str): Logging level for file output
        max_bytes (int): Maximum size of log file before rotation
        backup_count (int): Number of backup files to keep
        include_console (bool): Whether to include console output
    """
    # Create logger
    logger = logging.getLogger(f'utils.logging_setup.{log_name}')
    logger.setLevel(logging.DEBUG)
    
    # Skip if handlers already exist
    if logger.handlers:
        return logger
        
    # Create logs directory
    log_dir = Path(__file__).parent.parent / 'logs'
    log_dir.mkdir(exist_ok=True)
    
    # Handle UTF-8 encoding more safely
    if sys.platform == 'win32':
        try:
            # Only try to set UTF-8 if we're in a terminal
            if hasattr(sys.stdout, 'buffer'):
                import codecs
                sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
                sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')
        except Exception as e:
            # If UTF-8 encoding fails, just log it and continue
            print(f"Warning: Could not set UTF-8 encoding: {e}")

    # Set levels for other loggers
    logging.getLogger('urllib3.connectionpool').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('googleapiclient.discovery').setLevel(logging.WARNING)
    logging.getLogger('google.auth.transport.requests').setLevel(logging.WARNING)
    logging.getLogger('tzlocal').setLevel(logging.WARNING)
    logging.getLogger('apscheduler.scheduler').setLevel(logging.WARNING)
    logging.getLogger('googleapiclient.discovery_cache').setLevel(logging.WARNING)

    # File handler with rotation
    file_handler = RotatingFileHandler(
        f'logs/{log_name}.log',
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding='utf-8'
    )
    file_handler.setLevel(getattr(logging, file_level.upper()))
    
    # Detailed formatter for file output
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s\n'
        'File: [%(pathname)s:%(lineno)d]\n'
        'Function: [%(funcName)s]\n'
        '%(threadName)s - %(process)d\n'
    )
    file_handler.setFormatter(file_formatter)
    
    # Add smart filter
    smart_filter = SmartFilter(log_name)
    file_handler.addFilter(smart_filter)
    logger.addHandler(file_handler)
    
    # Console handler (optional)
    if include_console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, console_level.upper()))
        console_handler.setFormatter(CustomFormatter())
        console_handler.addFilter(smart_filter)
        logger.addHandler(console_handler)
    
    return logger

# Create the default logger instance
logger = setup_logging()

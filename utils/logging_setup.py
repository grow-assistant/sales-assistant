import logging
import sys
from logging.handlers import RotatingFileHandler
from typing import Optional
from contextlib import contextmanager

class StepLogger(logging.Logger):
    def step_complete(self, step_number: int, message: str):
        self.info(f"✓ Step {step_number}: {message}")
        
    def step_start(self, step_number: int, message: str):
        self.debug(f"Starting Step {step_number}: {message}")

def setup_logging():
    # Register custom logger class
    logging.setLoggerClass(StepLogger)
    
    # Clear existing handlers
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    
    # Configure formatters
    console_formatter = logging.Formatter('%(message)s')
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        '[%(pathname)s:%(lineno)d]'
    )
    
    # Console handler (INFO and above)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(logging.INFO)
    
    # File handler with rotation
    file_handler = RotatingFileHandler(
        'logs/app.log',
        maxBytes=1024*1024,
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(logging.DEBUG)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    
    # Suppress external library logs
    for logger_name in ['urllib3', 'googleapiclient', 'google.auth', 'openai']:
        logging.getLogger(logger_name).setLevel(logging.WARNING)
    
    return logging.getLogger(__name__)

logger = setup_logging()

@contextmanager
def workflow_step(step_number: int, description: str, logger=logger):
    """Context manager for workflow steps with proper logging"""
    logger.step_start(step_number, description)
    try:
        yield
        logger.step_complete(step_number, description)
    except Exception as e:
        logger.error(f"Step {step_number} failed: {description}", 
                    extra={"error": str(e)}, 
                    exc_info=True)
        raise

# Make it available for import
__all__ = ['logger', 'workflow_step']

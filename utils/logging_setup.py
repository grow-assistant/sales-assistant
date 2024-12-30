import logging
import sys
from logging.handlers import RotatingFileHandler
from typing import Optional
from contextlib import contextmanager
from config.settings import DEBUG_MODE

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

def workflow_step(step_num: int, description: str):
    """Context manager for workflow steps."""
    class WorkflowStep:
        def __enter__(self):
            logger.debug(f"Starting Step {step_num}: {description}")
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            if not exc_type:
                logger.info(f"✓ Step {step_num}: {description}")
            return False

    return WorkflowStep()

# Make it available for import
__all__ = ['logger', 'workflow_step']

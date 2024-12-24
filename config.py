from dotenv import load_dotenv
from utils.logging_setup import logger

# Load environment variables once
load_dotenv()

# Import all settings from centralized config
from config.settings import (
    # API Keys
    OPENAI_API_KEY,
    HUBSPOT_API_KEY,
    
    # Application Settings
    MAX_ITERATIONS,
    SUMMARY_INTERVAL,
    DEBUG_MODE,
    FORCE_SQL_UPDATE,
    
    # Model Configuration
    MODEL_FOR_EMAILS,
    MODEL_FOR_GENERAL,
    MODEL_FOR_ANALYSIS,
    DEFAULT_TEMPERATURE
)

# Log configuration status
logger.info("Configuration loaded successfully")
if DEBUG_MODE:
    logger.debug("Debug mode enabled")
if FORCE_SQL_UPDATE:
    logger.info("SQL update forced by environment variable")

# Export all imported settings
__all__ = [
    'OPENAI_API_KEY',
    'HUBSPOT_API_KEY',
    'MAX_ITERATIONS', 
    'SUMMARY_INTERVAL',
    'DEBUG_MODE',
    'FORCE_SQL_UPDATE',
    'MODEL_FOR_EMAILS',
    'MODEL_FOR_GENERAL',
    'MODEL_FOR_ANALYSIS',
    'DEFAULT_TEMPERATURE'
]

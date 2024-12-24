# config/settings.py
import os
from typing import Any, Optional
from dotenv import load_dotenv
from utils.logging_setup import logger
from pathlib import Path

# Load environment variables
load_dotenv()

class EnvironmentVariableError(Exception):
    """Custom exception for environment variable related errors."""
    pass

def get_env_var(
    var_name: str, 
    required: bool = True, 
    default: Any = None,
    var_type: type = str
) -> Any:
    """
    Retrieve and validate environment variables with type checking.
    
    Args:
        var_name: Name of the environment variable
        required: Whether the variable is required
        default: Default value if not required and not set
        var_type: Expected type of the variable
    
    Returns:
        The environment variable value cast to the specified type
    """
    try:
        value = os.getenv(var_name)
        
        if value is None or value.strip() == "":
            if required:
                logger.warning(f"Missing required environment variable: {var_name}")
                return default
            return default
            
        # Type conversion
        if var_type == bool:
            return value.lower() in ('true', '1', 'yes', 'y')
        return var_type(value)
        
    except ValueError as e:
        msg = f"Error converting environment variable {var_name} to type {var_type.__name__}: {str(e)}"
        logger.error(msg)
        return default
    except Exception as e:
        msg = f"Error accessing environment variable {var_name}: {str(e)}"
        logger.error(msg)
        return default

# Determine if we're in development mode
DEV_MODE = get_env_var("DEV_MODE", required=False, default=True, var_type=bool)

# API Keys and Authentication - Make HUBSPOT_API_KEY optional in dev mode
OPENAI_API_KEY = get_env_var("OPENAI_API_KEY", required=not DEV_MODE, default="sk-dummy")
HUBSPOT_API_KEY = get_env_var("HUBSPOT_API_KEY", required=not DEV_MODE, default="pat-dummy")

# OpenAI Model Configuration
MODEL_FOR_EMAILS = get_env_var("MODEL_FOR_EMAILS", default="gpt-4o-mini")
MODEL_FOR_GENERAL = get_env_var("MODEL_FOR_GENERAL", default="gpt-4o-mini")
MODEL_FOR_ANALYSIS = get_env_var("MODEL_FOR_ANALYSIS", default="gpt-4o-mini")
DEFAULT_TEMPERATURE = get_env_var("DEFAULT_TEMPERATURE", default=0.7, var_type=float)

# Additional Temperature Config
EMAIL_TEMPERATURE = 0.7
ANALYSIS_TEMPERATURE = 0.5

# Application Settings
MAX_ITERATIONS = get_env_var("MAX_ITERATIONS", default=20, var_type=int)
SUMMARY_INTERVAL = get_env_var("SUMMARY_INTERVAL", default=5, var_type=int)
DEBUG_MODE = get_env_var("DEBUG_MODE", default=False, var_type=bool)

# API Headers
HEADERS = {
    "Authorization": f"Bearer {HUBSPOT_API_KEY}",
    "Content-Type": "application/json"
}

# Define missing constants referenced in config/validate.py
API_KEYS = {}
API_ENDPOINTS = {}
PROJECT_ROOT = Path(__file__).parent.parent

# Google Search API Keys (optional)
GOOGLE_SEARCH_API_KEY = get_env_var("GOOGLE_SEARCH_API_KEY", required=False, default="")
GOOGLE_CSE_ID = get_env_var("GOOGLE_CSE_ID", required=False, default="")
ENABLE_GOOGLE_SEARCH = get_env_var("ENABLE_GOOGLE_SEARCH", required=False, default=True, var_type=bool)

# SQL Update Settings
FORCE_SQL_UPDATE = os.getenv('FORCE_SQL_UPDATE', 'False').lower() == 'true'

# Export all settings
__all__ = [
    'DEV_MODE',
    'OPENAI_API_KEY',
    'HUBSPOT_API_KEY',
    'MODEL_FOR_EMAILS',
    'MODEL_FOR_GENERAL',
    'MODEL_FOR_ANALYSIS',
    'DEFAULT_TEMPERATURE',
    'EMAIL_TEMPERATURE',
    'ANALYSIS_TEMPERATURE',
    'MAX_ITERATIONS',
    'SUMMARY_INTERVAL',
    'DEBUG_MODE',
    'HEADERS',
    'API_KEYS',
    'API_ENDPOINTS',
    'PROJECT_ROOT',
    'GOOGLE_SEARCH_API_KEY',
    'GOOGLE_CSE_ID',
    'ENABLE_GOOGLE_SEARCH',
    'FORCE_SQL_UPDATE'
]

# Log configuration status
if DEV_MODE:
    logger.warning("Running in DEVELOPMENT mode with dummy API keys")

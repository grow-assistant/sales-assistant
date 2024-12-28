# config/settings.py
import os
from typing import Any, Optional, Dict
from dotenv import load_dotenv
from utils.logger_base import logger
from pathlib import Path

# Load environment variables
load_dotenv()

# Project Root - this should be at the top level of your settings
PROJECT_ROOT = Path(__file__).parent.parent

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
DEBUG_MODE = get_env_var("DEBUG_MODE", default=False, var_type=bool)

# API Keys and Authentication
OPENAI_API_KEY = get_env_var("OPENAI_API_KEY", required=not DEV_MODE, default="sk-dummy")
HUBSPOT_API_KEY = get_env_var("HUBSPOT_API_KEY", required=not DEV_MODE, default="pat-dummy")
GOOGLE_SEARCH_API_KEY = get_env_var("GOOGLE_SEARCH_API_KEY", required=False, default="")
GOOGLE_CSE_ID = get_env_var("GOOGLE_CSE_ID", required=False, default="")
XAI_TOKEN = get_env_var("XAI_TOKEN", required=False, default="")
MARKET_RESEARCH_API = get_env_var("MARKET_RESEARCH_API", required=False, default="")

# Database Configuration
DB_SERVER = get_env_var("DB_SERVER", required=False, default="localhost")
DB_NAME = get_env_var("DB_NAME", required=False, default="swoop")
DB_USER = get_env_var("DB_USER", required=False, default="postgres")
DB_PASSWORD = get_env_var("DB_PASSWORD", required=False, default="")

# Model Configuration
MODEL_FOR_EMAILS = get_env_var("MODEL_FOR_EMAILS", default="gpt-4")
MODEL_FOR_GENERAL = get_env_var("MODEL_FOR_GENERAL", default="gpt-4")
MODEL_FOR_ANALYSIS = get_env_var("MODEL_FOR_ANALYSIS", default="gpt-4")
XAI_MODEL = get_env_var("XAI_MODEL", default="xai-latest")
XAI_API_URL = get_env_var("XAI_API_URL", default="https://api.xai.com/v1")

# Temperature Settings
DEFAULT_TEMPERATURE = get_env_var("DEFAULT_TEMPERATURE", default=0.7, var_type=float)
EMAIL_TEMPERATURE = get_env_var("EMAIL_TEMPERATURE", default=0.7, var_type=float)
ANALYSIS_TEMPERATURE = get_env_var("ANALYSIS_TEMPERATURE", default=0.5, var_type=float)

# Application Settings
MAX_ITERATIONS = get_env_var("MAX_ITERATIONS", default=20, var_type=int)
MAX_RETRIES = get_env_var("MAX_RETRIES", default=3, var_type=int)
RETRY_DELAY = get_env_var("RETRY_DELAY", default=1, var_type=int)
SUMMARY_INTERVAL = get_env_var("SUMMARY_INTERVAL", default=5, var_type=int)
ENABLE_COMPETITOR_ANALYSIS = get_env_var("ENABLE_COMPETITOR_ANALYSIS", default=True, var_type=bool)
ENABLE_GOOGLE_SEARCH = get_env_var("ENABLE_GOOGLE_SEARCH", default=True, var_type=bool)
FORCE_SQL_UPDATE = get_env_var("FORCE_SQL_UPDATE", default=False, var_type=bool)

# Cache Configuration
CACHE_ENABLED = get_env_var("CACHE_ENABLED", default=True, var_type=bool)
CACHE_TTL = get_env_var("CACHE_TTL", default=3600, var_type=int)  # 1 hour default

# Logging Configuration
LOG_LEVEL = get_env_var("LOG_LEVEL", default="INFO")

# API Headers
HEADERS: Dict[str, str] = {
    "Authorization": f"Bearer {HUBSPOT_API_KEY}",
    "Content-Type": "application/json"
}

# API Keys and Endpoints (for future use)
API_KEYS: Dict[str, str] = {
    "openai": OPENAI_API_KEY,
    "hubspot": HUBSPOT_API_KEY,
    "google_search": GOOGLE_SEARCH_API_KEY,
    "xai": XAI_TOKEN
}

API_ENDPOINTS: Dict[str, str] = {
    "xai": XAI_API_URL,
    "hubspot": "https://api.hubapi.com"
}

# Export all settings
__all__ = [
    'PROJECT_ROOT',  # Add PROJECT_ROOT to exports
    'DEV_MODE',
    'DEBUG_MODE',
    'OPENAI_API_KEY',
    'HUBSPOT_API_KEY',
    'GOOGLE_SEARCH_API_KEY',
    'GOOGLE_CSE_ID',
    'XAI_TOKEN',
    'DB_SERVER',
    'DB_NAME',
    'DB_USER',
    'DB_PASSWORD',
    'MODEL_FOR_EMAILS',
    'MODEL_FOR_GENERAL',
    'MODEL_FOR_ANALYSIS',
    'XAI_MODEL',
    'XAI_API_URL',
    'DEFAULT_TEMPERATURE',
    'EMAIL_TEMPERATURE',
    'ANALYSIS_TEMPERATURE',
    'MAX_ITERATIONS',
    'MAX_RETRIES',
    'RETRY_DELAY',
    'SUMMARY_INTERVAL',
    'ENABLE_COMPETITOR_ANALYSIS',
    'ENABLE_GOOGLE_SEARCH',
    'FORCE_SQL_UPDATE',
    'CACHE_ENABLED',
    'CACHE_TTL',
    'LOG_LEVEL',
    'HEADERS',
    'API_KEYS',
    'API_ENDPOINTS',
    'MARKET_RESEARCH_API'
]

# Log configuration status
if DEV_MODE:
    logger.warning("Running in DEVELOPMENT mode with dummy API keys")

# Add debug logging for PROJECT_ROOT
if DEBUG_MODE:
    logger.debug(f"PROJECT_ROOT set to: {PROJECT_ROOT}")

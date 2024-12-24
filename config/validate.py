"""
Configuration validation utilities.
"""
import os
from typing import List, Dict, Optional
from pathlib import Path
from .settings import API_KEYS, API_ENDPOINTS, PROJECT_ROOT

def validate_api_keys() -> List[str]:
    """Validate that all required API keys are set.
    
    Returns:
        List of missing API key names
    """
    missing_keys = []
    for key_name, value in API_KEYS.items():
        if not value:
            missing_keys.append(key_name)
    return missing_keys

def validate_api_endpoints() -> List[str]:
    """Validate that all required API endpoints are set.
    
    Returns:
        List of missing endpoint names
    """
    missing_endpoints = []
    for endpoint_name, value in API_ENDPOINTS.items():
        if not value:
            missing_endpoints.append(endpoint_name)
    return missing_endpoints

def validate_project_structure() -> Dict[str, bool]:
    """Validate that required project directories exist.
    
    Returns:
        Dictionary mapping directory names to existence status
    """
    required_dirs = ['docs', 'data', 'agents', 'utils']
    return {
        dir_name: (PROJECT_ROOT / dir_name).exists()
        for dir_name in required_dirs
    }

def check_configuration() -> Optional[str]:
    """Check the entire configuration for validity.
    
    Returns:
        Error message if configuration is invalid, None otherwise
    """
    # Check API keys
    missing_keys = validate_api_keys()
    if missing_keys:
        return f"Missing API keys: {', '.join(missing_keys)}"
    
    # Check API endpoints
    missing_endpoints = validate_api_endpoints()
    if missing_endpoints:
        return f"Missing API endpoints: {', '.join(missing_endpoints)}"
    
    # Check project structure
    structure = validate_project_structure()
    missing_dirs = [dir_name for dir_name, exists in structure.items() if not exists]
    if missing_dirs:
        return f"Missing directories: {', '.join(missing_dirs)}"
    
    return None 
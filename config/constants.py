"""
Static configuration values that don't change during runtime.
"""
from typing import Dict, Any

# Request retry configuration
RETRY_CONFIG = {
    'max_attempts': 3,
    'initial_delay': 1,
    'max_delay': 10,
    'backoff_factor': 2,
}

# API rate limits
RATE_LIMITS = {
    'hubspot': {
        'requests_per_second': 10,
        'burst_limit': 100,
    },
    'openai': {
        'requests_per_minute': 60,
        'tokens_per_minute': 90000,
    }
}

# Document categories
DOC_CATEGORIES = {
    'brand': ['guidelines', 'voice', 'messaging'],
    'case_studies': ['success_stories', 'testimonials'],
    'templates': ['email', 'objection_handling'],
}

# Default values
DEFAULTS: Dict[str, Any] = {
    'batch_size': 5,
    'retry_count': 3,
    'timeout': 30,
    'MAX_ITERATIONS': 20,      # Added
    'SUMMARY_INTERVAL': 5      # Added
}
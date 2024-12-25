"""
Custom exceptions for the Swoop Golf application.
"""

class SwoopError(Exception):
    """Base exception class for Swoop Golf application."""
    def __init__(self, message: str, details: dict = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)

class LeadContextError(SwoopError):
    """Raised when there's an error preparing lead context."""
    pass

class HubSpotError(SwoopError):
    """Raised when there's an error interacting with HubSpot API."""
    pass

class EmailGenerationError(SwoopError):
    """Raised when there's an error generating email content."""
    pass

class ConfigurationError(SwoopError):
    """Raised when there's a configuration-related error."""
    pass

class ExternalAPIError(SwoopError):
    """Raised when there's an error with external API calls."""
    pass

class OpenAIError(SwoopError):
    """Raised when there's an error with OpenAI API calls."""
    pass

"""
Test standardized exception handling.
"""
import pytest
from utils.exceptions import SwoopError, LeadContextError
from services.hubspot_service import HubspotService

def test_swoop_error_base():
    """Test that SwoopError properly stores details."""
    details = {"error_type": "TestError", "status_code": 404}
    error = SwoopError("Test error", details=details)
    
    assert error.message == "Test error"
    assert error.details == details
    assert str(error) == "Test error"

def test_lead_context_error():
    """Test that LeadContextError properly inherits from SwoopError."""
    details = {"email": "test@example.com"}
    error = LeadContextError("Invalid lead data", details=details)
    
    assert isinstance(error, SwoopError)
    assert error.message == "Invalid lead data"
    assert error.details == details

if __name__ == "__main__":
    pytest.main([__file__, "-v"])

# agents/context_preparer.py

from typing import Dict, Any
from services.leads_service import LeadsService

# Initialize the leads service
_leads_service = LeadsService()

def prepare_lead_context(lead_email: str) -> Dict[str, Any]:
    """
    Prepare comprehensive lead context including HubSpot data,
    company information, and external research.
    
    This function now delegates to the LeadsService while maintaining
    the same interface for backward compatibility.
    """
    return _leads_service.prepare_lead_context(lead_email)

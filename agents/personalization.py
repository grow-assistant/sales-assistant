"""
Module for lead personalization functionality.
Note: Core implementation has been moved to LeadsService.
This module maintains the original interface for backward compatibility.
"""

from typing import Dict, Any
from services.leads_service import LeadsService

# Initialize the leads service
_leads_service = LeadsService()


def generate_lead_summary(lead_email: str) -> Dict[str, Any]:
    """
    Gather and summarize lead's info according to the personalization steps.
    
    This function delegates to the LeadsService while maintaining
    the same interface for backward compatibility.
    
    Args:
        lead_email: Email address of the lead
        
    Returns:
        Dict containing lead summary information including:
        - lead_summary: List of bullet points about the lead
        - club_context: List of bullet points about the club
        - subject: Email subject
        - body: Email body
    """
    return _leads_service.generate_lead_summary(lead_email)

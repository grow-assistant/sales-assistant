"""
services/orchestrator_service.py

Service for managing orchestration-related operations including:
- Lead data fetching
- Previous interaction review
- Market research
- Competitor analysis
"""

from typing import Dict, Any

from services.leads_service import LeadsService
from services.hubspot_service import HubspotService
from external.external_api import (
    market_research,
    review_previous_interactions
)
from utils.logging_setup import logger
from utils.exceptions import LeadContextError, HubSpotError


class OrchestratorService:
    """Service for managing orchestration-related operations."""

    def __init__(self, leads_service: LeadsService, hubspot_service: HubspotService):
        """Initialize the orchestrator service."""
        self.leads_service = leads_service
        self.hubspot_service = hubspot_service
        self.logger = logger

    def get_lead_data(self, contact_id: str) -> Dict[str, Any]:
        """Get lead data from HubSpot."""
        return self.hubspot_service.get_lead_data_from_hubspot(contact_id)

    def review_interactions(self, contact_id: str) -> Dict[str, Any]:
        """Review previous interactions for a lead."""
        try:
            return review_previous_interactions(contact_id)
        except Exception as e:
            raise LeadContextError(f"Error reviewing interactions for {contact_id}: {e}")

    def analyze_competitors(self, company_name: str) -> Dict[str, Any]:
        """Analyze competitors for a company."""
        try:
            return market_research(company_name)
        except Exception as e:
            raise LeadContextError(f"Error analyzing competitors for {company_name}: {e}")

    def personalize_message(self, lead_data: Dict[str, Any]) -> Dict[str, Any]:
        """Personalize a message for a lead."""
        lead_email = lead_data.get("email")
        if not lead_email:
            raise LeadContextError("No email found in lead data")
        return self.leads_service.generate_lead_summary(lead_email)

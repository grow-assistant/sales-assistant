"""
services/leads_service.py

Handles lead-related business logic, including generating lead summaries
(not the full data gathering, which now lives in DataGathererService).
"""

from typing import Dict, Any, Optional
import json
from datetime import datetime
from pathlib import Path

from config.settings import PROJECT_ROOT, DEBUG_MODE
from config.settings import HUBSPOT_API_KEY
from services.hubspot_service import HubspotService
from external.link_summarizer import summarize_recent_news
from external.external_api import market_research, determine_club_season
from hubspot_integration.data_enrichment import check_competitor_on_website
from utils.logging_setup import logger
from utils.doc_reader import read_doc


class LeadContextError(Exception):
    """Custom exception for lead context preparation errors."""
    pass


class LeadsService:
    """
    Responsible for higher-level lead operations such as generating
    specialized summaries or reading certain docs for personalization.
    NOTE: The actual data gathering is now centralized in DataGathererService.
    """
    
    def __init__(self):
        self._hubspot = HubspotService(api_key=HUBSPOT_API_KEY)

    def generate_lead_summary(self, lead_email: str) -> Dict[str, Any]:
        """
        Generate a short summary for personalization (subject/body).
        This does NOT fetch all lead data; for full data, use DataGathererService.
        """
        contact_id = self._hubspot.get_contact_by_email(lead_email)
        if not contact_id:
            logger.warning("No contact found for given email in generate_lead_summary().")
            return {}

        props = self._hubspot.get_contact_properties(contact_id)
        notes = self._hubspot.get_all_notes_for_contact(contact_id)
        company_id = self._hubspot.get_associated_company_id(contact_id)
        company_data = self._hubspot.get_company_data(company_id) if company_id else {}

        job_title = (props.get("jobtitle", "") or "").strip()
        filename_job_title = (
            job_title.lower()
            .replace("&", "and")
            .replace("/", "_")
            .replace(" ", "_")
            .replace(",", "")
        )

        # Example: fetch template if it exists
        template_path = f"docs/templates/{filename_job_title}_initial_outreach.md"
        try:
            template_content = read_doc(template_path)
            subject = template_content.get("subject", "Default Subject")
            body = template_content.get("body", "Default Body")
        except Exception as e:
            logger.warning(
                f"Could not read document '{template_path}', using fallback content. "
                f"Reason: {str(e)}"
            )
            subject = "Fallback Subject"
            body = "Fallback Body..."

        # Basic company/season data as an example
        city = company_data.get("city", "")
        state = company_data.get("state", "")
        season_data = determine_club_season(city, state)

        # Try a quick market research call if needed
        company_name = company_data.get("name", "")
        external_insights = market_research(company_name) if company_name else {"recent_news": []}
        industry_trends = external_insights.get("recent_news", [])

        top_trend_title = (
            industry_trends[0]["title"]
            if (industry_trends and "title" in industry_trends[0])
            else "N/A"
        )

        # Basic summary example
        lead_care_about = [f"- Potential interest in: {top_trend_title}"]
        club_context = [
            f"- Club Name: {company_name or 'N/A'}",
            f"- Location: {city}, {state}",
            f"- Peak Season: {season_data.get('peak_season_start', 'Unknown')} to {season_data.get('peak_season_end', 'Unknown')}"
        ]

        return {
            "lead_summary": lead_care_about,
            "club_context": club_context,
            "subject": subject,
            "body": body
        }

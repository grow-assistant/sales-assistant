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
    
    def __init__(self, data_gatherer_service=None):
        """Initialize LeadsService with optional DataGathererService."""
        if data_gatherer_service is None:
            from services.data_gatherer_service import DataGathererService
            data_gatherer_service = DataGathererService()
        self._data_gatherer = data_gatherer_service

    def generate_lead_summary(self, lead_email: str) -> Dict[str, Any]:
        """
        Generate a short summary for personalization (subject/body).
        Uses DataGathererService to fetch required data.
        """
        try:
            # Get all lead data from DataGathererService
            lead_sheet = self._data_gatherer.gather_lead_data(lead_email)
            if not lead_sheet:
                logger.warning("No lead data found for given email in generate_lead_summary().")
                return {}

            lead_data = lead_sheet.get("lead_data", {})
            company_data = lead_data.get("company_data", {})
            analysis = lead_sheet.get("analysis", {})

            # Extract job title for template selection
            job_title = (lead_data.get("jobtitle", "") or "").strip()
            filename_job_title = (
                job_title.lower()
                .replace("&", "and")
                .replace("/", "_")
                .replace(" ", "_")
                .replace(",", "")
            )

            # Fetch template if it exists
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

            # Get season data from analysis
            season_data = analysis.get("season_data", {})
            research_data = analysis.get("research_data", {})
            industry_trends = research_data.get("recent_news", [])

            top_trend_title = (
                industry_trends[0]["title"]
                if (industry_trends and "title" in industry_trends[0])
                else "N/A"
            )

            # Build summary using gathered data
            lead_care_about = [f"- Potential interest in: {top_trend_title}"]
            club_context = [
                f"- Club Name: {company_data.get('name', 'N/A')}",
                f"- Location: {company_data.get('city', '')}, {company_data.get('state', '')}",
                f"- Peak Season: {season_data.get('peak_season_start', 'Unknown')} to {season_data.get('peak_season_end', 'Unknown')}"
            ]

            return {
                "subject": subject,
                "body": body,
                "lead_care_about": lead_care_about,
                "club_context": club_context
            }

        except Exception as e:
            logger.error(f"Error generating lead summary: {str(e)}")
            return {
                "subject": "Default Subject",
                "body": "Default Body",
                "lead_care_about": [],
                "club_context": []
            }

        return {
            "lead_summary": lead_care_about,
            "club_context": club_context,
            "subject": subject,
            "body": body
        }

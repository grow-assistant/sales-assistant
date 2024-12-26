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
from services.data_gatherer_service import DataGathererService
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
    
    def __init__(self, data_gatherer_service: DataGathererService):
        """
        Initialize LeadsService.
        
        Args:
            data_gatherer_service: Service for gathering lead data
        """
        self.data_gatherer = data_gatherer_service

    def prepare_lead_context(self, lead_email: str) -> Dict[str, Any]:
        """
        Prepare lead context for personalization (subject/body).
        Uses DataGathererService for data retrieval.
        """
        # Get comprehensive lead data from DataGathererService
        lead_sheet = self.data_gatherer.gather_lead_data(lead_email)
        if not lead_sheet: 
            logger.warning(
                "No lead data found for lead summary generation",
                extra={
                    "email_domain": lead_email.split('@')[1] if '@' in lead_email else 'unknown'
                }
            )
            return {}

        # Extract relevant data
        lead_data = lead_sheet.get("lead_data", {})
        company_data = lead_data.get("company_data", {})
        analysis = lead_sheet.get("analysis", {})
        
        # Get job title for template selection
        props = lead_data.get("properties", {})
        job_title = (props.get("jobtitle", "") or "").strip()
        filename_job_title = (
            job_title.lower()
            .replace("&", "and")
            .replace("/", "_")
            .replace(" ", "_")
            .replace(",", "")
        )

        # Get template content
        template_path = f"docs/templates/{filename_job_title}_initial_outreach.md"
        try:
            template_content = read_doc(template_path)
            subject = template_content.get("subject", "Default Subject")
            body = template_content.get("body", "Default Body")
        except Exception as e:
            logger.warning(
                "Template read failed, using fallback content",
                extra={
                    "template": template_path,
                    "error_type": type(e).__name__,
                    "error": str(e)
                }
            )
            subject = "Fallback Subject"
            body = "Fallback Body..."

        # Extract season and research data from analysis
        season_data = analysis.get("season_data", {})
        research_data = analysis.get("research_data", {})
        industry_trends = research_data.get("recent_news", [])
        
        top_trend_title = (
            industry_trends[0]["title"]
            if (industry_trends and "title" in industry_trends[0])
            else "N/A"
        )

        # Build summary
        lead_care_about = [f"- Potential interest in: {top_trend_title}"]
        club_context = [
            f"- Club Name: {company_data.get('name', 'N/A')}",
            f"- Location: {company_data.get('city', 'N/A')}, {company_data.get('state', 'N/A')}",
            f"- Peak Season: {season_data.get('peak_season_start', 'Unknown')} to {season_data.get('peak_season_end', 'Unknown')}"
        ]

        # Add metadata about the lead context generation
        metadata = {
            "status": "success",
            "timestamp": datetime.utcnow().isoformat(),
            "email": lead_email,
            "job_title": job_title
        }

        return {
            "metadata": metadata,
            "lead_summary": lead_care_about,
            "club_context": club_context,
            "subject": subject,
            "body": body
        }

"""
services/leads_service.py

Handles lead-related business logic, including data fetching, enrichment,
and preparation of lead context for AI personalization.
"""

from typing import Dict, Any, Optional, TYPE_CHECKING
import json
from datetime import datetime
from pathlib import Path

from config.settings import PROJECT_ROOT, DEBUG_MODE
from config.settings import HUBSPOT_API_KEY
from services.hubspot_service import HubspotService

# Initialize HubSpot service
_hubspot = HubspotService(api_key=HUBSPOT_API_KEY)
from external.external_api import determine_club_season
from hubspot_integration.data_enrichment import check_competitor_on_website
from utils.logging_setup import logger
from utils.doc_reader import read_doc
from external.link_summarizer import summarize_recent_news

if TYPE_CHECKING:
    from services.orchestrator_service import OrchestratorService


class LeadContextError(Exception):
    """Custom exception for lead context preparation errors."""
    pass


class LeadsService:
    """
    Service class for managing lead-related operations including:
    - Lead context preparation
    - Lead data enrichment
    - Lead summary generation
    """
    
    def __init__(self, orchestrator: 'OrchestratorService' = None):
        """Initialize the leads service."""
        self.orchestrator = orchestrator
    
    @staticmethod
    def get_domain_from_email(email: str) -> Optional[str]:
        """Extract domain from email address."""
        if "@" in email:
            return email.split("@", 1)[1].lower().strip()
        return None

    @staticmethod
    def create_context_directory() -> Path:
        """Create and return the directory for storing lead contexts."""
        context_dir = PROJECT_ROOT / "test_data" / "lead_contexts"
        context_dir.mkdir(parents=True, exist_ok=True)
        return context_dir

    @staticmethod
    def generate_context_filename(contact_id: str) -> str:
        """Generate a unique filename for storing lead context."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"lead_context_{contact_id}_{timestamp}.json"

    def _save_lead_context(self, lead_sheet: Dict[str, Any], contact_id: str) -> None:
        """Save the lead context to disk for debugging/inspection."""
        try:
            context_dir = self.create_context_directory()
            filename = self.generate_context_filename(contact_id)
            file_path = context_dir / filename
            with file_path.open("w", encoding="utf-8") as f:
                json.dump(lead_sheet, f, indent=2, ensure_ascii=False)
            logger.info(f"Lead context saved for testing: {file_path.resolve()}")
        except Exception as e:
            logger.warning(f"Failed to save lead context (non-critical): {e}")

    def prepare_lead_context(self, email: str) -> Dict[str, Any]:
        """
        Prepare comprehensive lead context including HubSpot data,
        company information, and interactions.
        """
        # Initialize interactions list
        interactions = []


        # Get lead data from HubSpot
        try:
            lead_data = _hubspot.get_lead_data_from_hubspot(email)
            if not lead_data:
                raise LeadContextError(f"No lead found for email: {email}")

            contact_id = _hubspot.get_contact_by_email(email)
            if contact_id:
                # Get associated company data
                company_id = _hubspot.get_associated_company_id(contact_id)
                company_data = _hubspot.get_company_data(company_id) if company_id else {}
                lead_data["company_data"] = company_data

                # Get interactions/notes
                interactions = _hubspot.get_all_notes_for_contact(contact_id)

                # Get additional lead emails
                try:
                    lead_emails = _hubspot.get_all_emails_for_contact(contact_id)
                    lead_data["emails"] = lead_emails
                except Exception as e:
                    logger.error(f"Error fetching emails for contact {contact_id}: {e}")
                    lead_data["emails"] = []

                # Get season info if company data exists
                try:
                    season_info = determine_club_season(
                        company_data.get("city", ""),
                        company_data.get("state", "")
                    )
                except Exception as e:
                    logger.warning(f"Error determining season info for contact {contact_id}: {e}")
                    season_info = {"peak_season_start": "Unknown", "peak_season_end": "Unknown"}
                
                lead_data["season_info"] = season_info

            return {
                "metadata": {"status": "success"},
                "lead_data": lead_data,
                "interactions": interactions
            }

        except Exception as e:
            logger.error(f"Error preparing lead context: {e}")
            return {
                "metadata": {"status": "error", "message": str(e)},
                "lead_data": {},
                "interactions": []
            }

    def generate_lead_summary(self, lead_email: str) -> Dict[str, Any]:
        """
        Generate a summary of lead information for personalization.
        
        This maintains the same interface as the original function
        in personalization.py for backward compatibility.
        """
        contact_id = _hubspot.get_contact_by_email(lead_email)
        if not contact_id:
            logger.warning("No contact found for given email.")
            return {}

        props = _hubspot.get_contact_properties(contact_id)
        notes = _hubspot.get_all_notes_for_contact(contact_id)
        company_id = _hubspot.get_associated_company_id(contact_id)
        company_data = _hubspot.get_company_data(company_id) if company_id else {}

        job_title = props.get("jobtitle", "").strip()
        filename_job_title = (
            job_title.lower()
            .replace("&", "and")
            .replace("/", "_")
            .replace(" ", "_")
            .replace(",", "")
        )

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

        club_type = company_data.get("club_type")
        competitor_presence = company_data.get("competitor_presence")
        peak_season_start = company_data.get("peak_season_start")
        peak_season_end = company_data.get("peak_season_end")

        known_pain_points = props.get("known_pain_points")
        engagement_focus = props.get("engagement_focus")
        last_personalization_update = props.get("last_personalization_update_date")

        if not club_type:
            club_type = self.infer_club_type(company_data)

        city = company_data.get("city", "")
        state = company_data.get("state", "")
        season_data = determine_club_season(city, state)
        if peak_season_start:
            season_data["peak_season_start"] = peak_season_start
        if peak_season_end:
            season_data["peak_season_end"] = peak_season_end

        needs_pain_points = not known_pain_points or known_pain_points.strip() == ""
        needs_engagement_focus = not engagement_focus or engagement_focus.strip() == ""

        engagement_points = {}
        if needs_pain_points or needs_engagement_focus:
            engagement_points = self.extract_engagement_points(props, notes)
            if needs_pain_points and engagement_points.get("pain_points"):
                known_pain_points = ";".join(engagement_points["pain_points"])
            if needs_engagement_focus and engagement_points.get("engagement_focus"):
                engagement_focus = ";".join(engagement_points["engagement_focus"])
        else:
            engagement_points = {
                "pain_points": known_pain_points.split(";") if known_pain_points else [],
                "engagement_focus": engagement_focus.split(";") if engagement_focus else [],
                "assets": [],
                "tone": "Neutral"
            }

        company_name = company_data.get("name", "")
        external_insights = (
            self.orchestrator.analyze_competitors(company_name).get("data", {})
            if company_name and self.orchestrator
            else {"recent_news": []}
        )
        industry_trends = external_insights.get("recent_news", [])

        engaged_assets = engagement_points.get("assets", [])
        pain_points_list = engagement_points.get("pain_points", [])
        tone = engagement_points.get("tone", "Neutral")

        lead_care_about = [
            f"- Engaged with: {', '.join(engaged_assets) if engaged_assets else 'N/A'}",
            f"- Pain Points: {', '.join(pain_points_list) if pain_points_list else 'Not explicitly mentioned'}",
            f"- Interaction Tone: {tone}",
        ]


        top_trend_title = industry_trends[0]['title'] if industry_trends and 'title' in industry_trends[0] else 'N/A'
        club_context = [
            f"- Club Type: {club_type or 'Unknown'}",
            f"- Peak Season: {season_data.get('peak_season_start', 'Unknown')} to {season_data.get('peak_season_end', 'Unknown')}",
            f"- Industry Trend Mentioned: {top_trend_title}"
        ]

        return {
            "lead_summary": lead_care_about,
            "club_context": club_context,
            "subject": subject,
            "body": body
        }

    @staticmethod
    def infer_club_type(company_data: Dict[str, Any]) -> str:
        """Infer the club type from company data."""
        name = company_data.get("name", "").lower()
        if "country" in name:
            return "Country Club"
        elif "resort" in name:
            return "Resort Course"
        else:
            return "Golf Club"

    @staticmethod
    def extract_engagement_points(props: Dict[str, Any], notes: Any) -> Dict[str, Any]:
        """
        Analyze properties and notes to identify engaged assets and pain points.
        """
        engaged_assets = []
        pain_points = []
        engagement_focus = []
        tone = "Neutral"

        if int(props.get("hs_analytics_num_page_views", 0)) > 5:
            engaged_assets.append("Pricing Page")

        pain_keywords = [
            "slow beverage service", "staffing issues", "low member engagement",
            "limited menu options", "high operational costs",
            "inefficient technology systems", "long wait times on course"
        ]

        for note in notes:
            note_text = note.get("body", "").lower()
            for keyword in pain_keywords:
                if keyword in note_text and keyword not in pain_points:
                    pain_points.append(keyword)

        return {
            "assets": engaged_assets,
            "pain_points": pain_points,
            "engagement_focus": engagement_focus,
            "tone": tone
        }

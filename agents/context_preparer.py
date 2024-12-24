# agents/context_preparer.py

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

from config.settings import PROJECT_ROOT, DEBUG_MODE
from hubspot_integration.hubspot_api import (
    get_contact_by_email,
    get_lead_data_from_hubspot,
    get_associated_company_id,
    get_company_data,
    get_all_emails_for_contact
)
from external.external_api import (
    market_research,
    review_previous_interactions,
    determine_club_season
)
from hubspot_integration.data_enrichment import check_competitor_on_website
from utils.logging_setup import logger

# NEW IMPORT HERE:
from external.link_summarizer import summarize_recent_news

class LeadContextError(Exception):
    """Custom exception for lead context preparation errors."""
    pass

def get_domain_from_email(email: str) -> Optional[str]:
    if "@" in email:
        return email.split("@", 1)[1].lower().strip()
    return None

def create_context_directory() -> Path:
    context_dir = PROJECT_ROOT / "test_data" / "lead_contexts"
    context_dir.mkdir(parents=True, exist_ok=True)
    return context_dir

def generate_context_filename(contact_id: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"lead_context_{contact_id}_{timestamp}.json"

def _save_lead_context(lead_sheet: Dict[str, Any], contact_id: str) -> None:
    """
    Saves the lead_sheet to disk as JSON for local inspection or debugging.
    """
    try:
        context_dir = create_context_directory()
        filename = generate_context_filename(contact_id)
        file_path = context_dir / filename
        with file_path.open("w", encoding="utf-8") as f:
            json.dump(lead_sheet, f, indent=2, ensure_ascii=False)
        logger.info(f"Lead context saved for testing: {file_path.resolve()}")
    except Exception as e:
        logger.error(f"Failed to save lead context: {e}")
        # Not fatal, just log the error

def prepare_lead_context(lead_email: str) -> Dict[str, Any]:
    contact_id = get_contact_by_email(lead_email)
    if not contact_id:
        msg = f"No contact found for email: {lead_email}"
        logger.error(msg)
        raise LeadContextError(msg)

    lead_data = get_lead_data_from_hubspot(contact_id)
    if not lead_data:
        msg = f"No HubSpot lead data found for contact: {contact_id}"
        logger.error(msg)
        raise LeadContextError(msg)

    # NEW: Fetch all associated emails
    try:
        lead_emails = get_all_emails_for_contact(contact_id)
    except Exception as e:
        logger.error(f"Error fetching emails for contact {contact_id}: {e}")
        lead_emails = []

    # Store them in lead_data
    lead_data["emails"] = lead_emails

    company_id = get_associated_company_id(contact_id)
    company_data = get_company_data(company_id) if company_id else {}

    # Get competitor info (optional)
    domain = get_domain_from_email(lead_email)
    competitor = ""
    # if domain:
    #     try:
    #         competitor = check_competitor_on_website(domain)
    #     except Exception as e:
    #         logger.warning(f"Error checking competitor on domain {domain}: {e}")

    # Perform market research
    research_data = {}
    company_name = company_data.get("name", "").strip()
    if company_name:
        try:
            research_data = market_research(company_name)
        except Exception as e:
            logger.warning(f"Market research failed for '{company_name}': {e}")
            research_data = {"error": str(e)}

    # Summarize links if they exist
    if "recent_news" in research_data:
        link_summaries = summarize_recent_news(research_data["recent_news"])
        for item in research_data["recent_news"]:
            link = item.get("link", "")
            item["summary"] = link_summaries.get(link, "No summary available")

    # Interactions
    try:
        interactions = review_previous_interactions(contact_id)
    except Exception as e:
        logger.warning(f"Error reviewing interactions for {contact_id}: {e}")
        interactions = {"status": "error", "error": str(e)}

    # Season info
    try:
        season_info = determine_club_season(
            company_data.get("city", ""), 
            company_data.get("state", "")
        )
    except Exception as e:
        logger.warning(f"Error determining season info for contact {contact_id}: {e}")
        season_info = {"peak_season_start": "Unknown", "peak_season_end": "Unknown"}

    # IMPORTANT: Merge company_data into lead_data so we can reference it later
    lead_data["company_data"] = company_data

    # Create a summary string
    summary_text = (
        f"Lead: {lead_email}\n"
        f"Contact ID: {contact_id}\n"
        f"Competitor found: {competitor if competitor else 'None'}\n"
        f"Peak season: {season_info.get('peak_season_start', 'Unknown')} to "
        f"{season_info.get('peak_season_end', 'Unknown')}\n"
        f"Last response: {interactions.get('last_response', 'Unknown')}\n"
        "Overall lead readiness: [Apply custom lead scoring here]\n"
    )

    lead_sheet = {
        "metadata": {
            "created_at": datetime.now().isoformat(),
            "contact_id": contact_id,
            "company_id": company_id,
            "lead_email": lead_email,
            "status": "success"
        },
        "lead_data": lead_data,
        "company_data": company_data,
        "analysis": {
            "competitor_analysis": competitor,
            "research_data": research_data,
            "previous_interactions": interactions,
            "season_data": season_info
        },
        "summary_overview": summary_text
    }


    _save_lead_context(lead_sheet, contact_id)
    return lead_sheet
